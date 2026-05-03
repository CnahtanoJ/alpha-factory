import os
import json
import time
import sqlite3
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from ta.momentum import RSIIndicator
from ta.trend import MACD

from bot.config import AWS_BUCKET, TESTNET_MODE, BASE_URL
from bot.utils import S3Interface, send_telegram_message, send_telegram_receipt
from bot.data_feed import AssetManager, fetch_daily_receipt
from bot.indicators import get_local_poc, get_cvd_slope
from bot.strategies import STRATEGY_CONFIG
from bot.risk_engine import RiskEngine
from data_pipeline.hyperliquid_sync import (
    get_hl_top_by_volume, 
    get_live_meta_ctx, 
    get_latest_candles, 
    get_bulk_latest_candles
)
from data_pipeline.database import DB_PATH, get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LiveInferenceEngine:
    def __init__(self, info, conn=None):
        self.info = info
        self.conn = conn

    def build_live_features(self, symbol, candles, ctx, index_df=None):
        if not candles: return None
        df = pd.DataFrame(candles)
        # candles have t, o, h, l, c, v
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if index_df is not None and not index_df.empty:
            df = pd.merge_asof(df, index_df[['timestamp', 'close']].rename(columns={'close': 'hist_index_close'}), on='timestamp', direction='backward')
            df['corr_to_index'] = df['close'].pct_change().rolling(20).corr(df['hist_index_close'].pct_change())
        else:
            df['corr_to_index'] = 0

        
        # M-5 Fix: Use historical index prices from DB if available for basis_pct
        if self.conn and symbol:
            try:
                # Fetch index klines for this specific symbol
                idx_q = "SELECT timestamp as idx_ts, close as index_close FROM index_ohlcv WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100"
                idx_df = pd.read_sql_query(idx_q, self.conn, params=(symbol,))
                if not idx_df.empty:
                    idx_df['idx_ts'] = pd.to_datetime(idx_df['idx_ts'], unit='ms').dt.floor('s')
                    df = pd.merge_asof(df, idx_df, left_on='timestamp', right_on='idx_ts', direction='backward')
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch historical index data for {symbol}: {e}")

        # Live row always uses the most recent oracle price from Hyperliquid context
        df['live_index_close'] = float(ctx.get('oraclePx', df['close'].iloc[-1]))
        
        # If we have historical index data, use it for all but the last row
        if 'index_close' in df.columns:
            df['final_index_close'] = df['index_close'].fillna(df['live_index_close'])
            # Override the last row with the live oracle price
            df.loc[df.index[-1], 'final_index_close'] = df['live_index_close']
        else:
            df['final_index_close'] = df['live_index_close']

        df['sum_open_interest'] = ctx.get('openInterest', 0)
        df['last_funding_rate'] = ctx.get('funding', 0)
        
        # ── Derivative Fuel (P0-2 FIX: use real data, not zeros) ──
        # The training model now uses raw OI in USD, so we match it exactly.
        oi_value = float(ctx.get('openInterest', 0)) * float(ctx.get('oraclePx', 0))
        df['oi_usd'] = oi_value  # Raw OI in USD; will be ranked across all assets

        # Funding rate: training model now uses raw absolute funding rate.
        df['funding_rate'] = float(ctx.get('funding', 0))

        # HL doesn't provide top-trader long/short ratio. This feature will rank
        # uniformly (all ~0.5) and contribute minimal signal — acceptable since the
        # training model also saw many NaN-filled values for this feature.
        df['sum_toptrader_long_short_ratio'] = np.nan

        df['basis_pct'] = (df['close'] - df['final_index_close']) / df['final_index_close']
        
        # Cyclic Time Features
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_num'] = df['timestamp'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Strategy loop
        for strat_name, strat_info in STRATEGY_CONFIG.items():
            try:
                strat_class = strat_info['class']
                strat_instance = strat_class()
                df[f"sig_{strat_name}"] = strat_instance.get_signal_column(df)
            except Exception as e:
                logger.error(f"⚠️ Strategy {strat_name} failed for {symbol}: {e}")
                df[f"sig_{strat_name}"] = 0
                
        return df.iloc[-1:] # Return only the latest row

# ==========================================
if TESTNET_MODE:
    KEY = os.environ.get("TESTNET_PRIVATE_KEY")
    ADDR = os.environ.get("TESTNET_ACCOUNT_ADDRESS")
else:
    KEY = os.environ.get("MAINNET_PRIVATE_KEY")
    ADDR = os.environ.get("MAINNET_ACCOUNT_ADDRESS")

if not KEY or not ADDR:
    raise ValueError(f"❌ MISSING CREDENTIALS! Mode is {TESTNET_MODE}, but keys not found in Env Vars.")

def executor_handler(event, context):
    logger.info(f"🚀 Waking up Live Engine...")
    info = Info(BASE_URL, skip_ws=True)
    
    task = event.get("task", "execute_trades")
    if task == "send_daily_report":
        stats = fetch_daily_receipt(info, ADDR)
        msg = send_telegram_receipt(stats)
        send_telegram_message(msg)
        return {'statusCode': 200, 'body': 'Daily report sent'}

    try:
        # 1. 🛡️ ACCOUNT SAFETY FIRST
        account = Account.from_key(KEY)
        exchange = Exchange(account, BASE_URL, account_address=ADDR)
        risk = RiskEngine(exchange=exchange, info=info, account_address=ADDR, bucket=AWS_BUCKET)
        
        user_state = info.user_state(ADDR)
        if not risk.check_safety(): return {'statusCode': 400, 'body': 'Safety failed'}
        
        all_mids = info.all_mids()
        portfolio = risk.parse_portfolio(user_state, all_mids)
        open_orders = info.frontend_open_orders(ADDR)
        
        # Clean Zombies
        risk.clean_global_zombies(portfolio, open_orders)

        # 2. 🌐 THE LIVE SNAPSHOT
        logger.info("📡 Pinging Hyperliquid for Top 50 assets & Live Meta Context...")
        top_50_symbols = get_hl_top_by_volume(50)
        live_ctx = get_live_meta_ctx()
        
        db_conn = get_connection()
        
        engine = LiveInferenceEngine(info, conn=db_conn)
        live_rows = []
        
        # 3. 🧠 FEATURE GENERATION
        logger.info("🧬 Generating Live Features (Klines + Strategies)...")
        
        index_candles = get_latest_candles('BTC', interval='15m', limit=100)
        if index_candles:
            index_df = pd.DataFrame(index_candles)
            index_df['timestamp'] = pd.to_datetime(index_df['timestamp'], unit='ms')
            index_df = index_df.sort_values('timestamp').reset_index(drop=True)
        else:
            index_df = None
        
        # Fetch all 50 assets concurrently in batches (takes ~5-10 seconds total)
        bulk_candles = get_bulk_latest_candles(top_50_symbols, interval='15m', limit=100)
        
        for sym, candles in bulk_candles.items():
            ctx = live_ctx.get(sym, {})
            row_df = engine.build_live_features(sym, candles, ctx, index_df=index_df)
            if row_df is not None and not row_df.empty:
                row_df['symbol'] = sym
                
                # Need ATR for risk engine later
                high, low = pd.Series([c['high'] for c in candles]), pd.Series([c['low'] for c in candles])
                prev_close = pd.Series([c['close'] for c in candles]).shift(1)
                tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                atr_pct = (tr.rolling(14).mean() / pd.Series([c['close'] for c in candles])).iloc[-1]
                row_df['atr_pct'] = atr_pct
                
                live_rows.append(row_df)
                
        if not live_rows:
            logger.error("❌ Failed to build any live features.")
            return
            
        mega_df = pd.concat(live_rows, ignore_index=True)
        
        # 4. 🥇 THE LIVE RANKING
        logger.info("⚖️ Applying Cross-Sectional Ranking...")
        continuous_features = ['rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_usd', 'funding_rate', 'sum_toptrader_long_short_ratio', 'corr_to_index']
        
        # Fill NaN with 0 before ranking (NaN would be excluded from rank())
        for col in continuous_features:
            mega_df[col] = mega_df[col].fillna(0)
        
        for col in continuous_features:
            mega_df[f'rank_{col}'] = mega_df[col].rank(pct=True)

        # 5. 🤖 ENSEMBLE INFERENCE
        s3 = S3Interface(AWS_BUCKET)
        lgb_path = '/tmp/cross_sectional_lgbm.txt'
        xgb_path = '/tmp/cross_sectional_xgboost.json'
        
        logger.info("🧠 Downloading Ensemble Models (LGBM + XGB) from S3...")
        success_lgb = s3.download_file('models/cross_sectional_lgbm.txt', lgb_path)
        success_xgb = s3.download_file('models/cross_sectional_xgboost.json', xgb_path)
        
        ml_success = False
        if success_lgb and success_xgb:
            try:
                # Load Models
                model_lgb = lgb.Booster(model_file=lgb_path)
                model_xgb = xgb.XGBRegressor()
                model_xgb.load_model(xgb_path)
                
                # Prepare Features
                feature_cols = ['rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_usd', 'rank_funding_rate', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index']
                time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
                strategy_cols = [col for col in mega_df.columns if col.startswith('sig_')]
                X_live = mega_df[feature_cols + time_features + strategy_cols]
                
                # Combined Prediction
                preds_lgb = model_lgb.predict(X_live)
                preds_xgb = model_xgb.predict(X_live)
                mega_df['predicted_rank'] = (preds_lgb + preds_xgb) / 2.0
                
                logger.info("✅ Ensemble Inference Complete.")
                ml_success = True
            except Exception as e:
                logger.error(f"⚠️ Ensemble Inference Failed: {e}")
                
        if not ml_success:
            logger.warning("⚠️ Falling back to simple Momentum Rank (RSI + MACD).")
            mega_df['predicted_rank'] = (mega_df['rank_rsi'] + mega_df['rank_macd']) / 2

        # 6. 🔪 SORT AND SLICE (Market Neutral Basket with Hysteresis)
        BASKET_N = 3
        HYSTERESIS_FACTOR = 3.0
        buffer_n = int(BASKET_N * HYSTERESIS_FACTOR)  # =9, keeps coins if they stay in top 9/bottom 9
        
        mega_df = mega_df.sort_values('predicted_rank', ascending=False)
        
        # Get currently open positions to apply hysteresis
        current_longs = set(coin for coin, pos in portfolio.items() if pos.get('szi', 0) > 0)
        current_shorts = set(coin for coin, pos in portfolio.items() if pos.get('szi', 0) < 0)
        
        # --- LONG SELECTION with Hysteresis ---
        eligible_longs = mega_df.head(buffer_n)
        kept_longs = eligible_longs[eligible_longs['symbol'].isin(current_longs)]['symbol'].tolist()
        needed_longs = BASKET_N - len(kept_longs)
        new_longs = eligible_longs[~eligible_longs['symbol'].isin(current_longs)].head(needed_longs)['symbol'].tolist()
        target_longs = kept_longs + new_longs
        
        # --- SHORT SELECTION with Hysteresis ---
        eligible_shorts = mega_df.tail(buffer_n)
        kept_shorts = eligible_shorts[eligible_shorts['symbol'].isin(current_shorts)]['symbol'].tolist()
        needed_shorts = BASKET_N - len(kept_shorts)
        new_shorts = eligible_shorts[~eligible_shorts['symbol'].isin(current_shorts)].tail(needed_shorts)['symbol'].tolist()
        target_shorts = kept_shorts + new_shorts
        
        target_basket = target_longs + target_shorts
        logger.info(f"🎯 TARGET LONGS: {target_longs} (kept: {kept_longs})")
        logger.info(f"🎯 TARGET SHORTS: {target_shorts} (kept: {kept_shorts})")
        
        # --- PHASE 5: MARKET-NEUTRAL EXECUTION ---
        
        # 1. PORTFOLIO RECONCILIATION
        temp_assets = AssetManager(info)
        for active_coin in list(portfolio.keys()):
            # If open position is NOT in our new basket, KILL IT.
            if active_coin not in target_basket:
                logger.info(f"🧹 RECONCILIATION: {active_coin} dropped out of buffer zone. Closing position.")
                send_telegram_message(f"🧹 RECONCILIATION: {active_coin} lost its edge. Closing.")
                risk.close_active_position(active_coin, all_mids, temp_assets, portfolio, AWS_BUCKET)
                
        # Refresh state after closes
        time.sleep(2)
        user_state = info.user_state(ADDR)
        portfolio = risk.parse_portfolio(user_state, all_mids)
        open_orders = info.frontend_open_orders(ADDR)

        # 2. MARKET-NEUTRAL ENTRY (Risk Parity Optimized)
        target_rows = mega_df[mega_df['symbol'].isin(target_basket)].copy()
        
        if not target_rows.empty:
            # A. Calculate Risk Parity Weights
            # Weight = (1/ATR) / sum(1/ATR)
            # P3-3: Failsafe against NaN/Zero ATR preventing all execution
            target_rows['inv_vol'] = 1.0 / (target_rows['atr_pct'].fillna(0.001).clip(lower=0.001))
            total_inv_vol = target_rows['inv_vol'].sum()
            target_rows['rp_weight'] = target_rows['inv_vol'] / total_inv_vol
            
            # B. Target Total Portfolio Exposure
            # We want 90% total leverage (0.15 * 6)
            equity = float(user_state.get('marginSummary', {}).get('accountValue', 0))
            total_target_usd = equity * 0.90
            
            # Map symbol -> target_usd
            symbol_to_usd = dict(zip(target_rows['symbol'], target_rows['rp_weight'] * total_target_usd))
            
            for target_coin in target_basket:
                row = mega_df[mega_df['symbol'] == target_coin].iloc[0]
                signal = "BULLISH" if target_coin in target_longs else "BEARISH"
                atr_pct = float(row['atr_pct'])
                
                if target_coin not in portfolio:
                    # H-5: PRE-TRADE SAFETY CHECK
                    is_buy = (signal == "BULLISH")
                    current_price = float(all_mids[target_coin])
                    coin_candles = bulk_candles.get(target_coin, [])
                    # C-5 FIX: get_local_poc expects a DataFrame, not a list of dicts
                    poc_price = get_local_poc(pd.DataFrame(coin_candles)) if coin_candles else 0.0
                    
                    if risk.check_execution_safety(target_coin, is_buy, current_price, poc_price):
                        logger.info(f"🚀 BASKET ENTRY: {signal} {target_coin}")
                        risk_parity_usd = symbol_to_usd.get(target_coin)
                        risk.execute_logic(target_coin, signal, "15m", atr_pct, portfolio, user_state, open_orders, override_usd=risk_parity_usd)
                    else:
                        logger.warning(f"🛑 SAFETY REJECTION: {target_coin} failed pre-trade checks.")
                else:
                    # We already have it. Just sync Breakeven logic.
                    risk.sync_break_even(target_coin, atr_pct, portfolio, open_orders)
                
        # 3. THE SHIELD
        # Refresh one last time to ensure we have the new positions
        time.sleep(2)
        portfolio = risk.parse_portfolio(info.user_state(ADDR), all_mids)
        for active_coin in portfolio:
            row = mega_df[mega_df['symbol'] == active_coin]
            if not row.empty:
                atr_pct = float(row.iloc[0]['atr_pct'])
                # Place Unified TP/SL
                risk.sync_unified_orders(active_coin, atr_pct, portfolio)

    except Exception as e:
        logger.error(f"💥 GLOBAL EXECUTOR ERROR: {e}")
        send_telegram_message(f"💥 GLOBAL EXECUTOR ERROR: {e}")
    finally:
        # 4. BATCH SAVE STATE (P1-6 Fix: Prevent S3 Write Storm)
        if 'db_conn' in locals():
            db_conn.close()
            
        if 'risk' in locals():
            risk.memory.save()

    logger.info("✅ Live Engine Cycle Complete.")
    return {'statusCode': 200, 'body': json.dumps('Basket Executed')}
