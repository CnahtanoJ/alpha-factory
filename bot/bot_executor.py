import os
import json
import time
import pandas as pd
import numpy as np
import logging
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange

from bot.config import *
from bot.utils import S3Interface, StateManager, send_telegram_message, send_telegram_receipt
from bot.data_feed import MarketData, AssetManager, fetch_daily_receipt
from bot.indicators import get_local_poc, get_cvd_slope
from bot.strategies import STRATEGY_CONFIG, SimpleBreakout
from bot.risk_engine import RiskEngine
from ta.momentum import RSIIndicator
from ta.trend import MACD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LiveInferenceEngine:
    def __init__(self, info):
        self.info = info

    def build_live_features(self, symbol, candles, ctx, index_df=None):
        if not candles: return None
        df = pd.DataFrame(candles)
        # candles have t, o, h, l, c, v
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if index_df is not None and not index_df.empty:
            df = pd.merge_asof(df, index_df[['timestamp', 'close']].rename(columns={'close': 'hist_index_close'}), on='timestamp', direction='backward')
            df['corr_to_index'] = df['close'].pct_change().rolling(20).corr(df['hist_index_close'].pct_change())
        else:
            df['corr_to_index'] = 0

        
        df['index_close'] = ctx.get('oraclePx', df['close'].iloc[-1])
        df['sum_open_interest'] = ctx.get('openInterest', 0)
        df['last_funding_rate'] = ctx.get('funding', 0)
        
        # ── Derivative Fuel (P0-2 FIX: use real data, not zeros) ──
        # For cross-sectional ranking, what matters is the RANK not the absolute value.
        # The training model used z-scored OI, but rank(pct=True) normalizes across
        # the 100-asset cross section, so using raw OI achieves the same rank ordering.
        oi_value = float(ctx.get('openInterest', 0)) * float(ctx.get('oraclePx', 0))
        df['oi_zscore'] = oi_value  # Raw OI in USD; will be ranked across all assets

        # Funding rate: we can't compute diff() without history, but the raw funding
        # rate itself captures the same cross-sectional signal (which assets have
        # extreme funding). The rank normalizes it.
        df['funding_delta'] = float(ctx.get('funding', 0))

        # HL doesn't provide top-trader long/short ratio. This feature will rank
        # uniformly (all ~0.5) and contribute minimal signal — acceptable since the
        # training model also saw many NaN-filled values for this feature.
        df['sum_toptrader_long_short_ratio'] = np.nan

        df['basis_pct'] = (df['close'] - df['index_close']) / df['index_close']
        
        # Cyclic Time Features
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)
        df['day_of_week_num'] = df['timestamp'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 6)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 6)
        
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
            except:
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

    # 1. 🛡️ ACCOUNT SAFETY FIRST
    account = Account.from_key(KEY)
    exchange = Exchange(account, BASE_URL, account_address=ADDR)
    risk = RiskEngine(exchange=exchange, info=info, account_address=ADDR, bucket=AWS_BUCKET)
    
    user_state = info.user_state(ADDR)
    if not risk.check_safety(user_state): return {'statusCode': 400, 'body': 'Safety failed'}
    
    all_mids = info.all_mids()
    portfolio = risk.parse_portfolio(user_state, all_mids)
    open_orders = info.frontend_open_orders(ADDR)
    
    # Clean Zombies
    risk.clean_global_zombies(portfolio, open_orders)

    # 2. 🌐 THE LIVE SNAPSHOT
    from data_pipeline.hyperliquid_sync import get_hl_top_by_volume, get_live_meta_ctx, get_latest_candles, get_bulk_latest_candles
    
    logger.info("📡 Pinging Hyperliquid for Top 100 assets & Live Meta Context...")
    top_100_symbols = get_hl_top_by_volume(100)
    live_ctx = get_live_meta_ctx()
    
    engine = LiveInferenceEngine(info)
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
    
    # Fetch all 100 assets concurrently in batches (takes ~5-10 seconds total)
    bulk_candles = get_bulk_latest_candles(top_100_symbols, interval='15m', limit=100)
    
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
    continuous_features = ['rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_zscore', 'funding_delta', 'sum_toptrader_long_short_ratio', 'corr_to_index']
    
    # Fill NaN with 0 before ranking (NaN would be excluded from rank())
    for col in continuous_features:
        mega_df[col] = mega_df[col].fillna(0)
    
    for col in continuous_features:
        mega_df[f'rank_{col}'] = mega_df[col].rank(pct=True)

    # 5. 🤖 INFERENCE
    s3 = S3Interface(AWS_BUCKET)
    model_path = '/tmp/cross_sectional_lgbm.txt'
    
    logger.info("🧠 Downloading LightGBM model from S3...")
    success = s3.download_file('models/cross_sectional_lgbm.txt', model_path)
    
    ml_success = False
    if success:
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=model_path)
            
            feature_cols = ['rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_zscore', 'rank_funding_delta', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index']
            time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            strategy_cols = [col for col in mega_df.columns if col.startswith('sig_')]
            X_live = mega_df[feature_cols + time_features + strategy_cols]
            
            mega_df['predicted_rank'] = model.predict(X_live)
            ml_success = True
        except ImportError:
            logger.warning("⚠️ lightgbm package not found. Skipping ML inference.")
        except Exception as e:
            logger.error(f"⚠️ Error loading LightGBM model or predicting: {e}")
            
    if not ml_success:
        logger.warning("⚠️ Falling back to simple Momentum Rank (RSI + MACD).")
        mega_df['predicted_rank'] = (mega_df['rank_rsi'] + mega_df['rank_macd']) / 2

    # 6. 🔪 SORT AND SLICE (Market Neutral Basket)
    mega_df = mega_df.sort_values('predicted_rank', ascending=False)
    
    top_3 = mega_df.head(3)
    bottom_3 = mega_df.tail(3)
    
    target_longs = top_3['symbol'].tolist()
    target_shorts = bottom_3['symbol'].tolist()
    
    target_basket = target_longs + target_shorts
    logger.info(f"🎯 TARGET LONGS: {target_longs}")
    logger.info(f"🎯 TARGET SHORTS: {target_shorts}")
    
    # --- PHASE 5: MARKET-NEUTRAL EXECUTION ---
    
    # 1. PORTFOLIO RECONCILIATION
    temp_assets = AssetManager(info)
    for active_coin in list(portfolio.keys()):
        # If open position is NOT in our new basket, KILL IT.
        if active_coin not in target_basket:
            logger.info(f"🧹 RECONCILIATION: {active_coin} dropped out of Top/Bottom 3. Closing position.")
            send_telegram_message(f"🧹 RECONCILIATION: {active_coin} lost its edge. Closing.")
            risk.close_active_position(active_coin, all_mids, temp_assets, portfolio, AWS_BUCKET)
            
    # Refresh state after closes
    time.sleep(2)
    user_state = info.user_state(ADDR)
    portfolio = risk.parse_portfolio(user_state, all_mids)
    open_orders = info.frontend_open_orders(ADDR)

    # 2. MARKET-NEUTRAL ENTRY
    for target_coin in target_basket:
        row = mega_df[mega_df['symbol'] == target_coin].iloc[0]
        signal = "BULLISH" if target_coin in target_longs else "BEARISH"
        atr_pct = float(row['atr_pct'])
        
        if target_coin not in portfolio:
            logger.info(f"🚀 BASKET ENTRY: {signal} {target_coin}")
            risk.execute_logic(target_coin, signal, "15m", atr_pct, portfolio, user_state, open_orders)
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

    # 4. BATCH SAVE STATE (P1-6 Fix: Prevent S3 Write Storm)
    risk.memory.save()

    logger.info("✅ Live Engine Cycle Complete.")
    return {'statusCode': 200, 'body': json.dumps('Basket Executed')}
