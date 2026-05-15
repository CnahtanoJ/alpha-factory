import os
import json
import time
import sqlite3
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
import joblib
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from ta.momentum import RSIIndicator
from ta.trend import MACD

from bot.config import AWS_BUCKET, TESTNET_MODE, BASE_URL
from bot.utils import S3Interface, send_telegram_message, send_telegram_receipt
from bot.data_feed import AssetManager, fetch_daily_receipt
from bot.indicators import get_local_poc, get_cvd_slope
from bot.risk_engine import RiskEngine
from data_pipeline.hyperliquid_sync import (
    get_hl_top_by_volume, 
    get_live_meta_ctx, 
    get_latest_candles, 
    get_bulk_latest_candles
)
from data_pipeline.database import DB_PATH, get_connection
from data_pipeline.binance_live import get_bulk_binance_sentiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LiveInferenceEngine:
    def __init__(self, info, conn=None, live_sentiment=None):
        self.info = info
        self.conn = conn
        self.live_sentiment = live_sentiment

    def build_live_features(self, symbol, candles, ctx, index_df=None, htf_candles=None):
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

        # --- LIVE SENTIMENT INJECTION ---
        # Map Binance dict keys to canonical model feature names
        top_trader = np.nan
        global_retail = np.nan
        
        if self.live_sentiment and symbol in self.live_sentiment:
            # Sync with Binance API dict keys
            top_trader = self.live_sentiment[symbol].get('top_trader_ratio')
            global_retail = self.live_sentiment[symbol].get('long_short_ratio')
            
        df['sum_toptrader_long_short_ratio'] = top_trader
        df['sum_long_short_ratio'] = global_retail
        
        # Fallback to neutral cross-sectional mean (later in the pipeline)
        if pd.notna(top_trader) and pd.notna(global_retail):
            df['sentiment_divergence'] = top_trader - global_retail
        else:
            df['sentiment_divergence'] = 0.0

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
        
        # --- NEW CONTINUOUS FEATURES ---
        
        # 1. Derivatives Velocity
        df['oi_delta_4'] = df['oi_usd'].pct_change(4)
        df['funding_delta_4'] = df['funding_rate'].diff(4)
        df['sum_toptrader_ls_delta_4'] = df['sum_toptrader_long_short_ratio'].diff(4)
        
        # Net Taker Volume Z-score Proxy (from CVD delta)
        vol_ma_20 = df['volume'].rolling(20).mean()
        df['net_taker_volume_zscore'] = (df['cvd_slope_5'] / (vol_ma_20 + 1e-9)).fillna(0.0)
        
        # 2. Momentum & Mean Reversion Refinements
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        df['distance_from_ema_50'] = (df['close'] - ema_50) / ema_50
        
        vol_mean = df['volatility_20'].rolling(100).mean()
        vol_std = df['volatility_20'].rolling(100).std()
        df['volatility_zscore'] = (df['volatility_20'] - vol_mean) / (vol_std + 1e-9)
        
        vol_ma = df['volume'].rolling(50).mean()
        vol_sd = df['volume'].rolling(50).std()
        df['volume_zscore'] = (df['volume'] - vol_ma) / (vol_sd + 1e-9)
        
        # 3. Relative Strength (vs BTC) & Market Beta
        if index_df is not None and not index_df.empty:
            df['ret_12'] = df['close'].pct_change(12)
            
            # Use 'hist_index_close' which was mapped from index_df in earlier step
            df['btc_ret_12'] = df['hist_index_close'].pct_change(12)
            df['relative_strength_btc'] = df['ret_12'] - df['btc_ret_12']
            
            # Phase 5: Market Beta
            asset_ret_1 = df['close'].pct_change()
            idx_ret_1 = df['hist_index_close'].pct_change()
            cov = asset_ret_1.rolling(20).cov(idx_ret_1)
            var = idx_ret_1.rolling(20).var()
            df['market_beta'] = (cov / (var + 1e-9)).fillna(0.0)
        else:
            df['relative_strength_btc'] = 0.0
            df['market_beta'] = 0.0
            
        # 4. Proxy CVD & Divergence
        candle_range = df['high'] - df['low']
        candle_range = candle_range.replace(0, 1e-9)
        delta = df['volume'] * ((df['close'] - df['open']) / candle_range)
        cvd = delta.cumsum()
        
        df['cvd_slope_5'] = cvd.diff(5)
        
        # Normalize CVD slope and price return for divergence
        price_ret_5 = df['close'].pct_change(5)
        norm_cvd_slope = df['cvd_slope_5'] / (df['volume'].rolling(20).mean() * 5 + 1e-9)
        df['price_cvd_divergence'] = price_ret_5 - norm_cvd_slope
        
        # 5. HTF Features & Divergences
        # Trend Convergence
        ema_50_slope = df['close'].ewm(span=50, adjust=False).mean().pct_change(5)
        
        if htf_candles:
            htf_df = pd.DataFrame(htf_candles)
            htf_df['timestamp'] = pd.to_datetime(htf_df['timestamp'], unit='ms').dt.floor('s')
            htf_df = htf_df.sort_values('timestamp').reset_index(drop=True)
            
            htf_df['ema_50_4h'] = htf_df['close'].ewm(span=50, adjust=False).mean()
            htf_df['htf_ts'] = htf_df['timestamp'] + pd.Timedelta(hours=4)
            htf_df['htf_ts'] = htf_df['htf_ts'].astype(df['timestamp'].dtype)
            
            df = pd.merge_asof(df, htf_df[['htf_ts', 'ema_50_4h']], left_on='timestamp', right_on='htf_ts', direction='backward')
            ema_50_4h_slope = df.get('ema_50_4h', df['close']).pct_change(16)
        else:
            ema_50_4h_slope = df['close'].pct_change(16)
            
        df['trend_convergence'] = (ema_50_slope * ema_50_4h_slope).fillna(0.0)
        
        # BBW Squeeze (Normalized over 100 periods)
        sma_20 = df['close'].rolling(20).mean()
        bbw_20 = df['volatility_20'] / (sma_20 + 1e-9)
        bbw_100_min = bbw_20.rolling(100).min()
        bbw_100_max = bbw_20.rolling(100).max()
        df['bbw_squeeze'] = ((bbw_20 - bbw_100_min) / (bbw_100_max - bbw_100_min + 1e-9)).fillna(0.0)
        
        # Funding / Basis Divergence
        fund_100_mean = df['funding_rate'].rolling(100).mean()
        fund_100_std = df['funding_rate'].rolling(100).std()
        fund_z = (df['funding_rate'] - fund_100_mean) / (fund_100_std + 1e-9)
        
        basis_100_mean = df['basis_pct'].rolling(100).mean()
        basis_100_std = df['basis_pct'].rolling(100).std()
        
        # Phase 5: Additional Missing Features for Parity
        # VPT Slope (Volume Price Trend)
        vpt = (df['volume'] * df['close'].pct_change()).cumsum()
        df['vpt_slope'] = vpt.diff(5) / (df['volume'].rolling(20).mean() + 1e-9)
        
        # Range Expansion
        atr = df['volatility_20'] # Use 20-period std as proxy for volatility if ATR isn't ready
        df['range_expansion'] = (df['high'] - df['low']) / (atr + 1e-9)
        
        # RSI Divergence Proxy
        rsi_delta_5 = df['rsi'].diff(5)
        price_delta_5 = df['close'].pct_change(5)
        df['rsi_divergence'] = rsi_delta_5 - price_delta_5
        basis_z = (df['basis_pct'] - basis_100_mean) / (basis_100_std + 1e-9)
        
        df['funding_basis_divergence'] = (fund_z - basis_z).fillna(0.0)
        
        # 6. Phase 5 Features
        # Volume to Volatility Ratio
        df['vol_volatility_ratio'] = df['volume'] / (df['volatility_20'] + 1e-9)
        
        # RSI Timeframe Divergence
        rsi_7 = RSIIndicator(close=df['close'], window=7).rsi()
        rsi_21 = RSIIndicator(close=df['close'], window=21).rsi()
        df['rsi_divergence'] = rsi_7 - rsi_21
        
        # Volume Price Trend (VPT) Slope
        prev_close = df['close'].shift(1)
        vpt = df['volume'] * ((df['close'] - prev_close) / (prev_close + 1e-9))
        df['vpt_slope'] = vpt.cumsum().diff(5)
        
        # Intraday Range Expansion
        tr_live = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14_live = tr_live.rolling(14).mean()
        df['range_expansion'] = (df['high'] - df['low']) / (atr_14_live + 1e-9)
        
        # --- NEW PHASE 2 DELTA FEATURES ---
        df['rsi_delta_4'] = df['rsi'].diff(4)
        df['macd_delta_4'] = df['macd'].diff(4)
        df['volatility_delta_4'] = df['volatility_20'].diff(4)
        df['volume_delta_4'] = df['volume'].pct_change(4)
        df['oi_change_pct_12'] = df['oi_usd'].pct_change(12)
        df['funding_acceleration'] = df['funding_delta_4'].diff(1)
        
        # Return only the latest row, but keep features needed for ranking and regime
        return df.iloc[-1:] 

# ==========================================
# Credentials will be loaded dynamically inside executor_handler based on timeframe

def executor_handler(event, context):
    logger.info(f"🚀 Waking up Live Engine...")
    info = Info(BASE_URL, skip_ws=True)
    
    timeframe = event.get("timeframe", "15m")
    env_suffix = timeframe.upper()
    
    # 0. 🔑 DYNAMIC CREDENTIAL LOADING (Sub-Account Architecture)
    if TESTNET_MODE:
        KEY = os.environ.get(f"TESTNET_PRIVATE_KEY_{env_suffix}", os.environ.get("TESTNET_PRIVATE_KEY"))
        ADDR = os.environ.get(f"TESTNET_ACCOUNT_ADDRESS_{env_suffix}", os.environ.get("TESTNET_ACCOUNT_ADDRESS"))
    else:
        KEY = os.environ.get(f"MAINNET_PRIVATE_KEY_{env_suffix}", os.environ.get("MAINNET_PRIVATE_KEY"))
        ADDR = os.environ.get(f"MAINNET_ACCOUNT_ADDRESS_{env_suffix}", os.environ.get("MAINNET_ACCOUNT_ADDRESS"))

    if not KEY or not ADDR:
        raise ValueError(f"❌ MISSING CREDENTIALS! Could not find API keys for timeframe '{timeframe}'. Checked suffix '_{env_suffix}' and default.")
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

        logger.info(f"🌐 THE LIVE SNAPSHOT ({timeframe})")
        logger.info(f"📡 Pinging Hyperliquid for Top 50 assets & Live Meta Context...")
        top_50_symbols = get_hl_top_by_volume(50)
        live_ctx = get_live_meta_ctx()
        
        logger.info(f"🐳 Fetching Live Sentiment from Binance ({timeframe})...")
        live_sentiment = get_bulk_binance_sentiment(top_50_symbols, period=timeframe)
        
        db_conn = get_connection()
        
        engine = LiveInferenceEngine(info, conn=db_conn, live_sentiment=live_sentiment)
        live_rows = []
        
        # 3. 🧠 FEATURE GENERATION
        logger.info(f"🧬 Generating Live Features ({timeframe})...")
        
        index_candles = get_latest_candles('BTC', interval=timeframe, limit=100)
        if index_candles:
            index_df = pd.DataFrame(index_candles)
            index_df['timestamp'] = pd.to_datetime(index_df['timestamp'], unit='ms')
            index_df = index_df.sort_values('timestamp').reset_index(drop=True)
        else:
            index_df = None
        
        # Fetch all 50 assets concurrently in batches (takes ~5-10 seconds total)
        bulk_candles = get_bulk_latest_candles(top_50_symbols, interval=timeframe, limit=100)
        bulk_htf_candles = get_bulk_latest_candles(top_50_symbols, interval='4h', limit=50)
        
        for sym, candles in bulk_candles.items():
            ctx = live_ctx.get(sym, {})
            htf_candles = bulk_htf_candles.get(sym, [])
            row_df = engine.build_live_features(sym, candles, ctx, index_df=index_df, htf_candles=htf_candles)
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
        
        # 3.5 🌍 Inject Market Regime Features
        logger.info("🌍 Injecting Market Regime features...")
        if index_df is not None:
            # We need at least 24 bars to compute these correctly
            index_df['btc_ret_24'] = index_df['close'].pct_change(24)
            index_df['btc_volatility_24'] = index_df['close'].pct_change().rolling(24).std()
            
            btc_ret_24 = index_df['btc_ret_24'].iloc[-1]
            btc_vol_24 = index_df['btc_volatility_24'].iloc[-1]
            
            mega_df['btc_ret_24'] = btc_ret_24
            mega_df['btc_volatility_24'] = btc_vol_24
        else:
            mega_df['btc_ret_24'] = 0.0
            mega_df['btc_volatility_24'] = 0.0

        if 'ret_12' in mega_df.columns:
            mega_df['market_breadth'] = (mega_df['ret_12'] > 0).mean()
        else:
            mega_df['market_breadth'] = 0.5
        
        # PHASE 4: Regime Score
        btc_ret_val = mega_df['btc_ret_24'].iloc[0] if 'btc_ret_24' in mega_df.columns else 0.0
        btc_vol_val = mega_df['btc_volatility_24'].iloc[0] if 'btc_volatility_24' in mega_df.columns else 1e-9
        mega_df['regime_score'] = abs(btc_ret_val) / (btc_vol_val + 1e-9)
        
        # PHASE 4: Macro Conviction Injection (4h model predictions)
        if timeframe != '4h':
            try:
                macro_lgb_path = f'/tmp/cross_sectional_lgbm_4h.txt'
                macro_s3 = S3Interface(AWS_BUCKET)
                if macro_s3.download_file('models/cross_sectional_lgbm_4h.txt', macro_lgb_path):
                    macro_model = lgb.Booster(model_file=macro_lgb_path)
                    macro_features = macro_model.feature_name()
                    # Build 4h-compatible features from live data and predict
                    avail_macro = [f for f in macro_features if f in mega_df.columns]
                    if len(avail_macro) >= len(macro_features) * 0.5:
                        X_macro = mega_df[avail_macro].fillna(0.0)
                        mega_df['macro_conviction_4h'] = macro_model.predict(X_macro)
                        logger.info(f"🔮 Macro conviction injected ({len(avail_macro)}/{len(macro_features)} features)")
                    else:
                        mega_df['macro_conviction_4h'] = 0.0
                        logger.warning(f"⚠️ Macro conviction: only {len(avail_macro)}/{len(macro_features)} features, using neutral")
                else:
                    mega_df['macro_conviction_4h'] = 0.0
            except Exception as e:
                logger.warning(f"⚠️ Macro conviction injection failed: {e}")
                mega_df['macro_conviction_4h'] = 0.0
        else:
            mega_df['macro_conviction_4h'] = 0.0
        
        # 4. 🥇 THE LIVE RANKING
        logger.info("⚖️ Applying Cross-Sectional Ranking...")
        # M-3 FIX: Use canonical raw features for ranking
        from analytics.cross_sectional import RAW_CONTINUOUS
        continuous_features = RAW_CONTINUOUS
        
        # Fill NaN with 0 before ranking (NaN would be excluded from rank())
        for col in continuous_features:
            if col in mega_df.columns:
                mega_df[col] = mega_df[col].fillna(0)
        
        for col in continuous_features:
            if col in mega_df.columns:
                # SAFE FALLBACK: If we have some values but not all (e.g. rate limit), 
                # use the cross-sectional mean instead of 0 or NaN.
                col_mean = mega_df[col].mean()
                if pd.isna(col_mean): col_mean = 0.0
                mega_df[col] = mega_df[col].fillna(col_mean)
                
                mega_df[f'rank_{col}'] = mega_df[col].rank(pct=True)

        # 5. 🤖 ENSEMBLE INFERENCE
        s3 = S3Interface(AWS_BUCKET)
        lgb_path = f'/tmp/cross_sectional_lgbm_{timeframe}.txt'
        xgb_path = f'/tmp/cross_sectional_xgboost_{timeframe}.json'
        ridge_path = f'/tmp/cross_sectional_ridge_{timeframe}.joblib'
        
        logger.info(f"🧠 Downloading Ensemble Models (LGBM + XGB + Ridge) from S3 for timeframe {timeframe}...")
        success_lgb = s3.download_file(f'models/cross_sectional_lgbm_{timeframe}.txt', lgb_path)
        success_xgb = s3.download_file(f'models/cross_sectional_xgboost_{timeframe}.json', xgb_path)
        success_ridge = s3.download_file(f'models/cross_sectional_ridge_{timeframe}.joblib', ridge_path)
        
        ml_success = False
        if success_lgb and success_xgb and success_ridge:
            try:
                # Load Models
                model_lgb = lgb.Booster(model_file=lgb_path)
                model_xgb = xgb.XGBRegressor()
                model_xgb.load_model(xgb_path)
                model_ridge = joblib.load(ridge_path)
                
                # Prepare Features (Asymmetric Pruning Match)
                # M-3 FIX: Use canonical feature retrieval
                from analytics.cross_sectional import get_feature_names
                feature_cols, time_features = get_feature_names(timeframe)

                X_live = mega_df[feature_cols + time_features]
                
                # Combined Prediction (Weighted Average: 40/40/20)
                lgb_preds = model_lgb.predict(X_live)
                xgb_preds = model_xgb.predict(X_live)
                ridge_preds = model_ridge.predict(X_live)
                
                mega_df['predicted_rank'] = (0.4 * lgb_preds) + (0.4 * xgb_preds) + (0.2 * ridge_preds)
                
                # 🛡️ PHASE 4: Macro Safeguard
                # If macro conviction is strong, dampen conflicting micro signals
                if 'macro_conviction_4h' in mega_df.columns:
                    # Conviction scale is usually around -1.0 to 1.0 (z-score like)
                    # We dampen if they are opposite signs
                    macro_sign = np.sign(mega_df['macro_conviction_4h'])
                    micro_sign = np.sign(mega_df['predicted_rank'])
                    
                    # If they conflict, dampen micro signal by 30%
                    conflict_mask = (macro_sign != micro_sign) & (abs(mega_df['macro_conviction_4h']) > 0.5)
                    mega_df.loc[conflict_mask, 'predicted_rank'] *= 0.7
                    
                    if conflict_mask.any():
                        logger.info(f"🛡️ Macro Safeguard: Dampened {conflict_mask.sum()} conflicting signals.")

                logger.info("✅ Ensemble Inference Complete.")
                ml_success = True
            except Exception as e:
                logger.error(f"⚠️ Ensemble Inference Failed: {e}")
                
        if not ml_success:
            logger.warning("⚠️ Falling back to simple Momentum Rank (RSI + MACD).")
            mega_df['predicted_rank'] = (mega_df['rank_rsi'] + mega_df['rank_macd']) / 2

        # 6. 🔪 SORT AND SLICE (Market Neutral Basket with Hysteresis)
        BASKET_N = 5
        HYSTERESIS_FACTOR = 4.0
        buffer_n = int(BASKET_N * HYSTERESIS_FACTOR)  # keeps coins if they stay in the buffer zone
        
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
                        risk.execute_logic(target_coin, signal, timeframe, atr_pct, portfolio, user_state, open_orders, override_usd=risk_parity_usd)
                    else:
                        logger.warning(f"🛑 SAFETY REJECTION: {target_coin} failed pre-trade checks.")
                else:
                    # We already have it. Just sync Trailing Stop logic.
                    risk.sync_trailing_stop(target_coin, atr_pct, portfolio, open_orders)
                
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
