import os
import json
import sqlite3
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
import gc
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from data_pipeline.database import DB_PATH

# --- CANONICAL FEATURE SETS (Single Source of Truth) ---
RAW_CONTINUOUS = [
    'rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_usd', 'funding_rate', 
    'sum_toptrader_long_short_ratio', 'corr_to_index',
    'oi_delta_4', 'funding_delta_4', 'sum_toptrader_ls_delta_4',
    'distance_from_ema_50', 'volatility_zscore', 
    'volume_zscore', 'relative_strength_btc',
    'cvd_slope_5', 'price_cvd_divergence', 'sentiment_divergence',
    'trend_convergence', 'bbw_squeeze', 'funding_basis_divergence',
    'vol_volatility_ratio', 'market_beta', 'rsi_divergence', 'vpt_slope',
    'range_expansion', 'net_taker_volume_zscore',
    # NEW PHASE 2 DELTA FEATURES
    'rsi_delta_4', 'macd_delta_4', 'volatility_delta_4', 'volume_delta_4',
    'oi_change_pct_12', 'funding_acceleration'
]

FULL_CONTINUOUS = [f'rank_{f}' for f in RAW_CONTINUOUS]

PRUNED_CONTINUOUS = [
    'rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_usd', 
    'rank_funding_rate', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index',
    'rank_oi_delta_4', 'rank_distance_from_ema_50', 'rank_volatility_zscore', 
    'rank_volume_zscore', 'rank_relative_strength_btc',
    'rank_cvd_slope_5', 'rank_price_cvd_divergence', 'rank_sentiment_divergence',
    'rank_trend_convergence', 'rank_bbw_squeeze', 'rank_funding_basis_divergence',
    'rank_vol_volatility_ratio', 'rank_market_beta', 'rank_rsi_divergence', 'rank_vpt_slope'
]

FULL_TIME = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
PRUNED_TIME = ['day_sin', 'day_cos']

# NEW PHASE 2 MARKET REGIME FEATURES (Unranked)
MARKET_REGIME_BASE = ['btc_ret_24', 'btc_volatility_24', 'market_breadth', 'regime_score']
MARKET_REGIME_MACRO = ['macro_conviction_4h']  # Only for 15m/1h (injected from 4h model)

def get_feature_names(timeframe='15m'):
    """Returns the canonical feature list used by the LightGBM model for a given timeframe."""
    if timeframe == '4h':
        cont, time = PRUNED_CONTINUOUS, PRUNED_TIME
    else:
        cont, time = FULL_CONTINUOUS, FULL_TIME
    
    # Include Market Regime features
    regime = MARKET_REGIME_BASE[:]
    if timeframe != '4h':
        regime += MARKET_REGIME_MACRO
    
    return cont + regime, time

try:
    import boto3
    from bot.config import AWS_BUCKET
except ImportError:
    boto3 = None
    AWS_BUCKET = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def get_params_path(timeframe):
    return os.path.join(MODEL_DIR, f'best_params_{timeframe}.json')

def get_fwd_return_bars(timeframe):
    """
    Returns the target prediction horizon (in bars) for a given timeframe.
    """
    mapping = {
        '15m': 12, # 3 hours
        '1h': 12,  # 12 hours
        '4h': 12,  # 48 hours
    }
    return mapping.get(timeframe, 6) # Default to 6 if unknown

def fetch_and_merge_symbol_data(symbol, conn, timeframe='15m'):
    """
    Fetches ohlcv, index_ohlcv, symbol_metrics, and funding_rate for a single symbol
    and merges them into one DataFrame using forward fill.
    """
    # 1. Fetch OHLCV
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp", conn, params=(symbol, timeframe))
    if df.empty: return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('s')

    # 2. Fetch Index OHLCV (M-1 FIX: filter by timeframe to avoid cross-timeframe pollution)
    idx_df = pd.read_sql_query("SELECT timestamp as idx_ts, close as index_close FROM index_ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp", conn, params=(symbol, timeframe))
    if not idx_df.empty:
        idx_df['idx_ts'] = pd.to_datetime(idx_df['idx_ts'], unit='ms').dt.floor('s')
        df = pd.merge_asof(df, idx_df, left_on='timestamp', right_on='idx_ts', direction='backward', tolerance=pd.Timedelta(hours=2))
    else:
        df['index_close'] = df['close'] # Fallback

    # 3. Fetch Symbol Metrics (C-4 FIX: fetch sum_open_interest_value which is already in USD)
    metrics_df = pd.read_sql_query("SELECT timestamp as met_ts, sum_open_interest, sum_open_interest_value, sum_toptrader_long_short_ratio, sum_long_short_ratio FROM symbol_metrics WHERE symbol = ? ORDER BY timestamp", conn, params=(symbol,))
    if not metrics_df.empty:
        metrics_df['met_ts'] = pd.to_datetime(metrics_df['met_ts'], unit='ms').dt.floor('s')
        df = pd.merge_asof(df, metrics_df, left_on='timestamp', right_on='met_ts', direction='backward', tolerance=pd.Timedelta(hours=2))
    else:
        df['sum_open_interest'] = np.nan
        df['sum_open_interest_value'] = np.nan
        df['sum_toptrader_long_short_ratio'] = np.nan

    # 4. Fetch Funding Rate
    fund_df = pd.read_sql_query("SELECT calc_time, last_funding_rate FROM funding_rate WHERE symbol = ? ORDER BY calc_time", conn, params=(symbol,))
    if not fund_df.empty:
        fund_df['calc_time'] = pd.to_datetime(fund_df['calc_time'], unit='ms').dt.floor('s')
        df = pd.merge_asof(df, fund_df, left_on='timestamp', right_on='calc_time', direction='backward', tolerance=pd.Timedelta(hours=16))
    else:
        df['last_funding_rate'] = np.nan

    # 5. Fetch HTF Data (4h)
    htf_df = pd.read_sql_query("SELECT timestamp as htf_ts, close as htf_close FROM ohlcv WHERE symbol = ? AND timeframe = '4h' ORDER BY timestamp", conn, params=(symbol,))
    if not htf_df.empty:
        htf_df['ema_50_4h'] = htf_df['htf_close'].ewm(span=50, adjust=False).mean()
        htf_df['rsi_4h'] = RSIIndicator(close=htf_df['htf_close'], window=14).rsi()
        # Prevent Lookahead: A 4h candle that opens at 12:00 closes at 16:00.
        htf_df['htf_ts'] = pd.to_datetime(htf_df['htf_ts'], unit='ms').dt.floor('s') + pd.Timedelta(hours=4)
        htf_df['htf_ts'] = htf_df['htf_ts'].astype(df['timestamp'].dtype)
        df = pd.merge_asof(df, htf_df[['htf_ts', 'ema_50_4h', 'rsi_4h']], left_on='timestamp', right_on='htf_ts', direction='backward')
    else:
        df['ema_50_4h'] = np.nan
        df['rsi_4h'] = 50.0

    # C-3 FIX: Bounded forward fill — propagate lower-frequency data (metrics/funding)
    # within a safe window, but never across unfillable gaps
    fill_cols = ['index_close', 'sum_open_interest', 'sum_open_interest_value', 'sum_toptrader_long_short_ratio', 'last_funding_rate']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=8)  # Max 8 periods (2h for 15m candles)
    
    df['symbol'] = symbol
    return df

def add_time_series_features(df, btc_df=None):
    """
    Adds indicators, derivative fuel, and strategy signals to a single symbol's DataFrame.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Base Indicators
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    
    df['volatility_20'] = df['close'].rolling(window=20).std()
    
    # C-1 FIX: Add ATR% for backtester risk_parity weighting
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / df['close']
    
    # Derivative Fuel
    df['basis_pct'] = (df['close'] - df['index_close']) / df['index_close']
    # C-4 FIX: Use sum_open_interest_value (already in USD from Binance) to match
    # Hyperliquid's openInterest * oraclePx calculation in live inference
    df['oi_usd'] = df.get('sum_open_interest_value', df.get('sum_open_interest', np.nan) * df['close'])
    df['funding_rate'] = df['last_funding_rate']
    
    # Market Correlation (Fixed to use BTC like Phase 2)
    if btc_df is not None and not btc_df.empty:
        df = pd.merge_asof(df, btc_df, on='timestamp', direction='backward')
        df['corr_to_index'] = df['close'].pct_change().rolling(20).corr(df['btc_close'].pct_change())
    else:
        df['corr_to_index'] = 0
    
    # Cyclic Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)

    # --- NEW CONTINUOUS FEATURES ---
    
    # 1. Derivatives Velocity
    # 4-bar delta (1 hour for 15m candles)
    df['oi_delta_4'] = df['oi_usd'].pct_change(4)
    df['funding_delta_4'] = df['funding_rate'].diff(4)
    
    # Taker buy/sell ratio
    df['taker_buy_sell_ratio'] = df.get('sum_taker_long_short_vol_ratio', np.nan)
    
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
    if btc_df is not None and not btc_df.empty:
        df['ret_12'] = df['close'].pct_change(12)
        df['btc_ret_12'] = df['btc_close'].pct_change(12)
        df['relative_strength_btc'] = df['ret_12'] - df['btc_ret_12']
        
        # Phase 5: Market Beta
        asset_ret_1 = df['close'].pct_change()
        btc_ret_1 = df['btc_close'].pct_change()
        cov = asset_ret_1.rolling(20).cov(btc_ret_1)
        var = btc_ret_1.rolling(20).var()
        df['market_beta'] = (cov / (var + 1e-9)).fillna(0.0)
    else:
        df['relative_strength_btc'] = 0.0
        df['market_beta'] = 0.0
        
    # Sentiment Divergence (Whales vs Retail)
    top_trader = df.get('sum_toptrader_long_short_ratio', np.nan)
    global_retail = df.get('sum_long_short_ratio', top_trader)
    df['sentiment_divergence'] = top_trader - global_retail
        
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
    # Trend Convergence (Micro vs Macro)
    ema_50_slope = df['close'].ewm(span=50, adjust=False).mean().pct_change(5)
    ema_50_4h_slope = df.get('ema_50_4h', df['close']).pct_change(16) # 16 * 15m = 4h
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
    basis_z = (df['basis_pct'] - basis_100_mean) / (basis_100_std + 1e-9)
    
    df['funding_basis_divergence'] = (fund_z - basis_z).fillna(0.0)

    # 6. Phase 5 Features (Parity Fix)
    vol_ma_20 = df['volume'].rolling(20).mean()
    
    # Volume to Volatility Ratio (Matched to Bot)
    df['vol_volatility_ratio'] = (vol_ma_20 / (df['volatility_20'] + 1e-9)).fillna(0.0)
    
    # RSI Divergence Proxy (Matched to Bot: RSI Momentum vs Price Momentum)
    df['rsi_divergence'] = df['rsi'].diff(5) - df['close'].pct_change(5)
    
    # VPT Slope (Volume Price Trend)
    vpt = (df['volume'] * df['close'].pct_change()).cumsum()
    df['vpt_slope'] = vpt.diff(5) / (vol_ma_20 + 1e-9)
    
    # Range Expansion
    df['range_expansion'] = (df['high'] - df['low']) / (df['volatility_20'] + 1e-9)
    
    # Top Trader Delta
    df['sum_toptrader_ls_delta_4'] = df['sum_toptrader_long_short_ratio'].diff(4)
    
    # Net Taker Volume Z-score Proxy (from CVD delta)
    df['net_taker_volume_zscore'] = (df['cvd_slope_5'] / (vol_ma_20 + 1e-9)).fillna(0.0)
    

    # --- NEW PHASE 2 DELTA FEATURES ---
    df['rsi_delta_4'] = df['rsi'].diff(4)
    df['macd_delta_4'] = df['macd'].diff(4)
    df['volatility_delta_4'] = df['volatility_20'].diff(4)
    df['volume_delta_4'] = df['volume'].pct_change(4)
    df['oi_change_pct_12'] = df['oi_usd'].pct_change(12)
    df['funding_acceleration'] = df['funding_delta_4'].diff(1)
    
    # Forward Return (Target Calculation)
    # Dynamic horizon based on timeframe
    tf_val = df['timeframe_name'].iloc[0] if 'timeframe_name' in df.columns else '15m'
    fwd_bars = get_fwd_return_bars(tf_val)
    
    # Safely compute fwd_return avoiding gaps
    tf_timedelta = df['timestamp'].diff().mode()[0]
    expected_delta = tf_timedelta * fwd_bars
    
    shifted_close = df['close'].shift(-fwd_bars)
    shifted_ts = df['timestamp'].shift(-fwd_bars)
    
    valid_mask = (shifted_ts - df['timestamp']) == expected_delta
    raw_fwd = shifted_close / df['close'] - 1
    
    # PHASE 3 FIX: Risk-Adjusted Target (Return / ATR)
    # We store it in a separate column to avoid corrupting the backtester's price data
    df['fwd_return'] = np.where(valid_mask, raw_fwd, np.nan)
    df['risk_adj_ret'] = np.where(valid_mask, raw_fwd / (df['atr_pct'] + 1e-9), np.nan)
    
    # Path-dependent metrics for simulator TP/SL
    fwd_highs = df['high'].iloc[::-1].rolling(window=fwd_bars, min_periods=1).max().iloc[::-1].shift(-1)
    fwd_lows = df['low'].iloc[::-1].rolling(window=fwd_bars, min_periods=1).min().iloc[::-1].shift(-1)
    
    df['fwd_max_ret'] = np.where(valid_mask, fwd_highs / df['close'] - 1, np.nan)
    df['fwd_min_ret'] = np.where(valid_mask, fwd_lows / df['close'] - 1, np.nan)
            
    return df

def inject_macro_conviction(mega_df, timeframe):
    """Inject 4h model predictions as a macro conviction feature for 15m/1h models."""
    if timeframe == '4h':
        mega_df['macro_conviction_4h'] = 0.0
        return mega_df
    
    # 1. Check if we are in "Live/OOS Mode" or "Training Mode"
    unique_ts = sorted(mega_df['timestamp'].unique())
    is_training = len(unique_ts) > 500 # Threshold to distinguish live/OOS from full training
    
    if not is_training:
        # LIVE/OOS MODE: Use the existing production 4h model
        lgb_path = os.path.join(MODEL_DIR, 'cross_sectional_lgbm_4h.txt')
        if not os.path.exists(lgb_path):
            mega_df['macro_conviction_4h'] = 0.0
            return mega_df
        
        print(f"🔮 Live/OOS Mode: Injecting Macro Conviction using production 4h model...")
        model_4h = lgb.Booster(model_file=lgb_path)
        mega_4h = build_mega_dataframe('4h') # This build is small in live mode
        if mega_4h.empty:
            mega_df['macro_conviction_4h'] = 0.0
            return mega_df
        
        X_4h = mega_4h[model_4h.feature_name()].fillna(0.0)
        pred_df = mega_4h[['timestamp', 'symbol']].copy()
        pred_df['macro_conviction_4h'] = model_4h.predict(X_4h).astype(np.float32)
        
        return _merge_macro_preds(mega_df, pred_df)

    # 2. TRAINING MODE: 3-Fold Walk-Forward to prevent Leakage
    print(f"🛡️ Training Mode: Implementing 3-Fold Walk-Forward OOF Injection (Leak-Free)...")
    
    # We need the full 4h data to train our "mini" models
    mega_4h = build_mega_dataframe('4h')
    if mega_4h.empty:
        mega_df['macro_conviction_4h'] = 0.0
        return mega_df
        
    # Split timestamps into 3 chunks
    ts_splits = np.array_split(unique_ts, 3)
    p1_ts, p2_ts, p3_ts = ts_splits[0], ts_splits[1], ts_splits[2]
    
    # Containers for predictions
    all_macro_preds = []
    
    # FOLD 1 (Part 1): Train on P1, Predict on P1 (Self-prediction to avoid zero-fuel cold start)
    print("   [Fold 1/3] Warming up: Self-prediction for Part 1...")
    df_train_p1 = mega_4h[mega_4h['timestamp'].isin(p1_ts)]
    model_p1, feats_p1 = train_mini_4h(df_train_p1)
    if model_p1:
        p1_preds = df_train_p1[['timestamp', 'symbol']].copy()
        p1_preds['macro_conviction_4h'] = model_p1.predict(df_train_p1[feats_p1].fillna(0.0))
        all_macro_preds.append(p1_preds)
    
    # FOLD 2 (Part 2): Train on P1, Predict on P2
    print("   [Fold 2/3] Training on 33%, Predicting next 33%...")
    df_train_f1 = mega_4h[mega_4h['timestamp'].isin(p1_ts)]
    df_test_f1 = mega_4h[mega_4h['timestamp'].isin(p2_ts)]
    
    model_f1, feats_f1 = train_mini_4h(df_train_f1)
    if model_f1:
        f1_preds = df_test_f1[['timestamp', 'symbol']].copy()
        f1_preds['macro_conviction_4h'] = model_f1.predict(df_test_f1[feats_f1].fillna(0.0))
        all_macro_preds.append(f1_preds)
    
    # FOLD 3 (Part 3): Train on P1+P2, Predict on P3
    print("   [Fold 3/3] Training on 66%, Predicting remaining 34%...")
    df_train_f2 = mega_4h[mega_4h['timestamp'].isin(np.concatenate([p1_ts, p2_ts]))]
    df_test_f2 = mega_4h[mega_4h['timestamp'].isin(p3_ts)]
    
    model_f2, feats_f2 = train_mini_4h(df_train_f2)
    if model_f2:
        f2_preds = df_test_f2[['timestamp', 'symbol']].copy()
        f2_preds['macro_conviction_4h'] = model_f2.predict(df_test_f2[feats_f2].fillna(0.0))
        all_macro_preds.append(f2_preds)
        
    if not all_macro_preds:
        print("   ⚠️ No macro predictions generated in any fold. Using neutral.")
        mega_df['macro_conviction_4h'] = 0.0
        return mega_df

    full_pred_df = pd.concat(all_macro_preds, ignore_index=True)
    
    del mega_4h
    gc.collect()
    
    return _merge_macro_preds(mega_df, full_pred_df)

def train_mini_4h(df):
    """Internal helper to train a fast, non-optimized 4h model for OOF injection."""
    from analytics.cross_sectional import prepare_training_data
    if len(df) < 1000: return None, []
    try:
        X, y, _, _, features = prepare_training_data(df, timeframe='4h')
        if X is None or X.empty: return None, []
        params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'learning_rate': 0.1, 'num_leaves': 31}
        ds = lgb.Dataset(X, label=y)
        model = lgb.train(params, ds, num_boost_round=50)
        return model, features
    except Exception as e:
        print(f"   ⚠️ Mini-model training failed: {e}")
        return None, []

def _merge_macro_preds(mega_df, pred_df):
    """Helper to merge macro predictions into the target dataframe symbol-by-symbol."""
    print("   Merging leak-free macro predictions...")
    pred_df = pred_df.sort_values('timestamp')
    merged_parts = []
    for sym, sym_df in mega_df.groupby('symbol', sort=False):
        sym_preds = pred_df[pred_df['symbol'] == sym][['timestamp', 'macro_conviction_4h']]
        if sym_preds.empty:
            sym_df = sym_df.copy()
            sym_df['macro_conviction_4h'] = 0.0
        else:
            sym_df = sym_df.sort_values('timestamp')
            sym_df = pd.merge_asof(sym_df, sym_preds, on='timestamp', direction='backward')
        merged_parts.append(sym_df)
    
    mega_df = pd.concat(merged_parts, ignore_index=True)
    mega_df['macro_conviction_4h'] = mega_df['macro_conviction_4h'].fillna(0.0)
    
    n_filled = (mega_df['macro_conviction_4h'] != 0.0).sum()
    print(f"   ✅ Macro conviction injected for {n_filled:,} / {len(mega_df):,} rows")
    
    del merged_parts, pred_df
    gc.collect()
    
    mega_df = mega_df.sort_values('timestamp').reset_index(drop=True)
    return mega_df

def build_mega_dataframe(timeframe='15m'):
    """
    Builds the massive cross-sectional dataframe.
    """
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ?", conn, params=(timeframe,))['symbol'].tolist()
    
    # Pre-fetch BTC data to use as the true "Market Index" for correlation (aligns with Phase 2)
    btc_df = pd.read_sql_query("SELECT timestamp, close as btc_close FROM ohlcv WHERE symbol IN ('BTC/USDT', 'BTCUSDT', 'BTC') AND timeframe = ? ORDER BY timestamp", conn, params=(timeframe,))
    if not btc_df.empty:
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    
    all_dfs = []
    print(f"Building Mega-DataFrame for {len(symbols)} symbols...")
    for sym in symbols:
        df = fetch_and_merge_symbol_data(sym, conn, timeframe=timeframe)
        if df.empty: continue
        df['timeframe_name'] = timeframe # Pass timeframe context for target calculation
        df = add_time_series_features(df, btc_df)
        
        # P3-4: Extreme Memory Optimization (16GB RAM Target)
        # 1. Early Drop: Remove useless rows before they enter the mega list
        df = df.dropna(subset=['fwd_return', 'rsi'])
        
        # 2. Downcast: Cut numerical memory in half (Float64 -> Float32)
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            # Keep timestamp as int64 to avoid overflow
            if col != 'timestamp':
                df[col] = df[col].astype(np.int32)
                
        if not df.empty:
            all_dfs.append(df)
        
    conn.close()
    
    if not all_dfs:
        print("No data found to build mega dataframe.")
        return pd.DataFrame()
        
    mega_df = pd.concat(all_dfs, ignore_index=True)
    
    # NEW PHASE 2: Market Regime Features
    print("Injecting Market Regime features...")
    if not btc_df.empty:
        btc_df = btc_df.sort_values('timestamp')
        btc_df['btc_ret_24'] = btc_df['btc_close'].pct_change(24)
        btc_df['btc_volatility_24'] = btc_df['btc_close'].pct_change().rolling(24).std()
        # Merge regime features into mega_df
        mega_df = pd.merge_asof(mega_df.sort_values('timestamp'), btc_df[['timestamp', 'btc_ret_24', 'btc_volatility_24']], on='timestamp', direction='backward')
    else:
        mega_df['btc_ret_24'] = 0.0
        mega_df['btc_volatility_24'] = 0.0
        
    # Market Breadth: % of assets with positive 12-bar returns at this timestamp
    if 'ret_12' in mega_df.columns:
        mega_df['market_breadth'] = mega_df.groupby('timestamp')['ret_12'].transform(lambda x: (x > 0).mean())
    else:
        mega_df['market_breadth'] = 0.5
    
    # PHASE 4: Regime Score — trend strength indicator
    mega_df['regime_score'] = abs(mega_df['btc_ret_24']) / (mega_df['btc_volatility_24'] + 1e-9)
    
    # PHASE 3 FIX: Outlier Winsorization
    # Clip extreme 1%/99% outliers for critical columns to protect the linear model
    # Optimized for speed (uses Cython-backed quantile transform)
    print("Applying Outlier Winsorization (1%/99%)...")
    critical_cols = ['fwd_return', 'funding_rate', 'volume_zscore', 'volatility_zscore']
    for col in critical_cols:
        if col in mega_df.columns:
            q_01 = mega_df.groupby('timestamp')[col].transform('quantile', 0.01)
            q_99 = mega_df.groupby('timestamp')[col].transform('quantile', 0.99)
            mega_df[col] = mega_df[col].clip(lower=q_01, upper=q_99)

    # PHASE 4: Macro Conviction Injection (4h → 15m/1h)
    mega_df = inject_macro_conviction(mega_df, timeframe)

    # P3-5: Explicit Garbage Collection
    print(f"Mega-DataFrame constructed: {len(mega_df)} rows. Purging intermediate memory...")
    del all_dfs
    gc.collect()
    
    # Cross-Sectional Ranking
    print("Applying cross-sectional ranking...")
    
    # Columns to rank
    continuous_features = RAW_CONTINUOUS
    
    for col in continuous_features:
        if col in mega_df.columns:
            mega_df[f'rank_{col}'] = mega_df.groupby('timestamp')[col].rank(pct=True)
        
    # PHASE 3.5: Magnitude-Aware Hybrid Target (Optimized Vectorized version)
    print("Calculating Magnitude-Aware Hybrid Target (50/50 Raw/Risk-Adj)...")
    g = mega_df.groupby('timestamp')
    z_raw = (mega_df['fwd_return'] - g['fwd_return'].transform('mean')) / (g['fwd_return'].transform('std') + 1e-9)
    z_risk = (mega_df['risk_adj_ret'] - g['risk_adj_ret'].transform('mean')) / (g['risk_adj_ret'].transform('std') + 1e-9)
    mega_df['target_magnitude'] = 0.5 * z_raw + 0.5 * z_risk
    
    # We also keep target_rank for the Spearman evaluation in the orchestrator
    mega_df['target_rank'] = g['target_magnitude'].rank(pct=True)
    
    # Drop rows where target or critical features are NaN
    mega_df = mega_df.dropna(subset=['target_magnitude', 'rank_rsi', 'rank_oi_delta_4'])
    
    return mega_df

def prepare_training_data(mega_df, timeframe='15m'):
    """
    Phase 3: Chronological Walk-Forward Split, LightGBM Training, and S3 Export.
    Selects feature sets based on timeframe (Asymmetric Pruning).
    """
    print(f"Preparing training data for {timeframe}...")
    
    # Sort chronologically for Walk-Forward split
    mega_df = mega_df.sort_values('timestamp')
    
    continuous, time_features = get_feature_names(timeframe)
    features = continuous + time_features
    
    if timeframe == '4h':
        print(f"✂️ Using PRUNED feature set for 4h ({len(features)} features)")
    else:
        print(f"🔥 Using FULL feature set for {timeframe} ({len(features)} features)")
    
    X = mega_df[features].copy()
    y = mega_df['target_magnitude']
    
    # PHASE 3 FIX: Fill NaNs for Linear Model (Ridge)
    # Tree models handle NaN, but Ridge requires finite values.
    # Since most features are ranks [0, 1], 0.5 is a safe neutral value.
    X = X.fillna(0.5)
    
    # Walk-Forward Split (85% Train, 15% Validation)
    split_idx = int(len(mega_df) * 0.85)
    
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    
    return X_train, y_train, X_val, y_val, features

def optimize_lgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50, timeframe='15m'):
    """
    Uses Optuna to find the best hyperparameters for LightGBM.
    """
    print(f"🚀 Starting Optuna HPO with {n_trials} trials for {timeframe}...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=25)]
        )
        
        return model.best_score['valid_0']['rmse']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Best Trial: RMSE {study.best_value:.4f}")
    
    # Persistent Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    params_path = get_params_path(timeframe)
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_params

def train_ensemble_models(mega_df, optimized_params=None, timeframe='15m'):
    """
    Train an ensemble of LightGBM + XGBoost + Ridge for maximum robustness.
    """
    X_train, y_train, X_val, y_val, features = prepare_training_data(mega_df, timeframe=timeframe)
    
    # 1. Train LightGBM
    print("🧠 Training LightGBM...")
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    
    # Load persisted params if none provided
    params_path = get_params_path(timeframe)
    if optimized_params is None and os.path.exists(params_path):
        print(f"📂 Loading persisted HPO parameters from {params_path}")
        with open(params_path, 'r') as f:
            optimized_params = json.load(f)

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        **(optimized_params if optimized_params else {'learning_rate': 0.05, 'num_leaves': 31})
    }
    
    model_lgb = lgb.train(
        lgb_params,
        train_data_lgb,
        num_boost_round=1000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 2. Train XGBoost
    print("🧠 Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.05, 
        n_jobs=-1,
        early_stopping_rounds=20
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 3. Train Ridge Regression (Phase 3: Linear Ensemble)
    print("🧠 Training Ridge Regression (Linear Perspective)...")
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    
    # 4. Evaluate Ensemble (Weighted Average)
    lgb_preds = model_lgb.predict(X_val)
    xgb_preds = model_xgb.predict(X_val)
    ridge_preds = model_ridge.predict(X_val)
    
    # Weighted Ensemble: 40% Trees, 20% Linear
    ensemble_preds = (0.4 * lgb_preds) + (0.4 * xgb_preds) + (0.2 * ridge_preds)
    
    spearman_corr, p_value = spearmanr(ensemble_preds, y_val)
    rmse = np.sqrt(np.mean((ensemble_preds - y_val)**2))
    print(f"✅ Validation Ensemble Spearman Correlation: {spearman_corr:.4f}")
    
    return (model_lgb, model_xgb, model_ridge), features, spearman_corr, p_value, rmse

def upload_ensemble_to_s3(timeframe='15m'):
    """Explicitly upload the trained ensemble to S3 after validation."""
    if not boto3:
        print("⚠️ boto3 not available. Skipping S3 upload.")
        return False
        
    lgb_path = os.path.join(MODEL_DIR, f'cross_sectional_lgbm_{timeframe}.txt')
    xgb_path = os.path.join(MODEL_DIR, f'cross_sectional_xgboost_{timeframe}.json')
    ridge_path = os.path.join(MODEL_DIR, f'cross_sectional_ridge_{timeframe}.joblib')
    meta_path = os.path.join(MODEL_DIR, f'cross_sectional_lgbm_{timeframe}_meta.json')
    
    s3 = boto3.client('s3')
    try:
        s3.upload_file(lgb_path, AWS_BUCKET, f'models/cross_sectional_lgbm_{timeframe}.txt')
        s3.upload_file(xgb_path, AWS_BUCKET, f'models/cross_sectional_xgboost_{timeframe}.json')
        s3.upload_file(ridge_path, AWS_BUCKET, f'models/cross_sectional_ridge_{timeframe}.joblib')
        s3.upload_file(meta_path, AWS_BUCKET, f'models/cross_sectional_lgbm_{timeframe}_meta.json')
        print(f"✅ Ensemble Models (LGBM+XGB+Ridge) uploaded to S3 bucket '{AWS_BUCKET}'.")
        return True
    except Exception as e:
        print(f"⚠️ S3 upload failed: {e}")
        return False

def train_cross_sectional_lgbm(mega_df, optimized_params=None, timeframe='15m', upload=True):
    """Main entry point for training and persisting the ensemble."""
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    lgb_path = os.path.join(MODEL_DIR, f'cross_sectional_lgbm_{timeframe}.txt')
    xgb_path = os.path.join(MODEL_DIR, f'cross_sectional_xgboost_{timeframe}.json')
    ridge_path = os.path.join(MODEL_DIR, f'cross_sectional_ridge_{timeframe}.joblib')
    meta_path = os.path.join(MODEL_DIR, f'cross_sectional_lgbm_{timeframe}_meta.json')
    
    ensemble, features, spearman_corr, p_value, rmse = train_ensemble_models(mega_df, optimized_params, timeframe=timeframe)
    model_lgb, model_xgb, model_ridge = ensemble
    
    # Feature Importance (LGBM + XGB)
    lgb_importance = dict(zip(features, model_lgb.feature_importance(importance_type='gain').tolist()))
    xgb_importance = dict(zip(features, [float(v) for v in model_xgb.feature_importances_]))
    
    total_lgb = sum(lgb_importance.values()) + 1e-9
    total_xgb = sum(xgb_importance.values()) + 1e-9
    
    combined_importance = []
    for f in features:
        combined_importance.append({
            'feature': f,
            'combined_importance': ((lgb_importance[f] / total_lgb) + (xgb_importance[f] / total_xgb)) / 2.0
        })
    combined_importance.sort(key=lambda x: x['combined_importance'], reverse=True)
    
    # Load actual parameters for metadata persistence
    actual_params = optimized_params
    if actual_params is None:
        params_path = get_params_path(timeframe)
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                actual_params = json.load(f)

    # Save Metadata
    with open(meta_path, 'w') as f:
        json.dump({
            'timestamp': timestamp_str,
            'is_ensemble': True,
            'best_params': actual_params,
            'validation_spearman': float(spearman_corr),
            'validation_rmse': float(rmse),
            'spearman_p_value': float(p_value),
            'feature_importance': combined_importance
        }, f, indent=4)
    
    # Save Models
    model_lgb.save_model(lgb_path)
    model_xgb.save_model(xgb_path)
    joblib.dump(model_ridge, ridge_path)
    
    # Upload to S3
    if upload and boto3:
        upload_ensemble_to_s3(timeframe)
            
    return model_lgb, features # Legacy return for some callers

if __name__ == "__main__":
    mega_df = build_mega_dataframe()
    if not mega_df.empty:
        model, features = train_cross_sectional_lgbm(mega_df)
