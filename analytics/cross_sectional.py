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
from ta.momentum import RSIIndicator
from ta.trend import MACD
from data_pipeline.database import DB_PATH

try:
    import boto3
    from bot.config import AWS_BUCKET
except ImportError:
    boto3 = None
    AWS_BUCKET = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PARAMS_PATH = os.path.join(MODEL_DIR, 'best_params.json')
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
    
    # 3. Relative Strength (vs BTC)
    if btc_df is not None and not btc_df.empty:
        df['ret_12'] = df['close'].pct_change(12)
        df['btc_ret_12'] = df['btc_close'].pct_change(12)
        df['relative_strength_btc'] = df['ret_12'] - df['btc_ret_12']
    else:
        df['relative_strength_btc'] = 0.0
        
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
    # Forward Return (Target Calculation)
    # Default: 6 bars (1.5h for 15m candles, 6h for 1h candles)
    FWD_RETURN_BARS = 6
    
    # Safely compute fwd_return avoiding gaps
    tf_timedelta = df['timestamp'].diff().mode()[0]
    expected_delta = tf_timedelta * FWD_RETURN_BARS
    
    shifted_close = df['close'].shift(-FWD_RETURN_BARS)
    shifted_ts = df['timestamp'].shift(-FWD_RETURN_BARS)
    
    valid_mask = (shifted_ts - df['timestamp']) == expected_delta
    df['fwd_return'] = np.where(valid_mask, shifted_close / df['close'] - 1, np.nan)
    
    # Path-dependent metrics for simulator TP/SL
    fwd_highs = df['high'].iloc[::-1].rolling(window=FWD_RETURN_BARS, min_periods=1).max().iloc[::-1].shift(-1)
    fwd_lows = df['low'].iloc[::-1].rolling(window=FWD_RETURN_BARS, min_periods=1).min().iloc[::-1].shift(-1)
    
    df['fwd_max_ret'] = np.where(valid_mask, fwd_highs / df['close'] - 1, np.nan)
    df['fwd_min_ret'] = np.where(valid_mask, fwd_lows / df['close'] - 1, np.nan)
            
    return df

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
    
    # P3-5: Explicit Garbage Collection
    print(f"Mega-DataFrame constructed: {len(mega_df)} rows. Purging intermediate memory...")
    del all_dfs
    gc.collect()
    
    # Cross-Sectional Ranking
    print("Applying cross-sectional ranking...")
    
    # Columns to rank
    continuous_features = [
        'rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_usd', 'funding_rate', 
        'sum_toptrader_long_short_ratio', 'corr_to_index',
        'oi_delta_4', 'funding_delta_4', 'taker_buy_sell_ratio', 
        'distance_from_ema_50', 'volatility_zscore', 'volume_zscore', 'relative_strength_btc',
        'cvd_slope_5', 'price_cvd_divergence', 'sentiment_divergence',
        'trend_convergence', 'bbw_squeeze', 'funding_basis_divergence'
    ]
    
    for col in continuous_features:
        if col in mega_df.columns:
            mega_df[f'rank_{col}'] = mega_df.groupby('timestamp')[col].rank(pct=True)
        
    # Rank Target
    mega_df['target_rank'] = mega_df.groupby('timestamp')['fwd_return'].rank(pct=True)
    
    # Drop rows where target or critical features are NaN
    mega_df = mega_df.dropna(subset=['target_rank', 'rank_rsi', 'rank_oi_delta_4'])
    
    return mega_df

def prepare_training_data(mega_df):
    """
    Phase 3: Chronological Walk-Forward Split, LightGBM Training, and S3 Export.
    """
    print("Preparing for LightGBM Training...")
    
    # Sort chronologically for Walk-Forward split
    mega_df = mega_df.sort_values('timestamp')
    
    # Define features
    continuous_features = [
        'rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_usd', 
        'rank_funding_rate', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index',
        'rank_oi_delta_4', 'rank_funding_delta_4', 'rank_taker_buy_sell_ratio',
        'rank_distance_from_ema_50', 'rank_volatility_zscore', 'rank_volume_zscore', 'rank_relative_strength_btc',
        'rank_cvd_slope_5', 'rank_price_cvd_divergence', 'rank_sentiment_divergence',
        'rank_trend_convergence', 'rank_bbw_squeeze', 'rank_funding_basis_divergence'
    ]
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = continuous_features + time_features
    
    X = mega_df[features]
    y = mega_df['target_rank']
    
    # Walk-Forward Split (85% Train, 15% Validation)
    split_idx = int(len(mega_df) * 0.85)
    
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    
    return X_train, y_train, X_val, y_val, features

def optimize_lgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Uses Optuna to find the best hyperparameters for LightGBM.
    """
    print(f"🚀 Starting Optuna HPO with {n_trials} trials...")
    
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
    with open(PARAMS_PATH, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_params

def train_ensemble_models(mega_df, optimized_params=None):
    """
    Phase 4: Train an ensemble of LightGBM + XGBoost for maximum robustness.
    """
    X_train, y_train, X_val, y_val, features = prepare_training_data(mega_df)
    
    # 1. Train LightGBM
    print("🧠 Training LightGBM...")
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    
    # Load persisted params if none provided
    if optimized_params is None and os.path.exists(PARAMS_PATH):
        print(f"📂 Loading persisted HPO parameters from {PARAMS_PATH}")
        with open(PARAMS_PATH, 'r') as f:
            optimized_params = json.load(f)

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        **(optimized_params if optimized_params else {'learning_rate': 0.05, 'num_leaves': 31})
    }
    
    lgb_model = lgb.train(
        lgb_params,
        train_data_lgb,
        num_boost_round=1000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 2. Train XGBoost
    print("🧠 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 3. Evaluate Ensemble (Averaging Ranks)
    lgb_preds = lgb_model.predict(X_val)
    xgb_preds = xgb_model.predict(X_val)
    ensemble_preds = (lgb_preds + xgb_preds) / 2.0
    
    spearman_corr, p_value = spearmanr(ensemble_preds, y_val)
    print(f"Validation Ensemble Spearman Correlation: {spearman_corr:.4f}")
    
    # Save Models Locally
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    lgb_path = os.path.join(model_dir, 'cross_sectional_lgbm.txt')
    xgb_path = os.path.join(model_dir, 'cross_sectional_xgboost.json')
    meta_path = os.path.join(model_dir, 'cross_sectional_lgbm_meta.json') # Kept as lgbm_meta for legacy compatibility
    
    lgb_model.save_model(lgb_path)
    xgb_model.save_model(xgb_path)
    
    # Save versioned
    lgb_model.save_model(os.path.join(model_dir, f'lgbm_{timestamp_str}.txt'))
    xgb_model.save_model(os.path.join(model_dir, f'xgboost_{timestamp_str}.json'))
    
    with open(meta_path, 'w') as f:
        json.dump({
            'validation_spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(p_value),
            'timestamp': timestamp_str,
            'is_ensemble': True
        }, f, indent=4)
    
    # S3 Export
    if boto3:
        try:
            s3_client = boto3.client('s3')
            s3_client.upload_file(lgb_path, AWS_BUCKET, 'models/cross_sectional_lgbm.txt')
            s3_client.upload_file(xgb_path, AWS_BUCKET, 'models/cross_sectional_xgboost.json')
            s3_client.upload_file(meta_path, AWS_BUCKET, 'models/cross_sectional_lgbm_meta.json')
            print(f"✅ Ensemble Models and Metadata uploaded to S3 bucket '{AWS_BUCKET}'.")
        except Exception as e:
            print(f"⚠️ S3 upload failed: {e}")
            
    return (lgb_model, xgb_model), features

def train_cross_sectional_lgbm(mega_df, optimized_params=None):
    """Legacy wrapper for weekly cycle compatibility."""
    ensemble, features = train_ensemble_models(mega_df, optimized_params)
    return ensemble[0], features # Return LightGBM for Step 3/4 which currently only support one model

if __name__ == "__main__":
    mega_df = build_mega_dataframe()
    if not mega_df.empty:
        model, features = train_cross_sectional_lgbm(mega_df)
