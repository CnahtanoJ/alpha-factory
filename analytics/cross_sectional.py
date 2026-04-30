import os
import json
import sqlite3
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
from scipy.stats import spearmanr
from ta.momentum import RSIIndicator
from ta.trend import MACD
from bot.strategies import STRATEGY_CONFIG
from data_pipeline.database import DB_PATH

try:
    import boto3
    from bot.config import AWS_BUCKET
except ImportError:
    boto3 = None
    AWS_BUCKET = None
def fetch_and_merge_symbol_data(symbol, conn, timeframe='15m'):
    """
    Fetches ohlcv, index_ohlcv, symbol_metrics, and funding_rate for a single symbol
    and merges them into one DataFrame using forward fill.
    """
    # 1. Fetch OHLCV
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp", conn, params=(symbol, timeframe))
    if df.empty: return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 2. Fetch Index OHLCV
    idx_df = pd.read_sql_query("SELECT timestamp as idx_ts, close as index_close FROM index_ohlcv WHERE symbol = ? ORDER BY timestamp", conn, params=(symbol,))
    if not idx_df.empty:
        idx_df['idx_ts'] = pd.to_datetime(idx_df['idx_ts'], unit='ms')
        df = pd.merge_asof(df, idx_df, left_on='timestamp', right_on='idx_ts', direction='backward')
    else:
        df['index_close'] = df['close'] # Fallback

    # 3. Fetch Symbol Metrics
    metrics_df = pd.read_sql_query("SELECT timestamp as met_ts, sum_open_interest, sum_toptrader_long_short_ratio FROM symbol_metrics WHERE symbol = ? ORDER BY timestamp", conn, params=(symbol,))
    if not metrics_df.empty:
        metrics_df['met_ts'] = pd.to_datetime(metrics_df['met_ts'], unit='ms')
        df = pd.merge_asof(df, metrics_df, left_on='timestamp', right_on='met_ts', direction='backward')
    else:
        df['sum_open_interest'] = np.nan
        df['sum_toptrader_long_short_ratio'] = np.nan

    # 4. Fetch Funding Rate
    fund_df = pd.read_sql_query("SELECT calc_time, last_funding_rate FROM funding_rate WHERE symbol = ? ORDER BY calc_time", conn, params=(symbol,))
    if not fund_df.empty:
        fund_df['calc_time'] = pd.to_datetime(fund_df['calc_time'], unit='ms')
        df = pd.merge_asof(df, fund_df, left_on='timestamp', right_on='calc_time', direction='backward')
    else:
        df['last_funding_rate'] = np.nan

    # Forward fill the lower frequency data (metrics, funding) that might have been NaN before their first timestamp
    df = df.ffill()
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
    
    # Derivative Fuel
    df['basis_pct'] = (df['close'] - df['index_close']) / df['index_close']
    df['oi_usd'] = df['sum_open_interest'] * df['close']
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

    
    # Forward Return (Target Calculation)
    # Default: 6 bars (1.5h for 15m candles, 6h for 1h candles)
    FWD_RETURN_BARS = 6
    df['fwd_return'] = df['close'].shift(-FWD_RETURN_BARS) / df['close'] - 1

    # Strategy Loop
    # Setting use_htf=False since we don't have htf_trend calculated yet
    # Using default parameters for strategy initialization
    for strat_name, strat_info in STRATEGY_CONFIG.items():
        try:
            strat_class = strat_info['class']
            # We initialize with default parameters
            strat_instance = strat_class()
            col_name = f"sig_{strat_name}"
            df[col_name] = strat_instance.get_signal_column(df)
        except Exception as e:
            # Some strategies might need specific columns or fail, fallback to 0
            df[f"sig_{strat_name}"] = 0
            
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
        all_dfs.append(df)
        
    conn.close()
    
    if not all_dfs:
        print("No data found to build mega dataframe.")
        return pd.DataFrame()
        
    mega_df = pd.concat(all_dfs, ignore_index=True)
    
    # Cross-Sectional Ranking
    print("Applying cross-sectional ranking...")
    
    # Columns to rank
    continuous_features = ['rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_usd', 'funding_rate', 'sum_toptrader_long_short_ratio', 'corr_to_index']
    
    for col in continuous_features:
        mega_df[f'rank_{col}'] = mega_df.groupby('timestamp')[col].rank(pct=True)
        
    # Rank Target
    mega_df['target_rank'] = mega_df.groupby('timestamp')['fwd_return'].rank(pct=True)
    
    # Drop rows where target or critical features are NaN
    mega_df = mega_df.dropna(subset=['target_rank', 'rank_rsi'])
    
    return mega_df

def prepare_training_data(mega_df):
    """
    Phase 3: Chronological Walk-Forward Split, LightGBM Training, and S3 Export.
    """
    print("Preparing for LightGBM Training...")
    
    # Sort chronologically for Walk-Forward split
    mega_df = mega_df.sort_values('timestamp')
    
    # Define features
    continuous_features = ['rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_usd', 'rank_funding_rate', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index']
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    # Extract strategy signal columns dynamically
    strategy_features = [col for col in mega_df.columns if col.startswith('sig_')]
    
    features = continuous_features + time_features + strategy_features
    
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
        
        return model.best_score['valid']['rmse']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Best Trial: RMSE {study.best_value:.4f}")
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
