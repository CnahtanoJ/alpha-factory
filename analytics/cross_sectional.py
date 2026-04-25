import pandas as pd
import numpy as np
import sqlite3
import os
import ta
import lightgbm as lgb
from bot.strategies import STRATEGY_CONFIG
from data_pipeline.database import DB_PATH
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
import pickle
import json
from scipy.stats import spearmanr
def fetch_and_merge_symbol_data(symbol, conn):
    """
    Fetches ohlcv, index_ohlcv, symbol_metrics, and funding_rate for a single symbol
    and merges them into one DataFrame using forward fill.
    """
    # 1. Fetch OHLCV
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol = ? ORDER BY timestamp", conn, params=(symbol,))
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
    df['oi_zscore'] = (df['sum_open_interest'] - df['sum_open_interest'].rolling(50).mean()) / df['sum_open_interest'].rolling(50).std()
    df['funding_delta'] = df['last_funding_rate'].diff()
    
    # Market Correlation (Fixed to use BTC like Phase 2)
    if btc_df is not None and not btc_df.empty:
        df = pd.merge_asof(df, btc_df, on='timestamp', direction='backward')
        df['corr_to_index'] = df['close'].pct_change().rolling(20).corr(df['btc_close'].pct_change())
    else:
        df['corr_to_index'] = 0
    
    # Cyclic Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 6)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 6)

    
    # Forward Return (Target Calculation)
    df['fwd_return'] = df['close'].shift(-6) / df['close'] - 1

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

def build_mega_dataframe():
    """
    Builds the massive cross-sectional dataframe.
    """
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM ohlcv", conn)['symbol'].tolist()
    
    # Pre-fetch BTC data to use as the true "Market Index" for correlation (aligns with Phase 2)
    btc_df = pd.read_sql_query("SELECT timestamp, close as btc_close FROM ohlcv WHERE symbol IN ('BTC/USDT', 'BTCUSDT', 'BTC') ORDER BY timestamp", conn)
    if not btc_df.empty:
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    
    all_dfs = []
    print(f"Building Mega-DataFrame for {len(symbols)} symbols...")
    for sym in symbols:
        df = fetch_and_merge_symbol_data(sym, conn)
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
    continuous_features = ['rsi', 'macd', 'volatility_20', 'basis_pct', 'oi_zscore', 'funding_delta', 'sum_toptrader_long_short_ratio', 'corr_to_index']
    
    for col in continuous_features:
        mega_df[f'rank_{col}'] = mega_df.groupby('timestamp')[col].rank(pct=True)
        
    # Rank Target
    mega_df['target_rank'] = mega_df.groupby('timestamp')['fwd_return'].rank(pct=True)
    
    # Drop rows where target or critical features are NaN
    mega_df = mega_df.dropna(subset=['target_rank', 'rank_rsi'])
    
    return mega_df

def train_cross_sectional_lgbm(mega_df):
    """
    Phase 3: Chronological Walk-Forward Split, LightGBM Training, and S3 Export.
    """
    print("Preparing for LightGBM Training...")
    
    # Sort chronologically for Walk-Forward split
    mega_df = mega_df.sort_values('timestamp')
    
    # Define features
    continuous_features = ['rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct', 'rank_oi_zscore', 'rank_funding_delta', 'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index']
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
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples...")
    
    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1
    }
    
    # Train Model
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    print(f"Best Validation RMSE: {model.best_score['valid']['rmse']:.4f}")
    
    # Calculate Spearman Correlation for Validation Set
    val_preds = model.predict(X_val)
    spearman_corr, p_value = spearmanr(val_preds, y_val)
    print(f"Validation Spearman Correlation: {spearman_corr:.4f} (p-value: {p_value:.4f})")
    
    # Save Model Locally
    import datetime
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cross_sectional_lgbm.txt')
    meta_path = os.path.join(model_dir, 'cross_sectional_lgbm_meta.json')
    
    versioned_model_path = os.path.join(model_dir, f'cross_sectional_lgbm_{timestamp_str}.txt')
    
    model.save_model(model_path)
    model.save_model(versioned_model_path)
    print(f"Model saved locally to {model_path} and {versioned_model_path}")
    
    with open(meta_path, 'w') as f:
        json.dump({
            'validation_rmse': float(model.best_score['valid']['rmse']),
            'validation_spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(p_value),
            'timestamp': timestamp_str
        }, f, indent=4)
    print(f"Metadata saved locally to {meta_path}")
    
    # S3 Export
    try:
        import boto3
        from bot.config import AWS_BUCKET
        s3_client = boto3.client('s3')
        
        # Upload the main file that the bot reads
        s3_client.upload_file(model_path, AWS_BUCKET, 'models/cross_sectional_lgbm.txt')
        s3_client.upload_file(meta_path, AWS_BUCKET, 'models/cross_sectional_lgbm_meta.json')
        
        # Upload the versioned file for history/rollback
        s3_client.upload_file(versioned_model_path, AWS_BUCKET, f'models/history/cross_sectional_lgbm_{timestamp_str}.txt')
            
        print(f"✅ Model and Metadata uploaded to S3 bucket '{AWS_BUCKET}'. History saved.")
            
    except ImportError:
        print("⚠️ boto3 not installed. Skipping S3 upload.")
    except Exception as e:
        print(f"⚠️ S3 upload failed: {e}. Model saved locally only.")
        
    return model, features

if __name__ == "__main__":
    mega_df = build_mega_dataframe()
    if not mega_df.empty:
        model, features = train_cross_sectional_lgbm(mega_df)
