import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier

# Use centralized DB path from data_pipeline
from data_pipeline.database import DB_PATH
import sqlite3

def add_indicators(df):
    """
    Add standard technical indicators using the 'ta' library.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Standard Indicators
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = indicator_bb.bollinger_hband()
    df['bb_low'] = indicator_bb.bollinger_lband()
    df['bb_mid'] = indicator_bb.bollinger_mavg()
    
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['z_score_20'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['volatility_20']

    # 2. Structural Features (Timeframe & Cyclic Time)
    # Map timeframes to minutes for numerical model input
    tf_map = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    df['timeframe_minutes'] = df['timeframe'].map(tf_map).fillna(60)

    # Cyclic Hour (0-23)
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

    # Cyclic Day of Week (0-6)
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 6)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 6)
    
    return df

def train_global_xgboost(market='futures', tune_hyperparams=False, force_train=False):
    """
    The Global AI Scout: Trains across the specified market's SQLite Database panel.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'xgboost_{market}.json')
    meta_path = os.path.join(model_dir, f'xgboost_{market}_meta.json')
    
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'ema_20', 'ema_50', 
                'ema_200', 'volatility_20', 'z_score_20', 'volume',
                'timeframe_minutes', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
                
    if not force_train and os.path.exists(model_path) and os.path.exists(meta_path):
        print(f"🧠 AI SCOUT: Loading cached XGBoost model for {market.upper()} market...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        with open(meta_path, 'r') as f:
            accuracy = json.load(f).get('accuracy', 0.5)
        return model, features, accuracy

    print(f"🧠 AI SCOUT: Training fresh XGBoost model for {market.upper()} market...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE market = ?", conn, params=(market,))
    conn.close()
    
    if df.empty:
        print(f"⚠️ Database has no data for market: {market}.")
        return None, None, 0.0
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp')
    
    dfs = []
    for symbol, group in df.groupby('symbol'):
        # Target variable (Multi-class):
        # 1: Bullish Breakout (>0.5% up in 5 candles)
        # 2: Bearish Breakdown (>0.5% down in 5 candles)
        # 0: Flat / Neutral
        future_close = group['close'].shift(-5)
        returns_5 = (future_close - group['close']) / group['close']
        
        group['target'] = 0
        group.loc[returns_5 > 0.005, 'target'] = 1
        group.loc[returns_5 < -0.005, 'target'] = 2
        
        processed = add_indicators(group.copy())
        
        processed = processed.dropna()
        dfs.append(processed)
        
    if not dfs:
        return None, None, 0.0
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    X = master_df[features]
    y = master_df['target']
    
    # Chronological Train/Validation/Test Split (70/15/15)
    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    if tune_hyperparams:
        print("🔍 AI SCOUT: Running RandomizedSearchCV for XGBoost... (Multi-class)")
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        base_model = xgb.XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='mlogloss', num_class=3)
        search = RandomizedSearchCV(base_model, param_grid, n_iter=5, scoring='accuracy', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        xgb_model = search.best_estimator_
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42,
            objective='multi:softprob', eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("🧠 AI SCOUT: Training Random Forest validator...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Final accuracy on the UNSEEN test set
    xgb_preds = xgb_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    accuracy = (xgb_acc + rf_acc) / 2
    
    print(f"Ensemble Results across {len(dfs)} symbols ({market.upper()}):")
    print(f"  XGBoost Accuracy:      {xgb_acc:.4f}")
    print(f"  Random Forest Accuracy: {rf_acc:.4f}")
    print(f"  Ensemble Mean:         {accuracy:.4f}")
    
    # Save models
    print("💾 AI SCOUT: Saving ensemble models to disk...")
    xgb_model.save_model(model_path)
    rf_path = model_path.replace('.json', '.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(meta_path, 'w') as f:
        json.dump({'accuracy': accuracy}, f)
    
    return (xgb_model, rf_model), features, float(accuracy)

def get_latest_probabilities(model, features, market='futures'):
    """
    Scans the live/latest edge of the database and predicts probability of a breakout.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE market = ?", conn, params=(market,))
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Support ensemble loading
    if isinstance(model, tuple):
        xgb_m, rf_m = model
    else:
        xgb_m = model
        rf_m = None

    probs = {}
    for symbol, group in df.groupby('symbol'):
        processed = add_indicators(group.copy()).dropna()
        if processed.empty:
            continue
            
        latest_state = processed.iloc[-1:]
        X_live = latest_state[features]
        
        xgb_p = xgb_m.predict_proba(X_live)[0]
        if rf_m:
            rf_p = rf_m.predict_proba(X_live)[0]
            # Average probabilities across ensemble for each class
            combined_p = (xgb_p + rf_p) / 2
        else:
            combined_p = xgb_p
            
        # Return a dictionary of class probabilities
        # 1: Bull, 2: Bear, 0: Flat
        probs[symbol] = {
            'bull': float(combined_p[1]),
            'bear': float(combined_p[2]),
            'flat': float(combined_p[0])
        }
        
    return probs

def calculate_seasonality(df=None):
    """
    Calculates broad market seasonality. If df is provided, uses it. 
    Otherwise, queries the whole DB.
    """
    if df is None:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM ohlcv", conn)
        conn.close()
    
    if df.empty: return {}
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    
    day_stats = df.groupby('day_of_week')['returns'].agg(['mean', 'std']).to_dict('index')
    hour_stats = df.groupby('hour')['returns'].agg(['mean']).to_dict('index')
    
    seasonality_summary = {
        'day_of_week': {day: {'return': float(stats['mean']), 'volatility': float(stats['std'])} for day, stats in day_stats.items()},
        'best_hours': sorted([{str(hour): float(stats['mean'])} for hour, stats in hour_stats.items()], key=lambda x: list(x.values())[0], reverse=True)[:5]
    }
    
    return seasonality_summary

def calculate_correlation(df, base_symbol='BTC/USDT'):
    """
    Calculates Pearson correlation between a base symbol and others in the provided DataFrame.
    """
    if df.empty: return {}
    
    pivot_df = df.pivot(index='timestamp', columns='symbol', values='close')
    if base_symbol not in pivot_df.columns:
        return {}
        
    correlations = pivot_df.corr()[base_symbol].to_dict()
    return correlations

def calculate_market_correlations(df):
    """
    Calculates a full correlation matrix and returns the top 5 most positive and top 5 most negative pairs.
    """
    if df.empty: return {}

    # Pivot to get a matrix of prices per timestamp
    pivot_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna(axis=1, how='all')
    if pivot_df.empty: return {}

    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Extract unique pairs
    pairs = []
    symbols = corr_matrix.columns
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            corr = corr_matrix.loc[s1, s2]
            if not np.isnan(corr):
                pairs.append({'p1': s1, 'p2': s2, 'corr': float(corr)})

    # Sort pairs
    top_pos = sorted(pairs, key=lambda x: x['corr'], reverse=True)[:5]
    top_neg = sorted(pairs, key=lambda x: x['corr'])[:5]

    return {
        'top_positive': top_pos,
        'top_negative': top_neg
    }

def get_db_timerange():
    """
    Returns the start and end dates from the database to inject timeframe context into the LLM.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
    min_ts, max_ts = cursor.fetchone()
    conn.close()
    
    if min_ts and max_ts:
        start_date = pd.to_datetime(min_ts, unit='ms').strftime('%Y-%m-%d')
        end_date = pd.to_datetime(max_ts, unit='ms').strftime('%Y-%m-%d')
        return f"{start_date} to {end_date}"
    return "Unknown Timeframe"

if __name__ == "__main__":
    print("Training Global ML Model...")
    model, features, accuracy = train_global_xgboost()
    if model:
        print(f"\nModel Test Accuracy: {accuracy:.2%}")
        print("\nScanning current market probabilities...")
        probs = get_latest_probabilities(model, features)
        for sym, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{sym}: {prob:.2%}")
        
        print("\nChecking Seasonality...")
        seasonality = calculate_seasonality()
        print("Best hours to trade (UTC):", seasonality['best_hours'])
