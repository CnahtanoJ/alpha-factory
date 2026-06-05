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
from data_pipeline.database import DB_PATH
from typing import Tuple, List, Dict, Any, Optional

# Shared feature engineering functions
from analytics.features import add_time_series_features, get_fwd_return_bars

# --- CANONICAL FEATURE SETS ---
RAW_CONTINUOUS = [
    "rsi",
    "macd",
    "volatility_20",
    "basis_pct",
    "oi_usd",
    "funding_rate",
    "sum_toptrader_long_short_ratio",
    "corr_to_index",
    "taker_buy_sell_ratio",
    "oi_delta_4",
    "funding_delta_4",
    "sum_toptrader_ls_delta_4",
    "distance_from_ema_50",
    "volatility_zscore",
    "volume_zscore",
    "relative_strength_btc",
    "cvd_slope_5",
    "price_cvd_divergence",
    "sentiment_divergence",
    "trend_convergence",
    "bbw_squeeze",
    "funding_basis_divergence",
    "vol_volatility_ratio",
    "market_beta",
    "rsi_divergence",
    "vpt_slope",
    "range_expansion",
    "net_taker_volume_zscore",
    "rsi_delta_4",
    "macd_delta_4",
    "volatility_delta_4",
    "volume_delta_4",
    "oi_change_pct_12",
    "funding_acceleration",
    "ret_1",
    "ret_2",
    "ret_3",
    "mom_accel_1_3",
]

FULL_CONTINUOUS = [f"rank_{f}" for f in RAW_CONTINUOUS]

PRUNED_CONTINUOUS = [
    "rank_rsi",
    "rank_macd",
    "rank_volatility_20",
    "rank_basis_pct",
    "rank_oi_usd",
    "rank_funding_rate",
    "rank_sum_toptrader_long_short_ratio",
    "rank_corr_to_index",
    "rank_taker_buy_sell_ratio",
    "rank_oi_delta_4",
    "rank_distance_from_ema_50",
    "rank_volatility_zscore",
    "rank_volume_zscore",
    "rank_relative_strength_btc",
    "rank_cvd_slope_5",
    "rank_price_cvd_divergence",
    "rank_sentiment_divergence",
    "rank_trend_convergence",
    "rank_bbw_squeeze",
    "rank_funding_basis_divergence",
    "rank_vol_volatility_ratio",
    "rank_market_beta",
    "rank_rsi_divergence",
    "rank_vpt_slope",
    "rank_range_expansion",
    "rank_net_taker_volume_zscore",
    "rank_funding_delta_4",
    "rank_sum_toptrader_ls_delta_4",
    "rank_rsi_delta_4",
    "rank_macd_delta_4",
    "rank_volatility_delta_4",
    "rank_volume_delta_4",
    "rank_oi_change_pct_12",
    "rank_funding_acceleration",
    "rank_ret_1",
    "rank_ret_2",
    "rank_ret_3",
    "rank_mom_accel_1_3",
]

FULL_TIME = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
PRUNED_TIME = ["day_sin", "day_cos"]

MARKET_REGIME_BASE = [
    "btc_ret_24",
    "btc_volatility_24",
    "market_breadth",
    "regime_score",
]
MARKET_REGIME_MACRO = ["macro_conviction_4h"]


def get_feature_names(timeframe: str = "15m") -> Tuple[List[str], List[str]]:
    """Returns the continuous and time feature lists for a given timeframe."""
    if timeframe == "4h":
        cont, time = PRUNED_CONTINUOUS, PRUNED_TIME
    else:
        cont, time = FULL_CONTINUOUS, FULL_TIME

    regime = MARKET_REGIME_BASE[:]
    if timeframe != "4h":
        regime += MARKET_REGIME_MACRO

    return cont + regime, time


try:
    import boto3
    from bot.config import AWS_BUCKET
except ImportError:
    boto3 = None
    AWS_BUCKET = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def get_params_path(timeframe: str) -> str:
    return os.path.join(MODEL_DIR, f"best_params_{timeframe}.json")


def fetch_and_merge_symbol_data(
    symbol: str, conn: sqlite3.Connection, timeframe: str = "15m"
) -> pd.DataFrame:
    """Fetches raw market and metrics data and merges them for a single symbol."""
    df = pd.read_sql_query(
        "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
        conn,
        params=(symbol, timeframe),
    )
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.floor("s")

    idx_df = pd.read_sql_query(
        "SELECT timestamp as idx_ts, close as index_close FROM index_ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
        conn,
        params=(symbol, timeframe),
    )
    if not idx_df.empty:
        idx_df["idx_ts"] = pd.to_datetime(idx_df["idx_ts"], unit="ms").dt.floor("s")
        df = pd.merge_asof(
            df,
            idx_df,
            left_on="timestamp",
            right_on="idx_ts",
            direction="backward",
            tolerance=pd.Timedelta(hours=2),
        )
    else:
        df["index_close"] = df["close"]

    metrics_df = pd.read_sql_query(
        "SELECT timestamp as met_ts, sum_open_interest, sum_open_interest_value, sum_toptrader_long_short_ratio, sum_long_short_ratio, sum_taker_long_short_vol_ratio FROM symbol_metrics WHERE symbol = ? ORDER BY timestamp",
        conn,
        params=(symbol,),
    )
    if not metrics_df.empty:
        metrics_df["met_ts"] = pd.to_datetime(metrics_df["met_ts"], unit="ms").dt.floor(
            "s"
        )
        df = pd.merge_asof(
            df,
            metrics_df,
            left_on="timestamp",
            right_on="met_ts",
            direction="backward",
            tolerance=pd.Timedelta(hours=2),
        )
    else:
        df["sum_open_interest"] = np.nan
        df["sum_open_interest_value"] = np.nan
        df["sum_toptrader_long_short_ratio"] = np.nan

    fund_df = pd.read_sql_query(
        "SELECT calc_time, last_funding_rate FROM funding_rate WHERE symbol = ? ORDER BY calc_time",
        conn,
        params=(symbol,),
    )
    if not fund_df.empty:
        fund_df["calc_time"] = pd.to_datetime(fund_df["calc_time"], unit="ms").dt.floor(
            "s"
        )
        df = pd.merge_asof(
            df,
            fund_df,
            left_on="timestamp",
            right_on="calc_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=8),
        )
    else:
        df["last_funding_rate"] = np.nan

    htf_df = pd.read_sql_query(
        "SELECT timestamp as htf_ts, close as htf_close FROM ohlcv WHERE symbol = ? AND timeframe = '4h' ORDER BY timestamp",
        conn,
        params=(symbol,),
    )
    if not htf_df.empty:
        htf_df["ema_50_4h"] = htf_df["htf_close"].ewm(span=50, adjust=False).mean()
        htf_df["rsi_4h"] = RSIIndicator(close=htf_df["htf_close"], window=14).rsi()
        htf_df["htf_ts"] = pd.to_datetime(htf_df["htf_ts"], unit="ms").dt.floor(
            "s"
        ) + pd.Timedelta(hours=4)
        htf_df["htf_ts"] = htf_df["htf_ts"].astype(df["timestamp"].dtype)
        df = pd.merge_asof(
            df,
            htf_df[["htf_ts", "ema_50_4h", "rsi_4h"]],
            left_on="timestamp",
            right_on="htf_ts",
            direction="backward",
        )
    else:
        df["ema_50_4h"] = np.nan
        df["rsi_4h"] = 50.0

    fill_cols = [
        "index_close",
        "sum_open_interest",
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "sum_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "last_funding_rate",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=8)

    df["symbol"] = symbol
    return df


def inject_macro_conviction(mega_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Injects 4h model predictions as a macro conviction feature for 15m/1h models."""
    if timeframe == "4h":
        mega_df["macro_conviction_4h"] = 0.0
        return mega_df

    unique_ts = sorted(mega_df["timestamp"].unique())
    is_training = len(unique_ts) > 500

    if not is_training:
        mega_df["macro_conviction_4h"] = 0.0
        return mega_df

    lgb_4h_path = os.path.join(MODEL_DIR, "cross_sectional_lgbm_4h.txt")
    xgb_4h_path = os.path.join(MODEL_DIR, "cross_sectional_xgboost_4h.json")
    ridge_4h_path = os.path.join(MODEL_DIR, "cross_sectional_ridge_4h.joblib")

    if not (
        os.path.exists(lgb_4h_path)
        and os.path.exists(xgb_4h_path)
        and os.path.exists(ridge_4h_path)
    ):
        print("   Warning: 4h models not found. macro_conviction_4h set to 0.")
        mega_df["macro_conviction_4h"] = 0.0
        return mega_df

    print("   Loading 4h models for macro conviction injection...")
    model_lgb = lgb.Booster(model_file=lgb_4h_path)
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(xgb_4h_path)
    model_ridge = joblib.load(ridge_4h_path)

    print("   Generating 4h predictions on 15m/1h timestamps...")
    conn = sqlite3.connect(DB_PATH)
    symbols = mega_df["symbol"].unique().tolist()

    btc_df_4h = pd.read_sql_query(
        "SELECT timestamp, close as btc_close FROM ohlcv WHERE symbol IN ('BTC/USDT', 'BTCUSDT', 'BTC') AND timeframe = '4h' ORDER BY timestamp",
        conn,
    )
    if not btc_df_4h.empty:
        btc_df_4h["timestamp"] = pd.to_datetime(btc_df_4h["timestamp"], unit="ms")

    dfs_4h = []
    for sym in symbols:
        df_4h = fetch_and_merge_symbol_data(sym, conn, timeframe="4h")
        if df_4h.empty:
            continue
        df_4h = add_time_series_features(df_4h, btc_df_4h)
        for col in df_4h.select_dtypes(include=["float64"]).columns:
            df_4h[col] = df_4h[col].astype(np.float32)
        dfs_4h.append(df_4h)

    conn.close()

    if not dfs_4h:
        mega_df["macro_conviction_4h"] = 0.0
        return mega_df

    mega_4h = pd.concat(dfs_4h, ignore_index=True)
    if not btc_df_4h.empty:
        btc_df_4h = btc_df_4h.sort_values("timestamp")
        btc_df_4h["btc_ret_24"] = btc_df_4h["btc_close"].pct_change(24)
        btc_df_4h["btc_volatility_24"] = (
            btc_df_4h["btc_close"].pct_change().rolling(24).std()
        )
        mega_4h = pd.merge_asof(
            mega_4h.sort_values("timestamp"),
            btc_df_4h[["timestamp", "btc_ret_24", "btc_volatility_24"]],
            on="timestamp",
            direction="backward",
        )
    else:
        mega_4h["btc_ret_24"] = 0.0
        mega_4h["btc_volatility_24"] = 0.0

    if "ret_12" in mega_4h.columns:
        mega_4h["market_breadth"] = mega_4h.groupby("timestamp")["ret_12"].transform(
            lambda x: (x > 0).mean()
        )
    else:
        mega_4h["market_breadth"] = 0.5

    mega_4h["regime_score"] = abs(mega_4h["btc_ret_24"]) / (
        mega_4h["btc_volatility_24"] + 1e-9
    )

    for col in RAW_CONTINUOUS:
        if col in mega_4h.columns:
            mega_4h[f"rank_{col}"] = mega_4h.groupby("timestamp")[col].rank(pct=True)

    features_4h, time_features_4h = get_feature_names("4h")
    X_4h = mega_4h[features_4h + time_features_4h].copy().fillna(0.5)

    lgb_preds = model_lgb.predict(X_4h)
    xgb_preds = model_xgb.predict(X_4h)
    ridge_preds = model_ridge.predict(X_4h)

    mega_4h["macro_conviction_4h"] = (
        (0.4 * lgb_preds) + (0.4 * xgb_preds) + (0.2 * ridge_preds)
    )

    pred_df = mega_4h[["timestamp", "symbol", "macro_conviction_4h"]].copy()

    merged_parts = []
    for sym in symbols:
        sym_df = mega_df[mega_df["symbol"] == sym]
        sym_preds = pred_df[pred_df["symbol"] == sym][
            ["timestamp", "macro_conviction_4h"]
        ]
        if sym_preds.empty:
            sym_df = sym_df.copy()
            sym_df["macro_conviction_4h"] = 0.0
        else:
            sym_df = sym_df.sort_values("timestamp")
            sym_df = pd.merge_asof(
                sym_df, sym_preds, on="timestamp", direction="backward"
            )
        merged_parts.append(sym_df)

    mega_df = pd.concat(merged_parts, ignore_index=True)
    mega_df["macro_conviction_4h"] = mega_df["macro_conviction_4h"].fillna(0.0)

    n_filled = (mega_df["macro_conviction_4h"] != 0.0).sum()
    print(f"   Macro conviction injected for {n_filled:,} / {len(mega_df):,} rows")

    del merged_parts, pred_df
    gc.collect()

    mega_df = mega_df.sort_values("timestamp").reset_index(drop=True)
    return mega_df


def _load_and_process_symbols_data(
    symbols: List[str], btc_df: pd.DataFrame, conn: sqlite3.Connection, timeframe: str
) -> List[pd.DataFrame]:
    """Loads, processes, and downcasts data for each symbol to optimize memory usage."""
    all_dfs = []
    for sym in symbols:
        df = fetch_and_merge_symbol_data(sym, conn, timeframe=timeframe)
        if df.empty:
            continue
        df["timeframe_name"] = timeframe
        df = add_time_series_features(df, btc_df)

        df = df.dropna(subset=["fwd_return", "rsi"])

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=["int64"]).columns:
            if col != "timestamp":
                df[col] = df[col].astype(np.int32)

        if not df.empty:
            all_dfs.append(df)
    return all_dfs


def _inject_regime_features(
    mega_df: pd.DataFrame, btc_df: pd.DataFrame
) -> pd.DataFrame:
    """Injects market regime features calculated from BTC index data."""
    if not btc_df.empty:
        btc_df = btc_df.sort_values("timestamp")
        btc_df["btc_ret_24"] = btc_df["btc_close"].pct_change(24)
        btc_df["btc_volatility_24"] = btc_df["btc_close"].pct_change().rolling(24).std()
        mega_df = pd.merge_asof(
            mega_df.sort_values("timestamp"),
            btc_df[["timestamp", "btc_ret_24", "btc_volatility_24"]],
            on="timestamp",
            direction="backward",
        )
    else:
        mega_df["btc_ret_24"] = 0.0
        mega_df["btc_volatility_24"] = 0.0

    if "ret_12" in mega_df.columns:
        mega_df["market_breadth"] = mega_df.groupby("timestamp")["ret_12"].transform(
            lambda x: (x > 0).mean()
        )
    else:
        mega_df["market_breadth"] = 0.5

    mega_df["regime_score"] = abs(mega_df["btc_ret_24"]) / (
        mega_df["btc_volatility_24"] + 1e-9
    )
    return mega_df


def _apply_winsorization(
    mega_df: pd.DataFrame, critical_cols: List[str]
) -> pd.DataFrame:
    """Applies outlier winsorization to keep feature values bounded."""
    for col in critical_cols:
        if col in mega_df.columns:
            counts = mega_df.groupby("timestamp")[col].transform("count")
            q_01_ts = mega_df.groupby("timestamp")[col].transform("quantile", 0.01)
            q_99_ts = mega_df.groupby("timestamp")[col].transform("quantile", 0.99)
            q_01_glob = mega_df[col].quantile(0.01)
            q_99_glob = mega_df[col].quantile(0.99)

            q_01 = np.where(counts >= 10, q_01_ts, q_01_glob)
            q_99 = np.where(counts >= 10, q_99_ts, q_99_glob)

            mega_df[col] = mega_df[col].clip(lower=q_01, upper=q_99)
    return mega_df


def _calculate_hybrid_target(mega_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates magnitude-aware hybrid targets for ensemble model training."""
    g = mega_df.groupby("timestamp")
    z_raw = (mega_df["fwd_return"] - g["fwd_return"].transform("mean")) / (
        g["fwd_return"].transform("std") + 1e-9
    )
    z_risk = (mega_df["risk_adj_ret"] - g["risk_adj_ret"].transform("mean")) / (
        g["risk_adj_ret"].transform("std") + 1e-9
    )
    mega_df["target_magnitude"] = 0.5 * z_raw + 0.5 * z_risk
    mega_df["target_rank"] = g["target_magnitude"].rank(pct=True)
    return mega_df


def build_mega_dataframe(timeframe: str = "15m") -> pd.DataFrame:
    """Builds the cross-sectional DataFrame by loading data and calculating all ranks/targets."""
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ?",
        conn,
        params=(timeframe,),
    )["symbol"].tolist()

    btc_df = pd.read_sql_query(
        "SELECT timestamp, close as btc_close FROM ohlcv WHERE symbol IN ('BTC/USDT', 'BTCUSDT', 'BTC') AND timeframe = ? ORDER BY timestamp",
        conn,
        params=(timeframe,),
    )
    if not btc_df.empty:
        btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], unit="ms")

    print(f"Building Aggregated DataFrame for {len(symbols)} symbols...")
    all_dfs = _load_and_process_symbols_data(symbols, btc_df, conn, timeframe)
    conn.close()

    if not all_dfs:
        print("No data found to build mega dataframe.")
        return pd.DataFrame()

    mega_df = pd.concat(all_dfs, ignore_index=True)

    print("Injecting Market Regime features...")
    mega_df = _inject_regime_features(mega_df, btc_df)

    print("Applying Outlier Winsorization (1%/99%)...")
    critical_cols = [
        "funding_rate",
        "volume_zscore",
        "volatility_zscore",
        "ret_1",
        "ret_2",
        "ret_3",
        "mom_accel_1_3",
    ]
    mega_df = _apply_winsorization(mega_df, critical_cols)

    mega_df = inject_macro_conviction(mega_df, timeframe)

    print(
        f"Aggregated DataFrame constructed: {len(mega_df)} rows. Purging intermediate memory..."
    )
    del all_dfs
    gc.collect()

    print("Applying cross-sectional ranking...")
    for col in RAW_CONTINUOUS:
        if col in mega_df.columns:
            mega_df[f"rank_{col}"] = mega_df.groupby("timestamp")[col].rank(pct=True)

    print("Calculating Magnitude-Aware Hybrid Target (50/50 Raw/Risk-Adj)...")
    mega_df = _calculate_hybrid_target(mega_df)
    mega_df = mega_df.dropna(subset=["target_magnitude", "rank_rsi", "rank_oi_delta_4"])

    return mega_df


def prepare_training_data(
    mega_df: pd.DataFrame, timeframe: str = "15m"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Splits data chronologically into training and validation sets."""
    print(f"Preparing training data for {timeframe}...")
    mega_df = mega_df.sort_values("timestamp")

    continuous, time_features = get_feature_names(timeframe)
    features = continuous + time_features

    if timeframe == "4h":
        print(f"Using PRUNED feature set for 4h ({len(features)} features)")
    else:
        print(f"Using FULL feature set for {timeframe} ({len(features)} features)")

    X = mega_df[features].copy()
    y = mega_df["target_magnitude"]
    X = X.fillna(0.5)

    split_idx = int(len(mega_df) * 0.85)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

    return X_train, y_train, X_val, y_val, features


def optimize_lgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeframe: str = "15m",
) -> Dict[str, Any]:
    """Runs hyperparameter tuning using Optuna and stores the best configuration."""
    print(f"Starting Optuna HPO with {n_trials} trials for {timeframe}...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=25)],
        )

        return model.best_score["valid_0"]["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best Trial: RMSE {study.best_value:.4f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    params_path = get_params_path(timeframe)
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    return study.best_params


def train_ensemble_models(
    mega_df: pd.DataFrame,
    optimized_params: Optional[Dict[str, Any]] = None,
    timeframe: str = "15m",
) -> Tuple[Tuple[lgb.Booster, xgb.XGBRegressor, Ridge], List[str], float, float, float]:
    """Trains LightGBM, XGBoost, and Ridge regressors and creates a weighted ensemble prediction."""
    X_train, y_train, X_val, y_val, features = prepare_training_data(
        mega_df, timeframe=timeframe
    )

    print("Training LightGBM...")
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

    params_path = get_params_path(timeframe)
    if optimized_params is None and os.path.exists(params_path):
        print(f"Loading HPO parameters from {params_path}")
        with open(params_path, "r") as f:
            optimized_params = json.load(f)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbose": -1,
        **(
            optimized_params
            if optimized_params
            else {"learning_rate": 0.05, "num_leaves": 31}
        ),
    }

    model_lgb = lgb.train(
        lgb_params,
        train_data_lgb,
        num_boost_round=1000,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    print("Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("Training Ridge Regression...")
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)

    lgb_preds = model_lgb.predict(X_val)
    xgb_preds = model_xgb.predict(X_val)
    ridge_preds = model_ridge.predict(X_val)

    ensemble_preds = (0.4 * lgb_preds) + (0.4 * xgb_preds) + (0.2 * ridge_preds)

    spearman_corr, p_value = spearmanr(ensemble_preds, y_val)
    rmse = np.sqrt(np.mean((ensemble_preds - y_val) ** 2))
    print(f"Validation Ensemble Spearman Correlation: {spearman_corr:.4f}")

    return (model_lgb, model_xgb, model_ridge), features, spearman_corr, p_value, rmse


def upload_ensemble_to_s3(timeframe: str = "15m", meta_only: bool = False) -> bool:
    """Uploads the trained models and meta information to S3."""
    try:
        import boto3
        from bot.config import AWS_BUCKET
    except ImportError:
        print("boto3 or AWS config not available. Skipping S3 upload.")
        return False

    lgb_path = os.path.join(MODEL_DIR, f"cross_sectional_lgbm_{timeframe}.txt")
    xgb_path = os.path.join(MODEL_DIR, f"cross_sectional_xgboost_{timeframe}.json")
    ridge_path = os.path.join(MODEL_DIR, f"cross_sectional_ridge_{timeframe}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"cross_sectional_lgbm_{timeframe}_meta.json")

    s3 = boto3.client("s3")
    try:
        if not meta_only:
            s3.upload_file(
                lgb_path, AWS_BUCKET, f"models/cross_sectional_lgbm_{timeframe}.txt"
            )
            s3.upload_file(
                xgb_path, AWS_BUCKET, f"models/cross_sectional_xgboost_{timeframe}.json"
            )
            s3.upload_file(
                ridge_path,
                AWS_BUCKET,
                f"models/cross_sectional_ridge_{timeframe}.joblib",
            )
            print("Ensemble Models uploaded to S3.")

        s3.upload_file(
            meta_path, AWS_BUCKET, f"models/cross_sectional_lgbm_{timeframe}_meta.json"
        )
        print("Metadata uploaded to S3.")
        return True
    except Exception as e:
        print(f"S3 upload failed: {e}")
        return False


def train_cross_sectional_lgbm(
    mega_df: pd.DataFrame,
    optimized_params: Optional[Dict[str, Any]] = None,
    timeframe: str = "15m",
    upload: bool = True,
) -> Tuple[lgb.Booster, List[str]]:
    """Trains and stores the final ensemble model suite."""
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    os.makedirs(MODEL_DIR, exist_ok=True)

    lgb_path = os.path.join(MODEL_DIR, f"cross_sectional_lgbm_{timeframe}.txt")
    xgb_path = os.path.join(MODEL_DIR, f"cross_sectional_xgboost_{timeframe}.json")
    ridge_path = os.path.join(MODEL_DIR, f"cross_sectional_ridge_{timeframe}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"cross_sectional_lgbm_{timeframe}_meta.json")

    ensemble, features, spearman_corr, p_value, rmse = train_ensemble_models(
        mega_df, optimized_params, timeframe=timeframe
    )
    model_lgb, model_xgb, model_ridge = ensemble

    lgb_importance = dict(
        zip(features, model_lgb.feature_importance(importance_type="gain").tolist())
    )
    xgb_importance = dict(
        zip(features, [float(v) for v in model_xgb.feature_importances_])
    )

    total_lgb = sum(lgb_importance.values()) + 1e-9
    total_xgb = sum(xgb_importance.values()) + 1e-9

    combined_importance = []
    for f in features:
        combined_importance.append(
            {
                "feature": f,
                "combined_importance": (
                    (lgb_importance[f] / total_lgb) + (xgb_importance[f] / total_xgb)
                )
                / 2.0,
            }
        )
    combined_importance.sort(key=lambda x: x["combined_importance"], reverse=True)

    actual_params = optimized_params
    if actual_params is None:
        params_path = get_params_path(timeframe)
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                actual_params = json.load(f)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp_str,
                "is_ensemble": True,
                "best_params": actual_params,
                "validation_spearman": float(spearman_corr),
                "validation_rmse": float(rmse),
                "spearman_p_value": float(p_value),
                "feature_importance": combined_importance,
            },
            f,
            indent=4,
        )

    model_lgb.save_model(lgb_path)
    model_xgb.save_model(xgb_path)
    joblib.dump(model_ridge, ridge_path)

    if upload and boto3:
        upload_ensemble_to_s3(timeframe)

    return model_lgb, features


if __name__ == "__main__":
    mega_df = build_mega_dataframe()
    if not mega_df.empty:
        model, features = train_cross_sectional_lgbm(mega_df)
