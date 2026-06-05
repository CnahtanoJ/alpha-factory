import numpy as np
import pandas as pd
from typing import Optional
from ta.momentum import RSIIndicator
from ta.trend import MACD


def get_fwd_return_bars(timeframe: str) -> int:
    """Get the number of forward return bars based on timeframe name."""
    mapping = {"15m": 12, "1h": 12, "4h": 6}
    return mapping.get(timeframe, 12)


def calculate_base_features(
    df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculates technical, derivatives, sentiment, and cyclical features on a DataFrame.
    This function is used for both historical training and live bot execution.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base Indicators
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["volatility_20"] = df["close"].rolling(window=20).std()

    # True Range & ATR % for risk parity weighting
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / df["close"]

    # Ensure all core and feature columns exist in the DataFrame (especially for live bot execution)
    fallback_cols = [
        "oi_usd",
        "funding_rate",
        "last_funding_rate",
        "sum_toptrader_long_short_ratio",
        "sum_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        "sum_open_interest",
        "index_close",
        "final_index_close",
        "sum_open_interest_value",
    ]
    for col in fallback_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Derivatives base logic
    # In live bot, final_index_close is used. In training, index_close is used.
    idx_col = (
        "final_index_close"
        if "final_index_close" in df.columns and df["final_index_close"].notna().any()
        else "index_close"
    )
    # Fallback to close if index price is completely missing
    df["temp_index_close"] = df[idx_col].fillna(df["close"])
    df["basis_pct"] = (df["close"] - df["temp_index_close"]) / df["temp_index_close"]
    df.drop(columns=["temp_index_close"], inplace=True)

    # Resolve oi_usd and funding_rate
    if (
        "sum_open_interest_value" in df.columns
        and df["sum_open_interest_value"].notna().any()
    ):
        df["oi_usd"] = df["sum_open_interest_value"].fillna(
            df["sum_open_interest"] * df["close"]
        )
    else:
        df["oi_usd"] = df["oi_usd"].fillna(df["sum_open_interest"] * df["close"])

    df["funding_rate"] = df["funding_rate"].fillna(df["last_funding_rate"])

    # Market Correlation
    if btc_df is not None and not btc_df.empty:
        df = pd.merge_asof(df, btc_df, on="timestamp", direction="backward")
        df["corr_to_index"] = (
            df["close"].pct_change().rolling(20).corr(df["btc_close"].pct_change())
        )
    else:
        # Check if corr_to_index is already computed or index_df merge was done externally
        if "corr_to_index" not in df.columns:
            df["corr_to_index"] = 0.0

    # Cyclic Time Features
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_num"] = df["timestamp"].dt.dayofweek
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)

    # 1. Derivatives Velocity
    df["oi_delta_4"] = df["oi_usd"].pct_change(4)
    df["funding_delta_4"] = df["funding_rate"].diff(4)
    df["taker_buy_sell_ratio"] = df["sum_taker_long_short_vol_ratio"]

    # 2. Momentum & Mean Reversion Refinements
    ema_50 = df["close"].ewm(span=50, adjust=False).mean()
    df["distance_from_ema_50"] = (df["close"] - ema_50) / ema_50

    vol_mean = df["volatility_20"].rolling(100).mean()
    vol_std = df["volatility_20"].rolling(100).std()
    df["volatility_zscore"] = (df["volatility_20"] - vol_mean) / (vol_std + 1e-9)

    vol_ma = df["volume"].rolling(50).mean()
    vol_sd = df["volume"].rolling(50).std()
    df["volume_zscore"] = (df["volume"] - vol_ma) / (vol_sd + 1e-9)

    # 3. Relative Strength (vs BTC) & Market Beta
    if btc_df is not None and not btc_df.empty:
        df["ret_12"] = df["close"].pct_change(12)
        df["btc_ret_12"] = df["btc_close"].pct_change(12)
        df["relative_strength_btc"] = df["ret_12"] - df["btc_ret_12"]

        # Market Beta
        asset_ret_1 = df["close"].pct_change()
        btc_ret_1 = df["btc_close"].pct_change()
        cov = asset_ret_1.rolling(20).cov(btc_ret_1)
        var = btc_ret_1.rolling(20).var()
        df["market_beta"] = (cov / (var + 1e-9)).fillna(0.0)
    else:
        # Check if they already exist from external index_df merge (e.g. live inference)
        if "relative_strength_btc" not in df.columns:
            df["relative_strength_btc"] = 0.0
        if "market_beta" not in df.columns:
            df["market_beta"] = 0.0

    # Sentiment Divergence (Whales vs Retail)
    top_trader = df["sum_toptrader_long_short_ratio"]
    global_retail = df["sum_long_short_ratio"].fillna(top_trader)
    df["sentiment_divergence"] = (top_trader - global_retail).fillna(0.0)

    # 4. Proxy CVD & Divergence
    candle_range = df["high"] - df["low"]
    candle_range = candle_range.replace(0, 1e-9)
    delta = df["volume"] * ((df["close"] - df["open"]) / candle_range)
    cvd = delta.cumsum()
    df["cvd_slope_5"] = cvd.diff(5)

    # Normalize CVD slope and price return for divergence
    price_ret_5 = df["close"].pct_change(5)
    vol_ma_20 = df["volume"].rolling(20).mean()
    norm_cvd_slope = df["cvd_slope_5"] / (vol_ma_20 * 5 + 1e-9)
    df["price_cvd_divergence"] = price_ret_5 - norm_cvd_slope

    # 5. HTF Features & Divergences
    ema_50_slope = df["close"].ewm(span=50, adjust=False).mean().pct_change(5)
    ema_50_4h_slope = df.get("ema_50_4h", df["close"]).pct_change(16)
    df["trend_convergence"] = (ema_50_slope * ema_50_4h_slope).fillna(0.0)

    # BBW Squeeze (Normalized over 100 periods)
    sma_20 = df["close"].rolling(20).mean()
    bbw_20 = df["volatility_20"] / (sma_20 + 1e-9)
    bbw_100_min = bbw_20.rolling(100).min()
    bbw_100_max = bbw_20.rolling(100).max()
    df["bbw_squeeze"] = (
        (bbw_20 - bbw_100_min) / (bbw_100_max - bbw_100_min + 1e-9)
    ).fillna(0.0)

    # Funding / Basis Divergence
    fund_100_mean = df["funding_rate"].rolling(100).mean()
    fund_100_std = df["funding_rate"].rolling(100).std()
    fund_z = (df["funding_rate"] - fund_100_mean) / (fund_100_std + 1e-9)

    basis_100_mean = df["basis_pct"].rolling(100).mean()
    basis_100_std = df["basis_pct"].rolling(100).std()
    basis_z = (df["basis_pct"] - basis_100_mean) / (basis_100_std + 1e-9)
    df["funding_basis_divergence"] = (fund_z - basis_z).fillna(0.0)

    # 6. Additional Continuous Features
    df["vol_volatility_ratio"] = (vol_ma_20 / (df["volatility_20"] + 1e-9)).fillna(0.0)
    df["rsi_divergence"] = df["rsi"].diff(5) - df["close"].pct_change(5)
    vpt = (df["volume"] * df["close"].pct_change()).cumsum()
    df["vpt_slope"] = vpt.diff(5) / (vol_ma_20 + 1e-9)
    df["range_expansion"] = (df["high"] - df["low"]) / (df["volatility_20"] + 1e-9)
    df["sum_toptrader_ls_delta_4"] = df["sum_toptrader_long_short_ratio"].diff(4)
    df["net_taker_volume_zscore"] = (df["cvd_slope_5"] / (vol_ma_20 + 1e-9)).fillna(0.0)

    # Delta & Acceleration Features
    df["rsi_delta_4"] = df["rsi"].diff(4)
    df["macd_delta_4"] = df["macd"].diff(4)
    df["volatility_delta_4"] = df["volatility_20"].diff(4)
    df["volume_delta_4"] = df["volume"].pct_change(4)
    df["oi_change_pct_12"] = df["oi_usd"].pct_change(12)
    df["funding_acceleration"] = df["funding_delta_4"].diff(1)

    # Lags & Momentum
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_3"] = df["close"].pct_change(3)
    df["mom_accel_1_3"] = df["ret_1"] - df["ret_3"]

    return df


def add_time_series_features(
    df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes all standard features, and adds training target variables.
    This function is used primarily for training pipelines.
    """
    # 1. Compute baseline features
    df = calculate_base_features(df, btc_df)

    # 2. Compute targets (for model training)
    # Dynamic horizon based on timeframe
    tf_val = df["timeframe_name"].iloc[0] if "timeframe_name" in df.columns else "15m"
    fwd_bars = get_fwd_return_bars(tf_val)

    # Safely compute fwd_return avoiding gaps
    tf_timedelta = df["timestamp"].diff().mode()[0]
    expected_delta = tf_timedelta * fwd_bars

    shifted_close = df["close"].shift(-fwd_bars)
    shifted_ts = df["timestamp"].shift(-fwd_bars)

    valid_mask = (shifted_ts - df["timestamp"]) == expected_delta
    raw_fwd = shifted_close / df["close"] - 1

    df["fwd_return"] = np.where(valid_mask, raw_fwd, np.nan)
    df["risk_adj_ret"] = np.where(valid_mask, raw_fwd / (df["atr_pct"] + 1e-9), np.nan)

    # Path-dependent metrics for simulator TP/SL
    fwd_highs = (
        df["high"]
        .iloc[::-1]
        .rolling(window=fwd_bars, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-1)
    )
    fwd_lows = (
        df["low"]
        .iloc[::-1]
        .rolling(window=fwd_bars, min_periods=1)
        .min()
        .iloc[::-1]
        .shift(-1)
    )

    df["fwd_max_ret"] = np.where(valid_mask, fwd_highs / df["close"] - 1, np.nan)
    df["fwd_min_ret"] = np.where(valid_mask, fwd_lows / df["close"] - 1, np.nan)

    return df
