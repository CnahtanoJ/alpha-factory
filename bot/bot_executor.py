import inspect

# Monkey-patch inspect.getargspec for compatibility with hyperliquid-python-sdk (v0.5.0),
# which uses the deprecated getargspec function removed in Python 3.11+.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = lambda func: inspect.getfullargspec(func)[:4]

import os
import json
import time
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
from typing import Tuple, Optional, Dict, Any, List

from bot.config import AWS_BUCKET, TESTNET_MODE, BASE_URL
from bot.utils import S3Interface, send_telegram_message, send_telegram_receipt
from bot.data_feed import AssetManager, fetch_daily_receipt
from bot.indicators import get_local_poc
from bot.risk_engine import RiskEngine
from data_pipeline.hyperliquid_sync import (
    get_hl_top_by_volume,
    get_live_meta_ctx,
    get_latest_candles,
    get_bulk_latest_candles,
)
from data_pipeline.database import get_connection
from data_pipeline.binance_live import get_bulk_binance_derivatives
from analytics.portfolio_optimizer import compute_hrp_weights

# Shared feature engineering functions
from analytics.features import calculate_base_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LiveInferenceEngine:
    def __init__(
        self,
        info: Info,
        conn: Optional[Any] = None,
        live_derivatives: Optional[Dict[str, Any]] = None,
    ):
        self.info = info
        self.conn = conn
        self.live_derivatives = live_derivatives

    def build_live_features(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        ctx: Dict[str, Any],
        index_df: Optional[pd.DataFrame] = None,
        htf_candles: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[pd.DataFrame]:
        """Calculates live feature matrix for model inference for a single symbol."""
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.floor("s")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Merge index close if available
        if index_df is not None and not index_df.empty:
            df = pd.merge_asof(
                df,
                index_df[["timestamp", "close"]].rename(
                    columns={"close": "hist_index_close"}
                ),
                on="timestamp",
                direction="backward",
            )
            df["corr_to_index"] = (
                df["close"]
                .pct_change()
                .rolling(20)
                .corr(df["hist_index_close"].pct_change())
            )
        else:
            df["corr_to_index"] = 0.0

        # Load historical database parameters
        if self.conn and symbol:
            try:
                idx_q = "SELECT timestamp as idx_ts, close as index_close FROM index_ohlcv WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100"
                idx_df = pd.read_sql_query(idx_q, self.conn, params=(symbol,))
                if not idx_df.empty:
                    idx_df["idx_ts"] = pd.to_datetime(
                        idx_df["idx_ts"], unit="ms"
                    ).dt.floor("s")
                    df = pd.merge_asof(
                        df,
                        idx_df,
                        left_on="timestamp",
                        right_on="idx_ts",
                        direction="backward",
                    )
            except Exception as e:
                logger.warning(
                    f"Could not fetch historical index data for {symbol}: {e}"
                )

            try:
                met_q = """
                    SELECT timestamp as met_ts, sum_open_interest, sum_open_interest_value, 
                           sum_toptrader_long_short_ratio, sum_long_short_ratio, sum_taker_long_short_vol_ratio 
                    FROM symbol_metrics 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """
                met_df = pd.read_sql_query(met_q, self.conn, params=(symbol,))
                if not met_df.empty:
                    met_df["met_ts"] = pd.to_datetime(
                        met_df["met_ts"], unit="ms"
                    ).dt.floor("s")
                    df = pd.merge_asof(
                        df,
                        met_df,
                        left_on="timestamp",
                        right_on="met_ts",
                        direction="backward",
                    )
            except Exception as e:
                logger.warning(f"Could not fetch historical metrics for {symbol}: {e}")

            try:
                fund_q = """
                    SELECT calc_time, last_funding_rate 
                    FROM funding_rate 
                    WHERE symbol = ? 
                    ORDER BY calc_time DESC 
                    LIMIT 100
                """
                fund_df = pd.read_sql_query(fund_q, self.conn, params=(symbol,))
                if not fund_df.empty:
                    fund_df["calc_time"] = pd.to_datetime(
                        fund_df["calc_time"], unit="ms"
                    ).dt.floor("s")
                    df = pd.merge_asof(
                        df,
                        fund_df,
                        left_on="timestamp",
                        right_on="calc_time",
                        direction="backward",
                    )
            except Exception as e:
                logger.warning(f"Could not fetch historical funding for {symbol}: {e}")

        # Set live values in final row
        live_data = (
            self.live_derivatives.get(symbol, {}) if self.live_derivatives else {}
        )
        binance_idx = float(live_data.get("index_close", 0.0))
        df["live_index_close"] = (
            binance_idx
            if binance_idx > 0
            else float(ctx.get("oraclePx", df["close"].iloc[-1]))
        )

        if "index_close" in df.columns:
            df["final_index_close"] = df["index_close"].fillna(df["live_index_close"])
            df.loc[df.index[-1], "final_index_close"] = df["live_index_close"]
        else:
            df["final_index_close"] = df["live_index_close"]

        binance_oi = float(live_data.get("oi_usd", 0.0))
        live_oi = (
            binance_oi
            if binance_oi > 0
            else float(ctx.get("openInterest", 0)) * float(ctx.get("oraclePx", 0))
        )
        df.loc[df.index[-1], "oi_usd"] = live_oi

        binance_funding = live_data.get("funding_rate", None)
        live_funding = float(
            binance_funding if binance_funding is not None else ctx.get("funding", 0)
        )
        df.loc[df.index[-1], "funding_rate"] = live_funding
        df.loc[df.index[-1], "last_funding_rate"] = live_funding
        df.loc[df.index[-1], "sum_open_interest"] = ctx.get("openInterest", 0)

        # Live Sentiment & Taker ratio mapping
        top_trader = live_data.get("top_long_short", np.nan)
        global_retail = live_data.get("global_long_short", np.nan)
        taker_ratio = live_data.get("taker_buy_sell_ratio", np.nan)

        if pd.notna(top_trader):
            df.loc[df.index[-1], "sum_toptrader_long_short_ratio"] = pd.to_numeric(
                top_trader, errors="coerce"
            )
        if pd.notna(global_retail):
            df.loc[df.index[-1], "sum_long_short_ratio"] = pd.to_numeric(
                global_retail, errors="coerce"
            )
        if pd.notna(taker_ratio):
            df.loc[df.index[-1], "sum_taker_long_short_vol_ratio"] = pd.to_numeric(
                taker_ratio, errors="coerce"
            )

        # Forward fill historical columns for base feature calculation
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

        # Inject Higher Timeframe Features
        if htf_candles:
            htf_df = pd.DataFrame(htf_candles)
            htf_df["timestamp"] = pd.to_datetime(
                htf_df["timestamp"], unit="ms"
            ).dt.floor("s")
            htf_df = htf_df.sort_values("timestamp").reset_index(drop=True)
            htf_df["ema_50_4h"] = htf_df["close"].ewm(span=50, adjust=False).mean()
            htf_df["htf_ts"] = htf_df["timestamp"] + pd.Timedelta(hours=4)
            htf_df["htf_ts"] = htf_df["htf_ts"].astype(df["timestamp"].dtype)
            df = pd.merge_asof(
                df,
                htf_df[["htf_ts", "ema_50_4h"]],
                left_on="timestamp",
                right_on="htf_ts",
                direction="backward",
            )

        # Calculate standard features using unified codebase module
        df = calculate_base_features(df, index_df)
        return df.iloc[-1:]


# Helper functions for decomposing executor_handler


def _determine_task_and_timeframe(
    event: Dict[str, Any], s3: S3Interface
) -> Tuple[str, str, Dict[str, Any], int]:
    """Autonomous routing logic to determine active task and timeframe."""
    try:
        config = s3.download_json("live_config.json")
        timeframe = config.get("active_timeframe", "15m")
        last_rebalance = config.get("last_rebalance_ts", 0)
    except Exception as e:
        logger.warning(f"Could not load live_config.json: {e}. Defaulting to 15m.")
        config = {"active_timeframe": "15m", "last_rebalance_ts": 0}
        timeframe = "15m"
        last_rebalance = 0

    window_map = {"15m": 180, "1h": 720, "4h": 2880}
    rebalance_window_mins = window_map.get(timeframe, 180)

    now_ts = int(time.time())
    mins_since_rebalance = (now_ts - last_rebalance) / 60

    if mins_since_rebalance >= rebalance_window_mins:
        task = "rebalance"
    else:
        task = event.get("task", "rebalance")

    logger.info(
        f"Task routing: {task} | Timeframe: {timeframe} | Minutes since rebalance: {mins_since_rebalance:.1f}m"
    )
    return task, timeframe, config, last_rebalance


def _load_credentials(timeframe: str) -> Tuple[str, str]:
    """Loads private key and address based on deployment environment."""
    env_suffix = timeframe.upper()
    if TESTNET_MODE:
        KEY = os.environ.get(
            f"TESTNET_PRIVATE_KEY_{env_suffix}", os.environ.get("TESTNET_PRIVATE_KEY")
        )
        ADDR = os.environ.get(
            f"TESTNET_ACCOUNT_ADDRESS_{env_suffix}",
            os.environ.get("TESTNET_ACCOUNT_ADDRESS"),
        )
    else:
        KEY = os.environ.get(
            f"MAINNET_PRIVATE_KEY_{env_suffix}", os.environ.get("MAINNET_PRIVATE_KEY")
        )
        ADDR = os.environ.get(
            f"MAINNET_ACCOUNT_ADDRESS_{env_suffix}",
            os.environ.get("MAINNET_ACCOUNT_ADDRESS"),
        )

    if not KEY or not ADDR:
        raise ValueError(f"Missing environment credentials for {timeframe}")
    return KEY, ADDR


def _run_manage_tpsl(
    risk: RiskEngine, portfolio: Dict[str, Any], open_orders: List[Any], timeframe: str
) -> Dict[str, Any]:
    """Executes trailing stop loss adjustments for active portfolio positions."""
    for coin in portfolio:
        candles = get_latest_candles(coin, timeframe, limit=20)
        if not candles:
            continue

        df_mini = pd.DataFrame(candles)
        df_mini["close"] = df_mini["close"].astype(float)
        df_mini["high"] = df_mini["high"].astype(float)
        df_mini["low"] = df_mini["low"].astype(float)

        tr = pd.concat(
            [
                df_mini["high"] - df_mini["low"],
                (df_mini["high"] - df_mini["close"].shift(1)).abs(),
                (df_mini["low"] - df_mini["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct = (tr.rolling(14).mean() / df_mini["close"]).iloc[-1]

        risk.sync_trailing_stop(coin, atr_pct, portfolio[coin], open_orders)

    logger.info("TP/SL management cycle complete.")
    return {"statusCode": 200, "body": "TP/SL Sync Complete"}


def _check_panic_switch(
    s3: S3Interface, timeframe: str, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Gating logic: checks validation metrics to suspend trading if score drops below threshold."""
    meta_path = f"/tmp/cross_sectional_lgbm_{timeframe}_meta.json"
    if s3.download_file(
        f"models/cross_sectional_lgbm_{timeframe}_meta.json", meta_path
    ):
        with open(meta_path, "r") as f:
            model_meta = json.load(f)
        validation_spearman = model_meta.get("validation_spearman", 0.0)
        if validation_spearman < 0.02:
            logger.error(
                f"Panic Switch Triggered: Validation Spearman ({validation_spearman:.4f}) < 0.02. Suspending trading."
            )
            send_telegram_message(
                f"PANIC: Model Validation Spearman ({validation_spearman:.4f}) < 0.02. Suspended for {timeframe}."
            )

            config["last_rebalance_ts"] = int(time.time())
            s3.upload_json("live_config.json", config)
            return {
                "statusCode": 400,
                "body": "Trading suspended due to low validation score.",
            }
    return None


def _fetch_market_data(
    timeframe: str, top_50_symbols: List[str]
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """Fetches real-time context, derivatives prices, and candles in parallel."""
    live_ctx = get_live_meta_ctx()
    live_derivatives = get_bulk_binance_derivatives(top_50_symbols, period=timeframe)

    index_candles = get_latest_candles("BTC", interval=timeframe, limit=100)
    index_df = pd.DataFrame(index_candles) if index_candles else None
    if index_df is not None:
        index_df["timestamp"] = pd.to_datetime(index_df["timestamp"], unit="ms")
        index_df = index_df.sort_values("timestamp").reset_index(drop=True)

    bulk_candles = get_bulk_latest_candles(
        top_50_symbols, interval=timeframe, limit=100
    )
    bulk_htf_candles = get_bulk_latest_candles(top_50_symbols, interval="4h", limit=50)
    return live_ctx, index_df, bulk_candles, bulk_htf_candles


def _generate_live_features(
    top_50_symbols: List[str],
    bulk_candles: Dict[str, Any],
    bulk_htf_candles: Dict[str, Any],
    live_ctx: Dict[str, Any],
    index_df: Optional[pd.DataFrame],
    engine: LiveInferenceEngine,
) -> pd.DataFrame:
    """Builds and consolidates features for all listed symbols."""
    live_rows = []
    for sym, candles in bulk_candles.items():
        ctx = live_ctx.get(sym, {})
        htf_candles = bulk_htf_candles.get(sym, [])
        row_df = engine.build_live_features(
            sym, candles, ctx, index_df=index_df, htf_candles=htf_candles
        )
        if row_df is not None and not row_df.empty:
            row_df["symbol"] = sym
            high = pd.Series([c["high"] for c in candles])
            low = pd.Series([c["low"] for c in candles])
            prev_close = pd.Series([c["close"] for c in candles]).shift(1)
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_pct = (
                tr.rolling(14).mean() / pd.Series([c["close"] for c in candles])
            ).iloc[-1]
            row_df["atr_pct"] = atr_pct
            live_rows.append(row_df)

    if not live_rows:
        raise ValueError("Failed to construct live features.")
    return pd.concat(live_rows, ignore_index=True)


def _inject_live_regime(
    mega_df: pd.DataFrame, index_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Computes and updates market regime classification in real-time."""
    if index_df is not None:
        index_df["btc_ret_24"] = index_df["close"].pct_change(24)
        index_df["btc_volatility_24"] = index_df["close"].pct_change().rolling(24).std()
        mega_df["btc_ret_24"] = index_df["btc_ret_24"].iloc[-1]
        mega_df["btc_volatility_24"] = index_df["btc_volatility_24"].iloc[-1]
    else:
        mega_df["btc_ret_24"] = 0.0
        mega_df["btc_volatility_24"] = 0.0

    mega_df["market_breadth"] = (
        (mega_df["ret_12"] > 0).mean() if "ret_12" in mega_df.columns else 0.5
    )
    btc_ret_val = (
        mega_df["btc_ret_24"].iloc[0] if "btc_ret_24" in mega_df.columns else 0.0
    )
    btc_vol_val = (
        mega_df["btc_volatility_24"].iloc[0]
        if "btc_volatility_24" in mega_df.columns
        else 1e-9
    )
    mega_df["regime_score"] = abs(btc_ret_val) / (btc_vol_val + 1e-9)
    return mega_df


def _inject_live_macro_conviction(
    mega_df: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    """Injects macro predictions from 4h model to align conviction trends."""
    if timeframe == "4h":
        mega_df["macro_conviction_4h"] = 0.0
        return mega_df

    try:
        macro_s3 = S3Interface(AWS_BUCKET)
        macro_meta_path = "/tmp/cross_sectional_lgbm_4h_meta.json"
        macro_valid = False

        if macro_s3.download_file(
            "models/cross_sectional_lgbm_4h_meta.json", macro_meta_path
        ):
            with open(macro_meta_path, "r") as f:
                m_meta = json.load(f)
            if m_meta.get("validation_spearman", 0.0) >= 0.02:
                macro_valid = True
            else:
                logger.warning(
                    "Macro (4h) model failed validation threshold. Setting macro conviction to neutral."
                )

        macro_lgb_path = "/tmp/cross_sectional_lgbm_4h.txt"
        if macro_valid and macro_s3.download_file(
            "models/cross_sectional_lgbm_4h.txt", macro_lgb_path
        ):
            macro_model = lgb.Booster(model_file=macro_lgb_path)
            macro_features = macro_model.feature_name()
            avail_macro = [f for f in macro_features if f in mega_df.columns]
            if len(avail_macro) >= len(macro_features) * 0.5:
                X_macro = mega_df[avail_macro].fillna(0.0)
                mega_df["macro_conviction_4h"] = macro_model.predict(X_macro)
            else:
                mega_df["macro_conviction_4h"] = 0.0
        else:
            mega_df["macro_conviction_4h"] = 0.0
    except Exception as e:
        logger.warning(f"Macro conviction injection failed: {e}")
        mega_df["macro_conviction_4h"] = 0.0
    return mega_df


def _predict_live_ranks(
    mega_df: pd.DataFrame, timeframe: str, s3: S3Interface
) -> pd.DataFrame:
    """Calculates predictions using the model ensemble or falls back to baseline indicators."""
    lgb_path = f"/tmp/cross_sectional_lgbm_{timeframe}.txt"
    xgb_path = f"/tmp/cross_sectional_xgboost_{timeframe}.json"
    ridge_path = f"/tmp/cross_sectional_ridge_{timeframe}.joblib"

    success_lgb = s3.download_file(
        f"models/cross_sectional_lgbm_{timeframe}.txt", lgb_path
    )
    success_xgb = s3.download_file(
        f"models/cross_sectional_xgboost_{timeframe}.json", xgb_path
    )
    success_ridge = s3.download_file(
        f"models/cross_sectional_ridge_{timeframe}.joblib", ridge_path
    )

    ml_success = False
    if success_lgb and success_xgb and success_ridge:
        try:
            model_lgb = lgb.Booster(model_file=lgb_path)
            model_xgb = xgb.XGBRegressor()
            model_xgb.load_model(xgb_path)
            model_ridge = joblib.load(ridge_path)
            from analytics.cross_sectional import get_feature_names

            feature_cols, time_features = get_feature_names(timeframe)
            X_live = mega_df[feature_cols + time_features]
            mega_df["predicted_rank"] = (
                (0.4 * model_lgb.predict(X_live))
                + (0.4 * model_xgb.predict(X_live))
                + (0.2 * model_ridge.predict(X_live))
            )
            ml_success = True
        except Exception as e:
            logger.error(f"Ensemble inference execution failed: {e}")

    if not ml_success:
        logger.warning("Falling back to RSI and MACD average ranking.")
        mega_df["predicted_rank"] = (mega_df["rank_rsi"] + mega_df["rank_macd"]) / 2
    return mega_df


def _execute_rebalance_trades(
    risk: RiskEngine,
    info: Info,
    user_state: Dict[str, Any],
    all_mids: Dict[str, Any],
    portfolio: Dict[str, Any],
    open_orders: List[Any],
    mega_df: pd.DataFrame,
    bulk_candles: Dict[str, Any],
    timeframe: str,
    ADDR: str,
    config: Dict[str, Any],
    s3: S3Interface,
) -> None:
    """Manages order execution and capital allocation weights based on HRP analysis."""
    BASKET_N, HYSTERESIS_FACTOR = 5, 4.0
    buffer_n = int(BASKET_N * HYSTERESIS_FACTOR)
    mega_df = mega_df.sort_values("predicted_rank", ascending=False)
    current_longs = set(c for c, p in portfolio.items() if p.get("szi", 0) > 0)
    current_shorts = set(c for c, p in portfolio.items() if p.get("szi", 0) < 0)

    eligible_longs = mega_df.head(buffer_n)
    target_longs = (
        eligible_longs[eligible_longs["symbol"].isin(current_longs)]["symbol"].tolist()
        + eligible_longs[~eligible_longs["symbol"].isin(current_longs)][
            "symbol"
        ].tolist()
    )[:BASKET_N]
    eligible_shorts = mega_df.tail(buffer_n)
    target_shorts = (
        eligible_shorts[eligible_shorts["symbol"].isin(current_shorts)][
            "symbol"
        ].tolist()
        + eligible_shorts[~eligible_shorts["symbol"].isin(current_shorts)][
            "symbol"
        ].tolist()
    )[-BASKET_N:]
    target_basket = target_longs + target_shorts

    temp_assets = AssetManager(info)
    for active_coin in list(portfolio.keys()):
        if active_coin not in target_basket:
            risk.close_active_position(
                active_coin, all_mids, temp_assets, portfolio, AWS_BUCKET
            )

    time.sleep(2)
    portfolio = risk.parse_portfolio(info.user_state(ADDR), all_mids)
    target_rows = mega_df[mega_df["symbol"].isin(target_basket)].copy()
    if not target_rows.empty:
        hist_returns = {}
        for coin in target_basket:
            coin_candles = bulk_candles.get(coin, [])
            if coin_candles:
                closes = pd.Series([float(c["close"]) for c in coin_candles])
                hist_returns[coin] = closes.pct_change()

        if hist_returns:
            returns_df = pd.DataFrame(hist_returns).tail(100)
            alphas = target_rows.set_index("symbol")["predicted_rank"]

            long_symbols = [c for c in target_basket if c in target_longs]
            short_symbols = [c for c in target_basket if c in target_shorts]

            long_w_dict = compute_hrp_weights(returns_df, long_symbols, alphas=alphas)
            short_w_dict = compute_hrp_weights(returns_df, short_symbols, alphas=alphas)

            total_target_usd = (
                float(user_state.get("marginSummary", {}).get("accountValue", 0)) * 0.90
            )
            symbol_to_usd = {}
            for sym, w in long_w_dict.items():
                symbol_to_usd[sym] = w * (total_target_usd / 2.0)
            for sym, w in short_w_dict.items():
                symbol_to_usd[sym] = w * (total_target_usd / 2.0)
        else:
            logger.warning(
                "Historical returns unavailable for HRP. Falling back to inverse ATR volatility."
            )
            target_rows["inv_vol"] = 1.0 / (
                target_rows["atr_pct"].fillna(0.001).clip(lower=0.001)
            )
            total_target_usd = (
                float(user_state.get("marginSummary", {}).get("accountValue", 0)) * 0.90
            )
            symbol_to_usd = dict(
                zip(
                    target_rows["symbol"],
                    (target_rows["inv_vol"] / target_rows["inv_vol"].sum())
                    * total_target_usd,
                )
            )

        for target_coin in target_basket:
            row = mega_df[mega_df["symbol"] == target_coin].iloc[0]
            atr_pct = float(row["atr_pct"])
            if target_coin not in portfolio:
                current_price = float(all_mids[target_coin])
                coin_candles = bulk_candles.get(target_coin, [])
                poc_price = (
                    get_local_poc(pd.DataFrame(coin_candles)) if coin_candles else 0.0
                )
                if risk.check_execution_safety(
                    target_coin, target_coin in target_longs, current_price, poc_price
                ):
                    risk.execute_logic(
                        target_coin,
                        "BULLISH" if target_coin in target_longs else "BEARISH",
                        timeframe,
                        atr_pct,
                        portfolio,
                        user_state,
                        open_orders,
                        override_usd=symbol_to_usd.get(target_coin),
                    )
            else:
                risk.sync_trailing_stop(target_coin, atr_pct, portfolio, open_orders)

    logger.info("Updating S3 State (last_rebalance_ts)...")
    config["last_rebalance_ts"] = int(time.time())
    s3.upload_json("live_config.json", config)

    time.sleep(2)
    portfolio = risk.parse_portfolio(info.user_state(ADDR), all_mids)
    for active_coin in portfolio:
        row = mega_df[mega_df["symbol"] == active_coin]
        if not row.empty:
            risk.sync_unified_orders(
                active_coin, float(row.iloc[0]["atr_pct"]), portfolio
            )


def executor_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Primary entry point for AWS Lambda scheduler events."""
    logger.info("Initializing execution cycle...")
    info = Info(BASE_URL, skip_ws=True)
    s3 = S3Interface(AWS_BUCKET)

    task, timeframe, config, last_rebalance = _determine_task_and_timeframe(event, s3)

    db_conn = None
    risk = None
    try:
        KEY, ADDR = _load_credentials(timeframe)
        account = Account.from_key(KEY)
        exchange = Exchange(account, BASE_URL, account_address=ADDR)
        risk = RiskEngine(
            exchange=exchange, info=info, account_address=ADDR, bucket=AWS_BUCKET
        )

        user_state = info.user_state(ADDR)
        all_mids = info.all_mids()
        portfolio = risk.parse_portfolio(user_state, all_mids)
        open_orders = info.frontend_open_orders(ADDR)

        if task == "send_daily_report":
            stats = fetch_daily_receipt(info, ADDR)
            msg = send_telegram_receipt(stats)
            send_telegram_message(msg)
            return {"statusCode": 200, "body": "Daily report sent"}

        elif task == "manage_tpsl":
            return _run_manage_tpsl(risk, portfolio, open_orders, timeframe)

        elif task == "rebalance":
            if not risk.check_safety():
                return {"statusCode": 400, "body": "Safety checks failed."}

            panic_response = _check_panic_switch(s3, timeframe, config)
            if panic_response is not None:
                return panic_response

            risk.clean_global_zombies(portfolio, open_orders)

            top_50_symbols = get_hl_top_by_volume(50)
            live_ctx, index_df, bulk_candles, bulk_htf_candles = _fetch_market_data(
                timeframe, top_50_symbols
            )

            try:
                db_conn = get_connection()
            except Exception as e:
                logger.warning(
                    f"Could not connect to database: {e}. Proceeding without DB features."
                )
                db_conn = None

            engine = LiveInferenceEngine(info, conn=db_conn, live_derivatives=None)

            mega_df = _generate_live_features(
                top_50_symbols,
                bulk_candles,
                bulk_htf_candles,
                live_ctx,
                index_df,
                engine,
            )
            mega_df = _inject_live_regime(mega_df, index_df)
            mega_df = _inject_live_macro_conviction(mega_df, timeframe)

            # Ranks for raw columns
            for col in mega_df.columns:
                if col in mega_df.columns:
                    col_mean = mega_df[col].mean()
                    mega_df[col] = mega_df[col].fillna(
                        col_mean if pd.notna(col_mean) else 0.0
                    )
                    mega_df[f"rank_{col}"] = mega_df[col].rank(pct=True)

            mega_df = _predict_live_ranks(mega_df, timeframe, s3)

            _execute_rebalance_trades(
                risk,
                info,
                user_state,
                all_mids,
                portfolio,
                open_orders,
                mega_df,
                bulk_candles,
                timeframe,
                ADDR,
                config,
                s3,
            )

            logger.info("Cycle completed successfully.")
            return {"statusCode": 200, "body": json.dumps("Basket rebalance executed.")}

        else:
            logger.warning(f"Unknown task: {task}")
            return {"statusCode": 400, "body": f"Unknown task: {task}"}

    except Exception as e:
        logger.error(f"Global execution error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        send_telegram_message(f"Global executor error: {e}")
        return {"statusCode": 500, "body": str(e)}

    finally:
        if db_conn:
            db_conn.close()
        if risk:
            risk.memory.save()
