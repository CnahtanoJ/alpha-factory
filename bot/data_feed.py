import math
import time
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timezone
from bot.config import AWS_BUCKET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from data_pipeline.database import DB_PATH

logger = logging.getLogger()


class AssetManager:
    def __init__(self, info=None):
        self.info = info
        self.universe = {}
        if info:
            self.refresh_universe()  # Abstracted into a retryable function

    def refresh_universe(self):
        """Fetches the universe rules, with a retry if the network lags."""
        for attempt in range(3):
            try:
                # Based on your SDK find, this is perfectly correct
                raw_meta = self.info.meta()
                self.universe = {
                    str(a["name"]).upper(): a for a in raw_meta["universe"]
                }
                return  # Success!
            except Exception as e:
                logger.warning(f"Meta fetch failed (Attempt {attempt+1}/3): {e}")
                time.sleep(0.5)

        logger.error(
            "FATAL: Could not load exchange rules. Bot will likely fail size routing."
        )

    def get_price_precision(self, coin, price):
        # Hyperliquid is extremely strict with price decimals.
        # For almost all assets, 5 significant figures is the maximum.
        # If the price is very small, we also cap the absolute decimal places.
        rounded = float(f"{price:.5g}")
        # Secondary safety: never exceed 6 decimal places for any asset
        return round(rounded, 6)

    def round_size(self, coin, size):
        safe_coin = str(coin).upper().strip()
        specs = self.universe.get(safe_coin)

        # The Loud Abort
        if not specs:
            logger.error(f"CRITICAL: {safe_coin} not found! Cannot route size safely.")
            return 0.0

        factor = 10 ** specs["szDecimals"]
        return float(math.floor(size * factor) / factor)

    def get_safe_tp_size(self, coin, total_pos_size, tp_pct):

        # 1. Fetch Price (Need to know $$ value)
        all_mids = self.info.all_mids()
        price = float(all_mids[coin])

        # 2. Calculate Proposed Size
        target_size = total_pos_size * tp_pct
        usd_value = target_size * price

        # 3. Define Safety Threshold ($11 to be safe above $10 min)
        MIN_NOTIONAL = 11.0

        # Check A: Is the TP itself too small?
        if usd_value < MIN_NOTIONAL:
            print(f"TP Adjustment: Chunk ${usd_value:.2f} too small. Selling ALL.")
            return total_pos_size  # Sell Everything

        # Check B: Will the leftovers be trapped dust?
        remaining_val = (total_pos_size - target_size) * price
        if 0 < remaining_val < MIN_NOTIONAL:
            print(f"TP Adjustment: Leftover ${remaining_val:.2f} is dust. Selling ALL.")
            return total_pos_size  # Sell Everything

        # 4. If safe, round it correctly and return
        return self.round_size(coin, target_size)


class MarketData:
    def __init__(self, info):
        self.info = info

    def get_clean_candles(self, coin, interval="15m", limit=5000, use_db=False):
        """
        Fetches OHLCV data.
        If use_db=True, pulls from local SQLite (fast, no rate limits).
        Otherwise, pulls live from exchange API.
        """
        if use_db:
            try:

                # Normalize symbol for DB query (e.g. BTC -> BTC/USDT)
                if "/" not in coin and "-" not in coin:
                    db_symbol = f"{coin}/USDT"
                else:
                    db_symbol = coin

                conn = sqlite3.connect(DB_PATH)
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(db_symbol, interval, limit))
                conn.close()

                if df.empty:
                    logger.warning(
                        f"MarketData: No local data for {db_symbol} [{interval}]."
                    )
                    return pd.DataFrame()

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.sort_values("timestamp").set_index("timestamp")
                return df
            except Exception as e:
                logger.error(f"MarketData DB Error: {e}")
                return pd.DataFrame()

        # --- FALLBACK: LIVE API ---
        try:
            current_time_ms = int(time.time() * 1000)
            start_time_ms = 0

            raw = self.info.candles_snapshot(
                coin, interval, start_time_ms, current_time_ms
            )

            if not raw:
                return pd.DataFrame()
            df = pd.DataFrame(raw).rename(
                columns={
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.tail(limit).set_index("timestamp")

        except:
            return pd.DataFrame()


def fetch_daily_receipt(info_client, wallet_address):
    # 1. Get Midnight UTC
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_day_ms = int(midnight.timestamp() * 1000)

    # 2. Fetch Fills
    # Note: If you have >2000 trades, use info_client.user_fills_by_time instead
    fills = info_client.user_fills(wallet_address)

    daily_trades = 0
    total_pnl = 0.0
    total_fees = 0.0
    total_volume = 0.0

    for f in fills:
        if f.get("time", 0) >= start_of_day_ms:
            daily_trades += 1
            total_pnl += float(f.get("closedPnl", 0))
            total_fees += float(f.get("fee", 0))
            total_volume += float(f.get("sz", 0)) * float(f.get("px", 0))

    # 3. Fetch Funding (The fix is here)
    fundings = info_client.user_funding_history(wallet_address, start_of_day_ms)
    total_funding = 0.0

    for fund in fundings:
        # Check 'time' first
        if fund.get("time", 0) >= start_of_day_ms:
            # IMPORTANT: The 'usdc' value is usually inside the 'delta' dictionary
            delta = fund.get("delta", {})
            amount = delta.get("usdc", 0)
            total_funding += float(amount)

    # 4. Final Calculation
    net_profit = total_pnl - total_fees + total_funding

    return {
        "net_profit": net_profit,
        "gross_pnl": total_pnl,
        "fees": total_fees,
        "funding": total_funding,
        "trades": daily_trades,
        "volume": total_volume,
    }
