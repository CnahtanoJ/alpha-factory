import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.database import get_connection
from data_pipeline.binance_vision import BinanceVision


class UniversalGapPatcher:
    """
    A robust standalone utility to scan and patch gaps across all
    Alpha Factory historical datasets using strictly Binance Vision.
    """

    def __init__(self):
        self.vision = BinanceVision()
        self.conn = get_connection()
        self.tf_ms_map = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS unfillable_gaps (
                table_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                start_ts INTEGER,
                end_ts INTEGER,
                PRIMARY KEY (table_name, symbol, timeframe, start_ts, end_ts)
            )
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def patch_ohlcv(self, dry_run=True):
        print("\n=== Auditing: ohlcv (Klines) ===")
        partitions = self.conn.execute(
            "SELECT DISTINCT symbol, timeframe, market FROM ohlcv"
        ).fetchall()

        for row in partitions:
            symbol = row["symbol"]
            timeframe = row["timeframe"]
            market = row["market"]
            tf_ms = self.tf_ms_map.get(timeframe)
            if not tf_ms:
                continue

            print(f"\n  [SCAN] Scanning {symbol} ({timeframe})...")

            query = """
                SELECT timestamp, next_ts 
                FROM (
                    SELECT timestamp, LEAD(timestamp) OVER (ORDER BY timestamp) as next_ts 
                    FROM ohlcv 
                    WHERE symbol=? AND timeframe=? AND market=?
                ) 
                WHERE next_ts - timestamp > ?
                AND NOT EXISTS (
                    SELECT 1 FROM unfillable_gaps u
                    WHERE u.table_name='ohlcv' AND u.symbol=? AND u.timeframe=?
                    AND u.start_ts=timestamp AND u.end_ts=next_ts
                )
            """
            gaps = self.conn.execute(
                query, (symbol, timeframe, market, tf_ms * 1.5, symbol, timeframe)
            ).fetchall()

            if not gaps:
                print(f"    [OK] No gaps found.")
                continue

            for gap in gaps:
                ts, next_ts = gap["timestamp"], gap["next_ts"]
                missing_count = int((next_ts - ts) // tf_ms - 1)
                print(
                    f"    [GAP] Gap: {pd.to_datetime(ts, unit='ms')} -> {pd.to_datetime(next_ts, unit='ms')} ({missing_count} bars)"
                )

                if not dry_run:
                    inserted = self._patch_daily_zip(
                        table="ohlcv",
                        symbol=symbol,
                        ts=ts,
                        next_ts=next_ts,
                        time_col="timestamp",
                        data_type="klines",
                        timeframe=timeframe,
                        extra_cols={"market": market},
                        insert_cols=[
                            "timestamp",
                            "symbol",
                            "timeframe",
                            "market",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ],
                    )
                    if inserted == 0:
                        print(
                            "    [INFO] Marking gap as unfillable to prevent future re-checks."
                        )
                        self.conn.execute(
                            "INSERT OR IGNORE INTO unfillable_gaps (table_name, symbol, timeframe, start_ts, end_ts) VALUES (?, ?, ?, ?, ?)",
                            ("ohlcv", symbol, timeframe, ts, next_ts),
                        )
                        self.conn.commit()

    def patch_index_ohlcv(self, dry_run=True):
        print("\n=== Auditing: index_ohlcv (Index Price) ===")
        partitions = self.conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM index_ohlcv"
        ).fetchall()

        for row in partitions:
            symbol = row["symbol"]
            timeframe = row["timeframe"]
            tf_ms = self.tf_ms_map.get(timeframe)
            if not tf_ms:
                continue

            print(f"\n  [SCAN] Scanning {symbol} ({timeframe})...")

            query = """
                SELECT timestamp, next_ts 
                FROM (
                    SELECT timestamp, LEAD(timestamp) OVER (ORDER BY timestamp) as next_ts 
                    FROM index_ohlcv 
                    WHERE symbol=? AND timeframe=?
                ) 
                WHERE next_ts - timestamp > ?
                AND NOT EXISTS (
                    SELECT 1 FROM unfillable_gaps u
                    WHERE u.table_name='index_ohlcv' AND u.symbol=? AND u.timeframe=?
                    AND u.start_ts=timestamp AND u.end_ts=next_ts
                )
            """
            gaps = self.conn.execute(
                query, (symbol, timeframe, tf_ms * 1.5, symbol, timeframe)
            ).fetchall()

            if not gaps:
                print(f"    [OK] No gaps found.")
                continue

            for gap in gaps:
                ts, next_ts = gap["timestamp"], gap["next_ts"]
                missing_count = int((next_ts - ts) // tf_ms - 1)
                print(
                    f"    [GAP] Gap: {pd.to_datetime(ts, unit='ms')} -> {pd.to_datetime(next_ts, unit='ms')} ({missing_count} bars)"
                )

                if not dry_run:
                    inserted = self._patch_daily_zip(
                        table="index_ohlcv",
                        symbol=symbol,
                        ts=ts,
                        next_ts=next_ts,
                        time_col="timestamp",
                        data_type="indexPriceKlines",
                        timeframe=timeframe,
                        extra_cols={},
                        insert_cols=[
                            "timestamp",
                            "symbol",
                            "timeframe",
                            "open",
                            "high",
                            "low",
                            "close",
                        ],
                    )
                    if inserted == 0:
                        print(
                            "    [INFO] Marking gap as unfillable to prevent future re-checks."
                        )
                        self.conn.execute(
                            "INSERT OR IGNORE INTO unfillable_gaps (table_name, symbol, timeframe, start_ts, end_ts) VALUES (?, ?, ?, ?, ?)",
                            ("index_ohlcv", symbol, timeframe, ts, next_ts),
                        )
                        self.conn.commit()

    def patch_symbol_metrics(self, dry_run=True):
        print("\n=== Auditing: symbol_metrics ===")
        partitions = self.conn.execute(
            "SELECT DISTINCT symbol FROM symbol_metrics"
        ).fetchall()
        tf_ms = 300000  # 5 minutes

        for row in partitions:
            symbol = row["symbol"]
            print(f"\n  [SCAN] Scanning {symbol} (5m)...")

            query = """
                SELECT timestamp, next_ts 
                FROM (
                    SELECT timestamp, LEAD(timestamp) OVER (ORDER BY timestamp) as next_ts 
                    FROM symbol_metrics 
                    WHERE symbol=?
                ) 
                WHERE next_ts - timestamp > ?
                AND NOT EXISTS (
                    SELECT 1 FROM unfillable_gaps u
                    WHERE u.table_name='symbol_metrics' AND u.symbol=? AND u.timeframe='5m'
                    AND u.start_ts=timestamp AND u.end_ts=next_ts
                )
            """
            gaps = self.conn.execute(query, (symbol, tf_ms * 1.5, symbol)).fetchall()

            if not gaps:
                print(f"    [OK] No gaps found.")
                continue

            for gap in gaps:
                ts, next_ts = gap["timestamp"], gap["next_ts"]
                missing_count = int((next_ts - ts) // tf_ms - 1)
                print(
                    f"    [GAP] Gap: {pd.to_datetime(ts, unit='ms')} -> {pd.to_datetime(next_ts, unit='ms')} ({missing_count} bars)"
                )

                if not dry_run:
                    inserted = self._patch_daily_zip(
                        table="symbol_metrics",
                        symbol=symbol,
                        ts=ts,
                        next_ts=next_ts,
                        time_col="create_time",
                        data_type="metrics",
                        timeframe="5m",
                        extra_cols={},
                        insert_cols=[
                            "timestamp",
                            "create_time",
                            "symbol",
                            "sum_open_interest",
                            "sum_open_interest_value",
                            "count_toptrader_long_short_ratio",
                            "sum_toptrader_long_short_ratio",
                            "count_long_short_ratio",
                            "sum_long_short_ratio",
                            "count_taker_long_short_vol_ratio",
                            "sum_taker_long_short_vol_ratio",
                        ],
                    )
                    if inserted == 0:
                        print(
                            "    [INFO] Marking gap as unfillable to prevent future re-checks."
                        )
                        self.conn.execute(
                            "INSERT OR IGNORE INTO unfillable_gaps (table_name, symbol, timeframe, start_ts, end_ts) VALUES (?, ?, ?, ?, ?)",
                            ("symbol_metrics", symbol, "5m", ts, next_ts),
                        )
                        self.conn.commit()

    def patch_funding_rate(self, dry_run=True):
        print("\n=== Auditing: funding_rate ===")
        partitions = self.conn.execute(
            "SELECT DISTINCT symbol FROM funding_rate"
        ).fetchall()
        tf_ms = 28800000  # 8 hours

        for row in partitions:
            symbol = row["symbol"]
            print(f"\n  [SCAN] Scanning {symbol} (8h)...")

            query = """
                SELECT calc_time, next_ts 
                FROM (
                    SELECT calc_time, LEAD(calc_time) OVER (ORDER BY calc_time) as next_ts 
                    FROM funding_rate 
                    WHERE symbol=?
                ) 
                WHERE next_ts - calc_time > ?
                AND NOT EXISTS (
                    SELECT 1 FROM unfillable_gaps u
                    WHERE u.table_name='funding_rate' AND u.symbol=? AND u.timeframe='8h'
                    AND u.start_ts=calc_time AND u.end_ts=next_ts
                )
            """
            gaps = self.conn.execute(query, (symbol, tf_ms * 1.5, symbol)).fetchall()

            if not gaps:
                print(f"    [OK] No gaps found.")
                continue

            for gap in gaps:
                ts, next_ts = gap["calc_time"], gap["next_ts"]
                missing_count = int((next_ts - ts) // tf_ms - 1)
                print(
                    f"    [GAP] Gap: {pd.to_datetime(ts, unit='ms')} -> {pd.to_datetime(next_ts, unit='ms')} ({missing_count} bars)"
                )

                if not dry_run:
                    inserted = self._patch_monthly_zip(
                        table="funding_rate",
                        symbol=symbol,
                        ts=ts,
                        next_ts=next_ts,
                        time_col="calc_time",
                        data_type="fundingRate",
                        timeframe="8h",
                        extra_cols={},
                        insert_cols=[
                            "calc_time",
                            "symbol",
                            "funding_interval_hours",
                            "last_funding_rate",
                        ],
                    )
                    if inserted == 0:
                        print(
                            "    [INFO] Marking gap as unfillable to prevent future re-checks."
                        )
                        self.conn.execute(
                            "INSERT OR IGNORE INTO unfillable_gaps (table_name, symbol, timeframe, start_ts, end_ts) VALUES (?, ?, ?, ?, ?)",
                            ("funding_rate", symbol, "8h", ts, next_ts),
                        )
                        self.conn.commit()

    def _patch_daily_zip(
        self,
        table,
        symbol,
        ts,
        next_ts,
        time_col,
        data_type,
        timeframe,
        extra_cols,
        insert_cols,
    ):
        start_date = datetime.fromtimestamp(ts / 1000).date()
        end_date = datetime.fromtimestamp(next_ts / 1000).date()
        total_inserted = 0

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"      [FETCH] Fetching {data_type} Daily ZIP for {date_str}...")

            # Fetch daily data
            df = self.vision.fetch_data(
                symbol,
                timeframe,
                current_date.year,
                current_date.month,
                daily_date=date_str,
                data_type=data_type,
            )

            if not df.empty:
                df_to_insert = df[(df[time_col] > ts) & (df[time_col] < next_ts)].copy()

                if not df_to_insert.empty:
                    df_to_insert["symbol"] = symbol
                    if timeframe:
                        df_to_insert["timeframe"] = timeframe
                    for k, v in extra_cols.items():
                        df_to_insert[k] = v
                    # Ensure timestamp exists if expected
                    if (
                        "timestamp" in insert_cols
                        and "timestamp" not in df_to_insert.columns
                        and time_col in df_to_insert.columns
                    ):
                        df_to_insert["timestamp"] = df_to_insert[time_col]

                    # Filter columns exactly to the schema
                    available_cols = [
                        c for c in insert_cols if c in df_to_insert.columns
                    ]
                    df_to_insert = df_to_insert[available_cols]

                    # M-2 FIX: Use INSERT OR REPLACE for idempotent inserts (safe re-runs)
                    cols = ", ".join(available_cols)
                    placeholders = ", ".join(["?"] * len(available_cols))
                    self.conn.executemany(
                        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
                        df_to_insert.values.tolist(),
                    )
                    self.conn.commit()
                    rows_inserted = len(df_to_insert)
                    total_inserted += rows_inserted
                    print(f"        [OK] Inserted {rows_inserted} rows.")
                else:
                    print(
                        f"        [INFO] ZIP contained no rows within the gap boundaries."
                    )
            else:
                print(f"        [ERROR] ZIP not found or empty.")

            current_date += timedelta(days=1)

        return total_inserted

    def _patch_monthly_zip(
        self,
        table,
        symbol,
        ts,
        next_ts,
        time_col,
        data_type,
        timeframe,
        extra_cols,
        insert_cols,
    ):
        # Funding rates only support monthly zips
        start_date = datetime.fromtimestamp(ts / 1000).date()
        end_date = datetime.fromtimestamp(next_ts / 1000).date()
        total_inserted = 0

        # Iterate over unique months
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            print(
                f"      [FETCH] Fetching {data_type} Monthly ZIP for {current_date.strftime('%Y-%m')}..."
            )

            df = self.vision.fetch_data(
                symbol, "1d", current_date.year, current_date.month, data_type=data_type
            )

            if not df.empty:
                df_to_insert = df[(df[time_col] > ts) & (df[time_col] < next_ts)].copy()

                if not df_to_insert.empty:
                    df_to_insert["symbol"] = symbol
                    for k, v in extra_cols.items():
                        df_to_insert[k] = v

                    # Ensure timestamp exists if expected
                    if (
                        "timestamp" in insert_cols
                        and "timestamp" not in df_to_insert.columns
                        and time_col in df_to_insert.columns
                    ):
                        df_to_insert["timestamp"] = df_to_insert[time_col]

                    available_cols = [
                        c for c in insert_cols if c in df_to_insert.columns
                    ]
                    df_to_insert = df_to_insert[available_cols]

                    # M-2 FIX: Use INSERT OR REPLACE for idempotent inserts (safe re-runs)
                    cols = ", ".join(available_cols)
                    placeholders = ", ".join(["?"] * len(available_cols))
                    self.conn.executemany(
                        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
                        df_to_insert.values.tolist(),
                    )
                    self.conn.commit()
                    rows_inserted = len(df_to_insert)
                    total_inserted += rows_inserted
                    print(f"        [OK] Inserted {rows_inserted} rows.")
                else:
                    print(
                        f"        [INFO] ZIP contained no rows within the gap boundaries."
                    )
            else:
                print(f"        [ERROR] ZIP not found or empty.")

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        return total_inserted


if __name__ == "__main__":
    patcher = UniversalGapPatcher()

    # Run in patch mode to fill the database gaps
    print("========================================")
    print(" UNIVERSAL GAP PATCHER - PATCH MODE ")
    print("========================================")
    patcher.patch_ohlcv(dry_run=False)
    patcher.patch_index_ohlcv(dry_run=False)
    patcher.patch_symbol_metrics(dry_run=False)
    patcher.patch_funding_rate(dry_run=False)

    patcher.close()
