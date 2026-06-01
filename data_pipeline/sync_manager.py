"""
Sync Manager — Local Data Synchronization Engine.

Syncs historical data (OHLCV, Index Klines, Metrics, Funding Rate)
to the local SQLite database using Binance Vision (free CSV archives).

For gap-filling, run universal_gap_patcher.py after ingestion.
"""

import time
import pandas as pd
from data_pipeline.database import get_connection
from data_pipeline.binance_vision import BinanceVision
from datetime import datetime, timedelta


class SyncManager:
    def __init__(self):
        self.conn = get_connection()
        
    def get_sync_state(self, symbol, timeframe, market, data_type='klines'):
        cursor = self.conn.execute(
            "SELECT earliest_timestamp, latest_timestamp FROM sync_state WHERE symbol = ? AND timeframe = ? AND market = ? AND data_type = ?",
            (symbol, timeframe, market, data_type)
        )
        row = cursor.fetchone()
        if row:
            return row['earliest_timestamp'], row['latest_timestamp']
        return None, None

    def update_sync_state(self, symbol, timeframe, market, data_type, earliest, latest):
        self.conn.execute(
            "INSERT OR REPLACE INTO sync_state (symbol, timeframe, market, data_type, earliest_timestamp, latest_timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, timeframe, market, data_type, earliest, latest)
        )
        self.conn.commit()

    def sync_from_binance_vision(self, symbol, timeframe='1h', market='futures', start_year=2020, data_type='klines'):
        """
        Downloads bulk historical data from Binance Vision (free public CSVs).
        This is the primary method for deep historical backfills.
        """
        print(f"📦 Downloading {symbol} ({timeframe}) [{market}] {data_type} from Binance Vision...")
        vision = BinanceVision()
        
        # Check if we already have some data (smart resume)
        earliest, latest = self.get_sync_state(symbol, timeframe, market, data_type)
        effective_start_year = start_year
        effective_start_month = 1
        effective_start_day = 1
        
        if latest:
            try:
                last_dt = datetime.fromtimestamp(latest / 1000)
                effective_start_year = last_dt.year
                effective_start_month = last_dt.month
                effective_start_day = last_dt.day
                print(f"  🔄 Resuming from {effective_start_year}-{effective_start_month:02d}-{effective_start_day:02d}")
            except (OSError, ValueError):
                print(f"  ⚠️ Corrupted sync state detected for {symbol}. Starting from scratch.")
                effective_start_year = start_year
                effective_start_month = 1
                effective_start_day = 1
        
        # Binance Vision uses clean symbol format (BTCUSDT)
        binance_symbol = symbol.replace("/", "").replace("-", "")
        
        # Map Hyperliquid 'k' prefix (kPEPE) to Binance '1000' prefix (1000PEPE)
        if binance_symbol.startswith('k') and len(binance_symbol) > 1 and binance_symbol[1].isupper():
            binance_symbol = "1000" + binance_symbol[1:]
            
        if not binance_symbol.endswith("USDT"):
            binance_symbol += "USDT"
        
        df = vision.fetch_history_range(
            binance_symbol, timeframe, 
            start_year=effective_start_year, 
            start_month=effective_start_month,
            start_day=effective_start_day,
            data_type=data_type
        )
        
        if df.empty:
            print(f"  ⚠️ No data found on Binance Vision for {symbol} ({data_type}).")
            # Mark it as checked up to today to prevent 72-month re-scans on every run
            now_ms = int(datetime.now().timestamp() * 1000)
            
            # Preserve earliest if we already had some data, otherwise use now_ms
            cur_earliest, _ = self.get_sync_state(symbol, timeframe, market, data_type)
            final_earliest = cur_earliest if cur_earliest else now_ms
            self.update_sync_state(symbol, timeframe, market, data_type, final_earliest, now_ms)
            return 0
        
        # Valid range: 2017-01-01 to 2027-01-01
        valid_min_ms = 1483228800000
        valid_max_ms = 1798761600000
        
        # Bulk insert to SQLite, aggressively filtering bad timestamps
        insert_data = []
        for _, row in df.iterrows():
            raw_ts = row.get('timestamp')
            if pd.isna(raw_ts):
                raw_ts = row.get('calc_time')
            if pd.isna(raw_ts):
                raw_ts = row.get('create_time')
                
            if pd.isna(raw_ts):
                continue
                
            ts = int(raw_ts)
            
            # Normalize 16-digit microseconds to 13-digit milliseconds
            if ts > 9999999999999:
                ts = ts // 1000
            
            # Normalize 10-digit seconds to 13-digit milliseconds
            if ts < 9999999999:
                ts = ts * 1000
                
            if valid_min_ms <= ts <= valid_max_ms:
                if data_type == 'klines':
                    insert_data.append(
                        (ts, symbol, timeframe, market,
                         float(row['open']), float(row['high']),
                         float(row['low']), float(row['close']), float(row['volume']))
                    )
                elif data_type == 'indexPriceKlines':
                    insert_data.append(
                        (ts, symbol, timeframe,
                         float(row['open']), float(row['high']),
                         float(row['low']), float(row['close']))
                    )
                elif data_type == 'metrics':
                    insert_data.append(
                        (ts, int(row.get('create_time', ts)), symbol,
                         float(row.get('sum_open_interest', 0)), float(row.get('sum_open_interest_value', 0)),
                         float(row.get('count_toptrader_long_short_ratio', 0)), float(row.get('sum_toptrader_long_short_ratio', 0)),
                         float(row.get('count_long_short_ratio', 0)), float(row.get('sum_long_short_ratio', 0)),
                         float(row.get('count_taker_long_short_vol_ratio', 0)), float(row.get('sum_taker_long_short_vol_ratio', 0)))
                    )
                elif data_type == 'fundingRate':
                    insert_data.append(
                        (ts, symbol, int(row['funding_interval_hours']), float(row['last_funding_rate']))
                    )
        
        if not insert_data:
            print(f"  ⚠️ No valid timestamps found in {symbol} Binance Vision data.")
            return 0
        
        if data_type == 'klines':
            self.conn.executemany(
                "INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                insert_data
            )
        elif data_type == 'indexPriceKlines':
            self.conn.executemany(
                "INSERT OR REPLACE INTO index_ohlcv (timestamp, symbol, timeframe, open, high, low, close) VALUES (?, ?, ?, ?, ?, ?, ?)",
                insert_data
            )
        elif data_type == 'metrics':
            self.conn.executemany(
                "INSERT OR REPLACE INTO symbol_metrics (timestamp, create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_long_short_ratio, count_taker_long_short_vol_ratio, sum_taker_long_short_vol_ratio) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                insert_data
            )
        elif data_type == 'fundingRate':
            self.conn.executemany(
                "INSERT OR REPLACE INTO funding_rate (calc_time, symbol, funding_interval_hours, last_funding_rate) VALUES (?, ?, ?, ?)",
                insert_data
            )
        
        # Update sync state based ONLY on the valid, filtered inserted data
        valid_timestamps = [x[0] for x in insert_data]
        new_earliest = min(valid_timestamps)
        new_latest = max(valid_timestamps)
        
        cur_earliest, cur_latest = self.get_sync_state(symbol, timeframe, market, data_type)
        final_earliest = min(new_earliest, cur_earliest) if cur_earliest else new_earliest
        final_latest = max(new_latest, cur_latest) if cur_latest else new_latest
        
        self.update_sync_state(symbol, timeframe, market, data_type, final_earliest, final_latest)
        self.conn.commit()
        
        print(f"  ✅ Synced {len(insert_data)} rows of {data_type} from Binance Vision.")
        return len(insert_data)



    def sync_symbol(self, symbol, timeframe='1h', market='futures', target_years=3, start_year=2020, **kwargs):
        """
        Full sync for a single symbol:
          1. Bulk download from Binance Vision (free, fast)
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"  Syncing: {symbol} | {timeframe} | {market}")
        print(f"{'='*60}")
        
        # Step 1: Binance Vision (bulk historical) — always fetches all data types for futures
        self.sync_from_binance_vision(symbol, timeframe, market, start_year, data_type='klines')
        self.sync_from_binance_vision(symbol, timeframe, market, start_year, data_type='indexPriceKlines')
        self.sync_from_binance_vision(symbol, timeframe, market, start_year, data_type='metrics')
        self.sync_from_binance_vision(symbol, timeframe, market, start_year, data_type='fundingRate')
        
        # Step 2 is removed (no longer using CCXT). 
        # For gap-filling, run universal_gap_patcher.py instead.
        
        # Summary
        earliest, latest = self.get_sync_state(symbol, timeframe, market)
        if earliest and latest:
            try:
                start = datetime.fromtimestamp(earliest/1000).strftime('%Y-%m-%d')
            except (OSError, ValueError):
                start = "UNKNOWN"
            try:
                end = datetime.fromtimestamp(latest/1000).strftime('%Y-%m-%d')
            except (OSError, ValueError):
                end = "UNKNOWN"
            
            cursor = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM ohlcv WHERE symbol = ? AND timeframe = ? AND market = ?",
                (symbol, timeframe, market)
            )
            count = cursor.fetchone()['cnt']
            print(f"  📊 Total: {count:,} candles from {start} to {end}")
            
        elapsed = time.time() - start_time
        print(f"  ⏱️ Time Spent: {elapsed:.2f} seconds")

    def bulk_sync(self, symbols, timeframe='1h', market='futures', target_years=3, start_year=2020, **kwargs):
        """Syncs multiple symbols sequentially."""
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] ", end="")
            self.sync_symbol(symbol, timeframe, market, target_years, start_year)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    sync = SyncManager()
    sync.sync_symbol('BTC/USDT', timeframe='1h', market='futures', target_years=1)
    sync.close()
