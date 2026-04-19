"""
Sync Manager — Local Data Synchronization Engine.

Syncs historical OHLCV data to the local SQLite database using:
  1. Binance Vision (free CSV archives - bulk historical)
  2. CCXT Exchange API (live data + gap filling)

No S3 involved in data storage. S3 is only used by the bot for
storing the blueprint (strategy_config.json).
"""

import time
import pandas as pd
from data_pipeline.database import get_connection
from data_pipeline.data_fetcher import get_exchange, get_top_symbols_by_volume
from data_pipeline.binance_vision import BinanceVision
from datetime import datetime, timedelta


class SyncManager:
    def __init__(self, exchange_id='binance'):
        self.exchange = get_exchange(exchange_id)
        self.conn = get_connection()
        
    def get_sync_state(self, symbol, timeframe, market):
        cursor = self.conn.execute(
            "SELECT earliest_timestamp, latest_timestamp FROM sync_state WHERE symbol = ? AND timeframe = ? AND market = ?",
            (symbol, timeframe, market)
        )
        row = cursor.fetchone()
        if row:
            return row['earliest_timestamp'], row['latest_timestamp']
        return None, None

    def update_sync_state(self, symbol, timeframe, market, earliest, latest):
        self.conn.execute(
            "INSERT OR REPLACE INTO sync_state (symbol, timeframe, market, earliest_timestamp, latest_timestamp) VALUES (?, ?, ?, ?, ?)",
            (symbol, timeframe, market, earliest, latest)
        )
        self.conn.commit()

    def sync_from_binance_vision(self, symbol, timeframe='1h', market='futures', start_year=2020):
        """
        Downloads bulk historical data from Binance Vision (free public CSVs).
        This is the primary method for deep historical backfills.
        """
        print(f"📦 Downloading {symbol} ({timeframe}) [{market}] from Binance Vision...")
        use_spot = (market == 'spot')
        vision = BinanceVision(use_spot=use_spot)
        
        # Check if we already have some data (smart resume)
        earliest, latest = self.get_sync_state(symbol, timeframe, market)
        effective_start_year = start_year
        effective_start_month = 1
        
        if latest:
            try:
                last_dt = datetime.fromtimestamp(latest / 1000)
                effective_start_year = last_dt.year
                effective_start_month = last_dt.month
                print(f"  🔄 Resuming from {effective_start_year}-{effective_start_month:02d}")
            except (OSError, ValueError):
                print(f"  ⚠️ Corrupted sync state detected for {symbol}. Starting from scratch.")
                effective_start_year = start_year
                effective_start_month = 1
        
        # Binance Vision uses clean symbol format (BTCUSDT)
        binance_symbol = symbol.replace("/", "").replace("-", "")
        if not binance_symbol.endswith("USDT"):
            binance_symbol += "USDT"
        
        df = vision.fetch_history_range(
            binance_symbol, timeframe, 
            start_year=effective_start_year, 
            start_month=effective_start_month
        )
        
        if df.empty:
            print(f"  ⚠️ No data found on Binance Vision for {symbol}.")
            return 0
        
        # Valid range: 2017-01-01 to 2027-01-01
        valid_min_ms = 1483228800000
        valid_max_ms = 1798761600000
        
        # Bulk insert to SQLite, aggressively filtering bad timestamps
        insert_data = []
        for _, row in df.iterrows():
            ts = int(row['timestamp'])
            
            # Normalize 16-digit microseconds to 13-digit milliseconds
            if ts > 9999999999999:
                ts = ts // 1000
                
            if valid_min_ms <= ts <= valid_max_ms:
                insert_data.append(
                    (ts, symbol, timeframe, market,
                     float(row['open']), float(row['high']),
                     float(row['low']), float(row['close']), float(row['volume']))
                )
        
        if not insert_data:
            print(f"  ⚠️ No valid timestamps found in {symbol} Binance Vision data.")
            return 0
        
        self.conn.executemany(
            "INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            insert_data
        )
        
        # Update sync state based ONLY on the valid, filtered inserted data
        valid_timestamps = [x[0] for x in insert_data]
        new_earliest = min(valid_timestamps)
        new_latest = max(valid_timestamps)
        
        cur_earliest, cur_latest = self.get_sync_state(symbol, timeframe, market)
        final_earliest = min(new_earliest, cur_earliest) if cur_earliest else new_earliest
        final_latest = max(new_latest, cur_latest) if cur_latest else new_latest
        
        self.update_sync_state(symbol, timeframe, market, final_earliest, final_latest)
        self.conn.commit()
        
        print(f"  ✅ Synced {len(insert_data)} rows from Binance Vision.")
        return len(insert_data)

    def sync_from_exchange(self, symbol, timeframe='1h', market='futures', target_years=3):
        """
        Fills gaps using the live CCXT exchange API.
        Walks backwards from the earliest known data point.
        """
        print(f"🔄 Gap-filling {symbol} ({timeframe}) [{market}] from exchange API...")
        
        self.exchange.options['defaultType'] = 'spot' if market == 'spot' else 'future'
        
        target_delta = timedelta(days=target_years * 365)
        now_ms = int(datetime.now().timestamp() * 1000)
        target_start_ms = now_ms - int(target_delta.total_seconds() * 1000)
        
        earliest, latest = self.get_sync_state(symbol, timeframe, market)
        limit = 1000
        total_synced = 0
        
        valid_min_ms = 1483228800000
        valid_max_ms = 1798761600000
        ms_per_candle = self.exchange.parse_timeframe(timeframe) * 1000

        # === 1. FORWARD GAP FILL (From last archive up to right now) ===
        if latest and latest < now_ms - ms_per_candle:
            print("  Forward gap-filling up to current minute...")
            current_forward = latest
            failsafe = 0
            while current_forward < now_ms and failsafe < 500:
                failsafe += 1
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_forward, limit=limit)
                    if not ohlcv:
                        break
                    
                    new_data = [c for c in ohlcv if c[0] > latest and valid_min_ms <= c[0] <= valid_max_ms]
                    if not new_data:
                        current_forward = ohlcv[-1][0] + ms_per_candle if ohlcv else current_forward + limit * ms_per_candle
                        continue
                        
                    insert_data = [(c[0], symbol, timeframe, market, c[1], c[2], c[3], c[4], c[5]) for c in new_data]
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        insert_data
                    )
                    self.conn.commit()
                    
                    # Update state
                    new_latest = max(new_data, key=lambda x: x[0])[0]
                    self.update_sync_state(symbol, timeframe, market, earliest, new_latest)
                    latest = new_latest
                    current_forward = new_latest
                    total_synced += len(new_data)
                    
                    time.sleep(self.exchange.rateLimit / 1000)
                except Exception as e:
                    print(f"  Error during forward exchange sync: {e}")
                    break

        # === 2. BACKWARD GAP FILL (To hit the 3-year target depth) ===
        current_since = earliest if earliest else now_ms
        
        while current_since > target_start_ms:
            ms_per_candle = self.exchange.parse_timeframe(timeframe) * 1000
            fetch_since = current_since - (limit * ms_per_candle)
            
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
                if not ohlcv:
                    print("  No more historical data from exchange.")
                    break
                
                # Filter for valid timestamps within [2017, 2027] AND before our current earliest
                upper_bound = earliest if earliest else now_ms + 1
                
                new_data = [
                    c for c in ohlcv 
                    if c[0] < upper_bound and valid_min_ms <= c[0] <= valid_max_ms
                ]
                
                if not new_data:
                    current_since = fetch_since
                    continue
                
                insert_data = [(c[0], symbol, timeframe, market, c[1], c[2], c[3], c[4], c[5]) for c in new_data]
                self.conn.executemany(
                    "INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    insert_data
                )
                self.conn.commit()
                
                new_earliest = min(new_data, key=lambda x: x[0])[0]
                if not earliest or new_earliest < earliest:
                    earliest = new_earliest
                if not latest:
                    latest = max(new_data, key=lambda x: x[0])[0]
                
                self.update_sync_state(symbol, timeframe, market, earliest, latest)
                current_since = earliest
                total_synced += len(new_data)
                
                try:
                    readable_date = datetime.fromtimestamp(earliest/1000).strftime('%Y-%m-%d')
                except (OSError, ValueError):
                    readable_date = "UNKNOWN"
                print(f"  Synced back to {readable_date} ({total_synced} total rows)")
                
                if earliest <= target_start_ms:
                    break
                    
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"  Error during exchange sync: {e}")
                break
        
        return total_synced

    def sync_symbol(self, symbol, timeframe='1h', market='futures', target_years=3, start_year=2020):
        """
        Full sync for a single symbol:
          1. Bulk download from Binance Vision (free, fast)
          2. Fill remaining gaps from exchange API
        """
        print(f"\n{'='*60}")
        print(f"  Syncing: {symbol} | {timeframe} | {market}")
        print(f"{'='*60}")
        
        # Step 1: Binance Vision (bulk historical)
        self.sync_from_binance_vision(symbol, timeframe, market, start_year)
        
        # Step 2: Exchange API (gap fill)
        self.sync_from_exchange(symbol, timeframe, market, target_years)
        
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

    def bulk_sync(self, symbols, timeframe='1h', market='futures', target_years=3, start_year=2020):
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
