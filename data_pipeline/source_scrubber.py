import sqlite3
import pandas as pd
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.getcwd())

from data_pipeline.database import DB_PATH
from data_pipeline.sync_manager import SyncManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SourceScrubber")

class SourceScrubber:
    def __init__(self):
        self.db_path = DB_PATH
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    def scrub_and_restore(self, lookback_candles=500):
        """
        1. Identifies symbols in the database.
        2. Deletes the most recent N candles for each.
        3. Calls SyncManager to fill the gaps from Binance.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Get all partitions
        cursor.execute("SELECT DISTINCT symbol, timeframe, market FROM ohlcv")
        partitions = cursor.fetchall()
        
        logger.info(f"🔍 Found {len(partitions)} market partitions to scrub.")

        for symbol, timeframe, market in partitions:
            logger.info(f"🧹 Scrubbing {symbol} [{timeframe}]...")

            # Find the split point
            cursor.execute(f"""
                SELECT timestamp FROM ohlcv 
                WHERE symbol = ? AND timeframe = ? AND market = ?
                ORDER BY timestamp DESC LIMIT 1 OFFSET {lookback_candles}
            """, (symbol, timeframe, market))
            
            result = cursor.fetchone()
            if result:
                split_timestamp = result[0]
                # Delete anything newer than the split point
                cursor.execute(f"""
                    DELETE FROM ohlcv 
                    WHERE symbol = ? AND timeframe = ? AND market = ?
                    AND timestamp > ?
                """, (symbol, timeframe, market, split_timestamp))
                logger.info(f"  ✅ Deleted rows newer than {split_timestamp}")
            else:
                logger.warning(f"  ⚠️ Not enough candles to scrub for {symbol} {timeframe}. Skipping.")

        conn.commit()
        conn.close()
        logger.info("✨ Scrubbing complete. Starting Binance Restoration...")

        # 2. Restoration Phase
        sync = SyncManager()
        for symbol, timeframe, market in partitions:
            logger.info(f"🔄 Restoring {symbol} [{timeframe}] from Binance...")
            try:
                # SyncManager uses Binance by default if not specified otherwise
                sync.sync_from_exchange(symbol, timeframe=timeframe, market=market)
            except Exception as e:
                logger.error(f"  ❌ Failed to restore {symbol}: {e}")

        logger.info("🏁 Database Restoration Finished.")

if __name__ == "__main__":
    scrubber = SourceScrubber()
    scrubber.scrub_and_restore(lookback_candles=500)
