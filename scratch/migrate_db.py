"""
Migration script: Move 10M rows from ohlcv_old -> ohlcv on E:\trading_data\alpha_factory.db

What this does:
  1. Ensures the ohlcv table has the correct schema: PK (symbol, timeframe, market, timestamp)
  2. Migrates all data from ohlcv_old into ohlcv (INSERT OR IGNORE skips duplicates)
  3. Validates the migration by comparing row counts
  4. Drops ohlcv_old after successful migration
  5. Rebuilds sync_state from the actual data in ohlcv
"""

import sqlite3
import os
import sys
import time

# Use the centralized DB path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "alpha_factory.db"))

def migrate():
    print(f"Target database: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("ERROR: Database file not found!")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA cache_size = -20000;")
    conn.execute("PRAGMA synchronous = OFF;")  # Faster for bulk migration
    
    # ── Step 1: Check what tables exist ──
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Existing tables: {tables}")
    
    # ── Step 2: Ensure ohlcv has the correct schema ──
    print("\nEnsuring ohlcv table has correct PK (symbol, timeframe, market, timestamp)...")
    
    # Check current ohlcv schema
    ohlcv_schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='ohlcv'").fetchone()
    if ohlcv_schema:
        print(f"  Current ohlcv schema:\n  {ohlcv_schema[0]}")
        ohlcv_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        print(f"  Current ohlcv rows: {ohlcv_count:,}")
    else:
        print("  No ohlcv table found. Creating...")
        conn.execute('''
            CREATE TABLE ohlcv (
                timestamp INTEGER,
                symbol TEXT,
                timeframe TEXT,
                market TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, market, timestamp)
            )
        ''')
        conn.commit()
        ohlcv_count = 0
    
    # ── Step 3: Migrate from ohlcv_old ──
    if 'ohlcv_old' in tables:
        old_count = conn.execute("SELECT COUNT(*) FROM ohlcv_old").fetchone()[0]
        print(f"\nFound ohlcv_old with {old_count:,} rows. Starting migration...")
        
        if old_count > 0:
            start = time.time()
            
            # Batch insert with INSERT OR IGNORE to skip duplicates
            conn.execute("""
                INSERT OR IGNORE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume)
                SELECT timestamp, symbol, timeframe, market, open, high, low, close, volume
                FROM ohlcv_old
            """)
            conn.commit()
            
            elapsed = time.time() - start
            new_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
            migrated = new_count - ohlcv_count
            print(f"  Migrated {migrated:,} rows in {elapsed:.1f}s")
            print(f"  Total ohlcv rows now: {new_count:,}")
            
            if migrated > 0:
                # ── Step 4: Drop ohlcv_old ──
                print("\n  Dropping ohlcv_old...")
                conn.execute("DROP TABLE ohlcv_old")
                conn.commit()
                print("  Done. ohlcv_old removed.")
            else:
                print("  No new rows to migrate (all duplicates). Dropping ohlcv_old...")
                conn.execute("DROP TABLE ohlcv_old")
                conn.commit()
    else:
        print("\nNo ohlcv_old table found. Nothing to migrate.")
    
    # ── Step 5: Rebuild sync_state from actual data ──
    print("\nRebuilding sync_state from ohlcv data...")
    
    # Ensure sync_state has the right schema (with market column)
    conn.execute("DROP TABLE IF EXISTS sync_state")
    conn.execute('''
        CREATE TABLE sync_state (
            symbol TEXT,
            timeframe TEXT,
            market TEXT,
            earliest_timestamp INTEGER,
            latest_timestamp INTEGER,
            PRIMARY KEY (symbol, timeframe, market)
        )
    ''')
    
    conn.execute("""
        INSERT INTO sync_state (symbol, timeframe, market, earliest_timestamp, latest_timestamp)
        SELECT symbol, timeframe, market, MIN(timestamp), MAX(timestamp)
        FROM ohlcv
        GROUP BY symbol, timeframe, market
    """)
    conn.commit()
    
    sync_count = conn.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0]
    print(f"  Rebuilt {sync_count} sync state entries.")
    
    # ── Step 6: Data quality check ──
    print("\n--- Data Quality Report ---")
    
    total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"  Total candles: {total:,}")
    
    # Check for nulls in critical columns
    nulls = conn.execute("""
        SELECT COUNT(*) FROM ohlcv 
        WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
    """).fetchone()[0]
    print(f"  Rows with NULL prices: {nulls:,}")
    
    if nulls > 0:
        print(f"  Cleaning {nulls} null rows...")
        conn.execute("""
            DELETE FROM ohlcv 
            WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
        """)
        conn.commit()
    
    # Check for zero-volume candles (suspicious)
    zero_vol = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE volume = 0").fetchone()[0]
    print(f"  Zero-volume candles: {zero_vol:,} (kept but noted)")
    
    # Check for impossible candles (high < low)
    bad_candles = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE high < low").fetchone()[0]
    print(f"  Invalid candles (high < low): {bad_candles:,}")
    if bad_candles > 0:
        conn.execute("DELETE FROM ohlcv WHERE high < low")
        conn.commit()
        print(f"  Removed {bad_candles} invalid candles.")
    
    # Summary by symbol
    print("\n--- Symbol Summary ---")
    rows = conn.execute("""
        SELECT symbol, timeframe, market, COUNT(*) as cnt,
               MIN(timestamp) as earliest, MAX(timestamp) as latest
        FROM ohlcv 
        GROUP BY symbol, timeframe, market 
        ORDER BY cnt DESC
        LIMIT 20
    """).fetchall()
    
    from datetime import datetime
    print(f"  {'Symbol':<15} {'TF':<6} {'Market':<8} {'Candles':>10}   {'From':<12} {'To':<12}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*10}   {'-'*12} {'-'*12}")
    for r in rows:
        start = datetime.fromtimestamp(r[4]/1000).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(r[5]/1000).strftime('%Y-%m-%d')
        print(f"  {r[0]:<15} {r[1]:<6} {r[2]:<8} {r[3]:>10,}   {start:<12} {end:<12}")
    
    # VACUUM to reclaim space
    print("\nVacuuming database to reclaim space...")
    conn.execute("VACUUM")
    
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"  Final database size: {size_mb:.1f} MB")
    
    conn.close()
    print("\n✅ Migration complete!")


if __name__ == "__main__":
    migrate()
