"""
Clean bad timestamps and junk symbols from the E:\trading_data database.
"""
import sqlite3
import os
import sys
from datetime import datetime

# Windows encoding fix
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = r"E:\trading_data\alpha_factory.db"

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL;")

# 1. Check the range of timestamps
print("=== Timestamp Analysis ===")
ts_range = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv").fetchone()
print(f"  Min timestamp: {ts_range[0]}")
print(f"  Max timestamp: {ts_range[1]}")

# Valid range: 2017-01-01 to 2026-12-31 in milliseconds
VALID_MIN = int(datetime(2017, 1, 1).timestamp() * 1000)   # ~1483228800000
VALID_MAX = int(datetime(2027, 1, 1).timestamp() * 1000)    # ~1798761600000
print(f"  Valid range: {VALID_MIN} to {VALID_MAX}")

bad_ts = conn.execute(f"SELECT COUNT(*) FROM ohlcv WHERE timestamp < {VALID_MIN} OR timestamp > {VALID_MAX}").fetchone()[0]
print(f"  Rows with out-of-range timestamps: {bad_ts:,}")

# 2. Check junk symbols
print("\n=== Symbol Analysis ===")
symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
print(f"  Total unique symbols: {len(symbols)}")

junk_symbols = []
for row in symbols:
    sym = row[0]
    # Check for non-ASCII characters (junk/test symbols)
    if not all(ord(c) < 128 for c in sym):
        count = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol = ?", (sym,)).fetchone()[0]
        junk_symbols.append((sym, count))
        print(f"  JUNK: '{sym}' ({count:,} rows)")

# 3. Clean
print("\n=== Cleaning ===")

# Remove junk symbols
for sym, count in junk_symbols:
    print(f"  Removing '{sym}' ({count:,} rows)...")
    conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (sym,))

# Remove bad timestamps
if bad_ts > 0:
    print(f"  Removing {bad_ts:,} rows with bad timestamps...")
    conn.execute(f"DELETE FROM ohlcv WHERE timestamp < {VALID_MIN} OR timestamp > {VALID_MAX}")

conn.commit()

# 4. Rebuild sync_state
print("\n  Rebuilding sync_state...")
conn.execute("DELETE FROM sync_state")
conn.execute("""
    INSERT INTO sync_state (symbol, timeframe, market, earliest_timestamp, latest_timestamp)
    SELECT symbol, timeframe, market, MIN(timestamp), MAX(timestamp)
    FROM ohlcv
    GROUP BY symbol, timeframe, market
""")
conn.commit()

# 5. Final count
total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
sync = conn.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0]
print(f"\n  Final total candles: {total:,}")
print(f"  Sync state entries: {sync}")

# 6. Verify timestamps are now valid
ts_range = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv").fetchone()
start = datetime.fromtimestamp(ts_range[0]/1000).strftime('%Y-%m-%d')
end = datetime.fromtimestamp(ts_range[1]/1000).strftime('%Y-%m-%d')
print(f"  Date range: {start} to {end}")

conn.close()
print("\n✅ Cleanup complete!")
