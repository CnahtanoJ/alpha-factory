import sqlite3
import sys
from datetime import datetime

# Fix for Windows terminal encoding
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = r"E:\trading_data\alpha_factory.db"

def run_db_audit():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # 1. Row count and variety
    total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"[STATUS] Total Rows: {total:,}")
    
    # 2. Check for out-of-range timestamps again
    VALID_MIN = int(datetime(2017, 1, 1).timestamp() * 1000)
    VALID_MAX = int(datetime(2027, 1, 1).timestamp() * 1000)
    
    bad_ts = conn.execute(f"SELECT COUNT(*) FROM ohlcv WHERE timestamp < {VALID_MIN} OR timestamp > {VALID_MAX}").fetchone()[0]
    print(f"🛑 Rogue Timestamps: {bad_ts}")
    
    # 3. Check for non-ASCII symbols (Junk discovery)
    symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()
    junk = []
    for s in symbols:
        sym = s['symbol']
        if not all(ord(c) < 128 for c in sym):
            junk.append(sym)
    
    print(f"👻 Junk Symbols found: {len(junk)}")
    if junk:
        print(f"   Example junk: {junk[:5]}")
        
    # 4. Check for timeframe gaps or overlaps
    print("\n🔍 Checking for data overlaps (Duplicates)...")
    overlaps = conn.execute("""
        SELECT symbol, timeframe, market, timestamp, COUNT(*) as cnt 
        FROM ohlcv 
        GROUP BY symbol, timeframe, market, timestamp 
        HAVING cnt > 1 
        LIMIT 5
    """).fetchall()
    
    if overlaps:
        print(f"⚠️ Warning: Found duplicate records (e.g., {overlaps[0]['symbol']} @ {overlaps[0]['timestamp']})")
    else:
        print("✅ No duplicate records found.")

    # 5. Check Sync State alignment
    print("\n🔄 Verifying Sync State consistency...")
    orphans = conn.execute("""
        SELECT s.symbol, s.timeframe FROM sync_state s
        LEFT JOIN ohlcv o ON s.symbol = o.symbol AND s.timeframe = o.timeframe AND s.market = o.market
        WHERE o.symbol IS NULL
    """).fetchall()
    
    if orphans:
        print(f"⚠️ Sync State contains {len(orphans)} entries with no data in ohlcv table.")
    else:
        print("✅ Sync state is perfectly aligned with data table.")

    conn.close()

if __name__ == "__main__":
    run_db_audit()
