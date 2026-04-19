import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data_pipeline.database import get_connection
from data_pipeline.s3_storage import S3Storage

def push_to_s3():
    """Reads all data from local SQLite and pushes/merges to S3 Master Files."""
    print("🚀 Starting Cloud Sync: Local DB -> S3 Master Files")
    
    conn = get_connection()
    storage = S3Storage()
    
    # 1. Discover all symbol/timeframe/market triples in the database
    cursor = conn.execute("SELECT DISTINCT symbol, timeframe, market FROM ohlcv")
    entries = cursor.fetchall()
    
    if not entries:
        print("❌ No data found in local database.")
        return

    print(f"📊 Found {len(entries)} symbol/timeframe/market combinations to sync.")
    
    for entry in entries:
        symbol = entry['symbol']
        tf = entry['timeframe']
        market = entry['market']
        
        print(f"\n>>> Preparing {symbol} ({tf}) - Market: {market}...")
        
        # 2. Fetch all data for this specific entry from SQLite
        query = "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? AND market = ? ORDER BY timestamp"
        df = pd.read_sql_query(query, conn, params=(symbol, tf, market))
        
        if df.empty:
            continue
            
        # 3. Direct Overwrite Upload
        storage.upload_ohlcv(df, symbol, tf, market)
        print(f"✅ {symbol} ({tf}) [{market}] successfully backed up to S3 ({len(df)} rows).")

    conn.close()
    print("\n✨ Cloud Sync Complete!")

if __name__ == "__main__":
    push_to_s3()
