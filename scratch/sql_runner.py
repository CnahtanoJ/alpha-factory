import sqlite3
import pandas as pd
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from data_pipeline.database import DB_PATH

def run_query(sql_file_path):
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # We'll run the 'Gap Finder' query which is query #1 in the .sql file
    query = """
    SELECT 
        symbol, 
        timeframe, 
        datetime(timestamp/1000, 'unixepoch') as gap_starts_at,
        datetime(next_ts/1000, 'unixepoch') as gap_ends_at,
        (next_ts - timestamp) / 60000 as gap_minutes
    FROM (
        SELECT 
            symbol, timeframe, timestamp,
            LEAD(timestamp) OVER (PARTITION BY symbol, timeframe ORDER BY timestamp) as next_ts
        FROM ohlcv
    ) 
    WHERE next_ts IS NOT NULL 
      AND (next_ts - timestamp) > (
          CASE timeframe 
              WHEN '15m' THEN 15 
              WHEN '1h' THEN 60 
              WHEN '4h' THEN 240 
              ELSE 1440 
          END
      ) * 60000
    LIMIT 15;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("Audit Result: No gaps found in the first 15 partitions.")
        else:
            # Simple text output instead of markdown
            print(df.to_string(index=False))
    except Exception as e:
        print(f"SQL Execution Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_query(None)
