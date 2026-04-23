import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.bulk_refactor_ingest import run_refactored_ingest
from data_pipeline.database import get_connection

def verify_data():
    conn = get_connection()
    print("\n--- Verifying Data ---")
    
    # Check ohlcv
    cnt = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    print(f"OHLCV rows: {cnt}")
    
    # Check index_ohlcv
    cnt = conn.execute("SELECT COUNT(*) FROM index_ohlcv").fetchone()[0]
    print(f"Index OHLCV rows: {cnt}")
    
    # Check symbol_metrics
    cnt = conn.execute("SELECT COUNT(*) FROM symbol_metrics").fetchone()[0]
    print(f"Symbol Metrics rows: {cnt}")
    
    if cnt > 0:
        sample = conn.execute("SELECT * FROM symbol_metrics LIMIT 1").fetchone()
        print(f"Sample Metric: {dict(sample)}")
        
    conn.close()

if __name__ == "__main__":
    # Run for just top 1 asset, starting 2025-01 for speed (less data)
    print("Running quick test for Top 1 asset...")
    run_refactored_ingest(top_n_hl=1, top_n_binance=1, start_year=2025)
    verify_data()
