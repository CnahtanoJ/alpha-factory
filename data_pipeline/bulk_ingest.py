import os
import sys

# Ensure project root is in path before local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from data_pipeline.binance_vision import BinanceVision
from data_pipeline.data_fetcher import get_exchange, get_top_symbols_by_volume
from data_pipeline.database import get_connection, init_db
from datetime import datetime

def ingest_symbol(symbol, timeframes, start_year=2017, use_spot=False):
    """Ingests multiple timeframes for a single symbol to the local database."""
    init_db()
    vision = BinanceVision(use_spot=use_spot)
    
    # Binance uses BTCUSDT, Hyperliquid often uses just BTC
    binance_symbol = symbol.replace("/", "").replace("-", "")
    if not binance_symbol.endswith("USDT"):
        binance_symbol += "USDT"
        
    market = 'spot' if use_spot else 'futures'
    
    for tf in timeframes:
        print(f"\n>>> [{symbol}] Processing {tf} (Market: {market})")
        
        # Smart Resume Check:
        conn = get_connection()
        cursor = conn.execute(
            "SELECT latest_timestamp FROM sync_state WHERE symbol = ? AND timeframe = ? AND market = ?",
            (symbol, tf, market)
        )
        row = cursor.fetchone()
        
        effective_start_year = start_year
        effective_start_month = 1
        
        if row and row['latest_timestamp']:
            last_dt = datetime.fromtimestamp(row['latest_timestamp'] / 1000)
            effective_start_year = last_dt.year
            effective_start_month = last_dt.month
            print(f"🔄 [{symbol}] Resuming from {effective_start_year}-{effective_start_month:02d}")
        conn.close()

        # 1. Fetch from Binance Vision
        df = vision.fetch_history_range(binance_symbol, tf, start_year=effective_start_year, start_month=effective_start_month)
        
        if not df.empty:
            # 2. Sync to Local Database
            print(f"💾 [{symbol}] Syncing {len(df)} rows to local SQLite database...")
            conn = get_connection()
            insert_data = []
            for _, row in df.iterrows():
                insert_data.append((
                    int(row['timestamp']), symbol, tf, market,
                    float(row['open']), float(row['high']), 
                    float(row['low']), float(row['close']), float(row['volume'])
                ))
            
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                insert_data
            )
            
            # Update sync_state
            earliest = int(df['timestamp'].min())
            latest = int(df['timestamp'].max())
            conn.execute(
                "INSERT OR REPLACE INTO sync_state (symbol, timeframe, market, earliest_timestamp, latest_timestamp) VALUES (?, ?, ?, ?, ?)",
                (symbol, tf, market, earliest, latest)
            )
            
            conn.commit()
            conn.close()
            print(f"✅ [{symbol}] Local database and sync state updated.")
        else:
            print(f"❌ [{symbol}] No new data found on Binance Vision.")

def mass_ingest(count=100, timeframes=['1h'], start_year=2017, use_spot=False):
    """Auto-discovers and ingests top symbols locally."""
    print(f"\n=== Starting Local Mass Ingestion (Top {count} symbols) ===")
    init_db()
    exchange = get_exchange('binance')
    symbols = get_top_symbols_by_volume(exchange, limit=count)
    
    for i, symbol in enumerate(symbols):
        print(f"\nProgress: {i+1}/{len(symbols)}")
        ingest_symbol(symbol, timeframes, start_year, use_spot)
        time.sleep(1.0)

def lambda_handler(event, context):
    """Entry point for AWS Lambda."""
    symbols = event.get('symbols', [])
    top_n = event.get('top_n', 0)
    timeframes = event.get('timeframes', ['1h'])
    start_year = event.get('start_year', 2022)
    use_spot = event.get('use_spot', False)
    
    if top_n > 0:
        mass_ingest(top_n, timeframes, start_year, use_spot)
    else:
        for symbol in symbols:
            ingest_symbol(symbol, timeframes, start_year, use_spot)
            
    return {'statusCode': 200, 'body': "Ingestion complete"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk Ingest Historical Data to Local DB")
    parser.add_argument("symbols", nargs="*", help="List of symbols (optional if --top is used)")
    parser.add_argument("--top", type=int, help="Fetch top N symbols by volume")
    parser.add_argument("--timeframes", default="1h,5m", help="Comma-separated timeframes")
    parser.add_argument("--start_year", type=int, default=2017, help="Year to start from")
    parser.add_argument("--spot", action="store_true", help="Use Spot data instead of Futures")
    
    args = parser.parse_args()
    tfs = args.timeframes.split(",")
    
    if args.top:
        mass_ingest(args.top, tfs, args.start_year, args.spot)
    elif args.symbols:
        for s in args.symbols:
            ingest_symbol(s, tfs, args.start_year, args.spot)
    else:
        parser.print_help()
