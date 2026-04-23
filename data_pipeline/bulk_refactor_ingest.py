import sys
import os
import pandas as pd
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.hyperliquid_sync import get_hl_top_by_volume
from data_pipeline.binance_vision import BinanceVision
from data_pipeline.database import get_connection, init_db

def run_refactored_ingest(top_n_hl=100, top_n_binance=100, start_year=2024):
    """
    1. Get top 100 Hyperliquid coins by volume.
    2. Get Binance Vision data for the top 10 of those.
    """
    print(f"--- Starting Refactored Ingest Flow ---")
    init_db()
    
    # 1. Get Top 100 HL
    print(f"Fetching top {top_n_hl} HL assets...")
    hl_top_all = get_hl_top_by_volume(limit=top_n_hl)
    if not hl_top_all:
        print("Failed to fetch HL assets.")
        return
    
    # 2. Focus on Top 10
    target_hl = hl_top_all[:top_n_binance]
    print(f"Targeting top {top_n_binance} for Binance Vision: {target_hl}")
    
    vision = BinanceVision(use_spot=False) # UM Futures
    conn = get_connection()
    
    for symbol in target_hl:
        binance_symbol = f"{symbol}USDT"
        print(f"\nProcessing {symbol} (Binance: {binance_symbol})...")
        
        # A. Klines
        print(f"  Fetching Klines...")
        df_klines = vision.fetch_history_range(binance_symbol, "1h", start_year=start_year, data_type='klines')
        if not df_klines.empty:
            df_klines['symbol'] = f"{symbol}/USDT" 
            df_klines['timeframe'] = '1h'
            df_klines['market'] = 'futures'
            # Filter to match DB schema
            cols = ['timestamp', 'symbol', 'timeframe', 'market', 'open', 'high', 'low', 'close', 'volume']
            df_klines[cols].to_sql('ohlcv', conn, if_exists='append', index=False)
            print(f"    Saved {len(df_klines)} klines.")
        
        # B. Index Price Klines
        print(f"  Fetching Index Price Klines...")
        df_index = vision.fetch_history_range(binance_symbol, "1h", start_year=start_year, data_type='indexPriceKlines')
        if not df_index.empty:
            df_index['symbol'] = f"{symbol}/USDT"
            df_index['timeframe'] = '1h'
            cols = ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close']
            df_index[cols].to_sql('index_ohlcv', conn, if_exists='append', index=False)
            print(f"    Saved {len(df_index)} index klines.")
            
        # C. Metrics
        print(f"  Fetching Metrics...")
        df_metrics = vision.fetch_history_range(binance_symbol, "1h", start_year=start_year, data_type='metrics')
        if not df_metrics.empty:
            df_metrics['symbol'] = f"{symbol}/USDT"
            cols = [
                'timestamp', 'create_time', 'symbol', 'sum_open_interest', 
                'sum_open_interest_value', 'count_toptrader_long_short_ratio', 
                'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 
                'sum_long_short_ratio', 'count_taker_long_short_vol_ratio', 
                'sum_taker_long_short_vol_ratio'
            ]
            df_metrics[cols].to_sql('symbol_metrics', conn, if_exists='append', index=False)
            print(f"    Saved {len(df_metrics)} metrics rows.")

    conn.close()
    print(f"\n✅ Refactored Ingest Complete.")

if __name__ == "__main__":
    # Test with top 100 HL, but only ingest top 1 for speed in this demo
    # The user wanted top 100 HL -> top 10 Binance.
    run_refactored_ingest(top_n_hl=100, top_n_binance=10, start_year=2024)
