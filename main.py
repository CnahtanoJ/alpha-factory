import argparse
import sys
from data_pipeline.sync_manager import SyncManager
from backtester.laboratory import Laboratory
from analytics.llm_analyzer import get_llm_verdict
from analytics.analytics import add_indicators, calculate_correlation, calculate_seasonality
from data_pipeline.data_fetcher import get_exchange, fetch_ohlcv_with_pagination

def main():
    parser = argparse.ArgumentParser(description="Alpha Factory: Trading Intelligence Engine")
    parser.add_argument("--mode", type=str, required=True, choices=['sync', 'lab', 'analyze'], 
                        help="Mode: sync (download data), lab (find blueprint), analyze (deep dive)")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated symbols")
    parser.add_argument("--all", action="store_true", help="Sync/Analyze all discovered symbols")
    parser.add_argument("--limit", type=int, default=100, help="Symbol limit for bulk operations")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--years", type=int, default=3, help="Years of history for sync")
    
    args = parser.parse_args()
    symbol_list = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.mode == 'sync':
        sync = SyncManager()
        if args.all:
            sync.bulk_sync_all(timeframe=args.timeframe, target_years=args.years, volume_limit=args.limit)
        else:
            for symbol in symbol_list:
                sync.sync_symbol(symbol, timeframe=args.timeframe, target_years=args.years)
        sync.close()

    elif args.mode == 'lab':
        lab = Laboratory()
        symbols = 'all' if args.all else symbol_list
        lab.generate_blueprint(symbols, timeframe=args.timeframe)


    elif args.mode == 'analyze':
        # Legacy deep dive
        symbol = symbol_list[0]
        exchange = get_exchange('binance')
        df = fetch_ohlcv_with_pagination(exchange, symbol, timeframe=args.timeframe, max_candles=1000)
        df = add_indicators(df)
        seasonality = calculate_seasonality(df)
        # Passing compressed context to LLM
        context = f"Symbol: {symbol}, Seasonality: {seasonality}"
        report = get_llm_verdict(context)
        print(report)

if __name__ == "__main__":
    main()
