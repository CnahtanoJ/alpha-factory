import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def get_exchange(exchange_id='binance'):
    """Initialize exchange with CCXT."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        'enableRateLimit': True,
    })

def discover_symbols(exchange, quote='USDT'):
    """Fetch all active spot trading pairs for a given quote currency."""
    exchange.load_markets()
    symbols = [
        symbol for symbol, market in exchange.markets.items()
        if market['quote'] == quote and market['spot'] and market['active']
    ]
    return symbols

def get_top_symbols_by_volume(exchange, limit=250):
    """Fetch the top symbols by 24h trading volume."""
    tickers = exchange.fetch_tickers()
    # Filter for USDT spot pairs
    usdt_tickers = {s: t for s, t in tickers.items() if s.endswith('/USDT') and 'quoteVolume' in t}
    # Sort by quoteVolume descending
    sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)
    return [s[0] for s in sorted_tickers[:limit]]

def fetch_ohlcv_with_pagination(exchange, symbol, timeframe='4h', limit=1000, max_candles=5000):
    """
    Fetch OHLCV data with pagination to get deep history.
    """
    all_ohlcv = []
    end_time = exchange.milliseconds()
    
    print(f"Fetching data for {symbol} on {exchange.id}...")
    
    while len(all_ohlcv) < max_candles:
        # Calculate since (startTime) to fetch older data
        # We fetch backwards from 'since'
        # For the first call, we can just use the most recent or calculate a starting point
        
        # Determine how many milliseconds to go back based on timeframe and limit
        # This is a bit tricky, easier to use CCXT's 'since' correctly.
        # Most exchanges return data *after* 'since'.
        # To fetch backwards, we need to know the startTime of the earliest data point we have.
        
        if not all_ohlcv:
            # First fetch: get the most recent data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        else:
            # Subsequent fetch: fetch data before our earliest data point
            earliest_timestamp = all_ohlcv[0][0]
            # Estimate how far back to ask
            # millisecond duration of timeframe (approx)
            duration = exchange.parse_timeframe(timeframe) * 1000
            since = earliest_timestamp - (limit * duration)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        if not ohlcv:
            break
            
        # Filter out candles we already have (just in case)
        new_candles = [c for c in ohlcv if not any(c[0] == existing[0] for existing in all_ohlcv)]
        if not new_candles:
            break
            
        all_ohlcv = new_candles + all_ohlcv
        # Sort by timestamp
        all_ohlcv.sort(key=lambda x: x[0])
        
        print(f"Fetched {len(all_ohlcv)} candles so far...")
        
        # Avoid hitting rate limits too hard
        time.sleep(exchange.rateLimit / 1000)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    ex = get_exchange('binance')
    data = fetch_ohlcv_with_pagination(ex, 'BTC/USDT', timeframe='1h', max_candles=2000)
    print(data.tail())
    print(f"Total rows: {len(data)}")
