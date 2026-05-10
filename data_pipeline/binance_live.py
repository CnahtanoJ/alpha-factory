import requests
import concurrent.futures
import time

def format_binance_symbol(hl_symbol):
    """Converts Hyperliquid symbol (e.g. BTC, kPEPE) to Binance symbol (e.g. BTCUSDT, 1000PEPEUSDT)."""
    sym = hl_symbol.replace("/", "").replace("-", "")
    if sym.startswith('k') and len(sym) > 1 and sym[1].isupper():
        sym = "1000" + sym[1:]
    
    sym = sym.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    return sym

def fetch_single_sentiment(hl_symbol, period="15m"):
    """Fetches Top Trader and Global Long/Short ratio for a single symbol."""
    binance_sym = format_binance_symbol(hl_symbol)
    
    top_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
    global_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    
    params = {
        "symbol": binance_sym,
        "period": period,
        "limit": 1
    }
    
    try:
        top_res = requests.get(top_url, params=params, timeout=5)
        global_res = requests.get(global_url, params=params, timeout=5)
        
        top_val = None
        global_val = None
        
        if top_res.status_code == 200 and len(top_res.json()) > 0:
            top_val = float(top_res.json()[0]['longShortRatio'])
            
        if global_res.status_code == 200 and len(global_res.json()) > 0:
            global_val = float(global_res.json()[0]['longShortRatio'])
            
        return hl_symbol, top_val, global_val
    except Exception as e:
        print(f"Error fetching live sentiment for {binance_sym}: {e}")
        return hl_symbol, None, None

def get_bulk_binance_sentiment(hl_symbols, period="15m"):
    """Concurrently fetches sentiment data for multiple symbols."""
    results = {}
    
    # 50 symbols * 2 requests = 100 requests. Safe to do in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_single_sentiment, sym, period): sym for sym in hl_symbols}
        for future in concurrent.futures.as_completed(futures):
            hl_symbol, top_val, global_val = future.result()
            results[hl_symbol] = {
                'top_long_short': top_val,
                'global_long_short': global_val
            }
            
    return results

if __name__ == "__main__":
    # Quick test
    symbols = ["BTC", "ETH", "kPEPE", "SOL"]
    start = time.time()
    res = get_bulk_binance_sentiment(symbols)
    print(f"Fetched {len(res)} symbols in {time.time() - start:.2f} seconds.")
    for k, v in res.items():
        print(f"{k}: Top={v['top_long_short']} | Global={v['global_long_short']}")
