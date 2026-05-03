import time
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from hyperliquid.info import Info
from hyperliquid.utils import constants

def get_hyperliquid_universe(testnet=False):
    """
    Fetches the live list of tradable perpetual contracts on Hyperliquid.
    """
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(url, skip_ws=True)
    
    try:
        meta = info.meta()
        universe = [asset['name'] for asset in meta['universe'] if not asset.get('isDelisted', False)]
        return universe
    except Exception as e:
        print(f"Error fetching Hyperliquid universe: {e}")
        return []

def get_hl_symbol_map(testnet=False):
    """
    Builds a lookup table mapping canonical symbol names
    to HL API strings for perpetual futures.
    """
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(url, skip_ws=True)
    
    symbol_map = {}
    
    try:
        meta = info.meta()
        for asset in meta['universe']:
            if not asset.get('isDelisted', False):
                name = asset['name']
                canonical = f"{name}/USDT"
                symbol_map[canonical] = name

        return symbol_map
    except Exception as e:
        print(f"Error building HL symbol map: {e}")
        return {}

def get_latest_candles(symbol, interval='1h', limit=100, testnet=False):
    """
    Fetches the latest OHLCV candles from Hyperliquid REST API (perpetual futures).
    Symbol can be either 'BTC' or 'BTC/USDT' format.
    """
    # Normalize: accept both 'BTC' and 'BTC/USDT'
    hl_api_string = symbol.split('/')[0]
    
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(url, skip_ws=True)
    
    # HL expects timestamps in ms
    end_time = int(time.time() * 1000)
    
    for attempt in range(3):
        try:
            candles = info.candles_snapshot(hl_api_string, interval, 0, end_time)
            if candles:
                # Take the most recent 'limit'
                latest = candles[-limit:]
                
                formatted = []
                for c in latest:
                    formatted.append({
                        'timestamp': c['t'],
                        'open': float(c['o']),
                        'high': float(c['h']),
                        'low': float(c['l']),
                        'close': float(c['c']),
                        'volume': float(c['v']),
                        'symbol': symbol,
                        'timeframe': interval,
                        'market': 'futures'
                    })
                return formatted
            
            # If empty, sleep and retry
            time.sleep(1.0 * (2 ** attempt))
            
        except Exception as e:
            print(f"Error fetching HL candles for {symbol} ({hl_api_string}) [Attempt {attempt+1}]: {e}")
            time.sleep(1.0 * (2 ** attempt))
            
    return []

def get_bulk_latest_candles(symbols, interval='15m', limit=100, testnet=False, batch_size=10, delay=0.5):
    """
    Fetches candles for multiple symbols concurrently in batches.
    Significantly faster than sequential fetching while respecting rate limits.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    results = {}
    
    def fetch_single(sym):
        return sym, get_latest_candles(sym, interval, limit, testnet)

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(fetch_single, sym) for sym in batch]
            for future in futures:
                try:
                    sym, candles = future.result()
                    if candles:
                        results[sym] = candles
                except Exception as e:
                    print(f"Bulk fetch error for {sym}: {e}")
                    
        # Rate limiting pause between batches
        if i + batch_size < len(symbols):
            time.sleep(delay)
            
    return results

def get_hl_top_by_volume(limit=100):
    """
    Fetches the top N assets on Hyperliquid by 24h volume.
    Returns a list of symbol names.
    """
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            data = response.json()
            # meta = data[0], asset_ctxs = data[1]
            meta = data[0]
            ctxs = data[1]
            
            assets = []
            for i, asset_meta in enumerate(meta['universe']):
                name = asset_meta['name']
                ctx = ctxs[i]
                # dayNtlVlm is the 24h volume
                day_vol = float(ctx.get('dayNtlVlm', 0))
                assets.append({'name': name, 'volume': day_vol})
                
            assets.sort(key=lambda x: x['volume'], reverse=True)
                
            return [a['name'] for a in assets[:limit]]
    except Exception as e:
        print(f"Error fetching top HL assets: {e}")
    return []

def get_live_meta_ctx():
    """
    Fetches the live metaAndAssetCtxs and returns a dictionary mapped by symbol name.
    Useful for live inference of funding, OI, and oracle price.
    Returns: {'BTC': {'funding': 0.0001, 'openInterest': 100.5, 'oraclePx': 65000.0}, ...}
    """
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}
    
    live_data = {}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            data = response.json()
            meta = data[0]
            ctxs = data[1]
            
            for i, asset_meta in enumerate(meta['universe']):
                name = asset_meta['name']
                ctx = ctxs[i]
                live_data[name] = {
                    'funding': float(ctx.get('funding', 0)),
                    'openInterest': float(ctx.get('openInterest', 0)),
                    'oraclePx': float(ctx.get('oraclePx', 0))
                }
    except Exception as e:
        print(f"Error fetching live meta ctx: {e}")
        
    return live_data

if __name__ == "__main__":
    uni = get_hyperliquid_universe()
    print(f"Found {len(uni)} active Hyperliquid markets.")
    test_candles = get_latest_candles("BTC", "1h", 5)
    print(f"Latest BTC 1h candles: {len(test_candles)}")
