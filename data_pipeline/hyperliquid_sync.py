from hyperliquid.info import Info
from hyperliquid.utils import constants
import time
import pandas as pd

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
    Builds a lookup table mapping (human_symbol, market) 
    to HL API strings (BTC, @107).
    """
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(url, skip_ws=True)
    
    symbol_map = {} # (Canonical, Market) -> HL_String
    
    try:
        # 1. Map Perps (Futures)
        meta = info.meta()
        for asset in meta['universe']:
            if not asset.get('isDelisted', False):
                name = asset['name']
                canonical = f"{name}/USDT"
                symbol_map[(canonical, 'futures')] = name
        
        # 2. Map Spot (Dynamic Discovery)
        sm = info.spot_meta()
        all_spot_tokens = {t['name'] for t in sm['tokens']}
        tokens_by_index = {t['index']: t['name'] for t in sm['tokens']}
        
        for idx, pair in enumerate(sm['universe']):
            api_name = pair['name'] # e.g. "PURR/USDC" or "@1"
            pair_index = pair['index']
            base_token_idx = pair['tokens'][0]
            base_l1_name = tokens_by_index.get(base_token_idx, "UNKNOWN")
            
            # Discovery Logic: Map human names to L1 names
            # If the exchange uses 'UBTC' but we want 'BTC', we check if 'UBTC' is the L1 name.
            # We look for:
            #   - Direct match (HYPE -> HYPE)
            #   - Unified match (BTC -> UBTC)
            #   - Purr match (PURR -> PURR)
            
            human_name = base_l1_name
            if base_l1_name.startswith('U') and len(base_l1_name) > 1:
                potential_human = base_l1_name[1:] # e.g. UBTC -> BTC
                # If the human name exists in Perps (like BTC), it's a valid remapping
                if any(asset['name'] == potential_human for asset in meta['universe']):
                    human_name = potential_human

            canonical = f"{human_name}/USDT"
            api_string = api_name if "/" in api_name else f"@{pair_index}"
            symbol_map[(canonical, 'spot')] = api_string

        return symbol_map
    except Exception as e:
        print(f"Error building HL symbol map: {e}")
        return {}

def get_latest_candles(symbol, interval='1h', limit=100, testnet=False, hl_map=None, market='futures'):
    """
    Fetches the latest OHLCV candles from Hyperliquid REST API.
    Handles the @index mapping for spot and direct names for perps.
    """
    if hl_map is None:
        hl_map = get_hl_symbol_map(testnet)
        
    hl_api_string = hl_map.get((symbol, market))
    if not hl_api_string:
        # Fallback for common perps if market is unspecified or missing
        if market == 'futures':
             hl_api_string = symbol.split('/')[0]
        else:
            return []
    
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(url, skip_ws=True)
    
    # HL expects timestamps in ms. Get approximately 'limit' candles back.
    # The API finds candles that END before or at the endTime.
    end_time = int(time.time() * 1000)
    
    try:
        # candle_snapshot returns a list of candles. NOTE: Typo fixed from 'candle_snapshot'
        candles = info.candles_snapshot(hl_api_string, interval, 0, end_time)
        if not candles: return []
        
        # Take the most recent 'limit'
        latest = candles[-limit:]
        
        # Format for our DB: timestamp, open, high, low, close, volume, symbol, timeframe, market
        formatted = []
        for c in latest:
            formatted.append({
                'timestamp': c['t'], # Opening time
                'open': float(c['o']),
                'high': float(c['h']),
                'low': float(c['l']),
                'close': float(c['c']),
                'volume': float(c['v']),
                'symbol': symbol,      # Preserve canonical symbol (slash format)
                'timeframe': interval,
                'market': market        # Preserve explicit market
            })
        return formatted
    except Exception as e:
        print(f"Error fetching HL candles for {symbol} ({hl_api_string}): {e}")
        return []

if __name__ == "__main__":
    uni = get_hyperliquid_universe()
    print(f"Found {len(uni)} active Hyperliquid markets.")
    test_candles = get_latest_candles("BTC", "1h", 5)
    print(f"Latest BTC 1h candles: {len(test_candles)}")
