import sys
import os

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

sys.path.append(os.getcwd())

from data_pipeline.hyperliquid_sync import get_hl_symbol_map, get_latest_candles
import sqlite3
from data_pipeline.database import DB_PATH

def test_dynamic_remapping():
    print("🔍 DYNAMIC DIAGNOSTIC: Testing Automatic Remapping (U-Tokens)...")
    
    # 1. Build Map
    hl_map = get_hl_symbol_map()
    print(f"✅ Mapping Engine: Loaded {len(hl_map)} symbols.")
    
    # Check these specific targets
    test_cases = [
        ("BTC/USDT", "spot"),
        ("ETH/USDT", "spot"),
        ("SOL/USDT", "spot"),
        ("BTC/USDT", "futures"),
        ("PURR/USDT", "spot")
    ]
    
    for sym, market in test_cases:
        hl_api_string = hl_map.get((sym, market))
        if hl_api_string:
            print(f"📍 {sym} ({market}) -> API: {hl_api_string}")
        else:
            print(f"❌ {sym} ({market}) -> [NOT FOUND]")

    # 2. Try an actual fetch for one of them
    target = ("SOL/USDT", "spot")
    print(f"\n📡 Testing Live Fetch for {target[0]} ({target[1]})...")
    candles = get_latest_candles(target[0], interval='1h', limit=1, hl_map=hl_map, market=target[1])
    if candles:
        c = candles[0]
        print(f"   ✅ Data Received: {c['symbol']} | market: {c['market']} | close: {c['close']}")
    else:
        print(f"   ❌ Fetch failed for {target[0]}. Check if token exists in HL spotMeta.")

if __name__ == "__main__":
    test_dynamic_remapping()
