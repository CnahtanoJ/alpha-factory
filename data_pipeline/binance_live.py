import requests
import concurrent.futures
import time


def format_binance_symbol(hl_symbol):
    """Converts Hyperliquid symbol (e.g. BTC, kPEPE) to Binance symbol (e.g. BTCUSDT, 1000PEPEUSDT)."""
    sym = hl_symbol.replace("/", "").replace("-", "")
    if sym.startswith("k") and len(sym) > 1 and sym[1].isupper():
        sym = "1000" + sym[1:]

    sym = sym.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    return sym


def fetch_global_binance_funding():
    """
    Fetches funding rate and index price for all symbols on Binance in a single call.
    Returns a dictionary mapping: {'BTCUSDT': {'funding_rate': 0.0001, 'index_price': 68200.5}, ...}
    """
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                mapping = {}
                for item in data:
                    sym = item.get("symbol")
                    if sym:
                        mapping[sym] = {
                            "funding_rate": float(item.get("lastFundingRate", 0.0)),
                            "index_price": float(item.get("indexPrice", 0.0)),
                        }
                return mapping
    except Exception as e:
        print(f"Error fetching global Binance funding rates: {e}")
    return {}


def fetch_single_derivatives(hl_symbol, period="15m", global_funding_map=None):
    """
    Fetches all derivatives, open interest, taker ratio, and sentiment metrics
    for a single symbol, merging them with the global funding/index map.
    """
    binance_sym = format_binance_symbol(hl_symbol)

    # 1. Grab funding and index price from global map if available
    funding_rate = 0.0
    index_price = 0.0

    if global_funding_map and binance_sym in global_funding_map:
        funding_rate = global_funding_map[binance_sym]["funding_rate"]
        index_price = global_funding_map[binance_sym]["index_price"]

    # Map supported periods to Binance limits
    valid_periods = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
    period_str = period if period in valid_periods else "15m"

    # Define endpoints
    oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
    top_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
    global_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    taker_url = "https://fapi.binance.com/futures/data/takerlongshortRatio"

    params = {"symbol": binance_sym}
    params_period = {"symbol": binance_sym, "period": period_str, "limit": 1}

    oi_usd = 0.0
    top_val = None
    global_val = None
    taker_ratio = None

    # Fetch Open Interest
    try:
        oi_res = requests.get(oi_url, params=params, timeout=5)
        if oi_res.status_code == 200:
            oi_data = oi_res.json()
            open_interest = float(oi_data.get("openInterest", 0.0))
            # Calculate oi_usd to match training set sum_open_interest_value
            # If global index price is missing, use mark price or close price from other sources later
            oi_usd = open_interest * (index_price if index_price > 0 else 1.0)
    except Exception as e:
        print(f"Error fetching open interest for {binance_sym}: {e}")

    # Fetch Top Trader Ratio
    try:
        top_res = requests.get(top_url, params=params_period, timeout=5)
        if top_res.status_code == 200 and len(top_res.json()) > 0:
            top_val = float(top_res.json()[0]["longShortRatio"])
    except Exception as e:
        print(f"Error fetching top trader sentiment for {binance_sym}: {e}")

    # Fetch Global Long/Short Ratio
    try:
        global_res = requests.get(global_url, params=params_period, timeout=5)
        if global_res.status_code == 200 and len(global_res.json()) > 0:
            global_val = float(global_res.json()[0]["longShortRatio"])
    except Exception as e:
        print(f"Error fetching global sentiment for {binance_sym}: {e}")

    # Fetch Taker Volume Ratio (Bugfix feature alignment)
    try:
        taker_res = requests.get(taker_url, params=params_period, timeout=5)
        if taker_res.status_code == 200 and len(taker_res.json()) > 0:
            taker_ratio = float(taker_res.json()[0]["buySellRatio"])
    except Exception as e:
        print(f"Error fetching taker ratio for {binance_sym}: {e}")

    return hl_symbol, {
        "funding_rate": funding_rate,
        "oi_usd": oi_usd,
        "top_long_short": top_val,
        "global_long_short": global_val,
        "taker_buy_sell_ratio": taker_ratio,
        "index_close": index_price,
    }


def get_bulk_binance_derivatives(hl_symbols, period="15m"):
    """
    Concurrently fetches sentiment, open interest, funding, and taker metrics for multiple symbols.
    Uses ThreadPoolExecutor for high-speed parallel fetches.
    """
    # 1. Single global fetch for funding rates & index prices
    global_funding = fetch_global_binance_funding()

    results = {}

    # 2. Concurrently fetch individual symbol derivatives metrics
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_single_derivatives, sym, period, global_funding): sym
            for sym in hl_symbols
        }
        for future in concurrent.futures.as_completed(futures):
            hl_symbol, metrics = future.result()
            results[hl_symbol] = metrics

    return results


if __name__ == "__main__":
    # Quick test
    symbols = ["BTC", "ETH", "kPEPE", "SOL"]
    start = time.time()
    res = get_bulk_binance_derivatives(symbols, period="1h")
    print(f"Fetched {len(res)} symbols in {time.time() - start:.2f} seconds.")
    for k, v in res.items():
        print(f"\n{k}:")
        print(f"  Funding Rate = {v['funding_rate']}")
        print(f"  Index Close  = {v['index_close']}")
        print(f"  OI in USD    = {v['oi_usd']:,}")
        print(f"  Top L/S      = {v['top_long_short']}")
        print(f"  Global L/S   = {v['global_long_short']}")
        print(f"  Taker Ratio  = {v['taker_buy_sell_ratio']}")
