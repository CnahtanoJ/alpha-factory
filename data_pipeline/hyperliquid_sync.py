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
        universe = [
            asset["name"]
            for asset in meta["universe"]
            if not asset.get("isDelisted", False)
        ]
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
        for asset in meta["universe"]:
            if not asset.get("isDelisted", False):
                name = asset["name"]
                canonical = f"{name}/USDT"
                symbol_map[canonical] = name

        return symbol_map
    except Exception as e:
        print(f"Error building HL symbol map: {e}")
        return {}


def get_latest_candles(symbol, interval="1h", limit=200, testnet=False, info=None):
    """
    Fetches the latest OHLCV candles from Hyperliquid REST API (perpetual futures).
    """
    hl_api_string = symbol.split("/")[0]

    if info is None:
        url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        info = Info(url, skip_ws=True)

    end_time = int(time.time() * 1000)

    for attempt in range(3):
        try:
            candles = info.candles_snapshot(hl_api_string, interval, 0, end_time)
            if candles:
                latest = candles[-limit:]
                formatted = []
                for c in latest:
                    formatted.append(
                        {
                            "timestamp": c["t"],
                            "open": float(c["o"]),
                            "high": float(c["h"]),
                            "low": float(c["l"]),
                            "close": float(c["c"]),
                            "volume": float(c["v"]),
                            "symbol": symbol,
                            "timeframe": interval,
                            "market": "futures",
                        }
                    )
                return formatted

            # If empty but no error, wait and retry
            time.sleep(1.0 * (2**attempt))

        except Exception as e:
            # Handle 429 Rate Limit specifically
            error_str = str(e)
            if "429" in error_str:
                # Heavy backoff for 429
                wait_time = 5.0 * (2**attempt)
                print(f"Rate limit hit for {symbol}. Backing off {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(
                    f"Error fetching HL candles for {symbol} ({hl_api_string}) [Attempt {attempt+1}]: {e}"
                )
                time.sleep(1.0 * (2**attempt))

    return []


def get_bulk_latest_candles(
    symbols, interval="15m", limit=200, testnet=False, delay=1.5
):
    """
    Fetches candles for multiple symbols strictly sequentially with a delay between every request.
    This is slower but ensures maximum data integrity and avoids rate limit issues.
    """
    results = {}
    url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    shared_info = Info(url, skip_ws=True)

    total = len(symbols)
    for i, sym in enumerate(symbols):
        print(f"Fetching {sym} ({i+1}/{total})...")
        sym, candles = sym, get_latest_candles(
            sym, interval, limit, testnet, info=shared_info
        )
        if candles:
            results[sym] = candles

        # Mandatory delay between EVERY single request
        if i < total - 1:
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
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            # meta = data[0], asset_ctxs = data[1]
            meta = data[0]
            ctxs = data[1]

            assets = []
            for i, asset_meta in enumerate(meta["universe"]):
                name = asset_meta["name"]
                ctx = ctxs[i]
                # dayNtlVlm is the 24h volume
                day_vol = float(ctx.get("dayNtlVlm", 0))
                assets.append({"name": name, "volume": day_vol})

            assets.sort(key=lambda x: x["volume"], reverse=True)

            return [a["name"] for a in assets[:limit]]
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
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            meta = data[0]
            ctxs = data[1]

            for i, asset_meta in enumerate(meta["universe"]):
                name = asset_meta["name"]
                ctx = ctxs[i]
                live_data[name] = {
                    "funding": float(ctx.get("funding", 0)),
                    "openInterest": float(ctx.get("openInterest", 0)),
                    "oraclePx": float(ctx.get("oraclePx", 0)),
                }
    except Exception as e:
        print(f"Error fetching live meta ctx: {e}")

    return live_data


if __name__ == "__main__":
    uni = get_hyperliquid_universe()
    print(f"Found {len(uni)} active Hyperliquid markets.")
    test_candles = get_latest_candles("BTC", "1h", 5)
    print(f"Latest BTC 1h candles: {len(test_candles)}")
