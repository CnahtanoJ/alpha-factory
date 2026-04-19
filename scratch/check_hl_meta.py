from hyperliquid.info import Info
from hyperliquid.utils import constants

def check_structure():
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    meta = info.meta()
    if meta and 'universe' in meta:
        delisted_assets = [a for a in meta['universe'] if a.get('isDelisted')]
        if delisted_assets:
            print(f"Sample delisted asset keys: {delisted_assets[0].keys()}")
            print(f"Sample delisted asset: {delisted_assets[0]}")

if __name__ == "__main__":
    check_structure()
