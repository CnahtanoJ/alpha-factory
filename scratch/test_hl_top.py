import requests
import json

def get_top_hl_assets(limit=100):
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        data = response.json()
        # The data is a list [meta, asset_contexts]
        # meta is at index 0, asset_contexts at index 1
        meta = data[0]
        ctxs = data[1]
        
        assets = []
        for i, asset_meta in enumerate(meta['universe']):
            name = asset_meta['name']
            ctx = ctxs[i]
            # Volume is 24h day volume
            day_vol = float(ctx.get('dayNtlVlm', 0))
            assets.append({'name': name, 'volume': day_vol})
            
        # Sort by volume descending
        assets.sort(key=lambda x: x['volume'], reverse=True)
        return assets[:limit]
    else:
        print(f"Error: {response.status_code}")
        return []

if __name__ == "__main__":
    top_assets = get_top_hl_assets(10)
    for i, asset in enumerate(top_assets):
        print(f"{i+1}. {asset['name']}: {asset['volume']}")
