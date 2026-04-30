import requests
import json
import os

def generate_top_100_file():
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            data = response.json()
            meta = data[0]
            ctxs = data[1]
            
            assets = []
            for i, asset_meta in enumerate(meta['universe']):
                name = asset_meta['name']
                ctx = ctxs[i]
                day_vol = float(ctx.get('dayNtlVlm', 0))
                assets.append({'name': name, 'volume': day_vol})
                
            assets.sort(key=lambda x: x['volume'], reverse=True)
            top_100 = assets[:100]
            
            output_file = "top_100_hl_volume.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# Top 100 Hyperliquid Assets by 24h Volume\n\n")
                f.write("| Rank | Asset | 24h Volume (USDC) |\n")
                f.write("|------|-------|-------------------|\n")
                for rank, asset in enumerate(top_100, 1):
                    f.write(f"| {rank} | **{asset['name']}** | ${asset['volume']:,.2f} |\n")
            
            print(f"Successfully wrote top 100 volumes to {output_file}")
        else:
            print(f"Failed to fetch data from Hyperliquid. Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_top_100_file()
