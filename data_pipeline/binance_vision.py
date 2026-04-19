import requests
import pandas as pd
import io
import zipfile
import time
from datetime import datetime, timedelta

class BinanceVision:
    def __init__(self, use_spot=False):
        self.market_type = "spot" if use_spot else "futures/um"
        self.base_url = f"https://data.binance.vision/data/{self.market_type}"
        
        # Binance Vision standard columns
        # Futures USD-M usually has 12, Spot often has 11. We use a slice.
        self.FUTURES_COLUMNS = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ]

    def _get_url(self, symbol, timeframe, year, month=None, daily_date=None):
        """Constructs the Binance Vision URL for monthly or daily klines."""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        
        if daily_date:
            # Daily: .../daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-01.zip
            return f"{self.base_url}/daily/klines/{clean_symbol}/{timeframe}/{clean_symbol}-{timeframe}-{daily_date}.zip"
        else:
            # Monthly: .../monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip
            month_str = f"{month:02d}"
            return f"{self.base_url}/monthly/klines/{clean_symbol}/{timeframe}/{clean_symbol}-{timeframe}-{year}-{month_str}.zip"

    def fetch_klines(self, symbol, timeframe, year, month=None, daily_date=None):
        """Downloads, unzips, and parses historical klines into a DataFrame."""
        url = self._get_url(symbol, timeframe, year, month, daily_date)
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch data (Status {response.status_code}). URL might not exist yet.")
                return pd.DataFrame()

            # Unzip in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Expecting exactly one CSV inside
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Load into Pandas
                    # We force header=None because Binance Vision CSVs are inconsistent.
                    # We then manually assign column names.
                    df = pd.read_csv(f, header=None)
            
            # Standard Binance OHLCV columns
            # [0:Open time, 1:Open, 2:High, 3:Low, 4:Close, 5:Volume, 6:Close time, ...]
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore']
            df.columns = cols[:len(df.columns)]
            
            # Clean up: only keep core columns to save space
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Basic cleanup: Convert to numeric
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Binance Vision: {e}")
            return pd.DataFrame()

    def fetch_history_range(self, symbol, timeframe, start_year=2017, start_month=1):
        """Fetches multiple months of data and returns a combined DataFrame."""
        all_dfs = []
        now = datetime.now()
        
        # Start from start_year up to current month
        for year in range(start_year, now.year + 1):
            # If we are in the start_year, start from start_month. Otherwise, start from Jan (1).
            m_start = start_month if year == start_year else 1
            
            for month in range(m_start, 13):
                # Don't fetch future months
                if year == now.year and month > now.month:
                    break
                
                # Attempt to fetch monthly file first
                df = self.fetch_klines(symbol, timeframe, year, month)
                
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  Collected Monthly ZIP: {year}-{month:02d}")
                else:
                    # If monthly fails and it's a recent/ongoing month, try Daily files
                    if year >= now.year - 1: # Only try daily for the last 2 years
                        print(f"  Monthly ZIP missing for {year}-{month:02d}. Attempting Daily Bridge...")
                        
                        # Loop through days 1 to 31
                        for day in range(1, 32):
                            # Don't fetch today's date (it's not archived yet)
                            try:
                                test_date = datetime(year, month, day)
                            except ValueError:
                                break # Invalid date (e.g., Feb 30)
                            
                            if test_date >= now - timedelta(days=1):
                                break
                                
                            daily_str = test_date.strftime("%Y-%m-%d")
                            df_daily = self.fetch_klines(symbol, timeframe, year, month, daily_date=daily_str)
                            
                            if not df_daily.empty:
                                all_dfs.append(df_daily)
                                print(f"    + Daily ZIP: {daily_str}")
                            
                # Small sleep between months
                time.sleep(0.5) 
        
        if not all_dfs:
            return pd.DataFrame()
            
        return pd.concat(all_dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

if __name__ == "__main__":
    vision = BinanceVision()
    # Test with a single month
    df = vision.fetch_klines("BTCUSDT", "1h", 2024, 1)
    if not df.empty:
        print(f"Success! Sample BTC data:\n{df.head()}")
    else:
        print("Data fetch failed or empty.")
