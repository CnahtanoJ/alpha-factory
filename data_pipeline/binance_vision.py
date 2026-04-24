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
        
        # Column Definitions
        self.COLUMNS = {
            'klines': [
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
            ],
            'indexPriceKlines': [
                'timestamp', 'open', 'high', 'low', 'close', 'ignore_vol',
                'close_time', 'ignore_qav', 'ignore_trades', 'ignore_tbbav', 
                'ignore_tbqav', 'ignore_last'
            ],
            'metrics': [
                'create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value',
                'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                'count_long_short_ratio', 'sum_long_short_ratio',
                'count_taker_long_short_vol_ratio', 'sum_taker_long_short_vol_ratio',
                'timestamp'
            ],
            'fundingRate': [
                'calc_time', 'funding_interval_hours', 'last_funding_rate'
            ]
        }

    def _get_url(self, symbol, timeframe, year, month=None, daily_date=None, data_type='klines'):
        """Constructs the Binance Vision URL for monthly or daily data."""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        
        # Metrics and IndexPriceKlines have different folder structures
        if data_type == 'metrics':
            if daily_date:
                return f"{self.base_url}/daily/metrics/{clean_symbol}/{clean_symbol}-metrics-{daily_date}.zip"
            else:
                month_str = f"{month:02d}"
                return f"{self.base_url}/monthly/metrics/{clean_symbol}/{clean_symbol}-metrics-{year}-{month_str}.zip"
                
        # FundingRate is monthly only and has no timeframe parameter
        if data_type == 'fundingRate':
            month_str = f"{month:02d}"
            return f"{self.base_url}/monthly/fundingRate/{clean_symbol}/{clean_symbol}-fundingRate-{year}-{month_str}.zip"
        
        # For klines and indexPriceKlines
        if daily_date:
            return f"{self.base_url}/daily/{data_type}/{clean_symbol}/{timeframe}/{clean_symbol}-{timeframe}-{daily_date}.zip"
        else:
            month_str = f"{month:02d}"
            return f"{self.base_url}/monthly/{data_type}/{clean_symbol}/{timeframe}/{clean_symbol}-{timeframe}-{year}-{month_str}.zip"

    def fetch_data(self, symbol, timeframe, year, month=None, daily_date=None, data_type='klines'):
        """Downloads, unzips, and parses historical data into a DataFrame."""
        url = self._get_url(symbol, timeframe, year, month, daily_date, data_type)
        print(f"Downloading: {url}")
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                return pd.DataFrame()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f, header=None)
            
            # Map columns
            expected_cols = self.COLUMNS.get(data_type, [])
            df.columns = expected_cols[:len(df.columns)]
            
            # Convert timestamp columns to numeric
            ts_cols = ['timestamp', 'create_time', 'calc_time']
            for col in ts_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error fetching {data_type} from Binance Vision: {e}")
            return pd.DataFrame()

    def fetch_history_range(self, symbol, timeframe, start_year=2020, start_month=1, data_type='klines'):
        """Fetches multiple months of data and returns a combined DataFrame."""
        all_dfs = []
        now = datetime.now()
        
        for year in range(start_year, now.year + 1):
            m_start = start_month if year == start_year else 1
            for month in range(m_start, 13):
                if year == now.year and month > now.month:
                    break
                
                # Try monthly first
                df = self.fetch_data(symbol, timeframe, year, month, data_type=data_type)
                
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  Collected Monthly {data_type}: {year}-{month:02d}")
                else:
                    # Daily fallback for current/last year (not for fundingRate which is monthly only)
                    if year >= now.year - 1 and data_type != 'fundingRate':
                        print(f"  Monthly missing for {year}-{month:02d}. Trying Daily...")
                        for day in range(1, 32):
                            try:
                                test_date = datetime(year, month, day)
                            except ValueError: break
                            
                            if test_date >= now - timedelta(days=1): break
                                
                            daily_str = test_date.strftime("%Y-%m-%d")
                            df_daily = self.fetch_data(symbol, timeframe, year, month, daily_date=daily_str, data_type=data_type)
                            if not df_daily.empty:
                                all_dfs.append(df_daily)
                
                time.sleep(0.2) 
        
        if not all_dfs:
            return pd.DataFrame()
            
        # Standardize on 'timestamp' for sorting and dedup
        sort_col = 'timestamp' 
        if 'timestamp' not in all_dfs[0].columns:
            sort_col = 'calc_time' if 'calc_time' in all_dfs[0].columns else 'create_time'
        
        return pd.concat(all_dfs).drop_duplicates(subset=[sort_col]).sort_values(sort_col)

if __name__ == "__main__":
    vision = BinanceVision()
    # Test klines
    df = vision.fetch_data("BTCUSDT", "1h", 2024, 1, data_type='klines')
    print(f"Klines: {len(df)}")
    # Test metrics
    df_m = vision.fetch_data("BTCUSDT", "1h", 2024, 1, data_type='metrics')
    print(f"Metrics: {len(df_m)}")
