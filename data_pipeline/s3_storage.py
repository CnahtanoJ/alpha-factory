import boto3
import pandas as pd
import io
import os
from dotenv import load_dotenv

load_dotenv()

class S3Storage:
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "flaminghotcheetos")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

    def _get_key(self, symbol, timeframe, market):
        """Standardized naming for Master Files."""
        # Sanitize symbol for filename (e.g. BTC/USDT -> BTCUSDT)
        clean_symbol = symbol.replace("/", "").replace("-", "")
        # Prefix with market type to keep directories clean
        return f"historical-data/{market}/{clean_symbol}/{timeframe}/{clean_symbol}-{timeframe}.parquet"

    def upload_ohlcv(self, df, symbol, timeframe, market):
        """Uploads a DataFrame to S3 as a Parquet file."""
        key = self._get_key(symbol, timeframe, market)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow')
        buffer.seek(0)
        
        print(f"Uploading {len(df)} rows to s3://{self.bucket_name}/{key}...")
        self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=buffer.getvalue())

    def get_latest_timestamp(self, symbol, timeframe, market):
        """Checks S3 to find the latest timestamp for a given symbol/timeframe."""
        key = self._get_key(symbol, timeframe, market)
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=key)
            df = self.download_ohlcv(symbol, timeframe, market)
            if not df.empty:
                return int(df['timestamp'].max())
            return None
        except self.s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] in ["404", "NoSuchKey"]:
                return None
            raise e

    def download_ohlcv(self, symbol, timeframe, market):
        """Downloads OHLCV data from S3 into a DataFrame."""
        key = self._get_key(symbol, timeframe, market)
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            buffer = io.BytesIO(response['Body'].read())
            return pd.read_parquet(buffer, engine='pyarrow')
        except self.s3.exceptions.NoSuchKey:
            return pd.DataFrame()
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return pd.DataFrame()

    def merge_and_upload(self, new_df, symbol, timeframe, market):
        """Downloads existing data, merges with new data (deduplicating), and re-uploads."""
        existing_df = self.download_ohlcv(symbol, timeframe, market)
        if not existing_df.empty:
            # Combine and deduplicate by timestamp
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            self.upload_ohlcv(combined_df, symbol, timeframe, market)
            return combined_df
        else:
            self.upload_ohlcv(new_df, symbol, timeframe, market)
            return new_df

if __name__ == "__main__":
    # Simple Connectivity Test
    storage = S3Storage()
    test_df = pd.DataFrame([{'timestamp': 1712911200000, 'open': 70000, 'high': 71000, 'low': 69000, 'close': 70500, 'volume': 100}])
    # storage.upload_ohlcv(test_df, 'BTC/USDT', '1h')
    # print("Test complete.")
