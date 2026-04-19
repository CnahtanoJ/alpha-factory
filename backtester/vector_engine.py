import pandas as pd
import numpy as np
import ta
from data_pipeline.database import get_connection

class VectorEngine:
    def __init__(self):
        self.conn = get_connection()

    def load_data(self, symbol, timeframe='1h'):
        """Loads OHLCV data from the SQLite database."""
        query = "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, self.conn, params=(symbol, timeframe))
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def add_features(self, df):
        """Calculates indicators in a vectorized way."""
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # EMA
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Volatility (Rolling Std Dev)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Target: 1% and 5% forward moves (for backtesting)
        # Shift close price backwards to see future prices
        for p in [1, 4, 8, 24]: # Hours ahead
            df[f'future_return_{p}h'] = df['close'].shift(-p) / df['close'] - 1
            
        return df

    def run_simulation(self, df, rsi_threshold=30, vol_threshold=0.02):
        """
        Runs a vectorized backtest on a single symbol.
        Finds 'Buy' signals where RSI < threshold AND Volatility > window.
        """
        # Logic: RSI < threshold AND Price above EMA 200 (uptrend)
        df['signal'] = (df['rsi'] < rsi_threshold) & (df['close'] > df['ema_200'])
        
        # Performance of signals
        signal_results = df[df['signal'] == True]
        
        metrics = {
            'total_signals': len(signal_results),
            'avg_1h_return': signal_results['future_return_1h'].mean(),
            'avg_4h_return': signal_results['future_return_4h'].mean(),
            'avg_24h_return': signal_results['future_return_24h'].mean(),
            'win_rate_24h': (signal_results['future_return_24h'] > 0).mean()
        }
        
        return metrics

if __name__ == "__main__":
    engine = VectorEngine()
    df = engine.load_data('BTC/USDT', '1d')
    if not df.empty:
        df = engine.add_features(df)
        results = engine.run_simulation(df)
        print(f"Results for BTC:")
        for k, v in results.items():
            print(f"  {k}: {v}")
