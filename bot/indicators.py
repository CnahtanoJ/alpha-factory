import numpy as np
import pandas as pd

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates ADX, +DI, and -DI using Wilder's Smoothing.
    Expects a DataFrame with 'high', 'low', and 'close' columns.
    """
    df = df.copy()
    
    # 1. Calculate +DM and -DM
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    df['+dm'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    df['-dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # 2. Calculate True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 3. Apply Wilder's Smoothing (Using EMA with alpha = 1/period)
    # The 'adjust=False' flag is critical for matching TradingView
    wilders_alpha = 1 / period
    
    df['smooth_tr'] = df['tr'].ewm(alpha=wilders_alpha, adjust=False).mean()
    df['smooth_+dm'] = df['+dm'].ewm(alpha=wilders_alpha, adjust=False).mean()
    df['smooth_-dm'] = df['-dm'].ewm(alpha=wilders_alpha, adjust=False).mean()
    
    # 4. Calculate +DI and -DI
    df['+di'] = 100 * (df['smooth_+dm'] / df['smooth_tr'])
    df['-di'] = 100 * (df['smooth_-dm'] / df['smooth_tr'])
    
    # 5. Calculate DX
    dx = 100 * (df['+di'] - df['-di']).abs() / (df['+di'] + df['-di'])
    
    # 6. Calculate ADX (Smoothed DX)
    df['adx'] = dx.ewm(alpha=wilders_alpha, adjust=False).mean()
    
    # Clean up intermediate columns to keep the DataFrame light
    df.drop(columns=['+dm', '-dm', 'tr', 'smooth_tr', 'smooth_+dm', 'smooth_-dm'], inplace=True)
    
    return df
    
# ==========================================
def get_local_poc(df, num_bins=50, lookback=200):
    """Calculates the Point of Control for the most recent 'lookback' candles."""
    # 1. Isolate the recent price action 
    recent_df = df.tail(lookback).copy()
    if recent_df.empty: return 0.0
        
    # 2. Use Typical Price
    tp = (recent_df['high'] + recent_df['low'] + recent_df['close']) / 3
    min_px, max_px = recent_df['low'].min(), recent_df['high'].max()
    
    # Safety catch for flatlines
    if min_px == max_px: return min_px
        
    # 3. Bin the volume and find the heaviest node
    bins = np.linspace(min_px, max_px, num_bins)
    price_bins = pd.cut(tp, bins=bins, include_lowest=True)
    volume_profile = recent_df.groupby(price_bins, observed=True)['volume'].sum()
    
    poc_bin = volume_profile.idxmax()
    return poc_bin.mid

def get_cvd_slope(df, lookback=5):
    """Calculates the Proxy CVD and returns the recent momentum slope."""
    if len(df) < lookback + 1:
        return 0.0
        
    # 1. Calculate Proxy Delta for the whole dataframe
    candle_range = df['high'] - df['low']
    candle_range = candle_range.replace(0, 1e-9) # Prevent division by zero
    delta = df['volume'] * ((df['close'] - df['open']) / candle_range)
    
    # 2. Calculate CVD (Running Total)
    cvd = delta.cumsum()
    
    # 3. Calculate the Slope (Current CVD minus the CVD 'lookback' periods ago)
    slope = cvd.iloc[-1] - cvd.iloc[-(lookback + 1)]
    
    return slope

# ==========================================
