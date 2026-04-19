import pandas as pd
import numpy as np
from bot.indicators import calculate_adx

class VectorStrategy:
    def get_signal_column(self, df: pd.DataFrame) -> pd.Series: raise NotImplementedError

class SimpleBreakout(VectorStrategy):
    """
    Classic Donchian-style breakout strategy.
    Buys when price breaks above the previous n-period high.
    Includes an optional Institutional VWAP filter to prevent fake-outs.
    """
    def __init__(self, n=50, use_vwap=False, use_htf=False):
        self.n = n
        self.use_vwap = use_vwap
        self.use_htf = use_htf

    def get_signal_column(self, df):
        df = df.copy()

        # 1. THE CEILING AND FLOOR 
        # Shifted by 1 so the current candle is trying to break out of the PAST n candles.
        ceiling = df['close'].rolling(self.n).max().shift(1)
        floor = df['close'].rolling(self.n).min().shift(1)
        
        # 2. PURE BREAKOUT LOGIC (Evaluating the raw, live data)
        buy_cond = df['close'] > ceiling
        sell_cond = df['close'] < floor

        # 3. OPTIONAL VWAP FILTER (Evaluating raw data)
        if self.use_vwap:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            # Daily VWAP Calculation
            vwap = (typical_price * df['volume']).groupby(df.index.date).cumsum() / df['volume'].groupby(df.index.date).cumsum()
            
            buy_cond = buy_cond & (df['close'] > vwap)
            sell_cond = sell_cond & (df['close'] < vwap)

        # 4. HIGHER TIMEFRAME FILTER
        if self.use_htf and 'htf_trend' in df.columns:
            buy_cond = buy_cond & (df['htf_trend'] == 1)
            sell_cond = sell_cond & (df['htf_trend'] == -1)

        # 5. ASSIGN SIGNALS (Breakout Purity = No Exits, rely on Flips & ATR!)
        signals = pd.Series(0, index=df.index)
        
        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        return signals

# ------------------------------------------------------------------
# 1. TREND STRATEGIES
# ------------------------------------------------------------------

class EMACrossover(VectorStrategy):
    """
    Catches the start of a trend. Event-based (only triggers on the exact cross).
    Includes an optional Institutional VWAP filter to prevent fake-outs.
    """
    def __init__(self, fast=9, slow=21, use_vwap=False, use_htf=False):
        self.fast = fast
        self.slow = slow
        self.use_vwap = use_vwap
        self.use_htf = use_htf

    def get_signal_column(self, df):
        df = df.copy()

        # 1. Base Math (Raw live data)
        fast_ema = df['close'].ewm(span=self.fast, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow, adjust=False).mean()
        
        # 2. PURE EVENT LOGIC
        # Buy: Fast is currently > Slow, but on the previous candle it was <= Slow
        buy_cond = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        
        # Short: Fast is currently < Slow, but on the previous candle it was >= Slow
        sell_cond = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        # 3. OPTIONAL FILTERS (Evaluating raw data)
        if self.use_vwap:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).groupby(df.index.date).cumsum() / df['volume'].groupby(df.index.date).cumsum()
            
            # Ensure we are trading on the right side of daily volume
            buy_cond = buy_cond & (df['close'] > vwap)
            sell_cond = sell_cond & (df['close'] < vwap)
        
        if self.use_htf and 'htf_trend' in df.columns:
            buy_cond = buy_cond & (df['htf_trend'] == 1)
            sell_cond = sell_cond & (df['htf_trend'] == -1)

        # 4. ASSIGN SIGNALS (Pure Offense, No 2s!)
        signals = pd.Series(0, index=df.index)

        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1
        
        return signals

class MACDStrategy(VectorStrategy):
    """
    Trend Following with Safety Filter.
    Only trades if ADX > threshold (avoids chop).
    Uses a Smart 2 Exit if the trend dies but it's too choppy to reverse.
    """
    def __init__(self, fast=12, slow=26, signal=9, adx_threshold=25, use_vwap=False, use_htf=False):
        self.fast = fast
        self.slow = slow
        self.signal_span = signal
        self.adx_threshold = adx_threshold
        self.use_vwap = use_vwap
        self.use_htf = use_htf

    def get_signal_column(self, df):
        df = df.copy()
        
        # 1. Calculate Indicators (Using raw, live data)
        exp1 = df['close'].ewm(span=self.fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=self.signal_span, adjust=False).mean()

        df_temp = calculate_adx(df, period=7)
        adx = df_temp['adx']
        is_trending = adx > self.adx_threshold

        # 2. ENTRY LOGIC (Exact crossover event on raw data)
        # Buy: MACD is currently > Signal, but previously was <= Signal
        buy_cond = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)) & is_trending
        sell_cond = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)) & is_trending
        
        # 3. SMART EXIT LOGIC (Thesis Invalidation)
        # Fires EXACTLY ONCE on the downward cross to exit a Long if conditions are too choppy to Short
        exit_long = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        exit_short = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))

        # 4. OPTIONAL FILTERS (Evaluating raw data)
        if self.use_vwap:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).groupby(df.index.date).cumsum() / df['volume'].groupby(df.index.date).cumsum()
            
            buy_cond = buy_cond & (df['close'] > vwap)
            sell_cond = sell_cond & (df['close'] < vwap)

        if self.use_htf and 'htf_trend' in df.columns:
            buy_cond = buy_cond & (df['htf_trend'] == 1)
            sell_cond = sell_cond & (df['htf_trend'] == -1)

        # 5. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)

        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        # Only assign 2 if the signal isn't already taking a new position (Flip)
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2

        return signals

class BollingerSqueezeBreakout(VectorStrategy):
    """
    Volatility Squeeze Breakout.
    Enters when Bollinger Band Width (BBW) contracts (Squeeze), 
    then fires as soon as raw price shatters the bands.
    """
    def __init__(self, bb_window=20, bb_std=2, squeeze_lookback=50, use_vwap=False, use_htf=False):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.use_vwap = use_vwap
        self.use_htf = use_htf

    def get_signal_column(self, df):
        df = df.copy()
        
        # 1. Calculate Volatility Math (Raw Data)
        roll = df['close'].rolling(self.bb_window)
        mean = roll.mean()
        std = roll.std()
        
        upper = mean + (std * self.bb_std)
        lower = mean - (std * self.bb_std)
        
        bbw = (upper - lower) / mean
        lowest_bbw = bbw.rolling(self.squeeze_lookback).min()
        
        # The Squeeze: Volatility is at the bottom 10% of its recent range
        is_squeeze = bbw <= (lowest_bbw * 1.1)
        
        # Memory buffer: Was there a squeeze in the last 3 candles?
        squeeze_recently = is_squeeze.rolling(3).max() > 0

        # 2. PURE BREAKOUT LOGIC (Using raw df['close'])
        buy_cond = (df['close'] > upper) & squeeze_recently
        sell_cond = (df['close'] < lower) & squeeze_recently

        # 3. OPTIONAL FILTERS
        if self.use_vwap:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).groupby(df.index.date).cumsum() / df['volume'].groupby(df.index.date).cumsum()
            
            buy_cond = buy_cond & (df['close'] > vwap)
            sell_cond = sell_cond & (df['close'] < vwap)

        if self.use_htf and 'htf_trend' in df.columns:
            buy_cond = buy_cond & (df['htf_trend'] == 1)
            sell_cond = sell_cond & (df['htf_trend'] == -1)

        # 4. ASSIGN SIGNALS (No Exits, pure momentum play!)
        signals = pd.Series(0, index=df.index)

        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1
        
        return signals

class OrderFlowReclaim(VectorStrategy):
    """
    Official Strategy: Order Flow Reclaim (OFR).
    Formerly known as the 'Accidental Sweep'. 
    Captures high-momentum breakouts that 'dip a toe' below recent 
    support bodies before exploding with CVD confirmation.
    """
    def __init__(self, window=20, cvd_lookback=3, vol_multiplier=1.2):
        self.window = window
        self.cvd_lookback = cvd_lookback
        self.vol_multiplier = vol_multiplier 
        
    def get_signal_column(self, df):
        df = df.copy()
        
        # Define 'Fair Value' boundaries based on candle bodies (Closes)
        # This allows the 'Floor' to move dynamically with the trend
        ceiling = df['close'].rolling(self.window).max().shift(1)
        floor = df['close'].rolling(self.window).min().shift(1)
        
        avg_vol = df['volume'].rolling(self.window).mean().shift(1)
        dynamic_thresh = avg_vol * self.vol_multiplier
        
        # CVD X-Ray
        candle_range = (df['high'] - df['low']).replace(0, 1e-9)
        delta = df['volume'] * ((df['close'] - df['open']) / candle_range)
        cvd_slope = delta.cumsum().diff(self.cvd_lookback)
        
        # THE CORE LOGIC (The 'Accidental' Winner)
        # 1. Price dipped below a recent body-close (Low < Floor)
        # 2. Price is currently above that body-close (Close >= Floor)
        # 3. Aggressive buying volume is present
        # 4. CVD confirms whales are hitting the market buy button
        buy_signal = (
            (df['low'] < floor) & 
            (df['close'] >= floor) & 
            (df['volume'] > dynamic_thresh) & 
            (cvd_slope > 0)
        )
        
        sell_signal = (
            (df['high'] > ceiling) & 
            (df['close'] <= ceiling) & 
            (df['volume'] > dynamic_thresh) & 
            (cvd_slope < 0)
        )
        
        signals = pd.Series(0, index=df.index)
        signals.loc[buy_signal] = 1
        signals.loc[sell_signal] = -1

        return signals

# ------------------------------------------------------------------
# 2. MEAN REVERSION STRATEGIES (Use these when ADX < 25)
# ------------------------------------------------------------------

class RSIStrategy_Turbo(VectorStrategy):
    """
    REPLACES Wilder/Cutler.
    - Faster Periods (9 vs 14)
    - Single Hook (Fast reaction on raw data)
    - Crash Logic (Ignores ADX if price is below Bollinger)
    - Smart Exit (Aborts if RSI crosses back into the danger zone)
    """
    def __init__(self, period=9, lower=30, upper=70, adx_threshold=30):
        self.period = period
        self.lower = lower
        self.upper = upper
        self.adx_threshold = adx_threshold 

    def get_signal_column(self, df):
        df = df.copy()

        # 1. Base Math (Raw, live data)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=self.period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df_temp = calculate_adx(df, period=7)
        adx = df_temp['adx']

        roll = df['close'].rolling(20)
        lower_bb = roll.mean() - (roll.std() * 2)
        upper_bb = roll.mean() + (roll.std() * 2)

        # 2. RAW MOMENTUM HOOKS
        # Evaluating current live candle vs the previous closed candle
        hook_up = rsi > rsi.shift(1)
        hook_down = rsi < rsi.shift(1)

        # 3. PURE OFFENSE LOGIC
        # Normal Chop Farming
        buy_normal = (rsi < self.lower) & (adx < self.adx_threshold) & hook_up
        sell_normal = (rsi > self.upper) & (adx < self.adx_threshold) & hook_down
        
        # Crash Override (Ignores ADX during extreme standard deviation breaks)
        buy_crash = (df['close'] < lower_bb) & (rsi < 25) & hook_up
        sell_pump = (df['close'] > upper_bb) & (rsi > 75) & hook_down
        
        buy_cond = (buy_normal | buy_crash)
        sell_cond = (sell_normal | sell_pump)

        # 4. SMART EXIT LOGIC (Thesis Invalidation)
        # Event Trigger: Emit a 2 ONLY on the exact candle where RSI drops back into the danger zone
        exit_long = (rsi < self.lower) & (rsi.shift(1) >= self.lower)
        exit_short = (rsi > self.upper) & (rsi.shift(1) <= self.upper)

        # 6. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)
        
        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        # Apply Smart 2s without clashing with entries
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2

        return signals

class BollingerReversion(VectorStrategy):
    """
    Mean-Reversion with Bounce Confirmation.
    Evaluates raw live data. Waits for the price to drop below the Bollinger Band, 
    but only buys when the live candle confirms a green bounce.
    Uses a Smart 2 Exit if the bounce immediately fails.
    """
    def __init__(self, window=20, std=2, min_bbw=0.015):
        self.window = window
        self.std = std
        self.min_bbw = min_bbw

    def get_signal_column(self, df):
        df = df.copy()

        # 1. Calculate the Bands (Raw, live data)
        roll = df['close'].rolling(self.window)
        mean = roll.mean()
        std_dev = roll.std()
        
        upper = mean + (std_dev * self.std)
        lower = mean - (std_dev * self.std)
        bbw = (upper - lower) / mean

        # We still need shift(1) purely to check if the current candle is green/red!
        prev_close = df['close'].shift(1)
        prev_prev_close = df['close'].shift(2)

        # 2. PURE OFFENSE (Evaluating raw data)
        # Buy: Live price is below the lower band, volatility is high, AND the live candle is green
        buy_cond = (df['close'] < lower) & (bbw > self.min_bbw) & (df['close'] >= prev_close)
        
        # Short: Live price is above the upper band, volatility is high, AND the live candle is red
        sell_cond = (df['close'] > upper) & (bbw > self.min_bbw) & (df['close'] <= prev_close)

        # 3. SMART EXIT LOGIC (Thesis Invalidation)
        # Event Trigger: Emit a 2 ONLY on the exact candle where a bounce fails.
        # (e.g., Long Exit: Still below the band, current candle is RED, but yesterday's was GREEN)
        exit_long = (df['close'] < lower) & (df['close'] < prev_close) & (prev_close >= prev_prev_close)
        exit_short = (df['close'] > upper) & (df['close'] > prev_close) & (prev_close <= prev_prev_close)

        # 5. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)

        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        # Apply Smart 2s without clashing with entries
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2

        return signals

class OrderFlowStrategy(VectorStrategy):
    """
    Weaponized Order Flow Strategy + ADX Shield.
    1. Uses a Rolling Value Area (VWAP + StdDev) to find the 'Box' (Chop Zone).
    2. ADX Shield prevents taking trades if the market is violently trending.
    3. Enters at the edges (VAH/VAL) ONLY if Cumulative Volume Delta (CVD)
       shows smart money is absorbing the move (Divergence/Exhaustion).
    """
    def __init__(self, lookback=200, cvd_lookback=5, adx_period=7, adx_threshold=25):
        self.lookback = lookback
        self.cvd_lookback = cvd_lookback
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def get_signal_column(self, df):
        df = df.copy()
        
        # 1. CALCULATE INDICATORS (Raw Data)
        df_temp = calculate_adx(df, period=self.adx_period)
        adx = df_temp['adx']

        candle_range = df['high'] - df['low']
        candle_range = candle_range.replace(0, 1e-9)
        
        delta = df['volume'] * ((df['close'] - df['open']) / candle_range)
        cvd = delta.cumsum()
        cvd_slope = cvd.diff(self.cvd_lookback)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        poc = (typical_price * df['volume']).rolling(self.lookback).sum() / df['volume'].rolling(self.lookback).sum()
        
        price_std = df['close'].rolling(self.lookback).std()
        vah = poc + price_std
        val = poc - price_std

        # 2. ENTRY LOGIC (Raw Data)
        buy_cond = (df['close'] <= val) & (cvd_slope > 0) & (adx < self.adx_threshold)
        sell_cond = (df['close'] >= vah) & (cvd_slope < 0) & (adx < self.adx_threshold)

        # 3. SMART 2 EXIT (Thesis Invalidation)
        # Triggered ONLY when the CVD slope flips direction while price is at the edges.
        # This is an event-based 'Smart 2' that fires once.
        exit_long = (df['close'] <= val) & (cvd_slope < 0) & (cvd_slope.shift(1) >= 0)
        exit_short = (df['close'] >= vah) & (cvd_slope > 0) & (cvd_slope.shift(1) <= 0)

        # 5. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)

        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        # Apply Logical Exit 2 (Thesis failed before ATR hit)
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2

        return signals

class OrderFlowSweep(VectorStrategy):
    """
    The Ultimate Trap Hunter.
    Sweeps a Price Action level, but ONLY executes if 
    Cumulative Volume Delta (CVD) shows institutional absorption.
    """
    def __init__(self, window=20, cvd_lookback=3, vol_multiplier=1.2):
        self.window = window
        self.cvd_lookback = cvd_lookback
        self.vol_multiplier = vol_multiplier 
        
    def get_signal_column(self, df):
        df = df.copy()
        
        # 1. Define the PA Box
        ceiling = df['high'].rolling(self.window).max().shift(1)
        floor = df['low'].rolling(self.window).min().shift(1)

        # 2. Dynamic Volume Threshold
        avg_vol = df['volume'].rolling(self.window).mean().shift(1)
        dynamic_thresh = avg_vol * self.vol_multiplier
        
        # 3. THE X-RAY: Calculate CVD Slope
        candle_range = df['high'] - df['low']
        candle_range = candle_range.replace(0, 1e-9)
        delta = df['volume'] * ((df['close'] - df['open']) / candle_range)
        cvd = delta.cumsum()
        cvd_slope = cvd.diff(self.cvd_lookback) # Short lookback to catch immediate divergence
        
        # 4. ENTRY LOGIC: The "Confirmed Trap"
        # PA sweeps the floor + closes back inside + volume is high + CVD is POSITIVE (Absorption)
        bull_sweep = (
            (df['low'] < floor) & 
            (df['close'] >= floor) & 
            (df['volume'] > dynamic_thresh) & 
            (cvd_slope > 0) # <--- The Order Flow confirmation
        )
        
        # PA sweeps the ceiling + closes back inside + volume is high + CVD is NEGATIVE (Absorption)
        bear_sweep = (
            (df['high'] > ceiling) & 
            (df['close'] <= ceiling) & 
            (df['volume'] > dynamic_thresh) & 
            (cvd_slope < 0) # <--- The Order Flow confirmation
        )
        
        # 5. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)
        signals.loc[bull_sweep] = 1
        signals.loc[bear_sweep] = -1

        return signals

# ------------------------------------------------------------------
# 4. HYBRID STRATEGIES (Confluence)
# ------------------------------------------------------------------

class HybridRSIBollinger_Wilder(VectorStrategy):
    def __init__(self, rsi_period=14, bb_window=20, bb_std=2):
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        
    def get_signal_column(self, df):
        df = df.copy()

        # 1. RSI Calculation (Wilder/EMA)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Bollinger Calculation
        roll = df['close'].rolling(self.bb_window)
        mean = roll.mean()
        upper = mean + (roll.std() * self.bb_std)
        lower = mean - (roll.std() * self.bb_std)

        # 3. Reference Points
        prev_close = df['close'].shift(1)
        prev_prev_close = df['close'].shift(2)

        # 4. PURE OFFENSE (Double Confirmation + Bounce)
        buy_cond = (df['close'] < lower) & (rsi < 30) & (df['close'] >= prev_close)
        sell_cond = (df['close'] > upper) & (rsi > 70) & (df['close'] <= prev_close)

        # 5. SMART 2 EXIT (Thesis Invalidation)
        # Triggered if: Still in danger zone + Current candle is RED + Previous was GREEN
        exit_long = (df['close'] < lower) & (df['close'] < prev_close) & (prev_close >= prev_prev_close)
        exit_short = (df['close'] > upper) & (df['close'] > prev_close) & (prev_close <= prev_prev_close)

        # 6. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)
        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1
        
        # Apply Logic Stop 2 (only if not flipping)
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2
        
        return signals

class HybridRSIBollinger_Cutler(VectorStrategy):
    def __init__(self, rsi_period=14, bb_window=20, bb_std=2):
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std

    def get_signal_column(self, df):
        df = df.copy()

        # 1. Cutler's RSI (SMA)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = (100 - (100 / (1 + rs))).fillna(50)
        
        # 2. Bollinger Calculation
        roll = df['close'].rolling(self.bb_window)
        mean = roll.mean()
        upper = mean + (roll.std() * self.bb_std)
        lower = mean - (roll.std() * self.bb_std)

        # 3. Reference Points
        prev_close = df['close'].shift(1)
        prev_prev_close = df['close'].shift(2)
        
        # 4. PURE OFFENSE
        buy_cond = (df['close'] < lower) & (rsi < 30) & (df['close'] >= prev_close)
        sell_cond = (df['close'] > upper) & (rsi > 70) & (df['close'] <= prev_close)

        # 5. SMART 2 EXIT
        exit_long = (df['close'] < lower) & (df['close'] < prev_close) & (prev_close >= prev_prev_close)
        exit_short = (df['close'] > upper) & (df['close'] > prev_close) & (prev_close <= prev_prev_close)

        # 6. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)
        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1
        
        signals.loc[exit_long & (signals == 0)] = 2
        signals.loc[exit_short & (signals == 0)] = 2

        return signals

# ==========================================
# NEW STRATEGY
# ==========================================

class PriceActionStrategy(VectorStrategy):
    """
    Pure Price Action Breakout & Deviation.
    Evaluates raw data for instantaneous signal generation.
    Relies on ATR Risk Manager and Flips for exits.
    """
    def __init__(self, window=20, vol_multiplier=2.0, use_htf=False):
        self.window = window
        self.vol_multiplier = vol_multiplier 
        self.use_htf = use_htf
        
    def get_signal_column(self, df):
        df = df.copy()
        
        # 1. Define the Box (Using the PAST window)
        ceiling = df['close'].rolling(self.window).max().shift(1)
        floor = df['close'].rolling(self.window).min().shift(1)
        
        # 2. Dynamic Volume Threshold (Institutional Weight)
        avg_vol = df['volume'].rolling(self.window).mean().shift(1)
        dynamic_thresh = avg_vol * self.vol_multiplier
        
        # 3. ENTRY LOGIC (Raw, live data)
        bull_breakout = (df['close'] > ceiling) & (df['close'] > df['open']) & (df['volume'] > dynamic_thresh)
        bear_breakout = (df['close'] < floor) & (df['close'] < df['open']) & (df['volume'] > dynamic_thresh)
        
        bull_bounce = (df['low'] < floor) & (df['close'] >= floor) & (df['close'] > df['open']) & (df['volume'] > dynamic_thresh)
        bear_bounce = (df['high'] > ceiling) & (df['close'] <= ceiling) & (df['close'] < df['open']) & (df['volume'] > dynamic_thresh)
        
        # Consolidate raw signals
        buy_cond = bull_breakout | bull_bounce
        sell_cond = bear_breakout | bear_bounce

        # 4. HIGHER TIMEFRAME FILTER (The Shield)
        if self.use_htf and 'htf_trend' in df.columns:
            buy_cond = buy_cond & (df['htf_trend'] == 1)
            sell_cond = sell_cond & (df['htf_trend'] == -1)
        
        # 5. ASSIGN SIGNALS
        signals = pd.Series(0, index=df.index)
        
        signals.loc[buy_cond] = 1
        signals.loc[sell_cond] = -1

        return signals

# ==========================================
STRATEGY_CONFIG = {

    # 1. MEAN REVERSION (For Chop) -------------------------------
    "RSIStrategyTurbo": {
        "class": RSIStrategy_Turbo, 
        "params": {
            "period": [6, 9, 14], 
            "adx_threshold": [15, 20, 25, 30]
        }
    },

    "BollingerBandit": {
        "class": BollingerReversion,
        "params": {
            "window": [15,20], 
            "std": [2,2.5],
        }
    },

    "OrderFlow": {
        "class": OrderFlowStrategy,
        "params": {
            "lookback": [50, 100, 200],
            "cvd_lookback": [5, 10], 
            "adx_period": [7], 
            "adx_threshold": [15, 20, 25, 30]
        }
    },

    "OrderFlowSweep": {
        "class": OrderFlowSweep,
        "params": {
            "window": [20, 30, 50, 100], 
            "cvd_lookback": [2, 3, 5], 
            "vol_multiplier": [1.2, 1.5, 1.8]
        }
    },

    # 2. TREND FOLLOWING (For Breakouts/Crashes) -----------------
    "MACDStrategy": {
        "class": MACDStrategy,
        "params": {
            "fast": [8, 12], 
            "slow": [21, 26], 
            "signal": [9],
            "adx_threshold": [15, 20, 25, 30],
            "use_vwap": [True, False], # <--- DO THIS
            "use_htf": [True, False]   # <--- AND THIS
        }
    },

    "BBSqueezeBreakout": {
        "class": BollingerSqueezeBreakout,
        "params": {
            "bb_window": [20], 
            "bb_std": [2], 
            "squeeze_lookback": [30, 50],
            "use_vwap":[True, False],
            "use_htf": [True, False]
        }
    },

    "EMACrossover": {
        "class": EMACrossover,
        "params": {
            "fast": [9, 20], 
            "slow": [21, 50],
            "use_vwap": [True, False],
            "use_htf": [True, False]
        }
    },

    "OrderFlowReclaim": {
        "class": OrderFlowReclaim,
        "params": {
            "window": [20, 30, 50], # Keep it fast to capture momentum
            "cvd_lookback": [3, 5], 
            "vol_multiplier": [1.2, 1.5, 1.8]
        }
    },

    # 3. HYBRIDS (Confluence) ------------------------------------
    "HybridWilder": {
        "class": HybridRSIBollinger_Wilder,
        "params": {
            "rsi_period": [14], 
            "bb_window": [20],
            "bb_std": [2],
        }
    },

    "HybridCutler": {
        "class": HybridRSIBollinger_Cutler,
        "params": {
            "rsi_period": [14], 
            "bb_window": [20],
            "bb_std": [2],
        }
    },
    
    # 4. WILDCARD (For Moonshots only) ---------------------------
    "SimpleBreakout": {
        "class": SimpleBreakout,
        "params": {
            "n": [20, 50],
            "use_vwap": [True, False],
            "use_htf": [True, False]
        } 
    },

    # NEW STRATEGY
    "PriceAction": {
        "class": PriceActionStrategy,
        "params": {
            "window": [20, 30, 50, 100], 
            "vol_multiplier": [1.5, 1.8, 2.0, 2.5, 3],
            "use_htf": [True, False]
        }
    }
}

# ==========================================
