import pandas as pd
import numpy as np
from bot.data_feed import MarketData

def inject_htf_trend(df_ltf, df_htf, ema_period=50):
    """
    Safely calculates the 1H trend and injects it into the lower timeframe (LTF)
    dataframe WITHOUT Lookahead Bias.
    """
    df_htf['htf_ema'] = df_htf['close'].ewm(span=ema_period, adjust=False).mean()
    df_htf['htf_trend'] = np.where(df_htf['close'] > df_htf['htf_ema'], 1, -1)
    
    # 2. 🚨 THE LOOKAHEAD SHIFT 🚨
    df_htf['htf_trend'] = df_htf['htf_trend'].shift(1)
    
    # 3. Merge into the LTF timeframe
    df_ltf['htf_trend'] = df_htf['htf_trend'].reindex(df_ltf.index, method='ffill')
    
    # Clean up any NaNs at the very beginning of the dataset
    df_ltf['htf_trend'] = df_ltf['htf_trend'].fillna(0) 
    
    return df_ltf

class HyperBacktester:
    def __init__(self, info_client, symbol, interval="1h", candles_lookback=5000, use_db=True):
        self.symbol = symbol
        self.interval = interval
        md = MarketData(info_client)
        self.data = md.get_clean_candles(symbol, interval, limit=candles_lookback, use_db=use_db)
        
        if not self.data.empty:
            self.data['pct_change'] = self.data['close'].pct_change()

    def run(self, strategy_instance, fee_rate=0.001): # Remember the new realistic fee!
        if self.data.empty: return {"return": -99, "buy_hold": 0, "trades": 0}
        df = self.data.copy()
        
        try:            
            # 1. Get raw signals (1=Buy, -1=Sell, 2=Exit, 0=Hold)
            df['signal'] = strategy_instance.get_signal_column(df)
            
            # CALCULATE ATR SIMULATION
            atr_tp_mult = 1.5 
            atr_sl_mult = 2.0
            
            high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()

            # SET THE SHADOW TARGETS
            df['long_tp'] = np.where(df['signal'] == 1, df['close'] + (df['atr'] * atr_tp_mult), np.nan)
            df['long_sl'] = np.where(df['signal'] == 1, df['close'] - (df['atr'] * atr_sl_mult), np.nan)
            
            df['short_tp'] = np.where(df['signal'] == -1, df['close'] - (df['atr'] * atr_tp_mult), np.nan)
            df['short_sl'] = np.where(df['signal'] == -1, df['close'] + (df['atr'] * atr_sl_mult), np.nan)

            # FORWARD FILL THE TARGETS
            # Drag the entry targets forward so future candles know where the finish line is
            # We group by the cumulative sum of signals to reset the fill on every new trade
            trade_blocks = df['signal'].replace(0, np.nan).ffill()
            
            df['long_tp'] = df['long_tp'].groupby(trade_blocks).ffill()
            df['long_sl'] = df['long_sl'].groupby(trade_blocks).ffill()
            df['short_tp'] = df['short_tp'].groupby(trade_blocks).ffill()
            df['short_sl'] = df['short_sl'].groupby(trade_blocks).ffill()

            # THE COLLISION DETECTION
            hit_long_exit = (df['high'] >= df['long_tp']) | (df['low'] <= df['long_sl'])
            hit_short_exit = (df['low'] <= df['short_tp']) | (df['high'] >= df['short_sl'])

            # EMIT THE ATR '2'
            df.loc[(hit_long_exit | hit_short_exit) & (df['signal'] == 0), 'signal'] = 2

            # 2. Convert Signals to Continuous Positions 🧠
            # Replace 0 (Hold) with NaN so we can forward-fill the previous active state
            # Replace 2 (Exit) with 0 so the position drops to flat
            pos_mapper = df['signal'].replace(0, np.nan).replace(2, 0)
            df['position'] = pos_mapper.ffill().fillna(0)

            df.loc[df.index[-1], 'position'] = 0
            
            # 3. Strategy Returns (Shifted 1 step forward)
            df['strategy_return'] = (df['position'].shift(1) * df['pct_change']).fillna(0)
            
            # 4. Calculate Position Changes for Fees
            position_change = df['position'].diff().fillna(0)
            position_change.iloc[0] = df['position'].iloc[0] 
            
            df['fees'] = position_change.abs() * fee_rate
            df['net_return'] = df['strategy_return'] - df['fees']

            # 5. Cumulative Returns (Safe Math)
            total_ret = ((1 + df['net_return']).cumprod().iloc[-1] - 1)
            hold_ret = ((1 + df['pct_change'].fillna(0)).cumprod().iloc[-1] - 1)

            # --- 24-HOUR RECENT BIAS CHECK ---
            # 5m = 288, 15m = 96, 1h = 24
            candles_24h = 288 if self.interval == '5m' else (96 if self.interval == '15m' else 24)
            
            if len(df) > candles_24h:
                recent_ret = ((1 + df['net_return'].iloc[-candles_24h:]).cumprod().iloc[-1] - 1)
            else:
                recent_ret = total_ret # Fallback if data is short

            trades = position_change.abs().sum() / 2 

            annual_map = {"5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
            annual_factor = annual_map.get(self.interval, 35040)
            sharpe = (df['net_return'].mean() / (df['net_return'].std() + 1e-9)) * np.sqrt(annual_factor)

            gross_profit = df.loc[df['net_return'] > 0, 'net_return'].sum()
            gross_loss = df.loc[df['net_return'] < 0, 'net_return'].abs().sum()
            
            if gross_loss == 0:
                profit_factor = 99.0 
            else:
                profit_factor = gross_profit / gross_loss

            return {
                "return": total_ret, 
                "recent_return": recent_ret,
                "buy_hold": hold_ret, 
                "trades": int(trades),
                "profit_factor": round(profit_factor, 2),
                'sharpe': sharpe
            }
            
        except Exception as e:
            print(f"Backtest Error: {e}") 
            return {"return": -99, "buy_hold": 0, "trades": 0}

# ==========================================
