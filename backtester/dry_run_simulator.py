"""
Dry Run Simulator — Vectorized Walk-Forward Portfolio Backtester.

Takes a trained LightGBM model's predictions on a mega-dataframe and simulates
a Top-N Long / Bottom-N Short portfolio, producing institutional-grade metrics.

Repurposed from the legacy engine.py — the Sharpe, PF, and transaction cost
math is preserved, but the execution is now fully vectorized across the
cross-sectional ranking output.

Usage:
    from backtester.dry_run_simulator import simulate_portfolio
    results = simulate_portfolio(mega_df, predictions, top_n=10, bottom_n=10)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DEFAULT_SLIPPAGE  = 0.0001    # 1 bps slippage (Hyperliquid on-chain book, liquid futures)
DEFAULT_TAKER_FEE = 0.00045   # Hyperliquid taker fee (0.045%)

def simulate_portfolio(
    mega_df: pd.DataFrame,
    predictions: np.ndarray,
    top_n: int = 10,
    bottom_n: int = 10,
    rebalance_freq: int = 6,
    fee_rate: float = DEFAULT_TAKER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    timeframe: str = '1h',
    weighting_mode: str = 'equal',
    mc_sims: int = 2500,
    hysteresis_factor: float = 4.0, # agile conviction mode
    long_atr_multiplier: float = 1.0,
    short_atr_multiplier: float = 1.0,
) -> dict:
    """
    Vectorized long/short portfolio simulation from LightGBM predictions.

    Parameters
    ----------
    mega_df : pd.DataFrame
        The full cross-sectional dataframe with 'timestamp', 'symbol', 'close',
        and 'fwd_return' columns. Must be the OOS (out-of-sample) slice only.
    predictions : np.ndarray
        The model's predicted target_rank for each row in mega_df.
    top_n : int
        Number of assets to go long (highest predicted rank).
    bottom_n : int
        Number of assets to go short (lowest predicted rank).
    rebalance_freq : int
        How often to rebalance in bars (default: 6 = every 6 hours for 1h candles).
    fee_rate : float
        Taker fee per side (applied on every rebalance).
    slippage : float
        Slippage per side.

    Returns
    -------
    dict with keys:
        'sharpe', 'profit_factor', 'win_rate', 'total_return',
        'n_rebalances', 'avg_daily_return', 'max_drawdown',
        'equity_curve' (pd.Series), 'trade_log' (pd.DataFrame),
        'top_assets' (list), 'bottom_assets' (list), 'mc_stats' (dict)
    """
    df = mega_df.copy()
    df['predicted_rank'] = predictions

    # P1-7: Log rebalance cadence for transparency
    print(f"Rebalance cadence: every {rebalance_freq} bars | Hysteresis buffer: {hysteresis_factor:.1f}x")

    # ─── Group by timestamp to get cross-sectional snapshots ───
    timestamps = np.sort(df['timestamp'].unique())

    # Rebalance at every `rebalance_freq`-th timestamp
    rebalance_points = timestamps[::rebalance_freq]

    portfolio_returns = []
    trade_log = []
    
    prev_longs = set()
    prev_shorts = set()
    total_basket_size = top_n + bottom_n

    for i, rb_ts in enumerate(rebalance_points):
        # Get the cross-section at this rebalance point
        snapshot = df[df['timestamp'] == rb_ts].copy()

        if len(snapshot) < total_basket_size:
            continue  # Not enough assets for a full basket

        # Implement Hysteresis (Friction) Buffer
        snapshot = snapshot.sort_values('predicted_rank', ascending=False)
        
        long_buffer_n = int(top_n * hysteresis_factor)
        short_buffer_n = int(bottom_n * hysteresis_factor)
        
        # --- LONG SELECTION ---
        eligible_longs = snapshot.head(long_buffer_n)
        kept_longs = eligible_longs[eligible_longs['symbol'].isin(prev_longs)]
        needed_longs = top_n - len(kept_longs)
        
        # Fill the remaining slots with the absolute highest ranked that aren't already kept
        new_longs = eligible_longs[~eligible_longs['symbol'].isin(prev_longs)].head(needed_longs)
        longs = pd.concat([kept_longs, new_longs])
        
        # --- SHORT SELECTION ---
        eligible_shorts = snapshot.tail(short_buffer_n)
        kept_shorts = eligible_shorts[eligible_shorts['symbol'].isin(prev_shorts)]
        needed_shorts = bottom_n - len(kept_shorts)
        
        new_shorts = eligible_shorts[~eligible_shorts['symbol'].isin(prev_shorts)].tail(needed_shorts)
        shorts = pd.concat([kept_shorts, new_shorts])
        
        curr_longs = set(longs['symbol'])
        curr_shorts = set(shorts['symbol'])

        # ─── VECTORIZED TRAILING STOP LOGIC ───
        # Trailing distance = multiplier * ATR
        long_trail_dist = longs['atr_pct'] * long_atr_multiplier
        
        # M-4 FIX: NaN Guard for trailing stop inputs
        long_fwd_min = longs['fwd_min_ret'].fillna(0)
        long_fwd_max = longs['fwd_max_ret'].fillna(0)
        long_fwd_ret = longs['fwd_return'].fillna(0)

        long_initial_sl_hit = long_fwd_min <= -long_trail_dist
        long_retrace_hit = (long_fwd_max - long_fwd_ret) >= long_trail_dist
        
        # Conservative: If initial SL hit, we take the full loss. Else, check if it retraced from the peak.
        long_actual_returns = np.where(
            long_initial_sl_hit, -long_trail_dist,
            np.where(long_retrace_hit, long_fwd_max - long_trail_dist, long_fwd_ret)
        )

        short_trail_dist = shorts['atr_pct'] * short_atr_multiplier
        
        # M-4 FIX: NaN Guard for trailing stop inputs
        short_fwd_min = shorts['fwd_min_ret'].fillna(0)
        short_fwd_max = shorts['fwd_max_ret'].fillna(0)
        short_fwd_ret = shorts['fwd_return'].fillna(0)

        short_initial_sl_hit = short_fwd_max >= short_trail_dist
        short_retrace_hit = (short_fwd_max - short_fwd_min) >= short_trail_dist
        
        # Simulated price movement of the underlying asset
        short_sim_move = np.where(
            short_initial_sl_hit, short_trail_dist,
            np.where(short_retrace_hit, short_fwd_min + short_trail_dist, short_fwd_ret)
        )

        # ─── RETURN CALCULATION ───
        if weighting_mode == 'risk_parity':
            # Weight is inversely proportional to volatility (ATR%)
            # We assume 'atr_pct' is available in mega_df
            long_weights = 1.0 / (longs['atr_pct'] + 1e-6)
            short_weights = 1.0 / (shorts['atr_pct'] + 1e-6)
            
            # Normalize weights to sum to 1.0 on each side
            long_weights /= long_weights.sum()
            short_weights /= short_weights.sum()
            
            # Weighted returns
            long_return = (long_actual_returns * long_weights).sum()
            short_return = (short_sim_move * short_weights).sum()
        else:
            # Equal-weighted
            long_return = long_actual_returns.mean()
            short_return = short_sim_move.mean()

        # Long/Short portfolio: profit on longs going up, shorts going down
        # Equal-weighted or Risk-Parity, dollar-neutral
        gross_return = (long_return - short_return) / 2.0

        # P2-10 FIX: Dynamic Transaction Costs based on actual turnover
        if not prev_longs and not prev_shorts:
            # First entry: 100% of portfolio enters
            cost_multiplier = 1.0
        else:
            long_exits = len(prev_longs - curr_longs)
            short_exits = len(prev_shorts - curr_shorts)
            long_entries = len(curr_longs - prev_longs)
            short_entries = len(curr_shorts - prev_shorts)
            cost_multiplier = (long_exits + short_exits + long_entries + short_entries) / total_basket_size

        net_return = gross_return - (cost_multiplier * (fee_rate + slippage))
        
        prev_longs = curr_longs
        prev_shorts = curr_shorts

        portfolio_returns.append({
            'timestamp': rb_ts,
            'gross_return': gross_return,
            'net_return': net_return,
            'long_return': long_return,
            'short_return': short_return,
            'long_symbols': longs['symbol'].tolist(),
            'short_symbols': shorts['symbol'].tolist(),
        })

        # Log individual trades
        for i, (_, row) in enumerate(longs.iterrows()):
            trade_log.append({
                'timestamp': rb_ts, 'symbol': row['symbol'],
                'side': 'LONG', 'predicted_rank': row['predicted_rank'],
                'actual_return': long_actual_returns[i]
            })
        for i, (_, row) in enumerate(shorts.iterrows()):
            trade_log.append({
                'timestamp': rb_ts, 'symbol': row['symbol'],
                'side': 'SHORT', 'predicted_rank': row['predicted_rank'],
                'actual_return': -short_sim_move[i]  # Shorts profit on decline
            })

    if not portfolio_returns:
        return _empty_result()

    returns_df = pd.DataFrame(portfolio_returns)
    returns_df = returns_df.set_index('timestamp')
    trade_log_df = pd.DataFrame(trade_log)

    # ─── Monte Carlo Simulation (The Bullshit Filter) ───
    mc_results = run_monte_carlo(returns_df['net_return'].values, sims=mc_sims)

    # ─── Metrics (repurposed from legacy engine.py) ───
    net_series = returns_df['net_return'].dropna()

    # Equity curve
    equity_curve = (1 + net_series).cumprod()

    # Total return
    total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0

    # Sharpe ratio (annualized, assuming rebalance every rebalance_freq bars)
    # 1h: 8760, 15m: 35040, 4h: 2190
    tf_map = {'15m': 35040, '1h': 8760, '4h': 2190, '1d': 365}
    annual_factor = tf_map.get(timeframe, 8760)
    
    bars_per_year = annual_factor / rebalance_freq
    if len(net_series) < 2:
        sharpe = 0.0
    else:
        sharpe = (net_series.mean() / (net_series.std() + 1e-9)) * np.sqrt(bars_per_year)

    # Profit factor
    gross_profit = net_series[net_series > 0].sum()
    gross_loss = net_series[net_series < 0].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Win rate
    n_wins = (net_series > 0).sum()
    n_total = len(net_series)
    win_rate = n_wins / n_total if n_total > 0 else 0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # Average daily return
    # 1h: 24, 15m: 96, 4h: 6
    day_map = {'15m': 96, '1h': 24, '4h': 6, '1d': 1}
    bars_per_day = day_map.get(timeframe, 24) / rebalance_freq
    avg_daily_return = net_series.mean() * bars_per_day

    # ─── Extract Top/Bottom assets by frequency ───
    if not trade_log_df.empty:
        long_freq = (
            trade_log_df[trade_log_df['side'] == 'LONG']
            .groupby('symbol')
            .agg(count=('timestamp', 'size'), avg_return=('actual_return', 'mean'))
            .sort_values('count', ascending=False)
            .head(top_n)
        )
        short_freq = (
            trade_log_df[trade_log_df['side'] == 'SHORT']
            .groupby('symbol')
            .agg(count=('timestamp', 'size'), avg_return=('actual_return', 'mean'))
            .sort_values('count', ascending=False)
            .head(bottom_n)
        )
        top_assets = long_freq.reset_index().to_dict('records')
        bottom_assets = short_freq.reset_index().to_dict('records')
    else:
        top_assets = []
        bottom_assets = []

    return {
        'sharpe': round(float(sharpe), 4),
        'profit_factor': round(float(profit_factor), 4),
        'win_rate': round(float(win_rate), 4),
        'total_return': round(float(total_return), 4),
        'n_rebalances': len(portfolio_returns),
        'avg_daily_return': round(float(avg_daily_return), 6),
        'max_drawdown': round(float(max_drawdown), 4),
        'equity_curve': equity_curve,
        'trade_log': trade_log_df,
        'top_assets': top_assets,
        'bottom_assets': bottom_assets,
        'mc_stats': mc_results,
    }


def run_monte_carlo(returns, sims=2500):
    """
    Performs bootstrap resampling on the return series to test robustness.
    """
    if len(returns) == 0:
        return {'prob_profit': 0, 'ci_lower': 0, 'ci_upper': 0, 'mean_return': 0}

    bootstrapped_returns = []
    for _ in range(sims):
        # Sample with replacement
        resampled = np.random.choice(returns, size=len(returns), replace=True)
        # Calculate cumulative return: prod(1 + r) - 1
        cum_ret = np.prod(1 + resampled) - 1
        bootstrapped_returns.append(cum_ret)

    bootstrapped_returns = np.array(bootstrapped_returns)
    
    prob_profit = (bootstrapped_returns > 0).mean()
    ci_lower = np.percentile(bootstrapped_returns, 5)
    ci_upper = np.percentile(bootstrapped_returns, 95)
    mean_ret = bootstrapped_returns.mean()

    return {
        'prob_profit': round(float(prob_profit), 4),
        'ci_lower': round(float(ci_lower), 4),
        'ci_upper': round(float(ci_upper), 4),
        'mean_return': round(float(mean_ret), 4),
        'sims': sims
    }


def _empty_result():
    """Returns a zeroed-out result dict when simulation can't run."""
    return {
        'sharpe': 0.0,
        'profit_factor': 0.0,
        'win_rate': 0.0,
        'total_return': 0.0,
        'n_rebalances': 0,
        'avg_daily_return': 0.0,
        'max_drawdown': 0.0,
        'equity_curve': pd.Series(dtype=float),
        'trade_log': pd.DataFrame(),
        'top_assets': [],
        'bottom_assets': [],
        'mc_stats': {'prob_profit': 0, 'ci_lower': 0, 'ci_upper': 0, 'mean_return': 0, 'sims': 0},
    }


def format_simulation_summary(results: dict) -> str:
    """Formats simulation results into a human-readable string for the report/LLM."""
    if results['n_rebalances'] == 0:
        return "⚠️ No simulation data available (insufficient OOS data)."

    return (
        f"=== OOS Dry Run Simulation ===\n"
        f"  Total Return:     {results['total_return']:+.2%}\n"
        f"  Sharpe Ratio:     {results['sharpe']:.2f}\n"
        f"  Profit Factor:    {results['profit_factor']:.2f}\n"
        f"  Win Rate:         {results['win_rate']:.1%}\n"
        f"  Max Drawdown:     {results['max_drawdown']:.2%}\n"
        f"  Rebalances:       {results['n_rebalances']}\n"
        f"  Avg Daily Return: {results['avg_daily_return']:+.4%}\n"
        f"=== Robustness (Monte Carlo) ===\n"
        f"  Prob. of Profit:  {results['mc_stats']['prob_profit']:.1%}\n"
        f"  95% CI Lower:     {results['mc_stats']['ci_lower']:+.2%}\n"
        f"  95% CI Upper:     {results['mc_stats']['ci_upper']:+.2%}\n"
    )
