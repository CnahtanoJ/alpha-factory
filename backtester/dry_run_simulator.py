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
DEFAULT_TAKER_FEE = 0.00055   # Hyperliquid taker fee (0.055%)
DEFAULT_SLIPPAGE  = 0.0005    # 5 bps slippage estimate
ANNUAL_FACTOR_1H  = 8760      # Hourly candles per year


def simulate_portfolio(
    mega_df: pd.DataFrame,
    predictions: np.ndarray,
    feature_names: list,
    top_n: int = 10,
    bottom_n: int = 10,
    rebalance_freq: int = 6,
    fee_rate: float = DEFAULT_TAKER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
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
    feature_names : list
        List of feature column names used by the model (for importance).
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
        'top_assets' (list), 'bottom_assets' (list)
    """
    df = mega_df.copy()
    df['predicted_rank'] = predictions

    # P1-7 FIX: Ensure rebalance_freq aligns with the fwd_return horizon
    if rebalance_freq != 6:
        print(f"⚠️ WARNING: rebalance_freq ({rebalance_freq}) != fwd_return horizon (6). Returns will be distorted!")

    # ─── Group by timestamp to get cross-sectional snapshots ───
    timestamps = df['timestamp'].unique()
    timestamps.sort()

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

        # Rank and select
        snapshot = snapshot.sort_values('predicted_rank', ascending=False)
        longs = snapshot.head(top_n)
        shorts = snapshot.tail(bottom_n)
        
        curr_longs = set(longs['symbol'])
        curr_shorts = set(shorts['symbol'])

        # Calculate the holding period return (until next rebalance)
        # Use the actual forward return from the data
        long_return = longs['fwd_return'].mean()
        short_return = shorts['fwd_return'].mean()

        # Long/Short portfolio: profit on longs going up, shorts going down
        # Equal-weighted, dollar-neutral
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
        for _, row in longs.iterrows():
            trade_log.append({
                'timestamp': rb_ts, 'symbol': row['symbol'],
                'side': 'LONG', 'predicted_rank': row['predicted_rank'],
                'actual_return': row['fwd_return']
            })
        for _, row in shorts.iterrows():
            trade_log.append({
                'timestamp': rb_ts, 'symbol': row['symbol'],
                'side': 'SHORT', 'predicted_rank': row['predicted_rank'],
                'actual_return': -row['fwd_return']  # Shorts profit on decline
            })

    if not portfolio_returns:
        return _empty_result()

    returns_df = pd.DataFrame(portfolio_returns)
    returns_df = returns_df.set_index('timestamp')
    trade_log_df = pd.DataFrame(trade_log)

    # ─── Metrics (repurposed from legacy engine.py) ───
    net_series = returns_df['net_return'].dropna()

    # Equity curve
    equity_curve = (1 + net_series).cumprod()

    # Total return
    total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) > 0 else 0

    # Sharpe ratio (annualized, assuming hourly rebalance * rebalance_freq)
    bars_per_year = ANNUAL_FACTOR_1H / rebalance_freq
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

    # Average daily return (assuming 24 bars per day for 1h)
    bars_per_day = 24 / rebalance_freq
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
    )
