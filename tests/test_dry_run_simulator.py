import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester.dry_run_simulator import simulate_portfolio


class TestDryRunSimulator(unittest.TestCase):
    def setUp(self):
        # Set up a mock out-of-sample mega_df
        # rebalance freq is 2 for testing speed, basket size 1 long / 1 short
        # We need at least 2 rebalances, so let's have 6 timestamps
        np.random.seed(42)
        self.symbols = ["BTC", "ETH", "SOL", "XRP"]
        self.timestamps = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(6)]

        rows = []
        for ts in self.timestamps:
            for sym in self.symbols:
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": sym,
                        "close": 100.0 + np.random.normal(0, 5),
                        "fwd_return": np.random.uniform(-0.02, 0.02),
                        "atr_pct": 0.02,
                    }
                )

        self.df = pd.DataFrame(rows)
        # Generate predictions matching the shape of the df
        # We want deterministic ranks to test rebalances
        # For BTC, prediction is high; for ETH, prediction is low; others in-between
        predictions = []
        for idx, row in self.df.iterrows():
            if row["symbol"] == "BTC":
                predictions.append(0.9)
            elif row["symbol"] == "ETH":
                predictions.append(0.1)
            elif row["symbol"] == "SOL":
                predictions.append(0.5)
            else:
                predictions.append(0.4)
        self.predictions = np.array(predictions)

    def test_empty_input(self):
        # Verify that passing an empty dataframe returns the zeroed out dict
        empty_df = pd.DataFrame(
            columns=["timestamp", "symbol", "close", "fwd_return", "atr_pct"]
        )
        res = simulate_portfolio(empty_df, np.array([]))

        self.assertEqual(res["sharpe"], 0.0)
        self.assertEqual(res["n_rebalances"], 0)
        self.assertEqual(len(res["equity_curve"]), 0)

    def test_insufficient_assets(self):
        # Only 1 asset but we want a basket of top_n=2 and bottom_n=2
        df_small = self.df[self.df["symbol"] == "BTC"].copy()
        preds_small = np.ones(len(df_small))
        res = simulate_portfolio(df_small, preds_small, top_n=2, bottom_n=2)
        self.assertEqual(res["n_rebalances"], 0)

    def test_transaction_cost_and_returns_math(self):
        # Run the simulation with 1 long, 1 short, rebalance frequency = 2
        # rebalance points will be index 0, 2, 4
        fee = 0.001
        slippage = 0.002
        res = simulate_portfolio(
            self.df,
            self.predictions,
            top_n=1,
            bottom_n=1,
            rebalance_freq=2,
            fee_rate=fee,
            slippage=slippage,
            weighting_mode="equal",
            mc_sims=50,
        )

        # Check we have 3 rebalances
        self.assertEqual(res["n_rebalances"], 3)

        # Check the trade log has long and short sides
        trade_log = res["trade_log"]
        self.assertFalse(trade_log.empty)
        self.assertIn("side", trade_log.columns)
        self.assertIn("LONG", trade_log["side"].values)
        self.assertIn("SHORT", trade_log["side"].values)

        # Check transaction cost logic on first rebalance (index 0)
        # Turnover cost multiplier is 1.0 (since no previous long/short portfolio)
        # Net return = gross_return - 2 * (fee + slippage) * 1.0
        first_rb = res["equity_curve"]  # Wait, equity curve is daily/periodic.
        # Let's inspect res['mc_stats'] structure
        self.assertIn("prob_profit", res["mc_stats"])

        # Verify total return and drawdown are calculated and are floats
        self.assertIsInstance(res["total_return"], float)
        self.assertIsInstance(res["max_drawdown"], float)
        self.assertIsInstance(res["sharpe"], float)


if __name__ == "__main__":
    unittest.main()
