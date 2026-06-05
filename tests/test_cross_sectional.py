import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analytics.cross_sectional import add_time_series_features, get_fwd_return_bars


class TestCrossSectional(unittest.TestCase):
    def setUp(self):
        # Generate 150 rows of dummy OHLCV data
        np.random.seed(42)
        timestamps = [
            datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(150)
        ]
        close_prices = 100.0 + np.cumsum(np.random.normal(0, 1, 150))
        high_prices = close_prices + np.random.uniform(0.1, 1.0, 150)
        low_prices = close_prices - np.random.uniform(0.1, 1.0, 150)
        open_prices = close_prices + np.random.normal(0, 0.5, 150)
        volume = np.random.uniform(1000, 5000, 150)

        self.df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
                "index_close": close_prices * 1.01,  # dummy index close
                "last_funding_rate": np.random.uniform(-0.001, 0.001, 150),
                "sum_open_interest_value": np.random.uniform(100000, 500000, 150),
                "sum_toptrader_long_short_ratio": np.random.uniform(0.8, 1.2, 150),
                "sum_long_short_ratio": np.random.uniform(0.9, 1.1, 150),
                "sum_taker_long_short_vol_ratio": np.random.uniform(0.9, 1.1, 150),
                "timeframe_name": ["15m"] * 150,
            }
        )

        # Dummy BTC df
        self.btc_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "btc_close": 40000.0 + np.cumsum(np.random.normal(0, 100, 150)),
            }
        )

    def test_feature_generation_shapes_and_columns(self):
        # Generate features
        featured_df = add_time_series_features(self.df.copy(), self.btc_df.copy())

        # Verify shape
        self.assertEqual(len(featured_df), len(self.df))

        # Verify some key columns are present
        expected_cols = [
            "rsi",
            "macd",
            "volatility_20",
            "atr_pct",
            "basis_pct",
            "oi_usd",
            "funding_rate",
            "corr_to_index",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "sentiment_divergence",
            "cvd_slope_5",
            "price_cvd_divergence",
            "trend_convergence",
            "bbw_squeeze",
            "fwd_return",
            "risk_adj_ret",
            "fwd_max_ret",
            "fwd_min_ret",
        ]
        for col in expected_cols:
            self.assertIn(col, featured_df.columns, f"Missing column: {col}")

        # Verify float32 casting wasn't messed up and data is populated
        self.assertFalse(featured_df["rsi"].isna().all(), "RSI is all NaN")
        self.assertFalse(
            featured_df["volatility_20"].isna().all(), "Volatility is all NaN"
        )

    def test_target_calculation(self):
        # Run feature generation
        featured_df = add_time_series_features(self.df.copy(), self.btc_df.copy())

        # 15m default horizon is 12 bars (3 hours)
        fwd_bars = get_fwd_return_bars("15m")
        self.assertEqual(fwd_bars, 12)

        # Pick an index where target should be valid
        idx = 50
        # fwd_return at idx should be (close[idx + 12] / close[idx]) - 1
        expected_fwd = (
            self.df["close"].iloc[idx + fwd_bars] / self.df["close"].iloc[idx]
        ) - 1
        self.assertAlmostEqual(
            featured_df["fwd_return"].iloc[idx], expected_fwd, places=6
        )

        # risk_adj_ret should be fwd_return / (atr_pct + 1e-9)
        expected_risk_adj = expected_fwd / (featured_df["atr_pct"].iloc[idx] + 1e-9)
        self.assertAlmostEqual(
            featured_df["risk_adj_ret"].iloc[idx], expected_risk_adj, places=6
        )

        # The last fwd_bars rows should have NaN targets because there's no future data
        for i in range(len(featured_df) - fwd_bars, len(featured_df)):
            self.assertTrue(
                np.isnan(featured_df["fwd_return"].iloc[i]),
                f"Index {i} should be NaN target",
            )

    def test_target_calculation_with_gaps(self):
        # Introduce a gap: skip elements 80 to 90
        df_with_gap = self.df.copy()
        df_with_gap = df_with_gap.drop(range(80, 90)).reset_index(drop=True)

        featured_df = add_time_series_features(df_with_gap, self.btc_df.copy())
        fwd_bars = get_fwd_return_bars("15m")

        # Check index 75: 75 + 12 is 87, which was dropped, so the time diff should not match expected_delta
        # Therefore fwd_return at index 75 (or corresponding timestamps that span the gap) must be NaN
        # Find timestamp at index 75
        ts_75 = featured_df["timestamp"].iloc[75]
        # The next timestamp in the df after 12 bars should be 12 * 15m = 3 hours later
        # But due to the drop, the timestamp 12 bars later will be further in the future
        expected_ts = ts_75 + timedelta(minutes=15 * fwd_bars)
        actual_future_ts = featured_df["timestamp"].iloc[75 + fwd_bars]

        self.assertNotEqual(actual_future_ts, expected_ts)
        self.assertTrue(
            np.isnan(featured_df["fwd_return"].iloc[75]),
            "Target should be NaN across timestamp gaps",
        )


if __name__ == "__main__":
    unittest.main()
