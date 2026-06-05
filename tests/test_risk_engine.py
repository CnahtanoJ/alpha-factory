import unittest
from unittest.mock import MagicMock, patch
import os
import sys

from bot.risk_engine import RiskEngine


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        # Mock StateManager before instantiating RiskEngine to avoid real S3 calls
        self.patcher = patch("bot.risk_engine.StateManager")
        self.mock_state_manager = self.patcher.start()
        self.mock_memory = MagicMock()
        self.mock_state_manager.return_value = self.mock_memory

        # Mock components
        self.mock_exchange = MagicMock()
        self.mock_info = MagicMock()

        # Initialize RiskEngine with mocked dependencies
        self.engine = RiskEngine(
            exchange=self.mock_exchange,
            info=self.mock_info,
            account_address="0x1234567890abcdef",
            bucket="mock-bucket",
        )

    def tearDown(self):
        self.patcher.stop()

    def test_margin_check_bankruptcy(self):
        # Account value is 0 (bankruptcy)
        user_state = {
            "marginSummary": {"accountValue": "0.0", "totalMarginUsed": "0.0"}
        }
        self.mock_info.user_state.return_value = user_state

        # check_safety should return False
        self.assertFalse(self.engine.check_safety())

    def test_margin_check_excess_leverage(self):
        # Margin limit is 44% (0.44)
        # Account value: 10,000, Margin used: 5,000 (50% usage > 44% limit)
        user_state = {
            "marginSummary": {"accountValue": "10000.0", "totalMarginUsed": "5000.0"}
        }
        self.mock_info.user_state.return_value = user_state

        # check_safety should return False
        self.assertFalse(self.engine.check_safety())

    def test_margin_check_safe(self):
        # Account value: 10,000, Margin used: 3,000 (30% usage <= 44% limit)
        user_state = {
            "marginSummary": {"accountValue": "10000.0", "totalMarginUsed": "3000.0"}
        }
        self.mock_info.user_state.return_value = user_state

        # check_safety should return True
        self.assertTrue(self.engine.check_safety())

    def test_circuit_breaker_blocks_entries(self):
        # Mock consecutive losses >= 3 in state manager
        self.mock_memory.get.return_value = 3

        portfolio = {}  # Empty portfolio means we want a new entry
        user_state = {
            "marginSummary": {"accountValue": "10000.0", "totalMarginUsed": "0.0"}
        }
        open_orders = []

        # Try to execute long entry logic
        self.engine.execute_logic(
            coin="SOL",
            signal="BULLISH",
            timeframe="15m",
            current_atr_pct=0.02,
            portfolio=portfolio,
            user_state=user_state,
            open_orders=open_orders,
        )

        # Verify that exchange.order was NOT called because of circuit breaker
        self.mock_exchange.order.assert_not_called()

    def test_consecutive_losses_update_loss(self):
        # Test closing a position at a loss increases consecutive_losses streak
        portfolio = {"SOL": {"szi": "10.0", "entryPx": "150.0"}}  # Long position
        all_mids = {"SOL": "140.0"}  # Price went down, so it's a loss

        # Mock memory streak retrieval
        self.mock_memory.get.return_value = 1

        # Mock exchange order response
        self.mock_exchange.order.return_value = {
            "status": "ok",
            "response": {"data": {"statuses": [{"filled": {"totalSz": "10.0"}}]}},
        }

        mock_assets = MagicMock()
        mock_assets.get_price_precision.side_effect = lambda coin, px: px

        # Call close active position
        res = self.engine.close_active_position(
            "SOL", all_mids, mock_assets, portfolio, "mock-bucket"
        )

        # Verify close succeeded
        self.assertTrue(res)

        # Verify consecutive_losses streak was incremented to 2
        self.mock_memory.set.assert_called_with("GLOBAL", "consecutive_losses", 2)

    def test_consecutive_losses_update_profit(self):
        # Test closing a position in profit resets consecutive_losses streak to 0
        portfolio = {"SOL": {"szi": "10.0", "entryPx": "150.0"}}  # Long position
        all_mids = {"SOL": "160.0"}  # Price went up, so it's a profit

        # Mock memory streak retrieval
        self.mock_memory.get.return_value = 2

        # Mock exchange order response
        self.mock_exchange.order.return_value = {
            "status": "ok",
            "response": {"data": {"statuses": [{"filled": {"totalSz": "10.0"}}]}},
        }

        mock_assets = MagicMock()
        mock_assets.get_price_precision.side_effect = lambda coin, px: px

        # Call close active position
        res = self.engine.close_active_position(
            "SOL", all_mids, mock_assets, portfolio, "mock-bucket"
        )

        # Verify close succeeded
        self.assertTrue(res)

        # Verify consecutive_losses streak was reset to 0
        self.mock_memory.set.assert_called_with("GLOBAL", "consecutive_losses", 0)


if __name__ == "__main__":
    unittest.main()
