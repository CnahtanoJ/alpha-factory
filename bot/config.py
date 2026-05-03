import os
from hyperliquid.utils import constants

AWS_BUCKET = "flaminghotcheetos"
CONFIG_FILE = "champion_blueprint.json" 
LEADERBOARD_FILE = "leaderboard_results.json"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MARGIN_LIMIT = 0.44
# M-3: DCA_LIMIT=1 is intentional. The market-neutral basket strategy does not DCA;
# each position gets a single entry sized by risk parity. The DCA logic in risk_engine
# is dormant but preserved in case we revert to a single-asset strategy.
DCA_LIMIT = 1
TESTNET_MODE = os.environ.get("TESTNET_MODE", "True").lower() == "true"
BASE_URL = constants.TESTNET_API_URL if TESTNET_MODE else constants.MAINNET_API_URL
