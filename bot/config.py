import os
# pyrefly: ignore [missing-import]
from hyperliquid.utils import constants

# --- CLOUD STORAGE ---
AWS_BUCKET = os.environ.get("AWS_BUCKET", "alpha-factory-models")

# --- NOTIFICATIONS ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- RISK PARAMETERS ---
MARGIN_LIMIT = 0.44
DCA_LIMIT = 1

# --- NETWORK ---
TESTNET_MODE = os.environ.get("TESTNET_MODE", "False").lower() == "true"
BASE_URL = constants.TESTNET_API_URL if TESTNET_MODE else constants.MAINNET_API_URL