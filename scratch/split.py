import os

with open('core/hyperliquidbot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def get_block(start_str, end_str=None):
    start_idx = -1
    for i, l in enumerate(lines):
        if start_str in l:
            start_idx = i
            break
    if start_idx == -1: return []
    
    if end_str is None:
        return lines[start_idx:]
        
    end_idx = -1
    for i in range(start_idx, len(lines)):
        if end_str in lines[i]:
            end_idx = i
            break
            
    if end_idx == -1:
        return lines[start_idx:]
        
    return lines[start_idx:end_idx]

config_code = """import os
from hyperliquid.utils import constants

AWS_BUCKET = "flaminghotcheetos"
CONFIG_FILE = "strategy_config.json" 
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MARGIN_LIMIT = 0.44
DCA_LIMIT = 1
TESTNET_MODE = os.environ.get("TESTNET_MODE", "True").lower() == "true"
BASE_URL = constants.TESTNET_API_URL if TESTNET_MODE else constants.MAINNET_API_URL
"""
with open('bot/config.py', 'w', encoding='utf-8') as f: f.write(config_code)

utils_code = """import json
import boto3
from botocore.exceptions import ClientError
import requests
import logging
from bot.config import AWS_BUCKET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger()

""" + "".join(get_block('class S3Interface:', 'class AssetManager:')) + "".join(get_block('def send_telegram_message(text):', 'def calculate_adx'))

with open('bot/utils.py', 'w', encoding='utf-8') as f: f.write(utils_code)

data_feed_code = """import math
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from bot.config import AWS_BUCKET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger()

""" + "".join(get_block('class AssetManager:', 'def fetch_daily_receipt')) + "".join(get_block('def fetch_daily_receipt(info_client', 'def send_telegram_message'))

with open('bot/data_feed.py', 'w', encoding='utf-8') as f: f.write(data_feed_code)

indicators_code = """import numpy as np
import pandas as pd

""" + "".join(get_block('def calculate_adx(df', '2. STRATEGIES (The Candidates)')) + "".join(get_block('def get_local_poc', '5. MAIN CONTROLLER'))

with open('bot/indicators.py', 'w', encoding='utf-8') as f: f.write(indicators_code)

strategies_code = """import pandas as pd
import numpy as np
from bot.indicators import calculate_adx

""" + "".join(get_block('class VectorStrategy:', '3. BACKTESTER')) + "".join(get_block('STRATEGY_CONFIG = {', '7. AWS LAMBDA HANDLERS'))

with open('bot/strategies.py', 'w', encoding='utf-8') as f: f.write(strategies_code)

backtester_code = """import pandas as pd
import numpy as np
from bot.data_feed import MarketData

""" + "".join(get_block('def inject_htf_trend(', '4. EXECUTION'))

with open('backtester/engine.py', 'w', encoding='utf-8') as f: f.write(backtester_code)

risk_engine_code = """import time
import logging
from bot.utils import StateManager, send_telegram_message
from bot.data_feed import AssetManager
from bot.config import MARGIN_LIMIT, DCA_LIMIT

logger = logging.getLogger()

""" + "".join(get_block('class RiskEngine:', 'def get_local_poc'))

with open('bot/risk_engine.py', 'w', encoding='utf-8') as f: f.write(risk_engine_code)

bot_executor_code = """import os
import json
import time
import logging
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange

from bot.config import *
from bot.utils import S3Interface, StateManager, send_telegram_message, send_telegram_receipt
from bot.data_feed import MarketData, AssetManager, fetch_daily_receipt
from bot.indicators import get_local_poc, get_cvd_slope
from bot.strategies import STRATEGY_CONFIG, SimpleBreakout
from bot.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

""" + "".join(get_block('class HyperliquidBot:', '6. STRATEGIES')) + "".join(get_block('if TESTNET_MODE:'))

with open('bot/bot_executor.py', 'w', encoding='utf-8') as f: f.write(bot_executor_code)
