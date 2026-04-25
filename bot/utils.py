import json
import boto3
from botocore.exceptions import ClientError
import requests
import logging
from bot.config import AWS_BUCKET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger()

class S3Interface:
    def __init__(self, bucket):
        self.bucket = bucket
        self.s3 = boto3.client('s3')

    def load_json(self, filename):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=filename)
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError: return {}

    def save_json(self, filename, data):
        try:
            self.s3.put_object(Bucket=self.bucket, Key=filename, Body=json.dumps(data, indent=4))
        except Exception as e: logger.error(f"S3 Save Error: {e}")

    def download_file(self, s3_key, local_path):
        """Downloads a raw file (like a .txt model) from S3 to local disk."""
        try:
            self.s3.download_file(self.bucket, s3_key, local_path)
            return True
        except Exception as e:
            logger.error(f"S3 Download Error for {s3_key}: {e}")
            return False

def is_authorized(chat_id):
    """
    The Gatekeeper: Verifies if a chat_id is the owner designated in .env.
    Use this to lock down command handlers if you add interactivity.
    """
    if not TELEGRAM_CHAT_ID:
        return False
    return str(chat_id) == str(TELEGRAM_CHAT_ID)

class StateManager:
    def __init__(self, bucket_name, filename="bot_state.json"):
        self.filename = filename
        self.s3 = S3Interface(bucket_name)
        self.state = self.s3.load_json(filename)

    def save(self): self.s3.save_json(self.filename, self.state)
    def get(self, coin, key, default=0): return self.state.get(f"{coin}_{key}", default)
    def set(self, coin, key, value, defer_save=True): 
        self.state[f"{coin}_{key}"] = value
        if not defer_save: self.save()
    def clear(self, coin, defer_save=True):
        for k in [k for k in self.state if k.startswith(f"{coin}_")]: del self.state[k]
        if not defer_save: self.save()

def send_telegram_message(text):
    """
    Sends a message to Telegram, automatically splitting it if it exceeds the 4096-char limit.
    Uses Markdown formatting.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing. Skipping message.")
        return

    MAX_LENGTH = 4000 # Safety margin
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Split message into chunks if necessary
    chunks = [text[i:i + MAX_LENGTH] for i in range(0, len(text), MAX_LENGTH)]
    
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": chunk, 
            "parse_mode": "Markdown" # MarkdownV2 is too strict for long automated reports
        }
        try:
            requests.post(url, json=payload, timeout=15)
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

def send_telegram_receipt(stats):
    # Emojis based on profitability 
    header_emoji = "🟢" if stats['net_profit'] > 0 else "🔴"
    
    msg = f"{header_emoji} **Vault Daily Close** {header_emoji}\n\n"
    msg += f"💵 **Net Profit:** `${stats['net_profit']:.2f}`\n"
    msg += f"📊 **Gross PnL:** `${stats['gross_pnl']:.2f}`\n"
    msg += f"💸 **Exchange Fees:** `-${stats['fees']:.2f}`\n"
    msg += f"⏳ **Funding Paid/Earned:** `${stats['funding']:.2f}`\n\n"
    msg += f"🔄 **Trades Executed:** `{stats['trades']}`\n"
    msg += f"🌊 **Volume Traded:** `${stats['volume']:,.2f}`\n"
    
    # Send this 'msg' string to your Telegram API
    return msg

