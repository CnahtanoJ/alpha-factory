import os
import json
import time
import pandas as pd
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

class HyperliquidBot:
    def __init__(self, exchange, info, coin, timeframe, bucket, strategy, risk_engine):
        self.exchange = exchange
        self.info = info
        self.address = exchange.account_address 
        self.coin = coin
        self.timeframe = timeframe
        self.strategy = strategy
        self.md = MarketData(self.info)
        self.risk = risk_engine

    def run_tick(self, portfolio, user_state):
        
        # 1. 🛡️ ACCOUNT SAFETY FIRST 
        if not self.risk.check_safety(user_state): 
            logger.warning("❌ Safety Check Failed. Aborting tick.")
            return
            
        # 2. 📥 SINGLE DATA FETCH
        df = self.md.get_clean_candles(self.coin, interval=self.timeframe, limit=1000)
        if df.empty: 
            logger.warning(f"⚠️ {self.coin}: Failed to fetch data.")
            return

        # 3. 🫀 THE HEARTBEAT: Calculate ATR immediately
        high, low = df['high'], df['low']
        prev_close = df['close'].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df['atr_pct'] = (tr.rolling(14).mean() / df['close'])
        
        current_atr_pct = float(df['atr_pct'].iloc[-1]) if pd.notna(df['atr_pct'].iloc[-1]) else 0.015

        open_orders = self.info.frontend_open_orders(self.address)
        # 4. 🧹 RISK CLEANUP
        self.risk.clean_global_zombies(portfolio, open_orders)
        self.risk.sync_break_even(self.coin, current_atr_pct, portfolio, open_orders)

        # 5. 🧠 THE BRAIN (Get Signals First!)
        signals = self.strategy.get_signal_column(df)
        last_closed_time = df.index[-2]
        current_signal = signals.iloc[-2] 

        # 🚀 EARLY EXIT: Save server CPU if there is no signal to act on
        if current_signal == 0:
            logger.info(f"Neutral for {self.coin} | Last Closed: {last_closed_time}")
            return 

        # 6. 🔍 HEAVY EXECUTION MATH (Only runs if a signal fired)
        logger.info(f"🔍 Analyzing Candle: {last_closed_time} | Signal: {current_signal}")
        
        poc_price = get_local_poc(df, num_bins=50, lookback=200)
        signal_price = df['close'].iloc[-2]
        current_price = df['close'].iloc[-1]
        cvd_slope = get_cvd_slope(df, lookback=5)
        cvd_text = f"CVD Slope: {cvd_slope:,.0f}"

        # 7. 🚦 THE BOUNCER INTEGRATION
        if current_signal == 1: 
            if self.risk.check_execution_safety(self.coin, is_buy=True, current_price=current_price, poc_price=poc_price):
                msg = f"🟢 BULLISH FOR {self.coin}\n📊 Signal Formed At: ${signal_price:.4f}\n💵 Live Entry: ~${current_price:.4f}\n🧲 POC Magnet: ${poc_price:.4f}\n{cvd_text}"
                send_telegram_message(msg)
                self.risk.execute_logic(self.coin, "BULLISH", self.timeframe, current_atr_pct, portfolio, user_state, open_orders)
            else:
                logger.info(f"🚫 BULLISH signal on {self.coin} blocked by Live Sensor.")

        elif current_signal == -1: 
            if self.risk.check_execution_safety(self.coin, is_buy=False, current_price=current_price, poc_price=poc_price):
                msg = f"🔴 BEARISH FOR {self.coin}\n📊 Signal Formed At: ${signal_price:.4f}\n💵 Live Entry: ~${current_price:.4f}\n🧲 POC Magnet: ${poc_price:.4f}\n{cvd_text}"
                send_telegram_message(msg)
                self.risk.execute_logic(self.coin, "BEARISH", self.timeframe, current_atr_pct, portfolio, user_state, open_orders)
            else:
                logger.info(f"🚫 BEARISH signal on {self.coin} blocked by Live Sensor.")

        elif current_signal == 2:
            if self.coin in portfolio:
                send_telegram_message(f"💨 THESIS INVALIDATED. FULLY EXITING {self.coin}!")
                self.risk.execute_logic(self.coin, "EXIT", self.timeframe, current_atr_pct, portfolio, user_state, open_orders)

# ==========================================
if TESTNET_MODE:
    KEY = os.environ.get("TESTNET_PRIVATE_KEY")
    ADDR = os.environ.get("TESTNET_ACCOUNT_ADDRESS")
else:
    KEY = os.environ.get("MAINNET_PRIVATE_KEY")
    ADDR = os.environ.get("MAINNET_ACCOUNT_ADDRESS")

if not KEY or not ADDR:
    raise ValueError(f"❌ MISSING CREDENTIALS! Mode is {TESTNET_MODE}, but keys not found in Env Vars.")

# --- HANDLER 1: THE SOLDIER (Executor) ---
def executor_handler(event, context):
    logger.info(f"Received event: {event}")
    task = event.get("task", "execute_trades")
    info = Info(BASE_URL, skip_ws=True)
    
    if task == "send_daily_report":
        logger.info("Generating daily Telegram receipt...")
      
        # Run the fetcher and formatter functions
        stats = fetch_daily_receipt(info, ADDR)
        msg = send_telegram_receipt(stats)
        
        # Use your existing Telegram helper!
        send_telegram_message(msg)
        
        return {'statusCode': 200, 'body': 'Daily report sent'}

    elif task == "execute_trades":
        # 1. Load Orders & Intelligence
        s3 = S3Interface(AWS_BUCKET)
        risk = RiskEngine(bucket=AWS_BUCKET)
        
        # Load the Live Scout Intelligence (The Fused Ranking)
        scout_data = s3.load_json("live_scout_intelligence.json")
        if scout_data:
            # The Scout's #1 pick is our new target
            best_scout = scout_data[0]
            target_coin = best_scout.get("target_coin", "BTC").split("/")[0]
            target_tf = best_scout.get("timeframe", "15m")
            target_strat = best_scout.get("strategy", "SimpleBreakout")
            target_params = best_scout.get("params", {"n": 50})
            target_conviction = best_scout.get("live_conviction", 1.0)
            
            logger.info(f"🛰️ BOT: Using Live Scout Intelligence for {target_coin} (Conviction: {target_conviction:.2%})")
        else:
            # Standard Fallback to Weekly Blueprint
            config = s3.load_json(CONFIG_FILE)
            target_coin = config.get("target_coin", "BTC").split("/")[0]
            target_tf = config.get("timeframe", "15m")
            target_strat = config.get("strategy", "SimpleBreakout")
            target_params = config.get("params", {"n": 50})
            target_conviction = 1.0
            logger.warning("🛰️ BOT: No Scout data found. Falling back to Weekly Blueprint.")

        user_state = info.user_state(ADDR)
        all_mids = info.all_mids()
        
        account = Account.from_key(KEY)
        exchange = Exchange(account, BASE_URL, account_address=ADDR)
        
        # Attach dependencies to risk before use
        risk.exchange = exchange
        risk.info = info
        
        portfolio = risk.parse_portfolio(user_state, all_mids)
        temp_assets = AssetManager(info)
        needs_state_refresh = False
        
        if target_coin == "SLEEP":
            logger.warning("💤 MARKET SLEEP MODE ACTIVATED.")

            for coin, pos_data in portfolio.items():
                sz = float(pos_data['szi'])
                
                logger.info(f"🧹 FLUSHING: Closing {coin} position due to Market Sleep.")
                send_telegram_message(f"🧹 FLUSHING: Closing {coin} to preserve capital (Market Toxic).")
                
                try: risk.cancel_all_orders(coin)
                except: pass
                
                # IOC IOC
                is_buy = sz < 0
                slippage = 0.002
                raw_price = float(all_mids[coin])
                curr_px = temp_assets.get_price_precision(coin, raw_price)
                
                raw_limit_px = curr_px * (1 + slippage) if is_buy else curr_px * (1 - slippage)
                limit_px = temp_assets.get_price_precision(coin, raw_limit_px)
                
                exchange.order(coin, is_buy, abs(sz), limit_px, {"limit": {"tif": "Ioc"}}, reduce_only=True)
                
                StateManager(AWS_BUCKET).clear(coin)
                    
            return {'statusCode': 200, 'body': json.dumps('Sleep Mode - Positions Flushed')}

        # 2. Check for "Regime Change" (Switching Coins)        
        active_coin = next(iter(portfolio), None)
        if active_coin:
            pass # logger.info(f"🎯 FOUND REAL POSITION: {active_coin}")
                    
        # 3. CONFLICT RESOLUTION (Dynamic Opportunity Cost) ⚖️
        if active_coin and active_coin != target_coin:
            
            market_scan = s3.load_json(LEADERBOARD_FILE)
            
            current_holding_score = -999 
            new_target_score = -999
            
            if market_scan and active_coin in market_scan:
                current_holding_score = market_scan[active_coin].get('score', -999)
            
            if market_scan and target_coin in market_scan:
                new_target_score = market_scan[target_coin].get('score', -999)
            
            SCORE_SWITCH_THRESHOLD = 1.15
            should_swap = False

            if current_holding_score <= 0:
                logger.info(f"🚨 BAILOUT: Current score is {current_holding_score:.2f}. Instant swap to target!")
                should_swap = True
                
            else: 
                score_ratio = new_target_score / current_holding_score 
                logger.info(f"⚖️ COMPARISON: Holding {active_coin} (Score: {current_holding_score:.2f}) vs Target {target_coin} (Score: {new_target_score:.2f}) | Ratio: {score_ratio:.2f}")

                if score_ratio > SCORE_SWITCH_THRESHOLD:
                    logger.info(f"🔄 SWAP APPROVED: Target is {((score_ratio - 1) * 100):.1f}% better. Executing swap.")
                    should_swap = True
                else:
                    logger.info(f"🧲 STICKY HOLD: {active_coin} is still good enough.")

            if should_swap:
                success = risk.close_active_position(
                    active_coin=active_coin, 
                    all_mids=all_mids, 
                    temp_assets=temp_assets,
                    portfolio=portfolio, 
                    aws_bucket=AWS_BUCKET
                )
                
                if success:
                    needs_state_refresh = True
                else:
                    # Brain cancels the swap
                    should_swap = False

            if not should_swap:
                target_coin = active_coin
                
                if market_scan and active_coin in market_scan:
                    target_strat = market_scan[active_coin]['strategy']
                    target_params = market_scan[active_coin]['params']
                    target_tf = market_scan[active_coin]['timeframe']

        if needs_state_refresh:
            user_state = info.user_state(ADDR)
            portfolio = risk.parse_portfolio(user_state, all_mids)

        # 4. EXECUTE NEW ORDERS (Only if no legacy position blocks us)
        if target_strat in STRATEGY_CONFIG:
            strat_class = STRATEGY_CONFIG[target_strat]["class"]
            chosen_strat = strat_class(**target_params)
        else:
            chosen_strat = SimpleBreakout(n=50)

        # 5. CONVICTION BAILOUT
        # If the target is SLEEP or conviction is too low, we stop
        if target_conviction < 0.30: # 30% conviction floor
             logger.warning(f"📉 BAILOUT: AI Conviction for {target_coin} too low ({target_conviction:.2%}). Staying flat.")
             # Add logic to close existing pos if target_coin matches active_coin
             return

        bot = HyperliquidBot(exchange, info, target_coin, target_tf, AWS_BUCKET, chosen_strat, risk)
        bot.run_tick(portfolio, user_state)
        
        return {'statusCode': 200, 'body': json.dumps('Executed')}
