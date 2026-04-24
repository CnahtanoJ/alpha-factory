import time
import logging
from bot.utils import StateManager, send_telegram_message
from bot.data_feed import AssetManager
from bot.config import MARGIN_LIMIT, DCA_LIMIT

logger = logging.getLogger()

class RiskEngine:
    def __init__(self, exchange=None, info=None, account_address=None, bucket=None):
        self.exchange = exchange
        self.info = info
        self.address = account_address
        self.assets = AssetManager(info) 
        self.memory = StateManager(bucket_name=bucket)
        # DCA SETTINGS
        self.dca_spacing = 0.02

    def check_safety(self, user_state):
        try:
            used = float(user_state['marginSummary']['totalMarginUsed'])
            val = float(user_state['marginSummary']['accountValue'])
            
            # 1. Check for Bankruptcy
            if val == 0: 
                logger.error("🚨 CRITICAL: Account Value is 0! Stopping.")
                return False
            
            # 2. Check Margin Usage
            curr_usage = used / val
            if curr_usage > MARGIN_LIMIT: 
                logger.warning(f"⚠️ HIGH RISK: Margin Usage {curr_usage:.2f} > {MARGIN_LIMIT}. Pausing new actions.")
                send_telegram_message(f"⚠️ High Margin Alert: {curr_usage:.2%}. Pausing new actions.")
                return False
            
            return True

        except Exception as e:
            logger.error(f"❌ SAFETY CHECK FAILED: {e}")
            return False

    def clean_global_zombies(self, portfolio, open_orders):
        # 1. HEARTBEAT LOG (Optional: Remove later if too noisy)
        # logger.info("🧹 Scanning for Zombies...")

        if not self.info or not self.exchange:
            logger.error("❌ Cannot clean zombies: RiskEngine missing 'info' or 'exchange'.")
            return

        if not open_orders: return
        active_positions = set(portfolio.keys())

        orders_by_coin = {}
        for o in open_orders:
            c = o['coin']
            if c not in orders_by_coin: orders_by_coin[c] = []
            orders_by_coin[c].append(o)
            
        for coin, orders in orders_by_coin.items():
            
            # A. If it's a Real Position, these orders are likely valid TP/SL. Skip.
            if coin in active_positions:
                continue
            
            # B. It's not a Real Position. Check for "Pending Entries".
            is_pending_entry = False
            for o in orders:
                # 1. Check Order Type (Must be Limit)
                if o['orderType'] == 'Limit':
                    
                    # 2. ROBUST REDUCE-ONLY CHECK
                    is_reduce_only = o.get('reduceOnly', False) or o.get('r', False)
                    if not is_reduce_only:
                        is_pending_entry = True
                        break
            
            if is_pending_entry:
                logger.info(f"⏳ {coin}: Ignoring Global Clean (Active Limit Entry detected)")
                continue
            
            # C. KILL ZOMBIES
            logger.warning(f"🧟 GLOBAL ZOMBIE DETECTED: {coin} has orders but only dust/no position.")
            send_telegram_message(f"🧟 GLOBAL CLEAN: Canceling all {coin} orders (Zombie/Dust).")
            
            self.cancel_all_orders(coin)
            time.sleep(1) # Small sleep to avoid rate limits
    
    def get_live_sensor_data(self, target_coin):
        # 🛡️ GUARD CLAUSE: Ensure the tool exists
        if not self.info:
            logger.error("❌ Cannot get sensor data: RiskEngine missing 'info'.")
            return None

        try:
            # 1. Fetch the master payload
            meta, ctxs = self.info.meta_and_asset_ctxs()

            # 2. Find the index number of our target coin
            coin_index = next(i for i, asset in enumerate(meta['universe']) if asset['name'] == target_coin)
            
            # 3. Pull the live data using that exact index
            coin_ctx = ctxs[coin_index]
            
            # 4. Format the institutional metrics
            funding_rate = float(coin_ctx['funding'])
            open_interest = float(coin_ctx['openInterest'])
            oracle_price = float(coin_ctx['oraclePx'])
            
            oi_usd = open_interest * oracle_price
            
            return {
                "funding_rate": funding_rate, 
                "oi_usd": oi_usd,
                "oracle_price": oracle_price
            }
            
        except Exception as e:
            logger.error(f"⚠️ Sensor error for {target_coin}: {e}")
            return None

    def check_execution_safety(self, coin, is_buy, current_price, poc_price):
        
        # 1. Pull the live data using the sensor (passing the coin down)
        sensor = self.get_live_sensor_data(coin)
        
        if not sensor:
            logger.error(f"⚠️ Sensor failure on {coin}. Aborting execution for safety.")
            return False

        oi_usd = sensor['oi_usd']
        funding = sensor['funding_rate']
        
        # 2. RULE 1: The Liquidity Floor ($10M Minimum)
        MIN_OI = 10_000_000 
        if oi_usd < MIN_OI:
            logger.warning(f"🛑 REJECTED: {coin} OI is only ${oi_usd:,.0f} (Needs ${MIN_OI:,.0f}). Danger of slippage.")
            return False

        # 3. RULE 2: The Funding Trap
        MAX_FUNDING = 0.0200  
        
        if is_buy and funding > MAX_FUNDING:
            logger.warning(f"🛑 REJECTED: {coin} is overcrowded. Funding rate too high ({funding:.4f}).")
            return False
            
        if not is_buy and funding < -MAX_FUNDING:
            logger.warning(f"🛑 REJECTED: {coin} short is overcrowded. Funding rate too negative ({funding:.4f}).")
            return False

        # 4. RULE 3: POC Gravity
        if is_buy and current_price > poc_price:
            logger.warning(f"🛑 REJECTED: {coin} Long at ${current_price:.4f} is above the POC (${poc_price:.4f}). Danger of downward gravity.")
            return False

        if not is_buy and current_price < poc_price:
            logger.warning(f"🛑 REJECTED: {coin} Short at ${current_price:.4f} is below the POC (${poc_price:.4f}). Danger of upward snapback.")
            return False        

        # If it passes the bouncer, greenlight the trade
        logger.info(f"🟢 CLEARED: {coin} execution safe. OI: ${oi_usd:,.0f} | Funding: {funding:.4f}")
        return True

    def parse_portfolio(self, user_state, all_mids, dust_threshold=1.0):
        """
        Runs ONCE per tick. Converts messy Hyperliquid JSON into a clean dictionary.
        Returns: {'SOL': {'szi': '10.5', 'entryPx': '150.2', ...}, 'HYPE': {...}}
        """
        portfolio = {}
        
        if not user_state or 'assetPositions' not in user_state:
            return portfolio
            
        for p in user_state['assetPositions']:
            pos = p['position']
            coin = pos['coin']
            sz = float(pos['szi'])
            
            if sz == 0:
                continue
                
            # Filter out the exchange dust
            price = float(all_mids.get(coin, 0))
            if abs(sz * price) > dust_threshold:
                portfolio[coin] = pos  # Save the entire clean dictionary under the coin's name
                
        return portfolio

    def cancel_all_orders(self, coin):
        orders = self.info.open_orders(self.address)
        coin_orders = [o for o in orders if o['coin'] == coin]
        
        if not coin_orders:
            return 

        logger.info(f"🗑️ Found {len(coin_orders)} orders to cancel for {coin}...")

        for o in coin_orders:
            try:
                self.exchange.cancel(coin, int(o['oid']))
                logger.info(f"   -> Cancelled order {o['oid']}")
            except Exception as e:
                logger.error(f"❌ Failed to cancel {o['oid']}: {e}")

    def execute_logic(self, coin, signal, timeframe, current_atr_pct, portfolio, user_state, open_orders):

        orders = [o for o in open_orders if o['coin'] == coin]
        
        has_position = coin in portfolio
        pos_details = portfolio.get(coin)

        if pos_details:
            pos_size = float(pos_details.get('szi', 0.0))
            avg_entry_px = float(pos_details.get('entryPx', 0.0))
            current_count = int(round(abs(pos_size) / base_sz_unit))
        else:
            pos_size = 0.0
            avg_entry_px = 0.0
            current_count = 0

        # THE STRATEGY EXIT OVERRIDE
        if signal == "EXIT" and has_position:
            logger.info(f"🏁 STRATEGY EXIT: {coin} hit the middle band. Closing position.")
            send_telegram_message(f"🏁 STRATEGY EXIT: {coin} | Closing position at Middle Band.")
            
            # Use your existing market close logic here:
            is_buy_close = (pos_size < 0)
            curr_mid = float(self.info.all_mids()[coin])
            close_px = curr_mid * (1 + 0.015) if is_buy_close else curr_mid * (1 - 0.015)
            close_px = self.assets.get_price_precision(coin, close_px)

            try:
                self.exchange.order(
                    coin, is_buy_close, abs(pos_size), close_px, 
                    {"limit": {"tif": "Ioc"}}, reduce_only=True
                )

            except Exception as e:
                logger.error(f"💥 FAILED TO CLOSE POSITION{e}")
                return
            
            self.cancel_all_orders(coin)
            self.memory.clear(coin)
            return # Stop processing, we are out!

        # 2. PENDING ORDER CHECKS
        if orders:
            has_limit = any(o['orderType'] == 'Limit' for o in orders)            
            if has_limit:
                logger.info(f"⏳ {coin}: Pending Limit Order. Waiting...")
                return
            
        # 3. MATH & STATE CALCULATION 🧮
        equity = float(user_state['marginSummary']['accountValue'])
        mid_px = float(self.info.all_mids()[coin])
        if mid_px == 0: return
        
        slippage = 0.002
                
        raw_limit_buy = mid_px * (1 + slippage)
        raw_limit_sell = mid_px * (1 - slippage)
        
        # Pre-calculate precise prices
        limit_buy_px = self.assets.get_price_precision(coin, raw_limit_buy)
        limit_sell_px = self.assets.get_price_precision(coin, raw_limit_sell)
        
        entry_usd = (equity * 0.25) # 25% of equity per leg for market-neutral basket (1.5x total leverage across 6 assets)
        base_sz_unit = entry_usd / mid_px
                
        # 4. DECISION LOGIC 🧠

        just_flipped = False
        if has_position:
            if current_count < 1: current_count = 1
            # logger.info(f"📊 Status {coin}: Layer {current_count} | Avg Entry {avg_entry_px} | Price {mid_px}")
            curr_side = "BULLISH" if pos_size > 0 else "BEARISH"
            is_long = pos_size > 0

            HOLD_MAP = {
                "5m": 900,
                "15m": 2700,
                "1h": 10800
            }
    
            MIN_HOLD_SECONDS = HOLD_MAP.get(timeframe, 2700)

            if signal != curr_side:

                last_entry_time = self.memory.get(coin, 'entry_time', 0)
                current_time = int(time.time())
                MIN_HOLD_SECONDS = 1800
                time_held = current_time - last_entry_time

                current_pnl = (mid_px - avg_entry_px) / avg_entry_px
                if not is_long: current_pnl *= -1

                cond_time = time_held > MIN_HOLD_SECONDS
                cond_profit = self.memory.get(coin, 'tp1_hit')
                cond_emergency = current_pnl < -0.01

                # 💎 THE "SAFE FLIP" LOGIC
                if cond_time or cond_profit or cond_emergency:
                    
                    logger.info(f"🔄 FLIP SIGNAL: {curr_side} -> {signal} after {time_held/60} minutes. Switching sides!")
                    send_telegram_message(f"🔄 FLIP: Closing {curr_side} to go {signal}!")

                    # 1. CLOSE THE OLD POSITION (Market Close)
                    is_buy_close = (pos_size < 0)
                    slippage = 0.015
                    curr_mid = float(self.info.all_mids()[coin])
                    close_px = curr_mid * (1 + slippage) if is_buy_close else curr_mid * (1 - slippage)
                    close_px = self.assets.get_price_precision(coin, close_px)

                    try:
                        self.exchange.order(
                            coin, is_buy_close, abs(pos_size), close_px, 
                            {"limit": {"tif": "Ioc"}}, reduce_only=True
                        )

                    except Exception as e:
                        logger.error(f"💥 FAILED TO CLOSE POSITION DURING FLIP: {e}")
                        return
                    
                    self.cancel_all_orders(coin)
                    self.memory.clear(coin)
                    
                    has_position = False
                    just_flipped = True
                    logger.info("⚡ Ready to enter new position immediately...")
                    
                else:
                    logger.info(f"🛡️ IGNORING {signal}: Holding {curr_side} (TP1 not hit yet).")
                    return

            # Guard B: Max Layers
            if not just_flipped and current_count >= DCA_LIMIT: 
                logger.info(f"✋ {coin}: Max Layers ({DCA_LIMIT}) reached.")
                return

            # Guard C: THE SPACER (Prevent Machine Gun) 📏
            if not just_flipped:
                is_long = (curr_side == "BULLISH")
                dist_threshold = avg_entry_px * (1 - self.dca_spacing) if is_long else avg_entry_px * (1 + self.dca_spacing)
                should_dca = (mid_px < dist_threshold) if is_long else (mid_px > dist_threshold)
                
                if not should_dca:
                    current_pullback = (avg_entry_px - mid_px) / avg_entry_px if is_long else (mid_px - avg_entry_px) / avg_entry_px
                    direction_str = "DROP" if is_long else "RISE"
                    msg = f"⏳ {coin}: Spacing too tight. Price {direction_str} {current_pullback:.2%}. Waiting for {self.dca_spacing:.2%}."
                    
                    if current_pullback > (self.dca_spacing * 0.5):
                        logger.info(msg)
                        send_telegram_message(msg)

                    return
                
                logger.info(f"{coin}: DCA Triggered! Price gap sufficient.")
                send_telegram_message(f"{coin}: DCA Triggered at {mid_px}! Price gap sufficient.")
                
                pending_dca = next((
                    o for o in open_orders 
                    if o['coin'] == coin and o['isTrigger'] == False # Limit/Market orders
                    and o['side'] == ('B' if is_long else 'A') # Same side as us
                ), None)

                if pending_dca:
                    logger.warning(f"⚠️ DCA Aborted: Found pending order {pending_dca['oid']}. Waiting for fill.")
                    return

                logger.info(f"🧹 PRE-DCA CLEANUP: Cancelling all orders for {coin} to reset TP/SL.")
                try:
                    self.cancel_all_orders(coin)
                    time.sleep(1) 
                except Exception as e:
                    logger.error(f"⚠️ Failed to cancel orders: {e}")

        else:
            self.memory.clear(coin)

        # 5. EXECUTION 🚀
        sz = self.assets.round_size(coin, base_sz_unit)
        if sz == 0: return
        
        is_buy = (signal == "BULLISH")
        raw_limit_px = limit_buy_px if is_buy else limit_sell_px
        final_limit_px = self.assets.get_price_precision(coin, raw_limit_px)

        logger.info(f"🚀 Executing {coin}: {sz} @ {final_limit_px}")
        send_telegram_message(f"⏳ ATTEMPTING {signal} {coin}: {sz} @ {final_limit_px}")

        try:
            res = self.exchange.order(coin, is_buy, sz, final_limit_px, {"limit": {"tif": "Ioc"}})
            logger.info(f"📬 API RESPONSE: {res}") 

            if res['status'] == 'ok':
                filled_val = res['response']['data']['statuses'][0].get('filled', {}).get('totalSz', 0)
                
                if float(filled_val) > 0:
                    send_telegram_message(f"✅ SUCCESS: Filled {coin} instantly!")
                    self.memory.set(coin, 'tp1_hit', False)
                    self.memory.set(coin, 'entry_time', int(time.time()))
                    time.sleep(3)
                    self.sync_unified_orders(coin, current_atr_pct, portfolio)
                else:
                    logger.warning(f"⚠️ Entry Failed.")
                    send_telegram_message(f"⚠️ Entry Failed.")

            else:
                error_msg = res.get('response', {}).get('data', 'Unknown Error')
                logger.error(f"❌ ORDER FAILED: {error_msg}")
                send_telegram_message(f"❌ ORDER REJECTED: {error_msg}")

        except Exception as e:
            logger.error(f"💥 CRASH DURING ORDER: {e}")
            send_telegram_message(f"💥 BOT CRASHED: {e}")

    def sync_unified_orders(self, coin, current_atr_pct, portfolio):

        pos_details = portfolio.get(coin)
        if not pos_details:
            send_telegram_message(f"❌ {coin}: Position closed or failed to fill.")
            return

        sz_raw = float(pos_details['szi'])
        total_sz = abs(sz_raw)
        avg_entry = float(pos_details['entryPx'])
        is_buy_pos = sz_raw > 0
        
        # 2. Cancel EVERYTHING (Clear the board)
        try: self.cancel_all_orders(coin)
        except: pass
        time.sleep(5)
        
        # 3. Place NEW Unified TP/SL
        logger.info(f"🛡️ REFRESHING TP/SL: Total {total_sz} @ {avg_entry} | ATR: {current_atr_pct:.2%}")
        
        # A. Unified Take Profits (Dynamic ATR Multipliers)
        # Format: (Size Percentage, ATR Multiplier)
        tp_levels = [(1, 1.5)] 
        d = 1 if is_buy_pos else -1
        
        for (pct, atr_mult) in tp_levels:
            # Calculate the required price move based on the current volatility
            move = current_atr_pct * atr_mult
            tp_px = self.assets.get_price_precision(coin, avg_entry * (1 + (move * d)))
            tp_sz = self.assets.round_size(coin, total_sz * pct)
            
            if tp_sz <= 0: continue # Failsafe against zero-size API rejections

            self.exchange.order(
                coin, not is_buy_pos, tp_sz, tp_px, 
                {"trigger": {"isMarket": True, "triggerPx": tp_px, "tpsl": "tp"}},
                reduce_only=True
            )
            send_telegram_message(f"🎯 TP ({atr_mult}x ATR): {tp_sz} @ {tp_px}")

        # B. Unified Stop Loss (Dynamic Volatility Shield)
        sl_mult = 2.0
        hard_sl_pct = current_atr_pct * sl_mult
        sl_px = self.assets.get_price_precision(coin, avg_entry * (1 - (hard_sl_pct * d)))
        
        self.exchange.order(
            coin, not is_buy_pos, total_sz, sl_px, 
            {"trigger": {"isMarket": True, "triggerPx": sl_px, "tpsl": "sl"}},
            reduce_only=True
        )
        send_telegram_message(f"🛑 SL ({sl_mult}x ATR): {total_sz} @ {sl_px}")

    def sync_break_even(self, coin, current_atr_pct, portfolio, open_orders):        
        # =========================================================
        # 💀 CASE A: POST-MORTEM (Position is Gone)
        # =========================================================
        if coin not in portfolio:
            if self.memory.get(coin, 'count') > 0: 
                logger.info(f"🔔 DETECTED EXIT: {coin} position is gone.")
                send_telegram_message(f"🔔 NOTIFICATION: {coin} Position Closed (Hard SL, TP, or Manual).")
                self.memory.clear(coin)
            return

        pos_details = portfolio[coin]
        current_entry = float(pos_details['entryPx'])
        stored_entry = self.memory.get(coin, 'last_known_entry', 0)
        
        if stored_entry != 0 and abs(current_entry - stored_entry) / current_entry > 0.005: # 0.5% diff
            logger.info(f"🆕 NEW TRADE DETECTED! Entry changed {stored_entry} -> {current_entry}")
            logger.info("🧹 Wiping stale memory (resetting TP1/SL flags).")
            
            # Reset everything for the fresh start
            self.memory.clear(coin)
            self.memory.set(coin, 'last_known_entry', current_entry)
                
        # If this is the first time we see it, just save it
        if stored_entry == 0:
             self.memory.set(coin, 'last_known_entry', current_entry)

        # =========================================================
        # 🟢 CASE B: LIVE MONITORING (Position Exists)
        # =========================================================
        entry_px = current_entry
        pos_size = float(pos_details['szi'])
        curr_px = float(self.info.all_mids()[coin])
        is_long = pos_size > 0

        # ---------------------------------------------------------
        # 🛡️ SAFETY NET: THE "ONE LINE" FIX
        # ---------------------------------------------------------
        existing_sl = next((
            o for o in open_orders 
            if o['coin'] == coin 
            and o['isTrigger'] == True
            and "stop" in o['orderType'].lower()
        ), None)

        # IF NAKED -> RESET EVERYTHING
        if not existing_sl:
            logger.warning(f"😱 NAKED POSITION: {coin} missing SL. Resetting Orders!")
            self.sync_unified_orders(coin, current_atr_pct, portfolio)
            return

        # 1. ⚠️ SOFT STOP WATCHDOG (Notify Only)
        soft_dist = 0.01 
        soft_limit = entry_px * (1 - soft_dist) if is_long else entry_px * (1 + soft_dist)
        soft_hit = (curr_px < soft_limit) if is_long else (curr_px > soft_limit)
        
        if soft_hit:
            if not self.memory.get(coin, 'soft_warned'):
                logger.warning(f"⚠️ SOFT STOP: {coin} breached 1%!")
                send_telegram_message(f"⚠️ SOFT STOP ALERT: {coin} is down > 1% @ {curr_px}. Hard Stop is at 1.2%.")
                self.memory.set(coin, 'soft_warned', True)

        # 2. ✅ TP1 BREAK EVEN CHECK (Robust "Order-Gone" Version)
        if not self.memory.get(coin, 'tp1_hit'):
            
            # A. Calculate where TP1 *should* be to identify the order
            tp_dist = 0.015
            target_px = entry_px * (1 + tp_dist) if is_long else entry_px * (1 - tp_dist)
            
            # B. Look for the TP1 Limit Order
            tp1_order_active = False
            required_side = 'A' if is_long else 'B'
            
            for o in open_orders:
                # 1. Basic coin and side check
                if o['coin'] == coin and o['side'] == required_side:
                    
                    # 2. Extract price (TP Market uses 'triggerPx', Limit uses 'limitPx')
                    raw_px = o.get('triggerPx') or o.get('limitPx')
                    if not raw_px: continue
                    
                    order_px = float(raw_px)
                    
                    # 3. Check if this order is our TP1 (within 0.2% tolerance)
                    if abs(order_px - target_px) / target_px < 0.002:
                        tp1_order_active = True
                        logger.info(f"🎯 Found active TP1 order at {order_px}")
                        break
            
            # D. If position exists but TP1 order is GONE, it means TP hit
            if not tp1_order_active:
                logger.info(f"🚀 TP1 DETECTED: Limit order at {target_px} is gone. Moving SL to Breakeven.")

                # A. Find Existing SL (Avoid Duplicates)
                existing_sl = next((
                    o for o in open_orders 
                    if o['coin'] == coin and o['isTrigger'] and "stop" in o['orderType'].lower()
                ), None)

                # If SL is ALREADY at Breakeven (Tolerance 0.2%)
                if existing_sl:
                    sl_trigger = float(existing_sl['triggerPx'])
                    if abs(sl_trigger - entry_px) / entry_px < 0.002:
                        logger.info("✅ SL already at Breakeven. Syncing memory.")
                        self.memory.set(coin, 'tp1_hit', True)
                        return

                # B. Execute Move
                safe_entry_px = self.assets.get_price_precision(coin, entry_px)
                
                logger.info(f"🛡️ MOVING SL TO: {safe_entry_px}")
                send_telegram_message(f"✅ TP1 Hit (Order Filled)! Securing {coin} at {safe_entry_px}")

                try:
                    if existing_sl:
                        self.exchange.cancel(coin, existing_sl['oid'])
                        logger.info('Old SL Cancelled.')
                        # Reduced sleep to 1s; 5s is quite long for high-speed markets
                        time.sleep(1)

                    res = self.exchange.order(
                        coin, 
                        not is_long, 
                        abs(pos_size), 
                        safe_entry_px, 
                        {"trigger": {"isMarket": True, "triggerPx": safe_entry_px, "tpsl": "sl"}},
                        reduce_only=True
                    )
                    
                    logger.info(f"📬 SL RESPONSE: {res}")

                    if res['status'] == 'ok':
                        logger.info("✅ SL Move Confirmed by Exchange.")
                        self.memory.set(coin, 'tp1_hit', True)
                    else:
                        err_msg = res.get('response', {}).get('data', 'Unknown Error')
                        logger.error(f"❌ SL MOVE REJECTED: {err_msg}")
                        send_telegram_message(f"⚠️ SL Failed: {err_msg}")

                except Exception as e:
                    logger.error(f"💥 CRASH MOVING SL: {e}")
            else:
                pass
                
    def close_active_position(self, active_coin, all_mids, temp_assets, portfolio, aws_bucket):
        """
        Safely closes an open position and clears its state.
        Returns True if successful, False if it failed.
        """
        raw_price = float(all_mids[active_coin])
        curr_px = temp_assets.get_price_precision(active_coin, raw_price)

        pos_data = portfolio.get(active_coin)
        if not pos_data:
            logger.error(f"❌ KILL SWITCH FAILED: No position found for {active_coin}!")
            return False

        pos_size = float(pos_data['szi'])
        is_buy = pos_size < 0  # If Short (-), we Buy to close. If Long (+), we Sell to close.
        slippage = 0.002
        
        raw_limit_px = curr_px * (1 + slippage) if is_buy else curr_px * (1 - slippage)
        limit_px = temp_assets.get_price_precision(active_coin, raw_limit_px)

        try:
            res = self.exchange.order(
                active_coin, 
                is_buy, 
                abs(pos_size), 
                limit_px,
                {"limit": {"tif": "Ioc"}},
                reduce_only=True
            )
            
            if res['status'] != 'ok':
                error_msg = res.get('response', {}).get('data', 'Unknown Error')
                logger.error(f"❌ CLOSE FAILED. Aborting switch. Error: {error_msg}")
                send_telegram_message(f"❌ SWITCH ABORTED: Could not close {active_coin}. Error: {error_msg}")
                return False
                
            logger.info(f"✅ CLOSING current {active_coin} position.")
            StateManager(aws_bucket).clear(active_coin)
            return True

        except Exception as e:
            logger.error(f"💥 CRASH DURING SWITCH: {e}")
            return False

