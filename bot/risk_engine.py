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

    def get_unified_equity(self):
        """
        Calculates true net equity for Unified Accounts.
        Sums Spot USDC balance + Perps Unrealized PnL.
        """
        try:
            # 1. Get Spot Balances
            spot_state = self.info.spot_user_state(self.address)
            spot_usdc = sum(float(b['total']) for b in spot_state.get('balances', []) if b['coin'] == 'USDC')
            
            # 2. Get Perps State (for Margin and UnPnL)
            user_state = self.info.user_state(self.address)
            summary = user_state.get('crossMarginSummary') or user_state.get('marginSummary', {})
            used_margin = float(summary.get('totalMarginUsed', 0))
            
            # 3. Aggregate Unrealized PnL
            unrealized_pnl = 0
            for p in user_state.get('assetPositions', []):
                unrealized_pnl += float(p['position'].get('unrealizedPnl', 0))
            
            total_equity = spot_usdc + unrealized_pnl
            return total_equity, used_margin, user_state
        except Exception as e:
            logger.error(f"❌ Failed to fetch unified equity: {e}")
            return 0, 0, {}

    def check_safety(self):
        """Checks for bankruptcy and margin limits using Unified Equity."""
        try:
            val, used, _ = self.get_unified_equity()
            
            # 1. Check for Bankruptcy
            if val <= 0: 
                logger.error("🚨 CRITICAL: Unified Account Value is 0 or negative! Stopping.")
                return False
            
            # 2. Check Margin Usage
            curr_usage = used / val
            if curr_usage > MARGIN_LIMIT: 
                logger.warning(f"⚠️ HIGH RISK: Unified Margin Usage {curr_usage:.2%}. Limit: {MARGIN_LIMIT:.2%}")
                send_telegram_message(f"⚠️ High Margin Alert: {curr_usage:.2%}. Pausing new entries.")
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
            logger.warning(f"⚠️ POC WARNING: {coin} Long at ${current_price:.4f} is above the POC (${poc_price:.4f}). Ignoring filter for A/B testing.")
            # return False

        if not is_buy and current_price < poc_price:
            logger.warning(f"⚠️ POC WARNING: {coin} Short at ${current_price:.4f} is below the POC (${poc_price:.4f}). Ignoring filter for A/B testing.")
            # return False        

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

    def execute_logic(self, coin, signal, timeframe, current_atr_pct, portfolio, user_state, open_orders, override_usd=None):

        orders = [o for o in open_orders if o['coin'] == coin]
        
        # 1. MATH & STATE CALCULATION 🧮
        equity, _, _ = self.get_unified_equity()
        
        mid_px = float(self.info.all_mids()[coin])
        if mid_px == 0: return
        
        slippage = 0.015
                
        raw_limit_buy = mid_px * (1 + slippage)
        raw_limit_sell = mid_px * (1 - slippage)
        
        # Pre-calculate precise prices
        limit_buy_px = self.assets.get_price_precision(coin, raw_limit_buy)
        limit_sell_px = self.assets.get_price_precision(coin, raw_limit_sell)
        
        if override_usd:
            entry_usd = override_usd
            logger.info(f"📊 Using RISK PARITY sizing: ${entry_usd:.2f} for {coin}")
        else:
            entry_usd = (equity * 0.15) # 15% of total equity per leg
            
        base_sz_unit = entry_usd / mid_px

        # 2. POSITION STATE
        has_position = coin in portfolio
        pos_details = portfolio.get(coin)

        if pos_details:
            pos_size = float(pos_details.get('szi', 0.0))
            avg_entry_px = float(pos_details.get('entryPx', 0.0))
            
            # P3-1: Use actual entry price to reconstruct the original layer size
            # Prevents DCA inflation if the token pumped significantly since entry
            original_base_unit = entry_usd / avg_entry_px if avg_entry_px > 0 else base_sz_unit
            current_count = int(round(abs(pos_size) / original_base_unit)) if original_base_unit > 0 else 1
        else:
            pos_size = 0.0
            avg_entry_px = 0.0
            current_count = 0
            
            # P2-8: Circuit Breaker Check (Only block NEW entries, not existing position management)
            consecutive_losses = self.memory.get('GLOBAL', 'consecutive_losses', default=0)
            if consecutive_losses >= 3:
                logger.warning(f"🛑 CIRCUIT BREAKER TRIPPED: {consecutive_losses} consecutive losses. Pausing {coin} entry.")
                return

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

        # 3. PENDING ORDER CHECKS
        if orders:
            has_limit = any(o['orderType'] == 'Limit' for o in orders)            
            if has_limit:
                logger.info(f"⏳ {coin}: Pending Limit Order. Waiting...")
                return
                
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
                # MIN_HOLD_SECONDS is already defined above from HOLD_MAP (L-2 Fix)
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
                filled_float = float(filled_val)
                
                if filled_float > 0:
                    if filled_float < sz:
                        send_telegram_message(f"⚠️ PARTIAL FILL: {coin} intended {sz}, got {filled_float}!")
                        logger.warning(f"⚠️ PARTIAL FILL: {coin} intended {sz}, got {filled_float}!")
                    else:
                        send_telegram_message(f"✅ SUCCESS: Filled {coin} instantly!")
                    self.memory.set(coin, 'tp1_hit', False)
                    self.memory.set(coin, 'entry_time', int(time.time()))
                    time.sleep(3)
                    self.sync_unified_orders(coin, current_atr_pct, portfolio)
                else:
                    logger.warning(f"⚠️ Entry Failed (0 filled).")
                    send_telegram_message(f"⚠️ Entry Failed (0 filled).")

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
        
        # P3-2: Failsafe against Dust - If total size is unmanageable, abort completely.
        if self.assets.round_size(coin, total_sz) <= 0:
            logger.warning(f"⚠️ {coin} position size {total_sz} is dust. Skipping Unified Orders.")
            return
            
        avg_entry = float(pos_details['entryPx'])
        is_buy_pos = sz_raw > 0
        
        # 2. Cancel EVERYTHING (Clear the board)
        try: self.cancel_all_orders(coin)
        except: pass
        time.sleep(1) # L-4 Fix: Reduce naked exposure
        
        # 3. Place NEW Unified SL (Trailing Initial)
        logger.info(f"🛡️ PLACING INITIAL SL: Total {total_sz} @ {avg_entry} | ATR: {current_atr_pct:.2%}")
        
        sl_mult = 1.0
        hard_sl_pct = current_atr_pct * sl_mult
        d = 1 if is_buy_pos else -1
        sl_px = self.assets.get_price_precision(coin, avg_entry * (1 - (hard_sl_pct * d)))
        
        self.exchange.order(
            coin, not is_buy_pos, total_sz, sl_px, 
            {"trigger": {"isMarket": True, "triggerPx": sl_px, "tpsl": "sl"}},
            reduce_only=True
        )
        send_telegram_message(f"🛑 INITIAL SL ({sl_mult}x ATR): {total_sz} @ {sl_px}")

    def sync_trailing_stop(self, coin, current_atr_pct, portfolio, open_orders):        
        # =========================================================
        # 💀 CASE A: POST-MORTEM (Position is Gone)
        # =========================================================
        if coin not in portfolio:
            if self.memory.get(coin, 'entry_time', default=0) > 0: 
                streak = self.memory.get('GLOBAL', 'consecutive_losses', default=0)
                # Since we don't have a fixed TP anymore, we assume it's a loss if it hit SL,
                # UNLESS the exit price was higher than entry. For now, we assume loss to be safe.
                streak += 1
                self.memory.set('GLOBAL', 'consecutive_losses', streak)
                logger.info(f"📉 {coin} position closed. Consecutive losses metric: {streak}")
                
                logger.info(f"🔔 DETECTED EXIT: {coin} position is gone.")
                send_telegram_message(f"🔔 NOTIFICATION: {coin} Position Closed. Current Streak Metric: {streak}")
                self.memory.clear(coin)
            return

        pos_details = portfolio[coin]
        current_entry = float(pos_details['entryPx'])
        stored_entry = self.memory.get(coin, 'last_known_entry', 0)
        
        if stored_entry != 0 and abs(current_entry - stored_entry) / current_entry > 0.005: # 0.5% diff
            logger.info(f"🆕 NEW TRADE DETECTED! Entry changed {stored_entry} -> {current_entry}")
            logger.info("🧹 Wiping stale memory (resetting SL).")
            self.memory.clear(coin)
            self.memory.set(coin, 'last_known_entry', current_entry)
                
        if stored_entry == 0:
             self.memory.set(coin, 'last_known_entry', current_entry)

        # =========================================================
        # 🟢 CASE B: LIVE MONITORING (Position Exists)
        # =========================================================
        entry_px = current_entry
        pos_size = float(pos_details['szi'])
        curr_px = float(self.info.all_mids()[coin])
        is_long = pos_size > 0

        # Find existing SL
        existing_sl = next((
            o for o in open_orders 
            if o['coin'] == coin 
            and o['isTrigger'] == True
            and "stop" in str(o.get('orderType', '')).lower()
        ), None)

        if not existing_sl:
            logger.warning(f"😱 NAKED POSITION: {coin} missing SL. Resetting Orders!")
            self.sync_unified_orders(coin, current_atr_pct, portfolio)
            return

        # 1. ⚠️ SOFT STOP WATCHDOG
        soft_dist = 0.01 
        soft_limit = entry_px * (1 - soft_dist) if is_long else entry_px * (1 + soft_dist)
        soft_hit = (curr_px < soft_limit) if is_long else (curr_px > soft_limit)
        
        if soft_hit:
            if not self.memory.get(coin, 'soft_warned'):
                logger.warning(f"⚠️ SOFT STOP: {coin} breached 1%!")
                send_telegram_message(f"⚠️ SOFT STOP ALERT: {coin} is down > 1% @ {curr_px}.")
                self.memory.set(coin, 'soft_warned', True)

        # 2. 📈 RATCHETING TRAILING STOP
        trail_dist = current_atr_pct * 1.0
        d = 1 if is_long else -1
        
        # Calculate new potential SL
        new_sl_px = curr_px * (1 - (trail_dist * d))
        current_sl_px = float(existing_sl['triggerPx'])
        
        # Check if the new SL is better than the current SL by at least 0.2% (to prevent API spam)
        sl_improved = (new_sl_px > current_sl_px * 1.002) if is_long else (new_sl_px < current_sl_px * 0.998)
        
        if sl_improved:
            safe_new_sl = self.assets.get_price_precision(coin, new_sl_px)
            logger.info(f"📈 TRAILING STOP TRIGGERED: Moving {coin} SL from {current_sl_px} to {safe_new_sl}")
            
            try:
                self.exchange.cancel(coin, existing_sl['oid'])
                time.sleep(1)

                res = self.exchange.order(
                    coin, 
                    not is_long, 
                    abs(pos_size), 
                    safe_new_sl, 
                    {"trigger": {"isMarket": True, "triggerPx": safe_new_sl, "tpsl": "sl"}},
                    reduce_only=True
                )
                
                if res['status'] == 'ok':
                    logger.info("✅ Trailing SL Move Confirmed.")
                    send_telegram_message(f"📈 TRAILING STOP: {coin} SL raised to {safe_new_sl} (Current Price: {curr_px})")
                else:
                    err_msg = res.get('response', {}).get('data', 'Unknown Error')
                    logger.error(f"❌ SL MOVE REJECTED: {err_msg}")
            except Exception as e:
                logger.error(f"💥 CRASH MOVING TRAILING SL: {e}")
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
        slippage = 0.015
        
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
                
            filled_val = float(res['response']['data']['statuses'][0].get('filled', {}).get('totalSz', 0))
            if filled_val == 0:
                logger.error(f"❌ CLOSE FAILED. 0 filled for {active_coin}.")
                send_telegram_message(f"❌ SWITCH ABORTED: 0 filled on close for {active_coin}.")
                return False
            elif filled_val < abs(pos_size):
                logger.warning(f"⚠️ PARTIAL CLOSE: {active_coin} intended {abs(pos_size)}, got {filled_val}.")
                send_telegram_message(f"⚠️ PARTIAL CLOSE: {active_coin} intended {abs(pos_size)}, got {filled_val}.")
                return False  # Still not fully closed, so we return False
                
            # P2-8: Calculate PnL for circuit breaker
            entry_px = float(pos_data['entryPx'])
            if pos_size > 0:
                pnl_pct = (curr_px - entry_px) / entry_px
            else:
                pnl_pct = (entry_px - curr_px) / entry_px

            streak = self.memory.get('GLOBAL', 'consecutive_losses', default=0)
            if pnl_pct < 0:
                streak += 1
                self.memory.set('GLOBAL', 'consecutive_losses', streak)
                logger.info(f"📉 {active_coin} closed at a loss ({pnl_pct:.2%}). Consecutive losses: {streak}")
            else:
                self.memory.set('GLOBAL', 'consecutive_losses', 0)
                logger.info(f"📈 {active_coin} closed in profit/BE ({pnl_pct:.2%}). Streak reset.")
                
            logger.info(f"✅ CLOSING current {active_coin} position.")
            self.memory.clear(active_coin)
            return True

        except Exception as e:
            logger.error(f"💥 CRASH DURING SWITCH: {e}")
            return False

