import sys
import os
import json
import itertools
import math
import logging
import pandas as pd
import numpy as np
from hyperliquid.info import Info

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import AWS_BUCKET, BASE_URL, TESTNET_MODE, CONFIG_FILE, LEADERBOARD_FILE
from bot.utils import S3Interface, send_telegram_message
from bot.strategies import STRATEGY_CONFIG
from backtester.engine import HyperBacktester, inject_htf_trend

# Note: AI scoring is now handled by the Scout phase, not the grid search.

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def strategist_handler(event, context, ai_probs=None, ai_accuracy=0.0):
    """Global grid search across all DB symbols. Scores by Pure Math only."""
    logger.info("🧠 STRATEGIST: Starting Global Alpha Factory Grid Search...")
    
    # 1. Initialize S3
    s3 = S3Interface(AWS_BUCKET)
    
    info_client = Info(BASE_URL, skip_ws=True)
    
    # 🌟 PIPELINE A: THE GLOBAL MARKET HUB
    logger.info("📡 Fetching Global Market Universe from Database...")
    import sqlite3
    from data_pipeline.database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv")
    TICKERS = [r[0] for r in cursor.fetchall()]
    conn.close()
    
    if not TICKERS:
        TICKERS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"] # Minimum fallback
    
    logger.info(f"🧬 ALPHA FACTORY: Researching {len(TICKERS)} symbols for the Elite Squad.")

    # NOTE: AI probabilities are accepted for legacy compatibility but NOT used in scoring.
    # The grid search uses Pure Math only. AI fusion happens in the Scout phase.
    if ai_probs:
        logger.info(f"📊 AI probs received ({len(ai_probs)} symbols) — will be used by Scout, not here.")
    
    # UPGRADE 1: Adjusted Timeframes (Agile Focus)
    TIMEFRAMES = ["15m", "1h", "4h"]

    MIN_TPD_TARGETS = {
    '1h': 0.33,  # ~1 trade every 3 days
    '15m': 1.5,  # ~1.5 trades per day
    '5m': 4.0    # ~4 trades per day
    }

    MIN_PROFIT_FACTOR = 1.2
    
    results = []
    
    for ticker in TICKERS:
        for tf in TIMEFRAMES:
            if tf == "5m":
                lookback = 10000 
            elif tf == "15m": 
                lookback = 8000
            elif tf == "1h": 
                lookback = 5000
            else: 
                lookback = 8000  # Fallback

            try:
                # Fetch data ONCE per ticker/tf
                engine = HyperBacktester(info_client, ticker, interval=tf, candles_lookback=lookback)
                
                # Safety check for empty data
                if engine.data.empty: continue
                
                if tf in ["5m", "15m"]:
                    htf_engine = HyperBacktester(info_client, ticker, interval="1h", candles_lookback=5000)
                    
                    if not htf_engine.data.empty:
                        engine.data = inject_htf_trend(engine.data, htf_engine.data)
                else:
                    htf_engine = HyperBacktester(info_client, ticker, interval="4h", candles_lookback=5000)
                    if not htf_engine.data.empty:
                        engine.data = inject_htf_trend(engine.data, htf_engine.data)

                # TPD Calculation
                total_days = (engine.data.index[-1] - engine.data.index[0]).total_seconds() / 86400
                total_days = max(1, total_days)
                target_tpd = MIN_TPD_TARGETS.get(tf, 0.5)
                dynamic_min_trades = int(target_tpd * total_days)

                # Iterate through Strategy Config
                for strat_name, cfg in STRATEGY_CONFIG.items():
                    keys, values = zip(*cfg['params'].items())
                    for v in itertools.product(*values):
                        params = dict(zip(keys, v))
                        strat_instance = cfg['class'](**params)
                        res = engine.run(strat_instance)
                        
                        raw_return = res['return']
                        recent_return = res.get('recent_return', 0)
                        alpha = raw_return - res['buy_hold']
                        
                        # 🧬 THE UPGRADE: Extract new metrics
                        trades = res.get('trades', 0)
                        sharpe = res.get('sharpe', 0)
                        profit_factor = res.get('profit_factor',0)

                        if raw_return > 0:
                            if trades < dynamic_min_trades:
                                logger.info(f"🗑️ REJECTED: {ticker} {tf} | {strat_name} -> Not enough trades ({trades}/{dynamic_min_trades})")
                                score = -999 # Disqualify
                            
                            elif profit_factor < MIN_PROFIT_FACTOR:
                                logger.info(f"📉 REJECTED: {ticker} {tf} | {strat_name} -> Low Profit Factor ({profit_factor:.2f})")
                                score = -999 # Disqualify
                            
                            elif recent_return < 0:
                                logger.info(f"💀 KILL SWITCH: {ticker} {tf} | {strat_name} -> BLOCKED (Bad 24h)")
                                score = -999 # Disqualify
                                
                            else:
                                # ⚖️ THE NEW JUDGE: Institutional 'Pure Math' Score
                                # This score reflects the 3-year historical stability of the strategy.
                                # It is agnostic of current AI probabilities (which are handled by the Scout).
                                if trades <= 1:
                                    score = 0
                                else:
                                    # Pure Math Score = (Return * ProfitFactor * Sharpe) * Frequency_Factor
                                    score = (raw_return * profit_factor * sharpe) * math.log10(trades)
                                
                                logger.info(
                                    f"✅ PASSED: {ticker} {tf} | {strat_name} | "
                                    f"Trades: {trades} | PF: {profit_factor:.2f} | "
                                    f"Score: {score:.4f}"
                                )
                        else:
                            score = -999

                        if score != -999:
                            results.append({
                                "target_coin": ticker,
                                "timeframe": tf,
                                "strategy": strat_name,
                                "params": params,
                                "alpha": alpha,
                                "raw_return": raw_return,
                                "trades": trades,
                                "sharpe": sharpe,
                                "profit_factor": profit_factor,
                                "score": score
                            })

            except Exception as e:
                logger.error(f"Error testing {ticker} {tf}: {e}")

    # Check if we found ANY viable strategies
    if not results:
        msg = "❌ MARKET WATCH: No strategies met the profitability floor. Bot going to SLEEP."
        logger.warning(msg)
        send_telegram_message(msg)
        
        sleep_config = {
            "target_coin": "SLEEP",
            "timeframe": "1h",
            "strategy": "None",
            "params": {},
            "raw_return": 0,
            "sharpe": -999,
            "score": -999
        }
        s3.save_json(CONFIG_FILE, sleep_config)
        
        return {'statusCode': 200, 'body': json.dumps('Market Crash - Bot Sleeping')}

    # ---------------------------------------------------------
    # SCAN COMPLETE: Save ALL Passing Results + Build Elite Squad
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='score', ascending=False)

    # Helper: clean numpy types for JSON serialization
    def to_json_safe(row_dict):
        return {k: v.item() if isinstance(v, np.generic) else v for k, v in row_dict.items()}

    # 1. Save ALL passing results (for Pipeline B distribution analysis)
    all_results_list = [to_json_safe(row.to_dict()) for _, row in df_res.iterrows()]
    ALL_RESULTS_FILE = "all_grid_results.json"
    with open(ALL_RESULTS_FILE, 'w') as f:
        json.dump(all_results_list, f, indent=2)
    s3.save_json(ALL_RESULTS_FILE, all_results_list)
    logger.info(f"📊 Saved {len(all_results_list)} passing strategies to {ALL_RESULTS_FILE}")

    # 2. Build Top 20 Elite Squad
    SQUAD_SIZE = 20
    top_n = df_res.head(SQUAD_SIZE)
    elite_squad = [to_json_safe(row.to_dict()) for _, row in top_n.iterrows()]

    # 3. HL TRADABILITY GUARANTEE
    # Ensure at least one squad member is tradable on Hyperliquid.
    # If none made it organically, inject the top-scoring HL-tradable candidates.
    from data_pipeline.hyperliquid_sync import get_hyperliquid_universe
    try:
        hl_universe = get_hyperliquid_universe()
    except Exception:
        hl_universe = []

    if hl_universe:
        squad_coins = {r['target_coin'].split('/')[0] for r in elite_squad}
        hl_in_squad = squad_coins & set(hl_universe)

        if not hl_in_squad:
            logger.warning("⚠️ No HL-tradable tokens in Top 20! Injecting best HL candidates...")
            # Find top HL-tradable entries from the full results
            for _, row in df_res.iterrows():
                coin_clean = row['target_coin'].split('/')[0]
                if coin_clean in hl_universe:
                    elite_squad.append(to_json_safe(row.to_dict()))
                    logger.info(f"   ✅ Injected HL-tradable: {row['target_coin']} (Score: {row['score']:.4f})")
                    if len(elite_squad) - SQUAD_SIZE >= 3:  # Inject up to 3 HL candidates
                        break
        else:
            logger.info(f"✅ HL-tradable tokens in squad: {hl_in_squad}")

    best = elite_squad[0]  # Global Champion

    logger.info(f"🏆 GLOBAL CHAMPION: {best['target_coin']} | {best['strategy']} | Score: {best['score']:.4f}")
    send_telegram_message(
        f"🏆 WEEKLY FACTORY COMPLETE\n"
        f"Global Champion: {best['target_coin']} | {best['strategy']}\n"
        f"Elite Squad: {len(elite_squad)} | Total Passing: {len(all_results_list)}"
    )

    # 4. Save Elite Squad (S3 + Local)
    SQUAD_FILE = "elite_squad.json"
    s3.save_json(SQUAD_FILE, elite_squad)
    with open(SQUAD_FILE, 'w') as f:
        json.dump(elite_squad, f, indent=4)

    # Legacy compatibility
    s3.save_json(CONFIG_FILE, best)

    return {'statusCode': 200, 'body': json.dumps(f'Elite Squad ({len(elite_squad)}) + {len(all_results_list)} total results saved')}

if __name__ == '__main__':
    strategist_handler(None, None)

