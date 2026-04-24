#!/usr/bin/env python3
"""
Alpha Factory — Master CLI

Single entry point to run every pipeline step-by-step.
Designed for local execution on your machine.

Usage:
  python master.py ingest   --symbols BTC/USDT,ETH/USDT --timeframe 1h
  python master.py ingest   --top 50 --timeframe 15m
  python master.py backtest
  python master.py report
  python master.py full
  python master.py status
"""

import sys
import os

# Fix Windows console encoding (cp1252 can't handle emoji)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import sqlite3
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def cmd_sync_live(args):
    """Fetch the latest 100 candles for active universe from Binance API (Pure Binance policy)."""
    print(f"\n🔄 Running Live Edge Sync (Binance Only)...")
    import requests
    import time
    
    from data_pipeline.database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    pairs_to_refresh = conn.execute("SELECT DISTINCT symbol, market FROM ohlcv").fetchall()
    conn.close()
    
    if not pairs_to_refresh:
        print("   ⚠️ No data found in DB. Ingest some historical data first.")
        return

    tf_arg = args.timeframe if hasattr(args, 'timeframe') else '1h'
    timeframes = [t.strip() for t in tf_arg.split(',')]
    
    sync_count = 0
    for current_tf in timeframes:
        print(f"   🔄 Syncing {current_tf} edge...")
        for symbol, market in pairs_to_refresh:
            candles = []
            binance_symbol = symbol.replace('/', '')  # e.g. BTC/USDT -> BTCUSDT
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {'symbol': binance_symbol, 'interval': current_tf, 'limit': 100}
                res = requests.get(url, params=params, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    for c in data:
                        candles.append({
                            'timestamp': c[0], 'open': float(c[1]), 'high': float(c[2]),
                            'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
                            'symbol': symbol, 'timeframe': current_tf, 'market': market
                        })
            except Exception as e:
                print(f"   ❌ Binance API failed for {symbol}: {e}")

            # Bulk upsert into DB
            if candles:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv (timestamp, symbol, timeframe, market, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(c['timestamp'], c['symbol'], c['timeframe'], c['market'], c['open'], c['high'], c['low'], c['close'], c['volume']) for c in candles])
                conn.commit()
                conn.close()
                sync_count += 1

            time.sleep(0.1)  # Rate limit courtesy
            
    print(f"   ✅ Refreshed {sync_count} symbols with live Binance edge data.")


def cmd_ingest(args):
    """Step 1: Download historical data into alpha_factory.db"""
    from data_pipeline.database import init_db
    from data_pipeline.sync_manager import SyncManager
    
    init_db()
    sync = SyncManager()
    
    market = 'spot' if args.spot else 'futures'
    timeframes = [t.strip() for t in args.timeframe.split(',')]
    
    if args.top:
        # Auto-discover top symbols by volume from Hyperliquid
        from data_pipeline.hyperliquid_sync import get_hl_top_by_volume
        print(f"\n🔍 Discovering top {args.top} HL symbols by volume...")
        hl_symbols = get_hl_top_by_volume(limit=args.top)
        # Map HL symbol to Binance symbol format
        symbols = [f"{s}/USDT" for s in hl_symbols]
        print(f"   Found: {symbols[:5]}... ({len(symbols)} total)")
    else:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"  INGESTING: {len(symbols)} symbols | {tf} | {market}")
        print(f"{'='*60}")
        sync.bulk_sync(symbols, timeframe=tf, market=market, 
                       target_years=args.years, start_year=args.start_year)
    
    sync.close()
    print("\n✅ Data ingestion complete!")


def cmd_backtest(args):
    """Step 2: Run Pipeline A — Extreme Grid Search + AI Scoring"""
    # Auto-sync live edge before backtesting
    cmd_sync_live(args)
    
    from analytics.analytics import train_global_xgboost, get_latest_probabilities
    from backtester.build_bot_blueprint import strategist_handler
    
    print("\n🧠 Training XGBoost model...")
    model, features, accuracy = train_global_xgboost(market=args.market, tune_hyperparams=args.tune, force_train=args.force_train)
    
    ai_probs = {}
    if model:
        print(f"   Model accuracy: {accuracy:.2%}")
        ai_probs = get_latest_probabilities(model, features, market=args.market)
        print(f"   Scored {len(ai_probs)} symbols.")
    else:
        print("   ⚠️ No data in database. Run 'ingest' first.")
        return
    
    print("\n🚀 Running Pipeline A (Grid Search)...")
    result = strategist_handler(None, None, ai_probs=ai_probs, ai_accuracy=accuracy)
    print(f"\n✅ Pipeline A complete: {result}")


def cmd_report(args):
    """Step 3: Run Pipeline B — Weekly Intelligence Report"""
    # Auto-sync live edge before report
    cmd_sync_live(args)
    
    from analytics.analytics import train_global_xgboost, get_latest_probabilities
    from analytics.generate_report import generate_report
    
    print("\n🧠 Training XGBoost model...")
    model, features, accuracy = train_global_xgboost(market=args.market, tune_hyperparams=args.tune, force_train=args.force_train)
    
    ai_probs = {}
    if model:
        print(f"   Model accuracy: {accuracy:.2%}")
        ai_probs = get_latest_probabilities(model, features, market=args.market)
    else:
        print("   ⚠️ No data in database. Run 'ingest' first.")
        return
    
    print("\n📊 Generating Weekly Intelligence Report...")
    report_path = generate_report(ai_probs=ai_probs, ai_accuracy=accuracy)
    
    if report_path:
        print(f"\n✅ Report saved to: {report_path}")


def cmd_full(args):
    """Run the full weekly cycle: Ingest (Optional) → Train once → Pipeline A → Pipeline B"""
    from analytics.weekly_orchestrator import run_weekly_cycle
    from analytics.generate_report import generate_report
    
    # 0. Check if we need to Ingest first
    if args.top > 0 or args.symbols != 'BTC/USDT,ETH/USDT,SOL/USDT':
        print("\n" + "="*60)
        print("  📥 STEP 0: BOOTSTRAP INGESTION")
        print("="*60)
        # Map market to spot flag for cmd_ingest
        args.spot = (args.market == 'spot')
        cmd_ingest(args)
    
    # 1. 🔄 SMART SYNC: Freshness Mode (The Hyperliquid Symmetry Mapping)
    # This catches the "Live Edge" so we trade on the current minute.
    cmd_sync_live(args)
    
    print("\n" + "="*60)
    print("  🚀 STARTING FULL WEEKLY CYCLE")
    if args.tune or args.force_train:
        print("     🧠 MODE: ELITE RE-TRAINING (High Fidelity Intelligence)")
    print("="*60)
    
    # 1. Orchestrate AI Training & Strategy Discovery (Pipeline A)
    result = run_weekly_cycle(market=args.market, tune_hyperparams=args.tune, force_train=args.force_train)
    
    probs = result['probs']
    accuracy = result['accuracy']
    
    print(f"\n✅ AI Intelligence Cycle Complete!")
    print(f"   Symbols scored: {len(probs)}")
    print(f"   Model accuracy: {accuracy:.2%}")
    
    # 2. Trigger Intelligence Report (Pipeline B)
    print("\n📊 Generating Master Intelligence Report...")
    report_path = generate_report(ai_probs=probs, ai_accuracy=accuracy)
    
def cmd_status(args):
    """Show what data you have in the local database"""
    from data_pipeline.database import DB_PATH
    db_path = DB_PATH
    
    if not os.path.exists(db_path):
        print(f"❌ No database found at {db_path}. Run 'python master.py ingest' first.")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Count total rows
    total = conn.execute("SELECT COUNT(*) as cnt FROM ohlcv").fetchone()['cnt']
    
    if total == 0:
        print("📭 Database is empty. Run 'python master.py ingest' first.")
        conn.close()
        return
    
    print(f"\n{'='*60}")
    print(f"  DATABASE STATUS: alpha_factory.db")
    print(f"{'='*60}")
    print(f"  Total candles: {total:,}")
    
    # Per-symbol breakdown
    rows = conn.execute("""
        SELECT symbol, timeframe, market, 
               COUNT(*) as candles,
               MIN(timestamp) as earliest,
               MAX(timestamp) as latest
        FROM ohlcv 
        GROUP BY symbol, timeframe, market
        ORDER BY candles DESC
    """).fetchall()
    
    print(f"\n  {'Symbol':<15} {'TF':<6} {'Market':<8} {'Candles':>10}   {'From':<12} {'To':<12}")
    print(f"  {'─'*15} {'─'*6} {'─'*8} {'─'*10}   {'─'*12} {'─'*12}")
    
    for r in rows:
        try:
            start = datetime.fromtimestamp(r['earliest']/1000).strftime('%Y-%m-%d')
        except (OSError, ValueError):
            start = "BAD_TS"
        try:
            end = datetime.fromtimestamp(r['latest']/1000).strftime('%Y-%m-%d')
        except (OSError, ValueError):
            end = "BAD_TS"
        print(f"  {r['symbol']:<15} {r['timeframe']:<6} {r['market']:<8} {r['candles']:>10,}   {start:<12} {end:<12}")
    
    # Check sync state
    sync_rows = conn.execute("SELECT COUNT(*) as cnt FROM sync_state").fetchone()['cnt']
    print(f"\n  Tracked sync states: {sync_rows}")
    
    # DB file size
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Database file size: {size_mb:.1f} MB")
    
    conn.close()


def cmd_audit(args):
    """Diagnose data integrity: gaps, spikes, and health scores."""
    from data_pipeline.data_auditor import DataAuditor
    auditor = DataAuditor()
    
    print(f"\n{'='*60}")
    print(f"  🔍 DATA INTEGRITY AUDIT")
    print(f"{'='*60}")
    
    partitions = auditor.get_all_partitions()
    # Filter by market if provided
    if args.market:
        partitions = [p for p in partitions if p[2] == args.market]
    # Filter by symbol if provided
    if args.symbols:
        s_list = [s.strip() for s in args.symbols.split(',')]
        partitions = [p for p in partitions if p[0] in s_list]

    if not partitions:
        print("  ⚠️ No matching data partitions found to audit.")
        return

    print(f"  Auditing {len(partitions)} partitions...\n")
    print(f"  {'Symbol':<15} {'TF':<6} {'Grade':<6} {'Gaps':>5} {'Spikes':>6} {'Health':>7}")
    print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*7}")

    grand_total_gaps = 0
    grand_total_anomalies = 0

    for p in partitions:
        res = auditor.audit_pair(*p)
        if res['status'] == 'EMPTY': continue
        
        grade_color = "🟢" if res['health_score'] >= 90 else "🟡" if res['health_score'] >= 70 else "🔴"
        
        print(f"  {res['symbol']:<15} {res['timeframe']:<6} {grade_color} {res['grade']:<3} {res['gaps_found']:>5} {res['anomalies']:>6} {res['health_score']:>6.1f}%")
        
        grand_total_gaps += res['gaps_found']
        grand_total_anomalies += res['anomalies']

    print(f"\n{'─'*60}")
    print(f"  ✅ Audit Complete.")
    print(f"  Gaps Found: {grand_total_gaps} | Anomalies: {grand_total_anomalies}")
    if grand_total_gaps > 0:
        print(f"  TIP: Run 'ingest' or 'sync' to attempt gap-filling for 'Dark Zones'.")


def cmd_scout(args):
    """Hourly Scout Loop: Fuses Static Math with Live AI Conviction."""
    from bot.utils import S3Interface, send_telegram_message
    from bot.config import AWS_BUCKET
    from analytics.analytics import get_latest_probabilities, train_global_xgboost
    from data_pipeline.hyperliquid_sync import get_hyperliquid_universe
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("📡 SCOUT: Pulling Elite Squad from S3...")
    s3 = S3Interface(AWS_BUCKET)
    squad = s3.load_json("elite_squad.json")
    
    if not squad:
        logger.error("❌ SCOUT: No Elite Squad found on S3. Run 'weekly' first.")
        return

    # 1. Filter for Hyperliquid Tradability
    hl_universe = get_hyperliquid_universe()
    executable_squad = []
    for member in squad:
        clean_name = member['target_coin'].split("/")[0]
        if clean_name in hl_universe:
            executable_squad.append(member)
    
    logger.info(f"🎯 SCOUT: Found {len(executable_squad)} tradable candidates on Hyperliquid.")

    # 2. Get Live AI Conviction
    # Load the existing trained model from analytics/models/
    import pickle
    import xgboost as xgb
    
    model_dir = os.path.join(os.path.dirname(__file__), 'analytics', 'models')
    xgb_path = os.path.join(model_dir, 'xgboost_futures.json')
    rf_path = os.path.join(model_dir, 'xgboost_futures.pkl')
    
    # Feature list must match what analytics.py uses for training
    xgb_features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'ema_20', 'ema_50',
                     'ema_200', 'volatility_20', 'z_score_20', 'volume',
                     'timeframe_minutes', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    ensemble = None
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        rf_model = None
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
        ensemble = (xgb_model, rf_model) if rf_model else xgb_model
        logger.info(f"SCOUT: Loaded cached AI model from {xgb_path}")
    else:
        logger.warning("SCOUT: No cached model found. Training a fresh one...")
        ensemble, xgb_features, _ = train_global_xgboost(force_train=False)

    if not ensemble:
        logger.error("❌ SCOUT: AI Brain not found. Fusing with neutral index.")
        ai_probs = {m['target_coin']: {'bull': 0.5, 'bear': 0.5} for m in executable_squad}
    else:
        # Get probabilities for ONLY the squad members to save time
        ai_probs = get_latest_probabilities(ensemble, xgb_features)

    # 3. Perform the Fusion
    fused_results = []
    for member in executable_squad:
        probs = ai_probs.get(member['target_coin'], {'bull': 0.5, 'bear': 0.5})
        conviction = (probs.get('bull', 0) + probs.get('bear', 0))
        
        # FINAL FORMULA: Static Math * Live Conviction
        fused_score = member['score'] * conviction
        
        member['live_conviction'] = conviction
        member['fused_score'] = fused_score
        fused_results.append(member)

    # 4. Rank and Upload Intelligence
    fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
    
    s3.save_json("live_scout_intelligence.json", fused_results)
    
    best = fused_results[0] if fused_results else None
    if best:
        msg = f"🛰️ SCOUT REFRESH\nNew Leader: {best['target_coin']}\nFused Score: {best['fused_score']:.4f}\nAI Conviction: {best['live_conviction']:.2%}"
        logger.info(msg)
        # send_telegram_message(msg)

    logger.info("✅ SCOUT COMPLETE: Live Intelligence updated on S3.")


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Factory — Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master.py status                              Check your database
  python master.py ingest --symbols BTC/USDT,ETH/USDT  Ingest specific coins
  python master.py ingest --top 20 --timeframe 1h,15m   Ingest top 20 by volume
  python master.py backtest                             Run grid search (Pipeline A)
  python master.py report                               Generate intelligence report (Pipeline B)
  python master.py full                                 Run complete weekly cycle
  python master.py scout                                Run hourly AI fusion

Recommended first-time workflow:
  1. python master.py ingest --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h
  2. python master.py status
  3. python master.py backtest
  4. python master.py report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ── ingest ──
    p_ingest = subparsers.add_parser('ingest', help='Download historical data to local DB')
    p_ingest.add_argument('--symbols', default='BTC/USDT,ETH/USDT,SOL/USDT', 
                          help='Comma-separated symbols (default: BTC,ETH,SOL)')
    p_ingest.add_argument('--top', type=int, default=0,
                          help='Auto-discover top N symbols by volume (overrides --symbols)')
    p_ingest.add_argument('--timeframe', default='1h', 
                          help='Comma-separated timeframes (default: 1h)')
    p_ingest.add_argument('--years', type=int, default=3, 
                          help='Target years of history (default: 3)')
    p_ingest.add_argument('--start-year', type=int, default=2020, dest='start_year',
                          help='Start year for Binance Vision download (default: 2020)')
    p_ingest.add_argument('--spot', action='store_true', 
                          help='Use Spot market data instead of Futures')
    p_ingest.set_defaults(func=cmd_ingest)
    
    # Common ML Arguments
    def add_ml_args(p):
        p.add_argument('--market', choices=['spot', 'futures'], default='futures',
                       help='Which market data to train/scan on (default: futures)')
        p.add_argument('--tune', action='store_true',
                       help='Run RandomizedSearchCV to find optimal XGBoost hyperparameters')
        p.add_argument('--force-train', '--force', action='store_true', dest='force_train',
                       help='Force retraining the model even if a cached version exists')

    # ── backtest ──
    p_backtest = subparsers.add_parser('backtest', help='Run Pipeline A: Grid Search + AI Scoring')
    add_ml_args(p_backtest)
    p_backtest.set_defaults(func=cmd_backtest)
    
    # ── report ──
    p_report = subparsers.add_parser('report', help='Run Pipeline B: Weekly Intelligence Report')
    add_ml_args(p_report)
    p_report.set_defaults(func=cmd_report)
    
    
    # ── full ──
    p_full = subparsers.add_parser('full', help='Run complete end-to-end cycle (Ingest? → Train → A → B)')
    # Include Ingest Args
    p_full.add_argument('--symbols', default='BTC/USDT,ETH/USDT,SOL/USDT', 
                          help='Symbols to ingest if bootstrapping')
    p_full.add_argument('--top', type=int, default=0,
                          help='Auto-discover top N symbols to bootstrap')
    p_full.add_argument('--timeframe', default='1h', 
                          help='Timeframes to ingest (default: 1h)')
    p_full.add_argument('--years', type=int, default=3, 
                          help='Years of history to fetch (default: 3)')
    p_full.add_argument('--start-year', type=int, default=2020, dest='start_year',
                          help='Binance Vision start year')
    
    add_ml_args(p_full)
    p_full.set_defaults(func=cmd_full)
    
    # ── audit ──
    p_audit = subparsers.add_parser('audit', help='Check data integrity (gaps and spikes)')
    p_audit.add_argument('--market', choices=['spot', 'futures'], help='Market to audit')
    p_audit.add_argument('--symbols', help='Specific symbols to audit (comma-separated)')
    p_audit.set_defaults(func=cmd_audit)
    
    # ── status ──
    p_status = subparsers.add_parser('status', help='Check database contents')
    p_status.set_defaults(func=cmd_status)

    # ── scout ──
    p_scout = subparsers.add_parser('scout', help='Hourly AI re-inference and fusion')
    p_scout.set_defaults(func=cmd_scout)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
