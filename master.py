#!/usr/bin/env python3
"""
Alpha Factory — Master CLI

Single entry point to run every pipeline step-by-step.
Designed for local execution on your machine.

Usage:
  python master.py ingest   --symbols BTC/USDT,ETH/USDT --timeframe 1h
  python master.py ingest   --top 50 --timeframe 15m
  python master.py report
  python master.py full
  python master.py status
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding (cp1252 can't handle emoji)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import sqlite3
import json
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.database import DB_PATH

class Tee:
    """Redirects stdout/stderr to both the console and a file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def cmd_ingest(args):
    """Step 1: Download historical data into alpha_factory.db"""
    # Setup timestamped logging
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_file = os.path.join('logs', f"ingest_{timestamp}.log")
    
    log_f = open(log_file, 'a', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Start Teeing
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    
    try:
        print(f"📝 Logging session started: {log_file}")
        _run_ingest_logic(args)
    finally:
        print(f"📝 Logging session finished: {log_file}")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_f.close()

def _run_ingest_logic(args):
    """The actual ingestion logic, now wrapped in logging."""
    from data_pipeline.database import init_db
    from data_pipeline.sync_manager import SyncManager
    
    init_db()
    sync = SyncManager()
    
    market = 'futures'
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
    
    # Automate Gap Patcher
    print("\n" + "="*60)
    print("  🩹 UNIVERSAL GAP PATCHER")
    print("="*60)
    from data_pipeline.universal_gap_patcher import UniversalGapPatcher
    patcher = UniversalGapPatcher()
    patcher.patch_ohlcv(dry_run=False)
    patcher.patch_index_ohlcv(dry_run=False)
    patcher.patch_symbol_metrics(dry_run=False)
    patcher.patch_funding_rate(dry_run=False)
    patcher.close()
    
    # Automate Auditor
    args.market = market
    cmd_audit(args)
    
    print("\n✅ Full Data Ingestion, Patching, and Auditing Complete!")
    
def save_active_regime(all_results):
    """
    Evaluates all timeframe results and saves the best one to live_config.json.
    """
    best_tf = None
    best_sharpe = -float('inf')
    best_spearman = -float('inf')

    for tf, res in all_results.items():
        sim = res.get('simulation_results')
        meta = res.get('model_meta', {})
        
        sharpe = sim.get('sharpe', -float('inf')) if sim else -float('inf')
        spearman = meta.get('validation_spearman', -float('inf'))
        
        # Primary sort: Sharpe Ratio. Secondary: Spearman
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_spearman = spearman
            best_tf = tf
        elif sharpe == best_sharpe and spearman > best_spearman:
            best_spearman = spearman
            best_tf = tf

    if best_tf:
        config_path = os.path.join(os.path.dirname(__file__), 'bot', 'live_config.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump({
                "active_timeframe": best_tf,
                "sharpe": float(best_sharpe) if best_sharpe != -float('inf') else 0.0,
                "spearman": float(best_spearman) if best_spearman != -float('inf') else 0.0,
                "updated_at": datetime.now().isoformat()
            }, f, indent=4)
        print(f"\n👑 REGIME SWITCHER: Selected '{best_tf}' as the optimal timeframe (Sharpe: {best_sharpe:.2f}, Spearman: {best_spearman:.4f})")
        print(f"   Saved to {config_path}")

        # PHASE 2: Upload to S3 so Lambda knows which timeframe is active
        try:
            import boto3
            from bot.config import AWS_BUCKET
            s3 = boto3.client('s3')
            with open(config_path, 'rb') as data:
                s3.put_object(Bucket=AWS_BUCKET, Key='live_config.json', Body=data.read())
            print(f"   ✅ live_config.json uploaded to S3 bucket '{AWS_BUCKET}'.")
        except Exception as e:
            print(f"   ⚠️ live_config.json S3 upload failed: {e}")

def cmd_report(args):
    """Run the Weekly Intelligence Cycle: OOS Simulate → Train → Report"""
    

    from analytics.weekly_orchestrator import run_weekly_cycle
    from analytics.generate_report import generate_report
    
    timeframes = args.timeframe.split(',')
    # PHASE 4: Enforce 4h-first training order (macro conviction needs 4h model)
    tf_order = {'4h': 0, '1h': 1, '15m': 2}
    timeframes = sorted([t.strip() for t in timeframes], key=lambda x: tf_order.get(x, 99))
    all_results = {}
    
    print(f"\n🧠 Running Full Intelligence Cycle for timeframes: {timeframes}")
    
    for tf in timeframes:
        tf = tf.strip()
        print(f"\n--- Processing Timeframe: {tf} ---")
        cycle_results = run_weekly_cycle(
            market=args.market,
            timeframe=tf,
            force_train=args.force_train,
            dry_run_weeks=args.dry_run_weeks,
            optimize=getattr(args, 'optimize', False),
            n_trials=getattr(args, 'trials', 50)
        )
        if isinstance(cycle_results, dict) and 'status' not in cycle_results:
            all_results[tf] = cycle_results
        else:
            print(f"⚠️ Skipping {tf} due to training failure or no data.")
    
    if not all_results:
        print("\n❌ No successful cycles completed. Report aborted.")
        return

    print("\n📊 Generating Aggregated Intelligence Report...")
    report_path = generate_report(all_results)
    
    save_active_regime(all_results)
    
    if report_path:
        print(f"\n✅ Report saved to: {report_path}")
        
        # PHASE 3: Send to Telegram
        if args.ping_telegram:
            print("📱 Sending report to Telegram...")
            from bot.utils import send_telegram_message
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            send_telegram_message(report_content)
            print("   ✅ Telegram transmission complete.")


def cmd_full(args):
    """Run the full weekly cycle: Ingest (Optional) → Sync → OOS Simulate → Train → Report"""
    from analytics.weekly_orchestrator import run_weekly_cycle
    from analytics.generate_report import generate_report
    
    # 0. Check if we need to Ingest first
    if args.top > 0 or args.symbols != 'BTC/USDT,ETH/USDT,SOL/USDT':
        print("\n" + "="*60)
        print("  📥 STEP 0: BOOTSTRAP INGESTION")
        print("="*60)
        cmd_ingest(args)
        
    print("\n" + "="*60)
    print("  🚀 STARTING FULL WEEKLY CYCLE")
    if args.force_train:
        print("     🧠 MODE: FORCE RE-TRAINING")
    print("="*60)
    
    # 2. Run the intelligence cycle
    timeframes = args.timeframe.split(',')
    # PHASE 4: Enforce 4h-first training order (macro conviction needs 4h model)
    tf_order = {'4h': 0, '1h': 1, '15m': 2}
    timeframes = sorted([t.strip() for t in timeframes], key=lambda x: tf_order.get(x, 99))
    all_results = {}
    
    for tf in timeframes:
        tf = tf.strip()
        cycle_results = run_weekly_cycle(
            market=args.market,
            timeframe=tf,
            force_train=args.force_train,
            dry_run_weeks=args.dry_run_weeks,
            optimize=getattr(args, 'optimize', False),
            n_trials=getattr(args, 'trials', 50)
        )
        if isinstance(cycle_results, dict) and 'status' not in cycle_results:
            all_results[tf] = cycle_results

    if all_results:
        print("\n📊 Generating Aggregated Intelligence Report...")
        report_path = generate_report(all_results)
        save_active_regime(all_results)

        # PHASE 3: Send to Telegram
        if args.ping_telegram:
            print("📱 Sending report to Telegram...")
            from bot.utils import send_telegram_message
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            send_telegram_message(report_content)
            print("   ✅ Telegram transmission complete.")
    
    print(f"\n✅ AI Intelligence Cycle Complete!")

def cmd_status(args):
    """Show what data you have in the local database"""

    print("\n" + "="*60)
    print("  📊 LOCAL DATABASE STATUS")
    print("="*60)

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get symbols and timeframes
    cursor.execute("SELECT symbol, timeframe, COUNT(*) as count, MIN(timestamp), MAX(timestamp) FROM ohlcv GROUP BY symbol, timeframe")
    rows = cursor.fetchall()

    if not rows:
        print("📭 Database is empty.")
    else:
        print(f"{'Symbol':<15} | {'TF':<5} | {'Rows':<8} | {'Start':<20} | {'End':<20}")
        print("-" * 75)
        for row in rows:
            symbol, tf, count, start, end = row
            start_dt = datetime.fromtimestamp(start/1000).strftime('%Y-%m-%d')
            end_dt = datetime.fromtimestamp(end/1000).strftime('%Y-%m-%d')
            print(f"{symbol:<15} | {tf:<5} | {count:<8,} | {start_dt:<20} | {end_dt:<20}")

    conn.close()
    print("="*60)

def cmd_health(args):
    """Verify all sub-account credentials and connectivity"""
    from hyperliquid.info import Info
    from bot.config import BASE_URL, AWS_BUCKET
    from bot.utils import send_telegram_message

    print("\n" + "="*60)
    print("  🩺 INFRASTRUCTURE HEALTH CHECK")
    print("="*60)

    timeframes = ['15m', '1h', '4h']
    info = Info(BASE_URL, skip_ws=True)

    for tf in timeframes:
        suffix = tf.upper()
        print(f"\n📡 Checking {tf} Sub-Account...")
        
        # Resolve keys (matching bot_executor logic)
        if os.environ.get("TESTNET_MODE", "False").lower() == "true":
            key = os.environ.get(f"TESTNET_PRIVATE_KEY_{suffix}", os.environ.get("TESTNET_PRIVATE_KEY"))
            addr = os.environ.get(f"TESTNET_ACCOUNT_ADDRESS_{suffix}", os.environ.get("TESTNET_ACCOUNT_ADDRESS"))
            mode = "TESTNET"
        else:
            key = os.environ.get(f"MAINNET_PRIVATE_KEY_{suffix}", os.environ.get("MAINNET_PRIVATE_KEY"))
            addr = os.environ.get(f"MAINNET_ACCOUNT_ADDRESS_{suffix}", os.environ.get("MAINNET_ACCOUNT_ADDRESS"))
            mode = "MAINNET"

        if not key or not addr:
            print(f"  ❌ Status: OFFLINE (Missing Keys for _{suffix})")
            continue
            
        key = key.strip()
        addr = addr.strip()

        try:
            # Try to fetch user state
            user_state = info.user_state(addr)
            margin = user_state.get('marginSummary', {}).get('accountValue', '0')
            print(f"  ✅ Status: ONLINE ({mode})")
            print(f"  📍 Address: {addr[:6]}...{addr[-4:]}")
            print(f"  💰 Account Value: ${float(margin):,.2f}")
        except Exception as e:
            print(f"  ❌ Status: ERROR (Connection failed: {e})")

    # Check S3
    print("\n📦 Checking AWS S3 Connectivity...")
    try:
        import boto3
        client = boto3.client('s3')
        client.list_objects_v2(Bucket=AWS_BUCKET, MaxKeys=1)
        print(f"  ✅ Status: CONNECTED (Bucket: {AWS_BUCKET})")
    except Exception as e:
        print(f"  ❌ Status: FAILED ({e})")

    # Check Telegram
    print("\n📱 Checking Telegram Bot...")
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        print(f"  ✅ Status: CONFIGURED")
        if args.ping_telegram:
            print("  🔔 Sending test message...")
            send_telegram_message("🩺 *Alpha Factory Health Check*: Connectivity Verified.")
    else:
        print("  ⚠️ Status: NOT CONFIGURED")

    print("\n" + "="*60)
    print("  ✅ HEALTH CHECK COMPLETE")
    print("="*60)

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



def main():
    parser = argparse.ArgumentParser(
        description="Alpha Factory — Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master.py status                              Check your database
  python master.py ingest --symbols BTC/USDT,ETH/USDT  Ingest specific coins
  python master.py ingest --top 100 --timeframe 1h,15m  Ingest top 100 by HL volume
  python master.py report                               Run intelligence cycle + report
  python master.py full                                 Run complete weekly cycle

Recommended first-time workflow:
  1. python master.py ingest --top 100 --timeframe 1h
  2. python master.py status
  3. python master.py report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ── ingest ──
    p_ingest = subparsers.add_parser('ingest', help='Download historical data to local DB')
    p_ingest.add_argument('--symbols', default='BTC/USDT,ETH/USDT,SOL/USDT', 
                          help='Comma-separated symbols (default: BTC,ETH,SOL)')
    p_ingest.add_argument('--top', type=int, default=0,
                          help='Auto-discover top N symbols by volume (overrides --symbols)')
    p_ingest.add_argument('--timeframe', default='15m,1h,4h', 
                          help='Comma-separated timeframes (default: 15m,1h,4h)')
    p_ingest.add_argument('--years', type=int, default=3, 
                          help='Target years of history (default: 3)')
    p_ingest.add_argument('--start-year', type=int, default=2020, dest='start_year',
                          help='Start year for Binance Vision download (default: 2020)')
    p_ingest.set_defaults(func=cmd_ingest)
    
    # Common ML Arguments
    def add_ml_args(p):
        p.add_argument('--market', choices=['futures'], default='futures',
                       help='Which market data to train on (default: futures)')
        p.add_argument('--timeframe', default='15m,1h,4h', 
                       help='Comma-separated timeframes to run report on (default: 15m,1h,4h)')
        p.add_argument('--force-train', '--force', action='store_true', dest='force_train',
                       help='Force retraining the model even if a cached version exists')
        p.add_argument('--dry-run-weeks', type=int, default=4, dest='dry_run_weeks',
                       help='Number of weeks to simulate in OOS dry run (default: 4)')
        p.add_argument('--optimize', action='store_true', dest='optimize',
                       help='Run Optuna Hyperparameter Optimization before training')
        p.add_argument('--trials', type=int, default=50, dest='trials',
                       help='Number of Optuna trials to run if --optimize is set (default: 50)')
        p.add_argument('--ping', action='store_true', dest='ping_telegram',
                       help='Send the generated report to Telegram')

    # ── report ──
    p_report = subparsers.add_parser('report', help='Run intelligence cycle + generate report')
    add_ml_args(p_report)
    p_report.set_defaults(func=cmd_report)
    
    
    # ── full ──
    p_full = subparsers.add_parser('full', help='Run complete end-to-end cycle (Ingest? → Sync → Train → Report)')
    # Include Ingest Args
    p_full.add_argument('--symbols', default='BTC/USDT,ETH/USDT,SOL/USDT', 
                          help='Symbols to ingest if bootstrapping')
    p_full.add_argument('--top', type=int, default=0,
                          help='Auto-discover top N symbols to bootstrap')
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
    
    # ── health ──
    p_health = subparsers.add_parser('health', help='Verify all sub-account credentials and connectivity')
    p_health.add_argument('--ping', action='store_true', dest='ping_telegram',
                          help='Send a test message to Telegram if configured')
    p_health.set_defaults(func=cmd_health)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
