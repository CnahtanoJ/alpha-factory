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

# Fix Windows console encoding (cp1252 can't handle emoji)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import sqlite3
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def cmd_ingest(args):
    """Step 1: Download historical data into alpha_factory.db"""
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


def cmd_report(args):
    """Run the Weekly Intelligence Cycle: OOS Simulate → Train → Report"""
    

    from analytics.weekly_orchestrator import run_weekly_cycle
    from analytics.generate_report import generate_report
    
    print("\n🧠 Running Full Intelligence Cycle...")
    cycle_results = run_weekly_cycle(
        market=args.market,
        force_train=args.force_train,
        dry_run_weeks=args.dry_run_weeks,
        optimize=getattr(args, 'optimize', False),
        n_trials=getattr(args, 'trials', 50)
    )
    
    print("\n📊 Generating Weekly Intelligence Report...")
    report_path = generate_report(cycle_results)
    
    if report_path:
        print(f"\n✅ Report saved to: {report_path}")


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
    cycle_results = run_weekly_cycle(
        market=args.market,
        force_train=args.force_train,
        dry_run_weeks=args.dry_run_weeks,
        optimize=getattr(args, 'optimize', False),
        n_trials=getattr(args, 'trials', 50)
    )
    
    sim = cycle_results.get('simulation_results')
    meta = cycle_results.get('model_meta', {})
    
    print(f"\n✅ AI Intelligence Cycle Complete!")
    if sim:
        print(f"   OOS Sharpe: {sim['sharpe']:.2f} | PF: {sim['profit_factor']:.2f} | Win Rate: {sim['win_rate']:.1%}")
    spearman = meta.get('validation_spearman_correlation', 'N/A')
    if isinstance(spearman, float):
        print(f"   Model Spearman ρ: {spearman:.4f}")
    
    # 3. Generate the intelligence report
    print("\n📊 Generating Master Intelligence Report...")
    report_path = generate_report(cycle_results)

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
        p.add_argument('--force-train', '--force', action='store_true', dest='force_train',
                       help='Force retraining the model even if a cached version exists')
        p.add_argument('--dry-run-weeks', type=int, default=4, dest='dry_run_weeks',
                       help='Number of weeks to simulate in OOS dry run (default: 4)')
        p.add_argument('--optimize', action='store_true', dest='optimize',
                       help='Run Optuna Hyperparameter Optimization before training')
        p.add_argument('--trials', type=int, default=50, dest='trials',
                       help='Number of Optuna trials to run if --optimize is set (default: 50)')

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
    p_full.add_argument('--timeframe', default='15m,1h,4h', 
                          help='Timeframes to ingest (default: 15m,1h,4h)')
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
