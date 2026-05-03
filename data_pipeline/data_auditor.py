import sqlite3
import pandas as pd
import numpy as np
from data_pipeline.database import DB_PATH

class DataAuditor:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.tf_ms_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }

    def audit_structural_integrity(self):
        """Runs the structural audit originally from audit_db_structure.py"""
        print("\n========================================")
        print(" DATABASE STRUCTURAL AUDIT ")
        print("========================================")
        
        conn = sqlite3.connect(self.db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        for table in tables['name']:
            print(f"\n--- Schema: {table} ---")
            info = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
            print(info[['name', 'type']].to_string(index=False))
            
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
            print(f"Total Rows: {count:,}")
            
            if table == 'ohlcv':
                print("\n  Timeframe Distribution:")
                dist = pd.read_sql_query("SELECT timeframe, COUNT(*) as count FROM ohlcv GROUP BY timeframe", conn)
                print(dist.to_string(index=False))

        print("\n=== Sync Source Audit ===")
        sync_df = pd.read_sql_query("SELECT * FROM sync_state", conn)
        if sync_df.empty:
            print("  ⚠️ No sync state found.")
        elif sync_df[sync_df['data_type'] == 'klines'].empty:
            print("  ⚠️ Warning: No 'klines' (Binance Vision) sync state found.")
        else:
            print("  ✅ 'klines' (Binance Vision) detected as primary source.")
            
        conn.close()

    def audit_join_integrity(self, symbol='BTC/USDT', timeframe='1h'):
        """Tests the ability to join OHLCV, Metrics, and Funding Rate cleanly."""
        print("\n========================================")
        print(" JOIN INTEGRITY AUDIT ")
        print("========================================")
        
        conn = sqlite3.connect(self.db_path)
        join_query = """
        SELECT 
            o.timestamp, o.symbol, o.close, 
            m.sum_open_interest, 
            f.last_funding_rate 
        FROM ohlcv o
        LEFT JOIN symbol_metrics m ON o.timestamp = m.timestamp AND o.symbol = m.symbol
        LEFT JOIN funding_rate f ON o.timestamp = f.calc_time AND o.symbol = f.symbol
        WHERE o.symbol = ? AND o.timeframe = ?
        ORDER BY o.timestamp DESC
        LIMIT 5
        """
        joined_data = pd.read_sql_query(join_query, conn, params=(symbol, timeframe))
        print(f"Sample Joined Rows ({symbol} {timeframe}):")
        print(joined_data.to_string())
        
        index_query = """
        SELECT 
            o.timestamp, o.symbol as asset, o.close as asset_close, 
            i.close as index_close
        FROM ohlcv o
        JOIN index_ohlcv i ON o.timestamp = i.timestamp
        WHERE o.symbol = ? AND o.timeframe = ? AND i.symbol = ?
        ORDER BY o.timestamp DESC
        LIMIT 5
        """
        index_joined = pd.read_sql_query(index_query, conn, params=(symbol, timeframe, symbol))
        print(f"\nSample Index Join ({symbol} {timeframe}):")
        print(index_joined.to_string())
        
        conn.close()

    def audit_pair(self, symbol, timeframe, market, anomaly_threshold=0.20):
        """
        Runs a full health check on a specific (symbol, timeframe, market) partition.
        Returns a dict containing health metrics.
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv 
            WHERE symbol = ? AND timeframe = ? AND market = ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, market))

        if df.empty:
            conn.close()
            return {"status": "EMPTY", "symbol": symbol, "timeframe": timeframe}

        results = {
            "status": "OK",
            "symbol": symbol,
            "timeframe": timeframe,
            "market": market,
            "total_rows": len(df),
            "earliest": df['timestamp'].min(),
            "latest": df['timestamp'].max(),
            "gaps_found": 0,
            "missing_candles": 0,
            "anomalies": 0,
            "max_spike": 0.0,
            "health_score": 100.0,
            "grade": "A"
        }

        # 1. Temporal Check (Gaps)
        expected_delta = self.tf_ms_map.get(timeframe)
        if expected_delta:
            diffs = df['timestamp'].diff()
            gaps = diffs[diffs > expected_delta]
            
            if not gaps.empty:
                results["gaps_found"] = len(gaps)
                raw_missing = int((gaps // expected_delta - 1).sum())
                
                # C-2 FIX: Safely query unfillable_gaps (table may not exist on fresh DBs)
                unfillable_candles = 0
                try:
                    unfillable_df = pd.read_sql_query(
                        "SELECT start_ts, end_ts FROM unfillable_gaps WHERE table_name='ohlcv' AND symbol=? AND timeframe=?", 
                        conn, params=(symbol, timeframe)
                    )
                    if not unfillable_df.empty:
                        for _, row in unfillable_df.iterrows():
                            gap_dur = row['end_ts'] - row['start_ts']
                            unfillable_candles += int(gap_dur // expected_delta - 1)
                except Exception:
                    pass  # Table doesn't exist yet — treat all gaps as fillable
                        
                results["missing_candles"] = max(0, raw_missing - unfillable_candles)
                
        conn.close()

        # 2. Mathematical Check (Anomalies)
        df['pct_move'] = (df['close'] - df['open']).abs() / df['open'].replace(0, np.nan)
        spikes = df[df['pct_move'] > anomaly_threshold]
        
        results["anomalies"] = len(spikes)
        results["max_spike"] = df['pct_move'].max() if not df['pct_move'].empty else 0.0

        # 3. Calculate Health Grade (REVISED FOR FAIRNESS)
        # We no longer apply a massive flat penalty per gap instance (which unfairly punished missing singular candles).
        # We only penalize based on the percentage of TOTAL TIME missing.
        gap_ratio = results["missing_candles"] / (results["total_rows"] + results["missing_candles"]) if results["total_rows"] > 0 else 0
        
        # Penalty formula: 
        # - Anomaly penalty: 5 points per flash crash (anomalies are dangerous to ML)
        # - Gap ratio penalty: 1% missing data = 2 points lost.
        penalty = (results["anomalies"] * 5) + ((gap_ratio * 100) * 2)
        
        results["gap_ratio"] = gap_ratio
        results["health_score"] = max(0, 100 - penalty)
        
        s = results["health_score"]
        if s >= 98: results["grade"] = "A+"
        elif s >= 90: results["grade"] = "A"
        elif s >= 80: results["grade"] = "B"
        elif s >= 70: results["grade"] = "C"
        elif s >= 50: results["grade"] = "D"
        else: results["grade"] = "F"

        return results

    def get_all_partitions(self):
        """Lists all (symbol, timeframe, market) combinations in the DB."""
        conn = sqlite3.connect(self.db_path)
        partitions = conn.execute("SELECT DISTINCT symbol, timeframe, market FROM ohlcv").fetchall()
        conn.close()
        return partitions

if __name__ == "__main__":
    auditor = DataAuditor()
    
    auditor.audit_structural_integrity()
    auditor.audit_join_integrity()
    
    print("\n========================================")
    print(" PAIR HEALTH AUDIT (Top 5) ")
    print("========================================")
    parts = auditor.get_all_partitions()
    for p in parts[:5]:
        print(f"\nAuditing {p}...")
        res = auditor.audit_pair(*p)
        if res["status"] == "OK":
            print(f"  Health : {res['grade']} (Score: {res['health_score']:.1f})")
            print(f"  Rows   : {res['total_rows']:,} | Gaps: {res['gaps_found']} instances ({res['missing_candles']} candles)")
            print(f"  Anom.  : {res['anomalies']} | Max Spike: {res['max_spike']:.2%}")
        else:
            print("  Empty data partition.")
