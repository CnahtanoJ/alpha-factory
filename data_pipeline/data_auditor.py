import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timezone
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
        conn.close()

        if df.empty:
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
            # The first diff is NaN, so we start from the second row
            gaps = diffs[diffs > expected_delta]
            
            if not gaps.empty:
                results["gaps_found"] = len(gaps)
                results["missing_candles"] = int((gaps // expected_delta - 1).sum())

        # 2. Mathematical Check (Anomalies)
        # Price spikes (abs change > 20%)
        df['pct_move'] = (df['close'] - df['open']).abs() / df['open'].replace(0, np.nan)
        spikes = df[df['pct_move'] > anomaly_threshold]
        
        results["anomalies"] = len(spikes)
        results["max_spike"] = df['pct_move'].max() if not df['pct_move'].empty else 0.0

        # 3. Calculate Health Grade
        # Gaps are more critical than spikes (penalize 2pts per gap, 10pts per anomaly)
        penalty = (results["gaps_found"] * 2) + (results["anomalies"] * 10)
        
        # Also factor in gap density
        gap_ratio = results["missing_candles"] / (results["total_rows"] + results["missing_candles"]) if results["total_rows"] > 0 else 0
        penalty += gap_ratio * 100
        
        results["gap_ratio"] = gap_ratio
        results["health_score"] = max(0, 100 - penalty)
        
        # Assign Grade
        s = results["health_score"]
        if s >= 95: results["grade"] = "A+"
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
    print("Listing partitions...")
    parts = auditor.get_all_partitions()
    for p in parts[:5]:
        print(f"Auditing {p}...")
        res = auditor.audit_pair(*p)
        print(f"  Health: {res['grade']} (Score: {res['health_score']:.1f}) | Gaps: {res['gaps_found']}")
