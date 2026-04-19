import sqlite3
conn = sqlite3.connect(r"E:\trading_data\alpha_factory.db")

total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
print(f"Total rows: {total:,}")

tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"Tables: {tables}")

print(f"\n{'Symbol':<15} {'TF':<6} {'Market':<8} {'Candles':>10}")
print(f"{'-'*15} {'-'*6} {'-'*8} {'-'*10}")

rows = conn.execute("""
    SELECT symbol, timeframe, market, COUNT(*) as cnt
    FROM ohlcv GROUP BY symbol, timeframe, market ORDER BY cnt DESC LIMIT 20
""").fetchall()
for r in rows:
    print(f"{r[0]:<15} {r[1]:<6} {r[2]:<8} {r[3]:>10,}")

sync = conn.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0]
print(f"\nSync state entries: {sync}")

conn.close()
