import sqlite3

# Check E: drive schema
print("=== E:\\trading_data\\alpha_factory.db ===")
try:
    conn = sqlite3.connect(r"E:\trading_data\alpha_factory.db")
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    for row in cursor.fetchall():
        print(row[0])
        print()
    
    # Row counts
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{t[0]}]").fetchone()[0]
        print(f"  {t[0]}: {count} rows")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

print("\n=== asset-analysis\\alpha_factory.db ===")
try:
    conn = sqlite3.connect(r"alpha_factory.db")
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    for row in cursor.fetchall():
        print(row[0])
        print()
    
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{t[0]}]").fetchone()[0]
        print(f"  {t[0]}: {count} rows")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
