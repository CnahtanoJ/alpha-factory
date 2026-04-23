import sqlite3
import os

DB_PATH = "alpha_factory.db"
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    for table in tables:
        t_name = table[0]
        cursor.execute(f"PRAGMA table_info({t_name});")
        print(f"\nSchema for {t_name}:")
        for col in cursor.fetchall():
            print(col)
    conn.close()
else:
    print("DB not found.")
