import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alpha_factory.db")
DB_PATH = os.getenv("DB_PATH", DEFAULT_DB_PATH)

def get_connection():
    """Returns a connection to the SQLite database with WAL and performance settings."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Logging for better concurrent performance
    conn.execute("PRAGMA journal_mode=WAL;")
    # Increase cache size to improve performance
    conn.execute("PRAGMA cache_size = -10000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def init_db():
    conn = get_connection()
    # OHLCV Table - Stores the actual candle data
    # market: 'spot' or 'futures'
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp INTEGER,
            symbol TEXT,
            timeframe TEXT,
            market TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, market, timestamp)
        )
    ''')

    # Sync State Table - Tracks progress for smart resuming
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sync_state (
            symbol TEXT,
            timeframe TEXT,
            market TEXT,
            earliest_timestamp INTEGER,
            latest_timestamp INTEGER,
            PRIMARY KEY (symbol, timeframe, market)
        )
    ''')
    
    # Blueprints Table: Stores the "Winning" strategies
    conn.execute("""
    CREATE TABLE IF NOT EXISTS blueprints (
        id TEXT PRIMARY KEY,
        created_at TEXT,
        config_json TEXT,
        performance_metrics TEXT,
        is_active INTEGER DEFAULT 0
    )
    """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
