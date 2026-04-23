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

    # Index Price OHLCV Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS index_ohlcv (
            timestamp INTEGER,
            symbol TEXT,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    ''')

    # Symbol Metrics Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS symbol_metrics (
            timestamp INTEGER,
            create_time INTEGER,
            symbol TEXT,
            sum_open_interest REAL,
            sum_open_interest_value REAL,
            count_toptrader_long_short_ratio REAL,
            sum_toptrader_long_short_ratio REAL,
            count_long_short_ratio REAL,
            sum_long_short_ratio REAL,
            count_taker_long_short_vol_ratio REAL,
            sum_taker_long_short_vol_ratio REAL,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')

    # Sync State Table - Tracks progress for smart resuming
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sync_state (
            symbol TEXT,
            timeframe TEXT,
            market TEXT,
            data_type TEXT DEFAULT 'klines',
            earliest_timestamp INTEGER,
            latest_timestamp INTEGER,
            PRIMARY KEY (symbol, timeframe, market, data_type)
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
