-- ============================================================================
-- SQL Verification Suite for Alpha Factory Data Integrity
-- ============================================================================

-- 1. THE GAP FINDER: Identify exact "Dark Zones" in the database
-- This looks for any jump between candles larger than the expected timeframe.
SELECT 
    symbol, 
    timeframe, 
    market,
    datetime(timestamp/1000, 'unixepoch', 'localtime') as gap_starts_at,
    datetime(next_ts/1000, 'unixepoch', 'localtime') as gap_ends_at,
    (next_ts - timestamp) / 60000 as gap_duration_minutes
FROM (
    SELECT 
        symbol, timeframe, market, timestamp,
        LEAD(timestamp) OVER (PARTITION BY symbol, timeframe, market ORDER BY timestamp) as next_ts
    FROM ohlcv
) 
WHERE next_ts IS NOT NULL 
  AND (next_ts - timestamp) > (
      CASE timeframe 
          WHEN '15m' THEN 15 
          WHEN '1h' THEN 60 
          WHEN '4h' THEN 240 
          ELSE 1440 
      END
  ) * 60000
ORDER BY gap_duration_minutes DESC
LIMIT 100;


-- 2. THE SPIKE DETECTOR: Flag anomalous price moves (> 20%)
-- Useful for finding "Bad Handover" spikes where Binance and HL data meet.
SELECT 
    symbol, 
    timeframe, 
    datetime(timestamp/1000, 'unixepoch', 'localtime') as candle_time,
    open, 
    close,
    ROUND(ABS(close - open) / open * 100, 2) as pct_move
FROM ohlcv
WHERE open > 0 
  AND ABS(close - open) / open > 0.20
ORDER BY pct_move DESC
LIMIT 100;


-- 3. THE COVERAGE SUMMARY: High-level health check per partition
SELECT 
    symbol, 
    timeframe, 
    market,
    COUNT(*) as total_candles,
    datetime(MIN(timestamp)/1000, 'unixepoch', 'localtime') as data_starts,
    datetime(MAX(timestamp)/1000, 'unixepoch', 'localtime') as data_ends
FROM ohlcv
GROUP BY symbol, timeframe, market
ORDER BY total_candles DESC;
