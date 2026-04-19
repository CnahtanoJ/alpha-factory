# Alpha Factory

> AI-driven autonomous crypto trading pipeline. Ingests historical data, trains ML models, discovers optimal strategies via exhaustive grid search, and deploys them as a fully autonomous trading bot on Hyperliquid.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OPENROUTER_API_KEY, AWS credentials, etc.)
```

### Step-by-Step Pipeline

```bash
# 1. Check database status
python master.py status

# 2. Ingest historical data (Binance Vision + CCXT)
python master.py ingest --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h

# 3. Run grid search + AI scoring (Pipeline A)
python master.py backtest

# 4. Generate weekly intelligence report (Pipeline B)
python master.py report

# 5. Run full weekly cycle (all pipelines)
python master.py full
```

## Architecture

Three pipelines powered by a shared XGBoost model:

| Pipeline | What It Does | Output |
|----------|-------------|--------|
| **A — Strategist** | Grid search across 12 strategies × all coins × all params | `champion_blueprint.json` (S3) |
| **B — Report** | Weekly intelligence with LLM verdict | `latest_market_report.md` |

The **bot** runs independently on AWS Lambda, reading the champion strategy from S3.

**For the complete system walkthrough, see [ARCHITECTURE.md](ARCHITECTURE.md).**

## Requirements

- Python 3.9+
- No GPU needed — XGBoost runs on CPU
- AWS credentials (for bot deployment only)
- OpenRouter API key (for LLM verdicts)

## Key Files

| File | Purpose |
|------|---------|
| `master.py` | CLI entry point for all local operations |
| `analytics/weekly_orchestrator.py` | Trains ML once, feeds all pipelines |
| `backtester/build_bot_blueprint.py` | Pipeline A grid search |
| `analytics/generate_report.py` | Pipeline B report generation |
| `bot/bot_executor.py` | Lambda trading bot |
