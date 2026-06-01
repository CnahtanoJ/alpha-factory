# Alpha Factory 🚀

> A professional-grade quantitative crypto research and autonomous trading system. Pure Binance data powers an exhaustive strategy grid search, an AI ensemble scout, and an autonomous executor on Hyperliquid.

---

## 🏛️ Architecture: The Two-Phase Loop

Alpha Factory operates in two distinct phases to maximize research depth and execution speed:

### Phase 1: Local Research Hub (Weekly)
*   **Data Ingestion**: Pulls years of high-fidelity data from **Binance Vision** archives and fills the "live edge" via Binance API.
*   **AI Training**: Trains an **XGBoost + Random Forest Ensemble** to identify market regimes and movement conviction.
*   **The Strategist (Grid Search)**: Performs an exhaustive backtest across **100+ symbols** and **12 strategy families**.
*   **Outputs**: 
    *   `elite_squad.json`: Top 20 strategies with a Hyperliquid-tradability guarantee.
    *   `all_grid_results.json`: Full market distribution data.
    *   `weekly_intelligence_report.md`: High-level strategic breakdown for human review.

### Phase 2: Cloud Execution (Hourly)
*   **The Scout**: Re-scores the Elite Squad using live market data and AI conviction.
*   **Fusion Logic**: Combines historical math (Sharpe/PF) with live AI probabilities to find the absolute best trade at that moment.
*   **The Executor**: An autonomous **HyperliquidBot** (AWS Lambda) that manages positions, risk, and order flow.

---

## 🛠️ Quick Start

```bash
# 1. Setup Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env # Add your API keys and AWS bucket
```

---

## 🕹️ CLI Usage (`master.py`)

The single entry point for all local operations:

| Command | Action |
|:---|:---|
| `python master.py status` | Check database health and row counts. |
| `python master.py ingest --top 50` | Ingest bulk historical data from Binance Vision. |
| `python master.py backtest` | Run the weekly ML training + Global Grid Search. |
| `python master.py report` | Generate the Weekly Intelligence Markdown report. |
| `python master.py full` | Run the entire weekly cycle from Sync to Report. |
| `python master.py scout` | Re-score the elite squad (Hourly loop). |
| `python master.py audit` | Run a non-destructive health check (Gaps/Spikes) on the DB. |

---

## 💎 Features

*   **100% Data Purity**: Enforces a "Pure Binance" policy — no Hyperliquid data injection for research.
*   **12 Strategy Families**: Trend Following, Mean Reversion, Order Flow, and Hybrid systems.
*   **AI Ensemble**: Multi-model (XGB+RF) movement conviction scoring.
*   **Risk Engine**: Integrated funding traps, OI floors, and Point-of-Control (POC) gravity checks.
*   **Telegram Guard**: Real-time private reporting of all trades and PnL receipts.

---

## 📁 Directory Structure

*   `backtester/`: Vectorized backtest engine & grid search orchestration.
*   `analytics/`: ML model training and intelligence reporting.
*   `data_pipeline/`: Binance Vision ingest, CCXT gap-filling, and DB auditing.
*   `bot/`: AWS Lambda execution logic (Risk Engine, Strategies, Data Feed).
*   `master.py`: The central CLI.

---

**For more details on the math and logic, see the [System Architecture Document](ARCHITECTURE.md).**
