"""
Weekly Intelligence Report Generator.

Produces a comprehensive Markdown report from:
  - OOS Dry Run Simulation results (Sharpe, PF, Win Rate)
  - LightGBM Feature Importance (model's top drivers)
  - Top 10 / Bottom 10 asset rankings with per-asset feature attribution
  - AI Executive Verdict from OpenRouter LLM

Now supports multiple timeframes in a single report.
"""
import os
from datetime import datetime
from analytics.llm_analyzer import get_llm_verdict, build_llm_context

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')

def build_report_section(timeframe: str, cycle_results: dict) -> str:
    """Builds a single timeframe's section of the report."""
    sim = cycle_results.get('simulation_results')
    feat_imp = cycle_results.get('feature_importance', {})
    drivers = cycle_results.get('per_asset_drivers', {})
    model_meta = cycle_results.get('model_meta', {})
    top_n = cycle_results.get('top_n', 10)
    bottom_n = cycle_results.get('bottom_n', 10)

    md = f"## ⏱️ Timeframe: {timeframe}\n\n"

    # --- Section 1: Model Health ---
    md += "### 📊 Model Training Summary\n"
    rmse = model_meta.get('validation_rmse', 'N/A')
    oos_spearman = sim.get('oos_spearman', 'N/A') if sim else 'N/A'

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    if isinstance(oos_spearman, float):
        if oos_spearman > 0.05:
            health = "🟢 Strong"
        elif oos_spearman > 0.03:
            health = "🟡 Moderate"
        elif oos_spearman > 0.02:
            health = "🟠 Weak"
        else:
            health = "🔴 No Signal"
        md += f"> Model Health: {health} | OOS Spearman ρ = **{oos_spearman:.4f}** | RMSE = {fmt(rmse)}\n\n"
    else:
        md += f"> Model Health: Unknown | RMSE = {fmt(rmse)} | OOS Spearman = {oos_spearman}\n\n"

    # --- Section 2: OOS Simulation Results ---
    md += "### 🔬 Out-of-Sample Dry Run\n"
    md += "*Results use LAST week's model on THIS week's data.*\n\n"

    if sim and sim.get('n_rebalances', 0) > 0:
        md += "| Metric | Value |\n"
        md += "| :--- | :--- |\n"
        md += f"| **Total Return** | {sim['total_return']:+.2%} |\n"
        md += f"| **Raw Sharpe** | {sim['sharpe']:.2f} |\n"
        md += f"| **Ann. Sharpe** | {sim.get('annualized_sharpe', 0.0):.2f} |\n"
        md += f"| **Profit Factor** | {sim['profit_factor']:.2f} |\n"
        md += f"| **Win Rate** | {sim['win_rate']:.1%} |\n"
        md += f"| **Max Drawdown** | {sim['max_drawdown']:.2%} |\n"
        md += f"| **Rebalances** | {sim['n_rebalances']} |\n"
        md += f"| **Avg Daily Return** | {sim['avg_daily_return']:+.4%} |\n\n"

        if 'mc_stats' in sim:
            mc = sim['mc_stats']
            md += "#### 🛡️ Robustness Analysis (Monte Carlo)\n"
            md += f"- **Probability of Profit**: {mc['prob_profit']:.1%}\n"
            md += f"- **95% CI**: [{mc['ci_lower']:+.2%}, {mc['ci_upper']:+.2%}]\n\n"
    else:
        md += "> ⚠️ No OOS simulation available. This is the first training run.\n\n"

    # --- Section 3: Top Assets ---
    top_symbols = drivers.get('top_symbols', [])
    top_drv = drivers.get('top_drivers', {})

    md += f"#### 🟢 Top {len(top_symbols)} Longs\n"
    md += "| Rank | Symbol | Score | Key Drivers |\n"
    md += "| :--- | :--- | :--- | :--- |\n"

    for i, entry in enumerate(top_symbols):
        sym = entry['symbol']
        rank = entry['predicted_rank']
        d = top_drv.get(sym, {})
        driver_str = ", ".join([f"`{k}`" for k in list(d.keys())[:3]]) if d else "—"
        md += f"| {i+1} | **{sym}** | {rank:.4f} | {driver_str} |\n"

    # --- Section 4: Bottom Assets ---
    bottom_symbols = drivers.get('bottom_symbols', [])
    bottom_drv = drivers.get('bottom_drivers', {})

    md += f"\n#### 🔴 Bottom {len(bottom_symbols)} Shorts\n"
    md += "| Rank | Symbol | Score | Key Drivers |\n"
    md += "| :--- | :--- | :--- | :--- |\n"

    for i, entry in enumerate(bottom_symbols):
        sym = entry['symbol']
        rank = entry['predicted_rank']
        d = bottom_drv.get(sym, {})
        driver_str = ", ".join([f"`{k}`" for k in list(d.keys())[:3]]) if d else "—"
        md += f"| {i+1} | **{sym}** | {rank:.4f} | {driver_str} |\n"

    md += "\n---\n\n"
    return md

def generate_report(multi_results: dict) -> str:
    """
    Generate the Weekly Intelligence Report for multiple timeframes.

    Parameters
    ----------
    multi_results : dict
        Dict mapping timeframe strings ('15m', '1h', etc.) to results dictionaries.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M")

    md = f"# Weekly Multi-Timeframe Intelligence Report\n"
    md += f"*Generated: {now_str} UTC*\n\n"

    # --- Table of Contents ---
    md += "## 📌 Summary of Metrics\n\n"
    md += "| Timeframe | Return | Sharpe | Win Rate | Spearman ρ |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for tf, res in multi_results.items():
        sim = res.get('simulation_results')
        meta = res.get('model_meta', {})
        ret = f"{sim['total_return']:+.2%}" if sim else "N/A"
        sha = f"{sim['sharpe']:.2f} ({sim.get('annualized_sharpe', 0.0):.2f} Ann.)" if sim else "N/A"
        win = f"{sim['win_rate']:.1%}" if sim else "N/A"
        rho = f"{sim.get('oos_spearman', 0):.4f}" if sim else "N/A"
        md += f"| **{tf}** | {ret} | {sha} | {win} | {rho} |\n"
    
    md += "\n"

    # --- Timeframe Sections ---
    for tf, res in multi_results.items():
        md += build_report_section(tf, res)

    # --- Section 6: AI Executive Verdict (Using the strongest timeframe as context) ---
    md += "## 🤖 AI Executive Verdict\n\n"
    
    # Heuristic: use the timeframe with the highest Spearman for the LLM context
    best_tf = max(multi_results.keys(), key=lambda k: multi_results[k].get('model_meta', {}).get('validation_spearman', 0))
    best_res = multi_results[best_tf]
    
    llm_context = build_llm_context(
        best_res.get('simulation_results'),
        best_res.get('feature_importance', {}),
        best_res.get('per_asset_drivers', {}),
        best_res.get('model_meta', {})
    )
    # Add a note about multiple timeframes to the context
    llm_context += f"\nNote: This verdict is primarily based on the {best_tf} timeframe, which showed the highest predictive accuracy."
    
    verdict = get_llm_verdict(llm_context)
    md += f"{verdict}\n"

    # --- Footer ---
    md += """
---
> **Note**: This report integrates results from multiple LightGBM Cross-Sectional models.
> OOS metrics use last week's model for each timeframe. 
> Rankings reflect predictions for the upcoming week.
"""

    # --- Save ---
    latest_path = os.path.join(REPORT_DIR, "latest_multi_tf_report.md")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(md)

    archive_path = os.path.join(REPORT_DIR, f"multi_tf_report_{timestamp_file}.md")
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"📄 Aggregated report saved to {latest_path}")
    return latest_path

if __name__ == "__main__":
    print("⚠️ Use 'python master.py report' to generate a report with full cycle data.")
