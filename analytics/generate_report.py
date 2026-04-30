"""
Weekly Intelligence Report Generator.

Produces a comprehensive Markdown report from:
  - OOS Dry Run Simulation results (Sharpe, PF, Win Rate)
  - LightGBM Feature Importance (model's top drivers)
  - Top 10 / Bottom 10 asset rankings with per-asset feature attribution
  - AI Executive Verdict from OpenRouter LLM

This replaces the legacy grid-search-based report. No more elite_squad.json
or all_grid_results.json — the report is driven entirely by the LightGBM
cross-sectional ranking model.
"""
import os
from datetime import datetime

from analytics.llm_analyzer import get_llm_verdict, build_llm_context

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')


def generate_report(cycle_results: dict) -> str:
    """
    Generate the Weekly Intelligence Report from the orchestrator's output.

    Parameters
    ----------
    cycle_results : dict
        Output from weekly_orchestrator.run_weekly_cycle(), containing:
        - simulation_results
        - feature_importance
        - per_asset_drivers
        - model_meta
        - top_n, bottom_n

    Returns
    -------
    str
        Path to the generated report file.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    sim = cycle_results.get('simulation_results')
    feat_imp = cycle_results.get('feature_importance', {})
    drivers = cycle_results.get('per_asset_drivers', {})
    model_meta = cycle_results.get('model_meta', {})
    top_n = cycle_results.get('top_n', 10)
    bottom_n = cycle_results.get('bottom_n', 10)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M")

    # ─── Build Report ───
    md = f"""# Weekly Market Intelligence Report
*Generated: {now_str} UTC*
*Model: LightGBM Cross-Sectional Ranking Engine*

"""

    # ─── Section 1: Model Health ───
    md += "## 📊 Model Training Summary\n\n"
    rmse = model_meta.get('validation_rmse', 'N/A')
    spearman = model_meta.get('validation_spearman_correlation', 'N/A')
    p_val = model_meta.get('spearman_p_value', 'N/A')

    if isinstance(spearman, float):
        if spearman > 0.10:
            health = "🟢 Strong"
        elif spearman > 0.05:
            health = "🟡 Moderate"
        else:
            health = "🔴 Weak"
        md += f"> Model Health: {health} | Spearman ρ = **{spearman:.4f}** (p={p_val:.4f}) | RMSE = {rmse:.4f}\n\n"
    else:
        md += f"> Model Health: Unknown | RMSE = {rmse} | Spearman = {spearman}\n\n"

    # ─── Section 2: OOS Simulation Results ───
    md += "## 🔬 Out-of-Sample Dry Run\n"
    md += "*These results use LAST week's model on THIS week's data — no lookahead bias.*\n\n"

    if sim and sim.get('n_rebalances', 0) > 0:
        md += "| Metric | Value |\n"
        md += "| :--- | :--- |\n"
        md += f"| **Total Return** | {sim['total_return']:+.2%} |\n"
        md += f"| **Sharpe Ratio** | {sim['sharpe']:.2f} |\n"
        md += f"| **Profit Factor** | {sim['profit_factor']:.2f} |\n"
        md += f"| **Win Rate** | {sim['win_rate']:.1%} |\n"
        md += f"| **Max Drawdown** | {sim['max_drawdown']:.2%} |\n"
        md += f"| **Rebalances** | {sim['n_rebalances']} |\n"
        md += f"| **Avg Daily Return** | {sim['avg_daily_return']:+.4%} |\n\n"
    else:
        md += "> ⚠️ No OOS simulation available. This is the first training run.\n\n"

    # ─── Section 2.5: Robustness (Monte Carlo) ───
    if sim and 'mc_stats' in sim:
        mc = sim['mc_stats']
        md += "### 🛡️ Robustness Analysis (Monte Carlo)\n"
        md += f"*Bootstrapped over {mc['sims']} randomized return sequences.*\n\n"
        md += f"- **Probability of Profit**: {mc['prob_profit']:.1%}\n"
        md += f"- **95% CI Lower Bound**: {mc['ci_lower']:+.2%}\n"
        md += f"- **95% CI Upper Bound**: {mc['ci_upper']:+.2%}\n"
        
        if mc['prob_profit'] > 0.90:
            verdict = "✅ **High Confidence**: Model edge is likely structural."
        elif mc['prob_profit'] > 0.70:
            verdict = "⚠️ **Moderate Confidence**: Model edge shows some sequence sensitivity."
        else:
            verdict = "🚨 **Low Confidence**: Model edge may be a chronological fluke."
        md += f"\n> **Robustness Verdict**: {verdict}\n\n"

    # ─── Section 3: Top 10 Longs ───
    top_symbols = drivers.get('top_symbols', [])
    top_drv = drivers.get('top_drivers', {})

    md += f"## 🟢 Top {top_n} Longs (Highest Predicted Rank)\n\n"
    md += "| Rank | Symbol | Predicted Score | Key Drivers |\n"
    md += "| :--- | :--- | :--- | :--- |\n"

    for i, entry in enumerate(top_symbols):
        sym = entry['symbol']
        rank = entry['predicted_rank']
        d = top_drv.get(sym, {})
        driver_str = ", ".join([f"`{k}`" for k in list(d.keys())[:3]]) if d else "—"
        md += f"| {i+1} | **{sym}** | {rank:.4f} | {driver_str} |\n"

    # ─── Section 4: Bottom 10 Shorts ───
    bottom_symbols = drivers.get('bottom_symbols', [])
    bottom_drv = drivers.get('bottom_drivers', {})

    md += f"\n## 🔴 Bottom {bottom_n} Shorts (Lowest Predicted Rank)\n\n"
    md += "| Rank | Symbol | Predicted Score | Key Drivers |\n"
    md += "| :--- | :--- | :--- | :--- |\n"

    for i, entry in enumerate(bottom_symbols):
        sym = entry['symbol']
        rank = entry['predicted_rank']
        d = bottom_drv.get(sym, {})
        driver_str = ", ".join([f"`{k}`" for k in list(d.keys())[:3]]) if d else "—"
        md += f"| {i+1} | **{sym}** | {rank:.4f} | {driver_str} |\n"

    # ─── Section 5: Feature Importance ───
    md += "\n## 🧠 Model Feature Importance (Top 10 by Gain)\n\n"
    md += "| Feature | Importance (%) |\n"
    md += "| :--- | :--- |\n"

    for feat, imp in list(feat_imp.items())[:10]:
        bar = "█" * int(imp / 2)
        md += f"| `{feat}` | {imp:.1f}% {bar} |\n"

    # ─── Section 6: AI Executive Verdict ───
    md += "\n## 🤖 AI Executive Verdict\n\n"

    llm_context = build_llm_context(sim, feat_imp, drivers, model_meta)
    verdict = get_llm_verdict(llm_context)
    md += f"{verdict}\n"

    # ─── Footer ───
    md += """
---
> **Note**: This report is generated from the Alpha Factory's LightGBM Cross-Sectional
> Ranking Engine. OOS metrics use the previous week's model on current data.
> The Top/Bottom rankings reflect the NEW model's predictions for the upcoming week.
"""

    # ─── Save ───
    # Save as latest
    latest_path = os.path.join(REPORT_DIR, "latest_market_report.md")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(md)

    # Save timestamped archive
    archive_path = os.path.join(REPORT_DIR, f"report_{timestamp_file}.md")
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"📄 Report saved to {latest_path}")
    print(f"📄 Archive saved to {archive_path}")

    return latest_path


if __name__ == "__main__":
    # Standalone test — requires a full cycle to have run first
    print("⚠️ Use 'python master.py report' to generate a report with full cycle data.")
