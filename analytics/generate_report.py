"""
Global Intelligence Report Generator.

Produces a comprehensive Markdown report from the Alpha Factory's
grid search results and Elite Squad data.

Data sources:
  - elite_squad.json        — Top 20 ranked strategies (for the leaderboard table)
  - all_grid_results.json   — ALL passing strategies (for distribution analysis)
"""
import json
import os
from datetime import datetime


SQUAD_FILE = "elite_squad.json"
ALL_RESULTS_FILE = "all_grid_results.json"


def generate_report(ai_probs=None, ai_accuracy=0.0):
    """
    Backward-compatible entry point called by master.py cmd_report / cmd_full.
    Reads the Elite Squad + full grid results and produces the weekly intelligence report.
    Returns the path to the generated report file.
    """
    reporter = IntelligenceReporter()
    reporter.generate(ai_probs=ai_probs, ai_accuracy=ai_accuracy)
    return reporter.report_path


class IntelligenceReporter:
    def __init__(self):
        self.report_path = "weekly_intelligence_report.md"

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def generate(self, ai_probs=None, ai_accuracy=0.0):
        """Generates a professional Markdown report of the Alpha Factory results."""
        squad = self._load_json(SQUAD_FILE)
        all_results = self._load_json(ALL_RESULTS_FILE)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        md = f"""# Global Market Intelligence Report
*Generated At: {now_str}*

"""
        if ai_accuracy > 0:
            md += f"> AI Ensemble Accuracy: **{ai_accuracy:.2%}**\n\n"

        # --- Summary ---
        md += f"> **Grid Search**: {len(all_results)} strategies passed filters across all symbols.\n"
        md += f"> **Elite Squad**: Top {len(squad)} ranked by Pure Math Score.\n\n"

        # --- Section 1: Elite Squad Leaderboard ---
        md += f"""## The Elite Squad (Top {len(squad)} Leaders)
These represent the highest historical math performers across the entire database.

| Rank | Symbol | TF | Strategy | Alpha | Sharpe | PF | Pure Math Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for i, r in enumerate(squad):
            md += (
                f"| {i+1} "
                f"| {r.get('target_coin', 'N/A')} "
                f"| {r.get('timeframe', '-')} "
                f"| {r.get('strategy', '-')} "
                f"| {r.get('alpha', 0):.2%} "
                f"| {r.get('sharpe', 0):.2f} "
                f"| {r.get('profit_factor', 0):.2f} "
                f"| **{r.get('score', 0):.4f}** |\n"
            )

        # --- Section 2: Strategy Distribution (from ALL passing results) ---
        source = all_results if all_results else squad
        source_label = f"All {len(source)} Passing Strategies" if all_results else "Elite Squad"

        if source:
            strat_counts = {}
            tf_counts = {}
            coin_counts = {}
            for r in source:
                s = r.get('strategy', 'Unknown')
                t = r.get('timeframe', 'Unknown')
                c = r.get('target_coin', 'Unknown')
                strat_counts[s] = strat_counts.get(s, 0) + 1
                tf_counts[t] = tf_counts.get(t, 0) + 1
                coin_counts[c] = coin_counts.get(c, 0) + 1

            md += f"\n## Strategy Distribution ({source_label})\n"
            md += "| Strategy | Count | Share |\n| :--- | :--- | :--- |\n"
            total = len(source)
            for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
                md += f"| {s} | {c} | {c/total:.0%} |\n"

            md += f"\n## Timeframe Distribution ({source_label})\n"
            md += "| Timeframe | Count | Share |\n| :--- | :--- | :--- |\n"
            for t, c in sorted(tf_counts.items(), key=lambda x: -x[1]):
                md += f"| {t} | {c} | {c/total:.0%} |\n"

            md += f"\n## Top Tokens by Strategy Count ({source_label})\n"
            md += "| Token | Passing Strategies | Best Score |\n| :--- | :--- | :--- |\n"
            # For each top coin, find its best score
            for coin, count in sorted(coin_counts.items(), key=lambda x: -x[1])[:20]:
                best_score = max(
                    (r.get('score', 0) for r in source if r.get('target_coin') == coin),
                    default=0
                )
                md += f"| {coin} | {count} | {best_score:.4f} |\n"

        # --- Section 3: AI Conviction (if available) ---
        if ai_probs:
            sorted_probs = sorted(
                ai_probs.items(),
                key=lambda x: x[1].get('bull', 0) + x[1].get('bear', 0),
                reverse=True
            )
            md += "\n## AI Movement Conviction (Top 15)\n"
            md += "| Symbol | P(Bull) | P(Bear) | P(Flat) | Movement |\n"
            md += "| :--- | :--- | :--- | :--- | :--- |\n"
            for sym, p in sorted_probs[:15]:
                movement = p.get('bull', 0) + p.get('bear', 0)
                md += (
                    f"| {sym} "
                    f"| {p.get('bull', 0):.2%} "
                    f"| {p.get('bear', 0):.2%} "
                    f"| {p.get('flat', 0):.2%} "
                    f"| **{movement:.2%}** |\n"
                )

        md += """
---
> **Note**: This report covers the **entire database** (unfiltered).
> To see which candidates are tradable on Hyperliquid, run `python master.py scout`.
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"Report saved to {self.report_path}")
        return self.report_path


if __name__ == "__main__":
    generate_report()
