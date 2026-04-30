"""
LLM Analyzer — OpenRouter AI Verdict with Glass-Box Context.

Feeds the LLM not just numbers, but the model's feature importance
and per-asset drivers so it can write an intelligent, deeply technical
intelligence report instead of hallucinating narratives.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are a highly cynical, extremely experienced Quantitative Hedge Fund Risk Manager who specializes in crypto derivatives.

You are reviewing the weekly output of an autonomous AI trading system called "Alpha Factory". This system uses a LightGBM cross-sectional ranking model to rank ~100 crypto futures assets and simulate a market-neutral portfolio (Top 10 Longs, Bottom 10 Shorts).

You will receive:
1. OUT-OF-SAMPLE (OOS) simulation metrics: Sharpe, Profit Factor, Win Rate, Total Return, Max Drawdown. These were generated using LAST week's model on THIS week's data — they are genuinely predictive, not overfit.
2. FEATURE IMPORTANCE: The model's top drivers ranked by gain. This tells you what market dynamics the model is exploiting.
3. PER-ASSET DRIVERS: For the Top 10 and Bottom 10 assets, the extreme features that explain why the model ranked them where it did.
4. MODEL METADATA: Validation RMSE and Spearman correlation from training.
5. ROBUSTNESS ANALYSIS (Monte Carlo): Probability of Profit and 95% Confidence Intervals from 2,500+ bootstrap simulations. This tells you if the result is a "chronological fluke."

YOUR JOB:
Write a 3-paragraph Executive Intelligence Verdict:

Paragraph 1 — REGIME ANALYSIS: Based on the feature importance, identify what market regime the model is detecting (e.g., momentum regime, mean-reversion, funding arbitrage, correlation breakdown). Be specific about which features dominate and what that implies.

Paragraph 2 — CONVICTION & ROBUSTNESS: Evaluate the OOS simulation metrics and the Monte Carlo results. Is the Sharpe realistic? More importantly, does the Monte Carlo CI Lower Bound stay positive? If Probability of Profit is < 80%, be extremely skeptical regardless of the Sharpe.

Paragraph 3 — RISK WARNINGS: Identify the top 3 risks you see. These could include: concentration risk (same sector in top/bottom), regime change fragility, features that might be stale, macro catalysts that could invalidate the model's thesis.

CRITICAL RULES:
- Do NOT cheerlead. If the numbers are mediocre, say so.
- Do NOT hallucinate specific price targets or macro events you don't have data for.
- DO reference specific feature names and assets from the data.
- Write in direct, institutional prose. No emoji, no hype."""


def get_llm_verdict(context_str: str) -> str:
    """
    Calls OpenRouter with Glass-Box context to generate an institutional AI verdict.

    Parameters
    ----------
    context_str : str
        The formatted string containing OOS metrics, feature importance,
        and per-asset drivers.

    Returns
    -------
    str
        The LLM's executive verdict, or an error message.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "⚠️ OPENROUTER_API_KEY not found in environment variables. Add it to .env to enable the AI Verdict."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the full Glass-Box context from this week's Alpha Factory cycle:\n\n{context_str}\n\nDeliver your Executive Intelligence Verdict."}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ LLM Analysis Failed: {str(e)}"


def build_llm_context(
    simulation_results: dict,
    feature_importance: dict,
    per_asset_drivers: dict,
    model_meta: dict,
) -> str:
    """
    Builds the full context string to feed to the LLM.

    This is the "Glass Box" — the LLM sees exactly what the model sees.
    """
    lines = []

    # Section 1: OOS Simulation Metrics
    lines.append("## 1. OUT-OF-SAMPLE SIMULATION METRICS")
    if simulation_results:
        lines.append(f"  Total Return:     {simulation_results['total_return']:+.2%}")
        lines.append(f"  Sharpe Ratio:     {simulation_results['sharpe']:.2f}")
        lines.append(f"  Profit Factor:    {simulation_results['profit_factor']:.2f}")
        lines.append(f"  Win Rate:         {simulation_results['win_rate']:.1%}")
        lines.append(f"  Max Drawdown:     {simulation_results['max_drawdown']:.2%}")
        lines.append(f"  Rebalance Count:  {simulation_results['n_rebalances']}")
        
        if 'mc_stats' in simulation_results:
            mc = simulation_results['mc_stats']
            lines.append(f"  MC Prob. Profit:  {mc['prob_profit']:.1%}")
            lines.append(f"  MC 95% CI Lower:  {mc['ci_lower']:+.2%}")
            lines.append(f"  MC 95% CI Upper:  {mc['ci_upper']:+.2%}")
    else:
        lines.append("  No OOS simulation available (first run or force-train).")

    # Section 2: Feature Importance
    lines.append("\n## 2. MODEL FEATURE IMPORTANCE (by gain)")
    for feat, imp in list(feature_importance.items())[:10]:
        bar = "█" * int(imp / 2)  # Visual bar
        lines.append(f"  {feat:45s} {imp:5.1f}% {bar}")

    # Section 3: Per-Asset Drivers
    lines.append("\n## 3. TOP 10 LONGS — Asset Drivers")
    top_symbols = per_asset_drivers.get('top_symbols', [])
    top_drivers = per_asset_drivers.get('top_drivers', {})
    for entry in top_symbols:
        sym = entry['symbol']
        rank = entry['predicted_rank']
        drivers = top_drivers.get(sym, {})
        driver_str = ", ".join([f"{k}: {v}" for k, v in drivers.items()]) if drivers else "No extreme features"
        lines.append(f"  {sym:15s} (rank: {rank:.4f}) → {driver_str}")

    lines.append("\n## 4. BOTTOM 10 SHORTS — Asset Drivers")
    bottom_symbols = per_asset_drivers.get('bottom_symbols', [])
    bottom_drivers = per_asset_drivers.get('bottom_drivers', {})
    for entry in bottom_symbols:
        sym = entry['symbol']
        rank = entry['predicted_rank']
        drivers = bottom_drivers.get(sym, {})
        driver_str = ", ".join([f"{k}: {v}" for k, v in drivers.items()]) if drivers else "No extreme features"
        lines.append(f"  {sym:15s} (rank: {rank:.4f}) → {driver_str}")

    # Section 4: Model Metadata
    lines.append("\n## 5. MODEL TRAINING METADATA")
    lines.append(f"  Validation RMSE:       {model_meta.get('validation_rmse', 'N/A')}")
    lines.append(f"  Spearman Correlation:  {model_meta.get('validation_spearman_correlation', 'N/A')}")
    lines.append(f"  Spearman p-value:      {model_meta.get('spearman_p_value', 'N/A')}")

    return "\n".join(lines)
