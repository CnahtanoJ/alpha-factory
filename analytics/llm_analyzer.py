import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_llm_verdict(market_data_str: str) -> str:
    """
    Calls OpenRouter with a strictly anti-sycophantic system prompt to generate a Hedge Fund Executive Summary.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "⚠️ OPENROUTER_API_KEY not found in environment variables. Add it to .env to enable the AI Verdict."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    system_prompt = """You are a highly cynical, extremely experienced Quantitative Hedge Fund Risk Manager.
Your job is to read raw mathematical data (ML probabilities, seasonality) and summarize the current market regime in exactly 2 paragraphs.
CRITICAL INSTRUCTION 1: You are strictly ANTI-SYCOPHANT. Do not cheerlead. Actively hunt for warning signs (e.g., low volume, mixed signals, over-extension). If the data is weak or flat, say so ruthlessly.
CRITICAL INSTRUCTION 2: Contextualize the data within current macro narratives. If you notice structural bullishness or bearishness, hypothesize if it's driven by external forces (e.g. rate cuts, manipulation, structural bull season, war). Factor the timeframe into your verdict.
Write a 2-paragraph 'Executive Verdict' that starts directly with your analysis. No fluff."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the raw data from the Pipeline B scan:\n{market_data_str}\n\nWhat is your verdict?"}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ LLM Analysis Failed: {str(e)}"
