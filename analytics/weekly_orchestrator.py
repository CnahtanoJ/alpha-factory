"""
Weekly Orchestrator — The Single Source of Truth for ML Intelligence.

Trains the XGBoost model ONCE per cycle, then feeds the results into:
  - Pipeline A (Strategist): ai_probs + accuracy used for institutional scoring
  - Pipeline B (Report):     ai_probs + accuracy + grid search results for the intelligence report

Usage:
  As a Lambda handler:  weekly_orchestrator.handler(event, context)
  As a CLI:             python -m analytics.weekly_orchestrator
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.analytics import train_global_xgboost, get_latest_probabilities, calculate_seasonality, get_db_timerange
from backtester.build_bot_blueprint import strategist_handler
from bot.config import AWS_BUCKET
from bot.utils import S3Interface
import json

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def run_weekly_cycle(market='futures', tune_hyperparams=False, force_train=False, skip_grid_search=False):
    """
    The master orchestrator:
      1. Train Ensemble Scout once
      2. Get probabilities once
      3. Feed into Pipeline A (Strategist) and/or Pipeline B (Report)
    """
    # =========================================================
    # STEP 1: TRAIN THE AI SCOUT (Once per cycle)
    # =========================================================
    logger.info(f"🧠 WEEKLY CYCLE: Loading/Training Global Scout Ensemble for {market.upper()}...")
    ensemble, xgb_features, ai_accuracy = train_global_xgboost(market=market, tune_hyperparams=tune_hyperparams, force_train=force_train)

    ai_probs = {}
    if ensemble:
        logger.info("🔍 WEEKLY CYCLE: Generating ensemble probabilities...")
        ai_probs = get_latest_probabilities(ensemble, xgb_features, market=market)
        logger.info(f"✅ AI Scout ready: {len(ai_probs)} symbols scored | Ensemble Accuracy: {ai_accuracy:.2%}")
        
        # 🏆 Save Probabilities (Dual Save: S3 + Local)
        prob_file = "latest_ai_probabilities.json"
        try:
            s3 = S3Interface(AWS_BUCKET)
            s3.save_json(prob_file, ai_probs)
            with open(prob_file, 'w') as f:
                json.dump(ai_probs, f, indent=4)
            logger.info(f"💾 Saved AI probabilities to {prob_file} (Local + S3)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save probabilities: {e}")
            
    else:
        logger.warning("⚠️ AI SCOUT FAILED: No DB data. All pipelines proceed without ML boost.")
        ai_accuracy = 0.0

    # =========================================================
    # STEP 2: PIPELINE A — The Strategist (Grid Search)
    # =========================================================
    pipeline_a_result = "Skipped"
    if not skip_grid_search:
        logger.info("🚀 WEEKLY CYCLE: Starting Pipeline A (Strategist)...")
        pipeline_a_result = strategist_handler(None, None, ai_probs=ai_probs, ai_accuracy=ai_accuracy)
        logger.info(f"✅ Pipeline A complete: {pipeline_a_result}")

    # =========================================================
    # STEP 3: PIPELINE B — Weekly Intelligence Report
    # =========================================================
    logger.info("📊 WEEKLY CYCLE: Starting Pipeline B (Report)...")
    # This might be called again if generate_report calls this function
    # but the generate_report CLI entry point uses this safely.
    return {
        'probs': ai_probs,
        'accuracy': ai_accuracy,
        'pipeline_a_result': pipeline_a_result
    }


def handler(event, context):
    """AWS Lambda entry point."""
    return run_weekly_cycle()


if __name__ == '__main__':
    run_weekly_cycle()
