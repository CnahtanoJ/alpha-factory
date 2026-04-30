"""
Weekly Orchestrator — Walk-Forward Out-of-Sample Pipeline.

The single source of truth for the weekly intelligence cycle.

Sequence (strict order to prevent lookahead bias):
  1. Build Mega-DataFrame from DB
  2. Load PREVIOUS week's LightGBM model (if exists)
  3. Run OOS Simulation on the most recent week using the old model
  4. Train NEW LightGBM model on the full dataset
  5. Extract Feature Importance for Top 10 / Bottom 10
  6. Generate Intelligence Report with AI Verdict

Usage:
  As a Lambda handler:  weekly_orchestrator.handler(event, context)
  As a CLI:             python -m analytics.weekly_orchestrator
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
from analytics.cross_sectional import build_mega_dataframe, train_cross_sectional_lgbm

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'cross_sectional_lgbm.txt')
META_PATH = os.path.join(MODEL_DIR, 'cross_sectional_lgbm_meta.json')


def get_feature_names():
    """Returns the canonical feature list used by the LightGBM model."""
    continuous_features = [
        'rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct',
        'rank_oi_usd', 'rank_funding_rate',
        'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index'
    ]
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    return continuous_features, time_features


def load_previous_model():
    """
    Load the previously trained LightGBM model.
    Returns (model, features) or (None, None) if no cached model exists.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠️ No cached LightGBM model found. First run — OOS simulation will be skipped.")
        return None, None

    logger.info(f"📦 Loading previous LightGBM model from {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)
    features = model.feature_name()
    return model, features


def extract_feature_importance(model, feature_names: list) -> dict:
    """
    Extract LightGBM native feature importance (gain-based).
    Returns a sorted dict of {feature: importance_pct}.
    """
    importance = model.feature_importance(importance_type='gain')
    total = importance.sum()
    if total == 0:
        return {}

    importance_pct = importance / total * 100
    importance_dict = dict(zip(feature_names, importance_pct))

    # Sort by importance descending
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def extract_per_asset_drivers(
    model, mega_df, predictions, feature_names, top_n=10, bottom_n=10
) -> dict:
    """
    For the Top N and Bottom N predicted assets at the LATEST timestamp,
    compute which features contributed most to their ranking.

    Uses a simple approach: for each asset in the basket, look at which
    ranked features are extreme (>90th or <10th percentile) to explain
    the model's conviction.
    """
    df = mega_df.copy()
    df['predicted_rank'] = predictions

    # Get the latest timestamp
    latest_ts = df['timestamp'].max()
    snapshot = df[df['timestamp'] == latest_ts].copy()

    if len(snapshot) < (top_n + bottom_n):
        return {'top_drivers': {}, 'bottom_drivers': {}}

    snapshot = snapshot.sort_values('predicted_rank', ascending=False)
    top_assets = snapshot.head(top_n)
    bottom_assets = snapshot.tail(bottom_n)

    def get_extreme_features(assets_df):
        """Find which ranked features are extreme for these assets."""
        rank_features = [f for f in feature_names if f.startswith('rank_')]
        drivers = {}
        for _, row in assets_df.iterrows():
            symbol = row['symbol']
            extreme = {}
            for feat in rank_features:
                val = row.get(feat, 0.5)
                if val > 0.85:
                    extreme[feat] = f"Very High ({val:.2f})"
                elif val < 0.15:
                    extreme[feat] = f"Very Low ({val:.2f})"
            drivers[symbol] = extreme
        return drivers

    return {
        'top_drivers': get_extreme_features(top_assets),
        'bottom_drivers': get_extreme_features(bottom_assets),
        'top_symbols': top_assets[['symbol', 'predicted_rank']].to_dict('records'),
        'bottom_symbols': bottom_assets[['symbol', 'predicted_rank']].to_dict('records'),
    }


def run_weekly_cycle(
    market='futures',
    timeframe='15m',
    force_train=False,
    dry_run_weeks=4,
    top_n=10,
    bottom_n=10,
):
    """
    The master orchestrator for the weekly intelligence cycle.

    Parameters
    ----------
    market : str
        Market to train on (only 'futures' supported).
    force_train : bool
        If True, always retrain even if cached model exists.
    dry_run_weeks : int
        Number of trailing weeks to simulate in OOS mode.
    top_n : int
        Number of top assets for the long basket.
    bottom_n : int
        Number of bottom assets for the short basket.

    Returns
    -------
    dict with keys:
        'simulation_results', 'feature_importance', 'per_asset_drivers',
        'model_meta', 'top_n', 'bottom_n'
    """
    # =========================================================
    # STEP 0: BUILD MEGA-DATAFRAME
    # =========================================================
    logger.info("📊 WEEKLY CYCLE: Building Mega-DataFrame...")
    mega_df = build_mega_dataframe(timeframe=timeframe)

    if mega_df.empty:
        logger.error("❌ WEEKLY CYCLE: No data available. Run 'ingest' first.")
        return {
            'simulation_results': None,
            'feature_importance': {},
            'per_asset_drivers': {},
            'model_meta': {},
            'top_n': top_n,
            'bottom_n': bottom_n,
        }

    logger.info(f"✅ Mega-DataFrame: {len(mega_df):,} rows, {mega_df['symbol'].nunique()} symbols")

    # =========================================================
    # STEP 1: LOAD PREVIOUS MODEL FOR OOS SIMULATION
    # =========================================================
    previous_model, prev_features = load_previous_model()

    simulation_results = None
    if previous_model is not None and not force_train:
        logger.info("🔬 WEEKLY CYCLE: Running OOS Simulation with PREVIOUS model...")

        from backtester.dry_run_simulator import simulate_portfolio

        # Define OOS window: the most recent `dry_run_weeks` weeks
        latest_ts = mega_df['timestamp'].max()
        oos_start = latest_ts - pd.Timedelta(weeks=dry_run_weeks)
        oos_df = mega_df[mega_df['timestamp'] >= oos_start].copy()

        if len(oos_df) > 0:
            # Predict with the OLD model
            X_oos = oos_df[prev_features]
            oos_predictions = previous_model.predict(X_oos)

            simulation_results = simulate_portfolio(
                oos_df, oos_predictions, prev_features,
                top_n=top_n, bottom_n=bottom_n,
                timeframe=timeframe,
                weighting_mode='risk_parity'
            )

            from backtester.dry_run_simulator import format_simulation_summary
            logger.info(f"\n{format_simulation_summary(simulation_results)}")
        else:
            logger.warning("⚠️ Not enough OOS data for simulation.")
    else:
        if force_train:
            logger.info("🔄 Force-train requested. Skipping OOS simulation with old model.")
        else:
            logger.info("🆕 First run — no previous model. OOS simulation skipped.")

    # =========================================================
    # STEP 2: TRAIN NEW LIGHTGBM MODEL
    # =========================================================
    logger.info("🧠 WEEKLY CYCLE: Training NEW LightGBM model...")

    model, features = train_cross_sectional_lgbm(mega_df)

    # =========================================================
    # STEP 3: EXTRACT FEATURE IMPORTANCE
    # =========================================================
    logger.info("🔍 WEEKLY CYCLE: Extracting feature importance...")
    feature_importance = extract_feature_importance(model, features)

    # Log top 5 features
    for feat, imp in list(feature_importance.items())[:5]:
        logger.info(f"   {feat}: {imp:.1f}%")

    # =========================================================
    # STEP 4: PER-ASSET FEATURE ATTRIBUTION
    # =========================================================
    logger.info("🎯 WEEKLY CYCLE: Computing per-asset feature drivers...")

    # Predict with the NEW model on the latest data for the report
    X_all = mega_df[features]
    all_predictions = model.predict(X_all)

    per_asset_drivers = extract_per_asset_drivers(
        model, mega_df, all_predictions, features,
        top_n=top_n, bottom_n=bottom_n
    )

    # Load model meta
    model_meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            model_meta = json.load(f)

    logger.info("✅ WEEKLY CYCLE COMPLETE!")
    return {
        'simulation_results': simulation_results,
        'feature_importance': feature_importance,
        'per_asset_drivers': per_asset_drivers,
        'model_meta': model_meta,
        'top_n': top_n,
        'bottom_n': bottom_n,
    }


def handler(event, context):
    """AWS Lambda entry point."""
    return run_weekly_cycle()


if __name__ == '__main__':
    run_weekly_cycle()
