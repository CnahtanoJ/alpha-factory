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
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.json')


def _bars_per_day(timeframe: str) -> int:
    """Return the number of bars in one day for a given timeframe string."""
    tf_map = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6}
    return tf_map.get(timeframe, 96)  # Default to 15m


def get_feature_names():
    """Returns the canonical feature list used by the LightGBM model."""
    continuous_features = [
        'rank_rsi', 'rank_macd', 'rank_volatility_20', 'rank_basis_pct',
        'rank_oi_usd', 'rank_funding_rate',
        'rank_sum_toptrader_long_short_ratio', 'rank_corr_to_index',
        'rank_oi_delta_4', 'rank_funding_delta_4', 'rank_taker_buy_sell_ratio',
        'rank_distance_from_ema_50', 'rank_volatility_zscore', 'rank_volume_zscore', 'rank_relative_strength_btc',
        'rank_cvd_slope_5', 'rank_price_cvd_divergence'
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
        """Find which ranked features or strategy signals are extreme for these assets."""
        driver_features = [f for f in feature_names if f.startswith('rank_') or f.startswith('sig_')]
        drivers = {}
        for _, row in assets_df.iterrows():
            symbol = row['symbol']
            extreme = {}
            for feat in driver_features:
                val = row.get(feat, 0.5)
                if feat.startswith('sig_') and val != 0:
                    extreme[feat] = f"Active Signal ({val})"
                elif feat.startswith('rank_'):
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
    top_n=3,
    bottom_n=3,
    optimize=False,
    n_trials=50,
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
    # ─── WALK-FORWARD SPLIT ───
    # Hold out the last N weeks from training so OOS is ALWAYS genuinely unseen.
    latest_ts = mega_df['timestamp'].max()
    oos_cutoff = latest_ts - pd.Timedelta(weeks=dry_run_weeks)
    train_df = mega_df[mega_df['timestamp'] < oos_cutoff].copy()
    oos_df = mega_df[mega_df['timestamp'] >= oos_cutoff].copy()
    
    logger.info(f"📐 Walk-Forward Split: Train={len(train_df):,} rows (before {oos_cutoff}) | OOS={len(oos_df):,} rows (after)")

    # =========================================================
    # STEP 1: OOS SIMULATION WITH PREVIOUS MODEL
    # =========================================================
    previous_model, prev_features = load_previous_model()
    simulation_results = None
    if previous_model is not None:
        logger.info("🔬 WEEKLY CYCLE: Running OOS Simulation with PREVIOUS model...")

        from backtester.dry_run_simulator import simulate_portfolio

        if len(oos_df) > 0:
            # Predict with the OLD model on truly unseen data
            # Handle feature schema changes gracefully
            missing_features = [f for f in prev_features if f not in oos_df.columns]
            if missing_features:
                logger.warning(f"⚠️ Missing features for OOS simulation (schema changed?): {missing_features}. Filling with 0.5 (median rank).")
                for f in missing_features:
                    oos_df[f] = 0.5
                    
            X_oos = oos_df[prev_features]
            oos_predictions = previous_model.predict(X_oos)

            simulation_results = simulate_portfolio(
                oos_df, oos_predictions,
                top_n=top_n, bottom_n=bottom_n,
                rebalance_freq=_bars_per_day(timeframe),
                timeframe=timeframe,
                weighting_mode='risk_parity'
            )

            from backtester.dry_run_simulator import format_simulation_summary
            logger.info(f"\n{format_simulation_summary(simulation_results)}")
        else:
            logger.warning("⚠️ Not enough OOS data for simulation.")
    else:
        logger.info("🆕 First run — no previous model. OOS simulation skipped.")

    # =========================================================
    # STEP 2: TRAIN NEW MODEL (on data EXCLUDING the OOS window)
    # =========================================================
    logger.info(f"🧠 WEEKLY CYCLE: Training NEW LightGBM model on {len(train_df):,} rows...")

    best_params = None
    if optimize:
        logger.info(f"🚀 Manual Optimization requested. Triggering Optuna HPO ({n_trials} trials)...")
        from analytics.cross_sectional import prepare_training_data, optimize_lgbm_hyperparameters
        X_train, y_train, X_val, y_val, features = prepare_training_data(train_df)
        best_params = optimize_lgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_trials)
        logger.info("✅ Optimization complete. Training model with best parameters...")

    model, features = train_cross_sectional_lgbm(train_df, optimized_params=best_params)

    # ─── ELITE GATEKEEPER ───
    # We load the meta to check the Spearman Correlation
    model_meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            model_meta = json.load(f)
            
    spearman_corr = model_meta.get('validation_spearman_correlation', 0)
    
    if spearman_corr < 0.05:
        logger.warning(f"🚨 GATEKEEPER REJECTION: Spearman={spearman_corr:.4f} is too low.")
        logger.info("🔄 Triggering Optuna HPO to find a more robust model...")
        
        from analytics.cross_sectional import prepare_training_data, optimize_lgbm_hyperparameters
        X_train, y_train, X_val, y_val, features = prepare_training_data(train_df)
        best_params = optimize_lgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30)
        
        # Re-train with optimized parameters
        model, features = train_cross_sectional_lgbm(train_df, optimized_params=best_params)
        
        # Reload meta after re-train
        with open(META_PATH, 'r') as f:
            model_meta = json.load(f)
        spearman_corr = model_meta.get('validation_spearman_correlation', 0)
        
        if spearman_corr < 0.05:
            logger.error(f"❌ OPTIMIZATION FAILED: Spearman={spearman_corr:.4f} still below threshold.")
            logger.error("🛑 CRITICAL: Deployment aborted to protect capital.")
            return {'status': 'FAIL', 'reason': 'Spearman correlation below threshold even after HPO.'}
        else:
            logger.info(f"✅ OPTIMIZATION SUCCESS: New Spearman={spearman_corr:.4f}")

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
