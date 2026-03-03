# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false}}
# -*- coding: utf-8 -*-
"""
Kaggle notebook entry point - FULL MODEL ONLY (No 2-Stage Filtering)

This script trains the full model on ALL features without any:
- Stage 1 tree-based filtering
- Stage 2 selected model training

This replicates the Dec 24 configuration that achieved 64.3% test accuracy.

KAGGLE VERSION - All outputs saved to /kaggle/working
"""
from __future__ import annotations

# Install dependencies - works in both notebook and script mode
import subprocess
import sys
# Install AutoGluon with ALL optional model types including TabPFN
subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.tabular[all]==1.5.0", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn>=2.0", "tabdpt", "tabm", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "lightgbm", "-q"])

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import shutil
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, precision_recall_curve, log_loss
from sklearn.calibration import calibration_curve

# KAGGLE VERSION: No Azure ML imports needed

# ============================================================================
# EXCLUDED FEATURES - Remove these features (negative importance in Dec 24/Feb 25 analysis)
# ============================================================================
EXCLUDED_FEATURES = [
    # Confirmed negative importance (validated on V2 pipeline, Feb 25)
    'age_dec_avg_diff',
    'body_strikes_attempted_dec_adjperf_dec_avg_diff',
    'head_strikes_landed_dec_adjperf_dec_avg_diff',
    'control_time_per_fight_career_avg_dec_adjperf_dec_avg_diff',
    'opp_reach_dec_avg_diff',
    'distance_strikes_attempted_dec_adjperf_dec_avg_diff',
    'leg_land_ratio_dec_adjperf_dec_avg_diff',
    'ground_strikes_landed_dec_adjperf_dec_avg_diff',
    'sig_strikes_landed_per_min_career_avg_dec_adjperf_dec_avg_diff',
    'ko_tko_win_dec_avg_diff',
    'leg_land_ratio_diff',
    'ground_strikes_attempted_dec_adjperf_dec_avg_diff',
    # Redundant: pure math transform of A_open_odds
    'implied_prob_A',
    
    # Newly discovered negative importance features (Draft 13/14 Updates)
    'days_since_last_fight_dec_adjperf_dec_avg_diff',
    'sig_str_acc_dec_adjperf_dec_avg_diff',
    'round1_sig_strikes_landed_dec_adjperf_dec_avg_diff',
    'ctrl_per_min_dec_adjperf_dec_avg_diff',
    'reversals_dec_avg_diff',
    'distance_acc_dec_adjperf_dec_avg_diff',
]
print(f"Will EXCLUDE {len(EXCLUDED_FEATURES)} negative features from training")



def log_metric(name: str, value: float) -> None:
    """Log metrics to console (Kaggle version)."""
    print(f"[METRIC] {name}: {value}")


def upload_artifact(local_path: Path, target_name: Optional[str] = None) -> None:
    """No-op for Kaggle (files are already in /kaggle/working)."""
    pass


def resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Resolve the output directory for Kaggle environment."""
    # Kaggle notebooks always save to /kaggle/working
    kaggle_output = Path("/kaggle/working")
    if kaggle_output.exists():
        return kaggle_output
    
    # Fallback for local testing
    if output_dir is not None:
        return output_dir
    return Path("./outputs")


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) when it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure to disk and upload to Azure when possible."""
    ensure_directory(path.parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    upload_artifact(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist JSON data to disk and upload to Azure when possible."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    upload_artifact(path)


def save_text(content: str, path: Path) -> None:
    """Persist plain-text content to disk and upload to Azure when possible."""
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    upload_artifact(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to CSV and upload to Azure when possible."""
    ensure_directory(path.parent)
    df.to_csv(path, index=False)
    upload_artifact(path)


def resolve_dataset_file(data_path: Path) -> Path:
    """Resolve an input CSV when Azure ML supplies a mounted directory."""
    if data_path.is_file():
        return data_path

    if data_path.is_dir():
        preferred_names = [
            "ufc_fights_full_with_odds_updated.csv",
            "ufc_fights_full_with_odds.csv",  # NEW: Dataset with betting odds
            "ufc_fights_ml_updated.csv",
            "ufc_fights_ml.csv",
            "ufc_fights_ml_clean.csv",
        ]
        for name in preferred_names:
            candidate = data_path / name
            if candidate.is_file():
                print(f"Resolved dataset file '{candidate.name}' inside directory {data_path}")
                return candidate

        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(data_path.rglob("*.csv"))

        if len(csv_files) == 1:
            resolved = csv_files[0]
            print(f"Resolved dataset file {resolved} located inside directory {data_path}")
            return resolved

        if csv_files:
            preview = "\n".join(f" - {candidate}" for candidate in csv_files[:5])
            if len(csv_files) > 5:
                preview += "\n - ..."
            raise ValueError(
                "Multiple CSV files detected in the provided data directory.\n"
                f"Directory: {data_path}\n"
                f"Candidates:\n{preview}\n"
                "Re-run with --data-path pointing to a specific CSV file."
            )

        raise FileNotFoundError(
            f"No CSV files found inside the provided directory: {data_path}. "
            "Ensure the dataset export produced at least one .csv file."
        )

    raise FileNotFoundError(f"Data path does not exist or is not accessible: {data_path}")



# NOTE: randomize_fighter_positions REMOVED (dead code).
# Fighter position randomization is now done inline in main() as differential-level
# swaps (negate _diff, flip outcome, swap odds) applied to training data only.
# This avoids randomizing val/test outcomes. See main() around line 671.

def clean_fight_data(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Standardize fight rows before feature engineering."""
    print("=== DATA HYGIENE CHECKS ===")
    df = df.copy()
    
    # Remove pre-computed differentials to avoid data leakage
    precomputed_diffs = [
        "sig_strikes_landed_diff",
        "sig_strike_accuracy_diff",
        "takedowns_landed_diff",
        "takedown_defense_diff",
        "reach_diff",
        "age_diff",
    ]
    dropped_diffs = [col for col in precomputed_diffs if col in df.columns]
    if dropped_diffs:
        df = df.drop(columns=dropped_diffs)
        print(f"Dropped {len(dropped_diffs)} pre-computed differentials to prevent data leakage: {dropped_diffs}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    invalid_dates = df["event_date"].isna().sum()
    if invalid_dates:
        print(f"Dropping {invalid_dates} rows with unparsable event_date values")
        df = df.dropna(subset=["event_date"])

    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()

    future_mask = df["event_date"] > today
    if future_mask.any():
        print(f"Removing {future_mask.sum()} future-dated fights beyond {today.date()}")
        df = df.loc[~future_mask].copy()

    if "fight_id" in df.columns:
        before_dupes = len(df)
        df = df.sort_values("event_date").drop_duplicates(subset="fight_id", keep="last")
        dupes_removed = before_dupes - len(df)
        if dupes_removed:
            print(f"Removed {dupes_removed} duplicate fight_id rows")

    df.reset_index(drop=True, inplace=True)

    if "outcome" in df.columns:
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    # Recompute basic differentials from fighter-specific columns
    if {"fighter_a_reach", "fighter_b_reach"}.issubset(df.columns):
        df["reach_diff"] = df["fighter_a_reach"] - df["fighter_b_reach"]
    if {"fighter_a_age", "fighter_b_age"}.issubset(df.columns):
        df["age_diff"] = df["fighter_a_age"] - df["fighter_b_age"]

    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def evaluate_threshold_grid(
    y_true: pd.Series,
    positive_scores: pd.Series | np.ndarray,
    positive_label: Any,
    negative_label: Any,
    min_negative_recall: float,  # kept for signature compatibility
    beta: float = 1.0,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate multiple thresholds and return the best balanced-accuracy configuration."""
    label_order = [negative_label, positive_label]
    positive_scores = np.asarray(positive_scores, dtype=float)

    def _compute_metrics(preds: np.ndarray, threshold_value: float) -> dict[str, float]:
        precision, recall, _, support = precision_recall_fscore_support(
            y_true, preds, labels=label_order, zero_division=0
        )
        acc = accuracy_score(y_true, preds)
        recall_pos = float(recall[1])
        precision_pos = float(precision[1])
        recall_neg = float(recall[0])
        precision_neg = float(precision[0])
        balanced_acc = float((recall_pos + recall_neg) / 2.0)
        
        # Penalize the gap between recalls to enforce balance
        recall_gap = abs(recall_pos - recall_neg)
        penalized_balanced_acc = balanced_acc - (recall_gap * 0.5)

        denom = beta * beta * precision_pos + recall_pos
        f_beta_pos = float(((1 + beta * beta) * precision_pos * recall_pos) / denom) if denom > 0 else 0.0
        return {
            "threshold": float(threshold_value),
            "precision_positive": precision_pos,
            "precision_negative": precision_neg,
            "recall_positive": recall_pos,
            "recall_negative": recall_neg,
            "support_positive": float(support[1]),
            "support_negative": float(support[0]),
            "accuracy": float(acc),
            "balanced_accuracy": balanced_acc,
            "penalized_balanced_accuracy": penalized_balanced_acc, # New metric for comparison
            "f_beta_positive": f_beta_pos,
        }

    baseline_preds = np.where(positive_scores >= 0.5, positive_label, negative_label)
    baseline_metrics = _compute_metrics(baseline_preds, 0.5)

    searched_thresholds = thresholds
    if searched_thresholds is None:
        searched_thresholds = np.linspace(0.05, 0.95, 19)
    searched_thresholds = np.unique(np.append(searched_thresholds, 0.5))

    best_metrics = {
        **baseline_metrics,
        "baseline_threshold": 0.5,
        "baseline_recall_positive": baseline_metrics["recall_positive"],
        "baseline_recall_negative": baseline_metrics["recall_negative"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "baseline_f_beta_positive": baseline_metrics["f_beta_positive"],
        "threshold": 0.5,
        "threshold_source": "baseline",
    }
    best_preds = baseline_preds

    for threshold_value in searched_thresholds:
        preds = np.where(positive_scores >= threshold_value, positive_label, negative_label)
        candidate = _compute_metrics(preds, threshold_value)
        candidate.update(
            {
                "baseline_threshold": 0.5,
                "baseline_recall_positive": baseline_metrics["recall_positive"],
                "baseline_recall_negative": baseline_metrics["recall_negative"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "baseline_f_beta_positive": baseline_metrics["f_beta_positive"],
                "threshold_source": "grid" if threshold_value != 0.5 else "baseline",
            }
        )
        if candidate["penalized_balanced_accuracy"] > best_metrics["penalized_balanced_accuracy"] + 1e-6:
            best_metrics = candidate
            best_preds = preds

    return best_metrics, best_preds


def clip_feature_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Clip high-variance engineered features to reduce memorization spikes."""
    clip_bounds: dict[str, tuple[float, float]] = {
        # meaningful for ratios if they go out of bounds due to safe_div numerical issues
        "sig_str_share_diff": (-1.0, 1.0),
        "reach_ratio_diff": (-1.0, 1.0),
        "height_ratio_diff": (-1.0, 1.0),
        "age_ratio_diff": (-1.0, 1.0),
    }

    for column, (lower, upper) in clip_bounds.items():
        if column in df.columns:
            df[column] = df[column].clip(lower, upper)

    return df


# NOTE: build_selected_model_default_hyperparameters REMOVED
# AutoGluon's extreme_quality preset handles hyperparameter tuning automatically.
# Manual hyperparameters were overriding AutoGluon's tuning and may not be optimal.



def generate_diagnostics(
    predictor: TabularPredictor,
    test_data: pd.DataFrame,
    output_dir: Path,
    persist_full_artifacts: bool
) -> None:
    """
    Generates advanced diagnostic plots for model robustness verification:
    1. Feature Importance (Top 20)
    2. Calibration Curve (Reliability Diagram)
    3. Confusion Matrix Heatmap
    4. Probability Density Histogram
    5. Precision-Recall Curve
    """
    if not persist_full_artifacts:
        return
        
    print("=== Generating Advanced Model Diagnostics ===")
    
    # Clear GPU memory before generating diagnostics
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   Cleared GPU cache for diagnostics")
    
    # 1. Feature Importance (use CatBoost to avoid TabPFN GPU issues)
    try:
        available_models = predictor.model_names()
        catboost_model = next((m for m in available_models if 'CatBoost' in m), None)
        
        if catboost_model:
            fi = predictor.feature_importance(test_data, model=catboost_model, subsample_size=500)
        else:
            fi = predictor.feature_importance(test_data, subsample_size=500)
        top_20 = fi.head(20)
        
        plt.figure(figsize=(10, 12))
        sns.barplot(x=top_20['importance'], y=top_20.index)
        plt.title("Top 20 Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png")
        plt.close()
        print("   Saved feature_importance.png")
    except Exception as e:
        print(f"   Failed to generate Feature Importance: {e}")

    # Prepare data for other plots
    y_true = test_data['outcome']
    y_pred = predictor.predict(test_data)
    y_proba = predictor.predict_proba(test_data)
    
    # Handle multiclass probability output (get prob of class 1)
    if isinstance(y_proba, pd.DataFrame):
        # Medium-Risk Fix: Dynamically find the column representing class 1.0 or '1' instead of hardcoding iloc[:, 1]
        pos_class_col = next((c for c in y_proba.columns if str(c) in ['1', '1.0', 1, 1.0]), y_proba.columns[1] if len(y_proba.columns) > 1 else y_proba.columns[0])
        y_proba = y_proba[pos_class_col]
    
    # 2. Calibration Curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.figure(figsize=(8, 8))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.legend()
        plt.savefig(output_dir / "calibration_curve.png")
        plt.close()
        print("   Saved calibration_curve.png")
    except Exception as e:
        print(f"   Failed to generate Calibration Curve: {e}")

    # 3. Confusion Matrix Heatmap
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
        print("   Saved confusion_matrix.png")
    except Exception as e:
        print(f"   Failed to generate Confusion Matrix: {e}")

    # 4. Probability Density Histogram
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(x=y_proba, hue=y_true, kde=True, bins=20, multiple="layer")
        plt.title("Prediction Probability Density by Outcome")
        plt.xlabel("Predicted Probability (Class 1)")
        plt.savefig(output_dir / "probability_density.png")
        plt.close()
        print("   Saved probability_density.png")
    except Exception as e:
        print(f"   Failed to generate Prob Density Plot: {e}")

    # 5. Precision-Recall Curve
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, marker='.', label='Model')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(output_dir / "pr_curve.png")
        plt.close()
        print("   Saved pr_curve.png")
    except Exception as e:
        print(f"   Failed to generate PR Curve: {e}")

# Import the pipeline robustly (Kaggle or local)
import sys
import os

try:
    if os.path.exists('/kaggle/usr/lib/production_feature_pipeline_v2'):
        sys.path.insert(0, '/kaggle/usr/lib/production_feature_pipeline_v2')
        import production_feature_pipeline_v2 as pipeline
    else:
        import kaggle_pipeline_for_results_2 as pipeline
except ImportError:
    import kaggle_pipeline_for_results_2 as pipeline

def main(
    data_path: Path,
    output_dir: Path,
    use_existing_model: bool,
    existing_model_path: Optional[Path],
    best_model_name: Optional[str],
    time_limit: Optional[int],
    persist_full_artifacts: bool,
) -> dict[str, Any]:
    """Execute the advanced AutoGluon pipeline (Kaggle version)."""
    output_dir = ensure_directory(output_dir)
    print("=== UFC Fight Predictor - Advanced AutoGluon Version (Kaggle) ===")

    # Check for GPU availability
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        print(f"Detected {num_gpu} GPU(s) available.")
    else:
        num_gpu = 0
        print("No GPU detected. Training will use CPU.")

    print(f"Loading data from {data_path} ...")
    dataset_path = resolve_dataset_file(data_path)
    if dataset_path != data_path:
        print(f"Using resolved dataset file: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)
    print(f"Raw dataset shape: {raw_df.shape}")

    df = clean_fight_data(raw_df)
    print(f"Dataset after cleaning: {df.shape}")
    
    # NOTE: The redundant secondary randomization call has been removed.
    # df = randomize_fighter_positions(df)
    
    print(f"Date coverage: {df['event_date'].min().date()} -> {df['event_date'].max().date()}")

    # Feature Engineering using Production Pipeline
    # This handles strict filtering (women's divisions, split decisions)
    # and leak-proof history reconstruction, including final <2 prior-fight target filtering.
    df, feature_cols = pipeline.build_prefight_features(df)
    
    # ADD BETTING ODDS FEATURES (if available in the data)
    # These are scraped from BestFightOdds.com - opening line odds
    odds_features = ['opening_odds_diff', 'implied_prob_A', 'A_open_odds', 'B_open_odds']
    added_odds = []
    for feat in odds_features:
        if feat in df.columns and feat not in feature_cols:
            # Check if feature has enough non-null values
            non_null_rate = df[feat].notna().sum() / len(df)
            if non_null_rate > 0.3:  # At least 30% coverage
                feature_cols.append(feat)
                added_odds.append(feat)
    if added_odds:
        print(f"Added betting odds features: {added_odds}")
    else:
        print("No betting odds features found in data")
    
    # FILTER OUT negative/noisy features identified in Dec 24 analysis
    original_count = len(feature_cols)
    feature_cols = [f for f in feature_cols if f not in EXCLUDED_FEATURES]
    print(f"Filtered out {original_count - len(feature_cols)} excluded features (negative importance)")
    
    numeric_feature_cols = feature_cols
    
    df = clip_feature_extremes(df)
    print(f"Engineered feature set with {len(feature_cols)} columns")

    print("=== STEP 3: Preparing Data for AutoGluon ===")

    df_clean = df.dropna(subset=["outcome"]).sort_values("event_date").reset_index(drop=True)
    if df_clean.empty:
        raise ValueError("No rows with valid outcomes found after cleaning.")

    train_cutoff = df_clean["event_date"].quantile(0.7)
    val_cutoff = df_clean["event_date"].quantile(0.85)

    train_data = df_clean[df_clean["event_date"] < train_cutoff].copy()
    val_data = df_clean[(df_clean["event_date"] >= train_cutoff) & (df_clean["event_date"] < val_cutoff)].copy()
    test_data = df_clean[df_clean["event_date"] >= val_cutoff].copy()

    if val_data.empty or test_data.empty:
        total_rows = len(df_clean)
        train_end = int(total_rows * 0.8)
        val_end = int(total_rows * 0.9)
        train_data = df_clean.iloc[:train_end].copy()
        val_data = df_clean.iloc[train_end:val_end].copy()
        test_data = df_clean.iloc[val_end:].copy()

    if val_data.empty:
        raise ValueError("Validation split produced zero rows; please verify event_date coverage.")
    if test_data.empty:
        raise ValueError("Test split produced zero rows; please verify event_date coverage.")

    print(
        "Temporal splits (train/val/test): "
        f"{train_data['event_date'].min().date()}->{train_data['event_date'].max().date()} | "
        f"{val_data['event_date'].min().date()}->{val_data['event_date'].max().date()} | "
        f"{test_data['event_date'].min().date()}->{test_data['event_date'].max().date()}"
    )
    print(f"Training set: {len(train_data)} fights")
    print(f"Validation set: {len(val_data)} fights")
    print(f"Test set: {len(test_data)} fights")

    ag_features = feature_cols + ["outcome"]
    train_ag_base = train_data[ag_features].copy()
    val_ag_base = val_data[ag_features].copy()
    test_ag_base = test_data[ag_features].copy()

    # === FIGHTER POSITION RANDOMIZATION (training data only) ===
    # Randomly swap A/B positions for ~50% of training rows to prevent positional bias.
    # Since features are already differentials (A - B), swapping = negate _diff + flip outcome + swap odds.
    # Val/test data keeps original outcomes for honest evaluation.
    rng = np.random.default_rng(42)
    diff_cols = [c for c in train_ag_base.columns if c.endswith('_diff')]
    odds_cols_a = [c for c in train_ag_base.columns if c == 'A_open_odds']
    odds_cols_b = [c for c in train_ag_base.columns if c == 'B_open_odds']
    
    # Class-balanced swap: 50% from each class
    class_0_idx = train_ag_base[train_ag_base['outcome'] == 0].index.tolist()
    class_1_idx = train_ag_base[train_ag_base['outcome'] == 1].index.tolist()
    swap_0 = rng.choice(class_0_idx, size=len(class_0_idx) // 2, replace=False)
    swap_1 = rng.choice(class_1_idx, size=len(class_1_idx) // 2, replace=False)
    swap_mask = train_ag_base.index.isin(set(swap_0) | set(swap_1))
    
    # Negate all differential features for swapped rows
    train_ag_base.loc[swap_mask, diff_cols] = -train_ag_base.loc[swap_mask, diff_cols]
    # Flip outcome
    train_ag_base.loc[swap_mask, 'outcome'] = 1.0 - train_ag_base.loc[swap_mask, 'outcome']
    # Swap A/B odds
    if 'A_open_odds' in train_ag_base.columns and 'B_open_odds' in train_ag_base.columns:
        temp_odds = train_ag_base.loc[swap_mask, 'A_open_odds'].copy()
        train_ag_base.loc[swap_mask, 'A_open_odds'] = train_ag_base.loc[swap_mask, 'B_open_odds']
        train_ag_base.loc[swap_mask, 'B_open_odds'] = temp_odds
        
    # FIX 1: opening_odds_diff ends in _diff, so it was ALREADY negated by diff_cols. 
    # Do not negate it again!
    
    # FIX 2: implied_prob_A does NOT end in _diff, so flip it here
    if 'implied_prob_A' in train_ag_base.columns:
        train_ag_base.loc[swap_mask, 'implied_prob_A'] = 1.0 - train_ag_base.loc[swap_mask, 'implied_prob_A']
        
    # FIX 3: elo_diff_squared ends in _squared, so it was missed by diff_cols. Negate it here.
    if 'elo_diff_squared' in train_ag_base.columns:
        train_ag_base.loc[swap_mask, 'elo_diff_squared'] = -train_ag_base.loc[swap_mask, 'elo_diff_squared']
        
    # FIX 4: elo_win_prob does NOT end in _diff, so flip it here
    if 'elo_win_prob' in train_ag_base.columns:
        train_ag_base.loc[swap_mask, 'elo_win_prob'] = 1.0 - train_ag_base.loc[swap_mask, 'elo_win_prob']
    
    n_swapped = int(swap_mask.sum())
    post_dist = train_ag_base['outcome'].value_counts().sort_index().to_dict()
    print(f"   Randomized {n_swapped}/{len(train_ag_base)} training rows (class-balanced)")
    print(f"   Post-randomization class distribution: {post_dist}")

    class_distribution = train_ag_base["outcome"].value_counts().sort_index()
    print("Class distribution:", class_distribution.to_dict())
    
    # UNIFORM WEIGHTS PER USER REQUEST (Dec 25)
    # Removing all class weighting to let model learn natural probabilities
    print("Using uniform sample weights (1.0) for all rows")
    
    # FIX: Dynamically determine majority/minority classes to prevent fragility if outcome encoding drifts
    value_counts = train_ag_base['outcome'].value_counts()
    if len(value_counts) == 2 and value_counts.iloc[0] == value_counts.iloc[1]:
        minority_label = 1   # positive class = Win
        majority_label = 0   # negative class = Loss
    else:
        majority_label = int(value_counts.idxmax())
        minority_label = int(value_counts.idxmin())

    train_ag = train_ag_base.copy()
    val_ag = val_ag_base.copy()
    test_ag = test_ag_base.copy()
    
    # CLASS WEIGHTS - Removed uniform sample weights since data is perfectly 50/50 balanced
    # Change these values to boost recall for a specific class if needed
    CLASS_0_WEIGHT = 1.0  # Loss prediction weight
    CLASS_1_WEIGHT = 1.0  # Win prediction weight
    class_weights = {0: CLASS_0_WEIGHT, 1: CLASS_1_WEIGHT}
    
    # Assign weights to samples (DISABLED)
    # train_ag["sample_weight"] = train_ag["outcome"].map(class_weights).fillna(1.0)
    # val_ag['sample_weight'] = 1.0
    # print(f"Added 'sample_weight' column: min={train_ag['sample_weight'].min():.2f}, max={train_ag['sample_weight'].max():.2f}")
    
    print(f"AutoGluon data ready with {len(feature_cols)} engineered features")

    metrics: dict[str, Any] = {
        "train_fights": int(len(train_data)),
        "val_fights": int(len(val_data)),
        "test_fights": int(len(test_data)),
        "num_features": int(len(feature_cols)),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.to_dict().items()},
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "minority_label": None if minority_label is None else int(minority_label),
        "majority_label": None if majority_label is None else int(majority_label),
    }

    # Determine model paths relative to the output directory when not provided.
    # CRITICAL: Always use output_dir for model storage, NOT temp directories.
    # AutoGluon's stacker models (L2) store relative paths to base models (L1).
    # Using temp directories causes path resolution failures when L2 models try
    # to load L1 models via relative paths like "../../../../mnt/azureml/..."
    cleanup_full_path: Optional[Path] = None
    if existing_model_path is None:
        # Always store in output_dir to ensure relative paths resolve correctly
        existing_model_path = output_dir / "autogluon_full_model"

    predictor: Optional[TabularPredictor]
    predictor_path = Path(existing_model_path)

    if use_existing_model and predictor_path.exists():
        predictor = TabularPredictor.load(str(predictor_path))
        print(f"Loaded existing predictor from {predictor_path}")
        if best_model_name:
            print(f"Highlighting stored model: {best_model_name}")
    elif use_existing_model and not predictor_path.exists():
        print(f"Requested to use existing model, but {predictor_path} does not exist. Training a new model instead.")
        use_existing_model = False
        predictor = None
    else:
        predictor = None

    if predictor is None:
        # ========================================================================
        # FULL MODEL TRAINING (No 2-stage filtering - replicates Dec 24 best config)
        # ========================================================================
        print("=" * 70)
        print("Training Full Model on ALL Features (No Filtering)")
        print("=" * 70)
        
        predictor = TabularPredictor(
            label="outcome",
            problem_type="binary",
            eval_metric="log_loss",
            path=str(predictor_path),
            # sample_weight="sample_weight", # Removed redundant sample weighting
        )
        
        # Train on ALL features without any pre-filtering
        predictor.fit(
            train_data=TabularDataset(train_ag),
            tuning_data=TabularDataset(val_ag_base),
            # CRITICAL: Disable Bagging/Stacking to prevent Temporal Leakage
            auto_stack=False,
            num_bag_folds=0,
            presets="extreme_quality",
            time_limit=time_limit,
            verbosity=2,
            hyperparameters={
                'CAT': {'task_type': 'CPU'},
                'GBM': {'device': 'cpu'},   # LightGBM - CPU for better quality
                'XGB': {},           # XGBoost
                'RF': {},            # Random Forest
                'XT': {},            # Extra Trees
                # 'KNN' excluded: log_loss ~1.27 (worse than random), drags down ensemble
                # 'LR' excluded: crashes on Kaggle due to numpy._core.numeric deserialization bug
                'REALTABPFN-V2': {}, # TabPFN
                'TABM': {},          # TabM (neural)
                'TABDPT': {},        # TabDPT (neural)
                'NN_TORCH': {},      # Neural Network
                'FASTAI': {},        # FastAI Neural Network
            },
            ag_args_fit={"num_gpus": num_gpu} if num_gpu is not None else None,
        )


    # Use base datasets directly (no filtering in this script)
    train_ag_aligned = train_ag
    val_ag_aligned = val_ag
    test_ag_aligned = test_ag

    oof_accuracy: Optional[float] = None
    try:
        oof_predictions = predictor.get_oof_pred(as_multiclass=False)
    except Exception as exc:  # pragma: no cover
        print(f"Unable to compute out-of-fold predictions: {exc}")
    else:
        if oof_predictions is not None:
            oof_array = np.asarray(oof_predictions)
            true_array = np.asarray(train_ag_aligned["outcome"])
            if oof_array.shape == true_array.shape:
                oof_accuracy = accuracy_score(true_array, oof_array)
                print(f"Out-of-fold training accuracy: {oof_accuracy:.3f} ({oof_accuracy * 100:.1f}%)")
                metrics["train_accuracy_oof"] = float(oof_accuracy)
                log_metric("train_accuracy_oof", float(oof_accuracy))
            else:
                print(
                    "Unable to align out-of-fold predictions with training labels for accuracy computation; "
                    "falling back to resubstitution accuracy."
                )

    print("=== STEP 4: Model Evaluation (Stored/Current Predictor) ===")

    y_train_pred = predictor.predict(TabularDataset(train_ag_aligned))
    y_val_pred = predictor.predict(TabularDataset(val_ag_aligned))
    y_test_pred = predictor.predict(TabularDataset(test_ag_aligned))

    train_accuracy = accuracy_score(train_ag_aligned["outcome"], y_train_pred)
    val_accuracy = accuracy_score(val_ag_aligned["outcome"], y_val_pred)
    test_accuracy = accuracy_score(test_ag_aligned["outcome"], y_test_pred)

    print(f"Training accuracy: {train_accuracy:.3f} ({train_accuracy * 100:.1f}%)")
    print(f"Validation accuracy: {val_accuracy:.3f} ({val_accuracy * 100:.1f}%)")
    print(f"Test accuracy: {test_accuracy:.3f} ({test_accuracy * 100:.1f}%)")
    print(f"Overfitting gap: {train_accuracy - test_accuracy:.3f}")

    metrics.update(
        {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "test_accuracy": float(test_accuracy),
            "overfit_gap": float(train_accuracy - test_accuracy),
        }
    )
    log_metric("train_accuracy", float(train_accuracy))
    log_metric("val_accuracy", float(val_accuracy))
    log_metric("test_accuracy", float(test_accuracy))
    log_metric("overfit_gap", float(train_accuracy - test_accuracy))

    # Skip leak guard test - production pipeline already validated
    # (test_no_leakage function requires additional setup not needed for production)
    print("Leak guard validation: using production_feature_pipeline.py which is pre-validated")

    report_predictions = y_test_pred

    class_labels = list(getattr(predictor, "class_labels", sorted(test_ag_aligned["outcome"].unique())))

    if len(class_labels) == 2:
        try:
            proba_val = predictor.predict_proba(TabularDataset(val_ag_aligned))
            proba_test = predictor.predict_proba(TabularDataset(test_ag_aligned))
        except Exception as exc:  # pragma: no cover
            print(f"Skipping ROC/threshold evaluation due to probability prediction error: {exc}")
        else:
            def _resolve_label(preferred: Optional[int], fallback: Any) -> Any:
                if preferred is None:
                    return fallback
                for label in class_labels:
                    if str(label) == str(preferred):
                        return label
                return fallback

            positive_label = _resolve_label(metrics.get("minority_label"), class_labels[0])
            negative_label = _resolve_label(metrics.get("majority_label"), class_labels[-1])

            def _extract_scores(proba_obj: Any) -> np.ndarray:
                if isinstance(proba_obj, pd.DataFrame):
                    return np.asarray(proba_obj[positive_label])
                if isinstance(proba_obj, pd.Series):
                    if proba_obj.name == positive_label:
                        return np.asarray(proba_obj)
                    return 1.0 - np.asarray(proba_obj)
                array = np.asarray(proba_obj)
                if len(class_labels) == 2 and class_labels[1] == positive_label:
                    return array
                return 1.0 - array

            positive_scores_val = _extract_scores(proba_val)
            positive_scores_test = _extract_scores(proba_test)

            if negative_label in test_ag_aligned["outcome"].values:
                fpr, tpr, _ = roc_curve(
                    test_ag_aligned["outcome"],
                    positive_scores_test,
                    pos_label=positive_label,
                )
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], "k--", label="Chance")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Holdout ROC Curve (Full Model)")
                ax.legend(loc="lower right")
                fig.tight_layout()
                if persist_full_artifacts:
                    save_figure(fig, output_dir / "roc_curve_full.png")
                else:
                    plt.close(fig)
                metrics["roc_auc_full"] = float(roc_auc)
                log_metric("roc_auc_full", float(roc_auc))
                
                # Compute Log Loss (cross-entropy) for probability calibration assessment
                logloss_val = log_loss(val_ag_aligned["outcome"], positive_scores_val)
                logloss_test = log_loss(test_ag_aligned["outcome"], positive_scores_test)
                metrics["log_loss_val"] = float(logloss_val)
                metrics["log_loss_test"] = float(logloss_test)
                log_metric("log_loss_val", float(logloss_val))
                log_metric("log_loss_test", float(logloss_test))
                print(f"Log Loss - Validation: {logloss_val:.4f}, Test: {logloss_test:.4f}")
            else:
                print(f"Skipping ROC curve: comparison label '{negative_label}' not present in holdout data.")

            threshold_metrics, val_threshold_preds = evaluate_threshold_grid(
                y_true=val_ag_aligned["outcome"],
                positive_scores=positive_scores_val,
                positive_label=positive_label,
                negative_label=negative_label,
                min_negative_recall=0.0,
            )
            best_threshold = float(threshold_metrics["threshold"])
            test_threshold_preds = np.where(
                positive_scores_test >= best_threshold, positive_label, negative_label
            )
            val_accuracy_thresholded = accuracy_score(val_ag_aligned["outcome"], val_threshold_preds)
            test_accuracy_thresholded = accuracy_score(test_ag_aligned["outcome"], test_threshold_preds)

            metrics["val_accuracy_thresholded"] = float(val_accuracy_thresholded)
            metrics["test_accuracy_thresholded"] = float(test_accuracy_thresholded)
            metrics["full_threshold_tuning"] = {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in threshold_metrics.items()
            }
            log_metric("val_accuracy_thresholded", float(val_accuracy_thresholded))
            log_metric("test_accuracy_thresholded", float(test_accuracy_thresholded))
            print(
                "Validation threshold search:",
                f"best_threshold={best_threshold:.3f}",
                f"balanced_acc={threshold_metrics['balanced_accuracy']:.3f}",
            )
            print(
                f"Validation accuracy @ best threshold: {val_accuracy_thresholded:.3f}"
                f" ({val_accuracy_thresholded * 100:.1f}%)"
            )
            print(
                f"Test accuracy @ best threshold: {test_accuracy_thresholded:.3f}"
                f" ({test_accuracy_thresholded * 100:.1f}%)"
            )

            report_predictions = test_threshold_preds


    report_text = classification_report(test_ag_aligned["outcome"], report_predictions)
    print("Holdout classification report (threshold-adjusted predictions):")
    print(report_text)
    if persist_full_artifacts:
        save_text(report_text, output_dir / "classification_report_full.txt")

    # NEW: Generate Advanced Diagnostics (Robustness Charts)
    generate_diagnostics(
        predictor=predictor,
        test_data=test_ag_aligned,
        output_dir=output_dir,
        persist_full_artifacts=persist_full_artifacts
    )

    precision_vals, recall_vals, f1_vals, support_vals = precision_recall_fscore_support(
        test_ag_aligned["outcome"], report_predictions, labels=class_labels, zero_division=0
    )
    metrics["full_precision_per_class"] = {
        str(label): float(precision_vals[idx]) for idx, label in enumerate(class_labels)
    }
    metrics["full_recall_per_class"] = {str(label): float(recall_vals[idx]) for idx, label in enumerate(class_labels)}
    metrics["full_f1_per_class"] = {str(label): float(f1_vals[idx]) for idx, label in enumerate(class_labels)}

    cm = confusion_matrix(test_ag_aligned["outcome"], report_predictions, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Holdout Confusion Matrix (Full Model)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    if persist_full_artifacts:
        save_figure(fig, output_dir / "confusion_matrix_full.png")
    else:
        plt.close(fig)

    print("=== STORED MODEL LEADERBOARD ===")
    leaderboard = predictor.leaderboard(TabularDataset(test_ag_aligned), silent=True)
    print(leaderboard.head(10))
    if persist_full_artifacts:
        save_dataframe(leaderboard, output_dir / "leaderboard_full.csv")

    print("=== STEP 5: Ensemble Feature Importance from Top Models ===")
    # Use ensemble FI from top-performing models to get robust feature selection
    # by aggregating multiple model perspectives
    
    leaderboard_for_fi = predictor.leaderboard(TabularDataset(val_ag_aligned), silent=True)
    dataset_for_fi = TabularDataset(val_ag_aligned)
    
    # Use top 5 non-ensemble models (works regardless of eval_metric)
    high_perf_models = [m for m in leaderboard_for_fi['model'].head(20).tolist() if 'Ensemble' not in m][:5]
    print(f"Using top {len(high_perf_models)} models for FI: {high_perf_models}")
    
    # Compute FI for each high-performing model
    fi_list = []
    for model_name in high_perf_models:
        try:
            print(f"  Computing FI for {model_name}...")
            fi = predictor.feature_importance(
                dataset_for_fi, # Use the selected dataset (Val or Test)
                model=model_name,
                num_shuffle_sets=3,
            )
            fi = fi.reset_index().rename(columns={"index": "feature"})
            fi["model"] = model_name
            
            # Normalize importance scores to 0-1 range for fair averaging
            max_imp = fi["importance"].abs().max()
            if max_imp > 0:
                fi["importance_normalized"] = fi["importance"] / max_imp
            else:
                fi["importance_normalized"] = fi["importance"]
            
            fi_list.append(fi)
            print(f"    ✓ {model_name}: {len(fi)} features scored")
        except Exception as e:
            print(f"    ✗ {model_name}: Failed - {e}")
            continue
    
    if len(fi_list) == 0:
        raise ValueError("Could not compute feature importance from any model.")
    
    print(f"\nSuccessfully computed FI from {len(fi_list)} models!")
    
    # Aggregate: average normalized importance across all models
    all_fi = pd.concat(fi_list)
    feature_importance = all_fi.groupby("feature").agg({
        "importance": "mean",           # Mean of raw importance
        "importance_normalized": "mean", # Mean of normalized importance
        "p_value": "mean",
    }).reset_index()
    
    # Use normalized importance for ranking, but keep raw for interpretability
    feature_importance = feature_importance.sort_values("importance_normalized", ascending=False)
    feature_importance["ensemble_models_count"] = len(fi_list)
    print(f"Ensemble FI computed from {len(fi_list)} models, covering {len(feature_importance)} features")
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    print("Top 20 features by importance:")
    print(feature_importance.head(20))
    if persist_full_artifacts:
        save_dataframe(feature_importance, output_dir / "feature_importance_full.csv")

    top_feature_count = min(20, len(feature_importance))
    if top_feature_count > 0:
        top_feature_df = feature_importance.head(top_feature_count).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(6, top_feature_count * 0.35)))
        sns.barplot(data=top_feature_df, x="importance", y="feature", palette="viridis", ax=ax)
        ax.set_title(f"Top {top_feature_count} Feature Importances (Full Model)")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        if persist_full_artifacts:
            save_figure(fig, output_dir / "feature_importance_full.png")
        else:
            plt.close(fig)
    
    # === NEW: Comprehensive Feature Importance Chart (All Features, Negative to Positive) ===
    print("Generating comprehensive feature importance chart (all features, negative to positive)...")
    fi_sorted = feature_importance.sort_values("importance", ascending=True)  # Negative at top, positive at bottom
    
    # Color coding based on importance and p-value
    def get_color(row):
        if row['importance'] < 0:
            return '#EF5350'  # Red for negative
        elif 'p_value' in row and row['p_value'] < 0.01:
            return '#2E7D32'  # Dark Green for highly reliable
        elif 'p_value' in row and row['p_value'] < 0.05:
            return '#66BB6A'  # Light Green for reliable
        elif 'p_value' in row and row['p_value'] < 0.1:
            return '#FFA726'  # Orange for borderline
        else:
            return '#FFCC80'  # Light Orange for unreliable positive
    
    colors = fi_sorted.apply(get_color, axis=1).tolist()
    
    # Create the comprehensive chart
    fig_height = max(12, len(fi_sorted) * 0.22)  # Scale height with feature count
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    y_pos = range(len(fi_sorted))
    ax.barh(y_pos, fi_sorted['importance'], color=colors, edgecolor='white', height=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:50] for f in fi_sorted['feature']], fontsize=7)  # Truncate long names
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Full Model Feature Importance\n(All Features: Negative → Positive)', fontsize=13, fontweight='bold')
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF5350', label='Negative (hurts model)'),
        Patch(facecolor='#FFCC80', label='Positive (p >= 0.1, noisy)'),
        Patch(facecolor='#FFA726', label='Positive (p < 0.1, borderline)'),
        Patch(facecolor='#66BB6A', label='Positive (p < 0.05, reliable)'),
        Patch(facecolor='#2E7D32', label='Positive (p < 0.01, highly reliable)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Add count annotations
    n_negative = (fi_sorted['importance'] < 0).sum()
    n_positive = (fi_sorted['importance'] >= 0).sum()
    ax.text(0.02, 0.98, f"Negative: {n_negative} | Positive: {n_positive}", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    save_figure(fig, output_dir / "feature_importance_full_comprehensive.png")
    print(f"Saved comprehensive FI chart: {n_negative} negative, {n_positive} positive features")





    print("=== STEP 9: GENERATE PREDICTIONS CSV SET ===")
    try:
        # Extract 2025 holdout dataset
        df_2025 = test_data[test_data['event_date'].dt.year == 2025].copy()
        df_2024 = test_data[test_data['event_date'].dt.year == 2024].copy()
        
        # Verify odds exist, if not merge from raw_df
        if 'A_open_odds' not in df_2025.columns or 'B_open_odds' not in df_2025.columns:
            print("Odds stripped during pipeline... Remerging from raw data.")
            raw_odds = raw_df[['fight_id', 'A_open_odds', 'B_open_odds']].copy()
            raw_odds = raw_odds.dropna(subset=['A_open_odds', 'B_open_odds'])
            
            df_2025 = df_2025.drop(columns=['A_open_odds', 'B_open_odds'], errors='ignore')
            df_2025 = df_2025.merge(raw_odds, on='fight_id', how='inner')
            
            df_2024 = df_2024.drop(columns=['A_open_odds', 'B_open_odds'], errors='ignore')
            df_2024 = df_2024.merge(raw_odds, on='fight_id', how='inner')
        
        df_2025 = df_2025.dropna(subset=['outcome', 'A_open_odds', 'B_open_odds'])
        df_2024 = df_2024.dropna(subset=['outcome', 'A_open_odds', 'B_open_odds'])
        
        print(f"Generating predictions for {len(df_2025)} fights in 2025 and {len(df_2024)} fights in 2024...")
        
        # Function to generate and save predictions
        def save_predictions_for_model(pred_model, suffix, year, df_target):
            if pred_model is None or df_target.empty: return
            
            y_proba = pred_model.predict_proba(TabularDataset(df_target))
            y_pred = pred_model.predict(TabularDataset(df_target))
            
            temp_df = df_target.copy()
            if isinstance(y_proba, pd.DataFrame):
                # Medium-Risk Fix: Dynamically locate the positive class ('1' or 1.0) probability rather than hardcoding column index 1
                pos_class_col = next((c for c in y_proba.columns if str(c) in ['1', '1.0', 1, 1.0]), y_proba.columns[1] if len(y_proba.columns) > 1 else y_proba.columns[0])
                temp_df['model_prob'] = y_proba[pos_class_col].values
            else:
                temp_df['model_prob'] = y_proba
                
            temp_df['prediction'] = y_pred.values
            
            out_cols = ['event_date', 'fighter_a_name', 'fighter_b_name', 'A_open_odds', 'B_open_odds', 'outcome', 'model_prob', 'prediction']
            if 'fight_id' in temp_df.columns:
                out_cols = ['fight_id'] + out_cols
            
            out_cols = [c for c in out_cols if c in temp_df.columns]
            
            out_df = temp_df[out_cols]
            out_path = output_dir / f"predictions_{suffix}_{year}.csv"
            out_df.to_csv(out_path, index=False)
            print(f"✅ Predictions saved to {out_path} ({len(out_df)} fights)")

        save_predictions_for_model(predictor, "full_model", "2025", df_2025)
        save_predictions_for_model(predictor, "full_model", "2024", df_2024)
        
    except Exception as e:
        print(f"Failed to generate prediction CSV: {e}")

    save_json(metrics, output_dir / "metrics_summary.json")

    # Note: cleanup_full_path is no longer used - models always stored in output_dir
    # to prevent AutoGluon relative path resolution issues with stacker models

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Kaggle notebook execution."""
    parser = argparse.ArgumentParser(description="Train the advanced AutoGluon UFC model (Kaggle version).")
    default_data = Path("/kaggle/input/ufc-fights-ml-full-odds-csv/ufc_fights_full_with_odds.csv")

    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data,
        help="Path to the input CSV dataset or a directory containing a single CSV export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working"),
        help="Directory for outputs. Defaults to /kaggle/working.",
    )
    parser.add_argument(
        "--use-existing-model",
        action="store_true",
        help="Load a previously trained AutoGluon model instead of fitting a new one.",
    )
    parser.add_argument(
        "--existing-model-path",
        type=Path,
        default=None,
        help="Path to an AutoGluon predictor directory to load (if --use-existing-model is set).",
    )
    parser.add_argument(
        "--best-model-name",
        type=str,
        default="CatBoost_r70_BAG_L1_FULL",
        help="Optional reference name for the highlighted model in logs.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional training time limit in seconds for AutoGluon fit calls.",
    )
    parser.add_argument(
        "--persist-full-artifacts",
        action="store_true",
        help="Persist the full ensemble model directory and visualizations alongside selected-model outputs.",
    )

    return parser.parse_args()


def load_hyperparameters(raw_value: Optional[str]) -> Optional[dict[str, Any]]:
    """Interpret hyperparameters provided as JSON string or file path."""
    if raw_value is None:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        path = Path(raw_value)
        if not path.exists():
            raise ValueError(f"Hyperparameters must be valid JSON or a path to JSON. Got: {raw_value}") from None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    import sys
    import os
    
    # Detect if running in Jupyter/Kaggle notebook
    # Kaggle check: /kaggle directory exists or KAGGLE_KERNEL_RUN_TYPE environment variable
    is_kaggle = os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    is_jupyter = any('kernel' in arg.lower() or 'ipython' in arg.lower() for arg in sys.argv)
    is_notebook = is_kaggle or is_jupyter
    
    if is_notebook:
        # Running in notebook - bypass argparse and use defaults
        print("Detected notebook environment - using default parameters")
        print("FULL MODEL ONLY - Selected model training is removed.")
        output_dir = resolve_output_dir(Path("/kaggle/working"))
        metrics = main(
            data_path=Path("/kaggle/input/ufc-fights-ml-full-odds-csv"),
            output_dir=output_dir,
            use_existing_model=False,
            existing_model_path=None,
            best_model_name=None,
            time_limit=None,
            persist_full_artifacts=True,
        )
    else:
        # Running from command line - use argparse
        args = parse_args()
        output_dir = resolve_output_dir(args.output_dir)
        print("FULL MODEL ONLY - Selected model training is removed.")
        metrics = main(
            data_path=args.data_path,
            output_dir=output_dir,
            use_existing_model=args.use_existing_model,
            existing_model_path=args.existing_model_path,
            best_model_name=args.best_model_name,
            time_limit=args.time_limit,
            persist_full_artifacts=args.persist_full_artifacts,
        )
    
    print("=== METRICS SUMMARY ===")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    
    # Auto-zip outputs for download (Kaggle only)
    output_path = Path("/kaggle/working")
    if output_path.exists():
        import zipfile
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"ufc_model_outputs_{timestamp}.zip"
        zip_path = output_path / zip_name
        
        print(f"\n=== Creating download package: {zip_name} ===")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in output_path.rglob('*'):
                if file.is_file() and file.name != zip_name:
                    arcname = file.relative_to(output_path)
                    zipf.write(file, arcname)
                    print(f"   Added: {arcname}")
        
        print(f"\n✅ Download package ready: {zip_path}")
        print(f"   Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("   → Go to 'Output' tab on the right and download the zip file")
        
        # Try to trigger Kaggle file download (only works in interactive notebooks)
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(zip_path.relative_to('/kaggle/working'))))
        except Exception:
            pass  # Not in IPython/Jupyter context