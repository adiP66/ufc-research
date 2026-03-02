"""
KAGGLE COMPREHENSIVE VALIDATION (ALTERNATE)
==========================================
Fixes:
- No double randomization
- Optional positional randomization (off by default)
- Proper isotonic calibration option
"""

from __future__ import annotations
import gc
import os
import pickle
from typing import Any
from pathlib import Path
import subprocess
import sys
# Install AutoGluon with ALL optional model types including TabPFN
subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.tabular[all]==1.5.0", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn>=2.0", "tabdpt", "tabm", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "lightgbm", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "fastcore>=1.8,<1.9", "-q"])

try:
    from autogluon.tabular import TabularPredictor
    print("AutoGluon already installed.")
except ImportError:
    print("Installing AutoGluon... (this may take 2-3 minutes)")
    import os
    # Install AutoGluon
    os.system("pip install -q autogluon.tabular")
    # FIX: Pin fastcore to restore rtoken_hex for nbdev_export
    os.system("pip install -q 'fastcore>=1.8,<1.9'")
    from autogluon.tabular import TabularPredictor
    print("AutoGluon and dependencies installed successfully!")






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

import sys
# Force python to look in the utility script directory
sys.path.insert(0, '/kaggle/usr/lib/production_feature_pipeline_v2')
sys.path.insert(0, '/kaggle/usr/lib/autogluon_model_v2')

import production_feature_pipeline_v2 as pipeline

# --- CONFIG ---
DATA_PATH = "/kaggle/input/ufc-fights-ml-full-odds-csv/ufc_fights_full_with_odds.csv"
MODEL_QUALITY = "extreme_quality"
YEARS_TO_VALIDATE = [2023, 2024, 2025]
CALIBRATION_METHOD = "temperature"  # "temperature" or "isotonic"
CALIBRATION_TRAIN_YEAR = 2024
CALIBRATION_TEST_YEAR = 2025
CALIBRATOR_PATH = Path(f"calibrator_{CALIBRATION_METHOD}.pkl")

RANDOMIZE_POSITIONS = False


def clean_fight_data(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Standardize fight rows before feature engineering (no randomization)."""
    df = df.copy()
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

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])

    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()

    future_mask = df["event_date"] > today
    if future_mask.any():
        df = df.loc[~future_mask].copy()

    if "fight_id" in df.columns:
        df = df.sort_values("event_date").drop_duplicates(subset="fight_id", keep="last")

    if "outcome" in df.columns:
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    return df.sort_values("event_date").reset_index(drop=True)


def randomize_fighter_positions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Optional positional randomization (no forced class balance)."""
    fighter_cols = [c for c in df.columns if c.startswith(("fighter_a_", "fighter_b_"))]
    if not fighter_cols:
        return df

    base_fields = set()
    for col in fighter_cols:
        parts = col.split("_", 2)
        if len(parts) == 3:
            base_fields.add(parts[2])

    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < 0.5
    if not mask.any():
        return df

    df = df.copy()
    for base in base_fields:
        a_col = f"fighter_a_{base}"
        b_col = f"fighter_b_{base}"
        if a_col in df.columns and b_col in df.columns:
            temp = df.loc[mask, a_col].copy()
            df.loc[mask, a_col] = df.loc[mask, b_col]
            df.loc[mask, b_col] = temp

    if "outcome" in df.columns:
        df.loc[mask, "outcome"] = 1.0 - df.loc[mask, "outcome"].astype(float)

    odds_swap_pairs = [("A_open_odds", "B_open_odds"), ("A_open_prob", "B_open_prob")]
    for col_a, col_b in odds_swap_pairs:
        if col_a in df.columns and col_b in df.columns:
            temp = df.loc[mask, col_a].copy()
            df.loc[mask, col_a] = df.loc[mask, col_b]
            df.loc[mask, col_b] = temp

    if "A_open_prob" in df.columns and "B_open_prob" in df.columns:
        df["opening_odds_diff"] = df["A_open_prob"] - df["B_open_prob"]
        df["implied_prob_A"] = df["A_open_prob"]

    return df


def temperature_scale(prob: np.ndarray, T: float = 1.38) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    scaled_logit = logit * T
    return 1 / (1 + np.exp(-scaled_logit))


def fit_temperature_scale(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fit a temperature scalar by minimizing log loss on a grid."""
    best_t = 1.0
    best_ll = float("inf")
    for t in np.linspace(0.6, 2.2, 17):
        p_cal = temperature_scale(y_prob, T=t)
        ll = log_loss(y_true, p_cal)
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)
    return best_t


def run_walk_forward_validation() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = clean_fight_data(df)
    if RANDOMIZE_POSITIONS:
        df = randomize_fighter_positions(df)

    df, feature_cols = pipeline.build_prefight_features(df)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    for test_year in YEARS_TO_VALIDATE:
        train_data = df[df["event_date"].dt.year < test_year].copy()
        test_data = df[df["event_date"].dt.year == test_year].copy()

        if test_data.empty:
            continue

        predictor = TabularPredictor(
            label="outcome",
            problem_type="binary",
            eval_metric="log_loss",
            path=f"autogluon_wfv_{test_year}",
        )
        predictor.fit(
            train_data[feature_cols + ["outcome"]],
            presets=MODEL_QUALITY,
            time_limit=1800 if MODEL_QUALITY == "high_quality" else 1200,
            auto_stack=False,
            num_bag_folds=0,
            excluded_model_types=["KNN", "NN_TORCH"],
        )

        y_pred_prob = predictor.predict_proba(test_data[feature_cols]).iloc[:, 1]
        y_true = test_data["outcome"]

        acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred_prob)
        ll = log_loss(y_true, y_pred_prob)
        print(f"YEAR {test_year}: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")

        test_data["model_prob"] = y_pred_prob
        test_data.to_csv(f"predictions_{test_year}.csv", index=False)

        del predictor
        gc.collect()


def run_calibration_analysis() -> None:
    train_path = f"predictions_{CALIBRATION_TRAIN_YEAR}.csv"
    test_path = f"predictions_{CALIBRATION_TEST_YEAR}.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Missing predictions files for calibration.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train["outcome"].to_numpy()
    p_train = df_train["model_prob"].to_numpy()
    y_test = df_test["outcome"].to_numpy()
    p_test = df_test["model_prob"].to_numpy()

    if CALIBRATION_METHOD == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train, y_train)
        y_calib = iso.transform(p_test)
        with CALIBRATOR_PATH.open("wb") as f:
            pickle.dump(iso, f)
    else:
        best_t = fit_temperature_scale(y_train, p_train)
        y_calib = temperature_scale(p_test, T=best_t)
        with CALIBRATOR_PATH.open("wb") as f:
            pickle.dump({"method": "temperature", "T": best_t}, f)

    df_test["calibrated_prob"] = y_calib

    for name, col in [("Raw", "model_prob"), ("Calibrated", "calibrated_prob")]:
        brier = brier_score_loss(y_test, df_test[col])
        ll = log_loss(y_test, df_test[col])
        auc = roc_auc_score(y_test, df_test[col])
        print(f"{name}: Brier={brier:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}")

    plt.figure(figsize=(8, 8))
    for name, col in [("Raw", "model_prob"), ("Calibrated", "calibrated_prob")]:
        fp, mv = calibration_curve(y_test, df_test[col], n_bins=5, strategy="quantile")
        plt.plot(mv, fp, marker="o", label=name)

    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title(f"Calibration {CALIBRATION_TEST_YEAR}")
    plt.savefig("calibration_comparison_kaggle.png")
    plt.close()


if __name__ == "__main__":
    run_walk_forward_validation()
    run_calibration_analysis()
    print("DONE")
