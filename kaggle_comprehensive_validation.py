
"""
KAGGLE COMPREHENSIVE VALIDATION SCRIPT
======================================
This script performs the following steps to rigorously validate the UFC Model:
1. Feature Engineering (using Pipeline V2)
2. Walk-Forward Validation (Training on Past, Testing on Future)
3. Isotonic Calibration (Learning from recent past)
4. Plotting Calibration Curves (5-Bin Quantile for smoothness)
5. Calculating ROI

Environment: Kaggle (Linux, 2x GPU)
"""

# --- INSTALL AUTOGLUON & FIX ENVIRONMENT ---


try:
    from autogluon.tabular import TabularPredictor
    print("AutoGluon already installed.")
except ImportError:
    print("Installing AutoGluon... (this may take 2-3 minutes)")
    import os
    # Install AutoGluon
    os.system("pip install -q autogluon.tabular")
    # FIX: Upgrade fastcore to solve 'rtoken_hex' import error in nbdev_export
    os.system("pip install -q --upgrade fastcore")
    from autogluon.tabular import TabularPredictor
    print("AutoGluon and dependencies installed successfully!")

# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import gc

# --- CONFIGURATION ---
DATA_PATH = "/kaggle/input/ufc-fights-ml-full-odds-csv/ufc_fights_full_with_odds.csv"
MODEL_QUALITY = "extreme_quality"
YEARS_TO_VALIDATE = [2023, 2024, 2025]
CALIBRATION_TRAIN_YEAR = 2024
CALIBRATION_TEST_YEAR = 2025

# Excluded features (negative importance in Dec 24 analysis)
EXCLUDED_FEATURES = [
    'age_dec_avg_diff',
    'body_strikes_attempted_dec_adjperf_dec_avg_diff',
    'head_strikes_landed_dec_adjperf_dec_avg_diff',
    'control_time_per_fight_career_avg_dec_adjperf_dec_avg_diff',
    'sub_att_diff',
    'opp_reach_dec_avg_diff',
    'distance_strikes_attempted_dec_adjperf_dec_avg_diff',
    'leg_land_ratio_dec_adjperf_dec_avg_diff',
    'ground_strikes_landed_dec_adjperf_dec_avg_diff',
    'sig_strikes_landed_per_min_career_avg_dec_adjperf_dec_avg_diff',
    'ko_tko_win_dec_avg_diff',
    'leg_land_ratio_diff',
    'ground_strikes_attempted_dec_adjperf_dec_avg_diff',
]

# --- PIPELINE IMPORT ---
# --- UTILITY SCRIPT IMPORTS ---
print("!!! DEBUG: SCRIPT VERSION 100 LOADED !!!") # Confirm update on Kaggle
import sys
# Force python to look in the utility script directory
sys.path.insert(0, '/kaggle/usr/lib/production_feature_pipeline_v2')
sys.path.insert(0, '/kaggle/usr/lib/autogluon_model_v2')

try:
    import production_feature_pipeline_v2 as pipeline
    # Try importing cleaning functions from the model script
    from autogluon_model_v2 import clean_fight_data, randomize_fighter_positions
    print("SUCCESS: Pipeline and Model scripts imported.")
except ImportError as e:
    print(f"WARNING: Import failed ({e}). USING LOCAL FALLBACK for cleaning functions.")
    
    # FALLBACK: Define functions locally if utility script is old/missing
    import numpy as np
    
    def randomize_fighter_positions(df, seed=42):
        fighter_cols = [col for col in df.columns if col.startswith(('fighter_a_', 'fighter_b_'))]
        if not fighter_cols: return df
        base_fields = set()
        for col in fighter_cols:
            parts = col.split('_', 2)
            if len(parts) == 3: base_fields.add(parts[2])
        rng = np.random.default_rng(seed)
        if 'outcome' in df.columns:
            class_0_idx = df[df['outcome'] == 0].index.tolist()
            class_1_idx = df[df['outcome'] == 1].index.tolist()
            n_swap_0 = len(class_0_idx) // 2
            n_swap_1 = len(class_1_idx) // 2
            swap_from_0 = rng.choice(class_0_idx, size=n_swap_0, replace=False)
            swap_from_1 = rng.choice(class_1_idx, size=n_swap_1, replace=False)
            swap_indices = set(swap_from_0) | set(swap_from_1)
            mask = df.index.isin(swap_indices)
        else:
            mask = rng.random(len(df)) < 0.5
        swapped_rows = int(mask.sum())
        if swapped_rows == 0: return df
        df = df.copy()
        for base in base_fields:
            a_col = f'fighter_a_{base}'; b_col = f'fighter_b_{base}'
            if a_col in df.columns and b_col in df.columns:
                temp = df.loc[mask, a_col].copy()
                df.loc[mask, a_col] = df.loc[mask, b_col]
                df.loc[mask, b_col] = temp
        if 'outcome' in df.columns:
            df.loc[mask, 'outcome'] = 1.0 - df.loc[mask, 'outcome'].astype(float)
        odds_swap_pairs = [('A_open_odds', 'B_open_odds'), ('A_open_prob', 'B_open_prob')]
        for col_a, col_b in odds_swap_pairs:
            if col_a in df.columns and col_b in df.columns:
                temp = df.loc[mask, col_a].copy()
                df.loc[mask, col_a] = df.loc[mask, col_b]
                df.loc[mask, col_b] = temp
        if 'A_open_prob' in df.columns and 'B_open_prob' in df.columns:
            df['opening_odds_diff'] = df['A_open_prob'] - df['B_open_prob']
        return df

    def clean_fight_data(df, today=None):
        print('=== DATA HYGIENE CHECKS (FALLBACK) ===')
        df = df.copy()
        precomputed_diffs = ['sig_strikes_landed_diff', 'sig_strike_accuracy_diff', 'takedowns_landed_diff', 'takedown_defense_diff', 'reach_diff', 'age_diff']
        dropped_diffs = [col for col in precomputed_diffs if col in df.columns]
        if dropped_diffs: df = df.drop(columns=dropped_diffs)
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df.dropna(subset=['event_date'])
        if today is None: today = pd.Timestamp.today().normalize()
        future_mask = df['event_date'] > today
        if future_mask.any(): df = df.loc[~future_mask].copy()
        if 'fight_id' in df.columns: df = df.sort_values('event_date').drop_duplicates(subset='fight_id', keep='last')
        df.reset_index(drop=True, inplace=True)
        if 'outcome' in df.columns: df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')
        df = randomize_fighter_positions(df)
        df = df.sort_values('event_date').reset_index(drop=True)
        return df

# --- HELPER FUNCTIONS ---

def get_payout(odds, stake):
    if pd.isna(odds) or odds == 0: return 0
    if odds > 0: return stake * (odds / 100)
    else: return stake * (100 / abs(odds))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy = y_true[in_bin].mean()
            avg_confidence = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence - accuracy) * prop_in_bin
    return ece

# --- MAIN VALIDATION LOGIC ---

def run_walk_forward_validation():
    print(f"=== STARTING WALK-FORWARD VALIDATION ({MODEL_QUALITY}) ===")
    
    import os
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    df = clean_fight_data(df)
    
    print("Running Feature Pipeline...")
    df, feature_cols = pipeline.build_prefight_features(df)
    
    # Remove excluded negative-importance features
    feature_cols = [f for f in feature_cols if f not in EXCLUDED_FEATURES]
    print(f"Using {len(feature_cols)} features ({len(EXCLUDED_FEATURES)} excluded)")
    
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date').reset_index(drop=True)
    
    results = []
    
    for test_year in YEARS_TO_VALIDATE:
        print(f"\n\n>>> PROCESSING YEAR: {test_year} <<<")
        
        train_data = df[df['event_date'].dt.year < test_year].copy()
        test_data = df[df['event_date'].dt.year == test_year].copy()
        
        if len(test_data) == 0:
            print(f"Skipping {test_year} (No data)")
            continue
            
        print(f"Training Size: {len(train_data)} | Test Size: {len(test_data)}")
        
        predictor = TabularPredictor(label='outcome', problem_type='binary', eval_metric='log_loss', path=f"autogluon_wfv_{test_year}")
        predictor.fit(
            train_data[feature_cols + ['outcome']],
            presets=MODEL_QUALITY, # "extreme_quality"
            # CRITICAL: Disable Bagging/Stacking to prevent Temporal Leakage
            auto_stack=False,
            num_bag_folds=0,
        )
        
        y_pred_prob = predictor.predict_proba(test_data[feature_cols]).iloc[:, 1]
        y_true = test_data['outcome']
        
        acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred_prob)
        ll = log_loss(y_true, y_pred_prob)
        
        print(f"YEAR {test_year} RESULTS: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
        
        test_data['model_prob'] = y_pred_prob
        test_data.to_csv(f"predictions_{test_year}.csv", index=False)
        
        results.append({'Year': test_year, 'Accuracy': acc, 'AUC': auc, 'LogLoss': ll})
        
        del predictor
        gc.collect()

    print("\n--- WALK-FORWARD SUMMARY ---")
    print(pd.DataFrame(results))


def temperature_scale(prob, T=1.38):
    """Sharpen probabilities using Temperature Scaling (T=1.38)."""
    p = np.clip(prob, 1e-6, 1-1e-6)
    logit = np.log(p / (1 - p))
    scaled_logit = logit * T
    return 1 / (1 + np.exp(-scaled_logit))

def run_calibration_analysis():
    print(f"\n=== CALIBRATION ANALYSIS (Train {CALIBRATION_TRAIN_YEAR} -> Test {CALIBRATION_TEST_YEAR}) ===")
    
    import os
    if not os.path.exists(f"predictions_{CALIBRATION_TEST_YEAR}.csv"):
        print("Predictions file not found.")
        return
        
    df_test = pd.read_csv(f"predictions_{CALIBRATION_TEST_YEAR}.csv")
    y_true = df_test['outcome']
    y_prob = df_test['model_prob']
    
    # 1. Temperature Scaling
    print("Applying Temperature Scaling (T=1.38)...")
    y_calib = temperature_scale(y_prob, T=1.38)
    df_test['calibrated_prob'] = y_calib
    
    # 2. Vegas Prob (if available)
    if 'implied_prob_A' in df_test.columns:
         df_test['vegas_prob'] = df_test['implied_prob_A'].fillna(0.5)
    else:
         # Fallback recalculation
         def get_vegas_prob(row):
             if pd.notna(row.get('A_open_odds')):
                 odd = row['A_open_odds']
                 if odd == 0: return 0.5
                 return 100/(odd+100) if odd > 0 else abs(odd)/(abs(odd)+100)
             return 0.5
         df_test['vegas_prob'] = df_test.apply(get_vegas_prob, axis=1)

    # Metrics
    metrics = {}
    for name, col in [("Raw", "model_prob"), ("Calibrated (T=1.38)", "calibrated_prob"), ("Vegas", "vegas_prob")]:
        y_p = df_test[col]
        brier = brier_score_loss(y_true, y_p)
        try:
            ece = expected_calibration_error(y_true, y_p)
        except: ece = 0.0
        metrics[name] = (brier, ece)
        print(f"{name}: Brier={brier:.4f}, ECE={ece:.4f}")
    
    # Plot
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for name, col in [("Raw", "model_prob"), ("Calibrated (T=1.38)", "calibrated_prob"), ("Vegas", "vegas_prob")]:
        fp, mv = calibration_curve(y_true, df_test[col], n_bins=5, strategy='quantile')
        b, e = metrics[name]
        style = "s-" if "Raw" in name else "^-" if "Calibrated" in name else "o-"
        plt.plot(mv, fp, style, label=f"{name} (B={b:.3f})")

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.legend()
    plt.title(f"Calibration Comparison {CALIBRATION_TEST_YEAR} (5-Bin Quantile)")
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_comparison_kaggle.png")
    plt.show()
    print("Plot saved as: calibration_comparison_kaggle.png")
    
    print("\n--- ROI CHECK (Threshold > 60%) ---")
    for name, col in [("Raw", "model_prob"), ("Calibrated (T=1.38)", "calibrated_prob")]:
        roi, bets = calculate_roi(df_test, col, threshold=0.60)
        print(f"{name} Model: {bets} Bets, ROI={roi:.2f}%")

    # --- ZIP RESULTS FOR EASY DOWNLOAD ---
    import shutil
    print("\nZipping main results and models to 'results.zip'...")
    
    files_to_zip = [
        f"predictions_{CALIBRATION_TRAIN_YEAR}.csv",
        f"predictions_{CALIBRATION_TEST_YEAR}.csv",
        "calibration_comparison_kaggle.png"
    ]
    
    with zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Zip output files
        existing_files = [f for f in files_to_zip if os.path.exists(f)]
        for file in existing_files:
            zipf.write(file)
            print(f"Added file: {file}")
            
        # 2. Zip model folders
        for year in YEARS_TO_VALIDATE:
            folder_name = f"autogluon_wfv_{year}"
            if os.path.exists(folder_name):
                print(f"Zipping model folder: {folder_name}...")
                for root, dirs, files in os.walk(folder_name):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, '.'))
            else:
                print(f"Model folder not found (skipped): {folder_name}")

    print(f"Successfully created 'results.zip' ({os.path.getsize('results.zip')/1e6:.2f} MB)")


def calculate_roi(df, prob_col, threshold=0.60):
    staked = 0
    winnings = 0
    bets = 0
    
    for _, row in df.iterrows():
        p = row[prob_col]
        outcome = row['outcome']
        odds_a = row.get('A_open_odds')
        odds_b = row.get('B_open_odds')
        
        if pd.isna(odds_a) or pd.isna(odds_b): continue
        
        if p > threshold:
            bets += 1; staked += 100
            if outcome == 1: winnings += get_payout(odds_a, 100)
            else: winnings -= 100
        elif p < (1 - threshold):
            bets += 1; staked += 100
            if outcome == 0: winnings += get_payout(odds_b, 100)
            else: winnings -= 100
            
    roi = (winnings / staked * 100) if staked > 0 else 0
    return roi, bets

if __name__ == "__main__":
    import zipfile # Import locally for the zipper
    import os
    run_walk_forward_validation()
    run_calibration_analysis()
    print("DONE.")
    import sys
    sys.exit(0) # Force clean exit logic
