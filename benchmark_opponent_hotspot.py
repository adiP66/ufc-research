from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

import kaggle_pipeline_for_results_2 as pipe


def get_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def derive_base_stats(long_df: pd.DataFrame) -> list[str]:
    ewm_cols = [c for c in long_df.columns if c.endswith("_dec_avg") and "roll3" not in c]
    base_stats = [c.replace("_dec_avg", "") for c in ewm_cols]

    physical_ratios = {"reach_ratio", "height_ratio", "age_ratio"}
    ratio_cols = [
        c
        for c in long_df.columns
        if (c.endswith("_ratio") or c.endswith("_acc") or "_per_" in c)
        and c not in physical_ratios
        and not c.endswith("_dec_avg")
        and not c.endswith("_career_avg")
    ]
    base_stats.extend(ratio_cols)
    return sorted(list(set(base_stats)))


def legacy_step2_merge(long_df: pd.DataFrame, base_stats: list[str]) -> pd.DataFrame:
    df = long_df.copy()

    opp_stats = df[["fight_id", "event_date", "opponent_id"]].copy()
    for base in base_stats:
        if base in df.columns:
            opp_stats[f"allowed_{base}"] = df[base]

    opp_stats = opp_stats.sort_values(["opponent_id", "event_date", "fight_id"])
    grouped_opp = opp_stats.groupby("opponent_id", group_keys=False)
    opp_stats["opp_n_fights"] = grouped_opp.cumcount()

    for base in base_stats:
        col_allowed = f"allowed_{base}"
        if col_allowed not in opp_stats.columns:
            continue

        opp_stats[f"{base}_opp_allowed"] = grouped_opp[col_allowed].transform(
            lambda x: x.shift(1).ewm(alpha=0.15, min_periods=1).mean()
        )
        opp_stats[f"{base}_opp_std"] = grouped_opp[col_allowed].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
        ).fillna(1.0)

    merge_cols = ["fight_id", "opponent_id", "opp_n_fights"] + [
        c for c in opp_stats.columns if c.endswith("_opp_allowed") or c.endswith("_opp_std")
    ]
    opp_stats_subset = opp_stats[merge_cols].drop_duplicates(subset=["fight_id", "opponent_id"])
    return df.merge(opp_stats_subset, on=["fight_id", "opponent_id"], how="left")


def mapped_step2(long_df: pd.DataFrame, base_stats: list[str]) -> pd.DataFrame:
    df = long_df.copy()

    available_base_stats = [base for base in base_stats if base in df.columns]
    opp_order = df.sort_values(["opponent_id", "event_date", "fight_id"]).index
    opp_history = df.loc[opp_order, ["opponent_id"] + available_base_stats].copy()
    grouped_opp = opp_history.groupby("opponent_id", group_keys=False)

    opp_history["opp_n_fights"] = grouped_opp.cumcount()
    df["opp_n_fights"] = 0.0
    df.loc[opp_order, "opp_n_fights"] = opp_history["opp_n_fights"].to_numpy()

    for base in available_base_stats:
        opp_allowed_hist = grouped_opp[base].transform(
            lambda x: x.shift(1).ewm(alpha=0.15, min_periods=1).mean()
        )
        opp_std_hist = grouped_opp[base].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
        ).fillna(1.0)

        allowed_col = f"{base}_opp_allowed"
        std_col = f"{base}_opp_std"
        df[allowed_col] = np.nan
        df[std_col] = np.nan
        df.loc[opp_order, allowed_col] = opp_allowed_hist.to_numpy()
        df.loc[opp_order, std_col] = opp_std_hist.to_numpy()

    return df


def benchmark(name: str, fn, long_df: pd.DataFrame, base_stats: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    gc.collect()
    rss_before = get_rss_mb()

    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(long_df, base_stats)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = get_rss_mb()

    return out, {
        "seconds": elapsed,
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before if not np.isnan(rss_before) and not np.isnan(rss_after) else float("nan"),
    }


def summarize_equivalence(legacy_df: pd.DataFrame, mapped_df: pd.DataFrame, base_stats: list[str]) -> None:
    print("\n=== SANITY CHECK (legacy vs mapped outputs) ===")
    checks = ["opp_n_fights"]
    for base in base_stats[:8]:
        checks.extend([f"{base}_opp_allowed", f"{base}_opp_std"])

    checked = 0
    for col in checks:
        if col in legacy_df.columns and col in mapped_df.columns:
            a = pd.to_numeric(legacy_df[col], errors="coerce")
            b = pd.to_numeric(mapped_df[col], errors="coerce")
            diff = (a - b).abs().fillna(0.0)
            print(f"{col}: max_abs_diff={diff.max():.12f}, mean_abs_diff={diff.mean():.12f}")
            checked += 1

    if checked == 0:
        print("No overlapping comparison columns found.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "ufc_fights_full_with_odds_updated.csv",
        root / "ufc_fights_full_with_odds.csv",
    ]
    dataset = next((p for p in candidates if p.exists()), None)
    if dataset is None:
        raise FileNotFoundError("Could not locate dataset CSV in workspace root.")

    print(f"Using dataset: {dataset}")
    raw_df = pd.read_csv(dataset)

    df = pipe._clean_and_sort(raw_df)
    df = pipe._apply_strict_filters(df)
    long_df = pipe._convert_to_long_format(df)
    long_df = pipe._compute_history_features(long_df)
    long_df = pipe._compute_ratio_features(long_df)
    long_df = long_df.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    base_stats = derive_base_stats(long_df)
    print(f"Prepared long_df shape: {long_df.shape}, base_stats: {len(base_stats)}")

    legacy_df, legacy_metrics = benchmark("legacy_merge", legacy_step2_merge, long_df, base_stats)
    mapped_df, mapped_metrics = benchmark("mapped_keys", mapped_step2, long_df, base_stats)

    print("\n=== BENCHMARK RESULTS (hotspot step only) ===")
    print(f"legacy_merge: {legacy_metrics}")
    print(f"mapped_keys: {mapped_metrics}")

    if legacy_metrics["seconds"] > 0:
        speedup = legacy_metrics["seconds"] / mapped_metrics["seconds"]
        print(f"Speedup (legacy/mapped): {speedup:.2f}x")

    if legacy_metrics["tracemalloc_peak_mb"] > 0:
        peak_ratio = mapped_metrics["tracemalloc_peak_mb"] / legacy_metrics["tracemalloc_peak_mb"]
        print(f"Peak memory ratio (mapped/legacy): {peak_ratio:.3f}")

    summarize_equivalence(legacy_df, mapped_df, base_stats)


if __name__ == "__main__":
    main()
