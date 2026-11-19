"""Select a #1-vs-#2 transformer score-gap cutoff using TEST labels.

WARNING: Test precision is optimized in-sample; it is NOT an unbiased
estimate of deployment performance. Use a separate untouched holdout for that.

Standalone:
  python scripts/select_test_score_gap.py PATH_TO_MODEL_DIR --target-precision 0.88 --min-changes 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def select_test_score_gap(
    model_directory,
    training_directory=None,  # compatibility with the pipeline signature
    *,
    target_precision: float = 0.88,
    min_changes: int = 100,
    max_distance_ft: float = 550,
):
    if not (0 < target_precision <= 1):
        raise ValueError("target_precision must be in (0, 1].")
    if min_changes < 1:
        raise ValueError("min_changes must be at least 1.")

    folder = Path(model_directory)
    path = folder / "test_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; train with evaluate_test=True.")
    predictions = pd.read_parquet(path)
    required = {
        "meter_number", "recommended_transformer", "current_transformer",
        "verified_transformer", "score_gap", "recommended_distance_ft",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing test prediction columns: {sorted(missing)}")
    if predictions.empty or predictions["meter_number"].isna().any() or predictions["meter_number"].duplicated().any():
        raise ValueError("Expected one non-null, unique meter_number per test prediction.")

    score_gap = pd.to_numeric(predictions["score_gap"], errors="coerce")
    distance = pd.to_numeric(predictions["recommended_distance_ft"], errors="coerce")
    truth_known = predictions["verified_transformer"].notna()
    top1_correct = predictions["recommended_transformer"].eq(predictions["verified_transformer"])
    current_correct = predictions["current_transformer"].eq(predictions["verified_transformer"])
    eligible = (
        predictions["current_transformer"].notna()
        & truth_known
        & predictions["recommended_transformer"].notna()
        & predictions["recommended_transformer"].ne(predictions["current_transformer"])
        & np.isfinite(score_gap)
        & score_gap.gt(0)
        & distance.between(0, max_distance_ft)
    )
    changes = predictions.loc[eligible].copy()
    changes["score_gap"] = score_gap.loc[eligible]
    changes["correct"] = top1_correct.loc[eligible]
    changes["wrongly_changed"] = current_correct.loc[eligible]

    if changes.empty:
        table = pd.DataFrame(columns=[
            "score_gap", "proposed_changes", "correct_changes",
            "correctly_assigned_meters_wrongly_changed", "precision",
        ])
    else:
        table = (
            changes.groupby("score_gap")
            .agg(
                proposed_changes=("correct", "size"),
                correct_changes=("correct", "sum"),
                correctly_assigned_meters_wrongly_changed=("wrongly_changed", "sum"),
            )
            .sort_index(ascending=False)
            .cumsum()
            .reset_index()
        )
        table["precision"] = table["correct_changes"] / table["proposed_changes"]

    table.to_csv(folder / "test_thresholds.csv", index=False)
    qualifying = table.loc[
        table["precision"].ge(target_precision)
        & table["proposed_changes"].ge(min_changes)
    ]
    # Choose the lowest eligible cutoff to maximize selected corrections.
    threshold = float(qualifying.iloc[-1]["score_gap"]) if not qualifying.empty else None
    settings = {
        "min_score_gap": threshold,
        "threshold_selected_on": "test",
        "target_precision": target_precision,
        "min_changes": min_changes,
        "max_distance_ft": max_distance_ft,
        "warning": "Test labels were used to tune this cutoff; test precision is in-sample.",
    }
    (folder / "correction_filter.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    selected = changes.loc[changes["score_gap"].ge(threshold)].copy() if threshold is not None else changes.iloc[0:0].copy()
    selected.to_parquet(folder / "test_selected_corrections.parquet", index=False)
    correct = int(selected["correct"].sum())
    incorrectly_assigned = int((truth_known & predictions["current_transformer"].notna() & ~current_correct).sum())
    summary = {
        **settings,
        "split": "test",
        "meters": int(len(predictions)),
        "labeled_meters": int(truth_known.sum()),
        "top1_accuracy": float(top1_correct.loc[truth_known].mean()) if truth_known.any() else None,
        "proposed_changes": int(len(selected)),
        "correct_changes": correct,
        "incorrect_changes": int(len(selected) - correct),
        "precision": correct / len(selected) if len(selected) else None,
        "incorrect_meters_recovered": correct / incorrectly_assigned if incorrectly_assigned else None,
        "correctly_assigned_meters_wrongly_changed": int(selected["wrongly_changed"].sum()),
    }
    (folder / "test_correction_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (folder / "test_tuned_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(folder / "test_tuned_report.csv", index=False)
    print(json.dumps(summary, indent=2))
    if threshold is None:
        print("No test cutoff met the requested precision and minimum change count.")
        return None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--target-precision", type=float, default=0.88)
    parser.add_argument("--min-changes", type=int, default=100)
    parser.add_argument("--max-distance-ft", type=float, default=550)
    args = parser.parse_args()
    select_test_score_gap(
        args.model_directory,
        target_precision=args.target_precision,
        min_changes=args.min_changes,
        max_distance_ft=args.max_distance_ft,
    )


if __name__ == "__main__":
    main()
