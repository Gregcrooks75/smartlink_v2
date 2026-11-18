from pathlib import Path
import json
import pandas as pd


def review_corrections(
    model_directory,
    training_directory,
    target_precision=0.95,
    min_changes=100,
    evaluate_test=False,
):
    folder = Path(model_directory)
    settings_file = folder / "correction_filter.json"
    split = "test" if evaluate_test else "validation"

    if evaluate_test:
        settings = json.loads(settings_file.read_text())
        threshold = settings["min_score_gap"]

        if threshold is None:
            raise ValueError("No validation threshold was approved.")

        prediction_file = folder / "test_predictions.parquet"
        if not prediction_file.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_file}. Run model training with evaluate_test=True first."
            )
        predictions = pd.read_parquet(prediction_file)

    else:
        predictions = pd.read_parquet(
            folder / "validation_predictions.parquet"
        )

    if predictions.empty or predictions["meter_number"].isna().any() or predictions["meter_number"].duplicated().any():
        raise ValueError(f"{split}: expected one prediction per unique non-null meter")
    required = {"recommended_transformer", "current_transformer", "verified_transformer", "score_gap", "recommended_distance_ft"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{split}: prediction columns missing: {sorted(missing)}")
    correct_all = predictions["recommended_transformer"].eq(predictions["verified_transformer"])

    known = predictions["current_transformer"].notna()

    was_correct = predictions["current_transformer"].eq(
        predictions["verified_transformer"]
    )

    incorrect_meters = int((known & ~was_correct).sum())

    changes = predictions.loc[
        known
        & predictions["recommended_transformer"].ne(
            predictions["current_transformer"]
        )
        & predictions["score_gap"].gt(0)
        & predictions["recommended_distance_ft"].between(0, 550)
    ].copy()

    changes["correct"] = changes["recommended_transformer"].eq(
        changes["verified_transformer"]
    )

    changes["wrongly_changed"] = changes["current_transformer"].eq(
        changes["verified_transformer"]
    )

    if not evaluate_test:
        table = (
            changes.groupby("score_gap")
            .agg(
                proposed_changes=("correct", "size"),
                correct_changes=("correct", "sum"),
                correctly_assigned_meters_wrongly_changed=(
                    "wrongly_changed",
                    "sum",
                ),
            )
            .sort_index(ascending=False)
            .cumsum()
            .reset_index()
        )

        table["precision"] = (
            table["correct_changes"] / table["proposed_changes"]
        )

        table.to_csv(
            folder / "validation_thresholds.csv",
            index=False,
        )

        eligible = table.loc[
            table["precision"].ge(target_precision)
            & table["proposed_changes"].ge(min_changes)
        ]

        threshold = (
            None
            if eligible.empty
            else float(eligible.iloc[-1]["score_gap"])
        )

        settings_file.write_text(
            json.dumps(
                {
                    "min_score_gap": threshold,
                    "target_precision": target_precision,
                    "min_changes": min_changes,
                },
                indent=2,
            )
        )

    if threshold is None:
        print("No cutoff meets the validation target and minimum volume.")
        return None

    selected = changes.loc[
        changes["score_gap"].ge(threshold)
    ]

    correct = int(selected["correct"].sum())

    summary = {
        "split": split,
        "threshold_selected_on": "validation",
        "meters": len(predictions),
        "correct_predictions": int(correct_all.sum()),
        "incorrect_predictions": int((~correct_all).sum()),
        "top1_accuracy": float(correct_all.mean()),
        "precision_definition": "Correct selected changes / all selected changes, within 550 ft and above the fixed validation cutoff",
        "min_score_gap": threshold,
        "proposed_changes": len(selected),
        "correct_changes": correct,
        "incorrect_changes": len(selected) - correct,
        "precision": (
            correct / len(selected)
            if len(selected)
            else None
        ),
        "incorrect_meters_recovered": (
            correct / incorrect_meters
            if incorrect_meters
            else None
        ),
        "correctly_assigned_meters_wrongly_changed": int(
            selected["wrongly_changed"].sum()
        ),
    }

    (folder / f"{split}_correction_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    selected.to_parquet(
        folder / f"{split}_selected_corrections.parquet",
        index=False,
    )

    if evaluate_test:
        # This is the final report. Do not optimize the cutoff on these outcomes.
        (folder / "final_test_report.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        pd.DataFrame([summary]).to_csv(folder / "final_test_report.csv", index=False)
        precision_text = "N/A (no selected test changes)" if summary["precision"] is None else f"{summary['precision']:.2%}"
        print("\nFINAL TEST RESULTS â€” cutoff fixed on validation")
        print(f"Test accuracy: {summary['top1_accuracy']:.2%} ({summary['correct_predictions']:,}/{summary['meters']:,} meters)")
        print(f"Final test correction precision: {precision_text} ({correct:,}/{len(selected):,} selected changes)")

    print(json.dumps(summary, indent=2))

    return summary