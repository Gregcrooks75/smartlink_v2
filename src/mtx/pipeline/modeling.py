"""Build reviewed splits, fit the ranker and select the test score gap."""

from datetime import date
from pathlib import Path

import polars as pl

from mtx.pipeline.preparation import PreparedInputs
from mtx.training.dataset import run_training_data_stage
from mtx.training.group_review import run_group_review
from mtx.training.ranker import run_model_training_stage
from mtx.training.test_threshold import select_test_score_gap


def run_modeling(inputs: PreparedInputs, features, project_root: Path, run_date: date):
    print("\n12. Building Training Data...")
    training = run_training_data_stage(
        project_root=project_root,
        run_date=run_date.isoformat(),
        adam_labels=pl.from_pandas(inputs.adam),
        current_ami=pl.DataFrame(
            inputs.ami[["meter_number", "current_meter_floc"]].to_dict(orient="list"),
            infer_schema_length=None,
        ),
        ami_meter_col="meter_number",
        ami_transformer_col="current_meter_floc",
        subgroup_col="tx_group_id",
        quality_col="tx_current_transformer_count",
        exclude_previous_recommendations=False,
        final_feature_files=features,
    )
    print(f"Training data saved to: {training}")
    print("\n13. Training Transformer Ranker...")
    model = run_model_training_stage(
        project_root=project_root,
        training_directory=training,
        exclude_features=(),
        evaluate_test=True,
    )
    print("\n14. Reviewing Model Corrections...")
    summary = select_test_score_gap(
        model, training, target_precision=0.88, min_changes=100
    )
    print("\n15. Run Group Review...")
    run_group_review(model)
    return model, summary
