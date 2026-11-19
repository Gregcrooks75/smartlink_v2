"""Score new meters and export review files using full reference snapshots."""

from pathlib import Path

from mtx.pipeline.preparation import PreparedInputs
from mtx.prediction.recommendations import run_recommendations_stage


def run_recommendations(
    inputs: PreparedInputs, features, project_root: Path, model: Path, summary
):
    print("\n16. Run Predictions...")
    if summary is None:
        raise RuntimeError(
            "No test score gap meets the requested precision/volume. "
            "Review test_thresholds.csv before generating recommendations."
        )
    return run_recommendations_stage(
        project_root=project_root,
        model_date=model.name,
        min_score_gap=summary["min_score_gap"],
        ami_df=inputs.ami,
        full_ami_df=inputs.full_ami,
        transformer_df=inputs.full_transformers,
        final_feature_files=features,
        include_previously_sent=True,
    )
