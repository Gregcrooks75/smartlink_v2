"""Build division groups, transformer candidates and regular features."""

from datetime import date
from pathlib import Path

from mtx.candidates.spatial import run_transformer_candidate_stage
from mtx.features.regular import run_candidate_feature_stage
from mtx.grouping.daily import run_meter_grouping_stage
from mtx.history.grouping_history import track_meter_grouping_history
from mtx.pipeline.preparation import PreparedInputs
from mtx.run_config import RunConfig


def run_spatial_processing(
    inputs: PreparedInputs, config: RunConfig, project_root: Path, run_date: date
):
    print("\n9. Building Meter Groups...")
    groups = run_meter_grouping_stage(
        config=config, ami_df=inputs.ami, itron_groups_df=inputs.itron
    )
    changes = track_meter_grouping_history(group_files=groups, ami_df=inputs.ami)
    print(f"Grouping-result changes detected: {len(changes):,}")

    print("\n10. Building Transformer Candidates...")
    candidates = run_transformer_candidate_stage(
        config=config,
        ami_df=inputs.ami,
        transformer_df=inputs.transformers,
    )
    print("\n11. Building Regular Final Features...")
    return run_candidate_feature_stage(
        config=config,
        candidate_files=candidates,
        group_files=groups,
        ami_df=inputs.ami,
        transformer_df=inputs.transformers,
        snapshot_date=run_date,
        cache_dir=project_root / "data" / "cache" / "final_features",
    )
