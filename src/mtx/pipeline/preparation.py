"""Standardize, reconcile history, then reject invalid coordinates."""

from dataclasses import dataclass

import pandas as pd

from mtx.data.coordinates import save_rejection_log, validate_coordinates
from mtx.data.standardization import standardize_all_inputs
from mtx.history.adam_history import track_adam_history
from mtx.history.ami_history import track_ami_history
from mtx.history.corrections import run_historical_corrections
from mtx.history.itron_history import track_grouping_history
from mtx.history.transformer_history import track_transformer_history
from mtx.pipeline.ingestion import Inputs


@dataclass
class PreparedInputs:
    ami: pd.DataFrame
    transformers: pd.DataFrame
    itron: pd.DataFrame
    adam: pd.DataFrame
    full_ami: pd.DataFrame
    full_transformers: pd.DataFrame


def run_preparation(
    inputs: Inputs, *, track_history: bool = True
) -> PreparedInputs:
    print("\n3. Standardizing Inputs...")
    ami, transformers, itron, divisions, adam = standardize_all_inputs(
        ami_df=inputs.ami,
        transformer_df=inputs.transformers,
        itron_groups_df=inputs.itron,
        division_gdf=inputs.divisions,
        adam_df=inputs.adam,
    )
    full_transformers = transformers.copy()

    if track_history:
        print("\n4. Reconciling Historical Recommendations...")
        run_historical_corrections(ami_df=ami)
        print("\n5. Tracking History...")
        for name, track, frame in (
            ("AMI", track_ami_history, ami),
            ("Adam", track_adam_history, adam),
            ("ITRON", track_grouping_history, itron),
            ("Transformer", track_transformer_history, transformers),
        ):
            changes = track(current_df=frame)
            print(f"{name} changes detected: {len(changes):,}")

    print("\n6. Validating Coordinates...")
    ami, invalid_meters = validate_coordinates(df=ami, division_gdf=divisions)
    transformers, invalid_transformers = validate_coordinates(
        df=transformers, division_gdf=divisions
    )
    print(f"Valid meters: {len(ami):,}")
    print(f"Rejected meters: {len(invalid_meters):,}")
    print("\n7. Saving Rejection Log...")
    if track_history:
        save_rejection_log(
            invalid_meters=invalid_meters, invalid_transformers=invalid_transformers
        )
    meters = invalid_meters["meter_number"]
    structures = invalid_transformers["transformer_floc"]
    print(f"Rejected meters to omit: {len(meters):,}")
    print(f"Rejected transformers to omit: {len(structures):,}")
    return PreparedInputs(
        ami=ami.loc[~ami["meter_number"].isin(meters)].reset_index(drop=True),
        transformers=transformers.loc[
            ~transformers["transformer_floc"].isin(structures)
        ].reset_index(drop=True),
        itron=itron.loc[~itron["meter_number"].isin(meters)].reset_index(drop=True),
        adam=adam.loc[~adam["meter_number"].isin(meters)].reset_index(drop=True),
        full_ami=inputs.full_ami,
        full_transformers=full_transformers,
    )
