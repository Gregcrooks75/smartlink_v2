"""Load the four source snapshots and the two division layers."""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from mtx.ingestion.adam_cache import load_or_refresh_adam_data
from mtx.ingestion.ami_cache import load_or_refresh_ny_ami
from mtx.ingestion.itron_group_cache import load_or_refresh_itron_groups
from mtx.ingestion.itron_groups import build_itron_tx_subgroups
from mtx.ingestion.transformer_cache import load_or_refresh_ny_transformers
from mtx.run_config import RunConfig


@dataclass
class Inputs:
    ami: pd.DataFrame
    transformers: pd.DataFrame
    itron: pd.DataFrame
    divisions: gpd.GeoDataFrame
    adam: pd.DataFrame
    full_ami: pd.DataFrame


def run_ingestion(
    config: RunConfig, project_root: Path, *, cache_only: bool = False
) -> Inputs:
    print("\n1. Loading Inputs...")
    if cache_only:
        # Never query SAP/GIS, refresh a cache, or build new ITRON groups.
        cache_root = Path(project_root) / "data" / "cache"
        paths = {
            "AMI": cache_root / "ami" / "ny_ami.parquet",
            "transformers": cache_root / "transformers" / "ny_transformers.parquet",
            "ITRON": cache_root / "itron" / "itron_tx_subgroups.parquet",
            "Adam": cache_root / "adam" / "adam_validated.parquet",
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {name} cache: {path}. Run the upstream pipeline first."
                )
        ami, transformers, itron, adam = (
            pd.read_parquet(path) for path in paths.values()
        )
        for name, frame in zip(paths, (ami, transformers, itron, adam)):
            if frame.empty and name != "Adam":
                raise ValueError(f"{name} cache is empty: {paths[name]}")
    else:
        ami = load_or_refresh_ny_ami(force_refresh=config.force_ami_refresh)
        transformers = load_or_refresh_ny_transformers(
            force_refresh=config.force_transformer_refresh
        )
        itron = load_or_refresh_itron_groups(
            refresh_fn=build_itron_tx_subgroups,
            force_refresh=config.force_itron_refresh,
            max_age_days=1,
            required_columns=(
                "meter_number",
                "Tx_Subgroup_ID",
                "tx_group_id",
                "tx_group_size",
                "group_source",
                "group_type",
            ),
        )
        adam = load_or_refresh_adam_data(force_refresh=False)
    full_ami = ami.copy()

    print("\n2. Loading Division Polygons...")
    directory = project_root / "data" / "data augmentation"
    nyseg = gpd.read_file(directory / "NYSEG_Electric_Divisions.shp").rename(
        columns={"DIVISION": "Division"}
    )
    rge = gpd.read_file(directory / "RGE_Electric_Divisions.shp").rename(
        columns={"DISTRICT": "Division"}
    )
    divisions = gpd.GeoDataFrame(
        pd.concat(
            [nyseg[["Division", "geometry"]], rge[["Division", "geometry"]]],
            ignore_index=True,
        ),
        geometry="geometry",
    ).to_crs(epsg=4326)
    transformers = transformers.loc[
        transformers["StructureType"] != "Vert"
    ].reset_index(drop=True)
    return Inputs(ami, transformers, itron, divisions, adam, full_ami)
