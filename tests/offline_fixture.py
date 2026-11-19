"""Deterministic, invented source snapshots for the production and baseline runners."""

from contextlib import ExitStack, contextmanager
from datetime import date
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def sources():
    meters, transformers, labels, memberships = [], [], [], []
    # Independent two-transformer sites keep both classes within 550 ft.
    for site in range(200):
        division = "Ithaca" if site < 100 else "Geneva"
        lon = -76 + (site % 10) * 0.006
        lat = 42 + (site // 10) * 0.008
        for side in range(2):
            floc = f"TX{site:03}-{side}"
            other = f"TX{site:03}-{1 - side}"
            y = lat + 0.0009 * side
            transformers.append(
                {
                    "FuncLoc": floc,
                    "SAPID": "S" + floc,
                    "GISID": "G" + floc,
                    "DivName": division,
                    "OPCO": "NYSEG",
                    "StructureType": "Pole",
                    "Location_X": lon,
                    "Location_Y": y,
                    "Transformer_kVA": 50,
                }
            )
            for j in range(5):
                meter = f"M{site:03}{side}{j}"
                current = floc if j == 0 else other
                live = site % 2 == 0 and side == 0 and j == 4
                if site == 4 and not live:
                    current = f"TX{site:03}-0"
                co = None if live and site % 10 == 0 else "C" + meter
                if site % 8 == 2 and (live or (side, j) == (1, 0)):
                    co = f"SHARED{site}"
                subgroup = None if live else "G" + floc
                if site % 8 in (0, 2) and (side, j) == (0, 0):
                    subgroup = None
                if site == 6 and live:
                    subgroup = "G" + floc
                meters.append(
                    {
                        "MeterID": meter,
                        "Division": division,
                        "SuperiorFLOC": current,
                        "LATITUDE": y + 0.00001 * (j + 1),
                        "LONGITUDE": lon,
                        "OPCO": "NYSEG",
                        "ConnectionObject": co,
                        "PremiseID": "P" + meter,
                        "Address": f"{j} Test St",
                        "Circuit": "A",
                        "City": division,
                        "ZipCode": "00000",
                    }
                )
                memberships.append(
                    {
                        "meter_number": meter,
                        "division": division,
                        "current_meter_floc": current,
                        "Tx_Subgroup_ID": subgroup,
                    }
                )
                if not live:
                    labels.append(
                        {
                            "meter_number": meter,
                            "Division": division,
                            "Correct_Transformer_FLOC": floc,
                            "ValidationSource": "MAP",
                            "ValidationDate": date(2020, 1, 1),
                        }
                    )
    # One invalid meter, one invalid transformer and one vertical structure.
    meters.append(dict(meters[0], MeterID="OUTSIDE", LATITUDE=0))
    transformers.extend(
        [
            dict(transformers[0], FuncLoc="OUT", SAPID="OUT", Location_Y=0),
            dict(transformers[0], FuncLoc="VERT", SAPID="VERT", StructureType="Vert"),
        ]
    )
    # Reviewed meters with an unapproved label are excluded from training and scoring.
    labels[0]["ValidationSource"] = "UNAPPROVED"
    return tuple(map(pd.DataFrame, (meters, transformers, memberships, labels)))


def seed_reference(root):
    shapes = root / "data" / "data augmentation"
    shapes.mkdir(parents=True)
    for field, division, name, bounds in (
        ("DIVISION", "Ithaca", "NYSEG", (-77, 41, -75, 42.078)),
        ("DISTRICT", "Geneva", "RGE", (-77, 42.078, -75, 43)),
    ):
        gpd.GeoDataFrame(
            {field: [division]}, geometry=[box(*bounds)], crs="EPSG:4326"
        ).to_file(shapes / f"{name}_Electric_Divisions.shp", index=False)
    history = root / "data" / "reference" / "Historical Corrections"
    history.mkdir(parents=True)
    # An integrated and an outstanding move, including a production meter.
    batches = pd.DataFrame(
        [("M00000", "TX000-1", "TX000-0"), ("M00204", "TX002-1", "TX002-0")],
        columns=["meter_number", "current_floc", "final_recommended_floc"],
    )
    batches.to_csv(history / "All_Batches_Sent.csv", index=False)
    archive = root / "data" / "historical_corrections"
    archive.mkdir()
    batches.to_csv(archive / "batch_model_2_fixture.csv", index=False)


@contextmanager
def offline_sources():
    """Patch only the four database boundaries. All cache I/O is real."""
    ami, transformers, itron, adam = sources()
    with ExitStack() as stack:
        for source, data in (
            ("mtx.ingestion.ami_cache.get_ny_ami_data", ami),
            (
                "mtx.ingestion.transformer_cache.get_devices_with_structures",
                transformers,
            ),
            ("mtx.db.db_AMI.get_itron_meter_data", itron),
            ("mtx.ingestion.adam_cache.get_adam_validated_data", adam),
        ):
            stack.enter_context(patch(source, return_value=data))
        yield


def run_checkout(checkout, root, mode):
    """Run in a child process so import-time paths belong to this fixture."""
    import os
    import sys
    from pathlib import Path

    checkout, root = Path(checkout).resolve(), Path(root).resolve()
    import pyodbc

    # The caller provides a fresh temporary checkout for the original script,
    # whose caches are anchored directly under its own source tree.
    if mode != "resume":
        seed_reference(root)
    os.environ["SMARTLINK_PROJECT_ROOT"] = str(root)
    sys.path[:0] = [str(checkout / "src"), str(checkout / "scripts")]
    with patch.object(
        pyodbc, "connect", side_effect=AssertionError("unexpected database access")
    ):
        import run_meter_to_transformer_pipeline as pipeline
        from meter_to_transformer.prediction import (
            smartlink_final_review as final_review,
        )

        if mode == "original":
            pipeline.PROJECT_ROOT = root
            for key in ("NYSEG_DIVISION_FILE", "RGE_DIVISION_FILE"):
                setattr(
                    pipeline,
                    key,
                    root / "data/data augmentation" / getattr(pipeline, key).name,
                )
            with offline_sources():
                try:
                    pipeline.main()
                except NameError as error:
                    if error.name != "MAP_HTML":
                        raise
                else:
                    raise AssertionError("Expected the original map export to fail")
            final_review.TRANSFORMER_COLUMNS["division"] = "DivName"
        elif mode == "resume":
            # With current caches, no external source should be consulted.
            pipeline.main()
            return
        else:
            with offline_sources():
                pipeline.main()
        final_review.run_final_review_stage()


if __name__ == "__main__":
    import sys

    run_checkout(*sys.argv[1:])
