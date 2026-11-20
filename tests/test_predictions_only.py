"""Focused checks for the opt-in read-only cached recommendation path."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mtx.pipeline.cached_predictions import load_cached_prediction_artifacts


def _artifacts(tmp_path, *, cutoff=0.03, selection="test"):
    day = date.today().isoformat()
    model_dir = tmp_path / "data/processed/models/transformer_ranker/2026-10-20"
    model_dir.mkdir(parents=True)
    (model_dir / "model.ubj").write_bytes(b"existing trained model")
    (model_dir / "metadata.json").write_text(
        json.dumps({"training_date": "2026-10-20"}), encoding="utf-8"
    )
    (model_dir / "correction_filter.json").write_text(
        json.dumps({"threshold_selected_on": selection, "min_score_gap": cutoff}),
        encoding="utf-8",
    )
    features = tmp_path / "data/cache/final_features" / day
    features.mkdir(parents=True)
    output = features / "Ithaca.parquet"
    output.write_bytes(b"already-generated features")
    (features / "_complete.json").write_text(
        json.dumps({
            "date": day,
            "divisions": {"Ithaca": {"sha256": hashlib.sha256(output.read_bytes()).hexdigest()}},
        }),
        encoding="utf-8",
    )
    return model_dir, output


def test_resolves_saved_test_threshold_and_daily_features(tmp_path):
    model, path = _artifacts(tmp_path)
    actual, summary, files = load_cached_prediction_artifacts(tmp_path)
    assert actual == model
    assert summary == {"min_score_gap": 0.03}
    assert files == {"Ithaca": path}


@pytest.mark.parametrize(
    ("cutoff", "selection"),
    [(None, "test"), (0.02, "validation"), (-0.1, "test")],
)
def test_refuses_missing_non_test_or_negative_cutoff(tmp_path, cutoff, selection):
    _artifacts(tmp_path, cutoff=cutoff, selection=selection)
    with pytest.raises(ValueError, match="cutoff|score-gap"):
        load_cached_prediction_artifacts(tmp_path)


def test_refuses_missing_or_modified_daily_features(tmp_path):
    _, path = _artifacts(tmp_path)
    path.write_bytes(b"stale or modified features")
    with pytest.raises(ValueError, match="differ"):
        load_cached_prediction_artifacts(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing cached features"):
        load_cached_prediction_artifacts(tmp_path)


def test_does_not_fall_back_to_older_features(tmp_path):
    _artifacts(tmp_path)
    other_day = date(2026, 11, 19)
    with pytest.raises(FileNotFoundError, match="_complete.json"):
        load_cached_prediction_artifacts(tmp_path, run_date=other_day)


def test_cached_ingestion_does_not_refresh_data(tmp_path, monkeypatch):
    from mtx.pipeline import ingestion
    from mtx.run_config import RunConfig
    import geopandas as gpd

    paths = {
        "ami/ny_ami.parquet": pd.DataFrame({"MeterID": ["1"]}),
        "transformers/ny_transformers.parquet": pd.DataFrame({
            "FuncLoc": ["TX"], "StructureType": ["Pole"]
        }),
        "itron/itron_tx_subgroups.parquet": pd.DataFrame({"meter_number": ["1"]}),
        "adam/adam_validated.parquet": pd.DataFrame({"MeterID": ["1"]}),
    }
    for relative, frame in paths.items():
        path = tmp_path / "data/cache" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("database or cache refresh attempted")

    monkeypatch.setattr(ingestion, "load_or_refresh_ny_ami", forbidden)
    monkeypatch.setattr(ingestion, "load_or_refresh_ny_transformers", forbidden)
    monkeypatch.setattr(ingestion, "load_or_refresh_itron_groups", forbidden)
    monkeypatch.setattr(ingestion, "load_or_refresh_adam_data", forbidden)
    polygons = gpd.GeoDataFrame(
        {"DIVISION": ["Ithaca"], "DISTRICT": ["Ithaca"]},
        geometry=gpd.points_from_xy([-76.0], [42.0]), crs="EPSG:4326"
    )
    monkeypatch.setattr(ingestion.gpd, "read_file", lambda _: polygons)
    result = ingestion.run_ingestion(RunConfig(), tmp_path, cache_only=True)
    assert result.full_ami.MeterID.tolist() == ["1"]
    assert result.transformers.FuncLoc.tolist() == ["TX"]


def test_cached_preparation_skips_all_history_writes(monkeypatch):
    from mtx.pipeline import preparation

    ami = pd.DataFrame({"meter_number": ["1"], "Division": ["Ithaca"]})
    tx = pd.DataFrame({"transformer_floc": ["TX"], "Division": ["Ithaca"]})
    itron = pd.DataFrame({"meter_number": ["1"], "Division": ["Ithaca"]})
    adam = pd.DataFrame({"meter_number": ["1"], "Division": ["Ithaca"]})
    input_frames = SimpleNamespace(ami=ami, transformers=tx, itron=itron,
                                   divisions=pd.DataFrame(), adam=adam, full_ami=ami)

    monkeypatch.setattr(
        preparation, "standardize_all_inputs",
        lambda **kwargs: (ami, tx, itron, pd.DataFrame(), adam)
    )
    monkeypatch.setattr(
        preparation, "validate_coordinates",
        lambda df, division_gdf: (df, df.iloc[0:0])
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("historical state was modified")
    monkeypatch.setattr(preparation, "run_historical_corrections", forbidden)
    monkeypatch.setattr(preparation, "track_adam_history", forbidden)
    monkeypatch.setattr(preparation, "save_rejection_log", forbidden)
    result = preparation.run_preparation(input_frames, track_history=False)
    assert result.ami.meter_number.tolist() == ["1"]
    assert result.full_transformers.transformer_floc.tolist() == ["TX"]


def test_predictions_only_skips_modeling_and_optional_final_review(monkeypatch, tmp_path):
    import mtx.config
    from scripts import run_meter_to_transformer_pipeline as entry
    from mtx.prediction import final_review

    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mtx.config, "PROJECT_ROOT", tmp_path)
    actions = []
    model = tmp_path / "data/processed/models/transformer_ranker/2026-10-20"
    features = {"Ithaca": tmp_path / "data/cache/final_features/day/Ithaca.parquet"}
    monkeypatch.setattr(
        entry, "load_cached_prediction_artifacts",
        lambda *args, **kwargs: (model, {"min_score_gap": 0.03}, features)
    )
    monkeypatch.setattr(
        entry, "run_ingestion",
        lambda *args, **kwargs: actions.append(("ingestion", kwargs)) or object()
    )
    monkeypatch.setattr(
        entry, "run_preparation",
        lambda *args, **kwargs: actions.append(("preparation", kwargs)) or object()
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("upstream rebuild invoked")
    monkeypatch.setattr(entry, "run_spatial_processing", forbidden)
    monkeypatch.setattr(entry, "run_modeling", forbidden)
    monkeypatch.setattr(
        entry, "run_recommendations",
        lambda *args: actions.append(("predictions", args[2], args[3], args[4]))
    )
    monkeypatch.setattr(
        final_review, "run_final_review_stage",
        lambda: actions.append(("final_review",))
    )
    entry.main(["--predictions-only"])
    assert [item[0] for item in actions] == ["ingestion", "preparation", "predictions"]
    assert actions[0][1] == {"cache_only": True}
    assert actions[1][1] == {"track_history": False}
    assert actions[2][3] == {"min_score_gap": 0.03}
    actions.clear()
    entry.main(["--predictions-only", "--final-review"])
    assert [item[0] for item in actions] == [
        "ingestion", "preparation", "predictions", "final_review"
    ]


def test_no_flags_preserves_full_pipeline(monkeypatch):
    from scripts import run_meter_to_transformer_pipeline as entry
    actions = []
    monkeypatch.setattr(entry, "run_ingestion",
                        lambda *args: actions.append("ingestion") or object())
    monkeypatch.setattr(entry, "run_preparation",
                        lambda *args: actions.append("preparation") or object())
    monkeypatch.setattr(entry, "run_spatial_processing",
                        lambda *args: actions.append("spatial") or {"Ithaca": "features"})
    monkeypatch.setattr(entry, "run_modeling",
                        lambda *args: actions.append("modeling") or (Path("model"), {"min_score_gap": 0.1}))
    monkeypatch.setattr(entry, "run_recommendations",
                        lambda *args: actions.append("predictions"))
    entry.main([])
    assert actions == ["ingestion", "preparation", "spatial", "modeling", "predictions"]
