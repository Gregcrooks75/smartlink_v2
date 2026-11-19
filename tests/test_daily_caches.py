from dataclasses import replace
import json

import pandas as pd
import pytest

from mtx.candidates import spatial
from mtx.features.regular import run_candidate_feature_stage
from mtx.grouping import daily
from mtx.run_config import RunConfig


def test_changed_itron_or_radius_refreshes_features(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "GROUPING_CACHE_DIRECTORY", tmp_path / "groups")
    ami = pd.DataFrame(
        dict(
            meter_number=["01", "02"],
            Division="Ithaca",
            latitude=[42.0, 42.00003],
            longitude=-76.0,
            current_meter_floc=["X", "Y"],
        )
    )
    tx = pd.DataFrame(
        dict(
            transformer_floc=["X", "Y"],
            Division="Ithaca",
            latitude=[42.0, 42.001],
            longitude=-76.0,
        )
    )
    itron = pd.DataFrame(dict(meter_number=["01", "02"], Tx_Subgroup_ID=["G", "G"]))
    config = RunConfig(transformer_division_col="Division")

    def run(config, memberships):
        groups = daily.run_meter_grouping_stage(ami, memberships, config)
        pairs = spatial.run_transformer_candidate_stage(
            config, ami, tx, cache_dir=tmp_path / "candidates"
        )
        features = run_candidate_feature_stage(
            config=config,
            candidate_files=pairs,
            group_files=groups,
            ami_df=ami,
            transformer_df=tx,
            cache_dir=tmp_path / "features",
        )
        return [mapping["Ithaca"] for mapping in (groups, pairs, features)]

    paths = run(config, itron)
    mtimes = [p.stat().st_mtime_ns for p in paths]
    assert pd.read_parquet(paths[2]).tx_group_size.tolist() == [2] * 4
    run(config, itron.iloc[::-1])
    assert mtimes == [p.stat().st_mtime_ns for p in paths]
    itron.loc[1, "Tx_Subgroup_ID"] = None
    run(config, itron)
    assert pd.read_parquet(paths[2]).tx_group_size.tolist() == [1] * 4
    assert paths[1].stat().st_mtime_ns == mtimes[1]
    narrow = replace(config, transformer_candidate_radius_ft=100)
    run(narrow, itron)
    assert set(pd.read_parquet(paths[2]).candidate_transformer) == {"X"}
    day = paths[1].parents[1]
    for folder in ("meters", "transformers", "no_candidates"):
        assert (
            pd.read_parquet(day / folder / "Ithaca.parquet").search_radius_ft == 100
        ).all()
    assert (
        json.loads((day / "_complete.json").read_text())["totals"]["candidate_pairs"]
        == 2
    )

    # Fail AFTER the pairs file is replaced and retry at the old radius.
    write = spatial.save_parquet

    def interrupt(frame, path):
        if path.parent.name == "meters":
            raise OSError("interrupted write")
        write(frame, path)

    with monkeypatch.context() as m:
        m.setattr(spatial, "save_parquet", interrupt)
        with pytest.raises(OSError, match="interrupted write"):
            run(config, itron)
    assert not (day / "_complete.json").exists()
    run(narrow, itron)
    assert len(pd.read_parquet(paths[1])) == 2
    assert json.loads((day / "_complete.json").read_text())["radius_ft"] == 100
