"""Paired regression against the original pipeline. All writes are under pytest's tmp_path."""

import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from compare_offline_outputs import compare_runs

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = "fee119ec81dbe003b493ad03cafdb8fdc6e9039c"


def run(checkout, root, mode):
    # Ignore user cache overrides, even if this test runs on a production workstation.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SMARTLINK_")}
    env.update(POLARS_MAX_THREADS="2", OMP_NUM_THREADS="2")
    subprocess.run(
        [
            sys.executable,
            Path(__file__).with_name("offline_fixture.py"),
            checkout,
            root,
            mode,
        ],
        cwd=root,
        env=env,
        check=True,
    )


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    run(REPO, root, "candidate")
    return root


def test_paired_outputs(candidate, tmp_path):
    if subprocess.run(
        ["git", "cat-file", "-e", ORIGINAL], cwd=REPO, capture_output=True, check=False
    ).returncode:
        pytest.skip(f"Paired comparison requires original commit {ORIGINAL}")
    original = tmp_path / "main"
    original.mkdir()
    archive = subprocess.check_output(
        ["git", "archive", ORIGINAL, "scripts", "src"], cwd=REPO
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(original)
    run(original, original, "original")
    compared = compare_runs(original / "data", candidate / "data")
    print(f"Matched {compared} baseline files")


def test_exports_and_same_day_resume(candidate):
    data = candidate / "data"
    folder = next((data / "processed/recommendations").glob("????-??-??"))
    rejected = pd.read_parquet(data / "cache/rejected/rejection_history.parquet")
    assert set(zip(rejected.asset_type, rejected.asset_id)) == {
        ("meter", "OUTSIDE"),
        ("transformer", "OUT"),
    }
    manifest = json.loads(
        next(data.glob("cache/candidates/*/_complete.json")).read_text()
    )
    assert manifest["radius_ft"] == 550
    assert manifest["totals"] == {
        "meters": 2000,
        "candidate_pairs": 4000,
        "meters_without_candidates": 0,
    }
    training = json.loads(
        (data / "processed/training/current/summary.json").read_text()
    )
    assert training["meters_by_status"] == {"INCLUDED": 1899, "NO_APPROVED_TRUTH": 1}
    report = json.loads(
        next(
            data.glob("processed/models/transformer_ranker/*/test_tuned_report.json")
        ).read_text()
    )
    assert (
        report["threshold_selected_on"] == "test" and report["proposed_changes"] == 212
    )
    decisions = pd.read_parquet(folder / "all_recommendations.parquet")
    completion = json.loads((folder / "_complete.json").read_text())
    assert completion["min_score_gap"] == report["min_score_gap"]
    assert completion["recommendations"] == len(decisions)
    assert completion["ready"] + completion["held"] == len(decisions)
    assert len(decisions.loc[decisions.recommendation_source.eq("MODEL")]) == 100
    adam = decisions.loc[decisions.recommendation_source.eq("ADAM_REVIEW")]
    assert len(adam) > 0
    assert completion["adam_review_rows"] == len(adam)
    assert completion["adam_ready"] == int(adam.ready_for_export.sum())
    assert adam.adam_validation_source.eq("MAP").all()
    assert adam.index.min() > decisions.loc[
        decisions.recommendation_source.eq("MODEL")
    ].index.max()
    assert not adam.loc[
        adam.review_flags.str.contains("PREVIOUSLY_SENT"), "ready_for_export"
    ].any()
    sheets = pd.ExcelFile(folder / "recommendations.xlsx").sheet_names
    assert "Adam Review" in sheets
    assert decisions.meter_number.is_unique and decisions.recommendation_id.is_unique
    assert not decisions.loc[
        decisions.would_empty_current_transformer, "ready_for_export"
    ].any()
    assert decisions.loc[decisions.ready_for_export, "current_customers_after_final"].gt(0).all()
    assert decisions.recommended_distance_ft.between(0, 550).all()
    html = (folder / "review_map.html").read_text()
    records = json.loads(
        re.search(
            r'<script type="application/json" id="records">(.*?)</script>',
            html,
            flags=re.DOTALL,
        ).group(1)
    )
    assert [
        r["recommendation_id"] for r in records
    ] == decisions.recommendation_id.tolist()
    review = pd.read_parquet(
        next(data.glob("processed/final_review/*/*/all_member_audit.parquet"))
    )
    assert review.status.notna().all()
    assert review.meter_number.is_unique
    assert not review.loc[review.already_sent, "status"].eq("READY_TO_SEND").any()
    sent = pd.read_csv(
        next(data.glob("processed/final_review/*/*/ready_to_send/*.csv"))
    )
    assert set(sent.meter_number) == set(
        review.loc[review.status.eq("READY_TO_SEND"), "meter_number"]
    )
    assert sent.meter_number.is_unique
    checkpoints = {
        p: p.stat().st_mtime_ns for p in (folder / "predictions").glob("*.parquet")
    }
    first = pd.read_parquet(folder / "scoring_summary.parquet")
    assert first.new_predictions.sum() == 101
    assert len(checkpoints) == 2
    run(REPO, candidate, "resume")
    assert checkpoints == {
        p: p.stat().st_mtime_ns for p in (folder / "predictions").glob("*.parquet")
    }
    reused = pd.read_parquet(folder / "scoring_summary.parquet")
    assert reused.new_predictions.sum() == 0 and reused.reused_predictions.sum() == 101
    pd.testing.assert_frame_equal(
        decisions, pd.read_parquet(folder / "all_recommendations.parquet")
    )


def test_prior_unsent_reconsidered_and_rejection_survives_new_id(candidate):
    from datetime import date, timedelta

    archive = candidate / "data/processed/recommendations"
    today = next(archive.glob("????-??-??"))
    yesterday = (date.fromisoformat(today.name) - timedelta(days=1)).isoformat()
    proposals = pd.read_parquet(today / "all_recommendations.parquet")
    old = proposals.loc[proposals.recommendation_source.eq("MODEL") & proposals.ready_for_export].iloc[0]
    tracker_path = archive / "history/recommendation_status.parquet"
    tracker = pd.read_parquet(tracker_path)
    selected = tracker.recommendation_id.eq(old.recommendation_id)
    tracker.loc[selected, ["recommendation_id", "prediction_date", "status"]] = ["older-id", yesterday, "REJECTED"]
    tracker.to_parquet(tracker_path, index=False)
    today.rename(archive / yesterday)
    run(REPO, candidate, "resume")
    refreshed = pd.read_parquet(today / "all_recommendations.parquet")
    row = refreshed.set_index("meter_number").loc[old.meter_number]
    assert row.recommendation_source == "MODEL" and row.recommended_transformer == old.recommended_transformer
    assert "PREVIOUSLY_REJECTED" in row.review_flags and not row.ready_for_export
    assert refreshed.loc[refreshed.recommendation_source.eq("MODEL"), "ready_for_export"].any()
