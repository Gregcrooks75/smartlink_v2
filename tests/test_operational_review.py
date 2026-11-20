import json

import pandas as pd
import pytest

from mtx.history.divisions import enrich_divisions
from mtx.prediction.final_review import read_history, review
from mtx.prediction.operational_reporting import revised_meters, summary
from mtx.prediction.recommendations import _protect_empty
from mtx.training.history_selection import select_run


@pytest.fixture
def site(tmp_path):
    ami = pd.DataFrame(
        [
            ("001", "O", "C", "P1"),
            ("002", "O", "C", "P2"),
            ("003", "O", "C", "P3"),
            ("004", "R", "D", "P4"),
            ("005", "S", "E", "P5"),
        ],
        columns=[
            "meter_number",
            "current_transformer",
            "connection_object",
            "premise_id",
        ],
    ).assign(division="Ithaca")
    itron = ami[["meter_number", "division"]].assign(
        itron_current_transformer=ami.current_transformer,
        subgroup_id=[None, None, "A", "B", "C"],
    )
    rec = ami.iloc[:2][["meter_number", "division"]].assign(
        recommended_transformer=["R", "S"],
        subgroup_id=None,
        recommendation_source=["ADAM_REVIEW", "MODEL"],
        ready_for_export=True,
        status="PENDING",
        review_flags="",
        adam_validation_source=["MAP", None],
        adam_validation_date=["2026-11-01", None],
    )
    tx = pd.DataFrame(
        {
            "floc": ["O", "R", "S"],
            "sapid": ["1", "2", "3"],
            "division": "ITHACA",
            "opco": "NY",
        }
    )
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    pd.DataFrame(
        [("001", "O", 20), ("001", "R", 25), ("002", "O", 21), ("002", "S", 26)],
        columns=["meter_number", "candidate_transformer", "distance_ft"],
    ).to_parquet(pairs / "Ithaca.parquet")
    archive = tmp_path / "New CO"
    archive.mkdir()
    return rec, ami, itron, tx, pairs, archive


def run(site):
    return review(*site, co_directory=site[-1])


def test_two_targets_split_and_archived_workbook_round_trip(site):
    result = run(site)
    es55, es64 = [result["co_tables"][tab] for tab in ("ES55", "ES64")]
    assert list(es55.final_recommended_floc) == ["R", "S"]
    assert dict(zip(es64.PremiseID, es64.Automation_ID)) == dict(
        zip(["P1", "P2"], es55.Automation_ID)
    )
    assert set(result["co_tables"]["Meters to Move"].meter_number) == {"001", "002"}
    assert result["plan"]["ready"].empty
    assert result["work"].connection_object_status.eq("READY_TO_SUBMIT").all()
    assert result["work"].review_reason.eq("").all()
    assert result["work"].loc[0, "upstream_adam_validation_source"] == "MAP"
    from mtx.prediction.final_review import write_excel

    write_excel(site[-1] / "previous_request.xlsx", result["co_tables"])
    repeated = run(site)
    assert repeated["co_tables"]["ES55"].empty
    assert repeated["ledger"].empty
    assert repeated["work"].connection_object_status.eq("SENT").all()


def test_held_manual_target_survives_and_blocks_unsafe_export(site):
    rec = site[0]
    rec.loc[0, ["ready_for_export", "status", "review_flags"]] = [
        False,
        "REJECTED",
        "ADAM_TARGET_CONFLICT",
    ]
    result = run(site)
    manual = result["work"].set_index("meter_number").loc["001"]
    assert (manual.recommendation_source, manual.upstream_status) == (
        "ADAM_REVIEW",
        "REJECTED",
    )
    assert manual.upstream_review_flags == "ADAM_TARGET_CONFLICT"
    assert (
        manual.proposed_final_transformer == "R"
        and manual.Final_Recommended_FLOC == "O"
    )
    assert manual.connection_object_status == "REVIEW_REQUIRED"
    assert result["co_tables"]["ES55"].empty and result["plan"]["ready"].empty


def test_unrelated_target_and_completed_request(site):
    archive = site[-1]
    pd.DataFrame(
        {
            "Connection_Object_Template": ["OTHER"],
            "recommended_floc": ["R"],
            "status": ["SENT"],
        }
    ).to_csv(archive / "request.csv", index=False)
    assert len(run(site)["co_tables"]["ES55"]) == 2
    pd.DataFrame(
        {
            "Connection_Object_Template": ["C"],
            "recommended_floc": ["R"],
            "co_status": ["COMPLETED"],
        }
    ).to_csv(archive / "completed.csv", index=False)
    r = run(site)
    row = r["work"].set_index("meter_number").loc["001"]
    assert (
        row.historical_co_status == "COMPLETED"
        and row.connection_object_status == "REVIEW_REQUIRED"
    )
    assert "COMPLETED_CO_STILL_REQUIRES_SPLIT" in row.review_reason


def test_sent_batch_is_not_sent_co_and_unavailable_archive_holds(site, tmp_path):
    directory = tmp_path / "history"
    directory.mkdir()
    pd.DataFrame({"meter_number": ["001"], "recommended_floc": ["R"]}).to_csv(
        directory / "batch_model_1.csv", index=False
    )
    r = review(*site[:-1], directory, co_directory=tmp_path / "missing")
    assert r["ledger"].meter_number.tolist() == ["001"]
    assert r["work"].set_index("meter_number").loc["001", "status"] == "SENT"
    assert r["co_tables"]["ES55"].empty
    pd.DataFrame({"unused": [1]}).to_csv(site[-1] / "unrecognized.csv", index=False)
    with pytest.raises(ValueError, match="No identifiable"):
        read_history(directory, co_directory=site[-1])


def test_divisions_are_one_to_one_with_conflict_and_unknown():
    f = pd.DataFrame(
        {
            "meter_number": ["01", "02", "03", "04"],
            "division": ["Ithaca", None, None, None],
        }
    )
    legacy = pd.DataFrame(
        {
            "MeterID": ["01", "02", "02", "03", "03"],
            "Division": ["Geneva", "Sodus", "Lakeshore/Sodus", "A", "B"],
        }
    )
    out = enrich_divisions(f, legacy)
    assert out.division.tolist() == ["ITHACA", "SODUS", "UNKNOWN", "UNKNOWN"]
    assert out.division_conflict.tolist() == [True, False, True, False]
    assert out.meter_number.is_unique


def test_daily_summary_separates_review_submission_revision_and_repeat():
    e = pd.DataFrame(
        {
            "meter_number": ["1"] * 5,
            "model_version": ["MODEL_1", "MODEL_2", "MODEL_2", None, None],
            "event_type": [
                "SUBMISSION",
                "MODEL_REVIEW",
                "MODEL_REVIEW",
                "RECOMMENDATION",
                "RECOMMENDATION",
            ],
            "event_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-02",
                "2026-01-03",
            ],
            "division": "ITHACA",
            "target_floc": "B",
            "original_transformer": "A",
            "review_outcome": None,
            "connection_object_status": None,
        }
    )
    snap = pd.DataFrame(
        {
            "meter_number": ["1"],
            "division": "ITHACA",
            "status": "TO_REVIEW",
            "connection_object_status": "REVIEW_REQUIRED",
        }
    )
    out = summary(e, snap, "2026-01-03", set()).iloc[0]
    assert (
        out.both_models,
        out.model_2_documented,
        out.submitted,
        out.model_1_revised_by_model_2,
    ) == (1, 1, 1, 0)
    assert (out.new_recommendations, out.held, out.ready, out.co_required) == (
        0,
        1,
        0,
        1,
    )
    records = pd.DataFrame(
        {
            "meter_number": ["1", "1"],
            "target_floc": ["A", "B"],
            "model_version": ["MODEL_1", "MODEL_2"],
            "submission_status": ["COMPLETED", "SENT"],
            "implementation_date": ["2026-01-01", "2026-02-01"],
        }
    )
    assert not revised_meters(records)
    assert revised_meters(records.assign(submission_status="COMPLETED")) == {"1"}


def test_empty_transformer_gate_with_one_departure():
    out = _protect_empty(
        pd.DataFrame(
            {
                "current_transformer": ["A"],
                "recommended_transformer": ["B"],
                "review_flags": [""],
                "status": ["PENDING"],
            }
        ),
        pd.DataFrame({"current_transformer": ["A", "B"]}),
    )
    assert out.would_empty_current_transformer.all() and not out.ready_for_export.any()


def test_training_history_selection_skips_invalid_latest(tmp_path):
    for time in ("080000_000001", "090000_000001"):
        folder = tmp_path / "2026-11-20" / time
        folder.mkdir(parents=True)
        for i, split in enumerate(("train", "validation", "test")):
            pd.DataFrame(
                {
                    "meter_number": [str(i) if time.startswith("08") else "same"],
                    "v": [3],
                }
            ).to_parquet(folder / f"{split}.parquet")
        (folder / "summary.json").write_text(json.dumps({"seed": 42}))
        (folder / "feature_columns.json").write_text('["v"]')
    selected = select_run(tmp_path)
    assert selected == select_run(tmp_path, "2026-11-20/080000_000001")
    assert selected["config"]["seed"] == 42 and len(selected["splits"]) == 3
    with pytest.raises(ValueError, match="leakage"):
        select_run(tmp_path, "2026-11-20/090000_000001")


def test_precision_uses_only_later_manual_or_explicit_validation(tmp_path):
    from mtx.prediction.operational_reporting import performance

    e = pd.DataFrame(
        [
            ("MODEL", "RECOMMENDATION", "M1", "B", "2026-01-01", "1"),
            ("MODEL", "RECOMMENDATION", "M2", "B", "2026-01-01", "2"),
            ("ADAM_REVIEW", "RECOMMENDATION", "M3", "C", "2026-01-01", "3"),
            ("ADAM_REVIEW", "ADAM_REVIEW", "M1", "C", "2026-02-01", None),
            ("ADAM_REVIEW", "ADAM_REVIEW", "M3", "C", "2026-02-01", None),
        ],
        columns=[
            "recommendation_source",
            "event_type",
            "meter_number",
            "target_floc",
            "event_date",
            "recommendation_id",
        ],
    ).assign(model_date="2026-01-01", division="ITHACA")
    (tmp_path / "reference").mkdir()
    pd.DataFrame(
        {
            "recommendation_id": ["2"],
            "evaluation_date": ["2026-02-03"],
            "verified_transformer": ["B"],
        }
    ).to_csv(tmp_path / "reference/operational_validation.csv", index=False)
    rows = performance(e, tmp_path, "2026-02-03")
    totals = rows.loc[
        rows.division.eq("ALL") & rows.evaluation_date.eq("ALL")
    ].set_index("basis")
    assert totals.loc["ADAM_FOLLOW_UP", ["evaluations", "precision"]].tolist() == [1, 0]
    assert totals.loc[
        "OPERATIONAL_VALIDATION", ["evaluations", "precision"]
    ].tolist() == [1, 1]


def test_completed_template_with_pending_assignment_needs_reconciliation(site):
    with pd.ExcelWriter(site[-1] / "archived.xlsx") as writer:
        pd.DataFrame(
            {
                "Connection_Object_Template": ["C"],
                "recommended_floc": ["R"],
                "co_status": ["COMPLETED"],
            }
        ).to_excel(writer, sheet_name="ES55", index=False)
        pd.DataFrame(
            {"PremiseID": ["P1"], "recommended_floc": ["R"], "co_status": ["SENT"]}
        ).to_excel(writer, sheet_name="ES64", index=False)
    result = run(site)
    row = result["work"].set_index("meter_number").loc["001"]
    assert row.historical_co_status == "REVIEW_REQUIRED"
    assert result["co_tables"]["ES55"].empty and result["plan"]["ready"].empty
