"""Focused tests for exporting Adam-verified labels alongside model proposals."""

from datetime import date

import pandas as pd

from mtx.prediction.adam_reviews import build_adam_review_predictions


def inputs(tmp_path):
    ami = pd.DataFrame([
        ("M1", "TX-A", "Ithaca"),
        ("M2", "TX-A", "Ithaca"),
        ("M3", "TX-A", "Ithaca"),
    ], columns=["meter_number", "current_transformer", "division"]).set_index(
        "meter_number", drop=False
    )
    tx = pd.DataFrame(
        {"transformer_floc": ["TX-A", "TX-B", "TX-C"]}
    ).set_index("transformer_floc", drop=False)
    features = pd.DataFrame([
        (meter, candidate, "G" + meter, 1, "TX-A")
        for meter in ("M1", "M2", "M3")
        for candidate in ("TX-A", "TX-B")
    ], columns=[
        "meter_number", "candidate_transformer", "tx_group_id",
        "tx_group_size", "current_meter_floc",
    ])
    path = tmp_path / "Ithaca.parquet"
    features.to_parquet(path)
    return ami, tx, {"Ithaca": path}


def review(meter, target, source="MAP", when=date(2020, 1, 1)):
    return {
        "meter_number": meter,
        "Correct_Transformer_FLOC": target,
        "ValidationSource": source,
        "ValidationDate": when,
    }


def build(reviews, inputs_):
    ami, tx, files = inputs_
    return build_adam_review_predictions(
        adam_reviews=pd.DataFrame(reviews),
        ami=ami,
        tx=tx,
        final_feature_files=files,
        run_date="2026-11-19",
    )


def test_verified_correction_includes_source_but_no_model_score(tmp_path):
    output = build([
        review("M1", "TX-B"),
        review("M2", "TX-A"),  # No actual correction.
        review("M3", "TX-B", source="UNAPPROVED"),
    ], inputs(tmp_path))
    assert output["meter_number"].tolist() == ["M1"]
    row = output.iloc[0]
    assert row["recommendation_source"] == "ADAM_REVIEW"
    assert row["adam_validation_source"] == "MAP"
    assert row["adam_validation_date"] == "2020-01-01"
    assert row["recommended_transformer"] == "TX-B"
    assert row["candidate_count"] == 2
    assert row["adam_review_flags"] == ""
    assert pd.isna(row["score_gap"])
    assert pd.isna(row["rank_score"])


def test_conflicting_truth_and_future_or_unapproved_labels_excluded(tmp_path):
    output = build([
        review("M1", "TX-B"),
        review("M1", "TX-C"),  # Conflict; do not pick arbitrarily.
        review("M2", "TX-B", source="UNAPPROVED"),
        review("M3", "TX-B", when=date(2027, 1, 1)),
    ], inputs(tmp_path))
    assert output.empty


def test_unverified_candidate_is_visible_but_held(tmp_path):
    output = build([review("M1", "TX-C")], inputs(tmp_path))
    assert len(output) == 1
    assert "ADAM_TARGET_NOT_IN_CANDIDATES" in output.iloc[0]["adam_review_flags"]


def test_subgroup_over_limit_is_held(tmp_path):
    ami, tx, files = inputs(tmp_path)
    feature_path = files["Ithaca"]
    features = pd.read_parquet(feature_path)
    features["tx_group_id"] = "SHARED"
    features["tx_group_size"] = 3
    features.loc[features.meter_number.eq("M2"), "current_meter_floc"] = "TX-B"
    features.loc[features.meter_number.eq("M3"), "current_meter_floc"] = "TX-C"
    features.to_parquet(feature_path)
    output = build_adam_review_predictions(
        adam_reviews=pd.DataFrame([review("M1", "TX-B")]),
        ami=ami,
        tx=tx,
        final_feature_files=files,
        run_date="2026-11-19",
        max_subgroup_transformers=2,
    )
    assert "ADAM_SUBGROUP_OVER_LIMIT" in output.iloc[0]["adam_review_flags"]


def test_explicit_rejection_is_auditable_and_held(tmp_path):
    output = build([dict(review("M1", "TX-B"), review_outcome="REJECTED")], inputs(tmp_path))
    row = output.iloc[0]
    assert row.adam_review_outcome == "REJECTED"
    assert "ADAM_OUTCOME_REQUIRES_REVIEW" in row.adam_review_flags
