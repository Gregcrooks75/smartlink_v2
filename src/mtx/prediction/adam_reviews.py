"""Prepare Adam's MAP-verified transformer corrections for the recommendation review.

The approved labels are the same source used by the training-data stage. These
rows are *not* model predictions. Their provenance and safety exceptions remain
visible for review; this module never dispatches a batch.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _identifier(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        .replace(["", "NONE", "NULL", "NAN", "N/A", "<NA>"], pd.NA)
    )


def _add_flag(frame: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    selected = mask.fillna(False)
    frame.loc[selected, "adam_review_flags"] += flag + ";"


def build_adam_review_predictions(
    *,
    adam_reviews: pd.DataFrame | None,
    ami: pd.DataFrame,
    tx: pd.DataFrame,
    final_feature_files: dict[str, Path],
    run_date: str,
    max_subgroup_transformers: int = 3,
) -> pd.DataFrame:
    """Build human-reviewed proposals without inventing model scores.

    Only MAP-approved, dated, unambiguous labels differing from current AMI
    are proposed. Candidate/subgroup evidence is read from today's features.
    Missing or unsafe evidence results in a review hold, not silent approval.
    """
    columns = [
        "meter_number", "recommended_transformer", "subgroup_id",
        "tx_group_size", "candidate_count", "score_gap", "rank_score",
        "second_score", "recommendation_source", "adam_validation_source",
        "adam_validation_date", "adam_review_flags",
    ]
    if adam_reviews is None or adam_reviews.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "meter_number", "Correct_Transformer_FLOC",
        "ValidationSource", "ValidationDate",
    }
    missing = required - set(adam_reviews.columns)
    if missing:
        raise ValueError(f"Adam reviews missing columns: {sorted(missing)}")

    labels = adam_reviews[list(required)].copy()
    labels["meter_number"] = _identifier(labels["meter_number"])
    labels["recommended_transformer"] = _identifier(
        labels["Correct_Transformer_FLOC"]
    )
    labels["adam_validation_source"] = _identifier(labels["ValidationSource"])
    labels["_date"] = pd.to_datetime(
        labels["ValidationDate"], errors="coerce"
    )
    labels = labels.loc[
        labels["meter_number"].notna()
        & labels["recommended_transformer"].notna()
        & labels["adam_validation_source"].eq("MAP").fillna(False)
        & labels["_date"].notna()
        & labels["_date"].le(pd.Timestamp(date.fromisoformat(run_date)))
    ].copy()

    if labels.empty:
        return pd.DataFrame(columns=columns)

    # Match the training dataset's conservative rule: conflicting verified
    # targets for one meter are excluded even if one was reviewed later.
    distinct = labels.groupby("meter_number")["recommended_transformer"].nunique()
    labels = labels.loc[
        labels["meter_number"].map(distinct).eq(1)
    ].sort_values("_date", kind="stable")
    labels = labels.drop_duplicates("meter_number", keep="last").copy()

    labels = labels.loc[labels["meter_number"].isin(ami.index)].copy()
    labels["current_transformer"] = labels["meter_number"].map(
        ami["current_transformer"]
    )
    labels = labels.loc[
        labels["current_transformer"].notna()
        & labels["current_transformer"].ne(labels["recommended_transformer"])
    ].copy()
    if labels.empty:
        return pd.DataFrame(columns=columns)

    labels["division"] = labels["meter_number"].map(ami["division"])
    labels["adam_validation_date"] = labels["_date"].dt.strftime("%Y-%m-%d")
    labels["recommendation_source"] = "ADAM_REVIEW"
    labels["adam_review_flags"] = ""
    labels["subgroup_id"] = pd.Series(pd.NA, index=labels.index, dtype="string")
    labels["tx_group_size"] = 1
    labels["candidate_count"] = 0
    labels["_candidate_match"] = False
    labels["_subgroup_quality"] = 0
    labels["_inconsistent_group"] = False

    # Project just the columns required for reviewing Adam's proposals.
    # This includes meters excluded from model scoring because they were
    # already used for training; no model scores are recomputed.
    feature_columns = [
        "meter_number", "candidate_transformer", "tx_group_id",
        "tx_group_size", "current_meter_floc",
    ]
    for division, feature_path in sorted(final_feature_files.items()):
        selected = labels["division"].eq(division).fillna(False)
        if not selected.any():
            continue
        meters = set(labels.loc[selected, "meter_number"])
        evidence = pd.read_parquet(feature_path, columns=feature_columns)
        for col in ("meter_number", "candidate_transformer", "tx_group_id", "current_meter_floc"):
            evidence[col] = _identifier(evidence[col])
        evidence["tx_group_size"] = pd.to_numeric(
            evidence["tx_group_size"], errors="coerce"
        )
        meter_rows = evidence.drop_duplicates("meter_number")
        group_quality = (
            meter_rows.loc[meter_rows["tx_group_id"].notna()]
            .groupby("tx_group_id")["current_meter_floc"]
            .nunique()
        )
        evidence = evidence.loc[evidence["meter_number"].isin(meters)].copy()
        if evidence.empty:
            continue

        metadata = evidence.groupby("meter_number").agg(
            subgroup_id=("tx_group_id", "first"),
            tx_group_size=("tx_group_size", "first"),
            candidate_count=("candidate_transformer", "nunique"),
            subgroup_variants=("tx_group_id", lambda x: x.nunique(dropna=False)),
        )
        candidate_pairs = set(zip(
            evidence["meter_number"], evidence["candidate_transformer"]
        ))
        for index in labels.index[selected]:
            meter = labels.at[index, "meter_number"]
            if meter not in metadata.index:
                continue
            info = metadata.loc[meter]
            labels.at[index, "subgroup_id"] = info["subgroup_id"]
            labels.at[index, "tx_group_size"] = (
                int(info["tx_group_size"])
                if pd.notna(info["tx_group_size"]) else 1
            )
            labels.at[index, "candidate_count"] = int(info["candidate_count"])
            labels.at[index, "_candidate_match"] = (
                meter, labels.at[index, "recommended_transformer"]
            ) in candidate_pairs
            labels.at[index, "_inconsistent_group"] = (
                int(info["subgroup_variants"]) > 1
            )
            group = info["subgroup_id"]
            labels.at[index, "_subgroup_quality"] = (
                int(group_quality.get(group, 0))
                if pd.notna(group) else 1
            )

    _add_flag(labels, ~labels["_candidate_match"], "ADAM_TARGET_NOT_IN_CANDIDATES")
    _add_flag(
        labels,
        ~labels["recommended_transformer"].isin(tx.index),
        "ADAM_TARGET_NOT_IN_TRANSFORMERS",
    )
    _add_flag(
        labels,
        labels["_inconsistent_group"],
        "ADAM_INCONSISTENT_SUBGROUP",
    )
    _add_flag(
        labels,
        labels["_subgroup_quality"].gt(max_subgroup_transformers),
        "ADAM_SUBGROUP_OVER_LIMIT",
    )

    grouped = labels.loc[labels["subgroup_id"].notna()]
    conflicting = (
        grouped.groupby(["division", "subgroup_id"])["recommended_transformer"]
        .nunique().gt(1)
    )
    if len(conflicting):
        key = pd.MultiIndex.from_frame(labels[["division", "subgroup_id"]])
        _add_flag(
            labels,
            pd.Series(key.map(conflicting).fillna(False), index=labels.index),
            "ADAM_SUBGROUP_TARGET_CONFLICT",
        )

    # No model-estimated score or precision is attributed to a manual label.
    labels["score_gap"] = np.nan
    labels["rank_score"] = np.nan
    labels["second_score"] = np.nan
    labels["adam_review_flags"] = labels["adam_review_flags"].str.rstrip(";")
    return labels[columns].reset_index(drop=True)
