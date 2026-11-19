"""Append authoritative ADAM/MAP corrections to model recommendations.

ADAM is a separately attributed source, not a model prediction. Its labels do
not receive model score gaps or model-based estimated precision. Unsafe or
unverifiable proposals are held for review, never silently marked ready.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _ids(series):
    """Use the same identifier canonicalization as the recommendation stage."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^(\\d+)\\.0+$", r"\\1", regex=True)
        .replace(["", "NONE", "NULL", "NAN", "N/A", "<NA>"], pd.NA)
    )


def append_adam_corrections(
    model_recommendations,
    adam_df,
    *,
    ami,
    transformers,
    feature_files,
    sent_meters,
    model_date,
    day,
    enrich,
):
    """Return model rows followed by eligible, separately labeled ADAM rows.

    Only MAP-validated, nonconflicting labels dated no later than the run date
    are authoritative (matching the training policy). Already-sent, already
    implemented and unresolved labels are not resubmitted. When an ADAM label
    disagrees with a model proposal, the ADAM target takes precedence but the
    row is held for review; each meter occurs at most once.
    """
    model = model_recommendations.copy()
    model["recommendation_source"] = "MODEL"
    for field in ("adam_validation_source", "adam_validation_date", "adam_verified_transformer"):
        model[field] = pd.NA

    required = {
        "meter_number",
        "Correct_Transformer_FLOC",
        "ValidationSource",
        "ValidationDate",
    }
    missing = required - set(adam_df.columns)
    if missing:
        raise ValueError(f"ADAM corrections missing columns: {sorted(missing)}")

    labels = adam_df[list(sorted(required))].copy()
    labels["meter_number"] = _ids(labels["meter_number"])
    labels["verified_transformer"] = _ids(labels["Correct_Transformer_FLOC"])
    labels["source"] = _ids(labels["ValidationSource"])
    labels["validation_date"] = pd.to_datetime(
        labels["ValidationDate"], errors="coerce", utc=True
    ).dt.date

    labels = labels.loc[
        labels["meter_number"].notna()
        & labels["verified_transformer"].notna()
        & labels["source"].eq("MAP").fillna(False)
        & labels["validation_date"].notna()
        & labels["validation_date"].le(date.fromisoformat(day))
    ].copy()

    if labels.empty:
        return model

    # A disagreement between two approved labels must not be resolved by
    # arbitrary row order or by the most recent observation.
    conflicts = labels.groupby("meter_number")["verified_transformer"].nunique()
    labels = labels.loc[
        ~labels["meter_number"].isin(conflicts.index[conflicts.gt(1)])
    ].sort_values("validation_date", ascending=False).drop_duplicates(
        "meter_number"
    )

    labels = labels.loc[labels["meter_number"].isin(ami.index)].copy()
    if labels.empty:
        return model

    labels["current_transformer"] = labels["meter_number"].map(
        ami["current_transformer"]
    )
    labels["division"] = labels["meter_number"].map(ami["division"])

    # A validated "no change" label suppresses a conflicting model change,
    # but cannot itself be exported as a correction.
    sent = {_id for _id in sent_meters if pd.notna(_id)}
    labels = labels.loc[~labels["meter_number"].isin(sent)].copy()
    no_change = labels["verified_transformer"].eq(
        labels["current_transformer"]
    ).fillna(False)
    model = model.loc[
        ~model["meter_number"].isin(labels.loc[no_change, "meter_number"])
    ].copy()
    labels = labels.loc[
        ~no_change & labels["current_transformer"].notna()
    ].copy()
    if labels.empty:
        return model

    model_targets = model.set_index("meter_number")["recommended_transformer"]
    labels["model_target"] = labels["meter_number"].map(model_targets)
    labels["model_agrees"] = labels["verified_transformer"].eq(
        labels["model_target"]
    ).fillna(False)
    labels["model_disagrees"] = labels["model_target"].notna() & ~labels[
        "model_agrees"
    ]

    # Give the human-verified target priority over one model proposal,
    # and append it after the remaining model recommendations.
    model = model.loc[
        ~model["meter_number"].isin(labels["meter_number"])
    ].copy()

    # Validate against *today's* final candidate features; failure to find
    # the pair holds the ADAM entry, it does not bypass the candidate policy.
    candidate_parts = []
    for division, path in sorted(feature_files.items()):
        meters = labels.loc[labels["division"].eq(division), "meter_number"].tolist()
        if not meters:
            continue
        part = pd.read_parquet(
            path,
            columns=[
                "meter_number",
                "candidate_transformer",
                "tx_group_id",
                "tx_group_size",
                "current_meter_floc",
            ],
            filters=[("meter_number", "in", meters)],
        )
        part["division"] = division
        candidate_parts.append(part)

    candidates = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame(
            columns=[
                "meter_number", "candidate_transformer", "tx_group_id",
                "tx_group_size", "current_meter_floc", "division",
            ]
        )
    )
    candidates["meter_number"] = _ids(candidates["meter_number"])
    candidates["candidate_transformer"] = _ids(candidates["candidate_transformer"])
    candidates["tx_group_id"] = _ids(candidates["tx_group_id"])
    candidates["current_meter_floc"] = _ids(candidates["current_meter_floc"])
    candidate_counts = candidates.groupby("meter_number").size()
    # This is sufficient for an *entire* ADAM subgroup; partial subgroups
    # are held below rather than treating sampled members as complete.
    member_info = candidates.drop_duplicates("meter_number").copy()
    member_info["tx_current_transformer_count"] = (
        member_info.groupby(["division", "tx_group_id"])
        ["current_meter_floc"].transform("nunique")
        .where(member_info["tx_group_id"].notna(), 1)
    )

    matching = candidates.merge(
        labels[["meter_number", "verified_transformer"]],
        left_on=["meter_number", "candidate_transformer"],
        right_on=["meter_number", "verified_transformer"],
        how="inner",
        validate="one_to_one",
    ).drop_duplicates("meter_number").set_index("meter_number")

    # Subgroup metadata is common to all pairs for a meter. Retain it even
    # when the verified target is not in the candidate list.
    member_info = member_info.set_index("meter_number")
    for col in ("tx_group_id", "tx_group_size", "tx_current_transformer_count"):
        labels[col] = labels["meter_number"].map(member_info[col])
    labels["has_candidate"] = labels["meter_number"].isin(matching.index)
    labels["candidate_count"] = labels["meter_number"].map(candidate_counts)
    labels["target_exists"] = labels["verified_transformer"].isin(
        transformers.index
    )

    predictions = pd.DataFrame({
        "meter_number": labels["meter_number"].to_numpy(),
        "recommended_transformer": labels["verified_transformer"].to_numpy(),
        "subgroup_id": labels["tx_group_id"].to_numpy(),
        "tx_group_size": labels["tx_group_size"].to_numpy(),
        "candidate_count": labels["candidate_count"].to_numpy(),
        "score_gap": np.nan,
        "rank_score": np.nan,
        "second_score": np.nan,
    })
    adam = enrich(predictions, ami, transformers, model_date, day)
    details = labels.set_index("meter_number")
    adam["recommendation_source"] = np.where(
        adam["meter_number"].map(details["model_agrees"]).fillna(False),
        "ADAM+MODEL",
        "ADAM",
    )
    adam["adam_validation_source"] = "MAP"
    adam["adam_validation_date"] = adam["meter_number"].map(
        details["validation_date"]
    ).astype("string")
    adam["adam_verified_transformer"] = adam["recommended_transformer"]
    adam["estimated_precision"] = np.nan
    adam["estimate_basis"] = "ADAM_VERIFIED_NOT_MODEL_ESTIMATE"
    adam["test_sample_size"] = 0
    adam["estimate_sample_size"] = 0
    adam["precision_reference_split"] = "ADAM"
    adam["review_flags"] = ""

    def flag(mask, reason):
        adam.loc[mask.fillna(True), "review_flags"] += reason + ";"

    flag(~adam["meter_number"].map(details["has_candidate"]), "ADAM_NOT_IN_FINAL_CANDIDATES")
    flag(~adam["meter_number"].map(details["target_exists"]), "ADAM_UNKNOWN_TRANSFORMER")
    flag(adam["meter_number"].map(details["model_disagrees"]), "ADAM_MODEL_DISAGREEMENT")
    flag(
        adam["meter_number"].map(details["tx_current_transformer_count"]).gt(3),
        "ADAM_SUBGROUP_OVER_LIMIT",
    )
    flag(
        adam["tx_group_size"].gt(1)
        & adam["group_predictions_partial"].fillna(True),
        "ADAM_PARTIAL_SUBGROUP",
    )
    flag(
        adam["group_recommended_transformers"].gt(1),
        "ADAM_SUBGROUP_TARGET_CONFLICT",
    )
    flag(~adam["recommended_distance_ft"].between(0, 550), "OUTSIDE_550FT")
    flag(adam["division_mismatch"], "DIVISION_MISMATCH")
    flag(adam["opco_mismatch"], "OPCO_MISMATCH")
    flag(
        adam[["meter_division", "recommended_division", "meter_opco", "recommended_opco"]]
        .isna().any(axis=1),
        "MISSING_DIVISION_OR_OPCO",
    )
    flag(
        adam["connection_object"].isna() | adam["premise_id"].isna(),
        "MISSING_SAP_IDENTIFIERS",
    )
    for prefix in ("meter", "current", "recommended"):
        coordinates = adam[[f"{prefix}_latitude", f"{prefix}_longitude"]]
        flag(
            ~coordinates.iloc[:, 0].between(-90, 90)
            | ~coordinates.iloc[:, 1].between(-180, 180),
            "MISSING_OR_INVALID_COORDINATES",
        )
    adam["review_flags"] = adam["review_flags"].str.rstrip(";")
    print(
        f"ADAM: appended {len(adam):,} verified corrections "
        f"({adam['review_flags'].ne('').sum():,} held for review); "
        f"{len(model_recommendations) - len(model):,} model rows superseded."
    )
    return pd.concat([model, adam], ignore_index=True, sort=False)
