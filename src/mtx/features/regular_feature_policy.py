"""Shared feature contract for SmartLink's regular-only M2T pipeline.

Capacity follows the analysis notebook's family: kVA, bank/unit/SAPID counts
and every interaction using those values. Customer counts are retained.
No geospatial or optional feature-stage imports belong in this module.
"""

FEATURE_POLICY_VERSION = "regular-no-tile-no-capacity-v1"

CAPACITY_TERMS = (
    "kva",
    "capacity",
    "transformer_unit_count",
    "transformer_bank",
    "sapid_count",
)

TILE_TERMS = (
    "obstacle",
    "building",
    "crossing",
    "blocked_length",
    "obstruction",
)

# This is also produced by the retired tile stage, despite its neutral name.
TILE_COLUMNS = frozenset({"straight_line_length_ft"})


def is_removed_feature(column: str) -> bool:
    name = str(column).strip().lower()
    return "tile" in name.split("_") or name in TILE_COLUMNS or any(
        term in name for term in CAPACITY_TERMS + TILE_TERMS
    )


def removed_feature_columns(columns) -> list[str]:
    return [column for column in columns if is_removed_feature(column)]


def assert_regular_features(columns, context="Features") -> None:
    removed = removed_feature_columns(columns)
    if removed:
        raise ValueError(
            f"{context}: retired tile/capacity features found: {removed}. "
            "Rebuild regular features and retrain the model."
        )
