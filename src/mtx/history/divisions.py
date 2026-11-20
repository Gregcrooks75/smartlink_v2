"""Optional legacy division enrichment for historical meters."""

import os
import warnings

import pandas as pd


def enrich_divisions(ledger, legacy):
    from mtx.data.standardization import DIVISION_MAPPING

    aliases = {key.upper(): value.upper() for key, value in DIVISION_MAPPING.items()}

    def division(values):
        return (
            values.astype("string")
            .str.strip()
            .str.upper()
            .replace({"": pd.NA, **aliases})
        )

    result = ledger.copy()
    result["division"] = division(result.division)
    result["division_source"] = "CURRENT_AMI"
    result["division_conflict"] = False
    if legacy is not None and not legacy.empty:
        old = (
            pd.DataFrame(
                {
                    "meter_number": legacy.MeterID.astype("string")
                    .str.strip()
                    .str.replace(r"^(\d+)\.0$", r"\1", regex=True),
                    "division": division(legacy.Division),
                }
            )
            .dropna()
            .drop_duplicates()
        )
        options = old.groupby("meter_number").division
        historical = result.meter_number.map(options.first())
        ambiguous = result.meter_number.map(options.nunique()).gt(1)
        result["division_conflict"] = ambiguous | (
            result.division.notna()
            & historical.notna()
            & result.division.ne(historical)
        ).fillna(False)
        recover = result.division.isna() & ~ambiguous & historical.notna()
        result.loc[recover, "division"] = historical[recover]
        result.loc[recover, "division_source"] = "LEGACY_AMI"
    result.loc[result.division.isna(), "division_source"] = "UNKNOWN"
    result["division"] = result.division.fillna("UNKNOWN")
    return result


def load_legacy_divisions():
    if os.getenv("SMARTLINK_LEGACY_DIVISIONS", "").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from mtx.data.extract_ami_data import SERVER, NY_DATABASE
        from mtx.db.db_trusted import run_query_trusted

        return run_query_trusted(
            "SELECT MeterID, ServiceCenterName AS Division FROM [NYAMI_SDE].[NYAMI].[ALL_METERS_NY_20260101]",
            SERVER,
            NY_DATABASE,
        )
    except Exception as exc:
        warnings.warn(
            f"Legacy division lookup unavailable ({type(exc).__name__})", RuntimeWarning
        )
        return None
