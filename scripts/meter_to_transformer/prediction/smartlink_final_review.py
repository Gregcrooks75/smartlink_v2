"""SmartLink final review: subgroup checks, then history/CO checks and exports.
Edit the paths below; run: python SmartLink_Final_Review.py
Use AFTER pipeline step 18. No SmartLink imports or pip install -e are needed.
Dependencies: pandas, numpy, pyarrow, openpyxl (xlrd for historical .xls files).
The script prepares files; it does not submit batches or create COs in SAP.
After submitting a batch/CO request, archive its batch_model_2 file in HISTORY_DIR
so the next run recognizes it. Keep unsubmitted exports outside that folder.
"""

from pathlib import Path
from datetime import datetime
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# 1. SETTINGS — use matching recommendation/candidate snapshots.

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

TODAY = datetime.now().strftime("%Y-%m-%d")

RECOMMENDATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / TODAY
    / "all_recommendations.parquet"
)

AMI_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ami"
    / "ny_ami.parquet"
)

ITRON_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "itron"
    / "itron_tx_subgroups.parquet"
)

TRANSFORMERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "transformers"
    / "ny_transformers.parquet"
)

CANDIDATE_PAIRS_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "candidates"
    / TODAY
    / "pairs"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical_corrections"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_review"
)

# None = all input divisions; or ["Ithaca", "Fillmore", "Canandaigua"].
DIVISIONS = None

MAX_DISTANCE_FT = 550.0

# Filename label for archiving submitted outputs.
BATCH_LABEL = "model_2"


# Receiving transformers must have group_count == distinct SAPID_count, at ANY size.

AMI_COLUMNS = {
    "meter_number": "MeterID",
    "division": "Division",
    "current_transformer": "SuperiorFLOC",
    "connection_object": "ConnectionObject",
    "premise_id": "PremiseID",
}

ITRON_COLUMNS = {
    "meter_number": "meter_number",
    "division": "division",
    "subgroup_id": "Tx_Subgroup_ID",
    "itron_current_transformer": "current_meter_floc",
}

TRANSFORMER_COLUMNS = {
    "floc": "FuncLoc",
    "sapid": "SAPID",
    "division": "Division",
    "opco": "OPCO",
}


RULES = """SmartLink final-review rules
1. Scope: input meters plus ALL current members of their ITRON subgroups.
Group IDs include division. Ungrouped input meters are checked individually.
This is a review of your already-selected model input; no new score/gap filter.

2. Every member votes for its model target when supplied, otherwise its current AMI
FLOC. With 1 distinct provisional FLOC, accept; with 2–3, the unique leader
needs >40% of ALL members; with >3, >=60%. Ties or missing/inconsistent
membership, divisions or assignments hold the ENTIRE subgroup. No >50% override.

3. Every member, including unchanged members, must have a candidate pair for the
proposed group target and a finite distance in [0, MAX_DISTANCE_FT]. Missing
pairs, missing/non-numeric/infinite/negative distances, conflicting duplicate
distances, or distances above the limit hold the entire subgroup.
A missing pair means UNVERIFIED, not proof of being farther than the limit.
Current, original-model and proposed-target distances are looked up separately.
Missing/large CURRENT distances alone do not reject a valid proposed target.
distance_flags.csv includes whole subgroups with a problem in ANY available
distance role; only proposed-target failures veto the recommendation.

4. AMI is authoritative for current FLOC/CO/premise. Current and target transformer
metadata must exist and agree on division and OPCO, including the meter's
division. Conflicting metadata goes to review; a prefix comparison alone is
insufficient. Score gaps are not confidence in a group-imposed transformer.

5. History: recursively read CSV, parquet and Excel, checking EVERY worksheet.
A filename containing 'batch' in the submitted-history folder is sent evidence;
an explicit is_sent/batch_sent/status column takes precedence when present.
Draft/review/pending files, CO workbooks and review/movement sheets are excluded
from filename-based sent classification. 'model_2' is a filename label ONLY;
a file must also qualify as sent to enter model_2_sent_meters.
CO creation/assignment requests are tracked separately, never automatically
counted as sent corrections. An archived request is treated as submitted.
Sent meters are never re-exported for sending. A sent member still needing the
proposed change, or a conflicting historical target, holds its entire subgroup.

6. Recompute the WHOLE AMI state after each hold. No selected correction may empty
an existing transformer. Receiving transformers MUST have subgroup_count ==
distinct SAPID_count, with no size cap (4/4 passes; 4/3 fails). Missing/zero SAPID
counts fail. This corrects the old groups_per_SAPID == group_count expression,
which actually selected one SAPID. SAPIDs are devices, not electrical phases.
Ungrouped meters get individual checks, but do not become invented ITRON groups;
a target with zero known groups and a positive SAPID count therefore fails.
>3 subgroups is informational and never an additional rejection rule.

7. COs: inspect ALL AMI meters in every affected CO. A CO must have one final FLOC.
Missing CO/premise data, unresolved premise conflicts, or overlapping historical
CO requests hold the affected subgroup. Create ONE new CO request per target
FLOC needing a split, using a source CO as template. ES64 has one row per premise.
A premise move includes ALL its AMI meters, including unchanged/out-of-input
meters. All must agree on the target, pass division/OPCO checks, and not already
be in a sent batch. Duplicate premises are exported for reference.

8. 1, 2, ... / New_Connection_Object_1, ... are run-local placeholders, not SAP
CO IDs. CO-dependent subgroup corrections and other changes receiving the same
target wait together. The separate ready batch is checked again for empty
transformers, subgroup/SAPID equality and CO consistency on its own.
CO files require creation/ES64 processing before their corrections are sent.

9. 'Correct' means accepted and unchanged in THIS reviewed population, not verified
ground truth or every AMI meter. Rejected fallbacks are never labelled correct.
Main statuses are exclusive; whole-group review/flag tables can also show sent
or unchanged members to preserve context. Full audit is parquet, compact lists
are CSV. Generating outputs does NOT mark anything sent.

10. Each run gets its own timestamped folder. A per-division-scope history stores
the latest completed snapshot, daily counts and per-run changes in decisions,
assignments and distances. 'Entered/left scope' is not an AMI deletion.
"""


def clean_id(series):
    result = series.astype("string").str.strip()

    result = result.mask(
        result.str.lower().isin(
            ["", "nan", "none", "null", "<na>"]
        )
    )

    return result.str.replace(
        r"^(\d+)\.0$",
        r"\1",
        regex=True,
    )


def key(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def unique_meters(frame, label):
    frame = frame.drop_duplicates().copy()

    if (
        frame.meter_number.isna().any()
        or frame.meter_number.duplicated().any()
    ):
        raise ValueError(
            f"{label}: missing meter IDs or conflicting duplicate meters."
        )

    return frame


def read_columns(path, mapping):
    available = pq.read_schema(path).names

    missing = set(mapping.values()) - set(available)

    if missing:
        raise ValueError(
            f"{path.name}: missing {sorted(missing)}; "
            f"edit the column map. Columns: {available}"
        )

    return pd.read_parquet(
        path,
        columns=list(mapping.values()),
    ).rename(
        columns={
            source: target
            for target, source in mapping.items()
        }
    )


HISTORY_ALIASES = {
    "meter_number": [
        "meter_number",
        "meter_id",
        "MeterID",
        "meter",
    ],
    "target_floc": [
        "Final_Recommended_FLOC",
        "final_recommended_floc",
        "recommended_transformer",
        "recommended_floc",
        "Corrected FLOC",
    ],
    "source_co": [
        "Connection_Object_Template",
        "connection_object",
        "Connection Object",
        "Connection_Obj",
    ],
    "premise_id": [
        "premise_id",
        "PremiseID",
    ],
    "created_co": [
        "New_Connection_Object_Created",
    ],
    "automation_id": [
        "Automation_ID",
        "Create CO",
    ],
    "sent_status": [
        "is_sent",
        "batch_sent",
        "sent_as_batch",
        "status",
    ],
}


def history_tables(path):
    wanted = {
        key(alias)
        for values in HISTORY_ALIASES.values()
        for alias in values
    }

    use = lambda c: key(c) in wanted

    if path.suffix.lower() == ".csv":
        yield "", pd.read_csv(
            path,
            dtype="string",
            usecols=use,
        )

    elif path.suffix.lower() == ".parquet":
        columns = [
            c
            for c in pq.read_schema(path).names
            if use(c)
        ]

        yield "", pd.read_parquet(
            path,
            columns=columns,
        )

    else:
        with pd.ExcelFile(path) as book:
            for sheet in book.sheet_names:
                yield sheet, pd.read_excel(
                    book,
                    sheet_name=sheet,
                    dtype="string",
                    usecols=use,
                )


def read_history(directory):
    """Historical corrections and CO submissions stay separate;
    unreadable files stop the run.
    """

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            "Set HISTORY_DIR to the actual step-4 historical folder: "
            f"{directory}"
        )

    sent = []
    co = []
    manifest = []

    suffixes = {
        ".csv",
        ".parquet",
        ".xlsx",
        ".xlsm",
        ".xls",
    }

    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("~$")
            or path.suffix.lower() not in suffixes
        ):
            continue

        if path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            continue

        stem = key(path.stem)

        file_batch = (
            "batch" in stem
            and not any(
                t in stem
                for t in [
                    "draft",
                    "review",
                    "pending",
                ]
            )
        )

        co_file = any(
            t in stem
            for t in [
                "connectionobject",
                "es55",
                "es64",
                "newcos",
                "needsco",
                "cocreation",
            ]
        )

        has_batch_data = False

        for sheet, raw in history_tables(path):
            columns = {}

            for target, aliases in HISTORY_ALIASES.items():
                hits = list(
                    dict.fromkeys(
                        c
                        for a in aliases
                        for c in raw
                        if key(c) == key(a)
                    )
                )

                if hits:
                    # Prefer the first documented alias;
                    # IDs are never inferred from row order.
                    columns[target] = clean_id(raw[hits[0]])

            frame = (
                pd.DataFrame(columns)
                .reindex(columns=list(HISTORY_ALIASES))
                .astype("string")
            )

            frame["source_file"] = str(
                path.relative_to(directory)
            )

            frame["source_sheet"] = sheet

            frame["target_floc"] = (
                frame.target_floc.str.upper()
            )

            model_2 = bool(
                re.search(
                    r"(?:^|[^a-z0-9])model[_ -]*2(?:$|[^a-z0-9])",
                    path.stem,
                    re.I,
                )
            )

            frame["is_model_2"] = model_2

            co_sheet = (
                "created_co" in columns
                or "automation_id" in columns
                or any(
                    t in key(sheet)
                    for t in [
                        "es55",
                        "es64",
                        "cocreation",
                    ]
                )
            )

            if co_sheet:
                co.append(
                    frame.loc[
                        frame.target_floc.notna()
                        & (
                            frame.source_co.notna()
                            | frame.premise_id.notna()
                        )
                    ].copy()
                )

            excluded_sheet = any(
                t in key(sheet)
                for t in [
                    "summary",
                    "review",
                    "validation",
                    "duplicate",
                    "meterstomove",
                    "es55",
                    "es64",
                    "cocreation",
                ]
            )

            if "sent_status" in columns:
                sent_mask = (
                    frame.sent_status.str.lower().isin(
                        [
                            "true",
                            "1",
                            "yes",
                            "sent",
                            "batch_sent",
                            "submitted",
                            "applied",
                            "completed",
                            "implemented",
                        ]
                    )
                )

            else:
                sent_mask = pd.Series(
                    file_batch
                    and not co_file
                    and not co_sheet
                    and not excluded_sheet,
                    index=frame.index,
                )

            selected = frame.loc[
                sent_mask
                & frame.meter_number.notna()
            ]

            has_batch_data |= (
                bool(len(selected))
                or co_sheet
                or "meter_number" in columns
            )

            sent.append(selected)

            manifest.append(
                {
                    "file": str(
                        path.relative_to(directory)
                    ),
                    "sheet": sheet,
                    "sent_meters": len(selected),
                    "CO_request_rows": (
                        int(co_sheet) * len(frame)
                    ),
                }
            )

        if (
            file_batch
            and not co_file
            and not has_batch_data
        ):
            raise ValueError(
                f"No meter column found in historical batch "
                f"{path.name}; check its headers."
            )

    fields = list(HISTORY_ALIASES) + [
        "source_file",
        "source_sheet",
        "is_model_2",
    ]

    combine = lambda parts: (
        pd.concat(
            parts,
            ignore_index=True,
        ).drop_duplicates()
        if parts
        else pd.DataFrame(columns=fields)
    )

    return (
        combine(sent),
        combine(co),
        pd.DataFrame(
            manifest,
            columns=[
                "file",
                "sheet",
                "sent_meters",
                "CO_request_rows",
            ],
        ),
    )


# 2. ITRON POPULATION AND VOTING


def normalize_group(ids, divisions):
    expected = (
        divisions
        + "::ITRON::"
        + ids.str.rsplit("::", n=1).str[-1]
    )

    bad = ids.notna() & (
        divisions.isna()
        | (
            ids.str.contains(
                "::",
                regex=False,
                na=False,
            )
            & ids.str.upper().ne(
                expected
            ).fillna(True)
        )
    )

    return (
        expected.where(ids.notna()),
        bad,
    )


def prepare_population(recommendations, ami, itron):
    ami, itron, rec = (
        ami.copy(),
        itron.copy(),
        recommendations.copy(),
    )

    for frame in (ami, itron, rec):
        for col in frame:
            if (
                "distance" not in col
                and col != "score_gap"
            ):
                frame[col] = clean_id(
                    frame[col]
                )

        frame["division"] = (
            frame.division.str.upper()
        )

    for col in [
        "current_transformer",
        "recommended_transformer",
        "itron_current_transformer",
    ]:
        if col in frame:
            frame[col] = (
                frame[col].str.upper()
            )

    if DIVISIONS is not None:
        rec = rec.loc[
            rec.division.isin(
                [
                    d.strip().upper()
                    for d in DIVISIONS
                ]
            )
        ].copy()

    ami, itron, rec = [
        unique_meters(f, label)
        for f, label in zip(
            [ami, itron, rec],
            [
                "AMI",
                "ITRON",
                "Recommendations",
            ],
        )
    ]

    (
        itron["subgroup_id"],
        itron["bad_group_id"],
    ) = normalize_group(
        itron.subgroup_id,
        itron.division,
    )

    (
        rec["subgroup_id"],
        rec["bad_group_id"],
    ) = normalize_group(
        rec.subgroup_id,
        rec.division,
    )

    itron = itron.rename(
        columns={
            "division": "itron_division",
        }
    )

    rec = rec.rename(
        columns={
            c: "model_" + c
            for c in rec
            if c != "meter_number"
        }
    )

    rec["in_input"] = True

    population = ami.merge(
        itron,
        on="meter_number",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    population["in_ami"] = (
        population._merge.ne("right_only")
    )

    population["division"] = (
        population.division.fillna(
            population.itron_division
        )
    )

    population = population.drop(
        columns="_merge"
    )

    touched = pd.concat(
        [
            population.loc[
                population.meter_number.isin(
                    rec.meter_number
                ),
                "subgroup_id",
            ],
            rec.model_subgroup_id,
        ]
    ).dropna().unique()

    work = population.loc[
        population.meter_number.isin(
            rec.meter_number
        )
        | population.subgroup_id.isin(
            touched
        )
    ]

    work = work.merge(
        rec,
        on="meter_number",
        how="outer",
        validate="one_to_one",
    )

    cached_group = work.subgroup_id.copy()

    work["subgroup_id"] = (
        cached_group.fillna(
            work.model_subgroup_id
        )
    )

    work["in_ami"] = (
        work.in_ami.eq(True)
    )

    work["in_input"] = (
        work.in_input.eq(True)
    )

    work["division"] = (
        work.division.fillna(
            work.model_division
        )
    )

    work["division_mismatch"] = (
        (
            work.itron_division.notna()
            & work.division.ne(
                work.itron_division
            )
        )
        | (
            work.model_division.notna()
            & work.division.ne(
                work.model_division
            )
        )
    ).fillna(False)

    work["membership_mismatch"] = (
        work.model_subgroup_id.notna()
        & (
            cached_group.isna()
            | cached_group.ne(
                work.model_subgroup_id
            )
        )
    ).fillna(False)

    work["provisional_transformer"] = (
        work.model_recommended_transformer.fillna(
            work.current_transformer
        )
    )

    work["invalid_member"] = (
        ~work.in_ami
        | work.division.isna()
        | work.current_transformer.isna()
        | work.division_mismatch
        | work.membership_mismatch
        | work.bad_group_id.eq(True)
        | work.model_bad_group_id.eq(True)
        | work.provisional_transformer.isna()
        | (
            work.in_input
            & work.model_recommended_transformer.isna()
        )
    )

    active_groups = work.loc[
        work.in_input,
        "subgroup_id",
    ].dropna()

    work = (
        work.loc[
            work.in_input
            | work.subgroup_id.isin(
                active_groups
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    work["unit"] = (
        "GROUP::" + work.subgroup_id
    ).fillna(
        "METER::" + work.meter_number
    )

    work["review_reason"] = ""

    return population, work


def hold(work, mask, reason):
    """Every subsequent filter also holds the complete subgroup, not selected members."""

    units = work.loc[
        pd.Series(
            mask,
            index=work.index,
        ).fillna(False),
        "unit",
    ]

    selected = work.unit.isin(units)

    existing = work.loc[
        selected,
        "review_reason",
    ]

    work.loc[
        selected,
        "review_reason",
    ] = existing.map(
        lambda value: (
            value
            if reason in value.split("; ")
            else (
                value + "; " + reason
            ).strip("; ")
        )
    )


def apply_votes(work):
    summary = work.groupby(
        "unit",
        sort=False,
    ).agg(
        subgroup_id=(
            "subgroup_id",
            "first",
        ),
        total_group_size=(
            "meter_number",
            "size",
        ),
        input_meters=(
            "in_input",
            "sum",
        ),
        distinct_flocs=(
            "provisional_transformer",
            "nunique",
        ),
        invalid_members=(
            "invalid_member",
            "sum",
        ),
    )

    votes = (
        work.groupby(
            [
                "unit",
                "provisional_transformer",
            ]
        )
        .size()
        .rename("votes")
        .reset_index()
    )

    leaders = votes.loc[
        votes.votes.eq(
            votes.groupby(
                "unit"
            ).votes.transform("max")
        )
    ]

    winner = (
        leaders
        .drop_duplicates("unit")
        .set_index("unit")
    )

    summary["leading_floc"] = (
        winner.provisional_transformer
    )

    summary["leading_votes"] = (
        winner.votes.reindex(
            summary.index,
            fill_value=0,
        )
    )

    summary["tied"] = (
        leaders.groupby("unit")
        .size()
        .reindex(
            summary.index,
            fill_value=0,
        )
        .gt(1)
    )

    summary["leading_percentage"] = (
        100
        * summary.leading_votes
        / summary.total_group_size
    )

    n = summary.distinct_flocs
    pct = summary.leading_percentage

    passed = (
        n.eq(1)
        | (
            n.isin([2, 3])
            & pct.gt(40)
        )
        | (
            n.gt(3)
            & pct.ge(60)
        )
    )

    for mask, reason in [
        (
            ~passed,
            "VOTE_THRESHOLD",
        ),
        (
            summary.tied,
            "TIED_LEADERS",
        ),
        (
            summary.invalid_members.gt(0),
            "INCOMPLETE_OR_INCONSISTENT_DATA",
        ),
    ]:
        hold(
            work,
            work.unit.isin(
                summary.index[mask]
            ),
            reason,
        )

    summary.loc[
        summary.tied,
        "leading_floc",
    ] = pd.NA

    work["proposed_final_transformer"] = (
        work.provisional_transformer.where(
            work.review_reason.ne(""),
            work.unit.map(
                summary.leading_floc
            ),
        )
    )

    for field in [
        "total_group_size",
        "distinct_flocs",
        "leading_percentage",
    ]:
        work[field] = work.unit.map(
            summary[field]
        )

    return work, summary


def check_distances(work, pairs_dir):
    """Look up all three targets, even for rejected groups;
    retain raw candidate distances.
    """

    if (
        not np.isfinite(MAX_DISTANCE_FT)
        or MAX_DISTANCE_FT < 0
    ):
        raise ValueError(
            "MAX_DISTANCE_FT must be finite and non-negative."
        )

    paths = {}

    for path in Path(pairs_dir).glob(
        "*.parquet"
    ):
        division = (
            path.stem
            .strip()
            .upper()
        )

        if division in paths:
            raise ValueError(
                f"Duplicate candidate division: {division}"
            )

        paths[division] = path

    parts = []

    for division, members in (
        work
        .dropna(subset=["division"])
        .groupby(
            "division",
            sort=False,
        )
    ):
        if division not in paths:
            raise FileNotFoundError(
                f"Candidate division file missing: "
                f"{division} in {pairs_dir}"
            )

        pairs = pd.read_parquet(
            paths[division],
            columns=[
                "meter_number",
                "candidate_transformer",
                "distance_ft",
            ],
        )

        pairs["meter_number"] = clean_id(
            pairs.meter_number
        )

        pairs = pairs.loc[
            pairs.meter_number.isin(
                members.meter_number
            )
        ].copy()

        pairs["candidate_transformer"] = (
            clean_id(
                pairs.candidate_transformer
            ).str.upper()
        )

        pairs["distance_raw"] = clean_id(
            pairs.distance_ft
        )

        pairs["distance_ft"] = (
            pd.to_numeric(
                pairs.distance_raw,
                errors="coerce",
            )
            .to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )

        pairs["division"] = division
        pairs["distance_status"] = "PASS"

        pairs.loc[
            pairs.distance_ft.gt(
                MAX_DISTANCE_FT
            ),
            "distance_status",
        ] = "DISTANCE_EXCEEDS_LIMIT"

        pairs.loc[
            pairs.distance_ft.lt(0),
            "distance_status",
        ] = "NEGATIVE_DISTANCE"

        pairs.loc[
            ~np.isfinite(
                pairs.distance_ft
            ),
            "distance_status",
        ] = "INVALID_DISTANCE"

        pairs.loc[
            pairs.distance_raw.isna(),
            "distance_status",
        ] = "MISSING_DISTANCE"

        parts.append(
            pairs.drop_duplicates()
        )

    fields = [
        "meter_number",
        "candidate_transformer",
        "division",
        "distance_ft",
        "distance_raw",
        "distance_status",
    ]

    raw = (
        pd.concat(
            parts,
            ignore_index=True,
        )
        if parts
        else pd.DataFrame(
            columns=fields
        )
    )

    ids = [
        "meter_number",
        "division",
        "candidate_transformer",
    ]

    conflict = (
        raw.drop_duplicates(
            ids + ["distance_raw"]
        )
        .duplicated(
            ids,
            keep=False,
        )
    )

    raw.loc[
        conflict,
        "distance_status",
    ] = "CONFLICTING_CANDIDATE_DISTANCES"

    lookup = raw.drop_duplicates(
        ids
    ).copy()

    lookup.loc[
        lookup.distance_status.eq(
            "CONFLICTING_CANDIDATE_DISTANCES"
        ),
        "distance_ft",
    ] = np.nan

    for role, target in [
        (
            "current",
            "current_transformer",
        ),
        (
            "model",
            "model_recommended_transformer",
        ),
        (
            "proposed",
            "proposed_final_transformer",
        ),
    ]:
        wanted = work[
            [
                "meter_number",
                "division",
                target,
            ]
        ].rename(
            columns={
                target: "candidate_transformer"
            }
        )

        found = wanted.merge(
            lookup[
                ids
                + [
                    "distance_ft",
                    "distance_status",
                ]
            ],
            on=ids,
            how="left",
            validate="many_to_one",
        )

        work[
            role + "_distance_ft"
        ] = found.distance_ft.to_numpy()

        status = (
            found.distance_status.fillna(
                "PAIR_NOT_IN_CANDIDATE_FILE"
            )
        )

        status = status.mask(
            wanted.candidate_transformer.isna(),
            "NO_TARGET",
        )

        status = status.mask(
            wanted.division.isna(),
            "MISSING_DIVISION",
        )

        work[
            role + "_distance_status"
        ] = status.to_numpy()

    hold(
        work,
        (
            work.review_reason.eq("")
            & work.proposed_distance_status.ne(
                "PASS"
            )
        ),
        "CANDIDATE_CHECK",
    )

    return work, raw


def transformer_metadata(transformers):
    tx = transformers.copy()

    for c in tx:
        tx[c] = (
            clean_id(tx[c])
            .str.upper()
        )

    values = tx[
        [
            "floc",
            "division",
            "opco",
        ]
    ].drop_duplicates()

    conflicts = (
        values.groupby("floc")
        .size()
        .gt(1)
    )

    meta = (
        values
        .drop_duplicates("floc")
        .set_index("floc")
    )

    meta["conflicting_metadata"] = conflicts

    return (
        meta,
        tx.groupby("floc")
        .sapid.nunique()
        .rename("SAPID_count"),
    )


def invalid_transformer_metadata(
    frame,
    meta,
    target="proposed_final_transformer",
):
    old = frame.current_transformer
    new = frame[target]

    bad = frame.division.isna()

    for field in [
        "division",
        "opco",
    ]:
        a = old.map(meta[field])
        b = new.map(meta[field])

        bad |= (
            a.isna()
            | b.isna()
            | a.ne(b).fillna(True)
        )

        if field == "division":
            bad |= (
                a.ne(
                    frame.division
                ).fillna(True)
                | b.ne(
                    frame.division
                ).fillna(True)
            )

    return (
        bad
        | old.map(
            meta.conflicting_metadata
        ).eq(True)
        | new.map(
            meta.conflicting_metadata
        ).eq(True)
    )


def attach_history(work, sent):
    joined = lambda values: "; ".join(
        sorted(
            set(
                values
                .dropna()
                .astype(str)
            )
        )
    )

    ledger = (
        sent.groupby(
            "meter_number",
            sort=False,
        )
        .agg(
            is_model_2=(
                "is_model_2",
                "max",
            ),
            historical_targets=(
                "target_floc",
                joined,
            ),
            source_files=(
                "source_file",
                joined,
            ),
        )
        .reset_index()
    )

    work = work.merge(
        ledger,
        on="meter_number",
        how="left",
        validate="one_to_one",
    )

    work["already_sent"] = (
        work.meter_number.isin(
            ledger.meter_number
        )
    )

    work["is_model_2"] = (
        work.is_model_2.eq(True)
    )

    work["is_change"] = (
        work.proposed_final_transformer.notna()
        & work.proposed_final_transformer.ne(
            work.current_transformer
        ).fillna(True)
    )

    conflicts = (
        sent
        .dropna(
            subset=["target_floc"]
        )
        .merge(
            work[
                [
                    "meter_number",
                    "proposed_final_transformer",
                ]
            ],
            on="meter_number",
            how="inner",
        )
    )

    conflicting_meters = conflicts.loc[
        conflicts.target_floc.ne(
            conflicts.proposed_final_transformer
        ).fillna(True),
        "meter_number",
    ]

    hold(
        work,
        (
            work.already_sent
            & work.is_change
        )
        | work.meter_number.isin(
            conflicting_meters
        ),
        "HISTORICAL_BATCH_PENDING_OR_DIFFERENT_TARGET",
    )

    return work, ledger


def post_state(population, batch):
    state = population.loc[
        population.in_ami,
        [
            "meter_number",
            "division",
            "connection_object",
            "premise_id",
            "current_transformer",
            "subgroup_id",
        ],
    ].copy()

    state["final_floc"] = (
        state.meter_number.map(
            batch
            .set_index("meter_number")
            .proposed_final_transformer
        )
        .fillna(
            state.current_transformer
        )
    )

    return state


def transformer_counts(state, sapids):
    counts = state.groupby(
        "final_floc"
    ).agg(
        group_count=(
            "subgroup_id",
            "nunique",
        ),
        meter_count=(
            "meter_number",
            "nunique",
        ),
        customer_count=(
            "premise_id",
            "nunique",
        ),
    )

    counts["ungrouped_meters"] = (
        state.loc[
            state.subgroup_id.isna()
        ]
        .groupby("final_floc")
        .size()
    )

    counts = (
        counts
        .join(
            state.groupby(
                "current_transformer"
            )
            .size()
            .rename("meters_before"),
            how="outer",
        )
        .join(sapids)
    )

    numeric = [
        "group_count",
        "meter_count",
        "customer_count",
        "ungrouped_meters",
        "meters_before",
    ]

    counts[numeric] = (
        counts[numeric]
        .fillna(0)
        .astype(int)
    )

    counts["groups_per_SAPID"] = (
        counts.group_count
        / counts.SAPID_count.replace(
            0,
            np.nan,
        )
    )

    counts["subgroup_SAPID_mismatch"] = (
        counts.group_count.ne(
            counts.SAPID_count
        )
    )

    counts["more_than_3_subgroups"] = (
        counts.group_count.gt(3)
    )

    counts["emptied"] = (
        counts.meters_before.gt(0)
        & counts.meter_count.eq(0)
    )

    counts.index.name = "floc"

    return counts

# 3. FINAL SELECTION — every hold is applied to a whole subgroup.


def select_batch(
    population,
    work,
    meta,
    sapids,
    co_history,
):
    hold(
        work,
        invalid_transformer_metadata(
            work,
            meta,
        ),
        "TRANSFORMER_DIVISION_OR_OPCO_METADATA",
    )

    hold(
        work,
        work.is_change
        & (
            work.connection_object.isna()
            | work.premise_id.isna()
        ),
        "MISSING_CO_OR_PREMISE",
    )

    previous_templates = (
        co_history.source_co.dropna()
    )

    previous_targets = (
        co_history.target_floc.dropna()
    )

    previous_premises = (
        co_history.premise_id.dropna()
    )

    events = []

    for _ in range(
        work.unit.nunique() + 1
    ):
        batch = work.loc[
            work.review_reason.eq("")
            & work.is_change
            & ~work.already_sent
        ].copy()

        state = post_state(
            population,
            batch,
        )

        counts = transformer_counts(
            state,
            sapids,
        )

        receiving = counts.reindex(
            batch.proposed_final_transformer
            .dropna()
            .unique()
        )

        flags = counts.loc[
            counts.emptied
            | (
                counts.index.isin(
                    receiving.index
                )
                & (
                    counts.subgroup_SAPID_mismatch
                    | counts.more_than_3_subgroups
                    | counts.SAPID_count.isna()
                )
            )
        ]

        events.append(
            flags.reset_index().assign(
                check="combined_selection",
                selection_round=_ + 1,
            )
        )

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & work.current_transformer.isin(
                counts.index[
                    counts.emptied
                ]
            ),
            "WOULD_EMPTY_CURRENT_TRANSFORMER",
        )

        good = (
            receiving.SAPID_count.gt(0)
            & receiving.group_count.eq(
                receiving.SAPID_count
            )
        )

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & work.proposed_final_transformer.isin(
                receiving.index[~good]
            ),
            "SUBGROUP_COUNT_NE_SAPID_COUNT",
        )

        if work.loc[
            batch.index,
            "review_reason",
        ].ne("").any():
            continue

        co_counts = (
            state.groupby(
                "connection_object"
            )
            .final_floc.nunique(
                dropna=False
            )
        )

        missing_cos = state.loc[
            state.final_floc.isna(),
            "connection_object",
        ].dropna()

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & work.connection_object.isin(
                missing_cos
            ),
            "CO_HAS_MISSING_ASSIGNMENTS",
        )

        needs = batch.loc[
            batch.connection_object.map(
                co_counts
            ).gt(1)
        ].copy()

        overlap = (
            needs.connection_object.isin(
                previous_templates
            )
            | needs.proposed_final_transformer.isin(
                previous_targets
            )
        )

        hold(
            work,
            work.index.isin(
                needs.index[overlap]
            )
            | (
                work.index.isin(
                    batch.index
                )
                & work.premise_id.isin(
                    previous_premises
                )
            ),
            "CO_REQUEST_ALREADY_IN_HISTORY",
        )

        links = needs[
            [
                "premise_id",
                "proposed_final_transformer",
            ]
        ].drop_duplicates()

        ambiguous = links.loc[
            links.premise_id.duplicated(
                False
            ),
            "premise_id",
        ]

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & work.premise_id.isin(
                ambiguous
            ),
            "PREMISE_HAS_MULTIPLE_TARGETS",
        )

        if work.loc[
            batch.index,
            "review_reason",
        ].ne("").any():
            continue

        moved = state.loc[
            state.premise_id.isin(
                links.premise_id
            )
        ].copy()

        target = (
            links.set_index(
                "premise_id"
            )
            .proposed_final_transformer
        )

        unsafe = (
            moved.final_floc.isna()
            | moved.final_floc.ne(
                moved.premise_id.map(
                    target
                )
            ).fillna(True)
            | moved.connection_object.isna()
            | moved.meter_number.isin(
                work.loc[
                    work.already_sent,
                    "meter_number",
                ]
            )
            | invalid_transformer_metadata(
                moved,
                meta,
                "final_floc",
            )
        )

        # Include sent meters outside the reviewed groups
        # in the premise safety check.
        unsafe |= moved.meter_number.isin(
            population.loc[
                population.historically_sent,
                "meter_number",
            ]
        )

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & work.premise_id.isin(
                moved.loc[
                    unsafe,
                    "premise_id",
                ]
            ),
            "WHOLE_PREMISE_MOVE_CONFLICT_OR_SENT_METER",
        )

        if work.loc[
            batch.index,
            "review_reason",
        ].ne("").any():
            continue

        links["planned_co"] = (
            "NEW::"
            + links.proposed_final_transformer
        )

        state["final_co"] = (
            state.premise_id.map(
                links.set_index(
                    "premise_id"
                ).planned_co
            )
            .fillna(
                state.connection_object
            )
        )

        affected_cos = pd.concat(
            [
                batch.connection_object,
                moved.connection_object,
            ]
        ).dropna().unique()

        affected = state.loc[
            state.connection_object.isin(
                affected_cos
            )
        ]

        bad_final_cos = (
            affected.groupby(
                "final_co"
            )
            .final_floc.nunique(
                dropna=False
            )
            .loc[
                lambda x: x.gt(1)
            ]
            .index
        )

        bad_rows = affected.loc[
            affected.final_co.isna()
            | affected.final_floc.isna()
            | affected.final_co.isin(
                bad_final_cos
            )
        ]

        hold(
            work,
            work.index.isin(
                batch.index
            )
            & (
                work.connection_object.isin(
                    bad_rows.connection_object
                )
                | work.premise_id.isin(
                    bad_rows.premise_id
                )
            ),
            "UNRESOLVED_CO_CONFLICT",
        )

        dependency_units = work.loc[
            work.meter_number.isin(
                moved.meter_number
            ),
            "unit",
        ]

        pending_targets = batch.loc[
            batch.unit.isin(
                dependency_units
            ),
            "proposed_final_transformer",
        ]

        # Keep receiving-FLOC counts intact
        # while CO work is pending.
        dependency_units = batch.loc[
            batch.proposed_final_transformer.isin(
                pending_targets
            ),
            "unit",
        ]

        cohort = batch.loc[
            batch.unit.isin(
                dependency_units
            )
        ].copy()

        ready = batch.loc[
            ~batch.unit.isin(
                dependency_units
            )
        ].copy()

        ready_state = post_state(
            population,
            ready,
        )

        ready_counts = transformer_counts(
            ready_state,
            sapids,
        )

        empty_ready = ready_counts.index[
            ready_counts.emptied
        ]

        if len(empty_ready):
            events.append(
                ready_counts.loc[
                    empty_ready
                ]
                .reset_index()
                .assign(
                    check="ready_batch_alone"
                )
            )

        hold(
            work,
            work.index.isin(
                ready.index
            )
            & work.current_transformer.isin(
                empty_ready
            ),
            "READY_BATCH_ALONE_WOULD_EMPTY_TRANSFORMER",
        )

        ready_receiving = (
            ready_counts.reindex(
                ready.proposed_final_transformer
                .dropna()
                .unique()
            )
        )

        bad_ready = ready_receiving.index[
            ~(
                ready_receiving.SAPID_count.gt(
                    0
                )
                & ready_receiving.group_count.eq(
                    ready_receiving.SAPID_count
                )
            )
        ]

        hold(
            work,
            work.index.isin(
                ready.index
            )
            & work.proposed_final_transformer.isin(
                bad_ready
            ),
            "READY_BATCH_ALONE_SUBGROUP_SAPID_MISMATCH",
        )

        ready_co = (
            ready_state.groupby(
                "connection_object"
            )
            .final_floc.nunique(
                dropna=False
            )
        )

        hold(
            work,
            work.index.isin(
                ready.index
            )
            & work.connection_object.isin(
                ready_co.index[
                    ready_co.gt(1)
                ]
            ),
            "READY_BATCH_ALONE_CO_CONFLICT",
        )

        if work.loc[
            batch.index,
            "review_reason",
        ].ne("").any():
            continue

        co_summary = (
            affected.groupby(
                "connection_object"
            )
            .agg(
                total_ami_meters=(
                    "meter_number",
                    "size",
                ),
                current_floc_count=(
                    "current_transformer",
                    "nunique",
                ),
                final_floc_count=(
                    "final_floc",
                    "nunique",
                ),
                planned_co_count=(
                    "final_co",
                    "nunique",
                ),
            )
            .reset_index()
        )

        return {
            "batch": batch,
            "state": state,
            "needs": needs,
            "co_cohort": cohort,
            "ready": ready,
            "links": links,
            "moved": moved,
            "counts": counts,
            "co_summary": co_summary,
            "count_events": (
                pd.concat(
                    [
                        e
                        for e in events
                        if not e.empty
                    ],
                    ignore_index=True,
                ).drop_duplicates()
                if any(
                    not e.empty
                    for e in events
                )
                else events[0]
            ),
        }

    raise RuntimeError(
        "Selection did not stabilize."
    )


COMPACT = [
    "meter_number",
    "division",
    "subgroup_id",
    "connection_object",
    "premise_id",
    "current_transformer",
    "model_recommended_transformer",
    "proposed_final_transformer",
    "Final_Recommended_FLOC",
    "current_distance_ft",
    "current_distance_status",
    "model_distance_ft",
    "model_distance_status",
    "proposed_distance_ft",
    "proposed_distance_status",
    "total_group_size",
    "distinct_flocs",
    "leading_percentage",
    "status",
    "review_reason",
    "already_sent",
    "source_files",
]


def write_csv(
    frame,
    path,
    compact=False,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if compact:
        frame = frame[
            [
                c
                for c in COMPACT
                if c in frame
            ]
        ]

    frame.to_csv(
        path,
        index=False,
    )


def write_excel(path, tables):
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:
        for name, frame in tables.items():
            if len(frame) > 1_048_575:
                raise ValueError(
                    f"Too many rows for Excel tab "
                    f"{name}; narrow DIVISIONS."
                )

            frame.to_excel(
                writer,
                sheet_name=name,
                index=False,
            )

            sheet = writer.sheets[name]

            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = (
                sheet.dimensions
            )

            for i, column in enumerate(
                frame.columns,
                1,
            ):
                sample = (
                    frame.iloc[
                        :100,
                        i - 1,
                    ]
                    .astype("string")
                    .str.len()
                    .max()
                )

                width = max(
                    len(str(column)),
                    (
                        0
                        if pd.isna(sample)
                        else sample
                    ),
                )

                sheet.column_dimensions[
                    sheet.cell(
                        1,
                        i,
                    ).column_letter
                ].width = min(
                    48,
                    max(
                        16,
                        width + 2,
                    ),
                )


def build_co_workbook(
    plan,
    work,
):
    needs = plan["needs"]

    links = plan[
        "links"
    ].copy()

    moved = plan[
        "moved"
    ].copy()

    es55 = (
        needs.groupby(
            "proposed_final_transformer",
            sort=True,
        )
        .connection_object.min()
        .reset_index()
    )

    es55 = es55.rename(
        columns={
            "proposed_final_transformer": (
                "final_recommended_floc"
            ),
            "connection_object": (
                "Connection_Object_Template"
            ),
        }
    )

    es55["Create CO"] = [
        f"_{i}"
        for i in range(
            1,
            len(es55) + 1,
        )
    ]

    es55["Automation_ID"] = [
        f"New_Connection_Object_{i}"
        for i in range(
            1,
            len(es55) + 1,
        )
    ]

    es55[
        "New_Connection_Object_Created"
    ] = ""

    automation = (
        es55.set_index(
            "final_recommended_floc"
        )
        .Automation_ID
    )

    links["Automation_ID"] = (
        links.proposed_final_transformer.map(
            automation
        )
    )

    es64 = links.rename(
        columns={
            "premise_id": "PremiseID",
            "proposed_final_transformer": (
                "final_recommended_floc"
            ),
        }
    )[
        [
            "Automation_ID",
            "PremiseID",
            "final_recommended_floc",
        ]
    ]

    es64[
        "New_Connection_Object_Created"
    ] = ""

    moved["Automation_ID"] = (
        moved.final_floc.map(
            automation
        )
    )

    moved["status"] = (
        "NEEDS_CONNECTION_OBJECT"
    )

    duplicates = (
        moved.loc[
            moved.premise_id.duplicated(
                False
            )
        ]
        .sort_values(
            [
                "premise_id",
                "meter_number",
            ]
        )
    )

    # Every subgroup correction waiting on CO work
    # appears here, even if its own CO is unchanged.
    dependent = plan[
        "co_cohort"
    ][
        [
            "meter_number",
            "connection_object",
            "current_transformer",
            "proposed_final_transformer",
        ]
    ].copy()

    new_by_meter = (
        moved.set_index(
            "meter_number"
        )
        .Automation_ID
    )

    dependent[
        "connection_object_after_ES64"
    ] = (
        dependent.meter_number.map(
            new_by_meter
        )
        .fillna(
            dependent.connection_object
        )
    )

    tables = {
        "ES55": es55,
        "ES64": es64,
        "Meters to Move": moved,
        "Duplicate Premises": duplicates,
        "Corrections After CO": dependent,
        "Validation Recommendations": work.loc[
            work.meter_number.isin(
                plan["co_cohort"].meter_number
            ),
            COMPACT,
        ],
        "CO Summary": plan["co_summary"],
    }

    return tables, moved


# 4. RUN / EXPORT — timestamped outputs; source files are never modified.


def review(
    recommendations,
    ami,
    itron,
    transformers,
    pairs_dir,
    history_dir,
):
    population, work = prepare_population(
        recommendations,
        ami,
        itron,
    )

    work, summary = apply_votes(
        work
    )

    work, candidates = check_distances(
        work,
        pairs_dir,
    )

    (
        sent,
        co_history,
        history_files,
    ) = read_history(
        history_dir
    )

    work, ledger = attach_history(
        work,
        sent,
    )

    population[
        "historically_sent"
    ] = population.meter_number.isin(
        ledger.meter_number
    )

    meta, sapids = transformer_metadata(
        transformers
    )

    plan = select_batch(
        population,
        work,
        meta,
        sapids,
        co_history,
    )

    approved = (
        work.review_reason.eq("")
        & ~work.already_sent
    )

    work[
        "Final_Recommended_FLOC"
    ] = (
        work.proposed_final_transformer.where(
            approved,
            work.current_transformer,
        )
    )

    work["is_change"] = (
        work.meter_number.isin(
            plan["batch"].meter_number
        )
    )

    work["added_by_subgroup"] = (
        work.is_change
        & ~work.in_input
    )

    work["final_matches_model"] = (
        approved
        & work.in_input
        & work.Final_Recommended_FLOC.eq(
            work.model_recommended_transformer
        ).fillna(False)
    )

    work["status"] = "CORRECT"

    work.loc[
        work.is_change,
        "status",
    ] = "READY_TO_SEND"

    co_meters = pd.concat(
        [
            plan[
                "co_cohort"
            ].meter_number,
            plan[
                "moved"
            ].meter_number,
        ]
    )

    work.loc[
        work.meter_number.isin(
            co_meters
        ),
        "status",
    ] = "NEEDS_CONNECTION_OBJECT"

    work.loc[
        work.review_reason.ne(""),
        "status",
    ] = "TO_REVIEW"

    work.loc[
        work.already_sent,
        "status",
    ] = "SENT"

    work.loc[
        work.already_sent
        & work.is_model_2,
        "status",
    ] = "MODEL_2_SENT"

    summary["review_reason"] = (
        work.groupby(
            "unit"
        )
        .review_reason.first()
    )

    summary["decision"] = np.where(
        summary.review_reason.eq(""),
        "ACCEPTED",
        "TO_REVIEW",
    )

    ledger = ledger.merge(
        population.loc[
            population.in_ami,
            [
                "meter_number",
                "division",
                "current_transformer",
            ],
        ],
        on="meter_number",
        how="left",
        validate="one_to_one",
    )

    ledger[
        "historical_target_matches_AMI"
    ] = (
        ledger.historical_targets.ne("")
        & ledger.historical_targets.eq(
            ledger.current_transformer
        ).fillna(False)
    )

    tables, moved = build_co_workbook(
        plan,
        work,
    )

    # ES64 may move unchanged AMI members that
    # were not in the initial review population.
    extra = (
        moved.loc[
            ~moved.meter_number.isin(
                work.meter_number
            )
        ]
        .rename(
            columns={
                "final_floc": (
                    "Final_Recommended_FLOC"
                )
            }
        )
        .copy()
    )

    extra[
        "proposed_final_transformer"
    ] = extra.Final_Recommended_FLOC

    extra["review_reason"] = ""
    extra["already_sent"] = False
    extra["in_input"] = False

    snapshot = (
        pd.concat(
            [
                work,
                extra,
            ],
            ignore_index=True,
            sort=False,
        )
        if not extra.empty
        else work.copy()
    )

    return {
        "work": work,
        "summary": summary.reset_index(),
        "candidates": candidates,
        "ledger": ledger,
        "co_history": co_history,
        "history_files": history_files,
        "plan": plan,
        "co_tables": tables,
        "snapshot": snapshot,
    }


def track_changes(
    snapshot,
    history_dir,
    output_dir,
    now,
    counts,
):
    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest = (
        history_dir
        / "latest_state.parquet"
    )

    fields = [
        "meter_number",
        "status",
        "subgroup_id",
        "connection_object",
        "premise_id",
        "current_transformer",
        "proposed_final_transformer",
        "Final_Recommended_FLOC",
        "current_distance_ft",
        "proposed_distance_ft",
        "proposed_distance_status",
        "review_reason",
        "already_sent",
    ]

    current = snapshot.reindex(
        columns=fields
    )

    existed = latest.exists()

    previous = (
        pd.read_parquet(
            latest
        )
        if existed
        else pd.DataFrame(
            columns=fields
        )
    )

    delta = previous.merge(
        current,
        on="meter_number",
        how="outer",
        suffixes=(
            "_before",
            "_after",
        ),
        indicator=True,
    )

    changed = delta._merge.ne(
        "both"
    )

    for field in fields[1:]:
        changed |= (
            delta[
                field + "_before"
            ]
            .astype("string")
            .fillna("<missing>")
            .ne(
                delta[
                    field + "_after"
                ]
                .astype("string")
                .fillna("<missing>")
            )
        )

    delta["change"] = (
        delta._merge
        .astype("string")
        .map(
            {
                "left_only": "LEFT_SCOPE",
                "right_only": (
                    "ENTERED_SCOPE"
                    if existed
                    else "INITIAL_SNAPSHOT"
                ),
                "both": "CHANGED",
            }
        )
    )

    write_csv(
        delta.loc[
            changed
        ].drop(
            columns="_merge"
        ),
        output_dir
        / "daily_changes.csv",
    )

    log_path = (
        history_dir
        / "daily_log.csv"
    )

    record = {
        "run_time": now.isoformat(),
        "output_folder": str(
            output_dir
        ),
        **counts,
        "changed_rows": int(
            changed.sum()
        ),
    }

    log = pd.concat(
        [
            (
                pd.read_csv(
                    log_path
                )
                if log_path.exists()
                else pd.DataFrame()
            ),
            pd.DataFrame(
                [record]
            ),
        ],
        ignore_index=True,
    )

    log.to_csv(
        history_dir
        / "daily_log.tmp.csv",
        index=False,
    )

    (
        history_dir
        / "daily_log.tmp.csv"
    ).replace(
        log_path
    )

    current.to_parquet(
        history_dir
        / "latest_state.tmp.parquet",
        index=False,
    )

    (
        history_dir
        / "latest_state.tmp.parquet"
    ).replace(
        latest
    )


def export_results(
    result,
    output_root=OUTPUT_ROOT,
):
    # Use the computer's local time,
    # including on Windows.
    now = datetime.now().astimezone()

    output_root = Path(
        output_root
    )

    output = (
        output_root
        / now.strftime(
            "%Y-%m-%d"
        )
        / now.strftime(
            "%H%M%S_%f"
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    work = result["work"]
    plan = result["plan"]

    for status, folder in [
        (
            "CORRECT",
            "correct_meters",
        ),
        (
            "NEEDS_CONNECTION_OBJECT",
            "needs_connection_object",
        ),
    ]:
        write_csv(
            result["snapshot"].loc[
                result["snapshot"].status.eq(
                    status
                )
            ],
            output
            / folder
            / "meters.csv",
            compact=True,
        )

    # All members of rejected groups, even
    # sent/unchanged members, remain visible here.
    held = work.loc[
        work.review_reason.ne("")
    ]

    write_csv(
        held,
        output
        / "to_review"
        / "meters.csv",
        compact=True,
    )

    distance_bad = (
        work.current_distance_status.ne(
            "PASS"
        )
        | work.proposed_distance_status.ne(
            "PASS"
        )
        | (
            work.in_input
            & ~work.model_distance_status.isin(
                [
                    "PASS",
                    "NO_TARGET",
                ]
            )
        )
    )

    distance_members = work.loc[
        work.unit.isin(
            work.loc[
                distance_bad,
                "unit",
            ]
        )
    ]

    write_csv(
        distance_members,
        output
        / "to_review"
        / "distance_flags.csv",
        compact=True,
    )

    distance_context = pd.concat(
        [
            held.meter_number,
            distance_members.meter_number,
        ]
    )

    write_csv(
        result["candidates"].loc[
            result[
                "candidates"
            ].meter_number.isin(
                distance_context
            )
        ],
        output
        / "to_review"
        / "all_candidate_distances.csv",
    )

    write_csv(
        result["summary"],
        output
        / "subgroup_summary.csv",
    )

    impacted = pd.concat(
        [
            work.current_transformer,
            work.proposed_final_transformer,
        ]
    ).dropna().unique()

    tx_counts = plan[
        "counts"
    ].loc[
        plan["counts"].index.isin(
            impacted
        )
    ]

    write_csv(
        tx_counts.reset_index(),
        output
        / "transformer_counts.csv",
    )

    write_csv(
        plan["count_events"],
        output
        / "to_review"
        / "transformer_count_flags.csv",
    )

    write_csv(
        plan["co_summary"],
        output
        / "connection_object_summary.csv",
    )

    co_history = result[
        "co_history"
    ]

    relevant = (
        co_history.source_co.isin(
            work.connection_object.dropna()
        )
        | co_history.target_floc.isin(
            work.proposed_final_transformer.dropna()
        )
        | co_history.premise_id.isin(
            work.premise_id.dropna()
        )
    )

    write_csv(
        co_history.loc[
            relevant
        ],
        output
        / "to_review"
        / "historical_CO_requests.csv",
    )

    write_csv(
        result["history_files"],
        output
        / "historical_files_checked.csv",
    )

    write_csv(
        result["ledger"],
        output
        / "sent_meters"
        / "sent_meters.csv",
    )

    write_csv(
        result["ledger"].loc[
            result["ledger"].is_model_2.eq(
                True
            )
        ],
        output
        / "model_2_sent_meters"
        / "meters.csv",
    )

    # The four-column corrections file is for
    # existing COs and contains no sent meters.
    corrections = plan["ready"][
        [
            "connection_object",
            "meter_number",
            "current_transformer",
            "proposed_final_transformer",
        ]
    ].rename(
        columns={
            "connection_object": (
                "Connection Object"
            ),
            "current_transformer": (
                "current_floc"
            ),
            "proposed_final_transformer": (
                "recommended_floc"
            ),
        }
    )

    batch_name = (
        f"batch_{BATCH_LABEL}_"
        f"{now:%Y%m%d_%H%M%S_%f}"
    )

    write_csv(
        corrections,
        output
        / "ready_to_send"
        / f"{batch_name}_Corrections.csv",
    )

    write_excel(
        output
        / "ready_to_send"
        / f"{batch_name}_Corrections.xlsx",
        {
            "Corrections": corrections
        },
    )

    write_excel(
        output
        / "needs_connection_object"
        / f"{batch_name}_New_Connection_Objects.xlsx",
        result["co_tables"],
    )

    result["snapshot"].to_parquet(
        output
        / "all_member_audit.parquet",
        index=False,
    )

    counts = {
        status: int(
            result[
                "snapshot"
            ].status.eq(
                status
            ).sum()
        )
        for status in [
            "TO_REVIEW",
            "CORRECT",
            "READY_TO_SEND",
            "NEEDS_CONNECTION_OBJECT",
            "SENT",
            "MODEL_2_SENT",
        ]
    }

    counts["new_CO_requests"] = len(
        result["co_tables"]["ES55"]
    )

    counts[
        "historical_sent_meters"
    ] = len(
        result["ledger"]
    )

    counts[
        "historical_model_2_sent_meters"
    ] = int(
        result[
            "ledger"
        ].is_model_2.eq(
            True
        ).sum()
    )

    settings = {
        "recommendations": str(
            RECOMMENDATIONS_PATH
        ),
        "ami": str(
            AMI_PATH
        ),
        "itron": str(
            ITRON_PATH
        ),
        "transformers": str(
            TRANSFORMERS_PATH
        ),
        "candidate_pairs": str(
            CANDIDATE_PAIRS_DIR
        ),
        "historical_folder": str(
            HISTORY_DIR
        ),
        "divisions": DIVISIONS,
        "max_distance_ft": (
            MAX_DISTANCE_FT
        ),
        "subgroup_SAPID_rule": (
            "group_count == distinct "
            "SAPID_count; no size cap"
        ),
        "counts": counts,
    }

    (
        output
        / "settings.json"
    ).write_text(
        json.dumps(
            settings,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output
        / "Conditions_Applied.txt"
    ).write_text(
        RULES.replace(
            "MAX_DISTANCE_FT",
            str(
                MAX_DISTANCE_FT
            ),
        ),
        encoding="utf-8",
    )

    scope = (
        "all"
        if DIVISIONS is None
        else "_".join(
            sorted(
                key(d)
                for d in DIVISIONS
            )
        )
    )

    track_changes(
        result["snapshot"],
        output_root
        / "history"
        / scope,
        output,
        now,
        counts,
    )

    print(
        pd.Series(
            counts,
            name="count",
        ).to_string()
    )

    print(
        f"\nSaved review: {output}"
    )

    return output

def run_final_review_stage():
    required = [
        "meter_number",
        "division",
        "recommended_transformer",
        "subgroup_id",
    ]

    optional = [
        "current_transformer",
        "score_gap",
        "current_distance_ft",
        "recommended_distance_ft",
    ]

    available = pq.read_schema(
        RECOMMENDATIONS_PATH
    ).names

    if not set(required).issubset(available):
        raise ValueError(
            f"Recommendations need "
            f"{required}; found {available}"
        )

    rec = pd.read_parquet(
        RECOMMENDATIONS_PATH,
        columns=(
            required
            + [
                c
                for c in optional
                if c in available
            ]
        ),
    )

    result = review(
        rec,
        read_columns(
            AMI_PATH,
            AMI_COLUMNS,
        ),
        read_columns(
            ITRON_PATH,
            ITRON_COLUMNS,
        ),
        read_columns(
            TRANSFORMERS_PATH,
            TRANSFORMER_COLUMNS,
        ),
        CANDIDATE_PAIRS_DIR,
        HISTORY_DIR,
    )

    output = export_results(
        result
    )

    return output


if __name__ == "__main__":
    run_final_review_stage()