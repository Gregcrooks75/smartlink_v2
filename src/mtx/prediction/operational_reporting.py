"""Read-only evidence aggregation for the timestamped final-review export."""

from pathlib import Path
import json
import re

import pandas as pd
import pyarrow.parquet as pq

from mtx.history.divisions import enrich_divisions


def columns(path, names):
    available = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in names if c in available]).reindex(
        columns=names
    )


def model_label(text):
    match = re.search(
        r"(?:^|[^a-z0-9])model[ _-]*([12])(?:$|[^a-z0-9])", str(text), re.I
    )
    return f"MODEL_{match[1]}" if match else "UNKNOWN"


def submission_events(result):
    frame = result["submissions"].copy()
    frame["model_version"] = frame.model_version.fillna(
        frame.source_file.map(model_label)
    )
    frame["submission_status"] = "SENT"
    frame.loc[
        frame.sent_status.str.upper().isin(
            ["APPLIED", "COMPLETED", "IMPLEMENTED", "INTEGRATED"]
        ),
        "submission_status",
    ] = "COMPLETED"
    # A master list often mirrors a dated batch. Keep the most specific evidence.
    action = ["meter_number", "original_transformer", "target_floc"]
    master = frame.source_file.str.lower().str.endswith("all_batches_sent.csv")
    mirrored = pd.MultiIndex.from_frame(frame[action]).isin(
        pd.MultiIndex.from_frame(frame.loc[~master, action])
    )
    frame = frame.loc[~(master & mirrored)].copy()
    stem = frame.source_file.map(lambda x: str(Path(x).with_suffix("")))
    frame["batch_id"] = frame.batch_id.fillna(stem)
    return frame.drop_duplicates(
        action + ["batch_id", "model_version", "submission_date", "submission_status"]
    )


def adam_events(path, asof):
    if not path.is_file():
        return pd.DataFrame()
    f = columns(
        path,
        [
            "meter_number",
            "Division",
            "ValidationSource",
            "ValidationDate",
            "Correct",
            "Correct_Transformer_FLOC",
        ],
    )
    f = f.rename(
        columns={
            "Division": "division",
            "Correct": "original_transformer",
            "Correct_Transformer_FLOC": "target_floc",
        }
    )
    f["event_date"] = pd.to_datetime(f.ValidationDate, errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    f = (
        f.loc[f.ValidationSource.eq("MAP") & f.event_date.le(asof)]
        .dropna(subset=["meter_number", "target_floc"])
        .copy()
    )
    f["meter_number"] = (
        f.meter_number.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    )
    for field in ["original_transformer", "target_floc"]:
        f[field] = f[field].astype("string").str.strip().str.upper()
    f["review_outcome"] = "UNKNOWN"
    known = f.original_transformer.notna()
    f.loc[known, "review_outcome"] = (
        f.loc[known, "original_transformer"]
        .eq(f.loc[known, "target_floc"])
        .map({True: "CURRENT_CONFIRMED", False: "CURRENT_INCORRECT"})
    )
    return f.assign(
        event_type="ADAM_REVIEW", recommendation_source="ADAM_REVIEW"
    ).drop_duplicates()


def revised_meters(events):
    # Only two documented implementations with dates and different FLOCs establish a revision.
    completed = (
        events.loc[events.submission_status.eq("COMPLETED")]
        .dropna(subset=["target_floc"])
        .copy()
    )
    completed["when"] = pd.to_datetime(
        completed.implementation_date, errors="coerce", utc=True
    )
    first, second = [
        completed.loc[
            completed.model_version.map(model_label).eq(label),
            ["meter_number", "when", "target_floc"],
        ]
        for label in ("MODEL_1", "MODEL_2")
    ]
    pairs = second.merge(first, on="meter_number", suffixes=("_second", "_first"))
    return set(
        pairs.loc[
            pairs.when_second.gt(pairs.when_first)
            & pairs.target_floc_second.ne(pairs.target_floc_first),
            "meter_number",
        ]
    )


def evidence(result, data, day):
    submissions = submission_events(result)
    sources = [
        submissions.assign(
            event_type="SUBMISSION", event_date=submissions.submission_date
        )
    ]
    legacy_reviews = data / "reference/model_review_history.csv"
    if legacy_reviews.is_file():
        legacy = pd.read_csv(
            legacy_reviews,
            dtype="string",
            usecols=["meter_number", "division", "model_version", "event_date"],
        )
        legacy["event_date"] = pd.to_datetime(
            legacy.event_date, errors="raise"
        ).dt.strftime("%Y-%m-%d")
        sources.append(
            legacy.loc[legacy.event_date.le(day)]
            .drop_duplicates()
            .assign(event_type="MODEL_REVIEW")
        )
    archive = data / "processed/recommendations"
    for run in sorted(archive.glob("????-??-??")):
        if run.name > day or not (run / "_complete.json").is_file():
            continue
        for path in sorted((run / "predictions").glob("*.parquet")):
            f = columns(path, ["meter_number", "model_date", "recommended_transformer"])
            sources.append(
                f.rename(columns={"recommended_transformer": "target_floc"}).assign(
                    event_type="MODEL_REVIEW",
                    model_version="MODEL_2",
                    event_date=run.name,
                    division=path.stem,
                )
            )
        path = run / "all_recommendations.parquet"
        if path.is_file():
            f = columns(
                path,
                [
                    "meter_number",
                    "division",
                    "model_date",
                    "recommendation_source",
                    "current_transformer",
                    "recommended_transformer",
                    "review_flags",
                    "status",
                    "recommended_distance_ft",
                    "recommendation_id",
                ],
            )
            sources.append(
                f.rename(
                    columns={
                        "current_transformer": "original_transformer",
                        "recommended_transformer": "target_floc",
                        "status": "upstream_status",
                    }
                ).assign(event_type="RECOMMENDATION", event_date=run.name)
            )
    adam_root = data / "cache/adam"
    manual = [adam_events(adam_root / "adam_validated.parquet", day)]
    committed = adam_root / "history/adam_daily_summary.parquet"
    if committed.exists():
        dates = pd.read_parquet(committed, columns=["date"]).date.astype(str)
        for date in dates[dates.le(day)].unique():
            manual.append(
                adam_events(adam_root / f"history/changed_rows/{date}.parquet", day)
            )
    sources.append(pd.concat(manual, ignore_index=True).drop_duplicates())
    co = result["co_history"]
    sources.append(
        co.rename(columns={"request_state": "connection_object_status"}).assign(
            event_type="CO_ARCHIVE"
        )
    )
    # The current full-AMI division (or recovered legacy division) takes precedence.
    events = pd.concat(sources, ignore_index=True, sort=False)
    for field in ("division", "review_outcome", "recommendation_source"):
        if field not in events:
            events[field] = pd.NA
    meter_ids = events[["meter_number"]].dropna().drop_duplicates()
    divisions = meter_ids.merge(
        result["ami_divisions"], on="meter_number", how="left", validate="one_to_one"
    )
    divisions = enrich_divisions(divisions, result["legacy_divisions"]).set_index(
        "meter_number"
    )
    events["division_conflict"] = events.meter_number.map(divisions.division_conflict)
    events["division_source"] = events.meter_number.map(divisions.division_source)
    recorded = events.division.astype("string").str.strip().str.upper()
    resolved = events.meter_number.map(divisions.division).replace("UNKNOWN", pd.NA)
    events.loc[resolved.isna() & recorded.notna(), "division_source"] = "EVENT_RECORD"
    events["division"] = resolved.fillna(recorded).fillna("UNKNOWN")
    return events, revised_meters(submissions)


def summary(events, snapshot, day, revised):
    def ids(f, mask):
        return set(f.loc[mask, "meter_number"].dropna())

    rows = []
    divisions = sorted(set(events.division) | set(snapshot.division.dropna()))
    for division in [None, *divisions]:
        e = events if division is None else events.loc[events.division.eq(division)]
        s = (
            snapshot
            if division is None
            else snapshot.loc[snapshot.division.fillna("UNKNOWN").eq(division)]
        )
        # A labeled submission proves coverage even without a historical scoring file.
        seen = e.event_type.isin(["SUBMISSION", "MODEL_REVIEW"])
        labels = e.model_version.map(model_label)
        m1, m2 = [ids(e, seen & labels.eq(m)) for m in ("MODEL_1", "MODEL_2")]
        adam = e.loc[e.event_type.eq("ADAM_REVIEW")].copy()
        # Different review dates are events. Report latest classification once per meter.
        latest = adam.loc[
            adam.event_date.eq(adam.groupby("meter_number").event_date.transform("max"))
        ]
        conflicts = latest.groupby("meter_number").target_floc.nunique()
        latest = latest.copy()
        latest.loc[
            latest.meter_number.isin(conflicts[conflicts.gt(1)].index), "review_outcome"
        ] = "UNKNOWN"
        outcomes = latest.groupby("meter_number").review_outcome.agg(
            lambda v: v.iloc[0] if v.nunique() == 1 else "UNKNOWN"
        )
        correct = int(outcomes.eq("CURRENT_CONFIRMED").sum())
        incorrect = int(outcomes.eq("CURRENT_INCORRECT").sum())
        new = (
            e.loc[e.event_type.eq("RECOMMENDATION")]
            .groupby(
                ["meter_number", "original_transformer", "target_floc"], dropna=False
            )
            .event_date.min()
        )
        manual_distances = (
            pd.to_numeric(
                s.loc[
                    s.get(
                        "recommendation_source",
                        pd.Series(index=s.index, dtype="string"),
                    ).eq("ADAM_REVIEW"),
                    "model_distance_ft",
                ],
                errors="coerce",
            ).dropna()
            if "model_distance_ft" in s
            else pd.Series(dtype=float)
        )
        sent = e.event_type.eq("SUBMISSION")
        completed = sent & e.get(
            "submission_status", pd.Series(index=e.index, dtype="string")
        ).eq("COMPLETED")
        rows.append(
            dict(
                date=day,
                model_1_submitted=len(ids(e, sent & labels.eq("MODEL_1"))),
                model_2_submitted=len(ids(e, sent & labels.eq("MODEL_2"))),
                model_1_implemented=len(ids(e, completed & labels.eq("MODEL_1"))),
                model_2_implemented=len(ids(e, completed & labels.eq("MODEL_2"))),
                adam_proposals_with_distance=len(manual_distances),
                adam_median_proposal_distance_ft=manual_distances.median()
                if len(manual_distances)
                else None,
                division=division or "ALL",
                model_1_documented=len(m1),
                model_2_documented=len(m2),
                both_models=len(m1 & m2),
                submitted=len(ids(e, e.event_type.eq("SUBMISSION"))),
                model_1_revised_by_model_2=len(revised & ids(e, seen)),
                new_recommendations=int(new.eq(day).sum()),
                ready=len(ids(s, s.status.eq("READY_TO_SEND"))),
                held=len(ids(s, s.status.eq("TO_REVIEW"))),
                co_required=len(
                    ids(
                        s,
                        s.connection_object_status.isin(
                            ["READY_TO_SUBMIT", "REVIEW_REQUIRED", "HELD", "SENT"]
                        ),
                    )
                ),
                co_sent_meters=len(
                    ids(
                        e,
                        e.event_type.eq("CO_ARCHIVE")
                        & e.connection_object_status.eq("SENT"),
                    )
                    | ids(s, s.connection_object_status.eq("SENT"))
                ),
                adam_reviewed=len(outcomes),
                adam_current_confirmed=correct,
                adam_current_incorrect=incorrect,
                adam_confirmed_rate=correct / (correct + incorrect)
                if correct + incorrect
                else None,
            )
        )
    return pd.DataFrame(rows)


def write_reports(result, data, output, day):
    events, revised = evidence(result, data, day)
    snapshot = result["snapshot"].copy()
    # Preserve authoritative final decisions separately from the upstream proposals.
    current = snapshot.rename(
        columns={
            "current_transformer": "original_transformer",
            "proposed_final_transformer": "target_floc",
            "upstream_model_date": "model_date",
            "upstream_adam_validation_date": "review_date",
        }
    ).assign(event_date=day, event_type="FINAL_REVIEW")
    report = summary(events, snapshot, day, revised)
    # Count distinct ES55 actions separately from affected meters and ES64 rows.
    requests = result["co_history"].dropna(subset=["source_co", "target_floc"])
    scope = snapshot[["connection_object", "division"]].dropna().drop_duplicates()
    known = (
        scope.loc[~scope.connection_object.duplicated(keep=False)]
        .set_index("connection_object")
        .division
    )
    actions = requests.loc[requests.request_state.eq("SENT")].drop_duplicates(
        ["source_co", "target_floc"]
    )
    counted = actions.source_co.map(known).fillna("UNKNOWN").value_counts()
    report["co_archived_sent_actions"] = (
        report.division.map(counted).fillna(0).astype(int)
    )
    report.loc[report.division.eq("ALL"), "co_archived_sent_actions"] = len(actions)
    previous = []
    for folder in sorted(output.parent.parent.glob("????-??-??")):
        if folder.name >= day:
            continue
        for candidate in sorted(
            folder.glob("*/all_member_audit.parquet"), reverse=True
        ):
            settings = candidate.parent / "settings.json"
            if not settings.is_file() or json.loads(
                settings.read_text(encoding="utf-8")
            ).get("divisions") != result.get("divisions"):
                continue
            frame = columns(
                candidate,
                [
                    "meter_number",
                    "division",
                    "current_transformer",
                    "proposed_final_transformer",
                    "Final_Recommended_FLOC",
                    "status",
                    "review_reason",
                    "connection_object_status",
                    "recommendation_source",
                    "upstream_model_date",
                    "proposed_distance_ft",
                ],
            )
            previous.append(
                frame.rename(
                    columns={
                        "current_transformer": "original_transformer",
                        "proposed_final_transformer": "target_floc",
                        "upstream_model_date": "model_date",
                    }
                ).assign(event_type="FINAL_REVIEW", event_date=folder.name)
            )
            break
    pd.concat([events, *previous, current], ignore_index=True, sort=False).to_parquet(
        output / "historical_events.parquet", index=False
    )
    report.to_csv(output / "daily_summary.csv", index=False)
    performance(events, data, day).to_csv(output / "model_performance.csv", index=False)


def performance(events, data, day):
    """Precision counts only proposals with a documented ground-truth target."""
    samples = []
    for path in sorted(
        (data / "processed/models/transformer_ranker").glob(
            "*/test_selected_corrections.parquet"
        )
    ):
        if path.parent.name <= day:
            test = columns(path, ["meter_number", "division", "correct"])
            samples.append(
                test.dropna(subset=["correct"]).assign(
                    basis="TEST_TUNED",
                    evaluation_date=path.parent.name,
                    model_date=path.parent.name,
                )
            )
    proposals = events.loc[
        events.event_type.eq("RECOMMENDATION")
        & events.recommendation_source.eq("MODEL")
    ].copy()
    labels = events.loc[
        events.event_type.eq("ADAM_REVIEW"),
        ["meter_number", "event_date", "target_floc"],
    ].drop_duplicates()
    # Compare one preceding proposal to each later unambiguous review.
    # Same-day ordering is unknown, and a manual proposal is never its own validation.
    labels = labels.loc[
        ~labels.duplicated(["meter_number", "event_date"], keep=False)
    ].rename(columns={"target_floc": "truth", "event_date": "evaluation_date"})
    if not proposals.empty and not labels.empty:
        pairs = proposals.merge(labels, on="meter_number", how="inner")
        pairs = pairs.loc[pairs.event_date.lt(pairs.evaluation_date)].sort_values(
            "event_date"
        )
        selected = pairs.drop_duplicates(
            ["meter_number", "evaluation_date"], keep="last"
        )
        samples.append(
            selected.assign(
                basis="ADAM_FOLLOW_UP", correct=selected.target_floc.eq(selected.truth)
            )
        )
    path = data / "reference" / "operational_validation.csv"
    if path.is_file() and not proposals.empty:
        explicit = pd.read_csv(
            path,
            dtype="string",
            usecols=["recommendation_id", "evaluation_date", "verified_transformer"],
        )
        explicit = explicit.dropna().drop_duplicates()
        if explicit.recommendation_id.duplicated().any():
            raise ValueError(
                "Conflicting operational validations for a recommendation_id"
            )
        selected = proposals.merge(
            explicit, on="recommendation_id", validate="one_to_one"
        )
        selected["evaluation_date"] = pd.to_datetime(
            selected.evaluation_date, errors="raise"
        ).dt.strftime("%Y-%m-%d")
        selected = selected.loc[
            selected.evaluation_date.between(selected.event_date, day)
        ]
        samples.append(
            selected.assign(
                basis="OPERATIONAL_VALIDATION",
                correct=selected.target_floc.eq(selected.verified_transformer),
            )
        )
    fields = [
        "basis",
        "model_date",
        "evaluation_date",
        "division",
        "validated_meters",
        "evaluations",
        "correct",
        "incorrect",
        "precision",
    ]
    if not samples:
        return pd.DataFrame(columns=fields)
    sample = pd.concat(samples, ignore_index=True)
    sample["division"] = sample.division.fillna("UNKNOWN").str.upper()
    groups = ["basis", "model_date", "evaluation_date", "division"]
    rows = []
    for periods in (sample, sample.assign(evaluation_date="ALL")):
        for parts in (periods, periods.assign(division="ALL")):
            stats = (
                parts.groupby(groups, dropna=False)
                .agg(
                    validated_meters=("meter_number", "nunique"),
                    evaluations=("correct", "size"),
                    correct=("correct", "sum"),
                )
                .reset_index()
            )
            stats["incorrect"] = stats.evaluations - stats.correct
            stats["precision"] = stats.correct / stats.evaluations
            rows.append(stats)
    return pd.concat(rows, ignore_index=True)[fields]
