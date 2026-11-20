# SmartLink recommendation and final-review outputs

Set `SMARTLINK_PROJECT_ROOT` to the same project directory in both stages. Run the existing pipeline to refresh recommendations, then run the existing final-review entry point against the matching AMI, ITRON, transformer and candidate snapshots.

```console
python scripts/run_meter_to_transformer_pipeline.py
python scripts/meter_to_transformer/prediction/smartlink_final_review.py
```

## Prediction-only runs (existing artifacts, no model retraining)

Set `SMARTLINK_PROJECT_ROOT` to your SmartLink data directory (the same root
used by all pipeline stages). The daily AMI, transformer, ITRON and Adam caches,
today's complete final-feature manifest, and at least one trained model must
already exist. No flag continues to run the full pipeline exactly as before.

```console
python scripts/run_meter_to_transformer_pipeline.py --predictions-only
python scripts/run_meter_to_transformer_pipeline.py --predictions-only --model-date 2026-10-20
python scripts/run_meter_to_transformer_pipeline.py --predictions-only --final-review
```

The default is the latest dated trained model; `--model-date` selects a specific
existing version. The score-gap cutoff comes **only** from that model's saved
`correction_filter.json` with `threshold_selected_on: test`. An absent or
unqualified cutoff fails instead of recalibrating on validation data. Features
are loaded from `data/cache/final_features/TODAY/` and checked against
`_complete.json`, including file hashes. The scorer also checks AMI assignments,
transformer references, geographic coordinates, division membership and model
feature compatibility. Old or incomplete feature snapshots cannot be reused.
Run the upstream stages first if today's final features do not exist.

In prediction-only mode source files are read directly from `data/cache` without
refreshing databases or rebuilding ITRON groups. Input standardization and
coordinate validation still run, but historical tracking and rejection-log writes
are skipped; existing submitted-batch and correction history is still consulted
by the recommendation stage. Previously unsent proposals and Adam MAP reviews
retain their existing scoring and attribution behavior. Re-running the same
date with different inputs may be rejected by the scoring checkpoint manifest.

`--final-review` is opt-in. With `--predictions-only --final-review` the
pipeline generates the original recommendations and then executes the existing
ITRON subgroup, candidate-distance, empty-transformer and CO checks. Without
`--final-review`, it only exports upstream recommendations. You can also
specify `--final-review` on a full pipeline run.

The model's test-derived cutoff was tuned on test labels; the resulting reported
test precision is not an unbiased deployment estimate.

## Which output to use

| Output under data/processed | Purpose |
| --- | --- |
| recommendations/DATE/predictions/DIVISION.parquet | Scored meters, including predictions that retain the current transformer. |
| recommendations/DATE/all_recommendations.parquet | One upstream proposal per meter. Includes model and eligible dated Adam MAP corrections, plus holds and their reasons. Division parquet files are partitions for existing consumers. |
| recommendations/DATE/final_recommendations.parquet | Proposals that passed the **upstream** gates. The Recommendations sheet is the same review stage. Review contains held proposals and Adam Review includes the manual proposals across both sets. |
| recommendations/DATE/division_summary.parquet | Upstream counts and test-derived estimates by division. |
| final_review/DATE/RUN/all_member_audit.parquet | Authoritative disposition by meter after subgroup, candidate, transformer, batch and whole-CO checks. Compare the input target, proposed final target and Final_Recommended_FLOC. Inspect upstream fields for the original decision, source, reason and review date. |
| final_review/DATE/RUN/ready_to_send | Four-column correction batch for existing COs. CSV and Excel are equivalent delivery formats. |
| final_review/DATE/RUN/needs_connection_object | ES55 creation rows keyed by source CO and target FLOC, ES64 rows per premise and all meters affected by the move. Validated requests bypass ordinary review. Dependent corrections wait here until the CO changes are reflected in the source data. |
| final_review/DATE/RUN/to_review | Held meters, distance evidence and CO_Review.xlsx. The workbook shows potential ES55 templates, premise associations and the other AMI members of affected COs for investigation. |
| final_review/DATE/RUN/sent_meters | Unique meters with submitted correction evidence, source files and known targets. |
| final_review/DATE/RUN/historical_events.parquet | Submission events, model coverage, proposals, historical Adam MAP evidence, archived CO records and final decisions. One meter can have many events. Use the event type and dates when tracing a revision. |
| final_review/DATE/RUN/daily_summary.csv | Overall and division counts as of the run. New proposals use the first observed meter/original/target association in completed recommendation archives. Generating the same proposal again does not make it new. |
| final_review/DATE/RUN/model_performance.csv | Validated proposal counts, correct/incorrect results and precision by evaluation date, model and division. ALL denotes the cumulative period or all divisions. |

Each final review writes a new timestamped directory. For a day's closing figures use the latest successful run for the same division scope. Keep dated archives to retain historical coverage. The existing daily_changes and history log track changes between runs.

## Review and submission evidence

- Adam proposals require a dated MAP-verified target that differs from current AMI. Conflicting targets are excluded. Explicit negative or ambiguous outcomes stay held. Original attribution and notes survive final review. A subgroup disagreement cannot authorize a different transformer on Adam's behalf. Archived Adam changes remain reporting evidence. A removed historical label alone cannot establish that a correction is still authorized.
- Prior unsent proposals can be scored again on a later day. Rejection follows the meter/original/target association across run IDs. The historical scoring option permits inspection of sent meters. Submission evidence still excludes them from ready exports.
- Put **actually submitted** CO workbooks in `data/reference/Historical Corrections/New CO` or set `SMARTLINK_CO_HISTORY_DIR` to the submitted archive. Keep drafts and generated workbooks in the output tree. A missing configured CO archive holds new split requests. An unreadable or unrecognized submitted workbook requires reconciliation.
- CO lifecycle values are NOT_REQUIRED, REVIEW_REQUIRED, READY_TO_SUBMIT, SENT, COMPLETED and HELD. HELD can identify a dependent correction. Explicit CO status is required for completion. An ES55 identifier alone does not demonstrate ES64 completion. A documented completion still inconsistent with AMI goes to review. The same target at an unrelated CO does not block a new request.
- Existing correction archives and All_Batches_Sent are read independently of the CO archive. An explicit implementation status is necessary to count a completed correction. A cross-model revision requires dated implementations by both models, a later Model 2 implementation and a different target. Supply `implementation_date`, `completed_date` or `applied_date` where documented. Archive rows without a model identifier retain unknown attribution.
- To supplement historical scoring coverage, supply `data/reference/model_review_history.csv` with meter_number, division, model_version (MODEL_1 or MODEL_2), and event_date. Without a full archive, documented coverage is a lower bound. Model 2 coverage uses operational scoring checkpoints, independently of whether a correction was proposed. Training and test examples do not count as operational coverage.
- To enable the optional read-only legacy NY division lookup, set `SMARTLINK_LEGACY_DIVISIONS=1`. Reporting prefers current AMI. A unique historical division can fill a gap. Conflicting historical names remain flagged. If the query is unavailable the run reports a warning and uses the other evidence. The lookup never supplies transformer or CO membership.
- The default subgroup majority rule is preserved. Set `SMARTLINK_GROUP_MAJORITY=0` to require consensus instead. Candidate, distance, subgroup membership, SAPID, whole-premise and empty-transformer checks still apply.

## Interpreting validation

TEST_TUNED uses the existing test-selected corrections. The threshold was selected on that test split. ADAM_FOLLOW_UP compares an unambiguous dated MAP target to the most recent earlier model proposal for the meter. Same-day ordering is unknown. OPERATIONAL_VALIDATION uses an optional `data/reference/operational_validation.csv` with recommendation_id, evaluation_date (YYYY-MM-DD) and verified_transformer. Provide a separately verified transformer for each ID. A missing evidence source produces no precision rows for that source.

Adam's current-confirmed and current-incorrect counts compare a recorded original assignment to the verified transformer. Where the recorded assignment is unavailable, the review contributes to coverage only. The distance summary uses available candidate distances for today's manual proposals. The pipeline has no verified transformer road identifier for a same-road metric.

## Rafa's offline checks

From the repository root in the SmartLink Python environment, run the full synthetic suite:

```console
python -m pytest -q
```

For focused coverage use `python -m pytest -q tests/test_adam_reviews.py tests/test_operational_review.py`. The offline integration tests run both production entry points with invented source snapshots. They also exercise same-day resume, prior-day reconsideration, the recommendation exports and original pipeline comparison.

To inspect saved training splits from PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m mtx.training.history_selection --run latest
python -m mtx.training.history_selection --run YYYY-MM-DD/HHMMSS_microseconds
```

Use `--history PATH` to select a different training archive. The utility checks readable train, validation and test files, declared feature columns and disjoint meter IDs. It prints each split path separately, the stored configuration and the recorded model reference when available. Older runs can be inspected without a model reference.
