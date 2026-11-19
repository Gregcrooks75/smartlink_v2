# SmartLink meter-to-transformer refactoring

**Status — implemented and partially validated.** The synthetic pipeline, separate final review and same-day resume ran offline. Production snapshot equivalence and production-volume behavior remain unverified.

| Revision | Reference |
| --- | --- |
| Original pipeline on main | `fee119ec81dbe003b493ad03cafdb8fdc6e9039c` |
| Integration base | `refactor/meter-to-transformer-stages` at `6277de0ed9fe1cfbfc2353b3a7b65a0bd6af247c` |
| Continued branch | `refactor/complete-mtx-modularization`, already at `4b7edc3819ebdcccd8f75ec85db3bc8d4ad6168b` when this session began |
| Validated implementation and regression suite | `9e4583cbe8351547f77997833b5d94bc69d36da7` |
| Pull request | [#5](https://github.com/Gregcrooks75/smartlink_v2/pull/5) targeting the staging branch |

The closing documentation commit follows the validated implementation. `git rev-parse HEAD` gives the exact revision containing this record.

## Audit and ownership

| Requirement | At staging base | At continuation start | At handoff |
| --- | --- | --- | --- |
| High-level orchestration | MISSING | COMPLETE | COMPLETE. The 40-line executable calls five stages in order. |
| Production modules and old import paths | PARTIAL | PARTIAL | COMPLETE for tracked module imports and tested entry points. External notebook and scheduler callers are unverified. |
| Map export | MISSING | PARTIAL | COMPLETE on synthetic inputs, including export, embedded records, review edits and a local tracker import. |
| Same-day subgroup and radius invalidation | PARTIAL | PARTIAL | COMPLETE for the identified changes, unchanged reuse and interrupted rebuild. |
| Functional equivalence and final review | MISSING | PARTIAL | COMPLETE on the paired fixture. Production validation remains PARTIAL. |
| Architecture, reproducible tests, progress and PR | MISSING | PARTIAL | COMPLETE. |

Merged PRs #1–4 simplified configuration and the entry point, added maintenance utilities, and introduced the test-selected cutoff. The staging commit moved implementations under `src/mtx`, retained 14 historical script paths and archived six modules. The next four commits created this continuation branch's progress record, extracted the five stages, repaired both caches and supplied the missing review-map template. The earlier report of a 2,000-meter run had no accompanying tracked fixture or outputs.

This session fixed the separate final-review reader for the raw GIS `DivName` column while retaining configured `Division` precedence. It corrected the ranker's cutoff wording and the historical-corrections CLI's obsolete argument, repaired imports in two older analysis modules and consolidated duplicate structure loaders behind a compatibility path. Database reads in the old candidate-analysis module now occur on execution. Maintenance previews resolve the checkout from their moved paths. The new tests generate invented input data and exercise the unchanged production functions.

## Executable and stages

| Call from `scripts/run_meter_to_transformer_pipeline.py` | Inputs, outputs and responsibilities |
| --- | --- |
| `pipeline.ingestion.run_ingestion` | Loads AMI, GIS, ITRON and ADAM snapshots and both division shapefiles. Retains the full AMI population. |
| `pipeline.preparation.run_preparation` | Standardizes identifiers and divisions, retains the full transformer reference, reconciles historical batches, records changes, and filters invalid coordinates. |
| `pipeline.spatial.run_spatial_processing` | Writes division-scoped group assignments and history, radius-limited candidate partitions, then regular features. |
| `pipeline.modeling.run_modeling` | Applies training eligibility, builds grouped splits, trains the XGBoost ranker, selects the cutoff on test predictions and runs validation-group review. |
| `pipeline.recommendations.run_recommendations` | Applies the selected cutoff through the existing checkpointed scorer and exports review data under its lock. A missing qualifying cutoff stops scoring. |

The entry point retains its production defaults. Set `SMARTLINK_PROJECT_ROOT` before starting Python to redirect the active workflow to an isolated data root. The historical `scripts/meter_to_transformer/prediction/smartlink_final_review.py` remains a separate local workflow. It checks full subgroup membership, distances, prior batches, post-move transformer counts and connection objects before writing final batch files. The intermediate recommendations stage retains `include_previously_sent=True`. Its ready count and the separate final batch count represent different decisions.

## Verification

Run `python -m pytest -q` from a checkout with the original commit available. The paired test extracts the original tracked code into a fresh temporary directory and redirects its hard-coded root and two shapefile paths. The four database extractors return deterministic source frames. Ingestion, cache I/O, polygon checks, history, feature engineering, training, threshold selection, scoring and exports run normally. ODBC connections are blocked. Main and staging both failed at the undefined `MAP_HTML` after writing their workbooks. Their standalone final reviews completed using the supported column-map override. The continuation branch completed both entry points with the GIS defaults.

| Check | Result |
| --- | --- |
| Automated suite | 12 passed. Python compilation also passed for `src`, `scripts` and `tests`. |
| Imports and commands | All 106 tracked Python package/script modules imported with database connections blocked. Static local imports formed an acyclic graph. The production path reached 35 modules and no archive or older analysis modules. All 14 historical wrappers were checked, as were the four maintenance previews. |
| Inputs and coordinates | 2,001 raw meters, 2,000 valid, two divisions. One of 402 GIS rows was vertical, one failed the polygon check, and 400 valid transformers remained. Rejected identities and history rows matched. |
| Spatial and training | 4,000 candidate pairs at 550 ft. Of 1,900 reviewed meters, 1,899 qualified, yielding 3,798 labeled pairs and 56 source features. Train, validation and test contained 1,374, 258 and 267 meters. |
| Ranker and cutoff | Model bytes, ranks, prediction scores and threshold tables matched exactly. The selected gap was 0.09802815318107605. All 212 selected synthetic test changes matched their labels. That precision is in-sample because the cutoff was selected on those same test labels. |
| Recommendation decisions | 101 scored meters, 100 proposed changes, 84 ready for review export and 16 held. A full same-day rerun reused both division checkpoints with zero new scores and identical decisions. |
| Separate operational review | Subgroup expansion produced 104 audited meters. It marked 20 ready to send, 24 needing connection-object work, 59 for review, and one previously sent. The final batch contained 20 unique meters. Missing COs, a pending historical target and an empty-transformer case exercised blocking paths. Exactly 550 ft passed. An excessive, missing or conflicting target distance held the entire subgroup. |
| Cache changes | Reordered ITRON rows reused caches. A reassignment to standalone rebuilt its group and feature partitions. A radius change rebuilt the pairs, status and no-candidate outputs. A simulated interruption followed by a retry restored a consistent completed snapshot. |
| File equivalence | 136 common files matched against main and, separately, against staging. Tables and workbook cells were compared exactly after sorting unordered rows and normalizing run-specific metadata. Model bytes matched. The branch also produced the HTML, scoring summary and recommendation completion marker after the point where the original failed. |
| Map workflow | The JavaScript ran in an offline DOM test with 100 embedded records. Meter navigation, three location markers, status and notes edits and CSV download passed. The existing importer accepted the CSV in a synthetic tracker and preserved a note with a comma and quotes. Native Chromium rendering and live map tiles were unverified in this environment. |

The tested Linux environment used Python 3.12.14, pandas 2.2.3, NumPy 2.3.5, Polars 1.44.2, PyArrow 26.0.1, GeoPandas 1.2.0, Shapely 2.2.0, scikit-learn 1.8.0 and XGBoost CPU 3.4.1. The checkout also needs its geospatial I/O, ODBC, Excel, RapidFuzz and pytest dependencies. No historical proprietary fixture, live database connection or SAP submission was used.

## Outstanding validation

1. Run both revisions on copies of the same approved production snapshots, division layers, prior checkpoints and historical batches in separate output roots. Compare the eligibility audit, candidate partitions, feature rows, split membership, scores, threshold, holds, prior-sent exclusions and final workbooks.
2. Verify refresh against the deployment's read-only source connections, Windows paths, shared-drive behavior and runtime at production volume. Check the offline review map in the deployed browser.
3. The stacked PR is ready for human review. Integrate it with the staging branch and assess the combined changes against main before deployment. A separate untouched holdout is needed for an unbiased estimate of deployment precision.
