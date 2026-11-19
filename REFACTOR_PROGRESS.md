# Meter-to-transformer refactoring

Branch `refactor/complete-mtx-modularization` starts at `6277de0ed9fe1cfbfc2353b3a7b65a0bd6af247c`. The comparison baseline is main at `fee119ec81dbe003b493ad03cafdb8fdc6e9039c`. Use `git rev-parse HEAD` for the exact checkout containing this record.

## Starting audit

| Requirement | Initial state | Evidence |
| --- | --- | --- |
| Preserve prior work | COMPLETE | Merged PRs #1–4 cover config/entry cleanup, maintenance utilities and test-selected thresholds. The unmerged base branch contains one subsequent commit with 14 compatibility wrappers and six archived modules. |
| Extract high-level stages | MISSING | The 594-line executable still orchestrates steps 1–16. The pipeline package contains only its initializer. |
| Organize modules and imports | PARTIAL | Canonical implementations exist. Compatibility and the runtime dependency graph need verification. |
| Export a review map | MISSING | Both main and the base reference an undefined MAP_HTML in the recommendation exporter. |
| Invalidate same-day caches | PARTIAL | Group reuse checks meter IDs but ignores ITRON assignment changes. The candidate manifest records the radius but reuse ignores it. Regular features already fingerprint group and candidate Parquet files. |
| Regression and operational equivalence | MISSING | No fixture, comparison outputs, test suite, or PR workflow run accompanies the staging commit. The reported earlier run is unverified. |
| Progress record, architecture and PR | PARTIAL | Continuation branch and this initial audit created. |

Next steps: extract orchestration, repair map and caches, run focused and paired offline regression checks, record results and open the stacked PR. No production connections or writes are necessary for offline validation.

## Architecture

The executable calls five stages in src/mtx/pipeline. Ingestion loads snapshots and division polygons. Preparation standardizes records, reconciles and records history, and filters invalid coordinates. Spatial processing writes groups, candidates and regular features. Modeling writes training splits, fits the ranker, selects the test cutoff and runs group review. Recommendations passes both full reference snapshots and filtered inputs to the existing checkpointed scorer/exporter. Exports remain inside that stage's existing lock. Production script paths and RunConfig defaults are retained. SMARTLINK_PROJECT_ROOT can explicitly direct the run to a separate data root.

The extraction compiled and the production entry point imported in an isolated Python environment. Current work proceeds to cache and export repairs, then fixture equivalence.

## Same-day cache repair

ITRON grouping now derives the canonical assignments once for the whole filtered population, compares the full division partition with the cached values, and replaces changed partitions. A radius mismatch invalidates the candidate completion marker before rewriting every partition, including meter/transformer status and no-candidate files. The existing single-writer cache convention, output locations and history writers are retained. A focused test passed for unchanged reuse, reassignment to standalone, downstream feature rebuilding, radius changes and interruption/retry.

## Review map

Main reproduced the missing-template NameError after writing its review workbooks on an offline fixture. The production exporter now reads a checked-in template and retains its existing JSON escaping and output path. Reviewers can inspect the meter and both transformer locations, edit status fields and download tracker-compatible CSV. Tile loading is optional. On the initial fixture the new entry point reached its completion marker (88 ready and 12 held). Paired output comparisons and separate final-review validation follow.
