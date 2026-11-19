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
