"""Audit and safely stage a meter-to-transformer-only SmartLink repository cleanup.

The default action is a read-only report. --apply only stages Git changes in
a local checkout (it does not commit, push, or delete untracked data).
Run from the repository root:
    python scripts/refactor_m2t_repository.py
    python scripts/refactor_m2t_repository.py --apply

Existing production modules must not be moved or renamed until their callers,
imports, and tests have been updated together. This script intentionally uses a
conservative allowlist of legacy files and checks tracked references first.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

# Outside the ITRON -> candidates -> regular features -> ranker -> review
# -> recommendations flow. These are only candidates for removal.
LEGACY_CANDIDATES = (
    "src/mtx/modules/Calculate_Distance_Between_2_Points.py",
    "src/mtx/modules/Extract_Coordinates.py",
    "src/mtx/modules/Getting_Candidate_Transformers.py",
    "src/mtx/modules/Itron_Metrics.py",
    "src/mtx/modules/Main.py",
    "src/mtx/modules/grouping_meters.py",
    "src/mtx/modules/reading_cluster_data.py",
    "src/mtx/modules/time_series_voltage_plotter.py",
    "src/mtx/processing/weekly_voltages.py",
    "src/mtx/clustering/meter_pairs.py",
    "src/mtx/features/meter_pair_features.py",
    "src/mtx/features/tile_features.py",
    "src/mtx/features/transformer_tile_features.py",
    "src/mtx/data/cust_volt_read_data_extraction.py",
)

# Never automatically remove these: their purpose and possible external
# callers need a manual review, even when no static import is found.
PROTECTED = (
    "scripts/run_meter_to_transformer_pipeline.py",
    "scripts/meter_to_transformer/",
    "src/mtx/ingestion/",
    "src/mtx/history/",
    "src/mtx/grouping/",
    "src/mtx/candidates/",
    "src/mtx/db/",
    "src/mtx/utils/",
    "src/mtx/config.py",
    "src/mtx/run_config.py",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True, text=True, capture_output=True,
    ).stdout.strip()


def tracked_python() -> list[str]:
    return [name for name in git("ls-files", "*.py").splitlines() if name]


def imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def module_name(path: str) -> str:
    path = path.removeprefix("src/").removeprefix("scripts/")
    return path.removesuffix("/__init__.py").removesuffix(".py").replace("/", ".")


def references(candidate: str, python_files: list[str]) -> list[str]:
    module = module_name(candidate)
    tail = module.rsplit(".", 1)[-1]
    callers: list[str] = []
    for path in python_files:
        if path == candidate:
            continue
        source = (ROOT / path).read_text(encoding="utf-8")
        imported = imports(source)
        # Also check dynamic import / path string references conservatively.
        if any(
            item == module or item.startswith(module + ".")
            or module.startswith(item + ".") and item != "mtx"
            for item in imported
        ) or re.search(r"(?<![\\w])" + re.escape(tail) + r"(?![\\w])", source):
            callers.append(path)
    return callers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Stage safe deletions in Git")
    options = parser.parse_args()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise SystemExit("This script must live in scripts/ in a Git checkout.")
    if options.apply and git("status", "--porcelain"):
        raise SystemExit("Refusing to stage deletions: the working tree is not clean.")

    tracked = set(git("ls-files").splitlines())
    python_files = tracked_python()
    safe: list[str] = []
    for candidate in LEGACY_CANDIDATES:
        if candidate not in tracked:
            continue
        if any(candidate == item or candidate.startswith(item) for item in PROTECTED):
            print(f"PROTECTED {candidate}")
            continue
        callers = references(candidate, python_files)
        if callers:
            print(f"KEEP {candidate} (referenced by {', '.join(callers)})")
        else:
            print(f"REVIEW {candidate} (no tracked Python references found)")
            safe.append(candidate)

    print(f"\n{len(safe)} candidates with no tracked Python references.")
    print("Check external notebooks, scheduled jobs, and downstream users before removal.")
    if options.apply and safe:
        git("rm", "--", *safe)
        print("Staged deletions only. Inspect git diff --cached before committing.")
    elif safe:
        print("Read-only preview; use --apply to stage the deletions.")


if __name__ == "__main__":
    main()
