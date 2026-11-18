"""Stage archival of verified unused legacy SmartLink Python files.

Preview: python scripts/remove_unused_m2t_files.py
Apply:   python scripts/remove_unused_m2t_files.py --apply

Only files in LEGACY_CANDIDATES can be archived. The script checks Python
imports from retained modules, conservatively blocks uncertain references,
and uses git mv into legacy/ (staged moves, never an automatic commit or push).
Review scheduled jobs, notebooks and external callers before committing.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Legacy meter-to-meter / older duplicate implementations, NOT a blanket
# deletion of everything absent from static imports.
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

# Preserve operational scripts even if imported dynamically / called directly.
ENTRYPOINTS = (
    "scripts/run_meter_to_transformer_pipeline.py",
    "scripts/run_correction_review.py",
    "scripts/run_group_review.py",
    "scripts/run_historical_recommendations.py",
    "scripts/pipeline_checkpoints.py",
    "scripts/meter_to_transformer/",
    "src/mtx/ingestion/",
    "src/mtx/history/",
    "src/mtx/grouping/",
    "src/mtx/candidates/",
    "src/mtx/db/",
    "src/mtx/utils/",
    "src/mtx/features/regular_feature_policy.py",
    "src/mtx/run_config.py",
    "src/mtx/config.py",
)
# Administrative scripts contain lists of filenames, not actual dependencies.
IGNORE_CALLERS = {
    "scripts/refactor_m2t_repository.py",
    "scripts/remove_unused_m2t_files.py",
    "scripts/cleanup_pipeline.py",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def module_name(path: str) -> str:
    name = path.removeprefix("src/").removeprefix("scripts/")
    if name.endswith("/__init__.py"):
        name = name[:-12]
    elif name.endswith(".py"):
        name = name[:-3]
    return name.replace("/", ".")


def imported_modules(path: str, source: str) -> set[str]:
    tree = ast.parse(source, filename=path)
    package = module_name(path)
    if not path.endswith("/__init__.py"):
        package = package.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bits = package.split(".") if package else []
                up = node.level - 1
                prefix = ".".join(bits[:len(bits) - up]) if up <= len(bits) else ""
                base = ".".join(x for x in (prefix, node.module or "") if x)
            else:
                base = node.module or ""
            if base:
                found.add(base)
                found.update(base + "." + alias.name for alias in node.names)
    return found


def is_entry(path: str) -> bool:
    return any(path == item or path.startswith(item) for item in ENTRYPOINTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Stage verified moves into legacy/ with git mv")
    args = parser.parse_args()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        sys.exit("ERROR: Run this script from a checkout of its own repository.")
    if args.apply and git("status", "--porcelain"):
        sys.exit("ERROR: Working tree is not clean. Commit/stash changes first.")

    tracked = set(git("ls-files").splitlines())
    python_files = sorted(p for p in tracked if p.endswith(".py") and (ROOT / p).is_file())
    sources: dict[str, str] = {}
    imports: dict[str, set[str]] = {}
    for path in python_files:
        if path in IGNORE_CALLERS:
            continue
        try:
            source = (ROOT / path).read_text(encoding="utf-8")
            sources[path] = source
            imports[path] = imported_modules(path, source)
        except (SyntaxError, UnicodeError) as exc:
            sys.exit(f"ERROR: Cannot safely audit {path}: {exc}")

    modules = {module_name(p): p for p in python_files}
    # Reachability from operational entrypoints; modules referenced by retained
    # code are kept, including modules referenced only by older non-entry scripts.
    active = {p for p in sources if is_entry(p)}
    queue = deque(active)
    while queue:
        caller = queue.popleft()
        for name in imports[caller]:
            target = modules.get(name)
            if target and target in sources and target not in active:
                active.add(target)
                queue.append(target)

    safe: list[str] = []
    blocked = 0
    for candidate in LEGACY_CANDIDATES:
        if candidate not in tracked:
            print(f"ABSENT  {candidate}")
            continue
        if candidate in active or is_entry(candidate):
            print(f"KEEP    {candidate} (operational import/entrypoint)")
            blocked += 1
            continue

        name = module_name(candidate)
        basename = Path(candidate).name
        # Check all retained code, not only the entrypoint closure. This avoids
        # breaking utility scripts that users may still run independently.
        callers = []
        for path, source in sources.items():
            if path == candidate or path in LEGACY_CANDIDATES:
                continue
            used = imports[path]
            if (
                any(mod == name or mod.startswith(name + ".") for mod in used)
                or name in source
                or basename in source
            ):
                callers.append(path)
        if callers:
            print(f"KEEP    {candidate} (referenced by: {', '.join(callers)})")
            blocked += 1
        else:
            print(f"ARCHIVE {candidate} -> legacy/{candidate}")
            safe.append(candidate)

    print(f"\nEligible for archival: {len(safe)}; blocked: {blocked}.")
    print("This static audit cannot detect external notebooks, scheduled jobs, or string-built imports.")
    if not args.apply:
        print("Preview only. Run with --apply to stage eligible moves.")
    elif safe:
        for candidate in safe:
            destination = ROOT / "legacy" / candidate
            if destination.exists():
                sys.exit(f"ERROR: Archive destination already exists: {destination}")
        for candidate in safe:
            destination = ROOT / "legacy" / candidate
            destination.parent.mkdir(parents=True, exist_ok=True)
            git("mv", "--", candidate, str(destination.relative_to(ROOT)))
        print("Staged moves to legacy/. Review: git diff --cached --name-status")
        print("Undo before committing: git reset --mixed HEAD, then move files back manually.")
    else:
        print("No files moved; no verified-safe candidates.")


if __name__ == "__main__":
    main()
