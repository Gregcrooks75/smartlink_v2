"""Safely reorganize SmartLink's scripts directory.

Run from the repository root:
    python scripts/organize_repository.py           # preview only
    python scripts/organize_repository.py --apply   # move + rewrite imports

Requires Git and a clean working tree. Does not delete code or touch outputs.
Review the diff and run pipeline smoke tests before committing.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOVES = {
    "scripts/run_correction_review.py": "scripts/meter_to_transformer/review/run_correction_review.py",
    "scripts/run_group_review.py": "scripts/meter_to_transformer/review/run_group_review.py",
    "scripts/select_test_score_gap.py": "scripts/meter_to_transformer/review/select_test_score_gap.py",
    "scripts/cleanup_pipeline.py": "scripts/maintenance/cleanup_pipeline.py",
    "scripts/refactor_m2t_repository.py": "scripts/maintenance/refactor_m2t_repository.py",
    "scripts/remove_unused_m2t_files.py": "scripts/maintenance/remove_unused_m2t_files.py",
}
MODULE_RENAMES = {
    "run_correction_review": "meter_to_transformer.review.run_correction_review",
    "run_group_review": "meter_to_transformer.review.run_group_review",
    "select_test_score_gap": "meter_to_transformer.review.select_test_score_gap",
}
INIT_FILES = [
    "scripts/meter_to_transformer/review/__init__.py",
    "scripts/maintenance/__init__.py",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rewritten(source: str) -> str:
    """Rewrite absolute imports only; avoid editing comments and string literals."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in MODULE_RENAMES:
            old = node.module
            line = lines[node.lineno - 1]
            # Replace the module token in the 'from X import' prefix only.
            import re
            pattern = r"(\\bfrom\\s+)" + re.escape(old) + r"(\\s+import\\b)"
            new_line, count = re.subn(pattern, lambda m: m.group(1) + MODULE_RENAMES[old] + m.group(2), line, count=1)
            if count != 1:
                raise RuntimeError(f"Cannot safely rewrite import on line {node.lineno}: {line.strip()}")
            edits.append((node.lineno - 1, new_line))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in MODULE_RENAMES:
                    raise RuntimeError(f"Unsupported plain import of {alias.name}; update it manually.")
    for index, new_line in edits:
        lines[index] = new_line
    result = "".join(lines)
    compile(result, "<reorganized>", "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform changes (default: preview)")
    args = parser.parse_args()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise SystemExit("Run inside the SmartLink Git repository.")
    if git("status", "--porcelain"):
        raise SystemExit("Working tree is not clean. Commit or stash changes before running.")

    missing = [src for src in MOVES if not (ROOT / src).is_file()]
    conflicts = [dst for dst in MOVES.values() if (ROOT / dst).exists()]
    conflicts += [p for p in INIT_FILES if (ROOT / p).exists()]
    if missing or conflicts:
        raise SystemExit(f"Cannot proceed. Missing sources: {missing}; destination conflicts: {conflicts}")

    # Check every tracked Python source, including imports in the moved files.
    tracked = [p for p in git("ls-files", "*.py").splitlines() if (ROOT / p).is_file()]
    changes = {}
    for path in tracked:
        source = (ROOT / path).read_text(encoding="utf-8-sig")
        new_source = rewritten(source)
        if new_source != source:
            changes[path] = new_source

    print("Moves:")
    for src, dst in MOVES.items():
        print(f"  {src} -> {dst}")
    print(f"Python files with import rewrites: {len(changes)}")
    for path in changes:
        print(f"  {path}")
    print("Preserved: main pipeline entrypoint, stages, training, prediction, src/mtx, validation, history, outputs.")
    if not args.apply:
        print("\\nPreview only. Run again with --apply to make these changes.")
        return

    for src, dst in MOVES.items():
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True)

    for src, new_source in changes.items():
        target = ROOT / MOVES.get(src, src)
        target.write_text(new_source, encoding="utf-8")

    for path in INIT_FILES:
        (ROOT / path).write_text('"""SmartLink package."""\\n', encoding="utf-8")

    print("\\nDone. No files were deleted. Next:")
    print("  git add scripts")
    print("  git diff --cached --stat")
    print("  git diff --cached --check")
    print("  python -m compileall -q scripts")
    print("  # Run your normal SmartLink smoke tests before committing.")
    print("  git commit -m \"Organize SmartLink scripts and update imports\"")


if __name__ == "__main__":
    main()
