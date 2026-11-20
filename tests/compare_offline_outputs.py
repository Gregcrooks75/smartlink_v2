"""Compare two fresh synthetic runs, including final-review exports."""

import json
import re
from pathlib import Path

import pandas as pd


# Run directories, status log names and batch names carry wall-clock times.
def inventory(root):
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = re.sub(
            r"(?<!\d)\d{6}_\d{6}(?!\d)", "<time>", path.relative_to(root).as_posix()
        )
        assert name not in files, f"Expected one cold run, found duplicate {name}"
        files[name] = path
    return files


def table(frame):
    frame = frame.copy()
    for col in ("run_time", "run_id", "changed_at", "updated_at"):
        if col in frame:
            # Preserve the column, type and null pattern. Its value identifies this run.
            frame.loc[frame[col].notna(), col] = "<run>"
    if "output_folder" in frame:

        def review_folder(value):
            path = Path(value)
            assert path.is_dir() and re.fullmatch(r"\d{6}_\d{6}", path.name)
            return path.parent.relative_to(path.parents[3]).as_posix()

        frame["output_folder"] = frame["output_folder"].map(review_folder)
    # Parallel aggregations and audit exports may emit rows in different orders.
    return frame.sort_values(list(frame.columns), ignore_index=True)


def compare_tables(left, right):
    pd.testing.assert_frame_equal(table(left), table(right), check_exact=True)


def metadata(path, root):
    # Only environment-dependent fields are normalized. The model bytes,
    # eligibility, settings, feature schema, score-gap reports and decisions
    # are compared independently.
    # Replace the JSON-escaped root so Windows paths normalize as well.
    project_prefix = json.dumps(str(root.parent))[1:-1]
    value = json.loads(
        path.read_text(encoding="utf-8").replace(project_prefix, "<project>")
    )
    if path.name == "_inputs.json":
        value.pop("metadata_sha256")  # metadata contains training time and location
    if path.name == "_complete.json" and "transformer_ranker" in path.parts:
        value.pop("signature")
    if path.name == "_complete.json" and "final_features" in path.parts:
        for entry in value["divisions"].values():
            entry.pop("input_signature")  # includes implementation hash
    for key in ("created_at", "trained_at", "snapshot_id", "training_signature"):
        if isinstance(value, dict):
            value.pop(key, None)
    return value


def compare_runs(original, candidate):
    original, candidate = Path(original).resolve(), Path(candidate).resolve()
    left, right = inventory(original), inventory(candidate)
    # Main stops in _map, after writing the workbooks and before these files.
    post_map = {"review_map.html", "scoring_summary.parquet", "_complete.json"}
    extra = {name for name in right if name not in left}
    final_extra = {name for name in extra if name.startswith("processed/final_review/")}
    assert {right[name].name for name in final_extra} == {
        "CO_Review.xlsx", "daily_summary.csv", "historical_events.parquet", "model_performance.csv",
    }
    training_extra = {name for name in extra if name.startswith("processed/training/history/")}
    assert len(training_extra) == 1 and right[next(iter(training_extra))].name == "model_reference.json"
    extra -= final_extra | training_extra
    assert len(extra) == 3 and {right[name].name for name in extra} == post_map
    assert all(name.startswith("processed/recommendations/") for name in extra)
    assert not left.keys() - right.keys()
    for name, first in sorted(left.items()):
        # Adam reviews intentionally change the final recommendation and
        # downstream final-review exports. Preserve strict parity for upstream
        # data, features, training, model and stored prediction checkpoints.
        if name.startswith("processed/final_review/"):
            continue
        if (
            name.startswith("processed/recommendations/")
            and "/predictions/" not in name
        ):
            continue
        second = right[name]
        try:
            if first.suffix == ".parquet":
                compare_tables(pd.read_parquet(first), pd.read_parquet(second))
            elif first.suffix == ".csv":
                # Compare text fields, preserving leading zeroes and empty strings.
                try:
                    a, b = (
                        pd.read_csv(p, dtype=str, keep_default_na=False)
                        for p in (first, second)
                    )
                except pd.errors.EmptyDataError:
                    assert first.read_bytes() == second.read_bytes()
                else:
                    compare_tables(a, b)
            elif first.suffix == ".xlsx":
                a, b = (
                    pd.read_excel(p, sheet_name=None, dtype=object)
                    for p in (first, second)
                )
                assert list(a) == list(b)
                for sheet in a:
                    compare_tables(a[sheet], b[sheet])
            elif first.suffix == ".json":
                assert metadata(first, original) == metadata(second, candidate)
            else:
                assert first.read_bytes() == second.read_bytes()
        except Exception as err:
            raise AssertionError(f"Regression in {name}: {err}") from err
    return len(left)
