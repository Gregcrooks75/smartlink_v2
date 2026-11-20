"""Inspect an immutable training run: python -m mtx.training.history_selection --help."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from mtx.config import PROJECT_ROOT


def inspect_run(folder):
    config = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    features = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
    seen = set()
    splits = {}
    for split in ("train", "validation", "test"):
        path = folder / f"{split}.parquet"
        schema = pq.read_schema(path)
        if set(features) - set(schema.names):
            raise ValueError(f"Missing model features in {path}")
        meters = set(
            pq.read_table(path, columns=["meter_number"]).column(0).to_pylist()
        )
        if not meters or None in meters or meters & seen:
            raise ValueError(f"Empty split, missing ID or meter leakage in {path}")
        seen.update(meters)
        splits[split] = {"path": str(path), "meters": len(meters)}
    reference = folder / "model_reference.json"
    model = (
        json.loads(reference.read_text(encoding="utf-8"))
        if reference.exists()
        else None
    )
    return {"run": str(folder), "config": config, "model": model, "splits": splits}


def select_run(history, run="latest"):
    history = Path(history)
    if run != "latest":
        selected = (history / run).resolve()
        if selected.parent.parent != history.resolve():
            raise ValueError("Select a run by date/time under the history directory")
        return inspect_run(selected)
    for folder in sorted(history.glob("????-??-??/*"), reverse=True):
        try:
            return inspect_run(folder)
        except (OSError, KeyError, ValueError):
            continue
    raise FileNotFoundError(f"No complete, disjoint training run in {history}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", type=Path, default=PROJECT_ROOT / "data/processed/training/history"
    )
    parser.add_argument(
        "--run", default="latest", help="latest or YYYY-MM-DD/HHMMSS_microseconds"
    )
    args = parser.parse_args()
    print(json.dumps(select_run(args.history, args.run), indent=2))
