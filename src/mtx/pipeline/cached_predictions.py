"""Resolve existing, validated artifacts for a prediction-only SmartLink run.

This module never generates features, trains a model, or changes source caches.
The prediction stage itself performs row-level snapshot/feature compatibility checks.
"""

from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path

from mtx.utils.daily_cache import file_digest


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required prediction artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_cached_prediction_artifacts(
    project_root: Path, *, model_date: str | None = None, run_date: date | None = None
) -> tuple[Path, dict, dict[str, Path]]:
    """Return an existing model, its test-selected cutoff, and today's features.

    Do not fall back to an older feature snapshot: the scoring stage explicitly
    requires today's features and checks assignments and coordinates against AMI.
    """
    root = Path(project_root)
    models_root = root / "data" / "processed" / "models" / "transformer_ranker"
    if model_date is not None:
        date.fromisoformat(model_date)
        model = models_root / model_date
    else:
        versions = (
            sorted(
                (p for p in models_root.iterdir() if p.is_dir() and _is_date(p.name)),
                reverse=True,
            )
            if models_root.is_dir()
            else []
        )
        if not versions:
            raise FileNotFoundError(
                f"No dated trained model under {models_root}. Run model training first."
            )
        model = versions[0]

    for required in ("model.ubj", "metadata.json"):
        if not (model / required).is_file():
            raise FileNotFoundError(f"Missing trained-model artifact: {model / required}")
    metadata = _json(model / "metadata.json")
    if metadata.get("training_date", model.name) != model.name:
        raise ValueError(f"Model training_date does not match folder: {model}")

    cutoff = _json(model / "correction_filter.json")
    if cutoff.get("threshold_selected_on") != "test":
        raise ValueError(
            f"Saved cutoff was not selected on the test set: {model / 'correction_filter.json'}"
        )
    gap = cutoff.get("min_score_gap")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)):
        raise ValueError(
            f"No valid test score-gap cutoff in {model / 'correction_filter.json'}"
        )
    if not math.isfinite(gap) or gap < 0:
        raise ValueError(f"Invalid test score-gap cutoff: {gap!r}")

    today = (run_date or date.today()).isoformat()
    directory = root / "data" / "cache" / "final_features" / today
    manifest = _json(directory / "_complete.json")
    if manifest.get("date") != today:
        raise ValueError(f"Feature manifest is not for {today}: {directory}")
    records = manifest.get("divisions")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"No completed division features in {directory}")

    features: dict[str, Path] = {}
    for division, record in sorted(records.items()):
        if not isinstance(division, str) or not division or Path(division).name != division:
            raise ValueError(f"Unsafe division in feature manifest: {division!r}")
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError(f"Incomplete feature manifest for {division}")
        path = directory / f"{division}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached features: {path}")
        if file_digest(path) != record["sha256"]:
            raise ValueError(f"Cached features differ from manifest: {path}")
        features[division] = path

    return model, {"min_score_gap": float(gap)}, features


def _is_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False
