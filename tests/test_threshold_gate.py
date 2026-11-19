import json

import pandas as pd
import pytest

from mtx.pipeline.recommendations import run_recommendations
from mtx.training.test_threshold import select_test_score_gap


def test_no_qualifying_test_cutoff_stops_before_scoring(tmp_path):
    pd.DataFrame(
        {
            "meter_number": ["01"],
            "current_transformer": ["A"],
            "recommended_transformer": ["B"],
            "verified_transformer": ["B"],
            "score_gap": [0.2],
            "recommended_distance_ft": [120.0],
        }
    ).to_parquet(tmp_path / "test_predictions.parquet")
    result = select_test_score_gap(tmp_path, min_changes=2)
    assert result is None
    assert (
        json.loads((tmp_path / "correction_filter.json").read_text())["min_score_gap"]
        is None
    )
    with pytest.raises(RuntimeError, match="No test score gap"):
        run_recommendations(None, None, tmp_path, tmp_path, result)
    assert not (tmp_path / "data/processed/recommendations").exists()
