import pandas as pd
import pytest

from mtx.prediction.final_review import TRANSFORMER_COLUMNS, read_columns


def test_raw_and_historical_gis_division_columns(tmp_path):
    path = tmp_path / "transformers.parquet"
    frame = pd.DataFrame({"FuncLoc": ["TX"], "SAPID": ["S"], "OPCO": ["NYSEG"]})

    def read(frame):
        frame.to_parquet(path)
        return read_columns(
            path, TRANSFORMER_COLUMNS, fallback_columns={"Division": "DivName"}
        )

    assert read(frame.assign(DivName="Ithaca")).division.tolist() == ["Ithaca"]
    assert read(
        frame.assign(Division="Geneva", DivName="Ithaca")
    ).division.tolist() == ["Geneva"]
    with pytest.raises(ValueError, match="missing"):
        read(frame)


@pytest.mark.parametrize(
    ("distances", "expected"),
    [
        ([550.0], "PASS"),
        ([550.01], "DISTANCE_EXCEEDS_LIMIT"),
        ([], "PAIR_NOT_IN_CANDIDATE_FILE"),
        ([10.0, 20.0], "CONFLICTING_CANDIDATE_DISTANCES"),
    ],
)
def test_target_distance_holds_whole_subgroup(tmp_path, distances, expected):
    from mtx.prediction.final_review import check_distances

    work = pd.DataFrame(
        {
            "meter_number": ["01", "02"],
            "division": "ITHACA",
            "unit": "G",
            "current_transformer": "OLD",
            "model_recommended_transformer": "NEW",
            "proposed_final_transformer": "NEW",
            "review_reason": "",
        }
    )
    pairs = [("01", "NEW", 15.0)] + [("02", "NEW", distance) for distance in distances]
    pd.DataFrame(
        pairs, columns=["meter_number", "candidate_transformer", "distance_ft"]
    ).to_parquet(tmp_path / "Ithaca.parquet", index=False)
    result, _ = check_distances(work, tmp_path)
    assert list(result.proposed_distance_status) == ["PASS", expected]
    reason = "" if expected == "PASS" else "CANDIDATE_CHECK"
    assert result.review_reason.tolist() == [reason, reason]
