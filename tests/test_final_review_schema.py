import pandas as pd
import pytest

from mtx.prediction.final_review import TRANSFORMER_COLUMNS, read_columns


def test_raw_and_historical_gis_division_columns(tmp_path):
    path = tmp_path / "transformers.parquet"
    frame = pd.DataFrame({"FuncLoc": ["TX"], "SAPID": ["S"], "OPCO": ["NYSEG"]})

    def read(frame):
        frame.to_parquet(path)
        return read_columns(path, TRANSFORMER_COLUMNS, fallback_columns={"Division": "DivName"})

    assert read(frame.assign(DivName="Ithaca")).division.tolist() == ["Ithaca"]
    assert read(frame.assign(Division="Geneva", DivName="Ithaca")).division.tolist() == ["Geneva"]
    with pytest.raises(ValueError, match="missing"):
        read(frame)
