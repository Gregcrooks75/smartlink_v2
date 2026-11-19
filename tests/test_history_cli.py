import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_history_cli_reads_snapshot_and_preserves_ids(tmp_path, extension):
    reference = tmp_path / "data/reference/Historical Corrections"
    reference.mkdir(parents=True)
    pd.DataFrame({"meter_number": ["001"], "current_floc": ["A"],
                  "final_recommended_floc": ["B"]}).to_csv(reference / "All_Batches_Sent.csv", index=False)
    ami = pd.DataFrame({"MeterID": ["001"], "Area": ["Ithaca"], "FLOC": ["B"]})
    snapshot = tmp_path / f"ami.{extension}"
    if extension == "csv":
        ami.to_csv(snapshot, index=False)
    else:
        ami.to_parquet(snapshot)
    script = Path(__file__).resolve().parents[1] / "scripts/meter_to_transformer/stages/run_historical_corrections.py"
    subprocess.run([sys.executable, script, "--ami-path", snapshot, "--meter-col", "MeterID",
                    "--division-col", "Area", "--floc-col", "FLOC"], cwd=tmp_path,
                   env={**os.environ, "SMARTLINK_PROJECT_ROOT": str(tmp_path)}, check=True)
    result = pd.read_parquet(next(tmp_path.glob("data/cache/historical_corrections/*/all_historical_corrections.parquet")))
    assert list(zip(result.meter_number, result.latest_floc)) == [("001", "B")]
