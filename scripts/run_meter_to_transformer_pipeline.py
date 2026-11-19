"""Run the SmartLink meter-to-transformer production pipeline."""

from datetime import date
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtx.pipeline.ingestion import run_ingestion
from mtx.pipeline.modeling import run_modeling
from mtx.pipeline.preparation import run_preparation
from mtx.pipeline.recommendations import run_recommendations
from mtx.pipeline.spatial import run_spatial_processing
from mtx.run_config import RunConfig


DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\u355394\OneDrive - IBERDROLA S.A"
    r"\Documents - Asset Data Strategy"
    r"\Asset Data Quality\SmartLink"
)

PROJECT_ROOT = Path(
    os.environ.get("SMARTLINK_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
).expanduser()
CONFIG = RunConfig(transformer_division_col="Division")


def main():
    print("\n" + "=" * 80 + "\nSMARTLINK METER-TO-TRANSFORMER PIPELINE\n" + "=" * 80)
    run_date = date.today()
    inputs = run_preparation(run_ingestion(CONFIG, PROJECT_ROOT))
    features = run_spatial_processing(inputs, CONFIG, PROJECT_ROOT, run_date)
    model, summary = run_modeling(inputs, features, PROJECT_ROOT, run_date)
    run_recommendations(inputs, features, PROJECT_ROOT, model, summary)


if __name__ == "__main__":
    main()
