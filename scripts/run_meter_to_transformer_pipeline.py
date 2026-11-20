"""Run the SmartLink meter-to-transformer production pipeline."""

import argparse
from datetime import date
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtx.pipeline.cached_predictions import load_cached_prediction_artifacts
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Reuse today's cached features, a trained model and its saved test cutoff.",
    )
    parser.add_argument(
        "--final-review",
        action="store_true",
        help="Run ITRON/CO deterministic final review after recommendations.",
    )
    parser.add_argument(
        "--model-date",
        help="Trained model date (YYYY-MM-DD); predictions-only defaults to latest.",
    )
    args = parser.parse_args(argv)
    if args.model_date and not args.predictions_only:
        parser.error("--model-date requires --predictions-only")

    print("\n" + "=" * 80 + "\nSMARTLINK METER-TO-TRANSFORMER PIPELINE\n" + "=" * 80)
    run_date = date.today()

    if args.predictions_only or args.final_review:
        # Other stages import mtx.config.PROJECT_ROOT for histories and review;
        # refuse a split checkout/data-root configuration rather than mixing data.
        from mtx.config import PROJECT_ROOT as configured_root

        if PROJECT_ROOT.resolve() != configured_root.resolve():
            raise RuntimeError(
                "SMARTLINK_PROJECT_ROOT must be set to the same data directory "
                "for the pipeline and its imported stages."
            )

    if args.predictions_only:
        model, summary, features = load_cached_prediction_artifacts(
            PROJECT_ROOT, model_date=args.model_date, run_date=run_date
        )
        # Read existing snapshots and reuse coordinate filtering, but never
        # refresh sources, reconcile history or write rejection logs.
        inputs = run_preparation(
            run_ingestion(CONFIG, PROJECT_ROOT, cache_only=True),
            track_history=False,
        )
    else:
        inputs = run_preparation(run_ingestion(CONFIG, PROJECT_ROOT))
        features = run_spatial_processing(inputs, CONFIG, PROJECT_ROOT, run_date)
        model, summary = run_modeling(inputs, features, PROJECT_ROOT, run_date)

    run_recommendations(inputs, features, PROJECT_ROOT, model, summary)
    if args.final_review:
        from mtx.prediction.final_review import run_final_review_stage

        run_final_review_stage()


if __name__ == "__main__":
    main()
