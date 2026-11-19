"""Compatibility entry point for mtx.prediction.final_review."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.prediction import final_review as implementation

if __name__ == "__main__":
    implementation.run_final_review_stage()
else:
    sys.modules[__name__] = implementation
