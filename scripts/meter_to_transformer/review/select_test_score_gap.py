"""Compatibility entry point for mtx.training.test_threshold."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.training import test_threshold as implementation

if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
