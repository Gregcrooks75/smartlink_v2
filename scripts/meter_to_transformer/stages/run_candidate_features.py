"""Compatibility entry point for mtx.features.regular."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.features import regular as implementation

sys.modules[__name__] = implementation
