"""Compatibility entry point for mtx.candidates.spatial."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.candidates import spatial as implementation

sys.modules[__name__] = implementation
