"""Compatibility entry point for mtx.data.coordinates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.data import coordinates as implementation

sys.modules[__name__] = implementation
