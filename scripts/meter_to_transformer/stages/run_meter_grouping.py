"""Compatibility entry point for mtx.grouping.daily."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.grouping import daily as implementation

sys.modules[__name__] = implementation
