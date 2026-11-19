"""Compatibility entry point for mtx.training.tile_dataset."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mtx.training import tile_dataset as implementation

sys.modules[__name__] = implementation
