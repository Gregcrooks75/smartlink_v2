import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modules_and_compatibility_paths_import_without_database():
    # A clean process detects dependencies hidden by pytest's import order.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import importlib
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path("src").resolve()))
with patch("pyodbc.connect", side_effect=AssertionError("database opened during import")):
    import scripts.run_meter_to_transformer_pipeline
    assert not any(name.startswith("mtx.legacy") for name in sys.modules)
    for root in ("src/mtx", "scripts"):
        for path in sorted(Path(root).rglob("*.py")):
            relative = path.relative_to("src") if root == "src/mtx" else path
            name = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
            module = importlib.import_module(name)
            if name.startswith("scripts.meter_to_transformer.") and path.name != "__init__.py":
                assert module.__name__.startswith("mtx."), name
""",
        ],
        cwd=ROOT,
        check=True,
    )
