"""Preview and optionally stage removal of reviewed SmartLink files.

Run python scripts/cleanup_pipeline.py --help from a Git checkout.
Git history retains committed versions. No commit or push is performed.
"""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
# Pin each reviewed Git blob so a later change requires a fresh review.
GROUPS = {
    "meter-to-meter": {
        "scripts/meter_to_meter/__init__.py":
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "scripts/meter_to_meter/stages/__init__.py":
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "scripts/meter_to_meter/stages/run_meter_correlations.py":
            "74a7d53aab4392c17aada959d06edc8e03478b82",
        "scripts/meter_to_meter/stages/run_meter_pair_features.py":
            "f38ac2c30e45a8399b5893dd9bf72dc9da416133",
        "scripts/meter_to_meter/stages/run_meter_to_meter_pipeline.py":
            "eb3eac7d668ff4d75fa5e4bc710a93bc57c9f318",
        "scripts/meter_to_meter/stages/run_tile_features.py":
            "edf4e50d99121acceeebc0aefebc637950472205",
        "scripts/meter_to_meter/stages/run_weekly_voltages.py":
            "b5096ea559f88cf3c6665a2c735d0b0ee2159994",
        "scripts/meter_to_meter/validation/__init__.py":
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "scripts/meter_to_meter/validation/checking_duplicates.ipynb":
            "3c015767f2f99c5e13d5fa136255809e3a0c00de",
        "scripts/meter_to_meter/validation/deduplicate_meter_pairs.py":
            "9f2327663a96abb03835676e34a89f5686411699",
        "scripts/meter_to_meter/validation/validate_meter_pair_features.py":
            "5ec7dc204ce67097fc051aeae55e690cd8c6271c",
        "scripts/meter_to_meter/validation/validate_meter_pairs.py":
            "a5dec6dff12d779948b2d70a263e01a776346c08",
    },
    "legacy-features": {
        "scripts/meter_to_transformer/stages/run_final_features.py":
            "fdebb55c93c0012a6bbaebacefd70e981d8e37fd",
        "scripts/meter_to_transformer/stages/run_tile_features.py":
            "eaff651799b2f5eff3f19c6fb790f00c34951dce",
        "scripts/meter_to_transformer/stages/run_transformer_candidate_features.py":
            "01ea5006707d0ce4f84ecd5f5ba5d43e9e4a39e9",
        "scripts/meter_to_transformer/stages/run_transformer_tile_features.py":
            "c29731f44d66d2f4c16d2638a1b7274d5474a5e0",
    },
    "duplicate": {
        "scripts/meter_to_transformer/training/build_ranker_dataset.py":
            "15067a8b085f1fcf9880579f613e2d84beb6359c",
    },
    "notebooks": {
        "scripts/rejection history.ipynb":
            "225a8f2d5603541f2596378be1d5a2421ea44b82",
        "src/mtx/data/Untitled-1.ipynb":
            "b15110e7f0e5a16d370eedffe00d0c9755096243",
    },
}


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--include", action="append", choices=sorted(GROUPS.keys() - {"duplicate"}),
        default=[], help="Also remove this optional workflow. Repeat for more groups.",
    )
    selection.add_argument("--all", action="store_true", help="Select all reviewed groups.")
    parser.add_argument("--apply", action="store_true", help="Stage the selected deletions.")
    args = parser.parse_args()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise SystemExit("Place this script in the scripts directory of the SmartLink checkout.")

    groups = list(GROUPS) if args.all else ["duplicate", *dict.fromkeys(args.include)]
    selected = []
    for group in groups:
        for path, expected in GROUPS[group].items():
            entry = git("ls-files", "--stage", "--", path)
            if not entry:
                continue
            if entry.split()[:3] != ["100644", expected, "0"]:
                raise SystemExit(f"Index differs from the reviewed version: {path}")
            if git("status", "--porcelain", "--", path):
                raise SystemExit(f"Local changes found: {path}")
            selected.append(path)
            print(f"[{group}] {path}")

    if not selected:
        print("No selected files remain tracked.")
        return
    if args.apply:
        git("rm", "--", *selected)
        print(f"Staged {len(selected)} deletions. Review with git diff --cached --stat.")
    else:
        print(f"Preview: {len(selected)} files. Repeat with --apply to stage deletions.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr.strip() or str(error)) from error

