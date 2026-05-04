"""Dump EXERCISE_STRENGTH_PRIORS + constants as a JSON asset for the Dart port.

The table itself is a hand-authored dict in strength_priors.py; we serialize
it instead of porting the literal so any edit to the Python source flows
through with one regen.

Output: ../../flutter_app/assets/model/strength_priors.json
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from strength_priors import (  # noqa: E402
    ANCHOR_ALIASES,
    ANCHOR_COLUMNS,
    ANCHOR_NAMES,
    DEFAULT_ANCHOR_VALUES_KG,
    DEFAULT_SESSION_GAP_HOURS,
    DEFAULT_UPDATE_ALPHA,
    DEFAULT_UPDATE_MAX_RELATIVE_CHANGE,
    DEFAULT_UPDATE_MAX_REPS,
    DEFAULT_UPDATE_MAX_RIR,
    DEFAULT_UPDATE_MIN_RELATIVE_LOAD,
    DEFAULT_UPDATE_TOP_K,
    EXERCISE_STRENGTH_PRIORS,
)


def main():
    out_path = REPO_ROOT / "flutter_app" / "assets" / "model" / "strength_priors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "anchor_names": list(ANCHOR_NAMES),
        "anchor_columns": dict(ANCHOR_COLUMNS),
        "anchor_aliases": {k: list(v) for k, v in ANCHOR_ALIASES.items()},
        "default_anchor_values_kg": dict(DEFAULT_ANCHOR_VALUES_KG),
        "exercise_priors": EXERCISE_STRENGTH_PRIORS,
        "constants": {
            "DEFAULT_UPDATE_ALPHA": DEFAULT_UPDATE_ALPHA,
            "DEFAULT_UPDATE_MAX_RELATIVE_CHANGE": DEFAULT_UPDATE_MAX_RELATIVE_CHANGE,
            "DEFAULT_UPDATE_MIN_RELATIVE_LOAD": DEFAULT_UPDATE_MIN_RELATIVE_LOAD,
            "DEFAULT_UPDATE_MAX_REPS": DEFAULT_UPDATE_MAX_REPS,
            "DEFAULT_UPDATE_MAX_RIR": DEFAULT_UPDATE_MAX_RIR,
            "DEFAULT_UPDATE_TOP_K": DEFAULT_UPDATE_TOP_K,
            "DEFAULT_SESSION_GAP_HOURS": DEFAULT_SESSION_GAP_HOURS,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"  {len(EXERCISE_STRENGTH_PRIORS)} exercise priors")
    print(f"  anchors: {ANCHOR_NAMES}")


if __name__ == "__main__":
    main()
