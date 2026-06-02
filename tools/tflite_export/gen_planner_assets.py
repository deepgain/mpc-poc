"""Dump planner constants (MUSCLE_INVOLVEMENT, EXERCISE_META, MAIN_EXERCISES,
DEFAULT_TARGET_ZONES) as JSON for the Dart port.

Output: ../../flutter_app/assets/model/planner_meta.json
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER_DIR = REPO_ROOT / "exercise_selection_algorithm"
sys.path.insert(0, str(PLANNER_DIR))
os.chdir(PLANNER_DIR)

from knapsack_planner import (  # noqa: E402
    DEFAULT_TARGET_ZONES,
    EXERCISE_META,
    MAIN_EXERCISES,
    MUSCLE_INVOLVEMENT,
    _BODYWEIGHT_FALLBACK_KG,
)


def main():
    out = REPO_ROOT / "flutter_app" / "assets" / "model" / "planner_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "muscle_involvement": MUSCLE_INVOLVEMENT,
        "exercise_meta": EXERCISE_META,
        "main_exercises": sorted(MAIN_EXERCISES),
        "default_target_zones": DEFAULT_TARGET_ZONES,
        "bodyweight_fallback_kg": _BODYWEIGHT_FALLBACK_KG,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    print(f"  {len(MUSCLE_INVOLVEMENT)} exercises with involvement")
    print(f"  {len(EXERCISE_META)} exercises with meta")
    print(f"  {len(MAIN_EXERCISES)} main exercises")


if __name__ == "__main__":
    main()
