"""Golden fixture generator for the Dart port of strength_priors.py.

Output: ../../flutter_app/assets/golden/strength_golden.json

Covers:
  - coerce_anchor_values (Map / List / null inputs)
  - project_exercise_1rm_kg (with and without anchors)
  - estimate_e1rm_candidate (edge cases: rir=0, large reps)
  - collect_strength_update_candidates (quality filtering)
  - update_strength_anchors (EMA blend, top-K, clip)
  - build_anchor_history_from_completed_sets (multi-session replay)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from strength_priors import (  # noqa: E402
    ANCHOR_NAMES,
    DEFAULT_ANCHOR_VALUES_KG,
    build_anchor_history_from_completed_sets,
    coerce_anchor_values,
    collect_strength_update_candidates,
    estimate_e1rm_candidate,
    project_exercise_1rm_kg,
    update_strength_anchors,
)

GOLDEN = REPO_ROOT / "flutter_app" / "assets" / "golden" / "strength_golden.json"


def _arr(x):
    return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def coerce_cases():
    return [
        {"name": "null → defaults", "input": None,
         "expected": _arr(coerce_anchor_values(None))},
        {"name": "full dict",
         "input": {"bench_press": 110.0, "squat": 150.0, "deadlift": 200.0},
         "expected": _arr(coerce_anchor_values(
             {"bench_press": 110.0, "squat": 150.0, "deadlift": 200.0}))},
        {"name": "partial dict (only bench)",
         "input": {"bench_press": 110.0},
         "expected": _arr(coerce_anchor_values({"bench_press": 110.0}))},
        {"name": "config_1rm_ aliases",
         "input": {"config_1rm_bench_press": 95.0, "config_1rm_squat": 130.0},
         "expected": _arr(coerce_anchor_values(
             {"config_1rm_bench_press": 95.0, "config_1rm_squat": 130.0}))},
        {"name": "list [bench, squat, deadlift]",
         "input": [120.0, 160.0, 220.0],
         "expected": _arr(coerce_anchor_values([120.0, 160.0, 220.0]))},
        {"name": "list with zero/negative entries falls back to defaults",
         "input": [-5.0, 0.0, 220.0],
         "expected": _arr(coerce_anchor_values([-5.0, 0.0, 220.0]))},
        {"name": "unknown alias ignored",
         "input": {"squat": 150.0, "unknown_lift": 999.0},
         "expected": _arr(coerce_anchor_values(
             {"squat": 150.0, "unknown_lift": 999.0}))},
    ]


def project_cases():
    cases = []
    anchors = {"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0}
    for ex in ["bench_press", "squat", "deadlift", "incline_bench", "rdl",
               "leg_press", "ohp", "pull_up", "reverse_fly"]:
        cases.append({
            "exercise": ex, "anchors_kg": anchors,
            "expected": project_exercise_1rm_kg(ex, anchors),
        })
    cases.append({
        "exercise": "made_up_exercise", "anchors_kg": anchors,
        "expected": project_exercise_1rm_kg("made_up_exercise", anchors),
    })
    cases.append({
        "exercise": "plank", "anchors_kg": anchors,  # bodyweight, no anchor
        "expected": project_exercise_1rm_kg("plank", anchors),
    })
    return cases


def e1rm_cases():
    return [
        {"weight_kg": 100.0, "reps": 5, "rir": 2.0,
         "expected": estimate_e1rm_candidate(100.0, 5, 2.0)},
        {"weight_kg": 60.0, "reps": 10, "rir": 0.0,
         "expected": estimate_e1rm_candidate(60.0, 10, 0.0)},
        {"weight_kg": 140.0, "reps": 1, "rir": 0.0,
         "expected": estimate_e1rm_candidate(140.0, 1, 0.0)},
        {"weight_kg": 80.0, "reps": 8, "rir": 3.0,
         "expected": estimate_e1rm_candidate(80.0, 8, 3.0)},
        {"weight_kg": 0.0, "reps": 5, "rir": 2.0,
         "expected": estimate_e1rm_candidate(0.0, 5, 2.0)},  # invalid → None
        {"weight_kg": 100.0, "reps": -1, "rir": 2.0,
         "expected": estimate_e1rm_candidate(100.0, -1, 2.0)},  # invalid → None
    ]


def _sets(specs, base=None):
    base = base or datetime(2026, 5, 1, 9, 0, 0)
    out = []
    for ex, w, r, rir, off in specs:
        out.append({
            "exercise": ex, "weight_kg": w, "reps": r, "rir": rir,
            "timestamp": (base + timedelta(minutes=off)).isoformat(),
        })
    return out


def update_cases():
    anchors = {"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0}

    def run(name, completed):
        new = update_strength_anchors(anchors, completed)
        return {
            "name": name,
            "anchors_kg": anchors,
            "completed_sets": completed,
            "expected_new_anchors": _arr(new),
        }

    return [
        run("no candidates → unchanged", []),
        run("light single set below min_relative_load → unchanged",
            _sets([("bench_press", 30.0, 3, 1, 0)])),
        run("clean PR-quality bench triple",
            _sets([("bench_press", 110.0, 3, 1, 0)])),
        run("multi-set bench session, top-K weighted average",
            _sets([
                ("bench_press", 100.0, 5, 2, 0),
                ("bench_press", 100.0, 5, 2, 3),
                ("bench_press", 95.0, 6, 1, 6),
                ("close_grip_bench", 90.0, 5, 1, 12),
            ])),
        run("squat session updates squat anchor only",
            _sets([
                ("squat", 150.0, 5, 1, 0),
                ("squat", 150.0, 5, 2, 4),
                ("low_bar_squat", 145.0, 5, 1, 10),
            ])),
        run("clip to ±4% — extreme jump capped",
            _sets([
                ("bench_press", 200.0, 1, 0, 0),  # would imply ~210kg 1RM
            ])),
        run("rir > max_rir filtered out",
            _sets([("bench_press", 80.0, 5, 5, 0)])),
        run("reps > max_reps filtered out",
            _sets([("bench_press", 80.0, 15, 1, 0)])),
        run("multi-anchor mixed session",
            _sets([
                ("bench_press", 100.0, 5, 2, 0),
                ("squat", 140.0, 5, 1, 8),
                ("deadlift", 180.0, 3, 0, 18),
            ])),
    ]


def history_cases():
    anchors = {"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0}
    base = datetime(2026, 5, 1, 9, 0, 0)

    def run(name, completed):
        history, final = build_anchor_history_from_completed_sets(anchors, completed)
        return {
            "name": name,
            "anchors_kg": anchors,
            "completed_sets": completed,
            # history[i] = anchors AT set i (before any update from set i)
            "expected_history": [_arr(h) for h in history],
            "expected_final": _arr(final),
        }

    return [
        run("empty history",
            []),
        run("single session, single set",
            _sets([("bench_press", 100.0, 5, 2, 0)])),
        run("single session, 3 sets — anchors stable mid-session",
            _sets([
                ("bench_press", 100.0, 5, 2, 0),
                ("bench_press", 100.0, 5, 2, 3),
                ("bench_press", 95.0, 6, 1, 6),
            ])),
        run("two sessions across 24h gap — anchor advances",
            [
                *_sets([
                    ("bench_press", 100.0, 5, 1, 0),
                    ("bench_press", 100.0, 5, 2, 3),
                ], base=base),
                *_sets([
                    ("bench_press", 102.5, 5, 2, 0),
                    ("bench_press", 102.5, 5, 1, 3),
                ], base=base + timedelta(hours=48)),
            ]),
        run("three sessions, mixed lifts",
            [
                *_sets([("bench_press", 100.0, 5, 1, 0)], base=base),
                *_sets([("squat", 140.0, 5, 1, 0)],
                       base=base + timedelta(hours=24)),
                *_sets([("deadlift", 180.0, 3, 0, 0)],
                       base=base + timedelta(hours=48)),
            ]),
    ]


def main():
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor_names": list(ANCHOR_NAMES),
        "default_anchors_kg": [
            DEFAULT_ANCHOR_VALUES_KG[n] for n in ANCHOR_NAMES
        ],
        "coerce": coerce_cases(),
        "project": project_cases(),
        "e1rm": e1rm_cases(),
        "update": update_cases(),
        "history": history_cases(),
    }
    GOLDEN.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {GOLDEN}")
    for k in ("coerce", "project", "e1rm", "update", "history"):
        print(f"  {k}: {len(payload[k])} cases")


if __name__ == "__main__":
    main()
