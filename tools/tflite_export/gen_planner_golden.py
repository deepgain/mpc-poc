"""Golden fixtures for the Dart KnapsackPlanner port.

Runs the Python KnapsackPlanner across deterministic scenarios using the
SAME model checkpoint (deepgain_model_muscle_ord.pt) that the Dart side
loads via TFLite, so block-by-block parity is achievable.

Output: ../../flutter_app/assets/golden/planner_golden.json

Scenarios mirror a subset of test_knapsack_planner.py:
  - 01: fresh user, 60 min, average anchors
  - 02: post-bench history (2 days ago), 60 min
  - 03: push day no legs, 30 min
  - 05: full-body yesterday, 45 min
  - 10: tired legs, 45 min
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
PLANNER_DIR = REPO_ROOT / "exercise_selection_algorithm"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(PLANNER_DIR))
os.chdir(MODELS_DIR)  # inference.py reads ../dataset/* at import time

from inference import load_model  # noqa: E402
from knapsack_planner import KnapsackPlanner  # noqa: E402

# Patch _select_main_exercise to iterate available mains in alphabetical order
# so the golden is deterministic across Python set-hash variations. The Dart
# port also sorts alphabetically — without this, ties in stimulus_score (e.g.
# fresh user) cause divergent main picks between languages.
_orig_select = KnapsackPlanner._select_main_exercise


def _patched_select_main(self, mpc_state, target_rir, exclusions, time_budget_sec):
    from knapsack_planner import MAIN_EXERCISES, EXERCISE_META, MUSCLE_INVOLVEMENT, ExerciseBlock
    best = None
    best_score = -1.0
    available = sorted((MAIN_EXERCISES & self._known_exercises) - exclusions)
    for ex_id in available:
        meta = EXERCISE_META[ex_id]
        sets = meta["sets"]
        set_sec = meta["set_sec"]
        time_cost = sets * set_sec + (sets - 1) * self.rest_sec
        if time_cost > time_budget_sec:
            continue
        weight_kg, reps, pred_rir = self._tune_weight_and_reps(ex_id, mpc_state, target_rir)
        score = self._compute_stimulus(ex_id, mpc_state)
        involvement = MUSCLE_INVOLVEMENT.get(ex_id, {})
        sorted_m = sorted(involvement.items(), key=lambda kv: kv[1], reverse=True)
        primary = [m for m, r in sorted_m[:2] if r >= 0.40]
        secondary = [m for m, r in sorted_m[2:] if r >= 0.20]
        if score > best_score:
            best_score = score
            best = ExerciseBlock(
                exercise_id=ex_id, weight_kg=weight_kg, reps=reps, sets_count=sets,
                rest_sec=self.rest_sec, predicted_rir=pred_rir, stimulus_score=score,
                time_cost_sec=time_cost, ex_type=meta["type"],
                primary_muscles=primary, secondary_muscles=secondary,
            )
    return best


KnapsackPlanner._select_main_exercise = _patched_select_main

GOLDEN = REPO_ROOT / "flutter_app" / "assets" / "golden" / "planner_golden.json"
CHECKPOINT = MODELS_DIR / "deepgain_model_best.pt"

ANCHORS_AVERAGE = {"bench_press": 80.0, "squat": 120.0, "deadlift": 160.0}
ANCHORS_STRONG = {"bench_press": 140.0, "squat": 200.0, "deadlift": 240.0}
ANCHORS_BEGINNER = {"bench_press": 40.0, "squat": 60.0, "deadlift": 80.0}

NOW = datetime(2026, 4, 28, 10, 0, 0)
TIME_30_MIN = 30 * 60
TIME_45_MIN = 45 * 60
TIME_60_MIN = 60 * 60

EXCLUSIONS_NO_LEGS = [
    "squat", "low_bar_squat", "high_bar_squat",
    "leg_press", "bulgarian_split_squat",
    "leg_curl", "leg_extension", "rdl",
]

HISTORY_BENCH_2D_AGO = [
    {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-26T18:00:00"},
    {"exercise": "incline_bench", "weight_kg": 65.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-26T18:25:00"},
    {"exercise": "close_grip_bench", "weight_kg": 55.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-26T18:50:00"},
]

HISTORY_FULL_BODY_YESTERDAY = [
    {"exercise": "squat", "weight_kg": 100.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:00:00"},
    {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:35:00"},
    {"exercise": "deadlift", "weight_kg": 140.0, "reps": 3, "rir": 1,
     "timestamp": "2026-04-27T18:10:00"},
    {"exercise": "ohp", "weight_kg": 50.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-27T18:40:00"},
]

HISTORY_LEGS_YESTERDAY = [
    {"exercise": "squat", "weight_kg": 100.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:00:00"},
    {"exercise": "leg_press", "weight_kg": 180.0, "reps": 10, "rir": 2,
     "timestamp": "2026-04-27T17:30:00"},
    {"exercise": "bulgarian_split_squat", "weight_kg": 40.0, "reps": 10, "rir": 2,
     "timestamp": "2026-04-27T18:00:00"},
    {"exercise": "leg_curl", "weight_kg": 45.0, "reps": 12, "rir": 2,
     "timestamp": "2026-04-27T18:25:00"},
]


def _block_to_json(b):
    return {
        "exercise_id": b.exercise_id,
        "weight_kg": b.weight_kg,
        "reps": b.reps,
        "sets_count": b.sets_count,
        "rest_sec": b.rest_sec,
        "predicted_rir": b.predicted_rir,
        "stimulus_score": b.stimulus_score,
        "time_cost_sec": b.time_cost_sec,
        "ex_type": b.ex_type,
        "primary_muscles": b.primary_muscles,
        "secondary_muscles": b.secondary_muscles,
    }


def _run(planner, *, name, history, time_budget, target_rir=2, exclusions=None,
         anchors=None):
    plan = planner.plan(
        user_history=history,
        time_budget_sec=time_budget,
        target_rir=target_rir,
        exclusions=exclusions,
        now=NOW,
    )
    return {
        "name": name,
        "anchors_kg": anchors,
        "history": history,
        "time_budget_sec": time_budget,
        "target_rir": target_rir,
        "exclusions": exclusions or [],
        "now": NOW.isoformat(),
        "expected_blocks": [_block_to_json(b) for b in plan.blocks],
        "expected_total_time_sec": plan.total_time_sec,
        "expected_total_stimulus": plan.total_stimulus,
        "expected_mpc_before": plan.mpc_before,
        "expected_mpc_after": plan.mpc_after,
        "expected_violations": plan.constraint_violations,
    }


def main():
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {CHECKPOINT}")
    model = load_model(str(CHECKPOINT))

    def planner_for(anchors):
        return KnapsackPlanner(
            model=model,
            strength_anchors=anchors,
            rest_between_sets_sec=120,
            time_resolution_sec=60,
        )

    fixtures = [
        _run(planner_for(ANCHORS_AVERAGE),
             name="01_fresh_user_60min_average",
             history=[], time_budget=TIME_60_MIN, target_rir=3,
             anchors=ANCHORS_AVERAGE),
        _run(planner_for(ANCHORS_AVERAGE),
             name="02_post_bench_history_60min",
             history=HISTORY_BENCH_2D_AGO, time_budget=TIME_60_MIN, target_rir=2,
             anchors=ANCHORS_AVERAGE),
        _run(planner_for(ANCHORS_AVERAGE),
             name="03_push_day_no_legs_30min",
             history=[], time_budget=TIME_30_MIN, target_rir=2,
             exclusions=EXCLUSIONS_NO_LEGS, anchors=ANCHORS_AVERAGE),
        _run(planner_for(ANCHORS_AVERAGE),
             name="05_full_body_yesterday_45min",
             history=HISTORY_FULL_BODY_YESTERDAY, time_budget=TIME_45_MIN,
             target_rir=2, anchors=ANCHORS_AVERAGE),
        _run(planner_for(ANCHORS_AVERAGE),
             name="10_tired_legs_45min",
             history=HISTORY_LEGS_YESTERDAY, time_budget=TIME_45_MIN,
             target_rir=2, anchors=ANCHORS_AVERAGE),
        _run(planner_for(ANCHORS_STRONG),
             name="04a_strong_user_compound_45min",
             history=[], time_budget=TIME_45_MIN, target_rir=2,
             anchors=ANCHORS_STRONG),
        _run(planner_for(ANCHORS_BEGINNER),
             name="04b_beginner_user_compound_45min",
             history=[], time_budget=TIME_45_MIN, target_rir=2,
             anchors=ANCHORS_BEGINNER),
    ]

    GOLDEN.write_text(json.dumps(fixtures, indent=2, default=str))
    print(f"Wrote {len(fixtures)} scenarios to {GOLDEN}")
    for f in fixtures:
        print(f"  {f['name']:45s} {len(f['expected_blocks'])} blocks "
              f"({f['expected_total_time_sec']}s, "
              f"stimulus={f['expected_total_stimulus']:.3f})")


if __name__ == "__main__":
    main()
