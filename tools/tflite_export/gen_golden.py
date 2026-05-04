"""Generate golden fixture JSON for the Dart parity test.

Picks a handful of deterministic scenarios (fresh user, single set,
multi-set replay, fully exhausted) and dumps inputs + expected MPC/RIR
outputs computed via inference.predict_mpc / predict_rir.

Output: ../../flutter_app/assets/golden/golden.json
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from inference import EXERCISE_TO_IDX, load_model, predict_mpc, predict_rir  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "flutter_app" / "assets" / "golden" / "golden.json"


def scenario_fresh_user():
    return {
        "name": "fresh_user_no_history",
        "history": [],
        "query_ts": "2026-05-01T10:00:00",
        "anchors_kg": [100.0, 140.0, 180.0],
        "rir_checks": [
            {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5},
            {"exercise": "squat", "weight_kg": 100.0, "reps": 5},
        ],
    }


def scenario_single_set():
    return {
        "name": "single_set_then_24h_recovery",
        "history": [
            {"exercise": "bench_press", "weight_kg": 100.0, "reps": 5,
             "rir": 1, "timestamp": "2026-05-01T10:00:00"},
        ],
        "query_ts": "2026-05-02T10:00:00",
        "anchors_kg": [100.0, 140.0, 180.0],
        "rir_checks": [
            {"exercise": "bench_press", "weight_kg": 95.0, "reps": 6},
        ],
    }


def scenario_full_session():
    base = datetime(2026, 5, 1, 9, 0, 0)
    sets = [
        ("bench_press", 100.0, 5, 1, 0),
        ("bench_press", 100.0, 5, 2, 3),
        ("bench_press", 95.0, 5, 2, 6),
        ("ohp", 60.0, 6, 2, 12),
        ("ohp", 60.0, 6, 3, 15),
        ("close_grip_bench", 80.0, 8, 2, 22),
    ]
    return {
        "name": "push_session_6_sets",
        "history": [
            {
                "exercise": ex, "weight_kg": w, "reps": r, "rir": rir,
                "timestamp": (base + timedelta(minutes=offset)).isoformat(),
            }
            for ex, w, r, rir, offset in sets
        ],
        "query_ts": (base + timedelta(minutes=30)).isoformat(),
        "anchors_kg": [100.0, 140.0, 180.0],
        "rir_checks": [
            {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5},
            {"exercise": "squat", "weight_kg": 120.0, "reps": 5},
            {"exercise": "lat_pulldown", "weight_kg": 70.0, "reps": 8},
        ],
    }


def scenario_full_session_then_recovery():
    base = datetime(2026, 5, 1, 9, 0, 0)
    sets = [
        ("squat", 120.0, 5, 1, 0),
        ("squat", 120.0, 5, 2, 4),
        ("rdl", 100.0, 6, 2, 10),
        ("leg_extension", 60.0, 12, 2, 18),
    ]
    return {
        "name": "leg_session_then_72h_recovery",
        "history": [
            {
                "exercise": ex, "weight_kg": w, "reps": r, "rir": rir,
                "timestamp": (base + timedelta(minutes=offset)).isoformat(),
            }
            for ex, w, r, rir, offset in sets
        ],
        "query_ts": (base + timedelta(hours=72)).isoformat(),
        "anchors_kg": [100.0, 140.0, 200.0],
        "rir_checks": [
            {"exercise": "squat", "weight_kg": 100.0, "reps": 5},
        ],
    }


def scenario_unknown_exercise_filtered():
    return {
        "name": "unknown_exercise_in_history_silently_skipped",
        "history": [
            {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5,
             "rir": 2, "timestamp": "2026-05-01T10:00:00"},
            {"exercise": "made_up_movement", "weight_kg": 80.0, "reps": 5,
             "rir": 2, "timestamp": "2026-05-01T10:05:00"},
            {"exercise": "ohp", "weight_kg": 50.0, "reps": 6,
             "rir": 3, "timestamp": "2026-05-01T10:15:00"},
        ],
        "query_ts": "2026-05-01T11:00:00",
        "anchors_kg": [100.0, 140.0, 180.0],
        "rir_checks": [],
    }


def main():
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = load_model("deepgain_model_muscle_ord.pt", device="cpu")
    scenarios = [
        scenario_fresh_user(),
        scenario_single_set(),
        scenario_full_session(),
        scenario_full_session_then_recovery(),
        scenario_unknown_exercise_filtered(),
    ]

    fixtures = []
    for sc in scenarios:
        import numpy as np
        anchors = np.array(sc["anchors_kg"], dtype=np.float32)
        mpc = predict_mpc(model, sc["history"], sc["query_ts"], strength_anchors=anchors)
        rir_results = []
        for r in sc["rir_checks"]:
            rir_val = predict_rir(
                model, mpc, r["exercise"], r["weight_kg"], r["reps"],
                strength_anchors=anchors,
            )
            rir_results.append({**r, "expected_rir": rir_val})

        fixtures.append({
            "name": sc["name"],
            "history": sc["history"],
            "query_ts": sc["query_ts"],
            "anchors_kg": sc["anchors_kg"],
            "expected_mpc": mpc,
            "rir_checks": rir_results,
        })

    GOLDEN_PATH.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {len(fixtures)} scenarios to {GOLDEN_PATH}")
    for f in fixtures:
        n_history = len(f["history"])
        n_rir = len(f["rir_checks"])
        print(f"  {f['name']:50s}  history={n_history}  rir_checks={n_rir}")


if __name__ == "__main__":
    main()
