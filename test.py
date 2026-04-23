from __future__ import annotations

import sys

from inference import get_exercises, get_muscles, load_model, predict_rir, update_strength_anchors


CHECKPOINT_PATH = "deepgain_model_best.pt"
BASE_ANCHORS = {
    "bench_press": 100.0,
    "squat": 140.0,
    "deadlift": 180.0,
}
PROFILE_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]
MONOTONIC_TOL = 0.05
MIN_RELEVANT_RANGE = 0.25

# (exercise, weight, reps, relevant_anchor)
TEST_CASES = [
    ("bench_press", 80.0, 5, "bench_press"),
    ("ohp", 50.0, 5, "bench_press"),
    ("incline_bench", 70.0, 6, "bench_press"),
    ("squat", 120.0, 5, "squat"),
    ("high_bar_squat", 105.0, 6, "squat"),
    ("leg_press", 180.0, 10, "squat"),
    ("deadlift", 140.0, 5, "deadlift"),
    ("sumo_deadlift", 140.0, 5, "deadlift"),
    ("rdl", 110.0, 8, "deadlift"),
]
UPDATE_DEMOS = [
    (
        "bench_press",
        [
            {"exercise": "bench_press", "weight_kg": 90.0, "reps": 5, "rir": 1},
            {"exercise": "incline_bench", "weight_kg": 75.0, "reps": 6, "rir": 1},
            {"exercise": "ohp", "weight_kg": 55.0, "reps": 5, "rir": 1},
        ],
        ("bench_press", 80.0, 5),
    ),
    (
        "squat",
        [
            {"exercise": "squat", "weight_kg": 130.0, "reps": 5, "rir": 1},
            {"exercise": "high_bar_squat", "weight_kg": 120.0, "reps": 5, "rir": 1},
            {"exercise": "leg_press", "weight_kg": 200.0, "reps": 8, "rir": 2},
        ],
        ("squat", 120.0, 5),
    ),
    (
        "deadlift",
        [
            {"exercise": "deadlift", "weight_kg": 160.0, "reps": 5, "rir": 1},
            {"exercise": "sumo_deadlift", "weight_kg": 150.0, "reps": 5, "rir": 1},
            {"exercise": "rdl", "weight_kg": 120.0, "reps": 8, "rir": 2},
        ],
        ("deadlift", 140.0, 5),
    ),
]


def build_state():
    return {muscle: 1.0 for muscle in get_muscles()}


def scaled_anchors(factor: float, only_anchor: str | None = None) -> dict[str, float]:
    anchors = dict(BASE_ANCHORS)
    if only_anchor is None:
        for key in anchors:
            anchors[key] *= factor
        return anchors
    anchors[only_anchor] *= factor
    return anchors


def predict_case(model, exercise: str, weight: float, reps: int, anchors: dict[str, float]) -> float:
    return predict_rir(model, build_state(), exercise, weight, reps, strength_anchors=anchors)


def is_monotonic_increasing(values: list[float], tol: float) -> bool:
    return all(next_val >= cur_val - tol for cur_val, next_val in zip(values, values[1:]))


def fmt_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def relevant_sweep(model, exercise: str, weight: float, reps: int, anchor: str) -> tuple[list[float], bool, float]:
    values = [
        predict_case(model, exercise, weight, reps, scaled_anchors(factor, only_anchor=anchor))
        for factor in PROFILE_FACTORS
    ]
    monotonic = is_monotonic_increasing(values, MONOTONIC_TOL)
    value_range = values[-1] - values[0]
    return values, monotonic, value_range


def unrelated_anchor_drifts(model, exercise: str, weight: float, reps: int, relevant_anchor: str) -> dict[str, float]:
    base_rir = predict_case(model, exercise, weight, reps, dict(BASE_ANCHORS))
    drifts = {}
    for anchor in BASE_ANCHORS:
        if anchor == relevant_anchor:
            continue
        values = [
            predict_case(model, exercise, weight, reps, scaled_anchors(factor, only_anchor=anchor))
            for factor in PROFILE_FACTORS
        ]
        drifts[anchor] = max(abs(v - base_rir) for v in values)
    return drifts


def print_profile_contrast(model, exercises: set[str]) -> bool:
    print("== Weak vs Base vs Strong ==")
    ok = True
    for exercise, weight, reps, _ in TEST_CASES:
        if exercise not in exercises:
            continue
        weak = predict_case(model, exercise, weight, reps, scaled_anchors(0.6))
        base = predict_case(model, exercise, weight, reps, scaled_anchors(1.0))
        strong = predict_case(model, exercise, weight, reps, scaled_anchors(1.4))
        spread = strong - weak
        passed = spread >= MIN_RELEVANT_RANGE
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status:4s} {exercise:15s} "
            f"weak={weak:.3f} base={base:.3f} strong={strong:.3f} spread={spread:.3f}"
        )
    print()
    return ok


def print_relevant_anchor_sweeps(model, exercises: set[str]) -> bool:
    print("== Relevant Anchor Sweeps ==")
    ok = True
    for exercise, weight, reps, anchor in TEST_CASES:
        if exercise not in exercises:
            continue
        values, monotonic, value_range = relevant_sweep(model, exercise, weight, reps, anchor)
        passed = monotonic and value_range >= MIN_RELEVANT_RANGE
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status:4s} {exercise:15s} anchor={anchor:11s} "
            f"range={value_range:.3f} values={fmt_values(values)}"
        )
    print()
    return ok


def print_isolation_diagnostics(model, exercises: set[str]) -> None:
    print("== Unrelated Anchor Drift (diagnostic only) ==")
    for exercise, weight, reps, anchor in TEST_CASES:
        if exercise not in exercises:
            continue
        drifts = unrelated_anchor_drifts(model, exercise, weight, reps, anchor)
        joined = ", ".join(f"{name}={drift:.3f}" for name, drift in drifts.items())
        print(f"{exercise:15s} relevant={anchor:11s} {joined}")
    print()


def print_anchor_update_demos(model) -> bool:
    print("== Anchor Update Demos ==")
    ok = True
    for anchor_name, session_sets, probe in UPDATE_DEMOS:
        updated_anchors, details = update_strength_anchors(
            dict(BASE_ANCHORS),
            session_sets,
            return_details=True,
        )
        old_value = BASE_ANCHORS[anchor_name]
        new_value = updated_anchors[anchor_name]
        unchanged_others = all(
            abs(updated_anchors[other] - BASE_ANCHORS[other]) < 1e-9
            for other in BASE_ANCHORS
            if other != anchor_name
        )
        probe_before = predict_case(model, probe[0], probe[1], probe[2], dict(BASE_ANCHORS))
        probe_after = predict_case(model, probe[0], probe[1], probe[2], updated_anchors)
        detail = details[anchor_name]
        passed = (
            detail["updated"]
            and new_value > old_value
            and unchanged_others
            and probe_after >= probe_before - 1e-6
        )
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status:4s} {anchor_name:11s} "
            f"old={old_value:.2f} new={new_value:.2f} "
            f"probe_before={probe_before:.3f} probe_after={probe_after:.3f} "
            f"selected={detail['selected_exercises']}"
        )

    low_quality_session = [
        {"exercise": "bench_press", "weight_kg": 40.0, "reps": 12, "rir": 4},
        {"exercise": "ohp", "weight_kg": 20.0, "reps": 12, "rir": 4},
    ]
    no_update_anchors, no_update_details = update_strength_anchors(
        dict(BASE_ANCHORS),
        low_quality_session,
        return_details=True,
    )
    no_update_pass = all(abs(no_update_anchors[key] - BASE_ANCHORS[key]) < 1e-9 for key in BASE_ANCHORS)
    ok = ok and no_update_pass
    status = "PASS" if no_update_pass else "FAIL"
    counts = {key: value["n_candidates"] for key, value in no_update_details.items()}
    print(f"{status:4s} low_quality_session anchors_unchanged={no_update_pass} candidates={counts}")
    print()
    return ok


def main() -> int:
    model = load_model(CHECKPOINT_PATH)
    exercises = set(get_exercises())

    print(f"checkpoint = {CHECKPOINT_PATH}")
    print(f"strength_feature_dim = {model.strength_feature_dim}")
    print()

    if model.strength_feature_dim <= 0:
        print("FAIL model was loaded without strength features; retraining is required.")
        return 1

    contrast_ok = print_profile_contrast(model, exercises)
    sweep_ok = print_relevant_anchor_sweeps(model, exercises)
    print_isolation_diagnostics(model, exercises)
    update_ok = print_anchor_update_demos(model)

    if contrast_ok and sweep_ok and update_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
