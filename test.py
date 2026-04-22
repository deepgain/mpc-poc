from __future__ import annotations

import sys

from inference import get_exercises, get_muscles, load_model, predict_rir


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

    if contrast_ok and sweep_ok:
        print("OVERALL: PASS")
        return 0

    print("OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
