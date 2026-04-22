"""Shared 1RM prior utilities for training and inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


ANCHOR_NAMES = ("bench_press", "squat", "deadlift")
ANCHOR_COLUMNS = {
    "bench_press": "config_1rm_bench_press",
    "squat": "config_1rm_squat",
    "deadlift": "config_1rm_deadlift",
}
ANCHOR_ALIASES = {
    "bench_press": ("bench_press", "bench_1rm", "config_1rm_bench_press"),
    "squat": ("squat", "squat_1rm", "config_1rm_squat"),
    "deadlift": ("deadlift", "deadlift_1rm", "config_1rm_deadlift"),
}
DEFAULT_ANCHOR_VALUES_KG = {
    "bench_press": 100.0,
    "squat": 140.0,
    "deadlift": 180.0,
}


EXERCISE_STRENGTH_PRIORS = {
    "bench_press": {"anchor_lift": "bench_press", "ratio_mean": 1.000, "ratio_sd": 0.000, "exercise_family": "bench_primary"},
    "incline_bench": {"anchor_lift": "bench_press", "ratio_mean": 0.815, "ratio_sd": 0.020, "exercise_family": "bench_variant"},
    "close_grip_bench": {"anchor_lift": "bench_press", "ratio_mean": 0.850, "ratio_sd": 0.017, "exercise_family": "bench_variant"},
    "spoto_press": {"anchor_lift": "bench_press", "ratio_mean": 0.900, "ratio_sd": 0.023, "exercise_family": "bench_variant"},
    "incline_bench_45": {"anchor_lift": "bench_press", "ratio_mean": 0.780, "ratio_sd": 0.023, "exercise_family": "bench_variant"},
    "decline_bench": {"anchor_lift": "bench_press", "ratio_mean": 0.900, "ratio_sd": 0.023, "exercise_family": "bench_variant"},
    "chest_press_machine": {"anchor_lift": "bench_press", "ratio_mean": 0.840, "ratio_sd": 0.035, "exercise_family": "machine_press"},
    "ohp": {"anchor_lift": "bench_press", "ratio_mean": 0.620, "ratio_sd": 0.029, "exercise_family": "vertical_press"},
    "dips": {"anchor_lift": "bench_press", "ratio_mean": 0.635, "ratio_sd": 0.049, "exercise_family": "press_assistance"},
    "dumbbell_flyes": {"anchor_lift": "bench_press", "ratio_mean": 0.290, "ratio_sd": 0.029, "exercise_family": "chest_isolation"},
    "pendlay_row": {"anchor_lift": "bench_press", "ratio_mean": 0.890, "ratio_sd": 0.052, "exercise_family": "upper_pull"},
    "seal_row": {"anchor_lift": "bench_press", "ratio_mean": 0.765, "ratio_sd": 0.049, "exercise_family": "upper_pull"},
    "lat_pulldown": {"anchor_lift": "bench_press", "ratio_mean": 0.680, "ratio_sd": 0.046, "exercise_family": "vertical_pull"},
    "pull_up": {"anchor_lift": "bench_press", "ratio_mean": 0.565, "ratio_sd": 0.049, "exercise_family": "vertical_pull"},
    "skull_crusher": {"anchor_lift": "bench_press", "ratio_mean": 0.295, "ratio_sd": 0.032, "exercise_family": "triceps_isolation"},
    "squat": {"anchor_lift": "squat", "ratio_mean": 1.000, "ratio_sd": 0.000, "exercise_family": "squat_primary"},
    "low_bar_squat": {"anchor_lift": "squat", "ratio_mean": 0.990, "ratio_sd": 0.030, "exercise_family": "squat_variant"},
    "high_bar_squat": {"anchor_lift": "squat", "ratio_mean": 0.960, "ratio_sd": 0.030, "exercise_family": "squat_variant"},
    "leg_press": {"anchor_lift": "squat", "ratio_mean": 1.450, "ratio_sd": 0.087, "exercise_family": "machine_lower_compound"},
    "bulgarian_split_squat": {"anchor_lift": "squat", "ratio_mean": 0.400, "ratio_sd": 0.035, "exercise_family": "unilateral_lower"},
    "leg_curl": {"anchor_lift": "squat", "ratio_mean": 0.340, "ratio_sd": 0.035, "exercise_family": "hamstring_isolation"},
    "leg_extension": {"anchor_lift": "squat", "ratio_mean": 0.420, "ratio_sd": 0.046, "exercise_family": "quad_isolation"},
    "deadlift": {"anchor_lift": "deadlift", "ratio_mean": 1.000, "ratio_sd": 0.000, "exercise_family": "hinge_primary"},
    "sumo_deadlift": {"anchor_lift": "deadlift", "ratio_mean": 0.975, "ratio_sd": 0.050, "exercise_family": "hinge_variant"},
    "rdl": {"anchor_lift": "deadlift", "ratio_mean": 0.700, "ratio_sd": 0.029, "exercise_family": "hinge_variant"},
    "reverse_fly": {"anchor_lift": "bodyweight", "ratio_mean": 0.120, "ratio_sd": 0.023, "exercise_family": "rear_delt_isolation"},
    "plank": {"anchor_lift": "bodyweight", "ratio_mean": 0.340, "ratio_sd": 0.035, "exercise_family": "core_bracing"},
    "farmers_walk": {"anchor_lift": "bodyweight", "ratio_mean": 0.750, "ratio_sd": 0.115, "exercise_family": "carry"},
    "leg_raises": {"anchor_lift": "bodyweight", "ratio_mean": 0.290, "ratio_sd": 0.029, "exercise_family": "core_flexion"},
    "ab_wheel": {"anchor_lift": "bodyweight", "ratio_mean": 0.375, "ratio_sd": 0.043, "exercise_family": "core_anti_extension"},
    "dead_bug": {"anchor_lift": "bodyweight", "ratio_mean": 0.250, "ratio_sd": 0.029, "exercise_family": "core_stability"},
    "trx_bodysaw": {"anchor_lift": "bodyweight", "ratio_mean": 0.290, "ratio_sd": 0.029, "exercise_family": "core_anti_extension"},
    "suitcase_carry": {"anchor_lift": "bodyweight", "ratio_mean": 0.440, "ratio_sd": 0.081, "exercise_family": "unilateral_carry"},
    "bird_dog": {"anchor_lift": "bodyweight", "ratio_mean": 0.250, "ratio_sd": 0.029, "exercise_family": "core_stability"},
}


def build_anchor_ratio_matrix(exercises: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-exercise ratios for the three supported strength anchors."""
    matrix = np.zeros((len(exercises), len(ANCHOR_NAMES)), dtype=np.float32)
    available = np.zeros((len(exercises),), dtype=np.float32)
    anchor_to_idx = {name: idx for idx, name in enumerate(ANCHOR_NAMES)}

    for exercise_idx, exercise in enumerate(exercises):
        prior = EXERCISE_STRENGTH_PRIORS.get(exercise)
        if not prior:
            continue
        anchor_name = prior["anchor_lift"]
        if anchor_name not in anchor_to_idx:
            continue
        matrix[exercise_idx, anchor_to_idx[anchor_name]] = float(prior["ratio_mean"])
        available[exercise_idx] = 1.0

    return matrix, available


def default_anchor_array_kg() -> np.ndarray:
    return np.array([DEFAULT_ANCHOR_VALUES_KG[name] for name in ANCHOR_NAMES], dtype=np.float32)


def coerce_anchor_values(anchor_values, defaults=None) -> np.ndarray:
    """Normalize anchors from dict/list/tuple into a dense [bench, squat, deadlift] array."""
    default_arr = default_anchor_array_kg() if defaults is None else np.asarray(defaults, dtype=np.float32).copy()

    if anchor_values is None:
        return default_arr

    if isinstance(anchor_values, Mapping):
        out = default_arr.copy()
        for idx, anchor_name in enumerate(ANCHOR_NAMES):
            value = None
            for key in ANCHOR_ALIASES[anchor_name]:
                if key in anchor_values:
                    value = anchor_values[key]
                    break
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                out[idx] = value
        return out

    if isinstance(anchor_values, np.ndarray):
        arr = anchor_values.astype(np.float32, copy=False).reshape(-1)
        out = default_arr.copy()
        n = min(arr.shape[0], len(ANCHOR_NAMES))
        mask = np.isfinite(arr[:n]) & (arr[:n] > 0.0)
        out[:n][mask] = arr[:n][mask]
        return out

    if isinstance(anchor_values, Iterable) and not isinstance(anchor_values, (str, bytes)):
        arr = np.asarray(list(anchor_values), dtype=np.float32).reshape(-1)
        out = default_arr.copy()
        n = min(arr.shape[0], len(ANCHOR_NAMES))
        mask = np.isfinite(arr[:n]) & (arr[:n] > 0.0)
        out[:n][mask] = arr[:n][mask]
        return out

    return default_arr


def resolve_anchor_values(anchor_values=None, records=None, defaults=None) -> np.ndarray:
    """Resolve anchors from explicit input first, then from records, else defaults."""
    default_arr = default_anchor_array_kg() if defaults is None else np.asarray(defaults, dtype=np.float32).copy()

    if anchor_values is not None:
        return coerce_anchor_values(anchor_values, defaults=default_arr)

    if records is not None:
        for record in records:
            resolved = coerce_anchor_values(record, defaults=default_arr)
            if np.any(resolved > 0.0):
                return resolved

    return default_arr
