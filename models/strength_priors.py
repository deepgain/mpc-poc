"""Shared 1RM prior utilities for training and inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

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
DEFAULT_UPDATE_ALPHA = 0.15
DEFAULT_UPDATE_MAX_RELATIVE_CHANGE = 0.04
DEFAULT_UPDATE_MIN_RELATIVE_LOAD = 0.50
DEFAULT_UPDATE_MAX_REPS = 10
DEFAULT_UPDATE_MAX_RIR = 3
DEFAULT_UPDATE_TOP_K = 3
DEFAULT_SESSION_GAP_HOURS = 6.0


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


def get_exercise_strength_prior(exercise: str) -> dict | None:
    return EXERCISE_STRENGTH_PRIORS.get(exercise)


def get_exercise_anchor_name(exercise: str) -> str | None:
    prior = get_exercise_strength_prior(exercise)
    if not prior:
        return None
    anchor_name = prior["anchor_lift"]
    if anchor_name not in ANCHOR_NAMES:
        return None
    return anchor_name


def get_exercise_anchor_ratio(exercise: str) -> float | None:
    prior = get_exercise_strength_prior(exercise)
    if not prior:
        return None
    anchor_name = prior["anchor_lift"]
    if anchor_name not in ANCHOR_NAMES:
        return None
    ratio = float(prior["ratio_mean"])
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None
    return ratio


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


def project_exercise_1rm_kg(exercise: str, anchor_values=None, defaults=None) -> float | None:
    anchors = coerce_anchor_values(anchor_values, defaults=defaults)
    anchor_name = get_exercise_anchor_name(exercise)
    ratio = get_exercise_anchor_ratio(exercise)
    if anchor_name is None or ratio is None:
        return None
    anchor_idx = ANCHOR_NAMES.index(anchor_name)
    projected = float(anchors[anchor_idx] * ratio)
    if not np.isfinite(projected) or projected <= 0.0:
        return None
    return projected


def estimate_e1rm_candidate(
    weight_kg: float,
    reps: int,
    rir: float,
) -> float | None:
    """Estimate e1RM from a completed set using Epley + RIR."""
    try:
        weight_kg = float(weight_kg)
        reps = float(reps)
        rir = float(rir)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(weight_kg) or not np.isfinite(reps) or not np.isfinite(rir):
        return None
    if weight_kg <= 0.0 or reps <= 0.0 or rir < 0.0:
        return None

    reps_to_failure = reps + rir
    if reps_to_failure <= 0.0:
        return None

    candidate = weight_kg * (1.0 + reps_to_failure / 30.0)
    if not np.isfinite(candidate) or candidate <= 0.0:
        return None
    return float(candidate)


def coerce_timestamp(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    to_pydatetime = getattr(ts, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime().replace(tzinfo=None)
    if isinstance(ts, np.datetime64):
        try:
            return datetime.fromisoformat(str(ts)).replace(tzinfo=None)
        except ValueError:
            return None
    s = str(ts).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def is_new_session(
    previous_timestamp,
    current_timestamp,
    *,
    session_gap_hours: float = DEFAULT_SESSION_GAP_HOURS,
) -> bool:
    prev_dt = coerce_timestamp(previous_timestamp)
    curr_dt = coerce_timestamp(current_timestamp)
    if prev_dt is None or curr_dt is None:
        return False
    gap_hours = (curr_dt - prev_dt).total_seconds() / 3600.0
    if not np.isfinite(gap_hours):
        return False
    return gap_hours > float(session_gap_hours)


def _score_update_candidate(
    reps: float,
    rir: float,
    relative_load: float,
    *,
    min_relative_load: float,
    max_rir: float,
) -> float:
    load_score = np.clip((relative_load - min_relative_load) / max(1.0 - min_relative_load, 1e-6), 0.0, 1.0)
    rir_score = np.clip((max_rir + 1.0 - rir) / (max_rir + 1.0), 0.0, 1.0)
    reps_score = np.clip(1.0 - abs(reps - 5.0) / 7.0, 0.25, 1.0)
    return float(0.50 * load_score + 0.30 * rir_score + 0.20 * reps_score)


def collect_strength_update_candidates(
    completed_sets: list[dict],
    strength_anchors=None,
    *,
    min_relative_load: float = DEFAULT_UPDATE_MIN_RELATIVE_LOAD,
    max_reps: int = DEFAULT_UPDATE_MAX_REPS,
    max_rir: int = DEFAULT_UPDATE_MAX_RIR,
) -> list[dict]:
    """Collect high-quality anchor update candidates from completed sets."""
    anchors_kg = resolve_anchor_values(anchor_values=strength_anchors)
    candidates = []

    for entry in completed_sets:
        exercise = entry.get("exercise", "")
        anchor_name = get_exercise_anchor_name(exercise)
        ratio = get_exercise_anchor_ratio(exercise)
        if anchor_name is None or ratio is None:
            continue

        try:
            weight_kg = float(entry["weight_kg"])
            reps = int(entry["reps"])
            rir = float(entry["rir"])
        except (KeyError, TypeError, ValueError):
            continue

        if weight_kg <= 0.0 or reps <= 0 or reps > max_reps or rir < 0.0 or rir > max_rir:
            continue

        projected_1rm = project_exercise_1rm_kg(exercise, anchors_kg)
        if projected_1rm is None or projected_1rm <= 0.0:
            continue

        relative_load = weight_kg / projected_1rm
        if not np.isfinite(relative_load) or relative_load < min_relative_load:
            continue

        e1rm_candidate = estimate_e1rm_candidate(weight_kg, reps, rir)
        if e1rm_candidate is None:
            continue

        anchor_candidate = e1rm_candidate / ratio
        if not np.isfinite(anchor_candidate) or anchor_candidate <= 0.0:
            continue

        candidates.append(
            {
                "exercise": exercise,
                "anchor_name": anchor_name,
                "weight_kg": weight_kg,
                "reps": reps,
                "rir": rir,
                "relative_load": float(relative_load),
                "projected_1rm": float(projected_1rm),
                "exercise_candidate_1rm": float(e1rm_candidate),
                "anchor_candidate_1rm": float(anchor_candidate),
                "quality": _score_update_candidate(
                    reps,
                    rir,
                    relative_load,
                    min_relative_load=min_relative_load,
                    max_rir=max_rir,
                ),
                "timestamp": entry.get("timestamp"),
            }
        )

    return candidates


def update_strength_anchors(
    strength_anchors,
    completed_sets: list[dict],
    *,
    alpha: float = DEFAULT_UPDATE_ALPHA,
    max_relative_change: float = DEFAULT_UPDATE_MAX_RELATIVE_CHANGE,
    min_relative_load: float = DEFAULT_UPDATE_MIN_RELATIVE_LOAD,
    max_reps: int = DEFAULT_UPDATE_MAX_REPS,
    max_rir: int = DEFAULT_UPDATE_MAX_RIR,
    top_k_per_anchor: int = DEFAULT_UPDATE_TOP_K,
    return_details: bool = False,
):
    """Update bench/squat/deadlift anchors from completed sets."""
    current_anchors = resolve_anchor_values(anchor_values=strength_anchors)
    new_anchors = current_anchors.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    max_relative_change = float(max(0.0, max_relative_change))
    top_k_per_anchor = max(1, int(top_k_per_anchor))

    candidates = collect_strength_update_candidates(
        completed_sets,
        current_anchors,
        min_relative_load=min_relative_load,
        max_reps=max_reps,
        max_rir=max_rir,
    )

    grouped = {anchor_name: [] for anchor_name in ANCHOR_NAMES}
    for candidate in candidates:
        grouped[candidate["anchor_name"]].append(candidate)

    details = {}
    for anchor_idx, anchor_name in enumerate(ANCHOR_NAMES):
        current_value = float(current_anchors[anchor_idx])
        anchor_candidates = sorted(
            grouped[anchor_name],
            key=lambda item: (item["quality"], item["relative_load"]),
            reverse=True,
        )

        if not anchor_candidates:
            details[anchor_name] = {
                "updated": False,
                "old_1rm": current_value,
                "new_1rm": current_value,
                "n_candidates": 0,
                "selected_exercises": [],
            }
            continue

        selected = anchor_candidates[:top_k_per_anchor]
        weights = np.array([max(item["quality"], 1e-6) for item in selected], dtype=np.float32)
        values = np.array([item["anchor_candidate_1rm"] for item in selected], dtype=np.float32)
        session_candidate = float(np.average(values, weights=weights))

        blended = (1.0 - alpha) * current_value + alpha * session_candidate
        lower_bound = current_value * (1.0 - max_relative_change)
        upper_bound = current_value * (1.0 + max_relative_change)
        new_value = float(np.clip(blended, lower_bound, upper_bound))
        new_anchors[anchor_idx] = new_value

        details[anchor_name] = {
            "updated": True,
            "old_1rm": current_value,
            "new_1rm": new_value,
            "session_candidate": session_candidate,
            "alpha": alpha,
            "n_candidates": len(anchor_candidates),
            "n_selected": len(selected),
            "selected_exercises": [item["exercise"] for item in selected],
            "selected_relative_loads": [float(item["relative_load"]) for item in selected],
        }

    if return_details:
        result = {anchor_name: float(new_anchors[idx]) for idx, anchor_name in enumerate(ANCHOR_NAMES)}
        return result, details
    return new_anchors


def build_anchor_history_from_completed_sets(
    strength_anchors,
    completed_sets: list[dict],
    *,
    session_gap_hours: float = DEFAULT_SESSION_GAP_HOURS,
    apply_trailing_session: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    current_anchors = resolve_anchor_values(anchor_values=strength_anchors)
    anchor_history = []
    pending_session = []
    previous_timestamp = None

    for entry in completed_sets:
        timestamp = entry.get("timestamp")
        if pending_session and is_new_session(
            previous_timestamp,
            timestamp,
            session_gap_hours=session_gap_hours,
        ):
            current_anchors = resolve_anchor_values(
                anchor_values=update_strength_anchors(current_anchors, pending_session)
            )
            pending_session = []

        anchor_history.append(current_anchors.copy())
        pending_session.append(entry)
        previous_timestamp = timestamp

    final_anchors = current_anchors.copy()
    if pending_session and apply_trailing_session:
        final_anchors = resolve_anchor_values(
            anchor_values=update_strength_anchors(current_anchors, pending_session)
        )

    if anchor_history:
        history_array = np.stack(anchor_history, axis=0).astype(np.float32, copy=False)
    else:
        history_array = np.zeros((0, len(ANCHOR_NAMES)), dtype=np.float32)

    return history_array, np.asarray(final_anchors, dtype=np.float32)
