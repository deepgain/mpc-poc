#!/usr/bin/env python3
"""
DeepGain Synthetic Training Data Generator v3 — Empirical, MPC-Free, RIR-Based
===============================================================================
Generates realistic resistance training logs:
    (user_id, exercise, weight_kg, reps, rir, timestamp)

DESIGN PHILOSOPHY:
    This simulator contains NO hidden fatigue state (no MPC, no latent variables).
    All performance predictions come from FIVE empirical lookup tables derived
    directly from peer-reviewed research:

      Table 1: Set-to-set retention ratios  (how reps drop across sets)
      Table 2: Cross-exercise transfer      (how prior exercises reduce performance)
      Table 3: Inter-session recovery        (how performance returns between days)
      Table 4: RIR noise model               (how inaccurately people report RIR)
      Table 5: Day-to-day variability        (how performance fluctuates session to session)

    The goal is to produce data that a DeepGain MPC model must then *learn* to
    explain — the simulator itself has no concept of per-muscle capacity.

OUTPUT FORMAT (DeepGain Data Standard v2):
    | Column    | Type     | Description                          |
    |-----------|----------|--------------------------------------|
    | user_id   | string   | User identifier                      |
    | exercise  | string   | Exercise name (e.g. bench_press)     |
    | weight_kg | float    | Weight on bar (kg)                   |
    | reps      | int      | Repetitions performed                |
    | rir       | int      | Reported Repetitions in Reserve (0-5)|
    | timestamp | datetime | When the set was performed           |

BIBLIOGRAPHY (57 peer-reviewed sources):
─────────────────────────────────────────
RM TABLE & REPS-TO-FAILURE:
  [1]  Nuzzo 2024, Sports Med 54:303-321 — meta-regression 269 studies, 7289 subjects
  [2]  Shimano 2006, JSCR 20(4):819 — exercise-specific reps at %1RM
  [3]  Hoeger 1990, JSCR 4(3):76 — reps to failure, 7 exercises
  [4]  Wolfe 2024 — trained females, reps at 65/75/85/95% SBD

SET-TO-SET RETENTION:
  [5]  Willardson & Burkett 2005, JSCR — 4 sets bench/squat 8RM, 1/2/5 min rest
  [6]  Willardson & Burkett 2006a, JSCR — 5 sets bench 80%/50%, 1/2/3 min rest
  [7]  Willardson & Burkett 2006b, JSCR — 5 sets 15RM, 0.5/1/2 min rest
  [8]  Richmond & Godard 2004 — 2 sets bench 75%, 1/3/5 min rest
  [9]  Kraemer 1997 — 3 sets bench 10RM, 1/3 min rest
  [10] Senna 2016 — 5 sets bench ~3RM, 1/2/3/5 min rest
  [11] Refalo 2023, Sports Med Open 9:10 — 6 sets bench 75%, 3 RIR conditions

RIR ACCURACY:
  [12] Zourdos 2016, JSCR 30(1):267 — RIR-RPE scale, accuracy by %1RM
  [13] Helms 2017a, JSCR 31(2):292 — RPE & velocity for SBD
  [14] Halperin 2022, Sports Med 52(2):377 — meta: underprediction ~1 rep, SD=1.45
  [15] Refalo 2024, JSCR 38(3):e78 — intraset RIR accuracy 0.65±0.78
  [16] Zourdos 2021, JSCR 35(2S):S158 — high-rep RIR accuracy degradation
  [17] Steele 2017, PeerJ 5:e4105 — experience effect on prediction
  [18] Hackett 2017, JSCR 31(8):2162 — exercise & proximity effects
  [19] Hackett 2012 — bodybuilder RIR accuracy r=0.93-0.95
  [20] Remmert 2023, Percept Mot Skills — no sex/exercise effect
  [21] Robinson 2024, Sports Med 54(9):2209 — dose-response meta-regression using RIR
  [22] Helms 2016, S&C Journal 38(4):42 — RIR-RPE application
  [23] Ruiz-Alias 2025 — sex differences at 65% but not 75/85%
  [24] Jukic 2024, Physiol Rep — RIR-velocity modeling

CROSS-EXERCISE FATIGUE TRANSFER:
  [25] Simão 2005, JSCR 19(1):152 — exercise order BP→LPD→SP→BC→TE
  [26] Senna 2019, J Human Kinetics — bench+fly order, 5×10RM
  [27] Spreuwenberg 2006, JSCR — squat after full-body: −32.5%
  [28] Sforzo 1996, JSCR — large→small vs small→large
  [29] Simão 2012, Sports Med 42(3):251 — exercise order review
  [30] Arazi 2015 — per-set data, 4 exercises, 2 orders
  [31] Dias 2010 — exercise order effects

RECOVERY BETWEEN SESSIONS:
  [32] Morán-Navarro 2017, Eur J Appl Physiol 117:2387 — failure vs non-failure
  [33] Pareja-Blanco 2019, Sports 7(3):59 — 60%/80% × 20%/40% VL recovery
  [34] Belcher 2019, Appl Physiol Nutr Metab — SBD recovery 96h
  [35] Raastad 2000, Eur J Appl Physiol 82:206 — biphasic recovery
  [36] Bartolomei 2017 — 8×3@90% vs 8×10@70% recovery
  [37] McLester 2003 — supercompensation at 72h
  [38] Häkkinen 1993, JSCR — MVC decline males/females
  [39] Häkkinen 1994, Eur J Appl Physiol — 10×10@70% recovery

SEX DIFFERENCES:
  [40] Refalo 2023 — males −29% vs females −21% velocity loss
  [41] Wolfe 2024 — female reps at various %1RM

DAY-TO-DAY VARIABILITY:
  [42] Grgic 2020 — 1RM test-retest: ICC=0.97, CV=3.5-4.2%
  [43] Day 2004 / Gearhart 2016 — session RPE ICC=0.88-0.895

WARM-UP:
  [44] Ribeiro 2014, Percept Mot Skills 119:133 — no warm-up fatigue effect
  [45] Barroso 2012 — warm-up protocols
  [46] Souza 2025 — warm-up confirmation

EMG / MUSCLE ACTIVATION:
  [47] Martín-Fuentes 2020 — deadlift EMG
  [48] Rodríguez-Ridao 2020 — bench press EMG
  [49] Escamilla 2002 — squat/deadlift EMG
  [50] Contreras 2015 — glute activation
  [51] Saeterbakken 2011 — OHP vs bench EMG
  [52] Signorile 2002 — tricep activation

PERIODIZATION:
  [53] Helms 2018, Front Physiol 9:247 — RPE-based periodization
  [54] Zourdos 2016b — DUP implementation

LOAD-VELOCITY:
  [55] González-Badillo 2010 — bench press load-velocity (R²=0.98)
  [56] Sánchez-Medina 2017 — squat load-velocity (R²=0.96)
  [57] Rodríguez-Rosell 2020 — velocity loss ↔ RIR (R²=0.93-0.97)
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 1: RM LOOKUP — Reps to Failure at %1RM                         ║
# ║  [1] Nuzzo 2024, [2] Shimano 2006, [3] Hoeger 1990, [4] Wolfe 2024   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_RM_PCT  = np.array([1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
_RM_REPS = np.array([1.0,  2.0,  4.5,  6.5,  9.0,  12.0, 15.0, 18.5, 22.0, 27.0, 33.0])
_RM_SD   = np.array([0.0,  1.0,  1.5,  2.2,  2.8,  3.5,  4.1,  4.8,  5.4,  6.0,  7.0])

# Exercise-specific RM multipliers at 60% 1RM [2]
EXERCISE_RM_MULT_60 = {
    "bench_press": 0.73, "incline_bench": 0.70, "close_grip_bench": 0.70,
    "dumbbell_bench": 0.72, "ohp": 0.68, "dumbbell_ohp": 0.65, "dips": 0.72,
    "barbell_row": 0.73, "lat_pulldown": 0.68, "cable_row": 0.70, "pull_up": 0.70,
    "squat": 1.00, "front_squat": 0.95, "deadlift": 0.85, "rdl": 0.78,
    "leg_press": 1.10, "bulgarian_split_squat": 0.90, "hip_thrust": 0.95,
    "tricep_pushdown": 0.63, "overhead_tricep_ext": 0.60, "bicep_curl": 0.63,
    "hammer_curl": 0.65, "lateral_raise": 0.55, "face_pull": 0.55,
    "leg_curl": 0.65, "leg_extension": 0.68, "calf_raise": 0.70,
}


def _interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Interpolate y at x. Handles descending xs via reversal."""
    if xs[0] > xs[-1]:
        return float(np.interp(x, xs[::-1], ys[::-1]))
    return float(np.interp(x, xs, ys))


def max_reps_at_pct(pct_1rm: float, exercise: str = "squat") -> float:
    """Expected reps to failure at a given %1RM for a given exercise. [1][2]"""
    pct = np.clip(pct_1rm, 0.50, 1.00)
    base = _interp(pct, _RM_PCT, _RM_REPS)
    mult_60 = EXERCISE_RM_MULT_60.get(exercise, 0.80)
    fade = max(0.0, min(1.0, (0.90 - pct) / 0.30))
    effective_mult = 1.0 + (mult_60 - 1.0) * fade
    return max(1.0, base * effective_mult)


def rm_between_individual_sd(pct_1rm: float) -> float:
    """Between-individual SD of reps to failure [1]. CV ranges 8.6-33.1%."""
    return _interp(np.clip(pct_1rm, 0.50, 1.00), _RM_PCT, _RM_SD)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 2: SET-TO-SET RETENTION RATIOS                                  ║
# ║  [5]-[11] Willardson, Richmond, Kraemer, Senna, Refalo                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# S2/S1 retention at failure by rest period [5][6][7][8][9]
_BENCH_S2S1_REST  = np.array([0.5,  1.0,  2.0,  3.0,  5.0])
_BENCH_S2S1_RATIO = np.array([0.33, 0.45, 0.62, 0.70, 0.86])

_SQUAT_S2S1_REST  = np.array([0.5,  1.0,  2.0,  3.0,  5.0])
_SQUAT_S2S1_RATIO = np.array([0.50, 0.65, 0.78, 0.82, 0.95])

_ISO_S2S1_REST  = np.array([0.5,  1.0,  1.5,  2.0,  3.0])
_ISO_S2S1_RATIO = np.array([0.55, 0.65, 0.72, 0.78, 0.88])


def s2_s1_retention(rest_minutes: float, exercise_category: str,
                    pct_1rm: float = 0.78) -> float:
    """S2/S1 retention ratio at failure. [5][6][7][8]

    Intensity adjustment [6]: lighter loads → worse retention because more
    reps = more metabolic fatigue. At 50% 1RM deficit is 15% worse,
    at 90% it's 15% better. Bench at 80% S2/S1=0.56@2min vs 50%=0.49@2min.
    """
    rest = np.clip(rest_minutes, 0.5, 5.0)
    if exercise_category == "lower_compound":
        base = _interp(rest, _SQUAT_S2S1_REST, _SQUAT_S2S1_RATIO)
    elif exercise_category == "isolation":
        base = _interp(rest, _ISO_S2S1_REST, _ISO_S2S1_RATIO)
    else:
        base = _interp(rest, _BENCH_S2S1_REST, _BENCH_S2S1_RATIO)

    # Intensity-dependent adjustment [6]
    intensity_factor = 1.0 + 0.375 * (pct_1rm - 0.70)  # 0.85@50%, 1.0@70%, 1.15@90%
    intensity_factor = np.clip(intensity_factor, 0.85, 1.15)
    deficit = 1.0 - base
    adjusted = 1.0 - deficit / intensity_factor
    return float(np.clip(adjusted, 0.20, 0.98))


def set_n_retention(set_number: int, s2_s1: float) -> float:
    """Retention ratio for set N given S2/S1. [5][6][7]
    S3/S2 ≈ S2/S1 + 0.12, S4/S3 converges to 0.85-1.0.
    """
    if set_number <= 1:
        return 1.0
    elif set_number == 2:
        return s2_s1
    elif set_number == 3:
        return min(0.98, s2_s1 + 0.12)
    elif set_number == 4:
        return min(0.98, s2_s1 + 0.20)
    else:
        return min(0.98, s2_s1 + 0.25)


@dataclass
class GeneratorConfig:
    """Configurable knobs for data realism and coverage control."""
    pair_block_rate: float = 0.12
    cross_overlap_base: float = 0.28
    session_penalty_per_exercise: float = 0.04
    max_session_penalty: float = 0.20
    rir_noise_scale: float = 1.0
    floor_fraction_per_exercise: float = 0.02
    floor_min_rows: int = 300


GEN_CONFIG = GeneratorConfig()


def set_generator_config(**kwargs):
    """Update generator config in place while preserving defaults."""
    for k, v in kwargs.items():
        if not hasattr(GEN_CONFIG, k):
            raise ValueError(f"Unknown GeneratorConfig field: {k}")
        setattr(GEN_CONFIG, k, v)


def compute_max_reps_set_n(set1_max_reps: float, set_number: int,
                           rest_minutes: float, exercise_category: str,
                           target_rir: int, pct_1rm: float = 0.78) -> float:
    """Max reps to failure on set N, accounting for non-failure training. [11]

    Refalo 2023: stopping at 3-RIR → 27% decline over 6 sets
    vs 54% at failure. Non-failure retains ~50% more capacity.
    """
    if set_number <= 1:
        return set1_max_reps
    base_s2s1 = s2_s1_retention(rest_minutes, exercise_category, pct_1rm)
    proximity_scale = 1.0 / (1.0 + 0.15 * target_rir)
    deficit = 1.0 - base_s2s1
    adjusted_s2s1 = 1.0 - deficit * proximity_scale
    cumulative = set1_max_reps
    for s in range(2, set_number + 1):
        ratio = set_n_retention(s, adjusted_s2s1)
        cumulative *= ratio
    return max(1.0, cumulative)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 3: CROSS-EXERCISE TRANSFER (within session)                     ║
# ║  [25]-[31] Simão, Senna, Spreuwenberg, Sforzo, Arazi                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]


def _load_scaled_weights(csv_path: str = "exercise_muscle_weights_scaled.csv") -> Dict[str, Dict[str, float]]:
    """Load exercise->muscle coefficients from transformed EMG CSV (same source as train.py)."""
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Missing required file: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [m for m in ALL_MUSCLES if m not in df.columns]
    if missing:
        raise RuntimeError(f"{csv_path} missing muscle columns: {missing}")

    weights: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        ex_id = str(row["exercise_id"])
        ex_weights = {
            m: float(np.clip(row[m], 0.0, 1.0))
            for m in ALL_MUSCLES
            if float(row[m]) > 0.0
        }
        if ex_weights:
            weights[ex_id] = ex_weights
    return weights


def _load_exercise_data(yaml_path: str = "exercise_muscle_order.yaml"):
    """Load canonical exercise list/order from YAML and coefficients from transformed CSV.

    This keeps generator identical to train.py regarding:
    - exercise IDs,
    - involved muscles,
    - numeric involvement coefficients.
    """
    if not os.path.exists(yaml_path):
        raise RuntimeError(f"Missing required file: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data or "exercises" not in data:
        raise RuntimeError(f"Invalid YAML structure in {yaml_path}: missing 'exercises'")

    scaled_weights = _load_scaled_weights()

    exercise_muscles: Dict[str, Dict[str, float]] = {}
    exercise_meta: Dict[str, Dict[str, object]] = {}

    for ex_id, ex_data in data["exercises"].items():
        if not isinstance(ex_data, dict):
            continue

        ranked = (
            ex_data.get("primary_muscles", [])
            + ex_data.get("secondary_muscles", [])
            + ex_data.get("tertiary_muscles", [])
        )
        ranked = [m for m in ranked if m in ALL_MUSCLES]
        if not ranked:
            continue

        exercise_meta[ex_id] = ex_data

        if ex_id in scaled_weights:
            exercise_muscles[ex_id] = scaled_weights[ex_id]
        else:
            # Keep fallback identical to train.py for consistency.
            exercise_muscles[ex_id] = {
                m: max(1.0 - 0.15 * r, 0.3)
                for r, m in enumerate(ranked)
            }

    return exercise_muscles, exercise_meta


EXERCISE_MUSCLES, EXERCISE_META = _load_exercise_data()
ALL_EXERCISES = list(EXERCISE_MUSCLES.keys())


def muscle_overlap(ex_a: str, ex_b: str) -> float:
    """Cosine similarity of EMG vectors between two exercises."""
    ma = EXERCISE_MUSCLES.get(ex_a, {})
    mb = EXERCISE_MUSCLES.get(ex_b, {})
    all_m = set(ma) | set(mb)
    if not all_m:
        return 0.0
    dot = sum(ma.get(m, 0) * mb.get(m, 0) for m in all_m)
    norm_a = math.sqrt(sum(v ** 2 for v in ma.values())) or 1.0
    norm_b = math.sqrt(sum(v ** 2 for v in mb.values())) or 1.0
    return dot / (norm_a * norm_b)


def cross_exercise_penalty(prior_exercises: List[Tuple[str, int, int, float]],
                           current_exercise: str,
                           exercise_position: int) -> float:
    """Rep reduction factor from prior exercises + session fatigue. [25]-[31]

    prior_exercises: list of (exercise_name, total_sets, avg_reps, avg_rir)
    exercise_position: 0-indexed position in session (0 = first exercise)

    Includes session-level fatigue accumulation [27]:
    each exercise beyond the 3rd adds -3% to -5% capacity, cap at -20%.
    """
    penalty = 0.0

    # Cross-exercise transfer from muscle overlap [25][26][29]
    for prior_ex, sets, avg_reps, avg_rir in prior_exercises:
        overlap = muscle_overlap(prior_ex, current_exercise)
        if overlap < 0.05:
            continue
        base_penalty = overlap * GEN_CONFIG.cross_overlap_base  # [25] ~28% at full overlap
        volume_scale = min(1.5, sets / 3.0)
        proximity_scale = 1.0 / (1.0 + 0.12 * avg_rir)
        penalty += base_penalty * volume_scale * proximity_scale

    # Session-level accumulated fatigue [27]
    # Spreuwenberg: -32.5% after full-body session
    # Model: -4% per exercise beyond the 3rd, cap at -20%
    if exercise_position >= 3:
        session_penalty = GEN_CONFIG.session_penalty_per_exercise * (exercise_position - 2)
        penalty += min(GEN_CONFIG.max_session_penalty, session_penalty)

    effective_multiplier = 1.0 - min(0.45, penalty)  # cap total at 45%
    return max(0.55, effective_multiplier)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 4: INTER-SESSION RECOVERY                                      ║
# ║  [32]-[39] Morán-Navarro, Pareja-Blanco, Belcher, Raastad, McLester   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Recovery curves for MUSCLE TISSUE (not total performance which includes CNS)
# Total performance recovers fast (CNS + metabolic clear in hours)
# Muscle tissue: structural damage takes 24-72h+ to repair
# Implied τ: upper ~18-22h, lower ~28-36h [32][34][38]

_UPPER_MOD_HOURS  = np.array([0, 6, 12, 24, 36, 48, 72, 96])
_UPPER_MOD_PERF   = np.array([0.80, 0.84, 0.87, 0.93, 0.97, 0.99, 1.00, 1.00])

_UPPER_HARD_HOURS = np.array([0, 6, 12, 24, 36, 48, 72, 96])
_UPPER_HARD_PERF  = np.array([0.73, 0.76, 0.80, 0.87, 0.93, 0.97, 1.00, 1.00])

_LOWER_MOD_HOURS  = np.array([0, 6, 12, 24, 36, 48, 72, 96])
_LOWER_MOD_PERF   = np.array([0.77, 0.79, 0.82, 0.86, 0.90, 0.94, 0.98, 1.00])

_LOWER_HARD_HOURS = np.array([0, 6, 12, 24, 36, 48, 72, 96])
_LOWER_HARD_PERF  = np.array([0.67, 0.70, 0.73, 0.78, 0.84, 0.90, 0.96, 1.00])

_LOWER_EXTREME_HOURS = np.array([0, 6, 12, 24, 36, 48, 72, 96, 120])
_LOWER_EXTREME_PERF  = np.array([0.60, 0.63, 0.66, 0.72, 0.78, 0.85, 0.92, 0.97, 1.00])

_DL_HARD_HOURS = np.array([0, 6, 12, 24, 36, 48, 72])
_DL_HARD_PERF  = np.array([0.80, 0.83, 0.86, 0.91, 0.95, 0.98, 1.00])


def session_recovery_multiplier(hours_since: float, body_region: str,
                                session_severity: str) -> float:
    """Performance multiplier based on time since last training. [32]-[39]"""
    h = max(0, hours_since)
    if body_region == "deadlift":
        return float(np.clip(_interp(h, _DL_HARD_HOURS, _DL_HARD_PERF), 0.6, 1.0))
    elif body_region == "upper":
        if session_severity == "hard":
            return float(np.clip(_interp(h, _UPPER_HARD_HOURS, _UPPER_HARD_PERF), 0.6, 1.0))
        return float(np.clip(_interp(h, _UPPER_MOD_HOURS, _UPPER_MOD_PERF), 0.6, 1.0))
    else:  # lower
        if session_severity == "extreme":
            return float(np.clip(_interp(h, _LOWER_EXTREME_HOURS, _LOWER_EXTREME_PERF), 0.5, 1.0))
        elif session_severity == "hard":
            return float(np.clip(_interp(h, _LOWER_HARD_HOURS, _LOWER_HARD_PERF), 0.6, 1.0))
        return float(np.clip(_interp(h, _LOWER_MOD_HOURS, _LOWER_MOD_PERF), 0.6, 1.0))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 5: RIR NOISE MODEL                                             ║
# ║  [14]-[20] Halperin, Refalo, Zourdos, Hackett, Remmert                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def rir_noise(actual_rir: float, total_reps_in_set: int,
              rng: np.random.Generator, user_bias: float = 0.0) -> int:
    """Convert actual RIR to reported (noisy) integer RIR. [14]-[20]

    Halperin 2022: mean underprediction = 0.95 reps, SD=1.45
    Refalo 2024: absolute error 0.65±0.78 at 75% 1RM bench
    Zourdos 2021: error 2.05±1.73 at 70% squat (~16 reps) at 1 RIR
    Remmert 2023: no sex/exercise effect; only proximity + set# matter

    Rep-count amplification [14]: +0.47 reps error per rep beyond 12
    """
    population_bias = -0.95  # [14] underprediction

    if total_reps_in_set <= 8:
        base_sd = 0.6   # [15]
    elif total_reps_in_set <= 12:
        base_sd = 0.8   # [15]
    else:
        base_sd = 0.8 + 0.20 * (total_reps_in_set - 12)  # [14][16]

    distance_amplifier = 1.0 + 0.15 * max(0, actual_rir - 1)  # [16][20]
    sd = base_sd * distance_amplifier * GEN_CONFIG.rir_noise_scale

    noise = population_bias + user_bias + rng.normal(0, sd)
    reported = actual_rir + noise

    # At failure: very accurate [18][19]
    if actual_rir < 0.5:
        return 0

    return int(np.clip(round(reported), 0, 5))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TABLE 6: DAY-TO-DAY VARIABILITY                                      ║
# ║  [42] Grgic 2020 — 1RM CV = 3.5-4.2%, ICC = 0.97                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def daily_1rm_multiplier(rng: np.random.Generator, cv: float = 0.035) -> float:
    """Day-to-day effective 1RM multiplier. [42]"""
    return float(np.clip(1.0 + rng.normal(0, cv), 0.88, 1.12))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE DATABASE — properties, categories, weight increments         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class ExerciseInfo:
    category: str        # 'upper_compound', 'lower_compound', 'isolation'
    body_region: str     # 'upper', 'lower', 'deadlift'
    equipment: str       # 'barbell', 'dumbbell', 'cable', 'machine', 'bodyweight'
    weight_increment: float
    min_weight_kg: float

EXERCISE_INFO: Dict[str, ExerciseInfo] = {
    # Upper body compounds/accessories
    "bench_press": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "close_grip_bench": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "spoto_press": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "incline_bench": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "incline_bench_45": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "decline_bench": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "chest_press_machine": ExerciseInfo("upper_compound", "upper", "machine", 5.0, 20),
    "dips": ExerciseInfo("upper_compound", "upper", "bodyweight", 2.5, 0),
    "ohp": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 15),
    "dumbbell_flyes": ExerciseInfo("isolation", "upper", "dumbbell", 2.0, 5),
    "skull_crusher": ExerciseInfo("isolation", "upper", "barbell", 2.5, 5),
    "pendlay_row": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "pull_up": ExerciseInfo("upper_compound", "upper", "bodyweight", 2.5, 0),
    "lat_pulldown": ExerciseInfo("upper_compound", "upper", "cable", 5.0, 20),
    "reverse_fly": ExerciseInfo("isolation", "upper", "machine", 2.5, 5),
    "seal_row": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),

    # Lower body compounds/accessories
    "squat": ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "low_bar_squat": ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "high_bar_squat": ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "deadlift": ExerciseInfo("lower_compound", "deadlift", "barbell", 2.5, 40),
    "sumo_deadlift": ExerciseInfo("lower_compound", "deadlift", "barbell", 2.5, 35),
    "rdl": ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 30),
    "leg_press": ExerciseInfo("lower_compound", "lower", "machine", 5.0, 40),
    "bulgarian_split_squat": ExerciseInfo("lower_compound", "lower", "dumbbell", 2.0, 0),
    "leg_curl": ExerciseInfo("isolation", "lower", "machine", 2.5, 10),
    "leg_extension": ExerciseInfo("isolation", "lower", "machine", 2.5, 10),

    # Core / stability
    "plank": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
    "farmers_walk": ExerciseInfo("isolation", "upper", "dumbbell", 2.5, 10),
    "leg_raises": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
    "ab_wheel": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
    "dead_bug": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
    "trx_bodysaw": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
    "suitcase_carry": ExerciseInfo("isolation", "upper", "dumbbell", 2.5, 8),
    "bird_dog": ExerciseInfo("isolation", "upper", "bodyweight", 2.5, 0),
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  USER PROFILE GENERATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAINING_LEVELS = {
    "beginner":     {"years": (0.3, 1.5),  "bench_bw": (0.55, 0.85), "squat_bw": (0.70, 1.05)},
    "intermediate": {"years": (1.5, 4.0),  "bench_bw": (0.85, 1.25), "squat_bw": (1.05, 1.55)},
    "advanced":     {"years": (4.0, 10.0), "bench_bw": (1.25, 1.75), "squat_bw": (1.55, 2.15)},
    "elite":        {"years": (8.0, 20.0), "bench_bw": (1.75, 2.40), "squat_bw": (2.15, 2.90)},
}
LEVEL_WEIGHTS = [0.20, 0.40, 0.30, 0.10]


@dataclass
class UserProfile:
    user_id: str
    sex: str
    age: int
    bodyweight_kg: float
    training_years: float
    training_level: str
    e1rm: Dict[str, float]
    rm_factor: float
    rir_bias: float
    daily_cv: float
    consistency: float
    preferred_rest: float
    session_time_pref: str                                    # 'morning', 'evening', 'flexible'
    excluded_exercises: List[str] = field(default_factory=list)
    program_switch_week: Optional[int] = None                 # week to switch programs
    stalling: bool = False                                    # no progression after week 30
    muscle_last_trained: Dict[str, datetime] = field(default_factory=dict)
    muscle_last_severity: Dict[str, str] = field(default_factory=dict)
    muscle_last_region: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        epoch = datetime(2024, 1, 1)
        if not self.muscle_last_trained:
            self.muscle_last_trained = {m: epoch for m in ALL_MUSCLES}
            self.muscle_last_severity = {m: "moderate" for m in ALL_MUSCLES}
            self.muscle_last_region = {m: "upper" for m in ALL_MUSCLES}


def generate_user(rng: np.random.Generator, user_id: str) -> UserProfile:
    """Generate a user with realistic anthropometrics and strength levels."""
    sex = rng.choice(["M", "F"], p=[0.62, 0.38])
    level = rng.choice(list(TRAINING_LEVELS.keys()), p=LEVEL_WEIGHTS)
    cfg = TRAINING_LEVELS[level]
    experience = rng.uniform(*cfg["years"])
    age = int(np.clip(rng.normal(27, 7), 18, 55))

    if sex == "M":
        bw = float(np.clip(rng.normal(82, 13), 58, 125))
        bench_mult = rng.uniform(*cfg["bench_bw"])
        squat_mult = rng.uniform(*cfg["squat_bw"])
    else:
        bw = float(np.clip(rng.normal(63, 10), 44, 95))
        bench_mult = rng.uniform(*cfg["bench_bw"]) * 0.57   # [40][41]
        squat_mult = rng.uniform(*cfg["squat_bw"]) * 0.64

    bench = bw * bench_mult
    squat = bw * squat_mult
    dl = squat * rng.uniform(1.05, 1.25)

    e1rm = {
        "bench_press": bench,
        "close_grip_bench": bench * rng.uniform(0.82, 0.90),
        "spoto_press": bench * rng.uniform(0.86, 0.94),
        "incline_bench": bench * rng.uniform(0.78, 0.86),
        "incline_bench_45": bench * rng.uniform(0.75, 0.83),
        "decline_bench": bench * rng.uniform(0.88, 0.97),
        "chest_press_machine": bench * rng.uniform(0.95, 1.20),
        "dips": bench * rng.uniform(0.58, 0.75),
        "ohp": bench * rng.uniform(0.57, 0.67),
        "dumbbell_flyes": bench * rng.uniform(0.30, 0.45),
        "skull_crusher": bench * rng.uniform(0.26, 0.40),
        "pendlay_row": bench * rng.uniform(0.75, 0.92),
        "pull_up": bench * rng.uniform(0.50, 0.68),
        "lat_pulldown": bench * rng.uniform(0.62, 0.82),
        "reverse_fly": bw * rng.uniform(0.12, 0.22),
        "seal_row": bench * rng.uniform(0.65, 0.88),
        "squat": squat,
        "low_bar_squat": squat * rng.uniform(0.95, 1.08),
        "high_bar_squat": squat * rng.uniform(0.90, 1.00),
        "deadlift": dl,
        "sumo_deadlift": dl * rng.uniform(0.92, 1.06),
        "rdl": dl * rng.uniform(0.65, 0.75),
        "leg_press": squat * rng.uniform(1.30, 1.65),
        "bulgarian_split_squat": squat * rng.uniform(0.34, 0.50),
        "leg_curl": squat * rng.uniform(0.28, 0.42),
        "leg_extension": squat * rng.uniform(0.34, 0.52),
        "plank": bw * rng.uniform(0.40, 0.70),
        "farmers_walk": bw * rng.uniform(0.70, 1.30),
        "leg_raises": bw * rng.uniform(0.30, 0.55),
        "ab_wheel": bw * rng.uniform(0.35, 0.60),
        "dead_bug": bw * rng.uniform(0.20, 0.45),
        "trx_bodysaw": bw * rng.uniform(0.35, 0.60),
        "suitcase_carry": bw * rng.uniform(0.45, 0.95),
        "bird_dog": bw * rng.uniform(0.18, 0.35),
    }

    # Exercise exclusions (injury simulation)
    excluded = []
    if rng.random() < 0.15:
        excluded.append("deadlift")
    if rng.random() < 0.10:
        excluded.append("squat")
    if rng.random() < 0.08:
        excluded.append("ohp")

    # Session time preference
    time_pref = rng.choice(["morning", "evening", "flexible"], p=[0.15, 0.55, 0.30])

    # Program switch mid-year (30% of users)
    switch_week = int(rng.integers(20, 36)) if rng.random() < 0.30 else None

    # Stalling for experienced lifters (20% of advanced/elite)
    stalling = (level in ("advanced", "elite")) and rng.random() < 0.20

    return UserProfile(
        user_id=user_id, sex=sex, age=age, bodyweight_kg=bw,
        training_years=experience, training_level=level, e1rm=e1rm,
        rm_factor=float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.30)),
        rir_bias=float(rng.normal(0, 0.25)),
        daily_cv=float(np.clip(rng.normal(0.035, 0.008), 0.015, 0.060)),
        consistency=float(rng.uniform(0.72, 1.0)),
        preferred_rest=float(np.clip(rng.normal(3.0, 0.7), 1.5, 5.0)),
        session_time_pref=time_pref,
        excluded_exercises=excluded,
        program_switch_week=switch_week,
        stalling=stalling,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WORKOUT TEMPLATES — canonical 34-exercise space                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Entry: (exercise, %1RM, sets, target_RIR, rest_multiplier)
WorkoutEntry = Tuple[str, float, int, int, float]

KEY_PAIR_BLOCKS: List[Tuple[WorkoutEntry, WorkoutEntry]] = [
    (("bench_press", 0.78, 3, 2, 1.1), ("skull_crusher", 0.67, 3, 2, 0.7)),
    (("bench_press", 0.76, 3, 2, 1.1), ("ohp", 0.71, 3, 2, 1.0)),
    (("rdl", 0.74, 3, 2, 1.0), ("leg_curl", 0.70, 3, 2, 0.7)),
    (("pendlay_row", 0.74, 3, 2, 1.0), ("lat_pulldown", 0.71, 3, 2, 0.8)),
    (("squat", 0.77, 3, 2, 1.2), ("leg_press", 0.72, 3, 2, 1.0)),
    (("deadlift", 0.79, 3, 2, 1.4), ("rdl", 0.73, 3, 2, 1.0)),
]

KEY_ORDERED_PAIRS: List[Tuple[str, str]] = [
    ("bench_press", "skull_crusher"),
    ("bench_press", "ohp"),
    ("rdl", "leg_curl"),
    ("pendlay_row", "lat_pulldown"),
    ("squat", "leg_press"),
    ("deadlift", "rdl"),
]


def _exercise_set(entries: List[WorkoutEntry]) -> set:
    return {ex for ex, _, _, _, _ in entries}


def inject_cross_muscle_pair(
    template_entries: List[WorkoutEntry],
    rng: np.random.Generator,
    user: UserProfile,
) -> List[WorkoutEntry]:
    """Inject one key pair block into a subset of sessions to strengthen transfer signal."""
    if rng.random() >= GEN_CONFIG.pair_block_rate:
        return template_entries

    existing = _exercise_set(template_entries)
    candidates: List[Tuple[WorkoutEntry, WorkoutEntry]] = []
    for first, second in KEY_PAIR_BLOCKS:
        a, b = first[0], second[0]
        if a in user.excluded_exercises or b in user.excluded_exercises:
            continue
        if a not in EXERCISE_INFO or b not in EXERCISE_INFO:
            continue
        if a in existing and b in existing:
            continue
        candidates.append((first, second))

    if not candidates:
        return template_entries

    first, second = candidates[int(rng.integers(0, len(candidates)))]
    return [first, second] + template_entries


def count_ordered_pair_sessions(df: pd.DataFrame, pair: Tuple[str, str]) -> int:
    """Count sessions where exercise A appears before B."""
    a, b = pair
    cnt = 0
    dc = df.copy()
    dc["date"] = dc["timestamp"].str[:10]
    for (_, _), grp in dc.groupby(["user_id", "date"]):
        seq = grp.sort_values("timestamp")["exercise"].tolist()
        if a in seq and b in seq and seq.index(a) < seq.index(b):
            cnt += 1
    return cnt

TEMPLATES: Dict[str, List[WorkoutEntry]] = {
    "push_vol": [
        ("bench_press", 0.75, 4, 2, 1.1), ("incline_bench", 0.72, 3, 2, 1.0),
        ("spoto_press", 0.74, 3, 2, 1.0), ("ohp", 0.70, 3, 2, 1.0),
        ("dips", 0.70, 3, 2, 0.8), ("skull_crusher", 0.68, 3, 3, 0.7),
    ],
    "push_str": [
        ("bench_press", 0.87, 5, 1, 1.5), ("close_grip_bench", 0.78, 3, 2, 1.2),
        ("incline_bench_45", 0.76, 3, 2, 1.1), ("decline_bench", 0.78, 3, 2, 1.0),
        ("ohp", 0.76, 3, 2, 1.1),
    ],
    "pull_vol": [
        ("pendlay_row", 0.75, 4, 2, 1.0), ("pull_up", 0.74, 3, 2, 1.0),
        ("lat_pulldown", 0.72, 3, 2, 0.8), ("seal_row", 0.70, 3, 2, 0.8),
        ("reverse_fly", 0.65, 3, 3, 0.5),
    ],
    "legs_quad": [
        ("squat", 0.78, 4, 2, 1.3), ("high_bar_squat", 0.74, 3, 2, 1.2),
        ("leg_press", 0.75, 3, 2, 1.0),
        ("bulgarian_split_squat", 0.72, 3, 2, 0.8), ("leg_extension", 0.70, 3, 2, 0.7),
        ("leg_curl", 0.68, 3, 2, 0.7),
    ],
    "legs_post": [
        ("deadlift", 0.80, 4, 2, 1.5), ("rdl", 0.75, 3, 2, 1.0),
        ("sumo_deadlift", 0.76, 3, 2, 1.3), ("leg_curl", 0.72, 3, 2, 0.7),
        ("leg_press", 0.70, 2, 2, 0.9),
    ],
    "upper_B": [
        ("incline_bench_45", 0.75, 4, 2, 1.2), ("seal_row", 0.75, 4, 2, 1.0),
        ("chest_press_machine", 0.72, 3, 2, 1.0), ("pull_up", 0.75, 3, 2, 1.0),
        ("dumbbell_flyes", 0.65, 3, 2, 0.5), ("skull_crusher", 0.66, 2, 2, 0.5),
    ],
    "upper_A": [
        ("bench_press", 0.78, 4, 2, 1.2), ("pendlay_row", 0.75, 4, 2, 1.0),
        ("ohp", 0.72, 3, 2, 1.0), ("lat_pulldown", 0.72, 3, 2, 0.8),
        ("skull_crusher", 0.68, 2, 2, 0.7), ("reverse_fly", 0.65, 2, 2, 0.5),
    ],
    "lower_A": [
        ("squat", 0.80, 4, 2, 1.3), ("rdl", 0.75, 3, 2, 1.1),
        ("leg_press", 0.72, 3, 2, 1.0), ("leg_curl", 0.70, 3, 2, 0.7),
        ("low_bar_squat", 0.74, 2, 2, 1.0),
    ],
    "lower_B": [
        ("deadlift", 0.82, 4, 2, 1.5), ("sumo_deadlift", 0.76, 3, 2, 1.2),
        ("rdl", 0.75, 3, 2, 1.0), ("leg_extension", 0.70, 3, 2, 0.7),
        ("leg_curl", 0.70, 3, 2, 0.7),
    ],
    "core_day": [
        ("plank", 0.60, 3, 3, 0.7), ("ab_wheel", 0.65, 3, 2, 0.7),
        ("dead_bug", 0.55, 3, 3, 0.7), ("trx_bodysaw", 0.62, 3, 2, 0.7),
        ("leg_raises", 0.60, 3, 2, 0.7), ("farmers_walk", 0.70, 3, 3, 0.7),
        ("suitcase_carry", 0.65, 3, 3, 0.7), ("bird_dog", 0.55, 2, 3, 0.7),
    ],
    "fb_A": [
        ("squat", 0.78, 3, 2, 1.3), ("bench_press", 0.78, 3, 2, 1.2),
        ("pendlay_row", 0.75, 3, 2, 1.0), ("rdl", 0.70, 2, 2, 1.0),
        ("ab_wheel", 0.65, 2, 3, 0.6),
    ],
    "fb_B": [
        ("deadlift", 0.80, 3, 2, 1.5), ("spoto_press", 0.75, 3, 2, 1.0),
        ("lat_pulldown", 0.72, 3, 2, 0.8), ("leg_press", 0.72, 2, 2, 1.0),
        ("plank", 0.60, 2, 3, 0.6),
    ],
    # ── DELOAD [53] ──
    "deload_upper": [
        ("bench_press", 0.62, 3, 4, 1.0), ("pendlay_row", 0.60, 3, 4, 1.0),
        ("ohp", 0.58, 2, 4, 0.8), ("skull_crusher", 0.55, 2, 5, 0.7),
    ],
    "deload_lower": [
        ("squat", 0.62, 3, 4, 1.2), ("rdl", 0.58, 2, 4, 1.0),
        ("leg_extension", 0.55, 2, 5, 0.7), ("leg_curl", 0.55, 2, 5, 0.7),
    ],
    "deload_full": [
        ("high_bar_squat", 0.60, 2, 4, 1.2), ("close_grip_bench", 0.60, 2, 4, 1.0),
        ("seal_row", 0.58, 2, 4, 1.0), ("dead_bug", 0.55, 2, 5, 0.8),
    ],
}


@dataclass(frozen=True)
class Schedule:
    name: str
    days: Tuple[Optional[str], ...]
    deload_days: Tuple[Optional[str], ...]

SCHEDULES: List[Schedule] = [
    Schedule("PPL_6", days=(
        "push_vol","pull_vol","legs_quad","push_str","upper_B","legs_post",None),
        deload_days=("deload_upper","deload_upper","deload_lower",None,None,None,None)),
    Schedule("UL_4", days=(
        "upper_A","lower_A",None,"upper_B","lower_B",None,None),
        deload_days=("deload_upper","deload_lower",None,"deload_upper",None,None,None)),
    Schedule("FB_3", days=(
        "fb_A",None,"fb_B",None,"core_day",None,None),
        deload_days=("deload_full",None,"deload_full",None,None,None,None)),
    Schedule("PL_4", days=(
        "push_str","legs_quad",None,"push_vol","legs_post",None,None),
        deload_days=("deload_upper","deload_lower",None,"deload_upper",None,None,None)),
    Schedule("Core_3", days=(
        "core_day",None,"fb_A",None,"core_day",None,None),
        deload_days=("deload_full",None,"deload_full",None,None,None,None)),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WARM-UP GENERATOR [44][45][46]                                        ║
# ║  Ribeiro 2014, Souza 2025: zero fatigue below ~75% working weight     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_warmup_sets(rng: np.random.Generator, user: UserProfile,
                         exercise: str, working_weight: float,
                         timestamp: datetime) -> Tuple[List[Dict], datetime]:
    """Generate 2-3 warm-up sets. Zero fatigue contribution [44][46]."""
    info = EXERCISE_INFO[exercise]
    rows = []
    t = timestamp

    # Only barbell/dumbbell compounds get warm-ups
    if info.category == "isolation" or info.equipment in ("cable", "machine"):
        return rows, t

    protocols = [
        (0.50, rng.integers(6, 9)),   # 50% × 6-8
        (0.70, rng.integers(3, 6)),   # 70% × 3-5
    ]
    # Heavy work (>82% 1RM) gets an extra warm-up
    e1rm = user.e1rm.get(exercise, 80)
    if e1rm > 0 and working_weight / e1rm > 0.82:
        protocols.append((0.85, rng.integers(1, 4)))  # 85% × 1-3

    for pct, reps in protocols:
        w = round_weight(working_weight * pct, info.weight_increment)
        w = max(w, info.min_weight_kg)
        if w >= working_weight:
            continue
        warm_rir = int(rng.integers(4, 6))  # RIR 4-5
        rows.append({
            "user_id": user.user_id,
            "exercise": exercise,
            "weight_kg": round(w, 1),
            "reps": int(reps),
            "rir": warm_rir,
            "timestamp": t.isoformat(),
        })
        t += timedelta(seconds=rng.uniform(60, 100))  # 60-100s rest [44]

    return rows, t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SET SIMULATOR — the core engine (no hidden state, pure lookup)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def round_weight(w: float, inc: float) -> float:
    return round(w / inc) * inc


def auto_adjust_weight(user: UserProfile, exercise: str, raw_weight: float,
                       target_rir: int, daily_mult: float,
                       recovery_mult: float, cross_mult: float) -> float:
    """Adjust weight so predicted reps are in a sensible 3-15 range."""
    info = EXERCISE_INFO[exercise]
    e1rm = user.e1rm.get(exercise, 80) * daily_mult
    effective_1rm = e1rm * recovery_mult * cross_mult
    if effective_1rm <= 0:
        return info.min_weight_kg
    pct = raw_weight / effective_1rm if effective_1rm > 0 else 1.0
    predicted_max = max_reps_at_pct(pct, exercise) * user.rm_factor
    predicted_working = predicted_max - target_rir
    w = raw_weight
    if predicted_working > 15:
        for _ in range(25):
            w += info.weight_increment
            pct = w / effective_1rm
            if pct >= 0.98:
                break
            test = max_reps_at_pct(pct, exercise) * user.rm_factor - target_rir
            if test <= 12:
                break
    elif predicted_working < 3 and w > info.min_weight_kg:
        for _ in range(20):
            w -= info.weight_increment
            if w < info.min_weight_kg:
                w = info.min_weight_kg
                break
            pct = w / effective_1rm
            test = max_reps_at_pct(pct, exercise) * user.rm_factor - target_rir
            if test >= 3:
                break
    return max(round_weight(w, info.weight_increment), info.min_weight_kg)


def simulate_set(rng: np.random.Generator, user: UserProfile,
                 exercise: str, weight: float, target_rir: int,
                 set_number: int, rest_minutes: float,
                 set1_max_reps: float, timestamp: datetime,
                 pct_1rm: float) -> Optional[Dict]:
    """Simulate one working set. Pure empirical — no hidden state.

    Pipeline:
      1. Set N max reps from retention model [5-11]
      2. Actual reps = max_reps - target_rir + noise
      3. True RIR = max_reps - actual_reps
      4. Reported RIR = true_rir + noise [14-20]
    """
    info = EXERCISE_INFO[exercise]

    max_reps_this_set = compute_max_reps_set_n(
        set1_max_reps, set_number, rest_minutes,
        info.category, target_rir, pct_1rm
    )

    if max_reps_this_set < 1:
        return None

    target_reps = max_reps_this_set - target_rir
    execution_noise = rng.normal(0, 0.6)  # [15]
    actual_reps = int(np.clip(round(target_reps + execution_noise),
                               1, round(max_reps_this_set)))

    true_rir = max(0.0, max_reps_this_set - actual_reps)
    reported_rir = rir_noise(true_rir, actual_reps, rng, user.rir_bias)

    return {
        "user_id": user.user_id,
        "exercise": exercise,
        "weight_kg": round(weight, 1),
        "reps": actual_reps,
        "rir": reported_rir,
        "timestamp": timestamp.isoformat(),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WORKOUT SIMULATOR                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_recovery_for_exercise(user: UserProfile, exercise: str,
                              current_time: datetime) -> float:
    """Lookup inter-session recovery multiplier. [32]-[39]"""
    muscles = EXERCISE_MUSCLES.get(exercise, {})
    if not muscles:
        return 1.0
    worst_recovery = 1.0
    for muscle in muscles:
        last_time = user.muscle_last_trained.get(muscle, datetime(2024, 1, 1))
        hours_since = (current_time - last_time).total_seconds() / 3600
        severity = user.muscle_last_severity.get(muscle, "moderate")
        region = user.muscle_last_region.get(muscle, "upper")
        rec = session_recovery_multiplier(hours_since, region, severity)
        worst_recovery = min(worst_recovery, rec)
    return worst_recovery


def update_muscle_history(user: UserProfile, exercise: str,
                          timestamp: datetime, total_sets: int,
                          avg_rir: float):
    """Record training for inter-session recovery tracking."""
    info = EXERCISE_INFO[exercise]
    muscles = EXERCISE_MUSCLES.get(exercise, {})
    if total_sets >= 5 and avg_rir <= 1:
        severity = "hard"
    elif total_sets >= 8:
        severity = "extreme"
    else:
        severity = "moderate"
    for muscle in muscles:
        existing = user.muscle_last_trained.get(muscle, datetime(2024, 1, 1))
        if timestamp > existing:
            user.muscle_last_trained[muscle] = timestamp
            user.muscle_last_severity[muscle] = severity
            user.muscle_last_region[muscle] = info.body_region


def simulate_workout(rng: np.random.Generator, user: UserProfile,
                     template_name: str, start_time: datetime,
                     week_in_meso: int,
                     include_warmups: bool = True) -> List[Dict]:
    """Generate all sets for one workout session."""
    template = list(TEMPLATES[template_name])
    template = inject_cross_muscle_pair(template, rng, user)
    rows: List[Dict] = []
    t = start_time
    daily_mult = daily_1rm_multiplier(rng, user.daily_cv)
    prior_exercises: List[Tuple[str, int, int, float]] = []

    # RIR periodization within mesocycle [53]
    rir_offset = max(-1, 1 - week_in_meso)

    for ex_position, (exercise, pct_1rm, n_sets, base_rir, rest_mult) in enumerate(template):
        # Skip excluded exercises
        if exercise in user.excluded_exercises:
            continue

        e1rm = user.e1rm.get(exercise, 0)
        if e1rm < 5:
            continue

        info = EXERCISE_INFO[exercise]
        target_rir = max(0, base_rir + rir_offset)
        rest_minutes = user.preferred_rest * rest_mult

        recovery_mult = get_recovery_for_exercise(user, exercise, t)
        cross_mult = cross_exercise_penalty(prior_exercises, exercise, ex_position)

        effective_1rm = e1rm * daily_mult * recovery_mult
        raw_weight = round_weight(e1rm * pct_1rm, info.weight_increment)
        raw_weight = max(raw_weight, info.min_weight_kg)

        weight = auto_adjust_weight(
            user, exercise, raw_weight, target_rir,
            daily_mult, recovery_mult, cross_mult
        )

        # Set 1 max reps
        effective_1rm_for_reps = effective_1rm * cross_mult
        pct_effective = weight / effective_1rm_for_reps if effective_1rm_for_reps > 0 else 1.0
        pct_effective = np.clip(pct_effective, 0.50, 0.999)
        set1_max = max_reps_at_pct(pct_effective, exercise) * user.rm_factor
        if user.sex == "F" and pct_effective < 0.80:
            set1_max *= 1.0 + 0.12 * (0.80 - pct_effective)  # [40][41]
        set1_max = min(set1_max, 30)

        # Warm-ups [44][46]
        if include_warmups:
            warmup_rows, t = generate_warmup_sets(rng, user, exercise, weight, t)
            rows.extend(warmup_rows)
        else:
            t += timedelta(minutes=rng.uniform(3, 6))

        set_reps_log = []
        set_rir_log = []
        weight_adjusted = False

        for set_idx in range(1, n_sets + 1):
            actual_rest = rest_minutes * rng.uniform(0.80, 1.25)
            row = simulate_set(
                rng, user, exercise, weight, target_rir,
                set_idx, actual_rest, set1_max, t, pct_effective
            )
            if row is None:
                weight = round_weight(weight * 0.90, info.weight_increment)
                weight = max(weight, info.min_weight_kg)
                row = simulate_set(
                    rng, user, exercise, weight, target_rir,
                    set_idx, actual_rest, set1_max, t, pct_effective
                )
                if row is None:
                    break

            rows.append(row)
            set_reps_log.append(row["reps"])
            set_rir_log.append(row["rir"])

            # Mid-exercise weight adjustment after set 1
            if set_idx == 1 and not weight_adjusted:
                if row["rir"] <= target_rir - 2 and target_rir >= 2:
                    weight = round_weight(weight * 0.92, info.weight_increment)
                    weight = max(weight, info.min_weight_kg)
                    weight_adjusted = True
                elif row["rir"] >= target_rir + 2 and row["rir"] >= 4:
                    weight = round_weight(weight * 1.05, info.weight_increment)
                    weight_adjusted = True

            t += timedelta(minutes=actual_rest, seconds=rng.uniform(25, 55))

        if set_reps_log:
            avg_reps = sum(set_reps_log) / len(set_reps_log)
            avg_rir = sum(set_rir_log) / len(set_rir_log)
            prior_exercises.append((exercise, len(set_reps_log), int(avg_reps), avg_rir))
            update_muscle_history(user, exercise, t, len(set_reps_log), avg_rir)

        t += timedelta(minutes=rng.uniform(1.5, 4.0))

    return rows


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PROGRAM SIMULATOR — multi-week with periodization                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_program(rng: np.random.Generator, user: UserProfile,
                     n_weeks: int, start_date: datetime,
                     include_warmups: bool = True) -> List[Dict]:
    """Generate n_weeks of training for one user. [53]"""
    schedule_idx = rng.integers(len(SCHEDULES))
    schedule = SCHEDULES[schedule_idx]

    # Handle program switch mid-year
    alt_schedule_idx = (schedule_idx + rng.integers(1, len(SCHEDULES))) % len(SCHEDULES)
    alt_schedule = SCHEDULES[alt_schedule_idx]

    all_rows: List[Dict] = []
    current_date = start_date

    for week_idx in range(n_weeks):
        week_in_meso = week_idx % 4
        is_deload = (week_in_meso == 3)

        # Switch programs if applicable
        active_schedule = schedule
        if user.program_switch_week and week_idx >= user.program_switch_week:
            active_schedule = alt_schedule

        for day_idx in range(7):
            template = (active_schedule.deload_days[day_idx] if is_deload
                        else active_schedule.days[day_idx])
            if template is None:
                current_date += timedelta(days=1)
                continue

            if rng.random() > user.consistency:
                current_date += timedelta(days=1)
                continue

            # Session start time based on preference
            wd = current_date.weekday()
            if user.session_time_pref == "morning":
                hour = int(np.clip(rng.normal(7.0, 1.0), 5, 10))
            elif user.session_time_pref == "evening":
                hour = int(np.clip(rng.normal(18.0, 1.5), 15, 22))
            else:  # flexible
                if wd < 5:
                    hour = int(np.clip(rng.normal(17.5, 1.8), 6, 21))
                else:
                    hour = int(np.clip(rng.normal(10.5, 2.0), 7, 17))

            minute = int(rng.integers(0, 60))
            session_start = current_date.replace(hour=hour, minute=minute)

            workout_rows = simulate_workout(
                rng, user, template, session_start, week_in_meso, include_warmups
            )
            all_rows.extend(workout_rows)
            current_date += timedelta(days=1)

        # Weekly e1RM progression [53]
        if user.stalling and week_idx >= 30:
            prog_rate = 0.0
        else:
            prog_rate = 0.004 / (1.0 + user.training_years * 0.1)
        if is_deload:
            prog_rate *= 0.15
        for ex in user.e1rm:
            user.e1rm[ex] *= (1.0 + rng.normal(prog_rate, prog_rate * 0.5))

    return all_rows


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET GENERATOR                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _find_donor_exercise(exercise: str, counts: Dict[str, int]) -> Optional[str]:
    """Pick a donor exercise for rare zero-count cases (same category preferred)."""
    if exercise not in EXERCISE_INFO:
        return None
    info = EXERCISE_INFO[exercise]
    candidates = [
        ex for ex, c in counts.items()
        if c > 0 and ex != exercise and ex in EXERCISE_INFO and EXERCISE_INFO[ex].category == info.category
    ]
    if not candidates:
        candidates = [ex for ex, c in counts.items() if c > 0 and ex != exercise]
    if not candidates:
        return None
    candidates.sort(key=lambda ex: muscle_overlap(exercise, ex), reverse=True)
    return candidates[0]


def enforce_exercise_floor(df: pd.DataFrame, min_rows_per_exercise: int, seed: int) -> pd.DataFrame:
    """Ensure each exercise has at least min_rows_per_exercise rows via conservative bootstrapping."""
    if len(df) == 0 or min_rows_per_exercise <= 0:
        return df

    rng = np.random.default_rng(seed + 1000)
    target_exercises = sorted(set(ALL_EXERCISES) | set(EXERCISE_INFO.keys()))
    counts = df.groupby("exercise").size().to_dict()
    augmented = [df]

    for ex in target_exercises:
        curr = int(counts.get(ex, 0))
        if curr >= min_rows_per_exercise:
            continue
        need = min_rows_per_exercise - curr

        src_ex = ex
        if curr == 0:
            donor = _find_donor_exercise(ex, counts)
            if donor is None:
                continue
            src_ex = donor

        source_rows = df[df["exercise"] == src_ex]
        if len(source_rows) == 0:
            continue

        sample_idx = rng.integers(0, len(source_rows), size=need)
        sample = source_rows.iloc[sample_idx].copy()
        sample["exercise"] = ex

        # Small perturbations avoid creating exact duplicates while preserving bounds.
        w_noise = rng.normal(1.0, 0.03, size=need)
        sample["weight_kg"] = np.clip(sample["weight_kg"].to_numpy() * w_noise, 0.0, None).round(1)
        rep_noise = rng.integers(-1, 2, size=need)
        sample["reps"] = np.clip(sample["reps"].to_numpy() + rep_noise, 1, 30).astype(int)
        rir_noise = rng.integers(-1, 2, size=need)
        sample["rir"] = np.clip(sample["rir"].to_numpy() + rir_noise, 0, 5).astype(int)

        ts = pd.to_datetime(sample["timestamp"], format="ISO8601", errors="coerce")
        delta_min = rng.integers(5, 240, size=need)
        ts = ts + pd.to_timedelta(delta_min, unit="m")
        sample["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")

        augmented.append(sample)
        counts[ex] = min_rows_per_exercise

    out = pd.concat(augmented, ignore_index=True)

    # Guarantee pass: fill any residual gaps using the already-augmented frame.
    counts2 = out.groupby("exercise").size().to_dict()
    extra = []
    for ex in target_exercises:
        curr = int(counts2.get(ex, 0))
        if curr >= min_rows_per_exercise:
            continue
        need = min_rows_per_exercise - curr
        donor = _find_donor_exercise(ex, counts2)
        if donor is None:
            continue
        source_rows = out[out["exercise"] == donor]
        if len(source_rows) == 0:
            continue
        sample_idx = rng.integers(0, len(source_rows), size=need)
        sample = source_rows.iloc[sample_idx].copy()
        sample["exercise"] = ex
        sample["weight_kg"] = np.clip(sample["weight_kg"].to_numpy() * rng.normal(1.0, 0.03, size=need), 0.0, None).round(1)
        sample["reps"] = np.clip(sample["reps"].to_numpy() + rng.integers(-1, 2, size=need), 1, 30).astype(int)
        sample["rir"] = np.clip(sample["rir"].to_numpy() + rng.integers(-1, 2, size=need), 0, 5).astype(int)
        ts = pd.to_datetime(sample["timestamp"], format="ISO8601", errors="coerce")
        ts = ts + pd.to_timedelta(rng.integers(5, 240, size=need), unit="m")
        sample["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
        extra.append(sample)

    if extra:
        out = pd.concat([out] + extra, ignore_index=True)

    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return out


def compute_quality_metrics(df: pd.DataFrame, floor_target: int) -> Dict[str, float]:
    """Compact quality metrics used for diagnostics and config search."""
    if len(df) == 0:
        return {
            "rows": 0.0,
            "rir_0_3_pct": 0.0,
            "mean_reps": 0.0,
            "floor_pass_pct": 0.0,
            "pair_mean": 0.0,
            "bench_s2s1": 0.0,
            "score": -1e9,
        }

    counts = df.groupby("exercise").size()
    floor_pass = sum(int(counts.get(ex, 0) >= floor_target) for ex in ALL_EXERCISES) / max(1, len(ALL_EXERCISES))

    dc = df.copy()
    dc["date"] = dc["timestamp"].str[:10]
    n_sessions = dc.groupby(["user_id", "date"]).ngroups

    pair_counts = [count_ordered_pair_sessions(df, pair) for pair in KEY_ORDERED_PAIRS]
    pair_mean = (sum(pair_counts) / len(pair_counts)) / max(1, n_sessions)

    rir_0_3_pct = float(((df["rir"] >= 0) & (df["rir"] <= 3)).mean())
    mean_reps = float(df["reps"].mean())

    bench = df[df["exercise"] == "bench_press"].copy()
    bench["date"] = bench["timestamp"].str[:10]
    s2_s1 = []
    for (_, _), grp in bench.groupby(["user_id", "date"]):
        sets = grp.sort_values("timestamp")
        working = sets[sets["rir"] < 4]
        if len(working) >= 2 and working["reps"].iloc[0] > 0:
            s2_s1.append(float(working["reps"].iloc[1] / working["reps"].iloc[0]))
        if len(s2_s1) >= 400:
            break
    bench_s2s1 = float(np.mean(s2_s1)) if s2_s1 else 0.0

    # Score near target bands: RIR0-3 ≈ 0.78, reps ≈ 8.5, S2/S1 ≈ 0.85.
    score = 0.0
    score -= abs(rir_0_3_pct - 0.78) * 4.0
    score -= abs(mean_reps - 8.5) * 0.15
    score -= abs(bench_s2s1 - 0.85) * 1.5
    score += floor_pass * 2.5
    score += min(pair_mean / 0.10, 1.0) * 2.0

    return {
        "rows": float(len(df)),
        "rir_0_3_pct": rir_0_3_pct,
        "mean_reps": mean_reps,
        "floor_pass_pct": float(floor_pass),
        "pair_mean": float(pair_mean),
        "bench_s2s1": bench_s2s1,
        "score": float(score),
    }

def generate_dataset(n_users: int = 200, n_weeks: int = 52,
                     seed: int = 42, include_warmups: bool = True,
                     target_rows: Optional[int] = None,
                     floor_min_rows: Optional[int] = None) -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)
    all_rows: List[Dict] = []
    report_every = 20
    i = 0

    while True:
        need_more_users = i < n_users
        need_more_rows = target_rows is not None and len(all_rows) < target_rows
        if not (need_more_users or need_more_rows):
            break

        if i % report_every == 0:
            row_str = f", rows={len(all_rows):,}" if target_rows is not None else ""
            print(f"\r  Users generated: {i:,}{row_str}", end="", flush=True)

        user = generate_user(rng, f"user_{i:05d}")
        rows = simulate_program(rng, user, n_weeks, start_date, include_warmups)
        all_rows.extend(rows)
        i += 1

        if target_rows is not None and i > max(n_users * 5, 5000):
            break

    print(f"\r  Users generated: {i:,}, rows={len(all_rows):,}    ")

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df

    per_ex_floor = floor_min_rows
    if per_ex_floor is None:
        per_ex_floor = max(
            GEN_CONFIG.floor_min_rows,
            int((GEN_CONFIG.floor_fraction_per_exercise * len(df)) / max(1, len(ALL_EXERCISES))),
        )

    if per_ex_floor > 0:
        df = enforce_exercise_floor(df, per_ex_floor, seed)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DIAGNOSTICS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_diagnostics(df: pd.DataFrame) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"Total sets:       {len(df):>10,}")
    print(f"Unique users:     {df['user_id'].nunique():>10,}")
    print(f"Unique exercises: {df['exercise'].nunique():>10,}")
    print(f"Date range:       {df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")

    print(f"\n{sep}")
    print("RIR distribution:")
    for rir_val in range(6):
        count = (df['rir'] == rir_val).sum()
        pct = count / len(df) * 100
        bar = '█' * int(pct / 1.5)
        print(f"  RIR {rir_val}: {count:>8,} ({pct:5.1f}%) {bar}")

    print(f"\n{sep}")
    print("Top 15 exercises:")
    for ex, cnt in df.groupby('exercise').size().sort_values(ascending=False).head(15).items():
        print(f"  {ex:28s} {cnt:>7,}")

    print(f"\n{sep}")
    print("Exercise floor coverage:")
    counts = df.groupby('exercise').size()
    floor_target = max(
        GEN_CONFIG.floor_min_rows,
        int((GEN_CONFIG.floor_fraction_per_exercise * len(df)) / max(1, len(ALL_EXERCISES))),
    )
    pass_cnt = 0
    for ex in ALL_EXERCISES:
        c = int(counts.get(ex, 0))
        ok = c >= floor_target
        pass_cnt += int(ok)
        marker = "OK" if ok else "LOW"
        print(f"  {ex:28s} {c:>7,} ({marker})")
    print(f"  Floor target/exercise: {floor_target:,}  pass: {pass_cnt}/{len(ALL_EXERCISES)}")

    print(f"\n{sep}")
    print("Key ordered pair coverage (session-level):")
    dc = df.copy()
    dc['date'] = dc['timestamp'].str[:10]
    n_sessions = max(1, dc.groupby(['user_id', 'date']).ngroups)
    for a, b in KEY_ORDERED_PAIRS:
        c = count_ordered_pair_sessions(df, (a, b))
        pct = 100.0 * c / n_sessions
        print(f"  {a:18s} -> {b:18s} {c:>6,} sessions ({pct:5.2f}%)")

    print(f"\n{sep}")
    print(f"Reps: mean={df['reps'].mean():.1f}  median={df['reps'].median():.0f}  "
          f"min={df['reps'].min()}  max={df['reps'].max()}")

    # Rep range distribution
    print(f"\n{sep}")
    print("Rep ranges:")
    for lo, hi in [(1,3), (4,6), (7,9), (10,12), (13,15), (16,20), (21,30)]:
        count = ((df['reps'] >= lo) & (df['reps'] <= hi)).sum()
        pct = count / len(df) * 100
        bar = '█' * int(pct / 1.5)
        print(f"  {lo:2d}-{hi:2d}: {count:>8,} ({pct:5.1f}%) {bar}")

    # Weight stats
    print(f"\n{sep}")
    print("Weight stats (top 8 exercises):")
    for ex in df.groupby('exercise').size().sort_values(ascending=False).head(8).index:
        w = df[df['exercise'] == ex]['weight_kg']
        print(f"  {ex:25s}  mean={w.mean():6.1f}kg  std={w.std():5.1f}  "
              f"range=[{w.min():.0f}-{w.max():.0f}]")

    # Session volume
    print(f"\n{sep}")
    print("Session stats:")
    dc = df.copy()
    dc['date'] = dc['timestamp'].str[:10]
    sessions = dc.groupby(['user_id', 'date']).agg(
        n_sets=('reps', 'count'), n_ex=('exercise', 'nunique'), total_reps=('reps', 'sum'))
    print(f"  Sets/session:     mean={sessions['n_sets'].mean():.1f}  std={sessions['n_sets'].std():.1f}")
    print(f"  Exercises/session: mean={sessions['n_ex'].mean():.1f}")
    print(f"  Reps/session:     mean={sessions['total_reps'].mean():.0f}")

    # Weekly sessions
    dc['week'] = pd.to_datetime(dc['timestamp'], format="ISO8601").dt.isocalendar().week.astype(int)
    weekly = dc.groupby(['user_id', 'week'])['date'].nunique()
    print(f"  Sessions/week:    mean={weekly.mean():.1f}  std={weekly.std():.1f}")

    # Rep drop-off validation
    print(f"\n{sep}")
    print("Intra-exercise rep drop-off validation:")
    validated = 0
    total_checked = 0
    for uid in df['user_id'].unique()[:20]:
        u = df[df['user_id'] == uid]
        for day in u['timestamp'].str[:10].unique()[:5]:
            d = u[u['timestamp'].str.startswith(day)]
            for ex in d['exercise'].unique():
                ex_sets = d[d['exercise'] == ex].sort_values('timestamp')
                if len(ex_sets) >= 3:
                    reps_list = ex_sets['reps'].tolist()
                    if reps_list[0] >= reps_list[-1]:
                        validated += 1
                    total_checked += 1
    if total_checked > 0:
        print(f"  {validated}/{total_checked} show first_set >= last_set "
              f"({validated/total_checked*100:.0f}%)")

    # Cross-exercise spot-check
    print(f"\n{sep}")
    print("Cross-exercise fatigue spot-check (bench → OHP):")
    found = False
    for uid in df['user_id'].unique()[:30]:
        if found:
            break
        u = df[df['user_id'] == uid]
        for day in u['timestamp'].str[:10].unique()[:10]:
            d = u[u['timestamp'].str.startswith(day)]
            exs = d['exercise'].unique()
            if 'bench_press' in exs and 'ohp' in exs:
                bp = d[d['exercise'] == 'bench_press']
                oh = d[d['exercise'] == 'ohp']
                if len(bp) >= 2 and len(oh) >= 2:
                    print(f"\n  {uid}, {day}:")
                    print(f"    Bench (first):")
                    for _, r in bp.iterrows():
                        print(f"      {r['weight_kg']}kg × {r['reps']} @ RIR {r['rir']}")
                    print(f"    OHP (after bench):")
                    for _, r in oh.iterrows():
                        print(f"      {r['weight_kg']}kg × {r['reps']} @ RIR {r['rir']}")
                    found = True
                    break

    # Sample workout
    print(f"\n{sep}")
    print("Sample workout (first user, first day):")
    u0 = df[df['user_id'] == df['user_id'].iloc[0]]
    first_day = u0['timestamp'].str[:10].iloc[0]
    sample = u0[u0['timestamp'].str.startswith(first_day)].head(25)
    print(sample.to_string(index=False))

    # Data quality
    print(f"\n{sep}")
    print("Data quality:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  RIR range: [{df['rir'].min()}, {df['rir'].max()}]")
    print(f"  No nulls: {df.isnull().sum().sum() == 0}")
    print(f"  All exercises valid: {all(ex in EXERCISE_MUSCLES for ex in df['exercise'].unique())}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EMPIRICAL PATTERN VALIDATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def validate_empirical_patterns(df: pd.DataFrame) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print("EMPIRICAL PATTERN VALIDATION")
    print(f"{sep}")

    # 1. S2/S1 retention [5-7]
    print("\n  1. Set-to-set rep drop-off (should match [5][6]):")
    bench = df[df['exercise'] == 'bench_press'].copy()
    bench['date'] = bench['timestamp'].str[:10]
    s2_s1 = []
    for (uid, day), grp in bench.groupby(['user_id', 'date']):
        sets = grp.sort_values('timestamp')
        # Skip warm-ups (RIR >= 4)
        working = sets[sets['rir'] < 4]
        if len(working) >= 2:
            reps = working['reps'].tolist()
            if reps[0] > 0:
                s2_s1.append(reps[1] / reps[0])
        if len(s2_s1) >= 500:
            break
    if s2_s1:
        print(f"     Bench S2/S1: {np.mean(s2_s1):.3f} "
              f"(expected ~0.75-0.95 for RIR 1-3 @ 3-5 min rest)")

    # 2. Cross-exercise transfer [25-31]
    print("\n  2. Cross-exercise transfer:")
    dc = df.copy()
    dc['date'] = dc['timestamp'].str[:10]
    ohp_after = []
    for (uid, day), grp in dc.groupby(['user_id', 'date']):
        exs = grp['exercise'].tolist()
        if 'bench_press' in exs and 'ohp' in exs:
            bi = exs.index('bench_press')
            oi = exs.index('ohp')
            if oi > bi:
                ohp_sets = grp[grp['exercise'] == 'ohp']
                ohp_after.extend(ohp_sets['reps'].tolist())
    if ohp_after:
        print(f"     OHP reps after bench: mean={np.mean(ohp_after):.1f}")

    # 3. RIR distribution
    print("\n  3. RIR distribution:")
    rir_0_3 = df[(df['rir'] >= 0) & (df['rir'] <= 3)].shape[0]
    print(f"     RIR 0-3: {rir_0_3/len(df)*100:.1f}% (expected ~70-85%)")

    # 4. Mesocycle periodization [53]
    print("\n  4. Mesocycle periodization (RIR by week in cycle):")
    dc['week_num'] = pd.to_datetime(dc['timestamp'], format="ISO8601").dt.isocalendar().week.astype(int)
    for w_off in range(4):
        w_data = dc[dc['week_num'] % 4 == (w_off + 1) % 4]
        if len(w_data) > 0:
            labels = ["Week 1 (easy)", "Week 2 (normal)",
                      "Week 3 (hard)", "Week 4 (deload)"]
            print(f"     {labels[w_off]}: mean RIR = {w_data['rir'].mean():.2f}")

    # 5. Per-user progression
    print("\n  5. Strength progression (bench, first vs last month):")
    dc['month'] = pd.to_datetime(dc['timestamp'], format="ISO8601").dt.month
    bench_prog = dc[(dc['exercise'] == 'bench_press') & (dc['rir'] < 4)]
    if len(bench_prog) > 0:
        first_month = bench_prog[bench_prog['month'] == bench_prog['month'].min()]
        last_month = bench_prog[bench_prog['month'] == bench_prog['month'].max()]
        if len(first_month) > 0 and len(last_month) > 0:
            print(f"     First month avg weight: {first_month['weight_kg'].mean():.1f} kg")
            print(f"     Last month avg weight:  {last_month['weight_kg'].mean():.1f} kg")
            change = (last_month['weight_kg'].mean() / first_month['weight_kg'].mean() - 1) * 100
            print(f"     Change: {change:+.1f}%")

    print(f"\n{sep}")
    print("Validation complete.")


def run_grid_search(
    seed: int,
    n_users: int,
    n_weeks: int,
    include_warmups: bool,
    max_trials: int,
    out_csv: str,
):
    """Run parameter search for generator realism/coverage trade-off."""
    base_cfg = GeneratorConfig(**vars(GEN_CONFIG))
    rng = np.random.default_rng(seed + 7000)

    grid = list(itertools.product(
        [0.06, 0.10, 0.14],     # pair_block_rate
        [0.24, 0.28, 0.32],     # cross_overlap_base
        [0.03, 0.04, 0.05],     # session_penalty_per_exercise
        [0.90, 1.00, 1.10],     # rir_noise_scale
    ))
    rng.shuffle(grid)
    trials = grid[:min(max_trials, len(grid))]

    records = []
    print(f"Running grid search: {len(trials)} trials")

    for i, (pair_rate, overlap, sess_pen, rir_scale) in enumerate(trials, start=1):
        set_generator_config(
            pair_block_rate=pair_rate,
            cross_overlap_base=overlap,
            session_penalty_per_exercise=sess_pen,
            rir_noise_scale=rir_scale,
        )

        trial_seed = seed + i * 17
        df = generate_dataset(
            n_users=n_users,
            n_weeks=n_weeks,
            seed=trial_seed,
            include_warmups=include_warmups,
        )

        floor_target = max(
            GEN_CONFIG.floor_min_rows,
            int((GEN_CONFIG.floor_fraction_per_exercise * len(df)) / max(1, len(ALL_EXERCISES))),
        )
        metrics = compute_quality_metrics(df, floor_target)
        rec = {
            "trial": i,
            "pair_block_rate": pair_rate,
            "cross_overlap_base": overlap,
            "session_penalty_per_exercise": sess_pen,
            "rir_noise_scale": rir_scale,
            **metrics,
        }
        records.append(rec)
        print(f"  Trial {i:02d}: score={metrics['score']:.3f}, rows={int(metrics['rows']):,}")

    out = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    best = out.iloc[0].to_dict()
    set_generator_config(
        pair_block_rate=float(best["pair_block_rate"]),
        cross_overlap_base=float(best["cross_overlap_base"]),
        session_penalty_per_exercise=float(best["session_penalty_per_exercise"]),
        rir_noise_scale=float(best["rir_noise_scale"]),
    )

    print("\nBest config from grid search:")
    print(f"  pair_block_rate={GEN_CONFIG.pair_block_rate:.3f}")
    print(f"  cross_overlap_base={GEN_CONFIG.cross_overlap_base:.3f}")
    print(f"  session_penalty_per_exercise={GEN_CONFIG.session_penalty_per_exercise:.3f}")
    print(f"  rir_noise_scale={GEN_CONFIG.rir_noise_scale:.3f}")
    print(f"Saved full results to: {out_csv}")

    # Restore immutable defaults for fields not optimized.
    set_generator_config(
        max_session_penalty=base_cfg.max_session_penalty,
        floor_fraction_per_exercise=base_cfg.floor_fraction_per_exercise,
        floor_min_rows=base_cfg.floor_min_rows,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    p = argparse.ArgumentParser(
        description="DeepGain v3 — Empirical, MPC-Free, RIR-Based Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Quick test:      python generate_training_data.py --num_users 10 --weeks 4
  Medium:          python generate_training_data.py --num_users 100 --weeks 16
  Full year:       python generate_training_data.py --num_users 200 --weeks 52
  Large:           python generate_training_data.py --num_users 500 --weeks 52
        """)
    p.add_argument("--num_users", type=int, default=200)
    p.add_argument("--weeks", type=int, default=52)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="training_data.csv")
    p.add_argument("--no_warmups", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--target_rows", type=int, default=0,
                   help="Generate until at least this many rows (0 = disabled).")
    p.add_argument("--floor_min_rows", type=int, default=300,
                   help="Minimum rows per exercise after floor balancing.")
    p.add_argument("--pair_block_rate", type=float, default=0.12,
                   help="Probability of injecting one key cross-muscle pair block per session.")
    p.add_argument("--cross_overlap_base", type=float, default=0.28,
                   help="Base cross-exercise penalty multiplier at full overlap.")
    p.add_argument("--session_penalty_per_exercise", type=float, default=0.04,
                   help="Extra fatigue penalty per exercise after the 3rd one.")
    p.add_argument("--rir_noise_scale", type=float, default=1.0,
                   help="Global scale for RIR-reporting noise SD.")
    p.add_argument("--grid_search", action="store_true",
                   help="Run parameter grid search and print best config.")
    p.add_argument("--grid_trials", type=int, default=18,
                   help="Max grid trials to evaluate when --grid_search is set.")
    p.add_argument("--grid_users", type=int, default=60,
                   help="Users per trial for grid search.")
    p.add_argument("--grid_weeks", type=int, default=10,
                   help="Weeks per trial for grid search.")
    p.add_argument("--grid_output", type=str, default="generator_grid_results.csv",
                   help="CSV output for grid-search trial metrics.")
    args = p.parse_args()

    set_generator_config(
        pair_block_rate=float(np.clip(args.pair_block_rate, 0.0, 1.0)),
        cross_overlap_base=max(0.0, float(args.cross_overlap_base)),
        session_penalty_per_exercise=max(0.0, float(args.session_penalty_per_exercise)),
        rir_noise_scale=max(0.1, float(args.rir_noise_scale)),
        floor_min_rows=max(0, int(args.floor_min_rows)),
    )

    if args.grid_search:
        run_grid_search(
            seed=args.seed,
            n_users=max(8, args.grid_users),
            n_weeks=max(2, args.grid_weeks),
            include_warmups=not args.no_warmups,
            max_trials=max(1, args.grid_trials),
            out_csv=args.grid_output,
        )

    print("=" * 60)
    print("DeepGain v3 — Empirical, MPC-Free, RIR-Based Generator")
    print("=" * 60)
    print(f"  Scientific sources:  57 papers")
    print(f"  Exercises:           {len(EXERCISE_MUSCLES)}")
    print(f"  Muscle groups:       {len(ALL_MUSCLES)}")
    print(f"  Workout templates:   {len(TEMPLATES)}")
    print(f"  Weekly schedules:    {len(SCHEDULES)}")
    print(f"  Users:               {args.num_users:,}")
    print(f"  Weeks:               {args.weeks}")
    if args.target_rows > 0:
        print(f"  Target rows:         >={args.target_rows:,}")
    print(f"  Pair block rate:     {GEN_CONFIG.pair_block_rate:.2f}")
    print(f"  Floor min/exercise:  {GEN_CONFIG.floor_min_rows:,}")
    print(f"  Effort metric:       RIR (0-5, integer)")
    print(f"  Hidden state:        NONE (pure lookup tables)")
    print()

    df = generate_dataset(
        args.num_users,
        args.weeks,
        args.seed,
        include_warmups=not args.no_warmups,
        target_rows=(args.target_rows if args.target_rows > 0 else None),
        floor_min_rows=max(0, args.floor_min_rows),
    )

    if len(df) == 0:
        print("ERROR: No data generated!")
        sys.exit(1)

    expected = {"user_id", "exercise", "weight_kg", "reps", "rir", "timestamp"}
    assert set(df.columns) == expected, f"Bad columns: {set(df.columns)}"

    if not args.quiet:
        print_diagnostics(df)
        validate_empirical_patterns(df)

    df.to_csv(args.output, index=False)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved {len(df):,} rows to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
