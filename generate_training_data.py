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
import hashlib
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    print("WARNING: PyYAML not installed — run: pip install pyyaml")


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
    "spoto_press": 0.71, "incline_bench_45": 0.68, "decline_bench": 0.74,
    "chest_press_machine": 0.76, "ohp": 0.68, "dips": 0.72,
    "dumbbell_flyes": 0.66, "skull_crusher": 0.60,
    "pendlay_row": 0.73, "seal_row": 0.70,
    "lat_pulldown": 0.68, "pull_up": 0.70, "reverse_fly": 0.55,
    "squat": 1.00, "low_bar_squat": 1.00, "high_bar_squat": 0.96,
    "deadlift": 0.85, "sumo_deadlift": 0.87, "rdl": 0.78,
    "leg_press": 1.10, "bulgarian_split_squat": 0.90,
    "leg_curl": 0.65, "leg_extension": 0.68,
    "plank": 0.62, "farmers_walk": 0.70, "leg_raises": 0.64,
    "ab_wheel": 0.62, "dead_bug": 0.60, "trx_bodysaw": 0.61,
    "suitcase_carry": 0.66, "bird_dog": 0.60,
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

EXERCISE_MUSCLES: Dict[str, Dict[str, float]] = {
    # Fallback map (used only when YAML is unavailable).
    # Keep IDs aligned with exercise_muscle_order.yaml.
    "bench_press":      {"chest": 1.00, "triceps": 0.654499, "anterior_delts": 0.203639, "lateral_delts": 0.047579, "rear_delts": 0.033305},
    "incline_bench":    {"chest": 0.70, "anterior_delts": 0.75, "triceps": 0.50},
    "close_grip_bench": {"chest": 0.65, "triceps": 0.75, "anterior_delts": 0.55},
    "spoto_press":      {"chest": 0.75, "triceps": 0.70, "anterior_delts": 0.45},
    "incline_bench_45": {"chest": 0.72, "triceps": 0.68, "anterior_delts": 0.55},
    "decline_bench":    {"chest": 0.78, "triceps": 0.72, "anterior_delts": 0.40},
    "chest_press_machine": {"chest": 0.72, "anterior_delts": 0.45, "triceps": 0.30},
    "dips":             {"chest": 1.00, "triceps": 0.876653, "anterior_delts": 0.716490, "lateral_delts": 0.275573, "abs": 0.407848},
    "ohp":              {"anterior_delts": 0.85, "triceps": 0.65, "lateral_delts": 0.45},
    "dumbbell_flyes":   {"chest": 0.82, "anterior_delts": 0.30},
    "skull_crusher":    {"triceps": 0.85, "anterior_delts": 0.25},
    "squat":            {"quads": 1.00, "hamstrings": 0.471318, "glutes": 0.420000, "erectors": 0.567194},
    "low_bar_squat":    {"adductors": 1.00, "calves": 0.729805, "erectors": 0.583398, "glutes": 0.232033},
    "high_bar_squat":   {"quads": 0.82, "glutes": 0.62, "erectors": 0.40},
    "deadlift":         {"hamstrings": 1.00, "erectors": 0.834134, "glutes": 0.821333, "quads": 0.600000},
    "sumo_deadlift":    {"quads": 1.00, "abs": 0.894167, "calves": 0.708333, "hamstrings": 0.604167, "glutes": 0.539583, "adductors": 0.479167, "erectors": 0.453334},
    "rdl":              {"hamstrings": 0.85, "glutes": 0.62, "erectors": 0.50},
    "bulgarian_split_squat": {"quads": 1.00, "glutes": 0.537209, "erectors": 0.425312, "hamstrings": 0.194614},
    "leg_press":        {"quads": 0.82, "glutes": 0.52, "hamstrings": 0.28},
    "leg_curl":         {"hamstrings": 0.85},
    "leg_extension":    {"quads": 0.85},
    "pendlay_row":      {"rear_delts": 1.00, "rhomboids": 0.640147, "erectors": 0.588446, "lats": 0.529988},
    "pull_up":          {"lats": 1.00, "biceps": 0.666667, "rhomboids": 0.500000},
    "lat_pulldown":     {"lats": 0.78, "rhomboids": 0.52, "rear_delts": 0.40, "biceps": 0.35},
    "reverse_fly":      {"rear_delts": 0.88, "lateral_delts": 0.65},
    "seal_row":         {"rear_delts": 1.00, "rhomboids": 0.713324, "lats": 0.399731, "erectors": 0.216904},
    "plank":            {"abs": 1.00},
    "farmers_walk":     {"erectors": 1.00, "abs": 0.828656},
    "leg_raises":       {"abs": 1.00, "lats": 0.184276, "quads": 0.131625, "erectors": 0.041769},
    "ab_wheel":         {"abs": 1.00},
    "dead_bug":         {"abs": 1.00},
    "trx_bodysaw":      {"abs": 1.00, "erectors": 0.021685},
    "suitcase_carry":   {"abs": 1.00, "erectors": 0.887049},
    "bird_dog":         {"glutes": 1.00, "erectors": 0.893714},
}

ALL_EXERCISES = list(EXERCISE_MUSCLES.keys())
ALL_MUSCLES = sorted(set(m for ex in EXERCISE_MUSCLES.values() for m in ex))


def _refresh_exercise_globals() -> None:
    """Refresh cached exercise/muscle lists after registry updates."""
    global ALL_EXERCISES, ALL_MUSCLES
    ALL_EXERCISES = sorted(EXERCISE_MUSCLES.keys())
    ALL_MUSCLES = sorted(set(m for ex in EXERCISE_MUSCLES.values() for m in ex))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ORDINAL MUSCLE INVOLVEMENT — Milestone 2                              ║
# ║  Loaded from exercise_muscle_order.yaml + transformed EMG CSV.         ║
# ║  Real transformed weights are the only source of numeric overlap.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Per-set MPC drop ranges sampled uniformly [Milestone 2 spec]
FATIGUE_DROP_RANGES: Dict[str, Tuple[float, float]] = {
    "primary":   (0.70, 1.00),
    "secondary": (0.30, 0.60),
    "tertiary":  (0.05, 0.20),
}

# Populated by load_exercise_yaml(); maps exercise → {muscle → tier str}
ORDINAL_MUSCLES: Dict[str, Dict[str, str]] = {}
MUSCLE_RECOVERY_REGION: Dict[str, str] = {
    "chest": "upper",
    "lats": "upper",
    "rear_delts": "upper",
    "anterior_delts": "upper",
    "lateral_delts": "upper",
    "biceps": "upper",
    "triceps": "upper",
    "forearms": "upper",
    "quads": "lower",
    "hamstrings": "lower",
    "glutes": "lower",
    "calves": "lower",
    "abs": "lower",
    "erectors": "lower",
    "adductors": "lower",
}

_DEFAULT_YAML_PATH = pathlib.Path(__file__).parent / "exercise_muscle_order.yaml"
_DEFAULT_SCALED_WEIGHTS_PATH = pathlib.Path(__file__).parent / "exercise_muscle_weights_scaled.csv"


def _load_scaled_weights(path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Load transformed EMG weights from CSV.

    This is the canonical numeric source for overlap / recovery logic.
    We intentionally avoid any tier-to-number fallback here so generator
    behavior cannot silently drift back to ordinal proxy weights.
    """
    p = pathlib.Path(path) if path else _DEFAULT_SCALED_WEIGHTS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Missing required transformed weights CSV: {p}")

    df = pd.read_csv(p)
    if "exercise_id" not in df.columns:
        raise RuntimeError(f"Invalid transformed weights CSV: missing exercise_id in {p}")

    ignored_cols = {"exercise_id", "csv_title", "source", "per_exercise_max_after_clip", "global_clip_p99"}
    muscle_cols = [c for c in df.columns if c not in ignored_cols and c not in {"upper_traps", "brachialis"}]
    weights: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        ex_id = str(row["exercise_id"])
        ex_weights = {
            m: float(np.clip(row[m], 0.0, 1.0))
            for m in muscle_cols
            if float(row[m]) > 0.0
        }
        if ex_weights:
            weights[ex_id] = ex_weights

    return weights


def load_exercise_yaml(path: Optional[str] = None) -> None:
    """Load exercise_muscle_order.yaml and populate ORDINAL_MUSCLES.

    Numeric exercise weights come strictly from exercise_muscle_weights_scaled.csv.
    YAML remains the canonical exercise registry and ordinal source.
    """
    global ORDINAL_MUSCLES
    if not _YAML_AVAILABLE:
        return
    p = pathlib.Path(path) if path else _DEFAULT_YAML_PATH
    if not p.exists():
        print(f"  WARNING: YAML not found at {p}. Using hardcoded EXERCISE_MUSCLES.")
        return

    with p.open("r", encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)

    exercises: Dict[str, Any] = data.get("exercises", {})
    ORDINAL_MUSCLES = {}
    scaled_weights = _load_scaled_weights()
    yaml_exercise_muscles: Dict[str, Dict[str, float]] = {}

    for ex_key, ex_data in exercises.items():
        tier_map: Dict[str, str] = {}
        for tier in ("primary", "secondary", "tertiary"):
            for muscle in ex_data.get(f"{tier}_muscles", []):
                tier_map[muscle] = tier
        ORDINAL_MUSCLES[ex_key] = tier_map

        if ex_key not in scaled_weights:
            raise RuntimeError(
                f"Missing transformed EMG weights for YAML exercise '{ex_key}'. "
                f"Update exercise_muscle_weights_scaled.csv before generating data."
            )
        yaml_exercise_muscles[ex_key] = scaled_weights[ex_key]

    # Keep only YAML-defined exercises so data generation cannot drift.
    EXERCISE_MUSCLES.clear()
    EXERCISE_MUSCLES.update(yaml_exercise_muscles)
    _refresh_exercise_globals()

    print(
        f"  Loaded ordinal muscles for {len(ORDINAL_MUSCLES)} exercises "
        f"from {p.name} and numeric EMG weights from {_DEFAULT_SCALED_WEIGHTS_PATH.name}"
    )


def fatigue_drop_for_muscle(exercise: str, muscle: str,
                             rng: np.random.Generator) -> float:
    """Sample a per-set MPC drop for (exercise, muscle) from its ordinal tier.

    Returns a float in [0.05, 1.0].  Falls back to secondary range when the
    (exercise, muscle) pair is not defined in ORDINAL_MUSCLES.

    Usage by train.py (Osoba B):
        drop = fatigue_drop_for_muscle("bench_press", "chest", rng)
        # drop ~ U[0.70, 1.00]
    """
    tier = ORDINAL_MUSCLES.get(exercise, {}).get(muscle)
    if tier is None:
        # Derive tier from position in EXERCISE_MUSCLES (legacy exercises)
        muscles_dict = EXERCISE_MUSCLES.get(exercise, {})
        if muscles_dict:
            ranked = sorted(muscles_dict, key=muscles_dict.get, reverse=True)
            idx = ranked.index(muscle) if muscle in ranked else len(ranked)
            tier = "primary" if idx == 0 else "secondary" if idx < 3 else "tertiary"
        else:
            tier = "secondary"
    lo, hi = FATIGUE_DROP_RANGES[tier]
    return float(rng.uniform(lo, hi))


def ordinal_set_capacity_multiplier(exercise: str, set_idx: int,
                                    rng: np.random.Generator) -> float:
    """Map ordinal per-muscle drop sampling to a bounded set capacity multiplier.

    We intentionally keep the effect moderate, because set-to-set retention,
    inter-session recovery and cross-exercise transfer already model fatigue.
    Numeric weighting by transformed EMG involvement keeps the ordinal sampler
    aligned with the same primary/secondary emphasis used elsewhere.
    """
    muscles = list(EXERCISE_MUSCLES.get(exercise, {}).keys())
    if not muscles:
        return 1.0

    weights = EXERCISE_MUSCLES.get(exercise, {})
    total_weight = sum(max(weights.get(m, 0.0), 0.0) for m in muscles) or 1.0
    normalized = {m: max(weights.get(m, 0.0), 0.0) / total_weight for m in muscles}
    sampled = {m: fatigue_drop_for_muscle(exercise, m, rng) for m in muscles}

    # Literature supports larger fatigue in more involved muscles, but not an
    # exact closed-form aggregator at exercise level, so we use a conservative
    # weighted mean + weighted-peak blend instead of equal-weight averaging.
    weighted_mean = sum(normalized[m] * sampled[m] for m in muscles)
    weighted_peak = max(sampled[m] * (0.65 + 0.35 * normalized[m]) for m in muscles)
    mean_drop = float(0.72 * weighted_mean + 0.28 * weighted_peak)

    # Increase the impact gradually with set number (first set least affected).
    set_scale = 0.05 + 0.025 * max(0, set_idx - 1)
    return float(np.clip(1.0 - set_scale * mean_drop, 0.82, 1.0))


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
    # Nearby prior exercises should affect current performance more strongly
    # than movements performed much earlier in the session.
    for prior_idx, (prior_ex, sets, avg_reps, avg_rir) in enumerate(prior_exercises):
        overlap = muscle_overlap(prior_ex, current_exercise)
        if overlap < 0.05:
            continue
        base_penalty = overlap * 0.28  # [25] ~28% at full overlap
        volume_scale = min(1.5, sets / 3.0)
        proximity_scale = 1.0 / (1.0 + 0.12 * avg_rir)
        distance = max(1, exercise_position - prior_idx)
        recency_scale = 0.82 ** (distance - 1)
        penalty += base_penalty * volume_scale * proximity_scale * recency_scale

    # Session-level accumulated fatigue [27]
    # Spreuwenberg: -32.5% after full-body session
    # Model: -4% per exercise beyond the 3rd, cap at -20%
    if exercise_position >= 3:
        session_penalty = 0.04 * (exercise_position - 2)
        penalty += min(0.20, session_penalty)

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
    sd = base_sd * distance_amplifier

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
    "bench_press":      ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "incline_bench":    ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "close_grip_bench": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "spoto_press":      ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "incline_bench_45": ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "decline_bench":    ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "chest_press_machine": ExerciseInfo("upper_compound", "upper", "machine", 5.0, 20),
    "ohp":              ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 15),
    "dips":             ExerciseInfo("upper_compound", "upper", "bodyweight", 2.5, 0),
    "dumbbell_flyes":   ExerciseInfo("isolation", "upper", "dumbbell", 2.0, 4),
    "pendlay_row":      ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "seal_row":         ExerciseInfo("upper_compound", "upper", "barbell", 2.5, 20),
    "lat_pulldown":     ExerciseInfo("upper_compound", "upper", "cable", 5.0, 20),
    "pull_up":          ExerciseInfo("upper_compound", "upper", "bodyweight", 2.5, 0),
    "squat":            ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "low_bar_squat":    ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "high_bar_squat":   ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 20),
    "deadlift":         ExerciseInfo("lower_compound", "deadlift", "barbell", 2.5, 40),
    "sumo_deadlift":    ExerciseInfo("lower_compound", "deadlift", "barbell", 2.5, 40),
    "rdl":              ExerciseInfo("lower_compound", "lower", "barbell", 2.5, 30),
    "leg_press":        ExerciseInfo("lower_compound", "lower", "machine", 5.0, 40),
    "bulgarian_split_squat": ExerciseInfo("lower_compound", "lower", "dumbbell", 2.0, 0),
    "skull_crusher":    ExerciseInfo("isolation", "upper", "barbell", 2.5, 10),
    "reverse_fly":      ExerciseInfo("isolation", "upper", "machine", 2.5, 5),
    "leg_curl":         ExerciseInfo("isolation", "lower", "machine", 2.5, 10),
    "leg_extension":    ExerciseInfo("isolation", "lower", "machine", 2.5, 10),
    "plank":            ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
    "farmers_walk":     ExerciseInfo("isolation", "lower", "dumbbell", 2.5, 0),
    "leg_raises":       ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
    "ab_wheel":         ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
    "dead_bug":         ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
    "trx_bodysaw":      ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
    "suitcase_carry":   ExerciseInfo("isolation", "lower", "dumbbell", 2.5, 0),
    "bird_dog":         ExerciseInfo("isolation", "lower", "bodyweight", 2.5, 0),
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
    split_preference: int
    lift_focus: str
    squat_style: str
    deadlift_style: str
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

    split_pref = int(rng.choice([3, 4, 5], p=[0.26, 0.54, 0.20]))
    lift_focus = str(rng.choice(
        ["balanced", "bench", "squat", "deadlift"],
        p=[0.52, 0.22, 0.16, 0.10]
    ))
    squat_style = str(rng.choice(["low_bar", "high_bar"], p=[0.72, 0.28]))
    deadlift_style = str(rng.choice(["conventional", "sumo"], p=[0.58, 0.42]))

    bench = bw * bench_mult
    squat = bw * squat_mult
    dl = squat * rng.uniform(1.05, 1.25)

    if lift_focus == "bench":
        bench *= rng.uniform(1.03, 1.09)
    elif lift_focus == "squat":
        squat *= rng.uniform(1.03, 1.08)
    elif lift_focus == "deadlift":
        dl *= rng.uniform(1.04, 1.10)

    low_bar_mult = rng.uniform(0.98, 1.03) if squat_style == "low_bar" else rng.uniform(0.92, 0.98)
    high_bar_mult = rng.uniform(0.92, 0.98) if squat_style == "low_bar" else rng.uniform(0.97, 1.01)
    sumo_mult = rng.uniform(0.98, 1.05) if deadlift_style == "sumo" else rng.uniform(0.90, 0.99)

    e1rm = {
        "bench_press": bench, "incline_bench": bench * rng.uniform(0.78, 0.85),
        "close_grip_bench": bench * rng.uniform(0.82, 0.88),
        "spoto_press": bench * rng.uniform(0.86, 0.94),
        "incline_bench_45": bench * rng.uniform(0.74, 0.82),
        "decline_bench": bench * rng.uniform(0.86, 0.94),
        "chest_press_machine": bench * rng.uniform(0.78, 0.90),
        "ohp": bench * rng.uniform(0.57, 0.67),
        "dips": bench * rng.uniform(0.55, 0.72),
        "dumbbell_flyes": bench * rng.uniform(0.24, 0.34),
        "pendlay_row": bench * rng.uniform(0.80, 0.98),
        "seal_row": bench * rng.uniform(0.68, 0.85),
        "lat_pulldown": bench * rng.uniform(0.60, 0.76),
        "pull_up": bench * rng.uniform(0.48, 0.65),
        "squat": squat,
        "low_bar_squat": squat * low_bar_mult,
        "high_bar_squat": squat * high_bar_mult,
        "deadlift": dl,
        "sumo_deadlift": dl * sumo_mult,
        "rdl": dl * rng.uniform(0.65, 0.75),
        "leg_press": squat * rng.uniform(1.30, 1.60),
        "bulgarian_split_squat": squat * rng.uniform(0.34, 0.46),
        "skull_crusher": bench * rng.uniform(0.24, 0.35),
        "reverse_fly": bw * rng.uniform(0.08, 0.16),
        "leg_curl": squat * rng.uniform(0.28, 0.40),
        "leg_extension": squat * rng.uniform(0.34, 0.50),
        "plank": bw * rng.uniform(0.28, 0.40),
        "farmers_walk": bw * rng.uniform(0.55, 0.95),
        "leg_raises": bw * rng.uniform(0.24, 0.34),
        "ab_wheel": bw * rng.uniform(0.30, 0.45),
        "dead_bug": bw * rng.uniform(0.20, 0.30),
        "trx_bodysaw": bw * rng.uniform(0.24, 0.34),
        "suitcase_carry": bw * rng.uniform(0.30, 0.58),
        "bird_dog": bw * rng.uniform(0.20, 0.30),
    }

    # Exercise exclusions (small probability - target users are still powerlifting-focused)
    excluded = []
    # Keep some exercise exclusions for realism, but avoid over-pruning the
    # deadlift family because posterior-chain transfer is one of the key
    # supervision signals we need in the dataset.
    if rng.random() < 0.03:
        excluded.append("deadlift")
    if rng.random() < 0.05:
        excluded.append("squat")
    if rng.random() < 0.04:
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
        split_preference=split_pref,
        lift_focus=lift_focus,
        squat_style=squat_style,
        deadlift_style=deadlift_style,
        excluded_exercises=excluded,
        program_switch_week=switch_week,
        stalling=stalling,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WORKOUT TEMPLATES — powerlifting-first archetypes                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Entry: (exercise, %1RM, sets, target_RIR, rest_multiplier)
WorkoutEntry = Tuple[str, float, int, int, float]

# Key ordered same-session exercise relations highlighted by Michał.
# These are not the only pairs we care about, but they are high-value transfer
# signals and deserve explicit coverage reporting in both train and validation.
KEY_SEQUENCE_PAIRS: List[Tuple[str, str]] = [
    ("bench_press", "skull_crusher"),
    ("bench_press", "ohp"),
    ("incline_bench", "dumbbell_flyes"),
    ("rdl", "leg_curl"),
    ("squat", "leg_press"),
    ("deadlift", "rdl"),
]

TEMPLATES: Dict[str, List[WorkoutEntry]] = {
    "bench_volume_day": [
        ("bench_press", 0.76, 5, 2, 1.2), ("spoto_press", 0.72, 3, 2, 1.0),
        ("incline_bench", 0.72, 3, 2, 0.9), ("dumbbell_flyes", 0.62, 3, 2, 0.6),
        ("pendlay_row", 0.74, 4, 2, 1.0), ("lat_pulldown", 0.72, 3, 2, 0.8),
        ("skull_crusher", 0.66, 3, 2, 0.6),
    ],
    "bench_intensity_day": [
        ("bench_press", 0.88, 5, 1, 1.6), ("close_grip_bench", 0.80, 3, 2, 1.2),
        ("ohp", 0.70, 3, 2, 0.9), ("decline_bench", 0.78, 3, 2, 1.0),
        ("chest_press_machine", 0.72, 2, 2, 0.7),
        ("reverse_fly", 0.65, 3, 3, 0.5),
    ],
    "bench_specialization_day": [
        ("incline_bench_45", 0.74, 4, 2, 1.0), ("close_grip_bench", 0.76, 4, 2, 1.0),
        ("dips", 0.72, 3, 2, 0.8), ("ohp", 0.72, 3, 2, 1.0),
        ("skull_crusher", 0.64, 3, 2, 0.6), ("reverse_fly", 0.62, 3, 3, 0.5),
        ("plank", 0.62, 2, 4, 0.5),
    ],
    "squat_volume_day": [
        ("squat", 0.78, 5, 2, 1.3), ("high_bar_squat", 0.72, 3, 2, 1.1),
        ("bulgarian_split_squat", 0.72, 3, 2, 0.8), ("leg_press", 0.75, 3, 2, 1.0),
        ("leg_curl", 0.68, 3, 2, 0.7),
    ],
    "squat_intensity_day": [
        ("low_bar_squat", 0.88, 5, 1, 1.7), ("squat", 0.80, 3, 2, 1.3),
        ("leg_press", 0.78, 3, 2, 1.0), ("leg_extension", 0.70, 3, 2, 0.7),
        ("leg_curl", 0.68, 3, 2, 0.7),
    ],
    # Posterior-chain ladder: deadlift -> rdl -> leg_curl.
    # This is a deliberate cross-muscle sequence to expose hamstrings/glutes
    # transfer, which M6 still struggles to infer from the older dataset.
    "deadlift_intensity_day": [
        ("deadlift", 0.87, 4, 1, 1.7), ("rdl", 0.74, 3, 2, 1.0),
        ("leg_curl", 0.68, 3, 2, 0.7), ("pendlay_row", 0.74, 4, 2, 1.0),
        ("lat_pulldown", 0.72, 3, 2, 0.8),
    ],
    "deadlift_sumo_day": [
        ("sumo_deadlift", 0.85, 4, 1, 1.6), ("rdl", 0.72, 3, 2, 1.0),
        ("leg_curl", 0.68, 3, 2, 0.7), ("seal_row", 0.72, 3, 2, 0.9),
        ("pull_up", 0.74, 3, 2, 0.9), ("farmers_walk", 0.70, 2, 2, 0.8),
    ],
    "upper_support_day": [
        ("ohp", 0.74, 4, 2, 1.0), ("pendlay_row", 0.74, 4, 2, 1.0),
        ("pull_up", 0.74, 3, 2, 0.9), ("chest_press_machine", 0.70, 3, 2, 0.8),
        ("reverse_fly", 0.65, 3, 3, 0.5), ("skull_crusher", 0.64, 3, 2, 0.6),
        ("leg_raises", 0.66, 2, 3, 0.6),
    ],
    "lower_accessory_day": [
        ("high_bar_squat", 0.74, 4, 2, 1.1), ("rdl", 0.70, 3, 2, 0.9),
        ("bulgarian_split_squat", 0.70, 3, 2, 0.8), ("leg_press", 0.72, 3, 2, 1.0),
        ("leg_extension", 0.68, 3, 2, 0.7), ("leg_curl", 0.68, 3, 2, 0.7),
        ("ab_wheel", 0.64, 2, 3, 0.6),
    ],
    "powerbuilding_upper_day": [
        ("incline_bench", 0.74, 4, 2, 0.9), ("dumbbell_flyes", 0.62, 3, 2, 0.6),
        ("ohp", 0.72, 4, 2, 0.9), ("pendlay_row", 0.72, 4, 2, 1.0),
        ("lat_pulldown", 0.70, 3, 2, 0.8),
        ("dips", 0.70, 3, 2, 0.8), ("reverse_fly", 0.62, 3, 3, 0.5),
        ("trx_bodysaw", 0.62, 2, 3, 0.6),
    ],
    "powerbuilding_lower_day": [
        ("high_bar_squat", 0.76, 4, 2, 1.2), ("sumo_deadlift", 0.76, 3, 2, 1.3),
        ("rdl", 0.68, 2, 3, 0.9), ("leg_press", 0.76, 3, 2, 1.0),
        ("bulgarian_split_squat", 0.70, 3, 2, 0.8), ("leg_curl", 0.68, 3, 2, 0.7),
        ("suitcase_carry", 0.64, 2, 3, 0.7),
    ],
    "peak_bench_day": [
        ("bench_press", 0.92, 3, 0, 2.0), ("spoto_press", 0.82, 2, 1, 1.3),
        ("close_grip_bench", 0.78, 2, 2, 1.1),
    ],
    "peak_squat_day": [
        ("low_bar_squat", 0.91, 3, 0, 2.0), ("squat", 0.82, 2, 1, 1.4),
        ("leg_press", 0.70, 2, 3, 0.9),
    ],
    "peak_deadlift_day": [
        ("deadlift", 0.91, 2, 0, 2.0), ("rdl", 0.68, 2, 3, 1.0),
        ("leg_curl", 0.60, 2, 4, 0.7), ("pendlay_row", 0.70, 2, 2, 0.9),
    ],
    "deload_upper_pl": [
        ("bench_press", 0.62, 3, 4, 1.0), ("pendlay_row", 0.60, 3, 4, 1.0),
        ("ohp", 0.58, 2, 4, 0.8), ("reverse_fly", 0.55, 2, 5, 0.5),
        ("dead_bug", 0.52, 2, 5, 0.5),
    ],
    "deload_lower_pl": [
        ("squat", 0.62, 3, 4, 1.2), ("rdl", 0.58, 2, 4, 1.0),
        ("leg_press", 0.58, 2, 4, 0.9), ("leg_curl", 0.55, 2, 5, 0.7),
        ("bird_dog", 0.50, 2, 5, 0.5),
    ],
    "deload_full_pl": [
        ("bench_press", 0.60, 2, 4, 1.0), ("squat", 0.60, 2, 4, 1.2),
        ("deadlift", 0.58, 1, 4, 1.6),
    ],
    "core_stability_day": [
        ("plank", 0.68, 3, 3, 0.5), ("ab_wheel", 0.66, 3, 2, 0.6),
        ("trx_bodysaw", 0.64, 3, 3, 0.6), ("dead_bug", 0.58, 2, 4, 0.5),
        ("bird_dog", 0.56, 2, 4, 0.5),
    ],
    "loaded_core_day": [
        ("farmers_walk", 0.72, 3, 2, 0.8), ("suitcase_carry", 0.66, 3, 2, 0.8),
        ("leg_raises", 0.68, 3, 2, 0.6),
    ],
    # Calibration template: expose Spoto earlier with mixed non-press context.
    "spoto_calibration_day": [
        ("spoto_press", 0.78, 4, 2, 1.1), ("pendlay_row", 0.72, 3, 2, 1.0),
        ("lat_pulldown", 0.70, 3, 2, 0.8), ("reverse_fly", 0.62, 2, 3, 0.5),
        ("skull_crusher", 0.62, 2, 2, 0.6),
    ],
    # Calibration template: make machine chest appear early in-session.
    "machine_chest_calibration_day": [
        ("chest_press_machine", 0.72, 4, 2, 0.9), ("seal_row", 0.70, 3, 2, 0.9),
        ("pull_up", 0.72, 3, 2, 0.9), ("ohp", 0.68, 2, 2, 0.9),
        ("plank", 0.60, 2, 4, 0.5),
    ],
}


@dataclass(frozen=True)
class Schedule:
    name: str
    days: Tuple[Optional[str], ...]
    deload_days: Tuple[Optional[str], ...]

SCHEDULES: List[Schedule] = [
    Schedule("PL_3_BASE", days=(
        "squat_volume_day", None, "bench_volume_day", None, "deadlift_intensity_day", None, None),
        deload_days=("deload_lower_pl", None, "deload_upper_pl", None, "deload_full_pl", None, None)),
    Schedule("PL_4_CLASSIC", days=(
        "bench_volume_day", "squat_intensity_day", None, "bench_intensity_day", "deadlift_intensity_day", None, None),
        deload_days=("deload_upper_pl", "deload_lower_pl", None, "deload_upper_pl", "deload_lower_pl", None, None)),
    Schedule("PL_4_BENCH", days=(
        "bench_intensity_day", "squat_volume_day", None, "bench_specialization_day", "deadlift_intensity_day", None, None),
        deload_days=("deload_upper_pl", "deload_lower_pl", None, "deload_upper_pl", "deload_lower_pl", None, None)),
    Schedule("PL_4_DEADLIFT", days=(
        "bench_volume_day", "squat_volume_day", None, "deadlift_sumo_day", "upper_support_day", None, None),
        deload_days=("deload_upper_pl", "deload_lower_pl", None, "deload_lower_pl", "deload_upper_pl", None, None)),
    Schedule("PL_4_OFFSEASON", days=(
        "powerbuilding_upper_day", "powerbuilding_lower_day", None, "bench_volume_day", "lower_accessory_day", "core_stability_day", None),
        deload_days=("deload_upper_pl", "deload_lower_pl", None, "deload_upper_pl", "deload_lower_pl", None, None)),
    Schedule("PL_5_POWERBUILDING", days=(
        "bench_volume_day", "squat_volume_day", "upper_support_day", "deadlift_intensity_day", "bench_specialization_day", "loaded_core_day", None),
        deload_days=("deload_upper_pl", "deload_lower_pl", "deload_upper_pl", "deload_lower_pl", "deload_upper_pl", None, None)),
    Schedule("PL_3_PEAK", days=(
        "peak_squat_day", None, "peak_bench_day", None, "peak_deadlift_day", None, None),
        deload_days=("deload_lower_pl", None, "deload_upper_pl", None, "deload_full_pl", None, None)),
]


def choose_schedule_pair(rng: np.random.Generator, user: UserProfile,
                         schedules: List[Schedule]) -> Tuple[Schedule, Schedule]:
    """Choose a primary and fallback schedule biased toward powerlifting behavior."""
    by_name = {s.name: s for s in schedules}

    if user.split_preference == 5:
        primary_name = "PL_5_POWERBUILDING"
        alt_name = "PL_4_OFFSEASON"
    elif user.split_preference == 3:
        primary_name = "PL_3_PEAK" if user.training_level in ("advanced", "elite") and rng.random() < 0.35 else "PL_3_BASE"
        alt_name = "PL_3_BASE" if primary_name == "PL_3_PEAK" else "PL_3_PEAK"
    else:
        if user.lift_focus == "bench":
            primary_name, alt_name = "PL_4_BENCH", "PL_4_CLASSIC"
        elif user.lift_focus == "deadlift":
            primary_name, alt_name = "PL_4_DEADLIFT", "PL_4_CLASSIC"
        elif user.lift_focus == "squat":
            primary_name, alt_name = "PL_4_CLASSIC", "PL_4_OFFSEASON"
        else:
            primary_name, alt_name = (
                ("PL_4_OFFSEASON", "PL_4_CLASSIC")
                if rng.random() < 0.35 else
                ("PL_4_CLASSIC", "PL_4_OFFSEASON")
            )

    return by_name[primary_name], by_name[alt_name]

# ── MINI TEMPLATES (MVP: bench_press, squat, deadlift only) ────────────────
# Used by --mini mode.  3-day upper/lower/posterior split.
MINI_TEMPLATES: Dict[str, List[WorkoutEntry]] = {
    "mini_bench_vol":   [("bench_press", 0.75, 4, 2, 1.1)],
    "mini_bench_str":   [("bench_press", 0.87, 5, 1, 1.5)],
    "mini_squat":       [("squat",       0.78, 4, 2, 1.3)],
    "mini_deadlift":    [("deadlift",    0.80, 3, 2, 1.5)],
    "mini_deload_upper":[("bench_press", 0.62, 3, 4, 1.0)],
    "mini_deload_lower":[("squat",       0.62, 3, 4, 1.2)],
}
MINI_SCHEDULE = Schedule(
    name="MINI_3",
    days=(
        "mini_bench_vol", "mini_squat", None,
        "mini_bench_str", "mini_deadlift", None, None,
    ),
    deload_days=(
        "mini_deload_upper", "mini_deload_lower", None,
        None, None, None, None,
    ),
)


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

def _weighted_recovery_multiplier(muscles: Dict[str, float],
                                  user: UserProfile,
                                  current_time: datetime) -> float:
    """Exercise-level recovery from weighted involved-muscle recovery.

    Recovery timing is literature-informed [32]-[39], while the exercise-level
    aggregation below is an explicit engineering heuristic: we blend a weighted
    mean with a weighted-worst component so a tiny tertiary muscle does not
    fully bottleneck the entire exercise, while primary muscles still dominate.
    """
    if not muscles:
        return 1.0

    total = sum(max(v, 0.0) for v in muscles.values()) or 1.0
    normalized = {m: max(v, 0.0) / total for m, v in muscles.items()}
    recoveries: List[Tuple[float, float]] = []

    for muscle, weight in normalized.items():
        last_time = user.muscle_last_trained.get(muscle, datetime(2024, 1, 1))
        hours_since = (current_time - last_time).total_seconds() / 3600
        severity = user.muscle_last_severity.get(muscle, "moderate")
        region = user.muscle_last_region.get(muscle, MUSCLE_RECOVERY_REGION.get(muscle, "upper"))
        rec = session_recovery_multiplier(hours_since, region, severity)
        recoveries.append((weight, rec))

    weighted_mean = sum(w * rec for w, rec in recoveries)
    weighted_worst = min(rec / max(w, 1e-6) ** 0.15 for w, rec in recoveries)
    return float(np.clip(0.72 * weighted_mean + 0.28 * weighted_worst, 0.55, 1.0))


def get_recovery_for_exercise(user: UserProfile, exercise: str,
                              current_time: datetime) -> float:
    """Lookup inter-session recovery multiplier. [32]-[39]"""
    muscles = EXERCISE_MUSCLES.get(exercise, {})
    return _weighted_recovery_multiplier(muscles, user, current_time)


def update_muscle_history(user: UserProfile, exercise: str,
                          timestamp: datetime, total_sets: int,
                          avg_rir: float):
    """Record training for inter-session recovery tracking.

    Severity is scaled by transformed involvement so compounds do not assign the
    same recovery debt to every stabilizer. Region mapping is anatomy-driven,
    with a deadlift-specific profile only for posterior-chain muscles.
    """
    info = EXERCISE_INFO[exercise]
    muscles = EXERCISE_MUSCLES.get(exercise, {})
    for muscle, involvement in muscles.items():
        # Highly involved muscles should accumulate more recovery debt than
        # weakly involved stabilizers from the same exercise.
        stress = total_sets * (1.35 - 0.12 * avg_rir) * involvement
        if stress >= 5.5:
            severity = "extreme"
        elif stress >= 2.8:
            severity = "hard"
        else:
            severity = "moderate"
        existing = user.muscle_last_trained.get(muscle, datetime(2024, 1, 1))
        if timestamp > existing:
            user.muscle_last_trained[muscle] = timestamp
            user.muscle_last_severity[muscle] = severity
            if info.body_region == "deadlift" and muscle in {"erectors", "glutes", "hamstrings"}:
                user.muscle_last_region[muscle] = "deadlift"
            else:
                user.muscle_last_region[muscle] = MUSCLE_RECOVERY_REGION.get(muscle, info.body_region)


def simulate_workout(rng: np.random.Generator, user: UserProfile,
                     template_name: str, start_time: datetime,
                     week_in_meso: int,
                     include_warmups: bool = True,
                     templates: Optional[Dict[str, List[WorkoutEntry]]] = None,
                     ) -> List[Dict]:
    """Generate all sets for one workout session."""
    _templates = templates if templates is not None else TEMPLATES

    # Controlled context diversification for chest auxiliaries.
    # Keeps powerlifting-first templates, but adds cleaner contexts where
    # Spoto/machine chest are not always preceded by another press.
    if template_name == "bench_volume_day" and rng.random() < 0.12:
        template_name = "spoto_calibration_day"
    elif template_name in {"upper_support_day", "bench_intensity_day"} and rng.random() < 0.12:
        template_name = "machine_chest_calibration_day"

    template = _templates[template_name]
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
            ordinal_mult = ordinal_set_capacity_multiplier(exercise, set_idx, rng)
            set1_max_for_set = set1_max * ordinal_mult
            row = simulate_set(
                rng, user, exercise, weight, target_rir,
                set_idx, actual_rest, set1_max_for_set, t, pct_effective
            )
            if row is None:
                weight = round_weight(weight * 0.90, info.weight_increment)
                weight = max(weight, info.min_weight_kg)
                row = simulate_set(
                    rng, user, exercise, weight, target_rir,
                    set_idx, actual_rest, set1_max_for_set, t, pct_effective
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
                     include_warmups: bool = True,
                     templates: Optional[Dict[str, List[WorkoutEntry]]] = None,
                     fixed_schedule: Optional[Schedule] = None,
                     ) -> List[Dict]:
    """Generate n_weeks of training for one user. [53]"""
    _templates = templates if templates is not None else TEMPLATES
    _schedules = SCHEDULES

    if fixed_schedule is not None:
        schedule = fixed_schedule
        alt_schedule = fixed_schedule
    else:
        schedule, alt_schedule = choose_schedule_pair(rng, user, _schedules)

    all_rows: List[Dict] = []
    current_date = start_date

    for week_idx in range(n_weeks):
        week_in_meso = week_idx % 4
        is_deload = (week_in_meso == 3)

        # Switch programs if applicable
        active_schedule = schedule
        if fixed_schedule is None and user.program_switch_week and week_idx >= user.program_switch_week:
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
                rng, user, template, session_start, week_in_meso,
                include_warmups, _templates,
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

def _user_in_val(user_id: str, val_ratio: float, seed: int) -> bool:
    """Deterministic train/val split by user_id hash. No data leakage.

    Uses MD5 of (seed, user_id) so assignment is stable across runs
    regardless of generation order.
    """
    digest = hashlib.md5(f"{seed}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    return bucket < int(val_ratio * 10_000)


def generate_dataset(
    n_users: int = 208,
    n_weeks: int = 65,
    seed: int = 42,
    include_warmups: bool = True,
    yaml_path: Optional[str] = None,
    mini: bool = False,
    val_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full synthetic dataset.

    Returns:
        (train_df, val_df) split at user level (no leakage).
        mini=True: 5 users × 4 weeks, only YAML-defined exercises.
    """
    # Load YAML (idempotent if already loaded)
    load_exercise_yaml(yaml_path)

    if mini:
        n_users = min(n_users, 5)
        n_weeks = min(n_weeks, 4)

    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)
    all_rows: List[Dict] = []
    user_profiles: Dict[str, Dict[str, str]] = {}
    report_every = max(1, n_users // 20)

    templates = MINI_TEMPLATES if mini else None
    fixed_schedule = MINI_SCHEDULE if mini else None

    for i in range(n_users):
        if i % report_every == 0:
            print(f"\r  Users: {i:,}/{n_users:,} ({i/n_users*100:.0f}%)",
                  end="", flush=True)
        user_id = f"user_{i:05d}"
        user = generate_user(rng, user_id)
        user_profiles[user_id] = {
            "training_level": user.training_level,
            "sex": user.sex,
            "split_preference": str(user.split_preference),
        }
        rows = simulate_program(
            rng, user, n_weeks, start_date, include_warmups,
            templates=templates, fixed_schedule=fixed_schedule,
        )
        all_rows.extend(rows)

    print(f"\r  Users: {n_users:,}/{n_users:,} (100%)    ")

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        empty = pd.DataFrame(columns=["user_id", "exercise", "weight_kg", "reps", "rir", "timestamp"])
        return empty, empty

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_df, val_df = _split_dataset_by_users_with_sequence_coverage(
        df=df,
        val_ratio=val_ratio,
        seed=seed,
        user_profiles=user_profiles,
    )
    return train_df, val_df


def _ordered_unique_exercises(exercises: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for exercise in exercises:
        if exercise not in seen:
            seen.add(exercise)
            ordered.append(exercise)
    return ordered


def _ordered_session_pairs(exercises: List[str]) -> List[Tuple[str, str]]:
    ordered = _ordered_unique_exercises(exercises)
    return [
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]


def _build_sequence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build same-session ordered exercise-pair coverage statistics.

    We intentionally track training-sensible ordered pairs within a session
    rather than every arbitrary cross-session combination.
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=["src", "dst", "session_count", "user_count", "sessions", "users"]
        )

    dc = df[["user_id", "exercise", "timestamp"]].copy()
    dc["timestamp_dt"] = pd.to_datetime(dc["timestamp"], format="ISO8601")
    dc = dc.sort_values(["user_id", "timestamp_dt"]).reset_index(drop=True)
    dc["prev_timestamp"] = dc.groupby("user_id")["timestamp_dt"].shift(1)
    dc["new_session"] = (
        dc["prev_timestamp"].isna()
        | ((dc["timestamp_dt"] - dc["prev_timestamp"]).dt.total_seconds() > 4 * 3600)
    )
    dc["session_index"] = dc.groupby("user_id")["new_session"].cumsum().astype(int)
    dc["session_key"] = (
        dc["timestamp_dt"].dt.strftime("%Y-%m-%d")
        + "#"
        + dc["session_index"].astype(str)
    )

    session_rows: List[Dict[str, Any]] = []
    for (user_id, session_key), group in dc.groupby(["user_id", "session_key"], sort=False):
        ordered = _ordered_session_pairs(group.sort_values("timestamp")["exercise"].tolist())
        if not ordered:
            continue
        for src, dst in ordered:
            session_rows.append(
                {
                    "user_id": user_id,
                    "session_key": session_key,
                    "src": src,
                    "dst": dst,
                }
            )

    if not session_rows:
        return pd.DataFrame(
            columns=["src", "dst", "session_count", "user_count", "sessions", "users"]
        )

    pair_df = pd.DataFrame(session_rows)
    stats = (
        pair_df.groupby(["src", "dst"])
        .agg(
            session_count=("session_key", "count"),
            user_count=("user_id", "nunique"),
            sessions=("session_key", lambda s: tuple(sorted(set(s)))),
            users=("user_id", lambda s: tuple(sorted(set(s)))),
        )
        .reset_index()
    )
    return stats.sort_values(["user_count", "session_count", "src", "dst"], ascending=[False, False, True, True])


def _build_user_sequence_map(df: pd.DataFrame) -> Dict[str, set[Tuple[str, str]]]:
    stats = _build_sequence_stats(df)
    user_map: Dict[str, set[Tuple[str, str]]] = {
        uid: set() for uid in sorted(df["user_id"].unique())
    }
    if len(stats) == 0:
        return user_map
    for row in stats.itertuples(index=False):
        pair = (row.src, row.dst)
        for user_id in row.users:
            user_map.setdefault(user_id, set()).add(pair)
    return user_map


def _initial_stratified_val_users(
    user_ids: List[str],
    target_val: int,
    seed: int,
    user_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> Set[str]:
    """Build initial validation users with lightweight profile stratification.

    Buckets are based on training level, sex and split preference so validation
    receives better representation of user archetypes before sequence repair.
    """
    if not user_ids:
        return set()

    if user_profiles is None:
        return {uid for uid in user_ids if _user_in_val(uid, target_val / max(len(user_ids), 1), seed)}

    ratio = target_val / max(len(user_ids), 1)
    by_bucket: Dict[Tuple[str, str, str], List[str]] = {}
    for uid in user_ids:
        p = user_profiles.get(uid, {})
        key = (
            p.get("training_level", "unknown"),
            p.get("sex", "unknown"),
            p.get("split_preference", "unknown"),
        )
        by_bucket.setdefault(key, []).append(uid)

    val_users: Set[str] = set()
    for members in by_bucket.values():
        members = sorted(members)
        target_bucket = int(round(len(members) * ratio))
        if len(members) >= 3:
            target_bucket = max(1, target_bucket)
        target_bucket = min(target_bucket, len(members))
        seeded = [uid for uid in members if _user_in_val(uid, ratio, seed)]
        picks = sorted(seeded)[:target_bucket]
        if len(picks) < target_bucket:
            for uid in members:
                if uid in picks:
                    continue
                picks.append(uid)
                if len(picks) >= target_bucket:
                    break
        val_users.update(picks)

    if len(val_users) > target_val:
        extras = sorted(val_users)
        for uid in extras[target_val:]:
            val_users.remove(uid)
    while len(val_users) < target_val:
        for uid in user_ids:
            if uid not in val_users:
                val_users.add(uid)
                if len(val_users) >= target_val:
                    break
    return val_users


def _split_dataset_by_users_with_sequence_coverage(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
    user_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """User-level split with sequence coverage repair.

    Base assignment is deterministic hash split. We then repair the split so
    every ordered same-session pair observed for at least two users appears in
    both train and val when feasible.
    """
    user_ids = sorted(df["user_id"].unique())
    if not user_ids:
        empty = df.copy()
        return empty, empty

    target_val = max(1, int(round(len(user_ids) * val_ratio)))
    user_pair_map = _build_user_sequence_map(df)

    all_pair_users: Dict[Tuple[str, str], set[str]] = {}
    for user_id, pairs in user_pair_map.items():
        for pair in pairs:
            all_pair_users.setdefault(pair, set()).add(user_id)

    required_pairs = {
        pair for pair, users in all_pair_users.items()
        if len(users) >= 2
    }

    val_users = _initial_stratified_val_users(
        user_ids=user_ids,
        target_val=target_val,
        seed=seed,
        user_profiles=user_profiles,
    )

    if len(val_users) == 0:
        val_users.add(user_ids[0])
    while len(val_users) > target_val:
        removable = [
            uid for uid in sorted(val_users)
            if len(user_pair_map.get(uid, set()) & required_pairs) == 0
        ]
        if not removable:
            break
        val_users.remove(removable[0])
    while len(val_users) < target_val:
        candidates = [
            uid for uid in user_ids
            if uid not in val_users
        ]
        if not candidates:
            break
        candidates.sort(key=lambda uid: len(user_pair_map.get(uid, set()) & required_pairs), reverse=True)
        val_users.add(candidates[0])

    def pair_side_counts(current_val_users: set[str]) -> Dict[Tuple[str, str], Tuple[int, int]]:
        counts: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for pair, users in all_pair_users.items():
            val_count = sum(1 for uid in users if uid in current_val_users)
            train_count = len(users) - val_count
            counts[pair] = (train_count, val_count)
        return counts

    # Greedy repair: if a required pair is missing in val or train, move a user
    # carrying it while trying to preserve the requested split size.
    for _ in range(len(user_ids) * 4):
        repaired = False
        counts = pair_side_counts(val_users)
        missing_in_val = [pair for pair in required_pairs if counts[pair][1] == 0]
        missing_in_train = [pair for pair in required_pairs if counts[pair][0] == 0]
        if not missing_in_val and not missing_in_train:
            break

        for pair in missing_in_val:
            candidates = sorted(all_pair_users[pair] - val_users)
            if not candidates:
                continue
            candidates.sort(
                key=lambda uid: (
                    -len(user_pair_map.get(uid, set()) & set(missing_in_val)),
                    -len(user_pair_map.get(uid, set()) & required_pairs),
                    uid,
                )
            )
            val_users.add(candidates[0])
            repaired = True
        counts = pair_side_counts(val_users)
        for pair in missing_in_train:
            candidates = sorted(all_pair_users[pair] & val_users)
            if not candidates:
                continue
            safe_candidates = [
                uid for uid in candidates
                if all(counts[other_pair][1] > 1 for other_pair in user_pair_map.get(uid, set()) & required_pairs)
            ]
            pool = safe_candidates or candidates
            pool.sort(
                key=lambda uid: (
                    len(user_pair_map.get(uid, set()) & required_pairs),
                    uid,
                )
            )
            val_users.remove(pool[0])
            repaired = True

        while len(val_users) > target_val:
            counts = pair_side_counts(val_users)
            removable = [
                uid for uid in sorted(val_users)
                if all(counts[pair][1] > 1 for pair in user_pair_map.get(uid, set()) & required_pairs)
            ]
            if not removable:
                break
            removable.sort(key=lambda uid: len(user_pair_map.get(uid, set()) & required_pairs))
            val_users.remove(removable[0])
            repaired = True

        while len(val_users) < target_val:
            counts = pair_side_counts(val_users)
            candidates = [
                uid for uid in user_ids
                if uid not in val_users
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda uid: (
                    -sum(1 for pair in user_pair_map.get(uid, set()) & required_pairs if counts[pair][1] == 0),
                    -len(user_pair_map.get(uid, set()) & required_pairs),
                    uid,
                )
            )
            val_users.add(candidates[0])
            repaired = True

        if not repaired:
            break

    # Final one-sided cleanup for any remaining feasible pairs. This prefers
    # correctness of sequence coverage over hitting the exact target ratio.
    final_counts = pair_side_counts(val_users)
    for pair in sorted(required_pairs):
        train_count, val_count = final_counts[pair]
        if val_count == 0 and train_count >= 2:
            candidates = sorted(all_pair_users[pair] - val_users)
            if candidates:
                val_users.add(candidates[0])
                final_counts = pair_side_counts(val_users)
        elif train_count == 0 and val_count >= 2:
            candidates = sorted(all_pair_users[pair] & val_users)
            if candidates:
                val_users.remove(candidates[0])
                final_counts = pair_side_counts(val_users)

    val_mask = df["user_id"].isin(val_users)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    return train_df, val_df


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


def write_dataset_report(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         output_path: str, seed: int,
                         n_users: int, n_weeks: int,
                         val_ratio: float) -> None:
    """Write a compact markdown report for coverage handoff."""
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    exercise_order = sorted(ORDINAL_MUSCLES.keys()) if ORDINAL_MUSCLES else sorted(all_df["exercise"].unique())

    train_counts = train_df.groupby("exercise").size().to_dict()
    val_counts = val_df.groupby("exercise").size().to_dict()
    total_counts = all_df.groupby("exercise").size().to_dict()
    train_seq = _build_sequence_stats(train_df)
    val_seq = _build_sequence_stats(val_df)
    all_seq = _build_sequence_stats(all_df)
    train_seq_map = {
        (row.src, row.dst): row for row in train_seq.itertuples(index=False)
    }
    val_seq_map = {
        (row.src, row.dst): row for row in val_seq.itertuples(index=False)
    }
    all_seq_map = {
        (row.src, row.dst): row for row in all_seq.itertuples(index=False)
    }

    rir_series = all_df["rir"]
    lines: List[str] = []
    lines.append("# Dataset Report")
    lines.append("")
    lines.append("## Generation")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Users requested: {n_users}")
    lines.append(f"- Weeks requested: {n_weeks}")
    lines.append(f"- Validation ratio (target): {val_ratio:.0%}")
    lines.append("")
    lines.append("## Size")
    lines.append(f"- Train rows: {len(train_df):,}")
    lines.append(f"- Val rows: {len(val_df):,}")
    lines.append(f"- Total rows: {len(all_df):,}")
    lines.append(f"- Train users: {train_df['user_id'].nunique()}")
    lines.append(f"- Val users: {val_df['user_id'].nunique()}")
    lines.append(f"- Unique exercises: {all_df['exercise'].nunique()}")
    lines.append("")
    lines.append("## Quality")
    lines.append(f"- RIR mean: {rir_series.mean():.2f}")
    lines.append(f"- RIR min/max: {int(rir_series.min())}/{int(rir_series.max())}")
    lines.append(f"- Reps mean: {all_df['reps'].mean():.2f}")
    lines.append(f"- Weight mean (kg): {all_df['weight_kg'].mean():.2f}")
    lines.append("")
    lines.append("## Exercise Coverage")
    lines.append("| exercise_id | train_sets | val_sets | total_sets |")
    lines.append("|---|---:|---:|---:|")
    for ex in exercise_order:
        t = int(train_counts.get(ex, 0))
        v = int(val_counts.get(ex, 0))
        tot = int(total_counts.get(ex, 0))
        lines.append(f"| {ex} | {t} | {v} | {tot} |")

    missing_total = [ex for ex in exercise_order if total_counts.get(ex, 0) == 0]
    lines.append("")
    lines.append("## Missing Coverage")
    if missing_total:
        for ex in missing_total:
            lines.append(f"- {ex}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Key Sequence Coverage")
    lines.append("| sequence | train_sessions | val_sessions | total_sessions | train_users | val_users | total_users |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair in KEY_SEQUENCE_PAIRS:
        tr = train_seq_map.get(pair)
        vr = val_seq_map.get(pair)
        ar = all_seq_map.get(pair)
        lines.append(
            f"| {pair[0]} -> {pair[1]} | "
            f"{0 if tr is None else int(tr.session_count)} | "
            f"{0 if vr is None else int(vr.session_count)} | "
            f"{0 if ar is None else int(ar.session_count)} | "
            f"{0 if tr is None else int(tr.user_count)} | "
            f"{0 if vr is None else int(vr.user_count)} | "
            f"{0 if ar is None else int(ar.user_count)} |"
        )

    lines.append("")
    lines.append("## Common Sequence Coverage")
    lines.append("| sequence | train_users | val_users | total_users | train_sessions | val_sessions | total_sessions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    common_pairs = [
        row for row in all_seq.itertuples(index=False)
        if int(row.user_count) >= 2
    ]
    for row in common_pairs[:40]:
        pair = (row.src, row.dst)
        tr = train_seq_map.get(pair)
        vr = val_seq_map.get(pair)
        lines.append(
            f"| {row.src} -> {row.dst} | "
            f"{0 if tr is None else int(tr.user_count)} | "
            f"{0 if vr is None else int(vr.user_count)} | "
            f"{int(row.user_count)} | "
            f"{0 if tr is None else int(tr.session_count)} | "
            f"{0 if vr is None else int(vr.session_count)} | "
            f"{int(row.session_count)} |"
        )

    rare_pairs = [
        row for row in all_seq.itertuples(index=False)
        if int(row.user_count) < 2
    ]
    lines.append("")
    lines.append("## Rare Sequences (<2 users)")
    if rare_pairs:
        for row in rare_pairs[:40]:
            lines.append(
                f"- {row.src} -> {row.dst}: {int(row.user_count)} user, {int(row.session_count)} session(s)"
            )
        if len(rare_pairs) > 40:
            lines.append(f"- ... plus {len(rare_pairs) - 40} more")
    else:
        lines.append("- None")

    missing_bidirectional = []
    for row in common_pairs:
        pair = (row.src, row.dst)
        tr = train_seq_map.get(pair)
        vr = val_seq_map.get(pair)
        if tr is None or vr is None:
            missing_bidirectional.append(pair)
    lines.append("")
    lines.append("## Split Sequence Audit")
    if missing_bidirectional:
        for src, dst in missing_bidirectional:
            lines.append(f"- Missing on one side despite >=2 users overall: {src} -> {dst}")
    else:
        lines.append("- All same-session ordered sequences with >=2 users are present in both train and val.")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    p = argparse.ArgumentParser(
        description="DeepGain v3 — Empirical, MPC-Free, RIR-Based Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Mini (MVP, 3 exercises):
      python generate_training_data.py --mini
  Quick test:
      python generate_training_data.py --num_users 10 --weeks 4
  Full year + split:
      python generate_training_data.py --num_users 208 --weeks 65
        """)
    p.add_argument("--num_users",    type=int,   default=208)
    p.add_argument("--weeks",        type=int,   default=65)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       type=str,   default="training_data.csv",
                   help="Legacy single-file output (ignored when --split is set).")
    p.add_argument("--train_output", type=str,   default="training_data_train.csv")
    p.add_argument("--val_output",   type=str,   default="training_data_val.csv")
    p.add_argument("--val_ratio",    type=float, default=0.20,
                   help="Fraction of users in val set (default 0.20).")
    p.add_argument("--yaml",         type=str,   default=None,
                   help="Path to exercise_muscle_order.yaml "
                        "(default: same directory as this script).")
    p.add_argument("--mini",         action="store_true",
                   help="Mini-dataset MVP: 5 users × 4 weeks, YAML exercises only. "
                        "Useful for Osoba B early pipeline test.")
    p.add_argument("--no_warmups",   action="store_true")
    p.add_argument("--quiet",        action="store_true")
    p.add_argument("--split",        action="store_true",
                   help="Write separate train/val CSV files (default when --mini).")
    p.add_argument("--report_output", type=str, default="dataset_report.md",
                   help="Path for autogenerated markdown coverage report.")
    p.add_argument("--no_report", action="store_true",
                   help="Disable autogenerated markdown coverage report.")
    args = p.parse_args()

    split = args.split or args.mini

    print("=" * 60)
    print("DeepGain v3 — Empirical, MPC-Free, RIR-Based Generator")
    if args.mini:
        print("  *** MINI / MVP MODE (3 exercises, 5 users, 4 weeks) ***")
    print("=" * 60)
    print(f"  Users:        {min(args.num_users, 5) if args.mini else args.num_users:,}")
    print(f"  Weeks:        {min(args.weeks, 4) if args.mini else args.weeks}")
    print(f"  Val ratio:    {args.val_ratio:.0%}")
    print(f"  Output mode:  {'train+val CSV' if split else 'single CSV'}")
    print()

    train_df, val_df = generate_dataset(
        n_users=args.num_users,
        n_weeks=args.weeks,
        seed=args.seed,
        include_warmups=not args.no_warmups,
        yaml_path=args.yaml,
        mini=args.mini,
        val_ratio=args.val_ratio,
    )

    if len(train_df) == 0:
        print("ERROR: No data generated!")
        sys.exit(1)

    expected = {"user_id", "exercise", "weight_kg", "reps", "rir", "timestamp"}
    assert set(train_df.columns) == expected, f"Bad columns: {set(train_df.columns)}"

    df_all = pd.concat([train_df, val_df]).sort_values(["user_id", "timestamp"])

    if not args.quiet and not args.mini:
        print_diagnostics(df_all)
        validate_empirical_patterns(df_all)
    elif args.mini and not args.quiet:
        _print_mini_diagnostics(train_df, val_df)

    if split:
        train_df.to_csv(args.train_output, index=False)
        val_df.to_csv(args.val_output, index=False)
        t_mb = os.path.getsize(args.train_output) / 1024 / 1024
        v_mb = os.path.getsize(args.val_output)   / 1024 / 1024
        print(f"\nSaved train: {len(train_df):,} rows -> {args.train_output} ({t_mb:.2f} MB)")
        print(f"Saved val:   {len(val_df):,} rows -> {args.val_output} ({v_mb:.2f} MB)")
        print(f"Train users: {train_df['user_id'].nunique()}  "
              f"Val users: {val_df['user_id'].nunique()}")
    else:
        df_all.to_csv(args.output, index=False)
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"\nSaved {len(df_all):,} rows to {args.output} ({size_mb:.1f} MB)")

    if not args.no_report:
        write_dataset_report(
            train_df=train_df,
            val_df=val_df,
            output_path=args.report_output,
            seed=args.seed,
            n_users=min(args.num_users, 5) if args.mini else args.num_users,
            n_weeks=min(args.weeks, 4) if args.mini else args.weeks,
            val_ratio=args.val_ratio,
        )
        print(f"Saved report: {args.report_output}")


def _print_mini_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Compact diagnostic for --mini mode."""
    df = pd.concat([train_df, val_df])
    sep = "-" * 50
    print(f"\n{sep}")
    print("MINI-DATASET DIAGNOSTIC")
    print(sep)
    print(f"  Total sets:  {len(df):,}   "
          f"(train {len(train_df):,} / val {len(val_df):,})")
    print(f"  Users:       {df['user_id'].nunique()}")
    print(f"  Exercises:   {sorted(df['exercise'].unique())}")
    print(f"  Date range:  {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
    print("\n  Sets per exercise:")
    for ex, cnt in df.groupby('exercise').size().items():
        print(f"    {ex:<30} {cnt:>5}")
    print("\n  RIR distribution:")
    for r in range(6):
        c = (df['rir'] == r).sum()
        print(f"    RIR {r}: {c:>4} ({c/len(df)*100:4.1f}%)")
    print(f"\n  Ordinal muscles loaded: {list(ORDINAL_MUSCLES.keys())}")
    print(sep)


if __name__ == "__main__":
    main()
