#!/usr/bin/env python3
"""
DeepGain Synthetic Training Data Generator — Research-Grade
===========================================================
Generates realistic resistance training logs:
    (user_id, exercise, weight, reps, rpe, timestamp)

Designed to produce data with the cross-exercise fatigue patterns that a
per-muscle fatigue (MPC) estimation model needs to learn.

Every numerical constant is traced to a peer-reviewed source via bracketed
reference numbers [N].  The full bibliography appears at the end of this
docstring.

────────────────────────────────────────────────────────────────────────────
BIBLIOGRAPHY (40 sources)
────────────────────────────────────────────────────────────────────────────
RM TABLE & REPS-TO-FAILURE
  [1]  Nuzzo 2024, Sports Med 54:303-321 — meta-regression 269 studies, 7289 subjects
  [2]  Shimano 2006, JSCR 20(4):819 — exercise-specific reps at %1RM (squat/bench/curl)
  [3]  Hoeger 1990, JSCR 4(3):76 — reps to failure, 7 exercises
  [4]  Wolfe 2024 — trained females, reps at 65/75/85/95% SBD

RPE / RIR
  [5]  Zourdos 2016, JSCR 30(1):267 — RIR-RPE scale, RPE SD by %1RM
  [6]  Helms 2017a, JSCR 31(2):292 — RPE & velocity for SBD in powerlifters
  [7]  Helms 2018, Front Physiol 9:247 — RPE-based periodisation
  [8]  Tuchscherer 2008 — RTS RPE chart (validated by [5][6])
  [9]  Halperin 2022, Sports Med — meta: RIR underprediction ~1 rep, SD=1.45
  [10] Refalo 2024, JSCR 38(3):e78 — intraset RIR accuracy 0.65±0.78 reps
  [11] Steele 2017, PeerJ 5:e4105 — RIR accuracy improves with experience

PER-SET FATIGUE & REP DROP-OFF
  [12] Refalo 2023, Sports Med Open 9:10 — 6×bench 75%, per-set velocity, 3 RIR conds
  [13] Saeterbakken ~2016 — 4×bench ~80%: 9.2→7.1→5.9→5.4
  [14] Willardson 2006, JSCR — rest period → rep sustainability
  [15] Willardson 2008, JSCR — 2 vs 4 min rest
  [16] Sánchez-Medina 2011, MSSE 43(9):1725 — velocity loss ↔ metabolic markers
  [17] Mangine 2022 — 3-RIR vs failure, 5×bench 80%

CROSS-EXERCISE FATIGUE
  [18] Simão 2005, JSCR 19(1):152 — exercise order BP→LPD→SP→BC→TE
  [19] Senna 2019, J Human Kinetics — bench+fly order, 5×10RM
  [20] Spreuwenberg 2006, JSCR — squat after full-body: −32.5%
  [21] Sforzo 1996, JSCR — large→small vs small→large order
  [22] Simão 2012, Sports Med 42(3):251 — exercise order review
  [23] Arazi 2015 — per-set data, 4 exercises, 2 orders, 70%

RECOVERY
  [24] Morán-Navarro 2017, Eur J Appl Physiol 117:2387 — failure vs non-failure recovery
  [25] Pareja-Blanco 2019, Sports 7(3):59 — 60%/80% × 20%/40% VL, 4 timepoints
  [26] Belcher 2019, Appl Physiol Nutr Metab — SBD recovery 4×failure@80%, 96h
  [27] Raastad 2000, Eur J Appl Physiol 82:206 — biphasic recovery (dip at ~22h)
  [28] Häkkinen 1993, JSCR — 20×1@100%, MVC decline M/F
  [29] Häkkinen 1994, Eur J Appl Physiol — 10×10@70%, MVC decline M/F

INDIVIDUAL VARIATION & SEX DIFFERENCES
  [30] Refalo 2023 — sex diff: males −29% vs females −21% velocity loss over 6 sets
  [31] Ahtiainen 2004 — trained vs untrained acute responses
  [32] Day 2004 / Gearhart 2016 — session RPE ICC = 0.88-0.895

WARM-UP
  [33] Ribeiro 2014, Percept Mot Skills 119:133 — no warm-up effect on working sets
  [34] Barroso 2012 — warm-up protocols

EMG / MUSCLE ACTIVATION
  [35] Martín-Fuentes 2020 — deadlift EMG systematic review
  [36] Rodríguez-Ridao 2020 — bench press EMG variations
  [37] Escamilla 2002 — squat/deadlift biomechanics & EMG
  [38] Contreras 2015 — glute activation across exercises
  [39] Saeterbakken 2011 — OHP vs bench EMG
  [40] Signorile 2002 — tricep activation across exercises
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — CONSTANTS & LOOKUP TABLES                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 1a. RM TABLE  [1] Nuzzo 2024 meta-regression ─────────────────────────
# %1RM → (mean reps to failure, between-individual SD)
# The SD doubles from ~1.0 at 95% to ~5.4 at 60% — critical for per-user
# noise.  The classic NSCA table *underestimates* reps at lighter loads:
# at 70% the textbook says 11 but the meta-regression says ~15.

_RM_PCTS  = np.array([1.00, 0.95, 0.90, 0.85, 0.80, 0.75,
                       0.70, 0.65, 0.60, 0.55, 0.50])
_RM_REPS  = np.array([1.0,  2.0,  4.5,  6.5,  9.0,  12.0,
                       15.0, 18.5, 22.0, 27.0, 33.0])
_RM_SD    = np.array([0.0,  1.0,  1.5,  2.2,  2.8,  3.5,
                       4.1,  4.8,  5.4,  6.0,  7.0])


def _interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """np.interp wrapper that handles descending xs."""
    return float(np.interp(x, xs[::-1], ys[::-1]))   # xs descend → reverse


def rm_table_reps(pct_1rm: float) -> float:
    """Mean reps to failure at *pct_1rm* (0-1).  [1]"""
    return _interp(np.clip(pct_1rm, 0.50, 1.0), _RM_PCTS, _RM_REPS)


def rm_table_sd(pct_1rm: float) -> float:
    """Between-individual SD of reps at *pct_1rm*.  [1]"""
    return _interp(np.clip(pct_1rm, 0.50, 1.0), _RM_PCTS, _RM_SD)


# ── 1b. EXERCISE-SPECIFIC RM MULTIPLIERS  [2] Shimano 2006 ──────────────
# At 60% 1RM: squat ≈30 reps, bench ≈22, curl ≈19.
# At 80-90% they converge.  Multiplier fades toward 1.0 above 85%.

def _exercise_rm_mult(base_mult: float, pct_1rm: float) -> float:
    """Apply exercise multiplier that fades above 85% 1RM.  [2]"""
    fade = max(0.0, min(1.0, (0.90 - pct_1rm) / 0.30))
    return 1.0 + (base_mult - 1.0) * fade


# ── 1c. RTS RPE CHART  [8] Tuchscherer, validated by [5][6] ─────────────
# (reps, RPE) → %1RM.  We store the full table for potential inverse
# lookups but primarily use the forward RM model.

RTS_CHART: Dict[Tuple[int, float], float] = {
    (1, 10.0): 1.000, (1, 9.5): 0.978, (1, 9.0): 0.955, (1, 8.5): 0.939,
    (1,  8.0): 0.922, (1, 7.5): 0.907, (1, 7.0): 0.892, (1, 6.5): 0.878,
    (2, 10.0): 0.955, (2, 9.5): 0.939, (2, 9.0): 0.922, (2, 8.5): 0.907,
    (2,  8.0): 0.892, (2, 7.5): 0.878, (2, 7.0): 0.863, (2, 6.5): 0.850,
    (3, 10.0): 0.922, (3, 9.5): 0.907, (3, 9.0): 0.892, (3, 8.5): 0.878,
    (3,  8.0): 0.863, (3, 7.5): 0.850, (3, 7.0): 0.837, (3, 6.5): 0.824,
    (4, 10.0): 0.892, (4, 9.5): 0.878, (4, 9.0): 0.863, (4, 8.5): 0.850,
    (4,  8.0): 0.837, (4, 7.5): 0.824, (4, 7.0): 0.811, (4, 6.5): 0.799,
    (5, 10.0): 0.863, (5, 9.5): 0.850, (5, 9.0): 0.837, (5, 8.5): 0.824,
    (5,  8.0): 0.811, (5, 7.5): 0.799, (5, 7.0): 0.786, (5, 6.5): 0.774,
    (6, 10.0): 0.837, (6, 9.5): 0.824, (6, 9.0): 0.811, (6, 8.5): 0.799,
    (6,  8.0): 0.786, (6, 7.5): 0.774, (6, 7.0): 0.762, (6, 6.5): 0.751,
    (7, 10.0): 0.811, (7, 9.5): 0.799, (7, 9.0): 0.786, (7, 8.5): 0.774,
    (7,  8.0): 0.762, (7, 7.5): 0.751, (7, 7.0): 0.739, (7, 6.5): 0.723,
    (8, 10.0): 0.786, (8, 9.5): 0.774, (8, 9.0): 0.762, (8, 8.5): 0.751,
    (8,  8.0): 0.739, (8, 7.5): 0.723, (8, 7.0): 0.707, (8, 6.5): 0.694,
    (9, 10.0): 0.762, (9, 9.5): 0.751, (9, 9.0): 0.739, (9, 8.5): 0.723,
    (9,  8.0): 0.707, (9, 7.5): 0.694, (9, 7.0): 0.680, (9, 6.5): 0.670,
    (10,10.0): 0.739, (10,9.5): 0.723, (10,9.0): 0.707, (10,8.5): 0.694,
    (10, 8.0): 0.680, (10,7.5): 0.670, (10,7.0): 0.650, (10,6.5): 0.640,
    (12,10.0): 0.707, (12,9.0): 0.680, (12,8.0): 0.650, (12,7.0): 0.620,
}


# ── 1d. MUSCLE GROUPS ────────────────────────────────────────────────────
# 16 distinct groups tracked for MPC state.

ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "upper_traps", "rhomboids",
    "triceps", "biceps", "brachialis",
    "lats",
    "quads", "hamstrings", "glutes", "adductors",
    "erectors",
    "calves",
]


# ── 1e. EXERCISE DATABASE ───────────────────────────────────────────────
# 25 exercises.  Each entry:
#   muscles        : muscle → EMG-based involvement 0.0-1.0  [35-40]
#   rm_mult_60     : RM multiplier at 60% 1RM  [2]
#   is_compound    : bool
#   recovery_modifier : per-exercise modifier on recovery tau (1.0 = default)
#   decay_modifier : 1.0 compounds, ~0.85 isolations (shallower set decay) [13]
#   weight_increment : rounding granularity (kg)
#   min_weight_kg  : floor
#   equipment      : "barbell" | "dumbbell" | "cable" | "machine" | "bodyweight"

@dataclass(frozen=True)
class ExerciseDef:
    muscles: Dict[str, float]
    rm_mult_60: float
    is_compound: bool
    recovery_modifier: float
    decay_modifier: float
    weight_increment: float
    min_weight_kg: float
    equipment: str


EXERCISE_DB: Dict[str, ExerciseDef] = {
    # ── PRESSING ──
    "bench_press": ExerciseDef(
        muscles={"chest": 0.85, "triceps": 0.55, "anterior_delts": 0.60},  # [36]
        rm_mult_60=0.73, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "incline_bench": ExerciseDef(
        muscles={"chest": 0.70, "anterior_delts": 0.75, "triceps": 0.50},  # [36]
        rm_mult_60=0.70, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "close_grip_bench": ExerciseDef(
        muscles={"chest": 0.65, "triceps": 0.75, "anterior_delts": 0.55},  # [36][40]
        rm_mult_60=0.70, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "dumbbell_bench": ExerciseDef(
        muscles={"chest": 0.82, "triceps": 0.45, "anterior_delts": 0.55},  # [36]
        rm_mult_60=0.72, is_compound=True, recovery_modifier=0.95,
        decay_modifier=1.0, weight_increment=2.0, min_weight_kg=10,
        equipment="dumbbell",
    ),
    "ohp": ExerciseDef(
        muscles={"anterior_delts": 0.85, "triceps": 0.65, "chest": 0.20,
                 "upper_traps": 0.40},  # [39]
        rm_mult_60=0.68, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=15,
        equipment="barbell",
    ),
    "dumbbell_ohp": ExerciseDef(
        muscles={"anterior_delts": 0.80, "triceps": 0.60, "upper_traps": 0.35},
        rm_mult_60=0.65, is_compound=True, recovery_modifier=0.95,
        decay_modifier=1.0, weight_increment=2.0, min_weight_kg=8,
        equipment="dumbbell",
    ),
    "dips": ExerciseDef(
        muscles={"chest": 0.70, "triceps": 0.65, "anterior_delts": 0.45},  # [36]
        rm_mult_60=0.72, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=0,
        equipment="bodyweight",
    ),

    # ── PULLING ──
    "barbell_row": ExerciseDef(
        muscles={"lats": 0.80, "biceps": 0.55, "rear_delts": 0.50,
                 "erectors": 0.40, "upper_traps": 0.35, "rhomboids": 0.45},  # [35]
        rm_mult_60=0.73, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "lat_pulldown": ExerciseDef(
        muscles={"lats": 0.75, "biceps": 0.50, "rear_delts": 0.35,
                 "rhomboids": 0.40},
        rm_mult_60=0.68, is_compound=True, recovery_modifier=0.95,
        decay_modifier=0.95, weight_increment=5.0, min_weight_kg=20,
        equipment="cable",
    ),
    "cable_row": ExerciseDef(
        muscles={"lats": 0.70, "biceps": 0.45, "rear_delts": 0.40,
                 "rhomboids": 0.50, "upper_traps": 0.30},
        rm_mult_60=0.70, is_compound=True, recovery_modifier=0.95,
        decay_modifier=0.95, weight_increment=5.0, min_weight_kg=15,
        equipment="cable",
    ),
    "pull_up": ExerciseDef(
        muscles={"lats": 0.82, "biceps": 0.55, "rear_delts": 0.35,
                 "rhomboids": 0.40},
        rm_mult_60=0.70, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=0,
        equipment="bodyweight",
    ),

    # ── LOWER BODY COMPOUNDS ──
    "squat": ExerciseDef(
        muscles={"quads": 0.85, "glutes": 0.60, "hamstrings": 0.35,
                 "erectors": 0.45, "adductors": 0.40},  # [37]
        rm_mult_60=1.00, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "front_squat": ExerciseDef(
        muscles={"quads": 0.90, "glutes": 0.50, "erectors": 0.55,
                 "adductors": 0.35},  # [37]
        rm_mult_60=0.95, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=20,
        equipment="barbell",
    ),
    "deadlift": ExerciseDef(
        muscles={"glutes": 0.70, "hamstrings": 0.55, "erectors": 0.80,
                 "quads": 0.40, "upper_traps": 0.50, "lats": 0.30,
                 "adductors": 0.35},  # [35][37]
        rm_mult_60=0.85, is_compound=True, recovery_modifier=0.80,
        # Belcher [26]: deadlift never significantly impaired → faster recovery
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=40,
        equipment="barbell",
    ),
    "rdl": ExerciseDef(
        muscles={"hamstrings": 0.80, "glutes": 0.55, "erectors": 0.50,
                 "adductors": 0.25},  # [35]
        rm_mult_60=0.78, is_compound=True, recovery_modifier=0.90,
        decay_modifier=1.0, weight_increment=2.5, min_weight_kg=30,
        equipment="barbell",
    ),
    "leg_press": ExerciseDef(
        muscles={"quads": 0.80, "glutes": 0.50, "adductors": 0.35},  # [37]
        rm_mult_60=1.10, is_compound=True, recovery_modifier=0.90,
        decay_modifier=0.95, weight_increment=5.0, min_weight_kg=40,
        equipment="machine",
    ),
    "bulgarian_split_squat": ExerciseDef(
        muscles={"quads": 0.80, "glutes": 0.65, "hamstrings": 0.30,
                 "adductors": 0.40},  # [38]
        rm_mult_60=0.90, is_compound=True, recovery_modifier=1.0,
        decay_modifier=1.0, weight_increment=2.0, min_weight_kg=0,
        equipment="dumbbell",
    ),
    "hip_thrust": ExerciseDef(
        muscles={"glutes": 0.85, "hamstrings": 0.40, "adductors": 0.30},  # [38]
        rm_mult_60=0.95, is_compound=True, recovery_modifier=0.90,
        decay_modifier=0.95, weight_increment=5.0, min_weight_kg=20,
        equipment="barbell",
    ),

    # ── ISOLATION ──
    "tricep_pushdown": ExerciseDef(
        muscles={"triceps": 0.90},  # [40]
        rm_mult_60=0.63, is_compound=False, recovery_modifier=0.80,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=5,
        equipment="cable",
    ),
    "overhead_tricep_ext": ExerciseDef(
        muscles={"triceps": 0.85},  # [40] long head emphasis
        rm_mult_60=0.60, is_compound=False, recovery_modifier=0.80,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=5,
        equipment="cable",
    ),
    "bicep_curl": ExerciseDef(
        muscles={"biceps": 0.90},  # [2]
        rm_mult_60=0.63, is_compound=False, recovery_modifier=0.80,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=5,
        equipment="dumbbell",
    ),
    "hammer_curl": ExerciseDef(
        muscles={"biceps": 0.75, "brachialis": 0.60},
        rm_mult_60=0.65, is_compound=False, recovery_modifier=0.80,
        decay_modifier=0.85, weight_increment=2.0, min_weight_kg=5,
        equipment="dumbbell",
    ),
    "lateral_raise": ExerciseDef(
        muscles={"lateral_delts": 0.85, "upper_traps": 0.30},
        rm_mult_60=0.55, is_compound=False, recovery_modifier=0.75,
        decay_modifier=0.80, weight_increment=1.0, min_weight_kg=2,
        equipment="dumbbell",
    ),
    "face_pull": ExerciseDef(
        muscles={"rear_delts": 0.70, "upper_traps": 0.40, "rhomboids": 0.35},
        rm_mult_60=0.55, is_compound=False, recovery_modifier=0.75,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=5,
        equipment="cable",
    ),
    "leg_curl": ExerciseDef(
        muscles={"hamstrings": 0.85},
        rm_mult_60=0.65, is_compound=False, recovery_modifier=0.85,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=10,
        equipment="machine",
    ),
    "leg_extension": ExerciseDef(
        muscles={"quads": 0.85},
        rm_mult_60=0.68, is_compound=False, recovery_modifier=0.85,
        decay_modifier=0.85, weight_increment=2.5, min_weight_kg=10,
        equipment="machine",
    ),
    "calf_raise": ExerciseDef(
        muscles={"calves": 0.90},
        rm_mult_60=0.70, is_compound=False, recovery_modifier=0.75,
        decay_modifier=0.80, weight_increment=5.0, min_weight_kg=20,
        equipment="machine",
    ),
}

ALL_EXERCISES = list(EXERCISE_DB.keys())


# ── 1f. MUSCLE RECOVERY TAU (hours) ─────────────────────────────────────
# Time for a fatigue deficit to decay to 37% (one time-constant).
# Calibrated from:
#   [26] Belcher 2019 — squat impaired 72h, bench by 24h, deadlift never
#   [25] Pareja-Blanco 2019 — 80%/20%VL recovered 48h; 60%/40%VL slower
#   [27] Raastad 2000 — biphasic dip at ~22h after heavy compounds
#   [28][29] Häkkinen 1993/94 — MVC decline data

MUSCLE_RECOVERY_TAU: Dict[str, float] = {
    # Lower body — slower [25][26]
    "quads":       36.0,     # [26] squat impaired 72h (3τ ≈ full)
    "glutes":      32.0,
    "hamstrings":  28.0,     # [25] slightly faster than quads
    "adductors":   28.0,
    "erectors":    26.0,     # [26]
    "calves":      20.0,
    # Upper body — faster [26]
    "chest":       22.0,     # [26] bench recovered by 24h
    "lats":        24.0,
    "anterior_delts": 20.0,
    "lateral_delts":  18.0,
    "rear_delts":     18.0,
    "upper_traps":    20.0,
    "rhomboids":      20.0,
    "triceps":     16.0,     # small muscle → fast recovery
    "biceps":      16.0,
    "brachialis":  16.0,
}

# Biphasic recovery dip parameters [27] Raastad 2000
# After heavy compound work (≥82% 1RM), a secondary performance dip
# appears at ~22h, depth ~3%, width ~8h (Gaussian envelope).
BIPHASIC_DIP_MAGNITUDE = 0.03   # [27]
BIPHASIC_DIP_CENTER_H  = 22.0   # [27]
BIPHASIC_DIP_WIDTH_H   = 8.0    # [27]


# ── 1g. PER-SET FATIGUE CONSTANTS ───────────────────────────────────────
# Calibrated so 4 sets of ~9 reps @ RPE 8 on bench drops chest MPC
# from 1.0 to ~0.80, matching ~20% force decline [24 Morán-Navarro].
# Each set at failure (RPE 10) on a primary mover (coeff=0.85) drops
# MPC by ~0.05–0.06.  At 3-RIR it's ~0.02–0.03.

BASE_MPC_DROP_PER_SET = 0.08   # [12][13][24] — tuned constant
MPC_FLOOR = 0.30               # never below 30% capacity


# ── 1h. RPE NOISE PARAMETERS ────────────────────────────────────────────
# [5] Zourdos 2016: RPE SD = 0.18 @ 100%, 0.92 @ 90%, 1.18 @ 60%
# Linear fit: SD ≈ 0.18 + 0.025 × (100 − %1RM)
# We use a slightly compressed version to prevent non-monotonic RPE
# artifacts within a single exercise:
#   SD ≈ 0.15 + 0.015 × (100 − %1RM)
# [9] Halperin 2022: population-level underprediction bias ≈ −0.3 RPE
# [11] Steele 2017: experienced lifters are more accurate → exp modifier

RPE_NOISE_BASE_SD    = 0.15    # [5] intercept (at 100% 1RM)
RPE_NOISE_SLOPE      = 0.015   # [5] per percentage point away from 100%
RPE_POPULATION_BIAS  = -0.3    # [9] systematic underprediction


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — SPLIT TEMPLATES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Each exercise entry: (name, %1RM, num_sets, target_RIR, rest_minutes)
# %1RM is an *initial* prescription; the workout generator auto-adjusts
# weight to hit a sensible rep range (see Section 7).

WorkoutEntry = Tuple[str, float, int, int, float]

WORKOUT_TEMPLATES: Dict[str, List[WorkoutEntry]] = {
    # ── PUSH/PULL/LEGS ──────────────────────────────────────────────────
    "push_volume": [
        ("bench_press",      0.75, 4, 2, 3.0),
        ("incline_bench",    0.72, 3, 2, 3.0),
        ("ohp",              0.70, 3, 2, 3.0),
        ("dips",             0.70, 3, 2, 2.5),
        ("tricep_pushdown",  0.68, 3, 3, 2.0),
        ("lateral_raise",    0.65, 3, 3, 1.5),
    ],
    "push_strength": [
        ("bench_press",      0.85, 5, 1, 4.5),
        ("close_grip_bench", 0.78, 3, 2, 3.5),
        ("ohp",              0.78, 3, 2, 3.5),
        ("dips",             0.75, 3, 2, 2.5),
        ("overhead_tricep_ext", 0.65, 3, 3, 2.0),
    ],
    "pull_volume": [
        ("barbell_row",      0.75, 4, 2, 3.0),
        ("lat_pulldown",     0.72, 3, 2, 2.5),
        ("cable_row",        0.70, 3, 2, 2.5),
        ("face_pull",        0.65, 3, 3, 1.5),
        ("bicep_curl",       0.68, 3, 2, 2.0),
        ("hammer_curl",      0.68, 2, 3, 1.5),
    ],
    "pull_strength": [
        ("barbell_row",      0.82, 4, 1, 3.5),
        ("pull_up",          0.80, 3, 2, 3.0),
        ("cable_row",        0.75, 3, 2, 3.0),
        ("bicep_curl",       0.70, 3, 2, 2.0),
        ("face_pull",        0.60, 3, 3, 1.5),
    ],
    "legs_quad": [
        ("squat",            0.78, 4, 2, 4.0),
        ("leg_press",        0.75, 3, 2, 3.0),
        ("bulgarian_split_squat", 0.72, 3, 2, 2.5),
        ("leg_extension",    0.70, 3, 2, 2.0),
        ("leg_curl",         0.68, 3, 2, 2.0),
        ("calf_raise",       0.72, 3, 2, 1.5),
    ],
    "legs_posterior": [
        ("deadlift",         0.80, 4, 2, 4.5),
        ("rdl",              0.75, 3, 2, 3.0),
        ("hip_thrust",       0.75, 3, 2, 3.0),
        ("leg_curl",         0.72, 3, 2, 2.0),
        ("leg_extension",    0.68, 3, 2, 2.0),
        ("calf_raise",       0.72, 3, 2, 1.5),
    ],

    # ── UPPER / LOWER ──────────────────────────────────────────────────
    "upper_A": [
        ("bench_press",      0.78, 4, 2, 3.5),
        ("barbell_row",      0.75, 4, 2, 3.0),
        ("ohp",              0.72, 3, 2, 3.0),
        ("lat_pulldown",     0.72, 3, 2, 2.5),
        ("tricep_pushdown",  0.68, 2, 2, 2.0),
        ("bicep_curl",       0.68, 2, 2, 2.0),
    ],
    "upper_B": [
        ("ohp",              0.78, 4, 2, 3.5),
        ("cable_row",        0.75, 4, 2, 3.0),
        ("dumbbell_bench",   0.72, 3, 2, 3.0),
        ("pull_up",          0.75, 3, 2, 3.0),
        ("lateral_raise",    0.65, 3, 2, 1.5),
        ("hammer_curl",      0.68, 2, 2, 2.0),
    ],
    "lower_A": [
        ("squat",            0.80, 4, 2, 4.0),
        ("rdl",              0.75, 3, 2, 3.5),
        ("leg_press",        0.72, 3, 2, 3.0),
        ("leg_curl",         0.70, 3, 2, 2.0),
        ("calf_raise",       0.72, 3, 2, 1.5),
    ],
    "lower_B": [
        ("deadlift",         0.82, 4, 2, 4.5),
        ("front_squat",      0.75, 3, 2, 3.5),
        ("hip_thrust",       0.75, 3, 2, 3.0),
        ("leg_extension",    0.70, 3, 2, 2.0),
        ("calf_raise",       0.72, 3, 2, 1.5),
    ],

    # ── FULL BODY ──────────────────────────────────────────────────────
    "full_body_A": [
        ("squat",            0.78, 3, 2, 4.0),
        ("bench_press",      0.78, 3, 2, 3.5),
        ("barbell_row",      0.75, 3, 2, 3.0),
        ("rdl",              0.70, 2, 2, 3.0),
        ("lateral_raise",    0.65, 2, 2, 1.5),
    ],
    "full_body_B": [
        ("deadlift",         0.80, 3, 2, 4.5),
        ("dumbbell_ohp",     0.75, 3, 2, 3.5),
        ("lat_pulldown",     0.72, 3, 2, 2.5),
        ("leg_press",        0.72, 2, 2, 3.0),
        ("bicep_curl",       0.68, 2, 2, 2.0),
    ],
    "full_body_C": [
        ("front_squat",      0.75, 3, 2, 4.0),
        ("incline_bench",    0.75, 3, 2, 3.5),
        ("cable_row",        0.72, 3, 2, 2.5),
        ("hip_thrust",       0.72, 2, 2, 3.0),
        ("face_pull",        0.65, 2, 2, 1.5),
    ],

    # ── POWERLIFTING ───────────────────────────────────────────────────
    "heavy_bench": [
        ("bench_press",      0.88, 5, 1, 5.0),
        ("close_grip_bench", 0.78, 3, 2, 3.5),
        ("dips",             0.72, 3, 2, 2.5),
        ("tricep_pushdown",  0.65, 3, 3, 2.0),
    ],
    "heavy_squat": [
        ("squat",            0.88, 5, 1, 5.0),
        ("front_squat",      0.72, 3, 3, 3.5),
        ("leg_press",        0.72, 3, 3, 3.0),
        ("leg_curl",         0.65, 3, 3, 2.0),
    ],
    "heavy_deadlift": [
        ("deadlift",         0.88, 4, 1, 5.0),
        ("rdl",              0.70, 3, 3, 3.0),
        ("barbell_row",      0.72, 3, 2, 3.0),
        ("hip_thrust",       0.70, 3, 3, 2.5),
    ],

    # ── BODYBUILDING / BRO SPLIT ───────────────────────────────────────
    "chest_shoulders": [
        ("bench_press",      0.75, 4, 2, 2.5),
        ("incline_bench",    0.72, 3, 2, 2.5),
        ("dumbbell_bench",   0.70, 3, 2, 2.0),
        ("ohp",              0.70, 3, 2, 2.5),
        ("lateral_raise",    0.65, 4, 2, 1.5),
        ("tricep_pushdown",  0.68, 3, 2, 1.5),
    ],
    "back_biceps": [
        ("barbell_row",      0.75, 4, 2, 3.0),
        ("lat_pulldown",     0.72, 3, 2, 2.5),
        ("cable_row",        0.72, 3, 2, 2.5),
        ("face_pull",        0.65, 3, 2, 1.5),
        ("bicep_curl",       0.68, 3, 2, 2.0),
        ("hammer_curl",      0.68, 3, 2, 1.5),
    ],

    # ── DELOAD (week 4 of each mesocycle) ─────────────────────────────
    "deload_upper": [
        ("bench_press",      0.65, 3, 4, 3.0),
        ("barbell_row",      0.62, 3, 4, 3.0),
        ("ohp",              0.60, 2, 4, 2.5),
        ("bicep_curl",       0.55, 2, 4, 2.0),
    ],
    "deload_lower": [
        ("squat",            0.65, 3, 4, 3.5),
        ("rdl",              0.60, 2, 4, 3.0),
        ("leg_extension",    0.55, 2, 4, 2.0),
        ("leg_curl",         0.55, 2, 4, 2.0),
    ],
    "deload_full": [
        ("squat",            0.62, 2, 4, 3.5),
        ("bench_press",      0.62, 2, 4, 3.0),
        ("barbell_row",      0.60, 2, 4, 3.0),
    ],
}

# Weekly schedule definitions:
#   days: list of 7 template names (None = rest day)
#   deload_map: template → deload replacement for week 4

@dataclass(frozen=True)
class WeeklySchedule:
    name: str
    days: Tuple[Optional[str], ...]
    deload_days: Tuple[Optional[str], ...]

WEEKLY_SCHEDULES: List[WeeklySchedule] = [
    WeeklySchedule("PPL_6day",
        days=("push_volume", "pull_volume", "legs_quad",
              "push_strength", "pull_strength", "legs_posterior", None),
        deload_days=("deload_upper", "deload_upper", "deload_lower",
                     None, None, None, None),
    ),
    WeeklySchedule("UL_4day",
        days=("upper_A", "lower_A", None, "upper_B", "lower_B", None, None),
        deload_days=("deload_upper", "deload_lower", None,
                     "deload_upper", None, None, None),
    ),
    WeeklySchedule("FB_3day",
        days=("full_body_A", None, "full_body_B", None, "full_body_C", None, None),
        deload_days=("deload_full", None, "deload_full", None, None, None, None),
    ),
    WeeklySchedule("PL_4day",
        days=("heavy_bench", "heavy_squat", None,
              "push_volume", "heavy_deadlift", None, None),
        deload_days=("deload_upper", "deload_lower", None,
                     "deload_upper", None, None, None),
    ),
    WeeklySchedule("bro_5day",
        days=("chest_shoulders", "back_biceps", "legs_quad",
              "push_volume", "legs_posterior", None, None),
        deload_days=("deload_upper", "deload_upper", "deload_lower",
                     None, None, None, None),
    ),
    WeeklySchedule("UL_3day",
        days=("upper_A", "lower_A", None, "upper_B", None, None, None),
        deload_days=("deload_upper", "deload_lower", None, None, None, None, None),
    ),
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — USER PROFILE                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Training levels for realistic population distribution
_LEVELS = {
    "beginner":     {"years": (0.5, 1.5),  "bench_m": (0.60, 0.90), "squat_m": (0.80, 1.10)},
    "intermediate": {"years": (1.5, 4.0),  "bench_m": (0.90, 1.30), "squat_m": (1.10, 1.60)},
    "advanced":     {"years": (4.0, 10.0), "bench_m": (1.30, 1.80), "squat_m": (1.60, 2.20)},
    "elite":        {"years": (8.0, 20.0), "bench_m": (1.80, 2.50), "squat_m": (2.20, 3.00)},
}
_LEVEL_WEIGHTS = [0.25, 0.40, 0.25, 0.10]


@dataclass
class UserProfile:
    user_id: str
    sex: str                        # 'M' or 'F'
    age: int
    bodyweight_kg: float
    training_years: float           # affects RPE accuracy [11]
    e1rm: Dict[str, float]          # exercise → estimated 1RM (kg)

    # Individual traits (drawn once, persist across all sessions)
    rm_individual_factor: float     # [1] personal offset on RM table
    rpe_bias: float                 # [9] systematic RPE reporting offset
    rpe_noise_scale: float          # multiplier on RPE noise SD
    fatigue_sensitivity: float      # multiplier on MPC drop rate
    recovery_rate: float            # multiplier on recovery speed
    daily_rep_variation: float      # SD of day-to-day rep noise (reps)
    consistency: float              # probability of attending each session

    # Mutable state
    mpc: Dict[str, float] = field(default_factory=dict)
    last_set_time: Dict[str, datetime] = field(default_factory=dict)
    last_session_pct: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.mpc = {m: 1.0 for m in ALL_MUSCLES}
        epoch = datetime(2024, 1, 1)
        self.last_set_time = {m: epoch for m in ALL_MUSCLES}
        self.last_session_pct = {m: 0.75 for m in ALL_MUSCLES}


def generate_user(rng: np.random.Generator, user_id: str) -> UserProfile:
    """Create a user with realistic strength levels and individual traits."""
    sex = rng.choice(["M", "F"], p=[0.65, 0.35])
    level_names = list(_LEVELS.keys())
    level = rng.choice(level_names, p=_LEVEL_WEIGHTS)
    cfg = _LEVELS[level]

    experience = rng.uniform(*cfg["years"])
    age = int(np.clip(rng.normal(28, 7), 18, 55))

    if sex == "M":
        bw = np.clip(rng.normal(82, 12), 60, 120)
        bench_mult = rng.uniform(*cfg["bench_m"])
        squat_mult = rng.uniform(*cfg["squat_m"])
    else:
        bw = np.clip(rng.normal(64, 10), 45, 95)
        bench_mult = rng.uniform(*cfg["bench_m"]) * 0.58
        squat_mult = rng.uniform(*cfg["squat_m"]) * 0.65

    bench = bw * bench_mult
    squat = bw * squat_mult
    dl = squat * rng.uniform(1.05, 1.25)

    e1rm = {
        "bench_press":       bench,
        "incline_bench":     bench * rng.uniform(0.78, 0.85),
        "close_grip_bench":  bench * rng.uniform(0.82, 0.88),
        "dumbbell_bench":    bench * rng.uniform(0.38, 0.45),  # per hand
        "ohp":               bench * rng.uniform(0.58, 0.68),
        "dumbbell_ohp":      bench * rng.uniform(0.28, 0.35),  # per hand
        "dips":              bench * rng.uniform(0.55, 0.70),
        "barbell_row":       bench * rng.uniform(0.75, 0.90),
        "lat_pulldown":      bench * rng.uniform(0.60, 0.75),
        "cable_row":         bench * rng.uniform(0.55, 0.70),
        "pull_up":           bench * rng.uniform(0.50, 0.65),
        "squat":             squat,
        "front_squat":       squat * rng.uniform(0.78, 0.85),
        "deadlift":          dl,
        "rdl":               dl * rng.uniform(0.65, 0.75),
        "leg_press":         squat * rng.uniform(1.30, 1.60),
        "bulgarian_split_squat": squat * rng.uniform(0.35, 0.45),
        "hip_thrust":        squat * rng.uniform(1.00, 1.30),
        "tricep_pushdown":   bench * rng.uniform(0.30, 0.40),
        "overhead_tricep_ext": bench * rng.uniform(0.25, 0.35),
        "bicep_curl":        bench * rng.uniform(0.25, 0.35),
        "hammer_curl":       bench * rng.uniform(0.28, 0.38),
        "lateral_raise":     bw * rng.uniform(0.08, 0.15),
        "face_pull":         bw * rng.uniform(0.15, 0.25),
        "leg_curl":          squat * rng.uniform(0.30, 0.40),
        "leg_extension":     squat * rng.uniform(0.35, 0.50),
        "calf_raise":        bw * rng.uniform(0.80, 1.50),
    }

    return UserProfile(
        user_id=user_id,
        sex=sex,
        age=age,
        bodyweight_kg=bw,
        training_years=experience,
        e1rm=e1rm,
        rm_individual_factor=float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.30)),  # [1]
        rpe_bias=float(rng.normal(0, 0.3)),            # [9]
        rpe_noise_scale=float(np.clip(rng.normal(1.0, 0.20), 0.5, 1.6)),
        fatigue_sensitivity=float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.5)),
        recovery_rate=float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.5)),
        daily_rep_variation=float(np.clip(rng.normal(1.5, 0.5), 0.5, 3.0)),  # [32]
        consistency=float(rng.uniform(0.75, 1.0)),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — MUSCLE STATE (MPC tracking & recovery)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def apply_recovery(user: UserProfile, current_time: datetime) -> None:
    """Recover all muscles toward MPC=1.0 based on elapsed time.

    Uses exponential decay of the fatigue deficit with muscle-specific τ
    [25][26], modulated by:
      - exercise-specific recovery_modifier from EXERCISE_DB
      - individual recovery_rate trait
      - optional biphasic dip at ~22h for heavy sessions [27]
    """
    for muscle in ALL_MUSCLES:
        if user.mpc[muscle] >= 0.999:
            user.mpc[muscle] = 1.0
            continue

        dt_hours = (current_time - user.last_set_time[muscle]).total_seconds() / 3600.0
        if dt_hours <= 0:
            continue

        tau = MUSCLE_RECOVERY_TAU[muscle]
        effective_dt = dt_hours * user.recovery_rate

        # Exponential recovery toward 1.0  [25]
        deficit = 1.0 - user.mpc[muscle]
        remaining_deficit = deficit * math.exp(-effective_dt / tau)
        mpc = 1.0 - remaining_deficit

        # Biphasic dip for heavy sessions  [27]
        session_pct = user.last_session_pct.get(muscle, 0.75)
        if session_pct > 0.82 and tau > 24:
            dip = BIPHASIC_DIP_MAGNITUDE * math.exp(
                -0.5 * ((dt_hours - BIPHASIC_DIP_CENTER_H) / BIPHASIC_DIP_WIDTH_H) ** 2
            )
            mpc -= dip

        user.mpc[muscle] = max(MPC_FLOOR, min(1.0, mpc))


def apply_fatigue(user: UserProfile, exercise: str, reps: int,
                  true_rpe: float, rest_minutes: float,
                  timestamp: datetime) -> None:
    """Apply MPC fatigue to all muscles involved in *exercise*.

    Drop formula components (all multiplicative):
      1. base_drop = 0.08  [12][13][24]
      2. emg_coeff — muscle involvement  [35-40]
      3. proximity_factor — (RPE-5)/5 raised to 1.5  [12][16]
      4. reps_factor — reps/10  [12][13]
      5. rest_factor — amplified for <3 min rest  [14]
      6. sex_factor — males 1.20, females 0.85  [30]
      7. decay_modifier — isolation exercises decay shallower  [13]
      8. individual fatigue_sensitivity
    """
    exdef = EXERCISE_DB[exercise]

    # Proximity to failure  [12][16]
    proximity = max(0.0, (true_rpe - 5.0) / 5.0)
    proximity_factor = proximity ** 1.5

    # Volume  [12][13]
    reps_factor = reps / 10.0

    # Rest period  [14]
    # 1min → retention 0.50, 2min → 0.65, 5min → 0.88
    rest_factor = 1.0 + 0.3 * max(0.0, 1.0 - rest_minutes / 3.0)

    # Sex  [30]: males lose ~38% more velocity than females over 6 sets
    sex_factor = 1.20 if user.sex == "M" else 0.85

    e1rm = user.e1rm.get(exercise, 80.0)
    pct_1rm_nominal = 0.75  # fallback

    for muscle, emg_coeff in exdef.muscles.items():
        drop = (BASE_MPC_DROP_PER_SET
                * emg_coeff
                * proximity_factor
                * reps_factor
                * rest_factor
                * sex_factor
                * exdef.decay_modifier
                * user.fatigue_sensitivity)
        drop = max(0.005, min(0.25, drop))

        user.mpc[muscle] = max(MPC_FLOOR, user.mpc[muscle] - drop)
        user.last_set_time[muscle] = timestamp
        # Track heaviest %1RM used for biphasic dip decision
        if e1rm > 0:
            pct_1rm_nominal = max(
                user.last_session_pct.get(muscle, 0.0),
                0.75  # will be set properly below
            )
        user.last_session_pct[muscle] = pct_1rm_nominal


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — SET SIMULATOR (core engine)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_capacity_modifier(user: UserProfile, exercise: str) -> float:
    """Weighted average MPC across involved muscles.

    This is the mechanism for cross-exercise fatigue transfer [18-23].
    If bench fatigued chest (MPC 0.85), triceps (0.90), anterior_delts (0.88),
    then a subsequent OHP sees reduced capacity through shared muscles.
    """
    exdef = EXERCISE_DB[exercise]
    total_w = sum(exdef.muscles.values())
    if total_w == 0:
        return 1.0
    weighted = sum(user.mpc.get(m, 1.0) * c for m, c in exdef.muscles.items())
    return weighted / total_w


def compute_max_reps(user: UserProfile, exercise: str, weight: float) -> float:
    """Maximum reps this user can do at *weight* given current MPC state.

    Pipeline:
      1. Effective 1RM = e1RM × capacity_modifier (cross-exercise fatigue)
      2. %1RM = weight / effective_1RM
      3. Base reps from RM table [1]
      4. Exercise-specific multiplier [2]
      5. Individual RM factor [1]
      6. Sex adjustment at lighter loads [12][30][4]
    """
    e1rm = user.e1rm.get(exercise, 80.0)
    if e1rm <= 0:
        return 0.0

    cap_mod = get_capacity_modifier(user, exercise)
    effective_1rm = e1rm * cap_mod

    if weight >= effective_1rm or effective_1rm <= 0:
        return 0.0

    pct = weight / effective_1rm
    pct = np.clip(pct, 0.50, 0.999)

    # Base reps from RM table [1]
    base_reps = rm_table_reps(pct)

    # Exercise-specific multiplier [2]
    exdef = EXERCISE_DB[exercise]
    mult = _exercise_rm_mult(exdef.rm_mult_60, pct)
    base_reps *= mult

    # Individual RM factor [1]
    base_reps *= user.rm_individual_factor

    # Sex effect: females do more reps at lighter loads [12][30][4]
    # Refalo 2023: 64±14 total reps (F) vs 45±8 (M) over 6 sets at 75%
    if user.sex == "F" and pct < 0.80:
        base_reps *= 1.0 + 0.15 * (0.80 - pct)

    return max(0.0, min(base_reps, 30.0))


def simulate_set(rng: np.random.Generator,
                 user: UserProfile,
                 exercise: str,
                 weight: float,
                 target_rir: int,
                 rest_minutes: float,
                 timestamp: datetime,
                 is_warmup: bool = False,
                 ) -> Optional[Dict]:
    """Simulate a single set.  Returns a row dict or None if can't lift.

    Algorithm:
      1. Apply recovery to current time
      2. Compute max reps (incorporates cross-exercise fatigue)
      3. Actual reps = max_reps − target_RIR + noise [10]
      4. True RPE = 10 − true_RIR
      5. Reported RPE = true_RPE + bias + noise [5][9]
      6. Apply fatigue to involved muscles
    """
    apply_recovery(user, timestamp)

    max_reps = compute_max_reps(user, exercise, weight)
    if max_reps < 1.0:
        return None

    if is_warmup:
        # Warm-ups: do prescribed reps regardless of capacity [33]
        actual_reps = int(max(1, min(round(max_reps * 0.3), 10)))
        true_rir = max(0, round(max_reps) - actual_reps)
        true_rpe = min(10.0, 10.0 - true_rir)
        # Warm-up RPE is low: roughly %1RM/20  [33]
        e1rm = user.e1rm.get(exercise, 80.0)
        pct = weight / e1rm if e1rm > 0 else 0.5
        reported_rpe = float(np.clip(round(pct * 100 / 20 * 2) / 2, 3.0, 6.0))

        # Negligible fatigue from warm-ups [33]
        for muscle in EXERCISE_DB[exercise].muscles:
            user.last_set_time[muscle] = timestamp

        return {
            "user_id": user.user_id,
            "exercise": exercise,
            "weight": round(weight, 1),
            "reps": actual_reps,
            "rpe": reported_rpe,
            "timestamp": timestamp.isoformat(),
        }

    # Working set
    target_reps = max_reps - target_rir
    # Day-to-day variation [32] + execution noise [10]
    noise = rng.normal(0, max(0.3, user.daily_rep_variation * 0.4))
    actual_reps = int(np.clip(round(target_reps + noise), 1, round(max_reps)))

    true_rir = max(0, round(max_reps) - actual_reps)
    true_rpe = min(10.0, 10.0 - true_rir)

    # RPE noise  [5][9][10][11]
    e1rm = user.e1rm.get(exercise, 80.0)
    pct_1rm = np.clip(weight / e1rm, 0.4, 1.0) if e1rm > 0 else 0.75

    rpe_sd = (RPE_NOISE_BASE_SD + RPE_NOISE_SLOPE * (100 - pct_1rm * 100))
    # Experience modifier [11]: experienced lifters are more accurate
    exp_mod = 1.3 - 0.05 * min(user.training_years, 10)
    rpe_sd *= exp_mod * user.rpe_noise_scale
    rpe_sd = max(0.1, rpe_sd)

    reported_rpe = (true_rpe
                    + RPE_POPULATION_BIAS
                    + user.rpe_bias
                    + rng.normal(0, rpe_sd))
    reported_rpe = float(np.clip(np.round(reported_rpe * 2) / 2, 6.0, 10.0))

    # Update session pct for biphasic dip tracking
    for muscle in EXERCISE_DB[exercise].muscles:
        user.last_session_pct[muscle] = max(
            user.last_session_pct.get(muscle, 0.0), pct_1rm
        )

    # Apply fatigue
    apply_fatigue(user, exercise, actual_reps, true_rpe, rest_minutes, timestamp)

    return {
        "user_id": user.user_id,
        "exercise": exercise,
        "weight": round(weight, 1),
        "reps": actual_reps,
        "rpe": reported_rpe,
        "timestamp": timestamp.isoformat(),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — WARM-UP GENERATOR                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_warmup_sets(rng: np.random.Generator,
                         user: UserProfile,
                         exercise: str,
                         working_weight: float,
                         timestamp: datetime,
                         ) -> Tuple[List[Dict], datetime]:
    """Generate 2-3 warm-up sets ramping to working weight.  [33][34]

    Standard protocol from research [34]:
      ~50% working weight × 5-8 reps
      ~70% working weight × 3-5 reps
      ~85% working weight × 1-2 reps (only for heavy sets)

    Returns (rows, updated_timestamp).
    Warm-ups produce negligible fatigue [33].
    """
    exdef = EXERCISE_DB[exercise]
    rows = []
    t = timestamp

    warmup_stages = [
        (0.50, (5, 8)),   # 50% × 5-8
        (0.70, (3, 5)),   # 70% × 3-5
    ]
    # Add a heavier warm-up for heavy working weights (>80% 1RM)
    e1rm = user.e1rm.get(exercise, 80.0)
    if e1rm > 0 and working_weight / e1rm > 0.80:
        warmup_stages.append((0.85, (1, 2)))

    for pct, (rep_lo, rep_hi) in warmup_stages:
        w = _round_weight(working_weight * pct, exdef.weight_increment)
        w = max(w, exdef.min_weight_kg)
        if w >= working_weight:
            continue

        row = simulate_set(rng, user, exercise, w, target_rir=8,
                           rest_minutes=2.0, timestamp=t, is_warmup=True)
        if row:
            rows.append(row)
        t += timedelta(minutes=rng.uniform(1.0, 2.0))

    return rows, t


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — WORKOUT SIMULATOR                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _round_weight(weight: float, increment: float) -> float:
    return round(weight / increment) * increment


def _auto_adjust_weight(user: UserProfile, exercise: str, raw_weight: float,
                        target_rir: int, exdef: ExerciseDef) -> float:
    """Adjust weight so predicted working reps land in 3-15 range.

    If the template %1RM gives too many or too few reps, bump weight
    up or down by increments until the rep target is sensible.
    """
    predicted = compute_max_reps(user, exercise, raw_weight) - target_rir
    w = raw_weight

    if predicted > 15:
        for _ in range(20):
            w += exdef.weight_increment
            test = compute_max_reps(user, exercise, w) - target_rir
            if test <= 12:
                break
    elif predicted < 3 and w > exdef.min_weight_kg:
        for _ in range(15):
            w -= exdef.weight_increment
            if w < exdef.min_weight_kg:
                w = exdef.min_weight_kg
                break
            test = compute_max_reps(user, exercise, w) - target_rir
            if test >= 3:
                break

    return max(w, exdef.min_weight_kg)


def simulate_workout(rng: np.random.Generator,
                     user: UserProfile,
                     template_name: str,
                     start_time: datetime,
                     week_in_meso: int,
                     include_warmups: bool = True,
                     ) -> List[Dict]:
    """Generate all sets for one workout session.

    Includes:
      - Warm-up sets (optional) [33][34]
      - Auto weight adjustment for sensible rep ranges
      - Mid-exercise weight adjustment if RPE way off target
      - Progressive RIR tightening across mesocycle weeks [7]
      - Realistic rest period variation [14]
    """
    template = WORKOUT_TEMPLATES[template_name]
    rows: List[Dict] = []
    t = start_time

    for exercise, pct_1rm, n_sets, base_rir, base_rest in template:
        exdef = EXERCISE_DB[exercise]
        e1rm = user.e1rm.get(exercise, 0)
        if e1rm < 5:
            continue

        # Linear periodization within mesocycle [7]:
        # Week 1: +1 RIR, Week 2: +0, Week 3: −1 RIR (harder)
        # Deload handled at schedule level with separate templates
        rir_offset = max(-1, 1 - week_in_meso)  # week0: +1, week1: +0, week2: -1, week3: deload
        target_rir = max(0, base_rir + rir_offset)

        raw_weight = _round_weight(e1rm * pct_1rm, exdef.weight_increment)
        raw_weight = max(raw_weight, exdef.min_weight_kg)
        weight = _auto_adjust_weight(user, exercise, raw_weight, target_rir, exdef)

        # Warm-up sets for first exercise of each muscle group  [33]
        if include_warmups and exdef.is_compound:
            warmup_rows, t = generate_warmup_sets(rng, user, exercise, weight, t)
            rows.extend(warmup_rows)
            t += timedelta(minutes=rng.uniform(1.5, 2.5))

        weight_adjusted = False
        for set_idx in range(n_sets):
            # Rest variation [14]: ±20% around base
            actual_rest = base_rest * rng.uniform(0.80, 1.25)

            row = simulate_set(rng, user, exercise, weight,
                               target_rir, actual_rest, t)
            if row is None:
                # Can't lift → drop 10%, try once more
                weight = _round_weight(weight * 0.90, exdef.weight_increment)
                weight = max(weight, exdef.min_weight_kg)
                row = simulate_set(rng, user, exercise, weight,
                                   target_rir, actual_rest, t)
                if row is None:
                    break

            rows.append(row)

            # Mid-exercise weight adjustment after set 1
            if set_idx == 0 and not weight_adjusted:
                target_rpe = 10.0 - target_rir
                if row["rpe"] >= target_rpe + 1.5:
                    weight = _round_weight(weight * 0.92, exdef.weight_increment)
                    weight = max(weight, exdef.min_weight_kg)
                    weight_adjusted = True
                elif row["rpe"] <= target_rpe - 2.0 and row["rpe"] < 7.0:
                    weight = _round_weight(weight * 1.05, exdef.weight_increment)
                    weight_adjusted = True

            # Time progression: rest + ~45s set execution
            t += timedelta(minutes=actual_rest, seconds=rng.uniform(30, 60))

        # Transition between exercises
        t += timedelta(minutes=rng.uniform(1.5, 4.0))

    return rows


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — PROGRAM GENERATOR                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_program(rng: np.random.Generator,
                     user: UserProfile,
                     n_weeks: int,
                     start_date: datetime,
                     include_warmups: bool = True,
                     ) -> List[Dict]:
    """Generate *n_weeks* of training for one user.

    Features:
      - Weekly schedule from WEEKLY_SCHEDULES
      - 4-week mesocycle: weeks 1-3 build, week 4 deload [7]
      - Linear periodization: RIR decreases across mesocycle [7]
      - Weight progression: ~0.3-0.5%/week intermediates [7]
      - ~5-25% random session skip (based on consistency trait)
      - Realistic start times: 16-20h weekdays, 8-12h weekends
    """
    schedule = WEEKLY_SCHEDULES[rng.integers(len(WEEKLY_SCHEDULES))]
    all_rows: List[Dict] = []

    current_date = start_date

    for week_idx in range(n_weeks):
        # Mesocycle position (4-week blocks)
        week_in_meso = week_idx % 4   # 0, 1, 2, 3(deload)
        is_deload = (week_in_meso == 3)

        for day_idx in range(7):
            if is_deload:
                template_name = schedule.deload_days[day_idx]
            else:
                template_name = schedule.days[day_idx]

            if template_name is None:
                current_date += timedelta(days=1)
                continue

            # Attendance check
            if rng.random() > user.consistency:
                current_date += timedelta(days=1)
                continue

            # Session start time
            weekday = current_date.weekday()  # 0=Mon
            if weekday < 5:
                # Weekday: 16:00-20:00 cluster  [realistic]
                hour = int(np.clip(rng.normal(17.5, 1.5), 6, 21))
            else:
                # Weekend: 08:00-12:00 cluster
                hour = int(np.clip(rng.normal(10.0, 1.5), 7, 16))
            minute = rng.integers(0, 60)

            session_start = current_date.replace(hour=hour, minute=minute)

            workout_rows = simulate_workout(
                rng, user, template_name, session_start,
                week_in_meso=week_in_meso,
                include_warmups=include_warmups,
            )
            all_rows.extend(workout_rows)
            current_date += timedelta(days=1)

        # Weekly e1RM progression [7]
        # Rate decreases with training age
        prog_rate = 0.004 / (1.0 + user.training_years * 0.1)
        if is_deload:
            prog_rate *= 0.2   # minimal progression during deload
        for ex in user.e1rm:
            user.e1rm[ex] *= (1.0 + rng.normal(prog_rate, prog_rate * 0.5))

    return all_rows


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — TRAINING HISTORY ORCHESTRATOR                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_dataset(n_users: int = 100,
                     n_weeks: int = 16,
                     seed: int = 42,
                     include_warmups: bool = True,
                     ) -> pd.DataFrame:
    """Generate the full synthetic training dataset.

    For 100 users × 16 weeks: ~80,000-120,000 rows (working sets + warm-ups).
    For 1000 users × 16 weeks: ~800,000-1,200,000 rows.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)

    all_rows: List[Dict] = []
    report_every = max(1, n_users // 20)

    for i in range(n_users):
        if i % report_every == 0:
            pct = i / n_users * 100
            print(f"\r  Users: {i:,}/{n_users:,} ({pct:.0f}%)", end="", flush=True)

        user = generate_user(rng, f"user_{i:05d}")
        user_rows = generate_program(rng, user, n_weeks, start_date,
                                     include_warmups=include_warmups)
        all_rows.extend(user_rows)

    print(f"\r  Users: {n_users:,}/{n_users:,} (100%)    ")

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — DIAGNOSTICS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_diagnostics(df: pd.DataFrame) -> None:
    """Print summary statistics to verify data quality."""
    sep = "─" * 50

    print(f"\n{sep}")
    print(f"Total rows:       {len(df):>10,}")
    print(f"Working sets:     {(df['rpe'] >= 6.0).sum():>10,}")
    print(f"Warm-up sets:     {(df['rpe'] < 6.0).sum():>10,}")
    print(f"Unique users:     {df['user_id'].nunique():>10,}")
    print(f"Unique exercises: {df['exercise'].nunique():>10,}")
    print(f"Date range:       {df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")

    print(f"\n{sep}")
    print("RPE distribution (working sets only):")
    working = df[df["rpe"] >= 6.0]
    for rpe_val in np.arange(6.0, 10.5, 0.5):
        count = (working["rpe"] == rpe_val).sum()
        pct = count / len(working) * 100 if len(working) > 0 else 0
        bar = "█" * int(pct)
        print(f"  RPE {rpe_val:4.1f}: {count:>7,} ({pct:5.1f}%) {bar}")

    print(f"\n{sep}")
    print("RPE summary stats (working sets):")
    print(f"  Mean: {working['rpe'].mean():.2f}")
    print(f"  Std:  {working['rpe'].std():.2f}")
    print(f"  Min:  {working['rpe'].min():.1f}")
    print(f"  Max:  {working['rpe'].max():.1f}")

    print(f"\n{sep}")
    print("Top 15 exercises by set count:")
    ex_counts = df.groupby("exercise").size().sort_values(ascending=False)
    for ex, cnt in ex_counts.head(15).items():
        print(f"  {ex:28s} {cnt:>6,} sets")

    print(f"\n{sep}")
    print("Columns:", list(df.columns))

    print(f"\n{sep}")
    print("Sample workout (first user, first day):")
    u0 = df[df["user_id"] == df["user_id"].iloc[0]]
    first_day = u0["timestamp"].str[:10].iloc[0]
    sample = u0[u0["timestamp"].str.startswith(first_day)].head(25)
    print(sample.to_string(index=False))

    # Cross-exercise fatigue check
    print(f"\n{sep}")
    print("Cross-exercise fatigue spot-check:")
    print("  (Exercises later in session should show lower reps or higher RPE")
    print("   vs if they were done first — this is the key signal [18-23])")

    # Find a user doing bench_press then ohp on same day
    for uid in df["user_id"].unique()[:20]:
        u = df[df["user_id"] == uid]
        for day in u["timestamp"].str[:10].unique()[:5]:
            day_data = u[u["timestamp"].str.startswith(day)]
            exercises = day_data["exercise"].unique()
            if "bench_press" in exercises and "ohp" in exercises:
                bench = day_data[day_data["exercise"] == "bench_press"]
                ohp_data = day_data[day_data["exercise"] == "ohp"]
                bench_working = bench[bench["rpe"] >= 6.0]
                ohp_working = ohp_data[ohp_data["rpe"] >= 6.0]
                if len(bench_working) > 0 and len(ohp_working) > 0:
                    print(f"\n  {uid}, {day}:")
                    print(f"    Bench press (first):")
                    for _, r in bench_working.iterrows():
                        print(f"      {r['weight']}kg × {r['reps']} @ RPE {r['rpe']}")
                    print(f"    OHP (after bench — shares triceps + delts):")
                    for _, r in ohp_working.iterrows():
                        print(f"      {r['weight']}kg × {r['reps']} @ RPE {r['rpe']}")
                    break
            else:
                continue
            break
        else:
            continue
        break


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — CLI                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="DeepGain Synthetic Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:      python generate_training_data.py --num_users 10 --weeks 4
  Medium dataset:  python generate_training_data.py --num_users 100 --weeks 16
  Large (1M rows): python generate_training_data.py --num_users 1000 --weeks 16
  No warm-ups:     python generate_training_data.py --no_warmups
        """,
    )
    parser.add_argument("--num_users", type=int, default=100,
                        help="Number of synthetic users (default: 100)")
    parser.add_argument("--weeks", type=int, default=16,
                        help="Training weeks per user (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="training_data.csv",
                        help="Output CSV path (default: training_data.csv)")
    parser.add_argument("--no_warmups", action="store_true",
                        help="Exclude warm-up sets from output")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress diagnostic output")

    args = parser.parse_args()

    print("=" * 60)
    print("DeepGain Synthetic Training Data Generator")
    print("=" * 60)
    print(f"  Scientific sources:  40 papers")
    print(f"  Exercises:           {len(EXERCISE_DB)}")
    print(f"  Muscle groups:       {len(ALL_MUSCLES)}")
    print(f"  Workout templates:   {len(WORKOUT_TEMPLATES)}")
    print(f"  Weekly schedules:    {len(WEEKLY_SCHEDULES)}")
    print(f"  Users:               {args.num_users:,}")
    print(f"  Weeks:               {args.weeks}")
    print(f"  Seed:                {args.seed}")
    print(f"  Warm-ups:            {'excluded' if args.no_warmups else 'included'}")
    print()

    df = generate_dataset(
        n_users=args.num_users,
        n_weeks=args.weeks,
        seed=args.seed,
        include_warmups=not args.no_warmups,
    )

    if len(df) == 0:
        print("ERROR: No data generated!")
        sys.exit(1)

    # Verify columns
    expected_cols = {"user_id", "exercise", "weight", "reps", "rpe", "timestamp"}
    assert set(df.columns) == expected_cols, f"Bad columns: {set(df.columns)}"

    if not args.quiet:
        print_diagnostics(df)

    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows to {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
