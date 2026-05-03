"""
knapsack_planner.py
===================
Dobiera ćwiczenia do sesji treningowej algorytmem plecakowym 0/1 (0/1 Knapsack).

Pojemność plecaka  : czas sesji [sekundy]
Wartość przedmiotu : stimulus_score = Σ(muscle_engagement × MPC_before) / Σ(engagement)
Ograniczenia twarde: MPC_after[muscle] ∈ [target_min, target_max]

Wagi treningowe i liczba reps wyznaczane wyłącznie przez inference.py:
  - weight_kg  = project_exercise_1rm(exercise, anchors) × intensity_factor
  - reps       = dobrane tak, by predict_rir() ≈ target_rir
Bez wzorów Epley / Brzycki.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Muscle involvement matrix
# Źródło: inference README (exercise → muscle → engagement_ratio)
# ---------------------------------------------------------------------------
MUSCLE_INVOLVEMENT: Dict[str, Dict[str, float]] = {
    # ── bench anchor ──────────────────────────────────────────────────────
    "bench_press":          {"chest": 0.80, "triceps": 0.60, "anterior_delts": 0.35, "lateral_delts": 0.15},
    "spoto_press":          {"chest": 0.70, "triceps": 0.55, "anterior_delts": 0.40},
    "decline_bench":        {"chest": 0.85, "triceps": 0.50, "anterior_delts": 0.20},
    "close_grip_bench":     {"chest": 0.65, "triceps": 0.85, "anterior_delts": 0.30},
    "chest_press_machine":  {"chest": 0.75, "triceps": 0.45, "anterior_delts": 0.35},
    "incline_bench":        {"chest": 0.75, "anterior_delts": 0.55, "triceps": 0.45, "lateral_delts": 0.15},
    "incline_bench_45":     {"chest": 0.70, "anterior_delts": 0.65, "triceps": 0.40},
    "dips":                 {"chest": 0.65, "triceps": 0.75, "anterior_delts": 0.40},
    "ohp":                  {"anterior_delts": 0.75, "lateral_delts": 0.55, "triceps": 0.55, "chest": 0.30},
    "skull_crusher":        {"triceps": 0.90, "anterior_delts": 0.10},
    "dumbbell_flyes":       {"chest": 0.85, "anterior_delts": 0.30},
    "pendlay_row":          {"lats": 0.60, "rhomboids": 0.75, "rear_delts": 0.55, "biceps": 0.45, "erectors": 0.50},
    "seal_row":             {"lats": 0.55, "rhomboids": 0.70, "rear_delts": 0.60, "biceps": 0.45},
    "lat_pulldown":         {"lats": 0.70, "rhomboids": 0.45, "biceps": 0.50, "rear_delts": 0.30},
    "pull_up":              {"lats": 0.80, "rhomboids": 0.55, "biceps": 0.55, "rear_delts": 0.30},
    "reverse_fly":          {"rear_delts": 0.85, "rhomboids": 0.60, "lateral_delts": 0.30},
    # ── squat anchor ──────────────────────────────────────────────────────
    "squat":                {"quads": 0.65, "glutes": 0.40, "hamstrings": 0.30, "erectors": 0.45, "adductors": 0.35, "abs": 0.20},
    "low_bar_squat":        {"quads": 0.50, "glutes": 0.50, "hamstrings": 0.40, "erectors": 0.50, "adductors": 0.40},
    "high_bar_squat":       {"quads": 0.75, "glutes": 0.45, "hamstrings": 0.25, "erectors": 0.35, "adductors": 0.30},
    "leg_press":            {"quads": 0.75, "glutes": 0.40, "hamstrings": 0.25},
    "leg_extension":        {"quads": 0.95},
    "bulgarian_split_squat":{"quads": 0.80, "glutes": 0.65, "hamstrings": 0.25},
    "leg_curl":             {"hamstrings": 0.95},
    # ── deadlift anchor ───────────────────────────────────────────────────
    "deadlift":             {"hamstrings": 0.60, "glutes": 0.55, "erectors": 0.70, "quads": 0.30, "lats": 0.40, "rhomboids": 0.30},
    "sumo_deadlift":        {"hamstrings": 0.45, "glutes": 0.50, "quads": 0.50, "adductors": 0.55, "erectors": 0.50},
    "rdl":                  {"hamstrings": 0.85, "glutes": 0.55, "erectors": 0.50},
    # ── bodyweight anchor ─────────────────────────────────────────────────
    "farmers_walk":         {"erectors": 0.45, "abs": 0.35, "glutes": 0.25, "calves": 0.30},
    "suitcase_carry":       {"abs": 0.75, "erectors": 0.55, "lats": 0.30},
    "ab_wheel":             {"abs": 0.90, "lats": 0.30, "triceps": 0.20},
    "plank":                {"abs": 0.75, "erectors": 0.30},
    "leg_raises":           {"abs": 0.85, "lats": 0.25},
    "dead_bug":             {"abs": 0.80, "erectors": 0.20},
    "bird_dog":             {"erectors": 0.70, "glutes": 0.50, "abs": 0.40},
    "trx_bodysaw":          {"abs": 0.85, "anterior_delts": 0.30, "lats": 0.30},
}

# Exercise metadata – typ, liczba serii, kandydaci na reps, czas seta, intensity factor
# intensity_factor: jaki % 1RM ustawiamy jako wagę treningową
EXERCISE_META: Dict[str, Dict] = {
    # id                        type          sets  reps_options           set_sec  intensity
    "bench_press":         {"type":"compound",   "sets":3, "reps":[3,5,6,8],     "set_sec":120, "intensity":0.78},
    "squat":               {"type":"compound",   "sets":3, "reps":[3,5,6,8],     "set_sec":150, "intensity":0.80},
    "low_bar_squat":       {"type":"compound",   "sets":3, "reps":[3,5,6,8],     "set_sec":150, "intensity":0.80},
    "high_bar_squat":      {"type":"compound",   "sets":3, "reps":[5,6,8],       "set_sec":150, "intensity":0.78},
    "deadlift":            {"type":"compound",   "sets":3, "reps":[3,5,6],       "set_sec":180, "intensity":0.82},
    "sumo_deadlift":       {"type":"compound",   "sets":3, "reps":[3,5,6],       "set_sec":180, "intensity":0.80},
    "spoto_press":         {"type":"variation",  "sets":3, "reps":[5,6,8],       "set_sec":120, "intensity":0.75},
    "decline_bench":       {"type":"variation",  "sets":3, "reps":[6,8,10],      "set_sec":120, "intensity":0.75},
    "close_grip_bench":    {"type":"variation",  "sets":3, "reps":[6,8,10],      "set_sec":120, "intensity":0.75},
    "incline_bench":       {"type":"variation",  "sets":3, "reps":[6,8,10],      "set_sec":120, "intensity":0.75},
    "incline_bench_45":    {"type":"variation",  "sets":3, "reps":[6,8,10],      "set_sec":120, "intensity":0.72},
    "ohp":                 {"type":"variation",  "sets":3, "reps":[5,6,8,10],    "set_sec":105, "intensity":0.72},
    "pendlay_row":         {"type":"variation",  "sets":3, "reps":[5,6,8],       "set_sec":105, "intensity":0.78},
    "seal_row":            {"type":"variation",  "sets":3, "reps":[6,8,10],      "set_sec":90,  "intensity":0.75},
    "chest_press_machine": {"type":"isolation",  "sets":3, "reps":[10,12,15],    "set_sec":75,  "intensity":0.70},
    "dips":                {"type":"isolation",  "sets":3, "reps":[8,10,12],     "set_sec":90,  "intensity":0.72},
    "lat_pulldown":        {"type":"isolation",  "sets":3, "reps":[8,10,12],     "set_sec":75,  "intensity":0.68},
    "pull_up":             {"type":"isolation",  "sets":3, "reps":[5,6,8,10],    "set_sec":90,  "intensity":0.70},
    "dumbbell_flyes":      {"type":"isolation",  "sets":3, "reps":[10,12,15],    "set_sec":75,  "intensity":0.55},
    "skull_crusher":       {"type":"isolation",  "sets":3, "reps":[10,12,15],    "set_sec":60,  "intensity":0.55},
    "reverse_fly":         {"type":"isolation",  "sets":3, "reps":[12,15],       "set_sec":60,  "intensity":0.50},
    "leg_press":           {"type":"isolation",  "sets":3, "reps":[8,10,12],     "set_sec":90,  "intensity":0.70},
    "leg_curl":            {"type":"isolation",  "sets":3, "reps":[10,12,15],    "set_sec":60,  "intensity":0.65},
    "leg_extension":       {"type":"isolation",  "sets":3, "reps":[10,12,15],    "set_sec":60,  "intensity":0.65},
    "bulgarian_split_squat":{"type":"isolation", "sets":3, "reps":[8,10,12],     "set_sec":120, "intensity":0.65},
    "rdl":                 {"type":"isolation",  "sets":3, "reps":[8,10,12],     "set_sec":120, "intensity":0.68},
    "farmers_walk":        {"type":"core",       "sets":2, "reps":[10,15],       "set_sec":60,  "intensity":0.60},
    "suitcase_carry":      {"type":"core",       "sets":2, "reps":[10,15],       "set_sec":60,  "intensity":0.40},
    "ab_wheel":            {"type":"core",       "sets":2, "reps":[8,10,12],     "set_sec":60,  "intensity":0.40},
    "plank":               {"type":"core",       "sets":2, "reps":[10,15,20],    "set_sec":45,  "intensity":0.30},
    "leg_raises":          {"type":"core",       "sets":2, "reps":[10,12,15],    "set_sec":60,  "intensity":0.30},
    "dead_bug":            {"type":"core",       "sets":2, "reps":[8,10,12],     "set_sec":45,  "intensity":0.25},
    "bird_dog":            {"type":"core",       "sets":2, "reps":[8,10,12],     "set_sec":45,  "intensity":0.25},
    "trx_bodysaw":         {"type":"core",       "sets":2, "reps":[10,12],       "set_sec":60,  "intensity":0.30},
}

# Ćwiczenia "główne" – jeden z trójboju i ich bliskie warianty.
# Planner zawsze wybiera dokładnie jedno z tej puli jako pierwsze ćwiczenie sesji.
MAIN_EXERCISES = {
    "squat", "low_bar_squat", "high_bar_squat",
    "bench_press",
    "deadlift", "sumo_deadlift",
}

# Default target MPC zones [min_after, max_after]
DEFAULT_TARGET_ZONES: Dict[str, List[float]] = {
    "chest":          [0.55, 0.85],
    "anterior_delts": [0.55, 0.85],
    "lateral_delts":  [0.55, 0.85],
    "rear_delts":     [0.55, 0.85],
    "rhomboids":      [0.55, 0.85],
    "triceps":        [0.45, 0.80],
    "biceps":         [0.45, 0.80],
    "lats":           [0.60, 0.85],
    "quads":          [0.60, 0.85],
    "hamstrings":     [0.60, 0.85],
    "glutes":         [0.60, 0.85],
    "adductors":      [0.50, 0.85],
    "erectors":       [0.60, 0.85],
    "calves":         [0.45, 0.80],
    "abs":            [0.50, 0.85],
}

# Bodyweight fallback [kg] dla ćwiczeń bez 1RM (project_exercise_1rm → None)
_BODYWEIGHT_FALLBACK_KG = 40.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExerciseBlock:
    """Jeden blok ćwiczenia (N serii) jako przedmiot plecaka."""
    exercise_id: str
    weight_kg: float          # waga treningowa [kg]
    reps: int                 # reps per set
    sets_count: int
    rest_sec: int             # odpoczynek między seriami [s]
    predicted_rir: float      # z predict_rir()
    stimulus_score: float     # wartość w plecaku
    time_cost_sec: int        # całkowity czas bloku: sets × set_sec + (sets-1) × rest
    ex_type: str              # compound / variation / isolation / core
    primary_muscles: List[str] = field(default_factory=list)
    secondary_muscles: List[str] = field(default_factory=list)

    # ---------- helpers --------

    def to_history_dicts(self, base_ts: datetime) -> List[dict]:
        """Generuje wpisy historii treningowej (format inference.predict_mpc)."""
        meta = EXERCISE_META.get(self.exercise_id, {})
        set_sec = meta.get("set_sec", 90)
        entries = []
        t = base_ts
        for _ in range(self.sets_count):
            entries.append({
                "exercise": self.exercise_id,
                "weight_kg": self.weight_kg,
                "reps": self.reps,
                "rir": max(0, int(round(self.predicted_rir))),
                "timestamp": t.isoformat(),
            })
            t += timedelta(seconds=set_sec + self.rest_sec)
        return entries

    def __repr__(self) -> str:
        return (
            f"ExerciseBlock({self.exercise_id!r}, "
            f"{self.sets_count}×{self.reps}@{self.weight_kg:.1f}kg, "
            f"RIR≈{self.predicted_rir:.1f}, score={self.stimulus_score:.3f}, "
            f"time={self.time_cost_sec}s)"
        )


@dataclass
class KnapsackPlan:
    """Wynik planowania – lista bloków + metadane."""
    blocks: List[ExerciseBlock]
    total_time_sec: int
    total_stimulus: float
    mpc_before: Dict[str, float]
    mpc_after: Dict[str, float]
    constraint_violations: List[str]   # mięśnie poza target zone
    notes: List[str]                   # pełny log stanu mięśni

    def summary(self) -> str:
        lines = [
            f"KnapsackPlan — {len(self.blocks)} exercises, "
            f"{self.total_time_sec // 60} min, "
            f"stimulus={self.total_stimulus:.3f}",
        ]
        for b in self.blocks:
            lines.append(
                f"  {b.exercise_id:28s}  {b.sets_count}×{b.reps} @ {b.weight_kg:.1f} kg"
                f"  RIR≈{b.predicted_rir:.1f}  score={b.stimulus_score:.3f}"
            )
        if self.constraint_violations:
            lines.append("Violations:")
            lines.extend(f"  {v}" for v in self.constraint_violations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KnapsackPlanner
# ---------------------------------------------------------------------------

class KnapsackPlanner:
    """
    Dobiera ćwiczenia algorytmem 0/1 Knapsack.

    Parameters
    ----------
    model
        Model załadowany przez inference.load_model().
    strength_anchors
        {"bench_press": float, "squat": float, "deadlift": float}
    target_zones
        Docelowe zakresy MPC po treningu (patrz DEFAULT_TARGET_ZONES).
    rest_between_sets_sec
        Odpoczynek między seriami [s].
    time_resolution_sec
        Dyskretyzacja czasu w DP [s] — kompromis między precyzją a szybkością.
    """

    def __init__(
        self,
        model,
        strength_anchors: Dict[str, float],
        target_zones: Optional[Dict[str, List[float]]] = None,
        rest_between_sets_sec: int = 120,
        time_resolution_sec: int = 60,
    ):
        # Importujemy tu, żeby nie wymagać inference.py przy samym imporcie modułu
        import inference as _inf
        self._inf = _inf
        self.model = model
        self.strength_anchors = strength_anchors
        self.target_zones = target_zones or deepcopy(DEFAULT_TARGET_ZONES)
        self.rest_sec = rest_between_sets_sec
        self.resolution = time_resolution_sec

        # Zbuduj zbiór ćwiczeń znanych modelowi
        known = set(self._inf.get_exercises())
        self._known_exercises = known & set(EXERCISE_META.keys())
        logger.info(
            f"KnapsackPlanner: {len(self._known_exercises)} exercises "
            f"known to model + defined in EXERCISE_META"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        user_history: List[dict],
        time_budget_sec: int = 3600,
        target_rir: int = 2,
        exclusions: Optional[List[str]] = None,
        now: Optional[datetime] = None,
    ) -> KnapsackPlan:
        """
        Planuje sesję treningową w dwóch fazach:

        Faza 1 – Ćwiczenie główne (zawsze jedno z trójboju)
            Wybiera najlepsze ćwiczenie z MAIN_EXERCISES wg stimulus_score.
            Odejmuje jego czas od budżetu.

        Faza 2 – Ćwiczenia akcesoryjne (knapsack DP)
            Na pozostały czas dobiera optymalny zestaw z puli variation +
            isolation + core (max 2 core).

        Parameters
        ----------
        user_history
            Historia treningów: [{exercise, weight_kg, reps, rir, timestamp}, ...]
        time_budget_sec
            Czas sesji [s].
        target_rir
            Docelowe RIR. Waga dobierana binarnie tak, by predict_rir ≈ target_rir.
        exclusions
            exercise_id do wykluczenia (kontuzje, brak sprzętu).
        now
            Timestamp sesji (default: datetime.now()).
        """
        now = now or datetime.now()
        exclusions = set(exclusions or [])

        # ── 1. MPC przed treningiem ──────────────────────────────────────────
        mpc_before = self._inf.predict_mpc(
            self.model,
            user_history=user_history,
            timestamp=now.isoformat(),
            strength_anchors=self.strength_anchors,
        )

        # ── 2. Faza 1: wybierz ćwiczenie główne z trójboju ──────────────────
        main_block = self._select_main_exercise(
            mpc_state=mpc_before,
            target_rir=target_rir,
            exclusions=exclusions,
            time_budget_sec=time_budget_sec,
        )

        selected = []
        remaining_time = time_budget_sec

        if main_block is not None:
            selected.append(main_block)
            remaining_time -= main_block.time_cost_sec
            logger.info(f"Main: {main_block.exercise_id} @ {main_block.weight_kg}kg")

        # ── 3. Faza 2: knapsack DP na ćwiczenia akcesoryjne ─────────────────
        # Wyklucz ćwiczenia główne z puli akcesoryjnej
        accessory_exclusions = exclusions | MAIN_EXERCISES
        if main_block:
            accessory_exclusions = accessory_exclusions | {main_block.exercise_id}

        accessories = self._build_candidates(
            mpc_state=mpc_before,
            target_rir=target_rir,
            exclusions=accessory_exclusions,
            time_budget_sec=remaining_time,
        )

        acc_selected = self._knapsack_dp(accessories, remaining_time)
        selected.extend(acc_selected)

        # ── 4. Repair constraints ────────────────────────────────────────────
        all_candidates = ([main_block] if main_block else []) + accessories
        selected = self._repair_constraints(
            selected=selected,
            candidates=all_candidates,
            user_history=user_history,
            mpc_before=mpc_before,
            time_budget_sec=time_budget_sec,
            now=now,
        )

        # ── 5. Finalna symulacja i ewaluacja ────────────────────────────────
        mpc_after = self._simulate_mpc(selected, user_history, now)
        violations, notes = self._evaluate_constraints(mpc_before, mpc_after)

        total_time = sum(b.time_cost_sec for b in selected)
        total_stimulus = sum(b.stimulus_score for b in selected)

        return KnapsackPlan(
            blocks=selected,
            total_time_sec=total_time,
            total_stimulus=total_stimulus,
            mpc_before=mpc_before,
            mpc_after=mpc_after,
            constraint_violations=violations,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Step 1: Candidate generation
    # ------------------------------------------------------------------

    def _select_main_exercise(
        self,
        mpc_state: Dict[str, float],
        target_rir: int,
        exclusions: set,
        time_budget_sec: int,
    ) -> Optional["ExerciseBlock"]:
        """
        Wybiera najlepsze ćwiczenie główne z trójboju (MAIN_EXERCISES).

        Kryterium: najwyższy stimulus_score (ważona MPC_before × engagement).
        Jeśli żadne ćwiczenie główne nie jest dostępne lub nie mieści się w czasie
        → zwraca None (sesja będzie składać się wyłącznie z akcesoriów).
        """
        best: Optional[ExerciseBlock] = None
        best_score = -1.0

        available_main = (
            (MAIN_EXERCISES & self._known_exercises) - exclusions
        )

        for ex_id in available_main:
            meta = EXERCISE_META[ex_id]
            sets    = meta["sets"]
            set_sec = meta["set_sec"]
            time_cost = sets * set_sec + (sets - 1) * self.rest_sec

            if time_cost > time_budget_sec:
                continue

            weight_kg, reps, pred_rir = self._tune_weight_and_reps(
                ex_id, mpc_state, target_rir
            )
            score = self._compute_stimulus(ex_id, mpc_state)

            involvement = MUSCLE_INVOLVEMENT.get(ex_id, {})
            sorted_m = sorted(involvement.items(), key=lambda kv: kv[1], reverse=True)
            primary   = [m for m, r in sorted_m[:2] if r >= 0.40]
            secondary = [m for m, r in sorted_m[2:] if r >= 0.20]

            if score > best_score:
                best_score = score
                best = ExerciseBlock(
                    exercise_id=ex_id,
                    weight_kg=weight_kg,
                    reps=reps,
                    sets_count=sets,
                    rest_sec=self.rest_sec,
                    predicted_rir=pred_rir,
                    stimulus_score=score,
                    time_cost_sec=time_cost,
                    ex_type=meta["type"],
                    primary_muscles=primary,
                    secondary_muscles=secondary,
                )

        return best

    def _build_candidates(
        self,
        mpc_state: Dict[str, float],
        target_rir: int,
        exclusions: set,
        time_budget_sec: int,
    ) -> List[ExerciseBlock]:
        candidates = []
        for ex_id in sorted(self._known_exercises):
            if ex_id in exclusions:
                continue
            meta = EXERCISE_META[ex_id]

            # Wyznacz weight i reps przez model (bez Brzycki)
            weight_kg, reps, pred_rir = self._tune_weight_and_reps(
                ex_id, mpc_state, target_rir
            )

            # Czas bloku
            sets = meta["sets"]
            set_sec = meta["set_sec"]
            time_cost = sets * set_sec + (sets - 1) * self.rest_sec
            if time_cost > time_budget_sec:
                # Spróbuj z jedną serią
                sets = 1
                time_cost = set_sec

            if time_cost > time_budget_sec:
                continue  # Nawet jedna seria nie mieści się

            # Stimulus score = ważona średnia (engagement × MPC_before)
            score = self._compute_stimulus(ex_id, mpc_state)

            involvement = MUSCLE_INVOLVEMENT.get(ex_id, {})
            sorted_m = sorted(involvement.items(), key=lambda kv: kv[1], reverse=True)
            primary = [m for m, r in sorted_m[:2] if r >= 0.40]
            secondary = [m for m, r in sorted_m[2:] if r >= 0.20]

            candidates.append(ExerciseBlock(
                exercise_id=ex_id,
                weight_kg=weight_kg,
                reps=reps,
                sets_count=sets,
                rest_sec=self.rest_sec,
                predicted_rir=pred_rir,
                stimulus_score=score,
                time_cost_sec=time_cost,
                ex_type=meta["type"],
                primary_muscles=primary,
                secondary_muscles=secondary,
            ))

        # Sortuj malejąco po gęstości wartości (score / czas) – heurystyka dla DP
        # Ogranicz liczbę ćwiczeń core do 2 – zapobiega stackowaniu
        # 5+ seriami abs naraz (overfatigue)
        MAX_CORE = 2
        core_count = 0
        filtered = []
        for b in candidates:
            if b.ex_type == "core":
                if core_count < MAX_CORE:
                    filtered.append(b)
                    core_count += 1
            else:
                filtered.append(b)
        candidates = filtered

        candidates.sort(
            key=lambda b: b.stimulus_score / max(b.time_cost_sec, 1),
            reverse=True,
        )
        return candidates

    # Stała liczba powtórzeń dla wszystkich ćwiczeń
    FIXED_REPS = 10

    def _tune_weight_and_reps(
        self,
        ex_id: str,
        mpc_state: Dict[str, float],
        target_rir: int,
    ) -> Tuple[float, int, float]:
        """
        Wyznacza (weight_kg, reps=10, predicted_rir) bez wzorów Brzycki.

        Reps jest zawsze stałe (FIXED_REPS=10).
        Waga dobierana binarnym przeszukiwaniem przedziału [0.30×1RM, 1.0×1RM]
        tak, żeby predict_rir(model, mpc, exercise, weight, 10) ≈ target_rir.
        """
        meta = EXERCISE_META[ex_id]

        # 1RM z modelu (bez Brzycki)
        e1rm = self._inf.project_exercise_1rm(
            ex_id, strength_anchors=self.strength_anchors
        )
        if e1rm is None or e1rm <= 0:
            e1rm = _BODYWEIGHT_FALLBACK_KG / max(meta["intensity"], 0.1)

        reps = self.FIXED_REPS

        # Ćwiczenia bez 1RM (bodyweight/core) → stała niska waga, bez binary search
        if e1rm is None or e1rm <= 0:
            fixed_weight = max(5.0, _BODYWEIGHT_FALLBACK_KG * meta["intensity"])
            fixed_weight = round(fixed_weight / 2.5) * 2.5
            try:
                pred = self._inf.predict_rir(
                    self.model, state=mpc_state, exercise=ex_id,
                    weight=fixed_weight, reps=reps,
                    strength_anchors=self.strength_anchors,
                )
            except Exception:
                pred = float(target_rir)
            return fixed_weight, reps, pred

        # Ćwiczenia z 1RM → binary search wagi w [30% 1RM, 95% 1RM]
        w_low  = max(5.0,  e1rm * 0.30)
        w_high = min(500.0, e1rm * 0.95)

        best_weight = round(e1rm * meta["intensity"] / 2.5) * 2.5
        best_pred   = float(target_rir)
        best_diff   = float("inf")

        for _ in range(12):
            w_mid = (w_low + w_high) / 2
            try:
                pred = self._inf.predict_rir(
                    self.model, state=mpc_state, exercise=ex_id,
                    weight=w_mid, reps=reps,
                    strength_anchors=self.strength_anchors,
                )
                diff = abs(pred - target_rir)
                if diff < best_diff:
                    best_diff   = diff
                    best_weight = w_mid
                    best_pred   = pred
                if pred > target_rir:
                    w_low = w_mid
                else:
                    w_high = w_mid
            except Exception as e:
                logger.debug(f"predict_rir failed {ex_id} w={w_mid:.1f}: {e}")
                break

        best_weight = max(5.0, round(best_weight / 2.5) * 2.5)
        return best_weight, reps, best_pred

    def _compute_stimulus(
        self,
        ex_id: str,
        mpc_state: Dict[str, float],
    ) -> float:
        """
        Stimulus score = Σ(engagement × MPC_before) / Σ(engagement)

        Interpretacja: jak bardzo świeże są mięśnie zaangażowane w to ćwiczenie.
        1.0 = wszystkie docelowe mięśnie w pełni zregenerowane.
        """
        involvement = MUSCLE_INVOLVEMENT.get(ex_id, {})
        total_w = sum(involvement.values())
        if total_w == 0:
            return 0.0
        score = sum(
            ratio * mpc_state.get(muscle, 1.0)
            for muscle, ratio in involvement.items()
        ) / total_w
        return score

    # ------------------------------------------------------------------
    # Step 2: 0/1 Knapsack DP
    # ------------------------------------------------------------------

    def _knapsack_dp(
        self,
        candidates: List[ExerciseBlock],
        time_budget_sec: int,
    ) -> List[ExerciseBlock]:
        """
        Klasyczny 0/1 Knapsack DP.

        Pojemność C = time_budget_sec // resolution (buckety)
        Wartość    = stimulus_score
        Waga       = time_cost_sec // resolution

        Złożoność: O(n × C) — dla n≈34, C≈3600/60=60: trivial.
        """
        R = self.resolution
        C = time_budget_sec // R
        n = len(candidates)

        # dp[i][c] = max stimulus używając pierwszych i kandydatów, czas ≤ c bucketów
        dp = [[0.0] * (C + 1) for _ in range(n + 1)]

        # Ceiling division: item zajmuje ⌈time/R⌉ bucketów.
        # Dzięki temu suma bucket_weights × R ≥ suma time_cost_sec, czyli
        # łączny czas wybranych ćwiczeń nigdy nie przekroczy budżetu.
        def _w(item: ExerciseBlock) -> int:
            return max(1, (item.time_cost_sec + R - 1) // R)

        for i, item in enumerate(candidates, start=1):
            w = _w(item)
            for c in range(C + 1):
                dp[i][c] = dp[i - 1][c]
                if c >= w:
                    alt = dp[i - 1][c - w] + item.stimulus_score
                    if alt > dp[i][c]:
                        dp[i][c] = alt

        # Backtracking – odtwórz wybrany zbiór
        selected = []
        c = C
        for i in range(n, 0, -1):
            if dp[i][c] != dp[i - 1][c]:
                item = candidates[i - 1]
                selected.append(item)
                c -= _w(item)

        selected.reverse()  # zachowaj kolejność (compound → isolation → core)
        selected.sort(key=lambda b: ["compound", "variation", "isolation", "core"].index(b.ex_type)
                      if b.ex_type in ["compound","variation","isolation","core"] else 3)
        return selected

    # ------------------------------------------------------------------
    # Step 3: Constraint repair
    # ------------------------------------------------------------------

    def _repair_constraints(
        self,
        selected: List[ExerciseBlock],
        candidates: List[ExerciseBlock],
        user_history: List[dict],
        mpc_before: Dict[str, float],
        time_budget_sec: int,
        now: datetime,
        max_iterations: int = 5,
    ) -> List[ExerciseBlock]:
        """
        Jeśli symulacja MPC po wybranym planie narusza ograniczenia (overfatigue),
        iteracyjnie usuwa najgorszy blok i reruns knapsack na pozostałych kandydatach.
        """
        for iteration in range(max_iterations):
            mpc_after = self._simulate_mpc(selected, user_history, now)
            violations, _ = self._evaluate_constraints(mpc_before, mpc_after)
            overfatigue = [v for v in violations if "OVERFATIGUE" in v]

            if not overfatigue:
                return selected  # Brak naruszeń

            logger.warning(
                f"[repair iter={iteration+1}] Overfatigue violations: {overfatigue}"
            )

            # Znajdź blok odpowiadający za największe naruszenie
            worst_item = self._find_worst_offender(selected, mpc_before, mpc_after)
            if worst_item is None:
                break
            logger.info(f"  Removing offender: {worst_item.exercise_id}")

            # Usuń z wybranych i z kandydatów (nie wróci do puli)
            selected = [b for b in selected if b.exercise_id != worst_item.exercise_id]
            candidates = [c for c in candidates if c.exercise_id != worst_item.exercise_id]

            # Reruns DP na pozostałych kandydatach
            selected = self._knapsack_dp(candidates, time_budget_sec)

        return selected

    def _find_worst_offender(
        self,
        selected: List[ExerciseBlock],
        mpc_before: Dict[str, float],
        mpc_after: Dict[str, float],
    ) -> Optional[ExerciseBlock]:
        """
        Znajdź blok, który najbardziej przyczynia się do overfatigue
        (największe ważone naruszenie target_min).
        """
        worst_block = None
        worst_penalty = 0.0

        for block in selected:
            involvement = MUSCLE_INVOLVEMENT.get(block.exercise_id, {})
            penalty = 0.0
            for muscle, ratio in involvement.items():
                mpc_val = mpc_after.get(muscle, 1.0)
                target_min = self.target_zones.get(muscle, [0.55, 0.85])[0]
                if mpc_val < target_min:
                    penalty += (target_min - mpc_val) * ratio

            if penalty > worst_penalty:
                worst_penalty = penalty
                worst_block = block

        return worst_block

    # ------------------------------------------------------------------
    # Step 4: MPC simulation
    # ------------------------------------------------------------------

    def _simulate_mpc(
        self,
        blocks: List[ExerciseBlock],
        user_history: List[dict],
        now: datetime,
    ) -> Dict[str, float]:
        """
        Symuluje MPC po wykonaniu wszystkich bloków.
        Buduje historię sesji i wywołuje predict_mpc z przesuniętym timestamp.
        """
        if not blocks:
            return self._inf.predict_mpc(
                self.model,
                user_history=user_history,
                timestamp=now.isoformat(),
                strength_anchors=self.strength_anchors,
            )

        combined_history = list(user_history)
        sim_ts = now
        for block in blocks:
            entries = block.to_history_dicts(sim_ts)
            combined_history.extend(entries)
            sim_ts += timedelta(seconds=block.time_cost_sec)

        return self._inf.predict_mpc(
            self.model,
            user_history=combined_history,
            timestamp=sim_ts.isoformat(),
            strength_anchors=self.strength_anchors,
        )

    # ------------------------------------------------------------------
    # Step 5: Constraint evaluation
    # ------------------------------------------------------------------

    def _evaluate_constraints(
        self,
        mpc_before: Dict[str, float],
        mpc_after: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        violations = []
        notes = []

        for muscle in sorted(mpc_after):
            after = mpc_after[muscle]
            before = mpc_before.get(muscle, 1.0)
            zone = self.target_zones.get(muscle, [0.55, 0.85])
            t_min, t_max = zone
            was_worked = after < before - 0.02

            if after < t_min:
                msg = f"⚠ OVERFATIGUE  {muscle}: MPC={after:.2f} < min={t_min:.2f}"
                violations.append(msg)
                notes.append(msg)
            elif was_worked and after > t_max:
                msg = f"⚠ UNDERFATIGUE {muscle}: MPC={after:.2f} > max={t_max:.2f}"
                violations.append(msg)
                notes.append(msg)
            elif was_worked:
                notes.append(f"✓ OK           {muscle}: MPC={after:.2f} in [{t_min:.2f}, {t_max:.2f}]")

        return violations, notes
