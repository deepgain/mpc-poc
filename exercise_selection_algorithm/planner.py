"""
WorkoutPlanner - Silnik rekomendacyjny dla treningu siłowego.

Używa modelu DeepGain (via models_wrapper) jako black-box:
  - predict_mpc(history, now) → dict[muscle_id, capacity]
  - predict_rir(state, exercise, weight, reps) → float

Kluczowe konwencje:
  - MPC = CAPACITY w [0.1, 1.0]:
      1.0 = fresh (fully recovered)
      0.1 = exhausted
  - Fresh user → MPC = 1.0 dla wszystkich mięśni
  - Target zones: capacity PO treningu (nie zmęczenie)

Algorytm: Greedy selection z symulacją
  1. Dla każdego kandydata:
     a. Skonstruuj PlannedSet (reps dobrane pod target RIR via predict_rir)
     b. Dodaj do history, wywołaj predict_mpc
     c. Oblicz score: reward(wysoki MPC_before × engagement) - penalty(poza target zone)
  2. Wybierz kandydata z max score
  3. Powtórz dla n_compound + n_isolation ćwiczeń
  4. Core na końcu (jeśli czas pozwala)

Kolejność: compound → isolation → core
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from data_structures import (
    WorkoutSet, PlannedSet, PlanResult, PlannerConfig,
)
import models_wrapper


logger = logging.getLogger(__name__)


# ============================================================================
# Domyślne kategorie ćwiczeń (gdy catalog nie zawiera metadanych)
# Używane do podziału na compound / isolation / core
# ============================================================================
DEFAULT_EXERCISE_CATEGORIES = {
    # Compound (main)
    "squat":                   {"type": "compound", "time_sec": 150},
    "low_bar_squat":           {"type": "compound", "time_sec": 150},
    "bench_press":             {"type": "compound", "time_sec": 120},
    "deadlift":                {"type": "compound", "time_sec": 180},

    # Compound variations
    "high_bar_squat":          {"type": "compound_variation", "time_sec": 150},
    "close_grip_bench":        {"type": "compound_variation", "time_sec": 120},
    "spoto_press":             {"type": "compound_variation", "time_sec": 120},
    "incline_bench":           {"type": "compound_variation", "time_sec": 120},
    "incline_bench_45":        {"type": "compound_variation", "time_sec": 120},
    "dumbbell_fly":            {"type": "compound_variation", "time_sec": 90},
    "sumo_deadlift":           {"type": "compound_variation", "time_sec": 180},
    "front_squat":             {"type": "compound_variation", "time_sec": 150},

    # Isolation - lower body
    "bulgarian_split_squat":   {"type": "isolation", "time_sec": 120},
    "leg_press":               {"type": "isolation", "time_sec": 90},
    "romanian_deadlift":       {"type": "isolation", "time_sec": 120},
    "leg_curl":                {"type": "isolation", "time_sec": 60},
    "leg_extension":           {"type": "isolation", "time_sec": 60},

    # Isolation - upper body
    "chest_press_machine":     {"type": "isolation", "time_sec": 75},
    "dips":                    {"type": "isolation", "time_sec": 90},
    "ohp":                     {"type": "isolation", "time_sec": 105},
    "decline_bench":           {"type": "isolation", "time_sec": 90},
    "french_press":            {"type": "isolation", "time_sec": 60},

    # Isolation - back
    "pendlay_row":             {"type": "isolation", "time_sec": 105},
    "pull_ups":                {"type": "isolation", "time_sec": 90},
    "lat_pulldown":            {"type": "isolation", "time_sec": 75},
    "reverse_fly":             {"type": "isolation", "time_sec": 60},
    "seal_row":                {"type": "isolation", "time_sec": 90},

    # Core / stability
    "plank":                   {"type": "core", "time_sec": 45},
    "farmers_walk":            {"type": "core", "time_sec": 60},
    "hanging_leg_raise":       {"type": "core", "time_sec": 60},
    "ab_wheel_rollout":        {"type": "core", "time_sec": 60},
    "dead_bug":                {"type": "core", "time_sec": 45},
    "trx_bodysaw":             {"type": "core", "time_sec": 60},
    "suitcase_carry":          {"type": "core", "time_sec": 60},
    "bird_dog":                {"type": "core", "time_sec": 45},
}


class WorkoutPlanner:
    """
    Główna klasa planera treningowego.

    Public API:
      - plan() -> PlanResult
      - replan() -> PlanResult
      - estimate_1rm_from_history() -> dict
    """

    def __init__(
        self,
        config: PlannerConfig,
        model_handle: Optional[models_wrapper.ModelHandle] = None,
        exercise_catalog: Optional[Dict[str, Dict]] = None,
    ):
        """
        Args:
            config: PlannerConfig (target zones, defaults, etc.)
            model_handle: opcjonalnie konkretny handle do modelu
                          (default: globalny z models_wrapper.get_model())
            exercise_catalog: metadata ćwiczeń {ex_id: {type, time_sec}}
                              (default: DEFAULT_EXERCISE_CATEGORIES)
        """
        self.config = config
        self.model = model_handle or models_wrapper.get_model()
        self.using_real_model = models_wrapper.is_using_real_model()

        # Weź listę ćwiczeń z modelu (canonical source of truth)
        self.all_exercises = set(self.model.get_exercises())
        self.all_muscles = self.model.get_muscles()

        # Merge catalogu: domyślny + custom
        self.exercise_catalog = dict(DEFAULT_EXERCISE_CATEGORIES)
        if exercise_catalog:
            self.exercise_catalog.update(exercise_catalog)

        # Odfiltruj ćwiczenia - tylko te rozpoznawane przez model
        self.exercise_catalog = {
            k: v for k, v in self.exercise_catalog.items()
            if k in self.all_exercises
        }

        # Podział na typy
        self.compound_exercises = [
            ex_id for ex_id, meta in self.exercise_catalog.items()
            if meta.get("type") in ("compound", "compound_variation")
        ]
        self.isolation_exercises = [
            ex_id for ex_id, meta in self.exercise_catalog.items()
            if meta.get("type") == "isolation"
        ]
        self.core_exercises = [
            ex_id for ex_id, meta in self.exercise_catalog.items()
            if meta.get("type") == "core"
        ]

        model_type = "DeepGain (real)" if self.using_real_model else "Mock"
        logger.info(f"✓ Planner initialized ({model_type})")
        logger.info(f"  - {len(self.compound_exercises)} compound exercises")
        logger.info(f"  - {len(self.isolation_exercises)} isolation exercises")
        logger.info(f"  - {len(self.core_exercises)} core exercises")

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def plan(
        self,
        state: Optional[Dict[str, float]] = None,
        n_compound: int = 2,
        n_isolation: int = 3,
        available_time_sec: int = 3600,
        user_history: Optional[List[WorkoutSet]] = None,
        exclusions: Optional[List[str]] = None,
        preferences: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> PlanResult:
        """
        Główna funkcja planowania.

        Args:
            state: dict[muscle_id, MPC_capacity] - aktualny stan mięśni (1.0=fresh)
                   Jeśli None, obliczymy z user_history via predict_mpc()
            n_compound: ile ćwiczeń compound (A. Main + B. Variations)
            n_isolation: ile ćwiczeń isolation (C. Accessories)
            available_time_sec: czas sesji (sekund)
            user_history: historia treningów (do estymacji 1RM i MPC)
            exclusions: list[exercise_id] - czego nie uwzględniać
            preferences: dict - ulubione ćwiczenia, etc.
            now: referenced timestamp (default: datetime.now())

        Returns:
            PlanResult
        """
        exclusions = set(exclusions or [])
        preferences = preferences or {}
        user_history = user_history or []
        now = now or datetime.now()

        # Jeśli state nie podany, oblicz z historii
        if state is None:
            state = self._get_state_from_history(user_history, now)

        # Estymuj 1RM
        estimated_1rm = self.estimate_1rm_from_history(user_history)

        # Working state (będziemy modyfikować)
        current_state = dict(state)
        current_history = list(user_history)
        used_exercises = set()
        total_time = 0
        plan = []

        logger.info(f"\n=== Planning: {n_compound}C + {n_isolation}I, time={available_time_sec}s ===")

        # Phase 1: Compound (N ćwiczeń × sets_count serii)
        logger.info(f"\n--- Phase 1: Compound ({n_compound} exercises) ---")
        for i in range(n_compound):
            exercise_sets, new_state, new_time = self._select_and_expand_exercise(
                candidate_pool=self.compound_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(plan) + 1,
                time_offset=total_time,
            )

            if not exercise_sets:
                logger.warning(f"  ⚠ Could not select compound #{i+1}")
                break

            plan.extend(exercise_sets)
            used_exercises.add(exercise_sets[0].exercise_id)

            # Update state & history (wszystkie serie tego ćwiczenia)
            for ps in exercise_sets:
                ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                current_history.append(ws)
                total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
            current_state = new_state

            first_set = exercise_sets[0]
            logger.info(
                f"  {first_set.exercise_id}: {len(exercise_sets)} sets × "
                f"{first_set.reps}×{first_set.weight_kg}kg (pred RIR={first_set.predicted_rir:.1f})"
            )

        # Phase 2: Isolation
        logger.info(f"\n--- Phase 2: Isolation ({n_isolation} exercises) ---")
        for i in range(n_isolation):
            exercise_sets, new_state, new_time = self._select_and_expand_exercise(
                candidate_pool=self.isolation_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(plan) + 1,
                time_offset=total_time,
            )

            if not exercise_sets:
                logger.warning(f"  ⚠ Could not select isolation #{i+1}")
                break

            plan.extend(exercise_sets)
            used_exercises.add(exercise_sets[0].exercise_id)

            for ps in exercise_sets:
                ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                current_history.append(ws)
                total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
            current_state = new_state

            first_set = exercise_sets[0]
            logger.info(
                f"  {first_set.exercise_id}: {len(exercise_sets)} sets × "
                f"{first_set.reps}×{first_set.weight_kg}kg (pred RIR={first_set.predicted_rir:.1f})"
            )

        # Phase 3: Core (always last, only if time)
        if total_time < available_time_sec and self.core_exercises:
            logger.info(f"\n--- Phase 3: Core ---")
            exercise_sets, new_state, new_time = self._select_and_expand_exercise(
                candidate_pool=self.core_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(plan) + 1,
                time_offset=total_time,
            )

            if exercise_sets:
                plan.extend(exercise_sets)
                used_exercises.add(exercise_sets[0].exercise_id)

                for ps in exercise_sets:
                    ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                    current_history.append(ws)
                    total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
                current_state = new_state

                logger.info(f"  [Core] {exercise_sets[0].exercise_id}: {len(exercise_sets)} sets")

        # Finalna predykcja MPC
        final_mpc = self._call_predict_mpc(current_history, now + timedelta(seconds=total_time))

        # Walidacja target zones
        notes = self._validate_target_zones(final_mpc, state)

        return PlanResult(
            plan=plan,
            predicted_mpc_after=final_mpc,
            total_time_estimated_sec=total_time,
            notes=notes,
            used_real_model=self.using_real_model,
        )

    def replan(
        self,
        session_so_far: List[PlannedSet],
        remaining_n_compound: int,
        remaining_n_isolation: int,
        current_state: Optional[Dict[str, float]] = None,
        available_time_sec: int = 3600,
        user_history: Optional[List[WorkoutSet]] = None,
        exclusions: Optional[List[str]] = None,
        preferences: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> PlanResult:
        """
        Replanning w locie - po wykonaniu/odrzuceniu/modyfikacji serii.

        Args:
            session_so_far: już wykonane (lub zmodyfikowane) serie
            remaining_n_compound: ile compound jeszcze
            remaining_n_isolation: ile isolation jeszcze
            current_state: aktualny MPC capacity (jeśli None, obliczymy)
            available_time_sec: pozostały czas
            user_history: historia przed tą sesją (do 1RM)
            exclusions: do pominięcia
            preferences: preferencje
            now: timestamp

        Returns:
            PlanResult z pełnym planem (session_so_far + new_plan)
        """
        logger.info(f"\n=== REPLAN: {len(session_so_far)} done, need {remaining_n_compound}C + {remaining_n_isolation}I ===")

        exclusions = set(exclusions or [])
        preferences = preferences or {}
        user_history = user_history or []
        now = now or datetime.now()

        # Zbuduj aktualną historię (pre-session + already done w session)
        combined_history = list(user_history)
        for ps in session_so_far:
            combined_history.append(ps.to_workout_set())

        # Jeśli current_state nie podany, oblicz
        if current_state is None:
            current_state = self._call_predict_mpc(combined_history, now)

        # Estymuj 1RM
        estimated_1rm = self.estimate_1rm_from_history(combined_history)

        # Używane ćwiczenia (nie powtarzaj!)
        used_exercises = set(ps.exercise_id for ps in session_so_far)

        # Czas już zużyty
        total_time = sum(ps.estimated_time_sec for ps in session_so_far)

        # Planuj pozostałe
        new_plan = []
        working_state = dict(current_state)
        working_history = list(combined_history)

        # Compound
        for i in range(remaining_n_compound):
            exercise_sets, new_state, _ = self._select_and_expand_exercise(
                candidate_pool=self.compound_exercises,
                current_state=working_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=working_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(session_so_far) + len(new_plan) + 1,
                time_offset=total_time,
            )
            if not exercise_sets:
                break

            new_plan.extend(exercise_sets)
            used_exercises.add(exercise_sets[0].exercise_id)

            for ps in exercise_sets:
                ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                working_history.append(ws)
                total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
            working_state = new_state

            logger.info(f"  [+] {exercise_sets[0].exercise_id}: {len(exercise_sets)} sets (compound)")

        # Isolation
        for i in range(remaining_n_isolation):
            exercise_sets, new_state, _ = self._select_and_expand_exercise(
                candidate_pool=self.isolation_exercises,
                current_state=working_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=working_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(session_so_far) + len(new_plan) + 1,
                time_offset=total_time,
            )
            if not exercise_sets:
                break

            new_plan.extend(exercise_sets)
            used_exercises.add(exercise_sets[0].exercise_id)

            for ps in exercise_sets:
                ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                working_history.append(ws)
                total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
            working_state = new_state

            logger.info(f"  [+] {exercise_sets[0].exercise_id}: {len(exercise_sets)} sets (isolation)")

        # Core (jeśli czas)
        if total_time < available_time_sec and self.core_exercises:
            exercise_sets, new_state, _ = self._select_and_expand_exercise(
                candidate_pool=self.core_exercises,
                current_state=working_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=working_history,
                remaining_time=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
                now=now,
                preferences=preferences,
                start_order=len(session_so_far) + len(new_plan) + 1,
                time_offset=total_time,
            )
            if exercise_sets:
                new_plan.extend(exercise_sets)

                for ps in exercise_sets:
                    ws = ps.to_workout_set(timestamp=now + timedelta(seconds=total_time))
                    working_history.append(ws)
                    total_time += ps.estimated_time_sec + self.config.rest_between_sets_sec
                working_state = new_state

                logger.info(f"  [+] {exercise_sets[0].exercise_id}: {len(exercise_sets)} sets (core)")

        # Finalna predykcja
        final_mpc = self._call_predict_mpc(working_history, now + timedelta(seconds=total_time))
        notes = self._validate_target_zones(final_mpc, current_state)

        # Pełny plan (already done + new)
        full_plan = list(session_so_far) + new_plan

        return PlanResult(
            plan=full_plan,
            predicted_mpc_after=final_mpc,
            total_time_estimated_sec=total_time,
            notes=notes,
            used_real_model=self.using_real_model,
        )

    def estimate_1rm_from_history(
        self,
        user_history: List[WorkoutSet],
    ) -> Dict[str, float]:
        """
        Szacuj 1RM z historii (Brzycki formula).
            1RM ≈ weight × (1 + reps / 30)
        Dla nieznanych ćwiczeń użyj rozsądnych defaults.
        """
        estimated = defaultdict(float)

        for ws in user_history:
            reps = max(1, ws.reps)
            # Skoryguj na RIR: dodaj RIR do reps (jak efektywny failure)
            effective_reps = reps + (ws.rir or 0)
            one_rm = ws.weight_kg * (1.0 + effective_reps / 30.0)
            if one_rm > estimated[ws.exercise_id]:
                estimated[ws.exercise_id] = one_rm

        # Defaults dla ćwiczeń bez historii
        for ex_id in self.exercise_catalog:
            if ex_id not in estimated:
                ex_type = self.exercise_catalog[ex_id].get("type", "isolation")
                estimated[ex_id] = self._default_1rm(ex_id, ex_type)

        return dict(estimated)

    # ========================================================================
    # INTERNAL: Selection logic
    # ========================================================================

    def _select_and_expand_exercise(
        self,
        candidate_pool: List[str],
        current_state: Dict[str, float],
        used_exercises: set,
        exclusions: set,
        current_history: List[WorkoutSet],
        remaining_time: int,
        estimated_1rm: Dict[str, float],
        now: datetime,
        preferences: Dict,
        start_order: int,
        time_offset: int,
    ) -> Tuple[List[PlannedSet], Dict[str, float], int]:
        """
        Wybiera NAJLEPSZE ćwiczenie i generuje N serii tego ćwiczenia.

        Returns:
            (lista PlannedSetów [N serii], predicted_mpc_after, total_time_added)
            Pustą listę jeśli nic nie znaleziono.
        """
        best_score = -float('inf')
        best_exercise_id = None
        best_sets_list = None
        best_final_state = current_state

        favorite_set = set(preferences.get("favorites", []))
        avoid_set = set(preferences.get("avoid", []))

        rest_sec = self.config.rest_between_sets_sec

        for ex_id in candidate_pool:
            if ex_id in used_exercises or ex_id in exclusions:
                continue

            meta = self.exercise_catalog.get(ex_id, {})
            ex_type = meta.get("type", "isolation")
            time_per_set = meta.get("time_sec", 120)
            sets_count = self.config.get_sets_count(ex_type)

            # Czas całkowity: sets × time + (sets-1) × rest
            total_block_time = sets_count * time_per_set + (sets_count - 1) * rest_sec
            if total_block_time > remaining_time:
                # Spróbuj zmniejszyć liczbę serii
                if time_per_set > remaining_time:
                    continue
                sets_count = max(1, (remaining_time + rest_sec) // (time_per_set + rest_sec))
                total_block_time = sets_count * time_per_set + (sets_count - 1) * rest_sec
                if sets_count < 1:
                    continue

            # Wybierz weight i reps (pod target RIR dla fresh state)
            candidate_prototype = self._construct_planned_set(
                exercise_id=ex_id,
                estimated_1rm=estimated_1rm,
                current_state=current_state,
            )

            # Generuj N PlannedSetów (te same parametry dla każdej serii)
            candidate_sets = []
            for set_idx in range(sets_count):
                ps = PlannedSet(
                    exercise_id=ex_id,
                    order=start_order + set_idx,
                    reps=candidate_prototype.reps,
                    weight_kg=candidate_prototype.weight_kg,
                    rir=None,
                    predicted_rir=candidate_prototype.predicted_rir,
                    estimated_time_sec=time_per_set,
                    primary_muscles=candidate_prototype.primary_muscles,
                    secondary_muscles=candidate_prototype.secondary_muscles,
                )
                candidate_sets.append(ps)

            # Symuluj: dodaj wszystkie serie do history → predict_mpc
            test_history = list(current_history)
            sim_time = time_offset
            for ps in candidate_sets:
                ws = ps.to_workout_set(timestamp=now + timedelta(seconds=sim_time))
                test_history.append(ws)
                sim_time += ps.estimated_time_sec + rest_sec

            predicted_mpc = self._call_predict_mpc(test_history, now + timedelta(seconds=sim_time))

            # Score
            score = self._calculate_score(
                exercise_id=ex_id,
                current_state=current_state,
                predicted_mpc=predicted_mpc,
                is_favorite=ex_id in favorite_set,
                should_avoid=ex_id in avoid_set,
            )

            if score > best_score:
                best_score = score
                best_exercise_id = ex_id
                best_sets_list = candidate_sets
                best_final_state = predicted_mpc

        if best_sets_list is None:
            return [], current_state, 0

        total_block_time = sum(ps.estimated_time_sec for ps in best_sets_list) + \
                           (len(best_sets_list) - 1) * rest_sec

        return best_sets_list, best_final_state, total_block_time

    def _calculate_score(
        self,
        exercise_id: str,
        current_state: Dict[str, float],
        predicted_mpc: Dict[str, float],
        is_favorite: bool = False,
        should_avoid: bool = False,
    ) -> float:
        """
        Objective function.

        Score = Reward - Penalty + Preferences

        Reward:
            Priorytet dla ćwiczeń które angażują ŚWIEŻE mięśnie (wysoki MPC_before).
            Im większe zaangażowanie × im większa capacity przed, tym lepszy reward.

        Penalty:
            Za mięśnie, które po dodaniu tego ćwiczenia znajdą się POZA target zone.
            OVERFATIGUE (MPC_after < target_min): duża kara (ryzyko kontuzji)
            UNDERFATIGUE (MPC_after > target_max): mniejsza kara (strata potencjału)
        """
        involvement = self._get_involvement(exercise_id)

        reward = 0.0
        penalty = 0.0

        for muscle_id, ratio in involvement.items():
            mpc_before = current_state.get(muscle_id, 1.0)  # Default: fresh
            mpc_after = predicted_mpc.get(muscle_id, 1.0)

            # Reward: świeże mięśnie zaangażowane w ćwiczeniu
            # MPC = capacity, więc wysoki MPC_before = świeży
            reward += ratio * mpc_before

            # Penalty: poza target zone
            target_zone = self.config.get_target_zone(muscle_id)
            target_min, target_max = target_zone

            if mpc_after < target_min:
                # OVERFATIGUE - niebezpieczne
                overfatigue = target_min - mpc_after
                penalty += 2.0 * overfatigue  # Wysoka kara
            elif mpc_after > target_max:
                # UNDERFATIGUE - stracona szansa na bodziec
                underfatigue = mpc_after - target_max
                # Tylko jeśli mięsień BYŁ zaangażowany (ratio > 0)
                if ratio > 0.1:
                    penalty += 0.3 * underfatigue

        score = reward - penalty

        # Preferencje
        if is_favorite:
            score += 0.2
        if should_avoid:
            score -= 0.5

        return score

    def _construct_planned_set(
        self,
        exercise_id: str,
        estimated_1rm: Dict[str, float],
        current_state: Dict[str, float],
    ) -> PlannedSet:
        """
        Konstruuj PlannedSet:
          1. Weight = ~75% 1RM (fallback)
          2. Reps dobrane pod target_rir (via predict_rir)
          3. primary_muscles z involvement matrix
        """
        meta = self.exercise_catalog.get(exercise_id, {})
        ex_type = meta.get("type", "isolation")
        time_sec = meta.get("time_sec", 90)

        # Domyślne reps
        default_reps = self.config.default_reps_by_type.get(ex_type, 10)

        # Weight ~75% 1RM
        one_rm = estimated_1rm.get(exercise_id, 50.0)
        weight = round(one_rm * 0.75, 1)

        # Dobierz reps pod target RIR
        target_rir = self.config.target_rir
        best_reps = default_reps
        best_predicted_rir = None

        try:
            # Próbuj różne reps counts, wybierz ten który daje RIR najbliżej target
            candidate_reps_options = self._get_reps_candidates(ex_type, default_reps)
            best_diff = float('inf')

            for candidate_reps in candidate_reps_options:
                predicted = self.model.predict_rir(
                    current_state, exercise_id, weight, candidate_reps
                )
                diff = abs(predicted - target_rir)
                if diff < best_diff:
                    best_diff = diff
                    best_reps = candidate_reps
                    best_predicted_rir = predicted
        except Exception as e:
            logger.debug(f"  predict_rir failed for {exercise_id}: {e}")
            best_predicted_rir = None

        # Primary muscles (top 3 by involvement)
        involvement = self._get_involvement(exercise_id)
        sorted_muscles = sorted(involvement.items(), key=lambda kv: kv[1], reverse=True)
        primary = [m for m, r in sorted_muscles[:2] if r >= 0.4]
        secondary = [m for m, r in sorted_muscles[2:] if r >= 0.2]

        return PlannedSet(
            exercise_id=exercise_id,
            order=0,  # Set later
            reps=best_reps,
            weight_kg=weight,
            rir=None,
            predicted_rir=best_predicted_rir,
            estimated_time_sec=time_sec,
            primary_muscles=primary,
            secondary_muscles=secondary,
        )

    def _get_reps_candidates(self, ex_type: str, default_reps: int) -> List[int]:
        """Zwróć kandydatów na liczbę reps w zależności od typu ćwiczenia"""
        if ex_type in ("compound", "compound_variation"):
            return [3, 5, 6, 8, 10]  # Strength to moderate
        elif ex_type == "isolation":
            return [8, 10, 12, 15]   # Hypertrophy
        elif ex_type == "core":
            return [8, 10, 12, 15, 20]
        return [default_reps]

    # ========================================================================
    # INTERNAL: Involvement matrix lookup
    # ========================================================================

    def _get_involvement(self, exercise_id: str) -> Dict[str, float]:
        """
        Pobierz muscle engagement dla ćwiczenia.
        Używa mock involvement matrix (fallback) - prawdziwy model ma to w checkpoint.

        NOTE: Prawdziwy DeepGain model ma INVOLVEMENT_MATRIX wbudowane,
              ale planner potrzebuje tej informacji osobno dla scoring.
        """
        # Z mock model handle - ma INVOLVEMENT_MATRIX
        if isinstance(self.model, models_wrapper.MockModelHandle):
            return self.model.INVOLVEMENT_MATRIX.get(exercise_id, {})

        # Jeśli real model - sprawdź czy jest involvement w wrapper
        # (Fallback to MockModelHandle.INVOLVEMENT_MATRIX bo to ta sama data)
        return models_wrapper.MockModelHandle.INVOLVEMENT_MATRIX.get(exercise_id, {})

    # ========================================================================
    # INTERNAL: Model wrapper calls
    # ========================================================================

    def _call_predict_mpc(
        self,
        history: List[WorkoutSet],
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Wrap predict_mpc - konwertuj WorkoutSet → dict"""
        history_dicts = [ws.to_model_dict() for ws in history]
        return self.model.predict_mpc(history_dicts, timestamp)

    def _get_state_from_history(
        self,
        user_history: List[WorkoutSet],
        now: datetime,
    ) -> Dict[str, float]:
        """Oblicz MPC z historii"""
        if not user_history:
            return {m: 1.0 for m in self.all_muscles}
        return self._call_predict_mpc(user_history, now)

    def _default_1rm(self, exercise_id: str, ex_type: str) -> float:
        """
        Domyślne 1RM dla ćwiczeń bez historii (średni amator, M, ~80kg).
        Użyte tylko jako fallback - w praktyce user powinien mieć historię.
        """
        specific = {
            "squat": 100.0, "low_bar_squat": 100.0, "high_bar_squat": 90.0,
            "bench_press": 80.0, "close_grip_bench": 70.0, "incline_bench": 65.0,
            "deadlift": 120.0, "sumo_deadlift": 120.0, "romanian_deadlift": 90.0,
            "ohp": 45.0, "pull_ups": 80.0,  # Bodyweight + extra
            "leg_press": 150.0, "leg_curl": 40.0, "leg_extension": 50.0,
            "bulgarian_split_squat": 30.0,
            "dumbbell_fly": 20.0, "french_press": 30.0, "dips": 80.0,
            "pendlay_row": 60.0, "lat_pulldown": 55.0, "seal_row": 50.0,
            "reverse_fly": 15.0, "chest_press_machine": 70.0,
            "plank": 0.0, "farmers_walk": 40.0,
            "hanging_leg_raise": 0.0, "ab_wheel_rollout": 0.0,
            "dead_bug": 0.0, "bird_dog": 0.0,
            "trx_bodysaw": 0.0, "suitcase_carry": 24.0,
        }
        if exercise_id in specific:
            return specific[exercise_id]
        # Fallback by type
        if ex_type in ("compound", "compound_variation"):
            return 80.0
        if ex_type == "core":
            return 0.0
        return 25.0

    # ========================================================================
    # INTERNAL: Validation
    # ========================================================================

    def _validate_target_zones(
        self,
        predicted_mpc: Dict[str, float],
        initial_state: Dict[str, float],
    ) -> List[str]:
        """
        Sprawdź które mięśnie trafiły w target capacity zone.

        Convention:
          - MPC_after < min_capacity → OVERFATIGUE (niebezpieczne)
          - MPC_after > max_capacity → UNDERFATIGUE (mało pracy)
          - min ≤ MPC_after ≤ max → OK

        Ale: underfatigue notujemy tylko dla mięśni które w ogóle były zaangażowane
        (jeśli MPC przed = 1.0 i MPC po = 1.0, to nie była to część treningu).
        """
        notes = []

        for muscle_id in sorted(predicted_mpc.keys()):
            mpc_after = predicted_mpc[muscle_id]
            mpc_before = initial_state.get(muscle_id, 1.0)
            target = self.config.get_target_zone(muscle_id)
            target_min, target_max = target

            # Czy mięsień był użyty w treningu?
            was_worked = mpc_after < mpc_before - 0.02  # >2% spadek capacity

            if mpc_after < target_min:
                notes.append(
                    f"⚠ {muscle_id}: OVERFATIGUE (MPC={mpc_after:.2f} < min={target_min})"
                )
            elif was_worked and mpc_after > target_max:
                notes.append(
                    f"⚠ {muscle_id}: UNDERFATIGUE (MPC={mpc_after:.2f} > max={target_max})"
                )
            elif was_worked:
                notes.append(
                    f"✓ {muscle_id}: OK (MPC={mpc_after:.2f} in [{target_min}, {target_max}])"
                )
            # Jeśli nie pracował, nie dodajemy noty (nie był częścią treningu)

        return notes
