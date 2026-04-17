"""
WorkoutPlanner - Silnik rekomendacyjny dla treningu

Algorytm: Greedy selection z symulacją
1. Podaj aktualny state MPC i preferencje (n_compound, n_isolation, czas, exclusions)
2. Planista iteracyjnie dobiera ćwiczenia maksymalizując objective function
3. Objective = suma(involvement_rank × MPC_before) - penalty za poza target zones
4. Na każdym kroku symuluje kandydatów, wybiera najlepszy
5. Wspiera replanning gdy user zmienia/odrzuca seriach
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import logging

from data_structures import (
    WorkoutSet, PlannedSet, PlanResult, PlannerConfig
)
from models_mock import predict_mpc


logger = logging.getLogger(__name__)


class WorkoutPlanner:
    """
    Główna klasa planera treningowego.

    Interface:
    - plan() -> PlanResult
    - replan() -> PlanResult
    - estimate_1rm() -> dict
    """

    def __init__(
        self,
        exercises_config: Dict,
        planner_config: PlannerConfig,
    ):
        """
        Args:
            exercises_config: Słownik z ćwiczeniami i ich parametrami
            planner_config: Konfiguracja (target zones, limity, itp.)
        """
        self.exercises = exercises_config.get('exercises', {})
        self.config = planner_config
        self.default_reps_by_type = exercises_config.get('default_reps_by_type', {})

        # Organize exercises by type
        self.compound_exercises = {}
        self.isolation_exercises = {}
        self.core_exercises = {}

        for ex_id, ex_data in self.exercises.items():
            ex_type = ex_data.get('type', 'other')
            if ex_type == 'compound' or ex_type == 'compound_variation':
                self.compound_exercises[ex_id] = ex_data
            elif ex_type == 'core':
                self.core_exercises[ex_id] = ex_data
            else:  # isolation, accessory
                self.isolation_exercises[ex_id] = ex_data

        logger.info(f"✓ Planner initialized")
        logger.info(f"  - {len(self.compound_exercises)} compound exercises")
        logger.info(f"  - {len(self.isolation_exercises)} isolation exercises")
        logger.info(f"  - {len(self.core_exercises)} core exercises")

    def plan(
        self,
        state: Dict[str, float],
        n_compound: int,
        n_isolation: int,
        available_time_sec: int,
        user_history: Optional[List[WorkoutSet]] = None,
        exclusions: Optional[List[str]] = None,
        preferences: Optional[Dict] = None,
    ) -> PlanResult:
        """
        Główna funkcja planowania.

        Args:
            state: dict[muscle_id, MPC] - aktualny stan mięśni
            n_compound: ile ćwiczeń compound
            n_isolation: ile ćwiczeń isolation
            available_time_sec: czas dostępny na trening (sekund)
            user_history: historia treningów użytkownika (dla estymacji 1RM)
            exclusions: lista exercise_id do pominięcia
            preferences: dict z ulubionymi ćwiczeniami itp.

        Returns:
            PlanResult z planem i predykcją MPC post-workout
        """
        exclusions = exclusions or []
        preferences = preferences or {}
        user_history = user_history or []

        # Estymuj 1RM dla każdego ćwiczenia
        estimated_1rm = self.estimate_1rm_from_history(user_history)

        # Greedy selection
        plan = []
        current_history = user_history.copy()
        current_state = state.copy()
        used_exercises = set()
        total_time = 0

        # Phase 1: Compound exercises
        logger.info(f"\n=== Phase 1: Selecting {n_compound} compound exercises ===")
        for i in range(n_compound):
            best_ex_id, best_set, simulation_mpc = self._select_best_exercise(
                exercise_pool=self.compound_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                available_time_sec=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
            )

            if not best_ex_id:
                logger.warning(f"  ⚠ Could not select compound exercise #{i+1}")
                break

            # Add to plan
            planned_set = self._create_planned_set(
                exercise_id=best_ex_id,
                order=len(plan) + 1,
                exercise_data=self.exercises[best_ex_id],
                estimated_1rm=estimated_1rm,
                state=current_state,
            )
            plan.append(planned_set)
            used_exercises.add(best_ex_id)

            # Update state and history
            workout_set = self._planned_set_to_workout_set(planned_set)
            current_history.append(workout_set)
            current_state = simulation_mpc
            total_time += planned_set.estimated_time_sec

            logger.info(f"  [{i+1}] {best_ex_id}: {planned_set.reps}x{planned_set.weight_kg}kg")

        # Phase 2: Isolation exercises
        logger.info(f"\n=== Phase 2: Selecting {n_isolation} isolation exercises ===")
        for i in range(n_isolation):
            best_ex_id, best_set, simulation_mpc = self._select_best_exercise(
                exercise_pool=self.isolation_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                available_time_sec=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
            )

            if not best_ex_id:
                logger.warning(f"  ⚠ Could not select isolation exercise #{i+1}")
                break

            planned_set = self._create_planned_set(
                exercise_id=best_ex_id,
                order=len(plan) + 1,
                exercise_data=self.exercises[best_ex_id],
                estimated_1rm=estimated_1rm,
                state=current_state,
            )
            plan.append(planned_set)
            used_exercises.add(best_ex_id)

            workout_set = self._planned_set_to_workout_set(planned_set)
            current_history.append(workout_set)
            current_state = simulation_mpc
            total_time += planned_set.estimated_time_sec

            logger.info(f"  [{i+1}] {best_ex_id}: {planned_set.reps}x{planned_set.weight_kg}kg")

        # Phase 3: Core (always last)
        logger.info(f"\n=== Phase 3: Core/Stability (if time allows) ===")
        if total_time < available_time_sec:
            best_ex_id, best_set, simulation_mpc = self._select_best_exercise(
                exercise_pool=self.core_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                available_time_sec=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
            )

            if best_ex_id:
                planned_set = self._create_planned_set(
                    exercise_id=best_ex_id,
                    order=len(plan) + 1,
                    exercise_data=self.exercises[best_ex_id],
                    estimated_1rm=estimated_1rm,
                    state=current_state,
                )
                plan.append(planned_set)
                used_exercises.add(best_ex_id)

                workout_set = self._planned_set_to_workout_set(planned_set)
                current_history.append(workout_set)
                current_state = simulation_mpc
                total_time += planned_set.estimated_time_sec

                logger.info(f"  [Core] {best_ex_id}: {planned_set.reps}x{planned_set.weight_kg}kg")

        # Finalna predykcja MPC
        final_mpc = predict_mpc(current_history, datetime.now(), {'exercises': self.exercises})

        # Walidacja target zones
        notes = self._validate_target_zones(final_mpc, state)

        return PlanResult(
            plan=plan,
            predicted_mpc_after=final_mpc,
            total_time_estimated_sec=total_time,
            notes=notes,
        )

    def replan(
        self,
        session_so_far: List[PlannedSet],
        remaining_n_compound: int,
        remaining_n_isolation: int,
        current_state: Dict[str, float],
        available_time_sec: int,
        user_history: Optional[List[WorkoutSet]] = None,
        exclusions: Optional[List[str]] = None,
    ) -> PlanResult:
        """
        Replanning w locie - gdy user zmieni/odrzuci ćwiczenie.

        Args:
            session_so_far: już zaplanowane/wykonane serie
            remaining_n_compound: ile compound jeszcze potrzeba
            remaining_n_isolation: ile isolation jeszcze potrzeba
            current_state: aktualny MPC (z predict_mpc)
            available_time_sec: czas zostały
            user_history: historia do estymacji
            exclusions: do pominięcia

        Returns:
            PlanResult z nowym planem na pozostałe serie
        """
        logger.info(f"\n=== REPLANNING ===")
        logger.info(f"  Session so far: {len(session_so_far)} sets")
        logger.info(f"  Remaining: {remaining_n_compound} compound + {remaining_n_isolation} isolation")

        used_exercises = set(s.exercise_id for s in session_so_far)
        exclusions = exclusions or []

        # Zbuduj aktualną historię
        current_history = user_history.copy() if user_history else []
        for planned_set in session_so_far:
            workout_set = self._planned_set_to_workout_set(planned_set)
            current_history.append(workout_set)

        # Estymuj 1RM
        estimated_1rm = self.estimate_1rm_from_history(current_history)

        # Zaplanuj pozostałe serie
        new_plan = []
        total_time = sum(s.estimated_time_sec for s in session_so_far)

        # Compound
        for i in range(remaining_n_compound):
            best_ex_id, _, simulation_mpc = self._select_best_exercise(
                exercise_pool=self.compound_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                available_time_sec=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
            )

            if not best_ex_id:
                logger.warning(f"  ⚠ Could not select compound exercise #{i+1}")
                break

            planned_set = self._create_planned_set(
                exercise_id=best_ex_id,
                order=len(session_so_far) + len(new_plan) + 1,
                exercise_data=self.exercises[best_ex_id],
                estimated_1rm=estimated_1rm,
                state=current_state,
            )
            new_plan.append(planned_set)
            used_exercises.add(best_ex_id)

            workout_set = self._planned_set_to_workout_set(planned_set)
            current_history.append(workout_set)
            current_state = simulation_mpc
            total_time += planned_set.estimated_time_sec

            logger.info(f"  [{i+1}] {best_ex_id}")

        # Isolation
        for i in range(remaining_n_isolation):
            best_ex_id, _, simulation_mpc = self._select_best_exercise(
                exercise_pool=self.isolation_exercises,
                current_state=current_state,
                used_exercises=used_exercises,
                exclusions=exclusions,
                current_history=current_history,
                available_time_sec=available_time_sec - total_time,
                estimated_1rm=estimated_1rm,
            )

            if not best_ex_id:
                logger.warning(f"  ⚠ Could not select isolation exercise #{i+1}")
                break

            planned_set = self._create_planned_set(
                exercise_id=best_ex_id,
                order=len(session_so_far) + len(new_plan) + 1,
                exercise_data=self.exercises[best_ex_id],
                estimated_1rm=estimated_1rm,
                state=current_state,
            )
            new_plan.append(planned_set)
            used_exercises.add(best_ex_id)

            workout_set = self._planned_set_to_workout_set(planned_set)
            current_history.append(workout_set)
            current_state = simulation_mpc
            total_time += planned_set.estimated_time_sec

            logger.info(f"  [{i+1}] {best_ex_id}")

        # Finalna predykcja
        final_mpc = predict_mpc(current_history, datetime.now(), {'exercises': self.exercises})
        notes = self._validate_target_zones(final_mpc, current_state)

        # Zwróć pełny plan (session_so_far + new_plan)
        full_plan = list(session_so_far) + new_plan

        return PlanResult(
            plan=full_plan,
            predicted_mpc_after=final_mpc,
            total_time_estimated_sec=total_time,
            notes=notes,
        )

    def _select_best_exercise(
        self,
        exercise_pool: Dict[str, Dict],
        current_state: Dict[str, float],
        used_exercises: set,
        exclusions: List[str],
        current_history: List[WorkoutSet],
        available_time_sec: int,
        estimated_1rm: Dict[str, float],
    ) -> Tuple[Optional[str], Optional[PlannedSet], Dict[str, float]]:
        """
        Greedy selection: spróbuj każdego kandydata, wybierz z najlepszym score.

        Returns:
            (exercise_id, planned_set, predicted_mpc_after)
        """
        best_score = -float('inf')
        best_ex_id = None
        best_simulation_mpc = current_state.copy()

        for ex_id, ex_data in exercise_pool.items():
            # Skip jeśli już użyty, excluded, lub brak czasu
            if ex_id in used_exercises or ex_id in exclusions:
                continue

            # Szacuj czas
            estimated_time = ex_data.get('estimated_time_per_set_sec', 120)
            if estimated_time > available_time_sec:
                continue

            # Symuluj dodanie tego ćwiczenia
            planned_set = self._create_planned_set(
                exercise_id=ex_id,
                order=999,  # Placeholder
                exercise_data=ex_data,
                estimated_1rm=estimated_1rm,
                state=current_state,
            )

            # Dodaj do historii i przedykuj
            test_history = current_history.copy()
            workout_set = self._planned_set_to_workout_set(planned_set)
            test_history.append(workout_set)

            predicted_mpc = predict_mpc(
                test_history,
                datetime.now(),
                {'exercises': self.exercises}
            )

            # Oblicz score (objective function)
            score = self._calculate_exercise_score(
                exercise_id=ex_id,
                exercise_data=ex_data,
                current_state=current_state,
                predicted_mpc=predicted_mpc,
            )

            if score > best_score:
                best_score = score
                best_ex_id = ex_id
                best_simulation_mpc = predicted_mpc

        return best_ex_id, None, best_simulation_mpc

    def _calculate_exercise_score(
        self,
        exercise_id: str,
        exercise_data: Dict,
        current_state: Dict[str, float],
        predicted_mpc: Dict[str, float],
    ) -> float:
        """
        Objective function dla selekcji ćwiczenia.

        Score = Reward - Penalty
        Reward: suma(engagement × MPC_before) - priorytet świeżych mięśni
        Penalty: suma dla mięśni poza target zone
        """
        reward = 0.0
        penalty = 0.0

        muscle_engagement = exercise_data.get('muscle_engagement', {})

        for muscle_id, engagement in muscle_engagement.items():
            mpc_before = current_state.get(muscle_id, 0.0)

            # Reward: priorytet dla świeżych mięśni z wysokim zaangażowaniem
            reward += engagement * (1.0 - mpc_before)  # Świeże = wysokie reward

            # Penalty: czy po dodaniu będzie poza target zone?
            target_zone = self.config.get_target_zone(muscle_id)
            mpc_after = predicted_mpc.get(muscle_id, 0.0)

            if mpc_after < target_zone[0]:
                penalty += 0.1  # Za mało zmęczenia
            elif mpc_after > target_zone[1]:
                penalty += 0.2 * (mpc_after - target_zone[1])  # Zbyt dużo - groźne

        score = reward - penalty
        return score

    def _create_planned_set(
        self,
        exercise_id: str,
        order: int,
        exercise_data: Dict,
        estimated_1rm: Dict[str, float],
        state: Dict[str, float],
    ) -> PlannedSet:
        """
        Stwórz PlannedSet z szacunkowych parametrów.
        """
        ex_type = exercise_data.get('type', 'other')
        reps = self.default_reps_by_type.get(ex_type, 10)

        # Szacuj weight (~75% 1RM)
        weight_1rm = estimated_1rm.get(exercise_id, 50.0)
        weight = weight_1rm * 0.75

        # Czas
        time_per_set = exercise_data.get('estimated_time_per_set_sec', 120)

        # Muscle engagement
        muscle_engagement = exercise_data.get('muscle_engagement', {})
        primary_muscles = sorted(
            muscle_engagement.keys(),
            key=lambda m: muscle_engagement[m],
            reverse=True
        )[:3]  # Top 3
        secondary_muscles = primary_muscles[1:] if len(primary_muscles) > 1 else []

        return PlannedSet(
            exercise_id=exercise_id,
            order=order,
            reps=reps,
            weight_kg=round(weight, 1),
            rir=None,  # User będzie pytany
            estimated_time_sec=time_per_set,
            primary_muscles=primary_muscles,
            secondary_muscles=secondary_muscles,
        )

    def _planned_set_to_workout_set(self, planned_set: PlannedSet) -> WorkoutSet:
        """Konwertuj PlannedSet do WorkoutSet dla historii"""
        return WorkoutSet(
            exercise_id=planned_set.exercise_id,
            weight_kg=planned_set.weight_kg,
            reps=planned_set.reps,
            rir=planned_set.rir,
            timestamp=datetime.now(),
            completed=True,
        )

    def estimate_1rm_from_history(
        self,
        user_history: List[WorkoutSet],
    ) -> Dict[str, float]:
        """
        Estymuj 1RM dla każdego ćwiczenia z historii.

        Heurystyka: 1RM ≈ weight × (1 + reps/30)  [Brzycki formula]
        Bierze max wartość z ostatnich treningów.
        """
        estimated_1rm = defaultdict(float)

        for workout_set in user_history:
            # Brzycki formula
            reps = max(1, workout_set.reps)
            estimated = workout_set.weight_kg * (1.0 + reps / 30.0)

            # Keep max
            if estimated > estimated_1rm[workout_set.exercise_id]:
                estimated_1rm[workout_set.exercise_id] = estimated

        # Default values for exercises without history
        for ex_id in self.exercises:
            if ex_id not in estimated_1rm:
                # Domyślne bazowe wartości (można dostroić)
                ex_type = self.exercises[ex_id].get('type', 'other')
                if ex_type == 'compound':
                    estimated_1rm[ex_id] = 100.0
                elif ex_type == 'isolation':
                    estimated_1rm[ex_id] = 30.0
                else:
                    estimated_1rm[ex_id] = 20.0

        return dict(estimated_1rm)

    def _validate_target_zones(
        self,
        predicted_mpc: Dict[str, float],
        initial_state: Dict[str, float],
    ) -> List[str]:
        """
        Walidacja czy finalne MPC trafia w target zones.
        Zwróć listę uwag/ostrzeżeń.
        """
        notes = []

        for muscle_id, mpc_final in predicted_mpc.items():
            target_zone = self.config.get_target_zone(muscle_id)
            mpc_initial = initial_state.get(muscle_id, 0.0)

            if mpc_final < target_zone[0]:
                notes.append(
                    f"⚠ {muscle_id}: underfatigue (MPC={mpc_final:.2f} < {target_zone[0]})"
                )
            elif mpc_final > target_zone[1]:
                notes.append(
                    f"⚠ {muscle_id}: overfatigue (MPC={mpc_final:.2f} > {target_zone[1]})"
                )
            else:
                notes.append(
                    f"✓ {muscle_id}: OK (MPC={mpc_final:.2f} in [{target_zone[0]}, {target_zone[1]}])"
                )

        return notes
