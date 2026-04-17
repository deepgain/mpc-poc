"""
Mock DeepGain Model - Stub implementacji

W przyszłości to będzie integracja z prawdziwym modelem DeepGain.
Na razie: prosta heurystyka bazująca na:
  - Recency decay (świeże serie mają większy wpływ)
  - RPE/RIR (bardziej intensywne serie = więcej zmęczenia)
  - Muscle engagement ratios z exercise catalog
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from data_structures import WorkoutSet


class MockDeepGainModel:
    """
    Heurystyka:
    - MPC_delta per muscle = sum(engagement_ratio * intensity_factor * decay_factor)
    - intensity_factor = f(reps, estimated_rir, weight_relative)
    - decay_factor = exp(-hours_since / tau) dla każdego mięśnia
    """

    # Time constants regeneracji (tau) - ile godzin aby MPC spadł do 37% (1/e)
    MUSCLE_TAU_HOURS = {
        'quadriceps': 48,      # Duże, wolna regen
        'hamstring': 48,
        'glutes': 48,
        'lats': 48,
        'erector_spinae': 48,
        'chest_upper': 36,
        'chest_lower': 36,
        'shoulder_front': 36,
        'shoulder_side': 36,
        'shoulder_rear': 36,
        'biceps': 24,          # Mniejsze, szybka regen
        'calves': 24,
        'hip_adductors': 24,
        'glute_med': 24,
        'rhomboid': 24,
        'abs': 20,
        'multifidus': 20,
        'oblique_ext': 20,
        'oblique_int': 20,
    }

    # Domyślny RIR dla szacowania intensywności
    DEFAULT_RIR = 2

    def __init__(self, exercises_config: Dict):
        """
        exercises_config: słownik z danymi ćwiczeń (muscle_engagement, itd.)
        """
        self.exercises = exercises_config.get('exercises', {})

    def predict_mpc(
        self,
        workout_history: List[WorkoutSet],
        now: datetime
    ) -> Dict[str, float]:
        """
        Predykcja MPC dla wszystkich mięśni na podstawie historii.

        Args:
            workout_history: lista wykonanych serii (WorkoutSet)
            now: aktualna data/czas

        Returns:
            dict[muscle_id, float] - MPC (Muscle Pathway Current state) [0, 1]
        """
        mpc = {}

        # Zbierz wszystkie mięśnie
        all_muscles = set()
        for ex in self.exercises.values():
            all_muscles.update(ex.get('muscle_engagement', {}).keys())

        # Inicjalizuj MPC na 0 (pełna regeneracja)
        for muscle in all_muscles:
            mpc[muscle] = 0.0

        if not workout_history:
            return mpc

        # Dla każdej serii w historii, dodaj zmęczenie z uwzględnieniem decay
        for workout_set in workout_history:
            exercise = self.exercises.get(workout_set.exercise_id)
            if not exercise:
                continue

            # Czas od serii
            time_diff = now - workout_set.timestamp
            hours_since = time_diff.total_seconds() / 3600.0

            # Muscle engagement dla tego ćwiczenia
            muscle_engagement = exercise.get('muscle_engagement', {})

            # Szacuj intensywność serii
            reps = workout_set.reps
            rir = workout_set.rir if workout_set.rir is not None else self.DEFAULT_RIR
            weight = workout_set.weight_kg

            # Heurystyka intensywności:
            # - Więcej reps = bardziej zmęczające
            # - Mniej RIR = bardziej zbliżone do zmęczenia
            # - Weight jest normalizacyjne (przyjmij liniową skalę)
            intensity = self._calculate_intensity(reps, rir, weight)

            # Dla każdego zaangażowanego mięśnia
            for muscle_id, engagement_ratio in muscle_engagement.items():
                # Decay based on time
                tau = self.MUSCLE_TAU_HOURS.get(muscle_id, 36)
                decay = self._exponential_decay(hours_since, tau)

                # Wkład tej serii do MPC tego mięśnia
                contribution = engagement_ratio * intensity * decay

                mpc[muscle_id] += contribution

        # Ogranicz do [0, 1]
        mpc = {k: min(1.0, max(0.0, v)) for k, v in mpc.items()}

        return mpc

    def _calculate_intensity(self, reps: int, rir: int, weight: float) -> float:
        """
        Heurystyka intensywności serii.

        Bazuje na RPE = 10 - RIR, gdzie:
        - RPE 10 = max effort (RIR=0)
        - RPE 8 = moderate (RIR=2)
        - RPE 6 = light (RIR=4)

        Intensity ∝ (reps/10) * (10 - rir) / 10

        Zakres: [0, 1]
        """
        rpe = max(1, 10 - rir)  # RPE w skali [1-10]

        # Więcej reps = większy effort dla danego RIR
        reps_factor = min(1.0, reps / 15.0)  # Normalize to [0, 1]

        # RPE wpływ
        rpe_factor = rpe / 10.0

        intensity = (reps_factor + rpe_factor) / 2.0

        # Weight factor - lekka korekta
        # (zakładaj, że ciężkie = gorsza technika, słabsza kontrola -> więcej zmęczenia)
        weight_factor = min(1.1, 1.0 + (weight / 100.0) * 0.1)

        return min(1.0, intensity * weight_factor)

    def _exponential_decay(self, hours_since: float, tau_hours: float) -> float:
        """
        Exponential decay: exp(-t / tau)

        Args:
            hours_since: ile godzin temu seria
            tau_hours: time constant (czas do 37% wartości)

        Returns:
            decay factor [0, 1]
        """
        import math
        if tau_hours <= 0:
            return 0.0
        decay = math.exp(-hours_since / tau_hours)
        return max(0.0, decay)


# Funkcja interfejsowa (globalna)
_model_instance = None


def get_model(exercises_config: Dict) -> MockDeepGainModel:
    """Lazy-load model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MockDeepGainModel(exercises_config)
    return _model_instance


def predict_mpc(
    workout_history: List[WorkoutSet],
    now: datetime,
    exercises_config: Dict
) -> Dict[str, float]:
    """
    Interfejs publiczny dla predykcji MPC.

    Te się będzie wywoływać z planner.py
    """
    model = get_model(exercises_config)
    return model.predict_mpc(workout_history, now)
