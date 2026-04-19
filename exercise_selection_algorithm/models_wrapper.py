"""
Model Wrapper - Unified interface dla DeepGain z fallback na mock.

Kluczowe:
  - MPC = Muscle Performance Capacity w [0.1, 1.0]
    - 1.0 = fully recovered (fresh)
    - 0.1 = exhausted
  - Semantyka jest ODWROTNA do fatigue!
  - Pusta historia → MPC = 1.0 dla wszystkich

Strategia:
  1. Spróbuj załadować prawdziwy model DeepGain (inference.py od Michała)
  2. Jeśli się nie uda → fallback na mock z tą samą semantyką (capacity)

Kontrakt (stable, nie zmieniać):
  - predict_mpc(model, user_history, timestamp) -> dict[muscle_id, float]
  - predict_rir(model, state, exercise, weight, reps) -> float
  - get_exercises() -> list[str]
  - get_muscles() -> list[str]
"""

import os
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Próba załadowania prawdziwego modelu DeepGain
# ============================================================================
_REAL_MODEL_AVAILABLE = False
_REAL_INFERENCE = None

try:
    import inference as _real_inference
    _REAL_INFERENCE = _real_inference
    _REAL_MODEL_AVAILABLE = True
    logger.info("✓ DeepGain inference module available")
except Exception as e:
    logger.warning(f"⚠ DeepGain inference not available: {e}")
    logger.warning("  Falling back to mock model")


# ============================================================================
# Lista mięśni i ćwiczeń (15 muscles / 34 exercises - jak u Michała)
# ============================================================================
ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]

# Fallback lista ćwiczeń (używana gdy prawdziwy model niedostępny)
# Zmapowane z Ratios.xlsx na konwencję Michała (bench_press, squat, ohp, ...)
FALLBACK_EXERCISES = [
    # A. Main compounds
    "squat", "low_bar_squat", "bench_press", "deadlift",
    # B. Variations
    "high_bar_squat", "close_grip_bench", "spoto_press",
    "incline_bench", "dumbbell_fly", "sumo_deadlift",
    # C. Accessories - lower
    "bulgarian_split_squat", "leg_press", "romanian_deadlift",
    "leg_curl", "leg_extension",
    # C. Accessories - upper
    "incline_bench_45", "chest_press_machine", "dips",
    "ohp", "decline_bench", "french_press",
    # C. Accessories - back
    "pendlay_row", "pull_ups", "lat_pulldown",
    "reverse_fly", "seal_row",
    # D. Core/stability
    "plank", "farmers_walk", "hanging_leg_raise",
    "ab_wheel_rollout", "dead_bug", "trx_bodysaw",
    "suitcase_carry", "bird_dog",
]


# ============================================================================
# Stan globalny modelu (lazy-loaded)
# ============================================================================
_MODEL_INSTANCE = None
_USING_REAL_MODEL = False


def initialize_model(
    checkpoint_path: str = "deepgain_model_muscle_ord.pt",
    force_mock: bool = False,
    tau_scale: float = 1.0,
) -> "ModelHandle":
    """
    Załaduj model (real lub mock).

    Args:
        checkpoint_path: Ścieżka do .pt file z wag modelu Michała
        force_mock: Jeśli True, wymusza użycie mock modelu
        tau_scale: Skala tau dla MockModel (user calibration).
                   Real DeepGain używa własnych tau z checkpointa.

    Returns:
        Handle do modelu (opakowany)
    """
    global _MODEL_INSTANCE, _USING_REAL_MODEL

    if force_mock or not _REAL_MODEL_AVAILABLE:
        logger.info(f"Using MockModel (fallback, tau_scale={tau_scale})")
        _MODEL_INSTANCE = MockModelHandle(tau_scale=tau_scale)
        _USING_REAL_MODEL = False
        return _MODEL_INSTANCE

    # Sprawdź dostępność pliku checkpoint
    if not os.path.exists(checkpoint_path):
        logger.warning(f"⚠ Checkpoint not found: {checkpoint_path}")
        logger.warning(f"  Falling back to MockModel (tau_scale={tau_scale})")
        _MODEL_INSTANCE = MockModelHandle(tau_scale=tau_scale)
        _USING_REAL_MODEL = False
        return _MODEL_INSTANCE

    try:
        real_model = _REAL_INFERENCE.load_model(checkpoint_path)
        logger.info(f"✓ Loaded DeepGain from {checkpoint_path}")
        _MODEL_INSTANCE = RealModelHandle(real_model)
        _USING_REAL_MODEL = True
    except Exception as e:
        logger.error(f"✗ Failed to load real model: {e}")
        logger.warning(f"  Falling back to MockModel (tau_scale={tau_scale})")
        _MODEL_INSTANCE = MockModelHandle(tau_scale=tau_scale)
        _USING_REAL_MODEL = False

    return _MODEL_INSTANCE


def get_model() -> "ModelHandle":
    """Zwróć aktualnie załadowany model (lub załaduj default)"""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = initialize_model()
    return _MODEL_INSTANCE


def is_using_real_model() -> bool:
    """Sprawdź czy używamy prawdziwego DeepGain, czy mock"""
    return _USING_REAL_MODEL


# ============================================================================
# Model Handles (Adapter Pattern)
# ============================================================================

class ModelHandle:
    """Abstrakcyjny interface dla modelu"""

    def predict_mpc(
        self,
        user_history: List[Dict],
        timestamp: Union[str, datetime],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def predict_rir(
        self,
        state: Dict[str, float],
        exercise: str,
        weight: float,
        reps: int,
    ) -> float:
        raise NotImplementedError

    def get_exercises(self) -> List[str]:
        raise NotImplementedError

    def get_muscles(self) -> List[str]:
        raise NotImplementedError


class RealModelHandle(ModelHandle):
    """Wrapper wokół prawdziwego DeepGain modelu Michała"""

    def __init__(self, real_model):
        self.model = real_model

    def predict_mpc(self, user_history, timestamp):
        return _REAL_INFERENCE.predict_mpc(self.model, user_history, timestamp)

    def predict_rir(self, state, exercise, weight, reps):
        return _REAL_INFERENCE.predict_rir(self.model, state, exercise, weight, reps)

    def get_exercises(self):
        return _REAL_INFERENCE.get_exercises()

    def get_muscles(self):
        return _REAL_INFERENCE.get_muscles()


class MockModelHandle(ModelHandle):
    """
    Mock DeepGain - fallback heurystyka.

    Używa TEJ SAMEJ semantyki co prawdziwy model:
      - MPC = CAPACITY w [0.1, 1.0]
      - 1.0 = fresh, 0.1 = exhausted
      - Empty history → MPC = 1.0

    Model:
      1. Zacznij z MPC = 1.0 dla wszystkich
      2. Dla każdej serii w historii (posortowanej po czasie):
         a. Apply recovery: MPC_new = 1 - (1 - MPC) * exp(-dt/(tau*tau_scale))
         b. Apply fatigue: MPC_new = MPC * (1 - involvement * drop)
            gdzie drop = f(reps, rir, weight)
      3. Final recovery od ostatniej serii do timestamp

    tau_scale:
      - 1.0 = baseline (default)
      - <1 = szybsza regeneracja (e.g., 0.85 dla advanced)
      - >1 = wolniejsza regeneracja (e.g., 1.2 dla beginner / starszy user)
    """

    def __init__(self, tau_scale: float = 1.0):
        self.tau_scale = tau_scale

    # Time constants regeneracji (hours) - jak w inference.py Michała
    MUSCLE_TAU = {
        "chest":          16.0,
        "anterior_delts": 13.0,
        "lateral_delts":   9.0,
        "rear_delts":      8.0,
        "rhomboids":      10.0,
        "triceps":         9.0,
        "biceps":         13.0,
        "lats":           13.0,
        "quads":          19.0,
        "hamstrings":     18.0,
        "glutes":         15.0,
        "adductors":      12.0,
        "erectors":       12.0,
        "calves":          8.0,
        "abs":            10.0,
    }

    # Involvement matrix: exercise → muscle → ratio [0, 1]
    # (Zmapowane z Ratios.xlsx na 15 mięśni Michała)
    INVOLVEMENT_MATRIX = {
        # === A. MAIN COMPOUNDS ===
        "squat": {
            "quads": 0.65, "glutes": 0.40, "hamstrings": 0.30,
            "erectors": 0.45, "adductors": 0.35, "abs": 0.20,
        },
        "low_bar_squat": {
            "quads": 0.50, "glutes": 0.50, "hamstrings": 0.40,
            "erectors": 0.50, "adductors": 0.40, "abs": 0.20,
        },
        "bench_press": {
            "chest": 0.80, "triceps": 0.60, "anterior_delts": 0.35,
            "lateral_delts": 0.15,
        },
        "deadlift": {
            "hamstrings": 0.60, "glutes": 0.55, "erectors": 0.70,
            "quads": 0.30, "lats": 0.40, "rhomboids": 0.30, "abs": 0.25,
        },

        # === B. VARIATIONS ===
        "high_bar_squat": {
            "quads": 0.75, "glutes": 0.45, "hamstrings": 0.25,
            "erectors": 0.35, "adductors": 0.30, "abs": 0.20,
        },
        "close_grip_bench": {
            "chest": 0.65, "triceps": 0.85, "anterior_delts": 0.30,
        },
        "spoto_press": {
            "chest": 0.70, "triceps": 0.55, "anterior_delts": 0.40,
        },
        "incline_bench": {
            "chest": 0.75, "anterior_delts": 0.55, "triceps": 0.45,
            "lateral_delts": 0.15,
        },
        "incline_bench_45": {
            "chest": 0.70, "anterior_delts": 0.65, "triceps": 0.40,
        },
        "dumbbell_fly": {
            "chest": 0.85, "anterior_delts": 0.30,
        },
        "sumo_deadlift": {
            "hamstrings": 0.45, "glutes": 0.50, "quads": 0.50,
            "adductors": 0.55, "erectors": 0.50, "abs": 0.25,
        },

        # === C. ACCESSORIES - LOWER BODY ===
        "bulgarian_split_squat": {
            "quads": 0.80, "glutes": 0.65, "hamstrings": 0.25,
            "erectors": 0.35, "adductors": 0.30,
        },
        "leg_press": {
            "quads": 0.75, "glutes": 0.40, "hamstrings": 0.25,
        },
        "romanian_deadlift": {
            "hamstrings": 0.85, "glutes": 0.55, "erectors": 0.50,
        },
        "leg_curl": {
            "hamstrings": 0.95,
        },
        "leg_extension": {
            "quads": 0.95,
        },

        # === C. ACCESSORIES - UPPER BODY ===
        "chest_press_machine": {
            "chest": 0.75, "triceps": 0.45, "anterior_delts": 0.35,
        },
        "dips": {
            "chest": 0.65, "triceps": 0.75, "anterior_delts": 0.40,
        },
        "ohp": {
            "anterior_delts": 0.75, "lateral_delts": 0.55, "triceps": 0.55,
            "chest": 0.30, "abs": 0.20,
        },
        "decline_bench": {
            "chest": 0.85, "triceps": 0.50, "anterior_delts": 0.20,
        },
        "french_press": {
            "triceps": 0.90, "anterior_delts": 0.15,
        },

        # === C. ACCESSORIES - BACK ===
        "pendlay_row": {
            "lats": 0.60, "rhomboids": 0.75, "rear_delts": 0.55,
            "biceps": 0.45, "erectors": 0.50, "abs": 0.15,
        },
        "pull_ups": {
            "lats": 0.80, "rhomboids": 0.55, "biceps": 0.55, "rear_delts": 0.30,
        },
        "lat_pulldown": {
            "lats": 0.70, "rhomboids": 0.45, "biceps": 0.50, "rear_delts": 0.30,
        },
        "reverse_fly": {
            "rear_delts": 0.85, "rhomboids": 0.60, "lateral_delts": 0.30,
        },
        "seal_row": {
            "lats": 0.55, "rhomboids": 0.70, "rear_delts": 0.60, "biceps": 0.45,
        },

        # === D. CORE & STABILITY ===
        "plank": {
            "abs": 0.75, "erectors": 0.30,
        },
        "farmers_walk": {
            "erectors": 0.45, "abs": 0.35, "glutes": 0.25, "calves": 0.30,
        },
        "hanging_leg_raise": {
            "abs": 0.85, "lats": 0.25, "erectors": 0.15,
        },
        "ab_wheel_rollout": {
            "abs": 0.90, "lats": 0.30, "triceps": 0.20, "anterior_delts": 0.15,
        },
        "dead_bug": {
            "abs": 0.80, "erectors": 0.20,
        },
        "trx_bodysaw": {
            "abs": 0.85, "anterior_delts": 0.30, "lats": 0.30,
        },
        "suitcase_carry": {
            "abs": 0.75, "erectors": 0.55, "lats": 0.30,
        },
        "bird_dog": {
            "erectors": 0.70, "glutes": 0.50, "abs": 0.40,
        },
    }

    def predict_mpc(
        self,
        user_history: List[Dict],
        timestamp: Union[str, datetime],
    ) -> Dict[str, float]:
        """
        Predict MPC (capacity) używając heurystyki naśladującej DeepGain.

        Semantyka: MPC = capacity w [0.1, 1.0]
        """
        # Parsuj timestamp
        ts_query = self._parse_timestamp(timestamp)

        # Inicjalizuj MPC = 1.0 (full capacity)
        mpc = {m: 1.0 for m in ALL_MUSCLES}

        # Filtruj i sortuj historię
        valid = []
        for entry in user_history:
            ts = self._parse_timestamp(entry["timestamp"])
            if ts > ts_query:
                continue
            exercise = entry.get("exercise", "")
            if exercise not in self.INVOLVEMENT_MATRIX:
                continue  # Skip unknown exercises
            valid.append({
                "exercise": exercise,
                "weight_kg": float(entry["weight_kg"]),
                "reps": int(entry["reps"]),
                "rir": int(entry.get("rir", 2)),
                "timestamp": ts,
            })

        if not valid:
            return mpc

        valid.sort(key=lambda x: x["timestamp"])

        # Replay historii
        prev_ts = valid[0]["timestamp"]

        for i, s in enumerate(valid):
            # 1. Recovery od poprzedniej serii
            if i > 0:
                dt_hours = (s["timestamp"] - prev_ts).total_seconds() / 3600.0
                if dt_hours > 0:
                    mpc = self._apply_recovery(mpc, dt_hours)

            # 2. Fatigue od tej serii
            mpc = self._apply_fatigue(mpc, s)

            prev_ts = s["timestamp"]

        # 3. Final recovery od ostatniej serii do query timestamp
        dt_final = (ts_query - prev_ts).total_seconds() / 3600.0
        if dt_final > 0:
            mpc = self._apply_recovery(mpc, dt_final)

        # Clamp
        mpc = {k: max(0.1, min(1.0, v)) for k, v in mpc.items()}

        return mpc

    def _apply_recovery(self, mpc: Dict[str, float], dt_hours: float) -> Dict[str, float]:
        """
        Recovery: MPC_new = 1 - (1 - MPC) * exp(-dt / (tau * tau_scale))
        (Im dłużej, tym bliżej 1.0)

        tau_scale skaluje regeneracje per user (z UserProfile)
        """
        new_mpc = {}
        for muscle, current in mpc.items():
            tau = self.MUSCLE_TAU.get(muscle, 12.0) * self.tau_scale
            new_mpc[muscle] = 1.0 - (1.0 - current) * math.exp(-dt_hours / tau)
        return new_mpc

    def _apply_fatigue(self, mpc: Dict[str, float], set_data: Dict) -> Dict[str, float]:
        """
        Fatigue: MPC_new = MPC * (1 - involvement * drop)
        gdzie drop = f(reps, rir, weight)
        """
        exercise = set_data["exercise"]
        involvement = self.INVOLVEMENT_MATRIX.get(exercise, {})

        # Intensywność serii (drop factor)
        reps = set_data["reps"]
        rir = set_data["rir"]
        weight = set_data["weight_kg"]

        # Heurystyka intensywności [0, 1]:
        # - RPE = 10 - RIR (effort)
        # - reps_factor (więcej = bardziej zmęczające)
        # - weight_factor (lżejsze = mniejszy drop)
        rpe_factor = max(0, (10 - rir)) / 10.0
        reps_factor = min(1.0, reps / 12.0)  # Hipertrofia sweet spot ~8-12
        weight_factor = min(1.0, max(0.3, weight / 100.0))

        drop = 0.15 * rpe_factor * (0.5 + 0.5 * reps_factor) * (0.5 + 0.5 * weight_factor)
        drop = min(0.4, max(0.02, drop))  # Clamp [2%, 40%]

        new_mpc = dict(mpc)
        for muscle, ratio in involvement.items():
            if muscle in new_mpc:
                new_mpc[muscle] = new_mpc[muscle] * (1.0 - ratio * drop)
                new_mpc[muscle] = max(0.1, new_mpc[muscle])  # Floor na 0.1 (jak Michał)

        return new_mpc

    def predict_rir(
        self,
        state: Dict[str, float],
        exercise: str,
        weight: float,
        reps: int,
    ) -> float:
        """
        Predict RIR for a planned set.

        Heurystyka:
          - Świeży mięsień + lekki weight + małe reps → high RIR
          - Zmęczony mięsień + ciężki weight + dużo reps → low RIR
        """
        if exercise not in self.INVOLVEMENT_MATRIX:
            raise ValueError(f"Unknown exercise: {exercise}")

        involvement = self.INVOLVEMENT_MATRIX[exercise]

        # Weighted average capacity dla zaangażowanych mięśni
        total_weight = sum(involvement.values())
        if total_weight == 0:
            avg_capacity = 1.0
        else:
            avg_capacity = sum(
                state.get(m, 1.0) * r for m, r in involvement.items()
            ) / total_weight

        # Bazowe RIR [0, 5]:
        # - Pełna capacity + niskie reps + niska waga = ~4 RIR
        # - Niska capacity + dużo reps + duża waga = ~0 RIR
        reps_difficulty = min(1.0, reps / 15.0)
        weight_difficulty = min(1.0, weight / 120.0)
        capacity_factor = avg_capacity  # 1.0 = fresh, 0.1 = exhausted

        # RIR ≈ capacity * (5 - 3*reps_diff - 2*weight_diff)
        base_rir = 5.0 * capacity_factor - 3.0 * reps_difficulty - 2.0 * weight_difficulty

        # Clamp
        return max(0.0, min(5.0, base_rir))

    def get_exercises(self) -> List[str]:
        return list(self.INVOLVEMENT_MATRIX.keys())

    def get_muscles(self) -> List[str]:
        return list(ALL_MUSCLES)

    def _parse_timestamp(self, ts) -> datetime:
        """Parse ISO string or datetime to naive datetime"""
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        s = str(ts)
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


# ============================================================================
# Public API (stable contract)
# ============================================================================

def predict_mpc(
    user_history: List[Dict],
    timestamp: Union[str, datetime],
    model: Optional[ModelHandle] = None,
) -> Dict[str, float]:
    """
    Predict MPC (Muscle Performance Capacity) dla wszystkich mięśni.

    Args:
        user_history: Lista dict-ów: {exercise, weight_kg, reps, rir, timestamp}
        timestamp: Moment predykcji (ISO string lub datetime)
        model: Opcjonalnie konkretny handle (default: globalny)

    Returns:
        dict[muscle_id, MPC] gdzie MPC w [0.1, 1.0], 1.0 = fresh
    """
    if model is None:
        model = get_model()
    return model.predict_mpc(user_history, timestamp)


def predict_rir(
    state: Dict[str, float],
    exercise: str,
    weight: float,
    reps: int,
    model: Optional[ModelHandle] = None,
) -> float:
    """
    Predict RIR (Reps in Reserve) dla planowanej serii.

    Args:
        state: dict[muscle_id, MPC] (z predict_mpc)
        exercise: ID ćwiczenia
        weight: kg
        reps: planned reps

    Returns:
        Predicted RIR w [0.0, 5.0]
    """
    if model is None:
        model = get_model()
    return model.predict_rir(state, exercise, weight, reps)


def get_exercises(model: Optional[ModelHandle] = None) -> List[str]:
    """Lista ćwiczeń rozpoznawalnych przez model"""
    if model is None:
        model = get_model()
    return model.get_exercises()


def get_muscles(model: Optional[ModelHandle] = None) -> List[str]:
    """Lista mięśni (15)"""
    if model is None:
        model = get_model()
    return model.get_muscles()
