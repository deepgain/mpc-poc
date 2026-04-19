"""
Data structures dla WorkoutPlanner

WAŻNE - Semantyka MPC:
    MPC = Muscle Performance Capacity w [0.1, 1.0]
    - 1.0 = fully recovered (fresh)
    - 0.1 = exhausted (max fatigue)
    - Empty history → MPC = 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class WorkoutSet:
    """Pojedyncza wykonana seria"""
    exercise_id: str              # Exercise ID (zgodne z get_exercises())
    weight_kg: float
    reps: int
    rir: Optional[int] = None     # Reps in Reserve (0-5), user poda czasem
    timestamp: datetime = field(default_factory=datetime.now)
    completed: bool = True

    def to_model_dict(self) -> dict:
        """
        Konwertuj do formatu dict wymaganego przez inference.predict_mpc():
          {exercise, weight_kg, reps, rir, timestamp}
        """
        return {
            "exercise": self.exercise_id,
            "weight_kg": self.weight_kg,
            "reps": self.reps,
            "rir": self.rir if self.rir is not None else 2,  # Default: RIR=2
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> dict:
        return {
            'exercise_id': self.exercise_id,
            'weight_kg': self.weight_kg,
            'reps': self.reps,
            'rir': self.rir,
            'timestamp': self.timestamp.isoformat(),
            'completed': self.completed
        }


@dataclass
class PlannedSet:
    """Zaplanowana seria"""
    exercise_id: str
    order: int
    reps: int
    weight_kg: float
    rir: Optional[int] = None
    predicted_rir: Optional[float] = None  # Z predict_rir()
    estimated_time_sec: int = 180
    primary_muscles: List[str] = field(default_factory=list)
    secondary_muscles: List[str] = field(default_factory=list)

    def to_workout_set(self, timestamp: Optional[datetime] = None) -> WorkoutSet:
        """Konwertuj planowaną serię na wykonaną (do historii)"""
        # Użyj predicted_rir jeśli user nie podał
        rir_to_use = self.rir
        if rir_to_use is None and self.predicted_rir is not None:
            rir_to_use = int(round(self.predicted_rir))

        return WorkoutSet(
            exercise_id=self.exercise_id,
            weight_kg=self.weight_kg,
            reps=self.reps,
            rir=rir_to_use,
            timestamp=timestamp or datetime.now(),
            completed=True,
        )

    def to_dict(self) -> dict:
        return {
            'exercise_id': self.exercise_id,
            'order': self.order,
            'reps': self.reps,
            'weight_kg': self.weight_kg,
            'rir': self.rir,
            'predicted_rir': self.predicted_rir,
            'estimated_time_sec': self.estimated_time_sec,
            'primary_muscles': self.primary_muscles,
            'secondary_muscles': self.secondary_muscles,
        }


@dataclass
class PlanResult:
    """Wynik planowania"""
    plan: List[PlannedSet]
    predicted_mpc_after: Dict[str, float]   # MPC per muscle po treningu (capacity)
    total_time_estimated_sec: int
    notes: List[str] = field(default_factory=list)
    used_real_model: bool = False           # Czy użyto prawdziwego DeepGain

    def to_dict(self) -> dict:
        return {
            'plan': [s.to_dict() for s in self.plan],
            'predicted_mpc_after': self.predicted_mpc_after,
            'total_time_estimated_sec': self.total_time_estimated_sec,
            'notes': self.notes,
            'used_real_model': self.used_real_model,
        }


@dataclass
class PlannerConfig:
    """
    Konfiguracja planner

    target_capacity_zones: dict[muscle_id, [min_capacity, max_capacity]]
        - min_capacity: najniższa akceptowalna capacity PO treningu (chroni przed over-fatigue)
        - max_capacity: najwyższa akceptowalna capacity PO treningu (wymusza wystarczający bodziec)

    Przykład dla quads: [0.60, 0.85]
        - Jeśli MPC_after < 0.60 → OVERFATIGUE (zbyt wyczerpany, ryzyko kontuzji)
        - Jeśli MPC_after > 0.85 → UNDERFATIGUE (nie było wystarczająco pracy)
        - Sweet spot: 0.60-0.85
    """
    target_capacity_zones: Dict[str, List[float]]
    default_reps_by_type: Dict[str, int]
    default_time_per_rep_sec: float = 2.5
    rest_between_sets_sec: int = 120
    target_rir: int = 2                                     # Planowanie do RIR=2 (moderate effort)
    volume_limit_per_muscle: Optional[Dict[str, float]] = None
    exercise_catalog: Optional[Dict[str, Dict]] = None      # metadata (type, czas, itd.)

    # Ile serii przypada na jedno ĆWICZENIE (każda seria = osobny PlannedSet w planie)
    sets_per_exercise_by_type: Dict[str, int] = field(default_factory=lambda: {
        "compound": 3,              # 3 serie compound (typowe)
        "compound_variation": 3,
        "isolation": 3,             # 3 serie isolation
        "core": 2,                  # 2 serie core (krócej)
    })

    def get_sets_count(self, ex_type: str) -> int:
        """Ile serii powinno być dla ćwiczenia tego typu"""
        return self.sets_per_exercise_by_type.get(ex_type, 3)

    def get_target_zone(self, muscle_id: str) -> List[float]:
        """Target capacity zone dla mięśnia (fallback to safe default)"""
        return self.target_capacity_zones.get(muscle_id, [0.55, 0.85])

    def get_volume_limit(self, muscle_id: str) -> float:
        if self.volume_limit_per_muscle:
            return self.volume_limit_per_muscle.get(muscle_id, 5000.0)
        return 5000.0


# ============================================================================
# Default target capacity zones (15 mięśni Michała)
# ============================================================================
DEFAULT_TARGET_CAPACITY_ZONES = {
    # Duże mięśnie (wolna regeneracja) - bardziej konserwatywnie
    "quads":          [0.60, 0.85],
    "hamstrings":     [0.60, 0.85],
    "glutes":         [0.60, 0.85],
    "lats":           [0.60, 0.85],
    "chest":          [0.55, 0.85],

    # Średnie
    "erectors":       [0.60, 0.85],
    "anterior_delts": [0.55, 0.85],
    "lateral_delts":  [0.55, 0.85],
    "rear_delts":     [0.55, 0.85],
    "rhomboids":      [0.55, 0.85],

    # Mniejsze (szybka regeneracja) - można agresywniej
    "biceps":         [0.45, 0.80],
    "triceps":        [0.45, 0.80],
    "calves":         [0.45, 0.80],
    "adductors":      [0.50, 0.85],

    # Core
    "abs":            [0.50, 0.85],
}


DEFAULT_DEFAULT_REPS_BY_TYPE = {
    "compound": 6,            # Main compounds - strength focus (6 reps @ RIR=2)
    "compound_variation": 8,  # Variations - hypertrophy (8 reps)
    "isolation": 12,          # Isolation - higher reps (12 reps)
    "core": 10,
}
