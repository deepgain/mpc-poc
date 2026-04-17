"""Data structures dla WorkoutPlanner"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class WorkoutSet:
    """Pojedyncza seria treningu"""
    exercise_id: str
    weight_kg: float
    reps: int
    rir: Optional[int] = None  # Reps in Reserve (user poda czasem)
    timestamp: datetime = field(default_factory=datetime.now)
    completed: bool = True

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
    """Planowana seria"""
    exercise_id: str
    order: int
    reps: int
    weight_kg: float
    rir: Optional[int] = None
    estimated_time_sec: int = 180
    primary_muscles: List[str] = field(default_factory=list)
    secondary_muscles: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'exercise_id': self.exercise_id,
            'order': self.order,
            'reps': self.reps,
            'weight_kg': self.weight_kg,
            'rir': self.rir,
            'estimated_time_sec': self.estimated_time_sec,
            'primary_muscles': self.primary_muscles,
            'secondary_muscles': self.secondary_muscles,
        }


@dataclass
class PlanResult:
    """Wynik planowania"""
    plan: List[PlannedSet]
    predicted_mpc_after: Dict[str, float]
    total_time_estimated_sec: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'plan': [s.to_dict() for s in self.plan],
            'predicted_mpc_after': self.predicted_mpc_after,
            'total_time_estimated_sec': self.total_time_estimated_sec,
            'notes': self.notes
        }


@dataclass
class PlannerConfig:
    """Konfiguracja planner"""
    target_fatigue_zones: Dict[str, List[float]]  # muscle_id -> [min, max]
    default_reps_by_type: Dict[str, int]
    default_time_per_rep_sec: float = 2.5
    rest_between_sets_sec: int = 120
    volume_limit_per_muscle: Optional[Dict[str, float]] = None  # weight*reps limit
    max_sessions_without_exercise: Dict[str, int] = field(default_factory=lambda: {})  # prevent monotony

    def get_target_zone(self, muscle_id: str) -> List[float]:
        return self.target_fatigue_zones.get(muscle_id, [0.2, 0.5])

    def get_volume_limit(self, muscle_id: str) -> float:
        if self.volume_limit_per_muscle:
            return self.volume_limit_per_muscle.get(muscle_id, 500.0)
        return 500.0  # default
