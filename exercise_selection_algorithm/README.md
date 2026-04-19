# WorkoutPlanner v3 — Silnik Rekomendacyjny Treningów

**Z integracją DeepGain (Michał)** — pełny silnik planujący trening siłowy z optymalizacją zmęczenia mięśni.

## ⚡ Co nowego w v3

- ✅ **Deadlift dominance fix** — ważona średnia zamiast sumy (reward normalization)
- ✅ **Beam search** — `exploration_temperature` daje różnorodne plany
- ✅ **Volume limits per muscle** — hard limit weight × reps × engagement
- ✅ **UserProfile** — personalizacja tau regeneracji (beginner/advanced, wiek)

## ⚡ Kluczowe Informacje

### MPC Semantyka

```
MPC = Muscle Performance Capacity w [0.1, 1.0]
  - 1.0 = fresh (fully recovered)
  - 0.1 = exhausted
```

**Fresh user → MPC = 1.0** dla wszystkich mięśni.

### Architektura

```
┌─────────────────────────────────────────────────┐
│  planner.py (WorkoutPlanner)                    │
│    plan() / replan() / estimate_1rm()           │
│    Beam search + volume limits                   │
│                    ↓                             │
│  models_wrapper.py                               │
│    - Try: inference.py (Michał's DeepGain)      │
│    - Fallback: MockModelHandle (heurystyka)     │
│    - UserProfile → tau_scale                    │
│                    ↓                             │
│  Model API:                                      │
│    predict_mpc(history, ts) → capacity          │
│    predict_rir(state, ex, w, r) → RIR           │
└─────────────────────────────────────────────────┘
```

## 🚀 Szybki Start

### Wymagania

```bash
# Minimalne (mock model):
# Tylko Python 3.8+ (standard library)

# Dla real DeepGain:
pip install torch numpy pandas pyyaml
```

### Uruchomienie

```bash
# Testy (12 scenariuszy)
python3 planner_tests.py

# Przykłady (9 praktycznych scenariuszy)
python3 example_usage.py
```

## 📦 Pliki

| Plik | Opis |
|------|------|
| `planner.py` | **`WorkoutPlanner`** — plan() i replan() z beam search |
| `models_wrapper.py` | Adapter DeepGain + mock fallback |
| `inference.py` | **DeepGain API Michała** (stable contract) |
| `data_structures.py` | `WorkoutSet`, `PlannedSet`, `PlanResult`, `PlannerConfig`, `UserProfile` |
| `planner_tests.py` | 12 scenariuszy testowych |
| `example_usage.py` | 9 praktycznych przykładów |

## 💡 Użycie API

### Podstawowe planowanie

```python
from planner import WorkoutPlanner
from data_structures import (
    PlannerConfig,
    DEFAULT_TARGET_CAPACITY_ZONES,
    DEFAULT_DEFAULT_REPS_BY_TYPE,
)
import models_wrapper

# 1. Załaduj model (real → fallback mock)
models_wrapper.initialize_model()

# 2. Stwórz planera
config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
    target_rir=2,
)
planner = WorkoutPlanner(config)

# 3. Fresh state
state = {m: 1.0 for m in planner.all_muscles}

# 4. Zaplanuj
result = planner.plan(
    state=state,
    n_compound=2,            # 2 ćwiczenia compound (× 3 serie każde)
    n_isolation=3,
    available_time_sec=3600,
)

for s in result.plan:
    print(f"{s.order}. {s.exercise_id}: {s.reps}×{s.weight_kg}kg (RIR~{s.predicted_rir:.1f})")
```

### 🆕 Beam Search — różnorodne plany

```python
config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
    exploration_temperature=0.5,   # <-- 0 = greedy, >0 = różnorodność
    beam_width=3,                  # top-K do rozważenia
)
planner = WorkoutPlanner(config)

# Każdy call da potencjalnie inny plan!
for _ in range(3):
    result = planner.plan(state=fresh_state, n_compound=2, n_isolation=2, available_time_sec=3600)
    # Plan 1: [sumo_deadlift, close_grip_bench, ...]
    # Plan 2: [squat, ohp, ...]
    # Plan 3: [deadlift, incline_bench, ...]
```

### 🆕 Volume Limits per Muscle

```python
# Rekonwalescent kolan - niski limit na nogi
config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
    volume_limit_per_muscle={
        "quads": 1000.0,       # kg × reps × engagement (normalnie 5000)
        "hamstrings": 800.0,
        "glutes": 1200.0,
    },
)
```

Planer **skipuje** kandydata jeśli dodanie go przekroczyłoby limit dla jakiegokolwiek mięśnia.

### 🆕 UserProfile — per-user recovery calibration

```python
from data_structures import UserProfile

# Starszy początkujący — wolniejsza regeneracja
beginner_older = UserProfile(
    experience_level="beginner",   # tau × 1.20
    age_years=55,                  # tau × 1.15
    recovery_factor=1.0,           # dodatkowy mnożnik (stres, sen)
)

# Młody zaawansowany — szybsza regeneracja
advanced_young = UserProfile(
    experience_level="advanced",   # tau × 0.85
    age_years=22,                  # tau × 0.90
)

config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
    user_profile=beginner_older,   # <-- personalizacja
)
planner = WorkoutPlanner(config)

# Planer automatycznie dostosuje predykcję MPC (po 12h leg day):
#   Beginner 55 lat:   quads = 0.88  (wolniejsza regeneracja)
#   Intermediate 30:   quads = 0.91
#   Advanced 22:       quads = 0.92  (szybsza regeneracja)
```

**Tau scaling factors:**
- `experience_level`: beginner=1.20, intermediate=1.00, advanced=0.85
- `age_years`: <25=0.90, 25-40=1.00, 40-55=1.15, >55=1.30
- `recovery_factor`: bezpośredni mnożnik (stres, sen, dieta), 0.5-2.0

**Uwaga:** UserProfile wpływa tylko na **Mock model**. Real DeepGain ma tau zaszyte w checkpoincie — per-user calibration wymaga fine-tuningu modelu.

### Replanning (zmiana w locie)

```python
# User wykonał pierwsze 3 serie, odrzuca następne
completed = result.plan[:3]

replanned = planner.replan(
    session_so_far=completed,
    remaining_n_compound=1,
    remaining_n_isolation=3,
    available_time_sec=3000,
    exclusions=["sumo_deadlift"],
)
```

### Z historią użytkownika (auto 1RM)

```python
from data_structures import WorkoutSet
from datetime import datetime, timedelta

user_history = [
    WorkoutSet('bench_press', 80.0, 5, rir=1,
               timestamp=datetime.now() - timedelta(days=7)),
]

result = planner.plan(
    n_compound=2,
    n_isolation=2,
    available_time_sec=3600,
    user_history=user_history,  # ← auto-oblicza MPC i estymuje 1RM
)
```

## 🎯 Target Capacity Zones

```python
# DEFAULT_TARGET_CAPACITY_ZONES
{
    "quads":       [0.60, 0.85],  # Duże — bardziej konserwatywnie
    "hamstrings":  [0.60, 0.85],
    "chest":       [0.55, 0.85],
    "biceps":      [0.45, 0.80],  # Małe — można agresywniej
    "triceps":     [0.45, 0.80],
    "abs":         [0.50, 0.85],
    # ... wszystkie 15 mięśni
}
```

## 📊 Volume Limits (default)

```python
# DEFAULT_VOLUME_LIMITS (kg × reps × engagement per session)
{
    "chest":          4000.0,
    "quads":          5000.0,    # Duże mięśnie wytrzymują więcej
    "hamstrings":     4000.0,
    "lats":           3500.0,
    "triceps":        2000.0,    # Średnie
    "biceps":         1800.0,
    "calves":         1200.0,    # Małe
    "abs":            1500.0,
    # ... wszystkie 15 mięśni
}
```

## 🔬 Integracja z DeepGain Michała

### Bezbolesna integracja

1. **Jeśli masz pliki Michała:**
   - `inference.py`, `deepgain_model_muscle_ord.pt`
   - `exercise_muscle_order.yaml`
   - `exercise_muscle_weights_scaled.csv`
   - `torch` + `pandas` + `pyyaml`

   → Wrapper **automatycznie** używa real DeepGain.

2. **Jeśli nie masz:** → wrapper fallback na MockModelHandle (heurystyka).

```python
import models_wrapper

models_wrapper.initialize_model()
print(models_wrapper.is_using_real_model())
# True = DeepGain, False = Mock
```

## 📊 Testy

```bash
python3 planner_tests.py
```

### 12 scenariuszy:

| # | Test | Status |
|---|------|--------|
| 1 | Fresh User | ✓ |
| 2 | Fatigued User (po leg day) | ✓ |
| 3 | Replanning | ✓ |
| 4 | Time Constraint (10 min) | ✓ |
| 5 | Exercise Variety | ✓ |
| 6 | Planning with User History | ✓ |
| 7 | Target Zones Verification | ✓ |
| 8 | Exclusions & Preferences | ✓ |
| **9** | **Deadlift Dominance Fix** | ✓ 0/5 jako #1 |
| **10** | **Beam Search Exploration** | ✓ 8/8 unique plans (temp=0.5) |
| **11** | **Volume Limits** | ✓ Leg volume = 0/500 przy niskim limicie |
| **12** | **UserProfile → Tau** | ✓ Advanced recovers +0.04 MPC szybciej |

## 🔄 Co się zmieniło (v2 → v3)

### 🧠 Algorytm

- **Reward = ważona średnia** (nie suma) × capacity — eliminuje deadlift dominance
- **Breadth bonus** ograniczony do 0.02-0.04 (było 0.05-0.15)
- **Beam search**: sortuj po score, wybierz z top-K via softmax sampling
- **Volume tracking** per muscle — `defaultdict(float)` w plan()/replan()

### 📋 PlannerConfig — nowe pola

```python
@dataclass
class PlannerConfig:
    # Existing
    target_capacity_zones: dict
    default_reps_by_type: dict
    target_rir: int = 2
    volume_limit_per_muscle: Optional[dict] = None

    # NEW in v3
    exploration_temperature: float = 0.0   # 0 = greedy, >0 = diverse
    beam_width: int = 3
    user_profile: Optional[UserProfile] = None
```

### 🆕 UserProfile

```python
@dataclass
class UserProfile:
    experience_level: str = "intermediate"  # beginner/intermediate/advanced
    age_years: Optional[int] = None
    recovery_factor: float = 1.0            # 0.5-2.0
    bodyweight_kg: Optional[float] = None

    def get_tau_scale(self) -> float:
        # Zwraca mnożnik tau (1.0 = baseline)
```

## 🛠️ TODO (nie w tej wersji)

- [ ] **Periodyzacja** — intensywność ↑↓ tygodniowo (wave/linear/DUP)
- [ ] Real DeepGain fine-tuning per-user (jeśli potrzebne — obecna calibracja działa w Mock)
- [ ] Auto-detect exercise difficulty dla beginners (redukcja weights przy małej historii)
- [ ] Cross-session volume tracking (weekly limits zamiast per-session)

## 📝 Stable Contract

Interfejs `plan()` / `replan()` jest stabilny. Zmiany modelu (DeepGain) są niewidoczne dla klientów — wrapper zapewnia kompatybilność.

## 👤 Współpraca

- **Michał** (`models/`): `inference.py` + `.pt` checkpoint
- **Aleksander** (`dataset/`): tau per mięsień, target zones z literatury, volume limits
- **Miłosz** (planner): `WorkoutPlanner` (ten kod)
