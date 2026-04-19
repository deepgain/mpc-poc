# WorkoutPlanner — Silnik Rekomendacyjny Treningów

**Z integracją DeepGain (Michał)** — pełny silnik planujący trening siłowy z optymalizacją zmęczenia mięśni.

## ⚡ Kluczowe Informacje

### MPC Semantyka (WAŻNE!)

```
MPC = Muscle Performance Capacity w [0.1, 1.0]
  - 1.0 = fresh (fully recovered)
  - 0.1 = exhausted
```

**Fresh user → MPC = 1.0 dla wszystkich mięśni.** Po treningu MPC spada, z czasem regeneruje się w górę.

### Architektura

```
┌─────────────────────────────────────────────────┐
│  planner.py (WorkoutPlanner)                    │
│    plan() / replan() / estimate_1rm()           │
│                    ↓                             │
│  models_wrapper.py                               │
│    - Try: inference.py (Michał's DeepGain)      │
│    - Fallback: MockModelHandle (heurystyka)     │
│                    ↓                             │
│  Model API:                                      │
│    predict_mpc(history, ts) → capacity          │
│    predict_rir(state, ex, w, r) → RIR           │
│    get_exercises() / get_muscles()              │
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
# Testy
python3 planner_tests.py

# Przykłady
python3 example_usage.py
```

## 📦 Pliki

| Plik | Opis |
|------|------|
| `planner.py` | **`WorkoutPlanner`** — plan() i replan() |
| `models_wrapper.py` | Adapter DeepGain + mock fallback |
| `inference.py` | **DeepGain API Michała** (stable contract) |
| `data_structures.py` | `WorkoutSet`, `PlannedSet`, `PlanResult`, `PlannerConfig` |
| `planner_tests.py` | 8 scenariuszy testowych |
| `example_usage.py` | 6 praktycznych przykładów |

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
models_wrapper.initialize_model()  # próbuje załadować deepgain_model_muscle_ord.pt

# 2. Stwórz planera
config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
    target_rir=2,  # Planuj pod RIR=2 (moderate)
)
planner = WorkoutPlanner(config)

# 3. Fresh state (pierwszy trening)
state = {m: 1.0 for m in planner.all_muscles}

# 4. Zaplanuj
result = planner.plan(
    state=state,
    n_compound=2,            # 2 ćwiczenia compound (× 3 serie każde)
    n_isolation=3,           # 3 ćwiczenia isolation (× 3 serie)
    available_time_sec=3600, # 1 godzina
)

# 5. Wyniki
for s in result.plan:
    print(f"{s.order}. {s.exercise_id}: {s.reps}×{s.weight_kg}kg "
          f"(predicted RIR: {s.predicted_rir:.1f})")

print(f"MPC after: {result.predicted_mpc_after}")
print(f"Total time: {result.total_time_estimated_sec/60:.1f} min")
print(f"Used real model: {result.used_real_model}")
```

### Replanning (zmiana w locie)

```python
# User wykonał pierwsze 3 serie (pierwsze ćwiczenie), odrzuca następne
completed = result.plan[:3]

replanned = planner.replan(
    session_so_far=completed,
    remaining_n_compound=1,
    remaining_n_isolation=3,
    available_time_sec=3000,
    exclusions=["sumo_deadlift"],  # To co odrzucił
)
```

### Z historią użytkownika

```python
from data_structures import WorkoutSet
from datetime import datetime, timedelta

user_history = [
    WorkoutSet('bench_press', 80.0, 5, rir=1,
               timestamp=datetime.now() - timedelta(days=7)),
    WorkoutSet('squat', 100.0, 5, rir=1,
               timestamp=datetime.now() - timedelta(days=5)),
]

# Planer sam obliczy state z historii
result = planner.plan(
    n_compound=2,
    n_isolation=2,
    available_time_sec=3600,
    user_history=user_history,  # ← auto-oblicza MPC i estymuje 1RM
)
```

### Exclusions i Preferences

```python
result = planner.plan(
    state=state,
    n_compound=2,
    n_isolation=3,
    available_time_sec=3600,
    exclusions=["deadlift", "sumo_deadlift"],  # Kontuzja lędźwi
    preferences={
        "favorites": ["incline_bench"],  # +0.2 score bonus
        "avoid": ["dips"],               # -0.5 score penalty
    },
)
```

## 🎯 Target Capacity Zones

```python
# Default (w data_structures.DEFAULT_TARGET_CAPACITY_ZONES)
{
    "quads":       [0.60, 0.85],  # Duże — bardziej konserwatywnie
    "hamstrings":  [0.60, 0.85],
    "chest":       [0.55, 0.85],  # Średnie
    "biceps":      [0.45, 0.80],  # Małe — można agresywniej
    "triceps":     [0.45, 0.80],
    "abs":         [0.50, 0.85],  # Core
    # ... wszystkie 15 mięśni
}
```

**Interpretacja:**
- MPC_after < `min` → **OVERFATIGUE** (niebezpieczne, duża kara w score)
- MPC_after > `max` → **UNDERFATIGUE** (mało bodźca, mniejsza kara)
- `min` ≤ MPC_after ≤ `max` → **SWEET SPOT** ✓

## 🔬 Integracja z DeepGain Michała

### Bezbolesna integracja

Planer używa `models_wrapper` jako adapter:

1. **Jeśli masz pliki Michała:**
   - `inference.py`
   - `deepgain_model_muscle_ord.pt`
   - `exercise_muscle_order.yaml`
   - `exercise_muscle_weights_scaled.csv`
   - `torch` + `pandas` + `pyyaml`

   → Wrapper **automatycznie** używa real DeepGain.

2. **Jeśli nie masz:** → wrapper fallback na MockModelHandle (heurystyka).

### Sprawdzenie który model używasz

```python
import models_wrapper

models_wrapper.initialize_model()
print(models_wrapper.is_using_real_model())
# True = DeepGain, False = Mock
```

### Forcing mock model (np. do testów)

```python
models_wrapper.initialize_model(force_mock=True)
```

## 📊 Testy

```bash
python3 planner_tests.py
```

8 scenariuszy:

| # | Test | Co sprawdza |
|---|------|-------------|
| 1 | Fresh User | Podstawowy workflow — upper + lower |
| 2 | Fatigued User | System oszczędza zmęczone mięśnie |
| 3 | Replanning | Zmiana planu w locie |
| 4 | Time Constraint | Respektowanie limitu czasu |
| 5 | Exercise Variety | Różnorodność w 10 scenariuszach |
| 6 | With User History | Auto-estymacja 1RM |
| 7 | Target Zones | Weryfikacja zakresów |
| 8 | Exclusions & Preferences | Filtry i preferencje |

## 🔄 Co się zmieniło (vs wcześniejsza wersja)

### ❗ MPC Semantyka ODWRÓCONA

| Stare | Nowe |
|-------|------|
| MPC = zmęczenie | MPC = capacity |
| 0.0 = fresh, 1.0 = zmęczone | **1.0 = fresh, 0.1 = exhausted** |
| fatigue zone [0.15, 0.40] | capacity zone [0.60, 0.85] |

### 🧠 Inne zmiany

- **15 mięśni** (nie 19) — zgodnie z DeepGain
- **34 ćwiczenia** z canonical names (`bench_press`, `squat`, `ohp`, ...)
- **PlannedSet = jedna seria** — `n_compound=2` znaczy 2 ĆWICZENIA (każde z `sets_per_exercise_by_type` serii, default 3)
- **predict_rir zintegrowane** — planer dobiera reps pod target RIR=2
- **to_model_dict()** — konwersja WorkoutSet → format dict Michała

## 🛠️ TODO / Future Work

- [ ] Hamulec na over-preferencję deadliftu (dominuje bo angażuje 7 mięśni)
- [ ] Beam search zamiast greedy (większa różnorodność)
- [ ] Volume limits per muscle (weight × reps)
- [ ] Per-user tau calibration (gdy user dostarczy dane)
- [ ] Periodyzacja (intensywność ↑↓ tygodniowo)

## 📝 Stable Contract

Interfejs `plan()` / `replan()` jest stabilny. Zmiany modelu (DeepGain) są niewidoczne dla klientów — wrapper zapewnia kompatybilność.

## 👤 Współpraca

- **Michał** (`models/`): `inference.py` + `.pt` checkpoint
- **Aleksander** (`dataset/`): tau per mięsień, target zones z literatury, volume limits
- **Miłosz** (planner): `WorkoutPlanner` (ten kod)
