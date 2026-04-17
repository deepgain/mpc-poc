# WorkoutPlanner - Silnik Rekomendacyjny Treningów

Pełna implementacja systemu planowania treningów z optimizacją stanu zmęczenia mięśni (MPC).

## 🚀 Szybki Start

### Instalacja
```bash
# Requirements: Python 3.8+, pandas, openpyxl
pip install pandas openpyxl
```

### Uruchomienie testów
```bash
python3 planner_tests.py
```

Zobaczy 6 scenariuszy testowych z logami:
1. **Fresh user** - nowy użytkownik (wszystko MPC=0)
2. **Fatigued user** - użytkownik po leg workout
3. **Replanning** - zmiana planu w locie
4. **Time constraint** - krótka sesja (10 min)
5. **Exercise variety** - różnorodność ćwiczeń
6. **With history** - estymacja 1RM z historii

---

## 📋 Struktura Plików

| Plik | Opis |
|------|------|
| `planner.py` | **Główna klasa `WorkoutPlanner`** - plan() i replan() |
| `models_mock.py` | Mock modelu DeepGain (heurystyka zmęczenia) |
| `data_structures.py` | Klasy: WorkoutSet, PlannedSet, PlanResult |
| `exercises_config.json` | Katalog 34 ćwiczeń + target fatigue zones |
| `planner_tests.py` | 6 scenariuszy testowych |
| `PLANNER_REPORT.md` | Raport detailowy (testy, problemy, ulepszenia) |

---

## 💡 Użycie API

### 1. Planowanie treningu
```python
from planner import WorkoutPlanner
from data_structures import PlannerConfig, WorkoutSet
import json
from datetime import datetime

# Wczytaj exercises i config
with open('exercises_config.json', 'r') as f:
    exercises_config = json.load(f)

planner_config = PlannerConfig(
    target_fatigue_zones=exercises_config['target_fatigue_zones'],
    default_reps_by_type=exercises_config['default_reps_by_type'],
)

planner = WorkoutPlanner(exercises_config, planner_config)

# Definiuj stan i preferencje
current_mpc_state = {
    'quadriceps': 0.2,
    'hamstring': 0.15,
    'chest_upper': 0.0,
    # ... wszystkie mięśnie
}

# Zaplanuj trening
result = planner.plan(
    state=current_mpc_state,
    n_compound=2,           # ile ćwiczeń compound
    n_isolation=3,          # ile ćwiczeń isolation
    available_time_sec=3600,  # 1 godzina
    user_history=[],        # opcjonalnie: historia treningów
    exclusions=['back_squat'],  # opcjonalnie: czego unikać
)

# Wyniki
print(f"Plan: {len(result.plan)} serii")
for planned_set in result.plan:
    print(f"  {planned_set.order}. {planned_set.exercise_id}: {planned_set.reps}x{planned_set.weight_kg}kg")

print(f"\nPredicted MPC after:")
for muscle, mpc in result.predicted_mpc_after.items():
    print(f"  {muscle}: {mpc:.2f}")

print(f"\nValidation notes:")
for note in result.notes:
    print(f"  {note}")
```

### 2. Replanning (zmiana w locie)
```python
# User wykonał pierwsze 2 serie, odrzuca trzecią
completed_sets = result.plan[:2]

# Przeplanuj resztę
remaining_result = planner.replan(
    session_so_far=completed_sets,
    remaining_n_compound=0,     # już miał 2
    remaining_n_isolation=2,    # potrzebuje jeszcze 2
    current_state=current_mpc_state,
    available_time_sec=1800,    # czas został
)

print(f"New plan:")
for s in remaining_result.plan:
    print(f"  {s.order}. {s.exercise_id}")
```

### 3. Estymacja 1RM z historii
```python
# Jeśli masz historię użytkownika
user_history = [
    WorkoutSet('back_squat', 80.0, 8, rir=2, timestamp=datetime.now()),
    WorkoutSet('bench_press', 60.0, 10, rir=1, timestamp=datetime.now()),
]

estimated_1rm = planner.estimate_1rm_from_history(user_history)
print(estimated_1rm)
# {'back_squat': 101.3, 'bench_press': 80.0, ...}
```

---

## 🔧 Konfiguracja

### Target Fatigue Zones (`exercises_config.json`)
```json
{
  "target_fatigue_zones": {
    "quadriceps": [0.15, 0.40],      // Conservative - duże mięśnie
    "hamstring": [0.15, 0.40],
    "chest_upper": [0.20, 0.45],     // Średnio agresywnie
    "biceps": [0.25, 0.55],          // Bardziej agresywnie - mniejsze
    "abs": [0.20, 0.50],
    "multifidus": [0.10, 0.35]       // Core - konserwatywnie
  }
}
```

Interpretacja:
- **Min**: wystarczający bodziec (musiał być zmęczony)
- **Max**: bezpieczna granica (nie przetrenowanie)

### Domyślne parametry
```json
{
  "default_reps_by_type": {
    "compound": 8,
    "compound_variation": 8,
    "isolation": 12,
    "core": 10
  },
  "default_time_per_rep_sec": 2.5,
  "rest_between_sets_sec": 120
}
```

---

## 🤖 Model DeepGain (Integration)

### Aktualnie: Mock heurystyka
```python
# models_mock.py
MPC_delta = sum(
    engagement_ratio × 
    intensity_factor × 
    exponential_decay
)

intensity = f(reps, RIR, weight)
decay = exp(-hours_since / tau_muscle)
```

### Docelowo: Real DeepGain
Swap imports:
```python
# Zmień to:
from models_mock import predict_mpc

# Na to:
from models_deepgain import predict_mpc  # (gdy Michał dostarczy)
```

**Interfejs nie zmieni się** - kontrakt jest stabilny.

---

## 📊 Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| Fresh User | ✓ PASS | Generuje 6-set plan |
| Fatigued User | ⚠ PARTIAL | Hamstring over-fatigue (data issue) |
| Replanning | ✓ PASS | Przeplanowuje bezproblemowo |
| Time Constraint | ✓ PASS | Respektuje 10-min limit |
| Exercise Variety | ⚠ LOW | 23% coverage (problem: greedy selects same) |
| With History | ✓ PASS | 1RM estimation works |

**Full results:** See `PLANNER_REPORT.md`

---

## ⚠️ Known Issues & TODOs

### High Priority
1. **Hamstring over-fatigue** w Test 1
   - Przyczyna: Ratios.xlsx data - normalize muscle_engagement

2. **Niska diversność ćwiczeń**
   - Powód: Greedy zawsze wybiera best score
   - Fix: Beam search (top-3) + random selection

3. **Brak volume limits**
   - Feature structure gotowy, ale nie implemented w `_select_best_exercise()`

### Medium Priority
4. RIR collection (brakuje UI)
5. Per-user 1RM defaults
6. Better soft preferences

### Low Priority
7. Stochastic planning (beam search)
8. Learning from feedback

---

## 🔗 Integracje

### Z DeepGain (Michał)
- Interface gotowy w `models_mock.py`
- Swap imports w `planner.py`

### Z Dataset (Aleksander)
- Exercise catalog z Ratios.xlsx ✓
- Czekam na: `exercise_muscle_order.yaml`
- Czekam na: Target zone charts z modelu

### Z UI (Frontend)
- API gotowe (plan/replan methods)
- Czekam na: RIR collection, logging, preferences UI

---

## 📚 Documentation

- **PLANNER_REPORT.md** - Detailed architecture, test analysis, improvements
- **Code docstrings** - In-file documentation

---

## 🎯 Next Steps

1. **Fix data issues** (hamstring over-fatigue)
2. **Implement volume limits** (per muscle)
3. **Improve diversification** (beam search)
4. **Integrate real DeepGain** (Michał)
5. **Integrate with UI** (frontend)

---

## 📝 License

Part of training recommendation engine for muscle fatigue optimization.

---

## 👤 Author

Implemented by Claude for [Project Name]

---

## 🚨 Support

Questions about:
- **Planner logic** → Check `planner.py` docstrings
- **Test failures** → Run `planner_tests.py` with logging
- **Integration** → See `PLANNER_REPORT.md` section "Integration"
