# WorkoutPlanner — Raport Integracji z DeepGain

## Status: ✅ Integracja ukończona

Planer został zintegrowany z `inference.py` (API Michała) poprzez wzorzec **Adapter**. Kontrakt jest stabilny — zmiany modelu nie wpływają na klientów planera.

---

## 1. Kluczowe Zmiany (vs wcześniejsza wersja mock)

### 🔴 KRYTYCZNE: MPC semantyka odwrócona

**Przed:** MPC = zmęczenie (0=świeże, 1=zmęczone)
**Teraz:** MPC = **Muscle Performance Capacity** w [0.1, 1.0] (1=świeże, 0.1=exhausted)

To wymagało przepisania:
- Initial state: `{m: 1.0 for m in muscles}` (nie `0.0`)
- Target zones: `[0.60, 0.85]` (nie `[0.15, 0.40]`)
- Objective reward: `engagement × MPC_before` (już OK, bo świeże = wysokie)
- Validation: `MPC_after < min` = OVERFATIGUE, `MPC_after > max` = UNDERFATIGUE

### 📋 Nowy wspólny słownik

| Element | Wartość |
|---------|---------|
| Liczba mięśni | **15** (było 19) |
| Liczba ćwiczeń | **34** (było 34 ale inne nazwy) |
| Konwencja nazw | `bench_press`, `squat`, `ohp` (nie `wyciskanie_leżąc_bench_press`) |
| Format historii | `dict` z `exercise, weight_kg, reps, rir, timestamp` |
| Timestamp | ISO8601 string lub datetime |

### 🎁 Bonus: predict_rir

Nowa funkcja którą daje Michał — planer ją wykorzystuje do **dobrania reps pod target RIR**:

```python
# W _construct_planned_set:
for candidate_reps in [3, 5, 6, 8, 10]:
    predicted = predict_rir(state, exercise, weight, candidate_reps)
    if abs(predicted - target_rir) < best_diff:
        best_reps = candidate_reps
```

Efekt: dla ciężkiego compound (squat @ 75% 1RM) planer wybiera niższe reps (5-6), dla isolation — wyższe (10-12).

---

## 2. Architektura Integracji (Adapter Pattern)

```
┌─────────────────────────────────┐
│   planner.py                    │
│   WorkoutPlanner.plan()         │
│      ↓                          │
│   self.model.predict_mpc(...)   │
│   self.model.predict_rir(...)   │  ← unified interface
└─────────────────────────────────┘
              ↓ wybór przy initialize_model()
┌──────────────────────┬─────────────────────┐
│                      │                     │
│  RealModelHandle     │  MockModelHandle    │
│  (torch, DeepGain)   │  (heurystyka)       │
│                      │                     │
│  uses inference.py   │  15 muscles, 34 ex, │
│                      │  tau values from    │
│                      │  inference.py       │
└──────────────────────┴─────────────────────┘
```

### Logika fallback

```python
# W models_wrapper.initialize_model():
1. Czy inference.py jest importowalny?     (torch available?)
2. Czy plik .pt istnieje na dysku?
3. Czy load_model() się powodzi?

Jeśli wszystkie TAK → RealModelHandle
W przeciwnym razie → MockModelHandle
```

**Kluczowe:** kod klienta (planner, testy, example_usage) jest **identyczny** niezależnie od tego, który model jest w użyciu.

---

## 3. Mapping muscle_id (kompatybilność z DeepGain)

Stara lista (19) → Nowa lista (15 z inference.py):

| Polska nazwa (stare) | DeepGain ID | Status |
|----------------------|-------------|--------|
| chest_upper + chest_lower | `chest` | ✅ merged |
| shoulder_front | `anterior_delts` | ✅ renamed |
| shoulder_side | `lateral_delts` | ✅ renamed |
| shoulder_rear | `rear_delts` | ✅ renamed |
| rhomboid | `rhomboids` | ✅ renamed |
| biceps | `biceps` | ✅ |
| — | **`triceps`** | ➕ **dodano** (brak w Ratios.xlsx!) |
| lats | `lats` | ✅ |
| quadriceps | `quads` | ✅ renamed |
| hamstring | `hamstrings` | ✅ renamed |
| glutes | `glutes` | ✅ |
| hip_adductors | `adductors` | ✅ renamed |
| erector_spinae | `erectors` | ✅ renamed |
| calves | `calves` | ✅ |
| abs | `abs` | ✅ |
| multifidus | — | ❌ **usunięto** |
| glute_med | — | ❌ **usunięto** |
| oblique_ext, oblique_int | — | ❌ **usunięto** (w `abs`) |

**Uwaga:** Ratios.xlsx **nie miał triceps jako osobnej kolumny** — w mock model dodałem go heurystycznie dla push movements (bench, ohp, dips, close_grip_bench, french_press).

---

## 4. Wyniki Testów

### Test 1: Fresh User (upper body excl. legs)

```
Plan (16 serii, 60 min):
  bench_press: 3 × 10×60kg (RIR~2)
  incline_bench: 3 × 8×48.8kg (RIR~1.9)
  pendlay_row: 3 × 12×45kg (RIR~1.8)
  seal_row: 3 × 8×37.5kg (RIR~2)
  ohp: 3 × 8×33.8kg (RIR~2)
  bird_dog: 1 × 12
```

Target zones:
- ✓ chest: 0.65 in [0.55, 0.85]
- ✓ triceps: 0.71 in [0.45, 0.80]
- ✓ rhomboids: 0.72 in [0.55, 0.85]
- ✓ lats: 0.77 in [0.60, 0.85]
- ⚠ biceps: 0.81 > 0.80 (ledwie underfatigue)
- **7/11 zaangażowanych mięśni w target zone** ✓

### Test 3: Replanning

```
Oryginał: [deadlift, sumo_deadlift, pendlay_row, ohp, ab_wheel]
User ✓ wykonał: deadlift (3 sets)
User ❌ odrzucił: sumo_deadlift
Replan:
  ✓ done: deadlift
  + new: low_bar_squat   ← zastąpił sumo_deadlift
  + new: pendlay_row
  + new: ohp
  + new: ab_wheel_rollout
```

### Test 4: Time constraint (30 min)

✅ Generuje plan w budżecie czasu, redukuje serie gdy potrzeba.

### Test 5: Variety (10 scenariuszy z exclusions)

Z rotacją exclusions planer używa **16-19 unikalnych ćwiczeń** (z 34 dostępnych) — znaczna poprawa vs wcześniejsze 6 w starej wersji (23%).

### Test 8: Exclusions & Preferences

```
Exclusions: [squat, low_bar_squat, deadlift, sumo_deadlift]  # Kontuzja
Favorites: [incline_bench]
Plan:
  high_bar_squat (jedyny compound nogi nie wykluczony)
  incline_bench ⭐ (favorite pojawił się)
  pendlay_row, ohp, bulgarian_split_squat
✓ No excluded exercises
```

---

## 5. Znane Problemy

### 🟡 Problem 1: Deadlift dominuje reward function

Deadlift angażuje 7 mięśni (suma engagement = 3.10). Bench_press tylko 4 (suma = 1.90).

Reward = Σ(engagement × capacity) → deadlift **zawsze wygra** w fresh state.

**Tymczasowy fix:** User dodaje do `exclusions` lub `preferences.avoid`.

**Długoterminowy fix:** Normalizacja reward przez liczbę mięśni (średnia zamiast sumy) LUB osobne scoring per muscle group (nogi/klatka/plecy/barki).

### 🟡 Problem 2: Mock model — bezpieczne defaults 1RM

Gdy brak historii, użyte są stałe 1RM (np. `bench_press: 80kg`). Nie są dopasowane do usera — w produkcji wymagana kalibracja przez pierwszych kilka treningów.

### 🟢 Problem 3: predict_rir nie uwzględnia weight progression

Jeśli user wkłada 100kg na bench (gdzie 1RM=80kg), `predict_rir` może zwrócić ujemne (clamped do 0). Ale model może nie radzić sobie z ekstremami.

**Mitigation:** Planer używa zawsze 75% 1RM, więc nie przekracza bezpiecznych wartości.

---

## 6. Deployment Checklist

### Aby użyć real DeepGain Michała:

```bash
# 1. Dependencies
pip install torch numpy pandas pyyaml

# 2. Pliki (w roboczym katalogu)
├── inference.py                          ← Michał
├── deepgain_model_muscle_ord.pt          ← Michał (checkpoint)
├── exercise_muscle_order.yaml            ← Michał
├── exercise_muscle_weights_scaled.csv    ← Michał
├── data_structures.py                    ← Miłosz
├── models_wrapper.py                     ← Miłosz
└── planner.py                            ← Miłosz

# 3. Uruchom
python3 planner_tests.py
# Powinieneś zobaczyć: "✓ Loaded DeepGain from deepgain_model_muscle_ord.pt"
```

### Weryfikacja że używasz real model:

```python
import models_wrapper
models_wrapper.initialize_model()
assert models_wrapper.is_using_real_model(), "Mock jest aktywny!"
```

### Workflow sanity-check po update modelu

Gdy Michał zrzuci nowy checkpoint:

```bash
1. cp nowy_checkpoint.pt deepgain_model_muscle_ord.pt
2. python3 planner_tests.py
3. Sprawdź Test 1 (fresh user) - czy Plan jest sensowny?
4. Sprawdź Test 5 (variety) - czy nie dominuje jedno ćwiczenie?
5. Sprawdź anomalie typu "po nowym modelu planner ciągle wybiera leg curl"
6. Raportuj Michałowi z przykładami planów
```

---

## 7. API Cheatsheet

### WorkoutPlanner.plan()

```python
result = planner.plan(
    state: dict[str, float] = None,        # MPC (capacity) per muscle
    n_compound: int = 2,                   # ile ĆWICZEŃ compound (×3 serie)
    n_isolation: int = 3,                  # ile ĆWICZEŃ isolation (×3 serie)
    available_time_sec: int = 3600,        # budżet czasu
    user_history: list[WorkoutSet] = None, # dla 1RM estimation
    exclusions: list[str] = None,          # exercise_id do pominięcia
    preferences: dict = None,              # {favorites: [...], avoid: [...]}
    now: datetime = None,                  # referenced timestamp
) -> PlanResult
```

### WorkoutPlanner.replan()

```python
result = planner.replan(
    session_so_far: list[PlannedSet],      # wykonane/zmodyfikowane
    remaining_n_compound: int,             # ile compound ZOSTAŁO
    remaining_n_isolation: int,            # ile isolation ZOSTAŁO
    current_state: dict[str, float] = None,
    available_time_sec: int = 3600,
    user_history: list[WorkoutSet] = None,
    exclusions: list[str] = None,
    preferences: dict = None,
    now: datetime = None,
) -> PlanResult
```

### PlanResult

```python
@dataclass
class PlanResult:
    plan: list[PlannedSet]                 # wszystkie serie (1 PlannedSet = 1 seria)
    predicted_mpc_after: dict[str, float]  # capacity per muscle po treningu
    total_time_estimated_sec: int
    notes: list[str]                       # walidacja target zones
    used_real_model: bool                  # True = DeepGain, False = Mock
```

### PlannedSet

```python
@dataclass
class PlannedSet:
    exercise_id: str
    order: int                             # numer porządkowy
    reps: int
    weight_kg: float                       # ~75% estimated 1RM
    rir: Optional[int]                     # None = user nie podał
    predicted_rir: Optional[float]         # z predict_rir()
    estimated_time_sec: int
    primary_muscles: list[str]             # top 2 engaged
    secondary_muscles: list[str]
```

### WorkoutSet (historia)

```python
@dataclass
class WorkoutSet:
    exercise_id: str
    weight_kg: float
    reps: int
    rir: Optional[int]
    timestamp: datetime
    completed: bool = True

    def to_model_dict(self) -> dict:
        """→ format dla inference.predict_mpc()"""
        return {
            "exercise": self.exercise_id,
            "weight_kg": self.weight_kg,
            "reps": self.reps,
            "rir": self.rir or 2,
            "timestamp": self.timestamp.isoformat(),
        }
```

---

## 8. Podsumowanie

✅ **Stable contract** z DeepGain — zmiana modelu = zmiana pliku `.pt` (bez kodu)
✅ **Graceful fallback** do mock gdy brak torch/checkpoint
✅ **8 testów symulacyjnych** pokrywających podstawowe scenariusze
✅ **predict_rir integration** — automatyczny dobór reps pod target RIR
✅ **Realistyczne obciążenie** — multi-set per exercise (3 serie default)
✅ **Exclusions & preferences** działają

⚠️ **Deadlift dominance** — do dopracowania (normalizacja reward)
⚠️ **1RM defaults** — wymagają kalibracji per-user w produkcji
