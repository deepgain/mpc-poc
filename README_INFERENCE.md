# Inference README

Ten dokument opisuje aktualny kontrakt integracyjny dla `inference.py`.
Jest przeznaczony głównie dla planera Miłosza.

## Zakres

`inference.py` odpowiada za:

- załadowanie checkpointu
- odtworzenie aktualnego `MPC` z historii użytkownika
- predykcję `RIR` dla planowanego seta
- projekcję `1RM` dla konkretnego ćwiczenia
- update anchorów `1RM` po zakończonej sesji

Aktualny wariant używa:

- 3 onboardingowych anchorów `1RM`
  - `bench_press`
  - `squat`
  - `deadlift`
- dynamicznego replayu `MPC`
- dynamicznego update'u anchorów `1RM` w trybie `per-session`

## Public API

```python
from inference import (
    load_model,
    predict_mpc,
    predict_rir,
    project_exercise_1rm,
    update_strength_anchors,
    get_exercises,
    get_muscles,
)
```

### `load_model(checkpoint_path, device=None) -> DeepGainModel`

Ładuje checkpoint `.pt` zapisany przez `train.py`.

Ważne:

- auto-wykrywa `embed_dim`, `hidden_dim` i `strength_feature_dim`
- działa zarówno ze starymi, jak i nowymi checkpointami
- zwraca model w `eval()`

Przykład:

```python
model = load_model("deepgain_model_best.pt")
```

### `predict_mpc(model, user_history, timestamp, strength_anchors=None) -> dict[str, float]`

Odtwarza historię użytkownika do chwili `timestamp` i zwraca aktualny `MPC` dla 15 mięśni.

Wymagany format historii:

```python
user_history = [
    {
        "exercise": "bench_press",
        "weight_kg": 80.0,
        "reps": 5,
        "rir": 2,
        "timestamp": "2026-04-20T18:05:00",
    },
]
```

Opcjonalnie rekordy mogą zawierać onboardingowe kolumny:

- `config_1rm_bench_press`
- `config_1rm_squat`
- `config_1rm_deadlift`

Ważne zachowanie:

- sety po `timestamp` są ignorowane
- nieznane ćwiczenia są pomijane
- jeśli nie ma historii, wszystkie mięśnie wracają jako `1.0`
- replay historii używa tej samej logiki `per-session` co trening
- nowa sesja jest wykrywana przy przerwie `> 6h`
- anchory `1RM` są aktualizowane dopiero po zakończeniu sesji, nie po każdym secie

Zwracany zakres:

- `MPC` w `[0.1, 1.0]`

Przykład:

```python
mpc = predict_mpc(
    model,
    user_history=history,
    timestamp="2026-04-23T09:00:00",
    strength_anchors={
        "bench_press": 100.0,
        "squat": 140.0,
        "deadlift": 180.0,
    },
)
```

### `predict_rir(model, state, exercise, weight, reps, strength_anchors=None) -> float`

Przewiduje `RIR` dla planowanego seta na podstawie:

- aktualnego `MPC`
- planowanego ćwiczenia
- ciężaru
- liczby powtórzeń
- aktualnych anchorów `1RM`

Ważne:

- `state` to zwykle wynik `predict_mpc(...)`
- brakujące mięśnie w `state` domyślnie dostają `1.0`
- jeśli `exercise` nie jest znane modelowi, funkcja rzuca `ValueError`
- zwracany `RIR` jest obcinany do zakresu `[0.0, 5.0]`

Przykład:

```python
rir = predict_rir(
    model,
    state=mpc,
    exercise="bench_press",
    weight=80.0,
    reps=5,
    strength_anchors={
        "bench_press": 100.0,
        "squat": 140.0,
        "deadlift": 180.0,
    },
)
```

### `project_exercise_1rm(exercise, strength_anchors=None) -> float | None`

Rzutuje 3 anchory `1RM` na konkretną wariację ćwiczenia przez priory `anchor + ratio`.

Przykład:

```python
projected = project_exercise_1rm(
    "incline_bench",
    strength_anchors={"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0},
)
```

### `update_strength_anchors(strength_anchors, completed_sets, ...)`

Aktualizuje anchory `1RM` po zakończonej sesji.

To jest ta sama logika, której używa:

- trening do budowy `estimated_1rm_before_set`
- `predict_mpc(...)` do replayu historii
- walidacja `validate_1rm_dynamics.py`

Minimalny format `completed_sets`:

```python
completed_sets = [
    {
        "exercise": "bench_press",
        "weight_kg": 85.0,
        "reps": 5,
        "rir": 1,
        "timestamp": "2026-04-23T18:15:00",
    },
]
```

Domyślne zasady update'u:

- aktualizacja `per-session`
- kandydat `e1RM` liczony wzorem `Epley + RIR`
- tylko sensowne sety:
  - `reps <= 10`
  - `RIR <= 3`
  - wystarczająco duży `relative_load`
- miękki update przez `EMA`
- cap względnej zmiany na jeden update

Przykład:

```python
new_anchors = update_strength_anchors(
    strength_anchors={
        "bench_press": 100.0,
        "squat": 140.0,
        "deadlift": 180.0,
    },
    completed_sets=session_sets,
)
```

## Zalecany flow integracyjny

### 1. Onboarding użytkownika

Przy zakładaniu konta zbierasz:

```python
strength_anchors = {
    "bench_press": 100.0,
    "squat": 140.0,
    "deadlift": 180.0,
}
```

Te wartości planner/backend powinny przechowywać jawnie.

### 2. Przed planowaniem / przed sesją

Wyznaczasz aktualny stan mięśni z historii:

```python
mpc = predict_mpc(
    model,
    user_history=history,
    timestamp=now,
    strength_anchors=strength_anchors,
)
```

### 3. Ocena kandydackiego seta

Planner pyta model o przewidywany `RIR`:

```python
rir = predict_rir(
    model,
    state=mpc,
    exercise="incline_bench",
    weight=72.5,
    reps=6,
    strength_anchors=strength_anchors,
)
```

### 4. Po zakończeniu sesji

Aktualizujesz anchory na podstawie realnie wykonanych setów:

```python
strength_anchors = update_strength_anchors(
    strength_anchors,
    completed_sets=session_sets,
)
```

Nowe anchory są używane dopiero od kolejnej sesji / kolejnego replayu historii.

## Rekomendacje praktyczne

- Planner powinien przekazywać `strength_anchors` jawnie przy `predict_mpc(...)` i `predict_rir(...)`.
- Nie polegaj tylko na tym, że anchory zostaną wyciągnięte z `user_history`.
- Do update'u przekazuj pełne, zakończone sesje, nie pojedyncze sety z połowy treningu.
- Traktuj `strength_anchors` jako stan użytkownika przechowywany po stronie backendu / planera.

## Minimalny przykład end-to-end

```python
from inference import load_model, predict_mpc, predict_rir, update_strength_anchors

model = load_model("deepgain_model_best.pt")

strength_anchors = {
    "bench_press": 100.0,
    "squat": 140.0,
    "deadlift": 180.0,
}

history = [
    {
        "exercise": "bench_press",
        "weight_kg": 80.0,
        "reps": 5,
        "rir": 2,
        "timestamp": "2026-04-20T18:00:00",
    },
    {
        "exercise": "incline_bench",
        "weight_kg": 70.0,
        "reps": 6,
        "rir": 2,
        "timestamp": "2026-04-20T18:12:00",
    },
]

mpc = predict_mpc(
    model,
    user_history=history,
    timestamp="2026-04-23T09:00:00",
    strength_anchors=strength_anchors,
)

rir = predict_rir(
    model,
    state=mpc,
    exercise="bench_press",
    weight=82.5,
    reps=5,
    strength_anchors=strength_anchors,
)

session_sets = [
    {
        "exercise": "bench_press",
        "weight_kg": 82.5,
        "reps": 5,
        "rir": 1,
        "timestamp": "2026-04-23T18:05:00",
    },
    {
        "exercise": "incline_bench",
        "weight_kg": 72.5,
        "reps": 6,
        "rir": 2,
        "timestamp": "2026-04-23T18:20:00",
    },
]

strength_anchors = update_strength_anchors(strength_anchors, session_sets)
```

## Co ten README świadomie pomija

- szczegóły architektury sieci
- szczegóły treningu i walidacji
- szczegóły priors `exercise -> anchor + ratio`

To jest opis kontraktu integracyjnego, nie dokument badawczy.
