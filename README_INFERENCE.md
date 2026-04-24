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

## Pitfalls — czego NIE robić

### Brak `strength_anchors` = brak personalizacji

Jeśli wywołasz `predict_rir(...)` lub `predict_mpc(...)` bez `strength_anchors`
i bez `config_1rm_*` w historii, model użyje **domyślnych mediańnych wartości
populacyjnych** zapisanych w checkpoincie. To oznacza:

- wszyscy użytkownicy są traktowani jako "przeciętny lifter"
- predykcje `RIR` przestają być personalizowane
- nie dostaniesz błędu ani warninga — model po prostu zachowa się jak stary
  wariant bez wariantu 2

Przykład:

```python
# ŹLE — brak anchorów, model używa mediany populacyjnej
rir = predict_rir(model, mpc, "bench_press", 80.0, 5)

# OK — personalizacja działa
rir = predict_rir(model, mpc, "bench_press", 80.0, 5,
                  strength_anchors={"bench_press": 120.0,
                                    "squat": 180.0,
                                    "deadlift": 210.0})
```

**Zawsze przekazuj `strength_anchors`** jeśli chcesz personalnych predykcji.

### Inne typowe błędy

- Aktualizacja anchorów po każdym secie, nie po sesji — rekomendowany tryb to
  pełna, zakończona sesja (co najmniej kilka dobrych setów).
- Zapisanie anchorów w dataset raw CSV — anchory to stan runtime użytkownika,
  nie część danych treningowych. Trzymaj je w backendzie / DB obok historii.
- Używanie `predict_rir(...)` jako wskaźnika `1RM` — od tego jest
  `project_exercise_1rm(...)`. `predict_rir` daje RIR przy konkretnym
  `(weight, reps)`, nie szacunek maksimum.

## Obsługiwane ćwiczenia

Model obsługuje 34 ćwiczenia zakotwiczone do jednego z 4 anchorów.
Anchor określa, przez który bój osobowy `1RM` danego ćwiczenia jest rzutowany
(`exercise_1rm = anchor_1rm * ratio`).

### Anchor: `bench_press`

| Exercise | Ratio | Family |
|---|---:|---|
| `bench_press` | 1.000 | bench_primary |
| `spoto_press` | 0.900 | bench_variant |
| `decline_bench` | 0.900 | bench_variant |
| `pendlay_row` | 0.890 | upper_pull |
| `close_grip_bench` | 0.850 | bench_variant |
| `chest_press_machine` | 0.840 | machine_press |
| `incline_bench` | 0.815 | bench_variant |
| `incline_bench_45` | 0.780 | bench_variant |
| `seal_row` | 0.765 | upper_pull |
| `lat_pulldown` | 0.680 | vertical_pull |
| `dips` | 0.635 | press_assistance |
| `ohp` | 0.620 | vertical_press |
| `pull_up` | 0.565 | vertical_pull |
| `skull_crusher` | 0.295 | triceps_isolation |
| `dumbbell_flyes` | 0.290 | chest_isolation |

### Anchor: `squat`

| Exercise | Ratio | Family |
|---|---:|---|
| `leg_press` | 1.450 | machine_lower_compound |
| `squat` | 1.000 | squat_primary |
| `low_bar_squat` | 0.990 | squat_variant |
| `high_bar_squat` | 0.960 | squat_variant |
| `leg_extension` | 0.420 | quad_isolation |
| `bulgarian_split_squat` | 0.400 | unilateral_lower |
| `leg_curl` | 0.340 | hamstring_isolation |

### Anchor: `deadlift`

| Exercise | Ratio | Family |
|---|---:|---|
| `deadlift` | 1.000 | hinge_primary |
| `sumo_deadlift` | 0.975 | hinge_variant |
| `rdl` | 0.700 | hinge_variant |

### Anchor: `bodyweight` (uwaga!)

Te ćwiczenia są zakotwiczone do `bodyweight`, ale **obecny model nie przyjmuje
masy ciała jako anchor** — onboarding zbiera tylko 3 boje siłowe. Dla tych
ruchów `project_exercise_1rm(...)` zwraca `None`, a model używa fallbacka
(populacyjne defaults z checkpointu) zamiast personalizowanej projekcji.

| Exercise | Ratio | Family |
|---|---:|---|
| `farmers_walk` | 0.750 | carry |
| `suitcase_carry` | 0.440 | unilateral_carry |
| `ab_wheel` | 0.375 | core_anti_extension |
| `plank` | 0.340 | core_bracing |
| `leg_raises` | 0.290 | core_flexion |
| `trx_bodysaw` | 0.290 | core_anti_extension |
| `dead_bug` | 0.250 | core_stability |
| `bird_dog` | 0.250 | core_stability |
| `reverse_fly` | 0.120 | rear_delt_isolation |

Predykcje `RIR`/`MPC` dla tych ćwiczeń dalej działają, tylko bez pełnej
personalizacji siły.

### Pełna lista źródłowa

Kanoniczne źródło ratio/anchor to `strength_priors.EXERCISE_STRENGTH_PRIORS`.
W razie wątpliwości:

```python
from strength_priors import EXERCISE_STRENGTH_PRIORS, get_exercise_anchor_name
get_exercise_anchor_name("incline_bench")       # -> "bench_press"
EXERCISE_STRENGTH_PRIORS["incline_bench"]
# {'anchor_lift': 'bench_press', 'ratio_mean': 0.815, ...}
```

Alternatywnie `get_exercises()` z `inference.py` zwraca wszystkie ćwiczenia
rozpoznawane przez załadowany model.

## Smoke test

W repo jest `test_inference_personalization.py` — skrypt end-to-end
sprawdzający kontrakt:

```bash
python test_inference_personalization.py
```

Testuje m.in.:

- personalizację (silny vs słaby user dostają różne RIR przy tym samym secie)
- monotoniczność (rosnący ciężar → spadający RIR)
- efekt zmęczenia MPC
- projekcje `1RM` przez anchor + ratio
- replay historii w `predict_mpc(...)` wraz z recovery
- dynamiczny update anchorów po sesji
- cold start (brak anchorów → domyślne populacyjne)

Dobry punkt startowy jak chcesz się upewnić, że integracja po Twojej stronie
dobrze wpina się w `inference.py`.

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
