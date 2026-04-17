# DeepGain Inference API

API for the workout planner (Milosz). Provides MPC estimation and RIR prediction.

## Quick Start

```python
from inference import load_model, predict_mpc, predict_rir

model = load_model("deepgain_model_muscle_ord.pt")
```

## Functions

### `predict_mpc(model, user_history, timestamp) -> dict[str, float]`

Estimates current Muscle Performance Capacity (MPC) for all 15 muscles.

**Input:**
- `user_history` — list of logged sets, each a dict:
  ```python
  {
      "exercise":   "bench_press",          # exercise_id (see get_exercises())
      "weight_kg":  80.0,                   # float
      "reps":       5,                      # int
      "rir":        2,                      # int, 0–5
      "timestamp":  "2024-01-01T10:00:00",  # ISO8601 string or datetime
  }
  ```
- `timestamp` — moment for which to estimate MPC (ISO8601 string or datetime)

**Output:** `dict[muscle_id, MPC]` — 15 muscles, values in [0.1, 1.0]

```python
history = [
    {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5, "rir": 2,
     "timestamp": "2024-01-01T10:00:00"},
    {"exercise": "ohp", "weight_kg": 50.0, "reps": 6, "rir": 3,
     "timestamp": "2024-01-01T10:20:00"},
]
mpc = predict_mpc(model, history, timestamp="2024-01-02T09:00:00")
# {"chest": 0.86, "triceps": 0.98, "quads": 1.0, "anterior_delts": 0.97, ...}
```

**Edge cases (handled silently):**
- Empty history → MPC = 1.0 for all muscles (fresh user)
- Sets after `timestamp` → automatically excluded
- Unknown exercise in history → skipped

---

### `predict_rir(model, state, exercise, weight, reps) -> float`

Predicts RIR for a planned set given current muscle state.

**Input:**
- `state` — dict from `predict_mpc()`. Missing muscles default to 1.0.
- `exercise` — exercise_id (see `get_exercises()`)
- `weight` — weight in kg
- `reps` — planned rep count

**Output:** predicted RIR in [0.0, 5.0]

**Raises:** `ValueError` if exercise is not in the model's exercise list.

```python
mpc = predict_mpc(model, history, timestamp)
rir = predict_rir(model, mpc, exercise="bench_press", weight=100.0, reps=5)
# 1.8
```

---

### `get_exercises() -> list[str]`

Returns all 34 exercise IDs recognized by the model.

```python
get_exercises()
# ["bench_press", "incline_bench", "ohp", "squat", "deadlift", ...]
```

### `get_muscles() -> list[str]`

Returns all 15 muscle IDs.

```python
get_muscles()
# ["chest", "anterior_delts", "lateral_delts", "rear_delts", "rhomboids",
#  "triceps", "biceps", "lats", "quads", "hamstrings", "glutes",
#  "adductors", "erectors", "calves", "abs"]
```

## Model

`deepgain_model_muscle_ord.pt` — M4 (Milestone 4)
- Val RMSE: **0.869 RIR**, MAE: 0.673, R: 0.878
- 34 exercises, 15 muscles
- Trained on 588k sets from 155 users

## Dependencies

```
torch
numpy
pandas
pyyaml
```

## Stable Contract

The function signatures above are frozen. Do not change them — both the planner
and (future) app depend on them. Report any issues to Michal.
