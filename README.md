# DeepGain — mpc-poc

Proof-of-concept implementacja estymacji **Muscle Performance Capacity (MPC)** z logów treningowych (weight, reps, RIR).

MPC to ukryty stan zmęczenia per mięsień w [0, 1] — nigdy bezpośrednio nieobserwowany, wnioskowany wyłącznie z błędu predykcji RIR.

---

## Struktura repo

```
mpc-poc/
├── models/          # Model ML — trening, inference, checkpointy (Michal)
├── dataset/         # Generator danych, YAML ćwiczeń, wagi EMG (Aleksander)
├── app/             # Algorytm planowania treningu (Miłosz)
└── exercise_selection_algorithm/
```

---

## Wyniki (aktualny model — Wariant 2, 100 epok)

Trenowany na 1.54M syntetycznych setów (320 userów, 34 ćwiczenia, 15 grup mięśniowych).

| Metryka | Wartość |
|---------|---------|
| Val RMSE | **0.841 RIR** |
| Test MAE | **0.658 RIR** |
| Pearson R | **0.882** |
| Ordering accuracy | **92%** |

Model rozróżnia siłę użytkowników przez 3 anchory 1RM (bench/squat/deadlift).

---

## Architektura

```
DeepGainModel
├── f_net  — FatigueNet (MLP): ile MPC spada per mięsień po secie
├── g_net  — RIRNet (MLP):    ile RIR przy danym stanie mięśni i sile usera
└── r      — ExponentialRecovery: MPC_new = 1 - (1-MPC)·exp(-dt/τ)
```

Szczegóły architektury, metryki, historia modeli → [`models/README.md`](models/README.md)

---

## Inference API

```python
from models.inference import load_model, predict_mpc, predict_rir, update_strength_anchors

model   = load_model("models/deepgain_model_best.pt")
anchors = {"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0}

mpc = predict_mpc(model, user_history, timestamp, strength_anchors=anchors)
rir = predict_rir(model, mpc, "bench_press", 80.0, 5, strength_anchors=anchors)
anchors = update_strength_anchors(anchors, completed_sets)
```

Pełna dokumentacja API → [`models/README_INFERENCE.md`](models/README_INFERENCE.md)

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r models/requirements.txt

# Generowanie danych treningowych
cd dataset && python generate_training_data.py

# Trening modelu
cd models && python train.py

# Smoke test inference
cd models && python test_inference_personalization.py
```

---

## Specyfikacja

Pełna specyfikacja MPC → [`DeepGain_specification.md`](DeepGain_specification.md)
