# DeepGain — Model (Michal / Osoba B)

Model ML predykujący zmęczenie mięśni (MPC) i trudność planowanego seta (RIR)
z historii treningowej użytkownika.

---

## Architektura

Trzy komponenty — dwa MLP-y + deterministyczna formuła recovery:

```
DeepGainModel
├── f_net  (FatigueNet)          — ile MPC spada per mięsień po secie
├── g_net  (RIRNet)              — ile RIR przy danym stanie mięśni i sile usera
└── r      (ExponentialRecovery) — regeneracja: 1 - (1-MPC)·exp(-dt/τ)
```

**FatigueNet** — input 138D → 512 → 512 → 256 → 1 (Sigmoid). Wywoływana 15× per set.
**RIRNet** — input 87D → 512 → 512 → 256 → 1 (Sigmoid). Wywoływana 1× per set.
**Recovery** — τ fixed z literatury: chest=16h, quads=19h, triceps=9h, ...

Obie sieci dostają `strength_feat` (6D) — anchory 1RM usera (bench/squat/deadlift)
sprojektowane na konkretne ćwiczenie przez tablicę ratio z `strength_priors.py`.
Kluczowy sygnał: `relative_load = weight / projected_1rm`.

Łącznie: **907,857 parametrów**.

---

## Metryki (finalny model, Wariant 2, 100 epok)

| Model | RMSE | MAE | Ordering |
|---|---:|---:|---:|
| M8 baseline (bez strength anchors, 100ep) | 0.848 | 0.656 | 93% |
| **Wariant 2 — finalny (100ep)** | **0.841** | **0.658** | **92%** |

Personalizacja — ten sam set (bench 80kg×5, MPC=1.0):

| User | 1RM bench | Predicted RIR |
|---|---:|---:|
| Silny | 140kg | 4.43 |
| Słaby | 60kg | 0.03 |

---

## Uruchamianie

Wszystkie skrypty uruchamiać z katalogu `models/`:

```bash
cd models/
python train.py                              # trening
python test_inference_personalization.py    # smoke test
python ablations/train_ablation_no_ord.py   # ablacja A1
```

---

## Publiczne API

```python
from inference import load_model, predict_mpc, predict_rir, update_strength_anchors

model   = load_model("deepgain_model_best.pt")
anchors = {"bench_press": 100.0, "squat": 140.0, "deadlift": 180.0}

mpc  = predict_mpc(model, user_history, timestamp, strength_anchors=anchors)
rir  = predict_rir(model, mpc, "bench_press", 80.0, 5, strength_anchors=anchors)
anchors = update_strength_anchors(anchors, completed_sets)
```

Pełna dokumentacja API → [`README_INFERENCE.md`](README_INFERENCE.md)

---

## Pliki

| Plik/Katalog | Opis |
|---|---|
| `train.py` | Główny skrypt treningu |
| `inference.py` | Publiczne API dla Miłosza |
| `strength_priors.py` | Logika anchorów 1RM, tablica ratio, update EMA |
| `deepgain_model_best.pt` | Best val checkpoint (100ep, RMSE 0.841) |
| `deepgain_model_muscle_ord.pt` | Final checkpoint (100ep) |
| `requirements.txt` | Zależności Pythona |
| `test_inference_personalization.py` | Smoke test — 7 scenariuszy |
| `train_report.md` | Historia modeli M1–M8, metryki, decyzje |
| `README_INFERENCE.md` | Kontrakt integracyjny dla Miłosza — flow, API, pitfalls |
| `VARIANT2_SUMMARY.md` | Opis problemu inter-person i implementacji Wariantu 2 |
| `ablations/` | Skrypty A1/A2/A3 + wyniki (pre-Wariant2) |
| `charts/` | Wykresy ze wszystkich runów treningowych |

---

## Znane ograniczenia

- `leg_curl` zakotwiczony do `squat` zamiast `deadlift` — do poprawki
- Pull exercises (row, pull-up, lat_pulldown) zakotwiczone do bench — słaba korelacja
- 9 ćwiczeń `bodyweight` bez personalizacji (plank, farmers_walk itd.) — wymaga masy ciała
- Ablacje A1/A2/A3 robione przed Wariantem 2, nieporównywalne z aktualnym modelem
