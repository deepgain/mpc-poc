# mpc-poc

Proof-of-concept implementation of **Muscle Performance Capacity (MPC)** estimation from minimal training logs (weight, reps, RIR).

MPC is a latent per-muscle fatigue state in [0, 1] — never directly observed, inferred purely from RIR prediction error across overlapping exercises.

## Why RIR over RPE

RIR (Repetitions in Reserve, 0–5 integer) replaces RPE (6–10) because:
- Maps directly to proximity-to-failure — the mechanistically relevant variable (Robinson 2024)
- Higher accuracy near failure: ±0.65 reps at RIR 1 (Refalo 2024) vs RPE's ordinal ambiguity
- No dead zone — every value 0–5 is used, unlike RPE where 6.0 is rarely reported
- Exercise-agnostic — RIR 2 means the same thing on squats and lateral raises

## POC Results

Trained on ~550K synthetic sets (200 users × 52 weeks, 27 exercises, 16 muscle groups).
Data generated with **no hidden fatigue state** — pure empirical lookup tables from 57 papers.

| Metric | Value |
|--------|-------|
| RMSE | 0.98 RIR |
| MAE | 0.76 RIR |
| Correlation | 0.83 |

70/30 train/test split by user (no data leakage).

## Architecture

Three learned neural network components:

- **f** (fatigue): `MPC_m' = f(w, r, RIR, MPC_m, exercise_embed, muscle_embed)` — how much each muscle's capacity drops after a set
- **g** (RIR predictor): `RIR = g(w, r, exercise_embed, all_MPC)` — predicted RIR from current MPC state of all 16 muscles
- **r** (recovery): `MPC_m' = r(MPC_m, delta_t, muscle_embed)` — how MPC recovers toward 1.0 over time

All three share weights across exercises/muscles via learned embeddings (16-dim). The involvement matrix (which muscles each exercise uses) is fixed from EMG literature.

Training signal: RIR prediction error only. Backprop flows through the full sequential chain: loss -> g -> MPC -> f -> previous MPC -> ...

See [DeepGain_specification.md](DeepGain_specification.md) for the full specification.

## Files

| File | Description |
|------|-------------|
| `generate_training_data.py` | MPC-free synthetic data generator (57 citations, 5 empirical lookup tables, NO hidden state) |
| `train.py` | Training script (50 epochs, ~25 min on CPU) |
| `explore_model.ipynb` | Notebook: load pretrained model, run all visualizations |
| `deepgain_model.pt` | Pretrained model weights (88K) |
| `DeepGain_specification.md` | Full MPC specification |

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate training data (200 users × 52 weeks, ~30MB CSV)
python generate_training_data.py

# Train the model (~25 min on CPU)
python train.py

# Or skip training and use the pretrained model in the notebook
jupyter notebook explore_model.ipynb
```

## Key Design Decisions

- **MPC-free data generator** — simulator uses only empirical lookup tables (retention ratios, cross-exercise transfer, recovery curves), NO hidden fatigue state. MPC must be *discovered* by the model, not reverse-engineered from the simulator.
- **RIR scale (0–5 integer)** — directly measures proximity to failure, supported by Robinson 2024, Halperin 2022, Refalo 2024
- **User-level train/test split** — full workout sequences stay intact, no data leakage
- **Teacher forcing** — f receives ground-truth RIR during training to prevent MPC state corruption
- **Fixed involvement matrix** — exercise-muscle coefficients from EMG literature, not learned
- **Multiplicative MPC update** — `new_mpc = mpc * (1 - involvement * drop)`, naturally bounded
- **Sequence chunking** (256 steps) — limits BPTT while carrying MPC state between chunks
