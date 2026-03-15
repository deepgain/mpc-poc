# mpc-poc

Proof-of-concept implementation of **Muscle Performance Capacity (MPC)** estimation from minimal training logs (weight, reps, RPE).

MPC is a latent per-muscle fatigue state in [0, 1] — never directly observed, inferred purely from RPE prediction error across overlapping exercises.

## POC Results

Trained on 1.16M synthetic sets (1000 users, 27 exercises, 16 muscle groups).

| Metric | Value |
|--------|-------|
| RMSE | 1.01 RPE |
| MAE | 0.70 RPE |
| Correlation | 0.90 |

70/30 train/test split by user (no data leakage).

## Architecture

Three learned neural network components:

- **f** (fatigue): `MPC_m' = f(w, r, RPE, MPC_m, exercise_embed, muscle_embed)` — how much each muscle's capacity drops after a set
- **g** (RPE predictor): `RPE = g(w, r, exercise_embed, all_MPC)` — predicted RPE from current MPC state of all 16 muscles
- **r** (recovery): `MPC_m' = r(MPC_m, delta_t, muscle_embed)` — how MPC recovers toward 1.0 over time

All three share weights across exercises/muscles via learned embeddings (16-dim). The involvement matrix (which muscles each exercise uses) is fixed from EMG literature.

Training signal: RPE prediction error only. Backprop flows through the full sequential chain: loss -> g -> MPC -> f -> previous MPC -> ...

See [DeepGain_specification.md](DeepGain_specification.md) for the full specification.

## Files

| File | Description |
|------|-------------|
| `generate_training_data.py` | Synthetic data generator (1000+ users, evidence-based, 40 citations) |
| `train.py` | Training script (50 epochs, ~35 min on CPU) |
| `explore_model.ipynb` | Notebook: load pretrained model, run all visualizations |
| `deepgain_model.pt` | Pretrained model weights (88K) |
| `DeepGain_specification.md` | Full MPC specification |

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate training data (1000 users, ~65MB CSV)
python generate_training_data.py

# Train the model (~35 min on CPU)
python train.py

# Or skip training and use the pretrained model in the notebook
jupyter notebook explore_model.ipynb
```

## Key Design Decisions

- **User-level train/test split** — full workout sequences stay intact, no data leakage
- **Teacher forcing** — f receives ground-truth RPE during training to prevent MPC state corruption
- **Fixed involvement matrix** — exercise-muscle coefficients from EMG literature, not learned
- **Multiplicative MPC update** — `new_mpc = mpc * (1 - involvement * drop)`, naturally bounded
- **Sequence chunking** (256 steps) — limits BPTT while carrying MPC state between chunks
