# TFLite Export — Phase 1 of the Flutter port

Converts `deepgain_model_muscle_ord.pt` (Michał's stable contract) to two
TFLite models plus a set of JSON constants that the Dart port consumes.

## What gets exported

| Output | Purpose |
|---|---|
| `assets/f_net.tflite` | FatigueModule — one set in, drop[15] out (called once per set during MPC replay) |
| `assets/g_net.tflite` | RIRModule — one planned set in, RIR scalar out |
| `assets/involvement_matrix.json` | (NUM_EXERCISES × 15) per-exercise muscle weights |
| `assets/anchor_ratio_matrix.json` | (NUM_EXERCISES × 3) per-exercise 1RM ratios + availability mask |
| `assets/fixed_tau.json` | [15] recovery time constants (hours) per muscle |
| `assets/exercises.json` | Ordered exercise IDs (index = `exercise_idx` used by f_net/g_net) |
| `assets/muscles.json` | Ordered muscle IDs (index = `muscle_idx`) |
| `assets/scales.json` | `WEIGHT_SCALE`, `REPS_SCALE`, `RIR_SCALE`, `DT_SCALE` normalization constants |
| `assets/default_anchors_kg.json` | Fallback bench/squat/deadlift 1RMs |

## Why two models, not one

`predict_mpc` calls `f_net` once per logged set (passing all 15 muscles),
and `predict_rir` calls `g_net` once per planned set. Both are tiny (<300 KB)
and take small fixed-shape inputs, so two specialized TFLite graphs is
cleaner than one monolithic graph with branching.

We deliberately did **not** try to TFLite-export the MPC replay loop itself
(history reconstruction, recovery, anchor history) — that orchestration
lives in Dart, mirroring `inference.py`.

## How to run

```bash
cd tools/tflite_export
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Export
python export.py --checkpoint ../../deepgain_model_muscle_ord.pt

# 2. Verify TFLite outputs match the live PyTorch path
python parity_test.py --checkpoint ../../deepgain_model_muscle_ord.pt
```

`parity_test.py` runs 50 random history scenarios through both pipelines
and asserts per-muscle MPC matches within `1e-4` and RIR within `1e-3`.

The `TFLiteOrchestrator` class in `parity_test.py` is the **Dart spec** —
the pure-NumPy reimplementation of `predict_mpc` / `predict_rir` that the
Dart port must reproduce 1:1.

## When to re-run

Whenever the underlying model changes:

- New `.pt` checkpoint from Michał (e.g. Variant 2 1RM anchors stabilises)
- Changes to `strength_priors.EXERCISE_STRENGTH_PRIORS` (anchor ratios)
- Changes to `inference.ExponentialRecovery.FIXED_TAU` (recovery constants)

Note: editing `exercise_muscle_weights_scaled.csv` does **not** affect the
exported assets. The involvement matrix is read from the model's saved
`involvement` buffer, which was frozen when the checkpoint was trained.
The export script prints the drift between checkpoint and current CSV so
you know when the model is "behind" what the live `inference.py` would
recompute. To pick up CSV changes you have to retrain.

After re-running, copy `assets/*` into `flutter_app/assets/model/` and
re-run the Dart golden tests in Phase 2.

## Diag scripts

Three one-off diagnostics from the parity-debugging session, kept around
because they're useful when parity breaks again after a model bump:

- `diag_torch.py` — original `predict_drop_norm` vs `FatigueModule` wrapper, both in PyTorch (catches wrapper bugs)
- `diag_tflite.py` — TFLite output for a single hand-picked input vs hardcoded baseline (catches conversion bugs)
- `diag_replay.py` — step-by-step replay of a 20-set scenario, prints per-set drop and mpc diffs (catches orchestration / asset-mismatch bugs — this one found the involvement-drift issue)
