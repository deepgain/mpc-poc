# Model parameters

| Parameter | Value |
|---|---|
| HIDDEN_DIM | 512 |
| EMBED_DIM | 64 |
| EPOCHS | 25 |
| LR | 0.001 |
| BATCH_SIZE | 16 |
| CHUNK_LEN | 512 |
| Dataset | training_data.csv |
| Parameters | 907,857 |
| Val RMSE (best) | 0.8894 |
| Test RMSE | 0.8894 |
| Test MAE | 0.6892 |
| Ordering MEAN | 95% |
| Penalties | ord=0.05 min_drop=0.10 mono=0.05 |

## Dynamic 1RM Validation

Holdout validation for the same checkpoint:

| Variant | RMSE | MAE | Corr |
|---|---:|---:|---:|
| dynamic_correct | 0.8832 | 0.6852 | 0.8707 |
| static_onboarding | 0.9079 | 0.7053 | 0.8636 |
| shuffled_dynamic | 1.5126 | 1.0552 | 0.6288 |

Dynamic-vs-static gain:

- `rmse_gain = 0.0247`
- `mae_gain = 0.0201`

Dynamic-vs-shuffled gain:

- `rmse_gain = 0.6294`
- `mae_gain = 0.3700`

## Anchor Dynamics Stats

- `changed_user_fraction = 1.000`
- `changed_set_fraction = 0.995`
- `bench_press_median_final_rel_change = -0.0152`
- `squat_median_final_rel_change = 0.0280`
- `deadlift_median_final_rel_change = -0.0433`

## Key Charts

- `chart_strength_anchor_trajectories.png`
  Shows how estimated `bench/squat/deadlift` anchors change over time for sample holdout users.

- `chart_strength_sweeps.png`
  Shows how predicted `RIR` changes when only the relevant anchor is varied at fresh `MPC`.

- `chart_dynamic_vs_static_rir.png`
  Compares per-set predictions using dynamic anchors vs static onboarding anchors on real holdout sequences.
