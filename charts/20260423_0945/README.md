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
| Val RMSE (best) | 0.9185 |
| Test RMSE | 0.9274 |
| Test MAE | 0.7234 |
| Ordering MEAN | 95% |
| Penalties | ord=0.05 min_drop=0.10 mono=0.05 |

## Dynamic 1RM Validation

Holdout validation for the same checkpoint:

| Variant | RMSE | MAE | Corr |
|---|---:|---:|---:|
| dynamic_correct | 0.9129 | 0.7083 | 0.8620 |
| static_onboarding | 1.0231 | 0.7972 | 0.8296 |
| shuffled_dynamic | 1.2920 | 0.9317 | 0.7202 |

Dynamic-vs-static gain:

- `rmse_gain = 0.1102`
- `mae_gain = 0.0889`

Dynamic-vs-shuffled gain:

- `rmse_gain = 0.3790`
- `mae_gain = 0.2234`

## Anchor Dynamics Stats

- `changed_user_fraction = 1.000`
- `changed_set_fraction = 0.999`
- `bench_press_median_final_rel_change = -0.2036`
- `squat_median_final_rel_change = -0.3174`
- `deadlift_median_final_rel_change = -0.1222`

## Key Charts

- `chart_strength_anchor_trajectories.png`
  Shows how estimated `bench/squat/deadlift` anchors change over time for sample holdout users.

- `chart_strength_sweeps.png`
  Shows how predicted `RIR` changes when only the relevant anchor is varied at fresh `MPC`.

- `chart_dynamic_vs_static_rir.png`
  Compares per-set predictions using dynamic anchors vs static onboarding anchors on real holdout sequences.
