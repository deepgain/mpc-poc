# Train Ablations

Wszystkie ablacje: HIDDEN_DIM=512, EMBED_DIM=64, 30 epok, ten sam dataset (`training_data_full_generated.csv`, 212/91 userów).

## Porównanie

| | M8 (baseline) | A1: bez ord_pen |
|---|---|---|
| Val RMSE | **0.904** | 0.910 |
| MAE | 0.710 | **0.701** |
| R | 0.863 | **0.864** |
| Ordering MEAN | **99%** | 59% |
| Charts | `charts/20260418_1729/` | `charts/20260418_2116_no_ord/` |
| Checkpoint | `deepgain_model_best.pt` | `deepgain_ablation_no_ord_best.pt` |

## A1 — bez fatigue_ordering_penalty

**Wyłączone:** `loss = loss + 0.05 * model.fatigue_ordering_penalty()`

**Wynik:** Ordering spada z 99% do 59% — kara jest głównym czynnikiem odpowiadającym za poprawną hierarchię mięśni. RMSE prawie identyczny (0.910 vs 0.904), MAE minimalnie lepszy bez kary (0.701 vs 0.710).

**Wniosek:** `fatigue_ordering_penalty` daje +40pp ordering przy koszcie ~0.006 RMSE. Opłacalne — kara zostaje.

**Obserwacje z wykresów:**
- `chart_muscle_breakdown.png`: bench press collapse — triceps i anterior_delts bliskie zeru, chest wszystko przejmuje. Bez kary model nie ma powodu dystrybuować fatigue na mniejsze mięśnie.
- `chart_transfer_matrix.png`: niespójne transfery — bench_press→squat=-0.0 (brak sensownego transferu cross-muscle).
- `ord_pen` w logach: **nigdy nie schodzi do 0** (utrzymuje się ~0.15-0.18 przez wszystkie 30 epok) — model naturalnie narusza ordering bez kary.

### Ordering per ćwiczenie — A1 vs M8

| Ćwiczenie | M8 | A1 (no_ord) | Δ |
|---|---:|---:|---|
| high_bar_squat | 100% | **0%** | -100pp |
| trx_bodysaw | 100% | **0%** | -100pp |
| suitcase_carry | 100% | **0%** | -100pp |
| bird_dog | 100% | **0%** | -100pp |
| seal_row | 100% | **17%** | -83pp |
| leg_raises | 100% | **17%** | -83pp |
| chest_press_machine | 100% | **33%** | -67pp |
| leg_press | 100% | **33%** | -67pp |
| pendlay_row | 100% | **33%** | -67pp |
| ohp | 80% | **40%** | -40pp |
| deadlift | 100% | **67%** | -33pp |
| bench_press | 100% | **83%** | -17pp |
| spoto_press | 100% | **100%** | — |
| skull_crusher | 100% | **100%** | — |
| **MEAN** | **99%** | **59%** | **-40pp** |
