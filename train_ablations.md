# Train Ablations

30 epok, ten sam dataset (`training_data_full_generated.csv`, 212/91 userów). M8 baseline: EMBED_DIM=64, HIDDEN_DIM=512.

## Porównanie

| | M8 (baseline) | A1: bez ord_pen | A2: small | A3: learned τ |
|---|---|---|---|---|
| Val RMSE | **0.904** | 0.910 | 0.922 | 0.915 |
| Test RMSE | **0.904** | — | 0.923 | 0.930 |
| MAE | **0.710** | **0.701** | 0.711 | 0.724 |
| R | **0.863** | **0.864** | 0.859 | 0.856 |
| Ordering MEAN | **99%** | 59% | 97% | 94% |
| Params | 901k | 901k | **230k** | 901k |
| Charts | `charts/20260418_1729/` | `charts/20260418_2116_no_ord/` | `charts/20260418_2336_small/` | `charts/20260419_0114_learned_tau/` |
| Checkpoint | `deepgain_model_best.pt` | `deepgain_ablation_no_ord_best.pt` | `deepgain_ablation_small_best.pt` | `deepgain_ablation_learned_tau_best.pt` |

## A1 — bez fatigue_ordering_penalty

**Wyłączone:** `loss = loss + 0.05 * model.fatigue_ordering_penalty()`

**Wynik:** Ordering spada z 99% do 59% — kara jest głównym czynnikiem odpowiadającym za poprawną hierarchię mięśni. RMSE prawie identyczny (0.910 vs 0.904), MAE minimalnie lepszy bez kary (0.701 vs 0.710).

**Wniosek:** `fatigue_ordering_penalty` daje +40pp ordering przy koszcie ~0.006 RMSE. Opłacalne — kara zostaje.

**Obserwacje z wykresów:**
- `chart_muscle_breakdown.png`: bench press collapse — triceps i anterior_delts bliskie zeru, chest wszystko przejmuje. Bez kary model nie ma powodu dystrybuować fatigue na mniejsze mięśnie.
- `chart_transfer_matrix.png`: niespójne transfery — bench_press→squat=-0.0 (brak sensownego transferu cross-muscle).
- `ord_pen` w logach: **nigdy nie schodzi do 0** (utrzymuje się ~0.15-0.18 przez wszystkie 30 epok) — model naturalnie narusza ordering bez kary.

## A2 — mniejszy model (EMBED_DIM=32, HIDDEN_DIM=256)

**Zmiana:** 4× mniej parametrów (230k vs 901k).

**Wynik:** RMSE 0.922 vs 0.904 (+0.018), ordering 97% vs 99% (-2pp). MAE i R praktycznie identyczne.

**Wniosek:** Mały model zaskakująco bliski M8. Większa pojemność M8 opłaca się (+2pp ordering, -0.018 RMSE), ale różnica jest marginalna. Na urządzeniach mobilnych A2 byłby lepszym wyborem.

**Ordering slabe punkty A2:** chest_press_machine 67%, pendlay_row 83%, seal_row 83%, dips 90% — ćwiczenia compound z wieloma mięśniami drugorzędnymi trudniejsze dla mniejszego modelu.

## A3 — learned τ (clamped [8, 72h])

**Zmiana:** `requires_grad=True` na `log_tau`, forward: `tau = torch.exp(log_tau).clamp(8, 72)`.

**Wynik:** Gorszy od M8 na wszystkich metrykach. RMSE 0.930 vs 0.904, ordering 94% vs 99%.

**Co się stało z τ:**

| Mięsień | Literatura (M8) | Learned (ep.30) | Δ |
|---|---|---|---|
| chest | 16h | 12h | -4h |
| anterior_delts | 13h | 9h | -4h |
| **lateral_delts** | **9h** | **1h** | **-8h** ⚠️ |
| **rear_delts** | **8h** | **1h** | **-7h** ⚠️ |
| **rhomboids** | **10h** | **3h** | **-7h** ⚠️ |
| **triceps** | **9h** | **2h** | **-7h** ⚠️ |
| biceps | 13h | 16h | +3h |
| lats | 13h | 5h | -8h |
| quads | 19h | 13h | -6h |
| hamstrings | 18h | 7h | -11h |
| glutes | 15h | 12h | -3h |
| **adductors** | **12h** | **1h** | **-11h** ⚠️ |
| erectors | 12h | 7h | -5h |
| **calves** | **8h** | **1h** | **-7h** ⚠️ |
| **abs** | **10h** | **2h** | **-8h** ⚠️ |

**Dlaczego τ się posypało:** Mniejsze/drugorzędne mięśnie (lateral_delts, triceps, calves, abs) uczestniczą w wielu ćwiczeniach jako secondary. Model uczy się, że jeśli te mięśnie mają bardzo krótkie τ (szybko się regenerują), błąd predykcji spada — bo MPC tych mięśni zawsze blisko 1.0, więc model nie musi ich modelować. To "cheating" kosztem fizjologicznej sensowności.

**Uwaga:** Clamp [8h, 72h] jest na `torch.exp(log_tau)` w forward passie, ale `log_tau` parametr może zejść poniżej log(8). Wyświetlane wartości to `exp(log_tau)` bez clampa — stąd wartości 1-2h w logach. W obliczeniach faktycznie używane jest 8h, ale gradienty trafiają w "martfą strefę" za clampem.

**Wniosek:** Fixed τ z literatury jest lepszy przy 30 epokach na tym datasecie. Dane nie mają wystarczającego sygnału temporalnego żeby nauczyć τ — zbyt mało obserwacji per mięsień z różnymi przerwami między sesjami. τ fixed zostaje.

---

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
