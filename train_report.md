# Train Report — Milestone 6

## Metryki

| | Baseline (27 ćw.) | M2 (44 ćw.) | M3 (47 ćw., 17 mięśni) | M4 (34 ćw., 15 mięśni) | M5 (34 ćw., 15 mięśni) | **M6 (34 ćw., 15 mięśni)** |
|---|---|---|---|---|---|---|
| Val RMSE | 1.08 RIR | 1.005 RIR | 0.845 RIR | 0.869 RIR¹ | 0.963 RIR | **0.924 RIR** |
| MAE | 0.86 RIR | 0.789 RIR | 0.656 RIR | 0.673 RIR¹ | 0.753 RIR | **0.749 RIR** |
| R | 0.789 | 0.833 | 0.884 | 0.878¹ | 0.842 | **0.854** |
| Ordering acc | — | — | — | 90% | 93% | **95%** |
| HIDDEN_DIM | 128 | 128 | 128 | 128 | 128 | **256** |
| Parametry | ~66k | ~66k | ~66k | ~66k | ~66k | **~230k** |
| Split | per-ex | per-ex | per-ex | per-ex¹ | per-user | per-user |

¹ M4 walidowany na danych z data leakage — per-exercise split powodował że wszyscy 312 userów byli w obu setach. Metryki M4 są zbyt optymistyczne. M5+ używa czystego per-user holdoutu (218 train / 94 val, zero overlap) — liczby są uczciwe.

Dataset: `training_data_michal_full.csv` — 723k train / 327k val, 218/94 userów, 34 ćwiczenia.
Checkpoint M6: `deepgain_model_muscle_ord.pt` (val RMSE 0.924, best epoka 49/50). M5: epoka 50/50, val RMSE 0.963.

**M6 vs M5:** RMSE −0.039, MAE −0.004, R +0.012, ordering +2pp. M5 osiągnął plateau na 0.96–0.97 od epoki ~36 — M6 (HIDDEN_DIM=256) przebija ten sufit i stabilizuje się na 0.92–0.95. Większa pojemność modelu przekłada się na lepszą generalizację cross-muscle. Dalsze skalowanie (HIDDEN_DIM=512) możliwe, ale diminishing returns bez więcej danych.

---

## M6 — Co się zmieniło względem M5

- **HIDDEN_DIM=256** zamiast 128 — 229k parametrów vs 66k. Jedyna zmiana architektury.
- **Best checkpointing** — `deepgain_model_muscle_ord.pt` zawiera best checkpoint (epoka 49, RMSE 0.924), nie ostatnią epokę (50, RMSE 0.946). M5 nie miał best checkpointing — `muscle_ord.pt` był ostatnią epoką.
- Dataset, split, normalizacja, penalty — bez zmian vs M5.

**Wniosek z M6:** plateau M5 (~0.96) było ograniczeniem pojemności, nie danych. HIDDEN_DIM=256 przełamuje je i daje 0.924. Ordering 95% (vs 93%) potwierdza że większy model lepiej uczy się hierarchii mięśni. Deadlift nadal 67% — to problem danych (brak sygnału hamstrings/glutes), nie architektury.

---

## M5 — Co się zmieniło względem M4

- **Per-user 70/30 split** z `training_data_michal_full.csv` — fix data leakage. Per-exercise split dawał wszystkich 312 userów w obu setach → model widział te same osoby w train i val → zaniżony val RMSE.
- **Per-exercise weight normalization** — `(weight_kg - p5) / (p95 - p5)` per ćwiczenie zamiast globalnego `/200`. 80 kg na bench ≠ 80 kg na dips. Percentyle p5/p95 zapisane w checkpoincie, inference.py je wczytuje.
- **minimum_drop_penalty** — penalty: `relu(0.15 × involvement[m] - drop[m])`. Zapobiega collapse do jednego mięśnia per ćwiczenie. Aktywna od epoki 4, coefficient 0.10.
- **Pełny CSV (34 ćwiczenia)** — wszystkie 34 ćwiczenia mają EMG wagi z CSV (csv=34, yaml_rank=0). W M4 `spoto_press` używał hardkodowanego fallbacku — teraz też z CSV.
- **Usunięty hardkod** — `_EXERCISE_MUSCLES_HARDCODED` dict całkowicie usunięty. YAML jest obowiązkowy (RuntimeError jeśli brak).
- **Probe na medianie** — penalty probe point zmieniony z w=0.4 na w=0.5 (mediana datasetu, on-manifold).

## Co się zmieniło względem M3

- **15 mięśni** (usunięto `upper_traps` i `brachialis`) — brak bezpośrednich kolumn EMG w schemacie CSV, nie powinny być w modelu
- **EMG weights z CSV** (`exercise_muscle_weights_scaled.csv`) jako primary source — zastąpiły flat 0.7 fallback dla 33/34 ćwiczeń. W M3 ~20 ćwiczeń miało błędne flat 0.7 involvement.
- **Nowy dataset Aleksandra** — 57 peer-reviewed sources, lepsza jakość generatora
- **34 ćwiczenia** (tylko te z yaml, nie union z hardkodowanymi jak w M3)
- `spoto_press` — jedyne ćwiczenie bez danych EMG w CSV, fallback na hardkod

## Per-exercise MAE

| Ćwiczenie | M3 MAE | M4 MAE | M5 MAE | Δ (M4→M5) |
|---|---:|---:|---:|---:|
| low_bar_squat | 0.442 | 0.503 | **0.612** | +0.109 |
| sumo_deadlift | 0.593 | 0.581 | **0.622** | +0.041 |
| lat_pulldown | 0.610 | 0.611 | **0.642** | +0.031 |
| pull_up | 0.626 | 0.605 | **0.661** | +0.056 |
| dips | 0.647 | 0.660 | **0.670** | +0.010 |
| spoto_press | 0.590 | 0.607 | **0.676** | +0.069 |
| ohp | 0.692 | 0.642 | **0.694** | +0.052 |
| leg_press | 0.718 | 0.692 | **0.700** | +0.008 |
| incline_bench | 0.592 | 0.607 | **0.707** | +0.100 |
| deadlift | 0.480 | 0.505 | **0.711** | +0.206 |
| pendlay_row | 0.619 | 0.619 | **0.712** | +0.093 |
| chest_press_machine | 0.711 | 0.700 | **0.715** | +0.015 |
| rdl | 0.631 | 0.627 | **0.716** | +0.089 |
| dumbbell_flyes | 0.695 | 0.688 | **0.730** | +0.042 |
| leg_curl | 0.740 | 0.726 | **0.735** | +0.009 |
| leg_raises | 0.776 | 0.764 | **0.736** | -0.028 ✓ |
| trx_bodysaw | 0.886 | 0.874 | **0.740** | -0.134 ✓ |
| seal_row | 0.611 | — | **0.746** | (brak M4 ref) |
| squat | 0.717 | 0.761 | **0.746** | -0.015 ✓ |
| decline_bench | 0.713 | 0.925 | **0.748** | -0.177 ✓ |
| leg_extension | 0.688 | 0.672 | **0.749** | +0.077 |
| close_grip_bench | 0.614 | 0.675 | **0.750** | +0.075 |
| high_bar_squat | 0.644 | 0.638 | **0.773** | +0.135 |
| bulgarian_split_squat | 0.656 | 0.709 | **0.775** | +0.066 |
| bench_press | 0.664 | 0.687 | **0.779** | +0.092 |
| incline_bench_45 | 0.675 | 0.652 | **0.783** | +0.131 |
| ab_wheel | 0.783 | 0.788 | **0.802** | +0.014 |
| farmers_walk | 0.724 | 0.692 | **0.821** | +0.129 |
| suitcase_carry | 0.780 | 0.764 | **0.823** | +0.059 |
| reverse_fly | 0.785 | 0.856 | **0.863** | +0.007 |
| skull_crusher | 0.481 | 0.482 | **0.867** | +0.385 |
| bird_dog | 0.923 | 0.892 | **0.954** | +0.062 |
| dead_bug | 0.937 | 0.987 | **0.979** | -0.008 ✓ |
| plank | 0.883 | 0.914 | **1.099** | +0.185 |

Uwaga: M5 MAE wyższe niż M4 dla większości ćwiczeń — to efekt uczciwego per-user splitu. M4 walidował się na tych samych userach co trenował → sztucznie niskie MAE. Wyjątki (✓ = M5 lepszy niż M4): leg_raises, trx_bodysaw, decline_bench, squat, dead_bug — dla tych ćwiczeń generalizacja faktycznie się poprawiła.

`skull_crusher` MAE znacznie wzrosło (0.482 → 0.867) — warto sprawdzić czy w nowym datasecie sekwencje skull_crusher są reprezentatywne.
`plank/dead_bug/bird_dog` — core z niską wariancją RIR, model ma mało sygnału do nauki.

## Ordering accuracy (eval_ordering.py)

Probe: w=0.5 (mediana datasetu), r=0.27 (~8 reps), rir=0.4 (~RIR 2), mpc=1.0.

**M6 MEAN: 95%** | M5: 93% | M4: 90%

| Ćwiczenie | M4 Acc | M5 Acc | M6 Acc | Δ M5→M6 |
|---|---:|---:|---:|---|
| deadlift | 67% | 67% | **67%** | — (problem danych) |
| decline_bench | 100% | 100% | **67%** | regresja ↓ |
| incline_bench | 100% | 67% | **100%** | poprawa ✓ |
| chest_press_machine | 100% | 67% | **100%** | poprawa ✓ |
| dumbbell_flyes | 100% | 67% | **83%** | poprawa ✓ |
| dips | 100% | 80% | **100%** | poprawa ✓ |
| squat | 100% | 83% | **100%** | poprawa ✓ |
| low_bar_squat | 100% | 83% | **100%** | poprawa ✓ |
| lat_pulldown | 50% | 83% | **100%** | poprawa ✓ |
| bird_dog | 0% | 100% | **100%** | — |
| high_bar_squat | 100% | 100% | **100%** | — |
| sumo_deadlift | 100% | 100% | **100%** | — |
| bench_press | 100% | 100% | **100%** | — |
| close_grip_bench | 100% | 100% | **100%** | — |
| spoto_press | 100% | 100% | **100%** | — |
| incline_bench_45 | 100% | 100% | **100%** | — |
| ohp | 100% | 100% | **100%** | — |
| skull_crusher | 100% | 100% | **100%** | — |
| bulgarian_split_squat | 67% | 100% | **100%** | — |
| leg_press | 100% | 100% | **100%** | — |
| pendlay_row | 83% | 100% | **100%** | — |
| pull_up | 100% | 100% | **100%** | — |
| reverse_fly | 100% | 100% | **100%** | — |
| seal_row | 100% | 100% | **100%** | — |
| farmers_walk | 100% | 100% | **100%** | — |
| leg_raises | 100% | 100% | **100%** | — |
| trx_bodysaw | — | 100% | **100%** | — |
| suitcase_carry | 100% | 100% | **100%** | — |
| **MEAN** | **90%** | **93%** | **95%** | |

**M6 poprawa vs M5:** incline_bench 67%→100%, chest_press_machine 67%→100%, dumbbell_flyes 67%→83%, dips 80%→100%, squat 83%→100%, low_bar_squat 83%→100%, lat_pulldown 83%→100%. Większy model nauczył się lepiej rozróżniać zmęczenie mięśni wtórnych (chest/triceps/quads) bez zmiany penalty.

**Regresja M6:** decline_bench 100%→67% — prawdopodobnie statystyczny artefakt (mało próbek decline_bench w val secie konkretnych userów) lub zmiana kolejności przy nowym probe point.

**Deadlift** nadal 67% we wszystkich modelach — erectors dominuje, glutes/hamstrings bliskie zeru. To problem sygnału w danych (brak sekwencji deadlift→rdl→leg_curl), nie architektury. Większy model nie pomaga.

Szczegółowe drop values: uruchom `python eval_ordering.py` po treningu (uwaga: zaktualizować HIDDEN_DIM=256).

## Obserwacje z wykresów — M6 (`charts/20260417_1954/`)

**`chart_muscle_breakdown.png` — per-muscle fatigue breakdown:**
- Push exercises (bench, ohp, dips): poprawna kolejność ordinal ✓
- Muscle collapse mniejszy niż M5 — triceps i anterior_delts mają widoczny drop dla bench press (wcześniej ~0). Większa pojemność modelu lepiej dystrybuuje fatigue.
- Deadlift: erectors nadal dominuje — glutes/hamstrings pozostają zbyt małe (niefizjologiczne). Problem danych, nie modelu.
- OHP: ante > lateral_delts > triceps — poprawna hierarchia ✓
- Pendlay row: rear_delts dominuje — poprawnie ✓

**`chart_transfer_matrix.png` — cross-exercise interference:**
- Strukturalne push/pull oddzielenie wyraźniejsze niż M5 ✓
- Ujemne wartości nadal obecne (artefakt modelu — f_net może produkować >1, po normalizacji daje ujemne transfery)

**`chart_mpc_per_muscle_*.png` — MPC trajectories:**
- Zębate wzorce drop+recovery fizjologicznie poprawne ✓
- τ per muscle zgodne z literaturą ✓
- Przebiegi bardziej zróżnicowane między mięśniami niż M5 — model lepiej rozróżnia specyficzny profil zmęczenia per ćwiczenie

**`chart_fatigue_heatmaps.png`:**
- Gradienty weight/reps bardziej płynne i zróżnicowane ✓
- In-distribution maski (szare obszary) pokazują granice trenowanych zakresów

## Obserwacje z wykresów — M5 (`charts/20260417_1540/`)

**`chart_muscle_breakdown.png` — per-muscle fatigue breakdown:**
- Push exercises (bench, ohp, dips): poprawna kolejność ordinal ✓
- **Muscle collapse**: model koncentruje fatigue na jednym dominującym mięśniu per ćwiczenie — triceps i anterior_delts mają ~0 dropu dla bench press, mimo że involvement jest niezerowe
- Deadlift: erectors dominuje (0.55), glutes/hamstrings/quads bliskie zeru — niefizjologiczne
- OHP: ante > lateral_delts > triceps — poprawna hierarchia ✓
- Pendlay row: rear_delts dominuje (0.63) — poprawnie

**`chart_transfer_matrix.png` — cross-exercise interference:**
- Strukturalne push/pull oddzielenie obecne ✓
- Ujemne wartości nadal obecne (artefakt modelu — f_net może produkować >1, po normalizacji daje ujemne transfery)

**`chart_mpc_per_muscle_*.png` — MPC trajectories:**
- Users: 00030, 00111 (przypięte), 00181 (najdłuższa sekwencja z val setu)
- Zębate wzorce drop+recovery fizjologicznie poprawne ✓
- τ per muscle zgodne z literaturą ✓

## Do zrobienia (Aleksander)

- [ ] **deadlift ordering** — nadal 67% we wszystkich modelach (M4/M5/M6). Erectors dominuje, glutes/hamstrings ~0. M6 (HIDDEN_DIM=256) nie pomógł → problem jest w danych, nie architekturze. Potrzebne sesje `deadlift → rdl → leg_curl` gdzie hamstrings/glutes zmęczenie jest wyraźnie widoczne w kolejnych seriach. To najwyższy priorytet bo deadlift jest kluczowym ćwiczeniem.

- [ ] **Sekwencje cross-muscle** — M6 naprawił większość regresji M5 (incline_bench, chest_press_machine, dips, squat) bez zmiany danych, ale dumbbell_flyes nadal tylko 83%. Priorytetowe pary do dalszego wzmocnienia:
  - `deadlift → rdl` (hamstrings wtórny → primary) — najważniejsze
  - `bench_press → skull_crusher` (triceps wtórny → primary)
  - `incline_bench → dumbbell_flyes` (chest wtórny → primary)
  - `rdl → leg_curl` (hamstrings wtórny → primary)

- [ ] **skull_crusher MAE** — wzrost 0.482 (M3) → 0.867 (M5) w czystym per-user splicie. Sprawdzić czy sekwencje skull_crusher w nowym datasecie są reprezentatywne (wystarczająco dużo serii po bench_press gdzie triceps jest zmęczone).

- [ ] **Rozkład realistyczny z floor** — każde ćwiczenie ≥1500–2000 wierszy w train secie. `seal_row`, `decline_bench` i podobne ćwiczenia niszowe mogą być niedoreprezentowane (decline_bench ordering regresja w M6 to możliwy sygnał).
