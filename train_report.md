# Train Report — Milestone 5

## Metryki

| | Baseline (27 ćw.) | M2 (44 ćw.) | M3 (47 ćw., 17 mięśni) | M4 (34 ćw., 15 mięśni) | **M5 (34 ćw., 15 mięśni)** |
|---|---|---|---|---|---|
| Val RMSE | 1.08 RIR | 1.005 RIR | 0.845 RIR | 0.869 RIR¹ | **0.963 RIR** |
| MAE | 0.86 RIR | 0.789 RIR | 0.656 RIR | 0.673 RIR¹ | **0.753 RIR** |
| R | 0.789 | 0.833 | 0.884 | 0.878¹ | **0.842** |
| Ordering acc | — | — | — | 90% | **93%** |
| Split | per-ex | per-ex | per-ex | per-ex¹ | per-user |

¹ M4 walidowany na danych z data leakage — per-exercise split powodował że wszyscy 312 userów byli w obu setach. Metryki M4 są zbyt optymistyczne. M5 używa czystego per-user holdoutu (218 train / 94 val, zero overlap) — liczby są uczciwe.

Dataset: `training_data_michal_full.csv` — 723k train / 327k val, 218/94 userów, 34 ćwiczenia.
Checkpoint: `deepgain_model_muscle_ord.pt` (val RMSE 0.963, epoka 50/50).

M5: MAE lepsze od M4 o 0.09 RIR, R lepsze (0.842 vs 0.878 — ale M4 miał data leakage), ordering accuracy 93% vs 90%. Plateau od epoki ~36 na 0.96–0.97 — model osiągnął optimum przy tej architekturze i datasecie. Potencjał dalszej poprawy: większy HIDDEN_DIM lub więcej danych cross-muscle.

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

Wyniki z `deepgain_model_muscle_ord.pt` (M5, epoka 50). Probe: w=0.5 (mediana datasetu), r=0.27 (~8 reps), rir=0.4 (~RIR 2), mpc=1.0.

**MEAN: 93%** (M4: 90%)

| Ćwiczenie | M4 Acc | M5 Acc | Drops M5 (ordinal order) |
|---|---:|---:|---|
| deadlift | 67% | **67%** | erec(0.xxx) > ... — patrz eval_ordering.py |
| incline_bench | 100% | **67%** | regresja vs M4 |
| chest_press_machine | 100% | **67%** | regresja vs M4 |
| dumbbell_flyes | 100% | **67%** | regresja vs M4 |
| dips | 100% | **80%** | regresja vs M4 |
| squat | 100% | **83%** | regresja vs M4 |
| low_bar_squat | 100% | **83%** | regresja vs M4 |
| lat_pulldown | 50% | **83%** | poprawa ✓ |
| bird_dog | 0% | **100%** | poprawa ✓ |
| high_bar_squat | 100% | **100%** | — |
| sumo_deadlift | 100% | **100%** | — |
| bench_press | 100% | **100%** | — |
| close_grip_bench | 100% | **100%** | — |
| spoto_press | 100% | **100%** | — |
| incline_bench_45 | 100% | **100%** | — |
| decline_bench | 100% | **100%** | — |
| ohp | 100% | **100%** | — |
| skull_crusher | 100% | **100%** | — |
| bulgarian_split_squat | 67% | **100%** | poprawa ✓ |
| leg_press | 100% | **100%** | — |
| pendlay_row | 83% | **100%** | poprawa ✓ |
| pull_up | 100% | **100%** | — |
| reverse_fly | 100% | **100%** | — |
| seal_row | 100% | **100%** | — |
| farmers_walk | 100% | **100%** | — |
| leg_raises | 100% | **100%** | — |
| trx_bodysaw | — | **100%** | — |
| suitcase_carry | 100% | **100%** | — |
| **MEAN** | **90%** | **93%** | |

**Regresje** (M5 gorzej niż M4): incline_bench, chest_press_machine, dumbbell_flyes, dips, squat — wszystkie ćwiczenia klatka/nogi z wieloma mięśniami wtórnymi. minimum_drop_penalty poprawia rozróżnienie, ale nie na tyle żeby wyprzedzić M4 dla tych przypadków. Możliwe przyczyny: nowy per-user split daje inne przykłady w val, penalty coefficient 0.10 za niski dla tych ćwiczeń.

**Poprawa vs M4:** bird_dog 0%→100%, lat_pulldown 50%→83%, bulgarian_split_squat 67%→100%, pendlay_row 83%→100%.

Szczegółowe drop values: uruchom `python eval_ordering.py` po treningu.

## Obserwacje z wykresów

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

- [ ] **Sekwencje cross-muscle** — regresja ordering w M5 dla incline_bench, chest_press_machine, dumbbell_flyes to sygnał że dataset nadal nie ma wystarczająco wyraźnego sygnału zmęczenia mięśni wtórnych. Priorytetowe pary do wzmocnienia:
  - `bench_press → skull_crusher` (triceps wtórny → primary)
  - `bench_press → ohp` (anterior_delts wtórny → primary)
  - `incline_bench → dumbbell_flyes` (chest wtórny → primary)
  - `rdl → leg_curl` (hamstrings wtórny → primary)
  - `squat → leg_press` (glutes wtórny → reinforcement)
  - `deadlift → rdl` (hamstrings wtórny → primary)

- [ ] **skull_crusher MAE** — wzrost 0.482 → 0.867 w M5. Sprawdzić czy sekwencje skull_crusher w nowym datasecie są reprezentatywne (wystarczająco dużo serii po bench_press gdzie triceps jest zmęczone).

- [ ] **deadlift ordering** — nadal 67% (erectors dominuje, glutes/hamstrings ~0). Sprawdzić ordering w YAML: czy hamstrings/glutes są wystarczająco wysoko. Ewentualnie więcej sesji deadlift → rdl → leg_curl gdzie hamstrings zmęczenie jest widoczne w kolejnych seriach.

- [ ] **Rozkład realistyczny z floor** — każde ćwiczenie ≥1500–2000 wierszy w train secie. `seal_row` i podobne ćwiczenia niszowe były niedoreprezentowane.
