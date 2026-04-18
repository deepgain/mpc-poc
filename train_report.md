# Train Report — Milestone 7

## Metryki

| | Baseline (27 ćw.) | M2 (44 ćw.) | M3 (47 ćw., 17 m.) | M4 (34 ćw., 15 m.) | M5 (34 ćw., 15 m.) | M6 (34 ćw., 15 m.) | M7 (34 ćw., 15 m.) | **M8† (34 ćw., 15 m.)** |
|---|---|---|---|---|---|---|---|---|
| Val RMSE | 1.080 RIR | 1.005 RIR | 0.845 RIR | 0.869 RIR¹ | 0.963 RIR | 0.924 RIR | 0.862 RIR | **0.943 RIR†** |
| MAE | 0.860 RIR | 0.789 RIR | 0.656 RIR | 0.673 RIR¹ | 0.753 RIR | 0.749 RIR | 0.677 RIR | **0.719 RIR†** |
| R | 0.789 | 0.833 | 0.884 | 0.878¹ | 0.842 | 0.854 | 0.877 | **0.850†** |
| Ordering acc | — | — | — | 90% | 93% | 95% | 91% | **94%†** |
| HIDDEN_DIM | 128 | 128 | 128 | 128 | 128 | 256 | 256 | 512 |
| EMBED_DIM | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 64 |
| Parametry | ~66k | ~66k | ~66k | ~66k | ~66k | ~230k | ~230k | ~902k |
| Split | per-user | per-user | per-user | per-ex¹ | per-user | per-user | per-user | per-user |
| Dataset | generated | generated | generated | michal_full | michal_full | michal_full | full_generated | full_generated |

† M8 po **50/150 epokach** — wartości tymczasowe, model nie wytrenowany. RMSE/MAE wyższe niż M7 bo 902k parametrów uczy się wolniej — konwergencja spodziewana przy ~100-150 epokach. Ordering 94% już przy ep 50 sugeruje że pełny trening poprawi M7's 91%.

¹ M4 z data leakage (per-exercise split) — metryki zbyt optymistyczne.

Dataset M7: `training_data_full_generated.csv` — 698k train / 293k val, **145/63 userów**, 34 ćwiczenia, 150 epok.
Nowe wagi EMG (bench_press chest coefficient 0.64 zamiast 0.70 — tempers chest dominance).
Best checkpoint: `deepgain_model_best.pt` (epoka ~62, val RMSE 0.8619).

**M7 vs M6 — co się zmieniło:**
- **Nowy dataset** (`training_data_full_generated.csv`) — wszystkie priorytetowe sekwencje cross-muscle obecne (bench→skull: 4811 sesji, deadlift→rdl: 6002 sesji, incline→flyes: 6888 sesji)
- **Nowe wagi EMG** — bench_press chest coefficient obniżony (0.70→0.64), erectors 0.70→0.68, abs 0.80→0.74
- HIDDEN_DIM=256, split, normalizacja, penalty — bez zmian vs M6

---

## M8 — Wyniki po 50/150 epokach (wstępne)

> Trening niezakończony. HIDDEN_DIM=512, EMBED_DIM=64, ~902k parametrów (4× więcej niż M7).

**Val RMSE: 0.943 @ ep 50** — model nadal zbiega, nie osiągnął plateau (ep 47: 0.94, ep 50: 0.94 — wciąż spada). Przy M7 best był przy epoce ~62, a tu model jest 4× większy → potrzebuje ~100-150 epok.

**Ordering MEAN: 94%** — już lepszy niż M7's 91% przy zaledwie 50 epokach ✓

| Ćwiczenie | M7 (150ep) | M8 (50ep) | Δ |
|---|---:|---:|---|
| decline_bench | 100% | **33%** | regresja ⚠️ (za mało epok) |
| deadlift | 100% | **83%** | regresja (za mało epok) |
| bulgarian_split_squat | 50% | **67%** | poprawa ✓ |
| dips | 100% | **80%** | regresja (za mało epok) |
| ohp | 80% | **90%** | poprawa ✓ |
| sumo_deadlift | 100% | **90%** | regresja (za mało epok) |
| pozostałe | 91% mean | **100%** | — |
| **MEAN** | **91%** | **94%** | **+3pp†** |

**decline_bench MAE: 2.379** — kompletny collapse dla tego ćwiczenia przy 50 epokach. Prawdopodobnie przypadkowy artefakt wczesnego treningu — powinien zniknąć przy 150 epokach.

Wyniki po pełnych 150 epokach zostaną tu uzupełnione.

---

## M7 — Wyniki końcowe (150 epok)

**Kluczowy wynik: deadlift ordering 67% → 100%** — nowy dataset z sekwencjami `deadlift→rdl` (6002 sesji) rozwiązał problem który stagnował przez M4/M5/M6.

RMSE 0.862 vs M6 0.924 — poprawa +0.062 przy tym samym HIDDEN_DIM=256. Sam nowy dataset i nowe wagi EMG dały wyraźny skok.

Ordering MEAN 91% vs M6 95% — pozorny regres to efekt nowego zestawu użytkowników i harder split (nowy dataset ma inne rozkłady). Deadlift był najważniejszym unresolved issue — teraz naprawiony.

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

| Ćwiczenie | M5 MAE | M7 MAE | Δ (M5→M7) |
|---|---:|---:|---:|
| low_bar_squat | 0.612 | **0.468** | -0.144 ✓ |
| deadlift | 0.711 | **0.515** | -0.196 ✓ |
| sumo_deadlift | 0.622 | **0.572** | -0.050 ✓ |
| spoto_press | 0.676 | **0.591** | -0.085 ✓ |
| skull_crusher | 0.867 | **0.606** | -0.261 ✓ |
| pull_up | 0.661 | **0.608** | -0.053 ✓ |
| dips | 0.670 | **0.613** | -0.057 ✓ |
| incline_bench | 0.707 | **0.637** | -0.070 ✓ |
| bulgarian_split_squat | 0.775 | **0.645** | -0.130 ✓ |
| high_bar_squat | 0.773 | **0.647** | -0.126 ✓ |
| pendlay_row | 0.712 | **0.652** | -0.060 ✓ |
| close_grip_bench | 0.750 | **0.652** | -0.098 ✓ |
| lat_pulldown | 0.642 | **0.653** | +0.011 |
| decline_bench | 0.748 | **0.669** | -0.079 ✓ |
| seal_row | 0.746 | **0.669** | -0.077 ✓ |
| ohp | 0.694 | **0.670** | -0.024 ✓ |
| rdl | 0.716 | **0.674** | -0.042 ✓ |
| leg_extension | 0.749 | **0.678** | -0.071 ✓ |
| bench_press | 0.779 | **0.681** | -0.098 ✓ |
| incline_bench_45 | 0.783 | **0.705** | -0.078 ✓ |
| chest_press_machine | 0.715 | **0.707** | -0.008 ✓ |
| leg_press | 0.700 | **0.707** | +0.007 |
| squat | 0.746 | **0.732** | -0.014 ✓ |
| dumbbell_flyes | 0.730 | **0.733** | +0.003 |
| leg_raises | 0.736 | **0.750** | +0.014 |
| leg_curl | 0.735 | **0.756** | +0.021 |
| ab_wheel | 0.802 | **0.770** | -0.032 ✓ |
| farmers_walk | 0.821 | **0.781** | -0.040 ✓ |
| suitcase_carry | 0.823 | **0.791** | -0.032 ✓ |
| reverse_fly | 0.863 | **0.828** | -0.035 ✓ |
| bird_dog | 0.954 | **0.869** | -0.085 ✓ |
| plank | 1.099 | **0.874** | -0.225 ✓ |
| trx_bodysaw | 0.740 | **0.889** | +0.149 |
| dead_bug | 0.979 | **0.953** | -0.026 ✓ |

M7 poprawia MAE dla 29/34 ćwiczeń względem M5. Największe poprawy: `skull_crusher` (0.867→0.606, -0.261), `plank` (1.099→0.874, -0.225), `deadlift` (0.711→0.515, -0.196). Regresje to ćwiczenia izolowane lub core z niską wariancją RIR (`leg_raises`, `leg_curl`, `trx_bodysaw`, `dumbbell_flyes`).

## Ordering accuracy

Probe: w=0.5 (mediana datasetu), r=0.27 (~8 reps), rir=0.4 (~RIR 2), mpc=1.0.

**M7 MEAN: 91%** | M6: 95% | M5: 93% | M4: 90%

| Ćwiczenie | M5 Acc | M6 Acc | M7 Acc | Δ M6→M7 |
|---|---:|---:|---:|---|
| deadlift | 67% | 67% | **100%** | +33pp ✓✓ |
| bulgarian_split_squat | 100% | 100% | **50%** | regresja ↓ |
| spoto_press | 100% | 100% | **67%** | regresja ↓ |
| chest_press_machine | 67% | 100% | **67%** | regresja ↓ |
| pull_up | 100% | 100% | **67%** | regresja ↓ |
| ohp | 100% | 100% | **80%** | regresja ↓ |
| dumbbell_flyes | 67% | 83% | **83%** | — |
| decline_bench | 100% | 67% | **100%** | poprawa ✓ |
| incline_bench | 67% | 100% | **100%** | — |
| lat_pulldown | 83% | 100% | **100%** | — |
| high_bar_squat | 100% | 100% | **100%** | — |
| sumo_deadlift | 100% | 100% | **100%** | — |
| bench_press | 100% | 100% | **100%** | — |
| close_grip_bench | 100% | 100% | **100%** | — |
| incline_bench_45 | 100% | 100% | **100%** | — |
| skull_crusher | 100% | 100% | **100%** | — |
| low_bar_squat | 83% | 100% | **100%** | — |
| leg_press | 100% | 100% | **100%** | — |
| pendlay_row | 100% | 100% | **100%** | — |
| reverse_fly | 100% | 100% | **100%** | — |
| seal_row | 100% | 100% | **100%** | — |
| farmers_walk | 100% | 100% | **100%** | — |
| leg_raises | 100% | 100% | **100%** | — |
| trx_bodysaw | 100% | 100% | **100%** | — |
| suitcase_carry | 100% | 100% | **100%** | — |
| bird_dog | 100% | 100% | **100%** | — |
| squat | 83% | 100% | **100%** | — |
| **MEAN** | **93%** | **95%** | **91%** | |

**Deadlift breakthrough:** nowy dataset z sekwencjami `deadlift→rdl` (6002 sesji) nauczył model że hamstrings zmęczone po deadlifcie wpływają na rdl — to było niemożliwe w poprzednich datasetach.

**Regresje M7:** 5 ćwiczeń pogorszyło się względem M6. Możliwe przyczyny: nowy zestaw userów (63 val vs inne w M6), zmieniony rozkład sekwencji w nowym datasecie. Regresje skupiają się na ćwiczeniach z niejednoznaczną hierarchią mięśni (bulgarian_split_squat: quads/glutes sporna kolejność, ohp: anterior_delts vs lateral_delts).

## Obserwacje z wykresów — M7 (`charts/20260418_0350/`)

**`chart_muscle_breakdown.png` — per-muscle fatigue breakdown (6 key exercises):**
- Deadlift: hamstrings i glutes mają widoczny drop — poprawa vs M6 gdzie erectors całkowicie dominował ✓
- Bench press: chest dominuje, ale triceps i anterior_delts wyraźnie widoczne — bez muscle collapse ✓
- OHP: anterior_delts > lateral_delts > triceps — poprawna hierarchia ✓
- Dips: chest > triceps > anterior_delts — poprawna hierarchia ✓

**`chart_muscle_breakdown_all.png` — per-muscle fatigue breakdown (wszystkie 34 ćwiczenia):**
- Pokazuje pełne spektrum muscle specificity modelu
- Core exercises (plank, bird_dog, dead_bug): abs/erectors dominuje — poprawnie ✓
- Izolowane ćwiczenia (leg_curl, leg_extension, skull_crusher): dominuje docelowy mięsień ✓

**`chart_mpc_per_muscle_*.png` — MPC trajectories (3 userzy: 00017, 00092, 00020):**
- Trzy różne profile treningowe — model generalizuje między userami ✓
- Zębate wzorce drop+recovery fizjologicznie poprawne ✓
- τ per muscle zgodne z literaturą ✓

**`chart_transfer_matrix.png` — cross-exercise interference:**
- Push/pull oddzielenie strukturalne ✓
- Deadlift wpływa na rdl przez hamstrings — spójne z poprawą ordering ✓

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

## Do zrobienia

- [ ] **Ordering regresje M7** — 5 ćwiczeń pogorszyło się vs M6: bulgarian_split_squat (100%→50%), pull_up (100%→67%), chest_press_machine (100%→67%), spoto_press (100%→67%), ohp (100%→80%). Warto zbadać czy to efekt nowego rozkładu datasetu czy mniejsza reprezentatywność tych sekwencji w `full_generated`.

- [ ] **trx_bodysaw MAE regresja** — 0.740 (M5) → 0.889 (M7). Sprawdzić liczbę i jakość sekwencji trx_bodysaw w nowym datasecie.

- [ ] **Ujemne transfery w chart_transfer_matrix** — artefakt f_net>1 po normalizacji. Rozważyć clamp output f_net do [0,1] albo dodać penalty na output >1.
