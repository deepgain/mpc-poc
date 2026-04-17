# Train Report — Milestone 3

## Metryki

| | Baseline (27 ćw.) | Milestone 2 (44 ćw.) | Milestone 3 (47 ćw., 17 mięśni) |
|---|---|---|---|
| Val RMSE | 1.08 RIR | 1.005 RIR | **0.845 RIR** |
| MAE | 0.86 RIR | 0.789 RIR | **0.656 RIR** |
| R | 0.789 | 0.833 | **0.884** |

Dataset: `generated_datasets/baseline_main/` — 872k train / 225k val, 238/62 userów, 47 ćwiczeń.
Checkpoint: `deepgain_model_best.pt` (najlepsza epoka: 116, val RMSE 0.845).

## Co się zmieniło względem Milestone 2

- **`abs` jako 17. mięsień** (τ=10h) — ćwiczenia core teraz śledzą właściwy mięsień
- **150 epok** zamiast 20 — model w pełni zwergowany
- ord_pen → 0.0000 od ~epoki 90 — constraint ordinalny w pełni spełniony
- 3 nowe ćwiczenia z yaml: `sumo_deadlift`, `ab_wheel`, `dead_bug`
- Brak overfittingu: train RMSE 0.82 vs val RMSE 0.85 przez całe trenowanie

## Per-exercise MAE

| Ćwiczenie | M2 MAE | M3 MAE | Δ |
|---|---:|---:|---:|
| low_bar_squat | 0.565 | **0.442** | -0.123 |
| deadlift | 0.565 | **0.480** | -0.085 |
| skull_crusher | 0.698 | **0.481** | -0.217 |
| spoto_press | 0.746 | **0.590** | -0.156 |
| incline_bench | 0.863 | **0.592** | -0.271 |
| sumo_deadlift | — | **0.593** | nowe |
| lat_pulldown | 0.743 | **0.610** | -0.133 |
| seal_row | 0.886 | **0.611** | -0.275 |
| close_grip_bench | 0.823 | **0.614** | -0.209 |
| pendlay_row | 0.708 | **0.619** | -0.089 |
| pull_up | 0.722 | **0.626** | -0.096 |
| rdl | 0.747 | **0.631** | -0.116 |
| high_bar_squat | 0.748 | **0.644** | -0.104 |
| dips | 1.414 | **0.647** | **-0.767** |
| bulgarian_split_squat | 0.757 | **0.656** | -0.101 |
| bench_press | 0.783 | **0.664** | -0.119 |
| incline_bench_45 | 0.723 | **0.675** | -0.048 |
| leg_extension | 0.851 | **0.688** | -0.163 |
| ohp | 0.812 | **0.692** | -0.120 |
| dumbbell_flyes | 0.790 | **0.695** | -0.095 |
| chest_press_machine | 0.859 | **0.711** | -0.148 |
| decline_bench | 1.211 | **0.713** | **-0.498** |
| squat | 0.776 | **0.717** | -0.059 |
| leg_press | 0.863 | **0.718** | -0.145 |
| farmers_walk | 0.786 | **0.724** | -0.062 |
| leg_curl | 0.912 | **0.740** | -0.172 |
| leg_raises | 0.958 | **0.776** | -0.182 |
| suitcase_carry | 0.877 | **0.780** | -0.097 |
| ab_wheel | — | **0.783** | nowe |
| reverse_fly | 0.915 | **0.785** | -0.130 |
| plank | — | **0.883** | nowe |
| trx_bodysaw | 1.078 | **0.886** | -0.192 |
| bird_dog | 0.968 | **0.923** | -0.045 |
| dead_bug | — | **0.937** | nowe |

## Obserwacje z wykresów

**`chart_muscle_breakdown.png` — per-muscle fatigue breakdown:**
- Bench press: chest >> anterior_delts > triceps — kolejność ordinal zachowana ✓
- OHP: anterior_delts > triceps > upper_traps > chest — poprawna hierarchia ✓
- Pendlay row: rear_delts ≈ erectors ≈ rhomboids (flat 0.7 involvement dla nowego ćwiczenia) — brak hierarchii, ale symetryczne
- **Squat**: quads dominuje (poprawnie), ale glutes drop nadal bardzo mały mimo `secondary` w yaml. Ten sam problem co w M2 — niespójność między hardkodowanym `glutes=0.60` w INVOLVEMENT_MATRIX a ordinal rankiem secondary
- Dips: chest >> triceps > anterior_delts — fizjologicznie sensowne ✓

**`chart_transfer_matrix.png` — cross-exercise interference:**
- Struktura push/pull ogólnie poprawna ✓
- Ujemne wartości nadal obecne (bench_press→dips: -2.9, pendlay_row→pull_up: -3.2) — artefakt, fizjologicznie nielogiczne
- rdl→deadlift: 4.4, deadlift→deadlift: 4.3 — autocorrelacja wysoka, poprawna
- dips→pull_up: 0.6 (znacznie lepsza niż 3.7 z M2) ✓

**`chart_mpc_per_muscle_*.png` — MPC trajectories:**
- Zabki (drop + recovery) wyglądają fizjologicznie poprawnie ✓
- abs (17. mięsień) pojawia się jako osobny panel ✓
- τ per muscle zgodne z literaturą, bez anomalii

## Do zrobienia

- [x] **`abs` jako 17. mięsień** — done. Core exercises poprawiły się: leg_raises 0.958→0.776, trx_bodysaw 1.078→0.886, bird_dog 0.968→0.923. dead_bug i ab_wheel (~0.9) to nowe ćwiczenia bez M2 baseline.

- [ ] **Poprawić hardkodowane wagi dla squat i deadlift**
  - Problem: squat glutes drop nadal minimalny w M3 mimo 150 epok — potwierdzone że to niespójność INVOLVEMENT_MATRIX (glutes=0.60) vs ordinal yaml (glutes=secondary). Model nie ma sygnału żeby uczyć glutes dropu bo f_net dostaje wysoki involvement bez gradientu ordinal.
  - Co trzeba: Aleksander weryfikuje squat i deadlift w yaml (czy `primary=[quads, erectors]` dla squata jest zamierzone), potem aktualizujemy hardkody i retrenujemy

- [x] **Dips MAE 1.414 — rozwiązane**. MAE spadło z 1.414 do 0.647 po dodaniu abs i pełnym treningu. Prawdopodobna przyczyna M2: zbyt mało epok (20) + możliwy sygnał z core muscles overlap w datasecie.

- [ ] **Ujemne wartości w transfer matrix** — strukturalny problem modelu. f_net może produkować wartości > 1 dla niektórych kombinacji, co po normalizacji daje ujemne transfery. Potencjalnie wymaga clampowania `f_net` output do [0, 1] range.

- [ ] **bird_dog i dead_bug MAE ~0.93** — najgorsze pozostałe ćwiczenia. Flat 0.7 involvement (nowe ćwiczenia z yaml) może być zbyt słabym sygnałem. Warto sprawdzić czy Aleksander ma lepsze wagi dla tych ćwiczeń.
