# Train Report — Milestone 4

## Metryki

| | Baseline (27 ćw.) | M2 (44 ćw.) | M3 (47 ćw., 17 mięśni) | M4 (34 ćw., 15 mięśni) |
|---|---|---|---|---|
| Val RMSE | 1.08 RIR | 1.005 RIR | 0.845 RIR | **0.869 RIR** |
| MAE | 0.86 RIR | 0.789 RIR | 0.656 RIR | **0.673 RIR** |
| R | 0.789 | 0.833 | 0.884 | **0.878** |

Dataset: `training_data_train/val.csv` (Aleksander, olek/dev) — 588k train / 167k val, 155/45 userów, 34 ćwiczenia.
Checkpoint: `deepgain_model_best.pt` (val RMSE 0.869, epoka 150/150).

M4 nieznacznie gorszy od M3 — główna przyczyna: mniejszy dataset (588k vs 872k, ~33% mniej). Model jest za to teraz **poprawny architektonicznie** (patrz sekcja "Co się zmieniło").

## Co się zmieniło względem M3

- **15 mięśni** (usunięto `upper_traps` i `brachialis`) — brak bezpośrednich kolumn EMG w schemacie CSV, nie powinny być w modelu
- **EMG weights z CSV** (`exercise_muscle_weights_scaled.csv`) jako primary source — zastąpiły flat 0.7 fallback dla 33/34 ćwiczeń. W M3 ~20 ćwiczeń miało błędne flat 0.7 involvement.
- **Nowy dataset Aleksandra** — 57 peer-reviewed sources, lepsza jakość generatora
- **34 ćwiczenia** (tylko te z yaml, nie union z hardkodowanymi jak w M3)
- `spoto_press` — jedyne ćwiczenie bez danych EMG w CSV, fallback na hardkod

## Per-exercise MAE

| Ćwiczenie | M3 MAE | M4 MAE | Δ |
|---|---:|---:|---:|
| skull_crusher | 0.481 | **0.482** | +0.001 |
| low_bar_squat | 0.442 | **0.503** | +0.061 |
| deadlift | 0.480 | **0.505** | +0.025 |
| sumo_deadlift | 0.593 | **0.581** | -0.012 |
| pull_up | 0.626 | **0.605** | -0.021 |
| incline_bench | 0.592 | **0.607** | +0.015 |
| spoto_press | 0.590 | **0.607** | +0.017 |
| lat_pulldown | 0.610 | **0.611** | +0.001 |
| pendlay_row | 0.619 | **0.619** | 0.000 |
| rdl | 0.631 | **0.627** | -0.004 |
| high_bar_squat | 0.644 | **0.638** | -0.006 |
| ohp | 0.692 | **0.642** | -0.050 |
| incline_bench_45 | 0.675 | **0.652** | -0.023 |
| dips | 0.647 | **0.660** | +0.013 |
| leg_extension | 0.688 | **0.672** | -0.016 |
| close_grip_bench | 0.614 | **0.675** | +0.061 |
| bench_press | 0.664 | **0.687** | +0.023 |
| dumbbell_flyes | 0.695 | **0.688** | -0.007 |
| farmers_walk | 0.724 | **0.692** | -0.032 |
| leg_press | 0.718 | **0.692** | -0.026 |
| chest_press_machine | 0.711 | **0.700** | -0.011 |
| bulgarian_split_squat | 0.656 | **0.709** | +0.053 |
| leg_curl | 0.740 | **0.726** | -0.014 |
| squat | 0.717 | **0.761** | +0.044 |
| suitcase_carry | 0.780 | **0.764** | -0.016 |
| leg_raises | 0.776 | **0.764** | -0.012 |
| ab_wheel | 0.783 | **0.788** | +0.005 |
| reverse_fly | 0.785 | **0.856** | +0.071 |
| trx_bodysaw | 0.886 | **0.874** | -0.012 |
| bird_dog | 0.923 | **0.892** | -0.031 |
| plank | 0.883 | **0.914** | +0.031 |
| decline_bench | 0.713 | **0.925** | +0.212 |
| dead_bug | 0.937 | **0.987** | +0.050 |
| seal_row | 0.611 | — | brak w test secie |

`seal_row` — **0 wierszy w test secie**. Całkowity dataset ma tylko 315 wierszy dla tego ćwiczenia (potwierdzone). Wymaga floor distribution od Aleksandra (patrz sekcja "Do zrobienia").

## Ordering accuracy (eval_ordering.py)

Uruchomić: `python eval_ordering.py`

**Ważna obserwacja:** wiele ćwiczeń ma 100% accuracy, ale wtórne mięśnie mają drop ~0.000. Oznacza to że kolejność jest "technicznie poprawna" (0.409 > 0.000 > 0.000 ✓), ale model nie uczy się faktycznego zmęczenia mięśni wtórnych — to jest problem **muscle collapse** (patrz sekcja Obserwacje).

Pełne wyniki (posortowane od najgorszych, probe: 80kg × 8 reps × RIR 2):

| Ćwiczenie | Acc | Drops (ordinal order, primary→last) |
|---|---:|---|
| bird_dog | 0% | glut(0.002) > erec(0.004) — **odwrócone** |
| lat_pulldown | 50% | lats(0.001) > rhom(0.000) > rear(0.001) > bice(0.001) — wszystko ~0 |
| deadlift | 67% | erec(**0.647**) > glut(0.003) > hams(0.001) > quad(0.003) |
| rdl | 67% | hams(**0.773**) > glut(0.340) > erec(0.390) — erec > glut inwersja |
| bulgarian_split_squat | 67% | quad(**1.000**) > glut(0.999) > erec(**1.000**) — brak rozróżnienia |
| pendlay_row | 83% | rear(**0.629**) > erec(0.622) > rhom(0.005) > lats(0.005) |
| squat | 100% | quad(**0.538**) > erec(0.001) > glut(0.001) — **collapse: tylko quads** |
| low_bar_squat | 100% | addu(**0.658**) > erec(0.401) > calv(0.077) |
| high_bar_squat | 100% | quad(**0.576**) > glut(0.299) |
| sumo_deadlift | 100% | abs(**0.626**) > quad(0.068) > glut(0.004) > calv(0.001) > erec(0.001) — abs primary? |
| bench_press | 100% | ches(**0.409**) > tric(0.000) > ante(0.000) — **collapse: tylko chest** |
| close_grip_bench | 100% | ches(**0.947**) > tric(0.945) > ante(0.613) — OK, wszystkie wysokie |
| spoto_press | 100% | ches(**0.805**) > tric(0.787) > ante(0.000) — ante zerowe |
| incline_bench | 100% | tric(**0.848**) > ches(0.488) > ante(0.451) — tric primary? sprawdzić yaml |
| incline_bench_45 | 100% | ches(**0.429**) > tric(0.028) > ante(0.000) — tric/ante ~0 |
| decline_bench | 100% | tric(**0.270**) > ches(0.156) > ante(0.002) — ante ~0 |
| chest_press_machine | 100% | ches(**0.018**) > ante(0.005) — oba niskie |
| dips | 100% | ches(**0.157**) > tric(0.152) > ante(0.062) — OK |
| ohp | 100% | ante(**0.627**) > late(0.570) > tric(0.146) — OK |
| dumbbell_flyes | 100% | ches(**0.037**) > ante(0.003) |
| skull_crusher | 100% | tric(**0.008**) > ante(0.008) — oba niskie |
| leg_press | 100% | quad(**0.623**) > glut(0.024) — glut ~0 |
| pull_up | 100% | rear(**0.853**) > lats(0.845) > rhom(0.769) > bice(0.768) — **OK, wszystkie wysokie** |
| reverse_fly | 100% | rear(**0.001**) > late(0.001) — oba ~0 |
| seal_row | 100% | rear(**0.203**) > rhom(0.027) > lats(0.025) — rhom/lats niskie |
| farmers_walk | 100% | erec(**0.012**) > abs(0.011) |
| leg_raises | 100% | abs(**0.063**) > lats(0.003) > quad(0.002) |
| suitcase_carry | 100% | abs(**0.011**) > erec(0.002) |
| **MEAN** | **90%** | |

**Wzorce collapse (dla Aleksandra):** bench_press, squat, deadlift, leg_press, incline_bench_45 mają jeden dominujący mięsień, reszta ~0. To sygnał że dataset nie ma wystarczająco wyraźnych sekwencji cross-muscle interference (np. bench → skull crusher → ohp gdzie zmęczone triceps wpływają na kolejne ćwiczenia).

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

- [ ] **Więcej danych** — 588k vs 872k w M3 to ~33% mniej. Docelowo ≥800k train, wtedy spodziewane zejście poniżej 0.845. Główna przyczyna plateau na 0.87.

- [ ] **Floor distribution** — minimum ~2% udziału per ćwiczenie w datasecie. `seal_row` ma 315 wierszy w całym datasecie → 0 w test secie → brak oceny jakości. Każde ćwiczenie powinno mieć ≥1500–2000 wierszy.

- [ ] **Rozkład realistyczny z floor** — realistyczna częstotliwość ćwiczeń (compound częściej niż izolacje) zachowana, ale z minimalnym progiem. Nie uniform.

- [ ] **bird_dog i lat_pulldown** — ordering accuracy 0% i 50%. Słaby sygnał w datasecie dla tych ćwiczeń. Warto sprawdzić czy sekwencje gdzie bird_dog sąsiaduje z innymi ćwiczeniami core/glutes są wystarczająco liczne.

- [ ] **Sekwencje cross-muscle** — model collapse (bench press = chest only) wynika z braku wyraźnego sygnału że zmęczone triceps po bench psują skull crushery. Dataset powinien mieć więcej sesji: bench → skull crusher → ohp w tej samej sesji.

## Do zrobienia (Michal)

- [ ] **Minimum drop penalty** — dodać do loss: `relu(min_involvement * involvement[mi] - drop)` dla wszystkich mięśni z niezerowym involvement. Zapobiegnie collapse do jednego mięśnia per ćwiczenie. Implementować po nowym datasecie od Aleksandra.

- [ ] **inference.py** — aktualizacja do 15 mięśni / 34 ćwiczeń (bloker dla Miłosza).

- [ ] **Ablacje** — z/bez ordinal penalty, z/bez minimum drop penalty, różne embedding dims. Po stabilnym datasecie.
