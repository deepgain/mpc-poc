# Train Report — Milestone 2

## Metryki

| | Baseline (27 ćw.) | Milestone 2 (44 ćw.) |
|---|---|---|
| Val RMSE | 1.08 RIR | **1.005 RIR** |
| MAE | 0.86 RIR | **0.789 RIR** |
| R | 0.789 | **0.833** |

Dataset: `generated_datasets/baseline_main/` — 846k train / 218k val, 238/62 userów, 44 ćwiczenia.

## Co się zmieniło względem baseline

- Ordinal involvement z `exercise_muscle_order.yaml` (Aleksander) zamiast hardkodowanych wag numerycznych
- `fatigue_ordering_penalty()` używa kolejności ordinalnej (primary → secondary → tertiary)
- Nowe ćwiczenia z yaml dostają flat involvement 0.7 dla wszystkich mięśni z listy
- τ (czas regeneracji) zamrożone z literatury

## Per-exercise MAE

| Ćwiczenie | MAE |
|---|---:|
| low_bar_squat | 0.565 |
| deadlift | 0.565 |
| skull_crusher | 0.698 |
| pendlay_row | 0.708 |
| pull_up | 0.722 |
| incline_bench_45 | 0.723 |
| lat_pulldown | 0.743 |
| spoto_press | 0.746 |
| rdl | 0.747 |
| high_bar_squat | 0.748 |
| bulgarian_split_squat | 0.757 |
| squat | 0.776 |
| bench_press | 0.783 |
| farmers_walk | 0.786 |
| dumbbell_flyes | 0.790 |
| ohp | 0.812 |
| close_grip_bench | 0.823 |
| leg_extension | 0.851 |
| chest_press_machine | 0.859 |
| leg_press | 0.863 |
| incline_bench | 0.863 |
| suitcase_carry | 0.877 |
| seal_row | 0.886 |
| leg_curl | 0.912 |
| reverse_fly | 0.915 |
| leg_raises | 0.958 |
| bird_dog | 0.968 |
| trx_bodysaw | 1.078 |
| decline_bench | 1.211 |
| dips | 1.414 |

## Obserwacje z wykresów

**`chart_muscle_breakdown.png` — per-muscle fatigue breakdown:**
- Większość ćwiczeń: kolejność ordinal jest zachowana ✓
- **Squat**: `glutes` prawie zerowy drop mimo secondary w yaml. Przyczyną jest niezgodność — w hardkodowanych wagach `glutes=0.60` (wysoko), ale yaml mówi `primary=[quads, erectors], secondary=[glutes]`. Model widzi wysoki involvement glutes w INVOLVEMENT_MATRIX i daje mały drop przez f_net — niespójność między macierzą a ordinal rankingiem.
- **Deadlift**: `hamstrings` prawie zerowy mimo primary w yaml. Ta sama przyczyna — hardkodowane `hamstrings=0.55`, yaml primary=[erectors, glutes, hamstrings], ale model ignoruje hamstrings bo f_net nie dostaje sygnału żeby go uczyć proporcjonalnie.

**`chart_transfer_matrix.png` — cross-exercise interference:**
- Struktura fizjologiczna poprawna (pchanie interferuje z pchaniem, ciągnięcie z ciągnięciem) ✓
- `lat_pulldown → pull_up: -1.0` — artefakt, ujemna wartość jest nielogiczna (ćwiczenie nie może regenerować)
- `dips → pull_up: 3.7` — za wysoka, dips to pchanie (klatka/triceps), pull_up to ciągnięcie (lats/biceps), mięśniowy overlap jest minimalny

**`chart_mpc_per_muscle_*.png` — MPC trajectories:**
- Zabki (drop + recovery) wyglądają fizjologicznie poprawnie ✓
- τ per muscle zgodne z literaturą

## Do zrobienia

- [ ] **`abs` jako 17. mięsień**
  - Problem: ćwiczenia core (plank, ab_wheel, bird_dog, trx_bodysaw, leg_raises) mają MAE ~1.0 bo model ma tylko 16 mięśni bez `abs` — te ćwiczenia trenują mięsień którego model nie śledzi
  - Co trzeba: Aleksander potwierdza `abs` jako oficjalny muscle ID w yaml, ja dodaję go do `ALL_MUSCLES` w `train.py` (zmiana rozmiaru modelu z 16 na 17) i retrenujemy od zera
  - Uwaga: to zmienia rozmiar modelu, stary checkpoint będzie niekompatybilny

- [ ] **Poprawić hardkodowane wagi dla squat i deadlift** (i ewentualnie innych ćwiczeń z yaml)
  - Problem: `_EXERCISE_MUSCLES_HARDCODED` w `train.py` ma inne proporcje wag niż ordinal ranking w yaml Aleksandra. Dla squata: hardkodowane daje `glutes` wyżej niż `erectors`, ale yaml mówi odwrotnie. Dla deadlifta: `hamstrings` jest za nisko.
  - Co trzeba: Aleksander przejrzy yaml i wskaże które ćwiczenia mają ewidentnie złe rankingi, albo da mi aktualne wagi numeryczne dla tych ćwiczeń — ja aktualizuję `_EXERCISE_MUSCLES_HARDCODED` i retrenujemy
  - Alternatywnie: całkowite zastąpienie hardkodowanych wag przez wagi z yaml (wymaga dodania `involvement_weight` do yaml)

- [ ] **Dips MAE 1.414 — najgorsze ćwiczenie**
  - Problem: dips ma najwyższy błąd ze wszystkich ćwiczeń, i było też najgorsze w baseline (1.458)
  - Co sprawdzić po stronie Aleksandra: czy dane dla dips w datasecie wyglądają sensownie (rozkład RIR, wag, reps)? Czy ordinal ranking w yaml jest poprawny? Może dips ma zbyt duże RIR variance w danych?
  - Co można spróbować: zwiększyć liczbę serii dips w datasecie jeśli ich jest mało