# README — Dynamic 1RM Anchors Variant

## Cel

Ta gałąź naprawia lukę w modelu, który wcześniej nie rozróżniał dwóch użytkowników o różnym poziomie siły dla tego samego seta.

Wariant końcowy rozdziela dwa sygnały:

- `MPC` = bieżące zmęczenie / świeżość
- `1RM anchors` = poziom siły użytkownika

Użytkownik startuje z 3 anchorami:

- `bench_press 1RM`
- `squat 1RM`
- `deadlift 1RM`

Na ich podstawie model liczy dla każdego ćwiczenia:

- `projected_1rm`
- `relative_load = weight / projected_1rm`
- `projection_available`

## Co zostało zrobione

### 1. Wspólna logika priors i update'u

Plik [strength_priors.py](/Users/michal/Documents/WB2/mpc-poc/strength_priors.py) jest wspólnym modułem dla treningu, inference i walidacji.

Zawiera:

- mapowanie `exercise -> anchor`
- `ratio_mean` dla projekcji siły
- parsowanie onboardingowych anchorów `config_1rm_*`
- projekcję `exercise 1RM <- anchor`
- update anchorów z `weight + reps + RIR`

Aktualizacja anchorów działa przez:

- kandydat `e1RM` liczony z `Epley + RIR`
- filtrowanie jakości setów
- przeliczenie z wariacji ćwiczenia na właściwy anchor
- miękki update z limitem zmiany

### 2. Dynamiczne `estimated_1rm_before_set` w treningu

W [train.py](/Users/michal/Documents/WB2/mpc-poc/train.py):

- loader czyta `config_1rm_bench_press`, `config_1rm_squat`, `config_1rm_deadlift`
- dla każdego usera budowany jest kauzalny stan anchorów przed każdym setem
- model dostaje już nie stałe anchory onboardingowe, tylko `anchors_before_set`
- `f_net` i `g_net` korzystają z nowych cech siły w każdym kroku sekwencji

To oznacza, że trening jest już spójny z założeniem:

- `dynamic fatigue`
- `dynamic strength`

bez leakage, bo anchor dla seta `t` powstaje wyłącznie z przeszłości.

### 3. Inference zgodne z nową wersją

W [inference.py](/Users/michal/Documents/WB2/mpc-poc/inference.py):

- `predict_mpc(...)` przyjmuje opcjonalne `strength_anchors`
- `predict_rir(...)` przyjmuje opcjonalne `strength_anchors`
- `load_model(...)` wykrywa stare vs nowe checkpointy przez `strength_feature_dim`
- `update_strength_anchors(...)` umożliwia aktualizację anchorów po realnych setach / sesji

Inference jest więc gotowe do użycia z plannerem:

- onboarding daje 3 anchory
- planner używa ich przy predykcji
- po sesji anchory mogą zostać zaktualizowane na przyszłość

### 4. Walidacja

Są dwa poziomy walidacji:

- [test.py](/Users/michal/Documents/WB2/mpc-poc/test.py)
- [validate_1rm_dynamics.py](/Users/michal/Documents/WB2/mpc-poc/validate_1rm_dynamics.py)

#### `test.py`

Sprawdza:

1. czy checkpoint ma `strength_feature_dim > 0`
2. czy model rozróżnia profile `weak / base / strong`
3. czy `RIR` rośnie monotonicznie przy zmianie właściwego anchora
4. czy update anchorów działa w dobrą stronę

Uruchomienie:

```bash
./venv/bin/python test.py
```

#### `validate_1rm_dynamics.py`

Sprawdza na holdoucie trzy warianty:

- `dynamic_correct`
- `static_onboarding`
- `shuffled_dynamic`

Uruchomienie:

```bash
./venv/bin/python validate_1rm_dynamics.py --num-users 63
```

Wynik dla pełnego holdoutu:

- `dynamic_correct   rmse=0.9129 mae=0.7083 corr=0.8620`
- `static_onboarding rmse=1.0231 mae=0.7972 corr=0.8296`
- `shuffled_dynamic  rmse=1.2920 mae=0.9317 corr=0.7202`

Wniosek:

- dynamiczne anchory wygrywają ze statycznym onboardingiem
- poprawne anchory wygrywają z anchorami błędnie przypisanymi

## Najnowsze wykresy

Ostatni pełny run jest w:

- [charts/20260423_0945](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945)

Najważniejsze pliki:

- [chart_loss_curves.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945/chart_loss_curves.png)
- [chart_rir_accuracy.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945/chart_rir_accuracy.png)
- [chart_strength_anchor_trajectories.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945/chart_strength_anchor_trajectories.png)
- [chart_strength_sweeps.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945/chart_strength_sweeps.png)
- [chart_dynamic_vs_static_rir.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260423_0945/chart_dynamic_vs_static_rir.png)

## Co ten wariant dziś potrafi

- model odróżnia użytkowników o różnym poziomie siły
- train i inference używają tej samej logiki priors
- anchor `1RM` może zmieniać się w czasie
- dynamiczne `estimated_1rm_before_set` poprawia predykcję względem stałych anchorów onboardingowych

## Ograniczenia obecnej wersji

- update anchorów w treningu jest obecnie `per-set`, nie `per-session`
- statystyki walidacyjne pokazują bias anchorów w dół, więc updater jest skuteczny, ale niekoniecznie idealnie skalibrowany
- ćwiczenia `bodyweight/core/carry` nie mają pełnego odpowiednika 4. anchora (`bodyweight`)

## Najbliższy kolejny krok

Najbardziej sensowny następny eksperyment:

- porównać `per-set update` vs `per-session update`
- ewentualnie dać osobne tempo update'u dla wzrostów i spadków anchorów

Ale na obecnym etapie wariant jest już technicznie i empirycznie obroniony.
