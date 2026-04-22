# README — Static 1RM Anchors Variant

## Cel

Ta gałąź dodaje do modelu informację o sile użytkownika przez 3 onboardingowe anchory:

- `bench_press 1RM`
- `squat 1RM`
- `deadlift 1RM`

Problem, który to adresuje:
- wcześniej model dostawał tylko `exercise + weight + reps + MPC`
- więc dla dwóch użytkowników o różnej sile ten sam set mógł wyglądać prawie identycznie
- model nie wiedział, czy `80 kg` to dla kogoś `80% 1RM`, czy `130% 1RM`

## Co zostało zrobione

### 1. Nowy sygnał siły użytkownika

Do danych wejściowych modelu dochodzą anchory:

- `config_1rm_bench_press`
- `config_1rm_squat`
- `config_1rm_deadlift`

Na ich podstawie dla każdego ćwiczenia liczona jest projekcja:

- `projected_1rm`
- `relative_load = weight / projected_1rm`
- `projection_available`

### 2. Wspólna tabela priors

Dodany został plik [strength_priors.py](/Users/michal/Documents/WB2/mpc-poc/strength_priors.py), który zawiera:

- mapowanie `exercise -> anchor`
- `ratio_mean` dla projekcji siły na pozostałe ćwiczenia
- helpery do parsowania anchorów z danych i inference

### 3. Zmiany w treningu

W [train.py](/Users/michal/Documents/WB2/mpc-poc/train.py):

- loader czyta nowe kolumny `config_1rm_*`
- batch trzyma anchory per user
- `f_net` i `g_net` dostały nowe feature’y siły
- trening korzysta z nowego pełnego datasetu `training_data.csv`

### 4. Zmiany w inference

W [inference.py](/Users/michal/Documents/WB2/mpc-poc/inference.py):

- `predict_rir(...)` przyjmuje opcjonalne `strength_anchors`
- `predict_mpc(...)` też może dostać anchory lub wyciągnąć je z historii
- `load_model(...)` wykrywa, czy checkpoint jest stary czy nowy

Stary checkpoint:
- ładuje się poprawnie
- ale nie korzysta z nowych feature’ów siły

Nowy checkpoint:
- ma `strength_feature_dim > 0`
- używa anchorów w predykcji

## Co ten wariant teraz potrafi

Model odróżnia użytkowników o różnym poziomie siły przy tym samym secie.

Przykład:
- ten sam `exercise`
- ta sama `weight`
- te same `reps`
- ten sam świeży `MPC`
- różne `1RM`

W takiej sytuacji model zwraca różne `RIR`.

To było sprawdzone probe’em w [test.py](/Users/michal/Documents/WB2/mpc-poc/test.py).

## Co testuje `test.py`

Plik [test.py](/Users/michal/Documents/WB2/mpc-poc/test.py) sprawdza:

1. Czy checkpoint faktycznie ma strength features:
   - `strength_feature_dim > 0`

2. Czy dla profili `weak / base / strong` model daje różne `RIR`

3. Czy `RIR` rośnie monotonicznie przy zwiększaniu właściwego anchora:
   - pressy reagują na `bench_press`
   - squat family reaguje na `squat`
   - deadlift family reaguje na `deadlift`

4. Diagnostycznie:
   - jak bardzo ćwiczenia reagują na niepowiązane anchory

Uruchomienie:

```bash
./venv/bin/python test.py
```

## Najnowsze wykresy

Wyniki ostatniego treningu są w:

- [charts/20260422_1759](/Users/michal/Documents/WB2/mpc-poc/charts/20260422_1759)

Najważniejsze pliki:

- [chart_loss_curves.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260422_1759/chart_loss_curves.png)
- [chart_rir_accuracy.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260422_1759/chart_rir_accuracy.png)
- [chart_transfer_matrix.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260422_1759/chart_transfer_matrix.png)
- [chart_rir_sensitivity.png](/Users/michal/Documents/WB2/mpc-poc/charts/20260422_1759/chart_rir_sensitivity.png)

## Czego jeszcze nie ma

Ten wariant jest **statyczny** po stronie siły użytkownika.

To znaczy:

- `MPC` zmienia się w czasie
- anchory `1RM` są na razie stałe
- nie ma jeszcze update’u `1RM` po sesjach treningowych

Czyli obecnie model zakłada:

- `dynamic fatigue`
- `static strength`

## Kolejny krok

Następny etap to dynamiczny update anchorów `1RM`, np.:

- po treningu
- na podstawie `weight + reps + RIR`
- bez data leakage

Docelowy kierunek:

- `MPC` dalej modeluje zmęczenie
- `1RM` anchory modelują siłę
- anchory są aktualizowane w czasie wraz z historią użytkownika
