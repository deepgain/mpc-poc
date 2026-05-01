# Personalizacja RIR przez 1RM

## Cel

Obecny problem jest taki, ze model zbyt slabo rozroznia realna sile konkretnych osob, przez co moze przewidywac zbyt podobny `RIR` dla roznych userow przy podobnym `(exercise, weight, reps)`.

Kierunek:
- user na poczatku podaje 3 bazowe wartosci:
  - `bench_press 1RM`
  - `squat 1RM`
  - `deadlift 1RM`
- z tych 3 wartosci budujemy startowa estymacje `1RM` dla pozostalych cwiczen
- potem te estymaty sa aktualizowane w trakcie uzywania aplikacji
- docelowo ma to sluzyc do bardziej personalnego liczenia `RIR`

## Co juz mamy

Do datasetu zostaly dodane 3 kolumny:
- `config_1rm_bench_press`
- `config_1rm_squat`
- `config_1rm_deadlift`

To jest OK jako punkt startowy, bo te 3 wartosci sa dokladnie tym, co user moze podac przy konfiguracji apki.

## Czy to robi data leakage

Same 3 startowe wartosci:
- nie robia `data leakage`

Statyczne zrzutowanie z tych 3 bojow na pozostale cwiczenia:
- tez nie robi `data leakage`

Dynamiczna aktualizacja `1RM`:
- tez nie robi `data leakage`, ale tylko jezeli dla seta `t` korzystamy wylacznie z danych dostepnych przed `t`

Leakage pojawia sie dopiero wtedy, gdy:
- dla wiersza / seta wykorzystujemy informacje z przyszlosci
- albo liczymy estymate na podstawie tego samego seta i podajemy ja modelowi jako ceche dla tego seta

Najwazniejsza zasada:
- feature dla seta musi byc policzony z przeszlosci, czyli semantycznie `estimated_1rm_before_set`

## Czy to kloci sie z tym, jak dziala generator

Nie.

Generator juz teraz wewnetrznie opiera sie na `e1rm`:
- losuje profil usera
- ustala bazowe `1RM` dla bench/squat/deadlift
- z tych wartosci aproksymuje `1RM` dla pozostalych cwiczen
- potem moduluje to przez day-to-day variability, recovery i fatigue transfer
- i dopiero z tego generuje sensowne ciezary, reps i `RIR`

Czyli nowy plan nie kloci sie z logika generatora.
On jest zgodny z tym, jak synthetic data juz teraz sa budowane.

## Jak najlepiej zrzutowac 3 boje na reszte cwiczen

Najbardziej sensowny wariant naukowo i inzyniersko:
- dla kazdego cwiczenia trzymac prior:
  - `anchor_lift`
  - `ratio_mean`
  - `ratio_sd`
  - `exercise_family`
- startowa sila na cwiczeniu:
  - `exercise_1rm_prior = anchor_1rm * ratio_mean`

Najwazniejsze zalozenie:
- to sa priory startowe, nie "prawda objawiona"
- po to dodajemy `ratio_sd`, zeby od razu modelowac niepewnosc
- im dalej cwiczenie jest od glownego boju, tym zwykle wyzsze `ratio_sd`

Najbardziej obroniony praktycznie punkt wyjscia na teraz:
- jako startowych priors uzyc tych samych relacji, na ktorych juz opiera sie generator synthetic data
- czyli zachowac spojna semantyke miedzy data generation i runtime personalization

Jak czytac `ratio_sd`:
- to nie jest "dokladne SD z literatury dla calej populacji"
- to jest praktyczny prior uncertainty do startowej inicjalizacji
- dla prostych priors z generatora mozna go traktowac jako szerokosc poczatkowej niepewnosci

### Gotowe priory dla cwiczen zakotwiczonych do bench press

| exercise_id | anchor_lift | ratio_mean | ratio_sd | exercise_family |
|---|---|---:|---:|---|
| `bench_press` | `bench_press` | 1.000 | 0.000 | `bench_primary` |
| `incline_bench` | `bench_press` | 0.815 | 0.020 | `bench_variant` |
| `close_grip_bench` | `bench_press` | 0.850 | 0.017 | `bench_variant` |
| `spoto_press` | `bench_press` | 0.900 | 0.023 | `bench_variant` |
| `incline_bench_45` | `bench_press` | 0.780 | 0.023 | `bench_variant` |
| `decline_bench` | `bench_press` | 0.900 | 0.023 | `bench_variant` |
| `chest_press_machine` | `bench_press` | 0.840 | 0.035 | `machine_press` |
| `ohp` | `bench_press` | 0.620 | 0.029 | `vertical_press` |
| `dips` | `bench_press` | 0.635 | 0.049 | `press_assistance` |
| `dumbbell_flyes` | `bench_press` | 0.290 | 0.029 | `chest_isolation` |
| `pendlay_row` | `bench_press` | 0.890 | 0.052 | `upper_pull` |
| `seal_row` | `bench_press` | 0.765 | 0.049 | `upper_pull` |
| `lat_pulldown` | `bench_press` | 0.680 | 0.046 | `vertical_pull` |
| `pull_up` | `bench_press` | 0.565 | 0.049 | `vertical_pull` |
| `skull_crusher` | `bench_press` | 0.295 | 0.032 | `triceps_isolation` |

### Gotowe priory dla cwiczen zakotwiczonych do squat

| exercise_id | anchor_lift | ratio_mean | ratio_sd | exercise_family |
|---|---|---:|---:|---|
| `squat` | `squat` | 1.000 | 0.000 | `squat_primary` |
| `low_bar_squat` | `squat` | 0.990 | 0.030 | `squat_variant` |
| `high_bar_squat` | `squat` | 0.960 | 0.030 | `squat_variant` |
| `leg_press` | `squat` | 1.450 | 0.087 | `machine_lower_compound` |
| `bulgarian_split_squat` | `squat` | 0.400 | 0.035 | `unilateral_lower` |
| `leg_curl` | `squat` | 0.340 | 0.035 | `hamstring_isolation` |
| `leg_extension` | `squat` | 0.420 | 0.046 | `quad_isolation` |

### Gotowe priory dla cwiczen zakotwiczonych do deadlift

| exercise_id | anchor_lift | ratio_mean | ratio_sd | exercise_family |
|---|---|---:|---:|---|
| `deadlift` | `deadlift` | 1.000 | 0.000 | `hinge_primary` |
| `sumo_deadlift` | `deadlift` | 0.975 | 0.050 | `hinge_variant` |
| `rdl` | `deadlift` | 0.700 | 0.029 | `hinge_variant` |

### Cwiczenia bodyweight / trunk / carry

Tu najbardziej poprawny naukowo wariant nie powinien byc na sile podpiety tylko pod 3 boje.
Dla tych ruchow lepiej miec dodatkowy anchor:
- `bodyweight`

Jesli apka zbiera mase ciala, to dla tych cwiczen rekomendowany jest taki prior:

| exercise_id | anchor_lift | ratio_mean | ratio_sd | exercise_family |
|---|---|---:|---:|---|
| `reverse_fly` | `bodyweight` | 0.120 | 0.023 | `rear_delt_isolation` |
| `plank` | `bodyweight` | 0.340 | 0.035 | `core_bracing` |
| `farmers_walk` | `bodyweight` | 0.750 | 0.115 | `carry` |
| `leg_raises` | `bodyweight` | 0.290 | 0.029 | `core_flexion` |
| `ab_wheel` | `bodyweight` | 0.375 | 0.043 | `core_anti_extension` |
| `dead_bug` | `bodyweight` | 0.250 | 0.029 | `core_stability` |
| `trx_bodysaw` | `bodyweight` | 0.290 | 0.029 | `core_anti_extension` |
| `suitcase_carry` | `bodyweight` | 0.440 | 0.081 | `unilateral_carry` |
| `bird_dog` | `bodyweight` | 0.250 | 0.029 | `core_stability` |

Wniosek praktyczny:
- dla cwiczen z tabel bench/squat/deadlift wystarcza 3 glówne boje
- dla bodyweight/core/carry warto miec 4. anchor, czyli `bodyweight`
- jezeli apka jeszcze nie zbiera bodyweight, to warto to dodac, bo dla tych ruchow bedzie to naukowo poprawniejsze niz agresywne wciskanie ich pod 3-boj

Skad wziete te liczby:
- `ratio_mean` sa ustawione jako srodek priors juz zaszytych w generatorze
- `ratio_sd` sa ustawione jako praktyczna niepewnosc startowa wynikajaca z szerokosci tych priors
- dla liftow style-dependent, jak `low_bar_squat`, `high_bar_squat`, `sumo_deadlift`, niepewnosc jest celowo troche szersza

To jest lepsze niz sztywne wpisywanie wszystkiego do raw datasetu, bo:
- 3 boje sa faktycznie obserwowalne od usera
- reszta jest wiedza pochodna
- mozna pozniej zmieniac algorytm lub ratio bez przebudowy calego raw CSV
- `ratio_sd` daje od razu miejsce na confidence-aware update

## Rekomendowana struktura danych do zrzutowania

Najlepiej trzymac osobna tabele priors, np. YAML / JSON / DB table:

- `exercise_id`
- `anchor_lift`
- `ratio_mean`
- `ratio_sd`
- `exercise_family`

Praktyczna uwaga:
- tych wartosci nie trzeba wpisywac do raw datasetu
- lepiej miec je jako osobna warstwe konfiguracyjna dla runtime state i preprocessingu

## Gdzie to przechowywac

Rekomendacja:
- w raw datasecie trzymac tylko 3 startowe `config_1rm_*`
- nie wpisywac na sztywno wszystkich dynamicznych `1RM` per exercise do raw CSV

Najlepsza opcja praktyczna:
- trzymac dynamiczne `1RM` per exercise jako runtime state usera poza raw datasetem

Czyli backend / apka trzyma dla usera:
- `config_1rm_bench_press`
- `config_1rm_squat`
- `config_1rm_deadlift`
- `estimated_1rm_per_exercise`
- `confidence_per_exercise`
- `n_observations`
- `last_update_at`

Dlaczego to jest najlepsze:
- latwiej uniknac leakage
- latwiej debugowac
- latwiej zmieniac algorytm update
- surowy dataset zostaje prostszy

## Czy model ma to sam aproksymowac

Sa 3 opcje:

### Opcja A: aproksymacja i update poza modelem

Model dostaje juz gotowy sygnal sily usera.

Plusy:
- najprostsze
- najbardziej kontrolowalne
- najmniejsze ryzyko leakage

Minus:
- trzeba utrzymywac osobny stan poza modelem

### Opcja B: sekwencyjny preprocessing

Przed treningiem modelu liczysz dla kazdego seta:
- `estimated_1rm_before_set`

I to trafia do modelu jako feature.

Plus:
- model widzi dynamiczna estymacje sily

Minus:
- trzeba bardzo pilnowac kauzalnosci

### Opcja C: model sam ma ukryty stan i sam dochodzi do estymaty sily

Najbardziej end-to-end.

Minusy:
- najtrudniejsze do stabilnego nauczenia
- najmniej interpretowalne
- najtrudniejsze do debugowania

Rekomendacja:
- v1: Opcja A
- potem ewentualnie v2: Opcja B

## Na czym opierac aktualizacje 1RM

Tak, glownie na `RIR`.

To jest najbardziej naturalne, bo:
- cel biznesowy tej zmiany to lepszy `RIR per person`
- najlepszy praktyczny sygnal bez velocity to:
  - `weight`
  - `reps`
  - `reported RIR`
  - `exercise`
  - `sex`

Wtedy mozna liczyc:
- `reps_to_failure = reps_done + RIR`
- z tego odczytywac oczekiwany `%1RM`
- i z tego liczyc kandydat na `e1RM`

To jest najbardziej spójne z literatura RIR-based / RTF-based profiling.

## Scientific metody aktualizacji 1RM

### 1. Single-set RIR-based e1RM

Dla dobrego seta:
- `RTF = reps + RIR`
- z lookup / curve bierzesz `%1RM` odpowiadajacy temu `RTF`
- liczysz:
  - `e1RM_candidate = weight / pct_1rm(RTF, sex, exercise_family)`

To jest podstawowy klocek do dalszego update'u.

### 2. Weighted moving average / weighted median

Z kilku ostatnich sensownych setow liczysz kandydatow `e1RM_candidate` i agregujesz je.

Wagi powinny preferowac sety:
- blisko upadku
- z umiarkowanym zakresem reps
- bez oczywistego bycia warm-upem
- z cwiczenia, dla ktorego ratio/prior jest wiarygodne

To jest najlepszy praktyczny wariant v1.

### 3. Bayesian update / Kalman-like filter

Stan ukryty:
- `true exercise 1RM`

Obserwacja:
- `e1RM_candidate` z seta

Nowy estimate:
- kompromis miedzy starym stanem i nowa obserwacja
- sila update zalezy od niepewnosci

To jest najbardziej "scientific" statystycznie, ale trudniejsze implementacyjnie.

## Kiedy aktualizowac

Nie warto robic twardego update'u po kazdym secie bez filtra, bo:
- `RIR` jest szumny
- lekkie serie daja slaby sygnal
- warm-upy i bardzo wysokie `RIR` malo mowia o realnym `1RM`

Sa 2 sensowne strategie:

### Strategia 1: event-triggered update

Update tylko, gdy pojawi sie dobry sygnal.

Warunki sensownego triggera:
- `RIR <= 3`
- brak warm-upu
- reps w sensownym zakresie dla danej rodziny cwiczen
- brak oczywistego outliera

To jest dobra opcja, jesli chcesz szybka reakcje na mocny set.

### Strategia 2: rolling-window update

Zbierasz ostatnie `N` sensownych setow / `2-4` ekspozycji na cwiczenie i dopiero wtedy robisz update.

To jest zwykle stabilniejsze.

Rekomendacja:
- produkcyjnie bardziej bezpieczny jest rolling-window update
- ewentualnie event-trigger mozna dodac dla bardzo dobrych, ciezkich setow

## Jak czesto aktualizowac

Najrozsadniej:
- nie po kazdym secie
- raczej po sesji albo po kilku ekspozycjach na dane cwiczenie

Propozycja v1:
- aktualizacja po kazdej sesji z danym cwiczeniem
- ale tylko jezeli w tej sesji byly high-quality sety

Propozycja v2:
- aktualizacja po uzbieraniu minimum `3-6` dobrych setow z ostatnich `2-4` ekspozycji

To powinno byc stabilniejsze niz aktualizacja na pojedynczym secie.

## Proponowany algorytm v1

1. User wpisuje 3 startowe `1RM`.
2. Z 3 bojow liczysz startowe `estimated_1rm_per_exercise` przez `anchor + ratio`.
3. Po kazdym secie liczysz `e1RM_candidate`, ale tylko dla setow dobrej jakosci.
4. Na koncu sesji albo po kilku ekspozycjach agregujesz kandydatow.
5. Aktualizujesz `estimated_1rm_per_exercise` z lekkim smoothingiem:
   - `new = (1 - alpha) * old + alpha * observed`
6. `alpha` zalezy od confidence i liczby dobrych obserwacji.
7. Tego stanu uzywasz potem do bardziej personalnego liczenia `RIR`.

## Czego nie robic

- nie wpisywac do raw datasetu dynamicznie zaktualizowanych `1RM` bez semantyki `before_set`
- nie liczyc feature dla seta z uzyciem informacji z tego samego seta lub przyszlosci
- nie opierac update'u glownie na lekkich setach z `RIR 4-5`
- nie robic brutalnego `1RM = last_observation`

## Rekomendacja koncowa

Najlepszy plan na teraz:
- raw dataset trzyma tylko 3 startowe `config_1rm_*`
- zrzutowanie na reszte cwiczen robimy poza raw datasetem przez `anchor + ratio`
- dynamiczne `1RM` trzymamy jako runtime state usera
- update opieramy glownie na `weight + reps + reported RIR`
- update robimy po sesji albo po kilku ekspozycjach, nie bezwarunkowo po kazdym secie
- pierwszy praktyczny wariant: weighted moving average / weighted median
- wariant bardziej researchowy: Bayesian / Kalman-like update
