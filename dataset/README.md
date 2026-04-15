# Dataset — Aleksander

## O co chodzi

Generowanie **syntetycznego datasetu** (train + val) do treningu sieci DeepGain. Dataset jest syntetyczny, bo nie mamy dostępu do real-world loggów z EMG/force plates — zastępujemy to **matematycznymi modelami zmęczenia i regeneracji** opartymi na literaturze (τ recovery per mięsień, exp decay, EMG-based involvement ordering).

Jeden wiersz CSV = jedna seria wykonana przez użytkownika:
`user_id, exercise, weight_kg, reps, rir, timestamp`

Realizm datasetu bezpośrednio determinuje jakość nauczonych parametrów modelu — to jest **fundament całego systemu**.

## Kluczowa zmiana vs Milestone 1

W Milestone 1 ćwiczenia miały liczbowe wagi zaangażowania (`chest: 0.85, triceps: 0.55, ...`). W Milestone 2 przechodzimy na **uporządkowaną listę** zaangażowanych mięśni (ordinal, nie kardynalny):
- bench press → `[chest, anterior_delts, triceps]`
- deadlift → `[erectors, glutes, hamstrings, upper_traps, quads, adductors, lats]`
- leg curl → `[hamstrings]`

Długość listy zmienna (1 do 7+). Model uczy się liczbowych współczynników sam — my dostarczamy tylko ranking.

## Taski

1. **Rozszerzyć słownik ćwiczeń** z `train.py:49` do pełnej listy z Milestone 1 PDF (~40 ćwiczeń): High Bar Squat, Spoto Press, Sumo DL, Bulgarian Split Squat, Leg Press, RDL, Leg Curl, Leg Extension, Dumbbell Press skos/płasko, JM Press, Wyciskanie Francuskie, Pendlay Row, Pull-ups, Lat Pulldown, Face Pulls, Seal Row, Plank, Farmer's Walk, Wznosy nóg, Ab Wheel, Dead Bug, Pallof Press, Suitcase Carry, Bird Dog, ...
2. **Ordinal involvement** — dla każdego ćwiczenia uporządkowana lista mięśni (primary → ... → last) o zmiennej długości. Bazować na źródłach z arkusza Milestone 1.
3. **i18n nazw** — każde ćwiczenie i każdy mięsień ma `id` (snake_case), `name_en`, `name_pl`. To są finalne nazwy dla UI — konsultować terminologię siłową (np. "Martwy ciąg", nie "Nieumarły ciąg"). Wszystko w `exercise_muscle_order.yaml`.
4. **Generator zmęczenia oparty o research** — konkretne zakresy dropu MPC per pozycja w rankingu muszą pochodzić z literatury (EMG, hipertrofia, velocity loss). Każdy zakres udokumentowany w `dataset_report.md` z cytowaniem. Tam gdzie literatura milczy — jawny assumption + uzasadnienie fizjologiczne.
5. **Coverage / sanity** — upewnić się że dataset ma sensowne pokrycie: każde ćwiczenie z Milestone 1 się pojawia, RIR ma rozsądny rozkład, mięśnie nie są pominięte. **Forma raportu i dobór metryk w gestii Aleksandra** — ważne żeby Michal mógł łatwo zobaczyć co jest w środku i żeby dało się wyłapać dziury (np. ćwiczenie występujące 3 razy w całym datasecie). Jakiekolwiek narzędzie: notebook, skrypt, markdown, plots — cokolwiek się sprawdzi.
6. **Split train/val** na poziomie użytkowników (brak wycieku). Proporcja do ustalenia eksperymentalnie — tyle, żeby val był statystycznie użyteczny.
7. **Dokumentacja** — modele matematyczne (τ z literatury, recovery exp, fatigue drop), seedy, jak odtworzyć dataset bit-for-bit.

## Artefakty (ewoluujące, nie finalne)

Te pliki będą się zmieniać **na bieżąco** w trakcie Milestone 2 — Michal konsumuje je wielokrotnie, nie raz na końcu:

- `training_data_train.csv` / `training_data_val.csv` — regenerowane po każdej zmianie generatora
- `exercise_muscle_order.yaml` — **jedno źródło prawdy** dla ćwiczeń, mięśni, i18n, rankingów. Zmiany przez PR.
- `generate_training_data.py` — kod generatora
- `dataset_report.md` — snapshot pokrycia i źródeł literaturowych, aktualizowany wraz z datasetem

Nie ma momentu "Aleksander oddał dataset Michałowi". Jest **ciągły strumień iteracji** — patrz sekcja niżej.

## Współpraca z `models/` (Michal) — CIĄGŁA, NIE SEKWENCYJNA

**To nie jest pipeline "zrobię dataset → oddam Michałowi → on trenuje".** To pętla ze stałym feedbackiem:

1. **Start:** Aleksander dostarcza *mini-dataset* (kilku userów, parę tyg. treningu) + draft `exercise_muscle_order.yaml`. Michal odpala na tym pełen training pipeline — nie żeby uzyskać dobry model, ale **żeby złapać problemy formatu/schema wcześnie**.
2. **Michal znajduje artefakty w krzywych τ / RIR MAE / fatigue transfer heatmap** → zgłasza Aleksandrowi: "τ dla calves zbiega do 1h, to niefizjologiczne — podejrzewam że twój generator za rzadko je trenuje / za mocno dropuje".
3. **Aleksander reaguje** — albo poprawia generator (realizm fatigue), albo rozkład częstotliwości ćwiczeń, albo współczynniki długości okien czasowych.
4. **Nowy dataset → nowy trening → nowe obserwacje.** Pętla biegnie często, nie rzadko.
5. **Aleksander również patrzy na wyniki Michala** — jeśli model uczy się ordering dla bench press poprawnie (chest > anterior_delts > triceps), ale dla deadlift myli kolejność glutes vs erectors → to sygnał że w datasecie sygnał dla tego ćwiczenia jest słaby (za mało setów, za mało wariancji RIR, za mało współwystępowania z innymi ćwiczeniami tych samych mięśni).

**Praktyczne zasady współpracy:**
- Wspólny plik `exercise_muscle_order.yaml` w repo — każda zmiana przez PR, oboje reviewują.
- Szybki kanał feedback — nie czekać do syncu.
- **Format danych zamrażamy wcześnie**, treść może ewoluować. Michal nie powinien co chwilę przepisywać data loadera.
- Wspólne metryki walidacyjne: Aleksander wie, na co Michal patrzy (τ w zakresie fizjologicznym, RIR MAE, ordering accuracy per exercise), i optymalizuje dataset pod to.
- **Nie stajemy** w ślepym zaułku "mój dataset jest idealny, twój model jest zły" ani odwrotnie. Jeśli model nie działa — wina jest rozłożona i winni go naprawiają razem.
