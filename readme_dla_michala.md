# Handoff A -> B (Michal)

Cel: szybki, praktyczny handoff co zostalo zrobione po stronie A i co dalej robic po stronie B.

## 1) Co zostalo zmodyfikowane po stronie A

1. Zweryfikowane i uporzadkowane EMG/CSV dla pull_up.
- Finalnie spojne z kierunkiem Youdas 2010 + Dickie 2016.
- Sprawnosc mapowania YAML <-> CSV sprawdzona (brak mismatchy operacyjnych).

2. Audyt roznic branchowych (A vs branch Michala).
- eval_ordering.py: brak roznic merytorycznych.
- train.py: roznice merytoryczne (hyperparametry, zrodlo CSV, fragmenty chartow).

3. Zmiany w generatorze danych (wprowadzone w kodzie).
- Dodany stratified initial val split po profilu usera:
  - training_level
  - sex
  - split_preference
- Dodane template'y kalibracyjne dla bardziej zroznicowanego kontekstu:
  - spoto_calibration_day
  - machine_chest_calibration_day
- Dodane kontrolowane podmiany template'ow, zeby Spoto/Machine nie byly prawie zawsze w tym samym kontekscie push.
- Profil usera przekazywany do splitu i dalej dziala sequence-repair.

4. Wygenerowany nowy dataset referencyjny 303 users.
- Folder aktywny:
  - generated_datasets/u303_stratified_calib/
- Kluczowe liczby:
  - users: 303 (train 242 / val 61)
  - total rows: 1,435,893
  - unique exercises: 34
  - split sequence audit: OK (brak brakow dla same-session ordered pairs >=2 users)

## 2) Co zostalo zarchiwizowane

Utworzony folder:
- archiveDataSets/legacy_pre_u303/

Przeniesione starsze artefakty (legacy):
- training_data_full_generated.csv
- training_data_train.csv
- training_data_val.csv
- root_dataset_report.md (snapshot rootowego dataset_report)
- baseline_main_dataset_report.md (z generated_datasets/baseline_main)

Zasada:
- Nowe datasety, ktore teraz generujemy/splitujemy, zostaja jako aktywne i nie sa wrzucane do archiveDataSets.

## 3) Kierunek pracy dla B (priorytety)

1. Trening M8 uruchamiac na aktywnym datasecie:
- generated_datasets/u303_stratified_calib/training_data_train.csv
- generated_datasets/u303_stratified_calib/training_data_val.csv

2. Najpierw sprawdzic czy stabilizuja sie regresy:
- spoto_press ordering
- chest_press_machine ordering

3. W eval raportowac nie tylko single probe, ale probe-grid (mean/min), zeby odseparowac lokalny przypadek od stabilnosci globalnej.

4. W train rozwazyc loss weighting per exercise (1/sqrt(freq), normalize do mean=1), zeby ograniczyc dominance najczestszych cwiczen.

5. Trzymac sequence coverage jako gate quality:
- key pairs po obu stronach splitu (train i val)
- brak one-sided holes dla par z >=2 users

6. Jezeli dalej beda regresy Spoto/Machine:
- lekko podniesc udzial calibration templates,
- ale bez psucia powerlifting-first rozkladu glownych template'ow.

## 4) Szybka checklista B (na start)

1. Potwierdz sciezki train/val na u303_stratified_calib.
2. Odpal trening M8 (ta sama metoda jak M7, bez mieszania wielu zmian naraz).
3. Porownaj M8 vs M7:
- global RMSE/MAE/corr,
- per-exercise MAE,
- ordering per exercise (szczegolnie Spoto/Machine).
4. Jezeli M8 jest stabilniejsze, ten dataset traktowac jako nowy baseline operacyjny.

## 5) Notatka o rozmiarach datasetow

Patrz porownanie w rozmowie A/B (ponizej wyslane tez liczby):
- aktywny u303 split jest wyraznie wiekszy od starego splitu train+val,
- ale mniejszy od sumy calego archiwum (bo archiwum zawiera dodatkowo plik full).
