# Milestone 2 — podział zadań

Kontekst: Milestone 1 zdefiniował ~40 ćwiczeń z PDF (A. Główne, B. Wariacje, C. Akcesoria, D. Stabilizacja/Core) + 16 grup mięśni + wagi zmęczeń z literatury. Obecny kod w `train.py` ma ~27 ćwiczeń i współczynniki `EXERCISE_MUSCLES` z liczbowymi proporcjami (0.85, 0.60...). Nowy paradygmat: **zamiast liczbowych wag — lista zaangażowanych mięśni w kolejności malejącej** (ordinal, nie kardynalny).

---

## Osoba A — Dataset (modelowanie + generator)

**Cel:** wygenerować train/val dataset `.csv` (user_id, exercise, weight_kg, reps, rir, timestamp) tak, by każde ćwiczenie z Milestone 1 miało wystarczającą reprezentację do nauczenia dynamiki.

**Taski:**
1. **Rozszerzyć słownik ćwiczeń** z `train.py:49` do pełnej listy z Milestone 1 PDF (str. 3–4): High Bar Squat, Spoto Press, Sumo DL, Bulgarian Split Squat, Leg Press, RDL, Leg Curl, Leg Extension, Dumbbell Press skos/płasko, JM Press, Wyciskanie Francuskie, Pendlay Row, Pull-ups, Lat Pulldown, Face Pulls, Seal Row, Plank, Farmer's Walk, Wznosy nóg, Ab Wheel, Dead Bug, Pallof Press, Suitcase Carry, Bird Dog — ~40 ćwiczeń łącznie.
2. **Przejście z wag liczbowych na ordinal**: dla każdego ćwiczenia zdefiniować uporządkowaną listę mięśni (primary → secondary → tertiary) bazując na źródłach z arkusza Milestone 1 (kolumna "Źródło (Autor Rok)"). Zapisać jako `exercise_muscle_order.yaml` lub analogiczny plik — jedno źródło prawdy dla całego zespołu.
3. **Zmodyfikować `generate_training_data.py`** by generator zmęczenia korzystał z kolejności (np. primary dostaje największy drop w zakresie [0.7, 1.0], secondary [0.3, 0.6], tertiary [0.05, 0.2]) z losowaniem w paśmie — nie sztywne liczby.
4. **Coverage audit**: wygenerować raport pokrycia — ile setów per ćwiczenie, per (ćwiczenie × mięsień), per użytkownik, rozkład RIR, tygodniowa objętość. Każde ćwiczenie z Milestone 1 musi mieć ≥ N setów w train i val.
5. **Split train/val** na poziomie użytkowników (nie setów) — minimum 20% użytkowników w val, brak wycieku.
6. **Dokumentacja** — jakie modele matematyczne użyte (τ per mięsień z literatury, recovery exp, fatigue drop), seedowanie, jak odtworzyć dataset.

**Deliverable:** `training_data_train.csv`, `training_data_val.csv`, `exercise_muscle_order.yaml`, `dataset_report.md`. **Deadline:** ~28.03 (by osoba B zdążyła trenować).

---

## Osoba B — Trening modelu

**Cel:** wytrenować sieć `f_{e,m}`, `g_e`, `r_m` (obecna architektura z `train.py`) tak, by predykcje RIR były dokładne i krzywe recovery/fatigue były interpretowalne.

**Taski:**
1. **Adaptacja loss-u do ordinal involvement**: obecny `L_f-order` w `train.py` używa `I_a > I_b` z konkretnych liczb. Przepisać na ranking pairwise: dla każdego ćwiczenia dla każdej pary (a, b) gdzie `rank(a) < rank(b)` wymuszać `f_a > f_b + margin`. Margin w jednostkach MPC drop.
2. **Baseline run** na nowym datasecie (pełne ~40 ćwiczeń) — zapisać metryki: val RIR MAE, val MPC trajectory smoothness, learned τ per mięsień vs literatura.
3. **Regularyzacja τ**: ograniczyć learned τ do zakresu fizjologicznego (8–72h) — clamp lub prior — żeby model się nie rozlatywał na ćwiczeniach z małą liczbą próbek (core, izolacje).
4. **Ablacje**: (a) z/bez fatigue ordering penalty, (b) różne embedding dim, (c) z/bez recovery path-consistency. Raport w `train_ablations.md`.
5. **Eksport modelu** + skrypt inference (`predict_mpc(user_history, timestamp) -> dict[muscle, mpc]` oraz `predict_rir(state, exercise, weight, reps) -> float`) — to jest API dla osoby C.
6. **Walidacja interpretacyjna**: wygenerować te same wykresy co w Milestone 1 PDF (str. 18, 21, 24 — Per-Muscle Breakdown, Cross-Exercise Fatigue Transfer) na nowych ~40 ćwiczeniach i sanity-checkować czy kolejność zgadza się z `exercise_muscle_order.yaml`.

**Deliverable:** `deepgain_model_final.pt`, `inference.py` z API, `chart_*.png` z walidacji, `train_report.md`. **Deadline:** 13.04.

**Koordynacja A↔B:** wspólny plik `exercise_muscle_order.yaml` + cotygodniowy sync — B wcześnie testuje pipeline na prototypowym datasecie A, nie czeka do końca.

---

## Osoba C — Algorytm planowania treningu

**Cel:** używając modelu z osoby B jako black-box, zaplanować sesję treningową o zadanej strukturze czasowej, która wykorzysta dostępną "energię" mięśni.

**Input algorytmu:**
- `state`: MPC per mięsień (z inference osoby B, z historii użytkownika)
- `n_compound`: ile ćwiczeń czasochłonnych (A. Główne + B. Wariacje)
- `n_isolation`: ile mniej czasochłonnych (C. Akcesoria, izolacje)
- (opcjonalnie) lista wykluczeń / preferencji użytkownika

**Output:** uporządkowana lista ćwiczeń z sugerowanymi seriami × powtórzeniami × RIR.

**Taski:**
1. **Definicja "target fatigue zone"** per mięsień — np. po sesji MPC powinno spaść do przedziału [0.2, 0.5] (wystarczające bodziec, nie przetrenowanie). Skonsultować zakresy z osobą A (bazując na literaturze recovery τ).
2. **Funkcja celu**: maksymalizuj sumę `involvement_rank_score` dla mięśni o najwyższym aktualnym MPC (priorytet "świeżych") minus penalty za mięśnie wypadające poza target zone (za mało lub za mocno zmęczone).
3. **Algorytm greedy / beam search**: w każdym kroku symuluj kandydatów (wywołanie `predict_mpc` po dodaniu ćwiczenia×setów), wybierz ten który maksymalizuje cel. Respektuj podział compound/isolation i typową kolejność treningu (compound najpierw).
4. **Re-planning on-the-fly**: API `replan(session_so_far, remaining_slots, new_state)` — po wykonaniu/odrzuceniu/modyfikacji ćwiczenia użytkownik aktualizuje stan (np. 3 serie zamiast 4), algorytm od nowa planuje resztę sesji z aktualnym MPC.
5. **Guardrails**: (a) nie powtarzać tego samego ćwiczenia w jednej sesji, (b) szanować volume limits per mięsień (max N setów/sesja), (c) uwzględnić że core/stabilizacja na końcu.
6. **Testy symulacyjne**: wygenerować syntetycznych użytkowników (różne MPC początkowe, różne historie), uruchomić planowanie, zweryfikować: czy wszystkie ćwiczenia z Milestone 1 mogą się pojawić w jakimś scenariuszu, czy po wykonaniu planu MPC trafiają w target zone.

**Deliverable:** `planner.py` z klasą `WorkoutPlanner` (metody `plan()`, `replan()`), `planner_tests.py`, `planner_report.md` z wykresami: rozkład MPC przed/po sesji, przykładowe plany dla różnych profili. **Deadline:** 13.04.

**Koordynacja B↔C:** stabilne API inference jak najwcześniej (tydzień 1). Osoba C może początkowo stubować modelem zastępczym, żeby nie być zablokowana.

---

## Timeline (kamienie milowe)

- **do 23.03** — A: lista ćwiczeń + `exercise_muscle_order.yaml` draft; B: inference API stub; C: szkic funkcji celu
- **do 30.03** — A: pierwszy generator + mini-dataset; B: trenuje na mini-dataset; C: MVP plannera na stub modelu
- **do 06.04** — A: pełny dataset + coverage report; B: pełny trening + ablacje; C: replanning + testy
- **13.04** — integracja end-to-end: historia użytkownika → MPC → plan → wykonanie → replan. Demo.
