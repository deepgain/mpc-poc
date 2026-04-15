# Exercise Selection Algorithm — Milosz

## O co chodzi

Silnik rekomendacyjny dla treningu. Używa modelu DeepGain (z `models/`) jako **black-box** do predykcji MPC i symulacji skutków serii. Na podstawie aktualnego stanu mięśni użytkownika i preferencji czasowych dobiera listę ćwiczeń, która:

- wykorzysta dostępną "energię" mięśni (wysoki MPC = priorytet),
- doprowadzi zaangażowane mięśnie do zakresu **target fatigue zone** (wystarczający bodziec, nie przetrenowanie),
- respektuje podział na ćwiczenia czasochłonne vs mniej czasochłonne zadany przez usera,
- **replanuje w locie** kiedy user odrzuci ćwiczenie lub zmodyfikuje serie/powtórzenia.

## Interfejs

**Input:**
- `state: dict[muscle_id, MPC]` — aktualny MPC per mięsień (z `inference.predict_mpc(history, now)`)
- `n_compound: int` — ile ćwiczeń czasochłonnych (A. Główne + B. Wariacje)
- `n_isolation: int` — ile mniej czasochłonnych (C. Akcesoria, izolacje)
- `exclusions: list[exercise_id]` — co user nie chce / nie może
- `preferences: dict` — opcjonalne: ulubione ćwiczenia, historia unikania

**Output:**
- `plan: list[PlannedSet]` z `(exercise_id, weight_kg, reps, rir, order)`
- `predicted_mpc_after: dict[muscle_id, MPC]` — prognoza stanu po sesji

**Replanning:**
- `replan(session_so_far, remaining_n_compound, remaining_n_isolation, new_state) -> plan`

## Taski

1. **Target fatigue zone** — przedział MPC per mięsień po sesji (np. [0.2, 0.5]). Konsultować z Aleksandrem (τ z literatury, mięśnie wolno regenerujące → mniej agresywny target).
2. **Funkcja celu** — suma `involvement_rank_score × MPC_before` (priorytet świeżych mięśni zaangażowanych wysoko w rankingu ćwiczenia) minus penalty za mięśnie wypadające poza target zone.
3. **Greedy / beam search** — w każdym kroku symuluj kandydatów (wywołanie `predict_mpc` po dodaniu kandydata), wybierz ten maksymalizujący cel. Compound przed isolation. Core na końcu.
4. **Replanning on-the-fly** — po każdej wykonanej/odrzuconej/zmodyfikowanej serii aktualizuj state, przeplanuj pozostałe sloty.
5. **Guardrails** — (a) bez powtórzeń tego samego ćwiczenia, (b) volume limits per mięsień, (c) core/stabilizacja na końcu, (d) kolejność typowa dla siłowni (duże grupy przed małymi w ramach compound).
6. **Testy symulacyjne** — syntetyczni userzy (różne MPC startowe, różne historie), uruchomić planowanie, zweryfikować: czy każde ćwiczenie z Milestone 1 może się pojawić w jakimś scenariuszu, czy po wykonaniu planu MPC trafia w target zone.

## Deliverables

- `planner.py` — klasa `WorkoutPlanner` z `plan()` i `replan()`
- `planner_tests.py` — testy symulacyjne
- `planner_report.md` — wykresy MPC przed/po sesji, przykładowe plany dla różnych profili

## Współpraca z `models/` (Michal)

- **Stabilne API inference od początku** — nawet jeśli model jest słaby, kontrakt wywołań nie może się zmieniać. Milosz może stubować modelem zastępczym (np. prosta heurystyka drop = f(reps, rir)) żeby nie być zablokowanym, ale integracja z prawdziwym modelem musi być bezbolesna.
- Po każdym re-treningu Michala — Milosz przepina model, robi sanity run plannera, raportuje anomalie (np. "po nowym modelu planner ciągle wybiera leg curl" → może τ dla hamstrings zbyt długie).

## Współpraca z `dataset/` (Aleksander)

- Target fatigue zones i volume limits — konsultacja z literaturą którą Aleksander zbiera.
- Lista ćwiczeń, ich `id`, nazwy i18n, podział compound/isolation — używać z `exercise_muscle_order.yaml`. Nie duplikować, nie trzymać własnej kopii.
