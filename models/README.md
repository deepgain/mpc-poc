# Models — Michal

## O co chodzi

Trening sieci neuronowej DeepGain — architektury opisanej w Milestone 1 PDF (str. 10–11). Cel: z loggów treningowych (weight, reps, RIR, timestamp) estymować **Muscle Performance Capacity (MPC)** — wartość [0,1] per mięsień, reprezentującą ile "energii" mięsień ma w danej chwili.

## Architektura (trzy komponenty)

1. **f_{e,m}** — sieć neuronowa predykująca drop MPC po serii ćwiczenia `e` dla mięśnia `m`. Input: `(weight, reps, RIR, MPC_current, exercise_embedding, muscle_embedding)`. Output: frakcja dropu (0,1), skalowana przez stały (wyuczony) współczynnik involvement.
2. **g_e** — sieć predykująca RIR dla ćwiczenia `e` przy stanie mięśni `(MPC_m1, MPC_m2, ...)` i parametrach `(weight, reps)`. Supervised target: obserwowane RIR w datasecie.
3. **r_m** — recovery: **nie sieć**, formuła exp decay z wyuczonym τ per mięsień. 16 parametrów τ (jeden per mięsień). Path-consistent by construction.

## Taski

1. **Adaptacja loss-u do ordinal involvement** — obecny `L_f-order` w `train.py` używa sztywnych `I_a > I_b`. Nowy loss: dla każdej pary mięśni (a, b) gdzie `rank(a) < rank(b)` w `exercise_muscle_order.yaml`, wymuszać `f_a > f_b + margin`.
2. **Baseline run** na pełnym datasecie (~40 ćwiczeń) — val RIR MAE, MPC trajectory smoothness, learned τ vs literatura.
3. **Regularyzacja τ** — clamp do zakresu fizjologicznego (8–72h) żeby model się nie rozlatywał na rzadkich ćwiczeniach.
4. **Ablacje** — z/bez fatigue ordering penalty, różne embedding dims, z/bez recovery path-consistency.
5. **Inference API** dla Milosza: `predict_mpc(user_history, timestamp) -> dict[muscle, mpc]` i `predict_rir(state, exercise, weight, reps) -> float`. Stabilny kontrakt od początku.
6. **Walidacja interpretacyjna** — odtworzyć wykresy z Milestone 1 PDF (Per-Muscle Breakdown, Cross-Exercise Fatigue Transfer) na nowych ~40 ćwiczeniach. Sanity-check: kolejność pasuje do `exercise_muscle_order.yaml`.

## Deliverables

- `deepgain_model_final.pt`
- `inference.py` — API dla Milosza
- `train.py` (zmodyfikowany)
- `train_report.md` + `train_ablations.md`
- `chart_*.png` — walidacja interpretacyjna

## Współpraca z `dataset/` (Aleksander) — CIĄGŁA, NIE SEKWENCYJNA

**Nie czekam aż Aleksander skończy dataset i "odda mi" go.** Pętla feedback od samego startu:

1. **Start:** dostaję mini-dataset + draft YAML-a. Odpalam pełen pipeline, szukam problemów schema/formatu, nie problemów modelu.
2. **Gdy widzę anomalie** — np. τ dla calves zbiega do 1h, RIR MAE bardzo wysoki dla konkretnych ćwiczeń, fatigue heatmap pokazuje transfer tam gdzie nie powinno być — **natychmiast raportuję Aleksandrowi** z konkretnym artefaktem (wykres + liczby). To nie jest "twój dataset jest zły", to jest "tu jest sygnał że coś w generatorze może wymagać spojrzenia, bo model uczy się X zamiast Y".
3. **Po każdej regeneracji datasetu** — przetrenowuję, porównuję metryki, wracam z obserwacjami.
4. **Sam też patrzę w dataset** — nie tylko w swoje krzywe lossu. Robię EDA: dystrybucja RIR per ćwiczenie, heatmapa współwystępowań par ćwiczeń w sesji, ile godzin między sesjami per user. Jeśli coś wygląda niefizjologicznie, pytam Aleksandra zanim zacznę oskarżać model.
5. **Wspólne metryki walidacyjne** — uzgadniam z Aleksandrem listę liczb na które oboje patrzymy przy każdej iteracji: `val_rir_mae`, `tau_within_physio_range_count`, `ordering_accuracy_per_exercise`, `fatigue_transfer_makes_sense`. Wspólny dashboard/notebook.

**Praktyczne zasady:**
- `exercise_muscle_order.yaml` zmieniany przez PR, oboje reviewują.
- Stabilny schema CSV od początku — Aleksander nie zmienia kolumn bez zgody.
- Stabilne API inference od początku — Milosz nie zmienia wywołań bez zgody.
- **Blame rozłożony** — jeśli end-to-end nie działa, nie szukamy winnego, szukamy co naprawić. Dataset i model są ze sobą zrośnięte.
