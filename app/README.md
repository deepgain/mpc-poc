# App — Milestone 3

## O co chodzi

Aplikacja mobilna / klient UI dla DeepGain. **Milestone 3** — następny etap po Milestone 2. W Milestone 2 powstają trzy klocki (dataset, model, planner); w Milestone 3 składamy je w produkt dla końcowego użytkownika.

## Zakres

- **Onboarding** — user wpisuje historię treningową (ostatnie 1–2 tyg.) albo zaczyna od zera (MPC = 1.0 wszędzie).
- **Logowanie serii** — weight, reps, RIR, timestamp. Schema identyczny jak `training_data.csv` (`exercise_id` z `exercise_muscle_order.yaml`).
- **Widok "stan mięśni"** — 16 mięśni z wizualizacją MPC (avatar ciała z kolorami, lista, wykresy recovery). Aktualizacja co klik.
- **Planowanie treningu** — user wpisuje "chcę zrobić N compound + M isolation" → planner zwraca listę → user akceptuje / odrzuca / modyfikuje pojedyncze ćwiczenia → replanning.
- **W trakcie treningu** — checklista serii, szybkie logowanie, aktualizacja MPC na bieżąco.
- **Historia** — przeszłe sesje, progresja obciążeń, wykresy MPC w czasie (jak `chart_mpc_trajectories.png`).

## Wymagania z Milestone 2

- **i18n** — używać `name_en` / `name_pl` z `exercise_muscle_order.yaml`. Wszystkie stringi wyświetlane w aplikacji pochodzą stamtąd, nie są hardkodowane w UI.
- **Inference** — wrapper na `inference.py` (Michal) albo reimplementacja w target stacku (jeśli mobile bez Pythona — TorchScript/ONNX/CoreML).
- **Planner** — wrapper na `planner.py` (Milosz). API `plan()` i `replan()`.

## Dlaczego katalog jest prawie pusty

Milestone 2 to **backend**: dane + model + algorytm. Appka przychodzi po. Ten katalog istnieje teraz jako **placeholder** żeby zespół widział gdzie Milestone 3 wyląduje i żeby decyzje w Milestone 2 (nazwy, schema, API) były podejmowane z perspektywą appki w głowie.

## Taski (TBD — Milestone 3)

Do rozpisania po konsultacji z fizjoterapeutą i decyzji o stack (Flutter vs native vs web). Wstępnie:
- Wybór stacku i architektury
- Figma / wireframes
- Integracja modelu (on-device inference vs backend API)
- Onboarding flow
- Widok stanu mięśni
- Flow planowania + replanning
- Logowanie serii
- Historia i analytics
- Testy z użytkownikami
