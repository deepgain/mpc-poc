# knapsack_planner — dokumentacja dla frontendu

## Co robi ten moduł?

Generuje spersonalizowany plan treningowy dla użytkownika. Plan zawsze ma tę samą strukturę:

```
ćwiczenie główne  (jedno z trójboju: squat / bench press / deadlift)
────────────────────────────────────────────────────
ćwiczenia akcesoryjne  (variation, isolation, max 2 core)
```

Wybór ćwiczeń i dobór ciężarów jest oparty na modelu AI (DeepGain), który uwzględnia:
- aktualne zmęczenie mięśni użytkownika (MPC)
- osobiste maksima siłowe użytkownika (1RM bench/squat/deadlift)
- dostępny czas sesji

Frontend **nie oblicza niczego** — tylko wysyła parametry i wyświetla gotowy plan.

---

## Słowniczek

| Pojęcie | Co to jest |
|---|---|
| **MPC** | Muscle Performance Capacity — jak wypoczęty jest mięsień. `1.0` = w pełni zregenerowany, `0.1` = wyczerpany |
| **RIR** | Reps in Reserve — ile powtórzeń do upadku mięśniowego zostało. `0` = upadek, `3` = 3 powtórzenia zapasu |
| **1RM** | One Rep Max — maksymalny ciężar na jedno powtórzenie. Użytkownik podaje przy onboardingu |
| **strength_anchors** | Trójka maksimów `{bench_press, squat, deadlift}` — stan użytkownika trzymany w bazie |
| **exercise_id** | Identyfikator ćwiczenia (string), np. `"bench_press"`, `"lat_pulldown"` |
| **muscle_id** | Identyfikator mięśnia (string), np. `"chest"`, `"quads"` |

---

## Endpointy

### 1. Onboarding — zapisz 1RM użytkownika

Wywołaj raz przy rejestracji (lub gdy user aktualizuje swoje maksima).

```
POST /api/user/onboarding
```

```json
{
  "bench_press_1rm": 100.0,
  "squat_1rm":       140.0,
  "deadlift_1rm":    180.0
}
```

Odpowiedź:
```json
{ "ok": true }
```

---

### 2. Generowanie planu treningowego

Wywołaj na początku każdej sesji.

```
POST /api/plan
```

**Request:**

```json
{
  "time_budget_sec": 3600,
  "target_rir": 3,
  "exclusions": [],
  "now": "2026-04-28T10:00:00"
}
```

| Pole | Typ | Domyślna | Kiedy ustawić |
|---|---|---|---|
| `time_budget_sec` | `number` | `3600` | User wybiera długość sesji (np. 30/45/60/90 min → mnożnik ×60) |
| `target_rir` | `number` | `3` | Intensywność. `1`–`2` = ciężko, `3` = umiarkowanie, `4`–`5` = lekko |
| `exclusions` | `string[]` | `[]` | Lista `exercise_id` do pominięcia — np. gdy user ma kontuzję lub nie ma sprzętu |
| `now` | `string` (ISO 8601) | teraz | Timestamp startu sesji — ważny do obliczenia MPC. Użyj `new Date().toISOString()` |

**Response:**

```json
{
  "blocks": [
    {
      "exercise_id":     "bench_press",
      "weight_kg":       77.5,
      "reps":            10,
      "sets_count":      3,
      "predicted_rir":   3.1,
      "time_cost_sec":   570,
      "ex_type":         "compound",
      "primary_muscles":   ["chest", "triceps"],
      "secondary_muscles": ["anterior_delts"]
    },
    {
      "exercise_id":     "lat_pulldown",
      "weight_kg":       37.5,
      "reps":            10,
      "sets_count":      3,
      "predicted_rir":   2.9,
      "time_cost_sec":   435,
      "ex_type":         "isolation",
      "primary_muscles":   ["lats"],
      "secondary_muscles": ["biceps", "rear_delts"]
    }
  ],
  "total_time_sec": 3240,
  "mpc_before": {
    "chest": 1.0, "quads": 0.82, "hamstrings": 1.0, "lats": 1.0,
    "triceps": 1.0, "biceps": 1.0, "anterior_delts": 1.0,
    "lateral_delts": 1.0, "rear_delts": 1.0, "rhomboids": 1.0,
    "glutes": 0.89, "adductors": 1.0, "erectors": 0.91,
    "calves": 1.0, "abs": 1.0
  },
  "mpc_after": {
    "chest": 0.68, "quads": 0.82, "hamstrings": 1.0, "lats": 0.71,
    "triceps": 0.72, "biceps": 0.74, "anterior_delts": 0.79,
    "lateral_delts": 1.0, "rear_delts": 0.88, "rhomboids": 0.91,
    "glutes": 0.89, "adductors": 1.0, "erectors": 0.91,
    "calves": 1.0, "abs": 1.0
  },
  "constraint_violations": []
}
```

**Opis pól response:**

| Pole | Typ | Opis |
|---|---|---|
| `blocks` | `Block[]` | Lista ćwiczeń w kolejności wykonania. `blocks[0]` to zawsze główne ćwiczenie z trójboju |
| `blocks[n].exercise_id` | `string` | ID ćwiczenia — patrz lista poniżej |
| `blocks[n].weight_kg` | `number` | Ciężar dobrany przez model [kg] |
| `blocks[n].reps` | `number` | Liczba powtórzeń (zawsze `10`) |
| `blocks[n].sets_count` | `number` | Liczba serii (3 dla compound/isolation, 2 dla core) |
| `blocks[n].predicted_rir` | `number` | Przewidywane RIR przy tym ciężarze |
| `blocks[n].time_cost_sec` | `number` | Szacowany czas całego bloku (serie + odpoczynki) [s] |
| `blocks[n].ex_type` | `string` | `"compound"` / `"variation"` / `"isolation"` / `"core"` |
| `blocks[n].primary_muscles` | `string[]` | Główne mięśnie (engagement ≥ 40%) |
| `blocks[n].secondary_muscles` | `string[]` | Wtórne mięśnie (engagement 20–40%) |
| `total_time_sec` | `number` | Łączny szacowany czas sesji [s] |
| `mpc_before` | `object` | MPC każdego mięśnia przed treningiem (wypoczęcie na starcie) |
| `mpc_after` | `object` | MPC każdego mięśnia po treningu (symulacja) |
| `constraint_violations` | `string[]` | Ostrzeżenia, gdy mięsień wyjdzie poza target zone. Pusta lista = OK |

---

### 3. Zakończenie sesji

Wywołaj po zakończeniu treningu. Backend aktualizuje 1RM i zapisuje historię.

```
POST /api/session/complete
```

```json
{
  "completed_sets": [
    {
      "exercise": "bench_press",
      "weight_kg": 77.5,
      "reps": 10,
      "rir": 3,
      "timestamp": "2026-04-28T10:05:00"
    },
    {
      "exercise": "bench_press",
      "weight_kg": 77.5,
      "reps": 10,
      "rir": 2,
      "timestamp": "2026-04-28T10:08:00"
    }
  ]
}
```

Każda **seria** (set) to osobny wpis. Jeśli ćwiczenie ma `sets_count: 3`, wysyłasz 3 wpisy.

Odpowiedź:
```json
{ "ok": true }
```

---

## Stan po stronie frontendu

Frontend trzyma tylko to co dostał z API i feedback od usera:

```typescript
interface SessionState {
  plan: Block[];               // z /api/plan
  currentBlockIndex: number;  // które ćwiczenie teraz
  currentSetIndex: number;    // która seria teraz
  completedSets: CompletedSet[]; // do wysłania na /api/session/complete
}

interface Block {
  exercise_id:     string;
  weight_kg:       number;
  reps:            number;
  sets_count:      number;
  predicted_rir:   number;
  time_cost_sec:   number;
  ex_type:         "compound" | "variation" | "isolation" | "core";
  primary_muscles:   string[];
  secondary_muscles: string[];
}

interface CompletedSet {
  exercise:  string;   // exercise_id
  weight_kg: number;   // może być zmieniony przez usera
  reps:      number;   // może być zmieniony przez usera
  rir:       number;   // user podaje po serii
  timestamp: string;   // ISO 8601, moment wykonania seta
}
```

---

## Flow sesji

```
User ustawia długość sesji
        │
        ▼
POST /api/plan { time_budget_sec, target_rir, exclusions, now }
        │
        ▼
Wyświetl plan (blocks[])
  blocks[0] = główne ćwiczenie z trójboju (zawsze pierwsze)
  blocks[1..] = akcesoryjne
        │
        ▼
User wykonuje serię
  → jeśli zmienił ciężar: zapisz lokalnie, nie wywołuj API
  → po serii: user podaje RIR → completedSets.push(...)
  → jeśli pomija ćwiczenie: po prostu przejdź do następnego
        │
        ▼
Kolejna seria / kolejne ćwiczenie
        │
        ▼
Koniec sesji
        │
        ▼
POST /api/session/complete { completed_sets: completedSets }
```

---

## Wyświetlanie MPC (opcjonalne)

MPC to "wypoczęcie" mięśnia w skali 0.1–1.0. Możesz to pokazać userowi jako wskaźnik zmęczenia.

```typescript
// Zmęczenie w procentach (0% = świeży, 90% = wyczerpany)
const fatigue = (mpc: number) => Math.round((1 - mpc) * 100);

// Kolor (czy mięsień był wystarczająco obciążony)
const muscleStatus = (mpc_before: number, mpc_after: number) => {
  const delta = mpc_before - mpc_after;
  if (delta < 0.02) return "not_trained";  // mięsień nie był używany
  if (mpc_after < 0.40) return "overworked"; // za duże zmęczenie
  return "trained_ok";
};
```

Mięśnie zwracane w `mpc_before` / `mpc_after`:

| muscle_id | Mięsień PL |
|---|---|
| `chest` | Klatka piersiowa |
| `quads` | Czworogłowy (udo) |
| `hamstrings` | Dwugłowy uda |
| `glutes` | Pośladki |
| `lats` | Najszerszy grzbietu |
| `erectors` | Prostownik kręgosłupa |
| `anterior_delts` | Bark przedni |
| `lateral_delts` | Bark boczny |
| `rear_delts` | Bark tylny |
| `rhomboids` | Równoległoboczny (środek grzbietu) |
| `triceps` | Trójgłowy ramienia |
| `biceps` | Dwugłowy ramienia |
| `adductors` | Przywodziciele |
| `calves` | Łydki |
| `abs` | Brzuch |

---

## Lista wszystkich exercise_id

Ćwiczenia które może zwrócić `blocks[n].exercise_id`:

### Główne (trójbój) — zawsze `blocks[0]`
| exercise_id | Ćwiczenie PL |
|---|---|
| `bench_press` | Wyciskanie sztangi na płaskiej ławce |
| `squat` | Przysiad ze sztangą (high bar) |
| `low_bar_squat` | Przysiad low bar |
| `high_bar_squat` | Przysiad high bar |
| `deadlift` | Martwy ciąg |
| `sumo_deadlift` | Martwy ciąg sumo |

### Akcesoryjne — `blocks[1..]`
| exercise_id | Ćwiczenie PL | Typ |
|---|---|---|
| `incline_bench` | Wyciskanie na skosie (30°) | variation |
| `incline_bench_45` | Wyciskanie na skosie (45°) | variation |
| `close_grip_bench` | Wyciskanie wąskim chwytem | variation |
| `spoto_press` | Spoto press | variation |
| `decline_bench` | Wyciskanie na ujemnym skosie | variation |
| `ohp` | Wyciskanie nad głowę (OHP) | variation |
| `pendlay_row` | Wiosłowanie Pendlay | variation |
| `seal_row` | Seal row | variation |
| `rdl` | Romanian deadlift (RDL) | isolation |
| `leg_press` | Prasa nożna | isolation |
| `leg_curl` | Uginanie nóg w leżeniu | isolation |
| `leg_extension` | Wyprostowanie nóg | isolation |
| `bulgarian_split_squat` | Wykrok bułgarski | isolation |
| `chest_press_machine` | Wyciskanie na maszynie | isolation |
| `dips` | Dipy na poręczach | isolation |
| `lat_pulldown` | Ściąganie drążka do klatki | isolation |
| `pull_up` | Podciąganie na drążku | isolation |
| `dumbbell_flyes` | Rozpiętki z hantlami | isolation |
| `skull_crusher` | Łamacze czaszki | isolation |
| `reverse_fly` | Odwrotne rozpiętki | isolation |
| `farmers_walk` | Spacer farmera | core |
| `suitcase_carry` | Suitcase carry | core |
| `ab_wheel` | Wałek do brzucha | core |
| `plank` | Plank | core |
| `leg_raises` | Unoszenie nóg w zwisie | core |
| `dead_bug` | Dead bug | core |
| `bird_dog` | Bird dog | core |
| `trx_bodysaw` | TRX bodysaw | core |

---

## Obsługa błędów

| HTTP | Znaczenie | Co zrobić |
|---|---|---|
| `200` | OK | Wyświetl plan |
| `422` | Błędne dane wejściowe | Sprawdź format `time_budget_sec`, `now` |
| `500` | Błąd modelu | Pokaż komunikat, spróbuj ponownie |

Jeśli `constraint_violations` nie jest pustą listą:
```typescript
if (plan.constraint_violations.length > 0) {
  // Plan jest nadal używalny, ale jakiś mięsień wyszedł poza optymalną strefę
  // Możesz pokazać subtelny komunikat, np. "Plan zoptymalizowany z ograniczeniami"
  // NIE blokuj użytkownika
}
```

---

## Przykładowy request w JavaScript

```javascript
// Generuj plan
const response = await fetch('/api/plan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    time_budget_sec: 3600,          // 60 minut
    target_rir: 3,
    exclusions: [],
    now: new Date().toISOString(),
  }),
});
const plan = await response.json();

// Zakończ sesję
await fetch('/api/session/complete', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    completed_sets: [
      {
        exercise:  plan.blocks[0].exercise_id,
        weight_kg: plan.blocks[0].weight_kg,
        reps:      plan.blocks[0].reps,
        rir:       userProvidedRir,           // user wpisuje po serii
        timestamp: new Date().toISOString(),
      },
      // ... pozostałe serie
    ],
  }),
});
```
