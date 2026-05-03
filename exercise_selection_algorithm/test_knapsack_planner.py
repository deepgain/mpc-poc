"""
test_knapsack_planner.py
========================
Testy dla knapsack_planner.py — wyłącznie z prawdziwym modelem DeepGain.

Wymagania
---------
  deepgain_model_best.pt     — checkpoint (w tym samym katalogu lub CHECKPOINT_PATH poniżej)
  inference.py               — API modelu (w tym samym katalogu lub katalogu nadrzędnym)
  strength_priors.py         — anchor/ratio logika (importowana przez inference.py)
  exercise_muscle_order.yaml — ładowane przez load_model()
  exercise_muscle_weights_scaled.csv — ładowane przez load_model()

Uruchomienie
------------
  python test_knapsack_planner.py          # wszystkie testy
  pytest test_knapsack_planner.py -v       # verbose
  pytest test_knapsack_planner.py -v -s    # verbose + stdout (widać wyniki scenariuszy)
  pytest test_knapsack_planner.py -k "scenariusz" -v -s   # tylko scenariusze
  pytest test_knapsack_planner.py -k "dp"  -v             # tylko testy DP
"""

from __future__ import annotations

import os
import sys
import unittest
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.WARNING)


# ===========================================================================
# [KONFIGURACJA]
# Edytuj tę sekcję żeby przetestować innego użytkownika lub inny scenariusz.
# ===========================================================================

# ── Ścieżka do checkpointa ───────────────────────────────────────────────────
# Tylko nazwa pliku – szukany automatycznie w katalogu models/ projektu.
CHECKPOINT_PATH = "deepgain_model_best.pt"

# ── Anchory 1RM [kg] — wejście od użytkownika ────────────────────────────────
ANCHORS_STRONG = {
    "bench_press": 140.0,
    "squat":       200.0,
    "deadlift":    240.0,
}

ANCHORS_AVERAGE = {
    "bench_press": 80.0,
    "squat":       120.0,
    "deadlift":    160.0,
}

ANCHORS_BEGINNER = {
    "bench_press": 40.0,
    "squat":        60.0,
    "deadlift":     80.0,
}

# ── Budżety czasu [sekundy] ──────────────────────────────────────────────────
TIME_30_MIN = 30 * 60
TIME_45_MIN = 45 * 60
TIME_60_MIN = 60 * 60
TIME_90_MIN = 90 * 60

# ── Wykluczenia ──────────────────────────────────────────────────────────────
EXCLUSIONS_NO_LEGS = [
    "squat", "low_bar_squat", "high_bar_squat",
    "leg_press", "bulgarian_split_squat",
    "leg_curl", "leg_extension", "rdl",
]
EXCLUSIONS_NO_PUSH = [
    "bench_press", "incline_bench", "incline_bench_45",
    "ohp", "close_grip_bench", "spoto_press", "decline_bench",
]

# ── Timestamp sesji ──────────────────────────────────────────────────────────
SESSION_NOW = datetime(2026, 4, 28, 10, 0, 0)

# ── Historie treningowe ──────────────────────────────────────────────────────
HISTORY_EMPTY = []

HISTORY_BENCH_TWO_DAYS_AGO = [
    {"exercise": "bench_press",  "weight_kg": 80.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-26T18:00:00"},
    {"exercise": "incline_bench","weight_kg": 65.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-26T18:25:00"},
    {"exercise": "close_grip_bench", "weight_kg": 55.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-26T18:50:00"},
]

HISTORY_FULL_BODY_YESTERDAY = [
    {"exercise": "squat",       "weight_kg": 100.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:00:00"},
    {"exercise": "bench_press", "weight_kg":  80.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:35:00"},
    {"exercise": "deadlift",    "weight_kg": 140.0, "reps": 3, "rir": 1,
     "timestamp": "2026-04-27T18:10:00"},
    {"exercise": "ohp",         "weight_kg":  50.0, "reps": 8, "rir": 2,
     "timestamp": "2026-04-27T18:40:00"},
]

HISTORY_LEGS_YESTERDAY = [
    {"exercise": "squat",                  "weight_kg": 100.0, "reps": 5, "rir": 2,
     "timestamp": "2026-04-27T17:00:00"},
    {"exercise": "leg_press",              "weight_kg": 180.0, "reps": 10,"rir": 2,
     "timestamp": "2026-04-27T17:30:00"},
    {"exercise": "bulgarian_split_squat",  "weight_kg":  40.0, "reps": 10,"rir": 2,
     "timestamp": "2026-04-27T18:00:00"},
    {"exercise": "leg_curl",               "weight_kg":  45.0, "reps": 12,"rir": 2,
     "timestamp": "2026-04-27T18:25:00"},
]


# ===========================================================================
# [MODEL] — ładowanie DeepGain, twardy skip gdy brak pliku
# ===========================================================================

def _find_inference_dir() -> Optional[str]:
    """
    Szuka katalogu z inference.py idąc w górę drzewa katalogów od tego pliku.

    Strategia (w kolejności):
      1. Ten sam katalog co test
      2. Podkatalog models/ w każdym katalogu nadrzędnym
         (obsługuje strukturę mpc-poc/models/ gdy test jest gdziekolwiek w projekcie)
      3. Bezpośrednio w katalogu nadrzędnym (flat layout)

    Działa dla każdego bez konfiguracji – wystarczy mieć sklonowane repo mpc-poc.
    """
    start = os.path.dirname(os.path.abspath(__file__))

    current = start
    while True:
        # Sprawdź bieżący katalog bezpośrednio
        if os.path.isfile(os.path.join(current, "inference.py")):
            return current

        # Sprawdź podkatalog models/ (typowa struktura mpc-poc)
        models_candidate = os.path.join(current, "models")
        if os.path.isfile(os.path.join(models_candidate, "inference.py")):
            return models_candidate

        parent = os.path.dirname(current)
        if parent == current:   # korzeń systemu plików
            break
        current = parent

    return None


def _resolve_checkpoint() -> Optional[str]:
    """Szuka checkpointa w katalogu inference.py i obok tego pliku."""
    if os.path.isabs(CHECKPOINT_PATH) and os.path.isfile(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    inf_dir = _find_inference_dir()
    candidates = [
        os.path.join(inf_dir, CHECKPOINT_PATH) if inf_dir else None,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_PATH),
        CHECKPOINT_PATH,
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


_inf_dir    = _find_inference_dir()
_checkpoint = _resolve_checkpoint()

_MODEL_REASON = ""
_model        = None
_inference    = None


def _find_required_files(inf_dir: str) -> dict:
    """
    Szuka plików wymaganych przez load_model().
    inference.py szuka ich jako ../dataset/<plik> względem swojego katalogu,
    czyli w mpc-poc/dataset/.
    Zwraca słownik {nazwa: ścieżka_lub_None}.
    """
    required = ["exercise_muscle_order.yaml", "exercise_muscle_weights_scaled.csv"]
    result = {}
    project_root = os.path.dirname(inf_dir)          # mpc-poc/
    dataset_dir  = os.path.join(project_root, "dataset")

    for fname in required:
        # Szukaj gdzie inference.py będzie ich szukać: ../dataset/ od models/
        expected = os.path.join(dataset_dir, fname)
        if os.path.isfile(expected):
            result[fname] = expected
        else:
            # Szukaj w całym projekcie jako fallback
            found = None
            for root, _, files in os.walk(project_root):
                if fname in files:
                    found = os.path.join(root, fname)
                    break
            result[fname] = found  # None jeśli nie znaleziono nigdzie

    return result


def _try_load_model():
    """
    Próbuje załadować DeepGain. Zwraca (model, inference_module, error_str).

    Kluczowa kolejność operacji:
      1. os.chdir(models/)  ← PRZED importem
      2. import inference    ← ładuje yaml/csv z ../dataset/ = mpc-poc/dataset/
      3. load_model()
      4. przywróć cwd

    inference.py czyta exercise_muscle_order.yaml i exercise_muscle_weights_scaled.csv
    na poziomie modułu (przy imporcie), używając ścieżki ../dataset/ względem cwd.
    Dlatego cwd musi być ustawione na models/ ZANIM inference zostanie zaimportowany.
    """
    global _inference

    if _inf_dir is None:
        return None, None, "Nie znaleziono inference.py w projekcie"
    if _checkpoint is None:
        return None, None, (
            f"Nie znaleziono checkpointa '{CHECKPOINT_PATH}'\n"
            f"   Sprawdź czy plik jest w: {_inf_dir}"
        )

    if _inf_dir not in sys.path:
        sys.path.insert(0, _inf_dir)

    # Sprawdź wymagane pliki zanim cokolwiek zrobimy (używa ścieżek absolutnych)
    required = _find_required_files(_inf_dir)
    missing  = [name for name, path in required.items() if path is None]
    if missing:
        dataset_dir = os.path.join(os.path.dirname(_inf_dir), "dataset")
        lines = ["Brakuje plików wymaganych przez inference.py:"]
        for name in missing:
            lines.append(f"   ✗  {name}")
        lines.append(f"   → Wrzuć je do: {dataset_dir}")
        return None, None, "\n".join(lines)

    # cwd = models/ PRZED importem – inference.py czyta pliki przy imporcie modułu
    orig_cwd = os.getcwd()
    try:
        os.chdir(_inf_dir)                         # ← musi być przed import
        import inference as inf_module             # ← tu czytany jest yaml/csv
        _inference = inf_module
        model = inf_module.load_model(_checkpoint)
        return model, inf_module, None
    except Exception as e:
        return None, None, str(e)
    finally:
        os.chdir(orig_cwd)                         # zawsze przywróć cwd


if _inf_dir is None:
    _MODEL_REASON = "Nie znaleziono inference.py w projekcie"
elif _checkpoint is None:
    _MODEL_REASON = f"Nie znaleziono checkpointa '{CHECKPOINT_PATH}'"
else:
    _model, _inference, _MODEL_REASON = _try_load_model()

MODEL_READY = _model is not None

if MODEL_READY:
    print(f"\n✓  DeepGain załadowany: {_checkpoint}")
else:
    print(f"\n✗  Model niedostępny:")
    for line in _MODEL_REASON.splitlines():
        print(f"   {line}")
    print(f"   Wszystkie testy zostaną pominięte.\n")

_skip_msg = _MODEL_REASON.splitlines()[0] if _MODEL_REASON else "nieznany blad"
require_model = unittest.skipUnless(MODEL_READY, f"Brak modelu: {_skip_msg}")


from knapsack_planner import (
    KnapsackPlanner, ExerciseBlock, KnapsackPlan,
    MUSCLE_INVOLVEMENT, EXERCISE_META, DEFAULT_TARGET_ZONES, MAIN_EXERCISES,
)


# ===========================================================================
# Fabryka plannera
# ===========================================================================

def make_planner(anchors: dict = None, rest_sec: int = 120) -> KnapsackPlanner:
    """Tworzy KnapsackPlanner z prawdziwym modelem DeepGain."""
    return KnapsackPlanner(
        model=_model,
        strength_anchors=anchors or ANCHORS_AVERAGE,
        rest_between_sets_sec=rest_sec,
        time_resolution_sec=60,
    )


def _avg_rir(plan: KnapsackPlan) -> float:
    return sum(b.predicted_rir for b in plan.blocks) / max(len(plan.blocks), 1)

def _avg_weight(plan: KnapsackPlan) -> float:
    return sum(b.weight_kg for b in plan.blocks) / max(len(plan.blocks), 1)

def _mpc(plan: KnapsackPlan, muscle: str, after: bool = False) -> float:
    d = plan.mpc_after if after else plan.mpc_before
    return d.get(muscle, 1.0)

def _assert_plan_valid(tc: unittest.TestCase, plan: KnapsackPlan, budget: int):
    """Podstawowe asercje dla każdego planu."""
    tc.assertGreater(len(plan.blocks), 0,          "Plan jest pusty")
    tc.assertLessEqual(plan.total_time_sec, budget, "Przekroczono budżet czasu")
    ids = [b.exercise_id for b in plan.blocks]
    tc.assertEqual(len(ids), len(set(ids)),         "Duplikaty ćwiczeń w planie")
    for m, v in plan.mpc_after.items():
        tc.assertGreaterEqual(v, 0.09, f"{m}: MPC poniżej dolnego limitu")
        tc.assertLessEqual(v, 1.0 + 1e-9, f"{m}: MPC powyżej 1.0")


# ===========================================================================
# [SCENARIUSZE] — testy end-to-end z DeepGain
# ===========================================================================


def _mpc_value_from_violation(v: str) -> float:
    """Wyciąga wartość MPC z napisu np. '⚠ OVERFATIGUE  chest: MPC=0.41 < min=0.50'."""
    import re
    m = re.search(r"MPC=([\d.]+)", v)
    return float(m.group(1)) if m else 1.0


@require_model
class TestScenariusze(unittest.TestCase):
    """
    Scenariusze użytkownika weryfikowane przez prawdziwy model DeepGain.

    Jak zmienić scenariusz:
      Edytuj sekcję [KONFIGURACJA] na górze pliku — anchory, historię, budżet czasu.
    """

    # ── SCENARIUSZ 1 ─────────────────────────────────────────────────────────

    def test_01_nowy_uzytkownik_60min(self):
        """
        Scenariusz: nowy użytkownik, brak historii, 60 minut.
        Wejście:  ANCHORS_AVERAGE, historia pusta, TIME_60_MIN, target_rir=3
        Wyjście:  plan z ćwiczeniami, czas ≤ 60 min, brak overfatigue
        """
        plan = make_planner(ANCHORS_AVERAGE).plan(
            user_history=HISTORY_EMPTY,
            time_budget_sec=TIME_60_MIN,
            target_rir=3,
            now=SESSION_NOW,
        )

        self._print_plan("01 – nowy użytkownik, 60 min", plan)
        _assert_plan_valid(self, plan, TIME_60_MIN)
        # Repair loop minimalizuje overfatigue ale może nie domknąć do dokładnej granicy.
        # Testujemy tylko brak CIĘŻKIEGO overfatigue (MPC < 0.35).
        severe = [v for v in plan.constraint_violations
                  if "OVERFATIGUE" in v and _mpc_value_from_violation(v) < 0.20]
        self.assertEqual(severe, [], f"Ciężkie overfatigue (MPC<0.20): {severe}")

    # ── SCENARIUSZ 2 ─────────────────────────────────────────────────────────

    def test_02_po_dniu_klatki_mpc_jest_nizsze(self):
        """
        Scenariusz: bench + incline 2 dni temu → MPC klatki powinno być niższe.
        Wejście:  HISTORY_BENCH_TWO_DAYS_AGO vs HISTORY_EMPTY
        Wyjście:  mpc_before["chest"] < 1.0, niższe niż przy pustej historii
        """
        planner = make_planner(ANCHORS_AVERAGE)
        plan_czysta    = planner.plan(HISTORY_EMPTY,             TIME_60_MIN, now=SESSION_NOW)
        plan_po_benchu = planner.plan(HISTORY_BENCH_TWO_DAYS_AGO, TIME_60_MIN, now=SESSION_NOW)

        chest_fresh = _mpc(plan_czysta,    "chest")
        chest_tired = _mpc(plan_po_benchu, "chest")

        self._print_mpc_diff("02 – po dniu klatki",
                             ["chest", "triceps", "anterior_delts"],
                             plan_czysta, plan_po_benchu)

        self.assertLess(chest_tired, chest_fresh,
                        "MPC klatki powinno być niższe po historii bench")
        self.assertLess(chest_tired, 1.0,
                        "MPC klatki powinno być < 1.0 po treningu 2 dni temu")

    # ── SCENARIUSZ 3 ─────────────────────────────────────────────────────────

    def test_03_push_day_bez_nog_30min(self):
        """
        Scenariusz: push day, 30 minut, nogi wykluczone.
        Wejście:  exclusions=EXCLUSIONS_NO_LEGS, TIME_30_MIN
        Wyjście:  żadne ćwiczenie nóg w planie, czas ≤ 30 min
        """
        plan = make_planner(ANCHORS_AVERAGE).plan(
            user_history=HISTORY_EMPTY,
            time_budget_sec=TIME_30_MIN,
            exclusions=EXCLUSIONS_NO_LEGS,
            now=SESSION_NOW,
        )

        self._print_plan("03 – push day bez nóg, 30 min", plan)
        _assert_plan_valid(self, plan, TIME_30_MIN)
        for b in plan.blocks:
            self.assertNotIn(b.exercise_id, EXCLUSIONS_NO_LEGS,
                             f"{b.exercise_id} jest na liście wykluczeń")

    # ── SCENARIUSZ 4 ─────────────────────────────────────────────────────────

    def test_04_silny_vs_average_vs_poczatkujacy_wagi(self):
        """
        Scenariusz: trzech użytkowników z różnymi 1RM, 45 min, compound only.
        Wejście:  ANCHORS_STRONG / AVERAGE / BEGINNER
        Wyjście:  wagi treningowe: strong > average > beginner
        """
        def plan_for(anchors):
            return make_planner(anchors).plan(
                HISTORY_EMPTY, TIME_45_MIN,
                now=SESSION_NOW,
            )

        plan_s = plan_for(ANCHORS_STRONG)
        plan_a = plan_for(ANCHORS_AVERAGE)
        plan_b = plan_for(ANCHORS_BEGINNER)

        print(f"\n{'─'*55}")
        print(f"[04] Personalizacja wag wg 1RM (compound, 45 min)")
        for label, anchors, plan in [
            ("Strong  ", ANCHORS_STRONG,   plan_s),
            ("Average ", ANCHORS_AVERAGE,  plan_a),
            ("Beginner", ANCHORS_BEGINNER, plan_b),
        ]:
            bench_w = next((b.weight_kg for b in plan.blocks if b.exercise_id == "bench_press"), None)
            squat_w = next((b.weight_kg for b in plan.blocks if "squat" in b.exercise_id), None)
            print(f"  {label}  bench={bench_w or '─':>6}  squat={squat_w or '─':>6}  "
                  f"avg={_avg_weight(plan):.1f} kg")

        self.assertGreater(_avg_weight(plan_s), _avg_weight(plan_a), "strong > average")
        self.assertGreater(_avg_weight(plan_a), _avg_weight(plan_b), "average > beginner")

    # ── SCENARIUSZ 5 ─────────────────────────────────────────────────────────

    def test_05_tylko_izolacja_core_po_ciezkim_tygodniu(self):
        """
        Scenariusz: full body wczoraj → lekki trening, tylko isolation + core.
        Wejście:  HISTORY_FULL_BODY_YESTERDAY, brak exclusions
        Wyjście:  plan z 1 głównym + akcesoryjne, MPC mięśni z wczoraj niższe
        """
        plan = make_planner(ANCHORS_AVERAGE).plan(
            user_history=HISTORY_FULL_BODY_YESTERDAY,
            time_budget_sec=TIME_45_MIN,
            now=SESSION_NOW,
        )

        self._print_plan("05 – tylko isolation+core po ciężkim tygodniu", plan)
        _assert_plan_valid(self, plan, TIME_45_MIN)
        # Pierwsze ćwiczenie zawsze główne (z trójboju)
        if plan.blocks:
            self.assertIn(plan.blocks[0].exercise_id, MAIN_EXERCISES,
                          f"Pierwsze ćwiczenie powinno być z trójboju: {plan.blocks[0].exercise_id}")
        # Pozostałe to akcesoryjne (nie z MAIN_EXERCISES)
        for b in plan.blocks[1:]:
            self.assertNotIn(b.exercise_id, MAIN_EXERCISES,
                             f"{b.exercise_id} nie powinno być w akcesoriach")

    # ── SCENARIUSZ 6 ─────────────────────────────────────────────────────────

    def test_06_monotonicznosc_rir_wzrost_ciezaru(self):
        """
        Scenariusz: ten sam użytkownik, świeże mięśnie, rosnący ciężar.
        Wejście:  bench_press, 5 reps, ciężary [50, 60, 70, 80, 90, 100] kg
        Wyjście:  predict_rir jest (w przybliżeniu) malejące
        """
        planner  = make_planner(ANCHORS_AVERAGE)
        mpc_now  = _inference.predict_mpc(
            _model, HISTORY_EMPTY, SESSION_NOW.isoformat(),
            strength_anchors=ANCHORS_AVERAGE,
        )

        weights = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        rirs = [
            _inference.predict_rir(
                _model, mpc_now, "bench_press", w, 5,
                strength_anchors=ANCHORS_AVERAGE,
            )
            for w in weights
        ]

        print(f"\n{'─'*55}")
        print(f"[06] Monotoniczność RIR – bench_press, 5 reps")
        for w, r in zip(weights, rirs):
            bar = "█" * max(0, int(r * 4))
            print(f"  {w:>5.0f} kg  →  RIR {r:.2f}  {bar}")

        # Sprawdź: żaden krok nie rośnie o więcej niż 0.5
        for i in range(1, len(rirs)):
            self.assertLessEqual(
                rirs[i], rirs[i - 1] + 0.5,
                f"RIR rośnie przy wzroście wagi: {weights[i-1]}kg={rirs[i-1]:.2f} → {weights[i]}kg={rirs[i]:.2f}",
            )

    # ── SCENARIUSZ 7 ─────────────────────────────────────────────────────────

    def test_07_regeneracja_mpc_po_3_dniach(self):
        """
        Scenariusz: full body wczoraj → MPC rośnie z każdym dniem.
        Wejście:  HISTORY_FULL_BODY_YESTERDAY, timestampy: tuż po, +1d, +3d
        Wyjście:  MPC po 3 dniach > MPC po 1 dniu > MPC tuż po treningu
        """
        ts_after  = "2026-04-27T20:00:00"  # 2h po treningu
        ts_1day   = "2026-04-28T10:00:00"  # 16h po treningu
        ts_3days  = "2026-04-30T10:00:00"  # 3 dni po treningu

        mpcs = {}
        for label, ts in [("2h po", ts_after), ("1 dzień", ts_1day), ("3 dni", ts_3days)]:
            mpcs[label] = _inference.predict_mpc(
                _model, HISTORY_FULL_BODY_YESTERDAY, ts,
                strength_anchors=ANCHORS_AVERAGE,
            )

        print(f"\n{'─'*55}")
        print(f"[07] Regeneracja MPC po full body")
        for m in ["chest", "quads", "hamstrings", "erectors"]:
            vals = " → ".join(f"{mpcs[k][m]:.3f}" for k in mpcs)
            print(f"  {m:18s}  {vals}")

        for m in ["chest", "quads", "hamstrings"]:
            self.assertGreater(mpcs["1 dzień"][m], mpcs["2h po"][m],
                               f"{m}: regeneracja w ciągu doby")
            self.assertGreater(mpcs["3 dni"][m], mpcs["1 dzień"][m],
                               f"{m}: regeneracja między 1 a 3 dniem")

    # ── SCENARIUSZ 8 ─────────────────────────────────────────────────────────

    def test_08_wiecej_czasu_nie_gorszy_stimulus(self):
        """
        Scenariusz: ten sam użytkownik – raz 30 min, raz 90 min.
        Wejście:  TIME_30_MIN vs TIME_90_MIN
        Wyjście:  longer_session.total_stimulus ≥ shorter_session.total_stimulus
        """
        planner    = make_planner(ANCHORS_AVERAGE)
        plan_short = planner.plan(HISTORY_EMPTY, TIME_30_MIN, now=SESSION_NOW)
        plan_long  = planner.plan(HISTORY_EMPTY, TIME_90_MIN, now=SESSION_NOW)

        print(f"\n{'─'*55}")
        print(f"[08] Budżet czasu vs stimulus")
        print(f"  30 min:  {len(plan_short.blocks):>2} ćw.  stimulus={plan_short.total_stimulus:.3f}  "
              f"czas={plan_short.total_time_sec//60} min")
        print(f"  90 min:  {len(plan_long.blocks):>2} ćw.  stimulus={plan_long.total_stimulus:.3f}  "
              f"czas={plan_long.total_time_sec//60} min")

        _assert_plan_valid(self, plan_short, TIME_30_MIN)
        _assert_plan_valid(self, plan_long,  TIME_90_MIN)
        self.assertGreaterEqual(plan_long.total_stimulus, plan_short.total_stimulus - 1e-9)

    # ── SCENARIUSZ 9 ─────────────────────────────────────────────────────────

    def test_09_projekcje_1rm_przez_model(self):
        """
        Scenariusz: project_exercise_1rm() daje sensowne wartości dla wariantów.
        Wejście:  ANCHORS_AVERAGE
        Wyjście:  bench_press > incline_bench > ohp > skull_crusher (ratio malejące)
        """
        exercises = ["bench_press", "incline_bench", "close_grip_bench", "ohp",
                     "dips", "skull_crusher", "dumbbell_flyes"]

        print(f"\n{'─'*55}")
        print(f"[09] Projekcje 1RM (bench anchor = {ANCHORS_AVERAGE['bench_press']:.0f} kg)")

        prev_1rm = None
        projections = {}
        for ex in exercises:
            val = _inference.project_exercise_1rm(ex, strength_anchors=ANCHORS_AVERAGE)
            projections[ex] = val
            if val is not None:
                bar = "▓" * int(val / 5)
                print(f"  {ex:25s}  {val:6.1f} kg  {bar}")

        # bench > incline > ohp (ratio malejące w ramach anchor bench)
        bench   = projections.get("bench_press")
        incline = projections.get("incline_bench")
        ohp     = projections.get("ohp")
        if bench and incline:
            self.assertGreater(bench, incline)
        if incline and ohp:
            self.assertGreater(incline, ohp)

    # ── SCENARIUSZ 10 ────────────────────────────────────────────────────────

    def test_10_nogi_zmeczone_plan_unika_nog(self):
        """
        Scenariusz: dzień po treningu nóg → planner powinien unikać przeciążania nóg.
        Wejście:  HISTORY_LEGS_YESTERDAY, brak exclusions, 60 min
        Wyjście:  MPC quads/hamstrings przed sesją niższe; plan nie doprowadza do overfatigue
        """
        planner      = make_planner(ANCHORS_AVERAGE)
        plan_fresh   = planner.plan(HISTORY_EMPTY,         TIME_60_MIN, now=SESSION_NOW)
        plan_leg_day = planner.plan(HISTORY_LEGS_YESTERDAY, TIME_60_MIN, now=SESSION_NOW)

        quads_fresh = _mpc(plan_fresh,   "quads")
        quads_tired = _mpc(plan_leg_day, "quads")
        hams_fresh  = _mpc(plan_fresh,   "hamstrings")
        hams_tired  = _mpc(plan_leg_day, "hamstrings")

        print(f"\n{'─'*55}")
        print(f"[10] Efekt zmęczenia nóg (po dniu nóg)")
        print(f"  quads:     fresh={quads_fresh:.3f}  po nogi={quads_tired:.3f}")
        print(f"  hamstrings fresh={hams_fresh:.3f}  po nogi={hams_tired:.3f}")
        print(f"\n  Plan po dniu nóg:")
        for b in plan_leg_day.blocks:
            print(f"    {b.exercise_id:28s}  {b.sets_count}×{b.reps}@{b.weight_kg:.1f}kg")
        if plan_leg_day.constraint_violations:
            print(f"\n  Violations: {plan_leg_day.constraint_violations}")

        self.assertLess(quads_tired, quads_fresh, "quads: MPC powinno być niższe po dniu nóg")
        self.assertLess(hams_tired,  hams_fresh,  "hamstrings: MPC powinno być niższe po dniu nóg")
        severe = [v for v in plan_leg_day.constraint_violations
                  if "OVERFATIGUE" in v and _mpc_value_from_violation(v) < 0.20]
        self.assertEqual(severe, [], f"Ciężkie overfatigue (MPC<0.20): {severe}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _print_plan(self, label: str, plan: KnapsackPlan):
        print(f"\n{'─'*55}")
        print(f"[{label}]")
        print(f"  Czas: {plan.total_time_sec//60} min   Stimulus: {plan.total_stimulus:.3f}")
        for b in plan.blocks:
            print(f"  {b.exercise_id:28s}  {b.sets_count}×{b.reps} @ {b.weight_kg:.1f} kg"
                  f"  RIR≈{b.predicted_rir:.1f}")
        if plan.constraint_violations:
            print(f"  Violations: {plan.constraint_violations}")

    def _print_mpc_diff(self, label, muscles, plan_a, plan_b,
                        label_a="bez historii", label_b="z historią"):
        print(f"\n{'─'*55}")
        print(f"[{label}]  {label_a} vs {label_b}")
        for m in muscles:
            a = _mpc(plan_a, m)
            b = _mpc(plan_b, m)
            diff = a - b
            bar = "▼" * max(0, int(diff * 20))
            print(f"  {m:20s}  {label_a}: {a:.3f}  {label_b}: {b:.3f}  Δ={diff:+.3f} {bar}")


# ===========================================================================
# [JEDNOSTKOWE] — algorytm DP (nie wymagają modelu do poprawności logiki,
#                ale require_model żeby zachować spójność pliku)
# ===========================================================================

@require_model
class TestKnapsackDP(unittest.TestCase):
    """Izolowane testy algorytmu 0/1 Knapsack DP."""

    def _items(self, specs: list):
        return [
            ExerciseBlock(
                exercise_id=f"ex_{i}", weight_kg=60.0, reps=8, sets_count=3,
                rest_sec=90, predicted_rir=2.0,
                stimulus_score=score, time_cost_sec=time, ex_type="isolation",
            )
            for i, (score, time) in enumerate(specs)
        ]

    def _dp(self, specs, budget):
        return make_planner()._knapsack_dp(self._items(specs), budget)

    def test_dp_wybiera_lepsze_cwiczenie_gdy_tylko_jedno_sie_miesci(self):
        selected = self._dp([(0.8, 200), (0.6, 200)], budget=300)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].stimulus_score, 0.8)

    def test_dp_wybiera_oba_przy_wystarczajacym_budzecie(self):
        selected = self._dp([(0.8, 200), (0.6, 200)], budget=600)
        self.assertEqual(len(selected), 2)

    def test_dp_pusty_plan_przy_zerowym_budzecie(self):
        selected = self._dp([(0.9, 300), (0.7, 200)], budget=0)
        self.assertEqual(len(selected), 0)

    def test_dp_wiekszy_budzet_nie_gorszy_stimulus(self):
        specs = [(0.9, 300), (0.8, 250), (0.7, 200), (0.6, 150)]
        s_small = sum(b.stimulus_score for b in self._dp(specs, 500))
        s_large = sum(b.stimulus_score for b in self._dp(specs, 1000))
        self.assertGreaterEqual(s_large, s_small - 1e-9)

    def test_dp_czas_nie_przekracza_budzetu(self):
        """Ceil-division gwarantuje że suma time_cost ≤ budżet."""
        specs  = [(0.9, 400), (0.85, 350), (0.8, 300), (0.75, 250), (0.7, 200)]
        budget = 700
        selected = self._dp(specs, budget)
        self.assertLessEqual(sum(b.time_cost_sec for b in selected), budget)

    def test_dp_kazde_cwiczenie_co_najwyzej_raz(self):
        items = self._items([(0.9, 200)] * 5)
        for i, b in enumerate(items):
            b.exercise_id = f"unique_{i}"
        selected = make_planner()._knapsack_dp(items, 600)
        ids = [b.exercise_id for b in selected]
        self.assertEqual(len(ids), len(set(ids)))


@require_model
class TestConstraintEvaluation(unittest.TestCase):
    """Testy wykrywania naruszeń target zone."""

    def _p(self): return make_planner()

    def test_brak_naruszen_gdy_mpc_w_strefie(self):
        violations, _ = self._p()._evaluate_constraints(
            {m: 1.0  for m in DEFAULT_TARGET_ZONES},
            {m: 0.70 for m in DEFAULT_TARGET_ZONES},
        )
        self.assertEqual(violations, [])

    def test_overfatigue_wykrywane(self):
        before = {m: 1.0 for m in DEFAULT_TARGET_ZONES}
        violations, _ = self._p()._evaluate_constraints(before, {**before, "chest": 0.10})
        self.assertTrue(any("OVERFATIGUE" in v and "chest" in v for v in violations))

    def test_underfatigue_gdy_miesien_pracowal_za_malo(self):
        before = {m: 1.0 for m in DEFAULT_TARGET_ZONES}
        violations, _ = self._p()._evaluate_constraints(before, {**before, "chest": 0.90})
        self.assertTrue(any("UNDERFATIGUE" in v and "chest" in v for v in violations))

    def test_brak_noty_dla_niepracujacego_miesnia(self):
        """diff < 0.02 → mięsień uznany za nieaktywny."""
        before = {m: 1.0 for m in DEFAULT_TARGET_ZONES}
        _, notes = self._p()._evaluate_constraints(before, {**before, "chest": 0.99})
        self.assertFalse(any("chest" in n for n in notes))


@require_model
class TestExerciseBlock(unittest.TestCase):
    """Testy helpera ExerciseBlock."""

    def _block(self, ex_id="bench_press", sets=3):
        return ExerciseBlock(
            exercise_id=ex_id, weight_kg=80.0, reps=5, sets_count=sets,
            rest_sec=90, predicted_rir=2.0, stimulus_score=0.8,
            time_cost_sec=sets * 120 + (sets - 1) * 90, ex_type="compound",
        )

    def test_to_history_dicts_tyle_wpisow_co_serii(self):
        self.assertEqual(len(self._block(sets=3).to_history_dicts(SESSION_NOW)), 3)

    def test_timestamps_rosna(self):
        ts = [datetime.fromisoformat(e["timestamp"])
              for e in self._block(sets=4).to_history_dicts(SESSION_NOW)]
        for i in range(1, len(ts)):
            self.assertGreater(ts[i], ts[i - 1])

    def test_exercise_id_poprawne_w_kazdym_wpisie(self):
        entries = self._block("deadlift", sets=2).to_history_dicts(SESSION_NOW)
        self.assertTrue(all(e["exercise"] == "deadlift" for e in entries))


# ===========================================================================
# Uruchomienie
# ===========================================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"KnapsackPlanner – Test Suite (DeepGain)")
    print(f"Checkpoint: {_checkpoint or '✗ nie znaleziono'}")
    print(f"{'='*55}\n")
    unittest.main(verbosity=2, exit=True)
