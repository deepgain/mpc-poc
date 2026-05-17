"""
test_multi_session.py
=====================
Test tygodniowego cyklu treningowego z prawdziwym modelem DeepGain.

Symuluje tydzień:
  Poniedziałek  — trening 1 (60 min, pełne zmęczenie po)
  Wtorek        — trening 2 (60 min, następny dzień, wysokie zmęczenie)
  Środa         — trening 3 (45 min, drugi dzień po treningu 1)
  Piątek        — trening 4 (60 min, 4 dni po treningu 1 — regeneracja)
  Niedziela     — trening 5 (60 min, 6 dni po treningu 1 — pełna regeneracja)

Co sprawdza:
  - Planner dobiera inne ćwiczenia gdy mięśnie są zmęczone vs wypoczęte
  - MPC mięśni rośnie między sesjami (regeneracja)
  - Przy zmęczeniu planner wybiera inne grupy mięśniowe
  - Ciężary są niższe przy zmęczonych mięśniach (model to uwzględnia)
  - Ćwiczenie główne zmienia się w zależności od stanu mięśni

Uruchomienie:
  python test_multi_session.py

Wymagania:
  - deepgain_model_best.pt (w katalogu models/ projektu)
  - inference.py + strength_priors.py + exercise_muscle_order.yaml + ...
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict

logging.basicConfig(level=logging.WARNING)


# ===========================================================================
# Konfiguracja — zmień na swoje wartości
# ===========================================================================

STRENGTH_ANCHORS = {
    "bench_press": 100.0,   # Twój 1RM bench press [kg]
    "squat":       140.0,   # Twój 1RM squat [kg]
    "deadlift":    180.0,   # Twój 1RM deadlift [kg]
}

TIME_60_MIN = 60 * 60
TIME_45_MIN = 45 * 60
TARGET_RIR  = 3


# ===========================================================================
# Załaduj model
# ===========================================================================

def _find_inference_dir():
    this = os.path.dirname(os.path.abspath(__file__))
    current = this
    while True:
        if os.path.isfile(os.path.join(current, "inference.py")):
            return current
        models_dir = os.path.join(current, "models")
        if os.path.isfile(os.path.join(models_dir, "inference.py")):
            return models_dir
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

inf_dir = _find_inference_dir()
if inf_dir is None:
    print("✗  Nie znaleziono inference.py — upewnij się że skrypt jest w projekcie mpc-poc")
    sys.exit(1)

# Ładuj inference.py bezpośrednio z pliku — omija konflikty z nazwą pakietu "models"
import importlib.util

def _load_module(name: str, path: str):
    """Ładuje moduł z podanej ścieżki i rejestruje w sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # rejestruj PRZED exec — obsługuje circular imports
    spec.loader.exec_module(mod)
    return mod


orig_cwd = os.getcwd()
os.chdir(inf_dir)   # musi być przed ładowaniem — inference.py czyta ../dataset/
try:
    # Kolejność: najpierw strength_priors (inference.py go importuje przy starcie)
    _load_module("strength_priors", os.path.join(inf_dir, "strength_priors.py"))
    inf = _load_module("inference",      os.path.join(inf_dir, "inference.py"))
finally:
    os.chdir(orig_cwd)
# sys.modules["inference"] jest teraz ustawione — knapsack_planner znajdzie go przez
# zwykłe "import inference" bez żadnych konfliktów z pakietem models/

checkpoint = os.path.join(inf_dir, "deepgain_model_best.pt")
if not os.path.isfile(checkpoint):
    print(f"✗  Nie znaleziono checkpointa: {checkpoint}")
    sys.exit(1)

MODEL = inf.load_model(checkpoint)
print(f"✓  Model załadowany: {checkpoint}\n")

from knapsack_planner import KnapsackPlanner, KnapsackPlan, MAIN_EXERCISES

PLANNER = KnapsackPlanner(
    model=MODEL,
    inference_module=inf,
    strength_anchors=STRENGTH_ANCHORS,
    rest_between_sets_sec=120,
)


# ===========================================================================
# Helpery
# ===========================================================================

# Daty tygodnia (poniedziałek = start)
WEEK_START = datetime(2026, 4, 27, 10, 0, 0)  # poniedziałek

SESSION_TIMES = {
    "Poniedziałek": WEEK_START,
    "Wtorek":       WEEK_START + timedelta(days=1),
    "Środa":        WEEK_START + timedelta(days=2),
    "Piątek":       WEEK_START + timedelta(days=4),
    "Niedziela":    WEEK_START + timedelta(days=6),
}

# Mięśnie które chcemy śledzić w raporcie
TRACKED_MUSCLES = ["chest", "quads", "hamstrings", "lats", "triceps",
                   "anterior_delts", "glutes", "erectors", "biceps", "abs"]

MUSCLE_PL = {
    "chest": "Klatka",
    "quads": "Czworo.",
    "hamstrings": "Dwugłowy",
    "lats": "Grzbiet",
    "triceps": "Triceps",
    "anterior_delts": "Bark prz.",
    "glutes": "Pośladki",
    "erectors": "Kręgosłup",
    "biceps": "Biceps",
    "abs": "Brzuch",
}


def format_mpc_bar(val: float) -> str:
    filled = int(val * 20)
    empty  = 20 - filled
    return f"{'█' * filled}{'░' * empty} {val:.2f}"


def print_plan(label: str, plan: KnapsackPlan):
    main = plan.blocks[0] if plan.blocks else None
    acc  = plan.blocks[1:]

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Czas: {plan.total_time_sec // 60} min   "
          f"Stimulus: {plan.total_stimulus:.2f}")

    if main:
        print(f"\n  GŁÓWNE:")
        print(f"    {main.exercise_id:28s}  {main.sets_count}×{main.reps} "
              f"@ {main.weight_kg:.1f} kg   RIR≈{main.predicted_rir:.1f}")

    if acc:
        print(f"\n  AKCESORYJNE:")
        for b in acc:
            print(f"    {b.exercise_id:28s}  {b.sets_count}×{b.reps} "
                  f"@ {b.weight_kg:.1f} kg   RIR≈{b.predicted_rir:.1f}")

    if plan.constraint_violations:
        print(f"\n  ⚠  {len(plan.constraint_violations)} violations:")
        for v in plan.constraint_violations:
            print(f"     {v}")


def print_mpc_table(header: str, mpc_dict: Dict[str, Dict[str, float]]):
    """Drukuje tabelę MPC dla kilku punktów czasowych."""
    sessions = list(mpc_dict.keys())
    col_w = 9

    print(f"\n{header}")
    print(f"  {'Mięsień':15s}", end="")
    for s in sessions:
        print(f"  {s[:col_w]:>{col_w}}", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in sessions:
        print(f"  {'─'*col_w}", end="")
    print()

    for muscle in TRACKED_MUSCLES:
        name = MUSCLE_PL.get(muscle, muscle)
        print(f"  {name:15s}", end="")
        prev = None
        for s in sessions:
            val = mpc_dict[s].get(muscle, 1.0)
            # Strzałka trendu
            if prev is not None:
                trend = "↑" if val > prev + 0.02 else ("↓" if val < prev - 0.02 else "→")
            else:
                trend = " "
            print(f"  {trend}{val:.2f}   ", end="")
            prev = val
        print()


def plan_to_history(plan: KnapsackPlan, session_now: datetime) -> List[dict]:
    """Konwertuje wykonany plan na historię serii dla predict_mpc."""
    history = []
    t = session_now
    for block in plan.blocks:
        entries = block.to_history_dicts(t)
        history.extend(entries)
        t += timedelta(seconds=block.time_cost_sec)
    return history


# ===========================================================================
# Główny test — tydzień treningowy
# ===========================================================================

def run_week():
    print("\n" + "═"*60)
    print("  TEST WIELOSESYJNY — TYDZIEŃ TRENINGOWY")
    print("  Anchory 1RM: bench={bench_press} kg | squat={squat} kg | "
          "deadlift={deadlift} kg".format(**STRENGTH_ANCHORS))
    print("═"*60)

    history: List[dict] = []
    plans: Dict[str, KnapsackPlan] = {}
    mpc_snapshots: Dict[str, Dict[str, float]] = {}

    for day_name, session_time in SESSION_TIMES.items():
        # MPC przed tą sesją
        mpc_before = inf.predict_mpc(
            MODEL,
            user_history=history,
            timestamp=session_time.isoformat(),
            strength_anchors=STRENGTH_ANCHORS,
        )
        mpc_snapshots[f"{day_name}\n(przed)"] = mpc_before

        # Dobierz czas sesji
        budget = TIME_45_MIN if day_name == "Środa" else TIME_60_MIN

        # Generuj plan
        plan = PLANNER.plan(
            user_history=history,
            time_budget_sec=budget,
            target_rir=TARGET_RIR,
            now=session_time,
        )
        plans[day_name] = plan
        print_plan(f"{day_name} ({session_time.strftime('%d.%m %H:%M')})", plan)

        # Dodaj sesję do historii (symulujemy wykonanie)
        session_sets = plan_to_history(plan, session_time)
        history.extend(session_sets)

    # MPC po ostatnim treningu i po weekend recovery
    for label, ts in [
        ("Po niedzieli\n(zaraz po)", SESSION_TIMES["Niedziela"] + timedelta(hours=2)),
        ("Poniedziałek\n(tydzień 2)", SESSION_TIMES["Niedziela"] + timedelta(days=1)),
    ]:
        mpc_snapshots[label] = inf.predict_mpc(
            MODEL,
            user_history=history,
            timestamp=ts.isoformat(),
            strength_anchors=STRENGTH_ANCHORS,
        )

    # ── Raport MPC tygodniowy ────────────────────────────────────────────
    print_mpc_table(
        "\nMPC MIĘŚNI PRZEZ CAŁY TYDZIEŃ  (↑ rośnie / ↓ spada / → stabilne)",
        mpc_snapshots,
    )

    # ── Analiza głównych ćwiczeń ─────────────────────────────────────────
    print("\n\nGŁÓWNE ĆWICZENIA W TYGODNIU:")
    for day, plan in plans.items():
        if plan.blocks:
            main = plan.blocks[0]
            print(f"  {day:12s}  {main.exercise_id:20s}  "
                  f"{main.weight_kg:.1f} kg × {main.reps} reps   "
                  f"RIR≈{main.predicted_rir:.1f}")

    # ── Analiza ciężarów przy zmęczeniu ──────────────────────────────────
    print("\n\nPORÓWNANIE CIĘŻARÓW: PONIEDZIAŁEK vs WTOREK (zmęczenie)")
    mon = plans.get("Poniedziałek")
    tue = plans.get("Wtorek")
    if mon and tue:
        mon_map = {b.exercise_id: b for b in mon.blocks}
        tue_map = {b.exercise_id: b for b in tue.blocks}
        common  = set(mon_map) & set(tue_map)
        if common:
            print(f"  {'Ćwiczenie':28s}  {'Pon':>8}  {'Wt':>8}  {'Δ':>8}")
            print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")
            for ex in sorted(common):
                w_mon = mon_map[ex].weight_kg
                w_tue = tue_map[ex].weight_kg
                delta = w_tue - w_mon
                sign  = "+" if delta > 0 else ""
                print(f"  {ex:28s}  {w_mon:>6.1f}kg  {w_tue:>6.1f}kg  "
                      f"{sign}{delta:+.1f}kg")
        else:
            print("  Brak wspólnych ćwiczeń — planner wybrał inne mięśnie")

    # ── Regeneracja ──────────────────────────────────────────────────────
    print("\n\nREGENERACJA KLUCZOWYCH MIĘŚNI:")
    after_mon_ts = SESSION_TIMES["Poniedziałek"] + timedelta(hours=1)
    checkpoints  = {
        "tuż po\ntrenigu 1": after_mon_ts,
        "24h":  after_mon_ts + timedelta(hours=23),
        "48h":  after_mon_ts + timedelta(hours=47),
        "96h":  after_mon_ts + timedelta(hours=95),
    }

    regen: Dict[str, Dict[str, float]] = {}
    for label, ts in checkpoints.items():
        regen[label] = inf.predict_mpc(
            MODEL,
            user_history=history[:len(plan_to_history(plans["Poniedziałek"],
                                                       SESSION_TIMES["Poniedziałek"]))],
            timestamp=ts.isoformat(),
            strength_anchors=STRENGTH_ANCHORS,
        )

    print(f"\n  (tylko historia z poniedziałku, bez kolejnych treningów)")
    for muscle in ["chest", "quads", "hamstrings", "lats", "triceps"]:
        name = MUSCLE_PL.get(muscle, muscle)
        print(f"\n  {name}:")
        for label, mpc in regen.items():
            val = mpc.get(muscle, 1.0)
            bar = format_mpc_bar(val)
            print(f"    {label:12s}  {bar}")

    print("\n" + "═"*60)
    print("  KONIEC TESTU")
    print("═"*60 + "\n")


if __name__ == "__main__":
    run_week()
