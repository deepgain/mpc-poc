"""
Praktyczne przykłady użycia WorkoutPlanner z integracją DeepGain.

WAŻNE: MPC = Muscle Performance Capacity w [0.1, 1.0]
    - 1.0 = fresh (fully recovered)
    - 0.1 = exhausted

Uruchom:
    python3 example_usage.py
"""

import logging
from datetime import datetime, timedelta

from data_structures import (
    WorkoutSet, PlannerConfig,
    DEFAULT_TARGET_CAPACITY_ZONES, DEFAULT_DEFAULT_REPS_BY_TYPE,
)
from planner import WorkoutPlanner
import models_wrapper

logging.basicConfig(level=logging.WARNING)  # Mniej logów w przykładach


def setup_planner():
    """Stwórz planer z domyślną konfiguracją."""
    # Próbuje załadować real model, fallback na mock
    models_wrapper.initialize_model()

    config = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        target_rir=2,
    )
    return WorkoutPlanner(config)


# ============================================================================
# EXAMPLE 1: Fresh user — pełny upper body
# ============================================================================
def example1_fresh_upper():
    print("\n" + "="*70)
    print("EXAMPLE 1: Fresh User — Upper Body Workout")
    print("="*70)

    planner = setup_planner()

    # Fresh state — wszystkie mięśnie na 1.0 (capacity)
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    # Wyłącz nogi (chcemy upper body)
    leg_exercises = ["squat", "low_bar_squat", "high_bar_squat", "deadlift",
                     "sumo_deadlift", "bulgarian_split_squat", "leg_press",
                     "romanian_deadlift", "leg_curl", "leg_extension"]

    result = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
        exclusions=leg_exercises,
    )

    print(f"\n📋 PLAN ({len(result.plan)} serii, {result.total_time_estimated_sec/60:.1f} min):")
    current_exercise = None
    for s in result.plan:
        if s.exercise_id != current_exercise:
            print(f"\n  🏋️  {s.exercise_id.upper()}  ({', '.join(s.primary_muscles)})")
            current_exercise = s.exercise_id
        rir_str = f"RIR~{s.predicted_rir:.1f}" if s.predicted_rir else ""
        print(f"     Seria {s.order}: {s.reps}×{s.weight_kg}kg  {rir_str}")

    print(f"\n📊 MPC (capacity) after — zaangażowane mięśnie:")
    for m in sorted(result.predicted_mpc_after.keys()):
        mpc = result.predicted_mpc_after[m]
        if mpc < 0.99:  # Zaangażowane
            target = planner.config.get_target_zone(m)
            status = "✓" if target[0] <= mpc <= target[1] else "⚠"
            print(f"   {status} {m}: {mpc:.2f}  (target: [{target[0]}, {target[1]}])")


# ============================================================================
# EXAMPLE 2: Po leg day — oszczędza nogi
# ============================================================================
def example2_post_leg_day():
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Po Leg Day (16h temu) — System Oszczędza Nogi")
    print("="*70)

    planner = setup_planner()

    # Symuluj leg day 16h temu
    now = datetime.now()
    yesterday = now - timedelta(hours=16)
    leg_history = [
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday + timedelta(minutes=3)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday + timedelta(minutes=6)),
        WorkoutSet('leg_curl', 40.0, 10, rir=2, timestamp=yesterday + timedelta(minutes=25)),
        WorkoutSet('leg_curl', 40.0, 10, rir=1, timestamp=yesterday + timedelta(minutes=28)),
        WorkoutSet('leg_extension', 50.0, 12, rir=2, timestamp=yesterday + timedelta(minutes=40)),
    ]

    # Pobierz MPC (capacity) teraz
    current_mpc = models_wrapper.predict_mpc(
        [ws.to_model_dict() for ws in leg_history],
        now
    )

    print(f"\n📊 Obecne MPC (po 16h odpoczynku):")
    for m in ['quads', 'hamstrings', 'glutes', 'chest', 'lats', 'biceps']:
        print(f"   {m}: {current_mpc[m]:.2f}  {'← zmęczone' if current_mpc[m] < 0.85 else ''}")

    # Zaplanuj — planer sam powinien unikać nóg
    result = planner.plan(
        state=current_mpc,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
        user_history=leg_history,
        now=now,
    )

    print(f"\n📋 PLAN:")
    current_exercise = None
    for s in result.plan:
        if s.exercise_id != current_exercise:
            print(f"   {s.exercise_id} (×{sum(1 for x in result.plan if x.exercise_id == s.exercise_id)} sets, "
                  f"muscles: {', '.join(s.primary_muscles[:2])})")
            current_exercise = s.exercise_id


# ============================================================================
# EXAMPLE 3: Replanning — user odrzuca ćwiczenie
# ============================================================================
def example3_replanning():
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Replanning — User Odrzuca Ćwiczenie w Trakcie")
    print("="*70)

    planner = setup_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    # Oryginalny plan
    original = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
    )

    print(f"\n📋 Oryginalny plan:")
    unique_ex = []
    for s in original.plan:
        if s.exercise_id not in unique_ex:
            unique_ex.append(s.exercise_id)
    for i, ex in enumerate(unique_ex):
        print(f"   {i+1}. {ex}")

    # Załóżmy: user zrobił pierwsze 3 serie (pierwsze ćwiczenie), ale odrzuca drugie
    first_exercise = unique_ex[0]
    completed = [s for s in original.plan if s.exercise_id == first_exercise]

    rejected_exercise = unique_ex[1] if len(unique_ex) > 1 else None
    print(f"\n⏳ User zrobił: {first_exercise} ({len(completed)} sets)")
    print(f"❌ User odrzuca: {rejected_exercise}")

    # Replan: brakuje 1 compound (bo 1 już zrobił), 2 isolation
    replanned = planner.replan(
        session_so_far=completed,
        remaining_n_compound=1,
        remaining_n_isolation=2,
        available_time_sec=3600,
        exclusions=[rejected_exercise] if rejected_exercise else [],
    )

    print(f"\n✅ Nowy plan:")
    seen = set()
    for s in replanned.plan:
        if s.exercise_id not in seen:
            seen.add(s.exercise_id)
            count = sum(1 for x in replanned.plan if x.exercise_id == s.exercise_id)
            status = "✓ done" if s.exercise_id == first_exercise else "+ new"
            print(f"   {status}: {s.exercise_id} (×{count} sets)")


# ============================================================================
# EXAMPLE 4: Short session (30 min)
# ============================================================================
def example4_short_session():
    print("\n\n" + "="*70)
    print("EXAMPLE 4: 30-Minutowy Trening")
    print("="*70)

    planner = setup_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    result = planner.plan(
        state=fresh_state,
        n_compound=1,
        n_isolation=2,
        available_time_sec=1800,  # 30 min
    )

    print(f"\n📋 Plan w 30 min:")
    print(f"   Całkowity czas: {result.total_time_estimated_sec/60:.1f} min")
    print(f"   Ilość serii: {len(result.plan)}")

    unique = {}
    for s in result.plan:
        unique.setdefault(s.exercise_id, []).append(s)
    for ex, sets in unique.items():
        first = sets[0]
        print(f"   - {ex}: {len(sets)}×{first.reps} @ {first.weight_kg}kg")


# ============================================================================
# EXAMPLE 5: Historia użytkownika + estymacja 1RM
# ============================================================================
def example5_with_history():
    print("\n\n" + "="*70)
    print("EXAMPLE 5: Z Historią Użytkownika — Auto 1RM Estimation")
    print("="*70)

    planner = setup_planner()

    now = datetime.now()
    user_history = [
        # Ostatnie 3 tygodnie progresji benchu
        WorkoutSet('bench_press', 70.0, 8, rir=2, timestamp=now - timedelta(days=21)),
        WorkoutSet('bench_press', 75.0, 6, rir=2, timestamp=now - timedelta(days=14)),
        WorkoutSet('bench_press', 80.0, 5, rir=1, timestamp=now - timedelta(days=7)),
        # Squat
        WorkoutSet('squat', 90.0, 8, rir=3, timestamp=now - timedelta(days=20)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=now - timedelta(days=6)),
        # Deadlift
        WorkoutSet('deadlift', 120.0, 5, rir=2, timestamp=now - timedelta(days=12)),
    ]

    estimated_1rm = planner.estimate_1rm_from_history(user_history)
    print(f"\n💪 Estimated 1RM (Brzycki formula z RIR correction):")
    for ex_id in ['bench_press', 'squat', 'deadlift']:
        if ex_id in estimated_1rm:
            print(f"   {ex_id}: {estimated_1rm[ex_id]:.1f} kg")

    # Planer użyje 75% 1RM jako default weight
    result = planner.plan(
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
        user_history=user_history,
        now=now,
    )

    print(f"\n📋 Dzisiejszy plan (weights = 75% estimated 1RM):")
    seen = set()
    for s in result.plan:
        if s.exercise_id in seen:
            continue
        seen.add(s.exercise_id)
        count = sum(1 for x in result.plan if x.exercise_id == s.exercise_id)
        one_rm = estimated_1rm.get(s.exercise_id, 0)
        pct = (s.weight_kg / one_rm * 100) if one_rm > 0 else 0
        print(f"   - {s.exercise_id}: {count}×{s.reps} @ {s.weight_kg}kg ({pct:.0f}% 1RM)")


# ============================================================================
# EXAMPLE 6: Sprawdzenie czy używamy real model
# ============================================================================
def example6_model_status():
    print("\n\n" + "="*70)
    print("EXAMPLE 6: Status Modelu")
    print("="*70)

    planner = setup_planner()

    print(f"\n🤖 Model status:")
    if models_wrapper.is_using_real_model():
        print(f"   ✓ Używam prawdziwego DeepGain (Michała)")
    else:
        print(f"   ⚠ Używam Mock model (fallback)")
        print(f"      Powód: brak torch lub brak deepgain_model_muscle_ord.pt")
        print(f"      Aby użyć prawdziwego modelu:")
        print(f"        1. pip install torch numpy pandas pyyaml")
        print(f"        2. Umieść deepgain_model_muscle_ord.pt w tym katalogu")
        print(f"        3. Umieść exercise_muscle_order.yaml")
        print(f"        4. Umieść exercise_muscle_weights_scaled.csv")

    print(f"\n📏 Model dostarcza:")
    print(f"   - {len(planner.all_muscles)} mięśni")
    print(f"   - {len(planner.all_exercises)} ćwiczeń (w catalogu planer widzi {len(planner.exercise_catalog)})")

    print(f"\n🎯 Target capacity zones (przykład):")
    for muscle in ['quads', 'chest', 'biceps', 'abs']:
        zone = planner.config.get_target_zone(muscle)
        print(f"   {muscle}: [{zone[0]}, {zone[1]}]  # MPC po treningu (capacity)")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n\n🏋️  WorkoutPlanner v2 — Przykłady Użycia (DeepGain Integration)\n")

    example6_model_status()
    example1_fresh_upper()
    example2_post_leg_day()
    example3_replanning()
    example4_short_session()
    example5_with_history()

    print("\n\n" + "="*70)
    print("✅ Wszystkie przykłady zakończone!")
    print("="*70)
    print("\nWięcej informacji: README.md lub PLANNER_REPORT.md\n")
