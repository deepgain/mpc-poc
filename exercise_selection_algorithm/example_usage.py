"""
Przykłady praktycznego użycia WorkoutPlanner

Uruchom:
    python3 example_usage.py
"""

import json
from datetime import datetime, timedelta
from planner import WorkoutPlanner
from data_structures import PlannerConfig, WorkoutSet


def load_planner():
    """Helper: wczytaj planner z config"""
    with open('exercises_config.json', 'r') as f:
        exercises_config = json.load(f)

    planner_config = PlannerConfig(
        target_fatigue_zones=exercises_config['target_fatigue_zones'],
        default_reps_by_type=exercises_config['default_reps_by_type'],
    )

    return WorkoutPlanner(exercises_config, planner_config), exercises_config, planner_config


# ============================================================================
# EXAMPLE 1: Nowy użytkownik - plan "push day" (klatka, ramiona)
# ============================================================================
def example1_push_day():
    """
    Scenariusz: Nowy użytkownik chce zrobić push day (bench press + shoulders).
    Wszystkie mięśnie świeże (MPC=0).
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Push Day (Fresh User)")
    print("="*70)

    planner, _, config = load_planner()

    # Świeży stan
    state = {m: 0.0 for m in config.target_fatigue_zones.keys()}

    # Zaplanuj: 1 compound (bench), 2 isolation (accessories)
    result = planner.plan(
        state=state,
        n_compound=1,
        n_isolation=2,
        available_time_sec=2400,  # 40 minut
    )

    print(f"\n📋 PLAN:")
    for s in result.plan:
        print(f"  {s.order}. {s.exercise_id}")
        print(f"      └─ {s.reps}x{s.weight_kg}kg | ~{s.estimated_time_sec}s")
        print(f"      └─ Muscles: {', '.join(s.primary_muscles[:2])}")

    print(f"\n⏱️  Total time: {result.total_time_estimated_sec / 60:.1f} min")

    print(f"\n📊 Predicted MPC after workout:")
    chest_muscles = ['chest_upper', 'chest_lower', 'shoulder_front', 'shoulder_side']
    for muscle in chest_muscles:
        if muscle in result.predicted_mpc_after:
            mpc = result.predicted_mpc_after[muscle]
            target = config.target_fatigue_zones[muscle]
            status = "✓" if target[0] <= mpc <= target[1] else "⚠"
            print(f"  {status} {muscle}: {mpc:.2f} (target: [{target[0]}, {target[1]}])")

    print(f"\n✅ Validation:")
    for note in result.notes:
        if muscle in note or "chest" in note or "shoulder" in note:
            print(f"  {note}")


# ============================================================================
# EXAMPLE 2: Powrót po dwudniowej przerwie - zmęczone nogi
# ============================================================================
def example2_upper_day_after_leg():
    """
    Scenariusz: User miał leg day 2 dni temu, teraz chce robić upper body.
    Nogi są jeszcze zmęczone, powinniśmy robić upper.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Upper Day After Leg Day (2 days ago)")
    print("="*70)

    planner, _, config = load_planner()

    # Stan: nogi zmęczone, reszta świeża
    state = {m: 0.0 for m in config.target_fatigue_zones.keys()}
    state['quadriceps'] = 0.35
    state['hamstring'] = 0.30
    state['glutes'] = 0.35
    state['erector_spinae'] = 0.20

    print(f"\n📊 Starting state (after leg day 2 days ago):")
    for muscle in ['quadriceps', 'hamstring', 'glutes', 'erector_spinae']:
        print(f"  {muscle}: {state[muscle]:.2f}")

    # Zaplanuj: upper body (no legs)
    result = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
    )

    print(f"\n📋 Suggested plan:")
    for s in result.plan:
        muscles = s.primary_muscles[:2]
        print(f"  {s.order}. {s.exercise_id} ({', '.join(muscles)})")

    # Sprawdź czy planner uniknął nóg
    legs = ['quadriceps', 'hamstring', 'glutes', 'calves']
    plan_muscles = []
    for s in result.plan:
        plan_muscles.extend(s.primary_muscles)

    legs_in_plan = [m for m in legs if m in plan_muscles]

    if legs_in_plan:
        print(f"\n⚠️  WARNING: Planner included leg muscles: {legs_in_plan}")
        print("  (Możesz je wyłączyć za pomocą exclusions)")
    else:
        print(f"\n✓ Good: Plan focuses on upper body, spares legs")


# ============================================================================
# EXAMPLE 3: Replanning - user odrzuca ćwiczenie w trakcie treningu
# ============================================================================
def example3_replanning_midworkout():
    """
    Scenariusz: User ma plan 4 serii.
    Wykonał 2, ale nie czuje się dobrze - chce zmienić trzecią serię.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Replanning - User Rejects Exercise Mid-Workout")
    print("="*70)

    planner, _, config = load_planner()

    state = {m: 0.0 for m in config.target_fatigue_zones.keys()}

    # Zaplanuj oryginalny trening
    original = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
    )

    print(f"\n📋 Original plan:")
    for s in original.plan:
        print(f"  {s.order}. {s.exercise_id}")

    # Symuluj: user wykonał pierwsze 2 serie
    print(f"\n⏳ User completed sets 1-2, starts set 3...")
    print(f"   Set 3 is: {original.plan[2].exercise_id}")
    print(f"   → User says: 'I don't feel like this one, give me something else'")

    completed = original.plan[:2]

    # Przeplanuj: został 0 compound, 1 isolation (bo już 2 razy robił isolation)
    replanned = planner.replan(
        session_so_far=completed,
        remaining_n_compound=0,
        remaining_n_isolation=1,
        current_state=state,
        available_time_sec=1200,
    )

    print(f"\n✅ Replanned workout:")
    for s in replanned.plan:
        if s.order <= 2:
            print(f"  {s.order}. {s.exercise_id} (completed)")
        else:
            print(f"  {s.order}. {s.exercise_id} (NEW)")

    print(f"\n💡 Changed: Set 3 from '{original.plan[2].exercise_id}' → '{replanned.plan[2].exercise_id}'")


# ============================================================================
# EXAMPLE 4: Krótka sesja (30 min) - time-constrained planning
# ============================================================================
def example4_quick_session():
    """
    Scenariusz: User ma tylko 30 minut na trening w pracy.
    Chce coś szybko, ale efektywnie.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Quick Session (30 min) - Time Constrained")
    print("="*70)

    planner, _, config = load_planner()

    state = {m: 0.0 for m in config.target_fatigue_zones.keys()}

    # Zaplanuj maksymalnie w 30 minut
    result = planner.plan(
        state=state,
        n_compound=2,   # ambitne, ale zobaczymy co się zmieści
        n_isolation=2,
        available_time_sec=30 * 60,  # 30 min
    )

    print(f"\n📋 30-Minute Workout Plan:")
    total = 0
    for s in result.plan:
        print(f"  {s.order}. {s.exercise_id}")
        print(f"      └─ Time: {s.estimated_time_sec}s + rest")
        total += s.estimated_time_sec

    print(f"\n⏱️  Total execution time: ~{total / 60:.1f} min (realistic: ~{(total + 90*len(result.plan)) / 60:.1f} with rest)")

    if total + 90 * len(result.plan) <= 30 * 60:
        print(f"✓ Fits in 30 minutes!")
    else:
        print(f"⚠️  Might be tight, need to reduce rest time or skip core")


# ============================================================================
# EXAMPLE 5: Z historią użytkownika - smart 1RM estimation
# ============================================================================
def example5_with_user_history():
    """
    Scenariusz: User trenuje ostatni miesiąc.
    Planner estymuje 1RM z historii i dostosowuje weights.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Planning with User Training History")
    print("="*70)

    planner, _, config = load_planner()

    # Symuluj historię: treningi z ostatniego miesiąca
    now = datetime.now()
    user_history = [
        # Tydzień 1
        WorkoutSet('back_squat', 60.0, 10, rir=3, timestamp=now - timedelta(days=28)),
        WorkoutSet('back_squat', 65.0, 10, rir=2, timestamp=now - timedelta(days=28)),
        WorkoutSet('bench_press', 50.0, 10, rir=3, timestamp=now - timedelta(days=27)),
        
        # Tydzień 2
        WorkoutSet('back_squat', 70.0, 8, rir=2, timestamp=now - timedelta(days=21)),
        WorkoutSet('back_squat', 70.0, 8, rir=2, timestamp=now - timedelta(days=21)),
        
        # Tydzień 3
        WorkoutSet('back_squat', 75.0, 6, rir=2, timestamp=now - timedelta(days=14)),
        WorkoutSet('bench_press', 55.0, 8, rir=2, timestamp=now - timedelta(days=14)),
        
        # Tydzień 4
        WorkoutSet('back_squat', 80.0, 5, rir=2, timestamp=now - timedelta(days=7)),
        WorkoutSet('bench_press', 60.0, 8, rir=1, timestamp=now - timedelta(days=7)),
    ]

    print(f"\n📚 User training history (last month):")
    for s in user_history:
        print(f"  {s.exercise_id}: {s.weight_kg}kg × {s.reps} (RIR={s.rir})")

    # Estymuj 1RM
    estimated_1rm = planner.estimate_1rm_from_history(user_history)
    print(f"\n💪 Estimated 1RM:")
    for ex_id in ['back_squat', 'bench_press']:
        if ex_id in estimated_1rm:
            print(f"  {ex_id}: {estimated_1rm[ex_id]:.1f}kg")

    # Zaplanuj nowy trening
    state = {m: 0.0 for m in config.target_fatigue_zones.keys()}
    result = planner.plan(
        state=state,
        n_compound=1,
        n_isolation=1,
        available_time_sec=1800,
        user_history=user_history,  # Pass history for smart planning
    )

    print(f"\n📋 Today's plan (based on estimated 1RM):")
    for s in result.plan:
        print(f"  {s.order}. {s.exercise_id}: {s.reps}x{s.weight_kg}kg")
        ex_1rm = estimated_1rm.get(s.exercise_id, 100)
        percentage = (s.weight_kg / ex_1rm) * 100
        print(f"      └─ {percentage:.0f}% of estimated 1RM")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n\n🏋️  WorkoutPlanner - Praktyczne Przykłady\n")

    example1_push_day()
    example2_upper_day_after_leg()
    example3_replanning_midworkout()
    example4_quick_session()
    example5_with_user_history()

    print("\n\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
    print("\nFor more details, see README.md or PLANNER_REPORT.md\n")
