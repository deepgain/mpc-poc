"""
Testy symulacyjne WorkoutPlanner

Scenariusze:
1. Fresh user (wszystkie mięśnie ~0 MPC)
2. Zmęczony user (niektóre mięśnie wysoki MPC)
3. Replanning - user odrzuca ćwiczenie
4. Walidacja target zones
5. Wyczerpanie czasu sesji
"""

import json
import logging
from datetime import datetime, timedelta
from data_structures import WorkoutSet, PlannerConfig
from planner import WorkoutPlanner

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config():
    """Wczytaj exercise catalog i config"""
    with open('/home/claude/exercises_config.json', 'r') as f:
        exercises_config = json.load(f)

    # Zbuduj PlannerConfig
    planner_config = PlannerConfig(
        target_fatigue_zones=exercises_config['target_fatigue_zones'],
        default_reps_by_type=exercises_config['default_reps_by_type'],
        default_time_per_rep_sec=exercises_config['default_time_per_rep_sec'],
        rest_between_sets_sec=exercises_config['rest_between_sets_sec'],
    )

    return exercises_config, planner_config


def test_fresh_user():
    """Test 1: Świeży user (first workout)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Fresh User (first workout)")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    # Fresh state - wszystko na 0
    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

    result = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,  # 1 godzina
        user_history=[],
    )

    logger.info(f"\n✓ Generated plan with {len(result.plan)} sets")
    logger.info(f"  Total time: {result.total_time_estimated_sec / 60:.1f} min")
    logger.info(f"\n  Exercises:")
    for s in result.plan:
        logger.info(f"    {s.order}. {s.exercise_id}: {s.reps}x{s.weight_kg}kg")

    logger.info(f"\n  Predicted MPC after workout:")
    for muscle, mpc in sorted(result.predicted_mpc_after.items()):
        target = planner_config.get_target_zone(muscle)
        in_zone = "✓" if target[0] <= mpc <= target[1] else "✗"
        logger.info(f"    {in_zone} {muscle}: {mpc:.3f} (target: {target})")

    logger.info(f"\n  Notes:")
    for note in result.notes:
        logger.info(f"    {note}")

    return result


def test_fatigued_user():
    """Test 2: Zmęczony user (some muscles already fatigued)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Fatigued User (some muscles already worked)")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    # Simulate user that did leg workout yesterday
    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}
    state['quadriceps'] = 0.35
    state['hamstring'] = 0.30
    state['glutes'] = 0.35

    logger.info(f"\nStarting state (from yesterday's leg workout):")
    for muscle in ['quadriceps', 'hamstring', 'glutes']:
        logger.info(f"  {muscle}: {state[muscle]:.2f}")

    result = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
        user_history=[],
    )

    logger.info(f"\n✓ Generated plan with {len(result.plan)} sets")
    logger.info(f"  Exercises:")
    for s in result.plan:
        logger.info(f"    {s.order}. {s.exercise_id}")

    logger.info(f"\n  Predicted MPC after workout:")
    for muscle in ['quadriceps', 'hamstring', 'glutes', 'chest_upper', 'lats']:
        if muscle in result.predicted_mpc_after:
            mpc = result.predicted_mpc_after[muscle]
            target = planner_config.get_target_zone(muscle)
            logger.info(f"    {muscle}: {mpc:.3f} → {state.get(muscle, 0):.2f} (target: {target})")

    return result


def test_replanning():
    """Test 3: Replanning - user odrzuca ćwiczenie"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Replanning (user rejects exercise)")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

    # Zaplanuj treninig
    original_plan = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
        user_history=[],
    )

    logger.info(f"\nOriginal plan:")
    for s in original_plan.plan:
        logger.info(f"  {s.order}. {s.exercise_id}")

    # User wykonał pierwsze 2 serie, ale odrzuca trzecie
    completed_sets = original_plan.plan[:2]
    remaining_compound = 0  # Już miał 2 compound
    remaining_isolation = 2  # Potrzebuje jeszcze 2 isolation

    # Uaktualnij state na podstawie wykonanych serii
    updated_history = []
    for s in completed_sets:
        updated_history.append(WorkoutSet(
            exercise_id=s.exercise_id,
            weight_kg=s.weight_kg,
            reps=s.reps,
            rir=None,
            timestamp=datetime.now() - timedelta(seconds=600),
        ))

    updated_state = state.copy()

    logger.info(f"\nReplanning after {len(completed_sets)} completed sets...")
    replanned = planner.replan(
        session_so_far=completed_sets,
        remaining_n_compound=remaining_compound,
        remaining_n_isolation=remaining_isolation,
        current_state=updated_state,
        available_time_sec=3600,
        user_history=updated_history,
    )

    logger.info(f"\nReplanned workout:")
    for s in replanned.plan:
        logger.info(f"  {s.order}. {s.exercise_id}")

    return replanned


def test_time_constraint():
    """Test 4: Time constraint - krótka sesja"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Time Constraint (short session)")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

    result = planner.plan(
        state=state,
        n_compound=3,
        n_isolation=3,
        available_time_sec=600,  # Tylko 10 minut!
        user_history=[],
    )

    logger.info(f"\n✓ Generated plan for 10-minute session")
    logger.info(f"  Sets: {len(result.plan)}")
    logger.info(f"  Actual time: {result.total_time_estimated_sec / 60:.1f} min")
    logger.info(f"  Exercises:")
    for s in result.plan:
        logger.info(f"    {s.order}. {s.exercise_id}")

    return result


def test_exercise_variety():
    """Test 5: Weryfikacja że wszystkie typy ćwiczeń się pojawiają"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Exercise Variety Check")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

    # Uruchom wiele scenariuszy
    all_exercises_used = set()

    for scenario_idx in range(5):
        # Różne starting states
        scenario_state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

        # Vary fatigue levels
        if scenario_idx % 2 == 0:
            scenario_state['quadriceps'] = 0.3
            scenario_state['hamstring'] = 0.2

        result = planner.plan(
            state=scenario_state,
            n_compound=2,
            n_isolation=2,
            available_time_sec=3600,
            user_history=[],
        )

        for s in result.plan:
            all_exercises_used.add(s.exercise_id)

    logger.info(f"\n✓ Across 5 scenarios, used {len(all_exercises_used)} unique exercises:")
    for ex_id in sorted(all_exercises_used):
        logger.info(f"    - {ex_id}")

    # Check coverage
    total_compound = len(planner.compound_exercises)
    total_isolation = len(planner.isolation_exercises)
    logger.info(f"\nExercise pool: {total_compound} compound, {total_isolation} isolation")
    logger.info(f"Coverage: {len(all_exercises_used) / (total_compound + total_isolation) * 100:.1f}%")

    return all_exercises_used


def test_with_user_history():
    """Test 6: Planning z historią użytkownika"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Planning with User History")
    logger.info("="*60)

    exercises_config, planner_config = load_config()
    planner = WorkoutPlanner(exercises_config, planner_config)

    # Simulate user history
    user_history = [
        WorkoutSet('back_squat', 80.0, 8, rir=2, timestamp=datetime.now() - timedelta(days=2)),
        WorkoutSet('back_squat', 80.0, 8, rir=2, timestamp=datetime.now() - timedelta(days=2)),
        WorkoutSet('bench_press', 60.0, 10, rir=1, timestamp=datetime.now() - timedelta(days=2)),
        WorkoutSet('bench_press', 60.0, 10, rir=1, timestamp=datetime.now() - timedelta(days=2)),
        WorkoutSet('leg_curl', 30.0, 12, rir=3, timestamp=datetime.now() - timedelta(days=2)),
    ]

    state = {m: 0.0 for m in planner_config.target_fatigue_zones.keys()}

    logger.info(f"\nUser history: {len(user_history)} sets from 2 days ago")

    result = planner.plan(
        state=state,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
        user_history=user_history,
    )

    logger.info(f"\n✓ Estimated 1RM from history:")
    estimated_1rm = planner.estimate_1rm_from_history(user_history)
    for ex_id, one_rm in sorted(estimated_1rm.items()):
        if ex_id in [s.exercise_id for s in user_history]:
            logger.info(f"    {ex_id}: ~{one_rm:.1f}kg")

    logger.info(f"\nGenerated plan:")
    for s in result.plan:
        logger.info(f"    {s.order}. {s.exercise_id}: {s.reps}x{s.weight_kg}kg")

    return result


def generate_report():
    """Wygeneruj raport z testów"""
    logger.info("\n\n" + "="*60)
    logger.info("RUNNING ALL TESTS")
    logger.info("="*60)

    test_fresh_user()
    test_fatigued_user()
    test_replanning()
    test_time_constraint()
    all_exercises = test_exercise_variety()
    test_with_user_history()

    logger.info("\n\n" + "="*60)
    logger.info("✓ ALL TESTS COMPLETED")
    logger.info("="*60)


if __name__ == '__main__':
    generate_report()
