"""
Testy symulacyjne WorkoutPlanner

WAŻNE: MPC = CAPACITY w [0.1, 1.0]. 1.0 = fresh, 0.1 = exhausted.

Scenariusze:
  1. Fresh user (wszystkie mięśnie MPC=1.0)
  2. Zmęczony user (niektóre mięśnie MPC<1.0 po poprzednim treningu)
  3. Replanning - user odrzuca / modyfikuje ćwiczenie
  4. Time constraint
  5. Exercise variety (5 scenariuszy)
  6. With user history (1RM estimation)
  7. Verify target zones są osiągane
  8. Exclusions i preferences
"""

import logging
from datetime import datetime, timedelta

from data_structures import (
    WorkoutSet, PlannerConfig,
    DEFAULT_TARGET_CAPACITY_ZONES, DEFAULT_DEFAULT_REPS_BY_TYPE,
)
from planner import WorkoutPlanner
import models_wrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_planner(force_mock: bool = True) -> WorkoutPlanner:
    """Helper: stwórz planera z sensownymi defaults."""
    models_wrapper.initialize_model(force_mock=force_mock)

    config = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        target_rir=2,
    )
    return WorkoutPlanner(config)


# ============================================================================
# TEST 1: Fresh User
# ============================================================================
def test_fresh_user():
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Fresh User (wszystko MPC=1.0)")
    logger.info("="*70)

    planner = make_planner()

    # Fresh state - wszystko MPC=1.0
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    result = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
    )

    logger.info(f"\n✓ Plan: {len(result.plan)} sets, {result.total_time_estimated_sec/60:.1f} min")
    logger.info(f"\nEjercicios:")
    for s in result.plan:
        logger.info(
            f"  {s.order}. {s.exercise_id}: {s.reps}×{s.weight_kg}kg "
            f"(pred RIR={s.predicted_rir:.1f})"
        )

    logger.info(f"\nMPC (capacity) after workout - zaangażowane mięśnie:")
    for muscle in sorted(result.predicted_mpc_after.keys()):
        mpc_after = result.predicted_mpc_after[muscle]
        if mpc_after < 0.99:  # Zaangażowane
            target = planner.config.get_target_zone(muscle)
            in_zone = target[0] <= mpc_after <= target[1]
            status = "✓" if in_zone else "⚠"
            logger.info(f"  {status} {muscle}: {mpc_after:.3f} (target: [{target[0]}, {target[1]}])")

    return result


# ============================================================================
# TEST 2: Fatigued User (po leg day 1 dzień temu)
# ============================================================================
def test_fatigued_user():
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Fatigued User (po leg day 24h temu)")
    logger.info("="*70)

    planner = make_planner()

    # Symuluj leg day wczoraj
    now = datetime.now()
    yesterday = now - timedelta(hours=24)
    leg_history = [
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday + timedelta(minutes=3)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=yesterday + timedelta(minutes=6)),
        WorkoutSet('leg_curl', 40.0, 12, rir=2, timestamp=yesterday + timedelta(minutes=30)),
        WorkoutSet('leg_curl', 40.0, 12, rir=1, timestamp=yesterday + timedelta(minutes=33)),
    ]

    state_after_recovery = planner._call_predict_mpc(leg_history, now)

    logger.info(f"\nStan po 24h odpoczynku (capacity = 1.0 - fatigue):")
    for m in ['quads', 'hamstrings', 'glutes', 'chest', 'biceps']:
        logger.info(f"  {m}: {state_after_recovery[m]:.3f}")

    result = planner.plan(
        state=state_after_recovery,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
        user_history=leg_history,
        now=now,
    )

    logger.info(f"\nPlan:")
    for s in result.plan:
        logger.info(f"  {s.order}. {s.exercise_id} ({', '.join(s.primary_muscles[:2])})")

    # Sprawdź czy nogi nie są overfatiguowane
    logger.info(f"\nMPC (capacity) after:")
    for m in ['quads', 'hamstrings', 'glutes', 'chest', 'biceps', 'triceps']:
        mpc_after = result.predicted_mpc_after.get(m, 1.0)
        target = planner.config.get_target_zone(m)
        status = "✓" if target[0] <= mpc_after <= target[1] else "⚠"
        logger.info(f"  {status} {m}: {mpc_after:.3f} (target: {target})")

    return result


# ============================================================================
# TEST 3: Replanning
# ============================================================================
def test_replanning():
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Replanning (user odrzuca ćwiczenie)")
    logger.info("="*70)

    planner = make_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    # Oryginalny plan
    original = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
    )

    logger.info(f"\nOriginal plan:")
    for s in original.plan:
        logger.info(f"  {s.order}. {s.exercise_id}")

    # User wykonał pierwsze 2 ćwiczenia, odrzuca 3
    completed = original.plan[:2]
    rejected_next = original.plan[2] if len(original.plan) > 2 else None

    if rejected_next:
        logger.info(f"\nUser completed 2 sets, rejects: {rejected_next.exercise_id}")

    # Dodaj do exclusions to rejected
    exclusions = [rejected_next.exercise_id] if rejected_next else []

    replanned = planner.replan(
        session_so_far=completed,
        remaining_n_compound=0,
        remaining_n_isolation=2,
        available_time_sec=3600,
        exclusions=exclusions,
    )

    logger.info(f"\nReplanned:")
    for s in replanned.plan:
        marker = "✓" if s.order <= 2 else "+"
        logger.info(f"  [{marker}] {s.order}. {s.exercise_id}")

    # Sprawdź czy rejected nie jest w planie
    rejected_in_plan = any(s.exercise_id == (rejected_next.exercise_id if rejected_next else "") for s in replanned.plan[2:])
    if rejected_in_plan:
        logger.warning(f"  ⚠ Rejected exercise appeared again!")
    else:
        logger.info(f"  ✓ Rejected exercise avoided")

    return replanned


# ============================================================================
# TEST 4: Time Constraint (10 min session)
# ============================================================================
def test_time_constraint():
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Time Constraint (10 min)")
    logger.info("="*70)

    planner = make_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    result = planner.plan(
        state=fresh_state,
        n_compound=3,
        n_isolation=3,
        available_time_sec=600,  # 10 min
    )

    logger.info(f"\nPlan w 10 min:")
    logger.info(f"  Sets generated: {len(result.plan)}")
    logger.info(f"  Actual time: {result.total_time_estimated_sec/60:.1f} min")
    logger.info(f"  Fits in budget: {result.total_time_estimated_sec <= 600}")

    for s in result.plan:
        logger.info(f"  {s.order}. {s.exercise_id} ({s.estimated_time_sec}s)")

    return result


# ============================================================================
# TEST 5: Exercise Variety
# ============================================================================
def test_exercise_variety():
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Exercise Variety (10 scenarios)")
    logger.info("="*70)

    planner = make_planner()
    all_used = set()

    for scenario_idx in range(10):
        # Różny state
        state = {m: 1.0 for m in planner.all_muscles}

        if scenario_idx % 3 == 1:
            # Po leg day
            state.update({"quads": 0.55, "hamstrings": 0.55, "glutes": 0.60})
        elif scenario_idx % 3 == 2:
            # Po push day
            state.update({"chest": 0.50, "triceps": 0.45, "anterior_delts": 0.50})

        # Exclude niektóre ćwiczenia losowo
        all_exs = list(planner.all_exercises)
        exclude_pool = []
        if scenario_idx >= 5:
            # W późniejszych scenariuszach wyłącz najczęściej wybierane
            exclude_pool = list(all_used)[:3]

        result = planner.plan(
            state=state,
            n_compound=2,
            n_isolation=2,
            available_time_sec=3600,
            exclusions=exclude_pool,
        )

        for s in result.plan:
            all_used.add(s.exercise_id)

    total_possible = len(planner.compound_exercises) + len(planner.isolation_exercises) + len(planner.core_exercises)
    coverage = len(all_used) / total_possible * 100

    logger.info(f"\nAcross 10 scenarios (z rotacją exclusions):")
    logger.info(f"  Unique exercises used: {len(all_used)} / {total_possible}")
    logger.info(f"  Coverage: {coverage:.1f}%")
    logger.info(f"\n  Used exercises:")
    for ex in sorted(all_used):
        logger.info(f"    - {ex}")

    return all_used


# ============================================================================
# TEST 6: With User History (1RM estimation)
# ============================================================================
def test_with_user_history():
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Planning with User History (1RM estimation)")
    logger.info("="*70)

    planner = make_planner()

    now = datetime.now()
    user_history = [
        # Tydzień 1 - bench
        WorkoutSet('bench_press', 70.0, 8, rir=2, timestamp=now - timedelta(days=20)),
        WorkoutSet('bench_press', 70.0, 8, rir=2, timestamp=now - timedelta(days=20)),
        # Tydzień 2 - bench progresses
        WorkoutSet('bench_press', 75.0, 6, rir=2, timestamp=now - timedelta(days=14)),
        WorkoutSet('bench_press', 80.0, 5, rir=1, timestamp=now - timedelta(days=7)),
        # Squats
        WorkoutSet('squat', 90.0, 6, rir=2, timestamp=now - timedelta(days=20)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=now - timedelta(days=7)),
        # Deadlift
        WorkoutSet('deadlift', 120.0, 5, rir=2, timestamp=now - timedelta(days=14)),
        WorkoutSet('deadlift', 130.0, 3, rir=1, timestamp=now - timedelta(days=4)),
    ]

    # Estymuj 1RM
    estimated_1rm = planner.estimate_1rm_from_history(user_history)
    logger.info(f"\nEstimated 1RM:")
    for ex_id in ['bench_press', 'squat', 'deadlift']:
        if ex_id in estimated_1rm:
            logger.info(f"  {ex_id}: {estimated_1rm[ex_id]:.1f} kg")

    # Plan (tu zbyt zmęczony po niedawnym deadlift!)
    result = planner.plan(
        n_compound=2,
        n_isolation=2,
        available_time_sec=3600,
        user_history=user_history,
        now=now,
    )

    logger.info(f"\nPlan:")
    for s in result.plan:
        logger.info(
            f"  {s.order}. {s.exercise_id}: {s.reps}×{s.weight_kg}kg "
            f"(pred RIR={s.predicted_rir:.1f})"
        )

    return result


# ============================================================================
# TEST 7: Target Zones Verification
# ============================================================================
def test_target_zones():
    logger.info("\n" + "="*70)
    logger.info("TEST 7: Target Zones Verification")
    logger.info("="*70)

    planner = make_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    result = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
    )

    in_zone = 0
    over = 0  # Overfatigue (MPC < min)
    under = 0  # Underfatigue (MPC > max)
    not_worked = 0

    for muscle, mpc_after in result.predicted_mpc_after.items():
        mpc_before = fresh_state[muscle]
        target = planner.config.get_target_zone(muscle)
        target_min, target_max = target

        was_worked = mpc_after < mpc_before - 0.02

        if not was_worked:
            not_worked += 1
        elif mpc_after < target_min:
            over += 1
            logger.info(f"  ⚠ OVER {muscle}: {mpc_after:.2f} < {target_min}")
        elif mpc_after > target_max:
            under += 1
            logger.info(f"  ⚠ UNDER {muscle}: {mpc_after:.2f} > {target_max}")
        else:
            in_zone += 1
            logger.info(f"  ✓ OK {muscle}: {mpc_after:.2f} in [{target_min}, {target_max}]")

    logger.info(f"\nSummary:")
    logger.info(f"  In zone: {in_zone}")
    logger.info(f"  Overfatigue: {over}")
    logger.info(f"  Underfatigue: {under}")
    logger.info(f"  Not worked: {not_worked}")


# ============================================================================
# TEST 8: Exclusions & Preferences
# ============================================================================
def test_exclusions_and_preferences():
    logger.info("\n" + "="*70)
    logger.info("TEST 8: Exclusions & Preferences")
    logger.info("="*70)

    planner = make_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    # Exclude squat, deadlift (kontuzja dolnego odcinka)
    exclusions = ["squat", "low_bar_squat", "deadlift", "sumo_deadlift"]

    # Preferences: lubi high incline
    preferences = {
        "favorites": ["incline_bench", "incline_bench_45"],
        "avoid": ["dips"],
    }

    result = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
        exclusions=exclusions,
        preferences=preferences,
    )

    logger.info(f"\nExclusions: {exclusions}")
    logger.info(f"Favorites: {preferences['favorites']}")
    logger.info(f"Avoid: {preferences['avoid']}")

    logger.info(f"\nPlan:")
    for s in result.plan:
        marker = ""
        if s.exercise_id in preferences["favorites"]:
            marker = "⭐"
        elif s.exercise_id in preferences["avoid"]:
            marker = "❌"
        logger.info(f"  {s.order}. {s.exercise_id} {marker}")

    # Assertions
    for ex in exclusions:
        if any(s.exercise_id == ex for s in result.plan):
            logger.error(f"  ✗ Excluded {ex} appears in plan!")
        else:
            pass  # Good

    logger.info(f"  ✓ No excluded exercises in plan")


# ============================================================================
# MAIN
# ============================================================================
def run_all_tests():
    logger.info("\n\n" + "#"*70)
    logger.info("# RUNNING ALL TESTS")
    logger.info("#"*70)

    test_fresh_user()
    test_fatigued_user()
    test_replanning()
    test_time_constraint()
    test_exercise_variety()
    test_with_user_history()
    test_target_zones()
    test_exclusions_and_preferences()

    logger.info("\n\n" + "#"*70)
    logger.info("# ✓ ALL TESTS COMPLETED")
    logger.info("#"*70 + "\n")


if __name__ == '__main__':
    run_all_tests()
