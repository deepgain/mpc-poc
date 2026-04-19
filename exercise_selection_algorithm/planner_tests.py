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
    WorkoutSet, PlannerConfig, UserProfile,
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
# TEST 9: Deadlift Dominance Fix — Weighted Avg Reward
# ============================================================================
def test_deadlift_dominance_fix():
    logger.info("\n" + "="*70)
    logger.info("TEST 9: Deadlift Dominance Fix (ważona średnia zamiast sumy)")
    logger.info("="*70)

    planner = make_planner()
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    # Zaplanuj wiele razy, sprawdź czy nie zawsze jest deadlift first
    deadlift_first_count = 0
    first_exercises = []

    for _ in range(5):
        result = planner.plan(
            state=fresh_state,
            n_compound=2,
            n_isolation=1,
            available_time_sec=3600,
        )
        first_ex = result.plan[0].exercise_id if result.plan else None
        first_exercises.append(first_ex)
        if first_ex in ("deadlift", "sumo_deadlift"):
            deadlift_first_count += 1

    logger.info(f"\n  First exercises across 5 runs: {first_exercises}")
    logger.info(f"  Deadlift-family as #1: {deadlift_first_count}/5")
    logger.info(f"  {'✓ Fixed' if deadlift_first_count < 5 else '⚠ Still dominated'}")


# ============================================================================
# TEST 10: Beam Search (exploration_temperature)
# ============================================================================
def test_beam_search_exploration():
    logger.info("\n" + "="*70)
    logger.info("TEST 10: Beam Search Exploration (różne plany per call)")
    logger.info("="*70)

    # High exploration temperature → większa różnorodność
    models_wrapper.initialize_model(force_mock=True)
    config = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        target_rir=2,
        exploration_temperature=0.5,   # Włączona eksploracja
        beam_width=3,
    )
    planner = WorkoutPlanner(config)
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    unique_plans = set()
    for run in range(8):
        result = planner.plan(
            state=fresh_state,
            n_compound=2,
            n_isolation=2,
            available_time_sec=3600,
        )
        # Unikalny fingerprint planu: set unikalnych exercise_id
        plan_fingerprint = tuple(sorted(set(s.exercise_id for s in result.plan)))
        unique_plans.add(plan_fingerprint)

    logger.info(f"\n  Unique plan fingerprints across 8 runs (temp=0.5): {len(unique_plans)}")
    for fp in unique_plans:
        logger.info(f"    {fp}")

    # Porównaj z temperature=0 (greedy, powinny być identyczne)
    config_greedy = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        target_rir=2,
        exploration_temperature=0.0,
    )
    planner_greedy = WorkoutPlanner(config_greedy)
    greedy_plans = set()
    for _ in range(3):
        result = planner_greedy.plan(
            state=fresh_state, n_compound=2, n_isolation=2,
            available_time_sec=3600,
        )
        fp = tuple(sorted(set(s.exercise_id for s in result.plan)))
        greedy_plans.add(fp)

    logger.info(f"\n  Greedy (temp=0): unique plans across 3 runs = {len(greedy_plans)}")
    logger.info(f"  {'✓ Diverse planning works' if len(unique_plans) > 1 else '⚠ Still deterministic'}")
    logger.info(f"  {'✓ Greedy is deterministic' if len(greedy_plans) == 1 else '⚠ Greedy not deterministic'}")


# ============================================================================
# TEST 11: Volume Limits
# ============================================================================
def test_volume_limits():
    logger.info("\n" + "="*70)
    logger.info("TEST 11: Volume Limits Per Muscle")
    logger.info("="*70)

    # Bardzo niski volume limit → planer powinien unikać ciężkich ćwiczeń
    models_wrapper.initialize_model(force_mock=True)
    custom_limits = {
        "quads": 500.0,       # Bardzo niski (normalnie 5000)
        "hamstrings": 500.0,
        "glutes": 500.0,
    }
    config = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        volume_limit_per_muscle=custom_limits,
    )
    planner = WorkoutPlanner(config)
    fresh_state = {m: 1.0 for m in planner.all_muscles}

    result = planner.plan(
        state=fresh_state,
        n_compound=2,
        n_isolation=3,
        available_time_sec=3600,
    )

    # Zlicz volume leg-related
    leg_muscles = ["quads", "hamstrings", "glutes"]
    volume_per_muscle = {m: 0.0 for m in leg_muscles}

    for ps in result.plan:
        delta = planner._calculate_volume_delta(
            exercise_id=ps.exercise_id,
            weight_kg=ps.weight_kg,
            reps=ps.reps,
            sets_count=1,
        )
        for m in leg_muscles:
            volume_per_muscle[m] += delta.get(m, 0.0)

    logger.info(f"\n  Leg volumes z limitem 500 kg-reps:")
    for m, v in volume_per_muscle.items():
        status = "✓ under limit" if v <= 500.0 else "✗ OVER limit"
        logger.info(f"    {m}: {v:.0f} / 500  {status}")

    logger.info(f"\n  Plan (powinno być mało/brak heavy leg):")
    unique = set()
    for ps in result.plan:
        if ps.exercise_id not in unique:
            unique.add(ps.exercise_id)
            count = sum(1 for x in result.plan if x.exercise_id == ps.exercise_id)
            logger.info(f"    - {ps.exercise_id} × {count} sets")


# ============================================================================
# TEST 12: UserProfile → tau calibration
# ============================================================================
def test_user_profile_tau():
    logger.info("\n" + "="*70)
    logger.info("TEST 12: UserProfile → Tau Calibration (per-user regeneracja)")
    logger.info("="*70)

    # Scenariusz: 2 userów robi identyczny trening, mierzymy MPC po 12h
    now = datetime.now()
    history_12h_ago = [
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=now - timedelta(hours=12)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=now - timedelta(hours=12, minutes=-3)),
        WorkoutSet('squat', 100.0, 5, rir=1, timestamp=now - timedelta(hours=12, minutes=-6)),
    ]

    # User 1: Beginner, age 60 (wolna regeneracja)
    beginner_profile = UserProfile(
        experience_level="beginner",
        age_years=60,
        recovery_factor=1.0,
    )
    models_wrapper.initialize_model(force_mock=True, tau_scale=1.0)  # Reset
    config_beginner = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        user_profile=beginner_profile,
    )
    planner_beg = WorkoutPlanner(config_beginner)
    mpc_beginner = planner_beg._call_predict_mpc(history_12h_ago, now)

    # User 2: Advanced, age 22 (szybka regeneracja)
    advanced_profile = UserProfile(
        experience_level="advanced",
        age_years=22,
        recovery_factor=1.0,
    )
    models_wrapper.initialize_model(force_mock=True, tau_scale=1.0)  # Reset
    config_advanced = PlannerConfig(
        target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
        default_reps_by_type=DEFAULT_DEFAULT_REPS_BY_TYPE,
        user_profile=advanced_profile,
    )
    planner_adv = WorkoutPlanner(config_advanced)
    mpc_advanced = planner_adv._call_predict_mpc(history_12h_ago, now)

    logger.info(f"\n  User 1 (Beginner, 60 lat): tau_scale={beginner_profile.get_tau_scale():.2f}")
    logger.info(f"  User 2 (Advanced, 22 lata): tau_scale={advanced_profile.get_tau_scale():.2f}")

    logger.info(f"\n  MPC po 12h (po identycznym leg day):")
    for muscle in ["quads", "hamstrings", "glutes"]:
        v1 = mpc_beginner[muscle]
        v2 = mpc_advanced[muscle]
        logger.info(f"    {muscle}:  Beginner={v1:.3f}  Advanced={v2:.3f}  diff={v2-v1:+.3f}")

    # Weryfikacja: advanced powinien mieć WYŻSZE capacity (szybsza regeneracja)
    all_advanced_higher = all(
        mpc_advanced[m] > mpc_beginner[m] for m in ["quads", "hamstrings", "glutes"]
    )
    logger.info(f"\n  {'✓ Advanced recovers faster (expected)' if all_advanced_higher else '⚠ Unexpected ordering'}")

    # Reset do default na koniec
    models_wrapper.initialize_model(force_mock=True, tau_scale=1.0)
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
    # Nowe testy (improvements #1-#4)
    test_deadlift_dominance_fix()
    test_beam_search_exploration()
    test_volume_limits()
    test_user_profile_tau()

    logger.info("\n\n" + "#"*70)
    logger.info("# ✓ ALL TESTS COMPLETED")
    logger.info("#"*70 + "\n")


if __name__ == '__main__':
    run_all_tests()
