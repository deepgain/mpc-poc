import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import models_wrapper
models_wrapper.initialize_model('deepgain_model_best.pt')

from data_structures import PlannerConfig, DEFAULT_TARGET_CAPACITY_ZONES, DEFAULT_TARGET_REPS_BY_TYPE
from planner import WorkoutPlanner
from datetime import datetime

anchors = {'bench_press': 100.0, 'squat': 140.0, 'deadlift': 180.0}
config = PlannerConfig(
    target_capacity_zones=DEFAULT_TARGET_CAPACITY_ZONES,
    target_reps_by_type=DEFAULT_TARGET_REPS_BY_TYPE,
    strength_anchors=anchors,
    target_rir=2,
)
planner = WorkoutPlanner(config)
fresh = {m: 1.0 for m in planner.all_muscles}

result = planner.plan(state=fresh, n_compound=2, n_isolation=2, available_time_sec=3600)

print()
print('=== PLAN ===')
prev = None
for s in result.plan:
    if s.exercise_id != prev:
        print(f'\n  {s.exercise_id.upper()}  ({", ".join(s.primary_muscles)})')
        prev = s.exercise_id
    print(f'    Seria {s.order}: {s.reps} reps × {s.weight_kg}kg  (pred RIR={s.predicted_rir:.1f})')

print(f'\nŁączny czas: {result.total_time_estimated_sec//60} min')
print(f'Używa real model: {result.used_real_model}')

print('\nMPC po treningu:')
for m, v in sorted(result.predicted_mpc_after.items()):
    if v < 0.99:
        t = planner.config.get_target_zone(m)
        ok = '✓' if t[0] <= v <= t[1] else '⚠'
        print(f'  {ok} {m}: {v:.3f}  (target [{t[0]},{t[1]}])')
