# Muscle Aggregation README

This file documents the aggregation formulas used when collapsing multiple raw
EMG readings into one modeled muscle channel.

## Source of truth

Numeric exercise weights used by training, inference, eval and the dataset
generator live in:

- `exercise_muscle_weights_scaled.csv`

These files load those numeric weights directly from the CSV:

- `train.py`
- `inference.py`
- `eval_ordering.py`
- `generate_training_data.py`

`generate_training_data.py` also contains a hardcoded fallback map that is only
used when YAML loading is unavailable. It has been synced to the same updated
values so there is no hidden mismatch.

## Aggregate definitions

The repo explicitly documents these merged muscle channels in
`exercise_muscle_order.yaml`:

- `chest = upper chest + mid/lower chest`
- `erectors = erector spinae + multifidus`
- `glutes = glute max + glute med`
- `abs = rectus abdominis + external oblique + internal oblique`

`x` below means the fixed number of source muscles inside the aggregate, not the
number of non-zero readings in one particular exercise.

## Current formulas

### Default aggregate rule

For unchanged aggregates:

```text
aggregate = sum(raw_signals) * (0.5 + 0.10 * x)
```

Applied currently to:

- `glutes`
- all other unchanged merged channels

Examples:

- `chest` default coefficient would be `0.5 + 0.10 * 2 = 0.70`
- `erectors` old/default coefficient would be `0.70`
- `abs` old/default coefficient would be `0.80`

### Custom rule for bench press chest only

For `bench_press` chest aggregation only:

```text
bench_press_chest = sum(chest_subsignals) * (0.5 + 0.07 * x)
```

With `x = 2`, the new bench-only chest coefficient is:

```text
0.5 + 0.07 * 2 = 0.64
```

This tempers chest dominance specifically for flat bench without changing the
rest of the chest-family.

### Custom rule for erectors

For all exercises using the `erectors` aggregate:

```text
erectors = sum(raw_signals) * (0.5 + 0.09 * x)
```

With `x = 2`, the new erectors coefficient is:

```text
0.5 + 0.09 * 2 = 0.68
```

### Custom rule for abs

For all exercises using the `abs` aggregate:

```text
abs = sum(raw_signals) * (0.5 + 0.08 * x)
```

With `x = 3`, the new abs coefficient is:

```text
0.5 + 0.08 * 3 = 0.74
```

## Exercises affected by the custom rules

### Bench press chest-only custom rule

- `bench_press`

### Erectors custom rule

- `bird_dog`
- `bulgarian_split_squat`
- `deadlift`
- `farmers_walk`
- `leg_raises`
- `low_bar_squat`
- `pendlay_row`
- `seal_row`
- `squat`
- `suitcase_carry`
- `sumo_deadlift`
- `trx_bodysaw`

### Abs custom rule

- `ab_wheel`
- `dead_bug`
- `dips`
- `farmers_walk`
- `leg_raises`
- `plank`
- `suitcase_carry`
- `sumo_deadlift`
- `trx_bodysaw`

## Notes

- YAML ordering (`exercise_muscle_order.yaml`) was kept unchanged because these
  updates changed magnitudes, not the intended primary/secondary/tertiary
  ordering.
- The CSV remains the numeric source of truth for all runtime paths.
