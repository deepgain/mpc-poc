# Exercise Muscle Weights Transformation

Source files:
- Ratios - Arkusz1.csv
- exercise_muscle_order.yaml

Output files:
- exercise_muscle_weights_scaled.csv (wide, model-ready)
- exercise_muscle_weights_scaled_long.csv (audit table)

Method:
1. Start from YAML canonical exercise list (exercise_id, csv_title).
2. For each exercise, read matching CSV row by exact csv_title.
3. Aggregate CSV columns to model muscle groups:
   - chest = Klatka (Gora) + Klatka (Srodek/Dol)
   - erectors = Prostownik Ledzwiowy + Multifidus
   - glutes = Posladkowy Wielki + Posladkowy Sredni
   - abs = Prosty Brzucha + Skosny zewnetrzny + Skosny wewnetrzny
   - upper_traps and brachialis are set to 0.0 (no direct columns in current CSV schema)
   - other groups map 1:1 to their CSV columns
4. For aggregated groups only (chest, erectors, glutes, abs), apply factor:
   - factor = 0.5 + 0.1 * x, where x = number of source muscles in that sum
   - chest / erectors / glutes: x=2 -> factor=0.7
   - abs: x=3 -> factor=0.8
   - non-aggregated groups are copied directly (no extra factor)
5. Clamp negatives to 0.
6. Compute global clipping threshold as p99 over all positive transformed values.
7. Clip each transformed value to global_clip_p99.
8. Per exercise: normalize all clipped muscle values by that exercise max clipped value, yielding scale [0,1].

Rationale:
- One consistent scale for all records, not only rows with values >100.
- Preserves within-exercise ranking (most important signal for transfer/fatigue overlap).
- Reduces influence of outliers from mixed EMG protocols.

Computed constants:
- global_clip_p99 = 135.618800
- exercises exported = 34
- muscles exported per exercise = 17
