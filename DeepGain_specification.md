# DeepGain — Muscle Performance Capacity Estimation from Training Logs

**deepgain.org** | Open-source standard for per-muscle fatigue tracking in resistance training.

---

## Problem

No existing system estimates per-muscle fatigue from minimal training logs (weight, reps, RIR). Current approaches require EMG sensors, force plates, or velocity encoders. DeepGain estimates **Muscle Performance Capacity (MPC)** per muscle group using only data every gym-goer already logs.

## MPC — Muscle Performance Capacity

$$MPC \in [0, 1]$$

- $MPC = 1.0$ → muscle at full capacity (fully recovered)
- $MPC = 0.5$ → muscle at 50% capacity
- MPC is **latent** — never directly observed, inferred from training data

Each muscle has its own MPC. Each (exercise, muscle) pair has its own learned dynamics.

---

## Data Standard

One row = one set performed by any user.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User identifier |
| `exercise` | string | Exercise name (e.g. `bench_press`) |
| `weight_kg` | float | Weight on bar (kg) |
| `reps` | int | Repetitions performed |
| `rir` | int | Repetitions in Reserve (0–5) |
| `timestamp` | datetime | When the set was performed |

RIR (Repetitions in Reserve) is used instead of RPE because it maps directly to proximity-to-failure — the mechanistically relevant variable for fatigue. RIR 0 = failure, RIR 1 = one rep left, etc. Integer scale 0–5 covers all productive training (Robinson 2024, Halperin 2022).

Minimum viable training example = **one workout with 2+ exercises sharing muscle groups**.

---

## Architecture

Three learned components:

### $f_{e,m}$ — MPC update (per exercise $e$, per muscle $m$)

$$MPC_m' = f_{e,m}(w, r, RIR, MPC_m)$$

After a set of exercise $e$, muscle $m$'s capacity drops. Each (exercise, muscle) pair has its own $f$ because bench press fatigues chest differently than it fatigues triceps.

### $g_e$ — RIR prediction (per exercise $e$)

$$\widehat{RIR} = g_e(w, r, MPC_{m_1}, MPC_{m_2}, \dots)$$

Predicts how many reps remain in reserve given current state of **all muscles involved** in exercise $e$. Bench press RIR depends on MPC of chest, triceps, and anterior delts simultaneously.

### $r_m$ — Recovery (per muscle $m$)

$$MPC_m' = r_m(MPC_m, \Delta t)$$

MPC recovers toward 1.0 over time. $r_m$ is a learned function — no assumed formula. The network may learn exponential decay, biphasic recovery (Raastad 2000), or something entirely different. Data decides.

---

## Training

### The key insight

MPC is never observed. RIR is always observed. **RIR prediction error is the only training signal** — but it trains all three components simultaneously.

### Forward pass (one workout)

```
Initialize: MPC = {chest: 1.0, triceps: 1.0, delts: 1.0, ...}

Set 1: bench 80kg × 10 @ RIR 3
  → g_bench(80, 10, 1.0, 1.0, 1.0)    → RIR_pred = 3.2
  → Loss: (3.2 - 3.0)² = 0.04
  → validates MPC₀ = 1.0 assumption (if RIR lower than expected → user came fatigued)
  → f_bench_chest(80, 10, 3.0, 1.0)   → MPC_chest = 0.91
  → f_bench_triceps(80, 10, 3.0, 1.0) → MPC_triceps = 0.94
  → f_bench_delts(80, 10, 3.0, 1.0)   → MPC_delts = 0.95

Set 2: bench 80kg × 8 @ RIR 2
  → g_bench(80, 8, 0.91, 0.94, 0.95)  → RIR_pred = 2.3
  → Loss: (2.3 - 2.0)² = 0.09
  → f updates MPC_chest, MPC_triceps, MPC_delts

Set 3: dips +20kg × 10 @ RIR 2
  → g_dips(20, 10, MPC_chest, MPC_triceps) → RIR_pred
  → Loss: (RIR_pred - 2.0)²
  → THIS IS THE KEY DATAPOINT:
    dips use chest + triceps (no delts)
    → forces model to decompose fatigue between muscles
  → f_dips_chest, f_dips_triceps update MPC

Set 4: OHP 40kg × 8 @ RIR 1
  → g_ohp(40, 8, MPC_delts, MPC_triceps) → RIR_pred
  → Loss: (RIR_pred - 1.0)²
  → OHP uses delts + triceps (no chest)
  → further triangulates per-muscle MPC
```

### Loss

$$\mathcal{L} = \sum_{i=1}^{n} (\widehat{RIR}_i - RIR_i)^2$$

Every set produces a loss — including the first (which validates the $MPC_0 = 1.0$ assumption).

### Backpropagation

RIR error at set $i$ propagates back through:
- $g_e$ (which produced the prediction)
- All $f_{e,m}$ from previous sets (which produced the MPC values that $g$ consumed)

$$\mathcal{L} \xrightarrow{\nabla} g \xrightarrow{\nabla} MPC_i \xrightarrow{\nabla} f_i \xrightarrow{\nabla} MPC_{i-1} \xrightarrow{\nabla} f_{i-1} \xrightarrow{\nabla} \dots$$

### What makes per-muscle MPC identifiable

**Overlapping muscle involvement across exercises within one workout.**

| Exercise | Muscles |
|----------|---------|
| bench press | chest + triceps + delts |
| dips | chest + triceps |
| OHP | delts + triceps |
| tricep pushdown | triceps |

If RIR on dips is high but RIR on OHP is low → triceps is fresh, delts are fatigued. The model **must** learn separate MPC values per muscle to minimize prediction error across all exercises.

Single-exercise workouts (only bench 5×5) provide **zero signal** for per-muscle decomposition. Multi-exercise workouts with overlapping muscles are essential.

### Training across users

All users train the **same** $f$ and $g$ networks. User A benching 60kg and User B benching 120kg both contribute to the same model. The network sees raw weight — it learns that heavier loads at same RIR drop MPC more.

### Recovery training

First set of a new workout validates recovery model $r$:

```
Monday: workout ends with MPC_chest = 0.72
Wednesday (Δt = 48h): first set bench → g predicts RIR
  → RIR error tells us if r(0.72, 48h) was correct
  → backprop adjusts τ_chest
```

Requires: user_id + timestamps (to link workouts of same user).

---

## Why this works

1. **RIR is the supervision signal** — available in every set, no special equipment, maps directly to proximity-to-failure
2. **Exercise overlap decomposes per-muscle MPC** — like solving N equations with N unknowns
3. **MPC is a learned latent variable** — emerges from training, not assumed from literature
4. **Recovery trains from inter-workout RIR** — first set of next workout validates recovery prediction
