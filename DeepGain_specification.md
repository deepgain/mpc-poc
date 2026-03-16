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

### $f_{e,m}$ — MPC update (per exercise $e$, per muscle $m$) — Neural Network

$$MPC_m' = MPC_m \cdot (1 - I_{e,m} \cdot f_{e,m}(w, r, RIR, MPC_m))$$

After a set of exercise $e$, muscle $m$'s capacity drops. $f$ is a neural network that outputs a drop fraction in $(0, 1)$, scaled by the fixed EMG-based involvement coefficient $I_{e,m}$.

$f$ is shared across all (exercise, muscle) pairs via learned embeddings — it takes exercise and muscle embeddings as input, allowing it to learn exercise-specific and muscle-specific fatigue patterns without separate networks.

**Inductive bias — fatigue ordering penalty**: The raw drop from $f$ should respect the involvement ordering. If exercise $e$ involves chest at 0.85 and triceps at 0.55, then $f$ should produce a larger drop for chest than triceps. This is enforced via a soft pairwise penalty during training:

$$\mathcal{L}_{f\text{-order}} = \frac{1}{|\text{pairs}|} \sum_{(a,b): I_a > I_b} \text{relu}(f_b - f_a)$$

### $g_e$ — RIR prediction (per exercise $e$) — Neural Network

$$\widehat{RIR} = g_e(w, r, MPC_{m_1}, MPC_{m_2}, \dots)$$

Predicts how many reps remain in reserve given current state of **all muscles involved** in exercise $e$. Bench press RIR depends on MPC of chest, triceps, and anterior delts simultaneously.

### $r_m$ — Recovery (per muscle $m$) — Exponential with learned $\tau_m$

$$MPC_m(t + \Delta t) = 1 - (1 - MPC_m(t)) \cdot e^{-\Delta t / \tau_m}$$

MPC recovers toward 1.0 over time with a learned per-muscle time constant $\tau_m$. This is **not** a neural network — it is an exponential decay formula with 16 learnable parameters (one $\tau$ per muscle).

**Path-consistent by construction**: $r(r(MPC, t_1), t_2) = r(MPC, t_1 + t_2)$. This means applying recovery in one big step or many small steps gives the same result — a property that neural network recovery functions cannot guarantee.

$\tau_m$ is interpretable: $\tau_{\text{chest}} = 22h$ means "chest recovers 63% of its deficit in 22 hours."

**Inductive bias — muscle size ordering penalty**: Smaller muscles should recover faster than larger muscles. This is enforced via a soft penalty on the group means of $\tau$:

$$\mathcal{L}_{\tau\text{-order}} = \text{relu}(\bar{\tau}_{\text{small}} - \bar{\tau}_{\text{medium}}) + \text{relu}(\bar{\tau}_{\text{medium}} - \bar{\tau}_{\text{large}})$$

Where:
- **Small**: triceps, biceps, brachialis, calves, rear delts, lateral delts
- **Medium**: chest, anterior delts, upper traps, rhomboids, lats, erectors
- **Large**: quads, hamstrings, glutes, adductors

Individual muscles within each group are free to take any value — only the group averages are constrained.

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
  → f_bench_chest(80, 10, 3.0, 1.0)   → MPC_chest = 0.91
  → f_bench_triceps(80, 10, 3.0, 1.0) → MPC_triceps = 0.94
  → f_bench_delts(80, 10, 3.0, 1.0)   → MPC_delts = 0.95

Set 2: bench 80kg × 8 @ RIR 2
  → g_bench(80, 8, 0.91, 0.94, 0.95)  → RIR_pred = 2.3
  → Loss: (2.3 - 2.0)² = 0.09
  → f updates MPC_chest, MPC_triceps, MPC_delts

Set 3: dips +20kg × 10 @ RIR 2
  → g_dips(20, 10, MPC_chest, MPC_triceps) → RIR_pred
  → THIS IS THE KEY DATAPOINT:
    dips use chest + triceps (no delts)
    → forces model to decompose fatigue between muscles

Set 4: OHP 40kg × 8 @ RIR 1
  → g_ohp(40, 8, MPC_delts, MPC_triceps) → RIR_pred
  → OHP uses delts + triceps (no chest)
  → further triangulates per-muscle MPC
```

### Loss

$$\mathcal{L} = \underbrace{\sum_{i=1}^{n} (\widehat{RIR}_i - RIR_i)^2}_{\text{RIR prediction}} + \lambda_1 \cdot \underbrace{\mathcal{L}_{\tau\text{-order}}}_{\text{recovery ordering}} + \lambda_2 \cdot \underbrace{\mathcal{L}_{f\text{-order}}}_{\text{fatigue ordering}}$$

Where $\lambda_1 = \lambda_2 = 0.01$ (soft penalties — the model can violate them if the data strongly disagrees).

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

---

## Inductive Biases (what we assume vs what we learn)

| Component | What is assumed | What is learned |
|-----------|----------------|-----------------|
| **f** (fatigue) | Involvement matrix from EMG literature; fatigue ordering matches involvement ordering | Drop magnitude as function of weight, reps, RIR, current MPC |
| **g** (RIR prediction) | Nothing — pure neural network | How MPC state maps to perceived effort |
| **r** (recovery) | Exponential shape; small muscles recover faster than large muscles (on average) | Per-muscle τ values |
| **MPC** | Starts at 1.0; bounded in [0.1, 1.0] | Everything else — MPC is a latent variable that emerges from training |

---

## Why this works

1. **RIR is the supervision signal** — available in every set, no special equipment, maps directly to proximity-to-failure
2. **Exercise overlap decomposes per-muscle MPC** — like solving N equations with N unknowns
3. **MPC is a learned latent variable** — emerges from training, not assumed from literature
4. **Recovery is path-consistent** — exponential formula guarantees mathematical consistency
5. **Soft inductive biases** — ordering penalties guide training without constraining final values
