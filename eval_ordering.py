#!/usr/bin/env python3
"""Compute ordering accuracy per exercise from a saved DeepGain checkpoint.

Usage:
    python eval_ordering.py                          # uses deepgain_model_best.pt
    python eval_ordering.py deepgain_model_best.pt
"""

import sys
import math
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "deepgain_model_muscle_ord.pt"

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Hyperparameters (must match train.py) ─────────────────────────────────────
EMBED_DIM  = 32
HIDDEN_DIM = 128
DT_SCALE   = math.log1p(168.0)

# ── Muscles ───────────────────────────────────────────────────────────────────
ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]
NUM_MUSCLES  = len(ALL_MUSCLES)
MUSCLE_TO_IDX = {m: i for i, m in enumerate(ALL_MUSCLES)}

# ── Exercise loading (same priority as train.py: CSV → YAML rank-derived) ─────
import os, pandas as pd

def _load_scaled_weights(csv_path="exercise_muscle_weights_scaled.csv"):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if any(m not in df.columns for m in ALL_MUSCLES):
        return {}
    weights = {}
    for _, row in df.iterrows():
        ex_id = str(row["exercise_id"])
        ex_w = {m: float(np.clip(row[m], 0.0, 1.0)) for m in ALL_MUSCLES if float(row[m]) > 0.0}
        if ex_w:
            weights[ex_id] = ex_w
    return weights

def _load_exercise_data(yaml_path="exercise_muscle_order.yaml"):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    scaled_weights = _load_scaled_weights()
    exercise_muscles = {}
    exercise_ordinal = {}

    for ex_id, ex_data in data["exercises"].items():
        if not isinstance(ex_data, dict):
            continue
        ranked = (
            (ex_data.get("primary_muscles") or [])
            + (ex_data.get("secondary_muscles") or [])
            + (ex_data.get("tertiary_muscles") or [])
        )
        valid_ranked = [m for m in ranked if m in MUSCLE_TO_IDX]
        if not valid_ranked:
            continue
        exercise_ordinal[ex_id] = valid_ranked

        if ex_id in scaled_weights:
            exercise_muscles[ex_id] = scaled_weights[ex_id]
        else:
            exercise_muscles[ex_id] = {m: max(1.0 - 0.15 * r, 0.3) for r, m in enumerate(valid_ranked)}

    return exercise_muscles, exercise_ordinal

EXERCISE_MUSCLES, EXERCISE_ORDINAL = _load_exercise_data()
ALL_EXERCISES = list(EXERCISE_MUSCLES.keys())
NUM_EXERCISES = len(ALL_EXERCISES)

# ── Model definition (must match train.py) ────────────────────────────────────
class FatigueNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 2 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
    def forward(self, weight, reps, rir, mpc, e_embed, m_embed):
        x = torch.cat([weight.unsqueeze(-1), reps.unsqueeze(-1),
                       rir.unsqueeze(-1), mpc.unsqueeze(-1),
                       e_embed, m_embed], dim=-1)
        return self.net(x).squeeze(-1)

class RIRNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_muscles):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + embed_dim + num_muscles, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, weight, reps, e_embed, mpc_all):
        x = torch.cat([weight.unsqueeze(-1), reps.unsqueeze(-1), e_embed, mpc_all], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

class ExponentialRecovery(nn.Module):
    FIXED_TAU = [16.0, 13.0, 9.0, 8.0, 10.0, 9.0, 13.0, 13.0,
                 19.0, 18.0, 15.0, 12.0, 12.0, 8.0, 10.0]
    def __init__(self, num_muscles):
        super().__init__()
        init_tau = torch.tensor([math.log(t) for t in self.FIXED_TAU])
        self.log_tau = nn.Parameter(init_tau, requires_grad=False)
    def forward(self, mpc, delta_t, muscle_idx):
        dt_hours = torch.expm1(delta_t * DT_SCALE)
        tau = torch.exp(self.log_tau[muscle_idx])
        return 1.0 - (1.0 - mpc) * torch.exp(-dt_hours / tau)

INVOLVEMENT_MATRIX = np.zeros((NUM_EXERCISES, NUM_MUSCLES), dtype=np.float32)
for ei, ex_id in enumerate(ALL_EXERCISES):
    for m_id, coeff in EXERCISE_MUSCLES[ex_id].items():
        if m_id in MUSCLE_TO_IDX:
            INVOLVEMENT_MATRIX[ei, MUSCLE_TO_IDX[m_id]] = coeff

class DeepGainModel(nn.Module):
    def __init__(self, num_exercises, num_muscles, embed_dim, hidden_dim):
        super().__init__()
        self.num_muscles  = num_muscles
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.muscle_embed   = nn.Embedding(num_muscles, embed_dim)
        self.f_net = FatigueNet(embed_dim, hidden_dim)
        self.g_net = RIRNet(embed_dim, hidden_dim, num_muscles)
        self.r     = ExponentialRecovery(num_muscles)
        self.register_buffer("involvement",
                             torch.tensor(INVOLVEMENT_MATRIX, dtype=torch.float32))

# ── Load checkpoint ───────────────────────────────────────────────────────────
model = DeepGainModel(NUM_EXERCISES, NUM_MUSCLES, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
print(f"Loaded: {CHECKPOINT}  ({NUM_EXERCISES} exercises, {NUM_MUSCLES} muscles)")

# ── Compute ordering accuracy ─────────────────────────────────────────────────
print(f"\nOrdering accuracy per exercise (% pairs where primary muscle drops more):")
print(f"  probe: w=0.4 (~80kg), r=0.27 (~8 reps), rir=0.4 (~RIR 2), mpc=1.0\n")

results = {}
with torch.no_grad():
    w   = torch.tensor([0.4],  dtype=torch.float32, device=DEVICE)
    r   = torch.tensor([0.27], dtype=torch.float32, device=DEVICE)
    rir = torch.tensor([0.4],  dtype=torch.float32, device=DEVICE)
    mpc = torch.tensor([1.0],  dtype=torch.float32, device=DEVICE)

    for ei, ex_id in enumerate(ALL_EXERCISES):
        ordinal = EXERCISE_ORDINAL.get(ex_id, [])
        if len(ordinal) < 2:
            continue
        e_embed = model.exercise_embed(torch.tensor([ei], device=DEVICE))
        drops = {}
        for m_id in ordinal:
            if m_id not in MUSCLE_TO_IDX:
                continue
            mi = MUSCLE_TO_IDX[m_id]
            m_embed = model.muscle_embed(torch.tensor([mi], device=DEVICE))
            drops[m_id] = model.f_net(w, r, rir, mpc, e_embed, m_embed).item()

        ranked = [m for m in ordinal if m in drops]
        if len(ranked) < 2:
            continue

        correct = total = 0
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                if drops[ranked[i]] > drops[ranked[j]]:
                    correct += 1
                total += 1

        results[ex_id] = (correct / total, ranked, drops)

for ex_id, (acc, ranked, drops) in sorted(results.items(), key=lambda x: x[1][0]):
    drop_str = " > ".join(f"{m[:4]}({drops[m]:.3f})" for m in ranked)
    print(f"  {ex_id:25s}: {acc*100:5.0f}%  [{drop_str}]")

overall = np.mean([v[0] for v in results.values()])
print(f"\n  {'MEAN':25s}: {overall*100:.0f}%")
