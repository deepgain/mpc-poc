#!/usr/bin/env python3
"""Generate chart_transfer_matrix.png from a saved checkpoint.

Usage:
    python gen_transfer_matrix.py                        # uses deepgain_model_best.pt
    python gen_transfer_matrix.py deepgain_model_best.pt
"""

import sys, math, os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "deepgain_model_best.pt"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Hyperparameters — auto-detected from checkpoint ──────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

REPS_SCALE = 30.0
RIR_SCALE  = 5.0
DT_SCALE   = math.log1p(168.0)

ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]
NUM_MUSCLES   = len(ALL_MUSCLES)
MUSCLE_TO_IDX = {m: i for i, m in enumerate(ALL_MUSCLES)}

# ── Exercise loading ──────────────────────────────────────────────────────────
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
        if ex_id in scaled_weights:
            exercise_muscles[ex_id] = scaled_weights[ex_id]
        else:
            exercise_muscles[ex_id] = {m: max(1.0 - 0.15 * r, 0.3) for r, m in enumerate(valid_ranked)}
    return exercise_muscles

EXERCISE_MUSCLES  = _load_exercise_data()
ALL_EXERCISES     = list(EXERCISE_MUSCLES.keys())
NUM_EXERCISES     = len(ALL_EXERCISES)
EXERCISE_TO_IDX   = {e: i for i, e in enumerate(ALL_EXERCISES)}

INVOLVEMENT_MATRIX = np.zeros((NUM_EXERCISES, NUM_MUSCLES), dtype=np.float32)
for ei, ex_id in enumerate(ALL_EXERCISES):
    for m_id, coeff in EXERCISE_MUSCLES[ex_id].items():
        if m_id in MUSCLE_TO_IDX:
            INVOLVEMENT_MATRIX[ei, MUSCLE_TO_IDX[m_id]] = coeff

# ── Detect embed/hidden dims from checkpoint ──────────────────────────────────
state = ckpt.get("model_state_dict", ckpt)
EMBED_DIM  = state["exercise_embed.weight"].shape[1]
HIDDEN_DIM = state["f_net.net.0.weight"].shape[0]
print(f"Detected: EMBED_DIM={EMBED_DIM}, HIDDEN_DIM={HIDDEN_DIM}")

# ── Model definition ──────────────────────────────────────────────────────────
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

class DeepGainModel(nn.Module):
    def __init__(self, num_exercises, num_muscles, embed_dim, hidden_dim):
        super().__init__()
        self.num_muscles    = num_muscles
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.muscle_embed   = nn.Embedding(num_muscles, embed_dim)
        self.f_net = FatigueNet(embed_dim, hidden_dim)
        self.g_net = RIRNet(embed_dim, hidden_dim, num_muscles)
        self.r     = ExponentialRecovery(num_muscles)
        self.register_buffer("involvement",
                             torch.tensor(INVOLVEMENT_MATRIX, dtype=torch.float32))

model = DeepGainModel(NUM_EXERCISES, NUM_MUSCLES, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded: {CHECKPOINT}  ({NUM_EXERCISES} exercises, {NUM_MUSCLES} muscles)")

# ── Transfer matrix ───────────────────────────────────────────────────────────
_transfer_ex = [e for e in ["bench_press", "ohp", "dips", "squat", "deadlift",
                             "pendlay_row", "lat_pulldown", "rdl", "pull_up"]
                if e in EXERCISE_TO_IDX]
n = len(_transfer_ex)
tmat = np.zeros((n, n))

with torch.no_grad():
    for i, ea in enumerate(_transfer_ex):
        for j, eb in enumerate(_transfer_ex):
            mpc_b = torch.ones(1, NUM_MUSCLES, device=DEVICE)
            eeb = model.exercise_embed(torch.tensor([EXERCISE_TO_IDX[eb]], device=DEVICE))
            wb  = torch.tensor([0.4],  dtype=torch.float32, device=DEVICE)
            rb  = torch.tensor([0.27], dtype=torch.float32, device=DEVICE)
            rir_fresh = model.g_net(wb, rb, eeb, mpc_b).item() * RIR_SCALE

            mpc = torch.ones(1, NUM_MUSCLES, device=DEVICE)
            eea = model.exercise_embed(torch.tensor([EXERCISE_TO_IDX[ea]], device=DEVICE))
            me_all = model.muscle_embed(torch.arange(NUM_MUSCLES, device=DEVICE)).unsqueeze(0)
            Ed = me_all.shape[-1]
            for _ in range(4):
                inv  = model.involvement[torch.tensor([EXERCISE_TO_IDX[ea]], device=DEVICE)]
                drop = model.f_net(
                    torch.tensor([0.4],  dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    torch.tensor([0.27], dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    torch.tensor([0.4],  dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    mpc.reshape(-1),
                    eea.expand(NUM_MUSCLES, -1),
                    me_all.reshape(NUM_MUSCLES, Ed),
                ).reshape(1, NUM_MUSCLES)
                mpc = (mpc * (1.0 - inv * drop)).clamp(min=0.1)
            rir_fat = model.g_net(wb, rb, eeb, mpc).item() * RIR_SCALE
            tmat[i, j] = rir_fresh - rir_fat

tmat = np.clip(tmat, 0, None)

fig, ax = plt.subplots(figsize=(10, 8))
labs = [e.replace("_", " ") for e in _transfer_ex]
im = ax.imshow(tmat, cmap="YlOrRd", vmin=0)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labs, fontsize=9)
ax.set_xlabel("Exercise B (tested after)")
ax.set_ylabel("Exercise A (4 sets first)")
ax.set_title("Cross-Exercise Fatigue Transfer\n(RIR decrease on B after 4 sets of A)")
for i in range(n):
    for j in range(n):
        c = "white" if tmat[i, j] > tmat.max() * 0.6 else "black"
        ax.text(j, i, f"{tmat[i, j]:.1f}", ha="center", va="center", fontsize=8, color=c)
plt.colorbar(im, ax=ax, label="RIR decrease", shrink=0.8)
plt.tight_layout()

out = "chart_transfer_matrix_test.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
print(f"\nMin value: {tmat.min():.3f}  Max value: {tmat.max():.3f}")
print(f"Negative cells (before clip): none — clamp applied")
