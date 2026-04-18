#!/usr/bin/env python3
"""Regenerate chart_fatigue_heatmaps.png from a saved checkpoint.

Usage:
    python gen_heatmap.py                          # uses deepgain_model_muscle_ord.pt
    python gen_heatmap.py deepgain_model_best.pt
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

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "deepgain_model_muscle_ord.pt"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Hyperparameters (must match train.py) ─────────────────────────────────────
EMBED_DIM  = 32
HIDDEN_DIM = 128
REPS_SCALE = 30.0
DT_SCALE   = math.log1p(168.0)

# ── Muscles ───────────────────────────────────────────────────────────────────
ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]
NUM_MUSCLES   = len(ALL_MUSCLES)
MUSCLE_TO_IDX = {m: i for i, m in enumerate(ALL_MUSCLES)}

# ── Exercise loading ───────────────────────────────────────────────────────────
def _load_scaled_weights(csv_path="exercise_muscle_weights_scaled.csv"):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if any(m not in df.columns for m in ALL_MUSCLES):
        return {}
    weights = {}
    for _, row in df.iterrows():
        ex_id = str(row["exercise_id"])
        ex_w  = {m: float(np.clip(row[m], 0.0, 1.0)) for m in ALL_MUSCLES if float(row[m]) > 0.0}
        if ex_w:
            weights[ex_id] = ex_w
    return weights

def _load_exercise_data(yaml_path="exercise_muscle_order.yaml"):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    scaled_weights   = _load_scaled_weights()
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

# ── Load checkpoint ───────────────────────────────────────────────────────────
model = DeepGainModel(NUM_EXERCISES, NUM_MUSCLES, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

weight_p5  = np.asarray(ckpt["weight_p5"],  dtype=np.float32) if "weight_p5"  in ckpt else None
weight_p95 = np.asarray(ckpt["weight_p95"], dtype=np.float32) if "weight_p95" in ckpt else None
print(f"Loaded: {CHECKPOINT}  ({NUM_EXERCISES} exercises, {NUM_MUSCLES} muscles)")

# ── Load training data for in-distribution ranges ────────────────────────────
_FULL_CSV = "training_data_full_generated.csv"
_TRAIN_CSV = "training_data_train.csv"
_src = _FULL_CSV if os.path.exists(_FULL_CSV) else _TRAIN_CSV
train_df = pd.read_csv(_src, usecols=["exercise", "weight_kg", "reps"])
if _src == _FULL_CSV:
    # Use only train split (70%) — same seed as train.py
    all_users_df = pd.read_csv(_FULL_CSV, usecols=["user_id", "exercise", "weight_kg", "reps"])
    uids = all_users_df["user_id"].unique()
    rng  = np.random.RandomState(42)
    rng.shuffle(uids)
    train_uids = set(uids[:int(0.7 * len(uids))])
    train_df = all_users_df[all_users_df["user_id"].isin(train_uids)].copy()
print(f"Training data for ranges: {len(train_df):,} rows from {_src}")

# ── Per-exercise weight ranges from checkpoint (or fallback from CSV) ─────────
WEIGHT_P5  = np.full(NUM_EXERCISES, 0.0,   dtype=np.float32)
WEIGHT_P95 = np.full(NUM_EXERCISES, 200.0, dtype=np.float32)
if weight_p5 is not None and len(weight_p5) == NUM_EXERCISES:
    WEIGHT_P5  = weight_p5
    WEIGHT_P95 = weight_p95
else:
    _stats = train_df.groupby("exercise")["weight_kg"].quantile([0.05, 0.95]).unstack()
    for _ex, _row in _stats.iterrows():
        if _ex in EXERCISE_TO_IDX:
            _ei = EXERCISE_TO_IDX[_ex]
            _p5, _p95 = float(_row[0.05]), float(_row[0.95])
            if _p95 > _p5 + 1.0:
                WEIGHT_P5[_ei]  = _p5
                WEIGHT_P95[_ei] = _p95

# ── Generate heatmap ──────────────────────────────────────────────────────────
_heatmap_exercises = ["bench_press", "squat", "deadlift", "ohp", "pendlay_row", "dips"]
_heatmap_exercises = [e for e in _heatmap_exercises if e in EXERCISE_TO_IDX]
_ncols = 3
_nrows = math.ceil(len(_heatmap_exercises) / _ncols)

_reps_stats = train_df.groupby("exercise")["reps"].quantile([0.05, 0.95]).unstack()
_REPS_P5  = {ex: max(1.0,  float(_reps_stats.loc[ex, 0.05])) for ex in _heatmap_exercises if ex in _reps_stats.index}
_REPS_P95 = {ex: min(30.0, float(_reps_stats.loc[ex, 0.95])) for ex in _heatmap_exercises if ex in _reps_stats.index}

_HM_N = 30
fig, axes = plt.subplots(_nrows, _ncols, figsize=(18, 5 * _nrows))
_axes_flat = np.array(axes).reshape(-1)

with torch.no_grad():
    for idx, ex_name in enumerate(_heatmap_exercises):
        ax = _axes_flat[idx]
        ei = EXERCISE_TO_IDX[ex_name]
        muscles = EXERCISE_MUSCLES[ex_name]
        primary_muscle = max(muscles, key=muscles.get)
        mi = MUSCLE_TO_IDX[primary_muscle]

        w_p5_kg  = float(WEIGHT_P5[ei])
        w_p95_kg = float(WEIGHT_P95[ei])
        r_p5     = _REPS_P5.get(ex_name, 1.0)
        r_p95    = _REPS_P95.get(ex_name, 15.0)

        weight_kg_arr = np.linspace(w_p5_kg, w_p95_kg, _HM_N)
        reps_arr      = np.linspace(r_p5, r_p95, _HM_N)

        e_embed = model.exercise_embed(torch.tensor([ei], device=DEVICE))
        m_embed = model.muscle_embed(torch.tensor([mi], device=DEVICE))

        drop_grid = np.zeros((_HM_N, _HM_N))
        w_range = max(w_p95_kg - w_p5_kg, 1.0)
        for ri, r_val in enumerate(reps_arr):
            r_norm = r_val / REPS_SCALE
            for wi, w_kg in enumerate(weight_kg_arr):
                w_norm = float(np.clip((w_kg - w_p5_kg) / w_range, 0.0, 1.0))
                w_t   = torch.tensor([w_norm], dtype=torch.float32, device=DEVICE)
                r_t   = torch.tensor([r_norm], dtype=torch.float32, device=DEVICE)
                rir_t = torch.tensor([0.4],    dtype=torch.float32, device=DEVICE)
                mpc_t = torch.tensor([1.0],    dtype=torch.float32, device=DEVICE)
                d = model.f_net(w_t, r_t, rir_t, mpc_t, e_embed, m_embed).item()
                drop_grid[ri, wi] = d * muscles[primary_muscle]

        im = ax.imshow(drop_grid, aspect="auto", origin="lower",
                       extent=[w_p5_kg, w_p95_kg, r_p5, r_p95],
                       cmap="YlOrRd", vmin=0)
        ax.set_xlabel("Weight (kg)")
        ax.set_ylabel("Reps")
        ax.set_title(f"{ex_name.replace('_', ' ').title()}\n(primary: {primary_muscle})")
        plt.colorbar(im, ax=ax, label="MPC drop")

        # Grey mask: cells with no training data (off-manifold)
        ex_train = train_df[train_df["exercise"] == ex_name]
        if len(ex_train) > 0:
            hist, _, _ = np.histogram2d(
                ex_train["weight_kg"].values, ex_train["reps"].values,
                bins=[_HM_N, _HM_N],
                range=[[w_p5_kg, w_p95_kg], [r_p5, r_p95]],
            )
            no_data = (hist.T == 0).astype(float)
            ax.imshow(no_data, aspect="auto", origin="lower",
                      extent=[w_p5_kg, w_p95_kg, r_p5, r_p95],
                      cmap="Greys", alpha=0.45, vmin=0, vmax=1)

for idx in range(len(_heatmap_exercises), len(_axes_flat)):
    _axes_flat[idx].set_visible(False)

plt.suptitle(
    "Learned Fatigue (f) — Per-Muscle Breakdown\n"
    "(per-exercise weight/reps range p5–p95; grey = no training data)",
    fontsize=13,
)
plt.tight_layout()

out_path = "chart_fatigue_heatmaps_new.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
