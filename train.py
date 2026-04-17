#!/usr/bin/env python3
"""DeepGain — Train f, g, r networks from training_data.csv (RIR-based, v3)"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
import time
from datetime import datetime
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ─── Hyperparameters ───
EMBED_DIM = 32
HIDDEN_DIM = 256
LR = 1e-3
EPOCHS = 50
CHUNK_LEN = 512
BATCH_SIZE = 16
WEIGHT_SCALE = 200.0
REPS_SCALE = 30.0
RIR_SCALE = 5.0
DT_SCALE = np.log1p(168.0)

# ─── Muscles (15 groups) ───
# upper_traps and brachialis removed: no direct EMG columns in current CSV schema
ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "rhomboids", "triceps", "biceps",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
    "abs",
]
NUM_MUSCLES = len(ALL_MUSCLES)
MUSCLE_TO_IDX = {m: i for i, m in enumerate(ALL_MUSCLES)}

# ─── Exercises are loaded from YAML + transformed CSV only ────────────────

_SCALED_WEIGHTS_PATH = "exercise_muscle_weights_scaled.csv"


def _load_scaled_weights(csv_path: str = _SCALED_WEIGHTS_PATH):
    """Load exercise->muscle weights in [0,1] from EMG-derived CSV.

    Priority source for numeric involvement coefficients.
    Expected schema: exercise_id + one column per muscle id in ALL_MUSCLES.
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Warning: could not read {csv_path}: {exc}")
        return {}
    missing = [m for m in ALL_MUSCLES if m not in df.columns]
    if missing:
        print(f"Warning: {csv_path} missing muscle columns: {missing}")
        return {}
    weights = {}
    for _, row in df.iterrows():
        ex_id = str(row["exercise_id"])
        ex_weights = {m: float(np.clip(row[m], 0.0, 1.0)) for m in ALL_MUSCLES if float(row[m]) > 0.0}
        if ex_weights:
            weights[ex_id] = ex_weights
    return weights

def _load_exercise_data(yaml_path: str = "exercise_muscle_order.yaml"):
    """Load exercise/muscle data strictly from exercise_muscle_order.yaml.

    Returns:
        exercise_muscles : dict[exercise_id -> dict[muscle_id -> float]]
            Numeric involvement coefficients used in the MPC update rule.
            Exercises missing in transformed CSV get rank-derived
            coefficients: coeff(rank r) = max(1.0 - 0.15*r, 0.3).
        exercise_ordinal : dict[exercise_id -> list[muscle_id]]
            Muscles ordered from most to least involved (primary first).
            Used by the ordinal ordering penalty.
    """
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required file: {yaml_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not load {yaml_path}: {exc}") from exc

    if not data or "exercises" not in data:
        raise RuntimeError(f"Invalid YAML structure in {yaml_path}: missing 'exercises'")

    scaled_weights = _load_scaled_weights()

    # YAML is the canonical exercise list — only exercises present in YAML are used.
    exercise_muscles = {}
    exercise_ordinal = {}
    scaled_used = yaml_rank_used = 0

    for ex_id, ex_data in data["exercises"].items():
        if not isinstance(ex_data, dict):
            continue
        ranked = (
            ex_data.get("primary_muscles", [])
            + ex_data.get("secondary_muscles", [])
            + ex_data.get("tertiary_muscles", [])
        )
        if not ranked:
            continue
        valid_ranked = [m for m in ranked if m in MUSCLE_TO_IDX]
        if not valid_ranked:
            continue
        exercise_ordinal[ex_id] = valid_ranked

        # Priority: EMG CSV → YAML rank-derived coefficients
        if ex_id in scaled_weights:
            exercise_muscles[ex_id] = scaled_weights[ex_id]
            scaled_used += 1
        else:
            exercise_muscles[ex_id] = {
                m: max(1.0 - 0.15 * r, 0.3)
                for r, m in enumerate(valid_ranked)
            }
            yaml_rank_used += 1

    print(f"Exercise data: {yaml_path} — {len(exercise_muscles)} exercises "
          f"(csv={scaled_used}, yaml_rank={yaml_rank_used})")
    return exercise_muscles, exercise_ordinal


EXERCISE_MUSCLES, EXERCISE_ORDINAL = _load_exercise_data()

ALL_EXERCISES = list(EXERCISE_MUSCLES.keys())
NUM_EXERCISES = len(ALL_EXERCISES)
EXERCISE_TO_IDX = {e: i for i, e in enumerate(ALL_EXERCISES)}

INVOLVEMENT_MATRIX = np.zeros((NUM_EXERCISES, NUM_MUSCLES), dtype=np.float32)
for ex_name, muscles in EXERCISE_MUSCLES.items():
    ei = EXERCISE_TO_IDX[ex_name]
    for m_name, coeff in muscles.items():
        mi = MUSCLE_TO_IDX.get(m_name)
        if mi is not None:
            INVOLVEMENT_MATRIX[ei, mi] = coeff

print(f"Exercises: {NUM_EXERCISES}, Muscles: {NUM_MUSCLES}")

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def _load_split(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["exercise_idx"] = df["exercise"].map(EXERCISE_TO_IDX)
    df = df.dropna(subset=["exercise_idx"])
    df["exercise_idx"] = df["exercise_idx"].astype(int)
    df["delta_t_hours"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 3600.0
    df["delta_t_hours"] = df["delta_t_hours"].fillna(0.0)
    return df

print("Loading data...")
# Per-user 70/30 split — preserves complete user sequences in val (no MPC state leakage).
# Falls back to pre-split files if full CSV not available.
_FULL_CSV = "training_data_michal_full.csv"
_SPLIT_CANDIDATES = [
    ("training_data_train.csv", "training_data_val.csv"),
    ("generated_datasets/baseline_main/training_data_train.csv",
     "generated_datasets/baseline_main/training_data_val.csv"),
]

if os.path.exists(_FULL_CSV):
    df = _load_split(_FULL_CSV)
    user_ids = df["user_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(user_ids)
    split = int(0.7 * len(user_ids))
    train_df = df[df["user_id"].isin(set(user_ids[:split]))].copy()
    test_df  = df[df["user_id"].isin(set(user_ids[split:]))].copy()
    print(f"Loaded {_FULL_CSV}, per-user 70/30 split ({len(set(user_ids[:split]))} train / {len(set(user_ids[split:]))} val users)")
else:
    _loaded = False
    for _train_path, _val_path in _SPLIT_CANDIDATES:
        if os.path.exists(_train_path) and os.path.exists(_val_path):
            train_df = _load_split(_train_path)
            test_df  = _load_split(_val_path)
            print(f"Loaded pre-split files: {_train_path}")
            _loaded = True
            break
    if not _loaded:
        df = _load_split("training_data.csv")
        user_ids = df["user_id"].unique()
        rng = np.random.RandomState(42)
        rng.shuffle(user_ids)
        split = int(0.7 * len(user_ids))
        train_df = df[df["user_id"].isin(set(user_ids[:split]))].copy()
        test_df  = df[df["user_id"].isin(set(user_ids[split:]))].copy()
        print(f"Loaded training_data.csv, per-user 70/30 split")

print(f"Train: {len(train_df):,} sets from {train_df['user_id'].nunique()} users")
print(f"Test:  {len(test_df):,} sets from {test_df['user_id'].nunique()} users")

# ── Per-exercise weight normalization (p5/p95 from train only) ────────────────
# Replaces global WEIGHT_SCALE=200. Makes weight=0.5 mean "median for this exercise"
# so bench 80kg and dips 80kg are no longer the same number.
_stats = train_df.groupby("exercise")["weight_kg"].quantile([0.05, 0.95]).unstack()
WEIGHT_P5  = np.full(NUM_EXERCISES, 0.0,   dtype=np.float32)
WEIGHT_P95 = np.full(NUM_EXERCISES, 200.0, dtype=np.float32)
for _ex, _row in _stats.iterrows():
    if _ex in EXERCISE_TO_IDX:
        _ei = EXERCISE_TO_IDX[_ex]
        _p5, _p95 = float(_row[0.05]), float(_row[0.95])
        if _p95 > _p5 + 1.0:  # skip degenerate ranges
            WEIGHT_P5[_ei]  = _p5
            WEIGHT_P95[_ei] = _p95

def normalize_weight(weight_kg_arr, exercise_idx_arr):
    """Normalize weight per-exercise to [0, 1] using p5/p95 from training data."""
    p5  = WEIGHT_P5[exercise_idx_arr]
    p95 = WEIGHT_P95[exercise_idx_arr]
    return np.clip((weight_kg_arr - p5) / (p95 - p5), 0.0, 1.0).astype(np.float32)

print("Weight ranges computed (per-exercise p5/p95)")

# ═══════════════════════════════════════════════════════════════════
# DATASET & DATALOADER
# ═══════════════════════════════════════════════════════════════════

def build_user_sequences(user_df):
    sequences = []
    for uid, grp in user_df.groupby("user_id"):
        seq = {
            "user_id": uid,
            "exercise_idx": torch.tensor(grp["exercise_idx"].values, dtype=torch.long),
            "weight": torch.tensor(normalize_weight(grp["weight_kg"].values, grp["exercise_idx"].values), dtype=torch.float32),
            "reps": torch.tensor(grp["reps"].values / REPS_SCALE, dtype=torch.float32),
            "rir": torch.tensor(grp["rir"].values / RIR_SCALE, dtype=torch.float32),
            "delta_t": torch.tensor(
                np.log1p(grp["delta_t_hours"].values) / DT_SCALE, dtype=torch.float32
            ),
            "timestamps": grp["timestamp"].values,
        }
        sequences.append(seq)
    return sequences

def chunk_sequence(seq, chunk_len):
    T = len(seq["exercise_idx"])
    if T <= chunk_len:
        return [seq]
    chunks = []
    for start in range(0, T, chunk_len):
        end = min(start + chunk_len, T)
        chunk = {
            "user_id": seq["user_id"],
            "exercise_idx": seq["exercise_idx"][start:end],
            "weight": seq["weight"][start:end],
            "reps": seq["reps"][start:end],
            "rir": seq["rir"][start:end],
            "delta_t": seq["delta_t"][start:end],
        }
        if "timestamps" in seq:
            chunk["timestamps"] = seq["timestamps"][start:end]
        chunks.append(chunk)
    return chunks

class ChunkedDataset(Dataset):
    def __init__(self, user_df, chunk_len):
        sequences = build_user_sequences(user_df)
        self.chunks = []
        for seq in sequences:
            self.chunks.extend(chunk_sequence(seq, chunk_len))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

def collate_fn(batch):
    max_len = max(len(b["exercise_idx"]) for b in batch)
    B = len(batch)
    out = {
        "exercise_idx": torch.zeros(B, max_len, dtype=torch.long),
        "weight": torch.zeros(B, max_len),
        "reps": torch.zeros(B, max_len),
        "rir": torch.zeros(B, max_len),
        "delta_t": torch.zeros(B, max_len),
        "mask": torch.zeros(B, max_len),
    }
    for i, b in enumerate(batch):
        T = len(b["exercise_idx"])
        out["exercise_idx"][i, :T] = b["exercise_idx"]
        out["weight"][i, :T] = b["weight"]
        out["reps"][i, :T] = b["reps"]
        out["rir"][i, :T] = b["rir"]
        out["delta_t"][i, :T] = b["delta_t"]
        out["mask"][i, :T] = 1.0
    return out

print("Building datasets...")
train_ds = ChunkedDataset(train_df, CHUNK_LEN)
test_ds = ChunkedDataset(test_df, CHUNK_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train chunks: {len(train_ds)}, Test chunks: {len(test_ds)}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class FatigueNet(nn.Module):
    """f: predicts MPC drop fraction after a set."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, weight, reps, rir, mpc, e_embed, m_embed):
        x = torch.cat([
            weight.unsqueeze(-1), reps.unsqueeze(-1),
            rir.unsqueeze(-1), mpc.unsqueeze(-1),
            e_embed, m_embed
        ], dim=-1)
        return self.net(x).squeeze(-1)


class RIRNet(nn.Module):
    """g: predicts RIR from current MPC state."""
    def __init__(self, embed_dim, hidden_dim, num_muscles):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + embed_dim + num_muscles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, weight, reps, e_embed, mpc_all):
        x = torch.cat([
            weight.unsqueeze(-1), reps.unsqueeze(-1),
            e_embed, mpc_all
        ], dim=-1)
        raw = self.net(x).squeeze(-1)
        return torch.sigmoid(raw)


class ExponentialRecovery(nn.Module):
    """r_m: MPC recovers toward 1.0 with learned per-muscle τ_m.

    r(MPC, Δt, muscle) = 1 - (1 - MPC) · exp(-Δt / τ_m)

    Path-consistent by construction. 15 learnable parameters.

    Soft ordering penalty: mean(small τ) < mean(medium τ) < mean(large τ)
    Small:  triceps, biceps, calves, rear_delts, lateral_delts, abs
    Medium: chest, anterior_delts, rhomboids, lats, erectors
    Large:  quads, hamstrings, glutes, adductors
    """
    # Muscle indices by size bucket (matches ALL_MUSCLES ordering, 15 muscles)
    # chest=0, anterior_delts=1, lateral_delts=2, rear_delts=3, rhomboids=4,
    # triceps=5, biceps=6, lats=7, quads=8, hamstrings=9, glutes=10,
    # adductors=11, erectors=12, calves=13, abs=14
    SMALL_IDX  = [5, 6, 13, 3, 2, 14]   # triceps, biceps, calves, rear_delts, lateral_delts, abs
    MEDIUM_IDX = [0, 1, 4, 7, 12]       # chest, anterior_delts, rhomboids, lats, erectors
    LARGE_IDX  = [8, 9, 10, 11]         # quads, hamstrings, glutes, adductors

    # Literature-derived τ values (hours) — averaged from two deep-research studies
    # Fixed, not learned. Based on MVC/velocity/torque recovery data from 15+ papers.
    FIXED_TAU = [
        16.0,  # chest — Ferreira 2017, Belcher 2019, Bartolomei 2021
        13.0,  # anterior_delts — bench synergist data
        9.0,   # lateral_delts — Korak 2015, training frequency data
        8.0,   # rear_delts — training frequency heuristics
        10.0,  # rhomboids — fiber type, rowing data
        9.0,   # triceps — Ferreira 2017a, Korak 2015
        13.0,  # biceps — Soares 2015, Clark 2020
        13.0,  # lats — Soares 2015, Korak 2015
        19.0,  # quads — Raastad 2000, Pareja-Blanco 2019, Thomas 2018
        18.0,  # hamstrings — Okazaki 2021, sprint studies
        15.0,  # glutes — squat/deadlift proxy, reduced eccentric susceptibility
        12.0,  # adductors — Korak 2015
        12.0,  # erectors — Belcher 2019 deadlift, fiber type
        8.0,   # calves — Lievens 2020, soleus fiber type
        10.0,  # abs — fast recovery, postural/stabilizer role
    ]

    def __init__(self, num_muscles):
        super().__init__()
        # Fixed τ from literature — NOT learned
        init_tau = torch.tensor([math.log(t) for t in self.FIXED_TAU])
        self.log_tau = nn.Parameter(init_tau, requires_grad=False)

    def forward(self, mpc, delta_t, muscle_idx):
        """mpc: (N,), delta_t: (N,) normalized, muscle_idx: (N,) long."""
        dt_hours = torch.expm1(delta_t * DT_SCALE)
        tau = torch.exp(self.log_tau[muscle_idx])
        decay = torch.exp(-dt_hours / tau)
        return 1.0 - (1.0 - mpc) * decay

    def ordering_penalty(self):
        """Soft penalty: mean(small τ) < mean(medium τ) < mean(large τ)."""
        tau = torch.exp(self.log_tau)
        mean_small = tau[self.SMALL_IDX].mean()
        mean_medium = tau[self.MEDIUM_IDX].mean()
        mean_large = tau[self.LARGE_IDX].mean()
        return (torch.relu(mean_small - mean_medium)
              + torch.relu(mean_medium - mean_large))


class DeepGainModel(nn.Module):
    def __init__(self, num_exercises, num_muscles, embed_dim, hidden_dim):
        super().__init__()
        self.num_muscles = num_muscles
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.muscle_embed = nn.Embedding(num_muscles, embed_dim)
        self.f_net = FatigueNet(embed_dim, hidden_dim)
        self.g_net = RIRNet(embed_dim, hidden_dim, num_muscles)
        self.r = ExponentialRecovery(num_muscles)
        self.register_buffer("involvement",
                             torch.tensor(INVOLVEMENT_MATRIX, dtype=torch.float32))

    def fatigue_ordering_penalty(self):
        """Penalize f when drop ordering doesn't match ordinal involvement ranking.

        For each exercise, muscles listed earlier in EXERCISE_ORDINAL (more primary)
        should have higher raw drops from f. Probes f at a reference point
        (w=0.4, r=0.27, rir=0.4, mpc=1.0) and penalizes pairwise ordering violations.
        """
        device = self.involvement.device
        penalty = torch.tensor(0.0, device=device)
        n_pairs = 0

        w = torch.tensor([0.4], dtype=torch.float32, device=device)
        r = torch.tensor([0.27], dtype=torch.float32, device=device)
        rir = torch.tensor([0.4], dtype=torch.float32, device=device)
        mpc = torch.tensor([1.0], dtype=torch.float32, device=device)

        for ei in range(len(ALL_EXERCISES)):
            inv = self.involvement[ei]  # (M,)
            involved = (inv > 0).nonzero(as_tuple=True)[0]
            if len(involved) < 2:
                continue

            ex_id = ALL_EXERCISES[ei]
            ordinal = EXERCISE_ORDINAL.get(ex_id, [])
            rank_map = {MUSCLE_TO_IDX[m]: rank for rank, m in enumerate(ordinal) if m in MUSCLE_TO_IDX}

            e_embed = self.exercise_embed(torch.tensor([ei], device=device))

            # Get raw drops and ordinal ranks for each involved muscle
            drops = []
            ranks = []
            for mi in involved:
                m_embed = self.muscle_embed(mi.unsqueeze(0))
                drop = self.f_net(w, r, rir, mpc, e_embed, m_embed)
                drops.append(drop)
                ranks.append(rank_map.get(mi.item(), len(ordinal)))

            # Pairwise: if rank[a] < rank[b] (a more primary), then drop[a] should > drop[b]
            for a in range(len(drops)):
                for b in range(a + 1, len(drops)):
                    if ranks[a] < ranks[b]:
                        penalty = penalty + torch.relu(drops[b] - drops[a])
                    elif ranks[b] < ranks[a]:
                        penalty = penalty + torch.relu(drops[a] - drops[b])
                    n_pairs += 1

        return penalty / max(n_pairs, 1)

    def minimum_drop_penalty(self, min_fraction: float = 0.15):
        """Penalize f when involved muscles have near-zero drops (muscle collapse).

        For each exercise, each muscle with non-zero involvement should produce
        a drop of at least min_fraction * involvement from f_net.
        Probes at the same reference point as fatigue_ordering_penalty.

        min_fraction=0.15 means: a muscle with involvement=0.6 should drop ≥ 0.09.
        """
        device = self.involvement.device
        penalty = torch.tensor(0.0, device=device)
        n_muscles = 0

        w   = torch.tensor([0.5],  dtype=torch.float32, device=device)
        r   = torch.tensor([0.27], dtype=torch.float32, device=device)
        rir = torch.tensor([0.4],  dtype=torch.float32, device=device)
        mpc = torch.tensor([1.0],  dtype=torch.float32, device=device)

        for ei in range(len(ALL_EXERCISES)):
            inv = self.involvement[ei]  # (M,)
            involved = (inv > 0).nonzero(as_tuple=True)[0]
            if len(involved) == 0:
                continue

            e_embed = self.exercise_embed(torch.tensor([ei], device=device))

            for mi in involved:
                m_embed = self.muscle_embed(mi.unsqueeze(0))
                drop = self.f_net(w, r, rir, mpc, e_embed, m_embed)
                min_expected = min_fraction * inv[mi]
                penalty = penalty + torch.relu(min_expected - drop)
                n_muscles += 1

        return penalty / max(n_muscles, 1)

    def forward(self, exercise_idx, weight, reps, rir_target, delta_t, mask):
        B, T = exercise_idx.shape
        M = self.num_muscles
        device = exercise_idx.device

        mpc = torch.ones(B, M, device=device)
        rir_preds = []

        all_m_idx = torch.arange(M, device=device)
        all_m_embed = self.muscle_embed(all_m_idx)
        all_m_embed_expanded = all_m_embed.unsqueeze(0).expand(B, -1, -1)
        E = all_m_embed.shape[-1]
        # Muscle indices for recovery: (B*M,) repeating [0,1,...,15] for each batch
        m_idx_flat = all_m_idx.unsqueeze(0).expand(B, -1).reshape(-1)

        for t in range(T):
            if t > 0:
                dt = delta_t[:, t]
                dt_exp = dt.unsqueeze(1).expand(-1, M).reshape(-1)
                mpc_flat = mpc.reshape(-1)
                mpc = self.r(mpc_flat, dt_exp, m_idx_flat).reshape(B, M)

            e_idx = exercise_idx[:, t]
            e_embed = self.exercise_embed(e_idx)
            rir_pred = self.g_net(weight[:, t], reps[:, t], e_embed, mpc)
            rir_preds.append(rir_pred)

            inv = self.involvement[e_idx]
            e_emb_exp = e_embed.unsqueeze(1).expand(-1, M, -1).reshape(-1, E)
            w_exp = weight[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            r_exp = reps[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            rir_exp = rir_target[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            mpc_flat = mpc.reshape(-1)
            m_emb_flat = all_m_embed_expanded.reshape(-1, E)

            drop = self.f_net(w_exp, r_exp, rir_exp, mpc_flat, e_emb_exp, m_emb_flat)
            drop = drop.reshape(B, M)
            mpc = (mpc * (1.0 - inv * drop)).clamp(min=0.1)

        rir_preds = torch.stack(rir_preds, dim=1)
        return rir_preds, mpc

    def forward_with_mpc_history(self, exercise_idx, weight, reps, rir_target, delta_t):
        B, T = exercise_idx.shape
        assert B == 1
        M = self.num_muscles
        device = exercise_idx.device

        mpc = torch.ones(1, M, device=device)
        rir_preds = []
        mpc_history = [mpc[0].detach().cpu().numpy().copy()]

        all_m_idx = torch.arange(M, device=device)
        all_m_embed = self.muscle_embed(all_m_idx)
        all_m_embed_expanded = all_m_embed.unsqueeze(0)
        E = all_m_embed.shape[-1]
        m_idx_flat = all_m_idx.unsqueeze(0).reshape(-1)

        for t in range(T):
            if t > 0:
                dt = delta_t[:, t]
                dt_exp = dt.unsqueeze(1).expand(-1, M).reshape(-1)
                mpc_flat = mpc.reshape(-1)
                mpc = self.r(mpc_flat, dt_exp, m_idx_flat).reshape(1, M)

            e_idx = exercise_idx[:, t]
            e_embed = self.exercise_embed(e_idx)
            rir_pred = self.g_net(weight[:, t], reps[:, t], e_embed, mpc)
            rir_preds.append(rir_pred.item())

            inv = self.involvement[e_idx]
            e_emb_exp = e_embed.unsqueeze(1).expand(-1, M, -1).reshape(-1, E)
            w_exp = weight[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            r_exp = reps[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            rir_exp = rir_target[:, t].unsqueeze(1).expand(-1, M).reshape(-1)
            mpc_flat = mpc.reshape(-1)
            m_emb_flat = all_m_embed_expanded.reshape(-1, E)
            drop = self.f_net(w_exp, r_exp, rir_exp, mpc_flat, e_emb_exp, m_emb_flat).reshape(1, M)
            mpc = (mpc * (1.0 - inv * drop)).clamp(min=0.1)
            mpc_history.append(mpc[0].detach().cpu().numpy().copy())

        return np.array(rir_preds), np.array(mpc_history)


model = DeepGainModel(NUM_EXERCISES, NUM_MUSCLES, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum()

@torch.no_grad()
def evaluate(mdl, loader):
    mdl.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        rir_pred, _ = mdl(batch["exercise_idx"], batch["weight"],
                          batch["reps"], batch["rir"], batch["delta_t"], batch["mask"])
        loss = masked_mse(rir_pred, batch["rir"], batch["mask"])
        n = batch["mask"].sum().item()
        total_loss += loss.item() * n
        total_count += n
    return total_loss / total_count

optimizer = optim.Adam([
    {'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR},
], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
best_val_loss = float("inf")

train_losses = []
val_losses = []

print("\n" + "=" * 70)
print("TRAINING (τ fixed from literature)")
print("=" * 70)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_count = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        rir_pred, _ = model(batch["exercise_idx"], batch["weight"],
                            batch["reps"], batch["rir"], batch["delta_t"], batch["mask"])
        loss = masked_mse(rir_pred, batch["rir"], batch["mask"])
        loss = loss + 0.05 * model.fatigue_ordering_penalty()
        loss = loss + 0.10 * model.minimum_drop_penalty()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = batch["mask"].sum().item()
        epoch_loss += loss.item() * n
        epoch_count += n

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{len(train_loader)}, "
                  f"loss: {loss.item():.6f}", flush=True)

    train_loss = epoch_loss / epoch_count
    val_loss = evaluate(model, test_loader)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    elapsed = time.time() - t0
    train_rmse = np.sqrt(train_loss) * RIR_SCALE
    val_rmse = np.sqrt(val_loss) * RIR_SCALE

    # Recovery probe: chest (idx 0) at MPC=0.5, sweep 2h/12h/48h
    with torch.no_grad():
        probe_results = []
        for dt_h in [2.0, 12.0, 48.0]:
            dt_norm = torch.tensor([np.log1p(dt_h) / DT_SCALE], dtype=torch.float32, device=DEVICE)
            mpc_in = torch.tensor([0.5], dtype=torch.float32, device=DEVICE)
            m_idx = torch.tensor([0], dtype=torch.long, device=DEVICE)  # chest
            mpc_out = model.r(mpc_in, dt_norm, m_idx).item()
            probe_results.append(f"{dt_h:.0f}h→{mpc_out:.3f}")
    tau_chest = math.exp(model.r.log_tau[0].item())
    recovery_str = f"  r(chest,0.5): " + " | ".join(probe_results) + f" | τ={tau_chest:.1f}h"

    # All τ values
    tau_strs = []
    for mi, m in enumerate(ALL_MUSCLES):
        tau_strs.append(f"{m[:4]}={math.exp(model.r.log_tau[mi].item()):.0f}")
    tau_all_str = "  τ: " + " | ".join(tau_strs)

    with torch.no_grad():
        ord_penalty = model.fatigue_ordering_penalty().item()
        min_penalty = model.minimum_drop_penalty().item()

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train RMSE: {train_rmse:.2f} RIR | Val RMSE: {val_rmse:.2f} RIR | "
          f"ord_pen: {ord_penalty:.4f} | min_pen: {min_penalty:.4f} | {elapsed:.0f}s", flush=True)
    print(recovery_str, flush=True)
    print(tau_all_str, flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "weight_p5":  WEIGHT_P5,
            "weight_p95": WEIGHT_P95,
        }, "deepgain_model_best.pt")

# Save final model
torch.save({
    "model_state_dict": model.state_dict(),
    "weight_p5":  WEIGHT_P5,
    "weight_p95": WEIGHT_P95,
}, "deepgain_model_muscle_ord.pt")
print(f"\nModel saved to deepgain_model_muscle_ord.pt")
print(f"Best val model saved to deepgain_model_best.pt (RMSE {np.sqrt(best_val_loss) * RIR_SCALE:.4f})")

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(mdl, loader):
    mdl.eval()
    all_preds, all_targets, all_exercises = [], [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        rir_pred, _ = mdl(batch["exercise_idx"], batch["weight"],
                          batch["reps"], batch["rir"], batch["delta_t"], batch["mask"])
        mask = batch["mask"].bool()
        all_preds.append((rir_pred[mask] * RIR_SCALE).cpu().numpy())
        all_targets.append((batch["rir"][mask] * RIR_SCALE).cpu().numpy())
        all_exercises.append(batch["exercise_idx"][mask].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_exercises)

print("\nCollecting test predictions...")
test_preds, test_targets, test_exercises = collect_predictions(model, test_loader)

mse = np.mean((test_preds - test_targets) ** 2)
mae = np.mean(np.abs(test_preds - test_targets))
rmse = np.sqrt(mse)
corr = np.corrcoef(test_preds, test_targets)[0, 1]

print(f"\nTest Metrics (RIR scale 0-5):")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R:    {corr:.4f}")

exercise_maes = {}
for ei in range(NUM_EXERCISES):
    mask = test_exercises == ei
    if mask.sum() > 0:
        exercise_maes[ALL_EXERCISES[ei]] = np.mean(np.abs(test_preds[mask] - test_targets[mask]))

print(f"\nPer-exercise MAE:")
for ex, mae_val in sorted(exercise_maes.items(), key=lambda x: x[1]):
    print(f"  {ex:25s}: {mae_val:.3f}")

# ── Ordering accuracy per exercise ───────────────────────────────────────────
print(f"\nOrdering accuracy per exercise (% pairs where primary muscle drops more):")
model.eval()
ordering_results = {}
with torch.no_grad():
    w   = torch.tensor([0.5],  dtype=torch.float32, device=DEVICE)
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
                # rank[i] < rank[j] → drops[ranked[i]] should be > drops[ranked[j]]
                if drops[ranked[i]] > drops[ranked[j]]:
                    correct += 1
                total += 1
        ordering_results[ex_id] = correct / total if total > 0 else float("nan")

for ex, acc in sorted(ordering_results.items(), key=lambda x: x[1]):
    print(f"  {ex:25s}: {acc*100:.0f}%")
overall_acc = np.mean(list(ordering_results.values()))
print(f"  {'MEAN':25s}: {overall_acc*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════

CHART_DIR = os.path.join("charts", datetime.now().strftime("%Y%m%d_%H%M"))
os.makedirs(CHART_DIR, exist_ok=True)
print(f"Charts → {CHART_DIR}/")

muscle_colors = plt.cm.tab20(np.linspace(0, 1, NUM_MUSCLES))

# --- Chart 1: Loss Curves ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
epochs_range = range(1, len(train_losses) + 1)
ax.plot(epochs_range, [np.sqrt(l) * RIR_SCALE for l in train_losses], label="Train RMSE", linewidth=2)
ax.plot(epochs_range, [np.sqrt(l) * RIR_SCALE for l in val_losses], label="Val RMSE", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE (RIR scale)")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_loss_curves.png", dpi=150)
print("Saved chart_loss_curves.png")

# --- Chart 2: RIR Prediction Accuracy ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
n_sample = min(15000, len(test_preds))
idx = np.random.choice(len(test_preds), n_sample, replace=False)
ax.scatter(test_targets[idx], test_preds[idx], alpha=0.05, s=8, c="steelblue")
ax.plot([0, 5], [0, 5], "r--", linewidth=2, label="Perfect")
r2 = corr ** 2
ax.set_xlabel("Actual RIR")
ax.set_ylabel("Predicted RIR")
ax.set_title(f"Predicted vs Actual RIR (R²={r2:.3f})")
ax.legend()
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 5.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
sorted_exercises = sorted(exercise_maes.items(), key=lambda x: x[1], reverse=True)
names = [e[0].replace("_", " ") for e in sorted_exercises]
vals = [e[1] for e in sorted_exercises]
colors = ["#e74c3c" if v > np.median(vals) else "#2ecc71" for v in vals]
ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("MAE (RIR)")
ax.set_title("Per-Exercise MAE")
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis="x")

ax = axes[2]
errors = test_preds - test_targets
ax.hist(errors, bins=80, color="steelblue", alpha=0.7, density=True)
ax.axvline(0, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("RIR Error (pred - actual)")
ax.set_ylabel("Density")
ax.set_title(f"Error Distribution (mean={np.mean(errors):.3f}, std={np.std(errors):.3f})")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_rir_accuracy.png", dpi=150)
print("Saved chart_rir_accuracy.png")

# --- Chart 3: MPC Trajectories ---
test_sequences = build_user_sequences(test_df)
_pinned_ids = {"user_00030", "user_00111"}
_pinned = [s for s in test_sequences if s["user_id"] in _pinned_ids]
_rest   = [s for s in test_sequences if s["user_id"] not in _pinned_ids]
_best_rest = sorted(_rest, key=lambda s: len(s["exercise_idx"]), reverse=True)[:1]
sample_users = (_pinned + _best_rest)[:3]

model.eval()
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

for row, seq in enumerate(sample_users):
    uid = seq["user_id"]
    T = len(seq["exercise_idx"])
    T_plot = min(T, 400)

    with torch.no_grad():
        ex = seq["exercise_idx"][:T_plot].unsqueeze(0).to(DEVICE)
        w = seq["weight"][:T_plot].unsqueeze(0).to(DEVICE)
        r = seq["reps"][:T_plot].unsqueeze(0).to(DEVICE)
        rir = seq["rir"][:T_plot].unsqueeze(0).to(DEVICE)
        dt = seq["delta_t"][:T_plot].unsqueeze(0).to(DEVICE)
        rir_preds_seq, mpc_hist = model.forward_with_mpc_history(ex, w, r, rir, dt)

    timestamps = seq["timestamps"][:T_plot]
    t0_ts = timestamps[0]
    hours = [(t - t0_ts) / np.timedelta64(1, "h") for t in timestamps]

    ax = axes[row, 0]
    used_exercises = seq["exercise_idx"][:T_plot].numpy()
    used_muscles = set()
    for ei_val in used_exercises:
        for mi in range(NUM_MUSCLES):
            if INVOLVEMENT_MATRIX[ei_val, mi] > 0:
                used_muscles.add(mi)
    muscle_activity = {}
    for mi in used_muscles:
        muscle_activity[mi] = np.sum(INVOLVEMENT_MATRIX[used_exercises, mi])
    top_muscles = sorted(muscle_activity, key=muscle_activity.get, reverse=True)[:6]

    for mi in top_muscles:
        ax.plot(hours, mpc_hist[:-1, mi], label=ALL_MUSCLES[mi].replace("_", " "),
                alpha=0.8, linewidth=1.2, color=muscle_colors[mi])
    ax.set_ylabel("MPC")
    ax.set_title(f"MPC Over Time — {uid} ({T_plot} sets)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    if row == 2:
        ax.set_xlabel("Hours from first set")

    ax = axes[row, 1]
    actual_rir = seq["rir"][:T_plot].numpy() * RIR_SCALE
    ax.scatter(hours, actual_rir, s=8, alpha=0.4, c="steelblue", label="Actual RIR")
    ax.scatter(hours, np.array(rir_preds_seq) * RIR_SCALE, s=8, alpha=0.4, c="coral", label="Predicted RIR")
    ax.set_ylabel("RIR")
    ax.set_title(f"RIR Prediction — {uid}")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, alpha=0.3)
    if row == 2:
        ax.set_xlabel("Hours from first set")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_mpc_trajectories.png", dpi=150)
print("Saved chart_mpc_trajectories.png")

# --- Chart 4: Recovery Curves ---
model.eval()
dt_hours = np.linspace(0, 72, 200)
dt_normalized = np.log1p(dt_hours) / DT_SCALE

fig, ax = plt.subplots(figsize=(12, 6))

with torch.no_grad():
    for mi in range(NUM_MUSCLES):
        m_idx = torch.full((len(dt_normalized),), mi, dtype=torch.long, device=DEVICE)
        mpc_start = torch.full((len(dt_normalized),), 0.5, device=DEVICE)
        dt_tensor = torch.tensor(dt_normalized, dtype=torch.float32, device=DEVICE)
        recovered = model.r(mpc_start, dt_tensor, m_idx).cpu().numpy()
        tau = math.exp(model.r.log_tau[mi].item())
        ax.plot(dt_hours, recovered, label=f"{ALL_MUSCLES[mi].replace('_', ' ')} (τ={tau:.0f}h)",
                color=muscle_colors[mi], linewidth=1.5, alpha=0.8)

ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("MPC")
ax.set_title("Learned Recovery Curves (starting MPC = 0.5) — Exponential τ per muscle")
ax.legend(fontsize=7, ncol=4, loc="lower right")
ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_recovery_curves.png", dpi=150)
print("Saved chart_recovery_curves.png")

# --- Chart 5: Fatigue Heatmaps ---
_heatmap_exercises = ["bench_press", "squat", "deadlift", "ohp", "pendlay_row", "dips"]
_heatmap_exercises = [e for e in _heatmap_exercises if e in EXERCISE_TO_IDX]
_ncols = 3
_nrows = math.ceil(len(_heatmap_exercises) / _ncols)

# Per-exercise reps p5/p95 from training data
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

        # Grey mask where no training data exists (off-manifold)
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

plt.suptitle("Learned Fatigue (f) — Per-Muscle Breakdown\n(per-exercise weight/reps range; grey = no training data)",
             fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_fatigue_heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved chart_fatigue_heatmaps.png")

# ── Chart 6: Cross-exercise fatigue transfer matrix ──────────────────────────
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
            wb = torch.tensor([0.4], dtype=torch.float32, device=DEVICE)
            rb = torch.tensor([0.27], dtype=torch.float32, device=DEVICE)
            rir_fresh = model.g_net(wb, rb, eeb, mpc_b).item() * RIR_SCALE

            mpc = torch.ones(1, NUM_MUSCLES, device=DEVICE)
            eea = model.exercise_embed(torch.tensor([EXERCISE_TO_IDX[ea]], device=DEVICE))
            me_all = model.muscle_embed(torch.arange(NUM_MUSCLES, device=DEVICE)).unsqueeze(0)
            Ed = me_all.shape[-1]
            for _ in range(4):
                inv = model.involvement[torch.tensor([EXERCISE_TO_IDX[ea]], device=DEVICE)]
                drop = model.f_net(
                    torch.tensor([0.4], dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    torch.tensor([0.27], dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    torch.tensor([0.4], dtype=torch.float32, device=DEVICE).expand(NUM_MUSCLES),
                    mpc.reshape(-1),
                    eea.expand(NUM_MUSCLES, -1),
                    me_all.reshape(NUM_MUSCLES, Ed),
                ).reshape(1, NUM_MUSCLES)
                mpc = (mpc * (1.0 - inv * drop)).clamp(min=0.1)
            rir_fat = model.g_net(wb, rb, eeb, mpc).item() * RIR_SCALE
            tmat[i, j] = rir_fresh - rir_fat

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
plt.savefig(f"{CHART_DIR}/chart_transfer_matrix.png", dpi=150)
print("Saved chart_transfer_matrix.png")

# ── Chart 7: Per-muscle fatigue breakdown ────────────────────────────────────
_probe_ex = [e for e in ["bench_press", "squat", "deadlift", "ohp", "pendlay_row", "dips"]
             if e in EXERCISE_TO_IDX]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
with torch.no_grad():
    for idx, ex in enumerate(_probe_ex):
        ax = axes[idx // 3, idx % 3]
        ei = EXERCISE_TO_IDX[ex]
        ms = EXERCISE_MUSCLES[ex]
        ee = model.exercise_embed(torch.tensor([ei], device=DEVICE))
        involved = sorted(ms.keys(), key=lambda m: ms[m], reverse=True)
        drops, cols = [], []
        for m in involved:
            me = model.muscle_embed(torch.tensor([MUSCLE_TO_IDX[m]], device=DEVICE))
            d = model.f_net(
                torch.tensor([0.4], dtype=torch.float32, device=DEVICE),
                torch.tensor([0.27], dtype=torch.float32, device=DEVICE),
                torch.tensor([0.4], dtype=torch.float32, device=DEVICE),
                torch.tensor([1.0], dtype=torch.float32, device=DEVICE),
                ee, me,
            ).item()
            drops.append(d * ms[m])
            cols.append(muscle_colors[MUSCLE_TO_IDX[m]])
        labs = [m.replace("_", " ") for m in involved]
        ax.barh(range(len(labs)), drops, color=cols, alpha=0.85)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=9)
        ax.set_xlabel("MPC Drop")
        ax.set_title(f"{ex.replace('_', ' ').title()}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        for i, (d, m) in enumerate(zip(drops, involved)):
            ax.text(d + 0.001, i, f"inv={ms[m]:.2f}", va="center", fontsize=7, color="gray")
plt.suptitle("Learned Fatigue (f) — Per-Muscle Breakdown", fontsize=14)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_muscle_breakdown.png", dpi=150)
print("Saved chart_muscle_breakdown.png")

# ── Chart 8: RIR sensitivity per exercise ────────────────────────────────────
_sens_ex = [e for e in ["bench_press", "squat", "deadlift", "ohp", "pendlay_row", "lat_pulldown"]
            if e in EXERCISE_TO_IDX]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
with torch.no_grad():
    for idx, ex in enumerate(_sens_ex):
        ax = axes[idx // 3, idx % 3]
        ei = EXERCISE_TO_IDX[ex]
        ee = model.exercise_embed(torch.tensor([ei], device=DEVICE))
        w = torch.tensor([0.4], dtype=torch.float32, device=DEVICE)
        r = torch.tensor([0.27], dtype=torch.float32, device=DEVICE)
        mpc_base = torch.ones(1, NUM_MUSCLES, device=DEVICE)
        rir_base = model.g_net(w, r, ee, mpc_base).item() * RIR_SCALE
        deltas = {}
        for mi in range(NUM_MUSCLES):
            mpc_test = torch.ones(1, NUM_MUSCLES, device=DEVICE)
            mpc_test[0, mi] = 0.5
            rir_test = model.g_net(w, r, ee, mpc_test).item() * RIR_SCALE
            deltas[ALL_MUSCLES[mi]] = rir_test - rir_base
        sd = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        names = [m.replace("_", " ") for m, _ in sd]
        vals = [d for _, d in sd]
        cols = ["#e74c3c" if d < 0 else "#2ecc71" for d in vals]
        ax.barh(range(len(names)), vals, color=cols, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("RIR change (muscle at 50%)")
        ax.set_title(f"{ex.replace('_', ' ').title()} (baseline RIR={rir_base:.1f})")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
plt.suptitle("RIR Sensitivity (g) — Which Muscle Fatigue Affects RIR Most?", fontsize=14)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart_rir_sensitivity.png", dpi=150)
print("Saved chart_rir_sensitivity.png")

# ── Chart 9: MPC per muscle per user (per-muscle subplots, smooth recovery) ──
INTERP_PER_GAP = 20
for seq in sample_users:
    ts = seq["timestamps"]
    t0 = ts[0]
    hours_all = np.array([(t - t0) / np.timedelta64(1, "h") for t in ts])
    mask_3w = hours_all <= 504
    T_plot = max(int(mask_3w.sum()), min(len(seq["exercise_idx"]), 300))

    with torch.no_grad():
        _, mh = model.forward_with_mpc_history(
            seq["exercise_idx"][:T_plot].unsqueeze(0).to(DEVICE),
            seq["weight"][:T_plot].unsqueeze(0).to(DEVICE),
            seq["reps"][:T_plot].unsqueeze(0).to(DEVICE),
            seq["rir"][:T_plot].unsqueeze(0).to(DEVICE),
            seq["delta_t"][:T_plot].unsqueeze(0).to(DEVICE))

    hours = hours_all[:T_plot]
    mpc_post = mh[1:]

    smooth_hours, smooth_mpc = [], []
    with torch.no_grad():
        for t in range(T_plot):
            smooth_hours.append(hours[t])
            smooth_mpc.append(mpc_post[t])
            if t < T_plot - 1:
                gap_h = hours[t + 1] - hours[t]
                if gap_h > 0.5:
                    n_pts = min(INTERP_PER_GAP, max(3, int(gap_h / 2)))
                    for dt_h in np.linspace(0.5, gap_h - 0.01, n_pts):
                        mpc_start = torch.tensor(mpc_post[t], dtype=torch.float32, device=DEVICE)
                        dt_norm = torch.full((NUM_MUSCLES,), float(np.log1p(dt_h) / DT_SCALE),
                                            dtype=torch.float32, device=DEVICE)
                        m_idx = torch.arange(NUM_MUSCLES, dtype=torch.long, device=DEVICE)
                        recovered = model.r(mpc_start, dt_norm, m_idx).cpu().numpy()
                        smooth_hours.append(hours[t] + dt_h)
                        smooth_mpc.append(recovered)

    order = np.argsort(smooth_hours)
    smooth_hours = np.array(smooth_hours)[order]
    smooth_mpc = np.array(smooth_mpc)[order]
    days = smooth_hours / 24.0

    used = seq["exercise_idx"][:T_plot].numpy()
    active_muscles = [mi for mi in range(NUM_MUSCLES) if np.any(INVOLVEMENT_MATRIX[used, mi] > 0)]
    n_cols = 4
    n_rows = math.ceil(len(active_muscles) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), sharex=True, sharey=True)
    uid = seq["user_id"]
    fig.suptitle(f"{uid} — MPC per Muscle ({days[-1]:.0f} days, {T_plot} sets)", fontsize=15, y=1.01)

    for idx, mi in enumerate(active_muscles):
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        tau = math.exp(model.r.log_tau[mi].item())
        ax.plot(days, smooth_mpc[:, mi], color=muscle_colors[mi], linewidth=1.5)
        ax.fill_between(days, smooth_mpc[:, mi], 1.0, alpha=0.08, color=muscle_colors[mi])
        ax.set_title(f"{ALL_MUSCLES[mi].replace('_', ' ')} (τ={tau:.0f}h)", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Days")
        if idx % n_cols == 0:
            ax.set_ylabel("MPC")

    for idx in range(len(active_muscles), n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/chart_mpc_per_muscle_{uid}.png", dpi=150)
    print(f"Saved chart_mpc_per_muscle_{uid}.png")

print("\nDone! All charts saved.")
