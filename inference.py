"""
DeepGain inference API — stable contract for external consumers.

Usage:
    from inference import load_model, predict_mpc, predict_rir

    model = load_model("deepgain_model_muscle_ord.pt")

    history = [
        {"exercise": "bench_press", "weight_kg": 80.0, "reps": 5, "rir": 2,
         "timestamp": "2024-01-01T10:00:00"},
        {"exercise": "ohp", "weight_kg": 50.0, "reps": 6, "rir": 3,
         "timestamp": "2024-01-01T10:20:00"},
    ]
    mpc = predict_mpc(model, history, timestamp="2024-01-02T09:00:00")
    # {"chest": 0.83, "triceps": 0.91, "anterior_delts": 0.88, ...}

    rir = predict_rir(model, mpc, exercise="bench_press", weight=80.0, reps=5)
    # 1.8
"""

import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# ─── Device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _DEFAULT_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    _DEFAULT_DEVICE = torch.device("mps")
else:
    _DEFAULT_DEVICE = torch.device("cpu")

# ─── Constants (must stay in sync with train.py) ────────────────────────────
EMBED_DIM = 32
HIDDEN_DIM = 128
WEIGHT_SCALE = 200.0
REPS_SCALE = 30.0
RIR_SCALE = 5.0
DT_SCALE = np.log1p(168.0)

ALL_MUSCLES = [
    "chest", "anterior_delts", "lateral_delts", "rear_delts",
    "upper_traps", "rhomboids", "triceps", "biceps", "brachialis",
    "lats", "quads", "hamstrings", "glutes", "adductors", "erectors", "calves",
]
NUM_MUSCLES = len(ALL_MUSCLES)
MUSCLE_TO_IDX = {m: i for i, m in enumerate(ALL_MUSCLES)}

# Hardcoded involvement — will be replaced by exercise_muscle_order.yaml once delivered
EXERCISE_MUSCLES = {
    "bench_press":           {"chest": 0.85, "triceps": 0.55, "anterior_delts": 0.60},
    "incline_bench":         {"chest": 0.70, "anterior_delts": 0.75, "triceps": 0.50},
    "close_grip_bench":      {"chest": 0.65, "triceps": 0.75, "anterior_delts": 0.55},
    "dumbbell_bench":        {"chest": 0.82, "triceps": 0.45, "anterior_delts": 0.55},
    "ohp":                   {"anterior_delts": 0.85, "triceps": 0.65, "chest": 0.20, "upper_traps": 0.40},
    "dumbbell_ohp":          {"anterior_delts": 0.80, "triceps": 0.60, "upper_traps": 0.35},
    "dips":                  {"chest": 0.70, "triceps": 0.65, "anterior_delts": 0.45},
    "barbell_row":           {"lats": 0.80, "biceps": 0.55, "rear_delts": 0.50, "erectors": 0.40, "upper_traps": 0.35, "rhomboids": 0.45},
    "lat_pulldown":          {"lats": 0.75, "biceps": 0.50, "rear_delts": 0.35, "rhomboids": 0.40},
    "cable_row":             {"lats": 0.70, "biceps": 0.45, "rear_delts": 0.40, "rhomboids": 0.50, "upper_traps": 0.30},
    "pull_up":               {"lats": 0.82, "biceps": 0.55, "rear_delts": 0.35, "rhomboids": 0.40},
    "squat":                 {"quads": 0.85, "glutes": 0.60, "hamstrings": 0.35, "erectors": 0.45, "adductors": 0.40},
    "front_squat":           {"quads": 0.90, "glutes": 0.50, "erectors": 0.55, "adductors": 0.35},
    "deadlift":              {"glutes": 0.70, "hamstrings": 0.55, "erectors": 0.80, "quads": 0.40, "upper_traps": 0.50, "lats": 0.30, "adductors": 0.35},
    "rdl":                   {"hamstrings": 0.80, "glutes": 0.55, "erectors": 0.50, "adductors": 0.25},
    "leg_press":             {"quads": 0.80, "glutes": 0.50, "adductors": 0.35},
    "bulgarian_split_squat": {"quads": 0.80, "glutes": 0.65, "hamstrings": 0.30, "adductors": 0.40},
    "hip_thrust":            {"glutes": 0.85, "hamstrings": 0.40, "adductors": 0.30},
    "tricep_pushdown":       {"triceps": 0.90},
    "overhead_tricep_ext":   {"triceps": 0.85},
    "bicep_curl":            {"biceps": 0.90},
    "hammer_curl":           {"biceps": 0.75, "brachialis": 0.60},
    "lateral_raise":         {"lateral_delts": 0.85, "upper_traps": 0.30},
    "face_pull":             {"rear_delts": 0.70, "upper_traps": 0.40, "rhomboids": 0.35},
    "leg_curl":              {"hamstrings": 0.85},
    "leg_extension":         {"quads": 0.85},
    "calf_raise":            {"calves": 0.90},
}

ALL_EXERCISES = list(EXERCISE_MUSCLES.keys())
NUM_EXERCISES = len(ALL_EXERCISES)
EXERCISE_TO_IDX = {e: i for i, e in enumerate(ALL_EXERCISES)}

INVOLVEMENT_MATRIX = np.zeros((NUM_EXERCISES, NUM_MUSCLES), dtype=np.float32)
for _ex, _ms in EXERCISE_MUSCLES.items():
    _ei = EXERCISE_TO_IDX[_ex]
    for _m, _c in _ms.items():
        INVOLVEMENT_MATRIX[_ei, MUSCLE_TO_IDX[_m]] = _c


# ─── Model Architecture (mirrors train.py) ──────────────────────────────────

class FatigueNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 2 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, weight, reps, rir, mpc, e_embed, m_embed):
        x = torch.cat([
            weight.unsqueeze(-1), reps.unsqueeze(-1),
            rir.unsqueeze(-1), mpc.unsqueeze(-1),
            e_embed, m_embed,
        ], dim=-1)
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
        x = torch.cat([
            weight.unsqueeze(-1), reps.unsqueeze(-1),
            e_embed, mpc_all,
        ], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


class ExponentialRecovery(nn.Module):
    # Literature-derived τ values (hours) — matches train.py exactly
    FIXED_TAU = [
        16.0,  # chest
        13.0,  # anterior_delts
         9.0,  # lateral_delts
         8.0,  # rear_delts
         9.0,  # upper_traps
        10.0,  # rhomboids
         9.0,  # triceps
        13.0,  # biceps
        13.0,  # brachialis
        13.0,  # lats
        19.0,  # quads
        18.0,  # hamstrings
        15.0,  # glutes
        12.0,  # adductors
        12.0,  # erectors
         8.0,  # calves
    ]

    def __init__(self, num_muscles):
        super().__init__()
        init_tau = torch.tensor([math.log(t) for t in self.FIXED_TAU])
        self.log_tau = nn.Parameter(init_tau, requires_grad=False)

    def forward(self, mpc, delta_t, muscle_idx):
        """mpc: (N,), delta_t: (N,) normalized, muscle_idx: (N,) long."""
        dt_hours = torch.expm1(delta_t * DT_SCALE)
        tau = torch.exp(self.log_tau[muscle_idx])
        return 1.0 - (1.0 - mpc) * torch.exp(-dt_hours / tau)


class DeepGainModel(nn.Module):
    def __init__(self, num_exercises, num_muscles, embed_dim, hidden_dim):
        super().__init__()
        self.num_muscles = num_muscles
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.muscle_embed = nn.Embedding(num_muscles, embed_dim)
        self.f_net = FatigueNet(embed_dim, hidden_dim)
        self.g_net = RIRNet(embed_dim, hidden_dim, num_muscles)
        self.r = ExponentialRecovery(num_muscles)
        self.register_buffer(
            "involvement",
            torch.tensor(INVOLVEMENT_MATRIX, dtype=torch.float32),
        )


# ─── Public API ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device=None) -> DeepGainModel:
    """Load a trained DeepGain model from a checkpoint file.

    Args:
        checkpoint_path: Path to a .pt file saved by train.py.
        device: torch.device to load onto. Defaults to best available GPU/MPS/CPU.

    Returns:
        DeepGainModel in eval mode, ready for inference.
    """
    if device is None:
        device = _DEFAULT_DEVICE
    model = DeepGainModel(NUM_EXERCISES, NUM_MUSCLES, EMBED_DIM, HIDDEN_DIM)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict_mpc(
    model: DeepGainModel,
    user_history: list[dict],
    timestamp: str | datetime,
) -> dict[str, float]:
    """Estimate Muscle Performance Capacity for all muscles at a given moment.

    Replays the user's training history through the model (recovery between
    sets, fatigue after each set), then applies recovery from the last set up
    to `timestamp`.

    Args:
        model: Loaded DeepGainModel (from load_model()).
        user_history: List of logged sets, each a dict with:
            - "exercise"   : str   — exercise_id (e.g. "bench_press")
            - "weight_kg"  : float — weight used
            - "reps"       : int   — reps performed
            - "rir"        : int   — reps in reserve actually logged (0-5)
            - "timestamp"  : str or datetime — when the set was performed
          Sets with unknown exercises are silently skipped.
          Sets after `timestamp` are excluded.
        timestamp: The moment for which to estimate MPC.
            Accepts a datetime object or an ISO8601 string.

    Returns:
        Dict mapping muscle_id -> MPC in [0.1, 1.0].
        All muscles are 1.0 (fully recovered) when there are no relevant sets.

    Example:
        mpc = predict_mpc(model, history, "2024-01-02T09:00:00")
        # {"chest": 0.72, "triceps": 0.88, "quads": 1.0, ...}
    """
    device = next(model.parameters()).device
    ts_query = _parse_timestamp(timestamp)

    # Filter to known exercises at or before the query timestamp, sort by time
    valid = []
    for entry in user_history:
        ts = _parse_timestamp(entry["timestamp"])
        if ts > ts_query:
            continue
        ex = entry.get("exercise", "")
        if ex not in EXERCISE_TO_IDX:
            continue
        valid.append({
            "exercise_idx": EXERCISE_TO_IDX[ex],
            "weight":  float(entry["weight_kg"]) / WEIGHT_SCALE,
            "reps":    float(entry["reps"]) / REPS_SCALE,
            "rir":     float(entry["rir"]) / RIR_SCALE,
            "timestamp": ts,
        })

    if not valid:
        return {m: 1.0 for m in ALL_MUSCLES}

    valid.sort(key=lambda x: x["timestamp"])

    M = model.num_muscles
    all_m_idx = torch.arange(M, device=device)
    all_m_embed = model.muscle_embed(all_m_idx)          # (M, E)
    E = all_m_embed.shape[-1]
    all_m_embed_exp = all_m_embed.unsqueeze(0)            # (1, M, E)
    m_idx_flat = all_m_idx                                # (M,)

    with torch.no_grad():
        mpc = torch.ones(1, M, device=device)             # (1, M)
        prev_ts = valid[0]["timestamp"]

        for i, s in enumerate(valid):
            # Recovery since previous set (skip for first set)
            if i > 0:
                dt_h = (s["timestamp"] - prev_ts).total_seconds() / 3600.0
                if dt_h > 0:
                    dt_norm = torch.tensor(
                        [np.log1p(dt_h) / DT_SCALE], dtype=torch.float32, device=device
                    ).expand(M)
                    mpc = model.r(mpc.reshape(-1), dt_norm, m_idx_flat).reshape(1, M)

            # Fatigue from this set
            ei = torch.tensor([s["exercise_idx"]], dtype=torch.long, device=device)
            e_embed = model.exercise_embed(ei)                            # (1, E)
            inv = model.involvement[ei]                                   # (1, M)

            e_emb_exp = e_embed.unsqueeze(1).expand(-1, M, -1).reshape(-1, E)  # (M, E)
            w_exp   = torch.full((M,), s["weight"], dtype=torch.float32, device=device)
            r_exp   = torch.full((M,), s["reps"],   dtype=torch.float32, device=device)
            rir_exp = torch.full((M,), s["rir"],    dtype=torch.float32, device=device)
            mpc_flat   = mpc.reshape(-1)                                  # (M,)
            m_emb_flat = all_m_embed_exp.reshape(-1, E)                   # (M, E)

            drop = model.f_net(w_exp, r_exp, rir_exp, mpc_flat, e_emb_exp, m_emb_flat)
            mpc = (mpc * (1.0 - inv * drop.reshape(1, M))).clamp(min=0.1)

            prev_ts = s["timestamp"]

        # Recovery from last set to query timestamp
        dt_final = (ts_query - prev_ts).total_seconds() / 3600.0
        if dt_final > 0:
            dt_norm = torch.tensor(
                [np.log1p(dt_final) / DT_SCALE], dtype=torch.float32, device=device
            ).expand(M)
            mpc = model.r(mpc.reshape(-1), dt_norm, m_idx_flat).reshape(1, M)

        mpc_np = mpc[0].cpu().numpy()

    return {muscle: float(mpc_np[i]) for i, muscle in enumerate(ALL_MUSCLES)}


def predict_rir(
    model: DeepGainModel,
    state: dict[str, float],
    exercise: str,
    weight: float,
    reps: int,
) -> float:
    """Predict RIR for a planned set given the current muscle state.

    Args:
        model: Loaded DeepGainModel (from load_model()).
        state: Dict mapping muscle_id -> MPC in [0.0, 1.0].
               Typically the output of predict_mpc().
               Muscles missing from the dict default to 1.0.
        exercise: Exercise ID (e.g. "bench_press").
        weight: Weight in kg.
        reps: Number of planned reps.

    Returns:
        Predicted RIR in [0.0, 5.0].

    Raises:
        ValueError: If exercise is not in the model's exercise list.

    Example:
        mpc = predict_mpc(model, history, now)
        rir = predict_rir(model, mpc, "bench_press", 100.0, 5)
        # 1.8
    """
    if exercise not in EXERCISE_TO_IDX:
        raise ValueError(
            f"Unknown exercise '{exercise}'. "
            f"Known: {ALL_EXERCISES}"
        )

    device = next(model.parameters()).device

    mpc_vals = [state.get(m, 1.0) for m in ALL_MUSCLES]
    mpc_t = torch.tensor([mpc_vals], dtype=torch.float32, device=device)   # (1, M)
    w_t   = torch.tensor([weight / WEIGHT_SCALE], dtype=torch.float32, device=device)
    r_t   = torch.tensor([reps / REPS_SCALE],     dtype=torch.float32, device=device)
    ei    = torch.tensor([EXERCISE_TO_IDX[exercise]], dtype=torch.long, device=device)
    e_emb = model.exercise_embed(ei)                                         # (1, E)

    with torch.no_grad():
        rir_norm = model.g_net(w_t, r_t, e_emb, mpc_t)

    return float(np.clip(float(rir_norm.item()) * RIR_SCALE, 0.0, 5.0))


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_muscles() -> list[str]:
    """Return the canonical list of muscle IDs."""
    return list(ALL_MUSCLES)


def get_exercises() -> list[str]:
    """Return all exercise IDs recognized by the current model."""
    return list(ALL_EXERCISES)


def _parse_timestamp(ts) -> datetime:
    """Coerce timestamp to a naive datetime (no tzinfo)."""
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    if isinstance(ts, np.datetime64):
        import pandas as pd
        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
    s = str(ts)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
