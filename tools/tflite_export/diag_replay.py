"""Step through a multi-set scenario in both pipelines, print MPC after each set."""

import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from inference import (  # noqa: E402
    ALL_MUSCLES, EXERCISE_TO_IDX, NUM_MUSCLES, REPS_SCALE, RIR_SCALE,
    WEIGHT_SCALE, load_model,
)
import torch  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
DT_SCALE = float(np.log1p(168.0))


def main():
    rng = random.Random(42)
    n_sets = rng.randint(0, 20)
    base = datetime(2026, 1, 1, 9, 0, 0) + timedelta(days=rng.randint(0, 30))
    pool = list(EXERCISE_TO_IDX.keys())
    history = []
    t = base
    for _ in range(n_sets):
        history.append({
            "exercise": rng.choice(pool),
            "weight_kg": round(rng.uniform(20, 180), 1),
            "reps": rng.randint(1, 12),
            "rir": rng.randint(0, 4),
            "timestamp": t.isoformat(),
        })
        t += timedelta(minutes=rng.randint(2, 90))

    print(f"Scenario 0: {len(history)} sets")
    for i, h in enumerate(history):
        print(f"  {i}: {h['exercise']:20s} {h['weight_kg']:5.1f}kg × {h['reps']:2d} @ rir={h['rir']}  t={h['timestamp']}")

    # ── PyTorch path (replicates inference.predict_mpc internals) ─────────────
    py_model = load_model("deepgain_model_muscle_ord.pt", device="cpu")
    M = NUM_MUSCLES
    all_m_idx = torch.arange(M)
    all_m_embed = py_model.muscle_embed(all_m_idx)
    valid = []
    for h in history:
        valid.append({
            "exercise": h["exercise"],
            "exercise_idx": EXERCISE_TO_IDX[h["exercise"]],
            "weight": float(h["weight_kg"]) / WEIGHT_SCALE,
            "reps": float(h["reps"]) / REPS_SCALE,
            "rir": float(h["rir"]) / RIR_SCALE,
            "timestamp": datetime.fromisoformat(h["timestamp"]),
        })
    valid.sort(key=lambda x: x["timestamp"])

    py_mpc = torch.ones(1, M)
    prev_ts = valid[0]["timestamp"]
    py_steps = []
    py_drops = []
    with torch.no_grad():
        for i, s in enumerate(valid):
            if i > 0:
                dt_h = (s["timestamp"] - prev_ts).total_seconds() / 3600.0
                if dt_h > 0:
                    dt_norm = torch.tensor([np.log1p(dt_h) / DT_SCALE], dtype=torch.float32).expand(M)
                    py_mpc = py_model.r(py_mpc.reshape(-1), dt_norm, all_m_idx).reshape(1, M)
            ei = torch.tensor([s["exercise_idx"]], dtype=torch.long)
            inv = py_model.involvement[ei]
            w_exp = torch.full((M,), s["weight"], dtype=torch.float32)
            r_exp = torch.full((M,), s["reps"], dtype=torch.float32)
            rir_exp = torch.full((M,), s["rir"], dtype=torch.float32)
            anchors_t = torch.tensor([[100.0/200, 140.0/200, 180.0/200]], dtype=torch.float32)
            drop = py_model.predict_drop_norm(
                ei.expand(M), w_exp, r_exp, rir_exp, py_mpc.reshape(-1),
                all_m_embed.reshape(-1, all_m_embed.shape[-1]), anchors_t,
            )
            py_drops.append(drop.numpy().copy())
            py_mpc = (py_mpc * (1.0 - inv * drop.reshape(1, M))).clamp(min=0.1)
            py_steps.append(py_mpc.numpy().copy())
            prev_ts = s["timestamp"]

    # ── TFLite path ───────────────────────────────────────────────────────────
    # Disable XNNPACK delegate to rule out fused-op precision drift
    interp = tf.lite.Interpreter(
        model_path=str(ASSETS / "f_net.tflite"),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    in_dets = interp.get_input_details()
    out_det = interp.get_output_details()[0]
    inv_mat = np.array(json.loads((ASSETS / "involvement_matrix.json").read_text()), dtype=np.float32)
    tau = np.array(json.loads((ASSETS / "fixed_tau.json").read_text()), dtype=np.float32)
    anchors_n = np.array([100.0/200, 140.0/200, 180.0/200], dtype=np.float32)

    def call_f(ex_idx, w_n, r_n, rir_n, mpc_arr, anchors_arr):
        vals = [np.int64(ex_idx), np.float32(w_n), np.float32(r_n),
                np.float32(rir_n), mpc_arr.astype(np.float32), anchors_arr.astype(np.float32)]
        for det, v in zip(in_dets, vals):
            interp.set_tensor(det["index"], np.asarray(v, dtype=det["dtype"]).reshape(det["shape"]))
        interp.invoke()
        return interp.get_tensor(out_det["index"])

    tf_mpc = np.ones(M, dtype=np.float32)
    prev_ts = valid[0]["timestamp"]
    tf_steps = []
    tf_drops = []
    for i, s in enumerate(valid):
        if i > 0:
            dt_h = (s["timestamp"] - prev_ts).total_seconds() / 3600.0
            if dt_h > 0:
                tf_mpc = 1.0 - (1.0 - tf_mpc) * np.exp(-dt_h / tau)
        inv = inv_mat[s["exercise_idx"]]
        drop = call_f(s["exercise_idx"], s["weight"], s["reps"], s["rir"], tf_mpc, anchors_n)
        tf_drops.append(drop.copy())
        tf_mpc = np.clip(tf_mpc * (1.0 - inv * drop), 0.1, None)
        tf_steps.append(tf_mpc.copy())
        prev_ts = s["timestamp"]

    # Drop diff at set 0 — should be ~1e-7 if everything's correct
    print(f"\n--- DROP DIFF at set 0 (no recovery yet, fresh mpc) ---")
    d0 = np.abs(py_drops[0] - tf_drops[0])
    print(f"py drops[0]: {py_drops[0]}")
    print(f"tf drops[0]: {tf_drops[0]}")
    print(f"max abs diff: {d0.max():.2e}")
    print(f"\n--- DROP DIFF max per set ---")
    for i in range(min(5, len(py_drops))):
        d = np.abs(py_drops[i] - tf_drops[i])
        print(f"  set {i}: max drop diff = {d.max():.2e}")

    # ── Diff ──────────────────────────────────────────────────────────────────
    print(f"\n{'set':>3} {'exercise':20s} {'max_diff':>12s}  {'worst muscle (py → tf)':40s}")
    for i, (py_s, tf_s) in enumerate(zip(py_steps, tf_steps)):
        diff = np.abs(py_s.flatten() - tf_s)
        worst_m = int(diff.argmax())
        print(f"{i:3d} {valid[i]['exercise']:20s} {diff.max():12.2e}  "
              f"{ALL_MUSCLES[worst_m]:15s} {py_s.flatten()[worst_m]:.4f} → {tf_s[worst_m]:.4f}")


if __name__ == "__main__":
    main()
