"""
Parity test: PyTorch inference.py vs exported TFLite + assets.

This is the contract the Dart port must also satisfy. We replay random
histories through both pipelines and assert per-muscle MPC / RIR outputs
match within tolerance.

Pipeline B (TFLite + assets) deliberately reimplements the predict_mpc
orchestration in pure NumPy — that's the spec the Dart side will mirror.

Run after export.py:
  python parity_test.py --checkpoint ../../deepgain_model_muscle_ord.pt
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import os
os.chdir(REPO_ROOT)

from inference import (  # noqa: E402
    EXERCISE_TO_IDX,
    load_model,
    predict_mpc,
    predict_rir,
)
from strength_priors import (  # noqa: E402
    DEFAULT_ANCHOR_VALUES_KG,
    build_anchor_history_from_completed_sets,
    coerce_anchor_values,
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# ──────────────────────────────────────────────────────────────────────────────
# TFLite-side reimplementation (the future Dart spec)
# ──────────────────────────────────────────────────────────────────────────────


class TFLiteOrchestrator:
    """Replicates predict_mpc / predict_rir using only the TFLite models +
    JSON assets. Pure NumPy — no torch import.
    """

    def __init__(self, assets_dir: Path):
        self.exercises = json.loads((assets_dir / "exercises.json").read_text())
        self.muscles = json.loads((assets_dir / "muscles.json").read_text())
        self.exercise_to_idx = {e: i for i, e in enumerate(self.exercises)}
        self.num_muscles = len(self.muscles)

        scales = json.loads((assets_dir / "scales.json").read_text())
        self.W = scales["WEIGHT_SCALE"]
        self.R = scales["REPS_SCALE"]
        self.RIR_SCALE = scales["RIR_SCALE"]
        self.DT = scales["DT_SCALE"]

        self.involvement = np.array(
            json.loads((assets_dir / "involvement_matrix.json").read_text()),
            dtype=np.float32,
        )
        self.tau = np.array(
            json.loads((assets_dir / "fixed_tau.json").read_text()),
            dtype=np.float32,
        )
        self.default_anchors = np.array(
            json.loads((assets_dir / "default_anchors_kg.json").read_text())["values_kg"],
            dtype=np.float32,
        )

        self.f_interp = tf.lite.Interpreter(model_path=str(assets_dir / "f_net.tflite"))
        self.f_interp.allocate_tensors()
        self.g_interp = tf.lite.Interpreter(model_path=str(assets_dir / "g_net.tflite"))
        self.g_interp.allocate_tensors()

    def _run(self, interp, named_inputs):
        details = interp.get_input_details()
        # Match inputs by shape — tflite input order is the wrapper's positional order.
        for det, value in zip(details, named_inputs):
            arr = np.asarray(value, dtype=_dtype_for(det))
            if arr.shape != tuple(det["shape"]):
                arr = arr.reshape(det["shape"])
            interp.set_tensor(det["index"], arr)
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]["index"])
        return out

    def predict_drop(self, exercise_idx, weight_n, reps_n, rir_n, mpc, anchors_n):
        return self._run(
            self.f_interp,
            [
                np.int64(exercise_idx),
                np.float32(weight_n),
                np.float32(reps_n),
                np.float32(rir_n),
                mpc.astype(np.float32),
                anchors_n.astype(np.float32),
            ],
        )

    def predict_rir_norm(self, exercise_idx, weight_n, reps_n, mpc_all, anchors_n):
        return self._run(
            self.g_interp,
            [
                np.int64(exercise_idx),
                np.float32(weight_n),
                np.float32(reps_n),
                mpc_all.astype(np.float32),
                anchors_n.astype(np.float32),
            ],
        )

    def recovery(self, mpc, dt_hours):
        if dt_hours <= 0:
            return mpc
        return 1.0 - (1.0 - mpc) * np.exp(-dt_hours / self.tau)

    def predict_mpc(self, history, timestamp, anchors_kg=None):
        ts_q = _parse_ts(timestamp)
        if anchors_kg is None:
            anchors_kg = self.default_anchors.copy()
        anchors_n = anchors_kg.astype(np.float32) / self.W

        valid = []
        for entry in history:
            ts = _parse_ts(entry["timestamp"])
            if ts > ts_q:
                continue
            ex = entry.get("exercise")
            if ex not in self.exercise_to_idx:
                continue
            valid.append({
                "exercise_idx": self.exercise_to_idx[ex],
                "weight_n": float(entry["weight_kg"]) / self.W,
                "reps_n": float(entry["reps"]) / self.R,
                "rir_n": float(entry["rir"]) / self.RIR_SCALE,
                "timestamp": ts,
            })

        if not valid:
            return {m: 1.0 for m in self.muscles}

        valid.sort(key=lambda x: x["timestamp"])

        # Replay dynamic per-set 1RM anchor updates the way inference.predict_mpc does.
        # In Phase 3 the Dart port reimplements this; for Phase 1 we reuse the Python
        # reference so we're isolating "did TFLite preserve the model's numerics?".
        completed_for_anchors = [
            {
                "exercise": [e for e, i in self.exercise_to_idx.items() if i == s["exercise_idx"]][0],
                "weight_kg": s["weight_n"] * self.W,
                "reps": int(round(s["reps_n"] * self.R)),
                "rir": s["rir_n"] * self.RIR_SCALE,
                "timestamp": s["timestamp"],
            }
            for s in valid
        ]
        anchor_history_kg, _ = build_anchor_history_from_completed_sets(
            anchors_kg, completed_for_anchors, apply_trailing_session=True,
        )

        mpc = np.ones(self.num_muscles, dtype=np.float32)
        prev_ts = valid[0]["timestamp"]

        for i, (s, anchors_step_kg) in enumerate(zip(valid, anchor_history_kg)):
            if i > 0:
                dt_h = (s["timestamp"] - prev_ts).total_seconds() / 3600.0
                mpc = self.recovery(mpc, dt_h)
            inv = self.involvement[s["exercise_idx"]]
            anchors_step_n = anchors_step_kg.astype(np.float32) / self.W
            drop = self.predict_drop(
                s["exercise_idx"], s["weight_n"], s["reps_n"], s["rir_n"], mpc, anchors_step_n
            )
            mpc = np.clip(mpc * (1.0 - inv * drop), 0.1, None)
            prev_ts = s["timestamp"]

        dt_final = (ts_q - prev_ts).total_seconds() / 3600.0
        mpc = self.recovery(mpc, dt_final)

        return {m: float(mpc[i]) for i, m in enumerate(self.muscles)}

    def predict_rir(self, state, exercise, weight_kg, reps, anchors_kg=None):
        if exercise not in self.exercise_to_idx:
            raise ValueError(f"Unknown exercise: {exercise}")
        if anchors_kg is None:
            anchors_kg = self.default_anchors.copy()
        anchors_n = anchors_kg.astype(np.float32) / self.W
        mpc_all = np.array([state.get(m, 1.0) for m in self.muscles], dtype=np.float32)
        rir_norm = self.predict_rir_norm(
            self.exercise_to_idx[exercise],
            float(weight_kg) / self.W,
            float(reps) / self.R,
            mpc_all,
            anchors_n,
        )
        return float(np.clip(float(rir_norm) * self.RIR_SCALE, 0.0, 5.0))


def _dtype_for(det):
    return np.dtype(det["dtype"]) if "dtype" in det else np.float32


def _parse_ts(ts):
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)


# ──────────────────────────────────────────────────────────────────────────────
# Random scenario generation
# ──────────────────────────────────────────────────────────────────────────────


def gen_history(rng, exercise_pool, n_sets, base_time):
    history = []
    t = base_time
    for _ in range(n_sets):
        history.append({
            "exercise": rng.choice(exercise_pool),
            "weight_kg": round(rng.uniform(20, 180), 1),
            "reps": rng.randint(1, 12),
            "rir": rng.randint(0, 4),
            "timestamp": t.isoformat(),
        })
        t += timedelta(minutes=rng.randint(2, 90))
    return history, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "deepgain_model_muscle_ord.pt"),
    )
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--rir-checks-per-scenario", type=int, default=3)
    parser.add_argument("--mpc-tol", type=float, default=1e-4)
    parser.add_argument("--rir-tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading PyTorch model from {args.checkpoint}")
    py_model = load_model(args.checkpoint, device="cpu")

    print(f"Loading TFLite assets from {ASSETS_DIR}")
    orch = TFLiteOrchestrator(ASSETS_DIR)

    rng = random.Random(args.seed)
    exercise_pool = list(EXERCISE_TO_IDX.keys())

    mpc_max_diff = 0.0
    rir_max_diff = 0.0
    failures = []

    for scenario in range(args.n_scenarios):
        n_sets = rng.randint(0, args.max_history)
        base = datetime(2026, 1, 1, 9, 0, 0) + timedelta(days=rng.randint(0, 30))
        history, last_t = gen_history(rng, exercise_pool, n_sets, base)
        query_t = last_t + timedelta(hours=rng.uniform(0, 72))

        # Randomize anchors so we exercise the strength-feature path.
        anchors_kg = np.array([
            rng.uniform(60, 180),
            rng.uniform(80, 250),
            rng.uniform(100, 320),
        ], dtype=np.float32)

        py_mpc = predict_mpc(py_model, history, query_t.isoformat(), strength_anchors=anchors_kg)
        tf_mpc = orch.predict_mpc(history, query_t.isoformat(), anchors_kg=anchors_kg)

        for muscle in py_mpc:
            diff = abs(py_mpc[muscle] - tf_mpc[muscle])
            mpc_max_diff = max(mpc_max_diff, diff)
            if diff > args.mpc_tol:
                failures.append(
                    f"scenario {scenario} muscle {muscle}: "
                    f"py={py_mpc[muscle]:.6f} tf={tf_mpc[muscle]:.6f} diff={diff:.6f}"
                )

        # RIR checks.
        for _ in range(args.rir_checks_per_scenario):
            ex = rng.choice(exercise_pool)
            w = round(rng.uniform(20, 180), 1)
            r = rng.randint(1, 12)
            py_rir = predict_rir(py_model, py_mpc, ex, w, r, strength_anchors=anchors_kg)
            tf_rir = orch.predict_rir(tf_mpc, ex, w, r, anchors_kg=anchors_kg)
            diff = abs(py_rir - tf_rir)
            rir_max_diff = max(rir_max_diff, diff)
            if diff > args.rir_tol:
                failures.append(
                    f"scenario {scenario} rir({ex}, {w}kg×{r}): "
                    f"py={py_rir:.4f} tf={tf_rir:.4f} diff={diff:.4f}"
                )

        if (scenario + 1) % 10 == 0:
            print(f"  {scenario + 1}/{args.n_scenarios} done | "
                  f"mpc_max_diff={mpc_max_diff:.2e} rir_max_diff={rir_max_diff:.2e}")

    print()
    print(f"MPC max abs diff: {mpc_max_diff:.2e}  (tol {args.mpc_tol})")
    print(f"RIR max abs diff: {rir_max_diff:.2e}  (tol {args.rir_tol})")

    if failures:
        print(f"\n{len(failures)} failures (showing first 10):")
        for f in failures[:10]:
            print(f"  {f}")
        sys.exit(1)
    print("\nPARITY OK")


if __name__ == "__main__":
    main()
