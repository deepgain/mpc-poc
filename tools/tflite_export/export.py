"""
Export FatigueNet (f_net) and RIRNet (g_net) from a DeepGain checkpoint to
TFLite, plus dump constants used by the Dart-side orchestration.

The wrappers are designed so the Dart `predict_mpc` loop can call f_net once
per set (returning drop[15] for all 15 muscles) and g_net once per planned
set, with no embedding lookup or strength-feature computation in Dart.

Outputs (all written to ./assets/):
  f_net.tflite             FatigueModule  (set-level fatigue → drop[15])
  g_net.tflite             RIRModule      (planned set → rir_norm scalar)
  involvement_matrix.json  (NUM_EXERCISES, 15) per-exercise muscle weights
  anchor_ratio_matrix.json (NUM_EXERCISES, 3)  per-exercise 1RM ratios
  fixed_tau.json           [15] recovery time constants per muscle
  exercises.json           ordered exercise IDs (index = exercise_idx)
  muscles.json             ordered muscle IDs   (index = muscle_idx)
  scales.json              WEIGHT_SCALE / REPS_SCALE / RIR_SCALE / DT_SCALE
  default_anchors_kg.json  fallback bench/squat/deadlift 1RMs

Run:
  cd tools/tflite_export
  python export.py --checkpoint ../../deepgain_model_muscle_ord.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ai-edge-torch was renamed to litert-torch in 2025; both ship together but the
# old shim has no `convert` attr, so import from the new package directly.
import litert_torch

# Make the repo root importable so `inference` and `strength_priors` resolve.
# inference.py also reads exercise_muscle_order.yaml + exercise_muscle_weights_scaled.csv
# from CWD, so chdir there before importing.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from inference import (  # noqa: E402
    ALL_EXERCISES,
    ALL_MUSCLES,
    DT_SCALE,
    INVOLVEMENT_MATRIX,
    NUM_MUSCLES,
    REPS_SCALE,
    RIR_SCALE,
    WEIGHT_SCALE,
    load_model,
)
from strength_priors import (  # noqa: E402
    ANCHOR_NAMES,
    DEFAULT_ANCHOR_VALUES_KG,
    build_anchor_ratio_matrix,
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper modules — what we actually export
# ──────────────────────────────────────────────────────────────────────────────


class FatigueModule(nn.Module):
    """One forward pass = one set's fatigue contribution to all 15 muscles.

    Inputs (all batch-free for simplicity; Dart calls one set at a time):
        exercise_idx : int64 scalar
        weight       : float32 scalar (already / WEIGHT_SCALE)
        reps         : float32 scalar (already / REPS_SCALE)
        rir          : float32 scalar (already / RIR_SCALE)
        mpc          : float32 [15]   (current per-muscle MPC, in [0.1, 1.0])
        anchors      : float32 [3]    (bench, squat, deadlift in kg / WEIGHT_SCALE)

    Output:
        drop : float32 [15]  raw fatigue model output, sigmoid'd, in [0, 1]
    """

    def __init__(self, source_model):
        super().__init__()
        self.f_net = source_model.f_net
        # Bake embeddings as plain tensors so TFLite gets clean Gather/MatMul.
        self.exercise_embed_w = source_model.exercise_embed.weight.detach().clone()
        self.muscle_embed_w = source_model.muscle_embed.weight.detach().clone()
        self.anchor_ratio_matrix = source_model.anchor_ratio_matrix.detach().clone()
        self.projection_available = source_model.projection_available.detach().clone()
        self.strength_feature_dim = source_model.strength_feature_dim
        self.num_muscles = source_model.num_muscles

    def forward(self, exercise_idx, weight, reps, rir, mpc, anchors):
        # Embeddings.
        e_embed = self.exercise_embed_w.index_select(0, exercise_idx.view(1)).squeeze(0)  # (E,)
        m_embed_all = self.muscle_embed_w  # (15, E) constant

        # Strength features (mirrors DeepGainModel.compute_strength_features).
        if self.strength_feature_dim > 0:
            ratios = self.anchor_ratio_matrix.index_select(0, exercise_idx.view(1)).squeeze(0)  # (3,)
            projected = (anchors * ratios).sum(dim=-1)  # scalar
            available = self.projection_available.index_select(0, exercise_idx.view(1)).squeeze(0)
            relative_load = torch.where(
                projected > 1e-6,
                weight / projected.clamp_min(1e-6),
                torch.zeros_like(weight),
            )
            strength_feat = torch.cat(
                [
                    anchors,
                    projected.unsqueeze(-1),
                    relative_load.unsqueeze(-1),
                    available.unsqueeze(-1),
                ],
                dim=-1,
            )[: self.strength_feature_dim]  # (S,)
        else:
            strength_feat = None

        # Build per-muscle input rows: shape (15, 4 + S + 2E)
        M = self.num_muscles
        weight_v = weight.view(1).expand(M)
        reps_v = reps.view(1).expand(M)
        rir_v = rir.view(1).expand(M)
        e_embed_v = e_embed.view(1, -1).expand(M, -1)

        pieces = [
            weight_v.unsqueeze(-1),
            reps_v.unsqueeze(-1),
            rir_v.unsqueeze(-1),
            mpc.unsqueeze(-1),
        ]
        if strength_feat is not None:
            pieces.append(strength_feat.view(1, -1).expand(M, -1))
        pieces.extend([e_embed_v, m_embed_all])
        x = torch.cat(pieces, dim=-1)  # (15, F)

        # Run f_net.net manually so we can feed pre-built rows.
        return self.f_net.net(x).squeeze(-1)  # (15,)


class RIRModule(nn.Module):
    """One forward pass = predicted RIR for one planned set.

    Inputs:
        exercise_idx : int64 scalar
        weight       : float32 scalar (already / WEIGHT_SCALE)
        reps         : float32 scalar (already / REPS_SCALE)
        mpc_all      : float32 [15]
        anchors      : float32 [3]    (already / WEIGHT_SCALE)

    Output:
        rir_norm : float32 scalar (sigmoid output, multiply by RIR_SCALE in Dart)
    """

    def __init__(self, source_model):
        super().__init__()
        self.g_net = source_model.g_net
        self.exercise_embed_w = source_model.exercise_embed.weight.detach().clone()
        self.anchor_ratio_matrix = source_model.anchor_ratio_matrix.detach().clone()
        self.projection_available = source_model.projection_available.detach().clone()
        self.strength_feature_dim = source_model.strength_feature_dim

    def forward(self, exercise_idx, weight, reps, mpc_all, anchors):
        e_embed = self.exercise_embed_w.index_select(0, exercise_idx.view(1)).squeeze(0)  # (E,)

        if self.strength_feature_dim > 0:
            ratios = self.anchor_ratio_matrix.index_select(0, exercise_idx.view(1)).squeeze(0)
            projected = (anchors * ratios).sum(dim=-1)
            available = self.projection_available.index_select(0, exercise_idx.view(1)).squeeze(0)
            relative_load = torch.where(
                projected > 1e-6,
                weight / projected.clamp_min(1e-6),
                torch.zeros_like(weight),
            )
            strength_feat = torch.cat(
                [
                    anchors,
                    projected.unsqueeze(-1),
                    relative_load.unsqueeze(-1),
                    available.unsqueeze(-1),
                ],
                dim=-1,
            )[: self.strength_feature_dim]
            pieces = [weight.view(1), reps.view(1), strength_feat, e_embed, mpc_all]
        else:
            pieces = [weight.view(1), reps.view(1), e_embed, mpc_all]

        x = torch.cat(pieces, dim=-1).unsqueeze(0)  # (1, F) — RIRNet expects batch dim
        # RIRNet.forward applies sigmoid manually; we replicate by walking .net.
        # Return shape (1,) instead of scalar — Dart typed lists always have rank ≥ 1.
        return torch.sigmoid(self.g_net.net(x).squeeze(-1))


# ──────────────────────────────────────────────────────────────────────────────
# Export driver
# ──────────────────────────────────────────────────────────────────────────────


def _sample_inputs_fatigue():
    # All scalars are shape (1,) instead of () — Dart typed lists always
    # have rank ≥ 1, so matching avoids GATHER_ND index reinterpretation.
    return (
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0.4], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.4], dtype=torch.float32),
        torch.full((NUM_MUSCLES,), 0.9, dtype=torch.float32),
        torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32),
    )


def _sample_inputs_rir():
    return (
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0.4], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.full((NUM_MUSCLES,), 0.9, dtype=torch.float32),
        torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32),
    )


def export_models(checkpoint_path: str):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint_path}")
    model = load_model(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    fatigue = FatigueModule(model).eval()
    rir = RIRModule(model).eval()

    # Sanity-check the wrappers work in plain PyTorch first.
    with torch.no_grad():
        f_out = fatigue(*_sample_inputs_fatigue())
        g_out = rir(*_sample_inputs_rir())
    assert f_out.shape == (NUM_MUSCLES,), f"FatigueModule output shape {f_out.shape}"
    assert g_out.shape == (1,), f"RIRModule output shape {g_out.shape}"
    print(f"  PyTorch sanity: f_out[0]={f_out[0].item():.4f}, g_out={g_out[0].item():.4f}")

    # Convert with ai-edge-torch.
    print("Converting FatigueModule → f_net.tflite")
    f_edge = litert_torch.convert(fatigue, _sample_inputs_fatigue())
    f_path = ASSETS_DIR / "f_net.tflite"
    f_edge.export(str(f_path))
    print(f"  wrote {f_path} ({f_path.stat().st_size / 1024:.1f} KB)")

    print("Converting RIRModule → g_net.tflite")
    g_edge = litert_torch.convert(rir, _sample_inputs_rir())
    g_path = ASSETS_DIR / "g_net.tflite"
    g_edge.export(str(g_path))
    print(f"  wrote {g_path} ({g_path.stat().st_size / 1024:.1f} KB)")

    return model, fatigue, rir


def dump_assets(model):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-exercise involvement weights — read from the LOADED model, not from
    # the freshly recomputed INVOLVEMENT_MATRIX. The checkpoint can carry an
    # involvement buffer that differs from the current
    # exercise_muscle_weights_scaled.csv (CSV has been edited post-training).
    inv_from_model = model.involvement.detach().cpu().numpy()
    inv_path = ASSETS_DIR / "involvement_matrix.json"
    inv_path.write_text(json.dumps(inv_from_model.tolist()))
    drift = float(np.abs(inv_from_model - INVOLVEMENT_MATRIX).max())
    print(f"  wrote {inv_path}  shape={inv_from_model.shape}  "
          f"(drift vs current CSV: max abs diff = {drift:.4f})")

    # Anchor ratio matrix (NUM_EXERCISES, 3) — re-derive so it matches model.
    ratio_matrix, availability = build_anchor_ratio_matrix(ALL_EXERCISES)
    (ASSETS_DIR / "anchor_ratio_matrix.json").write_text(json.dumps({
        "ratios": ratio_matrix.tolist(),
        "available": availability.tolist(),
    }))
    print(f"  wrote anchor_ratio_matrix.json shape={ratio_matrix.shape}")

    # Tau (recovery time constants per muscle, in hours) — read from the loaded
    # model so we get exp(log_tau) round-trip rather than raw FIXED_TAU constants.
    tau_from_model = torch.exp(model.r.log_tau).detach().cpu().numpy()
    (ASSETS_DIR / "fixed_tau.json").write_text(json.dumps(tau_from_model.tolist()))
    print(f"  wrote fixed_tau.json  len={len(tau_from_model)}  "
          f"(from exp(log_tau) — checkpoint may have trained values)")

    # Ordered ID lists — Dart code uses these to map name ↔ index.
    (ASSETS_DIR / "exercises.json").write_text(json.dumps(ALL_EXERCISES))
    (ASSETS_DIR / "muscles.json").write_text(json.dumps(list(ALL_MUSCLES)))
    print(f"  wrote exercises.json ({len(ALL_EXERCISES)}) and muscles.json ({len(ALL_MUSCLES)})")

    # Normalization constants.
    (ASSETS_DIR / "scales.json").write_text(json.dumps({
        "WEIGHT_SCALE": float(WEIGHT_SCALE),
        "REPS_SCALE": float(REPS_SCALE),
        "RIR_SCALE": float(RIR_SCALE),
        "DT_SCALE": float(DT_SCALE),
    }))
    print(f"  wrote scales.json")

    # Default 1RM anchors (when user hasn't provided onboarding values).
    (ASSETS_DIR / "default_anchors_kg.json").write_text(json.dumps({
        "anchor_names": list(ANCHOR_NAMES),
        "values_kg": [DEFAULT_ANCHOR_VALUES_KG[n] for n in ANCHOR_NAMES],
    }))
    print(f"  wrote default_anchors_kg.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "deepgain_model_muscle_ord.pt"),
        help="Path to the .pt file (default: deepgain_model_muscle_ord.pt at repo root)",
    )
    args = parser.parse_args()

    model, _, _ = export_models(args.checkpoint)
    print("\nDumping JSON assets")
    dump_assets(model)
    print(f"\nAll outputs in: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
