"""Torch-only diag: compare DeepGainModel.predict_drop_norm vs FatigueModule
wrapper. Isolates wrapper bugs from TFLite drift."""

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from inference import (  # noqa: E402
    EXERCISE_TO_IDX, NUM_MUSCLES, REPS_SCALE, RIR_SCALE, WEIGHT_SCALE, load_model,
)

# Import FatigueModule definition only (avoid the litert_torch import in export.py)
import importlib.util
spec = importlib.util.spec_from_file_location("export_mod", Path(__file__).parent / "export.py")


def load_fatigue_module():
    # Patch to skip litert_torch import
    src = (Path(__file__).parent / "export.py").read_text()
    src = src.replace("import litert_torch", "litert_torch = None")
    g = {"__file__": str(Path(__file__).parent / "export.py")}
    exec(compile(src, "export.py", "exec"), g)
    return g["FatigueModule"]


FatigueModule = load_fatigue_module()


def main():
    model = load_model("deepgain_model_muscle_ord.pt", device="cpu")
    fatigue = FatigueModule(model).eval()

    ex_idx = EXERCISE_TO_IDX["bench_press"]
    weight = 100.0 / WEIGHT_SCALE
    reps = 5.0 / REPS_SCALE
    rir = 2.0 / RIR_SCALE
    mpc = torch.full((NUM_MUSCLES,), 0.95, dtype=torch.float32)
    anchors = torch.tensor([100.0 / WEIGHT_SCALE, 140.0 / WEIGHT_SCALE, 180.0 / WEIGHT_SCALE], dtype=torch.float32)

    M = NUM_MUSCLES
    all_m_idx = torch.arange(M)
    all_m_embed = model.muscle_embed(all_m_idx)
    ei = torch.tensor([ex_idx], dtype=torch.long)
    w_exp = torch.full((M,), weight, dtype=torch.float32)
    r_exp = torch.full((M,), reps, dtype=torch.float32)
    rir_exp = torch.full((M,), rir, dtype=torch.float32)
    anchors_orig = anchors.unsqueeze(0)
    with torch.no_grad():
        orig_drop = model.predict_drop_norm(
            ei.expand(M), w_exp, r_exp, rir_exp, mpc, all_m_embed, anchors_orig,
        )
        wrapper_drop = fatigue(
            torch.tensor(ex_idx, dtype=torch.int64),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(reps, dtype=torch.float32),
            torch.tensor(rir, dtype=torch.float32),
            mpc,
            anchors,
        )

    print(f"orig_drop:    {orig_drop.numpy()}")
    print(f"wrapper_drop: {wrapper_drop.numpy()}")
    print(f"max abs diff: {(orig_drop - wrapper_drop).abs().max().item():.2e}")

    # Also dump the strength_feat seen by both paths.
    sf_orig = model.compute_strength_features(ei.expand(M), w_exp, anchors_orig)
    print(f"\norig strength_feat[0]: {sf_orig[0].numpy()}")
    sf_wrap_inputs = anchors  # what the wrapper sees
    print(f"wrap anchors input:    {sf_wrap_inputs.numpy()}")


if __name__ == "__main__":
    main()
