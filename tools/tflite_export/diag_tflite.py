"""TFLite-only diag: feed the same inputs the wrapper got, compare to a
saved PyTorch baseline."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ASSETS = Path(__file__).resolve().parent / "assets"

# Baseline from diag_torch.py (bench_press, 100kg×5@RIR=2, mpc=0.95, default anchors)
BASELINE = np.array([
    0.30886552, 0.02866242, 0.01306433, 0.00473181, 0.01001398,
    0.1203272, 0.00792676, 0.00625355, 0.01241068, 0.00401812,
    0.01015035, 0.00412566, 0.0046382, 0.01598512, 0.00540754,
])

W_SCALE = 200.0
R_SCALE = 30.0
RIR_SCALE = 5.0


def main():
    exercises = json.loads((ASSETS / "exercises.json").read_text())
    ex_idx = exercises.index("bench_press")

    interp = tf.lite.Interpreter(model_path=str(ASSETS / "f_net.tflite"))
    interp.allocate_tensors()

    print("Input details (in order):")
    for d in interp.get_input_details():
        print(f"  name={d['name']:60s} idx={d['index']} shape={d['shape']} dtype={d['dtype'].__name__}")

    print("\nOutput details:")
    for d in interp.get_output_details():
        print(f"  name={d['name']:60s} idx={d['index']} shape={d['shape']} dtype={d['dtype'].__name__}")

    # Build inputs by EXPECTED order (the order our wrapper.forward defined them)
    weight_n = 100.0 / W_SCALE
    reps_n = 5.0 / R_SCALE
    rir_n = 2.0 / RIR_SCALE
    mpc = np.full((15,), 0.95, dtype=np.float32)
    anchors = np.array([100.0 / W_SCALE, 140.0 / W_SCALE, 180.0 / W_SCALE], dtype=np.float32)

    expected = {
        "exercise_idx": np.int64(ex_idx),
        "weight": np.float32(weight_n),
        "reps": np.float32(reps_n),
        "rir": np.float32(rir_n),
        "mpc": mpc,
        "anchors": anchors,
    }

    # Inputs are in positional (args_0..args_5) order matching forward() args.
    ordered = list(expected.values())
    for det, val in zip(interp.get_input_details(), ordered):
        arr = np.asarray(val, dtype=det["dtype"]).reshape(det["shape"])
        interp.set_tensor(det["index"], arr)

    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    print(f"\ntflite_drop: {out}")
    print(f"baseline:    {BASELINE}")
    print(f"max abs diff: {np.abs(out - BASELINE).max():.2e}")


if __name__ == "__main__":
    main()
