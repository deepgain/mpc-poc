"""Ablation B: smaller model — EMBED_DIM=32, HIDDEN_DIM=256 (M7 architecture).

Runs train.py with reduced capacity. All other settings identical (penalties, dataset, epochs).
Model saved to deepgain_ablation_small*.pt.
"""

with open("train.py") as f:
    code = f.read()

# Smaller architecture
code = code.replace("EMBED_DIM = 64", "EMBED_DIM = 32")
code = code.replace("HIDDEN_DIM = 512", "HIDDEN_DIM = 256")

# Save to separate checkpoint files
code = code.replace(
    '}, "deepgain_model_best.pt")',
    '}, "deepgain_ablation_small_best.pt")',
)
code = code.replace(
    '}, "deepgain_model_muscle_ord.pt")',
    '}, "deepgain_ablation_small.pt")',
)
code = code.replace(
    'print(f"\\nModel saved to deepgain_model_muscle_ord.pt")',
    'print(f"\\nModel saved to deepgain_ablation_small.pt")',
)
code = code.replace(
    'print(f"Best val model saved to deepgain_model_best.pt',
    'print(f"Best val model saved to deepgain_ablation_small_best.pt',
)

# Suffix chart folder name
code = code.replace(
    'datetime.now().strftime("%Y%m%d_%H%M")',
    'datetime.now().strftime("%Y%m%d_%H%M") + "_small"',
)

# Label charts README
code = code.replace(
    '_f.write(f"| Penalties |',
    '_f.write("| Ablation | small (EMBED=32, HIDDEN=256) |\\n")\n    _f.write(f"| Penalties |',
)

exec(compile(code, "train.py", "exec"))
