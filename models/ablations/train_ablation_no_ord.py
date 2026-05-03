"""Ablation A: no fatigue_ordering_penalty.

Runs train.py with ordering penalty disabled. All other settings identical.
Same charts, same metrics output. Model saved to deepgain_ablation_no_ord*.pt.
"""

with open("train.py") as f:
    code = f.read()

# Disable ordering penalty
code = code.replace(
    "loss = loss + 0.05 * model.fatigue_ordering_penalty()",
    "pass  # ABLATION no_ord: ordering penalty disabled",
)

# Save to separate checkpoint files (avoid overwriting main model)
code = code.replace(
    '}, "deepgain_model_best.pt")',
    '}, "deepgain_ablation_no_ord_best.pt")',
)
code = code.replace(
    '}, "deepgain_model_muscle_ord.pt")',
    '}, "deepgain_ablation_no_ord.pt")',
)
code = code.replace(
    'print(f"\\nModel saved to deepgain_model_muscle_ord.pt")',
    'print(f"\\nModel saved to deepgain_ablation_no_ord.pt")',
)
code = code.replace(
    'print(f"Best val model saved to deepgain_model_best.pt',
    'print(f"Best val model saved to deepgain_ablation_no_ord_best.pt',
)

# Suffix chart folder name with ablation tag
code = code.replace(
    'datetime.now().strftime("%Y%m%d_%H%M")',
    'datetime.now().strftime("%Y%m%d_%H%M") + "_no_ord"',
)

# Label charts README as ablation
code = code.replace(
    '_f.write(f"| Penalties |',
    '_f.write("| Ablation | no_ord_penalty |\\n")\n    _f.write(f"| Penalties |',
)

exec(compile(code, "train.py", "exec"))
