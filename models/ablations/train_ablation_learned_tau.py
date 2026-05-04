"""Ablation C: learned tau, clamped to physiological range [8, 72h].

Runs train.py with tau unfrozen. All other settings identical.
Model saved to deepgain_ablation_learned_tau*.pt.
"""

import os
_TRAIN_PY = os.path.join(os.path.dirname(__file__), "..", "train.py")
with open(_TRAIN_PY) as f:
    code = f.read()

# Unfreeze tau + add clamp
code = code.replace(
    "        # Fixed τ from literature — NOT learned\n        init_tau = torch.tensor([math.log(t) for t in self.FIXED_TAU])\n        self.log_tau = nn.Parameter(init_tau, requires_grad=False)",
    "        # Learned τ, initialized from literature, clamped to [8, 72h]\n        init_tau = torch.tensor([math.log(t) for t in self.FIXED_TAU])\n        self.log_tau = nn.Parameter(init_tau, requires_grad=True)",
)
code = code.replace(
    "        tau = torch.exp(self.log_tau[muscle_idx])\n        decay = torch.exp(-dt_hours / tau)",
    "        tau = torch.exp(self.log_tau[muscle_idx]).clamp(8.0, 72.0)\n        decay = torch.exp(-dt_hours / tau)",
)

# Save to separate checkpoint files
code = code.replace(
    '}, "deepgain_model_best.pt")',
    '}, "deepgain_ablation_learned_tau_best.pt")',
)
code = code.replace(
    '}, "deepgain_model_muscle_ord.pt")',
    '}, "deepgain_ablation_learned_tau.pt")',
)
code = code.replace(
    'print(f"\\nModel saved to deepgain_model_muscle_ord.pt")',
    'print(f"\\nModel saved to deepgain_ablation_learned_tau.pt")',
)
code = code.replace(
    'print(f"Best val model saved to deepgain_model_best.pt',
    'print(f"Best val model saved to deepgain_ablation_learned_tau_best.pt',
)

# Suffix chart folder name
code = code.replace(
    'datetime.now().strftime("%Y%m%d_%H%M")',
    'datetime.now().strftime("%Y%m%d_%H%M") + "_learned_tau"',
)

# Label charts README
code = code.replace(
    '_f.write(f"| Penalties |',
    '_f.write("| Ablation | learned_tau (clamped 8-72h) |\\n")\n    _f.write(f"| Penalties |',
)

exec(compile(code, "train.py", "exec"))
