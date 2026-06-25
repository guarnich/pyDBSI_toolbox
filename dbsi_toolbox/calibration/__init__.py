"""
DBSI Calibration Module - Hyperparameter Optimization (v3, Hybrid Two-Stage)

Monte Carlo-based optimization to find optimal decoupled hyperparameters:
- lambda_aniso: Stage A regularization strength for the exhaustive
  detection dictionary's anisotropic (direction x AD/RD-pair) block
- lambda_iso: Stage A regularization strength for the isotropic spectrum
  block

The calibration loss is evaluated end-to-end through Stage A (direction
detection) AND Stage B (closed-form AD/RD estimation conditioned on the
detected direction), so the chosen hyperparameters reflect their actual
downstream effect on diffusivity recovery, not just fraction recovery.
"""

from .optimizer import optimize_hyperparameters, generate_synthetic_signal

__all__ = [
    "optimize_hyperparameters",
    "generate_synthetic_signal",
]
