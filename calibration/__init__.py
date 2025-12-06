
"""
DBSI Calibration Module - Hyperparameter Optimization

Monte Carlo-based optimization to find optimal hyperparameters:
- n_iso: Number of isotropic basis functions
- reg_lambda: L2 regularization strength

The calibration targets accuracy in restricted fraction estimation,
which is the primary inflammation biomarker.
"""

from .optimizer import optimize_hyperparameters, generate_synthetic_signal

__all__ = [
    "optimize_hyperparameters",
    "generate_synthetic_signal",
]