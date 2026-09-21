"""
DBSI Calibration Module - Regularization Parameter Selection (v3, Hybrid Two-Stage)
======================================================================================

TWO COMPLEMENTARY PATHS
----------------------------
1. DATA-DRIVEN (primary) — `calibration.data_driven`
   GCV (lambda_iso) + discrepancy principle (lambda_aniso), derived
   purely from the acquisition protocol and a sample of the dataset's
   own voxels. No tissue-fraction priors. Fast (a few dozen NNLS solves
   total). Recommended as the default calibration path — see
   `data_driven.py` module docstring for the full rationale and the
   empirical comparison against the Monte Carlo path.

2. MONTE CARLO — REMOVED 2026-09-21
   Both Monte Carlo entry points are gone: the grid search
   (`optimize_hyperparameters`, `calibration_method='monte_carlo'`) and
   the single-pair cross-check (`evaluate_lambda_pair`,
   `--mc-crosscheck`). Neither ever worked here — their shared scenario
   evaluator called `select_dominant_directions` with a pre-`fiber_dirs`
   argument list and the numba kernel never compiled. `calibration.optimizer`
   now only holds synthetic-signal generation.

3. MONTE CARLO SURE (cross-check) — `calibration.mc_sure`
   Stein's Unbiased Risk Estimate, evaluated via randomized probes
   (Ramani, Blu & Unser 2008), gives a risk criterion that is formally
   exact for the actual constrained NNLS estimator used in this
   toolbox — unlike GCV, which is derived for an unconstrained linear
   estimator and is therefore an approximation once the non-negativity
   constraint binds. Use `crosscheck_lambda_iso_sure` /
   `crosscheck_n_iso_sure` to verify that a candidate n_iso/lambda_iso
   (from the data-driven path) falls within a low-risk neighbourhood
   under this independent criterion. See `mc_sure.py` module docstring
   for the empirical finding that shaped this module's design: the risk
   landscape here is typically a flat valley, so these functions report
   AGREEMENT/DISAGREEMENT with a candidate, not a new point estimate.

Recommended usage pattern
-----------------------------
    from dbsi_toolbox.calibration import (
        select_lambdas_data_driven, sample_calibration_voxels,
    )

    y_voxels, sigma = sample_calibration_voxels(data, mask, bvals)
    lambda_aniso, lambda_iso, diag = select_lambdas_data_driven(
        bvals, bvecs, fiber_dirs, diff_pairs, iso_grid, y_voxels, sigma
    )
    # Check diag['discrepancy'].get('floor_applied'): True means the
    # discrepancy principle's raw answer was below the safety floor
    # (lambda_aniso < 10% of lambda_iso) and was clamped — usually a
    # sign the calibration voxel sample was too small/homogeneous (see
    # `select_lambda_aniso_discrepancy` docstring). Consider increasing
    # n_calibration_voxels if this triggers often.
"""

from .data_driven import (
    select_lambda_iso_gcv,
    select_lambda_aniso_discrepancy,
    select_lambdas_data_driven,
    sample_calibration_voxels,
)
from .adaptive_n_iso import (
    select_n_iso_svd,
    select_n_iso_with_gcv_crosscheck,
    select_n_iso_data_driven_sweep,
    select_n_iso_bootstrap,
)
from .optimizer import generate_synthetic_signal
from .mc_sure import (
    crosscheck_lambda_iso_sure,
    crosscheck_n_iso_sure,
)

__all__ = [
    # Data-driven (primary)
    "select_lambda_iso_gcv",
    "select_lambda_aniso_discrepancy",
    "select_lambdas_data_driven",
    "sample_calibration_voxels",
    # Adaptive n_iso
    "select_n_iso_svd",
    "select_n_iso_with_gcv_crosscheck",
    "select_n_iso_data_driven_sweep",
    "select_n_iso_bootstrap",
    # Synthetic signal generation (no longer used by the package itself)
    "generate_synthetic_signal",
    # Monte Carlo SURE (cross-check)
    "crosscheck_lambda_iso_sure",
    "crosscheck_n_iso_sure",
]
