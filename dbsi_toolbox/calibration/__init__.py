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

2. MONTE CARLO (cross-check) — `calibration.optimizer`
   14 physiologically grounded tissue scenarios (Wang et al. 2011, Ye et
   al. 2020, Vavasour et al. 2022), used to verify that a candidate
   (lambda_aniso, lambda_iso) pair — typically the data-driven path's
   output — behaves sensibly across known tissue regimes (e.g.
   adequately suppresses spurious fiber fraction in grey matter,
   preserves restricted-fraction sensitivity in lesions). Use
   `evaluate_lambda_pair` for this (cheaper, single-pair check); the
   full `optimize_hyperparameters` grid search remains available as a
   fallback if the data-driven method's assumptions are suspected to
   fail for a given protocol.

Recommended usage pattern
-----------------------------
    from dbsi_toolbox.calibration import (
        select_lambdas_data_driven, sample_calibration_voxels,
        evaluate_lambda_pair,
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

    # Cross-check against known tissue scenarios before trusting it:
    report = evaluate_lambda_pair(bvals, bvecs, snr, lambda_aniso, lambda_iso)
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
)
from .optimizer import (
    optimize_hyperparameters,
    evaluate_lambda_pair,
    generate_synthetic_signal,
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
    # Monte Carlo (cross-check)
    "optimize_hyperparameters",
    "evaluate_lambda_pair",
    "generate_synthetic_signal",
]
