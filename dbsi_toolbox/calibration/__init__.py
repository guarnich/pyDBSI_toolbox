"""
DBSI Calibration Module — regularisation parameter selection
=============================================================

- `calibration.data_driven` (the only selection path): GCV + discrepancy
  ceiling for lambda_iso, ridge-dof GCV for lambda_aniso, on a sample of the
  dataset's own voxels with the noise map's sigma. See its module docstring.
- `calibration.protocol`: aggregation of per-acquisition calibration records
  into one set of hyperparameters per protocol (cohort studies).
- `calibration.mc_sure` (optional, off by default): Monte Carlo SURE
  cross-check of lambda_iso / n_iso against the risk of the actual NNLS
  estimator; reports agreement, never a new value.
- `calibration.adaptive_n_iso` (optional): per-dataset n_iso selection
  ('bootstrap', 'svd_floor'). The default is n_iso = 6 by grid convergence
  ('quadrature'): n_iso is a discretisation parameter, not a model order.

Removed in 1.8.1 because nothing called them: `calibration.optimizer`
(synthetic tissue-scenario signals, the last remnant of the Monte Carlo
calibration removed on 2026-09-21), `select_n_iso_with_gcv_crosscheck` and
`select_n_iso_data_driven_sweep`.

Usage outside DBSI_Adaptive.fit:

    from dbsi_toolbox.calibration import select_lambdas_data_driven, sample_calibration_voxels
    y_voxels, sigma = sample_calibration_voxels(data_corr, mask, bvals, n_voxels=1000,
                                                seed=0, noise_map=noise_map)
    lambda_aniso, lambda_iso, diag = select_lambdas_data_driven(
        bvals, bvecs, fiber_dirs, diff_pairs, iso_grid, y_voxels, sigma)
"""

from .data_driven import (
    select_lambda_iso_gcv,
    select_lambda_aniso_discrepancy,
    select_lambdas_data_driven,
    sample_calibration_voxels,
)
from .adaptive_n_iso import (
    select_n_iso_svd,
    select_n_iso_bootstrap,
)
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
    "select_n_iso_bootstrap",
    # Monte Carlo SURE (cross-check)
    "crosscheck_lambda_iso_sure",
    "crosscheck_n_iso_sure",
]
