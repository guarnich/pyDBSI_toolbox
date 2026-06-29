"""
DBSI Data-Driven Regularization — GCV (lambda_iso) + Discrepancy Principle (lambda_aniso)
=============================================================================================

WHY THIS MODULE EXISTS
--------------------------
The Monte Carlo calibration in `calibration/optimizer.py` selects
(lambda_aniso, lambda_iso) by simulating signals from 14 physiologically
grounded tissue scenarios (Wang et al. 2011, Ye et al. 2020, Vavasour et
al. 2022) with hardcoded fraction/diffusivity priors. This is methodologically
sound as a CROSS-VALIDATION step (does a candidate lambda pair produce
reasonable fraction/diffusivity recovery across known tissue regimes?),
but using it as the SOURCE of lambda has a specific weakness: the chosen
lambda optimizes performance on literature-derived scenarios, which may
not match the actual fraction distribution of the protocol/population
being processed. Since lambda's optimum has already been shown (see
project validation records for v2/v3) to depend on SNR and the
acquisition's b-value scheme, calibrating against external literature
priors risks a lambda that is optimal for a hypothetical scenario set
rather than for the data actually being fit.

This module computes (lambda_aniso, lambda_iso) using ONLY quantities
already derived from the actual data being processed: the design matrix
A (depends only on bvals/bvecs, already known), the observed signal y
(via `sample_calibration_voxels`, drawn directly from the dataset's own
brain-mask voxels — no tissue model assumed), and the noise sigma
(already estimated, or computed directly from sampled b0 volumes). No
tissue-fraction priors are used. `calibration/optimizer.py`'s Monte
Carlo scenarios remain available as an INDEPENDENT cross-validation
check (see `calibration/optimizer.py` module docstring after this
change) rather than as the source of lambda.

TWO DIFFERENT METHODS FOR TWO DIFFERENT BLOCKS
----------------------------------------------------
lambda_iso (isotropic spectrum block) — Generalized Cross-Validation (GCV)
    The isotropic block is a classic Tikhonov-regularized linear inverse
    problem (no non-negativity constraint enforced at this stage of the
    estimate; the constraint is applied later by the actual NNLS solve,
    but GCV's ridge-regression selection of lambda is a standard,
    well-validated proxy for the constrained problem's regularization
    strength — see module docstring discussion of this approximation's
    scope). GCV chooses lambda to minimize a leave-one-out cross-
    validation estimate of out-of-sample prediction error, computed in
    closed form via the SVD of the design matrix — no need to refit N
    times.

        GCV(lambda) = (||y - A w_lambda||^2 / N) / (1 - tr(H_lambda)/N)^2

    where H_lambda = A (A^T A + lambda I)^-1 A^T is the "hat matrix".

lambda_aniso (Stage A anisotropic detection block) — Discrepancy Principle
    Stage A's job is sparse direction DETECTION, not spectral recovery
    (see model_Niso_adaptive_ff_thr.py module docstring) — a simpler,
    more binary objective (fiber present in this direction: yes/no) than
    the isotropic block's continuous spectrum recovery. For this simpler
    objective, the discrepancy principle (Morozov 1966) is an adequate
    and very cheap choice: select lambda such that the residual norm of
    the regularized fit matches the expected noise level,

        ||A w_lambda - y||^2 ~= N * sigma^2

    found via 1D bisection on lambda (a handful of NNLS solves, not
    thousands of Monte Carlo samples).

CAVEATS (stated explicitly per project methodology: no number is used
without a documented source, and limitations are not silently elided)
-----------------------------------------------------------------------------
- GCV and the discrepancy principle were both originally derived for
  UNCONSTRAINED linear least squares. Applying them to NNLS (non-
  negativity constrained) is common and empirically reasonable in the
  inverse-problems literature, but the asymptotic optimality guarantees
  of classical GCV theory do not strictly transfer to the constrained
  case. This should be validated empirically against synthetic recovery
  and against the Monte Carlo cross-validation scenarios before being
  trusted in place of the (slower but constraint-aware) Monte Carlo
  selection for a given protocol.
- GCV is known to occasionally under-regularize when the noise is not
  well-approximated as i.i.d. Gaussian (the Rician-corrected residual
  noise floor is not exactly Gaussian even after correction). The
  discrepancy principle requires an accurate sigma estimate — error in
  `estimate_snr_robust`'s sigma propagates directly into lambda_aniso.
- Both methods, as implemented here, choose a single GLOBAL
  lambda_aniso/lambda_iso per protocol (matching the existing
  calibration's scope), not a per-voxel value. Per-voxel adaptive
  regularization (e.g. SNR-weighted, voxel-local discrepancy) is a
  documented possible extension, not implemented here.
- EMPIRICAL VALIDATION SUMMARY (project records, synthetic testing):
  when both methods are calibrated FRESH on the exact same protocol and
  Stage A dictionary, lambda_aniso selected by the discrepancy principle
  (~2-3.6 in one tested configuration) landed close to the value
  independently found by a fresh Monte Carlo calibration (~3.6 for the
  same configuration) — i.e. the two methods are not in fundamental
  disagreement when compared fairly. Synthetic AD/RD recovery accuracy
  was comparable between the two (neither uniformly dominated the
  other across 5 independent test seeds), with the discrepancy-principle
  selection showing lower median RD error but higher median AD error in
  that specific comparison. The discrepancy-principle estimate's main
  weakness identified was VARIANCE, not bias: coefficient of variation
  of the selected lambda_aniso across repeated calibration samples was
  ~51% at 15 calibration voxels, dropping to ~14% at 150 voxels. With
  real datasets (hundreds of thousands of brain-mask voxels available
  for sampling via `sample_calibration_voxels`), this variance is
  expected to shrink further; this has not yet been confirmed on real
  (non-synthetic) data and should be checked before relying on the
  data-driven selection as the sole source of lambda for a given
  dataset — hence the recommendation to run the Monte Carlo
  cross-check (`calibration/optimizer.py`) alongside it, at least until
  this is validated on the project's actual acquisitions.

References
----------
Golub GH, Heath M, Wahba G (1979). Generalized cross-validation as a
    method for choosing a good ridge parameter. Technometrics, 21(2),
    215-223.
Hansen PC (1992). Analysis of discrete ill-posed problems by means of
    the L-curve. SIAM Review, 34(4), 561-580.
Morozov VA (1966). On the solution of functional equations by the
    method of regularization. Soviet Math. Dokl., 7, 414-417.
"""

import numpy as np
from ..core.basis import build_isotropic_dictionary
from ..core.solvers import nnls_coordinate_descent


# ─────────────────────────────────────────────────────────────────────────────
# GCV — lambda_iso
# ─────────────────────────────────────────────────────────────────────────────

def select_lambda_iso_gcv(bvals, iso_grid, y_voxels, lambda_grid=None):
    """
    Select lambda_iso via Generalized Cross-Validation on the isotropic
    spectrum design matrix.

    Uses the SVD of the isotropic design matrix A_iso to evaluate GCV(lambda)
    in closed form for every candidate lambda without refitting — standard
    technique for ridge-type regularization (Golub, Heath & Wahba 1979).

    Parameters
    ----------
    bvals : array (N,)
        B-values, s/mm^2 (same protocol used for fitting).
    iso_grid : array (K,)
        Isotropic ADC grid (same one that will be used in the actual
        fit's design matrix).
    y_voxels : array (N,) or (n_voxels, N)
        One or more normalised signals (S/S0) to average the GCV
        criterion over. Passing multiple representative voxels
        (e.g. a random sample of brain-mask voxels) gives a more stable
        protocol-level lambda_iso than a single voxel; passing one voxel
        is also valid (degenerates to per-voxel selection).
    lambda_grid : array or None
        Candidate lambda values to evaluate. Defaults to a log-spaced
        grid from 1e-5 to 10 (40 points), which comfortably spans the
        range used elsewhere in this toolbox's manual/MC-calibrated
        lambda_iso values (typically 1e-3 to 1).

    Returns
    -------
    best_lambda : float
        The lambda minimising the (voxel-averaged) GCV criterion.
    gcv_curve : dict
        {'lambda': array, 'gcv': array} — the full evaluated curve, for
        diagnostic plotting / sanity-checking the selection (e.g.
        confirming a clear minimum rather than a flat or monotonic
        curve, which would indicate GCV is not well-determined for this
        protocol).
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    iso_grid = np.asarray(iso_grid, dtype=np.float64)
    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))

    A_iso = build_isotropic_dictionary(bvals, iso_grid)
    N, K = A_iso.shape

    # SVD once; reused for every candidate lambda and every voxel.
    U, s, Vt = np.linalg.svd(A_iso, full_matrices=False)
    s2 = s ** 2

    if lambda_grid is None:
        lambda_grid = np.logspace(-5, 1, 40)
    lambda_grid = np.asarray(lambda_grid, dtype=np.float64)

    # Project every voxel's signal onto the left singular vectors once.
    # Uty[:, v] = U^T y_v  for voxel v.
    Uty = U.T @ y_voxels.T  # shape (K, n_voxels)

    gcv_vals = np.zeros(len(lambda_grid))

    for li, lam in enumerate(lambda_grid):
        # Filter factors for ridge regression in the SVD basis.
        filt = s2 / (s2 + lam)  # shape (K,)

        # Residual norm^2 per voxel, summed via Parseval (orthonormal U):
        # ||y - A w_lambda||^2 = ||y||^2 - sum_k filt_k * (1+filt_k... )
        # Simpler: compute residual coefficients directly.
        # Fitted part in U-basis: U @ (filt * Uty); residual = y - U@(filt*Uty)
        # ||residual||^2 = ||y||^2 - 2*sum(filt*Uty*Uty) + sum(filt^2 * Uty^2)
        # (since U columns orthonormal, cross term simplifies)
        y_norm2 = np.sum(y_voxels ** 2, axis=1)  # (n_voxels,)
        fitted_coef = filt[:, None] * Uty  # (K, n_voxels)
        # ||U @ fitted_coef||^2 = ||fitted_coef||^2 (U orthonormal columns)
        fitted_norm2 = np.sum(fitted_coef ** 2, axis=0)  # (n_voxels,)
        cross_term = np.sum(fitted_coef * Uty, axis=0)  # (n_voxels,)
        resid_norm2 = y_norm2 - 2 * cross_term + fitted_norm2
        resid_norm2 = np.maximum(resid_norm2, 0.0)  # guard fp negatives

        trace_H = np.sum(filt)  # tr(A(A^TA+lam I)^-1 A^T) = sum of filter factors
        denom = (1.0 - trace_H / N) ** 2
        denom = max(denom, 1e-12)

        gcv_per_voxel = (resid_norm2 / N) / denom
        gcv_vals[li] = np.mean(gcv_per_voxel)

    best_idx = int(np.argmin(gcv_vals))
    best_lambda = float(lambda_grid[best_idx])

    return best_lambda, dict(lambda_=lambda_grid, gcv=gcv_vals)


# ─────────────────────────────────────────────────────────────────────────────
# DISCREPANCY PRINCIPLE — lambda_aniso
# ─────────────────────────────────────────────────────────────────────────────

def select_lambda_aniso_discrepancy(AtA, At, y_voxels, n_aniso_cols,
                                     sigma, lambda_iso_fixed,
                                     lambda_lo=1e-6, lambda_hi=1e4,
                                     n_dirs=None, max_bisect_iter=40,
                                     tol=1e-3, min_floor_factor=0.1):
    """
    Select lambda_aniso via the discrepancy principle (Morozov 1966):
    find lambda such that the NNLS residual matches the expected noise
    level, ||A w_lambda - y||^2 ~= N * sigma^2.

    Operates on the FULL design matrix (anisotropic + isotropic blocks)
    since the actual fit always solves both jointly, but only
    lambda_aniso is searched here — lambda_iso is held fixed at the
    value already selected by `select_lambda_iso_gcv` (the discrepancy
    search asks "how much anisotropic regularization, given the
    isotropic block's regularization is already fixed", which matches
    how the two blocks are used together in `core.solvers.
    compute_regularization_matrix`).

    Uses 1D bisection on log(lambda_aniso): cheap (a few dozen NNLS
    solves total, vs. the thousands of Monte Carlo samples in
    `calibration/optimizer.py`), because Stage A's detection objective
    (sparse fiber/no-fiber per direction) is simpler than full spectral
    recovery and does not require evaluating multiple tissue scenarios.

    SAFETY FLOOR (min_floor_factor)
    ------------------------------------
    If the calibration voxel sample is small, near-noise-free, or
    unusually homogeneous (e.g. synthetic test signals with little
    inter-voxel variability), the residual at lambda_aniso -> 0 can
    already meet or undershoot the noise-matching target, causing the
    bisection to collapse to (near) zero — a pathological result: an
    unregularized anisotropic block with thousands of collinear columns
    has an enormous condition number (observed >1e9 in such a case
    during testing) and produces an ill-conditioned, numerically
    unstable fit even though it nominally "matches" the discrepancy
    target. This is NOT a useful solution: matching the residual to the
    noise level is necessary but not sufficient for a well-posed
    inverse problem when the unregularized system is singular or
    near-singular.

    `min_floor_factor` enforces lambda_aniso >= min_floor_factor *
    lambda_iso_fixed * n_aniso_cols_per_iso_col_ratio is NOT used here in
    favour of a simpler, more conservative floor:
    lambda_aniso >= min_floor_factor * lambda_iso_fixed (i.e. the
    anisotropic block's regularization is never allowed to drop below a
    fraction of the isotropic block's, by default 10%). This is a
    pragmatic safety net, not a principled derivation; it should be
    revisited if it is found to bind frequently on real datasets (which
    would suggest the calibration voxel sample needs to be larger or
    more diverse — see `sample_calibration_voxels` docstring).

    Parameters
    ----------
    AtA : ndarray (n_total_cols, n_total_cols)
        Gram matrix of the FULL design matrix (Stage A anisotropic
        block + isotropic block), unregularized.
    At : ndarray (n_total_cols, N)
        Transpose of the full design matrix.
    y_voxels : array (N,) or (n_voxels, N)
        One or more normalised signals (S/S0) to average the residual
        criterion over (same rationale as `select_lambda_iso_gcv`).
    n_aniso_cols : int
        Number of anisotropic (Stage A) columns; columns
        [0:n_aniso_cols) get lambda_aniso, columns [n_aniso_cols:) get
        lambda_iso_fixed.
    sigma : float
        Noise standard deviation (already estimated via
        `estimate_snr_robust`), in the same normalised (S/S0) units as
        y_voxels.
    lambda_iso_fixed : float
        The lambda_iso value to hold fixed during the lambda_aniso
        search (typically the output of `select_lambda_iso_gcv`).
    lambda_lo, lambda_hi : float
        Search bracket for lambda_aniso (log-scale bisection).
    n_dirs : int or None
        Unused directly here (kept for API symmetry / future per-
        direction discrepancy variants); the discrepancy principle as
        implemented operates on the aggregate anisotropic+isotropic
        residual, not per-direction.
    max_bisect_iter : int
        Maximum bisection iterations.
    tol : float
        Relative tolerance on the discrepancy match
        (|residual^2 - N*sigma^2| / (N*sigma^2) < tol stops early).
    min_floor_factor : float
        Minimum lambda_aniso, expressed as a fraction of
        lambda_iso_fixed (default 0.1, i.e. lambda_aniso is never
        allowed below 10% of lambda_iso_fixed). Set to 0.0 to disable
        the floor entirely (not recommended — see rationale above).

    Returns
    -------
    best_lambda_aniso : float
    diagnostics : dict
        {'target_residual2': float, 'achieved_residual2': float,
         'n_iter': int, 'floor_applied': bool} for sanity-checking the
         bisection converged to a residual close to the target rather
         than hitting a bracket edge, and whether the safety floor had
         to intervene.
    """
    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))
    n_voxels, N = y_voxels.shape

    target_residual2 = N * sigma ** 2
    lambda_aniso_floor = min_floor_factor * lambda_iso_fixed

    n_total_cols = AtA.shape[0]

    def _residual2_for_lambda(lambda_aniso):
        reg_vec = np.zeros(n_total_cols, dtype=np.float64)
        reg_vec[:n_aniso_cols] = lambda_aniso
        reg_vec[n_aniso_cols:] = lambda_iso_fixed
        AtA_reg = AtA + np.diag(reg_vec)

        total_resid2 = 0.0
        for v in range(n_voxels):
            y = y_voxels[v]
            Aty = At @ y
            w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
            # Residual computed via At/AtA identity to avoid needing A
            # explicitly: ||Aw - y||^2 = w^T AtA w - 2 w^T Aty + y^T y
            resid2 = float(w @ (AtA @ w) - 2.0 * w @ Aty + y @ y)
            total_resid2 += max(resid2, 0.0)

        return total_resid2 / n_voxels

    lo, hi = np.log10(lambda_lo), np.log10(lambda_hi)
    f_lo = _residual2_for_lambda(10 ** lo) - target_residual2
    f_hi = _residual2_for_lambda(10 ** hi) - target_residual2

    # Residual2 is monotonically increasing in lambda_aniso (more
    # regularization -> worse fit -> larger residual). If the bracket
    # does not contain a sign change, return the nearer edge rather than
    # bisecting blindly into a meaningless region.
    if f_lo > 0:
        # Even minimal regularization over-regularizes (residual already
        # above target) -- the protocol/sigma estimate may be
        # inconsistent with this dictionary; return the lower edge
        # (still subject to the safety floor below).
        result = float(10 ** lo)
        floor_applied = result < lambda_aniso_floor
        result = max(result, lambda_aniso_floor)
        return result, dict(
            target_residual2=target_residual2,
            achieved_residual2=f_lo + target_residual2,
            n_iter=0,
            floor_applied=floor_applied,
            warning="lambda_lo already exceeds target residual; "
                    "returning lambda_lo (subject to safety floor). "
                    "Check sigma estimate / dictionary size.",
        )
    if f_hi < 0:
        return float(10 ** hi), dict(
            target_residual2=target_residual2,
            achieved_residual2=f_hi + target_residual2,
            n_iter=0,
            floor_applied=False,
            warning="lambda_hi still under-regularizes relative to target "
                    "residual; returning lambda_hi. Consider widening the bracket.",
        )

    n_iter = 0
    mid_residual2 = None
    for n_iter in range(1, max_bisect_iter + 1):
        mid = 0.5 * (lo + hi)
        mid_residual2 = _residual2_for_lambda(10 ** mid)
        f_mid = mid_residual2 - target_residual2

        if abs(f_mid) / max(target_residual2, 1e-12) < tol:
            break

        if f_mid > 0:
            hi = mid
        else:
            lo = mid

    best_lambda_aniso = float(10 ** mid)
    floor_applied = best_lambda_aniso < lambda_aniso_floor
    best_lambda_aniso = max(best_lambda_aniso, lambda_aniso_floor)

    return best_lambda_aniso, dict(
        target_residual2=target_residual2,
        achieved_residual2=mid_residual2,
        n_iter=n_iter,
        floor_applied=floor_applied,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REAL-DATA VOXEL SAMPLING (for use with select_lambdas_data_driven)
# ─────────────────────────────────────────────────────────────────────────────

def sample_calibration_voxels(data, mask, bvals, b0_thr=100.0,
                              n_voxels=200, min_signal_fraction=0.05,
                              seed=None):
    """
    Sample a representative set of S0-normalised voxel signals from the
    actual brain volume being processed, for use as `y_voxels` in
    `select_lambda_iso_gcv` / `select_lambda_aniso_discrepancy` /
    `select_lambdas_data_driven`.

    Unlike the Monte Carlo cross-check in `calibration/optimizer.py`
    (which simulates signals from literature-derived tissue-fraction
    priors), this draws signals DIRECTLY from the dataset's own brain-
    mask voxels — no tissue model is assumed. With real datasets
    containing hundreds of thousands of brain voxels, a few hundred
    sampled voxels give a far more representative and far larger
    effective calibration sample than the per-protocol Monte Carlo
    scenario count, while remaining cheap to evaluate.

    Parameters
    ----------
    data : ndarray (X, Y, Z, N)
        Raw (or Rician-corrected) DWI signal.
    mask : ndarray (X, Y, Z), bool
        Brain mask.
    bvals : array (N,)
        B-values, s/mm^2 (used to identify b=0 volumes for S0
        normalisation).
    b0_thr : float
        B-value threshold below which a volume is treated as b=0.
    n_voxels : int
        Number of voxels to sample. 150-300 is a reasonable range based
        on the stability analysis in project validation records
        (coefficient of variation in the selected lambda dropped from
        ~51% at 15 voxels to ~14% at 150 voxels in synthetic testing);
        with real datasets' much larger voxel pools, going beyond ~300
        gives diminishing returns for added compute cost.
    min_signal_fraction : float
        Voxels with S0 below this fraction of the mask's median S0 are
        excluded (avoids near-zero-signal edge voxels distorting the
        GCV/discrepancy criteria).
    seed : int or None
        Random seed for reproducible sampling.

    Returns
    -------
    y_voxels : ndarray (n_sampled, N)
        S0-normalised signals (S/S0) for the sampled voxels.
    sigma_normalised : float
        An estimate of the noise standard deviation in the SAME
        normalised (S/S0) units as y_voxels, obtained by dividing the
        per-voxel temporal/spatial sigma by each voxel's own S0 and
        taking the median — consistent with how `estimate_snr_robust`
        characterises noise, but rescaled into normalised-signal units
        since that is the space `select_lambda_aniso_discrepancy`
        operates in.
    """
    rng = np.random.default_rng(seed)

    bvals = np.asarray(bvals, dtype=np.float64)
    b0_idx = np.where(bvals < b0_thr)[0]

    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("sample_calibration_voxels: mask is empty.")

    # Compute S0 for every mask voxel (mean of b0 volumes) to filter and
    # to normalise sampled signals.
    if len(b0_idx) > 0:
        s0_all = data[mask][:, b0_idx].mean(axis=1)
    else:
        s0_all = data[mask][:, 0]

    median_s0 = np.median(s0_all[s0_all > 0]) if np.any(s0_all > 0) else 1.0
    valid = s0_all > (min_signal_fraction * median_s0)

    valid_coords = coords[valid]
    valid_s0 = s0_all[valid]

    if len(valid_coords) == 0:
        raise ValueError(
            "sample_calibration_voxels: no voxels passed the "
            "min_signal_fraction filter. Check the mask and data scale."
        )

    n_sample = min(n_voxels, len(valid_coords))
    sample_idx = rng.choice(len(valid_coords), size=n_sample, replace=False)

    sampled_coords = valid_coords[sample_idx]
    sampled_s0 = valid_s0[sample_idx]

    xs, ys, zs = sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]
    raw_signals = data[xs, ys, zs, :].astype(np.float64)
    y_voxels = raw_signals / sampled_s0[:, None]

    # Noise sigma in normalised units: use the spread of the b0 volumes
    # themselves (already S0-normalised) as a simple, direct estimate
    # consistent with the temporal method in estimate_snr_robust, but
    # expressed in the normalised-signal space used here.
    if len(b0_idx) >= 2:
        b0_norm = raw_signals[:, b0_idx] / sampled_s0[:, None]
        sigma_normalised = float(np.median(np.std(b0_norm, axis=1, ddof=1)))
    else:
        # Fallback: residual spread around the per-voxel mean signal as
        # a rough proxy when too few b0 volumes are available for a
        # direct temporal estimate.
        sigma_normalised = float(np.median(np.std(y_voxels, axis=1)) * 0.3)

    return y_voxels, sigma_normalised


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def select_lambdas_data_driven(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid,
                                y_voxels, sigma,
                                lambda_iso_grid=None,
                                lambda_aniso_bracket=(1e-6, 1e4)):
    """
    Convenience wrapper: select (lambda_aniso, lambda_iso) for a given
    Stage A dictionary using GCV (lambda_iso) followed by the
    discrepancy principle (lambda_aniso), both purely from the data
    (design matrix + observed signal + noise sigma) with no tissue-
    fraction priors.

    This is intended as a drop-in alternative to `calibration.optimizer.
    optimize_hyperparameters` (the Monte Carlo scenario-based
    calibration) for callers who want a data-driven lambda selection.
    The Monte Carlo scenarios remain available separately as an
    independent cross-validation check — see `calibration/optimizer.py`
    module docstring.

    Parameters
    ----------
    bvals, bvecs : arrays
        Acquisition protocol.
    fiber_dirs : array (M, 3)
        Stage A hemisphere directions.
    diff_pairs : array (P, 2)
        Stage A (AD, RD) pairs.
    iso_grid : array (K,)
        Isotropic ADC grid.
    y_voxels : array (N,) or (n_voxels, N)
        Representative normalised signal(s) (S/S0) from the actual data
        being processed — e.g. a random sample of brain-mask voxels.
        Using multiple voxels (10-50 is typically sufficient) gives a
        more stable protocol-level estimate than a single voxel.
    sigma : float
        Noise standard deviation in normalised (S/S0) units (already
        estimated via `estimate_snr_robust`, then divided by the same
        S0 normalisation used for y_voxels).
    lambda_iso_grid : array or None
        Candidate grid for the GCV search (see `select_lambda_iso_gcv`).
    lambda_aniso_bracket : tuple (float, float)
        Search bracket for the discrepancy principle bisection (see
        `select_lambda_aniso_discrepancy`).

    Returns
    -------
    lambda_aniso : float
    lambda_iso : float
    diagnostics : dict
        Combined diagnostics from both sub-selections, for reporting /
        sanity-checking (e.g. in a methods supplement table).
    """
    from ..core.basis import build_design_matrix_exhaustive

    n_dirs = len(fiber_dirs)
    n_pairs = len(diff_pairs)
    n_aniso_cols = n_dirs * n_pairs

    lambda_iso, gcv_diag = select_lambda_iso_gcv(
        bvals, iso_grid, y_voxels, lambda_grid=lambda_iso_grid
    )

    A = build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid)
    AtA = A.T @ A
    At = A.T

    lambda_aniso, disc_diag = select_lambda_aniso_discrepancy(
        AtA, At, y_voxels, n_aniso_cols, sigma, lambda_iso,
        lambda_lo=lambda_aniso_bracket[0], lambda_hi=lambda_aniso_bracket[1],
        n_dirs=n_dirs,
    )

    diagnostics = dict(gcv=gcv_diag, discrepancy=disc_diag)

    return lambda_aniso, lambda_iso, diagnostics
