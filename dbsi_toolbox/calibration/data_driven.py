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
from ..core.solvers import nnls_coordinate_descent, compute_regularization_matrix


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
# Noise-referenced ceiling for lambda_iso (discrepancy principle)
# ─────────────────────────────────────────────────────────────────────────────

def lambda_iso_discrepancy_cap(bvals, iso_grid, y_iso_resid, sigma, lambda_grid):
    """Largest lambda_iso whose isotropic-block residual still sits AT the
    noise floor (discrepancy principle), used as an upper CEILING on the GCV
    pick for lambda_iso.

    WHY (low-SNR FF-leakage, discovered 2026-07-21 on the tumor-like complex
    voxel) -----------------------------------------------------------------
    `select_lambda_iso_gcv` minimises GCV over `lambda_grid` (default
    logspace(-5, 1), i.e. max 10). On a noise-dominated fiber-subtracted iso
    residual (low SNR), the GCV curve becomes monotone-decreasing in lambda
    and its argmin RAILS at the grid maximum. That over-regularizes the iso
    block so hard that every isotropic fraction is shrunk to ~0 and the fiber
    fraction leaks to ~1.0 — the exact failure the grid-fix cured, returning
    through the calibration. Empirically lambda_iso ran 0.29 (SNR30) -> 10.0
    (SNR10), tracking the grid ceiling rather than any real optimum.

    THE CEILING --------------------------------------------------------------
    The discrepancy principle says the iso block should fit the iso residual
    down to (not below) the noise level: mean ||y_iso_resid - A_iso w(lambda)||^2
    == N*sigma^2. The residual is monotone increasing in lambda, so there is a
    unique largest lambda meeting `residual^2 <= N*sigma^2`; beyond it the iso
    block is UNDER-fitting real isotropic signal (throwing it away). That
    lambda is a principled, noise-scaled ceiling: capping the GCV pick to it
    prevents the rail-to-grid-max collapse while never forcing MORE
    regularization than GCV already wanted at good SNR (there the GCV optimum
    is well below this ceiling, so the cap is inert).

    Returns
    -------
    float
        The ceiling lambda. == lambda_grid.max() if even the largest lambda
        keeps the residual within the noise floor (no iso signal above noise ->
        no cap needed); == lambda_grid.min() if even minimal regularization
        already exceeds the noise floor (poor fiber subtraction / sigma
        mismatch -> stay permissive, i.e. do not over-regularize).
    """
    A_iso = build_isotropic_dictionary(bvals, iso_grid)
    N, _ = A_iso.shape
    U, s, _ = np.linalg.svd(A_iso, full_matrices=False)
    s2 = s ** 2

    y = np.atleast_2d(np.asarray(y_iso_resid, dtype=np.float64))
    Uty = U.T @ y.T                       # (K, n_voxels)
    y_norm2 = np.sum(y ** 2, axis=1)      # (n_voxels,)
    target = N * sigma ** 2

    lambda_grid = np.sort(np.asarray(lambda_grid, dtype=np.float64))
    cap = float(lambda_grid[0])           # most-permissive fallback
    for lam in lambda_grid:
        filt = s2 / (s2 + lam)
        fitted = filt[:, None] * Uty
        resid2 = y_norm2 - 2 * np.sum(fitted * Uty, axis=0) + np.sum(fitted ** 2, axis=0)
        resid2 = np.maximum(resid2, 0.0)
        if np.mean(resid2) <= target:
            cap = float(lam)              # still within noise floor -> allowed
        else:
            break                         # monotone: all larger lambda under-fit
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo NULL calibration — fiber-detection concentration gate
# ─────────────────────────────────────────────────────────────────────────────

def _dominant_basin_concentration(w_aniso, n_dirs, n_pairs, fiber_dirs,
                                  neighbor_idx):
    """Fraction of the total anisotropic weight held by the LARGEST angular
    Voronoi basin (basins seeded at the local-maxima of the per-direction
    weight, assigned by nearest peak). This mirrors exactly the concentration
    quantity gated in `core.solvers.select_dominant_directions`. Pure-python
    (runs once at calibration, not per voxel).

    Column ordering must match `build_design_matrix_exhaustive` (pair-major,
    direction-minor): w_aniso[p*n_dirs + d].
    """
    dwt = np.asarray(w_aniso, dtype=np.float64).reshape(n_pairs, n_dirs).sum(axis=0)
    total = float(dwt.sum())
    if total < 1e-12:
        return 0.0
    k = neighbor_idx.shape[1]
    peaks = []
    for d in range(n_dirs):
        wd = dwt[d]
        if wd <= 0.0:
            continue
        is_pk = True
        for j in range(k):
            nb = neighbor_idx[d, j]
            if dwt[nb] > wd or (dwt[nb] == wd and nb < d):
                is_pk = False
                break
        if is_pk:
            peaks.append(d)
    if not peaks:
        return 0.0
    P = fiber_dirs[peaks]                          # (n_peaks, 3)
    basins = np.zeros(len(peaks))
    for d in range(n_dirs):
        wd = dwt[d]
        if wd <= 0.0:
            continue
        adots = np.abs(P @ fiber_dirs[d])
        basins[int(np.argmax(adots))] += wd
    return float(basins.max() / total)


def calibrate_concentration_gate_mc(bvals, At, AtA_reg, n_aniso_cols, n_dirs,
                                    n_pairs, fiber_dirs, neighbor_idx, sigma,
                                    iso_d_range, fiber_threshold,
                                    default_gate, percentile=99.0, n_mc=400,
                                    b0_thr=100.0, seed=0):
    """Monte-Carlo NULL calibration of the fiber-detection CONCENTRATION GATE
    used by `core.solvers.select_dominant_directions`.

    WHY (data-driven, protocol-general) ------------------------------------
    The gate must separate a real fiber (anisotropic weight CONCENTRATED in
    one angular basin) from isotropic (esp. restricted) leakage on a
    fiber-FREE voxel (weight DIFFUSE across many basins). The concentration a
    fiber-free voxel can produce is not a universal constant — it depends on
    the protocol (b-values, direction count), the Stage-A dictionary, the
    calibrated (lambda_aniso, lambda_iso) and the noise level. A fixed default
    (0.35) was tuned on one synthetic protocol/SNR and will not generalise.

    This routine builds the NULL distribution empirically FOR THE DATA AT
    HAND: it draws `n_mc` purely-isotropic signals (random 1-3 component
    log-uniform mixtures over `iso_d_range`), corrupts them with Rician noise
    at the estimated `sigma`, applies the same noise-floor bias correction as
    the pipeline, projects each onto the Stage-A anisotropic block with the
    SAME regularized NNLS the fit uses, and measures the dominant-basin
    concentration. The gate is the `percentile` (e.g. 99th) of that null: a
    fiber-free voxel essentially never exceeds it, so anything above is a
    genuinely concentrated fiber. Auto-adapts to protocol + dictionary +
    lambdas + SNR.

    Returns
    -------
    gate : float
        Concentration threshold in [0, 1].
    diag : dict
        Null-distribution summary (percentile, n_mc, sigma, conc p50/p95/max).
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    N = len(bvals)
    is_b0 = bvals < b0_thr
    lo, hi = np.log(iso_d_range[0]), np.log(iso_d_range[1])

    # Collect the concentration ONLY for fiber-free samples whose isotropic
    # leakage into the anisotropic block exceeds fiber_threshold — i.e. the
    # samples that would actually REACH the gate in the kernel (which only
    # calls select_dominant_directions when f_fib > fiber_threshold). Samples
    # well-explained by the iso block (f_fib below threshold) never trigger
    # detection, so including them (and their degenerate near-zero-weight
    # concentrations) would corrupt the null. Draw up to a bounded pool.
    concs = []
    max_attempts = n_mc * 8
    for _ in range(max_attempts):
        if len(concs) >= n_mc:
            break
        n_comp = int(rng.integers(1, 4))
        Ds = np.exp(rng.uniform(lo, hi, n_comp))
        fr = rng.dirichlet(np.ones(n_comp))
        s = np.zeros(N)
        for D, f in zip(Ds, fr):
            s += f * np.exp(-bvals * D)
        n1 = rng.normal(0.0, sigma, N)
        n2 = rng.normal(0.0, sigma, N)
        meas = np.sqrt((s + n1) ** 2 + n2 ** 2)
        meas = np.sqrt(np.maximum(meas ** 2 - 2.0 * sigma ** 2, 0.0))  # bias corr
        s0 = meas[is_b0].mean() if is_b0.any() else meas.mean()
        if s0 < 1e-6:
            continue
        w, _ = nnls_coordinate_descent(AtA_reg, At @ (meas / s0), 0.0)
        w_fib = float(w[:n_aniso_cols].sum())
        w_tot = float(w.sum())
        if w_tot < 1e-10 or (w_fib / w_tot) <= fiber_threshold:
            continue                       # would not reach the gate in the kernel
        concs.append(_dominant_basin_concentration(
            w[:n_aniso_cols], n_dirs, n_pairs, fiber_dirs, neighbor_idx))

    n_valid = len(concs)
    if n_valid < 30:
        # Too few fiber-free voxels leak past fiber_threshold for this
        # protocol/SNR -> false crossings are already unlikely; the null is
        # too sparse to trust. Fall back to the fixed default gate.
        diag = dict(percentile=float(percentile), n_mc=int(n_mc),
                    n_valid=n_valid, sigma=float(sigma), fell_back=True,
                    conc_p50=float(np.median(concs)) if concs else float('nan'),
                    conc_p95=float(np.percentile(concs, 95)) if concs else float('nan'),
                    conc_max=float(np.max(concs)) if concs else float('nan'))
        return float(default_gate), diag

    concs = np.asarray(concs)
    gate = float(np.percentile(concs, percentile))
    diag = dict(percentile=float(percentile), n_mc=int(n_mc), n_valid=n_valid,
                sigma=float(sigma), fell_back=False,
                conc_p50=float(np.median(concs)),
                conc_p95=float(np.percentile(concs, 95)),
                conc_max=float(concs.max()))
    return gate, diag


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo RESPONSE FUNCTION — data-driven restricted-fraction correction
# ─────────────────────────────────────────────────────────────────────────────

def _mc_signal(bvals, bvecs, iso_comps, fibers, sigma, rng, b0_thr=100.0):
    """One Rician-noised, bias-corrected, S0-normalised synthetic voxel.
    iso_comps: list of (fraction, D). fibers: list of (fraction, dir, AD, RD)."""
    N = len(bvals)
    s = np.zeros(N)
    for f, D in iso_comps:
        s += f * np.exp(-bvals * D)
    for f, fdir, ad, rd in fibers:
        cos2 = (bvecs @ fdir) ** 2
        dapp = rd + (ad - rd) * cos2
        contrib = f * np.exp(-bvals * dapp)
        contrib[bvals < b0_thr] = f            # b~0: no attenuation
        s += contrib
    n1 = rng.normal(0.0, sigma, N)
    n2 = rng.normal(0.0, sigma, N)
    meas = np.sqrt((s + n1) ** 2 + n2 ** 2)
    meas = np.sqrt(np.maximum(meas ** 2 - 2.0 * sigma ** 2, 0.0))
    s0 = meas[bvals < b0_thr].mean() if (bvals < b0_thr).any() else meas.mean()
    return meas / s0 if s0 > 1e-6 else meas


def build_rf_response_table(bvals, bvecs, At, AtA_reg, n_aniso_cols, iso_grid,
                            thresh_res, sigma, ff_levels, rf_levels, reps=40,
                            b0_thr=100.0, seed=0,
                            restricted_d_range=(0.1e-3, 0.28e-3),
                            hindered_d_range=(0.4e-3, 2.5e-3),
                            water_d_range=(3.0e-3, 5.0e-3),
                            fiber_ad_range=(1.2e-3, 2.0e-3),
                            fiber_rd_range=(0.1e-3, 0.8e-3)):
    """Build the per-dataset RESPONSE FUNCTION for the restricted fraction.

    WHY (data-driven bias correction) --------------------------------------
    At clinical b-max the estimated restricted fraction (RF) is systematically
    UNDER-recovered: the D~0.15e-3 signal is partly absorbed by the hindered
    band (and, in fiber voxels, the fiber block). An RF_true -> RF_est transfer
    was measured to be MONOTONE, smooth and low-variance -> the systematic
    part is INVERTIBLE. This routine measures that transfer FOR THE DATA AT
    HAND (same protocol, dictionary, calibrated lambdas via AtA_reg, and
    estimated noise sigma), so the correction re-tunes to every protocol/SNR.

    For a grid of (true fiber fraction FF, true restricted fraction RF) it
    draws `reps` synthetic voxels with the restricted D and ALL nuisance
    (hindered/water split + Ds, fiber count/directions/diffusivities)
    randomised — i.e. the nuisance is MARGINALISED — corrupts them with Rician
    noise at `sigma`, projects onto the Stage-A block with the SAME regularized
    NNLS the fit uses, and records the mean recovered RF_est (and mean FF_est).
    The result is a table RF_est[FF_true row, RF_true col] that
    `apply_rf_correction` inverts per voxel.

    Returns
    -------
    ff_est_rows : ndarray (n_ff,)   mean recovered FF for each FF_true row
    rf_levels   : ndarray (n_rf,)   the true-RF grid (columns)
    rf_est_grid : ndarray (n_ff, n_rf)  mean recovered RF (NaN where FF+RF>~1)
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    iso_D = np.asarray(iso_grid, dtype=np.float64)
    res_mask = iso_D <= thresh_res
    ff_levels = np.asarray(ff_levels, dtype=np.float64)
    rf_levels = np.asarray(rf_levels, dtype=np.float64)

    rf_est_grid = np.full((len(ff_levels), len(rf_levels)), np.nan)
    ff_est_acc = np.zeros(len(ff_levels))
    ff_est_cnt = np.zeros(len(ff_levels))

    def _rand_dir():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    for j, ff in enumerate(ff_levels):
        for i, rf in enumerate(rf_levels):
            if ff + rf > 0.95:
                continue
            rf_acc = 0.0
            for _ in range(reps):
                iso_comps = []
                if rf > 0:
                    iso_comps.append((rf, rng.uniform(*restricted_d_range)))
                rem = max(1.0 - rf - ff, 0.0)
                hf = rem * rng.uniform(0.4, 0.95)
                wf = rem - hf
                if hf > 0:
                    iso_comps.append((hf, rng.uniform(*hindered_d_range)))
                if wf > 0:
                    iso_comps.append((wf, rng.uniform(*water_d_range)))
                fibers = []
                if ff > 1e-6:
                    nf = int(rng.integers(1, 3))
                    fr = rng.dirichlet(np.ones(nf)) * ff
                    for kf in range(nf):
                        ad = rng.uniform(*fiber_ad_range)
                        rd = rng.uniform(fiber_rd_range[0], min(fiber_rd_range[1], ad / 1.2))
                        fibers.append((float(fr[kf]), _rand_dir(), ad, rd))
                y = _mc_signal(bvals, bvecs, iso_comps, fibers, sigma, rng, b0_thr)
                w, _ = nnls_coordinate_descent(AtA_reg, At @ y, 0.0)
                tot = float(w.sum())
                if tot < 1e-10:
                    continue
                rf_acc += float(w[n_aniso_cols:][res_mask].sum()) / tot
                ff_est_acc[j] += float(w[:n_aniso_cols].sum()) / tot
                ff_est_cnt[j] += 1
            rf_est_grid[j, i] = rf_acc / reps
    ff_est_rows = ff_est_acc / np.maximum(ff_est_cnt, 1)
    return ff_est_rows, rf_levels, rf_est_grid


def apply_rf_correction(rf_obs, ff_obs, ff_est_rows, rf_levels, rf_est_grid):
    """Invert the RF response table (`build_rf_response_table`) per voxel:
    map the OBSERVED (RF_est, FF_est) back to the RF_true that produced it.
    Vectorised over voxels. Values below the response's dead-zone map toward 0
    (the restricted signal was unrecoverable there — flag separately).

    Returns the corrected RF array (same shape as rf_obs)."""
    rf_obs = np.asarray(rf_obs, dtype=np.float64)
    ff_obs = np.asarray(ff_obs, dtype=np.float64)
    rf_levels = np.asarray(rf_levels, dtype=np.float64)
    n_rows = len(ff_est_rows)

    # Per-row inversion: RF_true = interp(RF_obs; xp=monotone RF_est_row, fp=RF_true)
    per_row = np.empty((n_rows, rf_obs.size))
    for j in range(n_rows):
        row = rf_est_grid[j]
        valid = ~np.isnan(row)
        xp = np.maximum.accumulate(row[valid])          # enforce non-decreasing
        fp = rf_levels[valid]
        per_row[j] = np.interp(rf_obs, xp, fp)

    # Interpolate across FF rows (vectorised linear).
    order = np.argsort(ff_est_rows)
    ffr = np.asarray(ff_est_rows)[order]
    per_row = per_row[order]
    idx = np.clip(np.searchsorted(ffr, ff_obs) - 1, 0, n_rows - 2)
    f0 = ffr[idx]; f1 = ffr[idx + 1]
    wgt = np.clip((ff_obs - f0) / np.maximum(f1 - f0, 1e-9), 0.0, 1.0)
    cols = np.arange(rf_obs.size)
    corrected = per_row[idx, cols] * (1.0 - wgt) + per_row[idx + 1, cols] * wgt
    return np.clip(corrected, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DISCREPANCY PRINCIPLE — lambda_aniso
# ─────────────────────────────────────────────────────────────────────────────

def select_lambda_aniso_discrepancy(AtA, At, y_voxels, n_aniso_cols,
                                     sigma, lambda_iso_fixed,
                                     lambda_lo=1e-6, lambda_hi=1e4,
                                     n_dirs=None, max_bisect_iter=40,
                                     tol=1e-3, min_floor_factor=0.1,
                                     max_eig_aniso=None,
                                     aniso_floor_fraction=0.075):
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

    SAFETY FLOOR — two components (min_floor_factor, aniso_floor_fraction)
    -----------------------------------------------------------------------
    If the calibration voxel sample is small, near-noise-free, or
    unusually homogeneous (e.g. synthetic test signals with little
    inter-voxel variability), the residual at lambda_aniso -> 0 can
    already meet or undershoot the noise-matching target, causing the
    bisection to collapse to (near) zero — a pathological result: an
    unregularized anisotropic block with hundreds of collinear columns
    (n_aniso_cols can exceed N, the number of measurements, for typical
    n_dirs) has an enormous condition number and produces an
    ill-conditioned, numerically unstable fit even though it nominally
    "matches" the discrepancy target. Matching the residual to the noise
    level is necessary but not sufficient for a well-posed inverse
    problem when the unregularized system is singular or near-singular.

    Originally this floor was a single term, `min_floor_factor *
    lambda_iso_fixed`. Synthetic validation (crossing-fiber phantoms,
    2 populations at 45/90 deg, tilted 25 deg out of the Stage A
    hemisphere's equatorial boundary, SNR 20-50, n_dirs in {40, 78, 120})
    found this collapses to a near-zero value whenever lambda_iso_fixed
    itself is small (e.g. GCV selecting a low-n_iso, low-lambda_iso
    regime) -- REGARDLESS of how ill-conditioned the anisotropic block
    is. The two blocks' regularizations are calibrated independently and
    have no principled relationship, so tying the anisotropic floor to
    the isotropic block's scale is a category error: the same
    min_floor_factor*lambda_iso_fixed floor was observed to leave the
    fiber fraction almost completely leaked out of the isotropic
    compartment (recovered FF approx 1.0 for a synthetic voxel with true
    FF=0.80) even though the true anisotropic block condition number was
    ~1e8.

    `aniso_floor_fraction * max_eig_aniso` is a second, independent floor
    term tied instead to the anisotropic block's OWN scale (its largest
    unregularized eigenvalue) -- see `max_eig_aniso` parameter below. The
    two floors are combined via max(), so either can bind. Anchoring the
    floor to the anisotropic block's own eigenvalue scale, rather than to
    the unrelated isotropic block's lambda_iso, remains the right
    structural fix: the two blocks have no principled regularization
    relationship, so the old min_floor_factor*lambda_iso_fixed-only floor
    was a category error regardless of the point below.

    ***CAVEAT ON THE DEFAULT CONSTANT (2026-07-16, unresolved)***
    aniso_floor_fraction=0.075 was fitted from a sweep that (by mistake)
    forced n_iso explicitly in the DBSI_Adaptive constructor for every
    test point. Explicit n_iso routes the fit through
    `core.basis.generate_isotropic_grid` (a plain linear grid), NOT
    `generate_anchored_isotropic_grid` (the threshold-anchored grid the
    real adaptive pipeline uses when n_iso is left at its default None).
    These are materially different dictionaries, and a follow-up check
    on the REAL pipeline (n_iso left at None, so the bootstrap-selected
    n_iso and the anchored grid are both exercised as they would be in
    production) found FF leakage essentially UNCHANGED across the entire
    lambda_aniso range up to 10 for the same synthetic crossing-fiber
    voxel that motivated this fix -- i.e. this floor term, at its
    current default, does not fix the leakage it was designed to fix on
    the actual pipeline. Root cause traced to the anchored grid itself:
    at the bootstrap-selected n_iso=4 for that protocol/SNR, the
    threshold-anchored grid is `[1e-4, 2.99e-4, 3.001e-3, 5e-3]` mm^2/s
    -- two of the four points are spent marking the RES/HIN and HIN/WAT
    boundaries, leaving NO column near a mid-HIN-range diffusivity like
    0.8e-3 mm^2/s. No amount of lambda_aniso regularization can compensate
    for a genuine coverage gap in the isotropic dictionary; the isotropic
    signal is then more cheaply explained by a near-isotropic anisotropic
    column than by the two far-away anchored columns. This is an
    isotropic-side problem (likely in `calibration.adaptive_n_iso.
    select_n_iso_bootstrap`'s bias-variance criterion selecting too
    coarse an n_iso for this regime -- confirmed to persist even with a
    tissue-heterogeneous calibration sample, so it is not simply a
    calibration-sample-diversity issue either), NOT something this
    anisotropic-side floor can or should try to compensate for.
    DO NOT tune aniso_floor_fraction further to try to mask this -- the
    two compartments were deliberately investigated together once (to
    find this coupling) and are being investigated separately from here:
    this docstring's job is to flag that the current default is
    unvalidated on the real pipeline, not to guess a patched value. See
    project notes for the isotropic-side follow-up.

    `min_floor_factor` is retained as a secondary, more conservative
    floor (still enforcing lambda_aniso >= min_floor_factor *
    lambda_iso_fixed as an absolute lower bound) for backward
    compatibility and as a fallback when `max_eig_aniso` is not supplied
    (callers that do not pass it get the original, pre-fix behaviour).

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
        this component of the floor (not recommended — see rationale
        above). Retained as a secondary/fallback floor alongside
        `aniso_floor_fraction`.
    max_eig_aniso : float or None
        Largest eigenvalue of the UNREGULARIZED anisotropic block
        (AtA[:n_aniso_cols, :n_aniso_cols]), i.e.
        `np.linalg.eigvalsh(AtA[:n_aniso_cols, :n_aniso_cols])[-1]`.
        Computed once per protocol by the caller (see
        `select_lambdas_data_driven`), not per voxel. If None, this
        floor component is disabled and behaviour falls back to the
        original min_floor_factor-only floor (pre-fix behaviour).
    aniso_floor_fraction : float
        Fraction of `max_eig_aniso` used as the (new) primary floor
        component: lambda_aniso >= aniso_floor_fraction * max_eig_aniso
        / n_aniso_cols. Default 0.075 -- ***UNVALIDATED on the real
        (anchored-isotropic-grid) pipeline as of 2026-07-16; see the
        CAVEAT in the SAFETY FLOOR section above before relying on this
        default.*** Only used if max_eig_aniso is not None.

    Returns
    -------
    best_lambda_aniso : float
    diagnostics : dict
        {'target_residual2': float, 'achieved_residual2': float,
         'n_iter': int, 'floor_applied': bool,
         'floor_component': str} for sanity-checking the bisection
         converged to a residual close to the target rather than hitting
         a bracket edge, whether the safety floor had to intervene, and
         (if so) which component of the floor bound
         ('min_floor_factor', 'aniso_floor_fraction', or 'none').
    """
    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))
    n_voxels, N = y_voxels.shape

    target_residual2 = N * sigma ** 2

    floor_from_iso = min_floor_factor * lambda_iso_fixed
    if max_eig_aniso is not None:
        floor_from_aniso = aniso_floor_fraction * max_eig_aniso / max(n_aniso_cols, 1)
    else:
        floor_from_aniso = 0.0
    lambda_aniso_floor = max(floor_from_iso, floor_from_aniso)
    floor_component = ('none' if lambda_aniso_floor == 0.0 else
                       'aniso_floor_fraction' if floor_from_aniso >= floor_from_iso else
                       'min_floor_factor')

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
            floor_component=floor_component if floor_applied else 'none',
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
            floor_component='none',
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
        floor_component=floor_component if floor_applied else 'none',
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

    A = build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid)
    AtA = A.T @ A
    At = A.T
    A_aniso = A[:, :n_aniso_cols]

    # Largest eigenvalue of the UNREGULARIZED anisotropic block, computed
    # once per protocol (not per voxel) -- feeds the aniso_floor_fraction
    # safety-floor component in select_lambda_aniso_discrepancy, which
    # anchors the lambda_aniso floor to the anisotropic block's own scale
    # instead of (as before the fix) to the unrelated isotropic block's
    # lambda_iso. See that function's docstring for the empirical basis
    # and caveats of this fix.
    AtA_aniso = AtA[:n_aniso_cols, :n_aniso_cols]
    max_eig_aniso = float(np.linalg.eigvalsh(AtA_aniso)[-1])

    y2d = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))

    # ── lambda_iso via GCV on the ISOTROPIC RESIDUAL (fiber-subtracted) ─────
    # Rationale: GCV for lambda_iso fits the isotropic block A_iso to the
    # signal. On fiber-containing voxels the raw signal is dominated by the
    # ANISOTROPIC component, which A_iso structurally cannot represent; that
    # model mismatch dominates the GCV residual, flattens the GCV curve and
    # pushes the selected lambda_iso far too high (it over-regularizes the
    # iso block, so the joint fit under-recovers the isotropic fraction).
    # Fix: first remove the fiber contribution with a preliminary joint NNLS
    # (first-pass lambdas), then run GCV on what the iso block must actually
    # explain. Shown to be insensitive to the preliminary lambda_aniso across
    # a >30x range on synthetic crossing-fiber + isotropic phantoms.
    lambda_iso_pass1, _gcv0 = select_lambda_iso_gcv(
        bvals, iso_grid, y2d, lambda_grid=lambda_iso_grid
    )
    lambda_aniso_pass1, _disc0 = select_lambda_aniso_discrepancy(
        AtA, At, y2d, n_aniso_cols, sigma, lambda_iso_pass1,
        lambda_lo=lambda_aniso_bracket[0], lambda_hi=lambda_aniso_bracket[1],
        n_dirs=n_dirs, max_eig_aniso=max_eig_aniso,
    )
    AtA_reg_prelim = compute_regularization_matrix(
        AtA, n_aniso_cols, lambda_aniso_pass1, lambda_iso_pass1
    )
    y_iso_resid = np.empty_like(y2d)
    for v in range(y2d.shape[0]):
        w, _ = nnls_coordinate_descent(AtA_reg_prelim, At @ y2d[v], 0.0)
        y_iso_resid[v] = y2d[v] - A_aniso @ w[:n_aniso_cols]

    lambda_iso_gcv, gcv_diag = select_lambda_iso_gcv(
        bvals, iso_grid, y_iso_resid, lambda_grid=lambda_iso_grid
    )

    # ── NOISE-REFERENCED CEILING on lambda_iso (fix low-SNR FF-leakage) ─────
    # At low SNR the GCV curve on the noise-dominated iso residual is monotone
    # and its argmin rails at the grid maximum, over-regularizing the iso block
    # until every isotropic fraction collapses to ~0 and FF leaks to ~1.0. The
    # discrepancy principle gives a principled ceiling: the largest lambda_iso
    # whose iso-block residual still matches the noise floor (N*sigma^2). Cap
    # the GCV pick to it. At good SNR the GCV optimum is well below the ceiling,
    # so this is inert; it only bites when GCV would otherwise diverge.
    lambda_iso_cap = lambda_iso_discrepancy_cap(
        bvals, iso_grid, y_iso_resid, sigma, gcv_diag['lambda_']
    )
    lambda_iso = min(lambda_iso_gcv, lambda_iso_cap)

    # ── lambda_aniso via discrepancy principle, using the refined lambda_iso.
    # (Uses the FULL signal — lambda_aniso governs the anisotropic fit.)
    lambda_aniso, disc_diag = select_lambda_aniso_discrepancy(
        AtA, At, y2d, n_aniso_cols, sigma, lambda_iso,
        lambda_lo=lambda_aniso_bracket[0], lambda_hi=lambda_aniso_bracket[1],
        n_dirs=n_dirs, max_eig_aniso=max_eig_aniso,
    )

    diagnostics = dict(
        gcv=gcv_diag, discrepancy=disc_diag,
        lambda_iso_pass1=lambda_iso_pass1, lambda_aniso_pass1=lambda_aniso_pass1,
        lambda_iso_gcv=lambda_iso_gcv, lambda_iso_cap=lambda_iso_cap,
        lambda_iso_capped=(lambda_iso_cap < lambda_iso_gcv),
    )

    return lambda_aniso, lambda_iso, diagnostics
