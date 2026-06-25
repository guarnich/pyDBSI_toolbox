"""
DBSI Core Solvers — v3: Hybrid Two-Stage Architecture
========================================================

WHY v2's SINGLE-STAGE EXHAUSTIVE APPROACH WAS REPLACED
----------------------------------------------------------
v2 attempted to fold orientation AND (AD, RD) estimation into a single
linear NNLS solve over an exhaustive (direction x AD/RD-pair) dictionary,
extracting AD_final/RD_final as weighted centroids over the activated
anisotropic columns. A systematic synthetic recovery validation
(`recovery_validation.py`, 55 swept configurations) showed this is NOT
numerically identifiable: median AD/RD relative errors ranged from ~20%
to >150% across every grid density tested (3x3 through 8x8 AD/RD pairs),
with WORSE recovery at finer grids, because the number of simultaneously
activated anisotropic columns grows roughly in proportion to dictionary
size — the centroid increasingly averages over an uninformative span of
the grid rather than concentrating on the true value. This is a genuine
structural collinearity problem, not a regularization-tuning problem: no
lambda_aniso/lambda_iso combination in the swept range fixed it.

v3 ARCHITECTURE: TWO STAGES, DIFFERENT PURPOSES
---------------------------------------------------
The insight preserved from the v2 design discussion (credited to Alonso
Ramirez-Manzanares) is correct and worth keeping: the dictionary DOES
need to "know" that pathology changes AD/RD, not just orientation.
What v2 got wrong was solving for orientation AND diffusivity
SIMULTANEOUSLY in one ill-conditioned linear system. v3 separates the
two questions, because they have different statistical character:

  STAGE A — "How many fiber populations, and in which directions?"
    This is fundamentally a SPARSE detection problem: a voxel contains a
    handful of distinct fiber populations at most. The exhaustive
    (direction x AD/RD-pair) dictionary is well suited to this question
    BECAUSE its richness is being used for what richness is good at —
    letting the data choose freely among many candidate explanations —
    while heavy regularization (lambda_aniso, sparsity-inducing) forces
    the solution toward a small number of active columns. We do not trust
    the centroid (AD, RD) value coming out of Stage A; we only trust
    WHICH DIRECTIONS were selected (see `select_dominant_directions`).

  STAGE B — "Given these directions, what are AD and RD?"
    Once the directions are fixed (typically 1, occasionally 2 for
    crossing fibers), this becomes a small, well-conditioned regression:
    at most a handful of orientation unknowns are now constants, and we
    are fitting 2 diffusivities (or 2 per population) against the full
    set of b-values/directions in the protocol — the same closed-form
    weighted-least-squares construction already validated in v1/v2 for
    the LINEAR initialisation step (`_estimate_AD_RD_2iso/3iso` in
    `model_Niso_adaptive_ff_thr.py`), just no longer treated as merely an
    "initial guess" to be refined by a non-linear grid search, since
    there is no longer a non-linear stage to refine it: Stage B's
    closed-form estimate IS the final AD/RD.

This resolves the v2 identifiability failure because Stage B's effective
number of free parameters (2 per active direction, typically 2-4 total)
is decoupled from the size of Stage A's dictionary — refining Stage A's
angular or (AD,RD) grid density no longer degrades Stage B's
conditioning.

Decoupled regularization (lambda_aniso vs lambda_iso) — Stage A only
------------------------------------------------------------------------
Unchanged rationale from v2 (see below), but now serves a narrower,
better-matched purpose: enforcing sparsity in Stage A's direction
selection, not also being asked to recover unbiased diffusivities (which
v2 incorrectly demanded of the same penalty).
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-7, max_iter=2000):
    """NNLS via Coordinate Descent with Active Set. Unchanged from v1/v2."""
    n = AtA.shape[0]
    x = np.zeros(n, dtype=np.float64)
    grad = -Aty.astype(np.float64)

    hess_diag = np.empty(n, dtype=np.float64)
    for k in range(n):
        hess_diag[k] = AtA[k, k] + reg_lambda + 1e-12

    for iteration in range(max_iter):
        max_update = 0.0

        for i in range(n):
            g_i = grad[i] + reg_lambda * x[i]

            if x[i] == 0.0 and g_i >= 0.0:
                continue

            x_new = max(0.0, x[i] - g_i / hess_diag[i])
            diff = x_new - x[i]

            if abs(diff) > 1e-14:
                if abs(diff) > max_update:
                    max_update = abs(diff)
                for k in range(n):
                    grad[k] += AtA[k, i] * diff
                x[i] = x_new

        if max_update < tol:
            break

    return x, iteration


def compute_regularization_matrix(AtA, n_aniso_cols, lambda_aniso, lambda_iso):
    """
    Build the decoupled regularization matrix for Stage A's design matrix.

    Gamma = diag(lambda_aniso * I_aniso, lambda_iso * I_iso). See module
    docstring: in v3 lambda_aniso only needs to enforce DIRECTIONAL
    sparsity (how many fiber populations), not also produce an unbiased
    (AD, RD) centroid — that job moved to Stage B.

    Parameters
    ----------
    AtA : ndarray (n_total_cols, n_total_cols)
    n_aniso_cols : int
        Number of anisotropic columns (n_dirs * n_pairs in the Stage A
        dictionary).
    lambda_aniso, lambda_iso : float

    Returns
    -------
    AtA_reg : ndarray (n_total_cols, n_total_cols)
    """
    n_total_cols = AtA.shape[0]
    reg_vec = np.zeros(n_total_cols, dtype=np.float64)

    reg_vec[:n_aniso_cols] = lambda_aniso
    reg_vec[n_aniso_cols:] = lambda_iso

    AtA_reg = AtA + np.diag(reg_vec)

    return AtA_reg


# ─────────────────────────────────────────────────────────────────────────────
# STAGE A — Direction selection from the exhaustive dictionary
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def select_dominant_directions(w_aniso, n_dirs, n_pairs, max_directions=2,
                               min_weight_fraction=0.05):
    """
    Stage A output interpretation: identify which hemisphere directions
    carry meaningful weight, collapsing across the (AD, RD)-pair axis.

    This function deliberately DISCARDS the per-pair breakdown of
    w_aniso and looks only at total weight per direction (summed over
    all n_pairs (AD,RD) pairs sharing that direction), because Stage A's
    only job is angular detection — see module docstring on why the
    per-pair centroid from Stage A is not trusted.

    Column ordering must match `core.basis.build_design_matrix_exhaustive`:
    pair-major, direction-minor (for p in pairs: for d in dirs: column).

    Parameters
    ----------
    w_aniso : array (n_aniso_cols,)
        NNLS weights restricted to the anisotropic (Stage A) block.
    n_dirs : int
        Number of hemisphere directions in the Stage A dictionary.
    n_pairs : int
        Number of (AD, RD) pairs in the Stage A dictionary.
    max_directions : int
        Maximum number of fiber populations to report (1 for a single
        dominant tract, 2 to allow simple crossing-fiber detection).
    min_weight_fraction : float
        A direction is only reported if its summed weight exceeds this
        fraction of the total anisotropic weight (filters numerical
        noise / near-zero spurious directions).

    Returns
    -------
    dir_indices : array (max_directions,), int64
        Hemisphere-direction indices of the selected populations, sorted
        by descending weight. Filled with -1 for unused slots if fewer
        than max_directions populations clear the weight threshold.
    dir_weights : array (max_directions,), float64
        Total weight (summed over all (AD,RD) pairs) for each selected
        direction. 0.0 for unused slots.
    """
    dir_weight_totals = np.zeros(n_dirs, dtype=np.float64)

    idx_col = 0
    for p in range(n_pairs):
        for d in range(n_dirs):
            dir_weight_totals[d] += w_aniso[idx_col]
            idx_col += 1

    total_weight = 0.0
    for d in range(n_dirs):
        total_weight += dir_weight_totals[d]

    dir_indices = np.full(max_directions, -1, dtype=np.int64)
    dir_weights = np.zeros(max_directions, dtype=np.float64)

    if total_weight < 1e-10:
        return dir_indices, dir_weights

    threshold = min_weight_fraction * total_weight

    remaining = dir_weight_totals.copy()
    for slot in range(max_directions):
        best_idx = -1
        best_val = -1.0
        for d in range(n_dirs):
            if remaining[d] > best_val:
                best_val = remaining[d]
                best_idx = d
        if best_idx == -1 or best_val < threshold:
            break
        dir_indices[slot] = best_idx
        dir_weights[slot] = best_val
        remaining[best_idx] = -1.0

    return dir_indices, dir_weights


# ─────────────────────────────────────────────────────────────────────────────
# STAGE B — Closed-form (AD, RD) estimation conditioned on direction
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def estimate_AD_RD_conditioned(bvals, bvecs, sig_norm, fiber_dir,
                               f_fib, f_res, f_hin, f_wat,
                               D_res, D_hin, D_wat, use_3iso):
    """
    Stage B: closed-form weighted-least-squares estimate of (AD, RD)
    for a SINGLE dominant fiber direction, given the isotropic
    compartment fractions/centroids already estimated.

    This is the same WLS construction validated as the v1/v2 analytical
    initialisation (`_estimate_AD_RD_2iso` / `_estimate_AD_RD_3iso` in
    `model_Niso_adaptive_ff_thr.py`), but in v3 it is the FINAL estimate
    rather than an initial guess for a subsequent non-linear refinement
    — there is no longer a non-linear stage. Unifying the 2-ISO/3-ISO
    cases into one function via the `use_3iso` flag avoids duplicating
    this analytical derivation a third time across the codebase.

    The fiber direction itself comes from Stage A
    (`select_dominant_directions`) — Stage B does NOT search over
    direction; it only fits the two diffusivities given a fixed
    direction. This is what keeps Stage B small and well-conditioned
    regardless of how rich Stage A's dictionary was.

    Parameters
    ----------
    bvals, bvecs : arrays
        Acquisition protocol.
    sig_norm : array
        Normalised signal (S / S0).
    fiber_dir : array (3,)
        Dominant fibre direction from Stage A (unit vector).
    f_fib : float
        Fiber fraction (from Stage A's NNLS solution, normalised).
    f_res : float
        Restricted fraction.
    f_hin, f_wat : float
        Hindered / free-water fractions if use_3iso, else f_hin is
        interpreted as the combined NRF fraction and f_wat is ignored
        (pass f_wat=0.0 in that case).
    D_res, D_hin, D_wat : float
        Centroid ADCs of the isotropic compartments (mm^2/s). If
        use_3iso is False, D_hin is the NRF centroid and D_wat is
        unused.
    use_3iso : bool
        Whether to treat (f_hin, f_wat, D_hin, D_wat) as three separate
        compartments (3-ISO) or merge hin into a single NRF term
        (2-ISO).

    Returns
    -------
    AD_est, RD_est : float
        Estimated diffusivities, or np.nan if the WLS system is
        singular (degenerate direction sampling or near-zero fiber
        fraction).
    """
    if use_3iso:
        ftot = f_fib + f_res + f_hin + f_wat + 1e-12
        ff = f_fib / ftot
        fr = f_res / ftot
        fh = f_hin / ftot
        fw = f_wat / ftot
    else:
        ftot = f_fib + f_res + f_hin + 1e-12
        ff = f_fib / ftot
        fr = f_res / ftot
        fh = f_hin / ftot
        fw = 0.0

    sum_AA = 0.0
    sum_AB = 0.0
    sum_BB = 0.0
    sum_Ay = 0.0
    sum_By = 0.0

    for i in range(len(bvals)):
        b = bvals[i]
        S_total = max(sig_norm[i], 0.01)

        if use_3iso:
            S_iso = fr * np.exp(-b * D_res) + fh * np.exp(-b * D_hin) + fw * np.exp(-b * D_wat)
        else:
            S_iso = fr * np.exp(-b * D_res) + fh * np.exp(-b * D_hin)

        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        S_fiber = max(min(S_fiber, 1.0), 0.01)
        log_S = np.log(S_fiber)

        g = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2 = cos_t * cos_t
        w = S_total * S_total

        sum_AA += w * b * b
        sum_AB += w * b * b * cos2
        sum_BB += w * b * b * cos2 * cos2
        sum_Ay += w * b * log_S
        sum_By += w * b * cos2 * log_S

    det = sum_AA * sum_BB - sum_AB * sum_AB
    if abs(det) < 1e-20:
        return np.nan, np.nan

    x = (sum_BB * sum_Ay - sum_AB * sum_By) / det
    y = (sum_AA * sum_By - sum_AB * sum_Ay) / det

    RD_est = max(0.05e-3, min(3.0e-3, -x))
    AD_est = max(0.05e-3, min(3.5e-3, -x - y))
    if AD_est < RD_est:
        m = (AD_est + RD_est) / 2.0
        AD_est = m
        RD_est = m

    return AD_est, RD_est


# ─────────────────────────────────────────────────────────────────────────────
# ISOTROPIC CENTROIDS AND FA — unchanged from v1/v2
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_weighted_centroids(w_iso, iso_grid):
    """Compute weighted centroids for isotropic components. Unchanged
    from v1/v2 — the isotropic block's centroid extraction is correct
    and is not affected by the Stage A/B split, which only concerns the
    anisotropic compartment.
    """
    THRESH_RES = 0.3e-3
    THRESH_WAT = 3.0e-3

    sum_w_res, sum_wd_res = 0.0, 0.0
    sum_w_hin, sum_wd_hin = 0.0, 0.0
    sum_w_wat, sum_wd_wat = 0.0, 0.0

    for k in range(len(iso_grid)):
        adc = iso_grid[k]
        w = w_iso[k]

        if adc <= THRESH_RES:
            sum_w_res += w
            sum_wd_res += w * adc
        elif adc <= THRESH_WAT:
            sum_w_hin += w
            sum_wd_hin += w * adc
        else:
            sum_w_wat += w
            sum_wd_wat += w * adc

    D_res = sum_wd_res / sum_w_res if sum_w_res > 1e-10 else 0.15e-3
    D_hin = sum_wd_hin / sum_w_hin if sum_w_hin > 1e-10 else 1.0e-3
    D_wat = sum_wd_wat / sum_w_wat if sum_w_wat > 1e-10 else 3.0e-3

    return D_res, D_hin, D_wat


@njit(cache=True, fastmath=True)
def compute_fiber_fa(AD, RD):
    """
    Compute FA for cylindrically symmetric tensor. Unchanged from v1/v2.
    Formula: FA = (AD - RD) / sqrt(AD^2 + 2*RD^2).
    """
    if AD < 1e-10 or RD < 1e-10:
        return 0.0

    if AD < RD:
        AD, RD = RD, AD

    diff = AD - RD
    if abs(diff) < 1e-10:
        return 0.0

    denom = np.sqrt(AD * AD + 2.0 * RD * RD)
    if denom < 1e-12:
        return 0.0

    FA_raw = diff / denom
    FA_raw = min(1.0, max(0.0, FA_raw))

    return FA_raw


# ─────────────────────────────────────────────────────────────────────────────
# v1/v2 LEGACY — single-stage centroid extraction and non-linear Step 2
# (DEPRECATED, NOT USED IN v3)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_aniso_centroids(w_aniso, diff_pairs, n_dirs, noise_floor=1e-5):
    """
    .. deprecated:: 3.0.0 (v3 hybrid two-stage release)
        This is the v2 single-stage centroid extraction, demonstrated by
        synthetic recovery validation to be non-identifiable (median
        AD/RD relative errors 20%-150%+ across all tested dictionary
        densities — see module docstring). Replaced by the
        `select_dominant_directions` (Stage A) +
        `estimate_AD_RD_conditioned` (Stage B) pair.

        Kept for backward compatibility / regression comparison against
        the v2 pipeline only. Do not call from new code.
    """
    n_pairs = len(diff_pairs)

    sum_w_fib = 0.0
    sum_ad_weight = 0.0
    sum_rd_weight = 0.0

    idx_col = 0
    for p in range(n_pairs):
        ad_base = diff_pairs[p, 0]
        rd_base = diff_pairs[p, 1]
        for d in range(n_dirs):
            wi = w_aniso[idx_col]
            if wi > noise_floor:
                sum_w_fib += wi
                sum_ad_weight += wi * ad_base
                sum_rd_weight += wi * rd_base
            idx_col += 1

    if sum_w_fib > 1e-10:
        AD_final = sum_ad_weight / sum_w_fib
        RD_final = sum_rd_weight / sum_w_fib
    else:
        AD_final = np.nan
        RD_final = np.nan

    return AD_final, RD_final, sum_w_fib


@njit(cache=True, fastmath=True)
def step2_refine_diffusivities_adaptive(bvals, bvecs, y_norm, fiber_dir,
                                        f_fiber, f_res, f_hin, f_wat,
                                        D_res, D_hin, D_wat,
                                        AD_init, RD_init):
    """
    .. deprecated:: 2.0.0 (v2 parametric-dictionary release)
        v1's non-linear Step 2 grid search. Not used in v2 or v3. Kept
        only for external code that may still import this symbol.
    """
    center_ax = AD_init
    center_rad = RD_init

    best_sse = 1e20
    best_ax = center_ax
    best_rad = center_rad

    ftot = f_fiber + f_res + f_hin + f_wat + 1e-12
    ff = f_fiber / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot

    n_ax, n_rad = 12, 10

    ax_min = max(0.5e-3, center_ax * 0.5)
    ax_max = min(2.5e-3, center_ax * 1.5)
    rad_min = max(0.1e-3, center_rad * 0.5)
    rad_max = min(1.2e-3, center_rad * 1.5)

    ax_step = (ax_max - ax_min) / (n_ax - 1) if n_ax > 1 else 0.0
    rad_step = (rad_max - rad_min) / (n_rad - 1) if n_rad > 1 else 0.0

    for i_ax in range(n_ax):
        ax = ax_min + i_ax * ax_step

        for i_rad in range(n_rad):
            rad = rad_min + i_rad * rad_step

            if ax < rad * 1.1:
                continue

            sse = 0.0
            for i in range(len(bvals)):
                b = bvals[i]
                if b < 50:
                    continue

                g = bvecs[i]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = rad + (ax - rad) * cos_t * cos_t

                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res) +
                          fh * np.exp(-b * D_hin) +
                          fw * np.exp(-b * D_wat))

                diff = y_norm[i] - s_pred
                sse += diff * diff

            if sse < best_sse:
                best_sse = sse
                best_ax = ax
                best_rad = rad

    ax_c, rad_c = best_ax, best_rad
    fine_ax = ax_step / 4 if ax_step > 0 else 0.05e-3
    fine_rad = rad_step / 4 if rad_step > 0 else 0.05e-3

    for di in range(-2, 3):
        ax = ax_c + di * fine_ax
        if ax < ax_min or ax > ax_max:
            continue

        for dj in range(-2, 3):
            rad = rad_c + dj * fine_rad
            if rad < rad_min or rad > rad_max:
                continue
            if ax < rad * 1.1:
                continue

            sse = 0.0
            for i in range(len(bvals)):
                b = bvals[i]
                if b < 50:
                    continue
                g = bvecs[i]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = rad + (ax - rad) * cos_t * cos_t

                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res) +
                          fh * np.exp(-b * D_hin) +
                          fw * np.exp(-b * D_wat))

                diff = y_norm[i] - s_pred
                sse += diff * diff

            if sse < best_sse:
                best_sse = sse
                best_ax = ax
                best_rad = rad

    return best_ax, best_rad


@njit(cache=True, fastmath=True)
def step2_refine_diffusivities(bvals, bvecs, y_norm, fiber_dir,
                               f_fiber, f_res, f_hin, f_wat,
                               D_res, D_hin, D_wat):
    """
    .. deprecated:: 2.0.0
        See `step2_refine_diffusivities_adaptive`. Not used in v2 or v3.
    """
    return step2_refine_diffusivities_adaptive(
        bvals, bvecs, y_norm, fiber_dir,
        f_fiber, f_res, f_hin, f_wat,
        D_res, D_hin, D_wat,
        1.5e-3,
        0.4e-3
    )
