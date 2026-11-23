"""
DBSI Core Solvers
========================================================

ARCHITECTURE: TWO STAGES, DIFFERENT PURPOSES
---------------------------------------------------

STAGE A 
SPARSE detection problem: a voxel contains a
handful of distinct fiber populations at most. The exhaustive
(direction x AD/RD-pair) dictionary is well suited to this question.

STAGE B 
Once the directions are fixed, this becomes a small, well-conditioned regression:
at most a handful of orientation unknowns are now constants, and we
are fitting 2 diffusivities (or 2 per population) against the full
set of b-values/directions in the protocol.

"""

import numpy as np
from numba import njit, prange
from .._numba_flags import FASTMATH


# ─── Bounds on the fiber tensorh ──────────────────────

_TENSOR_AD_FLOOR = 0.20e-3
_TENSOR_AD_CEIL = 3.50e-3
_TENSOR_RD_FLOOR = 0.15e-3  
_TENSOR_RD_CEIL = 3.00e-3
_NNLS_MAX_ITER = 20000
_STAGEC_N_REFINE = 3


@njit(cache=True, fastmath=FASTMATH, nogil=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-7,
                            max_iter=_NNLS_MAX_ITER):
    """NNLS via Coordinate Descent with Active Set."""
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


@njit(parallel=True, cache=True, fastmath=FASTMATH)
def nnls_batch(AtA, Aty_rows, reg_lambda):
    """`nnls_coordinate_descent` on every row of Aty_rows (n, cols) with the SAME
    matrix, voxels in parallel. Returns (W (n, cols), iterations (n,)).

    """
    n = Aty_rows.shape[0]
    W = np.empty((n, AtA.shape[0]), dtype=np.float64)
    it = np.empty(n, dtype=np.int64)
    for v in prange(n):
        w, k = nnls_coordinate_descent(AtA, Aty_rows[v], reg_lambda)
        W[v] = w
        it[v] = k
    return W, it


def compute_regularization_matrix(AtA, n_aniso_cols, lambda_aniso, lambda_iso):
    """
    Build the decoupled regularization matrix for Stage A's design matrix.

    Gamma = diag(lambda_aniso * I_aniso, lambda_iso * I_iso). 

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
#
# WHY LOCAL-MAXIMA PEAK-FINDING, 
# 
# A candidate direction is only counted as a detected population
# if its aggregated weight exceeds the weight of EVERY one of its k
# nearest geometric neighbours in the SAME dictionary. A true fiber's smeared neighbour is, by construction, always
# lower-weighted than the true peak itself, so it is never a local
# maximum and is correctly excluded. A genuine second, angularly
# well-separated fiber population has its own local neighbourhood and
# remains a local maximum independently.
#


def build_direction_neighbor_graph(fiber_dirs, k=6):
    """
    Precompute, once per protocol, the indices of each
    Stage A hemisphere direction's k angularly-nearest neighbours within
    the SAME dictionary. Feeds `select_dominant_directions`'s
    local-maxima peak-finding criterion.


    Parameters
    ----------
    fiber_dirs : ndarray (n_dirs, 3)
        Stage A's hemisphere direction dictionary.
    k : int
        Number of nearest neighbours per direction. Default 6.

    Returns
    -------
    neighbor_idx : ndarray (n_dirs, k_eff), int64
        Row d holds the indices of direction d's k_eff nearest
        neighbours by AXIAL distance (|dot|). k_eff = min(k, n_dirs - 1) to
        stay well-defined for very small dictionaries.
    """
    n = len(fiber_dirs)
    k_eff = max(1, min(k, n - 1))
    dots = np.abs(fiber_dirs @ fiber_dirs.T)
    np.fill_diagonal(dots, -2.0)
    neighbor_idx = np.argsort(-dots, axis=1)[:, :k_eff].astype(np.int64)
    return np.ascontiguousarray(neighbor_idx)


@njit(cache=True, fastmath=FASTMATH)
def select_dominant_directions(w_aniso, n_dirs, n_pairs, neighbor_idx,
                               fiber_dirs, max_directions=2,
                               min_weight_fraction=0.05,
                               min_separation_cos=0.8192,
                               min_peak_ratio=0.35,
                               min_dominant_concentration=0.35):
    """
    Stage A output interpretation: identify which hemisphere directions
    carry meaningful, GEOMETRICALLY DISTINCT weight, collapsing across
    the (AD, RD)-pair axis.

    This function deliberately DISCARDS the per-pair breakdown of
    w_aniso and looks only at total weight per direction (summed over
    all n_pairs (AD,RD) pairs sharing that direction), because Stage A's
    only job is angular detection.

    Detection is by LOCAL-MAXIMA peak-finding over the dictionary's
    k-nearest-neighbour graph (`build_direction_neighbor_graph`), then
    each local-maximum peak is scored by its ANGULAR-BASIN MASS (Voronoi
    assignment: every direction's weight is added to the nearest peak's
    basin), and peaks are accepted greedily in descending basin mass with
    angular NON-MAXIMUM SUPPRESSION.

    Parameters
    ----------
    w_aniso : array (n_aniso_cols,)
        NNLS weights restricted to the anisotropic (Stage A) block.
    n_dirs : int
        Number of hemisphere directions in the Stage A dictionary.
    n_pairs : int
        Number of (AD, RD) pairs in the Stage A dictionary.
    neighbor_idx : array (n_dirs, k), int64
        Precomputed once per protocol via `build_direction_neighbor_graph`.
    fiber_dirs : array (n_dirs, 3), float64
        Unit direction vectors of the hemisphere dictionary (used for the
        angular-basin assignment and the NMS separation test).
    max_directions : int
        Maximum number of fiber populations to report (1 for a single
        dominant tract, 2-3 to allow crossing-fiber detection).
    min_weight_fraction : float
        A peak is only reported if its BASIN MASS exceeds this fraction of
        the total anisotropic weight (filters numerical noise / spurious
        near-zero peaks).
    min_separation_cos : float
        cos of the minimum angular separation between two accepted
        populations. A candidate peak whose |cos| to an already-accepted
        peak EXCEEDS this (i.e. angle SMALLER than the separation) is
        suppressed as smearing skirt. Default 0.8192 = cos(35 deg).
    min_peak_ratio : float
        A secondary (non-dominant) population is only accepted if its basin
        mass is at least this fraction of the DOMINANT population's basin
        mass. Discriminates a genuine second fiber (carries a substantial
        share of the dominant's weight) from a single fiber's smearing
        skirt that noise pushed past `min_separation` (carries less). Default 0.35.
    min_dominant_concentration : float
        Minimum share of the TOTAL anisotropic weight that the dominant
        basin must hold for ANY fiber population to be reported. A real
        fiber concentrates its anisotropic weight along one direction
        (dominant basin ~0.4-0.6 of total); spurious anisotropic weight
        from isotropic (esp. restricted) leakage on a fiber-FREE voxel is
        DIFFUSE (dominant basin only ~0.2-0.24, spread across many similar
        peaks) and would otherwise be mis-read as a crossing (npop>=2,
        since every diffuse basin passes the relative test). Default 0.35: Below this
        concentration the whole anisotropic block is treated as leakage and
        NO population is returned (npop=0).

    Returns
    -------
    dir_indices : array (max_directions,), int64
        Hemisphere-direction indices of the selected populations, sorted
        by descending basin mass. Filled with -1 for unused slots.
    dir_weights : array (max_directions,), float64
        ANGULAR-BASIN MASS (summed weight in each peak's Voronoi basin)
        for each selected direction. 0.0 for unused slots. Used downstream
        to split the total fiber fraction among populations.
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
    k = neighbor_idx.shape[1]

    # ── Local-maxima peak-finding ────────────────────────────────────
    # A candidate counts as a peak if its weight exceeds ALL k geometric
    # nearest neighbours'. 

    is_peak = np.zeros(n_dirs, dtype=np.bool_)
    n_peaks = 0
    for d in range(n_dirs):
        wd = dir_weight_totals[d]
        if wd <= 0.0:
            continue
        peak = True
        for j in range(k):
            nb = neighbor_idx[d, j]
            wn = dir_weight_totals[nb]
            if wn > wd or (wn == wd and nb < d):
                peak = False
                break
        if peak:
            is_peak[d] = True
            n_peaks += 1

    if n_peaks == 0:
        return dir_indices, dir_weights

    peak_idx = np.empty(n_peaks, dtype=np.int64)
    t = 0
    for d in range(n_dirs):
        if is_peak[d]:
            peak_idx[t] = d
            t += 1

    # ── Angular-basin mass (Voronoi assignment to nearest peak) ──────
    basin_mass = np.zeros(n_peaks, dtype=np.float64)
    for d in range(n_dirs):
        wd = dir_weight_totals[d]
        if wd <= 0.0:
            continue
        best_adot = -1.0
        best_pk = 0
        for pk in range(n_peaks):
            pi = peak_idx[pk]
            dot = (fiber_dirs[d, 0] * fiber_dirs[pi, 0]
                   + fiber_dirs[d, 1] * fiber_dirs[pi, 1]
                   + fiber_dirs[d, 2] * fiber_dirs[pi, 2])
            adot = dot if dot >= 0.0 else -dot
            if adot > best_adot:
                best_adot = adot
                best_pk = pk
        basin_mass[best_pk] += wd

    # ── Concentration gate: reject DIFFUSE anisotropic weight ────────
    # A real fiber concentrates its weight in one basin; isotropic (esp.
    # restricted) leakage on a fiber-free voxel spreads it across many
    # near-equal basins. If no basin holds at least
    # `min_dominant_concentration` of the total anisotropic weight, the
    # whole block is leakage -> report NO population (npop=0), preventing
    # the diffuse-leakage-as-crossing false positive.

    max_basin = 0.0
    for pk in range(n_peaks):
        if basin_mass[pk] > max_basin:
            max_basin = basin_mass[pk]
    if max_basin < min_dominant_concentration * total_weight:
        return dir_indices, dir_weights

    # ── Greedy selection by basin mass + angular NMS ─────────────────
    used = np.zeros(n_peaks, dtype=np.bool_)
    n_sel = 0
    while n_sel < max_directions:
        best_val = -1.0
        best_pk = -1
        for pk in range(n_peaks):
            if used[pk]:
                continue
            if basin_mass[pk] > best_val:
                best_val = basin_mass[pk]
                best_pk = pk
        if best_pk == -1:
            break
        used[best_pk] = True
        # peaks are consumed in descending mass order; once below the
        # mass threshold, stop.
        if basin_mass[best_pk] < threshold:
            break
        # Relative threshold: a non-dominant population must carry at least
        # `min_peak_ratio` of the DOMINANT population's basin mass, else it
        # is a single fiber's smearing skirt (noise-shifted past the NMS
        # separation), not a distinct fiber. dir_weights[0] holds the
        # dominant's basin mass once the first peak is accepted.
        if n_sel >= 1 and basin_mass[best_pk] < min_peak_ratio * dir_weights[0]:
            continue
        # NMS: suppress a candidate too close to an already-accepted peak
        # (smearing skirt of a stronger fiber, not a distinct population).
        too_close = False
        qi = peak_idx[best_pk]
        for s in range(n_sel):
            si = dir_indices[s]
            dot = (fiber_dirs[si, 0] * fiber_dirs[qi, 0]
                   + fiber_dirs[si, 1] * fiber_dirs[qi, 1]
                   + fiber_dirs[si, 2] * fiber_dirs[qi, 2])
            adot = dot if dot >= 0.0 else -dot
            if adot > min_separation_cos:
                too_close = True
                break
        if too_close:
            continue
        dir_indices[n_sel] = qi
        dir_weights[n_sel] = basin_mass[best_pk]
        n_sel += 1

    return dir_indices, dir_weights


@njit(cache=True, fastmath=FASTMATH)
def dominant_basin_concentration(w_aniso, n_dirs, n_pairs, neighbor_idx,
                                 fiber_dirs):
    """
    Per-voxel ANGULAR CONCENTRATION of the anisotropic weight: the share of
    the total Stage-A anisotropic weight held by the single largest angular
    basin (Voronoi mass around the dominant local-maximum peak).

    This is EXACTLY the quantity the concentration gate in
    `select_dominant_directions` thresholds (max_basin / total_weight), but
    returned as a continuous scalar in [0, 1] for use as a per-voxel
    diagnostic channel and as the modulation variable for a per-voxel
    lambda_aniso adaptation.

    Interpretation (the discriminator Plan A rests on):
      - A REAL fiber -- even a demyelinated / low-FA one -- concentrates its
        anisotropic weight along one direction: high concentration (~0.4-0.6).
      - DIFFUSE anisotropic weight absorbed from isotropic (esp. restricted)
        signal on a fiber-free voxel spreads across many near-equal basins:
        low concentration (~0.2-0.25).

    Unlike n_pop (a binary/quantised detection count), concentration separates
    a weak-but-real fiber from leakage continuously.

    The peak-finding, basin-assignment and column-ordering conventions are
    IDENTICAL to `select_dominant_directions`.
    Returns 0.0 when the anisotropic block carries negligible weight or has no local maximum.

    Parameters
    ----------
    w_aniso : array (n_aniso_cols,)
        NNLS weights restricted to the anisotropic (Stage A) block.
    n_dirs, n_pairs : int
        Dictionary dimensions (n_aniso_cols == n_dirs * n_pairs).
    neighbor_idx : array (n_dirs, k), int64
        k-nearest-neighbour direction graph (from
        `build_direction_neighbor_graph`).
    fiber_dirs : array (n_dirs, 3), float64
        Hemisphere direction unit vectors.

    Returns
    -------
    concentration : float
        max_basin_mass / total_anisotropic_weight, in [0, 1]. 0.0 if the
        block is empty / has no peak.
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
    if total_weight < 1e-10:
        return 0.0

    k = neighbor_idx.shape[1]

    # Local-maxima peak-finding (same criterion as select_dominant_directions).
    is_peak = np.zeros(n_dirs, dtype=np.bool_)
    n_peaks = 0
    for d in range(n_dirs):
        wd = dir_weight_totals[d]
        if wd <= 0.0:
            continue
        peak = True
        for j in range(k):
            nb = neighbor_idx[d, j]
            wn = dir_weight_totals[nb]
            if wn > wd or (wn == wd and nb < d):
                peak = False
                break
        if peak:
            is_peak[d] = True
            n_peaks += 1

    if n_peaks == 0:
        return 0.0

    peak_idx = np.empty(n_peaks, dtype=np.int64)
    t = 0
    for d in range(n_dirs):
        if is_peak[d]:
            peak_idx[t] = d
            t += 1

    # Angular-basin mass (Voronoi assignment to nearest peak).
    basin_mass = np.zeros(n_peaks, dtype=np.float64)
    for d in range(n_dirs):
        wd = dir_weight_totals[d]
        if wd <= 0.0:
            continue
        best_adot = -1.0
        best_pk = 0
        for pk in range(n_peaks):
            pi = peak_idx[pk]
            dot = (fiber_dirs[d, 0] * fiber_dirs[pi, 0]
                   + fiber_dirs[d, 1] * fiber_dirs[pi, 1]
                   + fiber_dirs[d, 2] * fiber_dirs[pi, 2])
            adot = dot if dot >= 0.0 else -dot
            if adot > best_adot:
                best_adot = adot
                best_pk = pk
        basin_mass[best_pk] += wd

    max_basin = 0.0
    for pk in range(n_peaks):
        if basin_mass[pk] > max_basin:
            max_basin = basin_mass[pk]

    return max_basin / total_weight


# ─────────────────────────────────────────────────────────────────────────────
# STAGE B — Closed-form (AD, RD) estimation conditioned on direction
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=FASTMATH)
def estimate_AD_RD_conditioned(bvals, bvecs, sig_norm, fiber_dir,
                               f_fib, f_res, f_hin, f_wat,
                               D_res, D_hin, D_wat, use_3iso):
    """
    Stage B: closed-form weighted-least-squares estimate of (AD, RD)
    for a SINGLE dominant fiber direction, given the isotropic
    compartment fractions/centroids already estimated.

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

    RD_est = max(_TENSOR_RD_FLOOR, min(_TENSOR_RD_CEIL, -x))
    AD_est = max(_TENSOR_AD_FLOOR, min(_TENSOR_AD_CEIL, -x - y))
    if AD_est < RD_est:
        m = (AD_est + RD_est) / 2.0
        AD_est = m
        RD_est = m

    return AD_est, RD_est


# ─────────────────────────────────────────────────────────────────────────────
# STAGE C — constrained joint (VARPRO) mono-fiber fraction + tensor re-solve
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=FASTMATH)
def _stagec_scan(sig_norm, bvals, c2, iso_forward, iso_gram, iso_aty, yty,
                 ad_array, rd_array, aniso_ratio,
                 best_res, best_ad, best_rd, w_out):
    """Scan an (AD, RD) grid: for each admissible pair build the reduced Gram matrix
    of [fiber_col | iso_forward], NNLS-solve, keep the minimum-residual weights.
    Updates w_out (length 1+n_iso) with the best weights found. Helper for
    `stagec_varpro_single_fiber`."""
    N = len(bvals)
    n_iso = iso_forward.shape[1]
    ntot = 1 + n_iso
    AtA = np.zeros((ntot, ntot))
    Aty = np.empty(ntot)
    fibcol = np.empty(N)
    # iso block of the Gram / Aty is constant across (AD, RD) -- fill once.
    for k in range(n_iso):
        for j in range(n_iso):
            AtA[k + 1, j + 1] = iso_gram[k, j]
        Aty[k + 1] = iso_aty[k]
    for ia in range(ad_array.shape[0]):
        ad = ad_array[ia]
        for ir in range(rd_array.shape[0]):
            rd = rd_array[ir]
            if ad < rd * aniso_ratio:
                continue
            fib_fib = 0.0
            fib_y = 0.0
            for i in range(N):
                v = np.exp(-bvals[i] * (rd + (ad - rd) * c2[i]))
                fibcol[i] = v
                fib_fib += v * v
                fib_y += v * sig_norm[i]
            AtA[0, 0] = fib_fib
            for k in range(n_iso):
                s = 0.0
                for i in range(N):
                    s += fibcol[i] * iso_forward[i, k]
                AtA[0, k + 1] = s
                AtA[k + 1, 0] = s
            Aty[0] = fib_y
            w, _ = nnls_coordinate_descent(AtA, Aty, 0.0)
            # residual ||Aw - y||^2 = w^T AtA w - 2 w^T Aty + y^T y
            res = yty
            for a in range(ntot):
                res += -2.0 * w[a] * Aty[a]
                for b in range(ntot):
                    res += w[a] * AtA[a, b] * w[b]
            if res < best_res:
                best_res = res
                best_ad = ad
                best_rd = rd
                for a in range(ntot):
                    w_out[a] = w[a]
    return best_res, best_ad, best_rd


@njit(cache=True, fastmath=FASTMATH)
def stagec_varpro_single_fiber(sig_norm, bvals, bvecs, fiber_dir,
                               iso_forward, iso_gram, ad_grid, rd_grid,
                               aniso_ratio, w_out):
    """
    STAGE C — constrained joint (separable / VARPRO) re-solve of a SINGLE
    fiber's tensor AND all compartment fractions, given the Stage-A fiber
    direction.

    Parameters
    ----------
    sig_norm : array (N,)
        Normalised signal S/S0.
    bvals : array (N,)
    bvecs : array (N, 3)
    fiber_dir : array (3,)
        Stage-A detected fiber direction (unit vector).
    iso_forward : array (N, n_iso)
        Precomputed isotropic forward columns exp(-b * d_k) (constant across
        voxels; built once per protocol by the caller).
    iso_gram : array (n_iso, n_iso)
        iso_forward.T @ iso_forward (constant; built once).
    ad_grid, rd_grid : array
        Coarse (AD, RD) search grids.
    aniso_ratio : float
        Minimum AD/RD for an admissible fiber candidate.
    w_out : array (1 + n_iso,)
        OUTPUT: best-fit non-negative weights [fiber, iso_0..iso_{n_iso-1}].

    Returns
    -------
    best_ad, best_rd : float
        Residual-minimising fiber tensor. (Fractions are read from w_out by the
        caller: fiber_fraction = w_out[0] / sum(w_out), the iso compartments by
        binning w_out[1:] on iso-grid diffusivity thresholds.)
    """
    N = len(bvals)
    n_iso = iso_forward.shape[1]
    c2 = np.empty(N)
    for i in range(N):
        d = (bvecs[i, 0] * fiber_dir[0] + bvecs[i, 1] * fiber_dir[1]
             + bvecs[i, 2] * fiber_dir[2])
        c2[i] = d * d
    iso_aty = np.empty(n_iso)
    for k in range(n_iso):
        s = 0.0
        for i in range(N):
            s += iso_forward[i, k] * sig_norm[i]
        iso_aty[k] = s
    yty = 0.0
    for i in range(N):
        yty += sig_norm[i] * sig_norm[i]

    best_res = 1e30
    best_ad = ad_grid[0]
    best_rd = rd_grid[0]
    best_res, best_ad, best_rd = _stagec_scan(
        sig_norm, bvals, c2, iso_forward, iso_gram, iso_aty, yty,
        ad_grid, rd_grid, aniso_ratio, best_res, best_ad, best_rd, w_out)

    da = (ad_grid[1] - ad_grid[0]) if ad_grid.shape[0] > 1 else 0.1e-3
    dr = (rd_grid[1] - rd_grid[0]) if rd_grid.shape[0] > 1 else 0.1e-3
    ad_loc = np.empty(5)
    rd_loc = np.empty(5)
    for _ in range(_STAGEC_N_REFINE):
        da *= 0.5
        dr *= 0.5
        for j in range(5):
            av = best_ad + (j - 2) * da
            rv = best_rd + (j - 2) * dr
            ad_loc[j] = av if av > _TENSOR_AD_FLOOR else _TENSOR_AD_FLOOR
            rd_loc[j] = rv if rv > _TENSOR_RD_FLOOR else _TENSOR_RD_FLOOR
        best_res, best_ad, best_rd = _stagec_scan(
            sig_norm, bvals, c2, iso_forward, iso_gram, iso_aty, yty,
            ad_loc, rd_loc, aniso_ratio, best_res, best_ad, best_rd, w_out)

    return best_ad, best_rd


@njit(cache=True, fastmath=FASTMATH)
def iso_fraction_resolve(sig_norm, bvals, bvecs, fdirs, fad, frd, n_fib,
                         iso_d, w_out):
    """
    STAGE D — final constrained compartment-fraction re-solve.

    NNLS over the REDUCED, physically-anchored dictionary
    [fiber_col_1 .. fiber_col_{n_fib} | iso_col(d_1) .. iso_col(d_{n_iso})],
    where the fiber tensors (AD, RD) and directions are FIXED (from Stage A
    detection + Stage C / MRDS tensor estimation) and the isotropic
    diffusivities `iso_d` are a small FIXED set at physiological centroids
    (e.g. 0.15 / 1.0 / 3.0e-3 for RF / HF / WF). Returns the non-negative
    weights in `w_out` (length n_fib + n_iso); the caller reads fiber_fraction
    = sum(fiber weights)/sum and bins the iso weights on the THRESH_RES /
    THRESH_WAT diffusivity boundaries.

    Parameters
    ----------
    sig_norm : array (N,)
        Normalised CORRECTED signal S/S0.
    bvals, bvecs : arrays
    fdirs : array (n_fib_max, 3)
        Fiber unit directions (only the first n_fib rows are used).
    fad, frd : array (n_fib_max,)
        Fiber axial / radial diffusivities (first n_fib used).
    n_fib : int
        Number of fiber populations (0, 1, or 2).
    iso_d : array (n_iso,)
        Fixed isotropic diffusivities.
    w_out : array (>= n_fib + n_iso,)
        OUTPUT: non-negative weights [fiber_1..fiber_{n_fib}, iso_1..iso_{n_iso}].
    """
    N = len(bvals)
    n_iso = iso_d.shape[0]
    ntot = n_fib + n_iso
    A = np.empty((N, ntot))
    for k in range(n_fib):
        ad = fad[k]
        rd = frd[k]
        for i in range(N):
            d = (bvecs[i, 0] * fdirs[k, 0] + bvecs[i, 1] * fdirs[k, 1]
                 + bvecs[i, 2] * fdirs[k, 2])
            A[i, k] = np.exp(-bvals[i] * (rd + (ad - rd) * d * d))
    for j in range(n_iso):
        dj = iso_d[j]
        for i in range(N):
            A[i, n_fib + j] = np.exp(-bvals[i] * dj)
    AtA = np.zeros((ntot, ntot))
    Aty = np.zeros(ntot)
    for a in range(ntot):
        s = 0.0
        for i in range(N):
            s += A[i, a] * sig_norm[i]
        Aty[a] = s
        for b in range(a, ntot):
            v = 0.0
            for i in range(N):
                v += A[i, a] * A[i, b]
            AtA[a, b] = v
            AtA[b, a] = v
    w, _ = nnls_coordinate_descent(AtA, Aty, 0.0)
    for a in range(ntot):
        w_out[a] = w[a]


# ─────────────────────────────────────────────────────────────────────────────
# MRDS-LITE: MULTI-RESOLUTION CONE REFINEMENT OF THE STAGE A DIRECTION
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS
# -------------------
# Stage A only ever proposes a direction from its FIXED discrete Fibonacci
# hemisphere dictionary (built once per protocol, `n_dirs` directions).
# When the true fiber orientation falls between two dictionary nodes (which
# it will, for the majority of voxels, on any finite grid), Stage B's
# closed-form regression computes cos^2(angle to the WRONG axis), which
# biases the AD estimate systematically. 
#
# This module implements the idea from Coronado-Leija et al. (2017,
# Medical Image Analysis, "Multi-Resolution Discrete-Search" / MRDS)
# SCOPED DOWN to what actually matters for this toolbox's architecture:
# refining the SINGLE dominant fiber direction's angular precision. The
# parts of the original MRDS that do NOT apply here are deliberately left
# out:
#   - No multi-population model selection: this toolbox decides how
#     many populations a voxel has in `select_dominant_directions`, not by nested model
#     comparison.
#   - No iterative NNLS re-solving across resolution stages: the original
#     MRDS re-solves compartment SIZES at each resolution stage via a
#     small linear system. Here, Stage A's NNLS has ALREADY determined the
#     fractions (f_fib, f_res, f_hin, f_wat) and centroids for this voxel;
#     the refinement search only needs to re-evaluate ONE quantity (the
#     angle-dependent Stage B regression), which is far cheaper.
#
# References
# ----------
# Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Estimation
#     of individual axon bundle properties by a Multi-Resolution
#     Discrete-Search method. Medical Image Analysis, 42, 26-43.



@njit(cache=True, fastmath=FASTMATH)
def compute_cone_refinement_schedule(grid_spacing_rad, target_resolution_rad):
    """
    Derive the two-level cone search schedule (angles + candidate counts)
    from the Stage A dictionary's own measured angular spacing and a
    single target final angular resolution. No independently
    hardcoded angles or candidate counts.

    Parameters
    ----------
    grid_spacing_rad : float
        Measured nearest-neighbour angular spacing (radians) of the
        Stage A Fibonacci hemisphere dictionary actually in use for this
        protocol (compute once via `measure_hemisphere_spacing`, not a
        fixed constant).
    target_resolution_rad : float
        Desired final angular precision (radians) after two-level
        refinement. Default recommended: ~1 degree (0.01745 rad).

    Returns
    -------
    cone1_half_angle, n1, cone2_half_angle, n2 : (float, int, float, int)
    """
    cone1_half_angle = grid_spacing_rad
    if target_resolution_rad >= cone1_half_angle:
       
        return cone1_half_angle, 1, 0.0, 0

    intermediate_angle = np.sqrt(cone1_half_angle * target_resolution_rad)
    ratio1 = cone1_half_angle / intermediate_angle
    n1 = int(np.ceil(ratio1 * ratio1))
    n1 = max(n1, 4)

    cone2_half_angle = intermediate_angle
    ratio2 = cone2_half_angle / target_resolution_rad
    n2 = int(np.ceil(ratio2 * ratio2))
    n2 = max(n2, 4)

    return cone1_half_angle, n1, cone2_half_angle, n2


@njit(cache=True, fastmath=FASTMATH)
def _fibonacci_cone_point(axis0, axis1, axis2, x0, x1, x2, y0, y1, y2,
                          half_angle, i, n_points):
    """
    Compute the i-th of n_points directions in a deterministic
    solid-angle-uniform Fibonacci spiral sample of the cone of given
    half-angle around `axis`, using the orthonormal tangent basis
    (x, y, axis). Same construction principle as
    `generate_fibonacci_sphere_hemisphere` (cos(theta) uniform over the
    cone's solid angle range, golden-ratio azimuthal spacing), restricted
    to the cone instead of the full hemisphere.
    """
    cos_half = np.cos(half_angle)
    cos_theta = 1.0 - (i + 0.5) / n_points * (1.0 - cos_half)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    phi = 2.0 * np.pi * i / golden_ratio
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    d0 = cos_theta * axis0 + sin_theta * (cphi * x0 + sphi * y0)
    d1 = cos_theta * axis1 + sin_theta * (cphi * x1 + sphi * y1)
    d2 = cos_theta * axis2 + sin_theta * (cphi * x2 + sphi * y2)

    norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    return d0 / norm, d1 / norm, d2 / norm


@njit(cache=True, fastmath=FASTMATH)
def refine_fiber_direction_cone(bvals, bvecs, sig_norm, coarse_dir,
                                f_fib, f_res, f_hin, f_wat,
                                D_res, D_hin, D_wat, use_3iso,
                                cone1_half_angle, n1, cone2_half_angle, n2):
    """
    Two-level Fibonacci-cone refinement of Stage A's coarse dominant
    direction, scoring each candidate direction with Stage B's own
    closed-form regression + reconstructed SSE. 

    The isotropic fractions/centroids (f_res, f_hin, f_wat, D_res, D_hin,
    D_wat) and the fiber fraction f_fib are held FIXED at Stage A's
    values; refinement only searches over direction.

    Parameters
    ----------
    bvals, bvecs, sig_norm : arrays
        Same as `estimate_AD_RD_conditioned`.
    coarse_dir : array (3,)
        Stage A's dominant direction estimate (unit vector), the centre
        of the level-1 search cone.
    f_fib, f_res, f_hin, f_wat, D_res, D_hin, D_wat, use_3iso :
        Same as `estimate_AD_RD_conditioned`.
    cone1_half_angle, n1, cone2_half_angle, n2 : float, int, float, int
        Precomputed once per protocol via
        `compute_cone_refinement_schedule` (NOT per voxel — these depend
        only on the Stage A dictionary's spacing and the target
        resolution, both constant for the whole volume).

    Returns
    -------
    best_dir : array (3,)
        Refined direction (unit vector).
    best_AD, best_RD : float
        Refined diffusivities at best_dir (NaN if no candidate, including
        the coarse direction itself, produced a valid closed-form fit).
    """
    if abs(coarse_dir[0]) < 0.9:
        t0, t1, t2 = 1.0, 0.0, 0.0
    else:
        t0, t1, t2 = 0.0, 1.0, 0.0
    x0 = coarse_dir[1] * t2 - coarse_dir[2] * t1
    x1 = coarse_dir[2] * t0 - coarse_dir[0] * t2
    x2 = coarse_dir[0] * t1 - coarse_dir[1] * t0
    xn = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    x0 /= xn
    x1 /= xn
    x2 /= xn
    y0 = coarse_dir[1] * x2 - coarse_dir[2] * x1
    y1 = coarse_dir[2] * x0 - coarse_dir[0] * x2
    y2 = coarse_dir[0] * x1 - coarse_dir[1] * x0

    best_sse = 1e20
    best_d0, best_d1, best_d2 = coarse_dir[0], coarse_dir[1], coarse_dir[2]
    best_AD = np.nan
    best_RD = np.nan
    found = False

    # Always evaluate the coarse direction itself first, as candidate zero
    AD0, RD0 = estimate_AD_RD_conditioned(
        bvals, bvecs, sig_norm, coarse_dir,
        f_fib, f_res, f_hin, f_wat, D_res, D_hin, D_wat, use_3iso
    )
    if not np.isnan(AD0):
        sse0 = 0.0
        for i in range(len(bvals)):
            b = bvals[i]
            cos_t = (bvecs[i, 0] * coarse_dir[0] + bvecs[i, 1] * coarse_dir[1]
                     + bvecs[i, 2] * coarse_dir[2])
            D_app = RD0 + (AD0 - RD0) * cos_t * cos_t
            if use_3iso:
                s_iso = (f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
                         + f_wat * np.exp(-b * D_wat))
            else:
                s_iso = f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
            pred = f_fib * np.exp(-b * D_app) + s_iso
            diff = sig_norm[i] - pred
            sse0 += diff * diff
        best_sse = sse0
        best_d0, best_d1, best_d2 = coarse_dir[0], coarse_dir[1], coarse_dir[2]
        best_AD, best_RD = AD0, RD0
        found = True

    # ── Level 1: search the full grid-spacing cone ──────────────────────
    for k in range(n1):
        c0, c1, c2 = _fibonacci_cone_point(
            coarse_dir[0], coarse_dir[1], coarse_dir[2], x0, x1, x2, y0, y1, y2,
            cone1_half_angle, k, n1
        )
        AD_c, RD_c = estimate_AD_RD_conditioned(
            bvals, bvecs, sig_norm, np.array([c0, c1, c2]),
            f_fib, f_res, f_hin, f_wat, D_res, D_hin, D_wat, use_3iso
        )
        if np.isnan(AD_c):
            continue
        sse = 0.0
        for i in range(len(bvals)):
            b = bvals[i]
            cos_t = bvecs[i, 0] * c0 + bvecs[i, 1] * c1 + bvecs[i, 2] * c2
            D_app = RD_c + (AD_c - RD_c) * cos_t * cos_t
            if use_3iso:
                s_iso = (f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
                         + f_wat * np.exp(-b * D_wat))
            else:
                s_iso = f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
            pred = f_fib * np.exp(-b * D_app) + s_iso
            diff = sig_norm[i] - pred
            sse += diff * diff
        if sse < best_sse:
            best_sse = sse
            best_d0, best_d1, best_d2 = c0, c1, c2
            best_AD, best_RD = AD_c, RD_c
            found = True

    # ── Level 2: finer cone centred on the best level-1 candidate ───────
    if n2 > 0:
        level1_best = np.array([best_d0, best_d1, best_d2])
        if abs(level1_best[0]) < 0.9:
            t0, t1, t2 = 1.0, 0.0, 0.0
        else:
            t0, t1, t2 = 0.0, 1.0, 0.0
        x0b = level1_best[1] * t2 - level1_best[2] * t1
        x1b = level1_best[2] * t0 - level1_best[0] * t2
        x2b = level1_best[0] * t1 - level1_best[1] * t0
        xnb = np.sqrt(x0b * x0b + x1b * x1b + x2b * x2b)
        x0b /= xnb
        x1b /= xnb
        x2b /= xnb
        y0b = level1_best[1] * x2b - level1_best[2] * x1b
        y1b = level1_best[2] * x0b - level1_best[0] * x2b
        y2b = level1_best[0] * x1b - level1_best[1] * x0b

        for k in range(n2):
            c0, c1, c2 = _fibonacci_cone_point(
                level1_best[0], level1_best[1], level1_best[2],
                x0b, x1b, x2b, y0b, y1b, y2b,
                cone2_half_angle, k, n2
            )
            AD_c, RD_c = estimate_AD_RD_conditioned(
                bvals, bvecs, sig_norm, np.array([c0, c1, c2]),
                f_fib, f_res, f_hin, f_wat, D_res, D_hin, D_wat, use_3iso
            )
            if np.isnan(AD_c):
                continue
            sse = 0.0
            for i in range(len(bvals)):
                b = bvals[i]
                cos_t = bvecs[i, 0] * c0 + bvecs[i, 1] * c1 + bvecs[i, 2] * c2
                D_app = RD_c + (AD_c - RD_c) * cos_t * cos_t
                if use_3iso:
                    s_iso = (f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
                             + f_wat * np.exp(-b * D_wat))
                else:
                    s_iso = f_res * np.exp(-b * D_res) + f_hin * np.exp(-b * D_hin)
                pred = f_fib * np.exp(-b * D_app) + s_iso
                diff = sig_norm[i] - pred
                sse += diff * diff
            if sse < best_sse:
                best_sse = sse
                best_d0, best_d1, best_d2 = c0, c1, c2
                best_AD, best_RD = AD_c, RD_c
                found = True

    if not found:
        return coarse_dir, np.nan, np.nan

    return np.array([best_d0, best_d1, best_d2]), best_AD, best_RD


def measure_hemisphere_spacing(fiber_dirs):
    """
    Measure the actual nearest-neighbour angular spacing (radians) of a
    given Stage A Fibonacci hemisphere dictionary. Called ONCE per
    protocol.

    Used to derive the cone refinement schedule
    (`compute_cone_refinement_schedule`) directly from the ACTUAL
    dictionary in use.

    Parameters
    ----------
    fiber_dirs : ndarray (n_dirs, 3)

    Returns
    -------
    mean_spacing_rad : float
        Mean nearest-neighbour angular spacing across all directions.
    """
    n = len(fiber_dirs)
    dots = np.abs(fiber_dirs @ fiber_dirs.T)
    np.fill_diagonal(dots, -2.0)  # exclude self-match
    nearest_cos = np.clip(np.max(dots, axis=1), -1.0, 1.0)
    nearest_angle = np.arccos(nearest_cos)
    return float(np.mean(nearest_angle))


@njit(cache=True, fastmath=FASTMATH)
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


# ────────────────────────────────────────────────────────────────────────────
# MRDS MULTI-FIBER STAGE B — joint (AD, RD) for simultaneous populations
#
# References
# ----------
# Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Medical
#     Image Analysis, 42, 26-43.
# Levenberg K (1944); Marquardt DW (1963).
# Project synthetic validation: 2-fiber and 3-fiber joint Stage B sweep;
#     Stage A fraction-recovery-under-crossing sweep; Stage C re-fit
#     rejection (see this docstring and project records).
# ────────────────────────────────────────────────────────────────────────────


@njit(cache=True, fastmath=FASTMATH)
def _cos2_matrix(bvecs, directions):
    """cos^2 between every gradient direction and every candidate fiber
    population direction. Shape (N, n_pop)."""
    N = bvecs.shape[0]
    P = directions.shape[0]
    out = np.empty((N, P))
    for i in range(N):
        for p in range(P):
            c = (bvecs[i, 0] * directions[p, 0]
                 + bvecs[i, 1] * directions[p, 1]
                 + bvecs[i, 2] * directions[p, 2])
            out[i, p] = c * c
    return out


@njit(cache=True, fastmath=FASTMATH)
def _stageB_single_given_residual(bvals, bvecs, direction, f_fib,
                                  other_signal, sig_norm):
    """Same closed-form log-linear WLS as `estimate_AD_RD_conditioned`,
    generalised to accept an arbitrary pre-subtracted `other_signal`
    (isotropic compartments AND any other fiber populations currently
    held fixed). Single-population building block for
    `alternating_init_nfiber` ONLY -- not a replacement for
    `estimate_AD_RD_conditioned`, which remains the single-fiber (n_pop
    == 1) entry point.
    """
    N = len(bvals)
    S_fiber = np.empty(N)
    for i in range(N):
        s = (sig_norm[i] - other_signal[i]) / max(f_fib, 1e-6)
        if s < 0.01:
            s = 0.01
        elif s > 1.0:
            s = 1.0
        S_fiber[i] = s

    cos2 = np.empty(N)
    for i in range(N):
        c = (bvecs[i, 0] * direction[0] + bvecs[i, 1] * direction[1]
             + bvecs[i, 2] * direction[2])
        cos2[i] = c * c

    sum_AA = 0.0; sum_AB = 0.0; sum_BB = 0.0
    sum_Ay = 0.0; sum_By = 0.0
    for i in range(N):
        b = bvals[i]
        w = sig_norm[i] * sig_norm[i]
        log_S = np.log(S_fiber[i])
        sum_AA += w * b * b
        sum_AB += w * b * b * cos2[i]
        sum_BB += w * b * b * cos2[i] * cos2[i]
        sum_Ay += w * b * log_S
        sum_By += w * b * cos2[i] * log_S

    det = sum_AA * sum_BB - sum_AB * sum_AB
    if abs(det) < 1e-20:
        return np.nan, np.nan

    x = (sum_BB * sum_Ay - sum_AB * sum_By) / det
    y = (sum_AA * sum_By - sum_AB * sum_Ay) / det

    RD = max(_TENSOR_RD_FLOOR, min(_TENSOR_RD_CEIL, -x))
    AD = max(_TENSOR_AD_FLOOR, min(_TENSOR_AD_CEIL, -x - y))
    if AD < RD:
        m = (AD + RD) / 2.0
        AD = m; RD = m
    return AD, RD


@njit(cache=True, fastmath=FASTMATH)
def alternating_init_nfiber(bvals, bvecs, directions, fractions, iso_signal,
                            sig_norm, n_iter=3, AD0=1.6e-3, RD0=0.4e-3):
    
    """Cheap per-population initial guess for `estimate_AD_RD_nfiber_joint`. Deliberately not
    run to convergence -- its only job is to break the symmetry between
    populations before handing off to the joint LM solve.

    Parameters
    ----------
    bvals, bvecs : arrays
    directions : array (n_pop, 3)
        Stage A's detected directions.
    fractions : array (n_pop,)
        Stage A's fiber fractions for each population.
    iso_signal : array (N,)
        Precomputed isotropic-compartment contribution to the signal
        (already fraction-weighted, held FIXED throughout).
    sig_norm : array (N,)
        Normalised (S/S0) observed signal.
    n_iter : int
        Alternating iterations (default 3 -- a warm start, not a solve).
    AD0, RD0 : float
        Starting point for ALL populations before the first alternating
        pass differentiates them.

    Returns
    -------
    AD_init, RD_init : array (n_pop,)
    """
    n_pop = directions.shape[0]
    N = len(bvals)
    AD = np.full(n_pop, AD0)
    RD = np.full(n_pop, RD0)

    for _ in range(n_iter):
        for k in range(n_pop):
            other = iso_signal.copy()
            for j in range(n_pop):
                if j != k:
                    for i in range(N):
                        c = (bvecs[i, 0] * directions[j, 0]
                             + bvecs[i, 1] * directions[j, 1]
                             + bvecs[i, 2] * directions[j, 2])
                        cos2j = c * c
                        Dj = RD[j] + (AD[j] - RD[j]) * cos2j
                        other[i] += fractions[j] * np.exp(-bvals[i] * Dj)
            a, r = _stageB_single_given_residual(
                bvals, bvecs, directions[k], fractions[k], other, sig_norm
            )
            if not np.isnan(a):
                AD[k] = a
                RD[k] = r

    return AD, RD


@njit(cache=True, fastmath=FASTMATH)
def estimate_AD_RD_nfiber_joint(bvals, bvecs, sig_norm, directions, fractions,
                                iso_signal, AD_init, RD_init, max_iter=25):
    """
    Joint (AD, RD) estimation for n_pop (2 or 3) SIMULTANEOUS fiber
    populations via bounded Levenberg-Marquardt, given FIXED directions
    and fractions (from Stage A) and a fixed isotropic-compartment
    signal.

    Parameters
    ----------
    bvals, bvecs : arrays (N,), (N,3)
    sig_norm : array (N,)
    directions : array (n_pop, 3)
    fractions : array (n_pop,)
    iso_signal : array (N,)
    AD_init, RD_init : array (n_pop,)
        Population-differentiated starting point.
    max_iter : int
        Maximum LM iterations (default 25).

    Returns
    -------
    AD_out, RD_out : array (n_pop,)
        Bounded to [_TENSOR_AD_FLOOR, _TENSOR_AD_CEIL] (AD) /
        [_TENSOR_RD_FLOOR, _TENSOR_RD_CEIL] (RD), the same constants
        `estimate_AD_RD_conditioned` uses for the single-fiber case.
    """
    N = len(bvals)
    n_pop = directions.shape[0]
    n_par = 2 * n_pop

    cos2 = _cos2_matrix(bvecs, directions)

    p = np.empty(n_par)
    lb = np.empty(n_par)
    ub = np.empty(n_par)
    for k in range(n_pop):
        p[2 * k] = AD_init[k]
        p[2 * k + 1] = RD_init[k]
        lb[2 * k] = _TENSOR_AD_FLOOR
        ub[2 * k] = _TENSOR_AD_CEIL
        lb[2 * k + 1] = _TENSOR_RD_FLOOR
        ub[2 * k + 1] = _TENSOR_RD_CEIL
    for i in range(n_par):
        if p[i] < lb[i]:
            p[i] = lb[i]
        if p[i] > ub[i]:
            p[i] = ub[i]

    def _residual(pp):
        r = np.empty(N)
        for i in range(N):
            b = bvals[i]
            model = iso_signal[i]
            for k in range(n_pop):
                AD = pp[2 * k]
                RD = pp[2 * k + 1]
                D = RD + (AD - RD) * cos2[i, k]
                model += fractions[k] * np.exp(-b * D)
            r[i] = sig_norm[i] - model
        return r

    r = _residual(p)
    cost = np.sum(r * r)
    lam = 1e-3

    for _it in range(max_iter):
        J = np.empty((N, n_par))
        for i in range(N):
            b = bvals[i]
            for k in range(n_pop):
                AD = p[2 * k]
                RD = p[2 * k + 1]
                D = RD + (AD - RD) * cos2[i, k]
                e = np.exp(-b * D)
                J[i, 2 * k] = fractions[k] * e * b * cos2[i, k]
                J[i, 2 * k + 1] = fractions[k] * e * b * (1.0 - cos2[i, k])

        JTJ = J.T @ J
        JTr = J.T @ r

        improved = False
        for _try in range(10):
            A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-12)
            delta = np.linalg.solve(A, JTr)
            p_new = p - delta
            for i in range(n_par):
                if p_new[i] < lb[i]:
                    p_new[i] = lb[i]
                if p_new[i] > ub[i]:
                    p_new[i] = ub[i]
            r_new = _residual(p_new)
            cost_new = np.sum(r_new * r_new)
            if cost_new < cost:
                p = p_new
                r = r_new
                cost = cost_new
                lam = max(lam * 0.5, 1e-10)
                improved = True
                break
            else:
                lam = min(lam * 3.0, 1e10)

        if not improved:
            break

    AD_out = np.empty(n_pop)
    RD_out = np.empty(n_pop)
    for k in range(n_pop):
        AD_out[k] = p[2 * k]
        RD_out[k] = p[2 * k + 1]
    return AD_out, RD_out


@njit(cache=True, fastmath=FASTMATH)
def estimate_AD_RD_mrds(bvals, bvecs, sig_norm, directions, fractions,
                        iso_signal, init_n_iter=3, lm_max_iter=25,
                        AD0=1.6e-3, RD0=0.4e-3):
    """
    Full MRDS multi-fiber Stage B: `alternating_init_nfiber` (cheap,
    symmetry-breaking warm start) followed by
    `estimate_AD_RD_nfiber_joint`.

    Returns
    -------
    AD_out, RD_out : array (n_pop,)
    """
    AD_init, RD_init = alternating_init_nfiber(
        bvals, bvecs, directions, fractions, iso_signal, sig_norm,
        n_iter=init_n_iter, AD0=AD0, RD0=RD0
    )
    return estimate_AD_RD_nfiber_joint(
        bvals, bvecs, sig_norm, directions, fractions, iso_signal,
        AD_init, RD_init, max_iter=lm_max_iter
    )


@njit(cache=True, fastmath=FASTMATH)
def _mrds_sse(bvals, bvecs, sig_norm, directions, fractions, iso_signal, AD, RD):
    """Residual sum of squares of the multi-fiber model the LM fits."""
    sse = 0.0
    for i in range(len(bvals)):
        b = bvals[i]
        model = iso_signal[i]
        for k in range(directions.shape[0]):
            c = (bvecs[i, 0] * directions[k, 0] + bvecs[i, 1] * directions[k, 1]
                 + bvecs[i, 2] * directions[k, 2])
            model += fractions[k] * np.exp(-b * (RD[k] + (AD[k] - RD[k]) * c * c))
        d = sig_norm[i] - model
        sse += d * d
    return sse

@njit(cache=True, fastmath=FASTMATH)
def _crossing_cone_level(bvals, bvecs, sig_norm, dirs, fractions, iso_signal,
                         AD_b, RD_b, best, k, half_angle, n, min_separation_cos,
                         lm_max_iter):
    """One cone level for population k: `dirs` is updated in place, the
    (possibly new) tensors and residual are returned."""
    n_pop = dirs.shape[0]
    a0, a1, a2 = dirs[k, 0], dirs[k, 1], dirs[k, 2]
    if abs(a0) < 0.9:
        t0, t1, t2 = 1.0, 0.0, 0.0
    else:
        t0, t1, t2 = 0.0, 1.0, 0.0
    x0 = a1 * t2 - a2 * t1
    x1 = a2 * t0 - a0 * t2
    x2 = a0 * t1 - a1 * t0
    xn = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    x0 /= xn
    x1 /= xn
    x2 /= xn
    y0 = a1 * x2 - a2 * x1
    y1 = a2 * x0 - a0 * x2
    y2 = a0 * x1 - a1 * x0
    trial = dirs.copy()
    best_d0, best_d1, best_d2 = a0, a1, a2
    for q in range(n):
        c0, c1, c2 = _fibonacci_cone_point(a0, a1, a2, x0, x1, x2, y0, y1, y2,
                                           half_angle, q, n)
        ok = True
        for j in range(n_pop):
            if j != k and abs(c0 * dirs[j, 0] + c1 * dirs[j, 1] + c2 * dirs[j, 2]) > min_separation_cos:
                ok = False
        if not ok:
            continue
        trial[k, 0] = c0
        trial[k, 1] = c1
        trial[k, 2] = c2
        a_c, r_c = estimate_AD_RD_nfiber_joint(bvals, bvecs, sig_norm, trial, fractions,
                                               iso_signal, AD_b, RD_b, max_iter=lm_max_iter)
        sse = _mrds_sse(bvals, bvecs, sig_norm, trial, fractions, iso_signal, a_c, r_c)
        if sse < best:
            best = sse
            best_d0, best_d1, best_d2 = c0, c1, c2
            AD_b = a_c
            RD_b = r_c
    dirs[k, 0] = best_d0
    dirs[k, 1] = best_d1
    dirs[k, 2] = best_d2
    return AD_b, RD_b, best


@njit(cache=True, fastmath=FASTMATH)
def refine_crossing_directions_cone(bvals, bvecs, sig_norm, directions, fractions,
                                    iso_signal, AD, RD, min_separation_cos,
                                    cone1_half_angle, n1, cone2_half_angle, n2,
                                    lm_max_iter=10, n_sweeps=2):
    """Per-population two-level cone refinement of a crossing's directions.

    The populations are visited in turn, `n_sweeps` times: refining one while
    the other still sits on its grid node leaves part of the error, which the
    second sweep removes. Each level is centred on the current best direction.

    Returns (directions, AD, RD), each a new array; with no improving candidate
    they equal the inputs.
    """
    n_pop = directions.shape[0]
    dirs = directions.copy()
    AD_b = AD.copy()
    RD_b = RD.copy()
    best = _mrds_sse(bvals, bvecs, sig_norm, dirs, fractions, iso_signal, AD_b, RD_b)
    for _sweep in range(n_sweeps):
        for k in range(n_pop):
            if n1 > 0:
                AD_b, RD_b, best = _crossing_cone_level(
                    bvals, bvecs, sig_norm, dirs, fractions, iso_signal, AD_b, RD_b,
                    best, k, cone1_half_angle, n1, min_separation_cos, lm_max_iter)
            if n2 > 0:
                AD_b, RD_b, best = _crossing_cone_level(
                    bvals, bvecs, sig_norm, dirs, fractions, iso_signal, AD_b, RD_b,
                    best, k, cone2_half_angle, n2, min_separation_cos, lm_max_iter)
    return dirs, AD_b, RD_b


@njit(cache=True)
def crossing_ff_resolve(sig_norm, bvals, bvecs, fiber_dirs, directions, n_pop,
                        diff_pairs, iso_grid, ridge, iso_sig_out):
    """Returns the fiber fraction; writes the normalised iso signal to iso_sig_out."""
    N = bvals.shape[0]
    n_d = fiber_dirs.shape[0]
    n_p = diff_pairs.shape[0]
    n_iso = iso_grid.shape[0]
    n_near = 7 if n_d >= 7 else n_d
    nodes = np.full(n_pop * n_near, -1, dtype=np.int64)
    n_nodes = 0
    for k in range(n_pop):
        dots = np.empty(n_d)
        for j in range(n_d):
            dots[j] = -abs(fiber_dirs[j, 0] * directions[k, 0] + fiber_dirs[j, 1] * directions[k, 1]
                           + fiber_dirs[j, 2] * directions[k, 2])
        order = np.argsort(dots)
        for q in range(n_near):
            j = order[q]
            seen = False
            for r in range(n_nodes):
                if nodes[r] == j:
                    seen = True
                    break
            if not seen:
                nodes[n_nodes] = j
                n_nodes += 1
    na = n_nodes * n_p
    A = np.empty((N, na + n_iso))
    for r in range(n_nodes):
        j = nodes[r]
        for i in range(N):
            c = bvecs[i, 0] * fiber_dirs[j, 0] + bvecs[i, 1] * fiber_dirs[j, 1] + bvecs[i, 2] * fiber_dirs[j, 2]
            c2 = c * c
            for p in range(n_p):
                ad = diff_pairs[p, 0]
                rd = diff_pairs[p, 1]
                A[i, r * n_p + p] = np.exp(-bvals[i] * (rd + (ad - rd) * c2))
    for j in range(n_iso):
        for i in range(N):
            A[i, na + j] = np.exp(-bvals[i] * iso_grid[j])
    AtA = A.T @ A
    for a in range(na):
        AtA[a, a] += ridge
    Aty = A.T @ sig_norm
    w, _ = nnls_coordinate_descent(AtA, Aty, 0.0)
    tot = 0.0
    wa = 0.0
    for a in range(na + n_iso):
        tot += w[a]
        if a < na:
            wa += w[a]
    for i in range(N):
        iso_sig_out[i] = 0.0
    if tot <= 1e-10:
        return np.nan
    for j in range(n_iso):
        for i in range(N):
            iso_sig_out[i] += A[i, na + j] * w[na + j] / tot
    return wa / tot
