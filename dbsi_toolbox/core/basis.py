"""
DBSI Basis Functions: Stage A Detection Dictionary + Stage B Inputs
===========================================================================

The exhaustive (direction x AD/RD-pair) dictionary built here
(`build_design_matrix_exhaustive`, `generate_exhaustive_diffusivity_pairs`)
now serves ONLY as Stage A's detection dictionary: identifying which
hemisphere directions carry fiber signal (see
`core.solvers.select_dominant_directions`). AD/RD
are now estimated by Stage B (`core.solvers.estimate_AD_RD_conditioned`),
a small closed-form regression conditioned on Stage A's selected
direction(s).

Because Stage A's only job is angular detection, NOT diffusivity
recovery, its (AD, RD) grid can 
be a coarse grid that still reliably anchors to the active
direction(s) while keeping the NNLS solve fast.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601.
Discussion grounded in feedback from Alonso Ramirez-Manzanares on the
orientation-space vs. parameter-space sampling distinction (see
toolbox_v2.md design document) and the subsequent v3 hybrid redesign
motivated by synthetic recovery validation failures of the v2 single-
stage approach.
"""

import numpy as np
from numba import njit
from .._numba_flags import FASTMATH


# ─────────────────────────────────────────────────────────────────────────────
# HEMISPHERE DIRECTIONS (orientation sampling only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fibonacci_sphere_hemisphere(n_points):
    """
    Generate n_points uniformly distributed unit vectors on the upper
    hemisphere (z >= 0) using the direct Fibonacci spiral method.


    Parameters
    ----------
    n_points : int
        Number of hemisphere directions.

    Returns
    -------
    dirs : ndarray (n_points, 3), float64
        Unit vectors on the upper hemisphere (z >= 0).
    """
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(n_points, dtype=np.float64)

    z = 1.0 - (indices + 0.5) / n_points
    r = np.sqrt(1.0 - z * z)

    theta = 2.0 * np.pi * indices / golden_ratio

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    dirs = np.column_stack([x, y, z])

    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms

    return dirs.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# EXHAUSTIVE ANISOTROPIC PARAMETER GRID: (AD, RD) PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def generate_exhaustive_diffusivity_pairs(ad_min=0.5e-3, ad_max=2.2e-3, n_ad=3,
                                          rd_min=0.05e-3, rd_max=1.2e-3, n_rd=3,
                                          anisotropy_ratio=2.0):
    """
    DEFAULTS ALIGNED TO THE MODEL: `anisotropy_ratio=2.0` was
    introduced to remove, because it leaks isotropic signal into
    fiber_fraction. Now a bare call reproduces the model: 3x3, ratio 2.0, 5 admissible pairs.

    Generate the parametric (AD, RD) grid for the exhaustive anisotropic
    dictionary.

    Both AD and RD are swept on independent linear grids; only pairs
    satisfying the anisotropy constraint

        AD >= RD * anisotropy_ratio

    are retained. This constraint is the sole safeguard against near-
    isotropic cylinders entering the anisotropic block.

    Parameters
    ----------
    ad_min, ad_max : float
        Axial diffusivity range, mm^2/s.
    n_ad : int
        Number of AD steps.
    rd_min, rd_max : float
        Radial diffusivity range, mm^2/s.
    n_rd : int
        Number of RD steps.
    anisotropy_ratio : float
        Minimum AD/RD ratio for a pair to be physically admissible as an
        anisotropic basis element. Typical range 1.1-1.2 (see
        `autoconfigure_dictionary`).

    Returns
    -------
    pairs : ndarray (n_pairs, 2), float64
        Each row is (AD, RD). n_pairs <= n_ad * n_rd (filtered).
    """
    ad_grid = np.linspace(ad_min, ad_max, n_ad)
    rd_grid = np.linspace(rd_min, rd_max, n_rd)

    pairs = []
    for ad in ad_grid:
        for rd in rd_grid:
            if ad >= rd * anisotropy_ratio:
                pairs.append([ad, rd])

    if len(pairs) == 0:
        raise ValueError(
            "generate_exhaustive_diffusivity_pairs: no (AD, RD) pair "
            "satisfies the anisotropy_ratio constraint. Check ad_min/ad_max, "
            "rd_min/rd_max, and anisotropy_ratio for consistency "
            f"(got ad_max={ad_max:.2e}, rd_min={rd_min:.2e}, "
            f"ratio={anisotropy_ratio})."
        )

    return np.array(pairs, dtype=np.float64)


@njit(cache=True, fastmath=FASTMATH)
def build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid):
    """
    Build the full DBSI design matrix A = [A_aniso | A_iso].

    A_aniso has one column per (direction, AD/RD-pair) combination:
    n_aniso_cols = len(fiber_dirs) * len(diff_pairs).
    A_iso has one column per isotropic ADC value: n_iso_cols = len(iso_grid).

    Parameters
    ----------
    bvals : array (N,)
        B-values, s/mm^2.
    bvecs : array (N, 3)
        Normalized gradient directions.
    fiber_dirs : array (M, 3)
        Hemisphere directions.
    diff_pairs : array (P, 2)
        (AD, RD) pairs from `generate_exhaustive_diffusivity_pairs`.
    iso_grid : array (K,)
        Isotropic ADC values, mm^2/s.

    Returns
    -------
    A : array (N, M*P + K), float64
        Design matrix. First M*P columns anisotropic, last K isotropic.
    """
    n_meas = len(bvals)
    n_dirs = len(fiber_dirs)
    n_pairs = len(diff_pairs)
    n_iso = len(iso_grid)

    n_aniso_cols = n_dirs * n_pairs
    A = np.zeros((n_meas, n_aniso_cols + n_iso), dtype=np.float64)

    # Anisotropic block: pair-major, direction-minor ordering.
    col_idx = 0
    for p in range(n_pairs):
        ad = diff_pairs[p, 0]
        rd = diff_pairs[p, 1]
        for d in range(n_dirs):
            fdir = fiber_dirs[d]
            for i in range(n_meas):
                b = bvals[i]
                g = bvecs[i]
                cos_t = g[0]*fdir[0] + g[1]*fdir[1] + g[2]*fdir[2]
                D_app = rd + (ad - rd) * cos_t * cos_t
                A[i, col_idx] = np.exp(-b * D_app)
            col_idx += 1

    # Isotropic block.
    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            A[i, n_aniso_cols + j] = np.exp(-bvals[i] * D_iso)

    return A


# ─────────────────────────────────────────────────────────────────────────────
# ISOTROPIC GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def generate_log_uniform_isotropic_grid(d_min=0.1e-3, d_max=3.0e-3, n_steps=12):
    """
    Generate the isotropic ADC spectrum on a LOG-UNIFORM (geometric) grid.

    A log-uniform grid matches the sensitivity profile of the exponential kernel far more evenly across
    the range (Borgia et al. 1998; Whittall & MacKay's 25-points-per-
    decade convention for multi-exponential T2/diffusion spectra).

    Parameters
    ----------
    d_min, d_max : float
        Isotropic ADC range, mm^2/s. Default (0.1e-3, 3.0e-3) matches
        the range used by the SVD-based adaptive n_iso selection in
        `calibration.adaptive_n_iso`.
    n_steps : int
        Number of grid points. Typically set by
        `calibration.adaptive_n_iso.select_n_iso_svd` rather than left
        at the default here.

    Returns
    -------
    iso_grid : ndarray (n_steps,), float64
    """
    iso_grid = np.geomspace(d_min, d_max, n_steps)
    return np.array(iso_grid, dtype=np.float64)


def generate_anchored_isotropic_grid(d_min=0.1e-3, d_max=3.0e-3, n_steps=12,
                                     thresh_res=0.3e-3, thresh_wat=3.0e-3,
                                     epsilon=1e-6, max_ratio=2.0):
    """
    Generate the log-uniform isotropic ADC grid, but additionally force the
    grid point closest to each compartment threshold to sit EXACTLY at
    that threshold, offset by a small epsilon to the side that the
    production compartment-classification rule assigns to the INTENDED compartment.
  
    Anchoring does NOT remove the separate, genuine transition-zone
    bias documented in the methodological supplement.
    Anchoring only ensures the grid's own bookkeeping does not ADD a
    spurious, avoidable misclassification on top of that genuine
    physical ambiguity.

    Parameters
    ----------
    d_min, d_max : float
        Isotropic ADC range, mm^2/s.
    n_steps : int
        Number of grid points.
    thresh_res, thresh_wat : float
        Compartment boundaries to anchor to (must match the thresholds
        used by the actual fitting/classification code).
    epsilon : float
        Offset magnitude, mm^2/s. Default 1e-6 (0.001x10^-3) is far
        below any physiologically meaningful ADC difference and well
        below typical grid spacing, so it does not materially change
        the column's signal decay curve, while reliably placing it on
        the correct side of strict/non-strict threshold comparisons.
    max_ratio : float
        Maximum allowed geometric ratio between adjacent grid atoms.
        Oversized gaps are filled with geometric-midpoint atoms until
        every adjacent ratio is <= max_ratio, guaranteeing that any
        physiological isotropic D has a geometrically CLOSE atom (see
        "COVERAGE FLOOR" below). Set to np.inf to disable the fill.

    Returns
    -------
    iso_grid : ndarray, float64
        Sorted, unique. NOTE: length is generally GREATER than n_steps —
        anchoring adds up to two boundary atoms and the coverage floor
        adds as many midpoints as needed to satisfy max_ratio. Callers
        must read len(iso_grid) dynamically and must not assume the
        result has exactly n_steps entries.
    """
    grid = list(generate_log_uniform_isotropic_grid(
        d_min=d_min, d_max=d_max, n_steps=n_steps))


    for thresh, off in ((thresh_res, -epsilon), (thresh_wat, +epsilon)):
        
        if d_min <= thresh <= d_max:
            anc = thresh + off 
            if not any(abs(g - anc) < epsilon for g in grid):
                grid.append(anc)


    while True:
        g = np.sort(np.array(grid, dtype=np.float64))
        ratios = g[1:] / g[:-1]
        if len(ratios) == 0 or ratios.max() <= max_ratio:
            break
        j = int(np.argmax(ratios))
        grid.append(float(np.sqrt(g[j] * g[j + 1])))

    return np.array(np.unique(grid), dtype=np.float64)


@njit(cache=True, fastmath=FASTMATH)
def build_isotropic_dictionary(bvals, iso_grid):
    """
    Build the isotropic block of the design matrix in isolation.

    No orientation loop: each column is exp(-b * D_iso), independent of
    gradient direction, since AD = RD for every basis element here.

    Parameters
    ----------
    bvals : array (N,)
        B-values, s/mm^2.
    iso_grid : array (K,)
        Isotropic ADC values, mm^2/s.

    Returns
    -------
    A_iso : array (N, K), float64
    """
    n_meas = len(bvals)
    n_iso = len(iso_grid)

    A_iso = np.zeros((n_meas, n_iso), dtype=np.float64)

    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            b = bvals[i]
            A_iso[i, j] = np.exp(-b * D_iso)

    return A_iso
