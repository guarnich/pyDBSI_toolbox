"""
DBSI Basis Functions — v3: Stage A Detection Dictionary + Stage B Inputs
===========================================================================

v3 ARCHITECTURE NOTE
----------------------
The exhaustive (direction x AD/RD-pair) dictionary built here
(`build_design_matrix_exhaustive`, `generate_exhaustive_diffusivity_pairs`)
now serves ONLY as Stage A's detection dictionary: identifying which
hemisphere directions carry fiber signal (see
`core.solvers.select_dominant_directions`). It is NOT used to extract
final AD/RD values directly — synthetic recovery validation showed that
approach (the v2 architecture) is not numerically identifiable. AD/RD
are now estimated by Stage B (`core.solvers.estimate_AD_RD_conditioned`),
a small closed-form regression conditioned on Stage A's selected
direction(s).

Because Stage A's only job is angular detection, NOT diffusivity
recovery, its (AD, RD) grid can — and for performance reasons should —
be much coarser than the v2 architecture required (e.g. 3x3 pairs
rather than 6x6 or 8x8): a coarse grid still reliably anchors which
direction(s) are active (synthetic testing showed direction cosine
similarity ~1.0 essentially independent of pair-grid density), while
keeping the NNLS solve fast (solver cost scales worse than linearly
with total column count under coordinate descent).

The rest of this module (hemisphere generation, isotropic grid/
dictionary) is otherwise unchanged from v2; see individual function
docstrings below.

Why isotropic tensors (AD=RD) cannot share the orientation loop with the
anisotropic ones
-----------------------------------------------------------------------------
For a cylindrically symmetric tensor, D_app(g) = RD + (AD-RD)*cos^2(theta).
If AD = RD, the orientation-dependent term vanishes identically and D_app
collapses to a constant D_iso regardless of direction. Looping such a pair
over M hemisphere directions would produce M numerically IDENTICAL columns —
exact rank deficiency, not just high collinearity. The NNLS solver has no
way to choose among them and either spreads the weight arbitrarily or
becomes numerically unstable. This is why the dictionary is built as two
structurally distinct blocks:

    A = [ A_aniso (rotated cylinders, AD > RD by construction)
        | A_iso   (unrotated spheres, AD = RD, one column per radius) ]

Anisotropy is enforced at construction time via `anisotropy_ratio`
(AD >= RD * ratio, ratio typically 1.1-1.2), which both prevents near-rank-
deficient near-isotropic cylinders from entering the anisotropic block and
keeps the two blocks physically and mathematically disjoint.

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


# ─────────────────────────────────────────────────────────────────────────────
# HEMISPHERE DIRECTIONS (unchanged from v1 — orientation sampling only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fibonacci_sphere_hemisphere(n_points):
    """
    Generate n_points uniformly distributed unit vectors on the upper
    hemisphere (z >= 0) using the direct Fibonacci spiral method.

    Unchanged from v1. See module history for full derivation; retained
    here verbatim because the construction (uniform-in-z, golden-ratio
    azimuth, no post-hoc filtering) is independent of the v2 parametric
    dictionary change — only the diffusivities rotated over these
    directions change in v2, not the directions themselves.

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


def generate_fibonacci_sphere(n_points):
    """Full sphere (deprecated, kept for compatibility)."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)

    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack([x, y, z]).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# v2 — EXHAUSTIVE ANISOTROPIC PARAMETER GRID  (AD, RD) PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def generate_exhaustive_diffusivity_pairs(ad_min=0.5e-3, ad_max=2.2e-3, n_ad=8,
                                          rd_min=0.05e-3, rd_max=1.2e-3, n_rd=8,
                                          anisotropy_ratio=1.1):
    """
    Generate the parametric (AD, RD) grid for the exhaustive anisotropic
    dictionary.

    Both AD and RD are swept on independent linear grids; only pairs
    satisfying the anisotropy constraint

        AD >= RD * anisotropy_ratio

    are retained. This constraint is the sole safeguard against near-
    isotropic cylinders entering the anisotropic block (see module
    docstring for why AD ~ RD pairs are mathematically pathological here).

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


@njit(cache=True, fastmath=True)
def build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid):
    """
    Build the full v2 DBSI design matrix A = [A_aniso | A_iso].

    A_aniso has one column per (direction, AD/RD-pair) combination:
    n_aniso_cols = len(fiber_dirs) * len(diff_pairs).
    A_iso has one column per isotropic ADC value (no orientation loop,
    see module docstring): n_iso_cols = len(iso_grid).

    Column ordering in A_aniso is pair-major, direction-minor:
    for p in pairs: for d in fiber_dirs: column.
    This ordering must be matched exactly by any centroid-extraction code
    that re-derives (ad, rd) from a flat column index (see
    `core.solvers.compute_aniso_centroids`).

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

    # Isotropic block: no orientation loop (spheres have no orientation).
    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            A[i, n_aniso_cols + j] = np.exp(-bvals[i] * D_iso)

    return A


# ─────────────────────────────────────────────────────────────────────────────
# v2 — ISOTROPIC SPECTRUM (explicit standalone builder)
# ─────────────────────────────────────────────────────────────────────────────

def generate_isotropic_grid(d_min=0.0, d_max=3.0e-3, n_steps=31):
    """
    Generate the isotropic ADC spectrum ("sphere radii") on a LINEAR grid.

    A classic range is [0, 3.0e-3] mm^2/s with 31 steps (~0.1e-3 mm^2/s
    increments). Note: this linear grid is the v2-document default; see
    `generate_log_uniform_isotropic_grid` for the spectrally-motivated
    alternative (Borgia et al. 1998), used together with the SVD-based
    adaptive n_iso selection in `calibration.adaptive_n_iso`. The two
    are not interchangeable — callers should be explicit about which
    grid construction they need.

    Parameters
    ----------
    d_min, d_max : float
        Isotropic ADC range, mm^2/s.
    n_steps : int
        Number of grid points.

    Returns
    -------
    iso_grid : ndarray (n_steps,), float64
    """
    iso_grid = np.linspace(d_min, d_max, n_steps)
    return np.array(iso_grid, dtype=np.float64)


def generate_log_uniform_isotropic_grid(d_min=0.1e-3, d_max=3.0e-3, n_steps=12):
    """
    Generate the isotropic ADC spectrum on a LOG-UNIFORM (geometric) grid.

    For Laplace-inversion-type problems (fitting a sum of decaying
    exponentials exp(-b*D) — exactly the isotropic DBSI block), a linear
    ADC grid over-samples the high-ADC region (where the signal decays
    fast and is noise-dominated) and under-samples the low-ADC region
    (where decay is slow and biologically critical for quantifying
    cellularity/restriction). A log-uniform grid matches the
    sensitivity profile of the exponential kernel far more evenly across
    the range (Borgia et al. 1998; Whittall & MacKay's 25-points-per-
    decade convention for multi-exponential T2/diffusion spectra).

    d_min is deliberately NOT 0.0: a log-uniform grid cannot include 0
    (undefined), and including an ADC=0 column would in any case create
    a constant column in the design matrix (exp(0)=1 for every
    measurement), which is degenerate for any non-negative least
    squares fit — see project validation notes on why the grid must not
    start at ADC=0.

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
    Generate the log-uniform isotropic ADC grid (see
    `generate_log_uniform_isotropic_grid`), but additionally force the
    grid point closest to each compartment threshold to sit EXACTLY at
    that threshold, offset by a small epsilon to the side that the
    production compartment-classification rule (adc <= threshold, see
    `model_Niso_adaptive_ff_thr.py`) assigns to the INTENDED compartment.

    WHY THIS ANCHORING MATTERS (discovered during transition-zone
    validation, see project methodological supplement
    "isotropic_compartment_supplement.docx")
    -----------------------------------------------------------------------
    Without anchoring, a plain log-uniform grid can place its nearest
    point slightly on the WRONG side of a threshold purely by geometric
    coincidence of where geomspace() happens to land — e.g. a point at
    3.0000001e-3 when the true free-water signal should be assigned to
    the >3.0e-3 (WAT) bin, but the column itself, if generated to land
    at exactly 3.0e-3 or fractionally below, would be silently
    misclassified as HIN by the `adc <= THRESH_WAT` rule. This was
    caught as an artifact in early testing where it produced an
    apparent "complete free-water detection failure" that was purely a
    boundary-classification bug, not a genuine model limitation.

    Anchoring does NOT remove the separate, genuine transition-zone
    bias documented in the methodological supplement (Section 4): a
    true diffusivity near a threshold remains intrinsically hard to
    assign correctly regardless of grid anchoring, because the SIGNAL
    ITSELF is ambiguous there, not just the column's nominal position.
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

    # ── ANCHORING (AUGMENT, not relocate) ──────────────────────────────────
    # Historical bug: the previous implementation RELOCATED the grid point
    # closest to each threshold onto that threshold. At small n_steps over a
    # wide (extended) range, both interior log-uniform points get consumed by
    # the two anchors, leaving a coverage HOLE across the whole 0.3-3.0e-3
    # (hindered) band. An isotropic signal there (e.g. D~0.8e-3) then has NO
    # nearby atom and the NNLS dumps it onto the fiber columns -> the fiber
    # fraction leaks to ~1.0 with near-zero isotropic fractions. (Reproduced
    # on the synthetic crossing-fiber scenario; this is the "isotropic-grid
    # coverage gap" flagged in select_lambda_aniso_discrepancy's docstring.)
    #
    # Fix: ADD the threshold anchors as extra columns without displacing the
    # interior log-uniform points, so classification still gets an exact-
    # boundary column while band coverage is preserved.
    for thresh, off in ((thresh_res, -epsilon), (thresh_wat, +epsilon)):
        # Use <= / >= (not strict <) so a threshold coinciding with the
        # grid's own d_min/d_max boundary is still anchored — the anchor
        # offset (not distance from the edge) sets the compartment.
        if d_min <= thresh <= d_max:
            anc = thresh + off  # RES anchor sits just below thresh_res
                                # (adc<=thresh_res -> RES); WAT anchor just
                                # above thresh_wat (adc>thresh_wat -> WAT).
            if not any(abs(g - anc) < epsilon for g in grid):
                grid.append(anc)

    # ── COVERAGE FLOOR (resolution guarantee) ──────────────────────────────
    # Guarantee enough resolution that any physiological isotropic D has a
    # nearby atom: no geometric gap may exceed `max_ratio`. Because exp(-b*D)
    # is nonlinear in D, two atoms merely *bracketing* a true D is not enough
    # (a convex mix of exp(-b*D1), exp(-b*D2) fits worse than the fiber block,
    # so the signal still leaks); the neighbouring atom must be geometrically
    # CLOSE. Fill oversized gaps with geometric-midpoint atoms until every
    # adjacent ratio is <= max_ratio. (max_ratio=2.0 ~ 3 pts/octave: enough to
    # localize the iso peak without over-parameterizing.)
    while True:
        g = np.sort(np.array(grid, dtype=np.float64))
        ratios = g[1:] / g[:-1]
        if len(ratios) == 0 or ratios.max() <= max_ratio:
            break
        j = int(np.argmax(ratios))
        grid.append(float(np.sqrt(g[j] * g[j + 1])))

    return np.array(np.unique(grid), dtype=np.float64)


@njit(cache=True, fastmath=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# v1 LEGACY — single-(AD,RD) design matrix (DEPRECATED)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid, ad=1.5e-3, rd=0.5e-3):
    """
    .. deprecated:: 2.0.0 (superseded by v3 hybrid two-stage architecture)
        This is the v1 single-(AD,RD) orientation-only anisotropic
        dictionary. It samples ONLY the orientation space at a single
        fixed (AD, RD), which under-constrains DBSI's ability to explain
        pathological microstructural heterogeneity (axonal injury / RD
        changes look identical to a healthy fiber reoriented, under this
        construction).

        v3 uses `build_design_matrix_exhaustive` for Stage A (direction
        detection only) and `core.solvers.estimate_AD_RD_conditioned`
        for Stage B (closed-form diffusivity estimation conditioned on
        the detected direction) — see `model_Niso_adaptive_ff_thr.py`
        module docstring for the full v3 rationale.

        Kept for backward compatibility with external code and for
        regression comparisons against the v1 pipeline. Not used by
        `DBSI_Adaptive` in v2 or v3.

    Parameters
    ----------
    bvals : array (N,)
    bvecs : array (N, 3)
    fiber_dirs : array (M, 3)
    iso_grid : array (L,)
    ad, rd : float
        Fixed axial/radial diffusivity for every anisotropic column.

    Returns
    -------
    A : array (N, M+L)
    """
    n_meas = len(bvals)
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)

    A = np.zeros((n_meas, n_dirs + n_iso), dtype=np.float64)

    for j in range(n_dirs):
        fdir = fiber_dirs[j]
        for i in range(n_meas):
            b = bvals[i]
            g = bvecs[i]
            cos_t = g[0]*fdir[0] + g[1]*fdir[1] + g[2]*fdir[2]
            D_app = rd + (ad - rd) * cos_t * cos_t
            A[i, j] = np.exp(-b * D_app)

    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            A[i, n_dirs + j] = np.exp(-bvals[i] * D_iso)

    return A
