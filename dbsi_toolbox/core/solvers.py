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
#
# WHY LOCAL-MAXIMA PEAK-FINDING, NOT GLOBAL TOP-K BY WEIGHT
# -----------------------------------------------------------------------
# The original selection rule (ranking all n_dirs candidates globally by
# aggregated weight and keeping the top `max_directions`) has a
# structural false-positive failure mode, confirmed by project synthetic
# testing: for a TRUE single fiber, the NNLS solution's weight on the
# fixed Fibonacci grid routinely SMEARS across the true direction's
# immediate geometric neighbours (the true orientation almost never
# falls exactly on a grid node). The mean angular separation between the
# two highest-weight columns in single-fiber trials (~26 deg, measured)
# coincides almost exactly with the dictionary's own mean nearest-
# neighbour spacing (~25 deg) -- i.e. the "second population" the old
# rule reported was not a second fiber, it was the true fiber's own
# quantisation shadow on an adjacent grid node. Because global top-K only
# asks "is this among the 2 heaviest columns overall", it cannot tell
# that smearing apart from a genuine second peak.
#
# The fix: a candidate direction is only counted as a detected population
# if its aggregated weight exceeds the weight of EVERY one of its k
# nearest geometric neighbours in the SAME dictionary (a local maximum on
# the weight-over-sphere function), not merely high in the global
# ranking. A true fiber's smeared neighbour is, by construction, always
# lower-weighted than the true peak itself, so it is never a local
# maximum and is correctly excluded. A genuine second, angularly
# well-separated fiber population has its own local neighbourhood and
# remains a local maximum independently.
#
# RE-VALIDATED 2026-07-15 (this implementation, synthetic 3-shell
# Verona-like protocol, n_dirs=62, SNR=30, 30 seeds/condition; see
# project re-validation sweep for the script): single-fiber correct
# N_POP=1 rate rose from 0.0% (global top-K, confirmed on this same
# synthetic setup) to 73.3% at k=6 (the shipped default -- see
# `_DEFAULT_DIRECTION_PEAK_K` in model_Niso_adaptive_ff_thr.py for the
# k-sweep that selected 6 over the originally proposed 8). True-crossing
# sensitivity is angle-dependent, NOT uniformly 95-100% as an earlier,
# smaller-scale check suggested: 46.7% at 30 deg, 93.3% at 60 deg, 96.7%
# at 90 deg (k=6). The 30-degree regime is intrinsically hard at this
# dictionary density (mean node spacing ~25 deg -- a 30-degree crossing
# separates the two true peaks by barely more than one grid spacing, so
# their k-nearest-neighbourhoods overlap almost by construction) and is
# NOT much improved by retuning k alone (45-65% across k=4..12 tested);
# closing that gap requires a denser Stage A dictionary (larger n_dirs)
# for protocols where sub-45-degree crossings are expected to matter.
# Numbers above are specific to this synthetic setup (fraction splits,
# SNR, regularisation) and should be re-checked against your own
# protocol/tissue assumptions before relying on them clinically.
#
# NOTE ON REMAINING LIMITATION: local-maxima peak-finding fixes DETECTION
# (how many populations, which grid nodes) but not localisation -- the
# reported direction(s) are still raw grid nodes, accurate only to the
# dictionary's own angular resolution. For n_pop==1 this is separately
# addressed by the MRDS-lite cone refinement (`refine_fiber_direction_cone`).
# For n_pop>=2, per-population cone refinement is not implemented in this
# release (see MRDS multi-fiber module docstring further below).

def build_direction_neighbor_graph(fiber_dirs, k=6):
    """
    Precompute, once per protocol (NOT per voxel), the indices of each
    Stage A hemisphere direction's k angularly-nearest neighbours within
    the SAME dictionary. Feeds `select_dominant_directions`'s
    local-maxima peak-finding criterion.

    Plain NumPy (not Numba) -- called a handful of times per `fit()`
    call, exactly like `measure_hemisphere_spacing`, not per voxel.

    Parameters
    ----------
    fiber_dirs : ndarray (n_dirs, 3)
        Stage A's hemisphere direction dictionary.
    k : int
        Number of nearest neighbours per direction. Default 6 -- see the
        k-sweep in module notes above `select_dominant_directions` (k=6
        matched or beat k in {4,8,10,12} on both single-fiber correctness
        and crossing sensitivity in re-validation).

    Returns
    -------
    neighbor_idx : ndarray (n_dirs, k_eff), int64
        Row d holds the indices of direction d's k_eff nearest
        neighbours (by raw dot product, consistent with
        `measure_hemisphere_spacing`'s convention -- the hemisphere
        generator already resolves the +/- direction sign ambiguity, so
        no abs() is needed here). k_eff = min(k, n_dirs - 1) to stay
        well-defined for very small dictionaries.
    """
    n = len(fiber_dirs)
    k_eff = max(1, min(k, n - 1))
    dots = fiber_dirs @ fiber_dirs.T
    np.fill_diagonal(dots, -2.0)
    neighbor_idx = np.argsort(-dots, axis=1)[:, :k_eff].astype(np.int64)
    return np.ascontiguousarray(neighbor_idx)


@njit(cache=True, fastmath=True)
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
    only job is angular detection — see module docstring on why the
    per-pair centroid from Stage A is not trusted.

    Detection is by LOCAL-MAXIMA peak-finding over the dictionary's
    k-nearest-neighbour graph (`build_direction_neighbor_graph`), then
    each local-maximum peak is scored by its ANGULAR-BASIN MASS (Voronoi
    assignment: every direction's weight is added to the nearest peak's
    basin), and peaks are accepted greedily in descending basin mass with
    angular NON-MAXIMUM SUPPRESSION (a candidate within `min_separation`
    of an already-accepted, stronger peak is suppressed as smearing skirt,
    not a distinct population).

    WHY BASIN MASS, NOT PEAK HEIGHT (fixed 2026-07-21) -------------------
    The regularized Stage A NNLS + the (direction x AD/RD-pair) dictionary
    SMEAR a single fiber's weight across several neighbouring columns.
    Thresholding on a single direction's peak HEIGHT then rejects a
    genuine but spread SECOND fiber (esp. a low-FA / demyelinated one):
    on the tumor-like complex voxel the true 2nd fiber's peak height was
    only ~3% of total (below min_weight_fraction=5%) while its angular
    BASIN MASS was ~20% — clearly a real population. Thresholding on basin
    mass recovers it. The NMS separation criterion keeps the original
    protection against grid-quantisation smearing producing false N_POP>=2
    (skirt peaks near the true one are suppressed), replacing the previous
    reliance on peak-height alone.

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
        skirt that noise pushed past `min_separation` (carries less). Set
        from an empirical single-fiber-vs-crossing ratio sweep: the two
        distributions overlap, so 0.35 trades ~7% false crossings on single
        fibers for recovering the true 2nd fiber in ~40-80% of crossing
        voxels (SNR 30-20). Default 0.35.
    min_dominant_concentration : float
        Minimum share of the TOTAL anisotropic weight that the dominant
        basin must hold for ANY fiber population to be reported. A real
        fiber concentrates its anisotropic weight along one direction
        (dominant basin ~0.4-0.6 of total); spurious anisotropic weight
        from isotropic (esp. restricted) leakage on a fiber-FREE voxel is
        DIFFUSE (dominant basin only ~0.2-0.24, spread across many similar
        peaks) and would otherwise be mis-read as a crossing (npop>=2,
        since every diffuse basin passes the relative test). Below this
        concentration the whole anisotropic block is treated as leakage and
        NO population is returned (npop=0). Set from an empirical
        pure-iso-vs-fiber concentration sweep (SNR30: clean gap pure-iso
        p95~0.40 vs real-fiber p05~0.39; SNR15: overlap). 0.35 balances a
        ~89% cut in pure-iso false crossings against keeping low-SNR fiber
        detection. Default 0.35.

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
    # nearest neighbours'. Ties break toward the lower index (well-defined
    # at exact zero). NOTE: the min-mass filter is applied LATER on basin
    # mass, not here on peak height (a spread fiber's peak can be tiny).
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
        # mass threshold, every remaining peak is too -> stop.
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


@njit(cache=True, fastmath=True)
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
    lambda_aniso adaptation. It is factored out here (rather than returned
    from `select_dominant_directions`) so it can be computed even for voxels
    that never enter the direction-selection path (e.g. below fiber_threshold)
    and without perturbing that function's tuned return signature.

    Interpretation (the discriminator Plan A rests on):
      - A REAL fiber -- even a demyelinated / low-FA one -- concentrates its
        anisotropic weight along one direction: high concentration (~0.4-0.6).
      - DIFFUSE anisotropic weight absorbed from isotropic (esp. restricted)
        signal on a fiber-free voxel spreads across many near-equal basins:
        low concentration (~0.2-0.25).
    Unlike n_pop (a binary/quantised detection count), concentration separates
    a weak-but-real fiber from leakage continuously, which is why it is the
    right lever for a continuous per-voxel regularization (npop-gating was
    falsified: it crushed the demyelinated fiber, confusing it with GM).

    The peak-finding, basin-assignment and column-ordering conventions are
    IDENTICAL to `select_dominant_directions` (pair-major, direction-minor;
    local-maxima over the k-NN graph; Voronoi basin mass). Returns 0.0 when
    the anisotropic block carries negligible weight or has no local maximum.

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
# STAGE C — constrained joint (VARPRO) mono-fiber fraction + tensor re-solve
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _stagec_scan(sig_norm, bvals, c2, iso_forward, iso_gram, iso_aty, yty,
                 ad_array, rd_array, aniso_ratio,
                 best_res, best_ad, best_rd, w_out):
    """Scan an (AD, RD) grid: for each admissible pair build the reduced Gram
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


@njit(cache=True, fastmath=True)
def stagec_varpro_single_fiber(sig_norm, bvals, bvecs, fiber_dir,
                               iso_forward, iso_gram, ad_grid, rd_grid,
                               aniso_ratio, w_out):
    """
    STAGE C — constrained joint (separable / VARPRO) re-solve of a SINGLE
    fiber's tensor AND all compartment fractions, given the Stage-A fiber
    direction.

    WHY THIS EXISTS
    ---------------
    The raw v3 pipeline estimates the compartment fractions (Stage A NNLS over
    the OVER-COMPLETE anisotropic dictionary: fiber_fraction = sum of all ~468
    anisotropic weights) and the fiber tensor (Stage B closed-form given those
    fractions) in two DECOUPLED steps. On a concentrated single fiber that
    coexists with a restricted isotropic compartment the two are MUTUALLY
    biased: the many near-fiber anisotropic columns absorb restricted/hindered
    iso signal -> fiber_fraction inflated and restricted_fraction driven to ~0;
    the closed-form Stage B, fed those biased fractions, then UNDER-estimates AD
    (~1.1 vs a true 1.7), which makes the fiber column blunter/more isotropic
    and locks the inflation in. Synthetic single-fiber validation (FF_true 0.55,
    RF_true 0.10, AD 1.7, RD swept, SNR 30/50): the raw pipeline gave FF up to
    0.94, RF 0.00, AD ~1.1; this joint re-solve recovers FF 0.55-0.57, RF
    0.07-0.10, AD 1.64-1.85, RD within noise -- across all RD.

    The fix is to fit the fiber tensor and fractions JOINTLY by directly
    minimising the reconstruction residual. Because the fractions enter the
    forward model LINEARLY and only (AD, RD) enter non-linearly (the direction
    is fixed to Stage A's detection), this is a 2-D separable least squares
    (VARPRO): scan (AD, RD) on a coarse grid, and for each candidate solve the
    NON-NEGATIVE linear problem for [fiber_weight | iso_weights] on the REDUCED
    dictionary [fiber_col(AD,RD,dir) | iso_grid] -- ONE physically-anchored
    fiber column competing fairly with the isotropic columns, instead of the
    over-complete anisotropic block. A short local refine (half-spacing, 5x5)
    around the best grid node sharpens (AD, RD). An earlier ALTERNATING scheme
    (fractions <-> closed-form Stage B tensor) was tried and FALSIFIED: it
    inherits the closed-form's downward AD bias and converges to the same wrong
    fixed point (AD stuck ~1.0, RF ~0.01). Only the residual-minimising search
    recovers the truth, hence VARPRO.

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

    # Local refine: 5x5 at half the coarse spacing around the best node.
    da = (ad_grid[1] - ad_grid[0]) if ad_grid.shape[0] > 1 else 0.1e-3
    dr = (rd_grid[1] - rd_grid[0]) if rd_grid.shape[0] > 1 else 0.1e-3
    ad_loc = np.empty(5)
    rd_loc = np.empty(5)
    for j in range(5):
        av = best_ad + (j - 2) * 0.5 * da
        rv = best_rd + (j - 2) * 0.5 * dr
        ad_loc[j] = av if av > 0.2e-3 else 0.2e-3
        rd_loc[j] = rv if rv > 0.02e-3 else 0.02e-3
    best_res, best_ad, best_rd = _stagec_scan(
        sig_norm, bvals, c2, iso_forward, iso_gram, iso_aty, yty,
        ad_loc, rd_loc, aniso_ratio, best_res, best_ad, best_rd, w_out)

    return best_ad, best_rd


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
# biases the AD estimate systematically. Project validation (synthetic,
# Verona-protocol reconstruction, 25-30 trial pairs per condition) measured
# this directly: at n_dirs=62 (the protocol's autoconfigured value), median
# AD relative error was 6.8-7.0% using the raw grid direction, vs 2.7% for
# an oracle fit at the TRUE direction — i.e. roughly 60% of the total AD
# error at this n_dirs was attributable to angular grid quantisation alone,
# not noise. The gap closed monotonically with n_dirs (down to ~0.3
# percentage points at n_dirs=250) but only at a steep, unnecessary
# computational cost (Stage A's design matrix and NNLS solve scale with
# n_dirs x n_pairs columns).
#
# This module implements the idea from Coronado-Leija et al. (2017,
# Medical Image Analysis, "Multi-Resolution Discrete-Search" / MRDS)
# SCOPED DOWN to what actually matters for this toolbox's architecture:
# refining the SINGLE dominant fiber direction's angular precision. The
# parts of the original MRDS that do NOT apply here are deliberately left
# out (see rationale recorded in project discussion):
#   - No F-test / multi-population model selection: this toolbox currently
#     saves only the dominant direction in its output channels regardless
#     of how many populations Stage A detects, so refining a second
#     population's orientation would have no effect on any output.
#   - No iterative NNLS re-solving across resolution stages: the original
#     MRDS re-solves compartment SIZES at each resolution stage via a
#     small linear system. Here, Stage A's NNLS has ALREADY determined the
#     fractions (f_fib, f_res, f_hin, f_wat) and centroids for this voxel;
#     the refinement search only needs to re-evaluate ONE quantity (the
#     angle-dependent Stage B regression), which is far cheaper.
#   - No spatial regularisation / Simultaneous Denoising and Fitting: out
#     of scope for this addition; voxels remain fit independently.
#
# VALIDATION SUMMARY (project synthetic testing, Verona-like protocol,
# n_dirs=62, single dominant fiber, SNR=28)
# -----------------------------------------------------------------------
#   Grid direction only (current baseline):      AD err ~6.8-7.0%, RD err ~4-5%
#   Two-level cone refinement (this module):      AD err ~2.6-2.8%, RD err ~4.0-4.3%
#   Oracle (true direction, upper bound):          AD err ~2.7%,    RD err ~3.6-4.6%
# A further test at a DELIBERATELY SMALLER n_dirs=30 (half the Verona
# autoconfigured value, i.e. a much cheaper Stage A dictionary) plus this
# refinement reached AD err ~2.8% -- BETTER than the current n_dirs=62
# WITHOUT refinement (6.8-7.0%), at roughly half the anisotropic Stage A
# NNLS cost. This means the refinement, once enabled, allows the coarse
# n_dirs autoconfiguration heuristic (`utils/autoconfig.py`,
# max_dirs_per_shell x 1.3) to be treated purely as a DETECTION-resolution
# starting point (does this cone land near the right general orientation
# and avoid confusing nearby crossing populations?), not as the source of
# final angular precision -- that responsibility now belongs to this
# module. The 1.3x heuristic itself is NOT removed by this change (it
# remains a reasonable, literature-unconnected starting point per project
# discussion) but its role in the pipeline is now qualitatively different.
#
# NOT YET VALIDATED: real (non-synthetic) HCP/Verona data; multi-fiber
# crossing configurations (only single dominant fiber tested); the
# interaction between refinement and the isotropic block's own centroid
# estimates (refinement here holds Stage A's isotropic fractions/centroids
# fixed, exactly as Stage B already does).
#
# DATA-DRIVEN PARAMETERISATION (no fixed "magic number" cone angles or
# candidate counts)
# -----------------------------------------------------------------------
# The search cone's angular scale is tied directly to the ACTUAL measured
# nearest-neighbour spacing of the specific Stage A Fibonacci dictionary in
# use (computed once per protocol from `fiber_dirs`, not assumed) rather
# than a hardcoded degree value -- this ties the refinement's search
# radius to the dictionary's own known blind-spot size, so a denser or
# sparser n_dirs automatically gets a correspondingly narrower or wider
# initial search cone.
#
# The number of candidates per resolution level, and the second level's
# cone angle, are BOTH derived from a single user-facing quantity,
# `target_resolution_rad` (default corresponds to ~1 degree): assuming a
# geometric ("equal ratio") zoom across the two levels balances the
# angular reduction evenly,
#
#     intermediate_angle = sqrt(cone1_half_angle * target_resolution)
#     n1 = ceil( (cone1_half_angle / intermediate_angle)^2 )
#     cone2_half_angle   = intermediate_angle
#     n2 = ceil( (cone2_half_angle   / target_resolution)^2 )
#
# (the squared ratio reflects solid-angle, not linear-angle, coverage --
# doubling the linear angular ratio requires ~4x the candidates to keep
# the same point density over a 2D cone cross-section). This reproduces,
# almost exactly, the candidate counts (~20-25 per level) that were
# empirically found to close the AD accuracy gap to within noise of the
# oracle in project validation -- i.e. the formula is not an independent
# untested guess, it was checked against the same experiments that
# established the overall approach works.
#
# References
# ----------
# Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Estimation
#     of individual axon bundle properties by a Multi-Resolution
#     Discrete-Search method. Medical Image Analysis, 42, 26-43.

@njit(cache=True, fastmath=True)
def compute_cone_refinement_schedule(grid_spacing_rad, target_resolution_rad):
    """
    Derive the two-level cone search schedule (angles + candidate counts)
    from the Stage A dictionary's own measured angular spacing and a
    single target final angular resolution -- see module docstring for
    the geometric ("equal-ratio zoom") rationale. No independently
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
        # The dictionary is already finer than the requested target --
        # no refinement is meaningfully possible/needed; collapse to a
        # single trivial level (n1=1 candidate = the coarse direction
        # itself), n2=0 signals "skip level 2" to the caller.
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def refine_fiber_direction_cone(bvals, bvecs, sig_norm, coarse_dir,
                                f_fib, f_res, f_hin, f_wat,
                                D_res, D_hin, D_wat, use_3iso,
                                cone1_half_angle, n1, cone2_half_angle, n2):
    """
    Two-level Fibonacci-cone refinement of Stage A's coarse dominant
    direction, scoring each candidate direction with Stage B's own
    closed-form regression + reconstructed SSE — no NNLS re-solve. See
    module docstring ("MRDS-LITE") for full rationale and validation
    summary.

    The isotropic fractions/centroids (f_res, f_hin, f_wat, D_res, D_hin,
    D_wat) and the fiber fraction f_fib are held FIXED at Stage A's
    values throughout — exactly as the unrefined `estimate_AD_RD_conditioned`
    already does; refinement only searches over direction (and, as a
    byproduct of re-running the Stage B regression at each candidate,
    over AD/RD).

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
    # Tangent basis around coarse_dir
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

    # Always evaluate the coarse direction itself first, as candidate
    # zero -- guarantees refinement never performs WORSE than the
    # unrefined baseline (monotonic-improvement floor).
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
    protocol (not per voxel, not Numba — plain NumPy is fine here since
    it runs a handful of times per `fit()` call, not per voxel).

    Used to derive the cone refinement schedule
    (`compute_cone_refinement_schedule`) directly from the ACTUAL
    dictionary in use, rather than assuming a theoretical spacing formula
    — this stays correct even if `n_dirs` or the hemisphere generator
    itself changes in the future.

    Parameters
    ----------
    fiber_dirs : ndarray (n_dirs, 3)

    Returns
    -------
    mean_spacing_rad : float
        Mean nearest-neighbour angular spacing across all directions.
    """
    n = len(fiber_dirs)
    dots = fiber_dirs @ fiber_dirs.T
    np.fill_diagonal(dots, -2.0)  # exclude self-match
    nearest_cos = np.clip(np.max(dots, axis=1), -1.0, 1.0)
    nearest_angle = np.arccos(nearest_cos)
    return float(np.mean(nearest_angle))




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

"""
DBSI Core Solvers — MRDS Multi-Fiber Stage B (ADDITION to core/solvers.py)
=============================================================================
APPEND THE CONTENTS BELOW TO core/solvers.py (after the existing
estimate_AD_RD_conditioned / MRDS-lite cone-refinement sections). Add
`select_dominant_directions` already exists in that file -- do not
duplicate it; only the functions below are new.

WHY THIS EXISTS
-------------------
Stage A already detects up to `max_fiber_populations` directions per voxel
(`select_dominant_directions`), but prior to this addition only the
DOMINANT direction was ever passed to Stage B; a second detected
population was discarded before reaching the output. This module adds
joint (AD, RD) estimation for 2-3 SIMULTANEOUS fiber populations, given
their Stage-A-detected directions and fractions.

METHOD SELECTION — WHY JOINT NONLINEAR, NOT ALTERNATING-TO-CONVERGENCE
-----------------------------------------------------------------------------
A synthetic sweep (2-fiber crossings at 30/60/90 deg, fraction splits
45/45 and 63/27, SNR=30, 3-shell Verona-like protocol, isotropic
compartment present, 20 noise seeds/condition) compared:

  ALTERNATING (reuse the single-fiber closed-form WLS per population,
  holding the other population's CURRENT estimate fixed, iterate to
  convergence): found to be NUMERICALLY UNSTABLE, not just slow --
  median AD-of-minority-population relative error on one representative
  noisy realisation INCREASED from 24.8% at 4 iterations to 79.4% at 40
  iterations (oscillation, not convergence, in this poorly-separated
  block-coordinate-descent problem). No monotonic-improvement guarantee,
  unlike the existing single-fiber MRDS-lite cone refinement.

  JOINT NONLINEAR LEAST-SQUARES (this module): bounded Levenberg-Marquardt
  over all 2*n_pop diffusivity parameters simultaneously, INITIALISED from
  a SHORT (2-3 iteration, deliberately NOT converged) alternating pass.
  Matched a scipy.optimize.least_squares reference to within numerical
  noise across the sweep (median mean-abs-relative-error 10.75% vs
  scipy's 10.51%; max per-parameter difference 1.15e-3 mm^2/s across 120
  trials). Post-JIT-warmup: ~0.04 ms/voxel single-threaded.

  CORRECTED 2026-07-15: the update step below had a sign error
  (`p_new = p + delta` where `delta = (JtJ+lam*D)^-1 Jt r` with J the
  Jacobian of the RESIDUAL, not the model) that made the LM step always
  move in the ascent direction. Practical effect, confirmed by direct
  reproduction: the inner accept/reject loop rejected every trial step
  regardless of lambda, so `improved` was always False and the solver
  exited after the FIRST outer iteration on essentially every voxel --
  silently returning `alternating_init_nfiber`'s deliberately
  under-converged 3-iteration warm start as if it were the converged
  joint fit, with no error or flag. The scipy-agreement numbers above
  cannot have been measured against this exact function in this state;
  they either predate the sign flip or were produced by a differently
  configured comparison. The fix (`p_new = p - delta`) has been verified
  on a synthetic 2-population crossing: reproduces the exact ground
  truth (cost ~1e-32) in 8 iterations, versus zero cost improvement over
  1 iteration before the fix. See project re-validation sweep
  (post-fix) for updated accuracy numbers replacing the ones above.

CRITICAL: INITIALISATION MUST BREAK SYMMETRY BETWEEN POPULATIONS.
Identical starting (AD, RD) for every population causes the joint solver
to stall on a subset of populations (confirmed: in a 3-fiber test,
population 1 converged to 0% error while populations 2-3 did not move AT
ALL from an identical starting point -- a degenerate-Jacobian symmetry
stall, not slow convergence). `alternating_init_nfiber`'s short,
population-differentiating pass exists specifically to prevent this.

SCOPE — WHAT THIS MODULE DOES NOT DO (read before relying on it)
-----------------------------------------------------------------------
- Does NOT touch, correct, or re-derive the isotropic-compartment
  FRACTIONS (RF/HF/WF/NRF) or the anisotropic FRACTIONS (FF) computed by
  Stage A's NNLS solve. Those are computed and written to the output
  BEFORE this module ever runs and are architecturally independent of
  it -- this module only refines AD/RD/FA/direction reporting for
  populations Stage A already detected. An attempt to also correct
  Stage A's fraction leakage via a small unregularized re-fit conditioned
  on this module's output ("Stage C") was tested and REJECTED: it
  collapses the isotropic compartment to 1-2 fixed-diffusivity columns,
  discarding the spectral resolution that is the toolbox's core
  contribution, and made FF/NRF recovery WORSE in 5 of 6 tested
  synthetic conditions (median FF error increases of 0.03-0.09 vs. Stage
  A's raw, uncorrected fractions). A "targeted" variant (full isotropic
  spectrum retained, only the anisotropic block replaced by refined
  fiber columns) was less harmful but still inconsistent (improved 4/6
  conditions on FF, only 3/6 on NRF) and is NOT included here. Isotropic
  fraction accuracy under true crossing-fiber ground truth is TRACKED AS
  A DOCUMENTED OPEN LIMITATION, not something this module claims to fix.
- Does NOT refine direction (unlike the existing single-fiber MRDS-lite
  cone search): directions passed in are Stage A's raw discrete-grid
  detections. Extending per-population cone refinement to 2-3 populations
  is architecturally possible (call `refine_fiber_direction_cone` per
  population before the joint fit) but has not been implemented or
  validated here.
- Validated (synthetic): 2-fiber crossings at 30/60/90 deg, symmetric and
  2.3:1 splits; ONE 3-fiber configuration at well-separated (60 deg
  apart) angles and roughly balanced fractions (comparable 4-16% errors).
  NOT validated: 3-fiber with unbalanced fractions or narrow angular
  separation; real (non-synthetic) data.

References
----------
Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Medical
    Image Analysis, 42, 26-43.
Levenberg K (1944); Marquardt DW (1963).
Project synthetic validation: 2-fiber and 3-fiber joint Stage B sweep;
    Stage A fraction-recovery-under-crossing sweep; Stage C re-fit
    rejection (see this docstring and project records).
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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

    RD = max(0.05e-3, min(3.0e-3, -x))
    AD = max(0.05e-3, min(3.5e-3, -x - y))
    if AD < RD:
        m = (AD + RD) / 2.0
        AD = m; RD = m
    return AD, RD


@njit(cache=True, fastmath=True)
def alternating_init_nfiber(bvals, bvecs, directions, fractions, iso_signal,
                            sig_norm, n_iter=3, AD0=1.6e-3, RD0=0.4e-3):
    """Cheap per-population initial guess for `estimate_AD_RD_nfiber_joint`,
    via a SHORT (default 3-iteration) alternating pass. Deliberately not
    run to convergence -- its only job is to break the symmetry between
    populations before handing off to the joint LM solve (see module
    docstring: identical initial values for every population cause the
    joint solver to stall).

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


@njit(cache=True, fastmath=True)
def estimate_AD_RD_nfiber_joint(bvals, bvecs, sig_norm, directions, fractions,
                                iso_signal, AD_init, RD_init, max_iter=25):
    """
    Joint (AD, RD) estimation for n_pop (2 or 3) SIMULTANEOUS fiber
    populations via bounded Levenberg-Marquardt, given FIXED directions
    and fractions (from Stage A) and a fixed isotropic-compartment
    signal. See module docstring for the empirical comparison against
    the (rejected) alternating-only approach.

    MUST be called with population-DIFFERENTIATED AD_init/RD_init (e.g.
    from `alternating_init_nfiber`) -- identical initial values for every
    population cause the solver to stall on a subset of populations.

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
        Bounded to [0.05e-3, 3.5e-3] (AD) / [0.05e-3, 3.0e-3] (RD),
        matching `estimate_AD_RD_conditioned`'s single-fiber bounds.
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
        lb[2 * k] = 0.05e-3
        ub[2 * k] = 3.5e-3
        lb[2 * k + 1] = 0.05e-3
        ub[2 * k + 1] = 3.0e-3
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
            # NOTE: J here is the Jacobian of the RESIDUAL r = sig_norm -
            # model (dr/dp), not of the model itself -- confirmed by
            # direct differentiation: J[i,2k] = frac*e*b*cos2 =
            # -d(model)/d(AD_k) = +d(r)/d(AD_k). For that convention the
            # Gauss-Newton/LM normal equations (JtJ + lam*D) delta = Jtr
            # give the step in the ASCENT direction; the minimising step
            # is p_new = p - delta, not p + delta. (Verified empirically:
            # with the old `p + delta` sign, cost never decreases and the
            # solver silently no-ops after 1 iteration on every voxel,
            # always returning the un-refined alternating-init warm
            # start.)
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


@njit(cache=True, fastmath=True)
def estimate_AD_RD_mrds(bvals, bvecs, sig_norm, directions, fractions,
                        iso_signal, init_n_iter=3, lm_max_iter=25,
                        AD0=1.6e-3, RD0=0.4e-3):
    """
    Full MRDS multi-fiber Stage B: `alternating_init_nfiber` (cheap,
    symmetry-breaking warm start) followed by
    `estimate_AD_RD_nfiber_joint` (bounded LM refine). Single entry point
    for the fitting kernel to call once Stage A has detected n_pop >= 2
    populations in a voxel; for n_pop == 1, continue using the existing
    `estimate_AD_RD_conditioned` (+ optional MRDS-lite cone refinement).

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

