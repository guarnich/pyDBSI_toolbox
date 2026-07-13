"""
DBSI Monte Carlo SURE Cross-Check — Non-Asymptotic Risk Estimate for the NNLS Isotropic Fit
=================================================================================================

WHY THIS MODULE EXISTS
--------------------------
`calibration.data_driven.select_lambda_iso_gcv` selects lambda_iso via
Generalized Cross-Validation, using closed-form filter factors derived
from the SVD of the isotropic design matrix. That derivation is EXACT
for an unconstrained linear (ridge-regression-style) estimator, but the
actual estimator fit in this toolbox is NNLS — non-negativity constrained
and therefore nonlinear in the observed signal y. When many isotropic
weights are pinned at the zero boundary by the constraint (common in
this sparse spectral recovery problem), the constrained estimator's true
effective degrees of freedom differ from what the unconstrained GCV
formula assumes. GCV therefore remains a fast, reasonable APPROXIMATION,
but not a formally exact one for NNLS.

Stein's Unbiased Risk Estimate (SURE) gives an unbiased estimate of an
estimator's true risk (expected squared prediction error) under additive
Gaussian noise, for ANY estimator f(y) — linear or not — provided its
"divergence" (sum of partial derivatives d f_i / d y_i, i.e. the trace
of its Jacobian) can be computed or estimated:

    SURE(f) = -N*sigma^2 + ||y - f(y)||^2 + 2*sigma^2 * divergence(f)

Monte Carlo SURE (Ramani, Blu & Unser, IEEE Trans. Image Processing,
2008) estimates the divergence term via a randomized finite-difference
probe, without needing an analytical Jacobian:

    divergence(f) ~= (1/eps) * b^T [f(y + eps*b) - f(y)]

averaged over several independent random probe vectors b (Rademacher:
+-1 entries, mean 0, unit variance). This works for NNLS directly (just
re-solve NNLS at the perturbed signal), unlike GCV's closed-form filter
factors which have no analogue once the non-negativity constraint binds.

EMPIRICAL FINDING THAT SHAPED THIS MODULE'S DESIGN (project validation,
Verona-protocol reconstruction, n_iso=12, 60 sampled voxels)
-----------------------------------------------------------------------
A direct comparison of GCV's chosen lambda_iso against a full Monte
Carlo SURE sweep over the same lambda grid found:
  - GCV's argmin was very stable across independent voxel samples
    (0.043, 0.043, 0.078 across 3 seeds).
  - Monte Carlo SURE's RAW argmin was highly unstable (0.0002-0.004
    across the same 3 voxel samples, over an order of magnitude), even
    at a generous 50 probes per voxel per lambda (a materially more
    expensive setting than the default used elsewhere in this module).
  - Examining the full SURE curve directly explained why: risk values
    across lambda ~ 0.0003 to ~0.05 were essentially flat (within
    ~15% of each other), i.e. a genuine flat valley in the risk
    landscape (confirmed NOT an artifact of Monte Carlo probe noise,
    since it persisted at 50 probes), not merely a stable-but-different
    optimum from GCV's.
  - Critically, GCV's chosen lambda in every tested case fell WITHIN
    this SURE-estimated low-risk valley, clearly separated from the
    high-risk region (lambda > ~0.15, where SURE risk rose sharply and
    consistently across all tests).

CONCLUSION AND DESIGN CONSEQUENCE: Monte Carlo SURE, in this setting,
is informative about which REGIME of lambda is reasonable (confirming
or contradicting GCV's choice against a formally NNLS-correct risk
criterion), but its raw point-estimate argmin is no more reliable than
GCV's within the flat, statistically-indistinguishable region — same
overall lesson as this project's earlier findings for GCV-based n_iso
sweeps (`adaptive_n_iso.select_n_iso_data_driven_sweep`) and the
discrepancy-principle floor. This module therefore does NOT offer a
"select_lambda_iso_sure" function that returns a new point estimate.
Instead, it offers CROSS-CHECK functions that evaluate SURE risk in a
neighbourhood of an already-chosen candidate (from GCV, or from a
candidate n_iso) and report whether that candidate falls within the
low-risk region — consistent with this project's established pattern
for `select_n_iso_with_gcv_crosscheck` and the Monte Carlo
tissue-scenario cross-check (`calibration.optimizer.evaluate_lambda_pair`):
verification tools, not replacement selectors.

COMPUTATIONAL COST
-----------------------
Each SURE risk evaluation at one (n_iso, lambda) candidate requires,
per sampled voxel: 1 baseline NNLS solve + n_probes additional NNLS
solves (default 15) for the perturbed signals. This is more expensive
than GCV (near-free, closed-form) but cheaper than the full bootstrap
bias-variance sweep (`adaptive_n_iso.select_n_iso_bootstrap`, which
re-solves NNLS n_bootstrap=50 times per voxel per candidate, with no
opportunity to reuse the baseline solve). Because this module only
evaluates a candidate and a small neighbourhood around it (not a full
grid sweep), the total cost for a typical cross-check (5-7
neighbourhood points x 15 probes x ~60-500 voxels) is comparable to or
somewhat less than one bootstrap n_iso sweep pass.

References
----------
Stein CM (1981). Estimation of the mean of a multivariate normal
    distribution. Annals of Statistics, 9(6), 1135-1151.
Ramani S, Blu T, Unser M (2008). Monte-Carlo SURE: a black-box
    optimization of regularization parameters for general denoising
    algorithms. IEEE Trans. Image Processing, 17(9), 1540-1554.
"""

import numpy as np
from ..core.solvers import nnls_coordinate_descent
from ..core.basis import build_isotropic_dictionary, generate_anchored_isotropic_grid


# ─────────────────────────────────────────────────────────────────────────────
# CORE MC-SURE RISK ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def _mc_sure_risk(A, y_voxels, lam, sigma, n_probes=15, eps=None, seed=0):
    """
    Monte Carlo SURE risk estimate for the isotropic NNLS fit at a given
    lambda, averaged over the sampled voxels and over `n_probes` random
    Rademacher perturbations per voxel.

    Parameters
    ----------
    A : ndarray (N, K)
        Isotropic design matrix (same construction as
        `calibration.data_driven.select_lambda_iso_gcv`).
    y_voxels : ndarray (n_voxels, N)
        Sampled normalised (S/S0) real voxel signals.
    lam : float
        Candidate lambda_iso.
    sigma : float
        Noise standard deviation, in the same normalised (S/S0) units as
        y_voxels (same quantity used throughout `calibration.data_driven`).
    n_probes : int
        Number of independent Rademacher probes per voxel used to
        estimate the divergence term. Default 15 — chosen as a
        practical middle ground (project testing found n_probes=2 gives
        visibly unstable results across repeated runs; n_probes=15 vs 50
        gave similar overall risk LEVELS, though neither pins down an
        exact minimum in a flat valley — see module docstring). Increase
        for a smoother risk estimate at proportionally higher cost.
    eps : float or None
        Finite-difference step size for the divergence probe. Default
        None uses `0.01 * sigma`, tying the probe step to the actual
        noise scale of the data (a step much smaller than the noise
        floor risks numerical cancellation; much larger biases the
        finite-difference derivative estimate) — this is the standard
        recommendation from Ramani, Blu & Unser (2008).
    seed : int
        Seed for the probe random number generator (reproducibility).

    Returns
    -------
    mean_risk : float
        SURE risk estimate, averaged over voxels. Lower = better
        estimated prediction risk. NOT normalised for comparison across
        different N or different protocols — only meaningful for
        RELATIVE comparison across candidates on the SAME protocol/voxel
        sample.
    """
    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))
    N, K = A.shape
    AtA = A.T @ A
    AtA_reg = AtA + lam * np.eye(K)

    if eps is None:
        eps = 0.01 * sigma

    rng = np.random.default_rng(seed)
    total_risk = 0.0

    for y in y_voxels:
        Aty = A.T @ y
        w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
        yhat = A @ w
        resid2 = float(np.sum((y - yhat) ** 2))

        divergence = 0.0
        for _ in range(n_probes):
            b = rng.choice(np.array([-1.0, 1.0]), size=N)
            y_pert = y + eps * b
            w_p, _ = nnls_coordinate_descent(AtA_reg, A.T @ y_pert, 0.0)
            yhat_p = A @ w_p
            divergence += float(np.dot(b, yhat_p - yhat)) / eps
        divergence /= n_probes

        sure = -N * sigma ** 2 + resid2 + 2.0 * sigma ** 2 * divergence
        total_risk += sure

    return total_risk / len(y_voxels)


# ─────────────────────────────────────────────────────────────────────────────
# LAMBDA_ISO CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────

def crosscheck_lambda_iso_sure(bvals, iso_grid, y_voxels, sigma, lambda_candidate,
                               n_neighbors=3, neighbor_factor=4.0, n_probes=15,
                               flat_tolerance=0.15, verbose=True):
    """
    Cross-check a candidate lambda_iso (typically from
    `calibration.data_driven.select_lambda_iso_gcv`) against Monte Carlo
    SURE risk evaluated at that candidate and a small log-spaced
    neighbourhood around it. Does NOT return a new lambda — see module
    docstring for why a precise SURE-only point estimate is not reliable
    in this problem (the risk landscape's low-risk region is typically a
    flat valley, not a sharp minimum).

    Parameters
    ----------
    bvals : array-like (N,)
    iso_grid : array (K,)
        Isotropic ADC grid — MUST match the grid the candidate lambda
        was actually selected/will be used for.
    y_voxels : array (n_voxels, N)
        Sampled normalised (S/S0) real voxel signals (e.g. from
        `sample_calibration_voxels`).
    sigma : float
        Noise standard deviation in normalised units.
    lambda_candidate : float
        The lambda to cross-check (typically GCV's choice).
    n_neighbors : int
        Number of neighbours to evaluate on EACH side of the candidate
        (total evaluated points = 2*n_neighbors + 1).
    neighbor_factor : float
        Neighbours are log-spaced between
        lambda_candidate/neighbor_factor and lambda_candidate*neighbor_factor.
        Default 4.0 (i.e. neighbourhood spans a 16x range) is a
        deliberately wide net given the risk landscape's demonstrated
        flatness — a narrower neighbourhood would mostly just confirm
        the flatness without ever reaching the high-risk region that
        this cross-check is actually trying to detect.
    n_probes : int
        Passed to `_mc_sure_risk` (see its docstring for the rationale
        behind the default of 15).
    flat_tolerance : float
        Relative tolerance (default 15%) used to decide whether
        lambda_candidate's risk is "within" the low-risk region: True if
        risk(lambda_candidate) <= min(risk over neighbourhood) * (1 + flat_tolerance).
    verbose : bool
        Print the full neighbourhood risk table.

    Returns
    -------
    agrees : bool
        True if lambda_candidate's SURE risk is within `flat_tolerance`
        of the neighbourhood minimum (i.e. GCV's choice is NOT
        contradicted by the NNLS-correct risk criterion).
    report : dict
        {'lambdas': array, 'risks': array, 'candidate_risk': float,
         'neighborhood_min_risk': float, 'best_neighbor_lambda': float,
         'agrees': bool}
    """
    A = build_isotropic_dictionary(np.asarray(bvals, dtype=np.float64), iso_grid)

    neighbors = np.geomspace(
        lambda_candidate / neighbor_factor, lambda_candidate * neighbor_factor,
        2 * n_neighbors + 1
    )
    # Ensure the exact candidate is evaluated (geomspace midpoint may
    # differ from lambda_candidate by floating-point rounding).
    neighbors[n_neighbors] = lambda_candidate

    if verbose:
        print(f"\n[SURE CROSS-CHECK — lambda_iso]")
        print(f"  Candidate: {lambda_candidate:.5f}  |  "
              f"Neighbourhood: [{neighbors[0]:.5f}, {neighbors[-1]:.5f}]  |  "
              f"n_probes={n_probes}")

    risks = np.array([
        _mc_sure_risk(A, y_voxels, lam, sigma, n_probes=n_probes, seed=42)
        for lam in neighbors
    ])

    candidate_risk = risks[n_neighbors]
    min_risk = float(np.min(risks))
    best_neighbor_lambda = float(neighbors[np.argmin(risks)])
    agrees = candidate_risk <= min_risk * (1.0 + flat_tolerance)

    if verbose:
        for lam, risk in zip(neighbors, risks):
            marker = " <- candidate" if lam == lambda_candidate else ""
            marker += " <- SURE min in neighborhood" if lam == best_neighbor_lambda and lam != lambda_candidate else ""
            print(f"    lambda={lam:.5f}  SURE_risk={risk:.6f}{marker}")
        if agrees:
            print(f"  AGREEMENT: candidate lambda_iso's SURE risk is within "
                  f"{flat_tolerance*100:.0f}% of the neighbourhood minimum — "
                  f"the NNLS-correct risk criterion does not contradict GCV's choice.")
        else:
            print(f"  DISAGREEMENT: a neighbour (lambda={best_neighbor_lambda:.5f}) "
                  f"achieves SURE risk more than {flat_tolerance*100:.0f}% lower than "
                  f"the candidate. Consider investigating further before trusting "
                  f"GCV's choice unsupervised for this dataset/protocol.")

    return agrees, dict(
        lambdas=neighbors, risks=risks, candidate_risk=candidate_risk,
        neighborhood_min_risk=min_risk, best_neighbor_lambda=best_neighbor_lambda,
        agrees=agrees,
    )


# ─────────────────────────────────────────────────────────────────────────────
# N_ISO CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────

def crosscheck_n_iso_sure(bvals, y_voxels, sigma, n_iso_candidate,
                          d_min=0.1e-3, d_max=5.0e-3,
                          n_neighbors=2, n_probes=15,
                          flat_tolerance=0.15, verbose=True):
    """
    Cross-check a candidate n_iso (typically from
    `calibration.adaptive_n_iso.select_n_iso_bootstrap` or
    `select_n_iso_svd`) against Monte Carlo SURE risk evaluated at that
    candidate and its immediate integer neighbours, with lambda_iso
    re-tuned via GCV at each candidate (matching how
    `select_n_iso_with_gcv_crosscheck` handles the same issue for its
    own GCV-based comparison).

    Same non-replacement philosophy as `crosscheck_lambda_iso_sure` —
    see that function's and this module's docstrings.

    Parameters
    ----------
    bvals : array-like (N,)
    y_voxels : array (n_voxels, N)
    sigma : float
    n_iso_candidate : int
        The n_iso to cross-check.
    d_min, d_max : float
        ADC range for the candidate grids (must match what will
        actually be used for fitting).
    n_neighbors : int
        Number of integer neighbours evaluated on each side of the
        candidate (default 2, i.e. n_iso_candidate +/- 1, +/- 2).
    n_probes : int
        Passed to `_mc_sure_risk`.
    flat_tolerance : float
        Same role as in `crosscheck_lambda_iso_sure`.
    verbose : bool

    Returns
    -------
    agrees : bool
    report : dict
        {'n_iso_values': list, 'risks': array, 'candidate_risk': float,
         'neighborhood_min_risk': float, 'best_neighbor_n_iso': int,
         'agrees': bool}
    """
    from .data_driven import select_lambda_iso_gcv

    bvals = np.asarray(bvals, dtype=np.float64)
    n_iso_values = [
        n for n in range(n_iso_candidate - n_neighbors, n_iso_candidate + n_neighbors + 1)
        if n >= 3
    ]
    if n_iso_candidate not in n_iso_values:
        n_iso_values.append(n_iso_candidate)
        n_iso_values.sort()

    if verbose:
        print(f"\n[SURE CROSS-CHECK — n_iso]")
        print(f"  Candidate: {n_iso_candidate}  |  Neighbourhood: {n_iso_values}  |  "
              f"n_probes={n_probes}")

    risks = []
    for n_cand in n_iso_values:
        iso_grid = generate_anchored_isotropic_grid(d_min=d_min, d_max=d_max, n_steps=n_cand)
        lam_gcv, _ = select_lambda_iso_gcv(bvals, iso_grid, y_voxels)
        A = build_isotropic_dictionary(bvals, iso_grid)
        risk = _mc_sure_risk(A, y_voxels, lam_gcv, sigma, n_probes=n_probes, seed=42)
        risks.append(risk)
        if verbose:
            print(f"    n_iso={n_cand:3d}  lambda_iso(GCV)={lam_gcv:.5f}  SURE_risk={risk:.6f}")

    risks = np.array(risks)
    candidate_idx = n_iso_values.index(n_iso_candidate)
    candidate_risk = risks[candidate_idx]
    min_risk = float(np.min(risks))
    best_neighbor_n_iso = n_iso_values[int(np.argmin(risks))]
    agrees = candidate_risk <= min_risk * (1.0 + flat_tolerance)

    if verbose:
        if agrees:
            print(f"  AGREEMENT: candidate n_iso={n_iso_candidate}'s SURE risk is "
                  f"within {flat_tolerance*100:.0f}% of the neighbourhood minimum.")
        else:
            print(f"  DISAGREEMENT: n_iso={best_neighbor_n_iso} achieves SURE risk "
                  f"more than {flat_tolerance*100:.0f}% lower than the candidate. "
                  f"Consider investigating before trusting the candidate unsupervised.")

    return agrees, dict(
        n_iso_values=n_iso_values, risks=risks, candidate_risk=candidate_risk,
        neighborhood_min_risk=min_risk, best_neighbor_n_iso=best_neighbor_n_iso,
        agrees=agrees,
    )
