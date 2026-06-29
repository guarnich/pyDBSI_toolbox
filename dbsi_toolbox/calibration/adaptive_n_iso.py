"""
DBSI Adaptive Isotropic Basis Count — SVD Estimation + Empirical Floor + GCV Cross-Check
=============================================================================================

WHAT THIS MODULE DOES
-------------------------
Replaces the fixed n_iso=31 default with a protocol-aware, data-aware
estimate of how many isotropic basis functions the acquisition can
actually support, combining three checks rather than trusting any one
in isolation:

  1. SVD-BASED INFORMATION LIMIT (`select_n_iso_svd`)
     Counts singular values of a high-resolution isotropic design
     matrix above a noise-relative threshold. This directly measures
     the rank of the isotropic inverse problem for the GIVEN protocol
     (b-values) — see module docstring discussion below for why this is
     protocol-bounded rather than SNR-bounded for typical dMRI shell
     counts.

  2. EMPIRICAL FLOOR (min_n_iso, default 10)
     Project synthetic validation (600-cell sweep across 4 protocols x
     5 tissue scenarios x 5 SNR levels x 6 candidate n_iso values — see
     project records) showed that the SVD answer ALONE, with a low
     floor (3, as in early drafts of this approach), is frequently
     SUBOPTIMAL specifically on clinically relevant scenarios (NAWM,
     near-pure hindered grey matter, restricted lesions): RF recovery
     error was roughly DOUBLE compared to a floor of ~10-12 in those
     scenarios. n_iso=12 was the best-performing single fixed value
     across the full sweep, beating both n_iso=3 and the legacy
     n_iso=31. This is the empirical basis for `min_n_iso=10`: a
     conservative floor close to, but not above, the sweep's
     empirical optimum, leaving room for the SVD answer to exceed it
     when the protocol genuinely supports more resolution.

  3. GCV CROSS-CHECK (`select_n_iso_with_gcv_crosscheck`)
     After computing n_iso_opt = max(n_iso_svd, min_n_iso), evaluates
     GCV (see `calibration.data_driven.select_lambda_iso_gcv`) at a
     small range of candidate n_iso values around n_iso_opt using
     ACTUAL sampled voxels from the dataset being processed. If GCV's
     own (lambda, n_iso) joint score clearly favours a different n_iso
     in the tested range, this is flagged — the SVD+floor answer is
     informationally motivated but does not see the actual noise
     realisation in this specific dataset, whereas GCV does. This
     cross-check does not silently override the SVD+floor answer; it
     reports a recommendation and the caller decides.

WHY THE SVD ANSWER IS BOUNDED BY THE PROTOCOL, NOT BY SNR
----------------------------------------------------------------
For the isotropic block alone, the design matrix A_iso has rows
exp(-b_i * D_j); rows at the SAME b-value are IDENTICAL regardless of
gradient direction (no orientation term). The rank of A_iso is
therefore bounded by the number of DISTINCT b-values in the protocol,
not by the total number of measurements. Empirically, for typical
clinical/research dMRI protocols (1-7 distinct shells), the singular
value spectrum of even a 200-point high-resolution candidate grid
collapses to machine precision after only 3-5 singular values,
essentially independent of SNR in the 10-100 range (the noise floor is
reached far above machine-precision values for any realistic SNR, but
the protocol's intrinsic rank is reached even sooner). This matches the
Borgia et al. 1998 spectral-resolution bound already used elsewhere in
this toolbox (HF/WF separability requires b_eff >~ 7600 s/mm^2): the
same underlying mathematics, applied to the full isotropic spectrum
rather than just the HF/WF boundary.

References
----------
Borgia GC, Brown RJS, Fantazzini P (1998). Uniform-penalty inversion of
    multiexponential decay data. J Magn Reson, 132(1), 65-77.
Whittall KP, MacKay AL (1989). Quantitative interpretation of NMR
    relaxation data. J Magn Reson, 84(1), 134-152.
Project synthetic validation records: n_iso recovery sweep (4 protocols
    x 5 tissue scenarios x 5 SNR levels x 6 n_iso candidates).
"""

import numpy as np
from ..core.basis import build_isotropic_dictionary, generate_log_uniform_isotropic_grid


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Empirical floor: see module docstring. This is NOT a purely
# theoretical choice — it is set close to (12) but slightly below the
# best-performing fixed value found in project synthetic validation, to
# leave a small margin for the SVD answer to exceed it on richer
# protocols without the floor itself forcing an overly large value.
_DEFAULT_MIN_N_ISO = 10
_DEFAULT_MAX_N_ISO = 60   # safety cap on the other end; see select_n_iso_svd docstring
_DEFAULT_HIRES_N_CANDIDATE = 200
_DEFAULT_D_MIN = 0.1e-3
_DEFAULT_D_MAX = 3.0e-3


# ─────────────────────────────────────────────────────────────────────────────
# SVD-BASED INFORMATION LIMIT
# ─────────────────────────────────────────────────────────────────────────────

def select_n_iso_svd(bvals, snr, d_min=_DEFAULT_D_MIN, d_max=_DEFAULT_D_MAX,
                     n_hires_candidates=_DEFAULT_HIRES_N_CANDIDATE,
                     min_n_iso=_DEFAULT_MIN_N_ISO, max_n_iso=_DEFAULT_MAX_N_ISO):
    """
    Estimate the number of isotropic basis functions the acquisition
    protocol can support, via SVD of a high-resolution candidate
    isotropic design matrix, clipped to [min_n_iso, max_n_iso].

    Algorithm
    ---------
    1. Build a high-resolution (n_hires_candidates points, log-uniform)
       candidate isotropic design matrix for the GIVEN bvals.
    2. Compute its SVD; normalise singular values to the largest.
    3. Count how many normalised singular values exceed 1/snr (the
       noise-relative threshold: a spectral mode whose contribution to
       the signal is below the noise floor cannot be reliably resolved).
    4. Clip the count to [min_n_iso, max_n_iso] — see module docstring
       for why a floor above the raw SVD answer is necessary in
       practice (the raw answer is frequently 3-5 for typical protocols,
       which project validation showed under-performs on several
       clinically relevant tissue scenarios).

    IMPORTANT CAVEAT (stated explicitly, not silently absorbed into the
    floor): for the large majority of standard dMRI protocols (1-7
    distinct b-value shells), this function's RAW (pre-floor) answer
    will be small (often 3-5) and the floor will dominate the result.
    This is expected and is not a sign of a bug — it reflects a genuine
    property of the isotropic inverse problem (see module docstring) —
    but it does mean the "SVD-based" framing should not be oversold as
    determining n_iso in the typical case; in practice, for most
    protocols this function mainly serves to detect the (rarer) case of
    an unusually rich protocol that could support MORE than min_n_iso,
    and to provide a principled cap if SNR is very low.

    Parameters
    ----------
    bvals : array-like (N,)
        B-values, s/mm^2.
    snr : float
        Estimated SNR (e.g. from `estimate_snr_robust`).
    d_min, d_max : float
        ADC range for the high-resolution candidate grid, mm^2/s.
    n_hires_candidates : int
        Resolution of the candidate grid used only for the SVD estimate
        (not the grid actually used for fitting — that is built
        separately with n_iso_opt points via
        `generate_log_uniform_isotropic_grid`).
    min_n_iso, max_n_iso : int
        Clipping bounds. See module docstring for the empirical basis
        of the default min_n_iso=10.

    Returns
    -------
    n_iso_opt : int
        The clipped, final recommended n_iso.
    diagnostics : dict
        {'n_iso_raw': int (pre-clip SVD answer),
         'singular_values_normalised': ndarray,
         'threshold': float (1/snr),
         'floor_applied': bool,
         'cap_applied': bool}
    """
    bvals = np.asarray(bvals, dtype=np.float64)

    hires_grid = generate_log_uniform_isotropic_grid(
        d_min=d_min, d_max=d_max, n_steps=n_hires_candidates
    )
    A_hires = build_isotropic_dictionary(bvals, hires_grid)

    # Singular values only (cheaper than full SVD; sufficient for the
    # rank-counting criterion).
    s = np.linalg.svd(A_hires, compute_uv=False)
    s_norm = s / s[0]

    threshold = 1.0 / max(float(snr), 1.0)
    n_iso_raw = int(np.sum(s_norm > threshold))
    n_iso_raw = max(n_iso_raw, 1)  # guard degenerate all-below-threshold case

    n_iso_opt = int(np.clip(n_iso_raw, min_n_iso, max_n_iso))

    diagnostics = dict(
        n_iso_raw=n_iso_raw,
        singular_values_normalised=s_norm,
        threshold=threshold,
        floor_applied=(n_iso_raw < min_n_iso),
        cap_applied=(n_iso_raw > max_n_iso),
    )

    return n_iso_opt, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# GCV CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────

def select_n_iso_with_gcv_crosscheck(bvals, snr, y_voxels,
                                     d_min=_DEFAULT_D_MIN, d_max=_DEFAULT_D_MAX,
                                     min_n_iso=_DEFAULT_MIN_N_ISO,
                                     max_n_iso=_DEFAULT_MAX_N_ISO,
                                     gcv_search_radius=6, verbose=True):
    """
    Compute n_iso_opt via `select_n_iso_svd`, then cross-check it
    against GCV evaluated on actual sampled voxels from the dataset
    being processed, over a small range of candidate n_iso values
    around the SVD+floor answer.

    This does NOT replace the SVD+floor answer automatically — it
    reports both and flags a disagreement, leaving the decision (and
    the responsibility for it) explicit rather than silently picking
    whichever score is lowest. This mirrors the project's broader
    methodology of treating any single automatic criterion (Monte
    Carlo, SVD, GCV) as informative but not infallible, and preferring
    cross-checks over blind trust in any one of them.

    Parameters
    ----------
    bvals : array-like (N,)
    snr : float
        Estimated SNR.
    y_voxels : array (N,) or (n_voxels, N)
        Sampled normalised (S/S0) signals from the dataset being
        processed (e.g. via `calibration.data_driven.
        sample_calibration_voxels`), used to evaluate GCV at each
        candidate n_iso.
    d_min, d_max : float
        ADC range (must match what will actually be used for fitting).
    min_n_iso, max_n_iso : int
        Passed through to `select_n_iso_svd`.
    gcv_search_radius : int
        GCV is evaluated at n_iso values from
        max(min_n_iso, n_iso_opt - gcv_search_radius) to
        n_iso_opt + gcv_search_radius, in steps of 2 (to keep the search
        cheap — this is a cross-check, not a full optimisation).
    verbose : bool
        Print the comparison.

    Returns
    -------
    n_iso_opt : int
        The SVD+floor recommendation (NOT automatically replaced by the
        GCV cross-check's preference — see function docstring).
    report : dict
        {
          'svd_diagnostics': dict (from select_n_iso_svd),
          'gcv_candidates': list of int,
          'gcv_scores': list of float (best GCV score at each candidate
              n_iso, i.e. GCV minimised over lambda_iso at that n_iso),
          'gcv_preferred_n_iso': int (the candidate with lowest GCV
              score),
          'agreement': bool (True if gcv_preferred_n_iso == n_iso_opt
              or is within one search step of it),
        }
    """
    from .data_driven import select_lambda_iso_gcv

    n_iso_opt, svd_diag = select_n_iso_svd(
        bvals, snr, d_min=d_min, d_max=d_max,
        min_n_iso=min_n_iso, max_n_iso=max_n_iso,
    )

    lo = max(min_n_iso, n_iso_opt - gcv_search_radius)
    hi = min(max_n_iso, n_iso_opt + gcv_search_radius)
    candidates = sorted(set([lo, n_iso_opt, hi] + list(range(lo, hi + 1, 2))))

    gcv_scores = []
    for n_cand in candidates:
        iso_grid_cand = generate_log_uniform_isotropic_grid(
            d_min=d_min, d_max=d_max, n_steps=n_cand
        )
        _, gcv_diag = select_lambda_iso_gcv(bvals, iso_grid_cand, y_voxels)
        gcv_scores.append(float(np.min(gcv_diag['gcv'])))

    best_idx = int(np.argmin(gcv_scores))
    gcv_preferred_n_iso = candidates[best_idx]

    agreement = abs(gcv_preferred_n_iso - n_iso_opt) <= 2

    if verbose:
        print(f"\n[n_iso CROSS-CHECK]")
        print(f"  SVD+floor recommendation: n_iso={n_iso_opt} "
              f"(raw SVD answer: {svd_diag['n_iso_raw']}, "
              f"floor_applied={svd_diag['floor_applied']})")
        print(f"  GCV scores by candidate n_iso:")
        for n_cand, score in zip(candidates, gcv_scores):
            marker = " <- SVD+floor" if n_cand == n_iso_opt else ""
            marker += " <- GCV preferred" if n_cand == gcv_preferred_n_iso else ""
            print(f"    n_iso={n_cand:3d}  gcv={score:.6f}{marker}")
        if agreement:
            print(f"  AGREEMENT: GCV's preferred n_iso ({gcv_preferred_n_iso}) is "
                  f"close to the SVD+floor recommendation ({n_iso_opt}).")
        else:
            print(f"  DISAGREEMENT: GCV prefers n_iso={gcv_preferred_n_iso}, "
                  f"notably different from the SVD+floor recommendation "
                  f"({n_iso_opt}). Consider using the GCV-preferred value, "
                  f"or investigate why the two criteria disagree for this "
                  f"protocol/dataset before trusting either blindly.")

    report = dict(
        svd_diagnostics=svd_diag,
        gcv_candidates=candidates,
        gcv_scores=gcv_scores,
        gcv_preferred_n_iso=gcv_preferred_n_iso,
        agreement=agreement,
    )

    return n_iso_opt, report
