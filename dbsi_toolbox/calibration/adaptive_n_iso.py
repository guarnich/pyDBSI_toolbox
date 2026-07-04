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
from ..core.basis import (
    build_isotropic_dictionary,
    generate_log_uniform_isotropic_grid,
    generate_anchored_isotropic_grid,
)
from ..core.solvers import nnls_coordinate_descent


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


# ─────────────────────────────────────────────────────────────────────────────
# FULLY DATA-DRIVEN n_iso SWEEP (alternative to SVD+floor)
# ─────────────────────────────────────────────────────────────────────────────

def select_n_iso_data_driven_sweep(bvals, y_voxels, n_iso_candidates=None,
                                   d_min=_DEFAULT_D_MIN, d_max=_DEFAULT_D_MAX,
                                   relative_tolerance=0.02, flatness_threshold=0.005,
                                   verbose=True):
    """
    Sweep n_iso directly against GCV score on REAL sampled voxels (no
    synthetic tissue priors).

    CORRECTED ROLE OF THIS FUNCTION (important — read before use)
    -----------------------------------------------------------------------
    This function was initially proposed as a direct, fully data-driven
    REPLACEMENT for `select_n_iso_svd`'s empirical floor (10). Project
    validation (synthetic data WITH KNOWN ground-truth fractions,
    repeated across independent noise seeds, AND a reconstruction of
    the project's actual Verona acquisition protocol) showed this
    would be a methodological regression, not an improvement:

      The GCV(n_iso) curve for diffusion MRI isotropic-spectrum fitting
      is SYSTEMATICALLY FLAT — total relative variation across n_iso=4
      to n_iso=34 was under 0.3% on generic synthetic data and under
      0.15% on the Verona-protocol reconstruction, in both cases far
      smaller than the run-to-run variation in the raw arg-min itself
      (which moved between n_iso=7, 19, and 29 across three repeats of
      IDENTICAL underlying ground truth, differing only in noise seed).
      Any tolerance band wide enough to be robust to that noise (>=0.5%)
      ends up encompassing the ENTIRE candidate range, so a parsimony
      tie-break against such a band trivially returns the smallest
      candidate tested — which is not a genuine recommendation, it is
      an artifact of the tie-break rule applied to an uninformative
      curve. This was verified to reproduce exactly this failure mode
      (collapsing to n_iso=4, the smallest tested) on both test cases.

    This function therefore now performs an explicit FLATNESS CHECK
    before issuing any recommendation (see `flatness_threshold`). If
    the GCV curve's total relative span across all candidates is below
    `flatness_threshold`, `n_iso_recommended` is returned as None and
    the report's `curve_is_flat` field is True — signalling that GCV
    provides no reliable signal for n_iso selection on this dataset,
    and the caller should defer to `select_n_iso_svd`'s SVD+empirical-
    floor default rather than trust a tie-broken arg-min from a flat
    curve. A non-flat curve (curve_is_flat=False) is the exception, not
    the expected case, for typical diffusion MRI protocols; when it
    does occur, the parsimony-adjusted recommendation is returned as
    before.

    In short: this function is a DIAGNOSTIC — it checks whether THIS
    dataset's GCV curve contains a genuine, exploitable signal about
    n_iso, and only acts on it if so. It is not a default replacement
    for the empirical floor, which remains the primary recommendation
    for the (typical) flat-curve case.

    Parameters
    ----------
    bvals : array-like (N,)
        B-values, s/mm^2, for the protocol being processed.
    y_voxels : array (n_voxels, N)
        Sampled normalised (S/S0) signals from the actual dataset (e.g.
        via `calibration.data_driven.sample_calibration_voxels`). Using
        MORE voxels than the default calibration sample size elsewhere
        in this toolbox (e.g. 150-300) is recommended specifically for
        this sweep, since GCV's flat-minimum behaviour (see above)
        benefits from averaging over a larger, more representative
        voxel sample to narrow the noise-driven spread in the curve.
    n_iso_candidates : iterable of int or None
        Candidate n_iso values to test. Defaults to range(4, 36, 2).
    d_min, d_max : float
        ADC range for the candidate grids (should match what will
        actually be used for fitting).
    relative_tolerance : float
        Parsimony tie-break tolerance (default 2%, i.e. the smallest
        n_iso within 2% of the minimum GCV score is selected rather than
        the raw arg-min) — ONLY applied if the curve is not flat (see
        `flatness_threshold`).
    flatness_threshold : float
        If the GCV curve's total relative span (max-min)/min across all
        tested candidates is below this threshold (default 0.5%), the
        curve is judged uninformative and `n_iso_recommended` is
        returned as None (see function docstring — this was found to be
        the typical case for diffusion MRI protocols tested, including
        a reconstruction of the project's Verona acquisition).
    verbose : bool
        Print the full sweep table.

    Returns
    -------
    n_iso_recommended : int or None
        The parsimony-adjusted recommendation from this sweep, or None
        if the curve was judged flat/uninformative (see
        `flatness_threshold`) — in that case, defer to
        `select_n_iso_svd`.
    report : dict
        {'candidates': list, 'gcv_scores': list, 'raw_argmin_n_iso': int,
         'n_iso_recommended': int or None (parsimony-adjusted),
         'relative_tolerance': float, 'relative_span': float,
         'curve_is_flat': bool}
    """
    from .data_driven import select_lambda_iso_gcv

    if n_iso_candidates is None:
        n_iso_candidates = list(range(4, 36, 2))
    else:
        n_iso_candidates = list(n_iso_candidates)

    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))

    if verbose:
        print(f"\n[DATA-DRIVEN n_iso SWEEP — {len(y_voxels)} real sampled voxels, "
              f"NO synthetic tissue priors]")
        print(f"  Candidates: {n_iso_candidates}")
        print(f"  {'n_iso':<8}{'best_lambda_iso':<20}{'GCV score'}")

    gcv_scores = []
    for n_cand in n_iso_candidates:
        iso_grid = generate_log_uniform_isotropic_grid(d_min=d_min, d_max=d_max, n_steps=n_cand)
        best_lam, gcv_diag = select_lambda_iso_gcv(bvals, iso_grid, y_voxels)
        score = float(np.min(gcv_diag['gcv']))
        gcv_scores.append(score)
        if verbose:
            print(f"  {n_cand:<8}{best_lam:<20.6f}{score:.8f}")

    gcv_scores = np.array(gcv_scores)
    raw_argmin_idx = int(np.argmin(gcv_scores))
    raw_argmin_n_iso = n_iso_candidates[raw_argmin_idx]
    min_score = gcv_scores[raw_argmin_idx]
    max_score = float(np.max(gcv_scores))

    # Flatness diagnostic: if the ENTIRE curve spans less than
    # `flatness_threshold` relative variation, no n_iso is meaningfully
    # preferred by GCV over any other — the parsimony tie-break below
    # would then just return the smallest candidate tested, which is
    # NOT a genuine recommendation in that case (see function docstring;
    # confirmed on both generic synthetic data and a reconstruction of
    # the project's Verona protocol, where total curve spread was
    # <0.3% and <0.15% respectively — well under typical numerical
    # noise in this computation).
    relative_span = (max_score - min_score) / min_score
    curve_is_flat = relative_span < flatness_threshold

    if curve_is_flat:
        n_iso_recommended = None
    else:
        within_tol = np.where(gcv_scores <= min_score * (1.0 + relative_tolerance))[0]
        parsimony_idx = int(within_tol.min())
        n_iso_recommended = n_iso_candidates[parsimony_idx]

    if verbose:
        print(f"\n  Raw arg-min: n_iso={raw_argmin_n_iso} (score={min_score:.8f})")
        print(f"  Curve relative span: {relative_span*100:.4f}% "
              f"(flatness threshold: {flatness_threshold*100:.2f}%)")
        if curve_is_flat:
            print(f"  CURVE IS FLAT: GCV does not meaningfully distinguish between "
                  f"the tested n_iso candidates for this dataset (confirmed in "
                  f"project validation to be the typical case for diffusion MRI "
                  f"protocols — see function docstring). No reliable data-driven "
                  f"recommendation can be extracted from this sweep; defer to the "
                  f"SVD+empirical-floor default (`select_n_iso_svd`).")
        else:
            print(f"  Parsimony-adjusted recommendation (smallest n_iso within "
                  f"{relative_tolerance*100:.0f}% of minimum): n_iso={n_iso_recommended}")
            if n_iso_recommended != raw_argmin_n_iso:
                print(f"  (Differs from raw arg-min — see function docstring.)")

    report = dict(
        candidates=n_iso_candidates,
        gcv_scores=gcv_scores.tolist(),
        raw_argmin_n_iso=raw_argmin_n_iso,
        n_iso_recommended=n_iso_recommended,
        relative_tolerance=relative_tolerance,
        relative_span=relative_span,
        curve_is_flat=curve_is_flat,
    )

    return n_iso_recommended, report


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP BIAS-VARIANCE SELECTOR  (preferred data-driven method for n_iso)
# ─────────────────────────────────────────────────────────────────────────────

def select_n_iso_bootstrap(bvals, y_voxels, snr, sigma_normalised,
                           n_iso_candidates=None,
                           d_min=_DEFAULT_D_MIN, d_max=_DEFAULT_D_MAX,
                           n_bootstrap=50, lambda_iso=None,
                           thresh_res=0.3e-3, cv_threshold=None,
                           verbose=True):
    """
    Select n_iso by minimising the composite bias² + variance of the
    restricted fraction (RF) estimate across bootstrapped noise
    replicates of real sampled voxels from the dataset being processed.

    WHY THIS IS PREFERRED OVER GCV-SWEEP FOR n_iso
    -----------------------------------------------------------------------
    Project validation directly compared three approaches for selecting
    n_iso from data (SVD+floor, GCV-sweep, bootstrap bias-variance) on
    a reconstruction of the Verona acquisition protocol (4 shells,
    b_max=2000, SNR=28.08):

      GCV-sweep: curve is systematically flat (<0.3% relative span
          across n_iso=4-34), raw arg-min unstable across noise seeds
          (shifted between n_iso=7, 19, 29 on three repeats of identical
          ground truth). Returns None when flat — a correct but
          uninformative outcome for typical dMRI protocols.

      Bootstrap bias-variance: the composite score bias²+variance shows
          a clear, stable minimum at n_iso=8 for the Verona protocol
          across 5 independent voxel-sampling seeds (all five agreed),
          with the minimum 3-10x lower than surrounding candidates.
          Unlike GCV, the composite explicitly optimises the quantity we
          care about (RF fraction accuracy, not signal-domain prediction
          error), and the bias component correctly penalises very small
          n_iso values that GCV mistakenly ranks highly (e.g. n_iso=4
          has near-zero CV but 30%+ RF bias — GCV's parsimony tie-break
          would select it; bootstrap correctly rejects it).

    This function is therefore the recommended primary data-driven
    n_iso selector when a representative sample of the dataset's own
    voxels is available (via `sample_calibration_voxels`) and the
    computational budget allows for ~50 bootstrap replicates per voxel
    (typical runtime: a few seconds for 30-60 voxels, pure Python/NumPy,
    no Numba required — the matrices are small).

    Algorithm
    ---------
    For each candidate n_iso:
      1. Build the anchored isotropic grid and calibrate lambda_iso via
         GCV at THAT n_iso (so the composite score is never confounded
         by a mismatched lambda).
      2. For each sampled voxel:
           a. Inject synthetic Rician noise at the measured SNR
              (n_bootstrap times, each with a different seed).
           b. Solve the regularised NNLS and record the estimated RF
              fraction.
           c. Compute CV(RF) = std(RF_estimates)/mean(RF_estimates).
      3. Composite score = mean_voxel(bias²_voxel + variance_voxel),
         where bias = mean(RF_estimates) - true_rf (approximated by the
         noiseless signal's RF estimate at lambda_iso — no external
         ground truth assumed) and variance = (CV * mean_rf)².
      4. Return the n_iso with the minimum composite score.

    IMPORTANT CAVEATS (consistent with project methodology)
    ---------------------------------------------------------------
    - The "true_rf" used here is an APPROXIMATE, not a ground-truth
      value: it is the RF estimated from the CLEAN (noise-free) version
      of each voxel's signal at each candidate n_iso+lambda. This means
      the bias term captures "model/discretisation bias" (how far the
      noiseless fit deviates from the low-noise consensus), not bias
      from a known external ground truth. On synthetic data with known
      fractions, the method correctly identifies n_iso=8 for the Verona
      protocol; whether this transfers to real brain tissue is assumed
      reasonable but has not been directly validated.
    - CRITICAL: this bias proxy is only informative when the sampled
      voxels are TISSUE-HETEROGENEOUS. If the calibration sample is
      dominated by near-identical voxels (e.g. a small, spatially
      clustered sample, or synthetic test voxels sharing one tissue
      composition), the bias term collapses toward zero for essentially
      any n_iso (there is no diversity of true D_iso positions for
      discretisation bias to act on), and the composite score then
      selects the SMALLEST n_iso purely because it minimises variance
      — reproducing the exact n_iso=4 pathology this method was
      designed to avoid. This was directly observed during integration
      testing: a synthetic sample of 18 near-identical voxels (single
      tissue composition) caused the selector to return n_iso=4 despite
      a heterogeneous sample of the same protocol correctly returning
      n_iso=8 across 5 seeds. `sample_calibration_voxels` drawing from
      the actual brain mask should give adequate heterogeneity in
      practice (grey matter, white matter, and partial-volume voxels
      mixed together), but this should be confirmed — e.g. by checking
      that the sampled voxels' S0-normalised signals are not near-
      duplicates — rather than assumed.
    - The composite score is in fraction-squared units (not percent).
      Comparing absolute scores across protocols is not meaningful —
      only the RELATIVE ranking across n_iso candidates for a given
      protocol matters.
    - Not validated on HCP (3 shells, b_max=3000). Expected behaviour:
      the optimal n_iso may differ from Verona's n_iso=8; run this
      function on each protocol separately.

    Parameters
    ----------
    bvals : array (N,)
    y_voxels : array (n_voxels, N)
        Sampled normalised (S/S0) signals from the dataset (e.g. from
        `sample_calibration_voxels`). 30-60 voxels is sufficient in
        testing; more helps if the voxel population is heterogeneous.
    snr : float
        Estimated SNR (from `estimate_snr_robust`).
    sigma_normalised : float
        Noise sigma in S/S0 units (= 1/snr approximately, but use the
        value from `sample_calibration_voxels` for consistency).
    n_iso_candidates : list of int or None
        Default: [4, 6, 8, 10, 12, 16, 20, 31]
    d_min, d_max : float
        ADC range for the candidate grids (match what will be used for
        fitting — default uses _DEFAULT_D_MIN=0.1e-3 and the extended
        upper bound 5.0e-3 to include free-water representation).
    n_bootstrap : int
        Number of noise replicates per voxel (default 50; 30 is
        adequate, more than 100 gives diminishing returns).
    lambda_iso : float or None
        If None (default), lambda_iso is calibrated via GCV at each
        n_iso. Passing a fixed value skips GCV and uses that lambda for
        all candidates — faster but potentially mismatched.
    thresh_res : float
        RF threshold (default 0.3e-3 mm²/s), must match the fitting
        code's THRESH_RES.
    cv_threshold : float or None
        If not None, also report the smallest n_iso where mean CV(RF)
        drops below this threshold (as a secondary diagnostic, see
        document suggestion). Default None = not used.
    verbose : bool

    Returns
    -------
    n_iso_opt : int
        The n_iso minimising bias² + variance.
    report : dict
        {'candidates': list, 'composite_scores': list,
         'cv_per_candidate': list, 'bias_per_candidate': list,
         'n_iso_opt': int, 'curve_is_flat': bool,
         'cv_threshold_result': int or None}
    """
    from .data_driven import select_lambda_iso_gcv

    if n_iso_candidates is None:
        n_iso_candidates = [4, 6, 8, 10, 12, 16, 20, 31]

    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))
    n_vox, N = y_voxels.shape

    if verbose:
        print(f"\n[BOOTSTRAP n_iso SELECTION — {n_vox} voxels, "
              f"{n_bootstrap} bootstrap replicates, SNR={snr:.1f}]")
        print(f"  Candidates: {n_iso_candidates}")
        print(f"  {'n_iso':<8}{'lambda_iso':<14}{'CV(RF)%':<14}"
              f"{'Bias':<14}{'Bias²+Var'}")

    composite_scores = []
    cv_list_outer = []
    bias_list_outer = []

    for n_iso in n_iso_candidates:
        iso_grid = generate_anchored_isotropic_grid(
            d_min=d_min, d_max=d_max, n_steps=n_iso,
            thresh_res=thresh_res, thresh_wat=3.0e-3
        )
        n_cols = len(iso_grid)

        # Calibrate lambda at this specific n_iso
        if lambda_iso is None:
            lam, _ = select_lambda_iso_gcv(bvals, iso_grid, y_voxels)
        else:
            lam = lambda_iso

        A = build_isotropic_dictionary(bvals, iso_grid)
        AtA_reg = A.T @ A + lam * np.eye(n_cols)
        At = A.T

        # Noiseless reference fit (used as approximate "true_rf" proxy)
        rf_noiseless = []
        for v in range(n_vox):
            w_ref, _ = nnls_coordinate_descent(AtA_reg, At @ y_voxels[v], 0.0)
            ftot = w_ref.sum()
            if ftot > 1e-10:
                rf_noiseless.append(w_ref[iso_grid <= thresh_res].sum() / ftot)
            else:
                rf_noiseless.append(0.0)
        rf_noiseless = np.array(rf_noiseless)

        # Bootstrap loop
        cv_per_vox = []
        bias_per_vox = []
        for v in range(n_vox):
            y_clean = y_voxels[v]
            rf_boots = []
            for rep in range(n_bootstrap):
                rng = np.random.default_rng(v * 10007 + rep)
                noisy = np.sqrt(
                    (y_clean + rng.normal(0, sigma_normalised, N))**2 +
                    rng.normal(0, sigma_normalised, N)**2
                )
                w, _ = nnls_coordinate_descent(AtA_reg, At @ noisy, 0.0)
                ftot = w.sum()
                if ftot > 1e-10:
                    rf_boots.append(w[iso_grid <= thresh_res].sum() / ftot)

            if rf_boots:
                rf_arr = np.array(rf_boots)
                mean_rf = np.mean(rf_arr)
                cv_per_vox.append(
                    np.std(rf_arr) / (mean_rf + 1e-10) * 100
                )
                # Bias vs noiseless reference (discretisation + model bias)
                bias_per_vox.append(abs(mean_rf - rf_noiseless[v]))

        mean_cv = float(np.mean(cv_per_vox)) if cv_per_vox else np.nan
        mean_bias = float(np.mean(bias_per_vox)) if bias_per_vox else np.nan

        # Composite in fraction² units, with a reference mean RF of 0.15
        # (representative for brain) to convert CV% to variance fraction²
        var_frac2 = (mean_cv / 100.0 * 0.15) ** 2
        bias2 = mean_bias ** 2
        composite = bias2 + var_frac2

        composite_scores.append(composite)
        cv_list_outer.append(mean_cv)
        bias_list_outer.append(mean_bias)

        if verbose:
            print(f"  {n_iso:<8}{lam:<14.5f}{mean_cv:<14.2f}"
                  f"{mean_bias:<14.4f}{composite:.6f}")

    composite_arr = np.array(composite_scores)
    best_idx = int(np.argmin(composite_arr))
    n_iso_opt = n_iso_candidates[best_idx]

    # Flatness check: is the composite curve genuinely informative?
    span = (composite_arr.max() - composite_arr.min()) / (composite_arr.min() + 1e-15)
    # Threshold: if the MINIMUM is less than 3x any other candidate, the
    # signal is weak (empirically: Verona showed ~10x separation at n_iso=8).
    curve_is_flat = span < 0.5   # span < 50% relative spread = weak signal

    # Homogeneity diagnostic: if bias is near-zero for EVERY candidate
    # (not just the smallest), the sampled voxels likely lack tissue
    # diversity for the bias proxy to be meaningful — see function
    # docstring caveat on this failure mode (observed to select n_iso=4
    # spuriously on a near-homogeneous synthetic sample during project
    # integration testing). A bias ceiling of 0.01 (1 percentage point)
    # across ALL candidates is used as a conservative trigger.
    bias_arr = np.array(bias_list_outer)
    sample_looks_homogeneous = bool(np.all(bias_arr < 0.01))

    # Optional CV-threshold criterion (document Suggestion 1 secondary check)
    cv_threshold_result = None
    if cv_threshold is not None:
        below = [c for c, cv in zip(n_iso_candidates, cv_list_outer)
                 if cv <= cv_threshold]
        cv_threshold_result = min(below) if below else None

    if verbose:
        print(f"\n  Optimal n_iso (min bias²+var): {n_iso_opt}")
        if curve_is_flat:
            print(f"  [WARNING] Composite curve relative span={span*100:.1f}% "
                  f"is small — selection may not be reliable for this dataset.")
        else:
            print(f"  Composite curve relative span: {span*100:.1f}% "
                  f"(clear minimum — selection is reliable)")
        if sample_looks_homogeneous:
            print(f"  [WARNING] Bias is <1 percentage point for ALL candidate "
                  f"n_iso values — the sampled voxels may lack sufficient "
                  f"tissue diversity for the bias proxy to be meaningful. "
                  f"The selected n_iso={n_iso_opt} may be driven mainly by "
                  f"variance minimisation (favouring small n_iso) rather than "
                  f"a genuine bias-variance trade-off. Consider re-sampling "
                  f"with a larger/more spatially distributed voxel set.")
        if cv_threshold is not None:
            print(f"  CV-threshold ({cv_threshold}%) criterion: "
                  f"n_iso={cv_threshold_result}")

    return n_iso_opt, dict(
        candidates=n_iso_candidates,
        composite_scores=composite_scores,
        cv_per_candidate=cv_list_outer,
        bias_per_candidate=bias_list_outer,
        n_iso_opt=n_iso_opt,
        curve_is_flat=curve_is_flat,
        sample_looks_homogeneous=sample_looks_homogeneous,
        cv_threshold_result=cv_threshold_result,
    )
