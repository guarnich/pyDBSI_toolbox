"""
DBSI Adaptive Isotropic Basis Count — SVD Estimation + Empirical Floor + GCV Cross-Check
=============================================================================================

WHAT THIS MODULE DOES
-------------------------
OPTIONAL per-dataset n_iso selectors. The fit does NOT use them by default:
n_iso is fixed to 6 by grid convergence (`n_iso_method='quadrature'`, v1.3.9),
because Stage D re-solves the reported fractions on fixed centroids and n_iso
is a discretisation parameter, not a model order. They run only when asked:

  1. SVD-BASED INFORMATION LIMIT (`select_n_iso_svd`, `n_iso_method=
     'svd_floor'`, and the fallback of the bootstrap)
     Counts singular values of a high-resolution isotropic design matrix
     above a noise-relative threshold (1/SNR), clipped to [min_n_iso,
     max_n_iso] (floor 4, see _DEFAULT_MIN_N_ISO). It measures the rank of
     the isotropic inverse problem for the GIVEN b-values -- see below for
     why that is protocol-bounded rather than SNR-bounded. The caller passes
     the fit's grid range (v1.8.1).

  2. BOOTSTRAP BIAS-VARIANCE (`select_n_iso_bootstrap`, `n_iso_method=
     'bootstrap'`)
     Scores each candidate n_iso by bias^2 + variance of the restricted
     fraction over Rician bootstrap replicates of a fixed, richer reference
     fit, on the fiber-subtracted residual. See its docstring.

Removed in 1.8.1 because nothing called them: `select_n_iso_with_gcv_crosscheck`
(GCV cross-check around the SVD answer) and `select_n_iso_data_driven_sweep`
(pure GCV sweep over n_iso).

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

# Floor on the SVD answer (the raw count is often 3-5 on clinical protocols).
_DEFAULT_MIN_N_ISO = 4
# Era 10. Il valore non veniva da un guasto osservato: era prudenza, per non
# lasciare lo spettro isotropo troppo grossolano. Misurato il 2026-09-22, non
# protegge da niente di strutturale — l'ancoraggio alle soglie di comparto
# garantisce la copertura di RF/HF/WF gia' a n_steps=2 (griglia 8 colonne,
# 3/3/2). E costava caro: il pavimento non puo' mai concordare con un
# bootstrap che dice 6, quindi ogni volta che il fallback scattava n_iso
# saltava di almeno 4. Abbassato a 4 (9 colonne, 3/4/2) per lasciar parlare
# la SVD, che e' il limite informativo vero del protocollo; aggiungere
# colonne oltre quello aggiunge mal-condizionamento, non informazione.
_DEFAULT_MAX_N_ISO = 60   # safety cap on the other end; see select_n_iso_svd docstring
_DEFAULT_HIRES_N_CANDIDATE = 200
_DEFAULT_D_MIN = 0.1e-3
_DEFAULT_D_MAX = 3.0e-3    # the model passes its own grid range, [0.1, 5.0]e-3 (v1.8.1)


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
    4. Clip the count to [min_n_iso, max_n_iso].

    CAVEAT: for standard dMRI protocols (1-7 distinct b-value shells) the
    raw answer is small (often 3-5), because the isotropic block's rank is
    bounded by the number of distinct b-values (see module docstring), so
    the floor (4) often decides. The "SVD-based" framing should not be
    oversold as determining n_iso in the typical case.

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
        `generate_anchored_isotropic_grid`).
    min_n_iso, max_n_iso : int
        Clipping bounds (defaults 4 and 60, see the constants above).

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
# BOOTSTRAP BIAS-VARIANCE SELECTOR  (preferred data-driven method for n_iso)
# ─────────────────────────────────────────────────────────────────────────────

# Rapporto di separazione minimo (max/min del composito) perche' il ranking dei
# candidati sia considerato informativo. Verona misurava ~10x quando la curva ha
# un minimo chiaro. Espresso come RAPPORTO e non come span proprio perche' le due
# forme erano state confuse: il commento diceva 3x e il codice faceva 1.5x.
_FLAT_RATIO = 1.5


def _replica_corretta(base, sigma, rng):
    """Una replica bootstrap: rumore Rician su `base`, poi la STESSA correzione
    del fit, sqrt(max(S^2 - 2 sigma^2, 0)) (1.8.1).

    `y_voxels` arriva al bootstrap gia' corretto, quindi lo e' anche il
    riferimento liscio `base`. Fino alla 1.8.0 le repliche restavano grezze: il
    bootstrap le confrontava con un segnale di un altro tipo, e il bias
    misurato conteneva il pavimento del rumore oltre all'errore della griglia.
    """
    n = base.shape[0]
    raw = np.sqrt((base + rng.normal(0, sigma, n)) ** 2 + rng.normal(0, sigma, n) ** 2)
    return np.sqrt(np.maximum(raw ** 2 - 2.0 * sigma ** 2, 0.0))


def select_n_iso_bootstrap(bvals, y_voxels, snr, sigma_normalised,
                           n_iso_candidates=None,
                           d_min=_DEFAULT_D_MIN, d_max=_DEFAULT_D_MAX,
                           n_bootstrap=50, lambda_iso=None,
                           thresh_res=0.3e-3, cv_threshold=None,
                           n_ref_steps=48, verbose=True):
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
              (n_bootstrap times, each with a different seed) and apply
              the fit's Rician correction (`_replica_corretta`).
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

    THREE DEFECTS FIXED IN v1.3.6 — read before comparing with older runs
    -----------------------------------------------------------------------
    1. DOUBLE NOISE. Replicates used to be generated by adding noise at sigma
       to `y_voxels`, which is a REAL measured signal and therefore already
       carries noise at sigma. The replicates thus lived at ~2*sigma^2, i.e. the
       bootstrap probed SNR/sqrt(2), not SNR. Now the replicates are generated
       from a FITTED signal (see 2), so the noise level is the real one.

    2. THE REFERENCE MOVED WITH THE CANDIDATE. `rf_noiseless` used to be
       recomputed per candidate as that candidate's own fit of the observed
       signal, so every candidate was scored against ITSELF. That makes the
       "bias" term nearly vacuous, which is why the bias came out below 1
       percentage point for every candidate on real data and tripped the
       `sample_looks_homogeneous` warning: the flag was reporting a property of
       the ESTIMATOR, not of the sample.
       Now there is ONE reference per voxel, fitted on a grid FINER than any
       candidate (`n_ref_steps`), and every candidate is scored against it.
       Coarse grids therefore show genuine discretisation bias and fine grids
       show variance — an actual bias-variance trade-off. Generating from a
       candidate's own grid instead would make its bias zero by construction.

    3. lambda_iso COULD RAIL. The per-candidate GCV runs on the full signal,
       which the main path (`select_lambdas_data_driven`) documents as pushing
       lambda_iso far too high, because the anisotropic component the iso block
       cannot represent dominates the GCV residual. The fibre-subtracted
       residual is not available here (this runs BEFORE lambda_aniso exists),
       so instead the same noise-referenced ceiling the main path uses is
       applied: `lambda_iso_discrepancy_cap`. At good SNR it is inert; it only
       bites when GCV would otherwise diverge to the grid maximum and collapse
       every isotropic fraction.

    AND ONE THING THE CALLER MUST KNOW: n_iso IS NOT THE NUMBER OF COLUMNS.
    `generate_anchored_isotropic_grid` returns MORE points than `n_steps` (it
    anchors on the compartment thresholds and fills oversized geometric gaps),
    and the realised length is NOT monotone in n_iso: with the production
    range, n_iso=6 gives 11 columns while n_iso=8 gives 10, and the restricted
    column count drops from 4 to 3. The candidates are therefore a SET of
    grids, not a complexity ladder, and the report now carries `n_cols_per
    _candidate` so the selected model order is visible. A warning is printed
    when the realised lengths are non-monotone.

    Returns
    -------
    n_iso_opt : int
        The n_iso minimising bias² + variance.
    report : dict
        {'candidates': list, 'n_cols_per_candidate': list,
         'composite_scores': list, 'cv_per_candidate': list,
         'bias_per_candidate': list, 'lambda_iso_per_candidate': list,
         'n_iso_opt': int, 'n_cols_opt': int, 'curve_is_flat': bool,
         'cols_non_monotone': bool, 'n_ref_cols': int,
         'cv_threshold_result': int or None}
    """
    from .data_driven import select_lambda_iso_gcv, lambda_iso_discrepancy_cap

    if n_iso_candidates is None:
        n_iso_candidates = [4, 6, 8, 10, 12, 16, 20, 31]

    y_voxels = np.atleast_2d(np.asarray(y_voxels, dtype=np.float64))
    n_vox, N = y_voxels.shape

    # ── Riferimento comune, su una griglia PIU' FINE di ogni candidato ──────
    # Serve a una cosa sola: dare al bias qualcosa di fisso e piu' ricco contro
    # cui essere misurato. Se si generasse dalla griglia del candidato, il suo
    # bias sarebbe zero per costruzione (vedi difetto 2 nel docstring).
    ref_grid = generate_anchored_isotropic_grid(
        d_min=d_min, d_max=d_max, n_steps=n_ref_steps,
        thresh_res=thresh_res, thresh_wat=3.0e-3,
    )
    A_ref = build_isotropic_dictionary(bvals, ref_grid)
    lam_ref = (lambda_iso if lambda_iso is not None
               else select_lambda_iso_gcv(bvals, ref_grid, y_voxels)[0])
    AtA_ref = A_ref.T @ A_ref + lam_ref * np.eye(len(ref_grid))
    At_ref = A_ref.T
    y_hat = np.empty_like(y_voxels)      # segnale di riferimento, liscio
    rf_ref = np.empty(n_vox)             # la "verita'" contro cui misurare
    for v in range(n_vox):
        w, _ = nnls_coordinate_descent(AtA_ref, At_ref @ y_voxels[v], 0.0)
        y_hat[v] = A_ref @ w
        tot = w.sum()
        rf_ref[v] = (w[ref_grid <= thresh_res].sum() / tot) if tot > 1e-10 else 0.0

    # ── Il riferimento e' utilizzabile? ────────────────────────────────────
    # Se la RF di riferimento e' degenere, il termine di bias e' misurato contro
    # il nulla e la selezione scappa verso il candidato piu' grande. E' quello
    # che succede se si passa un segnale che contiene FIBRA a un dizionario
    # isotropo: la parte anisotropa viene assorbita e la RF collassa a zero.
    # Misurato su un fantoccio con RF vera 0.10 e FF 0.55: RF di riferimento
    # 0.000, n_iso scelto 31 (il massimo). Va gridato, non silenziato: il
    # chiamante deve passare un residuo con la fibra sottratta
    # (`fiber_subtracted_residual`), oppure accettare il ripiego SVD.
    rf_ref_mean = float(np.mean(rf_ref))
    reference_degenerate = (not np.isfinite(rf_ref_mean)) or rf_ref_mean < 0.02
    if reference_degenerate:
        if verbose:
            print(f"\n[BOOTSTRAP n_iso] REFERENCE DEGENERATE: mean reference "
                  f"RF = {rf_ref_mean:.4f} (< 0.02). The bias term would be "
                  f"measured against nothing and the selection would run away "
                  f"to the largest candidate. Refusing to select; the caller "
                  f"should pass a fibre-subtracted residual "
                  f"(calibration.data_driven.fiber_subtracted_residual) or use "
                  f"the SVD fallback.")
        return None, dict(candidates=list(n_iso_candidates),
                          n_cols_per_candidate=[], composite_scores=[],
                          cv_per_candidate=[], bias_per_candidate=[],
                          lambda_iso_per_candidate=[],
                          n_iso_opt=None, n_cols_opt=None,
                          n_ref_cols=int(len(ref_grid)),
                          reference_rf_mean=rf_ref_mean,
                          lambda_iso_reference=float(lam_ref),
                          curve_is_flat=True, separation_ratio=0.0,
                          cols_non_monotone=False,
                          reference_degenerate=True,
                          sample_looks_homogeneous=False,
                          cv_threshold_result=None)

    if verbose:
        print(f"\n[BOOTSTRAP n_iso SELECTION — {n_vox} voxels, "
              f"{n_bootstrap} bootstrap replicates, SNR={snr:.1f}]")
        print(f"  Candidates: {n_iso_candidates}")
        print(f"  Reference grid: n_steps={n_ref_steps} -> {len(ref_grid)} columns "
              f"(lambda_iso={lam_ref:.5f}); mean reference RF={rf_ref.mean():.3f}")
        print(f"  {'n_iso':<8}{'cols':<7}{'lambda_iso':<13}{'CV(RF)%':<12}"
              f"{'Bias':<12}{'Bias²+Var'}")

    composite_scores = []
    cv_list_outer = []
    bias_list_outer = []
    cols_list = []
    lam_list = []
    capped_any = False

    for n_iso in n_iso_candidates:
        iso_grid = generate_anchored_isotropic_grid(
            d_min=d_min, d_max=d_max, n_steps=n_iso,
            thresh_res=thresh_res, thresh_wat=3.0e-3
        )
        n_cols = len(iso_grid)

        A = build_isotropic_dictionary(bvals, iso_grid)
        At = A.T

        # lambda_iso al candidato, con il TETTO riferito al rumore (difetto 3)
        if lambda_iso is None:
            lam, _gdiag = select_lambda_iso_gcv(bvals, iso_grid, y_voxels)
            _cap = lambda_iso_discrepancy_cap(
                bvals, iso_grid, y_voxels, sigma_normalised, _gdiag['lambda_'])
            if _cap < lam:
                lam = _cap
                capped_any = True
        else:
            lam = lambda_iso
        AtA_reg = A.T @ A + lam * np.eye(n_cols)

        # Bootstrap: rumore UNA volta sola, sul segnale di riferimento liscio.
        # Il seme dipende solo da (v, rep), quindi tutti i candidati vedono le
        # STESSE realizzazioni di rumore -- numeri casuali comuni, che e' cio'
        # che rende il confronto fra candidati poco rumoroso.
        cv_per_vox = []
        bias_per_vox = []
        for v in range(n_vox):
            base = y_hat[v]
            rf_boots = []
            for rep in range(n_bootstrap):
                rng = np.random.default_rng(v * 10007 + rep)
                noisy = _replica_corretta(base, sigma_normalised, rng)
                w, _ = nnls_coordinate_descent(AtA_reg, At @ noisy, 0.0)
                ftot = w.sum()
                if ftot > 1e-10:
                    rf_boots.append(w[iso_grid <= thresh_res].sum() / ftot)

            if rf_boots:
                rf_arr = np.array(rf_boots)
                mean_rf = float(np.mean(rf_arr))
                cv_per_vox.append(np.std(rf_arr) / (mean_rf + 1e-10) * 100)
                # bias contro il riferimento FISSO, non contro se stesso
                bias_per_vox.append(abs(mean_rf - rf_ref[v]))

        mean_cv = float(np.mean(cv_per_vox)) if cv_per_vox else np.nan
        mean_bias = float(np.mean(bias_per_vox)) if bias_per_vox else np.nan

        # Composito in unita' di frazione². La CV% si converte in varianza con
        # la RF MISURATA sul riferimento, non con un 0.15 scritto a mano: quella
        # costante decideva il peso relativo fra bias e varianza, cioe' quale
        # n_iso vince, e descriveva un tessuto invece che questi dati.
        rf_scale = float(np.mean(rf_ref))
        var_frac2 = (mean_cv / 100.0 * rf_scale) ** 2
        bias2 = mean_bias ** 2
        composite = bias2 + var_frac2

        composite_scores.append(composite)
        cv_list_outer.append(mean_cv)
        bias_list_outer.append(mean_bias)
        cols_list.append(n_cols)
        lam_list.append(float(lam))

        if verbose:
            print(f"  {n_iso:<8}{n_cols:<7}{lam:<13.5f}{mean_cv:<12.2f}"
                  f"{mean_bias:<12.4f}{composite:.6f}")

    if verbose and capped_any:
        print("  [NOTE] the noise-referenced ceiling bit on at least one "
              "candidate: without it GCV on the full signal can rail to the "
              "grid maximum and collapse every isotropic fraction.")

    composite_arr = np.array(composite_scores)
    best_idx = int(np.argmin(composite_arr))
    n_iso_opt = n_iso_candidates[best_idx]

    # Flatness check: is the composite curve genuinely informative?
    #
    # `span` is (max-min)/min, i.e. span = R - 1 where R = max/min is the
    # separation RATIO. The comment here used to state the criterion as "the
    # minimum is less than 3x any other candidate" while the code tested
    # `span < 0.5`, which is a ratio of 1.5x, not 3x -- the two described
    # different tests, and this one decides whether the data-driven answer is
    # used at all or replaced by the SVD fallback. Stated as a ratio now, so
    # code and comment cannot drift again:
    #   Verona measured ~10x separation (span 9.0) when the curve is
    #   informative; below _FLAT_RATIO the ranking is not trusted.
    span = (composite_arr.max() - composite_arr.min()) / (composite_arr.min() + 1e-15)
    ratio = span + 1.0
    curve_is_flat = ratio < _FLAT_RATIO

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
            print(f"  [WARNING] separation ratio max/min = {ratio:.2f}x "
                  f"(< {_FLAT_RATIO}x) — the ranking is not reliable for this "
                  f"dataset; falling back to SVD+floor.")
        else:
            print(f"  Separation ratio max/min = {ratio:.2f}x "
                  f"(>= {_FLAT_RATIO}x — clear minimum, selection is reliable)")
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

    # n_iso NON e' il numero di colonne, e non e' nemmeno monotono in esso.
    cols_non_monotone = any(b <= a for a, b in zip(cols_list, cols_list[1:]))
    if verbose and cols_non_monotone:
        coppie = [f"n_iso={n}->{c}" for n, c in zip(n_iso_candidates, cols_list)]
        print("  [WARNING] the realised column count is NOT monotone in n_iso: "
              + ", ".join(coppie))
        print("            the candidates are a SET of grids, not a complexity "
              "ladder; read n_cols, not n_iso.")

    return n_iso_opt, dict(
        candidates=n_iso_candidates,
        n_cols_per_candidate=cols_list,
        lambda_iso_per_candidate=lam_list,
        composite_scores=composite_scores,
        cv_per_candidate=cv_list_outer,
        bias_per_candidate=bias_list_outer,
        n_iso_opt=n_iso_opt,
        n_cols_opt=int(cols_list[best_idx]),
        n_ref_cols=int(len(ref_grid)),
        reference_rf_mean=float(np.mean(rf_ref)),
        lambda_iso_reference=float(lam_ref),
        curve_is_flat=curve_is_flat,
        separation_ratio=float(ratio),
        cols_non_monotone=bool(cols_non_monotone),
        reference_degenerate=False,
        sample_looks_homogeneous=sample_looks_homogeneous,
        cv_threshold_result=cv_threshold_result,
    )
