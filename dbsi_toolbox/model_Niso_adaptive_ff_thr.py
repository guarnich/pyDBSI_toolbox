"""
DBSI Adaptive Model — Isotropic Compartment Selection Based on Acquisition Scheme
==================================================================================

This module implements a unified DBSI fitting pipeline that automatically selects
between a **two-compartment** (2-ISO) and a **three-compartment** (3-ISO) isotropic
model based on the maximum b-value and the number of distinct non-zero b-value
shells present in the acquisition protocol.

Compartment Definitions
-----------------------
Both models share the restricted compartment (RF) definition of Wang et al. (2011):

    RF  : ADC ≤ THRESH_RES  (0.3 × 10⁻³ mm²/s) — cells, inflammatory infiltrate
    HF  : THRESH_RES < ADC ≤ THRESH_WAT          — hindered extracellular water
    WF  : ADC > THRESH_WAT  (3.0 × 10⁻³ mm²/s)  — free water, CSF, oedema
    NRF : HF + WF  (Non-Restricted Fraction, 2-ISO mode only)

Model Selection Criterion
--------------------------
The two models are fundamentally constrained by signal physics:

    2-ISO model (NRF = HF + WF merged):
        Selected when  b_max < B_THRESH_3ISO  (default: 3000 s/mm²)
        OR             n_nonzero_shells < MIN_SHELLS_3ISO  (default: 3)

        Rationale: At b_max < 3000 s/mm² the free water signal
        (exp(-b × 3.0e-3)) drops below ~5% of S₀ at the highest shell,
        placing it near or below the noise floor at typical in-vivo SNR
        (~25–50). The NNLS cannot reliably separate HF from WF under
        these conditions; merging them into a single NRF compartment is
        both numerically stable and physiologically meaningful because
        the clinically relevant distinction — cellular (RF) vs. non-cellular
        (NRF) water — is preserved.

    3-ISO model (HF and WF estimated separately):
        Selected when  b_max ≥ B_THRESH_3ISO  AND  n_nonzero_shells ≥ MIN_SHELLS_3ISO

        Rationale: At b_max ≥ 3000 s/mm² with ≥ 3 diffusion-weighted shells,
        the NNLS has sufficient measurements to resolve the full isotropic
        spectrum. The hindered-to-free-water contrast at intermediate b-values
        (b ~ 300–700 s/mm²) combined with the extended ADC range visible at
        high b provides the necessary information for stable three-compartment
        estimation. The "efficient best" calibration criterion automatically
        selects higher N_iso and λ in this regime to match the expanded
        spectral resolution.

Restricted Fraction — Minimum b-value Requirements
----------------------------------------------------
The restricted compartment (ADC ≤ 0.3e-3 mm²/s) has the following detectability
constraints:

    • b_max < 1000 s/mm²: RF not quantifiable. The RF/Hindered signal ratio
      at the highest shell is < 2×, insufficient for NNLS disambiguation at
      typical SNR. RF may be detected (non-zero) but cannot be reliably
      quantified.

    • 1000 ≤ b_max < 1500 s/mm²: Qualitative estimation only. Signal ratio
      RF/Hindered ≈ 2–3×. Requires SNR > 40. Multi-shell design mandatory.

    • 1500 ≤ b_max < 2000 s/mm²: Marginal quantitative regime. Wang et al. 2011
      validated the RF threshold (0.3 mm²/ms) at b = 1500 as the lowest
      informative shell in a multi-shell design.

    • b_max ≥ 2000 s/mm²: Quantitatively reliable with ≥ 2 diffusion-weighted
      shells. This is the validated minimum for clinical applications per Wang
      et al. 2011 and Shirani et al. 2019.

    • b_max ≥ 3000 s/mm²: Optimal. Additional shell diversity allows separation
      of HF from WF. RF precision further improved by the high-b plateau.

Output Channels (11 total — unified across both model modes)
-------------------------------------------------------------
    0  : FF   — Fibre fraction                            (always valid)
    1  : RF   — Restricted fraction  (ADC ≤ 0.3e-3)      (always valid)
    2  : HF   — Hindered fraction    (0.3e-3 < ADC ≤ 3.0e-3) (NaN in 2-ISO mode)
    3  : WF   — Free-water fraction  (ADC > 3.0e-3)      (NaN in 2-ISO mode)
    4  : NRF  — Non-Restricted fraction = HF + WF        (always valid)
    5  : AD   — Axial diffusivity                         (NaN if FF ≤ fiber_threshold)
    6  : RD   — Radial diffusivity                        (NaN if FF ≤ fiber_threshold)
    7  : FA   — Intrinsic fibre FA                        (NaN if FF ≤ fiber_threshold)
    8  : ADC_iso — Mean isotropic ADC                     (always valid)
    9  : AD_lin  — Analytical AD estimate (Step 1)        (NaN if FF ≤ fiber_threshold)
    10 : RD_lin  — Analytical RD estimate (Step 1)        (NaN if FF ≤ fiber_threshold)

In 2-ISO mode: channels 2 (HF) and 3 (WF) are NaN; channel 4 (NRF) is the
directly estimated non-restricted fraction.

In 3-ISO mode: channels 2, 3, and 4 are all valid, with NRF = HF + WF enforced
by construction after renormalisation.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590–3601. doi:10.1093/brain/awr307
Shirani A, et al. (2019). Ann Clin Transl Neurol, 6(11):2323–2327.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33–47.
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from core.solvers import (
    nnls_coordinate_descent,
    compute_fiber_fa,
)
from calibration.optimizer import optimize_hyperparameters
from utils.tools import estimate_snr_robust, correct_rician_bias


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DESIGN_MATRIX_AD = 1.5e-3   # mm²/s  — fixed AD used to build design matrix
DESIGN_MATRIX_RD = 0.5e-3   # mm²/s  — fixed RD used to build design matrix
FIBER_THRESHOLD  = 0.15     # dimensionless — minimum FF for AD/RD estimation

# Isotropic compartment ADC boundaries (Wang et al. 2011)
THRESH_RES = 0.3e-3         # mm²/s  — restricted / hindered boundary
THRESH_WAT = 3.0e-3         # mm²/s  — hindered  / free-water boundary

# Adaptive model selection thresholds
B_THRESH_3ISO   = 3000.0    # s/mm²  — minimum b_max to activate 3-ISO model
MIN_SHELLS_3ISO = 3         # minimum distinct non-zero b-value shells for 3-ISO


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL ANALYSIS UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def analyse_protocol(bvals):
    """
    Analyse the diffusion acquisition scheme and determine which isotropic
    model is appropriate.

    Parameters
    ----------
    bvals : array-like (N,)
        B-values in s/mm².

    Returns
    -------
    b_max : float
        Maximum b-value in the protocol.
    n_nonzero_shells : int
        Number of distinct non-zero b-value shells (rounded to nearest 100).
    use_3iso : bool
        True  → three-compartment model (RF + HF + WF)
        False → two-compartment model  (RF + NRF)
    reason : str
        Human-readable explanation of the model selection decision.
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    b_max = float(np.max(bvals))

    # Round to nearest 100 s/mm² to count distinct shells
    rounded = np.round(bvals, -2)
    unique_nonzero = np.unique(rounded[rounded > 50])
    n_nonzero_shells = int(len(unique_nonzero))

    if b_max < B_THRESH_3ISO and n_nonzero_shells < MIN_SHELLS_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: b_max = {b_max:.0f} s/mm² < {B_THRESH_3ISO:.0f} s/mm² "
            f"AND only {n_nonzero_shells} non-zero shell(s) (minimum {MIN_SHELLS_3ISO} required "
            f"for 3-ISO). Free-water signal below noise floor; NRF = HF + WF merged."
        )
    elif b_max < B_THRESH_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: b_max = {b_max:.0f} s/mm² < {B_THRESH_3ISO:.0f} s/mm². "
            f"At this b_max, exp(-b_max × D_free) = {np.exp(-b_max * THRESH_WAT):.4f}, "
            f"placing the free-water signal near the noise floor at typical SNR. "
            f"NRF = HF + WF merged for numerical stability."
        )
    elif n_nonzero_shells < MIN_SHELLS_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: only {n_nonzero_shells} distinct non-zero b-shell(s) "
            f"detected (minimum {MIN_SHELLS_3ISO} required for 3-ISO). "
            f"Insufficient shell diversity to constrain HF/WF separation via NNLS. "
            f"NRF = HF + WF merged."
        )
    else:
        use_3iso = True
        reason = (
            f"3-ISO model selected: b_max = {b_max:.0f} s/mm² ≥ {B_THRESH_3ISO:.0f} s/mm² "
            f"with {n_nonzero_shells} distinct non-zero shells. "
            f"Sufficient b-range and shell diversity to resolve RF, HF, and WF separately."
        )

    return b_max, n_nonzero_shells, use_3iso, reason


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ANALYTICAL AD/RD INITIALISATION  (2-ISO and 3-ISO versions)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _estimate_AD_RD_2iso(bvals, bvecs, sig_norm, fiber_dir,
                         f_fib, f_res, f_nonrf, D_res, D_nonrf):
    """
    Analytical WLS estimate of (AD, RD) using a two-compartment isotropic model.

    The log-signal from the isolated fibre contribution is fitted by weighted
    least squares in (D_perp, D_parallel - D_perp), exploiting the cylindrical
    symmetry of the tensor model.

    Parameters
    ----------
    bvals, bvecs : arrays
        Acquisition protocol.
    sig_norm : array
        Normalised signal (S / S₀).
    fiber_dir : array (3,)
        Dominant fibre direction (unit vector).
    f_fib, f_res, f_nonrf : float
        Normalised fractions (sum to 1).
    D_res, D_nonrf : float
        Centroid ADCs of the two isotropic compartments (mm²/s).

    Returns
    -------
    AD_est, RD_est : float
        Estimated diffusivities, or np.nan if the WLS system is singular.
    """
    ftot = f_fib + f_res + f_nonrf + 1e-12
    ff = f_fib   / ftot
    fr = f_res   / ftot
    fn = f_nonrf / ftot

    sum_AA = 0.0; sum_AB = 0.0; sum_BB = 0.0
    sum_Ay = 0.0; sum_By = 0.0

    for i in range(len(bvals)):
        b       = bvals[i]
        S_total = max(sig_norm[i], 0.01)
        S_iso   = fr * np.exp(-b * D_res) + fn * np.exp(-b * D_nonrf)
        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        S_fiber = max(min(S_fiber, 1.0), 0.01)
        log_S   = np.log(S_fiber)

        g     = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2  = cos_t * cos_t
        w     = S_total * b          # amplitude-weighted, b-weighted

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
        m      = (AD_est + RD_est) / 2.0
        AD_est = m; RD_est = m

    return AD_est, RD_est


@njit(cache=True, fastmath=True)
def _estimate_AD_RD_3iso(bvals, bvecs, sig_norm, fiber_dir,
                         f_fib, f_res, f_hin, f_wat,
                         D_res, D_hin, D_wat):
    """
    Analytical WLS estimate of (AD, RD) using a three-compartment isotropic model.

    Identical mathematics to _estimate_AD_RD_2iso but subtracts three
    isotropic contributions (restricted, hindered, free-water) before the
    WLS step.

    Returns
    -------
    AD_est, RD_est : float
        Estimated diffusivities, or np.nan if the WLS system is singular.
    """
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot

    sum_AA = 0.0; sum_AB = 0.0; sum_BB = 0.0
    sum_Ay = 0.0; sum_By = 0.0

    for i in range(len(bvals)):
        b       = bvals[i]
        S_total = max(sig_norm[i], 0.01)
        S_iso   = (fr * np.exp(-b * D_res) +
                   fh * np.exp(-b * D_hin) +
                   fw * np.exp(-b * D_wat))
        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        S_fiber = max(min(S_fiber, 1.0), 0.01)
        log_S   = np.log(S_fiber)

        g     = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2  = cos_t * cos_t
        w     = S_total * b

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
        m      = (AD_est + RD_est) / 2.0
        AD_est = m; RD_est = m

    return AD_est, RD_est


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ADAPTIVE GRID SEARCH REFINEMENT  (2-ISO and 3-ISO versions)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _refine_AD_RD_2iso(bvals, bvecs, sig_norm, fiber_dir,
                       f_fib, f_res, f_nonrf, D_res, D_nonrf,
                       AD_init, RD_init):
    """
    Coarse-then-fine 2D grid search for (AD, RD) — two-compartment isotropic model.

    The search grid is centred adaptively on (AD_init, RD_init) from the
    analytical initialisation, with range factor scaled by the estimated
    anisotropy to minimise redundant evaluations in near-isotropic voxels.

    Returns np.nan if the initialisation is np.nan (propagates gracefully).
    """
    if np.isnan(AD_init) or np.isnan(RD_init):
        return np.nan, np.nan

    ftot = f_fib + f_res + f_nonrf + 1e-12
    ff = f_fib   / ftot
    fr = f_res   / ftot
    fn = f_nonrf / ftot

    anisotropy   = abs(AD_init - RD_init) / ((AD_init + RD_init) / 2.0 + 1e-12)
    range_factor = 0.25 if anisotropy > 0.5 else (0.35 if anisotropy > 0.2 else 0.50)

    AD_min = max(0.05e-3, AD_init * (1.0 - range_factor))
    AD_max = min(3.5e-3,  AD_init * (1.0 + range_factor))
    RD_min = max(0.05e-3, RD_init * (1.0 - range_factor))
    RD_max = min(3.0e-3,  RD_init * (1.0 + range_factor))

    n_AD = 8; n_RD = 6
    dAD  = (AD_max - AD_min) / (n_AD - 1) if n_AD > 1 else 0.0
    dRD  = (RD_max - RD_min) / (n_RD - 1) if n_RD > 1 else 0.0

    best_sse = 1e20
    best_AD  = AD_init
    best_RD  = RD_init

    # Coarse grid
    for i in range(n_AD):
        AD = AD_min + i * dAD
        for j in range(n_RD):
            RD = RD_min + j * dRD
            if AD < RD:
                continue
            sse = 0.0
            for k in range(len(bvals)):
                b     = bvals[k]
                g     = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res)  +
                          fn * np.exp(-b * D_nonrf))
                diff  = sig_norm[k] - s_pred
                sse  += diff * diff
            if sse < best_sse:
                best_sse = sse; best_AD = AD; best_RD = RD

    # Fine grid
    AD_c = best_AD; RD_c = best_RD
    fine_AD = dAD / 4.0 if dAD > 0.0 else 0.05e-3
    fine_RD = dRD / 4.0 if dRD > 0.0 else 0.05e-3

    for di in range(-2, 3):
        AD = AD_c + di * fine_AD
        if AD < AD_min or AD > AD_max:
            continue
        for dj in range(-2, 3):
            RD = RD_c + dj * fine_RD
            if RD < RD_min or RD > RD_max or AD < RD:
                continue
            sse = 0.0
            for k in range(len(bvals)):
                b     = bvals[k]
                g     = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res)  +
                          fn * np.exp(-b * D_nonrf))
                diff  = sig_norm[k] - s_pred
                sse  += diff * diff
            if sse < best_sse:
                best_sse = sse; best_AD = AD; best_RD = RD

    return best_AD, best_RD


@njit(cache=True, fastmath=True)
def _refine_AD_RD_3iso(bvals, bvecs, sig_norm, fiber_dir,
                       f_fib, f_res, f_hin, f_wat,
                       D_res, D_hin, D_wat,
                       AD_init, RD_init):
    """
    Coarse-then-fine 2D grid search for (AD, RD) — three-compartment isotropic model.

    Same adaptive logic as _refine_AD_RD_2iso but the residual is computed
    against a three-compartment isotropic model.
    """
    if np.isnan(AD_init) or np.isnan(RD_init):
        return np.nan, np.nan

    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot

    anisotropy   = abs(AD_init - RD_init) / ((AD_init + RD_init) / 2.0 + 1e-12)
    range_factor = 0.25 if anisotropy > 0.5 else (0.35 if anisotropy > 0.2 else 0.50)

    AD_min = max(0.05e-3, AD_init * (1.0 - range_factor))
    AD_max = min(3.5e-3,  AD_init * (1.0 + range_factor))
    RD_min = max(0.05e-3, RD_init * (1.0 - range_factor))
    RD_max = min(3.0e-3,  RD_init * (1.0 + range_factor))

    n_AD = 8; n_RD = 6
    dAD  = (AD_max - AD_min) / (n_AD - 1) if n_AD > 1 else 0.0
    dRD  = (RD_max - RD_min) / (n_RD - 1) if n_RD > 1 else 0.0

    best_sse = 1e20
    best_AD  = AD_init
    best_RD  = RD_init

    # Coarse grid
    for i in range(n_AD):
        AD = AD_min + i * dAD
        for j in range(n_RD):
            RD = RD_min + j * dRD
            if AD < RD:
                continue
            sse = 0.0
            for k in range(len(bvals)):
                b     = bvals[k]
                g     = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res)  +
                          fh * np.exp(-b * D_hin)  +
                          fw * np.exp(-b * D_wat))
                diff  = sig_norm[k] - s_pred
                sse  += diff * diff
            if sse < best_sse:
                best_sse = sse; best_AD = AD; best_RD = RD

    # Fine grid
    AD_c = best_AD; RD_c = best_RD
    fine_AD = dAD / 4.0 if dAD > 0.0 else 0.05e-3
    fine_RD = dRD / 4.0 if dRD > 0.0 else 0.05e-3

    for di in range(-2, 3):
        AD = AD_c + di * fine_AD
        if AD < AD_min or AD > AD_max:
            continue
        for dj in range(-2, 3):
            RD = RD_c + dj * fine_RD
            if RD < RD_min or RD > RD_max or AD < RD:
                continue
            sse = 0.0
            for k in range(len(bvals)):
                b     = bvals[k]
                g     = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                s_pred = (ff * np.exp(-b * D_app) +
                          fr * np.exp(-b * D_res)  +
                          fh * np.exp(-b * D_hin)  +
                          fw * np.exp(-b * D_wat))
                diff  = sig_norm[k] - s_pred
                sse  += diff * diff
            if sse < best_sse:
                best_sse = sse; best_AD = AD; best_RD = RD

    return best_AD, best_RD


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL FITTING KERNELS
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_2iso(data, coords, A, AtA, At, bvals, bvecs,
                     fiber_dirs, iso_grid, reg,
                     enable_step2, fiber_threshold, out):
    """
    Numba parallel fitting kernel — two-compartment isotropic model (2-ISO).

    Writes into output channels 0, 1, 4–10.  Channels 2 (HF) and 3 (WF)
    are left at 0.0 (caller pre-initialises them to NaN).

    Output layout
    -------------
    out[..., 0] = FF   (Fibre fraction)
    out[..., 1] = RF   (Restricted fraction, ADC ≤ THRESH_RES)
    out[..., 2] = HF   ← NaN (not estimated in 2-ISO)
    out[..., 3] = WF   ← NaN (not estimated in 2-ISO)
    out[..., 4] = NRF  (Non-Restricted fraction = HF + WF merged)
    out[..., 5] = AD   (NaN if FF ≤ fiber_threshold)
    out[..., 6] = RD   (NaN if FF ≤ fiber_threshold)
    out[..., 7] = FA   (NaN if FF ≤ fiber_threshold)
    out[..., 8] = ADC_iso  (mean isotropic ADC, always valid)
    out[..., 9] = AD_lin   (NaN if FF ≤ fiber_threshold)
    out[...,10] = RD_lin   (NaN if FF ≤ fiber_threshold)
    """
    n_voxels = coords.shape[0]
    n_dirs   = len(fiber_dirs)
    n_iso    = len(iso_grid)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]

        # S₀ normalisation (b ≈ 0 volumes)
        b_min = 1e10
        for i in range(len(bvals)):
            if bvals[i] < b_min:
                b_min = bvals[i]
        b0_thr = b_min + 100.0

        s0  = 0.0; cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < b0_thr:
                s0 += sig[i]; cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue

        sig_norm = sig / s0

        # ── Step 1: regularised NNLS ──────────────────────────────────────
        Aty = np.zeros(AtA.shape[0])
        for r in range(AtA.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val

        w, _ = nnls_coordinate_descent(AtA, Aty, reg)
        w_fib = w[:n_dirs]
        w_iso = w[n_dirs:]

        # Compartment accumulation (single pass over isotropic grid)
        f_fib_raw  = 0.0
        for i in range(n_dirs):
            f_fib_raw += w_fib[i]

        f_res_raw   = 0.0; f_nonrf_raw = 0.0
        sum_w_iso   = 0.0; sum_wd_iso  = 0.0
        sum_res_w   = 0.0; sum_res_wd  = 0.0
        sum_nonrf_w = 0.0; sum_nonrf_wd = 0.0

        for i in range(n_iso):
            adc = iso_grid[i]
            wi  = w_iso[i]
            if adc <= THRESH_RES:
                f_res_raw   += wi
                sum_res_w   += wi
                sum_res_wd  += wi * adc
            else:
                f_nonrf_raw  += wi
                sum_nonrf_w  += wi
                sum_nonrf_wd += wi * adc
            sum_w_iso  += wi
            sum_wd_iso += wi * adc

        mean_iso_adc = sum_wd_iso   / sum_w_iso   if sum_w_iso   > 1e-10 else 0.0
        D_res_c      = sum_res_wd   / sum_res_w   if sum_res_w   > 1e-10 else 0.15e-3
        D_nonrf_c    = sum_nonrf_wd / sum_nonrf_w if sum_nonrf_w > 1e-10 else 1.5e-3

        ftot = f_fib_raw + f_res_raw + f_nonrf_raw
        if ftot < 1e-10:
            continue

        f_fib   = f_fib_raw   / ftot
        f_res   = f_res_raw   / ftot
        f_nonrf = f_nonrf_raw / ftot

        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        # channels 2, 3 remain NaN (pre-initialised by caller)
        out[x, y, z, 4] = f_nonrf
        out[x, y, z, 8] = mean_iso_adc

        # ── Step 2: AD / RD estimation ────────────────────────────────────
        if f_fib > fiber_threshold:

            # Dominant fibre direction
            idx_max = 0; val_max = -1.0
            for i in range(n_dirs):
                if w_fib[i] > val_max:
                    val_max = w_fib[i]; idx_max = i
            fiber_dir = fiber_dirs[idx_max]

            AD_lin, RD_lin = _estimate_AD_RD_2iso(
                bvals, bvecs, sig_norm, fiber_dir,
                f_fib, f_res, f_nonrf, D_res_c, D_nonrf_c
            )

            AD = AD_lin; RD = RD_lin
            if enable_step2:
                AD, RD = _refine_AD_RD_2iso(
                    bvals, bvecs, sig_norm, fiber_dir,
                    f_fib, f_res, f_nonrf, D_res_c, D_nonrf_c,
                    AD_lin, RD_lin
                )

            FA = np.nan
            if not np.isnan(AD) and not np.isnan(RD):
                FA = compute_fiber_fa(AD, RD)

            out[x, y, z, 5]  = AD
            out[x, y, z, 6]  = RD
            out[x, y, z, 7]  = FA
            out[x, y, z, 9]  = AD_lin
            out[x, y, z, 10] = RD_lin


@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_3iso(data, coords, A, AtA, At, bvals, bvecs,
                     fiber_dirs, iso_grid, reg,
                     enable_step2, fiber_threshold, out):
    """
    Numba parallel fitting kernel — three-compartment isotropic model (3-ISO).

    Writes into output channels 0–10.  All channels are valid (no NaN from
    model choice; NaN may still appear in AD/RD/FA where FF ≤ fiber_threshold).

    Output layout
    -------------
    out[..., 0] = FF   (Fibre fraction)
    out[..., 1] = RF   (Restricted,  ADC ≤ THRESH_RES)
    out[..., 2] = HF   (Hindered,    THRESH_RES < ADC ≤ THRESH_WAT)
    out[..., 3] = WF   (Free-water,  ADC > THRESH_WAT)
    out[..., 4] = NRF  (= HF + WF, always consistent)
    out[..., 5] = AD   (NaN if FF ≤ fiber_threshold)
    out[..., 6] = RD   (NaN if FF ≤ fiber_threshold)
    out[..., 7] = FA   (NaN if FF ≤ fiber_threshold)
    out[..., 8] = ADC_iso
    out[..., 9] = AD_lin
    out[...,10] = RD_lin
    """
    n_voxels = coords.shape[0]
    n_dirs   = len(fiber_dirs)
    n_iso    = len(iso_grid)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]

        # S₀ normalisation
        b_min = 1e10
        for i in range(len(bvals)):
            if bvals[i] < b_min:
                b_min = bvals[i]
        b0_thr = b_min + 100.0

        s0 = 0.0; cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < b0_thr:
                s0 += sig[i]; cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue

        sig_norm = sig / s0

        # ── Step 1: regularised NNLS ──────────────────────────────────────
        Aty = np.zeros(AtA.shape[0])
        for r in range(AtA.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val

        w, _ = nnls_coordinate_descent(AtA, Aty, reg)
        w_fib = w[:n_dirs]
        w_iso = w[n_dirs:]

        # Compartment accumulation — three-compartment partition
        f_fib_raw = 0.0
        for i in range(n_dirs):
            f_fib_raw += w_fib[i]

        f_res_raw = 0.0; f_hin_raw = 0.0; f_wat_raw = 0.0
        sum_w_iso  = 0.0; sum_wd_iso = 0.0
        sum_res_w  = 0.0; sum_res_wd = 0.0
        sum_hin_w  = 0.0; sum_hin_wd = 0.0
        sum_wat_w  = 0.0; sum_wat_wd = 0.0

        for i in range(n_iso):
            adc = iso_grid[i]
            wi  = w_iso[i]
            if adc <= THRESH_RES:
                f_res_raw += wi
                sum_res_w += wi; sum_res_wd += wi * adc
            elif adc <= THRESH_WAT:
                f_hin_raw += wi
                sum_hin_w += wi; sum_hin_wd += wi * adc
            else:
                f_wat_raw += wi
                sum_wat_w += wi; sum_wat_wd += wi * adc
            sum_w_iso  += wi
            sum_wd_iso += wi * adc

        mean_iso_adc = sum_wd_iso / sum_w_iso if sum_w_iso > 1e-10 else 0.0
        D_res_c = sum_res_wd / sum_res_w if sum_res_w > 1e-10 else 0.15e-3
        D_hin_c = sum_hin_wd / sum_hin_w if sum_hin_w > 1e-10 else 1.00e-3
        D_wat_c = sum_wat_wd / sum_wat_w if sum_wat_w > 1e-10 else 3.00e-3

        ftot = f_fib_raw + f_res_raw + f_hin_raw + f_wat_raw
        if ftot < 1e-10:
            continue

        f_fib = f_fib_raw / ftot
        f_res = f_res_raw / ftot
        f_hin = f_hin_raw / ftot
        f_wat = f_wat_raw / ftot

        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 2] = f_hin
        out[x, y, z, 3] = f_wat
        out[x, y, z, 4] = f_hin + f_wat    # NRF = HF + WF, always consistent
        out[x, y, z, 8] = mean_iso_adc

        # ── Step 2: AD / RD estimation ────────────────────────────────────
        if f_fib > fiber_threshold:

            # Dominant fibre direction
            idx_max = 0; val_max = -1.0
            for i in range(n_dirs):
                if w_fib[i] > val_max:
                    val_max = w_fib[i]; idx_max = i
            fiber_dir = fiber_dirs[idx_max]

            AD_lin, RD_lin = _estimate_AD_RD_3iso(
                bvals, bvecs, sig_norm, fiber_dir,
                f_fib, f_res, f_hin, f_wat,
                D_res_c, D_hin_c, D_wat_c
            )

            AD = AD_lin; RD = RD_lin
            if enable_step2:
                AD, RD = _refine_AD_RD_3iso(
                    bvals, bvecs, sig_norm, fiber_dir,
                    f_fib, f_res, f_hin, f_wat,
                    D_res_c, D_hin_c, D_wat_c,
                    AD_lin, RD_lin
                )

            FA = np.nan
            if not np.isnan(AD) and not np.isnan(RD):
                FA = compute_fiber_fa(AD, RD)

            out[x, y, z, 5]  = AD
            out[x, y, z, 6]  = RD
            out[x, y, z, 7]  = FA
            out[x, y, z, 9]  = AD_lin
            out[x, y, z, 10] = RD_lin


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DBSI_Adaptive:
    """
    Adaptive DBSI model that automatically selects between the two-compartment
    (2-ISO) and three-compartment (3-ISO) isotropic decomposition based on the
    maximum b-value and the shell diversity of the input acquisition protocol.

    Selection rule
    --------------
    3-ISO (RF + HF + WF) if:  b_max ≥ B_THRESH_3ISO  AND  n_shells ≥ MIN_SHELLS_3ISO
    2-ISO (RF + NRF)     otherwise.

    The rule is evaluated automatically in ``fit()`` and is logged to stdout.
    It can be overridden by passing ``force_n_iso=2`` or ``force_n_iso=3``.

    Parameters
    ----------
    n_iso : int or None
        Number of isotropic ADC basis points (auto-calibrated if None).
    reg_lambda : float or None
        L2 regularisation strength (auto-calibrated if None).
    enable_step2 : bool
        Whether to refine AD/RD via adaptive grid search (Step 2). Default: True.
    n_dirs : int
        Number of fibre directions on the Fibonacci hemisphere. Default: 100.
    iso_range : tuple (float, float)
        ADC range [mm²/s] of the isotropic basis. Default: (0.0, 4.5e-3).
    fiber_threshold : float
        Minimum fibre fraction for AD/RD/FA estimation. Default: 0.15.
    force_n_iso : int or None
        Override automatic model selection (2 or 3). None = automatic.
    """

    # Output channel index map — identical for both model modes
    CH = {
        'FF':       0,
        'RF':       1,
        'HF':       2,   # NaN in 2-ISO mode
        'WF':       3,   # NaN in 2-ISO mode
        'NRF':      4,
        'AD':       5,
        'RD':       6,
        'FA':       7,
        'ADC_iso':  8,
        'AD_lin':   9,
        'RD_lin':   10,
    }
    N_CHANNELS = 11

    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True,
                 n_dirs=100, iso_range=(0.0, 4.5e-3),
                 fiber_threshold=FIBER_THRESHOLD, force_n_iso=None):
        self.n_iso           = n_iso
        self.reg_lambda      = reg_lambda
        self.enable_step2    = enable_step2
        self.n_dirs          = n_dirs
        self.iso_range       = iso_range
        self.fiber_threshold = fiber_threshold
        self.force_n_iso     = force_n_iso

        # Set after fit()
        self.model_mode_     = None    # 2 or 3
        self.b_max_          = None
        self.n_shells_       = None

    # ------------------------------------------------------------------
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fit the adaptive DBSI model to 4D diffusion MRI data.

        Parameters
        ----------
        data : ndarray (X, Y, Z, N)
            Raw DWI signal. Rician bias correction is applied internally.
        bvals : array (N,)
            B-values in s/mm².
        bvecs : array (N, 3) or (3, N)
            Gradient directions (unit vectors; orientation auto-detected).
        mask : ndarray (X, Y, Z) bool
            Brain mask.
        run_calibration : bool
            Whether to run Monte Carlo calibration for (n_iso, reg_lambda).

        Returns
        -------
        results : ndarray (X, Y, Z, 11)
            See class docstring for channel layout.
        model_mode : int
            2 or 3 — the isotropic model actually used.
        """
        print("\n" + "="*70)
        print("  DBSI ADAPTIVE PIPELINE")
        print("="*70)

        # ── Gradient normalisation ─────────────────────────────────────────
        bvecs = np.asarray(bvecs, dtype=np.float64)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms

        # ── Model selection ────────────────────────────────────────────────
        b_max, n_shells, use_3iso, reason = analyse_protocol(bvals)
        self.b_max_    = b_max
        self.n_shells_ = n_shells

        if self.force_n_iso is not None:
            if self.force_n_iso == 3:
                use_3iso = True
                print(f"\n  [MODEL OVERRIDE] force_n_iso=3 requested by user.")
            elif self.force_n_iso == 2:
                use_3iso = False
                print(f"\n  [MODEL OVERRIDE] force_n_iso=2 requested by user.")
            else:
                raise ValueError("force_n_iso must be 2, 3, or None.")

        model_mode = 3 if use_3iso else 2
        self.model_mode_ = model_mode

        print(f"\n  Model selection: {reason}")
        print(f"\n  Active model: {model_mode}-ISO "
              f"({'RF + HF + WF' if use_3iso else 'RF + NRF (HF+WF merged)'})")
        print(f"  b_max detected: {b_max:.0f} s/mm²  |  "
              f"Non-zero shells: {n_shells}")

        # ── SNR estimation ─────────────────────────────────────────────────
        print("\n1. Estimating SNR...")
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)

        # ── Monte Carlo calibration ────────────────────────────────────────
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            print("\n2. Running Monte Carlo Calibration...")
            self.n_iso, self.reg_lambda = optimize_hyperparameters(
                bvals, bvecs, snr, n_mc=1000
            )

        if self.n_iso    is None: self.n_iso    = 60
        if self.reg_lambda is None: self.reg_lambda = 0.05

        print(f"\n   Hyperparameters: n_iso={self.n_iso}, "
              f"λ={self.reg_lambda:.4f}")
        print(f"   Fibre threshold: {self.fiber_threshold:.2f}  "
              f"(AD/RD/FA valid only where FF > {self.fiber_threshold:.2f})")
        _thresh_str = (
            f"RF (ADC ≤ {THRESH_RES*1e3:.1f}×10⁻³ mm²/s) | "
            f"HF ({THRESH_RES*1e3:.1f}–{THRESH_WAT*1e3:.0f}×10⁻³ mm²/s) | "
            f"WF (ADC > {THRESH_WAT*1e3:.0f}×10⁻³ mm²/s)"
            if use_3iso else
            f"RF (ADC ≤ {THRESH_RES*1e3:.1f}×10⁻³ mm²/s) | "
            f"NRF (ADC > {THRESH_RES*1e3:.1f}×10⁻³ mm²/s)"
        )
        print(f"   Compartments: {_thresh_str}")

        # ── Rician bias correction ─────────────────────────────────────────
        print("\n3. Applying Rician Bias Correction...")
        coords    = np.argwhere(mask)
        data_corr = np.zeros_like(data, dtype=np.float32)
        for i in range(len(coords)):
            x, y, z = coords[i]
            data_corr[x, y, z] = correct_rician_bias(data[x, y, z], sigma)

        # ── Design matrix ──────────────────────────────────────────────────
        print("\n4. Building Design Matrix...")
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid   = np.linspace(self.iso_range[0], self.iso_range[1],
                                 self.n_iso)

        A       = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid,
                                      ad=DESIGN_MATRIX_AD, rd=DESIGN_MATRIX_RD)
        AtA     = A.T @ A
        AtA_reg = AtA + self.reg_lambda * np.eye(AtA.shape[0])
        At      = A.T

        cond = np.linalg.cond(AtA_reg)
        print(f"   Design matrix: {A.shape}  |  "
              f"Condition number: {cond:.2e}")

        # ── Allocate output ────────────────────────────────────────────────
        results = np.zeros(data.shape[:3] + (self.N_CHANNELS,), dtype=np.float32)
        # Pre-initialise diffusivity channels to NaN (valid only where FF > thr)
        results[..., 5]  = np.nan   # AD
        results[..., 6]  = np.nan   # RD
        results[..., 7]  = np.nan   # FA
        results[..., 9]  = np.nan   # AD_lin
        results[..., 10] = np.nan   # RD_lin
        # In 2-ISO mode HF and WF are meaningless → NaN
        if not use_3iso:
            results[..., 2] = np.nan   # HF
            results[..., 3] = np.nan   # WF

        # ── Parallel voxel fitting ─────────────────────────────────────────
        n_voxels  = len(coords)
        batch_sz  = 10_000
        n_batches = int(np.ceil(n_voxels / batch_sz))

        print(f"\n5. Fitting {n_voxels:,} voxels "
              f"[{model_mode}-ISO model]...")

        _kernel = _fit_voxels_3iso if use_3iso else _fit_voxels_2iso

        t0 = time.time()
        with tqdm(total=n_voxels, desc="   Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start = i * batch_sz
                end   = min((i + 1) * batch_sz, n_voxels)
                _kernel(
                    data_corr, coords[start:end], A, AtA_reg, At,
                    bvals, bvecs, fiber_dirs, iso_grid,
                    self.reg_lambda, self.enable_step2,
                    self.fiber_threshold, results
                )
                pbar.update(end - start)

        elapsed  = time.time() - t0
        n_fitted = int(np.sum(~np.isnan(results[..., 5]) & mask))
        pct      = n_fitted / n_voxels * 100 if n_voxels > 0 else 0.0

        print(f"\n   Completed: {elapsed:.1f}s  "
              f"({n_voxels / elapsed:.0f} vox/s)")
        print(f"   AD/RD estimated: {n_fitted:,} / {n_voxels:,} "
              f"({pct:.1f}%)")
        print(f"\n{'='*70}\n")

        return results, model_mode

    # ------------------------------------------------------------------
    @staticmethod
    def output_map_names(model_mode):
        """
        Return the ordered list of output map file names for the given model mode.

        Parameters
        ----------
        model_mode : int
            2 or 3 — as returned by fit().

        Returns
        -------
        list of str
        """
        if model_mode == 3:
            return [
                'fiber_fraction',       # ch 0
                'restricted_fraction',  # ch 1
                'hindered_fraction',    # ch 2
                'water_fraction',       # ch 3
                'nonrestricted_fraction',  # ch 4  (= HF + WF)
                'axial_diffusivity',    # ch 5
                'radial_diffusivity',   # ch 6
                'fiber_fa',             # ch 7
                'mean_iso_adc',         # ch 8
                'ad_linear',            # ch 9
                'rd_linear',            # ch 10
            ]
        else:  # 2-ISO
            return [
                'fiber_fraction',          # ch 0
                'restricted_fraction',     # ch 1
                'hindered_fraction_NaN',   # ch 2  (NaN, not estimated)
                'water_fraction_NaN',      # ch 3  (NaN, not estimated)
                'nonrestricted_fraction',  # ch 4
                'axial_diffusivity',       # ch 5
                'radial_diffusivity',      # ch 6
                'fiber_fa',                # ch 7
                'mean_iso_adc',            # ch 8
                'ad_linear',               # ch 9
                'rd_linear',               # ch 10
            ]
