"""
DBSI Fit Quality Maps
=====================

Post-processing module that computes voxel-wise goodness-of-fit metrics from
the parameter maps produced by DBSI_Adaptive.fit().

Two scalar maps are produced per voxel:

    R²   =  1  −  SS_res / SS_tot
          =  1  −  Σ(S_obs − S_pred)²  /  Σ(S_obs − S̄_obs)²

         Unitless, range (−∞, 1].  R² = 1 → perfect fit.
         Values near 0 indicate the model explains no more variance than the
         mean signal; negative values indicate the model is worse than the mean
         (pathological, typically in near-zero-signal voxels).

    RMSE =  √[ Σ(S_obs − S_pred)² / N ]   (in the same units as S_obs)

         Absolute residual. When S_obs has been normalised by S₀ (as in the
         fitting kernel), RMSE is dimensionless and directly interpretable as
         a fraction of S₀.

Both metrics are computed on the normalised signal (S/S₀), consistently with
the fitting step.

Signal reconstruction
---------------------
The predicted signal at each voxel is reconstructed from the stored parameter
maps using the same forward model as the fitting kernel:

    S_pred(b,g)/S₀ = FF · exp(−b · D_app(g, fdir, AD, RD))
                   + RF · exp(−b · D_res)
                   + compartment_iso_terms

where D_app = RD + (AD−RD)·cos²θ and the isotropic centroid ADCs are
recovered analytically from ADC_iso and the stored fractions (see
_recover_iso_adcs_2iso and _recover_iso_adcs_3iso).

For voxels where FF ≤ fiber_threshold (AD/RD are NaN), the fiber term is
omitted and the signal is reconstructed from the isotropic compartments only.
R² on isotropic-only voxels reflects the quality of the isotropic
decomposition.

Fiber direction
---------------
The fiber direction is not stored in the output maps.  It is recovered by a
fast grid search over the same Fibonacci hemisphere used during fitting, with
the coarseness appropriate for quality assessment (not for precision
estimation).

Usage
-----
    from dbsi_toolbox.fit_quality import compute_fit_quality

    r2, rmse = compute_fit_quality(
        data, bvals, bvecs, mask, results, model_mode,
        fiber_threshold=0.15, n_dirs=100, verbose=True
    )

    # Save maps
    import nibabel as nib
    nib.save(nib.Nifti1Image(r2,   affine), 'dbsi_r2.nii.gz')
    nib.save(nib.Nifti1Image(rmse, affine), 'dbsi_rmse.nii.gz')

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590–3601.
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from .core.basis import generate_fibonacci_sphere_hemisphere

# ── Constants — must mirror model_Niso_adaptive_ff_thr.py exactly ────────────
THRESH_RES  = 0.3e-3    # mm²/s
THRESH_WAT  = 3.0e-3    # mm²/s
_D_RES_NOM  = 0.15e-3   # nominal restricted centroid when RF≈0
_D_WAT_NOM  = 3.05e-3   # free water ADC at 37°C

# Output channel indices — must match DBSI_Adaptive.CH
_CH_FF      = 0
_CH_RF      = 1
_CH_HF      = 2
_CH_WF      = 3
_CH_NRF     = 4
_CH_AD      = 5
_CH_RD      = 6
_CH_ADC_ISO = 8


# ─────────────────────────────────────────────────────────────────────────────
# ISOTROPIC ADC RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _recover_iso_adcs_2iso(rf, nrf, adc_iso):
    """
    Recover D_res and D_nonrf from stored ADC_iso (2-ISO model).

    ADC_iso is the weighted centroid of all isotropic pools:
        ADC_iso = (RF·D_res + NRF·D_nonrf) / (RF + NRF)

    Solving for D_nonrf given D_res = _D_RES_NOM:
        D_nonrf = (ADC_iso·(RF+NRF) − RF·D_res) / NRF

    D_res is not stored but is always close to _D_RES_NOM since the RF pool
    is constrained to ADC ≤ 0.3e-3 mm²/s during fitting.

    Returns
    -------
    D_res, D_nonrf : float  (mm²/s)
    """
    ftot_iso = rf + nrf
    if ftot_iso < 1e-10:
        return _D_RES_NOM, 1.0e-3

    D_res = _D_RES_NOM

    if nrf > 1e-6:
        D_nonrf = (adc_iso * ftot_iso - rf * D_res) / nrf
        # Clamp to physiological range (hindered + free water combined)
        D_nonrf = max(THRESH_RES, min(_D_WAT_NOM, D_nonrf))
    else:
        D_nonrf = adc_iso

    return D_res, D_nonrf


@njit(cache=True, fastmath=True)
def _recover_iso_adcs_3iso(rf, hf, wf, adc_iso):
    """
    Recover D_res, D_hin, D_wat from stored ADC_iso (3-ISO model).

    ADC_iso = (RF·D_res + HF·D_hin + WF·D_wat) / (RF + HF + WF)

    With D_res = _D_RES_NOM and D_wat = _D_WAT_NOM fixed, D_hin is solved:
        D_hin = (ADC_iso·(RF+HF+WF) − RF·D_res − WF·D_wat) / HF

    Returns
    -------
    D_res, D_hin, D_wat : float  (mm²/s)
    """
    ftot_iso = rf + hf + wf
    if ftot_iso < 1e-10:
        return _D_RES_NOM, 0.9e-3, _D_WAT_NOM

    D_res = _D_RES_NOM
    D_wat = _D_WAT_NOM

    if hf > 1e-6:
        D_hin = (adc_iso * ftot_iso - rf * D_res - wf * D_wat) / hf
        # Clamp to hindered range
        D_hin = max(THRESH_RES, min(THRESH_WAT, D_hin))
    else:
        # No hindered pool — distribute remainder to free/restricted
        D_hin = 0.9e-3

    return D_res, D_hin, D_wat


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA KERNELS
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _quality_kernel_2iso(data, coords, bvals, bvecs, fiber_dirs,
                         params, b0_thr, fiber_threshold,
                         out_r2, out_rmse):
    """
    Parallel R²/RMSE kernel for the 2-ISO model.

    Parameters
    ----------
    data    : (X, Y, Z, N) float32
    coords  : (V, 3) int   — brain mask voxel coordinates
    bvals   : (N,) float64
    bvecs   : (N, 3) float64
    fiber_dirs : (M, 3) float64
    params  : (X, Y, Z, 11) float32  — DBSI_Adaptive output
    b0_thr  : float
    fiber_threshold : float
    out_r2, out_rmse : (X, Y, Z) float32  — pre-allocated outputs
    """
    n_voxels = coords.shape[0]
    n_dirs   = len(fiber_dirs)
    N        = len(bvals)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]

        # ── Retrieve stored parameters ────────────────────────────────────
        ff      = params[x, y, z, _CH_FF]
        rf      = params[x, y, z, _CH_RF]
        nrf     = params[x, y, z, _CH_NRF]
        ad      = params[x, y, z, _CH_AD]
        rd      = params[x, y, z, _CH_RD]
        adc_iso = params[x, y, z, _CH_ADC_ISO]

        # Skip voxels with no signal decomposition
        if (ff + rf + nrf) < 1e-6:
            continue

        # ── S₀ normalisation ──────────────────────────────────────────────
        sig = data[x, y, z]
        s0 = 0.0; cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]; cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue
        sig_norm = sig / s0

        # ── Recover isotropic ADCs from stored ADC_iso ────────────────────
        D_res, D_nonrf = _recover_iso_adcs_2iso(rf, nrf, adc_iso)

        # ── Fiber direction: grid search if FF > threshold ────────────────
        has_fiber = (not (ad != ad)) and ff > fiber_threshold  # NaN check
        best_dir  = fiber_dirs[0]

        if has_fiber:
            best_sse = 1e20
            for j in range(n_dirs):
                v   = fiber_dirs[j]
                sse = 0.0
                for i in range(N):
                    b     = bvals[i]
                    cos_t = bvecs[i, 0]*v[0] + bvecs[i, 1]*v[1] + bvecs[i, 2]*v[2]
                    D_app = rd + (ad - rd) * cos_t * cos_t
                    s_p   = (ff  * np.exp(-b * D_app)
                             + rf  * np.exp(-b * D_res)
                             + nrf * np.exp(-b * D_nonrf))
                    diff  = sig_norm[i] - s_p
                    sse  += diff * diff
                if sse < best_sse:
                    best_sse = sse
                    best_dir = v

        # ── Reconstruct signal and compute R², RMSE ───────────────────────
        ss_res = 0.0; ss_tot = 0.0; rmse_sum = 0.0
        s_mean = 0.0
        for i in range(N):
            s_mean += sig_norm[i]
        s_mean /= N

        for i in range(N):
            b = bvals[i]

            if has_fiber:
                cos_t = (bvecs[i, 0]*best_dir[0]
                         + bvecs[i, 1]*best_dir[1]
                         + bvecs[i, 2]*best_dir[2])
                D_app = rd + (ad - rd) * cos_t * cos_t
                s_pred = (ff  * np.exp(-b * D_app)
                          + rf  * np.exp(-b * D_res)
                          + nrf * np.exp(-b * D_nonrf))
            else:
                # Pure isotropic prediction (no fiber term)
                s_pred = (rf  * np.exp(-b * D_res)
                          + nrf * np.exp(-b * D_nonrf))

            res     = sig_norm[i] - s_pred
            ss_res += res * res
            ss_tot += (sig_norm[i] - s_mean) ** 2
            rmse_sum += res * res

        if ss_tot > 1e-14:
            out_r2[x, y, z]   = 1.0 - ss_res / ss_tot
        out_rmse[x, y, z] = np.sqrt(rmse_sum / N)


@njit(parallel=True, cache=True, fastmath=True)
def _quality_kernel_3iso(data, coords, bvals, bvecs, fiber_dirs,
                         params, b0_thr, fiber_threshold,
                         out_r2, out_rmse):
    """
    Parallel R²/RMSE kernel for the 3-ISO model.

    Parameters
    ----------
    Same as _quality_kernel_2iso. Additionally reads CH_HF and CH_WF from
    params to distinguish hindered from free-water fractions.
    """
    n_voxels = coords.shape[0]
    n_dirs   = len(fiber_dirs)
    N        = len(bvals)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]

        ff      = params[x, y, z, _CH_FF]
        rf      = params[x, y, z, _CH_RF]
        hf      = params[x, y, z, _CH_HF]
        wf      = params[x, y, z, _CH_WF]
        ad      = params[x, y, z, _CH_AD]
        rd      = params[x, y, z, _CH_RD]
        adc_iso = params[x, y, z, _CH_ADC_ISO]

        if (ff + rf + hf + wf) < 1e-6:
            continue

        sig = data[x, y, z]
        s0 = 0.0; cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]; cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue
        sig_norm = sig / s0

        D_res, D_hin, D_wat = _recover_iso_adcs_3iso(rf, hf, wf, adc_iso)

        has_fiber = (not (ad != ad)) and ff > fiber_threshold
        best_dir  = fiber_dirs[0]

        if has_fiber:
            best_sse = 1e20
            for j in range(n_dirs):
                v   = fiber_dirs[j]
                sse = 0.0
                for i in range(N):
                    b     = bvals[i]
                    cos_t = bvecs[i, 0]*v[0] + bvecs[i, 1]*v[1] + bvecs[i, 2]*v[2]
                    D_app = rd + (ad - rd) * cos_t * cos_t
                    s_p   = (ff * np.exp(-b * D_app)
                             + rf * np.exp(-b * D_res)
                             + hf * np.exp(-b * D_hin)
                             + wf * np.exp(-b * D_wat))
                    diff  = sig_norm[i] - s_p
                    sse  += diff * diff
                if sse < best_sse:
                    best_sse = sse
                    best_dir = v

        ss_res = 0.0; ss_tot = 0.0; rmse_sum = 0.0
        s_mean = 0.0
        for i in range(N):
            s_mean += sig_norm[i]
        s_mean /= N

        for i in range(N):
            b = bvals[i]
            if has_fiber:
                cos_t = (bvecs[i, 0]*best_dir[0]
                         + bvecs[i, 1]*best_dir[1]
                         + bvecs[i, 2]*best_dir[2])
                D_app = rd + (ad - rd) * cos_t * cos_t
                s_pred = (ff * np.exp(-b * D_app)
                          + rf * np.exp(-b * D_res)
                          + hf * np.exp(-b * D_hin)
                          + wf * np.exp(-b * D_wat))
            else:
                s_pred = (rf * np.exp(-b * D_res)
                          + hf * np.exp(-b * D_hin)
                          + wf * np.exp(-b * D_wat))

            res      = sig_norm[i] - s_pred
            ss_res  += res * res
            ss_tot  += (sig_norm[i] - s_mean) ** 2
            rmse_sum += res * res

        if ss_tot > 1e-14:
            out_r2[x, y, z]   = 1.0 - ss_res / ss_tot
        out_rmse[x, y, z] = np.sqrt(rmse_sum / N)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def compute_fit_quality(data, bvals, bvecs, mask, results, model_mode,
                        fiber_threshold=0.15, n_dirs=100, verbose=True):
    """
    Compute voxel-wise R² and RMSE goodness-of-fit maps from DBSI parameter maps.

    This function is designed to be called *after* DBSI_Adaptive.fit() has
    completed and the parameter maps have been saved or are still in memory.
    It does not require re-running the fitting step.

    Parameters
    ----------
    data : ndarray (X, Y, Z, N), float32
        Raw DWI signal (same array passed to DBSI_Adaptive.fit()).
        Rician bias correction is NOT re-applied here; the function uses the
        raw signal for R² computation, consistent with how R² is conventionally
        reported (against the measured, not corrected, data).
    bvals : ndarray (N,)
        B-values in s/mm².
    bvecs : ndarray (N, 3)
        Normalized gradient directions.
    mask : ndarray (X, Y, Z), bool
        Brain mask.
    results : ndarray (X, Y, Z, 11), float32
        Output of DBSI_Adaptive.fit() — the 11-channel parameter map array.
    model_mode : int
        2 or 3 — as returned by DBSI_Adaptive.fit().
    fiber_threshold : float
        Same value used during fitting. Voxels with FF ≤ fiber_threshold are
        reconstructed from isotropic compartments only (AD/RD are NaN).
        Default: 0.15.
    n_dirs : int
        Number of fiber directions for the grid search used to find the
        dominant fiber orientation during reconstruction. Should match the
        value used during fitting (default: 100).
    verbose : bool
        Print progress. Default: True.

    Returns
    -------
    r2_map : ndarray (X, Y, Z), float32
        Voxel-wise R². Initialized to NaN; written only for valid brain mask
        voxels with non-zero signal.
    rmse_map : ndarray (X, Y, Z), float32
        Voxel-wise RMSE in units of S/S₀ (dimensionless). Initialized to NaN.

    Notes
    -----
    R² is computed on the normalised signal S/S₀ in signal space (not
    log-space), consistent with the SSE objective minimised during fitting.

    For voxels where 0 < FF ≤ fiber_threshold, the fiber term is omitted
    from the reconstruction (AD/RD are NaN in these voxels). The resulting R²
    reflects only the isotropic decomposition quality and will be slightly
    below 1.0 even on noise-free data by an amount proportional to the
    omitted FF. This is correct and expected behavior: it provides a
    diagnostic signal that these voxels contain a small but unmodelled fiber
    contribution.

    Interpretation guidelines:
        R² > 0.99  : excellent fit
        R² > 0.95  : good fit
        R² > 0.90  : acceptable fit
        R² < 0.90  : inspect manually; possible model misspecification,
                     motion, gibbs ringing, or B1 inhomogeneity

    RMSE is expressed as a fraction of S₀. Typical values in well-fitted WM
    voxels are 0.01–0.03 (1–3% of S₀).
    """
    if verbose:
        print("\n" + "="*60)
        print("  DBSI FIT QUALITY — R² and RMSE")
        print("="*60)
        print(f"  Model mode: {model_mode}-ISO")
        print(f"  Fiber threshold: {fiber_threshold}")

    # ── Input validation ──────────────────────────────────────────────────────
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms

    bvals = np.asarray(bvals, dtype=np.float64)

    # ── Fiber directions (same hemisphere as fitting) ─────────────────────────
    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)

    # ── Fixed b0 threshold (consistent with fit()) ───────────────────────────
    b0_thr = 100.0

    # ── Allocate outputs ──────────────────────────────────────────────────────
    shape3d = data.shape[:3]
    r2_map   = np.full(shape3d, np.nan, dtype=np.float32)
    rmse_map = np.full(shape3d, np.nan, dtype=np.float32)

    # ── Parallel kernel ───────────────────────────────────────────────────────
    coords    = np.argwhere(mask)
    n_voxels  = len(coords)
    batch_sz  = 10_000
    n_batches = int(np.ceil(n_voxels / batch_sz))

    data_f32    = data.astype(np.float32)
    results_f32 = results.astype(np.float32)

    # Replace NaN in results with 0 for Numba (NaN check inside kernel handles
    # the fiber/no-fiber logic correctly via the `ad != ad` idiom)
    results_work = np.where(np.isnan(results_f32), 0.0, results_f32).astype(np.float32)
    # But keep original NaN pattern for AD channel so kernel detects no-fiber
    results_work[..., _CH_AD] = results_f32[..., _CH_AD]

    _kernel = _quality_kernel_3iso if model_mode == 3 else _quality_kernel_2iso

    if verbose:
        print(f"\n  Computing quality maps for {n_voxels:,} voxels...")

    t0 = time.time()
    with tqdm(total=n_voxels, desc="  Progress", unit="vox", disable=not verbose) as pbar:
        for i in range(n_batches):
            start = i * batch_sz
            end   = min((i + 1) * batch_sz, n_voxels)
            _kernel(
                data_f32, coords[start:end],
                bvals, bvecs, fiber_dirs,
                results_work, b0_thr, fiber_threshold,
                r2_map, rmse_map
            )
            pbar.update(end - start)

    elapsed = time.time() - t0

    # ── Summary statistics ────────────────────────────────────────────────────
    if verbose:
        valid = mask & ~np.isnan(r2_map)
        r2_vals   = r2_map[valid]
        rmse_vals = rmse_map[valid]
        print(f"\n  Completed: {elapsed:.1f}s  ({n_voxels/elapsed:.0f} vox/s)")
        print(f"\n  R² summary (brain mask, n={valid.sum():,}):")
        print(f"    Median : {np.median(r2_vals):.4f}")
        print(f"    Mean   : {np.mean(r2_vals):.4f}")
        print(f"    > 0.99 : {np.mean(r2_vals > 0.99)*100:.1f}%")
        print(f"    > 0.95 : {np.mean(r2_vals > 0.95)*100:.1f}%")
        print(f"    > 0.90 : {np.mean(r2_vals > 0.90)*100:.1f}%")
        print(f"    < 0.90 : {np.mean(r2_vals < 0.90)*100:.1f}%  ← inspect")
        print(f"\n  RMSE summary (fraction of S₀):")
        print(f"    Median : {np.median(rmse_vals):.4f}")
        print(f"    Mean   : {np.mean(rmse_vals):.4f}")
        print(f"    > 0.05 : {np.mean(rmse_vals > 0.05)*100:.1f}%  ← high residuals")
        print("="*60 + "\n")

    return r2_map, rmse_map


def save_fit_quality(r2_map, rmse_map, affine, output_dir):
    """
    Save R² and RMSE maps as compressed NIfTI files.

    Parameters
    ----------
    r2_map, rmse_map : ndarray (X, Y, Z), float32
        Maps returned by compute_fit_quality().
    affine : ndarray (4, 4)
        NIfTI affine matrix (same as the input DWI image).
    output_dir : str
        Directory where the files will be written (must exist).

    Returns
    -------
    paths : dict
        {'r2': path_to_r2_nii, 'rmse': path_to_rmse_nii}
    """
    import nibabel as nib
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for name, arr in [('r2', r2_map), ('rmse', rmse_map)]:
        fpath = os.path.join(output_dir, f'dbsi_fit_{name}.nii.gz')
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), fpath)
        paths[name] = fpath

    return paths
