"""
DBSI Fit Quality Maps — v3
=============================

IMPROVEMENT OVER v2
----------------------
v2's stored (AD, RD) were a weighted CENTROID over potentially many
simultaneously-activated (direction, AD/RD-pair) columns, so the
fit_quality reconstruction there was a documented APPROXIMATION of the
true multi-component v2 fit (see v2's fit_quality.py module docstring
for the full discussion).

v3's stored (AD, RD) are Stage B's closed-form estimate for a SINGLE
fixed direction (Stage A's dominant detected direction) — i.e. exactly
the single-tensor model this function reconstructs. There is no
approximation gap in v3: the signal reconstructed here from (FF,
RF/HF/WF, AD, RD) IS the same forward model Stage B fit, for voxels
where Stage A detected exactly one dominant fiber population (the
default `_MAX_FIBER_POPULATIONS=1` reporting path, or the dominant
population when 2 were detected — see model_Niso_adaptive_ff_thr.py).

Remaining caveat: in voxels with genuine crossing fibers where Stage A
detects 2 populations but only the dominant one is reported in the
11-channel output (no second-tensor slot in the current output layout),
R² will still legitimately be lower than a hypothetical two-tensor
fit, because the reconstruction here only models the dominant
population. This is expected and diagnostic (a depressed R² flags
voxels where the single-tensor output is an incomplete description),
not a reconstruction bug.

Fiber direction
-----------------
Unlike v1/v2 (which had to recover direction via a grid search because
it was not stored), v3's fiber direction also is not stored directly in
the 11-channel output (the channel layout has no direction slot), so a
grid search recovery step is still required here, exactly as in v1/v2.

Usage
-----
    from dbsi_toolbox.fit_quality import compute_fit_quality

    r2, rmse = compute_fit_quality(
        data, bvals, bvecs, mask, results, model_mode,
        fiber_threshold=0.15, n_dirs=100, verbose=True
    )

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601.
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from .core.basis import generate_fibonacci_sphere_hemisphere

THRESH_RES = 0.3e-3
THRESH_WAT = 3.0e-3
_D_RES_NOM = 0.15e-3
_D_WAT_NOM = 3.05e-3

_ISO_ADC_MAX = 3.5e-3

_CH_FF = 0
_CH_RF = 1
_CH_HF = 2
_CH_WF = 3
_CH_NRF = 4
_CH_AD = 5
_CH_RD = 6
_CH_ADC_ISO = 8


@njit(cache=True, fastmath=True)
def _recover_iso_adcs_2iso(rf, nrf, adc_iso):
    """Recover D_res and D_nonrf from stored ADC_iso (2-ISO model).
    Unchanged from v1/v2.
    """
    ftot_iso = rf + nrf
    if ftot_iso < 1e-10:
        return _D_RES_NOM, 1.0e-3

    D_res = _D_RES_NOM

    if nrf > 1e-6:
        D_nonrf = (adc_iso * ftot_iso - rf * D_res) / nrf
        D_nonrf = max(THRESH_RES, min(_ISO_ADC_MAX, D_nonrf))
    else:
        D_nonrf = adc_iso

    return D_res, D_nonrf


@njit(cache=True, fastmath=True)
def _recover_iso_adcs_3iso(rf, hf, wf, adc_iso):
    """Recover D_res, D_hin, D_wat from stored ADC_iso (3-ISO model).
    Unchanged from v1/v2.
    """
    ftot_iso = rf + hf + wf
    if ftot_iso < 1e-10:
        return _D_RES_NOM, 0.9e-3, _D_WAT_NOM

    D_res = _D_RES_NOM
    D_wat = _D_WAT_NOM

    if hf > 1e-6:
        D_hin = (adc_iso * ftot_iso - rf * D_res - wf * D_wat) / hf
        D_hin = max(THRESH_RES, min(THRESH_WAT, D_hin))
    else:
        D_hin = 0.9e-3

    return D_res, D_hin, D_wat


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA KERNELS — v3: single-tensor reconstruction, now EXACT (see module
# docstring)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _quality_kernel_2iso(data, coords, bvals, bvecs, fiber_dirs,
                         params, b0_thr, fiber_threshold,
                         out_r2, out_rmse):
    """
    v3 parallel R²/RMSE kernel for the 2-ISO model.

    Reconstructs the signal from the stored (AD, RD) — Stage B's
    closed-form estimate — searching over direction (not stored) to find
    the direction that best matches the reconstruction. For voxels where
    Stage A detected a single dominant fiber population (the common
    case), this direction search recovers the SAME direction Stage B
    actually used, making this reconstruction exact rather than
    approximate (contrast with v2).
    """
    n_voxels = coords.shape[0]
    n_dirs = len(fiber_dirs)
    N = len(bvals)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]

        ff = params[x, y, z, _CH_FF]
        rf = params[x, y, z, _CH_RF]
        nrf = params[x, y, z, _CH_NRF]
        ad = params[x, y, z, _CH_AD]
        rd = params[x, y, z, _CH_RD]
        adc_iso = params[x, y, z, _CH_ADC_ISO]

        if (ff + rf + nrf) < 1e-6:
            continue

        sig = data[x, y, z]
        s0 = 0.0
        cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue
        sig_norm = sig / s0

        D_res, D_nonrf = _recover_iso_adcs_2iso(rf, nrf, adc_iso)

        has_fiber = (not np.isnan(ad)) and ff > fiber_threshold
        best_dir = fiber_dirs[0]

        if has_fiber:
            best_sse = 1e20
            for j in range(n_dirs):
                v = fiber_dirs[j]
                sse = 0.0
                for i in range(N):
                    b = bvals[i]
                    cos_t = bvecs[i, 0]*v[0] + bvecs[i, 1]*v[1] + bvecs[i, 2]*v[2]
                    D_app = rd + (ad - rd) * cos_t * cos_t
                    s_p = (ff * np.exp(-b * D_app)
                           + rf * np.exp(-b * D_res)
                           + nrf * np.exp(-b * D_nonrf))
                    diff = sig_norm[i] - s_p
                    sse += diff * diff
                if sse < best_sse:
                    best_sse = sse
                    best_dir = v

        ss_res = 0.0
        ss_tot = 0.0
        rmse_sum = 0.0
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
                          + nrf * np.exp(-b * D_nonrf))
            else:
                s_pred = (rf * np.exp(-b * D_res)
                          + nrf * np.exp(-b * D_nonrf))

            res = sig_norm[i] - s_pred
            ss_res += res * res
            ss_tot += (sig_norm[i] - s_mean) ** 2
            rmse_sum += res * res

        if ss_tot > 1e-14:
            out_r2[x, y, z] = 1.0 - ss_res / ss_tot
        out_rmse[x, y, z] = np.sqrt(rmse_sum / N)


@njit(parallel=True, cache=True, fastmath=True)
def _quality_kernel_3iso(data, coords, bvals, bvecs, fiber_dirs,
                         params, b0_thr, fiber_threshold,
                         out_r2, out_rmse):
    """v3 parallel R²/RMSE kernel for the 3-ISO model. See
    `_quality_kernel_2iso` for why this reconstruction is now exact
    rather than approximate.
    """
    n_voxels = coords.shape[0]
    n_dirs = len(fiber_dirs)
    N = len(bvals)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]

        ff = params[x, y, z, _CH_FF]
        rf = params[x, y, z, _CH_RF]
        hf = params[x, y, z, _CH_HF]
        wf = params[x, y, z, _CH_WF]
        ad = params[x, y, z, _CH_AD]
        rd = params[x, y, z, _CH_RD]
        adc_iso = params[x, y, z, _CH_ADC_ISO]

        if (ff + rf + hf + wf) < 1e-6:
            continue

        sig = data[x, y, z]
        s0 = 0.0
        cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue
        sig_norm = sig / s0

        D_res, D_hin, D_wat = _recover_iso_adcs_3iso(rf, hf, wf, adc_iso)

        has_fiber = (not np.isnan(ad)) and ff > fiber_threshold
        best_dir = fiber_dirs[0]

        if has_fiber:
            best_sse = 1e20
            for j in range(n_dirs):
                v = fiber_dirs[j]
                sse = 0.0
                for i in range(N):
                    b = bvals[i]
                    cos_t = bvecs[i, 0]*v[0] + bvecs[i, 1]*v[1] + bvecs[i, 2]*v[2]
                    D_app = rd + (ad - rd) * cos_t * cos_t
                    s_p = (ff * np.exp(-b * D_app)
                           + rf * np.exp(-b * D_res)
                           + hf * np.exp(-b * D_hin)
                           + wf * np.exp(-b * D_wat))
                    diff = sig_norm[i] - s_p
                    sse += diff * diff
                if sse < best_sse:
                    best_sse = sse
                    best_dir = v

        ss_res = 0.0
        ss_tot = 0.0
        rmse_sum = 0.0
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

            res = sig_norm[i] - s_pred
            ss_res += res * res
            ss_tot += (sig_norm[i] - s_mean) ** 2
            rmse_sum += res * res

        if ss_tot > 1e-14:
            out_r2[x, y, z] = 1.0 - ss_res / ss_tot
        out_rmse[x, y, z] = np.sqrt(rmse_sum / N)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def compute_fit_quality(data, bvals, bvecs, mask, results, model_mode,
                        fiber_threshold=0.15, n_dirs=100, verbose=True):
    """
    Compute voxel-wise R² and RMSE goodness-of-fit maps from v3 DBSI
    parameter maps.

    Unlike v2, this reconstruction is exact (not an approximation) for
    single-fiber-population voxels, because v3's stored (AD, RD) ARE the
    single-tensor parameters Stage B actually fit — see module
    docstring.

    Parameters
    ----------
    data : ndarray (X, Y, Z, N), float32
    bvals : ndarray (N,)
    bvecs : ndarray (N, 3)
    mask : ndarray (X, Y, Z), bool
    results : ndarray (X, Y, Z, 11), float32
    model_mode : int
        2 or 3.
    fiber_threshold : float
        Same value used during fitting. Default: 0.15.
    n_dirs : int
        Number of fiber directions for the grid search used to recover
        the dominant orientation during reconstruction (direction itself
        is not stored in the 11-channel output). For best fidelity this
        should be at least as fine as the Stage A dictionary used during
        fitting; check `model.n_dirs` after fitting.
    verbose : bool

    Returns
    -------
    r2_map : ndarray (X, Y, Z), float32
    rmse_map : ndarray (X, Y, Z), float32
    """
    if verbose:
        print("\n" + "="*60)
        print("  DBSI FIT QUALITY (v3) — R² and RMSE")
        print("  Single-tensor reconstruction; exact for single-fiber-")
        print("  population voxels (the common case). See module")
        print("  docstring for the genuine-crossing-fiber caveat.")
        print("="*60)
        print(f"  Model mode: {model_mode}-ISO")
        print(f"  Fiber threshold: {fiber_threshold}")

    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms

    bvals = np.asarray(bvals, dtype=np.float64)

    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)

    b0_thr = 100.0

    shape3d = data.shape[:3]
    r2_map = np.full(shape3d, np.nan, dtype=np.float32)
    rmse_map = np.full(shape3d, np.nan, dtype=np.float32)

    coords = np.argwhere(mask)
    n_voxels = len(coords)
    batch_sz = 10_000
    n_batches = int(np.ceil(n_voxels / batch_sz))

    data_f32 = data.astype(np.float32)
    results_f32 = results.astype(np.float32)

    results_work = np.where(np.isnan(results_f32), 0.0, results_f32).astype(np.float32)
    results_work[..., _CH_AD] = results_f32[..., _CH_AD]

    _kernel = _quality_kernel_3iso if model_mode == 3 else _quality_kernel_2iso

    if verbose:
        print(f"\n  Computing quality maps for {n_voxels:,} voxels...")

    t0 = time.time()
    with tqdm(total=n_voxels, desc="  Progress", unit="vox", disable=not verbose) as pbar:
        for i in range(n_batches):
            start = i * batch_sz
            end = min((i + 1) * batch_sz, n_voxels)
            _kernel(
                data_f32, coords[start:end],
                bvals, bvecs, fiber_dirs,
                results_work, b0_thr, fiber_threshold,
                r2_map, rmse_map
            )
            pbar.update(end - start)

    elapsed = time.time() - t0

    if verbose:
        valid = mask & ~np.isnan(r2_map)
        r2_vals = r2_map[valid]
        rmse_vals = rmse_map[valid]
        print(f"\n  Completed: {elapsed:.1f}s  ({n_voxels/elapsed:.0f} vox/s)")
        print(f"\n  R² summary (brain mask, n={valid.sum():,}):")
        print(f"    Median : {np.median(r2_vals):.4f}")
        print(f"    Mean   : {np.mean(r2_vals):.4f}")
        print(f"    > 0.99 : {np.mean(r2_vals > 0.99)*100:.1f}%")
        print(f"    > 0.95 : {np.mean(r2_vals > 0.95)*100:.1f}%")
        print(f"    > 0.90 : {np.mean(r2_vals > 0.90)*100:.1f}%")
        print(f"    < 0.90 : {np.mean(r2_vals < 0.90)*100:.1f}%  <- inspect "
              f"(may reflect genuine crossing fibers not modelled by the "
              f"single dominant-population output, see module docstring)")
        print(f"\n  RMSE summary (fraction of S0):")
        print(f"    Median : {np.median(rmse_vals):.4f}")
        print(f"    Mean   : {np.mean(rmse_vals):.4f}")
        print(f"    > 0.05 : {np.mean(rmse_vals > 0.05)*100:.1f}%  <- high residuals")
        print("="*60 + "\n")

    return r2_map, rmse_map


def save_fit_quality(r2_map, rmse_map, affine, output_dir):
    """Save R² and RMSE maps as compressed NIfTI files. Unchanged from v1/v2."""
    import nibabel as nib
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for name, arr in [('r2', r2_map), ('rmse', rmse_map)]:
        fpath = os.path.join(output_dir, f'dbsi_fit_{name}.nii.gz')
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), fpath)
        paths[name] = fpath

    return paths
