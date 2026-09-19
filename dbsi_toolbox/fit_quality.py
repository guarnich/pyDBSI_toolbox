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

# Channel indices come from the model, which is the single source of truth for
# the output layout -- they used to be redeclared here, which is exactly how a
# layout change silently corrupts a downstream reader. The multi-population
# reconstruction below needs the whole per-population block so that genuine
# crossings are modelled with BOTH their populations (stored directions and
# per-population fractions/tensors), not only the dominant one.
from .model_Niso_adaptive_ff_thr import (          # noqa: E402
    _C_FF as _CH_FF, _C_RF as _CH_RF, _C_HF as _CH_HF, _C_WF as _CH_WF,
    _C_NRF as _CH_NRF, _C_ADC_ISO as _CH_ADC_ISO, _C_NPOP as _CH_NPOP,
    _C_FF1 as _CH_FF1, _C_AD1 as _CH_AD, _C_RD1 as _CH_RD, _C_DIR1 as _CH_DIR1,
    _C_FF2 as _CH_FF2, _C_AD2 as _CH_AD2, _C_RD2 as _CH_RD2,
    _C_DIR2 as _CH_DIR2,
)


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
# NUMBA KERNEL — MULTI-POPULATION reconstruction (Point B): models ALL detected
# fiber populations from their STORED directions/tensors + the isotropic block,
# so genuine crossings no longer depress R² as an artefact of the previous
# dominant-population-only reconstruction. No direction grid search needed.
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)   # NOT fastmath: this kernel relies on np.isnan
def _quality_kernel_multipop(data, coords, bvals, bvecs, params,     # to detect absent
                             b0_thr, fiber_threshold, model_mode,     # populations, and
                             out_r2, out_rmse):                       # fastmath assumes no NaNs
    n_voxels = coords.shape[0]
    N = len(bvals)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]

        ff_tot = params[x, y, z, _CH_FF]
        ff_tot_v = 0.0 if np.isnan(ff_tot) else ff_tot

        rf = params[x, y, z, _CH_RF]
        if np.isnan(rf):
            rf = 0.0
        if model_mode == 3:
            hf = params[x, y, z, _CH_HF]
            wf = params[x, y, z, _CH_WF]
            if np.isnan(hf):
                hf = 0.0
            if np.isnan(wf):
                wf = 0.0
            nrf = 0.0
            iso_tot = rf + hf + wf
        else:
            nrf = params[x, y, z, _CH_NRF]
            if np.isnan(nrf):
                nrf = 0.0
            hf = 0.0
            wf = 0.0
            iso_tot = rf + nrf

        if (ff_tot_v + iso_tot) < 1e-6:
            continue

        adc_iso = params[x, y, z, _CH_ADC_ISO]
        if np.isnan(adc_iso):
            adc_iso = 1.0e-3

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

        if model_mode == 3:
            D_res, D_hin, D_wat = _recover_iso_adcs_3iso(rf, hf, wf, adc_iso)
            D_nonrf = 0.0
        else:
            D_res, D_nonrf = _recover_iso_adcs_2iso(rf, nrf, adc_iso)
            D_hin = 0.0
            D_wat = 0.0

        # fiber populations (up to MAX_FIBER_POPULATIONS = 2), from stored
        # channels. FF_pop1 is now stored, not re-derived here.
        ad1 = params[x, y, z, _CH_AD]
        rd1 = params[x, y, z, _CH_RD]
        ff1 = params[x, y, z, _CH_FF1]
        ff2 = params[x, y, z, _CH_FF2]
        if np.isnan(ff1):
            ff1 = 0.0
        if np.isnan(ff2):
            ff2 = 0.0
        ad2 = params[x, y, z, _CH_AD2]
        rd2 = params[x, y, z, _CH_RD2]
        d1x = params[x, y, z, _CH_DIR1]
        d1y = params[x, y, z, _CH_DIR1 + 1]
        d1z = params[x, y, z, _CH_DIR1 + 2]
        d2x = params[x, y, z, _CH_DIR2]
        d2y = params[x, y, z, _CH_DIR2 + 1]
        d2z = params[x, y, z, _CH_DIR2 + 2]

        has_fiber = ff_tot_v > fiber_threshold
        use1 = has_fiber and (ff1 > 1e-6) and (not np.isnan(ad1)) and (not np.isnan(d1x))
        use2 = has_fiber and (ff2 > 1e-6) and (not np.isnan(ad2)) and (not np.isnan(d2x))

        s_mean = 0.0
        for i in range(N):
            s_mean += sig[i] / s0
        s_mean /= N

        ss_res = 0.0
        ss_tot = 0.0
        rmse_sum = 0.0
        for i in range(N):
            b = bvals[i]
            bx = bvecs[i, 0]
            by = bvecs[i, 1]
            bz = bvecs[i, 2]

            if model_mode == 3:
                s_pred = (rf * np.exp(-b * D_res)
                          + hf * np.exp(-b * D_hin)
                          + wf * np.exp(-b * D_wat))
            else:
                s_pred = (rf * np.exp(-b * D_res)
                          + nrf * np.exp(-b * D_nonrf))

            if use1:
                c1 = bx * d1x + by * d1y + bz * d1z
                s_pred += ff1 * np.exp(-b * (rd1 + (ad1 - rd1) * c1 * c1))
            if use2:
                c2 = bx * d2x + by * d2y + bz * d2z
                s_pred += ff2 * np.exp(-b * (rd2 + (ad2 - rd2) * c2 * c2))

            sn = sig[i] / s0
            res = sn - s_pred
            ss_res += res * res
            ss_tot += (sn - s_mean) ** 2
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
        print("  MULTI-POPULATION reconstruction: models ALL detected fiber")
        print("  populations (stored directions + tensors) + the isotropic")
        print("  block. Exact for single-fiber voxels; genuine crossings are")
        print("  now modelled, so a low R² reflects real misfit, not the old")
        print("  dominant-population-only artefact.")
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

    b0_thr = 100.0

    shape3d = data.shape[:3]
    r2_map = np.full(shape3d, np.nan, dtype=np.float32)
    rmse_map = np.full(shape3d, np.nan, dtype=np.float32)

    coords = np.argwhere(mask)
    n_voxels = len(coords)
    batch_sz = 10_000
    n_batches = int(np.ceil(n_voxels / batch_sz))

    data_f32 = data.astype(np.float32)
    # NaN is meaningful here (it marks absent populations/tensors), and the
    # kernel handles it explicitly, so pass results through UNMODIFIED — do not
    # zero out NaNs (that would fabricate zero-fraction populations/directions).
    results_f32 = results.astype(np.float32)

    if verbose:
        print(f"\n  Computing quality maps for {n_voxels:,} voxels...")

    t0 = time.time()
    with tqdm(total=n_voxels, desc="  Progress", unit="vox", disable=not verbose) as pbar:
        for i in range(n_batches):
            start = i * batch_sz
            end = min((i + 1) * batch_sz, n_voxels)
            _quality_kernel_multipop(
                data_f32, coords[start:end],
                bvals, bvecs, results_f32, b0_thr, fiber_threshold,
                int(model_mode), r2_map, rmse_map
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
              f"(genuine misfit now that all populations are modelled: low SNR, "
              f"partial volume, or >2-population configurations)")
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

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MAP SAVING
# ─────────────────────────────────────────────────────────────────────────────

# In 3-ISO mode nonrestricted_fraction is an exact duplicate: it equals
# hindered_fraction + water_fraction by construction. The other former
# duplicates, ad_linear/rd_linear, no longer exist as channels — they were
# byte-identical copies of the population-1 diffusivities and were dropped from
# the output layout itself.


def compute_fiber_validity_map(results, channel_names):
    """
    Voxel-level validity indicator for the fiber-tensor output channels: 1
    where a real fiber tensor was estimated, 0 elsewhere.

    The NaN/0 convention it companions is documented in
    `DBSI_Adaptive.output_map_names`. The short version: the per-population
    block and the FF-weighted aggregates use NaN for "absent / not estimated",
    because 0 is a physically plausible diffusivity and cannot be a sentinel.

    That convention is correct in native space but does not survive a naive
    interpolation: ANTs (and any linear resampler) turns NaN into 0 before
    interpolating, so a reprojected AD/RD/FA map is silently depressed in
    proportion to the local NaN density. The fix is a normalised convolution —
    resample `value * valid` and `valid` through the same transform and divide.
    This function returns the `valid` term, which `save_output_maps` writes
    alongside the channel maps so the correction is available to anyone who
    picks the maps up later.

    Parameters
    ----------
    results : ndarray (X, Y, Z, C)
    channel_names : sequence of str

    Returns
    -------
    ndarray (X, Y, Z), uint8
    """
    names = list(channel_names)
    ad1 = results[..., names.index('axial_diffusivity_pop1')]
    ff = results[..., names.index('fiber_fraction')]
    return (np.isfinite(ad1) & np.isfinite(ff) & (ff > 0)).astype(np.uint8)


def save_output_maps(results, channel_names, affine, output_dir,
                     skip_redundant=True, validity=True):
    """
    Save the DBSI output set for one dataset as compressed NIfTI: the channel
    maps plus the fiber-validity indicator. This is the single saving entry
    point — `scripts/run_dbsi.py` and the analysis notebooks both go through
    it, so the on-disk layout is defined in exactly one place.

    Layout written under `output_dir`::

        NN_<channel>.nii.gz    per-channel maps (NN = channel index)
        fiber_valid.nii.gz     uint8 validity mask for the tensor channels

    The FF-weighted aggregate maps are ordinary channels now (21-23:
    axial_/radial_diffusivity_weighted, fiber_fa_weighted), computed by the
    model itself, so they are written by the loop below like any other channel.
    They used to be a separate `compute_aggregate_fiber_maps` step that every
    caller had to remember; that function is gone.

    Channels skipped:

    * `*_NaN` — invalid for the model mode (hindered/water in 2-ISO);
    * with `skip_redundant`, `nonrestricted_fraction` in 3-ISO mode, where it
      is hindered_fraction + water_fraction by construction.

    Nothing else is trimmed: with at most two fiber populations, every
    remaining channel carries information, and a population-2 block that
    happens to be empty on one subject must still produce the same file set as
    on every other subject.

    This does NOT change the internal `results` array — channel positions are
    load-bearing for `compute_fit_quality`, and the saved `.npz` should keep
    every channel. Only the written files are trimmed.

    Returns the list of channel names actually written.
    """
    import nibabel as nib
    import os
    names = list(channel_names)
    skip = set()
    if skip_redundant and 'hindered_fraction' in names:   # 3-ISO: NRF == HF + WF
        skip.add('nonrestricted_fraction')
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i, nm in enumerate(names):
        if nm.endswith('_NaN') or nm in skip:
            continue
        nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine),
                 os.path.join(output_dir, f'{i:02d}_{nm}.nii.gz'))
        saved.append(nm)
    if validity:
        nib.save(nib.Nifti1Image(compute_fiber_validity_map(results, names), affine),
                 os.path.join(output_dir, 'fiber_valid.nii.gz'))
    return saved
