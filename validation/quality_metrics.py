"""
DBSI Quality Metrics and Validation Module

This module provides comprehensive validation tools for assessing DBSI fitting quality,
including R², RMSE, residual analysis, and physiological plausibility checks.

These metrics are designed for use with REAL DATA (not synthetic), and compute
voxel-wise and global quality indicators.

References:
- Wang et al. (2011) Brain - Original DBSI methodology
- Ye et al. (2020) Ann Clin Transl Neurol - DBSI in MS lesions
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Tuple, Optional


# =============================================================================
# ADC THRESHOLDS (from literature)
# =============================================================================
# Restricted: ADC ≤ 0.3 µm²/ms (cells, inflammation) - Ye et al. 2020
# Hindered: 0.3 < ADC ≤ 3.0 µm²/ms (edema, tissue disorganization) 
# Free: ADC > 3.0 µm²/ms (CSF) - Wang et al. 2011

THRESH_RESTRICTED = 0.3e-3  # mm²/s
THRESH_FREE = 3.0e-3        # mm²/s

# Physiological bounds for diffusivities (mm²/s)
# Based on literature values from Wang et al. 2011, Lavadi et al. 2025
AD_PHYSIOLOGICAL_RANGE = (0.5e-3, 2.5e-3)  # Typical WM: 1.0-2.0
RD_PHYSIOLOGICAL_RANGE = (0.1e-3, 1.2e-3)  # Typical WM: 0.2-0.6
FA_PHYSIOLOGICAL_RANGE = (0.0, 1.0)


# =============================================================================
# CORE FITTING QUALITY METRICS
# =============================================================================

def compute_r2_map(data: np.ndarray, predicted: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
    """
    Compute voxel-wise R² (coefficient of determination) for DBSI fit.
    
    R² = 1 - SS_res / SS_tot
    
    where:
    - SS_res = Σ(y_i - ŷ_i)² (residual sum of squares)
    - SS_tot = Σ(y_i - ȳ)² (total sum of squares)
    
    Parameters
    ----------
    data : ndarray (X, Y, Z, N)
        Original normalized DWI signal
    predicted : ndarray (X, Y, Z, N)
        Predicted signal from DBSI model
    mask : ndarray (X, Y, Z)
        Binary brain mask
        
    Returns
    -------
    r2_map : ndarray (X, Y, Z)
        R² values per voxel. Values close to 1 indicate good fit.
    """
    r2_map = np.zeros(data.shape[:3], dtype=np.float32)
    
    coords = np.argwhere(mask)
    for idx in range(len(coords)):
        x, y, z = coords[idx]
        y_true = data[x, y, z]
        y_pred = predicted[x, y, z]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot > 1e-10:
            r2_map[x, y, z] = 1.0 - ss_res / ss_tot
        else:
            r2_map[x, y, z] = 0.0
    
    return r2_map


def compute_rmse_map(data: np.ndarray, predicted: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """
    Compute voxel-wise Root Mean Square Error (RMSE).
    
    RMSE = sqrt(mean((y - ŷ)²))
    
    Parameters
    ----------
    data : ndarray (X, Y, Z, N)
        Original normalized DWI signal
    predicted : ndarray (X, Y, Z, N)
        Predicted signal from DBSI model
    mask : ndarray (X, Y, Z)
        Binary brain mask
        
    Returns
    -------
    rmse_map : ndarray (X, Y, Z)
        RMSE values per voxel. Lower is better.
    """
    rmse_map = np.zeros(data.shape[:3], dtype=np.float32)
    
    coords = np.argwhere(mask)
    for idx in range(len(coords)):
        x, y, z = coords[idx]
        y_true = data[x, y, z]
        y_pred = predicted[x, y, z]
        
        rmse_map[x, y, z] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return rmse_map


def compute_nrmse_map(data: np.ndarray, predicted: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """
    Compute voxel-wise Normalized RMSE (by signal range).
    
    NRMSE = RMSE / (max(y) - min(y))
    
    Returns values in [0, 1] where lower is better.
    """
    nrmse_map = np.zeros(data.shape[:3], dtype=np.float32)
    
    coords = np.argwhere(mask)
    for idx in range(len(coords)):
        x, y, z = coords[idx]
        y_true = data[x, y, z]
        y_pred = predicted[x, y, z]
        
        y_range = np.max(y_true) - np.min(y_true)
        if y_range > 1e-10:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            nrmse_map[x, y, z] = rmse / y_range
        else:
            nrmse_map[x, y, z] = 0.0
    
    return nrmse_map


# =============================================================================
# SIGNAL PREDICTION FROM DBSI PARAMETERS
# =============================================================================

@njit(cache=True, fastmath=True)
def predict_signal_single_voxel(bvals: np.ndarray, bvecs: np.ndarray,
                                fiber_dir: np.ndarray,
                                f_fiber: float, f_res: float, 
                                f_hin: float, f_wat: float,
                                AD: float, RD: float,
                                D_res: float = 0.15e-3,
                                D_hin: float = 1.0e-3,
                                D_wat: float = 3.0e-3) -> np.ndarray:
    """
    Predict normalized DWI signal from DBSI parameters for a single voxel.
    
    Uses the cylinder model for anisotropic component:
    S_aniso = f_fiber * exp(-b * (RD + (AD-RD)*cos²θ))
    
    And exponential decay for isotropic components:
    S_iso = Σ f_i * exp(-b * D_i)
    """
    n_meas = len(bvals)
    s_pred = np.zeros(n_meas, dtype=np.float64)
    
    # Normalize fractions
    f_tot = f_fiber + f_res + f_hin + f_wat + 1e-12
    ff = f_fiber / f_tot
    fr = f_res / f_tot
    fh = f_hin / f_tot
    fw = f_wat / f_tot
    
    for i in range(n_meas):
        b = bvals[i]
        g = bvecs[i]
        
        # Fiber component
        if ff > 0.01 and AD > 1e-10:
            cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
            D_app = RD + (AD - RD) * cos_t * cos_t
            s_fiber = ff * np.exp(-b * D_app)
        else:
            s_fiber = 0.0
        
        # Isotropic components
        s_res = fr * np.exp(-b * D_res) if fr > 0.01 else 0.0
        s_hin = fh * np.exp(-b * D_hin) if fh > 0.01 else 0.0
        s_wat = fw * np.exp(-b * D_wat) if fw > 0.01 else 0.0
        
        s_pred[i] = s_fiber + s_res + s_hin + s_wat
    
    return s_pred


def predict_signal_volume(bvals: np.ndarray, bvecs: np.ndarray,
                          dbsi_results: np.ndarray, mask: np.ndarray,
                          fiber_dirs: np.ndarray,
                          fiber_dir_indices: np.ndarray = None) -> np.ndarray:
    """
    Predict signal for entire volume from DBSI results.
    
    Parameters
    ----------
    bvals : ndarray (N,)
        B-values
    bvecs : ndarray (N, 3)
        Gradient directions
    dbsi_results : ndarray (X, Y, Z, 8)
        DBSI outputs: [f_fib, f_res, f_hin, f_wat, AD, RD, FA, mean_iso_adc]
    mask : ndarray (X, Y, Z)
        Brain mask
    fiber_dirs : ndarray (M, 3)
        Fiber direction basis (from model fitting)
    fiber_dir_indices : ndarray (X, Y, Z), optional
        Index of dominant fiber direction per voxel
        
    Returns
    -------
    predicted : ndarray (X, Y, Z, N)
        Predicted signal
    """
    shape = mask.shape
    n_meas = len(bvals)
    predicted = np.zeros(shape + (n_meas,), dtype=np.float32)
    
    # If no direction indices provided, use z-axis as default
    if fiber_dir_indices is None:
        default_dir = np.array([0.0, 0.0, 1.0])
    
    coords = np.argwhere(mask)
    for idx in range(len(coords)):
        x, y, z = coords[idx]
        
        f_fib = dbsi_results[x, y, z, 0]
        f_res = dbsi_results[x, y, z, 1]
        f_hin = dbsi_results[x, y, z, 2]
        f_wat = dbsi_results[x, y, z, 3]
        AD = dbsi_results[x, y, z, 4]
        RD = dbsi_results[x, y, z, 5]
        
        # Get fiber direction
        if fiber_dir_indices is not None:
            fdir = fiber_dirs[fiber_dir_indices[x, y, z]]
        else:
            fdir = default_dir
        
        predicted[x, y, z] = predict_signal_single_voxel(
            bvals, bvecs, fdir, f_fib, f_res, f_hin, f_wat, AD, RD
        )
    
    return predicted


# =============================================================================
# PHYSIOLOGICAL PLAUSIBILITY CHECKS
# =============================================================================

def check_physiological_plausibility(dbsi_results: np.ndarray, 
                                     mask: np.ndarray) -> Dict:
    """
    Check if DBSI results are physiologically plausible.
    
    Checks:
    1. Fraction sum = 1.0 (± tolerance)
    2. AD > RD for fiber voxels
    3. Diffusivities within physiological range
    4. FA in [0, 1]
    5. No negative fractions
    
    Parameters
    ----------
    dbsi_results : ndarray (X, Y, Z, 8)
        DBSI outputs
    mask : ndarray (X, Y, Z)
        Brain mask
        
    Returns
    -------
    report : dict
        Dictionary with plausibility metrics
    """
    f_fib = dbsi_results[..., 0][mask]
    f_res = dbsi_results[..., 1][mask]
    f_hin = dbsi_results[..., 2][mask]
    f_wat = dbsi_results[..., 3][mask]
    AD = dbsi_results[..., 4][mask]
    RD = dbsi_results[..., 5][mask]
    FA = dbsi_results[..., 6][mask]
    
    n_voxels = len(f_fib)
    
    # 1. Fraction sum check
    f_sum = f_fib + f_res + f_hin + f_wat
    frac_sum_ok = np.sum(np.abs(f_sum - 1.0) < 0.01) / n_voxels * 100
    
    # 2. AD > RD for fiber voxels
    fiber_mask = f_fib > 0.1
    if np.sum(fiber_mask) > 0:
        ad_gt_rd = np.sum(AD[fiber_mask] > RD[fiber_mask]) / np.sum(fiber_mask) * 100
    else:
        ad_gt_rd = 100.0
    
    # 3. Diffusivity range checks
    ad_in_range = np.sum((AD >= AD_PHYSIOLOGICAL_RANGE[0]) & 
                         (AD <= AD_PHYSIOLOGICAL_RANGE[1]) | 
                         (f_fib < 0.1)) / n_voxels * 100
    rd_in_range = np.sum((RD >= RD_PHYSIOLOGICAL_RANGE[0]) & 
                         (RD <= RD_PHYSIOLOGICAL_RANGE[1]) | 
                         (f_fib < 0.1)) / n_voxels * 100
    
    # 4. FA check
    fa_in_range = np.sum((FA >= 0) & (FA <= 1.0)) / n_voxels * 100
    
    # 5. No negative fractions
    no_neg_frac = np.sum((f_fib >= 0) & (f_res >= 0) & 
                         (f_hin >= 0) & (f_wat >= 0)) / n_voxels * 100
    
    # 6. CSF-like voxels (high water fraction) should have ~0 fiber
    csf_like = f_wat > 0.7
    if np.sum(csf_like) > 0:
        csf_low_fiber = np.sum(f_fib[csf_like] < 0.1) / np.sum(csf_like) * 100
    else:
        csf_low_fiber = 100.0
    
    # 7. High fiber voxels should have reasonable FA
    high_fiber = f_fib > 0.5
    if np.sum(high_fiber) > 0:
        wm_fa_reasonable = np.sum(FA[high_fiber] > 0.2) / np.sum(high_fiber) * 100
    else:
        wm_fa_reasonable = 100.0
    
    report = {
        'n_voxels_analyzed': n_voxels,
        'fraction_sum_ok_pct': frac_sum_ok,
        'ad_gt_rd_in_fiber_pct': ad_gt_rd,
        'ad_physiological_pct': ad_in_range,
        'rd_physiological_pct': rd_in_range,
        'fa_in_range_pct': fa_in_range,
        'no_negative_fractions_pct': no_neg_frac,
        'csf_low_fiber_pct': csf_low_fiber,
        'wm_fa_reasonable_pct': wm_fa_reasonable,
        'overall_plausibility_pct': np.mean([
            frac_sum_ok, ad_gt_rd, ad_in_range, rd_in_range,
            fa_in_range, no_neg_frac, csf_low_fiber, wm_fa_reasonable
        ])
    }
    
    return report


# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================

def compute_residual_statistics(data: np.ndarray, predicted: np.ndarray,
                                mask: np.ndarray) -> Dict:
    """
    Comprehensive residual analysis.
    
    Parameters
    ----------
    data : ndarray (X, Y, Z, N)
        Original normalized signal
    predicted : ndarray (X, Y, Z, N)
        Predicted signal
    mask : ndarray (X, Y, Z)
        Brain mask
        
    Returns
    -------
    stats : dict
        Residual statistics
    """
    residuals = (data - predicted)[mask]
    
    stats = {
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'median_residual': float(np.median(residuals)),
        'max_abs_residual': float(np.max(np.abs(residuals))),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'skewness': float(_compute_skewness(residuals.flatten())),
        'kurtosis': float(_compute_kurtosis(residuals.flatten())),
        'pct_residual_lt_5pct': float(np.sum(np.abs(residuals) < 0.05) / 
                                      residuals.size * 100),
        'pct_residual_lt_10pct': float(np.sum(np.abs(residuals) < 0.10) / 
                                       residuals.size * 100),
    }
    
    return stats


def _compute_skewness(x: np.ndarray) -> float:
    """Compute skewness of distribution."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.sum(((x - mean) / std) ** 3) / n


def _compute_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of distribution."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.sum(((x - mean) / std) ** 4) / n - 3.0


# =============================================================================
# TISSUE-SPECIFIC METRICS
# =============================================================================

def compute_tissue_specific_metrics(dbsi_results: np.ndarray,
                                    mask: np.ndarray,
                                    tissue_priors: Dict[str, np.ndarray] = None) -> Dict:
    """
    Compute metrics stratified by tissue type based on DBSI fractions.
    
    Tissue classification based on dominant fraction:
    - WM-like: fiber_fraction > 0.4
    - GM-like: hindered_fraction dominant, fiber_fraction < 0.3
    - CSF-like: water_fraction > 0.7
    - Lesion-like: restricted_fraction > 0.2
    
    Parameters
    ----------
    dbsi_results : ndarray (X, Y, Z, 8)
        DBSI outputs
    mask : ndarray (X, Y, Z)
        Brain mask
    tissue_priors : dict, optional
        Dictionary with 'wm', 'gm', 'csf' masks from segmentation
        
    Returns
    -------
    metrics : dict
        Tissue-specific summary statistics
    """
    f_fib = dbsi_results[..., 0]
    f_res = dbsi_results[..., 1]
    f_hin = dbsi_results[..., 2]
    f_wat = dbsi_results[..., 3]
    AD = dbsi_results[..., 4]
    RD = dbsi_results[..., 5]
    FA = dbsi_results[..., 6]
    mean_adc = dbsi_results[..., 7]
    
    metrics = {}
    
    # DBSI-based tissue classification
    wm_like = mask & (f_fib > 0.4)
    gm_like = mask & (f_fib < 0.3) & (f_hin > f_res) & (f_wat < 0.3)
    csf_like = mask & (f_wat > 0.7)
    lesion_like = mask & (f_res > 0.2)
    
    tissue_masks = {
        'WM-like (DBSI)': wm_like,
        'GM-like (DBSI)': gm_like,
        'CSF-like (DBSI)': csf_like,
        'Lesion-like (DBSI)': lesion_like
    }
    
    # Add prior-based masks if available
    if tissue_priors is not None:
        for name, prior_mask in tissue_priors.items():
            tissue_masks[f'{name} (prior)'] = prior_mask & mask
    
    for name, tmask in tissue_masks.items():
        n_vox = np.sum(tmask)
        if n_vox > 10:  # Minimum voxels for statistics
            metrics[name] = {
                'n_voxels': int(n_vox),
                'fiber_fraction': {
                    'mean': float(np.mean(f_fib[tmask])),
                    'std': float(np.std(f_fib[tmask])),
                },
                'restricted_fraction': {
                    'mean': float(np.mean(f_res[tmask])),
                    'std': float(np.std(f_res[tmask])),
                },
                'hindered_fraction': {
                    'mean': float(np.mean(f_hin[tmask])),
                    'std': float(np.std(f_hin[tmask])),
                },
                'water_fraction': {
                    'mean': float(np.mean(f_wat[tmask])),
                    'std': float(np.std(f_wat[tmask])),
                },
                'AD': {
                    'mean': float(np.mean(AD[tmask])),
                    'std': float(np.std(AD[tmask])),
                },
                'RD': {
                    'mean': float(np.mean(RD[tmask])),
                    'std': float(np.std(RD[tmask])),
                },
                'FA': {
                    'mean': float(np.mean(FA[tmask])),
                    'std': float(np.std(FA[tmask])),
                },
                'mean_iso_adc': {
                    'mean': float(np.mean(mean_adc[tmask])),
                    'std': float(np.std(mean_adc[tmask])),
                },
            }
    
    return metrics


# =============================================================================
# GLOBAL SUMMARY REPORT
# =============================================================================

def generate_validation_report(data_normalized: np.ndarray,
                               predicted: np.ndarray,
                               dbsi_results: np.ndarray,
                               mask: np.ndarray,
                               verbose: bool = True) -> Dict:
    """
    Generate comprehensive validation report for DBSI fitting.
    
    Parameters
    ----------
    data_normalized : ndarray (X, Y, Z, N)
        Normalized DWI signal (S/S0)
    predicted : ndarray (X, Y, Z, N)
        Predicted signal from DBSI model
    dbsi_results : ndarray (X, Y, Z, 8)
        DBSI outputs
    mask : ndarray (X, Y, Z)
        Brain mask
    verbose : bool
        Print report to console
        
    Returns
    -------
    report : dict
        Complete validation report
    """
    report = {}
    
    # 1. R² analysis
    r2_map = compute_r2_map(data_normalized, predicted, mask)
    r2_masked = r2_map[mask]
    report['r2'] = {
        'mean': float(np.mean(r2_masked)),
        'median': float(np.median(r2_masked)),
        'std': float(np.std(r2_masked)),
        'min': float(np.min(r2_masked)),
        'max': float(np.max(r2_masked)),
        'pct_gt_0.9': float(np.sum(r2_masked > 0.9) / len(r2_masked) * 100),
        'pct_gt_0.8': float(np.sum(r2_masked > 0.8) / len(r2_masked) * 100),
        'pct_gt_0.7': float(np.sum(r2_masked > 0.7) / len(r2_masked) * 100),
    }
    
    # 2. RMSE analysis
    rmse_map = compute_rmse_map(data_normalized, predicted, mask)
    rmse_masked = rmse_map[mask]
    report['rmse'] = {
        'mean': float(np.mean(rmse_masked)),
        'median': float(np.median(rmse_masked)),
        'std': float(np.std(rmse_masked)),
        'min': float(np.min(rmse_masked)),
        'max': float(np.max(rmse_masked)),
    }
    
    # 3. NRMSE analysis
    nrmse_map = compute_nrmse_map(data_normalized, predicted, mask)
    nrmse_masked = nrmse_map[mask]
    report['nrmse'] = {
        'mean': float(np.mean(nrmse_masked)),
        'median': float(np.median(nrmse_masked)),
        'pct_lt_5pct': float(np.sum(nrmse_masked < 0.05) / len(nrmse_masked) * 100),
        'pct_lt_10pct': float(np.sum(nrmse_masked < 0.10) / len(nrmse_masked) * 100),
    }
    
    # 4. Residual statistics
    report['residuals'] = compute_residual_statistics(data_normalized, predicted, mask)
    
    # 5. Physiological plausibility
    report['plausibility'] = check_physiological_plausibility(dbsi_results, mask)
    
    # 6. Tissue-specific metrics
    report['tissue_metrics'] = compute_tissue_specific_metrics(dbsi_results, mask)
    
    # 7. Overall quality score (0-100)
    quality_score = _compute_overall_quality_score(report)
    report['overall_quality_score'] = quality_score
    
    if verbose:
        _print_validation_report(report)
    
    return report


def _compute_overall_quality_score(report: Dict) -> float:
    """
    Compute overall quality score from 0-100.
    
    Weights:
    - R² (40%): Mean R² scaled to 0-40
    - NRMSE (20%): Percent < 10% scaled to 0-20
    - Plausibility (30%): Overall plausibility scaled to 0-30
    - Residual symmetry (10%): Based on skewness
    """
    # R² contribution (target: mean > 0.8)
    r2_score = min(40, report['r2']['mean'] / 0.8 * 40)
    
    # NRMSE contribution (target: > 80% below 10%)
    nrmse_score = min(20, report['nrmse']['pct_lt_10pct'] / 80 * 20)
    
    # Plausibility contribution
    plaus_score = report['plausibility']['overall_plausibility_pct'] / 100 * 30
    
    # Residual symmetry (target: |skewness| < 0.5)
    skew = abs(report['residuals']['skewness'])
    resid_score = max(0, 10 - skew * 10)
    
    return float(r2_score + nrmse_score + plaus_score + resid_score)


def _print_validation_report(report: Dict):
    """Print formatted validation report to console."""
    print("\n" + "=" * 70)
    print("           DBSI VALIDATION REPORT")
    print("=" * 70)
    
    print("\n[1] FITTING QUALITY (R²)")
    print("-" * 40)
    r2 = report['r2']
    print(f"  Mean R²:         {r2['mean']:.4f}")
    print(f"  Median R²:       {r2['median']:.4f}")
    print(f"  Std R²:          {r2['std']:.4f}")
    print(f"  % voxels R² > 0.9: {r2['pct_gt_0.9']:.1f}%")
    print(f"  % voxels R² > 0.8: {r2['pct_gt_0.8']:.1f}%")
    
    print("\n[2] ERROR METRICS")
    print("-" * 40)
    print(f"  Mean RMSE:       {report['rmse']['mean']:.4f}")
    print(f"  Mean NRMSE:      {report['nrmse']['mean']:.4f}")
    print(f"  % NRMSE < 5%:    {report['nrmse']['pct_lt_5pct']:.1f}%")
    print(f"  % NRMSE < 10%:   {report['nrmse']['pct_lt_10pct']:.1f}%")
    
    print("\n[3] RESIDUAL ANALYSIS")
    print("-" * 40)
    res = report['residuals']
    print(f"  Mean residual:   {res['mean_residual']:.6f}")
    print(f"  Std residual:    {res['std_residual']:.4f}")
    print(f"  Skewness:        {res['skewness']:.4f}")
    print(f"  Kurtosis:        {res['kurtosis']:.4f}")
    
    print("\n[4] PHYSIOLOGICAL PLAUSIBILITY")
    print("-" * 40)
    plaus = report['plausibility']
    print(f"  Fraction sum ≈ 1:    {plaus['fraction_sum_ok_pct']:.1f}%")
    print(f"  AD > RD (fiber):     {plaus['ad_gt_rd_in_fiber_pct']:.1f}%")
    print(f"  AD in range:         {plaus['ad_physiological_pct']:.1f}%")
    print(f"  RD in range:         {plaus['rd_physiological_pct']:.1f}%")
    print(f"  FA in [0,1]:         {plaus['fa_in_range_pct']:.1f}%")
    print(f"  No negative frac:    {plaus['no_negative_fractions_pct']:.1f}%")
    print(f"  CSF low fiber:       {plaus['csf_low_fiber_pct']:.1f}%")
    print(f"  WM FA reasonable:    {plaus['wm_fa_reasonable_pct']:.1f}%")
    print(f"  OVERALL:             {plaus['overall_plausibility_pct']:.1f}%")
    
    print("\n[5] TISSUE-SPECIFIC SUMMARY")
    print("-" * 40)
    for tissue, stats in report['tissue_metrics'].items():
        print(f"\n  {tissue} (n={stats['n_voxels']})")
        print(f"    Fiber: {stats['fiber_fraction']['mean']:.3f} ± {stats['fiber_fraction']['std']:.3f}")
        print(f"    Restr: {stats['restricted_fraction']['mean']:.3f} ± {stats['restricted_fraction']['std']:.3f}")
        print(f"    FA:    {stats['FA']['mean']:.3f} ± {stats['FA']['std']:.3f}")
    
    print("\n" + "=" * 70)
    print(f"  OVERALL QUALITY SCORE: {report['overall_quality_score']:.1f}/100")
    print("=" * 70 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK VALIDATION
# =============================================================================

def quick_r2_check(dbsi_results: np.ndarray, 
                   data_normalized: np.ndarray,
                   bvals: np.ndarray, bvecs: np.ndarray,
                   mask: np.ndarray,
                   fiber_dirs: np.ndarray = None) -> float:
    """
    Quick R² computation for validation without full report.
    
    Returns mean R² across all masked voxels.
    """
    if fiber_dirs is None:
        from core.basis import generate_fibonacci_sphere_hemisphere
        fiber_dirs = generate_fibonacci_sphere_hemisphere(100)
    
    predicted = predict_signal_volume(bvals, bvecs, dbsi_results, mask, fiber_dirs)
    r2_map = compute_r2_map(data_normalized, predicted, mask)
    
    return float(np.mean(r2_map[mask]))
