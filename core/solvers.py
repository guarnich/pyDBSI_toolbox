"""
DBSI Core Solvers - Balanced Final Implementation

Key principles:
1. Step 1: Linear NNLS to get fractions (fixed diffusivities, tissue-adaptive)
2. Step 2: Refine AD/RD only if fiber fraction is significant
3. FA scaled by fiber fraction to avoid artifacts
4. Extended parameter ranges to handle all tissue types

ADC Thresholds (Literature Standard):
    - Restricted: ≤ 0.3 µm²/ms (Ye et al. 2020)
    - Hindered: 0.3-3.0 µm²/ms (Wang et al. 2011)  
    - Free/Water: > 3.0 µm²/ms (CSF reference)

References:
    Wang Y et al. (2011) Brain 134:3590-3601
    Ye Z et al. (2020) Ann Clin Transl Neurol 7:695-706
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-7, max_iter=2000):
    """NNLS via Coordinate Descent with Active Set."""
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


@njit(cache=True, fastmath=True)
def compute_weighted_centroids(w_iso, iso_grid):
    """Compute weighted centroids for isotropic components."""
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
def step2_refine_diffusivities(bvals, bvecs, y_norm, fiber_dir,
                               f_fiber, f_res, f_hin, f_wat,
                               D_res, D_hin, D_wat):
    """
    Step 2: Refine fiber AD/RD using grid search.
    
    Extended range to handle various tissue types.
    """
    best_sse = 1e20
    best_ax = 1.5e-3
    best_rad = 0.4e-3
    
    ftot = f_fiber + f_res + f_hin + f_wat + 1e-12
    ff = f_fiber / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    # Extended grid for all tissue types
    n_ax, n_rad = 12, 10
    ax_min, ax_max = 0.5e-3, 2.5e-3
    rad_min, rad_max = 0.1e-3, 1.2e-3
    
    ax_step = (ax_max - ax_min) / (n_ax - 1)
    rad_step = (rad_max - rad_min) / (n_rad - 1)
    
    for i_ax in range(n_ax):
        ax = ax_min + i_ax * ax_step
        
        for i_rad in range(n_rad):
            rad = rad_min + i_rad * rad_step
            
            # Soft constraint: AD should be > RD
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
    
    # Local refinement
    ax_c, rad_c = best_ax, best_rad
    fine_ax = ax_step / 4
    fine_rad = rad_step / 4
    
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
def compute_fiber_fa(AD, RD, fiber_fraction):
    """
    Compute FA scaled by fiber fraction.
    
    Uses the cylindrically symmetric tensor formula:
        FA = sqrt(0.5) * (AD - RD) / sqrt(AD² + 2*RD²)
    
    This assumes λ₁ = AD and λ₂ = λ₃ = RD (cylindrical symmetry),
    which is consistent with the DBSI anisotropic tensor model.
    
    Note: This differs from standard DTI FA which uses full eigenvalue
    decomposition. For DBSI, this formula is appropriate as the model
    explicitly assumes cylindrically symmetric tensors.
    
    The FA is scaled by fiber fraction because when fiber content is low,
    the estimated diffusivities become unreliable, so we attenuate FA
    to avoid misleading high FA values in non-fiber regions (e.g., CSF).
    
    Parameters
    ----------
    AD : float
        Axial diffusivity (mm²/s)
    RD : float
        Radial diffusivity (mm²/s)
    fiber_fraction : float
        Fiber signal fraction (0-1)
        
    Returns
    -------
    FA : float
        Fractional anisotropy (0-1), scaled by fiber fraction
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
    
    FA_raw = np.sqrt(0.5) * diff / denom
    FA_raw = min(1.0, max(0.0, FA_raw))
    
    # Scale by fiber fraction: if f_fiber < 0.2, FA is questionable
    # Smooth transition from 0 at f=0 to full FA at f>=0.3
    scale = min(1.0, fiber_fraction / 0.3)
    
    return FA_raw * scale
