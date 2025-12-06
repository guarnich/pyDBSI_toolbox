"""
DBSI Core: Optimization Solvers

Corrected Version:
- Fixed ADC thresholds (0.3, 3.0 per literature)
- Dynamic isotropic diffusivity centroids from Step 1
- Finer grid search resolution (10x10 minimum)
- Improved numerical stability with diagonal regularization
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-6, max_iter=3000):
    """
    Non-Negative Least Squares via Coordinate Descent with Active Set.
    
    Solves: min ||Ax - y||² + λ||x||²  subject to x >= 0
    
    Uses coordinate descent with active set management for efficiency.
    KKT conditions are used to identify which variables should be active.
    
    Parameters
    ----------
    AtA : ndarray (n_features, n_features)
        Pre-computed A^T @ A matrix
    Aty : ndarray (n_features,)
        Pre-computed A^T @ y vector
    reg_lambda : float
        L2 regularization strength (Tikhonov)
    tol : float
        Convergence tolerance on maximum update
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    x : ndarray (n_features,)
        Non-negative solution vector
    n_iter : int
        Number of iterations performed
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    grad = -Aty.astype(np.float64)
    
    # Precompute regularized Hessian diagonal
    # Add small constant for numerical stability
    hessian_diag = np.empty(n_features, dtype=np.float64)
    for k in range(n_features):
        hessian_diag[k] = AtA[k, k] + reg_lambda + 1e-10
    
    n_iter = 0
    
    for iteration in range(max_iter):
        max_update = 0.0
        n_changes = 0
        
        for i in range(n_features):
            # Gradient for variable i including regularization
            g_i = grad[i] + reg_lambda * x[i]
            
            # Active Set Check (KKT conditions)
            # If x[i] = 0 and gradient >= 0, variable should stay at 0
            if x[i] == 0.0 and g_i >= 0.0:
                continue
            
            # Coordinate descent update
            x_new = max(0.0, x[i] - g_i / hessian_diag[i])
            diff = x_new - x[i]
            
            if abs(diff) > 1e-14:
                if abs(diff) > max_update:
                    max_update = abs(diff)
                
                # Update gradient: grad += AtA[:, i] * diff
                for k in range(n_features):
                    grad[k] += AtA[k, i] * diff
                    
                x[i] = x_new
                n_changes += 1
        
        n_iter += 1
        
        # Convergence check
        if n_changes == 0 or max_update < tol:
            break
            
    return x, n_iter


@njit(cache=True, fastmath=True)
def compute_weighted_centroids(w_iso, iso_grid):
    """
    Compute weighted centroid diffusivities for each compartment.
    
    Parameters
    ----------
    w_iso : ndarray (N_iso,)
        Isotropic spectrum weights from Step 1
    iso_grid : ndarray (N_iso,)
        Discrete ADC values
        
    Returns
    -------
    D_res_centroid : float
        Weighted mean ADC for restricted component (ADC ≤ 0.3e-3)
    D_hin_centroid : float
        Weighted mean ADC for hindered component (0.3e-3 < ADC ≤ 3.0e-3)
    D_wat_centroid : float
        Weighted mean ADC for free water component (ADC > 3.0e-3)
    """
    # CORRECTED: Use literature thresholds
    # Restricted: ADC ≤ 0.3 µm²/ms = 0.3e-3 mm²/s
    # Hindered: 0.3e-3 < ADC ≤ 3.0e-3 mm²/s
    # Free: ADC > 3.0e-3 mm²/s
    
    THRESH_RES = 0.3e-3   # Restricted/Hindered boundary
    THRESH_WAT = 3.0e-3   # Hindered/Free boundary
    
    # Compute weighted centroids for each compartment
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
    
    # Compute centroids with fallback defaults
    D_res_centroid = sum_wd_res / sum_w_res if sum_w_res > 1e-10 else 0.15e-3
    D_hin_centroid = sum_wd_hin / sum_w_hin if sum_w_hin > 1e-10 else 1.0e-3
    D_wat_centroid = sum_wd_wat / sum_w_wat if sum_w_wat > 1e-10 else 3.0e-3
    
    return D_res_centroid, D_hin_centroid, D_wat_centroid


@njit(cache=True, fastmath=True)
def step2_refine_diffusivities(bvals, bvecs, y_norm, fiber_dir, 
                               f_fiber, f_res, f_hin, f_wat,
                               D_res=0.15e-3, D_hin=1.0e-3, D_wat=3.0e-3):
    """
    Step 2: Non-Linear Refinement of Fiber AD/RD.
    
    Uses Grid Search over physiologically plausible range, then optionally
    refines with local search. The isotropic diffusivities should be
    pre-computed from Step 1 centroids.
    
    Parameters
    ----------
    bvals : ndarray (N_meas,)
        B-values in s/mm²
    bvecs : ndarray (N_meas, 3)
        Normalized gradient directions
    y_norm : ndarray (N_meas,)
        Normalized signal (S/S0)
    fiber_dir : ndarray (3,)
        Principal fiber direction from Step 1
    f_fiber, f_res, f_hin, f_wat : float
        Compartment fractions from Step 1
    D_res, D_hin, D_wat : float
        Isotropic diffusivities (should be centroids from Step 1)
        
    Returns
    -------
    best_ax : float
        Optimal axial diffusivity (mm²/s)
    best_rad : float
        Optimal radial diffusivity (mm²/s)
    """
    best_sse = 1e20
    best_ax = 1.7e-3
    best_rad = 0.3e-3
    
    # Normalize fractions for forward model
    ftot = f_fiber + f_res + f_hin + f_wat + 1e-12
    ff = f_fiber / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    # CORRECTED: Finer grid search (10x10 = 100 combinations)
    # Physiologically plausible ranges for white matter:
    # AD: 1.0 - 2.5 µm²/ms (healthy WM typically 1.5-1.7)
    # RD: 0.1 - 0.8 µm²/ms (healthy WM typically 0.3-0.5)
    n_ax, n_rad = 10, 10
    ax_min, ax_max = 1.0e-3, 2.5e-3
    rad_min, rad_max = 0.1e-3, 0.8e-3
    
    ax_step = (ax_max - ax_min) / (n_ax - 1)
    rad_step = (rad_max - rad_min) / (n_rad - 1)
    
    for i_ax in range(n_ax):
        ax = ax_min + i_ax * ax_step
        
        for i_rad in range(n_rad):
            rad = rad_min + i_rad * rad_step
            
            # Physics constraint: AD > RD
            if ax <= rad:
                continue
                
            # Compute SSE for this (AD, RD) pair
            sse = 0.0
            for i in range(len(bvals)):
                b = bvals[i]
                g = bvecs[i]
                
                # Fiber component with cylinder model
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = rad + (ax - rad) * cos_t * cos_t
                
                # Forward model: weighted sum of compartments
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
    
    # Local refinement around best grid point (optional but improves accuracy)
    # Search in ±1 step with finer resolution
    ax_center, rad_center = best_ax, best_rad
    fine_step_ax = ax_step / 4
    fine_step_rad = rad_step / 4
    
    for di in range(-2, 3):
        ax = ax_center + di * fine_step_ax
        if ax < ax_min or ax > ax_max:
            continue
            
        for dj in range(-2, 3):
            rad = rad_center + dj * fine_step_rad
            if rad < rad_min or rad > rad_max:
                continue
            if ax <= rad:
                continue
                
            sse = 0.0
            for i in range(len(bvals)):
                b = bvals[i]
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
def compute_fiber_fa(AD, RD):
    """
    Compute Fractional Anisotropy for cylindrically symmetric tensor.
    
    For a cylinder model with λ₁ = AD, λ₂ = λ₃ = RD:
    FA = sqrt(0.5) × (AD - RD) / sqrt(AD² + 2×RD²)
    
    Parameters
    ----------
    AD : float
        Axial diffusivity
    RD : float
        Radial diffusivity
        
    Returns
    -------
    FA : float
        Fractional anisotropy [0, 1]
    """
    if AD <= 0 or RD <= 0:
        return 0.0
        
    # For cylinder: λ1 = AD, λ2 = λ3 = RD
    # MD = (AD + 2*RD) / 3
    # FA = sqrt(3/2) * sqrt((λ1-MD)² + (λ2-MD)² + (λ3-MD)²) / sqrt(λ1² + λ2² + λ3²)
    # Simplifies to:
    numerator = AD - RD
    denominator = np.sqrt(AD * AD + 2.0 * RD * RD)
    
    if denominator < 1e-12:
        return 0.0
        
    FA = np.sqrt(0.5) * abs(numerator) / denominator
    
    # Clamp to [0, 1]
    if FA > 1.0:
        FA = 1.0
    if FA < 0.0:
        FA = 0.0
        
    return FA
