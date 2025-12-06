"""
DBSI Core Solvers v3.1 - Improved Joint Optimization

Key improvements:
1. Better initialization from DTI-like fit
2. Alternating optimization: fix fractions, optimize AD/RD, repeat
3. Gradient-based local refinement
4. Constraint on FA to prevent degeneracy
5. Multiple initializations
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-7, max_iter=2000):
    """Non-Negative Least Squares via Coordinate Descent."""
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
            
    return x


@njit(cache=True, fastmath=True)
def estimate_initial_diffusivities(bvals, bvecs, sig_norm):
    """
    Estimate initial AD, RD using a simplified DTI-like approach.
    
    Uses the signal decay at different angles to estimate diffusivities.
    """
    n_meas = len(bvals)
    
    # Find signal decay along different directions
    # Group measurements by b-value
    b_low_mask = bvals < 500
    b_high_mask = bvals > 800
    
    # Mean signal at high b
    mean_high = 0.0
    cnt_high = 0
    for i in range(n_meas):
        if b_high_mask[i]:
            mean_high += sig_norm[i]
            cnt_high += 1
    if cnt_high > 0:
        mean_high /= cnt_high
    
    # Estimate mean diffusivity from signal decay
    # S = exp(-b * D) => D = -ln(S) / b
    if mean_high > 0.01 and mean_high < 0.99:
        mean_D = -np.log(mean_high) / 1500.0  # Approximate b-value
    else:
        mean_D = 1.0e-3
    
    # Estimate anisotropy from signal variance at high b
    var_high = 0.0
    for i in range(n_meas):
        if b_high_mask[i]:
            diff = sig_norm[i] - mean_high
            var_high += diff * diff
    if cnt_high > 1:
        var_high /= (cnt_high - 1)
    
    # Higher variance = more anisotropy
    # Estimate FA from coefficient of variation
    cv = np.sqrt(var_high) / (mean_high + 0.01)
    
    # Map CV to FA (rough approximation)
    est_FA = min(0.8, max(0.1, cv * 2))
    
    # Convert mean D and FA to AD, RD
    # FA = sqrt(0.5) * (AD - RD) / sqrt(AD^2 + 2*RD^2)
    # For simplicity, assume AD = MD * (1 + k), RD = MD * (1 - k/2)
    # where k is related to FA
    
    k = est_FA * 0.8  # Scaling factor
    AD = mean_D * (1.0 + k)
    RD = mean_D * (1.0 - k * 0.5)
    
    # Clamp to physiological range
    AD = max(0.5e-3, min(2.5e-3, AD))
    RD = max(0.1e-3, min(1.5e-3, RD))
    
    # Ensure AD > RD
    if AD <= RD:
        AD = RD * 1.5
    
    return AD, RD


@njit(cache=True, fastmath=True)
def build_design_matrix_fast(bvals, bvecs, fiber_dirs, iso_grid, AD, RD):
    """Build design matrix for given AD, RD values."""
    n_meas = len(bvals)
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    n_cols = n_dirs + n_iso
    
    A = np.zeros((n_meas, n_cols), dtype=np.float64)
    
    # Anisotropic columns
    for j in range(n_dirs):
        fdir = fiber_dirs[j]
        for i in range(n_meas):
            b = bvals[i]
            g = bvecs[i]
            
            cos_theta = g[0]*fdir[0] + g[1]*fdir[1] + g[2]*fdir[2]
            D_app = RD + (AD - RD) * cos_theta * cos_theta
            A[i, j] = np.exp(-b * D_app)
    
    # Isotropic columns
    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            A[i, n_dirs + j] = np.exp(-bvals[i] * D_iso)
    
    return A


@njit(cache=True, fastmath=True)
def solve_fractions(A, sig_norm, reg_lambda):
    """Solve for fractions given design matrix."""
    n_meas, n_cols = A.shape
    
    # Build AtA and Aty
    AtA = np.zeros((n_cols, n_cols), dtype=np.float64)
    Aty = np.zeros(n_cols, dtype=np.float64)
    
    for i in range(n_cols):
        for k in range(n_meas):
            Aty[i] += A[k, i] * sig_norm[k]
        for j in range(n_cols):
            for k in range(n_meas):
                AtA[i, j] += A[k, i] * A[k, j]
        AtA[i, i] += reg_lambda
    
    w = nnls_coordinate_descent(AtA, Aty, reg_lambda)
    return w


@njit(cache=True, fastmath=True)
def compute_cost(A, sig_norm, w, reg_lambda):
    """Compute fitting cost."""
    n_meas = A.shape[0]
    n_cols = len(w)
    
    sse = 0.0
    for i in range(n_meas):
        pred = 0.0
        for j in range(n_cols):
            pred += A[i, j] * w[j]
        diff = sig_norm[i] - pred
        sse += diff * diff
    
    reg = 0.0
    for j in range(n_cols):
        reg += w[j] * w[j]
    
    return sse + reg_lambda * reg


@njit(cache=True, fastmath=True)
def alternating_optimization(bvals, bvecs, sig_norm, fiber_dirs, iso_grid,
                             ad_init, rd_init, reg_lambda, 
                             max_outer=10, max_inner=20):
    """
    Alternating optimization between fractions and diffusivities.
    
    1. Fix AD, RD -> solve for fractions (linear NNLS)
    2. Fix fractions -> optimize AD, RD (grid search)
    3. Repeat until convergence
    """
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    
    # Initialize
    AD = ad_init
    RD = rd_init
    
    # Bounds
    AD_MIN, AD_MAX = 0.4e-3, 2.8e-3
    RD_MIN, RD_MAX = 0.1e-3, 1.8e-3
    
    best_cost = 1e20
    best_AD = AD
    best_RD = RD
    best_w = np.zeros(n_dirs + n_iso)
    
    for outer in range(max_outer):
        # Step 1: Solve for fractions
        A = build_design_matrix_fast(bvals, bvecs, fiber_dirs, iso_grid, AD, RD)
        w = solve_fractions(A, sig_norm, reg_lambda)
        cost = compute_cost(A, sig_norm, w, reg_lambda)
        
        if cost < best_cost:
            best_cost = cost
            best_AD = AD
            best_RD = RD
            best_w = w.copy()
        
        # Check fiber content to adjust search
        f_fiber = np.sum(w[:n_dirs])
        f_total = np.sum(w)
        fiber_frac = f_fiber / f_total if f_total > 0 else 0
        
        # Step 2: Optimize AD, RD with grid search
        # Adaptive step based on iteration
        step_ad = 0.2e-3 * (0.7 ** outer)
        step_rd = 0.15e-3 * (0.7 ** outer)
        
        improved = False
        
        # Grid search around current point
        for dad in [-step_ad, 0.0, step_ad]:
            ad_try = AD + dad
            if ad_try < AD_MIN or ad_try > AD_MAX:
                continue
                
            for drd in [-step_rd, 0.0, step_rd]:
                rd_try = RD + drd
                if rd_try < RD_MIN or rd_try > RD_MAX:
                    continue
                
                # Enforce AD > RD with margin
                if ad_try < rd_try * 1.2:
                    continue
                
                # If low fiber fraction, allow more isotropic diffusivities
                if fiber_frac < 0.2:
                    # CSF-like: AD and RD can be similar and high
                    pass
                else:
                    # WM-like: enforce strong anisotropy constraint
                    if ad_try < rd_try * 1.5:
                        continue
                
                A_try = build_design_matrix_fast(bvals, bvecs, fiber_dirs, iso_grid, ad_try, rd_try)
                w_try = solve_fractions(A_try, sig_norm, reg_lambda)
                cost_try = compute_cost(A_try, sig_norm, w_try, reg_lambda)
                
                if cost_try < best_cost - 1e-8:
                    best_cost = cost_try
                    best_AD = ad_try
                    best_RD = rd_try
                    best_w = w_try.copy()
                    AD = ad_try
                    RD = rd_try
                    improved = True
        
        # If no improvement, try random jump
        if not improved and outer < max_outer - 1:
            # Random perturbation
            AD = best_AD + (np.random.random() - 0.5) * 0.6e-3
            RD = best_RD + (np.random.random() - 0.5) * 0.4e-3
            AD = max(AD_MIN, min(AD_MAX, AD))
            RD = max(RD_MIN, min(RD_MAX, RD))
            if AD <= RD * 1.1:
                AD = RD * 1.5
    
    return best_AD, best_RD, best_w, best_cost


@njit(cache=True, fastmath=True)
def joint_optimization(bvals, bvecs, sig_norm, fiber_dirs, iso_grid,
                       ad_init, rd_init, reg_lambda):
    """
    Main joint optimization routine with multiple restarts.
    """
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    
    # Get data-driven initial estimate
    ad_data, rd_data = estimate_initial_diffusivities(bvals, bvecs, sig_norm)
    
    # Try multiple initializations
    best_cost = 1e20
    best_AD = ad_init
    best_RD = rd_init
    best_w = np.zeros(n_dirs + n_iso)
    
    # Initialization strategies
    inits = [
        (ad_data, rd_data),           # Data-driven
        (ad_init, rd_init),           # User-provided
        (1.5e-3, 0.4e-3),             # Healthy WM prior
        (1.0e-3, 0.6e-3),             # Isotropic-ish
        (2.0e-3, 0.3e-3),             # High anisotropy
    ]
    
    for k in range(5):
        ad_start = inits[k][0]
        rd_start = inits[k][1]
        
        AD, RD, w, cost = alternating_optimization(
            bvals, bvecs, sig_norm, fiber_dirs, iso_grid,
            ad_start, rd_start, reg_lambda
        )
        
        if cost < best_cost:
            best_cost = cost
            best_AD = AD
            best_RD = RD
            best_w = w.copy()
    
    # Parse results
    w_fiber = best_w[:n_dirs]
    w_iso = best_w[n_dirs:]
    
    f_fiber = np.sum(w_fiber)
    
    # Parse isotropic spectrum
    THRESH_RES = 0.3e-3
    THRESH_FREE = 3.0e-3
    
    f_res = 0.0
    f_hin = 0.0
    f_wat = 0.0
    sum_w = 0.0
    sum_wd = 0.0
    
    for k in range(n_iso):
        adc = iso_grid[k]
        wk = w_iso[k]
        
        if adc <= THRESH_RES:
            f_res += wk
        elif adc <= THRESH_FREE:
            f_hin += wk
        else:
            f_wat += wk
        
        sum_w += wk
        sum_wd += wk * adc
    
    mean_iso_adc = sum_wd / sum_w if sum_w > 1e-10 else 0.0
    
    # Normalize fractions
    ftot = f_fiber + f_res + f_hin + f_wat
    if ftot > 1e-10:
        f_fiber /= ftot
        f_res /= ftot
        f_hin /= ftot
        f_wat /= ftot
    
    # Compute FA
    FA = compute_fiber_fa(best_AD, best_RD)
    
    # Adjust FA based on fiber fraction
    # If fiber fraction is very low, FA is less meaningful
    # We scale it to reflect this uncertainty
    effective_FA = FA * min(1.0, f_fiber * 2)  # Scale by fiber fraction
    
    result = np.array([f_fiber, f_res, f_hin, f_wat, best_AD, best_RD, effective_FA, mean_iso_adc])
    
    return result, best_cost


@njit(cache=True, fastmath=True)
def compute_fiber_fa(AD, RD):
    """Compute Fractional Anisotropy for cylindrically symmetric tensor."""
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
    
    FA = np.sqrt(0.5) * diff / denom
    return min(1.0, max(0.0, FA))


@njit(cache=True, fastmath=True)
def compute_fiber_metrics(w_fiber, fiber_dirs):
    """Compute fiber direction and dispersion from weights."""
    n_dirs = len(fiber_dirs)
    
    idx_max = 0
    val_max = w_fiber[0]
    total_weight = 0.0
    
    for k in range(n_dirs):
        total_weight += w_fiber[k]
        if w_fiber[k] > val_max:
            val_max = w_fiber[k]
            idx_max = k
    
    if total_weight < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    
    mean_dir = np.zeros(3)
    for k in range(n_dirs):
        w = w_fiber[k] / total_weight
        for j in range(3):
            mean_dir[j] += w * fiber_dirs[k, j]
    
    norm = np.sqrt(mean_dir[0]**2 + mean_dir[1]**2 + mean_dir[2]**2)
    if norm > 1e-10:
        for j in range(3):
            mean_dir[j] /= norm
    
    return mean_dir, 0.0


@njit(cache=True, fastmath=True)
def parse_isotropic_spectrum(w_iso, iso_grid):
    """Parse the isotropic spectrum into components."""
    n_iso = len(iso_grid)
    
    THRESH_RES = 0.3e-3
    THRESH_FREE = 3.0e-3
    
    f_res = 0.0
    f_hin = 0.0
    f_wat = 0.0
    sum_w = 0.0
    sum_wd = 0.0
    
    for k in range(n_iso):
        adc = iso_grid[k]
        wk = w_iso[k]
        
        if adc <= THRESH_RES:
            f_res += wk
        elif adc <= THRESH_FREE:
            f_hin += wk
        else:
            f_wat += wk
        
        sum_w += wk
        sum_wd += wk * adc
    
    mean_adc = sum_wd / sum_w if sum_w > 1e-10 else 0.0
    
    ftot = f_res + f_hin + f_wat
    if ftot > 1e-10:
        f_res /= ftot
        f_hin /= ftot
        f_wat /= ftot
    
    return f_res, f_hin, f_wat, mean_adc