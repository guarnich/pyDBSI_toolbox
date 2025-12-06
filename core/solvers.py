import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def nnls_coordinate_descent(AtA, Aty, reg_lambda, tol=1e-5, max_iter=2000):
    """
    Step 1: Fast NNLS Solver using Coordinate Descent with Active Set.
    Solves: min ||Ax - y||^2 + lambda||x||^2 s.t. x >= 0
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    grad = -Aty.astype(np.float64)
    
    # Precompute Hessian diagonal with regularization
    hessian_diag = np.empty(n_features, dtype=np.float64)
    for k in range(n_features):
        hessian_diag[k] = AtA[k, k] + reg_lambda
    
    n_iter = 0
    max_update = 0.0
    
    for iteration in range(max_iter):
        max_update = 0.0
        n_changes = 0
        
        for i in range(n_features):
            g_i = grad[i] + reg_lambda * x[i]
            
            # Active Set Check (KKT conditions)
            if x[i] == 0.0 and g_i >= 0.0:
                continue
            
            # Update
            x_new = max(0.0, x[i] - g_i / hessian_diag[i])
            diff = x_new - x[i]
            
            if abs(diff) > 1e-14:
                if abs(diff) > max_update: max_update = abs(diff)
                
                # Fast gradient update O(N)
                for k in range(n_features):
                    grad[k] += AtA[k, i] * diff
                x[i] = x_new
                n_changes += 1
        
        n_iter += 1
        if n_changes == 0 or max_update < tol:
            break
            
    return x, n_iter

@njit(cache=True, fastmath=True)
def step2_refine_diffusivities(bvals, bvecs, y_norm, fiber_dir, 
                               f_fiber, f_res, f_hin, f_wat):
    """
    Step 2: Non-Linear Refinement of AD/RD.
    Uses Grid Search to avoid local minima, followed by selecting the best pair.
    """
    best_sse = 1e20
    best_ax = 1.7e-3
    best_rad = 0.3e-3
    
    # Fixed isotropic diffusivities for the forward model
    D_res = 0.1e-3
    D_hin = 1.0e-3
    D_wat = 3.0e-3
    
    # Normalize fractions
    ftot = f_fiber + f_res + f_hin + f_wat + 1e-12
    ff, fr, fh, fw = f_fiber/ftot, f_res/ftot, f_hin/ftot, f_wat/ftot
    
    # Simplified Grid Search
    ax_range = np.linspace(1.0e-3, 2.5e-3, 5)
    rad_range = np.linspace(0.1e-3, 0.8e-3, 5)
    
    for ax in ax_range:
        for rad in rad_range:
            if ax <= rad: continue # Physics constraint
            
            sse = 0.0
            for i in range(len(bvals)):
                # Fiber model
                cos_t = bvecs[i, 0]*fiber_dir[0] + bvecs[i, 1]*fiber_dir[1] + bvecs[i, 2]*fiber_dir[2]
                D_app = rad + (ax - rad) * cos_t * cos_t
                
                s_pred = (ff * np.exp(-bvals[i]*D_app) + 
                          fr * np.exp(-bvals[i]*D_res) + 
                          fh * np.exp(-bvals[i]*D_hin) + 
                          fw * np.exp(-bvals[i]*D_wat))
                
                diff = y_norm[i] - s_pred
                sse += diff*diff
            
            if sse < best_sse:
                best_sse = sse
                best_ax = ax
                best_rad = rad
                
    return best_ax, best_rad