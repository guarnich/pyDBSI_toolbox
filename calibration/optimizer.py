import numpy as np
from ..core.basis import build_design_matrix, generate_fibonacci_sphere
from ..core.solvers import nnls_coordinate_descent

def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3):
    """Generates synthetic signal with Rician noise for calibration."""
    N = len(bvals)
    D_ax, D_rad = 1.7e-3, 0.3e-3
    D_cell, D_wat = 0.0, 3.0e-3
    f_wat = 1.0 - f_fiber - f_cell
    
    # Random fiber direction
    v = np.random.randn(3); v /= np.linalg.norm(v)
    
    s = np.zeros(N)
    for i in range(N):
        cos_t = np.dot(bvecs[i], v)
        D_app = D_rad + (D_ax - D_rad)*cos_t**2
        s[i] = (f_fiber * np.exp(-bvals[i]*D_app) + 
                f_cell * np.exp(-bvals[i]*D_cell) + 
                f_wat * np.exp(-bvals[i]*D_wat))
    
    # Add Rician noise
    sigma = 1.0/snr
    n1 = np.random.normal(0, sigma, N)
    n2 = np.random.normal(0, sigma, N)
    return np.sqrt((s+n1)**2 + n2**2)

def optimize_hyperparameters(bvals, bvecs, snr, n_mc=300):
    """
    Monte Carlo optimization to find best (n_iso, lambda).
    Target: Minimize error in estimating Restricted Fraction (Cellularity).
    
    Args:
        bvals: Acquisition b-values
        bvecs: Acquisition b-vectors
        snr: Estimated SNR
        n_mc: Number of Monte Carlo iterations
        
    Returns:
        tuple: (optimal_n_iso, optimal_lambda)
    """
    print(f"\n[Calibration] Running Monte Carlo Optimization (SNR={snr:.1f})...")
    
    bases_grid = [20, 40, 60]
    lambdas_grid = [0.01, 0.1, 0.5, 1.0]
    
    best_mae = np.inf
    best_config = (50, 0.1) # Default
    
    fiber_dirs = generate_fibonacci_sphere(150)
    
    # Pre-generate synthetic data (GT cellularity = 0.3)
    gt_cell = 0.3
    signals = [generate_synthetic_signal(bvals, bvecs, snr, f_cell=gt_cell) for _ in range(n_mc)]
    signals = np.array(signals)
    
    for n_iso in bases_grid:
        iso_grid = np.linspace(0, 3.0e-3, n_iso)
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid)
        AtA = A.T @ A
        At = A.T
        
        for reg in lambdas_grid:
            errors = []
            for i in range(n_mc):
                y = signals[i]
                w, _ = nnls_coordinate_descent(AtA, At @ y, reg)
                
                # Parse restricted fraction (D <= 0.3)
                # 150 fiber bases come first
                iso_w = w[150:] 
                f_res = np.sum(iso_w[iso_grid <= 0.3e-3])
                f_tot = np.sum(w)
                if f_tot > 0: f_res /= f_tot
                
                errors.append(abs(f_res - gt_cell))
            
            mae = np.mean(errors)
            if mae < best_mae:
                best_mae = mae
                best_config = (n_iso, reg)
                
    print(f"[Calibration] Optimal Parameters: n_iso={best_config[0]}, lambda={best_config[1]}")
    return best_config