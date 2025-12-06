"""
DBSI Calibration: Hyperparameter Optimization

Corrected Version:
- Fixed ADC thresholds (0.3, 3.0)
- Extended isotropic range
- Uses hemisphere directions
"""

import numpy as np
from ..core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from ..core.solvers import nnls_coordinate_descent


THRESH_RESTRICTED = 0.3e-3
THRESH_FREE = 3.0e-3


def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3):
    """
    Generates synthetic DBSI signal with Rician noise for calibration.
    
    Parameters
    ----------
    bvals : ndarray (N,)
        B-values
    bvecs : ndarray (N, 3)
        Gradient directions
    snr : float
        Signal-to-noise ratio
    f_fiber : float
        Fiber fraction (default 0.5)
    f_cell : float
        Cell/restricted fraction (default 0.3)
        
    Returns
    -------
    signal : ndarray (N,)
        Noisy signal with Rician distribution
    """
    N = len(bvals)
    D_ax, D_rad = 1.7e-3, 0.3e-3
    D_cell = 0.1e-3   # Restricted diffusion
    D_wat = 3.0e-3    # Free water
    f_wat = 1.0 - f_fiber - f_cell
    
    # Random fiber direction
    v = np.random.randn(3)
    v /= np.linalg.norm(v)
    
    # Ensure on upper hemisphere for consistency
    if v[2] < 0:
        v = -v
    
    s = np.zeros(N)
    for i in range(N):
        if bvals[i] < 50:  # b0
            cos_t = 0
        else:
            cos_t = np.dot(bvecs[i], v)
            
        D_app = D_rad + (D_ax - D_rad) * cos_t**2
        s[i] = (f_fiber * np.exp(-bvals[i] * D_app) + 
                f_cell * np.exp(-bvals[i] * D_cell) + 
                f_wat * np.exp(-bvals[i] * D_wat))
    
    # Add Rician noise
    sigma = 1.0 / snr
    n1 = np.random.normal(0, sigma, N)
    n2 = np.random.normal(0, sigma, N)
    return np.sqrt((s + n1)**2 + n2**2)


def optimize_hyperparameters(bvals, bvecs, snr, n_mc=1000):
    """
    Monte Carlo optimization to find best (n_iso, lambda) hyperparameters.
    
    Optimizes for accuracy in restricted fraction estimation, which is
    the primary inflammation biomarker in DBSI.
    
    Parameters
    ----------
    bvals : ndarray
        B-values
    bvecs : ndarray
        Gradient directions
    snr : float
        Estimated SNR
    n_mc : int
        Number of Monte Carlo iterations
        
    Returns
    -------
    best_n_iso : int
        Optimal number of isotropic bases
    best_lambda : float
        Optimal regularization strength
    """
    print(f"\n[CALIBRATION REPORT]")
    print(f"  Target: Minimize error in Restricted Fraction estimation")
    print(f"  Configuration: SNR={snr:.1f}, MC_Iterations={n_mc}")
    print("-" * 65)
    print(f"{'Bases':<6} | {'Lambda':<8} | {'MAE':<8} | {'MSE':<8} | {'Bias':<8}")
    print("-" * 65)
    
    # Grid of hyperparameters to test
    bases_grid = [25, 50, 75, 100, 125, 150]
    lambdas_grid = [0.01, 0.1, 0.5, 1.0, 2.0]
    
    results = []
    
    # CORRECTED: Use hemisphere directions
    n_dirs = 100
    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)
    
    # Ground truth for calibration
    gt_cell = 0.3
    
    # Pre-generate synthetic data
    np.random.seed(42)  # For reproducibility
    signals = [generate_synthetic_signal(bvals, bvecs, snr, f_cell=gt_cell) 
               for _ in range(n_mc)]
    signals = np.array(signals)
    
    for n_iso in bases_grid:
        # CORRECTED: Extended range
        iso_grid = np.linspace(0, 4.0e-3, n_iso)
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid)
        AtA = A.T @ A
        At = A.T
        
        for reg in lambdas_grid:
            # Add regularization to AtA
            AtA_reg = AtA + reg * np.eye(AtA.shape[0])
            
            errors = []
            raw_errors = []
            
            for i in range(n_mc):
                y = signals[i]
                Aty = At @ y
                w, _ = nnls_coordinate_descent(AtA_reg, Aty, reg)
                
                # Parse isotropic weights with CORRECTED thresholds
                iso_w = w[n_dirs:]
                f_res = np.sum(iso_w[iso_grid <= THRESH_RESTRICTED])
                f_tot = np.sum(w)
                
                if f_tot > 0:
                    f_res /= f_tot
                
                raw_errors.append(f_res - gt_cell)
                errors.append(abs(f_res - gt_cell))
            
            mae = np.mean(errors)
            mse = np.mean(np.array(raw_errors)**2)
            bias = np.mean(raw_errors)
            
            print(f"{n_iso:<6} | {reg:<8.2f} | {mae:<8.4f} | {mse:<8.4f} | {bias:+8.4f}")
            
            results.append({
                'n_iso': n_iso,
                'lambda': reg,
                'mae': mae,
                'mse': mse,
                'bias': abs(bias)
            })
            
    # Selection Logic: minimize MAE, then prefer lower bias
    best_res = min(results, key=lambda x: (x['mae'], x['bias']))
    
    # "Efficient" selection: prefer fewer bases if performance is close (within 5%)
    efficient_res = best_res
    threshold = best_res['mae'] * 1.05
    
    for res in sorted(results, key=lambda x: x['n_iso']):
        if res['mae'] <= threshold:
            efficient_res = res
            break
            
    print("-" * 65)
    print(f"  ABSOLUTE BEST:  n_iso={best_res['n_iso']}, lambda={best_res['lambda']:.2f} (MAE={best_res['mae']:.4f})")
    print(f"  EFFICIENT BEST: n_iso={efficient_res['n_iso']}, lambda={efficient_res['lambda']:.2f} (MAE={efficient_res['mae']:.4f})")
    print("  -> Using Efficient Best configuration.")
    print("=" * 65 + "\n")
    
    return efficient_res['n_iso'], efficient_res['lambda']
