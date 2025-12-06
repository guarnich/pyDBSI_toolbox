import numpy as np
import pandas as pd
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
    """
    print(f"\n[CALIBRATION REPORT]")
    print(f"  Target: Minimize error in Restricted Fraction estimation")
    print(f"  Configuration: SNR={snr:.1f}, MC_Iterations={n_mc}")
    print("-" * 65)
    print(f"{'Bases':<6} | {'Lambda':<8} | {'MAE':<8} | {'MSE':<8} | {'Bias':<8}")
    print("-" * 65)
    
    bases_grid = [20, 40, 60]
    lambdas_grid = [0.01, 0.1, 0.5, 1.0]
    
    results = []
    
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
            raw_errors = []
            
            for i in range(n_mc):
                y = signals[i]
                w, _ = nnls_coordinate_descent(AtA, At @ y, reg)
                
                iso_w = w[150:] 
                f_res = np.sum(iso_w[iso_grid <= 0.3e-3])
                f_tot = np.sum(w)
                if f_tot > 0: f_res /= f_tot
                
                raw_errors.append(f_res - gt_cell)
                errors.append(abs(f_res - gt_cell))
            
            mae = np.mean(errors)
            mse = np.mean(np.array(raw_errors)**2)
            bias = np.mean(raw_errors)
            
            print(f"{n_iso:<6} | {reg:<8.2f} | {mae:<8.4f} | {mse:<8.4f} | {bias:<8.4f}")
            
            results.append({
                'n_iso': n_iso,
                'lambda': reg,
                'mae': mae,
                'mse': mse
            })
            
    # Selection Logic
    best_res = min(results, key=lambda x: x['mae'])
    
    # "Efficient" selection: prefer fewer bases if performance is close (within 5%)
    efficient_res = best_res
    threshold = best_res['mae'] * 1.05
    for res in sorted(results, key=lambda x: x['n_iso']): # sort by complexity
        if res['mae'] <= threshold:
            efficient_res = res
            break
            
    print("-" * 65)
    print(f"  ABSOLUTE BEST:  n_iso={best_res['n_iso']}, lambda={best_res['lambda']} (MAE={best_res['mae']:.4f})")
    print(f"  EFFICIENT BEST: n_iso={efficient_res['n_iso']}, lambda={efficient_res['lambda']} (MAE={efficient_res['mae']:.4f})")
    print("  -> Using Efficient Best configuration.")
    print("=" * 65 + "\n")
    
    return efficient_res['n_iso'], efficient_res['lambda']