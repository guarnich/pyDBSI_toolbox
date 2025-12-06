"""
DBSI Fusion Model v3.0 - Faithful Implementation

Based on Wang et al. (2011) Brain 134:3587-3598

Key principles:
1. Joint optimization of AD, RD and fractions using iterative refinement
2. Continuous isotropic spectrum f(D) without hard boundaries during fitting
3. Thresholds (0.3, 3.0 µm²/ms) only for post-hoc interpretation
4. Data-driven parameter estimation
5. Multiple random restarts to avoid local minima

The model:
    S_k = Σ f_i * exp(-b_k * D_perp_i) * exp(-b_k * (D_par_i - D_perp_i) * cos²θ_ik)
        + ∫ f(D) * exp(-b_k * D) dD

Where the integral is discretized as a sum over L isotropic basis functions.
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from .core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from .core.solvers_v3 import (
    nnls_coordinate_descent,
    joint_optimization,
    compute_fiber_metrics,
    parse_isotropic_spectrum
)
from .calibration.optimizer import optimize_hyperparameters
from .utils.tools import estimate_snr_robust, correct_rician_bias


class DBSI_Fused:
    """
    Main DBSI Model - Faithful Implementation
    
    This version implements the DBSI algorithm as described in the original
    papers, with joint optimization of fiber diffusivities and fractions.
    
    Key improvements over simplified versions:
    1. AD and RD are optimized jointly with fractions, not separately
    2. Isotropic spectrum is continuous, thresholds are only for interpretation
    3. Multiple random restarts to find global minimum
    4. Tissue-adaptive initialization based on signal characteristics
    
    Outputs (8 channels):
        0: Fiber Fraction - apparent axonal/fiber density
        1: Restricted Fraction - cellularity (ADC ≤ 0.3 µm²/ms)
        2: Hindered Fraction - edema/demyelination (0.3 < ADC ≤ 3.0 µm²/ms)
        3: Water Fraction - CSF/free water (ADC > 3.0 µm²/ms)
        4: Fiber AD - axial diffusivity of fiber component
        5: Fiber RD - radial diffusivity of fiber component
        6: Fiber FA - fractional anisotropy
        7: Mean Isotropic ADC - weighted mean of isotropic spectrum
    """
    
    def __init__(self, n_iso=None, reg_lambda=None, n_dirs=100, 
                 iso_range=(0.0, 4.0e-3), n_restarts=3):
        """
        Parameters
        ----------
        n_iso : int
            Number of isotropic basis functions (default: auto)
        reg_lambda : float
            Regularization strength (default: auto)
        n_dirs : int
            Number of fiber directions on hemisphere
        iso_range : tuple
            (min, max) ADC range for isotropic spectrum
        n_restarts : int
            Number of random restarts for optimization
        """
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.n_dirs = n_dirs
        self.iso_range = iso_range
        self.n_restarts = n_restarts
        
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fits the DBSI model to 4D diffusion MRI data.
        """
        print(">>> Starting DBSI Pipeline v3.0 (Faithful Implementation)")
        print("    Based on Wang et al. (2011) Brain")
        
        # Normalize bvecs
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        bvals = np.asarray(bvals, dtype=np.float64)
        
        # 1. SNR Estimation
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
        
        # 2. Calibration
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            self.n_iso, self.reg_lambda = optimize_hyperparameters(bvals, bvecs, snr)
        
        if self.n_iso is None: 
            self.n_iso = 50
        if self.reg_lambda is None: 
            self.reg_lambda = 0.01  # Lower regularization for more data fidelity
            
        print(f"   Using n_iso={self.n_iso}, lambda={self.reg_lambda:.4f}")
        
        # 3. Data Preprocessing
        print("2. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        data_corr[mask] = correct_rician_bias(data[mask], sigma)
        
        # 4. Build fiber direction basis
        print("3. Building Basis Functions...")
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        print(f"   Fiber directions: {self.n_dirs}")
        print(f"   Isotropic grid: {self.n_iso} points from 0 to {self.iso_range[1]*1e3:.1f} µm²/ms")
        
        # 5. Fit
        mask_coords = np.argwhere(mask)
        n_total = len(mask_coords)
        print(f"4. Fitting {n_total:,} voxels with joint optimization...")
        
        # 8 output channels
        results = np.zeros(data.shape[:3] + (8,), dtype=np.float32)
        
        batch_size = 5000
        n_batches = int(np.ceil(n_total / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_total, desc="  Fitting", unit="vox") as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_total)
                batch_coords = mask_coords[start_idx:end_idx]
                
                self._fit_batch(
                    data_corr, batch_coords, bvals, bvecs,
                    fiber_dirs, iso_grid, self.reg_lambda, 
                    self.n_restarts, results
                )
                pbar.update(len(batch_coords))
        
        print(f"   Done in {time.time()-t0:.1f}s")
        
        # Statistics
        valid_mask = results[..., 0] > 0
        print("\n   Summary Statistics (valid voxels):")
        print(f"   Fiber Fraction: {results[valid_mask, 0].mean():.3f} ± {results[valid_mask, 0].std():.3f}")
        print(f"   Fiber AD: {results[valid_mask, 4].mean()*1e3:.2f} ± {results[valid_mask, 4].std()*1e3:.2f} µm²/ms")
        print(f"   Fiber RD: {results[valid_mask, 5].mean()*1e3:.2f} ± {results[valid_mask, 5].std()*1e3:.2f} µm²/ms")
        print(f"   Fiber FA: {results[valid_mask, 6].mean():.3f} ± {results[valid_mask, 6].std():.3f}")
        
        return results

    @staticmethod
    @njit(parallel=True)
    def _fit_batch(data, coords, bvals, bvecs, fiber_dirs, iso_grid, 
                   reg_lambda, n_restarts, out):
        """
        Parallel batch fitting with joint optimization.
        """
        n_voxels = coords.shape[0]
        n_dirs = len(fiber_dirs)
        n_iso = len(iso_grid)
        n_meas = len(bvals)
        
        for i in prange(n_voxels):
            x, y, z = coords[i]
            sig = data[x, y, z]
            
            # S0 normalization
            s0 = 0.0
            cnt = 0
            for k in range(n_meas):
                if bvals[k] < 50:
                    s0 += sig[k]
                    cnt += 1
            
            if cnt > 0:
                s0 /= cnt
            if s0 < 1e-6:
                continue
                
            sig_norm = sig / s0
            
            # Estimate tissue type from signal for initialization
            # High signal at high b = restricted, Low signal = free water
            mean_high_b = 0.0
            cnt_high_b = 0
            for k in range(n_meas):
                if bvals[k] > 800:
                    mean_high_b += sig_norm[k]
                    cnt_high_b += 1
            if cnt_high_b > 0:
                mean_high_b /= cnt_high_b
            
            # Joint optimization with multiple restarts
            best_cost = 1e20
            best_result = np.zeros(8)
            
            for restart in range(n_restarts):
                # Tissue-adaptive initialization
                if mean_high_b > 0.5:  # Likely WM or restricted
                    ad_init = 1.5e-3 + restart * 0.2e-3
                    rd_init = 0.3e-3 + restart * 0.1e-3
                elif mean_high_b > 0.2:  # Likely GM or mixed
                    ad_init = 1.0e-3 + restart * 0.15e-3
                    rd_init = 0.5e-3 + restart * 0.1e-3
                else:  # Likely CSF
                    ad_init = 2.0e-3 + restart * 0.2e-3
                    rd_init = 1.5e-3 + restart * 0.2e-3
                
                result, cost = joint_optimization(
                    bvals, bvecs, sig_norm, fiber_dirs, iso_grid,
                    ad_init, rd_init, reg_lambda
                )
                
                if cost < best_cost:
                    best_cost = cost
                    for j in range(8):
                        best_result[j] = result[j]
            
            # Store results
            for j in range(8):
                out[x, y, z, j] = best_result[j]