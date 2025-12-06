"""
DBSI Fusion Model - Main Pipeline

Corrected Version:
- Fixed ADC thresholds per literature (0.3e-3, 3.0e-3)
- Extended isotropic range to 4.0e-3
- Uses hemisphere fiber directions
- Added Fiber FA to outputs
- Dynamic isotropic centroids for Step 2
- Improved matrix conditioning with regularization
- Added output channel 6 for Fiber FA
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from .core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from .core.solvers import (
    nnls_coordinate_descent, 
    step2_refine_diffusivities,
    compute_weighted_centroids,
    compute_fiber_fa
)
from .calibration.optimizer import optimize_hyperparameters
from .utils.tools import estimate_snr_robust, correct_rician_bias


# CORRECTED: Literature-standard ADC thresholds
# References: Ye et al. 2020 (Annals Clin Trans Neurol), Wang et al. 2011 (Brain)
THRESH_RESTRICTED = 0.3e-3   # ADC ≤ 0.3 µm²/ms for restricted (cells)
THRESH_FREE = 3.0e-3          # ADC > 3.0 µm²/ms for free water (CSF)


class DBSI_Fused:
    """
    Main DBSI Model orchestrating Preprocessing, Basis Construction and Fitting.
    
    Implements the two-step DBSI approach:
    - Step 1: Linear decomposition to estimate fractions and fiber directions
    - Step 2: Non-linear refinement of fiber diffusivities (AD, RD)
    
    Outputs (7 channels):
        0: Fiber Fraction - apparent axonal density
        1: Restricted Fraction - cellularity marker (ADC ≤ 0.3)
        2: Hindered Fraction - edema/tissue loss (0.3 < ADC ≤ 3.0)
        3: Water Fraction - CSF contamination (ADC > 3.0)
        4: Axial Diffusivity (AD) - along fiber axis
        5: Radial Diffusivity (RD) - perpendicular to fiber
        6: Fiber FA - fractional anisotropy of fiber component
        
    Parameters
    ----------
    n_iso : int, optional
        Number of isotropic basis functions. If None, determined by calibration.
    reg_lambda : float, optional
        L2 regularization strength. If None, determined by calibration.
    enable_step2 : bool
        Whether to run Step 2 refinement (default True)
    n_dirs : int
        Number of fiber directions on hemisphere (default 100)
    iso_range : tuple
        (min, max) ADC range for isotropic spectrum in mm²/s
    """
    
    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True, 
                 n_dirs=100, iso_range=(0.0, 4.0e-3)):
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.enable_step2 = enable_step2
        self.n_dirs = n_dirs
        # CORRECTED: Extended range to 4.0e-3 to capture free water
        self.iso_range = iso_range
        
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fits the DBSI model to 4D diffusion MRI data.
        
        Parameters
        ----------
        data : ndarray (X, Y, Z, N_meas)
            4D diffusion-weighted image data
        bvals : ndarray (N_meas,)
            B-values in s/mm²
        bvecs : ndarray (N_meas, 3)
            Gradient directions (will be normalized)
        mask : ndarray (X, Y, Z)
            Boolean brain mask
        run_calibration : bool
            Whether to run Monte Carlo hyperparameter optimization
            
        Returns
        -------
        results : ndarray (X, Y, Z, 7)
            Parameter maps: [fiber_frac, restricted, hindered, water, AD, RD, FA]
        """
        print(">>> Starting DBSI Pipeline (Corrected Version)")
        
        # Ensure bvecs are normalized
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        
        # 1. SNR Estimation
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
        
        # 2. Calibration (if needed)
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            self.n_iso, self.reg_lambda = optimize_hyperparameters(bvals, bvecs, snr)
        
        # Default fallbacks
        if self.n_iso is None: 
            self.n_iso = 60
        if self.reg_lambda is None: 
            self.reg_lambda = 0.1
            
        print(f"   Using n_iso={self.n_iso}, lambda={self.reg_lambda:.3f}")
        
        # 3. Data Preprocessing
        print("2. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        data_corr[mask] = correct_rician_bias(data[mask], sigma)
        
        # 4. Basis Construction
        print("3. Building Design Matrix...")
        # CORRECTED: Use hemisphere directions for proper antipodal handling
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid)
        
        # CORRECTED: Add regularization to AtA for numerical stability
        AtA = A.T @ A
        # Add small diagonal term to improve conditioning
        reg_diag = self.reg_lambda * np.eye(AtA.shape[0])
        AtA_reg = AtA + reg_diag
        
        At = A.T
        
        # Report matrix properties
        cond_number = np.linalg.cond(AtA_reg)
        print(f"   Matrix conditioning (with reg): {cond_number:.2e}")
        
        # 5. Parallel Fitting with Progress Bar
        mask_coords = np.argwhere(mask)
        n_total = len(mask_coords)
        print(f"4. Fitting {n_total:,} voxels (Two-Step)...")
        
        # CORRECTED: 7 output channels (added Fiber FA)
        results = np.zeros(data.shape[:3] + (7,), dtype=np.float32)
        
        # Process in chunks to update tqdm
        batch_size = 10000 
        n_batches = int(np.ceil(n_total / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_total, desc="  Fitting Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_total)
                batch_coords = mask_coords[start_idx:end_idx]
                
                self._fit_batch_kernel(
                    data_corr, batch_coords, A, AtA_reg, At, bvals, bvecs,
                    fiber_dirs, iso_grid, self.reg_lambda, self.enable_step2, results
                )
                pbar.update(len(batch_coords))
        
        print(f"   Done in {time.time()-t0:.1f}s")
        return results

    @staticmethod
    @njit(parallel=True)
    def _fit_batch_kernel(data, coords, A, AtA, At, bvals, bvecs, 
                          fiber_dirs, iso_grid, reg, do_step2, out):
        """
        Numba-accelerated fitting kernel for a batch of voxels.
        
        Implements the full two-step DBSI pipeline per voxel.
        """
        n_voxels = coords.shape[0]
        n_dirs = len(fiber_dirs)
        n_iso = len(iso_grid)
        
        # CORRECTED thresholds per literature
        THRESH_RES = 0.3e-3   # Restricted/Hindered boundary
        THRESH_WAT = 3.0e-3   # Hindered/Free boundary
        
        for i in prange(n_voxels):
            x, y, z = coords[i]
            sig = data[x, y, z]
            
            # =============== S0 Normalization ===============
            s0 = 0.0
            cnt = 0
            for k in range(len(bvals)):
                if bvals[k] < 50:  # b0 volumes
                    s0 += sig[k]
                    cnt += 1
            
            if cnt > 0:
                s0 = s0 / cnt
            
            if s0 < 1e-6: 
                continue  # Skip voxel if no valid b0 signal
                
            sig_norm = sig / s0
            
            # =============== Step 1: NNLS ===============
            # Compute A^T y
            Aty = np.zeros(AtA.shape[0])
            for r in range(AtA.shape[0]):
                val = 0.0
                for c in range(len(sig_norm)):
                    val += At[r, c] * sig_norm[c]
                Aty[r] = val
                
            w, _ = nnls_coordinate_descent(AtA, Aty, reg)
            
            # =============== Parse Step 1 Results ===============
            w_fib = w[:n_dirs]
            w_iso = w[n_dirs:]
            
            f_fib = np.sum(w_fib)
            f_res = 0.0
            f_hin = 0.0
            f_wat = 0.0
            
            # CORRECTED: Use literature thresholds
            for k in range(n_iso):
                adc = iso_grid[k]
                if adc <= THRESH_RES:
                    f_res += w_iso[k]
                elif adc <= THRESH_WAT:
                    f_hin += w_iso[k]
                else:
                    f_wat += w_iso[k]
            
            # Normalize fractions
            ftot = f_fib + f_res + f_hin + f_wat
            if ftot > 1e-10:
                f_fib /= ftot
                f_res /= ftot
                f_hin /= ftot
                f_wat /= ftot
            
            # =============== Compute Isotropic Centroids ===============
            # For Step 2: use weighted centroids instead of fixed values
            D_res_centroid = 0.15e-3  # default
            D_hin_centroid = 1.0e-3
            D_wat_centroid = 3.0e-3
            
            sum_w_res, sum_wd_res = 0.0, 0.0
            sum_w_hin, sum_wd_hin = 0.0, 0.0
            sum_w_wat, sum_wd_wat = 0.0, 0.0
            
            for k in range(n_iso):
                adc = iso_grid[k]
                wk = w_iso[k]
                
                if adc <= THRESH_RES:
                    sum_w_res += wk
                    sum_wd_res += wk * adc
                elif adc <= THRESH_WAT:
                    sum_w_hin += wk
                    sum_wd_hin += wk * adc
                else:
                    sum_w_wat += wk
                    sum_wd_wat += wk * adc
                    
            if sum_w_res > 1e-10:
                D_res_centroid = sum_wd_res / sum_w_res
            if sum_w_hin > 1e-10:
                D_hin_centroid = sum_wd_hin / sum_w_hin
            if sum_w_wat > 1e-10:
                D_wat_centroid = sum_wd_wat / sum_w_wat
            
            # =============== Step 2: Refine Diffusivities ===============
            AD, RD = 1.7e-3, 0.3e-3  # Default values
            
            if do_step2 and f_fib > 0.1:
                # Find dominant fiber direction
                idx_max = 0
                val_max = -1.0
                for k in range(n_dirs):
                    if w_fib[k] > val_max:
                        val_max = w_fib[k]
                        idx_max = k
                
                f_dir = fiber_dirs[idx_max]
                
                # CORRECTED: Pass dynamic centroids to Step 2
                AD, RD = step2_refine_diffusivities(
                    bvals, bvecs, sig_norm, f_dir,
                    f_fib, f_res, f_hin, f_wat,
                    D_res_centroid, D_hin_centroid, D_wat_centroid
                )
            
            # =============== Compute Fiber FA ===============
            FA = compute_fiber_fa(AD, RD)
            
            # =============== Store Results ===============
            out[x, y, z, 0] = f_fib
            out[x, y, z, 1] = f_res
            out[x, y, z, 2] = f_hin
            out[x, y, z, 3] = f_wat
            out[x, y, z, 4] = AD
            out[x, y, z, 5] = RD
            out[x, y, z, 6] = FA  # NEW: Fiber FA output
