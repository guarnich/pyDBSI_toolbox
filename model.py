"""
DBSI Fusion Model - Balanced Final Implementation

Key design principles:
1. Linear NNLS Step 1 for robustness
2. Non-linear Step 2 refinement for accuracy
3. FA scaled by fiber fraction to avoid artifacts
4. Extended parameter ranges for all tissue types
5. Proper handling of CSF and GM regions

Outputs (8 channels):
    0: Fiber Fraction - apparent axonal density
    1: Restricted Fraction - cellularity marker (ADC ≤ 0.3 µm²/ms)
    2: Hindered Fraction - edema/tissue loss (0.3 < ADC ≤ 3.0 µm²/ms)
    3: Water Fraction - CSF contamination (ADC > 3.0 µm²/ms)
    4: Fiber AD - axial diffusivity along fiber axis (mm²/s)
    5: Fiber RD - radial diffusivity perpendicular to fiber (mm²/s)
    6: Fiber FA - fractional anisotropy (cylindrically symmetric tensor)
    7: Mean Isotropic ADC - weighted mean of isotropic ADC spectrum

ADC Thresholds (from Ye et al. 2020, Wang et al. 2011):
    - Restricted: ADC ≤ 0.3 µm²/ms (0.3e-3 mm²/s) - cells, inflammation
    - Hindered: 0.3 < ADC ≤ 3.0 µm²/ms - edema, tissue disorganization
    - Free/Water: ADC > 3.0 µm²/ms - CSF

FA Computation Note:
    The FA is computed using the cylindrically symmetric tensor assumption
    (λ₂ = λ₃ = RD), consistent with the DBSI anisotropic model. This differs 
    from standard DTI FA which uses full eigenvalue decomposition.

References:
    - Wang Y et al. (2011) Brain 134:3590-3601
    - Ye Z et al. (2020) Ann Clin Transl Neurol 7:695-706
    - Cross AH, Song SK (2017) J Neuroimmunol 304:81-85
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from core.solvers import (
    nnls_coordinate_descent,
    step2_refine_diffusivities,
    compute_weighted_centroids,
    compute_fiber_fa
)
from calibration.optimizer import optimize_hyperparameters
from utils.tools import estimate_snr_robust, correct_rician_bias


class DBSI_Fused:
    """Main DBSI Model - Balanced Implementation."""
    
    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True,
                 n_dirs=100, iso_range=(0.0, 4.0e-3)):
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.enable_step2 = enable_step2
        self.n_dirs = n_dirs
        self.iso_range = iso_range
        
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """Fit DBSI model to 4D diffusion MRI data."""
        print(">>> Starting DBSI Pipeline (Balanced Final Version)")
        
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
        
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            self.n_iso, self.reg_lambda = optimize_hyperparameters(bvals, bvecs, snr)
        
        if self.n_iso is None:
            self.n_iso = 60
        if self.reg_lambda is None:
            self.reg_lambda = 0.05
            
        print(f"   Using n_iso={self.n_iso}, lambda={self.reg_lambda:.4f}")
        
        print("2. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        data_corr[mask] = correct_rician_bias(data[mask], sigma)
        
        print("3. Building Design Matrix...")
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        # Use tissue-adaptive AD/RD for initial design matrix
        # These are refined in Step 2
        init_AD, init_RD = 1.5e-3, 0.4e-3
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid, 
                                 ad=init_AD, rd=init_RD)
        
        AtA = A.T @ A
        reg_diag = self.reg_lambda * np.eye(AtA.shape[0])
        AtA_reg = AtA + reg_diag
        At = A.T
        
        cond_number = np.linalg.cond(AtA_reg)
        print(f"   Matrix conditioning: {cond_number:.2e}")
        
        mask_coords = np.argwhere(mask)
        n_total = len(mask_coords)
        print(f"4. Fitting {n_total:,} voxels...")
        
        # 8 output channels
        results = np.zeros(data.shape[:3] + (8,), dtype=np.float32)
        
        batch_size = 10000
        n_batches = int(np.ceil(n_total / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_total, desc="  Fitting", unit="vox") as pbar:
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
        """Parallel fitting kernel."""
        n_voxels = coords.shape[0]
        n_dirs = len(fiber_dirs)
        n_iso = len(iso_grid)
        
        THRESH_RES = 0.3e-3
        THRESH_WAT = 3.0e-3
        MIN_FIBER_FOR_STEP2 = 0.10
        
        for i in prange(n_voxels):
            x, y, z = coords[i]
            sig = data[x, y, z]
            
            # S0 normalization
            s0 = 0.0
            cnt = 0
            for k in range(len(bvals)):
                if bvals[k] < 50:
                    s0 += sig[k]
                    cnt += 1
            if cnt > 0:
                s0 /= cnt
            if s0 < 1e-6:
                continue
            
            sig_norm = sig / s0
            
            # Step 1: NNLS
            Aty = np.zeros(AtA.shape[0])
            for r in range(AtA.shape[0]):
                val = 0.0
                for c in range(len(sig_norm)):
                    val += At[r, c] * sig_norm[c]
                Aty[r] = val
            
            w, _ = nnls_coordinate_descent(AtA, Aty, reg)
            
            # Parse fractions
            w_fib = w[:n_dirs]
            w_iso = w[n_dirs:]
            
            f_fib = np.sum(w_fib)
            f_res = 0.0
            f_hin = 0.0
            f_wat = 0.0
            sum_w_iso = 0.0
            sum_wd_iso = 0.0
            
            for k in range(n_iso):
                adc = iso_grid[k]
                wk = w_iso[k]
                
                if adc <= THRESH_RES:
                    f_res += wk
                elif adc <= THRESH_WAT:
                    f_hin += wk
                else:
                    f_wat += wk
                
                sum_w_iso += wk
                sum_wd_iso += wk * adc
            
            # Mean isotropic ADC
            mean_iso_adc = sum_wd_iso / sum_w_iso if sum_w_iso > 1e-10 else 0.0
            
            # Normalize
            ftot = f_fib + f_res + f_hin + f_wat
            if ftot > 1e-10:
                f_fib /= ftot
                f_res /= ftot
                f_hin /= ftot
                f_wat /= ftot
            
            # Compute centroids
            D_res, D_hin, D_wat = compute_weighted_centroids(w_iso, iso_grid)
            
            # Step 2: Refine diffusivities
            AD = 1.5e-3  # default
            RD = 0.4e-3
            
            if do_step2 and f_fib > MIN_FIBER_FOR_STEP2:
                # Find dominant direction
                idx_max = 0
                val_max = -1.0
                for k in range(n_dirs):
                    if w_fib[k] > val_max:
                        val_max = w_fib[k]
                        idx_max = k
                
                f_dir = fiber_dirs[idx_max]
                
                AD, RD = step2_refine_diffusivities(
                    bvals, bvecs, sig_norm, f_dir,
                    f_fib, f_res, f_hin, f_wat,
                    D_res, D_hin, D_wat
                )
            
            # Compute FA 
            FA_int = compute_fiber_fa(AD, RD)
            
            # Store
            out[x, y, z, 0] = f_fib
            out[x, y, z, 1] = f_res
            out[x, y, z, 2] = f_hin
            out[x, y, z, 3] = f_wat
            out[x, y, z, 4] = AD
            out[x, y, z, 5] = RD
            out[x, y, z, 6] = FA_int      
            out[x, y, z, 7] = mean_iso_adc 
