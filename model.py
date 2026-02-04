"""
DBSI Model

"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

# ABSOLUTE IMPORTS
from core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from core.solvers import (
    nnls_coordinate_descent,
    compute_weighted_centroids,
    compute_fiber_fa
)
from calibration.optimizer import optimize_hyperparameters
from utils.tools import estimate_snr_robust, correct_rician_bias


DESIGN_MATRIX_AD = 1.5e-3
DESIGN_MATRIX_RD = 0.5e-3


#  ANALYTICAL AD/RD ESTIMATION (NO THRESHOLDS)

@njit(cache=True, fastmath=True)
def estimate_AD_RD_pure(bvals, bvecs, sig_norm, fiber_dir,
                        f_fib, f_res, f_hin, f_wat,
                        D_res, D_hin, D_wat):
    """
    Analytical AD/RD estimation
    
    Returns
    -------
    AD_est, RD_est : float
        Estimated diffusivities OR np.nan if fit fails
    """
    
    # Normalize fractions (NO threshold checks)
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    # Build weighted least squares system
    sum_AA = 0.0
    sum_AB = 0.0
    sum_BB = 0.0
    sum_Ay = 0.0
    sum_By = 0.0
    n_meas = 0
    
    for i in range(len(bvals)):
        b = bvals[i]
        
        S_total = sig_norm[i]
        if S_total < 0.01:
            S_total = 0.01
        
        # Subtract isotropic contributions
        S_iso = (fr * np.exp(-b * D_res) +
                 fh * np.exp(-b * D_hin) +
                 fw * np.exp(-b * D_wat))
        
        # Isolate fiber component
        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        
        # Clamp to valid range
        if S_fiber < 0.01:
            S_fiber = 0.01
        if S_fiber > 1.0:
            S_fiber = 1.0
        
        log_S = np.log(S_fiber)
        
        # Compute angle
        g = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2 = cos_t * cos_t
        
        w = S_total * b  
        
        # Accumulate
        sum_AA += w * b * b
        sum_AB += w * b * b * cos2
        sum_BB += w * b * b * cos2 * cos2
        sum_Ay += w * b * log_S
        sum_By += w * b * cos2 * log_S
        n_meas += 1
    

    det = sum_AA * sum_BB - sum_AB * sum_AB
    
    if abs(det) < 1e-20:
        # Singular system → return NaN (transparent failure)
        return np.nan, np.nan
    
    # Cramer's rule
    x = (sum_BB * sum_Ay - sum_AB * sum_By) / det
    y = (sum_AA * sum_By - sum_AB * sum_Ay) / det
    
    RD_est = -x
    AD_est = -x - y
    
    # Apply ONLY physiological bounds
    RD_est = max(0.05e-3, min(3.0e-3, RD_est))  # Wider range
    AD_est = max(0.05e-3, min(3.5e-3, AD_est))  # Wider range
    
    # Physical constraint: AD >= RD
    if AD_est < RD_est:
        # If inverted, likely isotropic → make them equal
        mean_val = (AD_est + RD_est) / 2
        AD_est = mean_val
        RD_est = mean_val
    
    return AD_est, RD_est


# =============================================================================
# PURE GRID SEARCH REFINEMENT (NO THRESHOLDS)
# =============================================================================

@njit(cache=True, fastmath=True)
def refine_AD_RD_pure(bvals, bvecs, sig_norm, fiber_dir,
                      f_fib, f_res, f_hin, f_wat,
                      D_res, D_hin, D_wat,
                      AD_init, RD_init):
    """
    
    If AD_init or RD_init is NaN, returns NaN.
    """
    
    # If initialization failed, don't try to refine
    if np.isnan(AD_init) or np.isnan(RD_init):
        return np.nan, np.nan
    
    # Normalize fractions
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    best_sse = 1e20
    best_AD = AD_init
    best_RD = RD_init
    
    # Adaptive range based on initial estimate 
    
    anisotropy = abs(AD_init - RD_init) / ((AD_init + RD_init) / 2 + 1e-12)
    
    if anisotropy > 0.5:
        # Highly anisotropic → narrow range
        range_factor = 0.25
    elif anisotropy > 0.2:
        # Moderately anisotropic
        range_factor = 0.35
    else:
        # Nearly isotropic → wide range
        range_factor = 0.50
    
    AD_min = max(0.05e-3, AD_init * (1 - range_factor))
    AD_max = min(3.5e-3, AD_init * (1 + range_factor))
    RD_min = max(0.05e-3, RD_init * (1 - range_factor))
    RD_max = min(3.0e-3, RD_init * (1 + range_factor))
    
    # Coarse grid
    n_AD = 8
    n_RD = 6
    
    dAD = (AD_max - AD_min) / (n_AD - 1) if n_AD > 1 else 0
    dRD = (RD_max - RD_min) / (n_RD - 1) if n_RD > 1 else 0
    
    for i in range(n_AD):
        AD = AD_min + i * dAD
        
        for j in range(n_RD):
            RD = RD_min + j * dRD
            
            # Only constraint: AD >= RD (physical)
            if AD < RD:
                continue
            
            # Compute SSE
            sse = 0.0
            for k in range(len(bvals)):
                b = bvals[k]
                
                g = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                
                s_pred = (ff * np.exp(-b * D_app) +
                         fr * np.exp(-b * D_res) +
                         fh * np.exp(-b * D_hin) +
                         fw * np.exp(-b * D_wat))
                
                diff = sig_norm[k] - s_pred
                sse += diff * diff
            
            if sse < best_sse:
                best_sse = sse
                best_AD = AD
                best_RD = RD
    
    # Fine grid refinement
    AD_c = best_AD
    RD_c = best_RD
    
    fine_AD = dAD / 4 if dAD > 0 else 0.05e-3
    fine_RD = dRD / 4 if dRD > 0 else 0.05e-3
    
    for di in range(-2, 3):
        AD = AD_c + di * fine_AD
        if AD < AD_min or AD > AD_max:
            continue
        
        for dj in range(-2, 3):
            RD = RD_c + dj * fine_RD
            if RD < RD_min or RD > RD_max:
                continue
            if AD < RD:
                continue
            
            sse = 0.0
            for k in range(len(bvals)):
                b = bvals[k]
                
                g = bvecs[k]
                cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
                D_app = RD + (AD - RD) * cos_t * cos_t
                
                s_pred = (ff * np.exp(-b * D_app) +
                         fr * np.exp(-b * D_res) +
                         fh * np.exp(-b * D_hin) +
                         fw * np.exp(-b * D_wat))
                
                diff = sig_norm[k] - s_pred
                sse += diff * diff
            
            if sse < best_sse:
                best_sse = sse
                best_AD = AD
                best_RD = RD
    
    return best_AD, best_RD

# PARALLEL FITTING 

@njit(parallel=True, cache=True, fastmath=True)
def fit_voxels_pure(data, coords, A, AtA, At, bvals, bvecs,
                    fiber_dirs, iso_grid, reg, enable_step2, out):
    """
    
    Output channels:
    0-7: Standard DBSI outputs
    8-9: AD_linear, RD_linear (before Step 2 refinement)
    """
    
    n_voxels = coords.shape[0]
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    
    THRESH_RES = 0.3e-3
    THRESH_WAT = 3.0e-3
    
    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]
        
        # Adaptive b0 detection
        b_min = 1e10
        for i in range(len(bvals)):
            if bvals[i] < b_min:
                b_min = bvals[i]
        
        b0_threshold = b_min + 100
        
        # S0 normalization
        s0 = 0.0
        cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < b0_threshold:
                s0 += sig[i]
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
        
        for i in range(n_iso):
            adc = iso_grid[i]
            wi = w_iso[i]
            
            if adc <= THRESH_RES:
                f_res += wi
            elif adc <= THRESH_WAT:
                f_hin += wi
            else:
                f_wat += wi
            
            sum_w_iso += wi
            sum_wd_iso += wi * adc
        
        mean_iso_adc = sum_wd_iso / sum_w_iso if sum_w_iso > 1e-10 else 0.0
        
        # Normalize fractions
        ftot = f_fib + f_res + f_hin + f_wat
        if ftot > 1e-10:
            f_fib /= ftot
            f_res /= ftot
            f_hin /= ftot
            f_wat /= ftot
        else:
            continue
        
        # Step 2: Non-linear fit 

        AD = np.nan
        RD = np.nan
        FA = np.nan
        AD_linear = np.nan
        RD_linear = np.nan
        
        D_res, D_hin, D_wat = compute_weighted_centroids(w_iso, iso_grid)
        
        idx_max = 0
        val_max = -1.0
        for i in range(n_dirs):
            if w_fib[i] > val_max:
                val_max = w_fib[i]
                idx_max = i
        
        fiber_dir = fiber_dirs[idx_max]
        
        AD_linear, RD_linear = estimate_AD_RD_pure(
            bvals, bvecs, sig_norm, fiber_dir,
            f_fib, f_res, f_hin, f_wat,
            D_res, D_hin, D_wat
        )
        
        AD = AD_linear
        RD = RD_linear
        
        if enable_step2:  
            AD, RD = refine_AD_RD_pure(
                bvals, bvecs, sig_norm, fiber_dir,
                f_fib, f_res, f_hin, f_wat,
                D_res, D_hin, D_wat,
                AD_linear, RD_linear
            )
        
        if not np.isnan(AD) and not np.isnan(RD):
            FA = compute_fiber_fa(AD, RD)
        
        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 2] = f_hin
        out[x, y, z, 3] = f_wat
        out[x, y, z, 4] = AD
        out[x, y, z, 5] = RD
        out[x, y, z, 6] = FA
        out[x, y, z, 7] = mean_iso_adc
        out[x, y, z, 8] = AD_linear 
        out[x, y, z, 9] = RD_linear 


class DBSI_Fused:
    """
    DBSI Model

    """
    
    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True,
                 n_dirs=100, iso_range=(0.0, 4.5e-3)):
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.enable_step2 = enable_step2
        self.n_dirs = n_dirs
        self.iso_range = iso_range
    
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fit DBSI with ZERO THRESHOLDS approach.
        
        Returns
        -------
        results : ndarray (X, Y, Z, 10)
            0: Fiber fraction
            1: Restricted fraction (inflammation/cells)
            2: Hindered fraction (edema)
            3: Water fraction (CSF)
            4: Axial diffusivity (AD) - final (after Step 2 if enabled)
            5: Radial diffusivity (RD) - final (after Step 2 if enabled)
            6: Fiber FA
            7: Mean isotropic ADC
            8: AD_linear - from analytical estimation (before Step 2)
            9: RD_linear - from analytical estimation (before Step 2)
            
            AD/RD/FA will be NaN if fit fails
        """
        print("\n" + "="*70)
        print("  DBSI PIPELINE  ")
        print("="*70)

        # Normalize gradients
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        
        # SNR estimation
        print("\n1. Estimating SNR...")
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
        
        # Calibration
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            print("\n2. Running Monte Carlo Calibration...")
            self.n_iso, self.reg_lambda = optimize_hyperparameters(
                bvals, bvecs, snr, n_mc=1000
            )
        
        if self.n_iso is None:
            self.n_iso = 60
        if self.reg_lambda is None:
            self.reg_lambda = 0.05
        
        print(f"\n   Hyperparameters: n_iso={self.n_iso}, λ={self.reg_lambda:.4f}")
        
        # Rician correction
        print("\n3. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        coords = np.argwhere(mask)
        for i in range(len(coords)):
            x, y, z = coords[i]
            data_corr[x, y, z] = correct_rician_bias(data[x, y, z], sigma)
        
        # Design matrix
        print("\n4. Building Design Matrix...")
        print(f"   Using AD={DESIGN_MATRIX_AD*1e3:.2f}, RD={DESIGN_MATRIX_RD*1e3:.2f} µm²/ms")
        
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid,
                                ad=DESIGN_MATRIX_AD, rd=DESIGN_MATRIX_RD)
        
        AtA = A.T @ A
        reg_diag = self.reg_lambda * np.eye(AtA.shape[0])
        AtA_reg = AtA + reg_diag
        At = A.T
        
        cond = np.linalg.cond(AtA_reg)
        print(f"   Matrix condition number: {cond:.2e}")
        
        # Fit
        n_voxels = len(coords)
        print(f"\n5. Fitting {n_voxels:,} voxels...")
        print("   Approach: ZERO THRESHOLDS (pure data-driven)")
        print("   - AD/RD estimated for ALL voxel (even f_fib→0)")
        print("   - Step 2 applied to ALL if enabled (no f_fib check)")
        print("   - NaN indicates natural fit failure")
        
        results = np.zeros(data.shape[:3] + (10,), dtype=np.float32) 
        
        results[..., 4] = np.nan  # AD
        results[..., 5] = np.nan  # RD
        results[..., 6] = np.nan  # FA
        results[..., 8] = np.nan  # AD_linear
        results[..., 9] = np.nan  # RD_linear
        
        batch_size = 10000
        n_batches = int(np.ceil(n_voxels / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_voxels, desc="   Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_voxels)
                batch_coords = coords[start:end]
                
                fit_voxels_pure(
                    data_corr, batch_coords, A, AtA_reg, At,
                    bvals, bvecs, fiber_dirs, iso_grid,
                    self.reg_lambda, self.enable_step2, results
                )
                
                pbar.update(len(batch_coords))
        
        elapsed = time.time() - t0
        
        
        print(f"\n   Completed in {elapsed:.1f}s ({n_voxels/elapsed:.0f} vox/s)")
        print("\n" + "="*70 + "\n")
        
        return results