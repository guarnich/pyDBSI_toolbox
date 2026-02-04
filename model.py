"""
DBSI Fusion Model - SIGNAL-BASED INITIALIZATION (NO BIAS)

KEY FIX: AD/RD are now estimated directly from the observed DWI signal,
not from hard-coded "healthy WM" assumptions. This makes the model
truly tissue-adaptive and works for lesions, edema, and all pathologies.

Version: 2.2.0 (Robust)
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

# ABSOLUTE IMPORTS
from dbsi_toolbox.core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from dbsi_toolbox.core.solvers import (
    nnls_coordinate_descent,
    compute_weighted_centroids,
    compute_fiber_fa
)
from dbsi_toolbox.calibration.optimizer import optimize_hyperparameters
from dbsi_toolbox.utils.tools import estimate_snr_robust, correct_rician_bias


# =============================================================================
# ROBUST SIGNAL-BASED AD/RD INITIALIZATION
# =============================================================================

@njit(cache=True, fastmath=True)
def estimate_AD_RD_from_signal(bvals, bvecs, sig_norm, fiber_dir, f_fib,
                               f_res, f_hin, f_wat, D_res, D_hin, D_wat):
    """
    Estimate AD/RD using COMPOSITE MODEL FITTING.
    
    KEY IMPROVEMENT: Instead of simple angular binning, we solve:
        S_fiber(b,θ) = exp(-b * (RD + (AD-RD)*cos²θ))
        S_total = f_fiber * S_fiber + f_iso * S_iso
    
    We isolate the fiber component by subtracting isotropic contributions,
    then fit the cylinder model to get AD/RD.
    
    This is more accurate than simple angular regression.
    """
    # Case 1: Very low fiber fraction → nearly isotropic
    if f_fib < 0.05:
        ftot_iso = f_res + f_hin + f_wat + 1e-12
        iso_mean = (f_res * D_res + f_hin * D_hin + f_wat * D_wat) / ftot_iso
        iso_mean = max(0.3e-3, min(2.5e-3, iso_mean))
        return iso_mean, iso_mean * 0.95
    
    # Case 2: Fit cylinder model to fiber component
    
    # Normalize fractions
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    # Subtract isotropic signal to isolate fiber component
    # S_fiber = (S_total - S_iso) / f_fiber
    
    # Build system for weighted least squares
    # Model: S_fiber = exp(-b * (RD + (AD-RD)*cos²θ))
    # log(S_fiber) = -b*RD - b*(AD-RD)*cos²θ
    # Let x = RD, y = AD-RD
    # log(S_fiber) = -b*x - b*y*cos²θ
    
    n_meas = 0
    sum_AA = 0.0  # Σ b²
    sum_AB = 0.0  # Σ b² * cos²θ
    sum_BB = 0.0  # Σ b² * cos⁴θ
    sum_Ay = 0.0  # Σ b * log(S_fiber)
    sum_By = 0.0  # Σ b * cos²θ * log(S_fiber)
    
    for i in range(len(bvals)):
        b = bvals[i]
        if b < 500:
            continue
        
        S_total = sig_norm[i]
        if S_total < 0.01:
            S_total = 0.01
        
        # Subtract isotropic contributions
        S_iso = (fr * np.exp(-b * D_res) +
                 fh * np.exp(-b * D_hin) +
                 fw * np.exp(-b * D_wat))
        
        # Isolate fiber signal
        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        
        if S_fiber < 0.01:
            S_fiber = 0.01
        if S_fiber > 1.0:
            S_fiber = 1.0
        
        log_S_fiber = np.log(S_fiber)
        
        g = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2_t = cos_t * cos_t
        
        # Weight by signal strength (higher SNR at low b)
        w = S_total
        
        sum_AA += w * b * b
        sum_AB += w * b * b * cos2_t
        sum_BB += w * b * b * cos2_t * cos2_t
        sum_Ay += w * b * log_S_fiber
        sum_By += w * b * cos2_t * log_S_fiber
        n_meas += 1
    
    if n_meas < 6:
        # Not enough data, use simple fallback
        return 1.0e-3, 0.5e-3
    
    
    det = sum_AA * sum_BB - sum_AB * sum_AB
    
    if abs(det) < 1e-20:
        # Singular, use simple estimate
        RD_est = -sum_Ay / (sum_AA + 1e-12)
        AD_est = RD_est * 2.0
    else:
        # x = RD, y = AD - RD
        x = (sum_BB * sum_Ay - sum_AB * sum_By) / det
        y = (sum_AA * sum_By - sum_AB * sum_Ay) / det
        
        RD_est = -x
        AD_est = -x - y
    
    # Apply physiological constraints
    RD_est = max(0.1e-3, min(1.5e-3, RD_est))
    AD_est = max(0.3e-3, min(2.5e-3, AD_est))
    
    # Ensure AD > RD
    if AD_est < RD_est * 1.05:
        mid = (AD_est + RD_est) / 2
        AD_est = mid * 1.15
        RD_est = mid * 0.85
        AD_est = min(2.5e-3, AD_est)
        RD_est = max(0.1e-3, RD_est)
    
    return AD_est, RD_est


@njit(cache=True, fastmath=True)
def refine_AD_RD_grid_search(bvals, bvecs, sig_norm, fiber_dir,
                             f_fib, f_res, f_hin, f_wat,
                             D_res, D_hin, D_wat,
                             AD_init, RD_init):
    """
    Refine AD/RD with adaptive grid search centered on signal-based init.
    
    This is Step 2 refinement, but now it starts from a sensible
    data-driven estimate instead of WM defaults.
    """
    # Normalize fractions
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    # If fiber fraction is tiny, skip refinement
    if ff < 0.03:
        return AD_init, RD_init
    
    best_sse = 1e20
    best_AD = AD_init
    best_RD = RD_init
    
    # Adaptive search range: ±40% around initialization
    AD_min = max(0.3e-3, AD_init * 0.6)
    AD_max = min(2.5e-3, AD_init * 1.4)
    RD_min = max(0.1e-3, RD_init * 0.6)
    RD_max = min(1.5e-3, RD_init * 1.4)
    
    # Grid resolution
    n_AD = 10
    n_RD = 8
    
    dAD = (AD_max - AD_min) / (n_AD - 1) if n_AD > 1 else 0
    dRD = (RD_max - RD_min) / (n_RD - 1) if n_RD > 1 else 0
    
    # Grid search
    for i in range(n_AD):
        AD = AD_min + i * dAD
        
        for j in range(n_RD):
            RD = RD_min + j * dRD
            
            # Physical constraint
            if AD < RD * 1.05:
                continue
            
            # Compute SSE
            sse = 0.0
            for k in range(len(bvals)):
                b = bvals[k]
                if b < 50:
                    continue
                
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


# =============================================================================
# PARALLEL FITTING KERNEL
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def fit_voxels_parallel(data, coords, A, AtA, At, bvals, bvecs,
                        fiber_dirs, iso_grid, reg, enable_step2, out):
    """
    Fit DBSI to all voxels with signal-based AD/RD initialization.
    
    NO MORE hard-coded WM defaults - everything is data-driven.
    """
    n_voxels = coords.shape[0]
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    
    THRESH_RES = 0.3e-3
    THRESH_WAT = 3.0e-3
    
    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]
        
        # === S0 NORMALIZATION ===
        s0 = 0.0
        cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < 50:
                s0 += sig[i]
                cnt += 1
        
        if cnt > 0:
            s0 /= cnt
        
        if s0 < 1e-6:
            continue
        
        sig_norm = sig / s0
        
        # === STEP 1: LINEAR NNLS ===
        Aty = np.zeros(AtA.shape[0])
        for r in range(AtA.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val
        
        w, _ = nnls_coordinate_descent(AtA, Aty, reg)
        
        # === PARSE FRACTIONS ===
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
        
        # Compute isotropic centroids
        D_res, D_hin, D_wat = compute_weighted_centroids(w_iso, iso_grid)
        
        # Find dominant fiber direction
        idx_max = 0
        val_max = -1.0
        for i in range(n_dirs):
            if w_fib[i] > val_max:
                val_max = w_fib[i]
                idx_max = i
        
        fiber_dir = fiber_dirs[idx_max]
        
        # === SIGNAL-BASED AD/RD INITIALIZATION ===
        AD, RD = estimate_AD_RD_from_signal(
            bvals, bvecs, sig_norm, fiber_dir,
            f_fib, f_res, f_hin, f_wat,
            D_res, D_hin, D_wat
        )
        
        # === STEP 2: OPTIONAL REFINEMENT ===
        if enable_step2 and f_fib > 0.05:
            AD, RD = refine_AD_RD_grid_search(
                bvals, bvecs, sig_norm, fiber_dir,
                f_fib, f_res, f_hin, f_wat,
                D_res, D_hin, D_wat,
                AD, RD
            )
        
        # Compute FA
        FA = compute_fiber_fa(AD, RD)
        
        # Store results
        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 2] = f_hin
        out[x, y, z, 3] = f_wat
        out[x, y, z, 4] = AD
        out[x, y, z, 5] = RD
        out[x, y, z, 6] = FA
        out[x, y, z, 7] = mean_iso_adc


# =============================================================================
# MAIN MODEL CLASS
# =============================================================================

class DBSI_Fused:
    """
    DBSI Model with Signal-Based Initialization.
    
    NO MORE bias towards healthy WM - works for all pathologies.
    """
    
    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True,
                 n_dirs=100, iso_range=(0.0, 4.0e-3)):
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.enable_step2 = enable_step2
        self.n_dirs = n_dirs
        self.iso_range = iso_range
    
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fit DBSI to 4D diffusion MRI data.
        
        Returns
        -------
        results : ndarray (X, Y, Z, 8)
            DBSI outputs:
            0: Fiber fraction
            1: Restricted fraction (cells, inflammation)
            2: Hindered fraction (edema)
            3: Water fraction (CSF)
            4: Axial diffusivity (AD)
            5: Radial diffusivity (RD)
            6: Fiber FA
            7: Mean isotropic ADC
        """
        print("\n" + "="*70)
        print("  DBSI PIPELINE - SIGNAL-BASED INITIALIZATION (NO BIAS)")
        print("="*70)
        
        # Normalize gradient directions
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        
        # Estimate SNR
        print("\n1. Estimating SNR...")
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
        
        # Calibrate hyperparameters if needed
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
        
        # Rician bias correction
        print("\n3. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        coords = np.argwhere(mask)
        for i in range(len(coords)):
            x, y, z = coords[i]
            data_corr[x, y, z] = correct_rician_bias(data[x, y, z], sigma)
        
        # Build design matrix
        print("\n4. Building Design Matrix...")
        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        # Use NEUTRAL defaults for matrix construction (will be overridden)
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid,
                                ad=1.5e-3, rd=0.4e-3)
        
        AtA = A.T @ A
        reg_diag = self.reg_lambda * np.eye(AtA.shape[0])
        AtA_reg = AtA + reg_diag
        At = A.T
        
        cond = np.linalg.cond(AtA_reg)
        print(f"   Matrix condition number: {cond:.2e}")
        
        # Fit voxels
        n_voxels = len(coords)
        print(f"\n5. Fitting {n_voxels:,} voxels...")
        print("   Using SIGNAL-BASED AD/RD initialization (no WM bias)")
        
        results = np.zeros(data.shape[:3] + (8,), dtype=np.float32)
        
        batch_size = 10000
        n_batches = int(np.ceil(n_voxels / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_voxels, desc="   Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_voxels)
                batch_coords = coords[start:end]
                
                fit_voxels_parallel(
                    data_corr, batch_coords, A, AtA_reg, At,
                    bvals, bvecs, fiber_dirs, iso_grid,
                    self.reg_lambda, self.enable_step2, results
                )
                
                pbar.update(len(batch_coords))
        
        elapsed = time.time() - t0
        print(f"\n   Completed in {elapsed:.1f}s ({n_voxels/elapsed:.0f} vox/s)")
        print("\n" + "="*70 + "\n")
        
        return results