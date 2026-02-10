"""
DBSI Model with RMSE-Based Adaptive Threshold

Key Innovation:
- Threshold based on Step 1 fit quality (RMSE), not external SNR
- More robust and always available
- Intrinsic to the model (no external assumptions)

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


# ============================================================================
# RMSE-BASED ADAPTIVE THRESHOLD
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_adaptive_threshold_rmse(f_fib, f_res, rmse):
    """
    Compute adaptive threshold based on Step 1 fit quality.
    
    Parameters
    ----------
    f_fib : float
        Fiber fraction from Step 1
    f_res : float
        Restricted fraction (pathology marker)
    rmse : float
        Root mean square error from Step 1 NNLS fit
        Typical range: 0.02-0.20 for normalized signal
        
    Returns
    -------
    threshold : float
        Adaptive threshold for f_fib
        
    Typical Ranges:
    - Excellent fit (RMSE < 0.05): threshold = 0.15
    - Good fit (RMSE 0.05-0.08): threshold = 0.18
    - Fair fit (RMSE 0.08-0.12): threshold = 0.22
    - Poor fit (RMSE 0.12-0.18): threshold = 0.28
    - Very poor fit (RMSE > 0.18): threshold = 0.35
    
    """
    
    # PRIMARY FACTOR: FIT QUALITY (RMSE)
    # ========================================================================
    # Lower RMSE = better fit = more permissive threshold
    # Higher RMSE = poor fit = more conservative threshold
    
    if rmse < 0.05:
       
        base_thresh = 0.15
        
    elif rmse < 0.08:
       
        base_thresh = 0.18
        
    elif rmse < 0.12:
        
        base_thresh = 0.22
        
    elif rmse < 0.18:
        
        base_thresh = 0.28
        
    else:
        
        base_thresh = 0.35
    
    # SECONDARY FACTOR: PATHOLOGY ADJUSTMENT
    
    if f_res > 0.35:
        # Severe pathology (acute MS lesion, tumor)
        # Lower threshold to capture residual structure
        pathology_adjustment = -0.03
        
    elif f_res > 0.25:
        # Moderate pathology
        pathology_adjustment = -0.02
        
    elif f_res > 0.15:
        # Mild pathology
        pathology_adjustment = -0.01
        
    else:
        # Normal tissue
        pathology_adjustment = 0.0
    
    # TERTIARY FACTOR: ANISOTROPY RATIO
    
    anisotropy_adjustment = 0.0
    
    if f_fib > 0.01 and f_res > 0.01:
        aniso_ratio = f_fib / (f_fib + f_res)
        
        if aniso_ratio < 0.3:
            
            anisotropy_adjustment = 0.05
            
        elif aniso_ratio < 0.4:
            
            anisotropy_adjustment = 0.02
    
    # COMBINE FACTORS
    threshold = base_thresh + pathology_adjustment + anisotropy_adjustment
    
    # bounds: never go below 0.10 or above 0.40
    threshold = max(0.10, min(0.40, threshold))
    
    return threshold

# ANALYTICAL AD/RD ESTIMATION 

@njit(cache=True, fastmath=True)
def estimate_AD_RD_pure(bvals, bvecs, sig_norm, fiber_dir,
                        f_fib, f_res, f_hin, f_wat,
                        D_res, D_hin, D_wat):
    """Analytical AD/RD estimation via weighted least squares."""
    
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    sum_AA = 0.0
    sum_AB = 0.0
    sum_BB = 0.0
    sum_Ay = 0.0
    sum_By = 0.0
    
    for i in range(len(bvals)):
        b = bvals[i]
        
        S_total = sig_norm[i]
        if S_total < 0.01:
            S_total = 0.01
        
        S_iso = (fr * np.exp(-b * D_res) +
                 fh * np.exp(-b * D_hin) +
                 fw * np.exp(-b * D_wat))
        
        S_fiber = (S_total - S_iso) / (ff + 1e-12)
        
        if S_fiber < 0.01:
            S_fiber = 0.01
        if S_fiber > 1.0:
            S_fiber = 1.0
        
        log_S = np.log(S_fiber)
        
        g = bvecs[i]
        cos_t = g[0]*fiber_dir[0] + g[1]*fiber_dir[1] + g[2]*fiber_dir[2]
        cos2 = cos_t * cos_t
        
        w = S_total * b
        
        sum_AA += w * b * b
        sum_AB += w * b * b * cos2
        sum_BB += w * b * b * cos2 * cos2
        sum_Ay += w * b * log_S
        sum_By += w * b * cos2 * log_S
    
    det = sum_AA * sum_BB - sum_AB * sum_AB
    
    if abs(det) < 1e-20:
        return np.nan, np.nan
    
    x = (sum_BB * sum_Ay - sum_AB * sum_By) / det
    y = (sum_AA * sum_By - sum_AB * sum_Ay) / det
    
    RD_est = -x
    AD_est = -x - y
    
    RD_est = max(0.05e-3, min(3.0e-3, RD_est))
    AD_est = max(0.05e-3, min(3.5e-3, AD_est))
    
    if AD_est < RD_est:
        mean_val = (AD_est + RD_est) / 2
        AD_est = mean_val
        RD_est = mean_val
    
    return AD_est, RD_est

# GRID SEARCH REFINEMENT 

@njit(cache=True, fastmath=True)
def refine_AD_RD_pure(bvals, bvecs, sig_norm, fiber_dir,
                      f_fib, f_res, f_hin, f_wat,
                      D_res, D_hin, D_wat,
                      AD_init, RD_init):
    """Grid search refinement with adaptive range."""
    
    if np.isnan(AD_init) or np.isnan(RD_init):
        return np.nan, np.nan
    
    ftot = f_fib + f_res + f_hin + f_wat + 1e-12
    ff = f_fib / ftot
    fr = f_res / ftot
    fh = f_hin / ftot
    fw = f_wat / ftot
    
    best_sse = 1e20
    best_AD = AD_init
    best_RD = RD_init
    
    anisotropy = abs(AD_init - RD_init) / ((AD_init + RD_init) / 2 + 1e-12)
    
    if anisotropy > 0.5:
        range_factor = 0.25
    elif anisotropy > 0.2:
        range_factor = 0.35
    else:
        range_factor = 0.50
    
    AD_min = max(0.05e-3, AD_init * (1 - range_factor))
    AD_max = min(3.5e-3, AD_init * (1 + range_factor))
    RD_min = max(0.05e-3, RD_init * (1 - range_factor))
    RD_max = min(3.0e-3, RD_init * (1 + range_factor))
    
    n_AD = 8
    n_RD = 6
    
    dAD = (AD_max - AD_min) / (n_AD - 1) if n_AD > 1 else 0
    dRD = (RD_max - RD_min) / (n_RD - 1) if n_RD > 1 else 0
    
    # Coarse grid
    for i in range(n_AD):
        AD = AD_min + i * dAD
        
        for j in range(n_RD):
            RD = RD_min + j * dRD
            
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
    
    # Fine grid
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


# PARALLEL FITTING KERNEL WITH RMSE-BASED THRESHOLD

@njit(parallel=True, cache=True, fastmath=True)
def fit_voxels_rmse_threshold(data, coords, A, AtA, At, bvals, bvecs,
                               fiber_dirs, iso_grid, reg, enable_step2, out):
    """
    Parallel fitting with RMSE-based adaptive threshold.
    
    Output channels (11 total):
    0: Fiber fraction
    1: Restricted fraction
    2: Hindered fraction
    3: Water fraction
    4: AD (final, after Step 2 if enabled)
    5: RD (final, after Step 2 if enabled)
    6: Fiber FA
    7: Mean isotropic ADC
    8: AD_linear (before Step 2)
    9: RD_linear (before Step 2)
    10: Adaptive threshold used (RMSE-based)
    """
    
    n_voxels = coords.shape[0]
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    n_meas = len(bvals)
    
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
        
        # STEP 1: NNLS FIT

        Aty = np.zeros(AtA.shape[0])
        for r in range(AtA.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val
        
        w, _ = nnls_coordinate_descent(AtA, Aty, reg)
        
        # COMPUTE RMSE OF STEP 1 FIT
        # Predict signal from Step 1 weights

        sig_pred = np.zeros(n_meas)
        for i in range(n_meas):
            for j in range(len(w)):
                sig_pred[i] += A[i, j] * w[j]
        
        # Compute residual sum of squares
        sse = 0.0
        for i in range(n_meas):
            diff = sig_norm[i] - sig_pred[i]
            sse += diff * diff
        
        # RMSE (normalized by number of measurements)
        rmse = np.sqrt(sse / n_meas)
        
        # PARSE FRACTIONS FROM STEP 1

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
        
        # COMPUTE ADAPTIVE THRESHOLD (RMSE-BASED)

        adaptive_thresh = compute_adaptive_threshold_rmse(f_fib, f_res, rmse)
        
        # STEP 2: AD/RD ESTIMATION 
        AD = np.nan
        RD = np.nan
        FA = np.nan
        AD_linear = np.nan
        RD_linear = np.nan
        
        # Only estimate AD/RD if f_fib exceeds RMSE-based threshold
        if f_fib > adaptive_thresh:
            
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
            
            # Analytical estimation
            AD_linear, RD_linear = estimate_AD_RD_pure(
                bvals, bvecs, sig_norm, fiber_dir,
                f_fib, f_res, f_hin, f_wat,
                D_res, D_hin, D_wat
            )
            
            AD = AD_linear
            RD = RD_linear
            
            # Optional refinement via grid search
            if enable_step2:
                AD, RD = refine_AD_RD_pure(
                    bvals, bvecs, sig_norm, fiber_dir,
                    f_fib, f_res, f_hin, f_wat,
                    D_res, D_hin, D_wat,
                    AD_linear, RD_linear
                )
            
            # Compute FA if AD/RD valid
            if not np.isnan(AD) and not np.isnan(RD):
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
        out[x, y, z, 8] = AD_linear
        out[x, y, z, 9] = RD_linear
        out[x, y, z, 10] = adaptive_thresh


# MAIN MODEL CLASS

class DBSI_Fused:
    """
    DBSI Model with RMSE-Based Adaptive Threshold.
    
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
        Fit DBSI with RMSE-based adaptive threshold.
        
        Returns
        -------
        results : ndarray (X, Y, Z, 11)
            0: Fiber fraction
            1: Restricted fraction (inflammation/cells)
            2: Hindered fraction (edema)
            3: Water fraction (CSF)
            4: Axial diffusivity (AD) - final
            5: Radial diffusivity (RD) - final
            6: Fiber FA
            7: Mean isotropic ADC
            8: AD_linear (analytical estimate)
            9: RD_linear (analytical estimate)
            10: Adaptive threshold (RMSE-based)
            
        AD/RD/FA will be NaN if f_fib < adaptive_threshold
        """
        print("\n" + "="*70)
        print("  DBSI PIPELINE - RMSE-BASED ADAPTIVE THRESHOLD")
        print("="*70)
        
        # Normalize gradients
        bvecs = np.asarray(bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms
        
        # SNR estimation (for calibration and Rician correction)
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
        
        
        results = np.zeros(data.shape[:3] + (11,), dtype=np.float32)
        
        # Initialize with NaN
        results[..., 4] = np.nan  # AD
        results[..., 5] = np.nan  # RD
        results[..., 6] = np.nan  # FA
        results[..., 8] = np.nan  # AD_linear
        results[..., 9] = np.nan  # RD_linear
        results[..., 10] = np.nan  # Threshold
        
        batch_size = 10000
        n_batches = int(np.ceil(n_voxels / batch_size))
        
        t0 = time.time()
        
        with tqdm(total=n_voxels, desc="   Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_voxels)
                batch_coords = coords[start:end]
                
                fit_voxels_rmse_threshold(
                    data_corr, batch_coords, A, AtA_reg, At,
                    bvals, bvecs, fiber_dirs, iso_grid,
                    self.reg_lambda, self.enable_step2, results
                )
                
                pbar.update(len(batch_coords))
        
        elapsed = time.time() - t0
        
        # Report threshold statistics
        print(f"\n   Completed in {elapsed:.1f}s ({n_voxels/elapsed:.0f} vox/s)")
        
        thresh_map = results[..., 10]
        valid_thresh = thresh_map[mask]
        valid_thresh = valid_thresh[~np.isnan(valid_thresh)]
        
        if len(valid_thresh) > 0:
            print("\n   THRESHOLD STATISTICS:")
            print(f"     Mean:   {np.mean(valid_thresh):.3f}")
            print(f"     Median: {np.median(valid_thresh):.3f}")
            print(f"     Min:    {np.min(valid_thresh):.3f}")
            print(f"     Max:    {np.max(valid_thresh):.3f}")
            print(f"     25th %: {np.percentile(valid_thresh, 25):.3f}")
            print(f"     75th %: {np.percentile(valid_thresh, 75):.3f}")
            
            # Report RMSE distribution (diagnostic)
            # Note: RMSE not stored, but threshold gives indirect info
            rmse_bins = [(0.15, "Excellent"), (0.18, "Good"), 
                         (0.22, "Fair"), (0.28, "Poor"), (0.40, "Very Poor")]
            
            print("\n   ESTIMATED FIT QUALITY DISTRIBUTION:")
            for thresh_val, label in rmse_bins:
                pct = np.sum(valid_thresh <= thresh_val) / len(valid_thresh) * 100
                print(f"     {label:12s} (<{thresh_val:.2f}): {pct:5.1f}%")
        
        print("\n" + "="*70 + "\n")
        
        return results
