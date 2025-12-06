import numpy as np
from numba import njit, prange
import time

from .core.basis import build_design_matrix, generate_fibonacci_sphere
from .core.solvers import nnls_coordinate_descent, step2_refine_diffusivities
from .calibration.optimizer import optimize_hyperparameters
from .utils.tools import estimate_snr_robust, correct_rician_bias

class DBSI_Fused:
    """
    Main DBSI Model orchestrating Preprocessing, Basis Construction and Fitting.
    """
    def __init__(self, n_iso=None, reg_lambda=None, enable_step2=True):
        """
        Args:
            n_iso (int): Number of isotropic bases.
            reg_lambda (float): Regularization parameter.
            enable_step2 (bool): Enable non-linear refinement of diffusivities.
        """
        self.n_iso = n_iso
        self.reg_lambda = reg_lambda
        self.enable_step2 = enable_step2
        self.n_dirs = 150
        self.iso_range = (0.0, 3.0e-3)
        
    def fit(self, data, bvals, bvecs, mask, run_calibration=True):
        """
        Fits the model to the provided data.
        
        Args:
            data (4D array): DWI Volume
            bvals (1D array): B-values
            bvecs (Nx3 array): Gradient directions
            mask (3D array): Brain mask
            run_calibration (bool): If True and parameters are None, runs MC calibration.
            
        Returns:
            np.ndarray: 4D array containing parameter maps.
                        Channels: [f_fiber, f_rest, f_hind, f_water, AD, RD]
        """
        print(">>> Starting DBSI Pipeline")
        
        # 1. SNR Estimation
        snr, sigma = estimate_snr_robust(data, bvals, mask)
        print(f"1. SNR Estimated: {snr:.2f}")
        
        # 2. Calibration (if needed)
        if run_calibration and (self.n_iso is None or self.reg_lambda is None):
            self.n_iso, self.reg_lambda = optimize_hyperparameters(bvals, bvecs, snr)
        
        # Default fallbacks if calibration skipped/failed and no inputs
        if self.n_iso is None: self.n_iso = 50
        if self.reg_lambda is None: self.reg_lambda = 0.1
            
        # 3. Data Preprocessing (Rician Correction)
        print("2. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data)
        data_corr[mask] = correct_rician_bias(data[mask], sigma)
        
        # 4. Basis Construction
        print("3. Building Design Matrix...")
        fiber_dirs = generate_fibonacci_sphere(self.n_dirs)
        iso_grid = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso)
        
        A = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid)
        AtA = A.T @ A
        At = A.T
        
        # 5. Parallel Fitting
        print(f"4. Fitting {np.sum(mask)} voxels (Two-Step)...")
        t0 = time.time()
        
        coords = np.argwhere(mask)
        # Result layout: 6 channels
        results = np.zeros(data.shape[:3] + (6,), dtype=np.float32)
        
        self._fit_batch(
            data_corr, coords, A, AtA, At, bvals, bvecs,
            fiber_dirs, iso_grid, self.reg_lambda, self.enable_step2, results
        )
        
        print(f"   Done in {time.time()-t0:.1f}s")
        return results

    @staticmethod
    @njit(parallel=True)
    def _fit_batch(data, coords, A, AtA, At, bvals, bvecs, 
                   fiber_dirs, iso_grid, reg, do_step2, out):
        """Numba-accelerated batch fitting kernel."""
        n_voxels = coords.shape[0]
        n_dirs = len(fiber_dirs)
        
        for i in prange(n_voxels):
            x, y, z = coords[i]
            sig = data[x, y, z]
            
            # S0 Normalization (mean of b < 50)
            s0 = 0.0
            cnt = 0
            for k in range(len(bvals)):
                if bvals[k] < 50:
                    s0 += sig[k]
                    cnt += 1
            
            if s0 < 1e-6: continue
            sig_norm = sig / s0
            
            # Step 1: NNLS
            Aty = np.zeros(AtA.shape[0])
            for r in range(AtA.shape[0]):
                val = 0.0
                for c in range(len(sig_norm)):
                    val += At[r, c] * sig_norm[c]
                Aty[r] = val
                
            w, _ = nnls_coordinate_descent(AtA, Aty, reg)
            
            # Parse weights
            w_fib = w[:n_dirs]
            w_iso = w[n_dirs:]
            
            f_fib = np.sum(w_fib)
            f_res = 0.0
            f_hin = 0.0
            f_wat = 0.0
            
            for k in range(len(iso_grid)):
                adc = iso_grid[k]
                if adc <= 0.3e-3: f_res += w_iso[k]
                elif adc <= 2.0e-3: f_hin += w_iso[k]
                else: f_wat += w_iso[k]
            
            # Normalize fractions
            ftot = f_fib + f_res + f_hin + f_wat
            if ftot > 0:
                f_fib/=ftot; f_res/=ftot; f_hin/=ftot; f_wat/=ftot
            
            # Step 2: Refine AD/RD if fiber present
            AD, RD = 1.7e-3, 0.3e-3 # Defaults
            if do_step2 and f_fib > 0.1:
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
                    f_fib, f_res, f_hin, f_wat
                )
            
            out[x,y,z,0] = f_fib
            out[x,y,z,1] = f_res
            out[x,y,z,2] = f_hin
            out[x,y,z,3] = f_wat
            out[x,y,z,4] = AD
            out[x,y,z,5] = RD