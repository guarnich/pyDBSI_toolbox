import os
import numpy as np
import nibabel as nib
import warnings

def print_protocol_summary(bvals):
    """
    Analyzes and prints a summary of the acquisition protocol.
    """
    rounded_bvals = np.round(bvals, -2)
    unique_b, counts = np.unique(rounded_bvals, return_counts=True)
    
    print("\n" + "="*50)
    print("       ACQUISITION PROTOCOL SUMMARY")
    print("="*50)
    print(f" Total Volumes: {len(bvals)}")
    print(f" Max B-value:   {np.max(bvals):.0f} s/mm^2")
    print("-" * 50)
    print(f" {'Shell (b-val)':<15} | {'Directions':<10}")
    print("-" * 50)
    
    for b, count in zip(unique_b, counts):
        print(f" b = {int(b):<11} | {count:<10}")
    print("="*50 + "\n")

def load_data(dwi_path, bval_path, bvec_path, mask_path=None, verbose=True):
    """
    Unified function to load NIfTI data, gradients, and mask.
    """
    if not os.path.exists(dwi_path):
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")
    
    img = nib.load(dwi_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    try:
        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
    except Exception as e:
        raise ValueError(f"Error loading bvals/bvecs: {e}")

    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms

    if mask_path and os.path.exists(mask_path):
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)
    else:
        if verbose:
            print("Warning: No mask provided. Generating simple threshold mask.")
        mean_vol = np.mean(data, axis=-1)
        thresh = np.percentile(mean_vol[mean_vol > 0], 10)
        mask = mean_vol > thresh

    if verbose:
        print_protocol_summary(bvals)

    return data, affine, bvals, bvecs, mask

def estimate_snr_robust(data, bvals, mask, verbose=True):
    """
    Robust SNR estimation with detailed reporting.
    """
    if verbose:
        print("\n[SNR ESTIMATION REPORT]")
        print("-" * 30)

    bvals = np.array(bvals).flatten()
    b0_idx = np.where(bvals < 50)[0]
    n_b0 = len(b0_idx)
    
    if verbose:
        print(f"  Number of b0 volumes found: {n_b0}")

    if n_b0 >= 2:
        # Temporal method
        if verbose:
            print("  Method: TEMPORAL (Voxel-wise variance over time)")
            
        b0_data = data[..., b0_idx]
        mean_b0 = np.mean(b0_data, axis=-1)
        # MAD (Median Absolute Deviation) for robust noise estimation
        mad = np.median(np.abs(b0_data - mean_b0[..., None]), axis=-1)
        sigma_map = mad * 1.4826
        
        valid = mask & (sigma_map > 0)
        if np.sum(valid) == 0:
            if verbose: print("  ! No valid voxels for calculation. Returning default.")
            return 20.0, 1.0 
            
        snr = np.nanmedian(mean_b0[valid] / sigma_map[valid])
        sigma = np.nanmedian(sigma_map[valid])
        
        if verbose:
            print(f"  Estimated SNR: {snr:.2f}")
            print(f"  Estimated Noise Sigma: {sigma:.4f}")
            
        return snr, sigma
    else:
        # Background fallback
        if verbose:
            print("  Method: SPATIAL (Signal vs Background Air)")
            print("  ! Warning: Less accurate than temporal method.")
            
        bg_mask = (~mask) & (data[..., 0] > 0)
        if np.sum(bg_mask) == 0:
             return 20.0, 1.0 
             
        bg_signal = data[..., 0][bg_mask]
        sigma = np.mean(bg_signal) / 1.253 
        mean_signal = np.median(data[..., 0][mask])
        
        snr = mean_signal / sigma
        
        if verbose:
            print(f"  Estimated SNR: {snr:.2f}")
            print(f"  Estimated Noise Sigma: {sigma:.4f}")
            
        return snr, sigma

def correct_rician_bias(signal, sigma):
    """
    Koay-Basser approximation for Rician bias correction.
    """
    if sigma <= 0: return signal
    
    signal_sq = signal**2
    noise_floor = 2 * sigma**2
    mask_valid = signal_sq > noise_floor
    
    corrected = np.zeros_like(signal)
    corrected[mask_valid] = np.sqrt(signal_sq[mask_valid] - noise_floor)
    corrected[~mask_valid] = signal[~mask_valid]
    
    return corrected