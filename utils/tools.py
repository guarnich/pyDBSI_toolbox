import os
import numpy as np
import nibabel as nib
import warnings

def print_protocol_summary(bvals):
    """
    Analyzes and prints a summary of the acquisition protocol.
    
    Args:
        bvals (np.ndarray): Array of b-values.
    """
    # Round b-values to nearest 100 to group shells
    rounded_bvals = np.round(bvals, -2)
    unique_b, counts = np.unique(rounded_bvals, return_counts=True)
    
    print("\n" + "="*40)
    print("       ACQUISITION PROTOCOL SUMMARY")
    print("="*40)
    print(f" Total Volumes: {len(bvals)}")
    print(f" Max B-value:   {np.max(bvals):.0f} s/mm^2")
    print("-" * 40)
    print(f" {'Shell (b-val)':<15} | {'Directions':<10}")
    print("-" * 40)
    
    for b, count in zip(unique_b, counts):
        print(f" b = {int(b):<11} | {count:<10}")
    print("="*40 + "\n")

def load_data(dwi_path, bval_path, bvec_path, mask_path=None, verbose=True):
    """
    Unified function to load NIfTI data, gradients, and mask.
    
    Args:
        dwi_path (str): Path to 4D DWI NIfTI file.
        bval_path (str): Path to .bval file.
        bvec_path (str): Path to .bvec file.
        mask_path (str, optional): Path to binary brain mask.
        verbose (bool): If True, prints protocol summary.
        
    Returns:
        tuple: (data, affine, bvals, bvecs, mask)
    """
    # 1. Load NIfTI
    if not os.path.exists(dwi_path):
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")
    
    img = nib.load(dwi_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # 2. Load Gradients
    try:
        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
    except Exception as e:
        raise ValueError(f"Error loading bvals/bvecs: {e}")

    # Transpose bvecs if necessary (must be N x 3)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    
    # Normalize b-vectors
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms

    # 3. Handle Mask
    if mask_path and os.path.exists(mask_path):
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)
    else:
        if verbose:
            print("Warning: No mask provided. Generating simple threshold mask.")
        mean_vol = np.mean(data, axis=-1)
        thresh = np.percentile(mean_vol[mean_vol > 0], 10)
        mask = mean_vol > thresh

    # 4. Protocol Summary
    if verbose:
        print_protocol_summary(bvals)

    return data, affine, bvals, bvecs, mask

def estimate_snr_robust(data, bvals, mask):
    """
    Robust SNR estimation.
    Method 1 (Preferred): Temporal variance across b0 volumes (if >= 2).
    Method 2 (Fallback): Signal / Background Noise (Rayleigh distribution).
    """
    bvals = np.array(bvals).flatten()
    b0_idx = np.where(bvals < 50)[0]
    
    if len(b0_idx) >= 2:
        # Temporal method
        b0_data = data[..., b0_idx]
        mean_b0 = np.median(b0_data, axis=-1)
        # MAD (Median Absolute Deviation) for robust noise estimation
        mad = np.median(np.abs(b0_data - mean_b0[..., None]), axis=-1)
        sigma_map = mad * 1.4826
        
        valid = mask & (sigma_map > 0)
        if np.sum(valid) == 0:
            return 20.0, 1.0 # Safe fallback
            
        snr = np.nanmedian(mean_b0[valid] / sigma_map[valid])
        sigma = np.nanmedian(sigma_map[valid])
        return snr, sigma
    else:
        # Background fallback
        bg_mask = (~mask) & (data[..., 0] > 0)
        if np.sum(bg_mask) == 0:
             return 20.0, 1.0 # Safe fallback
             
        bg_signal = data[..., 0][bg_mask]
        # Sigma from Rayleigh background mean
        sigma = np.mean(bg_signal) / 1.253 
        mean_signal = np.median(data[..., 0][mask])
        return mean_signal / sigma, sigma

def correct_rician_bias(signal, sigma):
    """
    Koay-Basser approximation for Rician bias correction.
    Essential for accurate parameter estimation at high b-values.
    """
    if sigma <= 0: return signal
    
    # Simple correction: S_true = sqrt(S_measured^2 - 2*sigma^2)
    # Safe for SNR > 2
    signal_sq = signal**2
    noise_floor = 2 * sigma**2
    
    # Only correct where signal is sufficiently high to avoid complex numbers
    mask_valid = signal_sq > noise_floor
    
    corrected = np.zeros_like(signal)
    corrected[mask_valid] = np.sqrt(signal_sq[mask_valid] - noise_floor)
    corrected[~mask_valid] = signal[~mask_valid] # Leave low signal as is
    
    return corrected