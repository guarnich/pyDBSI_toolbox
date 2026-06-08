import os
import numpy as np
import nibabel as nib
import warnings

def print_protocol_summary(bvals):
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
    if verbose:
        print("\n[SNR ESTIMATION REPORT]")
        print("-" * 30)
    bvals = np.array(bvals).flatten()
    b0_idx = np.where(bvals < 50)[0]
    n_b0 = len(b0_idx)
    if verbose:
        print(f"  Number of b0 volumes found: {n_b0}")
    if n_b0 >= 2:
        if verbose:
            print("  Method: TEMPORAL (Voxel-wise STD + Iterative Correction)")
        b0_data = data[..., b0_idx]
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1, ddof=1)
        std_b0[std_b0 == 0] = 1e-10
        valid_mask = mask
        if np.sum(valid_mask) == 0:
            if verbose: print(" No valid voxels! Defaulting to 20.0")
            return 20.0, 1.0
        mean_masked = mean_b0[valid_mask]
        std_masked = std_b0[valid_mask]
        snr_apparent = mean_masked / std_masked
        snr_corrected = snr_apparent.copy()
        # Iteratively correct for Rician bias in the noise estimate, which is significant at low SNR.
        # 5 iterations is insufficient for convergence at typical in-vivo SNR
        # (e.g. SNR=30 requires ~14 iterations to converge to delta<0.01).
        # 20 iterations ensures convergence across the full physiological range
        # (SNR 10–100) with negligible additional cost.
        for i in range(20):
            snr_old = snr_corrected.copy()
            bias_term = mean_masked**2 / (2 * snr_corrected**2 + 1e-10)
            var_corrected = std_masked**2 - bias_term
            var_corrected[var_corrected < 0] = 1e-10
            snr_corrected = mean_masked / np.sqrt(var_corrected)
            diff = np.mean(np.abs(snr_corrected - snr_old))
            if diff < 0.01:
                break
        final_snr = np.nanmedian(snr_corrected)
        final_sigma = np.nanmedian(mean_masked / snr_corrected)
        if verbose:
            print(f"  Estimated SNR: {final_snr:.2f}")
            print(f"  Estimated Noise Sigma: {final_sigma:.4f}")
        return float(final_snr), float(final_sigma)
    else:
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
        return float(snr), float(sigma)

# def correct_rician_bias(signal, sigma):
#     """
#     Koay-Basser approximation for Rician bias correction.

#     For voxels where signal^2 <= 2*sigma^2 (SNR < ~1.4), the correction
#     cannot be applied (negative under sqrt). These are set to 0 rather than
#     retaining the biased original signal, which would be the dominant noise
#     contribution and not a meaningful signal estimate.
#     """
#     if sigma <= 0:
#         return signal
#     signal_sq = signal**2
#     noise_floor = 2.0 * sigma**2
#     mask_valid = signal_sq > noise_floor
#     corrected = np.zeros_like(signal)
#     corrected[mask_valid] = np.sqrt(signal_sq[mask_valid] - noise_floor)
#     # ~mask_valid: SNR too low to correct → 0 (noise floor; not the biased signal)
#     corrected[~mask_valid] = 0.0
#     return corrected
