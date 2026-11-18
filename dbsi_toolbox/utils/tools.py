import os
import numpy as np
import nibabel as nib
from scipy.stats import chi2

# ─────────────────────────────────────────────────────────────────────────────
# NOISE ESTIMATION 
# ─────────────────────────────────────────────────────────────────────────────


B0_THRESHOLD = 100.0


def _chi_median_factor(n_samples):
    """E[median] scale of a k-dof sample standard deviation, k = n_samples - 1.

    std_hat = sigma * sqrt(chi2_k / k), so median(std_hat) = sigma * factor with
    factor = sqrt(median(chi2_k) / k). 0.6745 at n_samples=2, -> 1 as k -> inf.
    """
    k = int(n_samples) - 1
    if k < 1:
        raise ValueError("at least 2 samples are needed to estimate sigma")
    return float(np.sqrt(chi2.ppf(0.5, k) / k))

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
    
    if not mask_path:
        raise ValueError(
            "Brain mask is required but was not provided (mask_path=None). "
            "The mask defines both the fitted voxels and the calibration sample."
        )
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")
    mask = nib.load(mask_path).get_fdata().astype(bool)
    if not mask.any():
        raise ValueError(f"Brain mask is empty (no True voxels): {mask_path}")
    if mask.shape != data.shape[:3]:
        raise ValueError(
            f"Brain mask shape {mask.shape} does not match DWI volume shape "
            f"{data.shape[:3]}: {mask_path}"
        )
    if verbose:
        print_protocol_summary(bvals)
    return data, affine, bvals, bvecs, mask

def estimate_snr_robust(data, bvals, mask, verbose=True):
    if verbose:
        print("\n[SNR ESTIMATION REPORT]")
        print("-" * 30)
    bvals = np.array(bvals).flatten()
    b0_idx = np.where(bvals < B0_THRESHOLD)[0]
    n_b0 = len(b0_idx)
    if verbose:
        print(f"  Number of b0 volumes found: {n_b0}")
    if n_b0 < 2:
        raise ValueError(
            f"At least 2 b=0 volumes are required to estimate the noise level "
            f"from the data; this acquisition has {n_b0}. The single-b0 spatial "
            f"fallback (signal vs background air) was removed in v1.2.0+: sigma "
            f"feeds the Rician correction and the concentration-gate calibration, "
            f"so an unreliable sigma propagates into every downstream stage."
        )
    b0_data = data[..., b0_idx]
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)
    valid_mask = mask
    if np.sum(valid_mask) == 0:
        raise ValueError(
            "Cannot estimate SNR: the brain mask contains no voxels. "
        )
    mean_masked = mean_b0[valid_mask]
    std_masked = std_b0[valid_mask]

    if verbose:
        print(f"  Method: TEMPORAL (median per-voxel sigma, chi-debiased, "
              f"k={n_b0 - 1})")
    factor = _chi_median_factor(n_b0)
    sigma_biased = float(np.nanmedian(std_masked))
    final_sigma = sigma_biased / factor
    signal_level = float(np.nanmedian(mean_masked))
    if final_sigma <= 0 or not np.isfinite(final_sigma):
        raise ValueError(
            "Noise sigma estimated as zero or non-finite from the b=0 volumes. "
            "This usually means the b=0 volumes are duplicates of one another "
            "(identical voxel values), in which case they carry no information "
            "about the noise level and SNR cannot be estimated from the data."
        )
    final_snr = signal_level / final_sigma

    if verbose:
        print(f"  chi median de-bias factor: {factor:.4f} "
              f"(raw median sigma {sigma_biased:.4f})")
        print(f"  Estimated SNR: {final_snr:.2f}")
        print(f"  Estimated Noise Sigma: {final_sigma:.4f}")

    return float(final_snr), float(final_sigma)
