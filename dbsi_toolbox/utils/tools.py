import os
import numpy as np
import nibabel as nib

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
    # La maschera e' OBBLIGATORIA. Fino alla v1.2.0, se mancava, se ne generava
    # una per soglia sul volume medio: ma la maschera non decide solo quali voxel
    # fittare, decide anche da dove si campionano i voxel di calibrazione, quindi
    # una maschera improvvisata cambia lambda, n_iso e il gate. Meglio fermarsi.
    if not mask_path:
        raise ValueError(
            "Brain mask is required but was not provided (mask_path=None). "
            "The mask defines both the fitted voxels and the calibration sample, "
            "so a fallback mask would silently change the calibrated hyperparameters."
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
    b0_idx = np.where(bvals < 50)[0]
    n_b0 = len(b0_idx)
    if verbose:
        print(f"  Number of b0 volumes found: {n_b0}")
    # Con un solo b=0 restava la stima SPAZIALE (segnale vs aria di fondo,
    # sigma = bg/1.253). E' poco affidabile — dipende da come il costruttore
    # ha filtrato lo sfondo — e sigma alimenta la calibrazione del gate e la
    # correzione Rician: un sigma sbagliato destabilizza tutto il resto.
    # Meglio fermarsi che produrre mappe che sembrano buone.
    if n_b0 < 2:
        raise ValueError(
            f"At least 2 b=0 volumes are required to estimate the noise level "
            f"from the data; this acquisition has {n_b0}. The single-b0 spatial "
            f"fallback (signal vs background air) was removed in v1.2.0+: sigma "
            f"feeds the Rician correction and the concentration-gate calibration, "
            f"so an unreliable sigma propagates into every downstream stage."
        )
    if verbose:
        print("  Method: TEMPORAL (Voxel-wise STD + Iterative Correction)")
    b0_data = data[..., b0_idx]
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)
    std_b0[std_b0 == 0] = 1e-10
    valid_mask = mask
    if np.sum(valid_mask) == 0:
        # Fino alla v1.2.0 qui si ritornava (SNR=20, sigma=1) inventati.
        raise ValueError(
            "Cannot estimate SNR: the brain mask contains no voxels. "
            "Returning a default SNR would silently mis-calibrate the fit."
        )
    mean_masked = mean_b0[valid_mask]
    std_masked = std_b0[valid_mask]
    snr_apparent = mean_masked / std_masked
    snr_corrected = snr_apparent.copy()
    # Iteratively correct for Rician bias in the noise estimate, which is significant at low SNR.
    # 5 iterations is insufficient for convergence at typical in-vivo SNR
    # (e.g. SNR=30 requires ~14 iterations to converge to delta<0.01).
    # 20 iterations ensures convergence across the full physiological range
    # (SNR 10-100) with negligible additional cost.
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
