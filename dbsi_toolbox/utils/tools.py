import os
import numpy as np
import nibabel as nib
from scipy.stats import chi2

# ─────────────────────────────────────────────────────────────────────────────
# NOISE ESTIMATION — bias of the pre-v1.3.5 estimator
# ─────────────────────────────────────────────────────────────────────────────
# `estimate_snr_robust` returned an SNR biased HIGH by x1.82 and a sigma biased
# LOW by x0.55 on a 2-b0 acquisition (the minimum this toolbox accepts, and what
# the Verona P3 protocol provides). Two independent defects, each measured by
# Monte Carlo against a known sigma (tests/test_snr_sigma_bias.py):
#
#  (1) MEDIAN OF A RATIO. The old code took the median over voxels of the
#      per-voxel ratio mean/std. With nb0 samples the per-voxel std is
#      sigma*sqrt(chi2_k / k), k = nb0-1, so the median of the RATIO is not the
#      ratio of the medians. At nb0=2, std = sigma*|z| and median(1/|z|) =
#      1/0.6745, so SNR came out x1.483 too high. The bias shrinks with nb0
#      (x1.051 at 8, x1.012 at 32) but never vanishes at nb0=2.
#
#  (2) THE "ITERATIVE RICIAN CORRECTION" WAS A CONSTANT. The loop
#          snr <- m / sqrt(s^2 - m^2/(2 snr^2))
#      has a closed-form fixed point: u = m^2/(s^2 - m^2/(2u)) gives
#      u = (3/2) m^2/s^2, i.e. snr = sqrt(3/2) * (m/s) = 1.2247 * (m/s).
#      So the 20 iterations only ever multiplied the estimate by 1.2247 — at
#      EVERY SNR and EVERY nb0. It is not a Rician correction, and the comment
#      about "how many iterations to converge" described convergence to that
#      constant. Measured: x1.2247 at nb0 = 2, 4, 8 and 32 alike.
#
#      Combined at nb0=2: SNR x1.817, sigma x0.554.
#
#  HOW BIG THE BIAS IS DEPENDS ON nb0, AND 2 IS THE FLOOR, NOT THE TYPICAL CASE.
#  Do not quote x1.82 as "the" bias: the Verona P3 cohort acquires NINE b=0
#  volumes (measured, n_b0=9 on all 178 subjects), where defect (1) has almost
#  vanished and essentially only the sqrt(3/2) remains:
#
#      nb0     chi factor    sigma x
#        2       0.6745       1.816    <- the toolbox MINIMUM
#        4       0.8881       1.379
#        9       0.9581       1.278    <- the real acquisition
#       16       0.9777       1.253
#
#  Also worth knowing when correcting numbers already produced: sigma can be
#  corrected EXACTLY from an old report, because the legacy fixed point makes
#  m_v/snr_v = s_v/sqrt(1.5) identically, so sigma_legacy = median(s_v)/sqrt(1.5)
#  and sigma_true = sigma_legacy * sqrt(1.5)/c(nb0) -- verified to 5 decimals and
#  independent of how heterogeneous the signal is across the mask. The SNR cannot:
#  it is a median of a RATIO, which does not decompose, and the measured factor at
#  nb0=9 moves between 1.268 and 1.295 with tissue heterogeneity. An old SNR must
#  be RECOMPUTED, not rescaled.
#
# WHY IT MATTERS BEYOND THE REPORTED NUMBER. sigma is not cosmetic:
#   - the Rician correction subtracts a noise floor 2*sigma^2, so at x0.554 it
#     removed only ~31% of the floor it should have;
#   - `lambda_iso_discrepancy_cap` and the pass-1 lambda_aniso target N*sigma^2;
#   - `select_n_iso_svd` thresholds the singular values at 1/snr;
#   - `select_n_iso_bootstrap` injects noise at sigma.
#
# THE REPLACEMENT is robust AND unbiased: take the MEDIAN of the per-voxel
# standard deviations (robust to motion/outlier voxels, unlike pooling the
# variances) and divide by the known median of the chi distribution with
# k = nb0-1 degrees of freedom, sqrt(median(chi2_k)/k) — 0.6745 at nb0=2. SNR is
# then the ratio of two medians, not the median of a ratio. Verified x1.001
# against a known sigma at nb0 = 2, 4, 8 and 32.
#
# `_SNR_LEGACY_BIASED = True` restores the old estimator bit for bit, so a
# regression can be bisected against <= v1.3.4 by flipping one constant. It is
# NOT a supported configuration: it is provably biased.
_SNR_LEGACY_BIASED = False


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
    b0_data = data[..., b0_idx]
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)
    valid_mask = mask
    if np.sum(valid_mask) == 0:
        # Fino alla v1.2.0 qui si ritornava (SNR=20, sigma=1) inventati.
        raise ValueError(
            "Cannot estimate SNR: the brain mask contains no voxels. "
            "Returning a default SNR would silently mis-calibrate the fit."
        )
    mean_masked = mean_b0[valid_mask]
    std_masked = std_b0[valid_mask]

    # ── Legacy (biased) path, kept only to bisect against <= v1.3.4 ─────────
    def _legacy():
        s = std_masked.copy()
        s[s == 0] = 1e-10
        snr_c = (mean_masked / s).copy()
        for _ in range(20):
            snr_old = snr_c.copy()
            bias_term = mean_masked**2 / (2 * snr_c**2 + 1e-10)
            var_corrected = s**2 - bias_term
            var_corrected[var_corrected < 0] = 1e-10
            snr_c = mean_masked / np.sqrt(var_corrected)
            if np.mean(np.abs(snr_c - snr_old)) < 0.01:
                break
        return (float(np.nanmedian(snr_c)),
                float(np.nanmedian(mean_masked / snr_c)))

    if _SNR_LEGACY_BIASED:
        if verbose:
            print("  Method: TEMPORAL (LEGACY, BIASED -- bisection only)")
        final_snr, final_sigma = _legacy()
        if verbose:
            print(f"  Estimated SNR: {final_snr:.2f}")
            print(f"  Estimated Noise Sigma: {final_sigma:.4f}")
        return float(final_snr), float(final_sigma)

    # ── Unbiased path (default from v1.3.5) ────────────────────────────────
    # sigma from the MEDIAN of the per-voxel standard deviations, de-biased by
    # the known chi median factor for k = n_b0 - 1 degrees of freedom; SNR as a
    # ratio of two medians. See the module header for the two defects this
    # replaces and the Monte Carlo that measured them.
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
        _lsnr, _lsig = _legacy()
        print(f"  [<=v1.3.4 would have reported SNR {_lsnr:.2f} "
              f"(x{_lsnr / final_snr:.3f}) and sigma {_lsig:.4f} "
              f"(x{_lsig / final_sigma:.3f}) -- biased, see tools.py header]")

    return float(final_snr), float(final_sigma)
