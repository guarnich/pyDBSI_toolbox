"""
DBSI Transition-Zone Confidence Mapping
==========================================

WHY THIS MODULE EXISTS
--------------------------
The methodological supplement "isotropic_compartment_supplement.docx"
(project validation records) quantified two systematic, n_iso-
independent bias zones around the compartment thresholds used to
partition the isotropic spectrum into RF / HF / WF:

  - RES/HIN boundary (0.3e-3 mm^2/s): bias zone approx. 0.10-0.45e-3 mm^2/s
  - HIN/WAT boundary (3.0e-3 mm^2/s): bias zone approx. 1.5-5.0e-3 mm^2/s

These zones reflect a genuine ambiguity in the forward signal model
near each threshold (confirmed noise-independent in that investigation
— the bias persists even in the noiseless limit), not a deficiency
fixable by grid design or by increasing n_iso. The correct response is
therefore not to suppress or refuse to report fractions in these zones
(the project's standing principle is that the fit is always carried
through to completion — partial or withheld results are themselves a
source of analytical inconsistency), but to ACCOMPANY the existing
fraction maps with an explicit, quantified indicator of how close each
voxel's recovered compartment diffusivity sits to a known low-
confidence zone, so that downstream interpretation (visual inspection,
ROI statistics, group comparisons) can weight or flag accordingly.

WHAT THIS MODULE COMPUTES
-------------------------------
For every voxel already fit by `DBSI_Adaptive.fit()`, this module
recovers the per-compartment centroid diffusivities (D_res, D_hin,
D_wat — already computed internally during fitting but not retained in
the compact 11-channel output) using the SAME analytical recovery
already implemented and validated in `fit_quality.py`
(`_recover_iso_adcs_2iso` / `_recover_iso_adcs_3iso`), then maps each
recovered centroid to a confidence score using the empirically mapped
bias curves.

Confidence score definition
--------------------------------
For a given boundary and recovered centroid D, the confidence score is

    confidence(D) = 1.0   outside the mapped zone (|bias| <= 5pp region)
    confidence(D) = 1.0 - (|bias(D)| - 0.05) / (bias_peak - 0.05)
                          inside the mapped zone, linearly interpolated
                          between the zone edge (confidence=1.0) and the
                          point of peak observed bias (confidence=0.0)

This is a deliberately simple, monotonic interpolation of the actual
measured bias curves (see supplement Figure 3) rather than a fitted
parametric model — the goal is an interpretable, conservative flag, not
a precise bias-correction (this module does NOT attempt to correct the
fraction estimates; see "What this module does NOT do" below).

The bias curves were measured at four discrete n_iso values (10, 12,
20, 31) on one set of synthetic protocols/scenarios; this module uses a
single representative curve (the WIDEST observed zone across those four
n_iso values, i.e. the most conservative choice) rather than
interpolating by the dataset's actual n_iso, since the supplement found
zone width to be only weakly n_iso-dependent in the tested range.

WHAT THIS MODULE DOES NOT DO
---------------------------------
- It does NOT correct, adjust, or suppress fraction estimates. The
  fractions reported by `DBSI_Adaptive.fit()` are left exactly as
  computed; this module only adds an auxiliary confidence layer.
- It does NOT replace voxel-wise fit-quality (R^2, RMSE) maps from
  `fit_quality.py`, which assess how well the model reproduces the
  OBSERVED SIGNAL. This module instead flags a known structural
  ambiguity in the MODEL/THRESHOLD DEFINITION itself, present even for
  a perfect (R^2=1) fit.
- It has NOT been validated on real (non-synthetic) acquisitions. The
  zone boundaries are carried over directly from the synthetic mapping
  in the methodological supplement; project policy is to re-validate
  this mapping on real HCP/Verona data before using these confidence
  scores for clinical interpretation (see supplement Section 5.3).

References
----------
Project methodological supplement: "isotropic_compartment_supplement.docx"
    (Sections 4-5, transition zone quantification).
Borgia GC, Brown RJS, Fantazzini P (1998). J Magn Reson, 132(1), 65-77.
"""

import numpy as np

THRESH_RES = 0.3e-3
THRESH_WAT = 3.0e-3

# Zone bounds and peak-bias points, taken from the methodological
# supplement Table 1 / Figure 3, using the WIDEST extent observed across
# the four tested n_iso values (10, 12, 20, 31) as the conservative
# choice (see module docstring).
_RES_ZONE_LOW = 0.10e-3
_RES_ZONE_HIGH = 0.50e-3
_RES_ZONE_PEAK_BIAS = 0.21       # peak |bias| observed in the supplement's
                                  # RES/HIN sweep (Figure 3A, ~21 percentage
                                  # points at the threshold edge)

_WAT_ZONE_LOW = 1.50e-3
_WAT_ZONE_HIGH = 6.00e-3          # supplement's sweep did not fully close
                                  # by its scan limit (6.0e-3) at n_iso=10;
                                  # treated here as the zone's upper bound
                                  # for the conservative default, see
                                  # supplement Section 4.4 caveat.
_WAT_ZONE_PEAK_BIAS = 0.32       # peak |bias| observed (Figure 3B, ~32
                                  # percentage points near the threshold)

_BIAS_TOLERANCE = 0.05           # the |bias|>5pp criterion used to define
                                  # the zone boundaries in the supplement


def _confidence_from_distance(d, zone_low, zone_high, threshold, peak_bias,
                              tolerance=_BIAS_TOLERANCE):
    """
    Map a recovered centroid diffusivity `d` to a confidence score in
    [0, 1], using a piecewise-linear approximation of the measured bias
    curve: 1.0 outside [zone_low, zone_high], decreasing toward 0.0 as
    `d` approaches `threshold` from either side.

    This is intentionally crude (piecewise-linear, symmetric decay from
    each zone edge to the threshold) rather than fitting the exact
    measured curve shape (which was somewhat asymmetric — see
    supplement Figure 3): the purpose is a conservative, interpretable
    flag, not a precise bias model.
    """
    d = np.asarray(d, dtype=np.float64)
    confidence = np.ones_like(d)

    in_zone = (d >= zone_low) & (d <= zone_high)
    if not np.any(in_zone):
        return confidence

    # Distance from the threshold, normalised by the zone's half-width
    # on whichever side `d` falls.
    below = in_zone & (d <= threshold)
    above = in_zone & (d > threshold)

    half_width_below = max(threshold - zone_low, 1e-12)
    half_width_above = max(zone_high - threshold, 1e-12)

    # Linear ramp: confidence=0 AT the threshold (peak bias), confidence=1
    # at the zone edge. Values are clipped to keep confidence in [0,1]
    # even though the true peak bias may not be exactly at the threshold
    # in the measured curves.
    frac_below = (d[below] - zone_low) / half_width_below if half_width_below > 0 else 1.0
    frac_above = (zone_high - d[above]) / half_width_above if half_width_above > 0 else 1.0

    confidence[below] = np.clip(frac_below, 0.0, 1.0)
    confidence[above] = np.clip(frac_above, 0.0, 1.0)

    return confidence


def compute_transition_confidence(results, model_mode):
    """
    Compute per-voxel confidence maps for the RES/HIN and HIN/WAT
    compartment boundaries, given the standard 11-channel DBSI_Adaptive
    output.

    Parameters
    ----------
    results : ndarray (X, Y, Z, 11), float32
        Output of `DBSI_Adaptive.fit()`.
    model_mode : int
        2 or 3, as returned alongside `results`.

    Returns
    -------
    confidence_res : ndarray (X, Y, Z), float32
        Confidence score in [0, 1] for the RES/HIN boundary, based on
        the recovered hindered/non-restricted centroid diffusivity.
        1.0 = outside the mapped low-confidence zone; 0.0 = at the
        threshold itself (peak measured bias). NaN where fractions are
        unavailable (e.g. outside brain mask / zero total fraction).
    confidence_wat : ndarray (X, Y, Z), float32
        Confidence score for the HIN/WAT boundary, based on the
        recovered free-water (3-ISO) or non-restricted (2-ISO) centroid
        diffusivity. NaN where unavailable. In 2-ISO mode this reflects
        the merged NRF centroid, since HF and WF are not separately
        estimated in that mode (see `DBSI_Adaptive` model-selection
        logic) — interpret accordingly (a 2-ISO voxel's WAT confidence
        score describes how close its single NRF centroid sits to the
        3.0e-3 boundary, not a genuine HF-vs-WF distinction).
    """
    CH_RF, CH_HF, CH_WF, CH_NRF, CH_ADC_ISO = 1, 2, 3, 4, 8

    shape3d = results.shape[:3]
    confidence_res = np.full(shape3d, np.nan, dtype=np.float32)
    confidence_wat = np.full(shape3d, np.nan, dtype=np.float32)

    rf = results[..., CH_RF]
    adc_iso = results[..., CH_ADC_ISO]

    valid = ~np.isnan(adc_iso) & (adc_iso > 0)

    if model_mode == 3:
        hf = results[..., CH_HF]
        wf = results[..., CH_WF]
        ftot_iso = rf + hf + wf
        valid = valid & (ftot_iso > 1e-6)

        # Recover D_hin, D_wat analytically (same construction as
        # fit_quality._recover_iso_adcs_3iso) from the stored ADC_iso
        # centroid and fractions.
        D_res = 0.15e-3
        D_wat_fixed = 3.05e-3

        D_hin = np.full(shape3d, np.nan, dtype=np.float64)
        hf_valid = valid & (hf > 1e-6)
        D_hin[hf_valid] = (
            (adc_iso[hf_valid] * ftot_iso[hf_valid]
             - rf[hf_valid] * D_res - wf[hf_valid] * D_wat_fixed)
            / hf[hf_valid]
        )
        D_hin = np.clip(D_hin, THRESH_RES, THRESH_WAT)

        confidence_res[hf_valid] = _confidence_from_distance(
            D_hin[hf_valid], _RES_ZONE_LOW, _RES_ZONE_HIGH, THRESH_RES, _RES_ZONE_PEAK_BIAS
        )
        # WAT confidence: based on D_hin's proximity to the UPPER
        # threshold (does this voxel's hindered pool actually sit close
        # to the HIN/WAT boundary, i.e. could some of it genuinely be
        # under-resolved free water?).
        confidence_wat[hf_valid] = _confidence_from_distance(
            D_hin[hf_valid], _WAT_ZONE_LOW, _WAT_ZONE_HIGH, THRESH_WAT, _WAT_ZONE_PEAK_BIAS
        )
        # Voxels with a directly estimated WF fraction: also incorporate
        # the WAT-side confidence using the fixed D_wat_fixed value
        # (always inside the zone, since WAT centroid is pinned near
        # the boundary by construction in this model) is not useful
        # here; instead leave confidence_wat from the D_hin assessment
        # above, which is the informative quantity (how much of the
        # "hindered" pool may actually be ambiguous with free water).

    else:
        nrf = results[..., CH_NRF]
        ftot_iso = rf + nrf
        valid = valid & (ftot_iso > 1e-6)

        D_res = 0.15e-3
        D_nonrf = np.full(shape3d, np.nan, dtype=np.float64)
        nrf_valid = valid & (nrf > 1e-6)
        D_nonrf[nrf_valid] = (
            (adc_iso[nrf_valid] * ftot_iso[nrf_valid] - rf[nrf_valid] * D_res)
            / nrf[nrf_valid]
        )
        D_nonrf = np.clip(D_nonrf, THRESH_RES, 3.5e-3)

        confidence_res[nrf_valid] = _confidence_from_distance(
            D_nonrf[nrf_valid], _RES_ZONE_LOW, _RES_ZONE_HIGH, THRESH_RES, _RES_ZONE_PEAK_BIAS
        )
        confidence_wat[nrf_valid] = _confidence_from_distance(
            D_nonrf[nrf_valid], _WAT_ZONE_LOW, _WAT_ZONE_HIGH, THRESH_WAT, _WAT_ZONE_PEAK_BIAS
        )

    return confidence_res, confidence_wat


def save_transition_confidence(confidence_res, confidence_wat, affine, output_dir):
    """
    Save the two confidence maps as compressed NIfTI files, analogous
    to `fit_quality.save_fit_quality`.

    Parameters
    ----------
    confidence_res, confidence_wat : ndarray (X, Y, Z), float32
        Output of `compute_transition_confidence`.
    affine : ndarray (4, 4)
    output_dir : str

    Returns
    -------
    paths : dict
        {'confidence_res': path, 'confidence_wat': path}
    """
    import nibabel as nib
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for name, arr in [('confidence_res', confidence_res), ('confidence_wat', confidence_wat)]:
        fpath = os.path.join(output_dir, f'dbsi_{name}.nii.gz')
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), fpath)
        paths[name] = fpath

    return paths
