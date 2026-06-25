"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging (v3, Hybrid Two-Stage)
=========================================================================

v3 ARCHITECTURE SUMMARY
-------------------------
v2 attempted to estimate fiber orientation AND (AD, RD) simultaneously
from a single exhaustive (direction x AD/RD-pair) linear NNLS solve.
Synthetic recovery validation (55 swept configurations) showed this is
NOT numerically identifiable — median AD/RD relative errors ranged from
~20% to >150% across all tested dictionary densities, worsening with
finer grids.

v3 separates the two questions into two appropriately-sized stages:

  Stage A (detection): a coarse exhaustive (direction x AD/RD-pair)
    dictionary, fit via heavily-regularized NNLS, used ONLY to detect
    which hemisphere direction(s) carry fiber signal.
  Stage B (estimation): given Stage A's detected direction, a small
    closed-form weighted-least-squares regression (2 free parameters)
    estimates the final AD/RD.

This preserves the design intent that motivated v2 (the dictionary
should "know" pathology changes AD/RD, not just orientation — credited
to feedback from Alonso Ramirez-Manzanares) while resolving the
identifiability failure: synthetic validation of the v3 architecture
showed direction-recovery cosine similarity ~1.0 and substantially
reduced (though not yet eliminated) AD/RD relative error versus v2; see
project validation records for the full sweep and appropriate caution
around RD precision specifically before reporting it as a precise
quantitative biomarker.

There is no non-linear Step 2 refinement stage in v3 (as in v2): Stage
B's closed-form estimate is final.

Main Components
---------------
DBSI_Adaptive
    Main model class. Automatically selects between a two-compartment
    (2-ISO: RF + NRF) and a three-compartment (3-ISO: RF + HF + WF)
    isotropic decomposition based on the acquisition protocol, and
    autoconfigures Stage A's detection dictionary (direction count) from
    the same protocol.

load_data
    Load NIfTI DWI data together with b-values, b-vectors, and an
    optional brain mask.

estimate_snr_robust
    Estimate SNR from b=0 volumes (temporal method) or background air
    (spatial fallback).

autoconfigure_dictionary
    Derive Stage A's hemisphere-direction count (and other diagnostic
    values) from the acquisition protocol.

optimize_hyperparameters
    Monte Carlo calibration of (lambda_aniso, lambda_iso), evaluated
    end-to-end through Stage A + Stage B, across 14 physiologically
    grounded tissue scenarios.

compute_fit_quality
    Compute voxel-wise R² and RMSE goodness-of-fit maps. v3: this
    reconstruction is exact (not approximate, unlike v2) for
    single-fiber-population voxels — see `fit_quality.py` module
    docstring.

save_fit_quality
    Save R² and RMSE maps as compressed NIfTI files.

Output Channels (11, unified across both model modes — unchanged layout
from v1/v2; channel semantics for 5-7, 9-10 updated, see
model_Niso_adaptive_ff_thr.py)
------------------------------------------------------------------------
    0 : FF      - Fiber fraction                          (always valid)
    1 : RF      - Restricted fraction / inflammation      (always valid)
    2 : HF      - Hindered fraction                       (NaN in 2-ISO mode)
    3 : WF      - Free-water fraction / CSF               (NaN in 2-ISO mode)
    4 : NRF     - Non-restricted fraction (= HF + WF)     (always valid)
    5 : AD      - Axial diffusivity (v3: Stage B closed-form estimate
                  conditioned on Stage A's detected direction)
                                                            (NaN if FF <= fiber_threshold)
    6 : RD      - Radial diffusivity (v3: Stage B closed-form estimate)
                                                            (NaN if FF <= fiber_threshold)
    7 : FA      - Intrinsic fiber FA                      (NaN if FF <= fiber_threshold)
    8 : ADC_iso - Mean isotropic ADC                      (always valid)
    9 : AD_lin  - v3: identical to channel 5 (retained for output-shape
                  compatibility; Stage B's estimate is the only
                  diffusivity estimate produced)
   10 : RD_lin  - v3: identical to channel 6 (see note above)

References
----------
Wang Y, et al. (2011). Quantification of increased cellularity during
    inflammatory demyelination. Brain, 134(12), 3590-3601.
Wang Y, et al. (2015). Differentiation and quantification of inflammation,
    demyelination and axon injury or loss in multiple sclerosis. Brain,
    138(5), 1223-1238.
Vavasour IM, et al. (2022). Characterisation of multiple sclerosis
    neuroinflammation and neurodegeneration with relaxation and diffusion
    basis spectrum imaging. Multiple Sclerosis Journal, 28(3), 418-428.
Design document: toolbox_v2.md (orientation-space vs. parameter-space
    sampling discussion, credited to feedback from Alonso
    Ramirez-Manzanares); v3 hybrid redesign motivated by synthetic
    recovery validation of the v2 single-stage approach.
"""

__version__ = "3.0.0-hybrid"
__author__ = "DBSI Toolbox Contributors"


from .model_Niso_adaptive_ff_thr import DBSI_Adaptive
from .utils.tools import load_data, estimate_snr_robust
from .utils.autoconfig import autoconfigure_dictionary
from .calibration.optimizer import optimize_hyperparameters
from .fit_quality import compute_fit_quality, save_fit_quality

# NOTE: `correct_rician_bias` is NOT imported here. In the source
# `utils/tools.py` this function is entirely commented out (Koay-Basser
# correction is currently applied inline inside DBSI_Adaptive.fit() via
# the Gudbjartsson & Patz 1995 formula instead). Importing a
# commented-out symbol here would raise ImportError at package import
# time. Re-add this import once `correct_rician_bias` is actually
# implemented and uncommented in utils/tools.py.

__all__ = [
    "DBSI_Adaptive",
    "load_data",
    "estimate_snr_robust",
    "autoconfigure_dictionary",
    "optimize_hyperparameters",
    "compute_fit_quality",
    "save_fit_quality",
]
