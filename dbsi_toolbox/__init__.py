"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging
================================================

Main Components
---------------
DBSI_Adaptive
    Main model class. Automatically selects between a two-compartment
    (2-ISO: RF + NRF) and a three-compartment (3-ISO: RF + HF + WF)
    isotropic decomposition based on the acquisition protocol
    (b_max and shell diversity).

load_data
    Load NIfTI DWI data together with b-values, b-vectors, and an
    optional brain mask.

estimate_snr_robust
    Estimate SNR from b=0 volumes (temporal method) or background air
    (spatial fallback).

correct_rician_bias
    Koay-Basser Rician bias correction (vectorized).

optimize_hyperparameters
    Monte Carlo calibration of (n_iso, reg_lambda) across 14
    physiologically grounded tissue scenarios.

compute_fit_quality
    Compute voxel-wise R² and RMSE goodness-of-fit maps from the
    parameter maps returned by DBSI_Adaptive.fit().

save_fit_quality
    Save R² and RMSE maps as compressed NIfTI files.

Output Channels (11, unified across both model modes)
------------------------------------------------------
    0 : FF      - Fiber fraction                          (always valid)
    1 : RF      - Restricted fraction / inflammation      (always valid)
    2 : HF      - Hindered fraction                       (NaN in 2-ISO mode)
    3 : WF      - Free-water fraction / CSF               (NaN in 2-ISO mode)
    4 : NRF     - Non-restricted fraction (= HF + WF)     (always valid)
    5 : AD      - Axial diffusivity                       (NaN if FF ≤ fiber_threshold)
    6 : RD      - Radial diffusivity                      (NaN if FF ≤ fiber_threshold)
    7 : FA      - Intrinsic fiber FA                      (NaN if FF ≤ fiber_threshold)
    8 : ADC_iso - Mean isotropic ADC                      (always valid)
    9 : AD_lin  - Analytical AD estimate (Step 1 init)    (NaN if FF ≤ fiber_threshold)
   10 : RD_lin  - Analytical RD estimate (Step 1 init)    (NaN if FF ≤ fiber_threshold)

References
----------
Wang Y, et al. (2011). Quantification of increased cellularity during
    inflammatory demyelination. Brain, 134(12), 3590–3601.
Wang Y, et al. (2015). Differentiation and quantification of inflammation,
    demyelination and axon injury or loss in multiple sclerosis. Brain,
    138(5), 1223–1238.
Vavasour IM, et al. (2022). Characterisation of multiple sclerosis
    neuroinflammation and neurodegeneration with relaxation and diffusion
    basis spectrum imaging. Multiple Sclerosis Journal, 28(3), 418–428.
"""

__version__ = "2.0.0"
__author__  = "DBSI Toolbox Contributors"


from .model_Niso_adaptive_ff_thr import DBSI_Adaptive
from .utils.tools import load_data, estimate_snr_robust, correct_rician_bias
from .calibration.optimizer import optimize_hyperparameters
from .fit_quality import compute_fit_quality, save_fit_quality

__all__ = [
    "DBSI_Adaptive",
    "load_data",
    "estimate_snr_robust",
    "correct_rician_bias",
    "optimize_hyperparameters",
    "compute_fit_quality",
    "save_fit_quality",
]
