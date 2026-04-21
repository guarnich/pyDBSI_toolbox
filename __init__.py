"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging

Main Components:
- DBSI_Adaptive: Main model class for fitting DBSI to diffusion MRI data. 
  It automatically adapts between 2-ISO (RF + NRF) and 3-ISO (RF + HF + WF) 
  compartmentalization based on the acquisition protocol.
- load_data: Utility for loading NIfTI data and gradients
- optimize_hyperparameters: Monte Carlo calibration

Outputs (11 channels):
            0: FF   - Fiber fraction (always valid)
            1: RF   - Restricted fraction / inflammation/cells (always valid)
            2: HF   - Hindered fraction (NaN in 2-ISO mode)
            3: WF   - Free-water fraction / CSF (NaN in 2-ISO mode)
            4: NRF  - Non-restricted fraction / edema/CSF (= HF + WF)
            5: AD   - Axial diffusivity (NaN if FF <= fiber_threshold)
            6: RD   - Radial diffusivity (NaN if FF <= fiber_threshold)
            7: FA   - Fiber FA (NaN if FF <= fiber_threshold)
            8: ADC_iso - Mean isotropic ADC (always valid)
            9: AD_lin  - Analytical AD estimate (NaN if FF <= fiber_threshold)
            10: RD_lin - Analytical RD estimate (NaN if FF <= fiber_threshold)
            
References:
    Wang Y, et al. (2011) Brain. Quantification of increased cellularity 
    during inflammatory demyelination.
"""

__version__ = "2.0.0"
__author__ = "DBSI Toolbox Contributors"

# Importa la classe dal file con il nome esatto che hai scelto
from .model_Niso_adaptive_ff_thr import DBSI_Adaptive
from .utils.tools import load_data, estimate_snr_robust, correct_rician_bias
from .calibration.optimizer import optimize_hyperparameters

__all__ = [
    "DBSI_Adaptive",
    "load_data",
    "estimate_snr_robust",
    "correct_rician_bias",
    "optimize_hyperparameters",
]