
"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging

Main Components:
- DBSI_Fused: Main model class for fitting DBSI to diffusion MRI data
- load_data: Utility for loading NIfTI data and gradients
- optimize_hyperparameters: Monte Carlo calibration

Outputs (10 channels):
            0: Fiber fraction
            1: Restricted fraction (inflammation/cells)
            2: Non-restricted fraction (edema/CSF)
            3: Axial diffusivity (AD) - final
            4: Radial diffusivity (RD) - final
            5: Fiber FA
            6: Mean isotropic ADC
            7: AD_linear (analytical estimate)
            8: RD_linear (analytical estimate)
            
        AD/RD/FA will be NaN if f_fib < f_fib threshold.

References:
    Wang Y, et al. (2011) Brain. Quantification of increased cellularity 
    during inflammatory demyelination.
    

"""

__version__ = "2.0.0"
__author__ = "DBSI Toolbox Contributors"

#from .model import DBSI_Fused
#from .model_adaptive_threshold import DBSI_Fused
from .model import DBSI_Fused
from .utils.tools import load_data, estimate_snr_robust, correct_rician_bias
from .calibration.optimizer import optimize_hyperparameters

__all__ = [
    "DBSI_Fused",
    "load_data",
    "estimate_snr_robust",
    "correct_rician_bias",
    "optimize_hyperparameters",
]