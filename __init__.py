
"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging

Corrected implementation of the two-step DBSI approach with Numba/JIT acceleration.

Main Components:
- DBSI_Fused: Main model class for fitting DBSI to diffusion MRI data
- load_data: Utility for loading NIfTI data and gradients
- optimize_hyperparameters: Monte Carlo calibration

Outputs (8 channels):
    0: Fiber Fraction - apparent axonal density
    1: Restricted Fraction - cellularity marker (ADC ≤ 0.3 µm²/ms)
    2: Hindered Fraction - edema/tissue loss (0.3 < ADC ≤ 3.0 µm²/ms)
    3: Water Fraction - CSF contamination (ADC > 3.0 µm²/ms)
    4: Axial Diffusivity (AD) - along fiber axis (mm²/s)
    5: Radial Diffusivity (RD) - perpendicular to fiber (mm²/s)
    6: Fiber FA - fractional anisotropy of fiber component
    7: Mean Isotropic ADC - weighted average of isotropic diffusivity

References:
    Wang Y, et al. (2011) Brain. Quantification of increased cellularity 
    during inflammatory demyelination.
    
    Ye Z, et al. (2020) Ann Clin Transl Neurol. Deep learning with DBSI 
    for classification of MS lesions.

Version: 2.1.1 (Corrected)
Date: December 2025
"""

__version__ = "2.1.1"
__author__ = "DBSI Toolbox Contributors"

from .model import DBSI_Fused
from .model_adaptive_threshold import DBSI_Fused
from .utils.tools import load_data, estimate_snr_robust, correct_rician_bias
from .calibration.optimizer import optimize_hyperparameters

__all__ = [
    "DBSI_Fused",
    "load_data",
    "estimate_snr_robust",
    "correct_rician_bias",
    "optimize_hyperparameters",
]