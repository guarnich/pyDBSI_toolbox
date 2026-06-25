"""
DBSI Utilities Module - Data Loading, Preprocessing, and Autoconfiguration (v3)

Contains utilities for:
- NIfTI data loading
- Gradient file parsing
- SNR estimation (temporal and spatial methods)
- Rician bias correction (currently inline in DBSI_Adaptive.fit(); see
  `correct_rician_bias` note in package __init__.py)
- Protocol-driven Stage A direction-count sizing (`autoconfig.py`)
"""

from .tools import (
    load_data,
    print_protocol_summary,
    estimate_snr_robust,
    # correct_rician_bias,
)
from .autoconfig import autoconfigure_dictionary

__all__ = [
    "load_data",
    "print_protocol_summary",
    "estimate_snr_robust",
    # "correct_rician_bias",
    "autoconfigure_dictionary",
]
