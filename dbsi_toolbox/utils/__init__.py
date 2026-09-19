"""
DBSI Utilities Module - Data Loading, Preprocessing, and Autoconfiguration (v3)

Contains utilities for:
- NIfTI data loading
- Gradient file parsing
- SNR estimation (temporal and spatial methods)
- Rician bias correction (applied inline in DBSI_Adaptive.fit())
- Protocol-driven Stage A direction-count sizing (`autoconfig.py`)
"""

from .tools import (
    load_data,
    print_protocol_summary,
    estimate_snr_robust,
)
from .autoconfig import autoconfigure_dictionary

__all__ = [
    "load_data",
    "print_protocol_summary",
    "estimate_snr_robust",
    "autoconfigure_dictionary",
]
