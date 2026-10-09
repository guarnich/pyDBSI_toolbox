"""
DBSI Utilities Module - Data Loading and Autoconfiguration

Contains utilities for:
- NIfTI data loading and gradient file parsing (`load_data`)
- `B0_THRESHOLD`, the single b=0 threshold (b < 100 s/mm^2)
- `estimate_snr_robust`, sigma from the b=0 volumes: NOT used in production,
  where sigma comes from the noise map passed to fit() (v1.7.1); kept for the
  synthetic tests
- Protocol-driven Stage A direction-count sizing (`autoconfig.py`)

The Rician correction is applied inline in DBSI_Adaptive.fit(), voxel by voxel
with the noise map.
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
