"""
DBSI Validation Module

Provides comprehensive quality metrics and validation tools for DBSI fitting:
- R² (coefficient of determination)
- RMSE/NRMSE (root mean square error)
- Residual analysis
- Physiological plausibility checks
- Tissue-specific metrics
"""

from .quality_metrics import (
    # Core metrics
    compute_r2_map,
    compute_rmse_map,
    compute_nrmse_map,
    
    # Signal prediction
    predict_signal_single_voxel,
    predict_signal_volume,
    
    # Validation
    check_physiological_plausibility,
    compute_residual_statistics,
    compute_tissue_specific_metrics,
    generate_validation_report,
    quick_r2_check,
    
    # Constants
    THRESH_RESTRICTED,
    THRESH_FREE,
    AD_PHYSIOLOGICAL_RANGE,
    RD_PHYSIOLOGICAL_RANGE,
    FA_PHYSIOLOGICAL_RANGE,
)

__all__ = [
    'compute_r2_map',
    'compute_rmse_map', 
    'compute_nrmse_map',
    'predict_signal_single_voxel',
    'predict_signal_volume',
    'check_physiological_plausibility',
    'compute_residual_statistics',
    'compute_tissue_specific_metrics',
    'generate_validation_report',
    'quick_r2_check',
    'THRESH_RESTRICTED',
    'THRESH_FREE',
    'AD_PHYSIOLOGICAL_RANGE',
    'RD_PHYSIOLOGICAL_RANGE',
    'FA_PHYSIOLOGICAL_RANGE',
]
