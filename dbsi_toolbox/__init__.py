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
    Compute voxel-wise R² and RMSE goodness-of-fit maps. `DBSI_Adaptive.fit`
    already calls this and stores the result in the `fit_r2` / `fit_rmse`
    output channels; call it directly only to recompute against different
    data or a different fiber_threshold. The reconstruction models ALL
    detected fiber populations from their stored directions and tensors —
    see `fit_quality.py` module docstring.

compute_transition_confidence
    Compute voxel-wise confidence maps for the RES/HIN and HIN/WAT
    compartment boundaries, based on the systematic, n_iso-independent
    bias zones quantified in the project methodological supplement
    "isotropic_compartment_supplement.docx". Does NOT correct fractions
    — only flags proximity to a known low-confidence zone. NOT YET
    VALIDATED ON REAL DATA (see `transition_confidence.py` module
    docstring caveats).

save_transition_confidence
    Save the two transition-confidence maps as compressed NIfTI files.

Output Channels (27 — see DBSI_Adaptive.output_map_names for the full
contract, and the `_C_*` constants in model_Niso_adaptive_ff_thr.py for
the indices, which are the single source of truth)
------------------------------------------------------------------------
    0 : FF      - Fiber fraction, TOTAL over both populations
    1 : RF      - Restricted fraction / inflammation
    2 : HF      - Hindered fraction                       (NaN in 2-ISO mode)
    3 : WF      - Free-water fraction / CSF               (NaN in 2-ISO mode)
    4 : NRF     - Non-restricted fraction (= HF + WF)
    5 : ADC_iso - Mean isotropic ADC
    6 : N_POP   - Number of fiber populations resolved. THREE-STATE:
                  NaN = no fiber compartment attempted, 0 = fiber present
                  but no direction resolved, 1-2 = populations found.
    ---- population 1 (dominant), NaN if absent ----
    7 : FF_POP1, 8: AD_POP1, 9: RD_POP1, 10: FA_POP1, 11-13: DIR1_XYZ
    ---- population 2, NaN if absent ----
   14 : FF_POP2, 15: AD_POP2, 16: RD_POP2, 17: FA_POP2, 18-20: DIR2_XYZ
    ---- FF-weighted over the populations present, NaN if no fiber ----
   21 : AD_W, 22: RD_W, 23: FA_W   (FA_W is the FA OF the weighted tensor,
                  NOT the mean of FA_POP1 and FA_POP2)
    ---- diagnostics ----
   24 : CONC    - Dominant-basin angular concentration
   25 : R2      - Goodness of fit of the reconstructed signal
   26 : RMSE    - Residual RMSE, as a fraction of S0

There is no third population and no AD_lin/RD_lin: the toolbox resolves at
most TWO fiber populations per voxel, and the linear channels were
byte-identical copies of AD_POP1/RD_POP1.

Compartment fractions (0-4) use 0 for "absent"; the fiber block (7-23)
uses NaN. `save_output_maps` writes the matching `fiber_valid.nii.gz`,
which you need before resampling any of the NaN-convention maps.

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

__version__ = "1.1.0"
__author__ = "DBSI Toolbox Contributors"


from .model_Niso_adaptive_ff_thr import DBSI_Adaptive
from .utils.tools import load_data, estimate_snr_robust
from .utils.autoconfig import autoconfigure_dictionary
from .calibration.optimizer import optimize_hyperparameters
from .fit_quality import (compute_fit_quality, compute_fiber_validity_map,
                          format_run_report, save_output_maps)
from .transition_confidence import compute_transition_confidence, save_transition_confidence

__all__ = [
    "DBSI_Adaptive",
    "load_data",
    "estimate_snr_robust",
    "autoconfigure_dictionary",
    "optimize_hyperparameters",
    "compute_fit_quality",
    "compute_fiber_validity_map",
    "format_run_report",
    "save_output_maps",
    "compute_transition_confidence",
    "save_transition_confidence",
]
