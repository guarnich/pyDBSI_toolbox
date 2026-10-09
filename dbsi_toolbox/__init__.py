"""
DBSI Toolbox - Diffusion Basis Spectrum Imaging (pyDBSI)
=========================================================

Input (fixed): a preprocessed DWI (topup, eddy, N4), bvals/bvecs, a brain mask
and the thermal noise map of dwidenoise divided by the N4 bias field. Output:
27 channels per voxel (compartment fractions, up to two fiber populations with
tensor and direction, FF-weighted tensor, diagnostics), per-voxel standard
errors and a run report.

Pipeline (see `model_Niso_adaptive_ff_thr` module docstring and the
"voxel journey" technical report for the mathematics):

  Stage A   regularised NNLS over [direction x (AD, RD) dictionary |
            anchored isotropic grid]: detects up to two fiber directions;
  Stage B/C single fiber: cone refinement of the direction, then a joint
            VARPRO re-solve of tensor and fractions on the raw signal;
            crossing: FF re-solved on the detected support, then a joint
            Levenberg-Marquardt fit of both tensors (MRDS);
  Stage D   final fractions by NNLS on [fitted fibers | three fixed
            isotropic centroids]; fiber_threshold applied to the final FF.

The two regularisation constants are calibrated from the data, per
acquisition (`DBSI_Adaptive.fit` / `calibrate`) or per protocol
(`calibrate_protocol` + `DBSI_Adaptive.from_calibration`).

Main Components
---------------
DBSI_Adaptive          the model; `fit(..., noise_map=...)`, `calibrate`,
                       `from_calibration`, `output_map_names`.
load_data              DWI + bvals/bvecs + brain mask (required).
save_output_maps       the single saving entry point: NN_<channel>.nii.gz,
                       validity masks, detection statistic, toolbox_report/.
calibrate_protocol, save_protocol_calibration, load_protocol_calibration
                       protocol-level calibration for cohort studies.
compute_fit_quality    R2 / RMSE of the reported model (already stored by
                       `fit` in channels 24-25).
estimate_snr_robust    b=0 temporal SNR/sigma; NOT used by the model since
                       1.7.1 (the noise map is), kept for comparisons and
                       synthetic tests.

Output Channels (27; the `_C_*` constants in model_Niso_adaptive_ff_thr.py
are the single source of truth -- read channels by name, via
`DBSI_Adaptive.output_map_names`)
-------------------------------------------------------------------------
    0 : fiber_fraction            TOTAL over both populations
    1 : restricted_fraction
    2 : hindered_fraction         (NaN in 2-ISO)
    3 : water_fraction            (NaN in 2-ISO)
    4 : nonrestricted_fraction    (= HF + WF in 3-ISO)
    5 : n_fiber_populations       THREE-STATE: NaN no fiber, 0 fiber mass
                                  without a resolved direction, 1-2
    6-12  population 1: fraction, AD, RD, FA, dir x/y/z   (NaN if absent)
    13-19 population 2: same                              (NaN if absent)
    20-22 FF-weighted AD, RD, and FA OF the weighted tensor
    23    dominant_basin_concentration
    24-25 fit_r2, fit_rmse
    26    nnls_iterations

Fractions (0-4) use 0 for "absent"; the fiber block (6-22) uses NaN.
`save_output_maps` writes `fiber_valid.nii.gz` / `fiber_valid_pop2.nii.gz`,
needed before resampling any NaN-convention map.

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
"""

__version__ = "1.8.1"
__author__ = "DBSI Toolbox Contributors"


from .model_Niso_adaptive_ff_thr import DBSI_Adaptive
from .utils.tools import load_data, estimate_snr_robust
from .utils.autoconfig import autoconfigure_dictionary
from .fit_quality import (compute_fit_quality, compute_fiber_validity_map,
                          format_run_report, save_output_maps, find_run_report)
from .toolbox_report import save_toolbox_report, plot_design_matrix
from .calibration.protocol import (protocol_fingerprint, fingerprint_mismatches,
                                   calibrate_protocol, save_protocol_calibration,
                                   load_protocol_calibration)

__all__ = [
    "DBSI_Adaptive",
    "load_data",
    "estimate_snr_robust",
    "autoconfigure_dictionary",
    "compute_fit_quality",
    "compute_fiber_validity_map",
    "format_run_report",
    "save_output_maps",
    "find_run_report",
    "save_toolbox_report",
    "plot_design_matrix",
    "protocol_fingerprint",
    "fingerprint_mismatches",
    "calibrate_protocol",
    "save_protocol_calibration",
    "load_protocol_calibration",
]
