# DBSI Toolbox Adaptive

**Diffusion Basis Spectrum Imaging (DBSI) - Adaptive Implementation with Numba Acceleration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DBSI is an advanced diffusion MRI technique that decomposes the diffusion-weighted signal into multiple components. This advanced toolbox includes an **Adaptive Model** that automatically selects the optimal isotropic compartmentalization based on the acquisition protocol (maximum b-value and number of shells):

- **3-ISO (Research Protocols)**: Separates Restricted (inflammation/cellularity), Hindered (vasogenic edema), and Free Water (CSF). Activated when $b_{max} \ge 3000 \text{ s/mm}^2$ and $\ge 3$ non-zero shells.
- **2-ISO (Clinical Protocols)**: Merges Hindered and Free Water into a single Non-Restricted fraction to guarantee numerical stability when the free water signal is near the noise floor ($b_{max} < 3000 \text{ s/mm}^2$).

## Outputs (11 Channels)

The toolbox dynamically generates the following parameter maps depending on the selected model mode:

1. **fiber_fraction (FF)**: Apparent axonal density.
2. **restricted_fraction (RF)**: Cellularity marker (inflammation).
3. **hindered_fraction (HF)**: Vasogenic edema *(NaN in 2-ISO mode)*.
4. **water_fraction (WF)**: CSF / Free water *(NaN in 2-ISO mode)*.
5. **nonrestricted_fraction (NRF)**: HF + WF combined.
6. **axial_diffusivity (AD)**: Axonal integrity *(NaN if FF < threshold)*.
7. **radial_diffusivity (RD)**: Demyelination marker *(NaN if FF < threshold)*.
8. **fiber_fa**: Intrinsic fiber fractional anisotropy *(NaN if FF < threshold)*.
9. **mean_iso_adc**: Mean isotropic ADC.
10. **ad_linear**: Initial analytical AD estimate.
11. **rd_linear**: Initial analytical RD estimate.

## Installation

```bash
# Clone the repository
git clone https://github.com/guarnich/pyDBSI_toolbox
cd pyDBSI_toolbox

# Install
pip install .
```

## Quick Start

### Python API

```python
import os
import nibabel as nib
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.utils.tools import load_data

# Load data
data, affine, bvals, bvecs, mask = load_data(
    'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'mask.nii.gz'
)

# Initialize adaptive model
model = DBSI_Adaptive(enable_step2=True)

# Fit data (returns both results array and the automatically chosen model_mode)
results, model_mode = model.fit(data, bvals, bvecs, mask, run_calibration=True)

# Save maps dynamically
map_names = DBSI_Adaptive.output_map_names(model_mode)
for i, name in enumerate(map_names):
    if not name.endswith('_NaN'):
        out_img = nib.Nifti1Image(results[..., i], affine)
        nib.save(out_img, f'dbsi_{name}.nii.gz')
```

### Command Line

```bash
python -m scripts.run_dbsi \
    --dwi data.nii.gz \
    --bval data.bval \
    --bvec data.bvec \
    --mask mask.nii.gz \
    --out results/ \
    --force-n-iso 3  # Optional: override automatic model selection (2 or 3)
```

## Algorithm Details

### Optimization Approach

* **Step 1 (Linear)**: Decomposes signal into fiber and isotropic components using NNLS with L2 regularization.
* **Step 2 (Non-linear)**: Refines fiber diffusivities (AD, RD) using grid search optimization. An integrity filter (`FIBER_THRESHOLD = 0.15`) is applied to prevent artifactual tensor estimation in pure fluid/inflammation voxels.

### Key Parameters

* `n_iso`: Number of isotropic basis functions (default: auto-calibrated).
* `reg_lambda`: L2 regularization strength (default: auto-calibrated).
* `n_dirs`: Number of fiber directions on the hemisphere (default: 100).
* `enable_step2`: Whether to refine diffusivities non-linearly (default: True).
* `force_n_iso`: Forces the algorithm into 2-ISO or 3-ISO mode, bypassing automatic protocol detection.

## Requirements

* Python >= 3.8
* NumPy >= 1.20
* Numba >= 0.55
* NiBabel >= 3.2
* SciPy >= 1.7
* tqdm >= 4.60

## References

1. Wang Y, et al. (2011). Quantification of increased cellularity during inflammatory demyelination. *Brain*, 134(12), 3590-3601.
2. Wang Y, et al. (2015). Differentiation and quantification of inflammation, demyelination and axon injury or loss in multiple sclerosis. *Brain*, 138(5), 1223-1238.
3. Vavasour IM, et al. (2022). Characterisation of multiple sclerosis neuroinflammation and neurodegeneration with relaxation and diffusion basis spectrum imaging. *Multiple Sclerosis Journal*, 28(3), 418-428.