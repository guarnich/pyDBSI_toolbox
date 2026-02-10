# DBSI Toolbox

**Diffusion Basis Spectrum Imaging (DBSI) - Two-Step Implementation with Numba Acceleration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DBSI is an advanced diffusion MRI technique that decomposes the diffusion-weighted signal into multiple components, distinguishing:

- **Anisotropic fiber diffusion** (axons)
- **Restricted isotropic diffusion** (cells/inflammation)
- **Hindered isotropic diffusion** (edema/tissue loss)
- **Free isotropic diffusion** (CSF)

This implementation follows the two-step approach described in Wang et al. (2011) Brain, with Numba/JIT acceleration for performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/guarnich/pyDBSI_toolbox
cd dbsi-toolbox

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

### Python API

```python
from dbsi_toolbox import DBSI_Fused, load_data

# Load data
data, affine, bvals, bvecs, mask = load_data(
    'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'mask.nii.gz'
)

# Initialize and fit model
model = DBSI_Fused(enable_step2=True)
results = model.fit(data, bvals, bvecs, mask)

```

### Command Line

```bash
python -m scripts.run_dbsi \
    --dwi data.nii.gz \
    --bval data.bval \
    --bvec data.bvec \
    --mask mask.nii.gz \
    --out results/
```

## Algorithm Details

### Two-Step Approach

1. **Step 1 (Linear)**: Decompose signal into fiber and isotropic components using NNLS with regularization
2. **Step 2 (Non-linear)**: Refine fiber diffusivities (AD, RD) using grid search optimization

### Key Parameters

- `n_iso`: Number of isotropic basis functions (default: auto-calibrated)
- `reg_lambda`: L2 regularization strength (default: auto-calibrated)
- `n_dirs`: Number of fiber directions on hemisphere (default: 100)
- `enable_step2`: Whether to refine diffusivities (default: True)

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Numba ≥ 0.55
- NiBabel ≥ 3.2
- SciPy ≥ 1.7
- tqdm ≥ 4.60

## References

1. Wang Y, et al. (2011). Quantification of increased cellularity during inflammatory demyelination. *Brain*, 134(12), 3590-3601.

2. Ye Z, et al. (2020). Deep learning with diffusion basis spectrum imaging for classification of multiple sclerosis lesions. *Ann Clin Transl Neurol*, 7(5), 695-706.

3. Cross AH, Song SK (2017). A new imaging modality to non-invasively assess multiple sclerosis pathology. *J Neuroimmunol*, 304, 81-85.

## License

MIT License - see LICENSE file for details.
