# DBSI Toolbox Adaptive (v3 — Hybrid Two-Stage Architecture)

**Diffusion Basis Spectrum Imaging (DBSI) - Adaptive Implementation with Numba Acceleration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What changed in v3

v2 attempted to estimate fiber orientation **and** (AD, RD) simultaneously
from a single linear NNLS solve over an exhaustive (direction × AD/RD-pair)
dictionary, taking AD/RD as a weighted centroid over the activated
columns. Systematic synthetic recovery validation (55 swept
configurations) showed this is **not numerically identifiable**: median
AD/RD relative errors ranged from ~20% to over 150% across every tested
dictionary density, getting *worse* with finer grids rather than better,
because the centroid increasingly averages over an uninformative span of
the grid as more columns become simultaneously active.

v3 separates the two questions into two appropriately-sized stages:

- **Stage A (detection)**: a coarse exhaustive (direction × AD/RD-pair)
  dictionary is fit via heavily-regularized NNLS — used **only** to
  detect which hemisphere direction(s) carry fiber signal. We trust
  *which direction* was selected, not the (AD, RD) breakdown that comes
  with it.
- **Stage B (estimation)**: given Stage A's detected direction, a small
  closed-form weighted-least-squares regression (2 free parameters: AD
  and RD) — the same analytical construction validated as the v1/v2
  linear AD/RD initialisation — produces the final diffusivities.

This preserves the design intent that motivated v2 (the dictionary must
"know" that pathology changes AD/RD, not just orientation — credited to
feedback from Alonso Ramirez-Manzanares) while resolving the
identifiability failure: Stage B's conditioning no longer depends on
Stage A's dictionary density, because direction is fixed before Stage B
runs.

**Synthetic validation summary** (coarse Stage A dictionary, ~30
directions × 3×3 AD/RD pairs, `lambda_base ≈ 0.005`): direction recovery
cosine similarity ≈ 1.0 across randomized ground truth; median AD
relative error ~10–20%, median RD relative error ~15–25% — a substantial
improvement over v2, though RD precision (the demyelination marker) still
warrants caution before being reported as a precise quantitative
biomarker without further protocol-specific validation.

There is still no non-linear Step 2 refinement stage (eliminated since
v2): Stage B's closed-form estimate is final.

## Overview

DBSI is an advanced diffusion MRI technique that decomposes the
diffusion-weighted signal into multiple components. This toolbox includes
an **Adaptive Model** that automatically selects the optimal isotropic
compartmentalization based on the acquisition protocol (maximum b-value
and number of shells):

- **3-ISO (Research Protocols)**: Separates Restricted (inflammation/cellularity), Hindered (vasogenic edema), and Free Water (CSF). Activated when $b_{max} \ge 3000 \text{ s/mm}^2$ and $\ge 3$ non-zero shells.
- **2-ISO (Clinical Protocols)**: Merges Hindered and Free Water into a single Non-Restricted fraction to guarantee numerical stability when the free water signal is near the noise floor ($b_{max} < 3000 \text{ s/mm}^2$).

## Outputs

25 channels. `DBSI_Adaptive.output_map_names(mode)` returns them in
order; the `_C_*` constants in `model_Niso_adaptive_ff_thr.py` are the
single source of truth for the indices.

**Isotropic block and total fiber fraction (0-5)**

1. **fiber_fraction (FF)**: apparent axonal density — the TOTAL over both populations.
2. **restricted_fraction (RF)**: cellularity marker (inflammation).
3. **hindered_fraction (HF)**: vasogenic edema *(NaN in 2-ISO mode)*.
4. **water_fraction (WF)**: CSF / free water *(NaN in 2-ISO mode)*.
5. **nonrestricted_fraction (NRF)**: HF + WF combined.
6. **mean_iso_adc**: mean isotropic ADC.

**Fiber block (6-23)** — at most TWO populations per voxel, each with its
own fraction, tensor and direction:

- **n_fiber_populations**: three-state (NaN = no fiber compartment attempted,
  0 = fiber present but no direction resolved, 1-2 = populations found).
- **fiber_fraction_pop1 / _pop2**: each population's share of FF.
- **axial_diffusivity_pop1 / _pop2**: axonal integrity — Stage B closed-form
  (single fiber) or MRDS joint estimate (crossing).
- **radial_diffusivity_pop1 / _pop2**: demyelination marker.
- **fiber_fa_pop1 / _pop2**: intrinsic fiber fractional anisotropy.
- **dir1_x/y/z, dir2_x/y/z**: unit direction vectors.
- **axial_/radial_diffusivity_weighted, fiber_fa_weighted**: the fiber tensor
  averaged over the populations present, weighted by their fractions. Use
  these for a single per-voxel fiber number; `fiber_fa_weighted` is the FA
  OF the weighted tensor, not the mean of the two FAs.

**Diagnostic (24)** — **dominant_basin_concentration**.

The fiber block is NaN wherever the population is absent, while the
compartment fractions use 0. `save_output_maps` writes `fiber_valid.nii.gz`
alongside the maps: resample `value * valid` and `valid` together and divide,
or linear interpolation will turn those NaNs into 0 and depress the result.

## Installation

```bash
git clone https://github.com/guarnich/pyDBSI_toolbox
cd pyDBSI_toolbox
pip install .
```

## Quick Start

### Python API

```python
import os
import nibabel as nib
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.utils.tools import load_data

data, affine, bvals, bvecs, mask = load_data(
    'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'mask.nii.gz'
)

# n_dirs defaults to None (protocol-derived autoconfiguration for Stage A).
# n_ad/n_rd default to a coarse 3x3 grid -- Stage A only needs direction
# detection, not diffusivity resolution (that's Stage B's job).
model = DBSI_Adaptive()

results, model_mode = model.fit(data, bvals, bvecs, mask, run_calibration=True)

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
    --force-n-iso 3  # Optional: override automatic isotropic-model selection (2 or 3)
```

Optional overrides for Stage A's detection dictionary (default: coarse
3×3 grid, protocol-derived direction count):

```bash
python -m scripts.run_dbsi \
    --dwi data.nii.gz --bval data.bval --bvec data.bvec --out results/ \
    --n-dirs 30 --n-ad 3 --n-rd 3 --anisotropy-ratio 1.15 \
    --lambda-aniso 0.6 --lambda-iso 0.005 --min-weight-fraction 0.05
```

## Algorithm Details

### Two-Stage Optimization Approach (v3)

1. **Stage A — direction detection**: regularized NNLS over a coarse
   exhaustive (direction × AD/RD-pair) detection dictionary, with heavy
   sparsity regularization (`lambda_aniso`) on the anisotropic block.
   Fractions (FF, RF, HF, WF) are sums of NNLS weights within each
   compartment's columns — unaffected by the Stage A→B split. Isotropic
   diffusivities (D_res, D_hin, D_wat) are NNLS-weight-weighted centroids,
   as in v1/v2. The dominant fiber direction(s) are identified by
   collapsing the anisotropic block's per-pair weight breakdown and
   looking only at total weight per direction.
2. **Stage B — diffusivity estimation**: given the dominant direction
   from Stage A, a closed-form weighted-least-squares regression (the
   same construction as the v1/v2 analytical AD/RD initialisation, now
   used as the *final* estimate) computes AD and RD.

### Key Parameters

* `n_iso`: Number of isotropic basis functions (default: 31).
* `lambda_aniso`, `lambda_iso`: Stage A regularization strengths
  (default: auto-calibrated; evaluated end-to-end through Stage A + B).
* `n_dirs`: Number of fiber directions on the hemisphere for Stage A
  (default: `None`, autoconfigured from the protocol).
* `n_ad`, `n_rd`: Stage A's AD/RD grid density (default: 3, 3 —
  deliberately coarse; Stage A does not benefit from finer diffusivity
  resolution).
* `anisotropy_ratio`: Minimum AD/RD ratio admissible into Stage A's
  detection dictionary (default: 1.15).
* `min_weight_fraction`: Minimum fraction of total Stage A anisotropic
  weight a direction must carry to be reported as a fiber population
  (default: 0.05).
* `ad_range`, `rd_range`: Physical bounds for Stage A's AD/RD grids,
  mm²/s (default: `(0.5e-3, 2.2e-3)` and `(0.05e-3, 1.2e-3)`).
* `iso_range`: ADC range of the isotropic basis in mm²/s (default:
  `(0.0, 3.0e-3)`).
* `fiber_threshold`: Minimum fiber fraction for AD/RD/FA estimation
  (default: 0.15).
* `force_n_iso`: Forces the algorithm into 2-ISO or 3-ISO mode.

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

v2's design grounded in feedback from Alonso Ramirez-Manzanares on
orientation-space vs. parameter-space dictionary sampling for DBSI; v3's
hybrid two-stage redesign motivated by synthetic recovery validation
showing v2's single-stage approach is not numerically identifiable.
