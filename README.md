# pyDBSI Toolbox (v1.0.0 — hybrid two-stage architecture)

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

Stage B's closed-form estimate is no longer the last word, though. Two
further stages, both ON by default, sit on top of it — see *Algorithm
details* below.

## Overview

DBSI is an advanced diffusion MRI technique that decomposes the
diffusion-weighted signal into multiple components. This toolbox includes
an **Adaptive Model** that automatically selects the optimal isotropic
compartmentalization based on the acquisition protocol (maximum b-value
and number of shells):

- **3-ISO**: separates Restricted (inflammation/cellularity), Hindered (vasogenic edema) and Free Water (CSF). Activated when $b_{max} \ge 2000 \text{ s/mm}^2$ **and** there are $\ge 2$ distinct non-zero shells (`B_THRESH_3ISO`, `MIN_SHELLS_3ISO`).
- **2-ISO**: merges Hindered and Free Water into a single Non-Restricted fraction, for numerical stability when the free-water signal sits near the noise floor. Selected whenever either 3-ISO condition fails.

## Outputs

27 channels. `DBSI_Adaptive.output_map_names(mode)` returns them in
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

**Diagnostics (24-26)**

- **dominant_basin_concentration**: angular concentration of the dominant basin.
- **fit_r2 / fit_rmse**: voxel-wise goodness of fit of the reconstructed signal
  (all compartments, both populations) and the residual RMSE as a fraction of
  S0. Computed by the fit itself — the maps always ship with the means to judge
  them.

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
from dbsi_toolbox import DBSI_Adaptive, load_data, save_output_maps

data, affine, bvals, bvecs, mask = load_data(
    'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'mask.nii.gz'
)

# The defaults are the validated configuration: n_dirs is protocol-derived,
# Stage A uses a coarse 3x3 (AD, RD) grid because it only needs to detect the
# fiber direction, and Stages C/D plus the concentration modulation are on.
model = DBSI_Adaptive()
results, model_mode = model.fit(data, bvals, bvecs, mask, run_calibration=True)

names = DBSI_Adaptive.output_map_names(model_mode)
save_output_maps(results, names, affine, 'results/')
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
    --n-dirs 30 --n-ad 3 --n-rd 3 --anisotropy-ratio 2.0 \
    --lambda-aniso 0.6 --lambda-iso 0.005 --min-weight-fraction 0.05
```

## Algorithm Details

The stages below run in this order. Stages C and D and the concentration
modulation are **on by default**; the full per-voxel walkthrough with formulas
is in `voxel_journey_report`.

1. **Stage A — direction detection**: regularized NNLS over a coarse
   exhaustive (direction × AD/RD-pair) detection dictionary, with heavy
   sparsity regularization (`lambda_aniso`) on the anisotropic block. Raw
   fractions (FF, RF, HF, WF) are sums of NNLS weights within each
   compartment's columns; the isotropic diffusivities are weighted
   centroids. The fiber direction(s) are found by collapsing the
   anisotropic block's per-pair weights and looking only at total weight
   per direction.
2. **Concentration modulation**: after the first NNLS, the *angular
   concentration* of the anisotropic weight is measured. Where it is low
   (diffuse weight = isotropic leakage rather than a real fiber),
   `lambda_aniso` is boosted and the voxel is re-solved. A continuous ramp,
   so low-FA fibers are not crushed along with the leakage.
3. **Population detection**: local maxima on the direction graph, Voronoi
   basin mass, a data-driven concentration gate, then greedy selection with
   angular non-maximum suppression. Yields 0, 1 or **at most 2**
   populations (`MAX_FIBER_POPULATIONS`, fixed).
4. **Per-voxel tensor estimation**, by branch:
   - **one population → Stage C (VARPRO)**: fiber tensor *and* compartment
     fractions are re-solved jointly on a reduced `[fiber column | isotropic
     grid]` dictionary, scanning a 14×12 (AD, RD) grid with a 5×5 local
     refine. This replaces both the raw Stage A fractions and the Stage B
     tensor, which poison each other: near-fiber columns absorb restricted
     signal, so FF inflates, RF collapses and AD is under-estimated. Uses
     the **raw** normalised signal — Rician noise-floor subtraction biases
     the tensor low.
   - **two populations → MRDS joint Stage B**: a short symmetry-breaking
     alternating warm start, then a bounded Levenberg–Marquardt fit of both
     tensors simultaneously, with directions and fractions held fixed.
   - **Stage B closed-form** — a weighted least-squares regression on the
     log-signal, no grid — is the fallback used for a single fiber only when
     Stage C is disabled.
5. **Stage D — constrained fraction re-solve** (whole volume): for every
   fitted voxel the compartment fractions are re-estimated by NNLS over
   `[detected fibers | fixed isotropic centroids]` on the **Rician-corrected**
   signal. Stage A's over-complete spectrum smears weight across nearly
   collinear atoms and mis-bins RF/HF/WF; a few well-placed columns are a
   far better-conditioned estimator. For crossings only the isotropic split
   is taken from Stage D, not the total FF.
6. **Derived channels**: the population-1 fraction, the FF-weighted fiber
   tensor, and the R²/RMSE fit-quality maps.

### Key Parameters

* `n_iso`: number of isotropic basis functions (default: `None` —
  selected per dataset by the bootstrap bias/variance method; typically 6–10).
* `lambda_aniso`, `lambda_iso`: Stage A regularization strengths
  (default: auto-calibrated; evaluated end-to-end through Stage A + B).
* `n_dirs`: Number of fiber directions on the hemisphere for Stage A
  (default: `None`, autoconfigured from the protocol).
* `n_ad`, `n_rd`: Stage A's AD/RD grid density (default: 3, 3 —
  deliberately coarse; Stage A does not benefit from finer diffusivity
  resolution).
* `anisotropy_ratio`: minimum AD/RD ratio admissible into Stage A's
  detection dictionary (default: **2.0**). Raised from 1.15 to drop the
  near-isotropic ratio-1.83 grid column, which leaked isotropic signal into
  `fiber_fraction`.
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
