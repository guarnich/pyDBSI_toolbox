# pyDBSI Toolbox — v1.11.1

**Diffusion Basis Spectrum Imaging (DBSI): adaptive implementation with Numba acceleration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DBSI decomposes the diffusion-weighted signal of each voxel into **anisotropic fiber
compartments** (up to two populations, each with its own fraction, tensor and direction) and an
**isotropic spectrum** split into restricted (cellularity, inflammation), hindered (vasogenic
edema, extracellular water) and free water (CSF). This toolbox is built for clinical multi-shell
protocols and for multiple sclerosis studies: every default below was chosen on synthetic data
with known ground truth and checked on real acquisitions.

- [Inputs](#inputs) — including the **required thermal noise map**
- [Quick start](#quick-start)
- [Outputs](#outputs)
- [Protocol calibration for a cohort](#protocol-calibration-for-a-cohort)

## Inputs

| input | notes |
|---|---|
| preprocessed DWI (4D NIfTI) | preprocessed through topup, eddy and N4 bias correction (recipe below) |
| b-values, b-vectors (FSL format) | the gradient table matching the preprocessed volumes; at least one b=0 (b < 100), used for S0 only |
| brain mask | defines the fitted voxels **and** the calibration sample |
| **thermal noise map** (3D NIfTI) | **required:** output of MP-PCA denoising algorithm |

**3-ISO vs 2-ISO.** With `b_max >= 2000` s/mm² and at least two distinct non-zero shells the
isotropic block is split into RF / HF / WF (3-ISO); otherwise HF and WF are merged into a single
non-restricted fraction NRF (2-ISO).

**Compartment classes (v1.10.0).** Default boundaries RF ≤ 0.3 < HF ≤ 3.0 < WF μm²/ms (Wang et al.
2015, *Brain* 138:1223; Ye et al. 2020, *Ann Clin Transl Neurol* 7:695). They are not a consensus:
tumour studies use RF ≤ 1.0, HF 1.0–1.5, non-restricted > 1.5 (Han et al. 2023, *Neuro-Oncol Adv*
5:vdad050). The reported fractions are the weights of the Stage D centroids (default 0.15 / 1.0 /
3.0 μm²/ms, a modelling choice), not bins of a spectrum, so thresholds and centroids are set
**together**, each centroid inside its class:

```bash
python scripts/run_dbsi.py ... --iso-thresholds 1.0 1.5 --iso-centroids 0.5 1.25 3.0
```

(`DBSI_Adaptive(iso_thresholds=(1.0e-3, 1.5e-3), iso_centroids=(0.5e-3, 1.25e-3, 3.0e-3))` in
Python, mm²/s.) The classes shape Stage A's dictionary, so the λ must be calibrated with them; a
protocol calibration records and imposes them. A single centroid cannot reproduce a wide class:
check new classes on simulations of the tissue of interest before using them.

### The thermal noise map (required)

The Rician correction uses the second moment of the Rice distribution, E[S²] = A² + 2σ², where A is
the noise-free signal: each measurement is replaced by √max(S² − 2σ², 0), which is unbiased for
A² and clamps to zero where the signal sits at the noise floor (Gudbjartsson & Patz 1995;
Koay & Basser 2006). So σ must be the **thermal** noise of the magnitude images and he toolbox therefore requires the
noise map of MP-PCA denoising (Veraart et al. 2016), brought into the intensity scale of the DWI.

The run report gives the **thermal SNR** (median of S0 / σ over the mask) and how many mask voxels
had no valid value in the map.

## Quick start

## Installation

```bash
git clone https://github.com/guarnich/pyDBSI_toolbox
cd pyDBSI_toolbox
pip install -e
python tests/test_noise_map.py        # every tests/test_*.py runs as a plain script
```

Requirements: Python ≥ 3.8, NumPy ≥ 1.20, Numba ≥ 0.55, NiBabel ≥ 3.2, SciPy ≥ 1.7, tqdm ≥ 4.60.
For the noise map: MRtrix3 (`dwidenoise`, `dwibiascorrect`, `mrcalc`).


### Command line

```bash
python scripts/run_dbsi.py \
    --dwi dwi_N4.nii.gz --bval dwi.bval --bvec dwi.bvec \
    --mask brain_mask.nii.gz --noise noise_N4.nii.gz \
    --protocol-calibration path_to_calibratio_file.json \
    --out results/
```

Without `--protocol-calibration` the hyperparameters are calibrated on the acquisition itself
(fine for a single scan, not for a cohort). Options you do not pass take the
constructor's own default;

### Python

```python
import nibabel as nib
from dbsi_toolbox import DBSI_Adaptive, load_data, save_output_maps

data, affine, bvals, bvecs, mask = load_data('dwi_N4.nii.gz', 'dwi.bval', 'dwi.bvec',
                                             'brain_mask.nii.gz')
noise = nib.load('noise_N4.nii.gz').get_fdata()

model = DBSI_Adaptive.from_calibration('P3_calibration.json')    # or DBSI_Adaptive()
results, mode = model.fit(data, bvals, bvecs, mask, noise_map=noise)
save_output_maps(results, DBSI_Adaptive.output_map_names(mode), affine, 'results/', model=model)
```

Passing the model to `save_output_maps` writes `toolbox_report/` next to the maps (provenance,
dictionary, uncertainty maps).

## Outputs

27 channels; read them **by name** (`DBSI_Adaptive.output_map_names(mode)`).

| # | channel | |
|---|---|---|
| 0 | `fiber_fraction` | total fiber fraction, both populations |
| 1–4 | `restricted_fraction`, `hindered_fraction`, `water_fraction`, `nonrestricted_fraction` | HF/WF are NaN in 2-ISO |
| 5 | `n_fiber_populations` | NaN = no fiber (Stage A or final FF ≤ threshold), 0 = fiber mass without a resolved direction (FF 0 after Stage D), 1–2 |
| 6–12 | `fiber_fraction_pop1`, `axial_/radial_diffusivity_pop1`, `fiber_fa_pop1`, `dir1_x/y/z` | NaN if absent |
| 13–19 | same for population 2 | NaN if absent |
| 20–22 | `axial_/radial_diffusivity_weighted`, `fiber_fa_weighted` | tensor averaged over the populations, weighted by their fractions; FA of that tensor |
| 23–26 | `dominant_basin_concentration`, `fit_r2`, `fit_rmse`, `nnls_iterations` | diagnostics |

Fractions use 0 for "compartment absent"; the per-population block and the weighted tensor use
NaN. `save_output_maps` writes `NN_<channel>.nii.gz`, the validity masks `fiber_valid.nii.gz` /
`fiber_valid_pop2.nii.gz` (resample `value × valid` and `valid` together and divide, or
interpolation turns NaN into 0), `fiber_detection_stat.nii.gz`, the fiber directions as
`dir1.nii.gz` / `dir2.nii.gz`, and `toolbox_report/`.

**Directions.** `dirK_x/y/z` are the three components of a unit vector along population K's
axis: each is the cosine of the angle with that image axis, hence in [-1, 1]. The sign is
arbitrary (v and −v are the same fiber). The vector is in the frame of the bvecs passed to the
fit, with no rotation: with FSL-convention bvecs (dcm2niix) it is the frame of `dtifit`'s V1.
`dir1.nii.gz` / `dir2.nii.gz` stack the three channels into one (X, Y, Z, 3) image, the layout
of V1, so FSLeyes shows them as RGB or line vectors (modulate by `fiber_fraction_pop1/2`).
Do not resample them like the scalar maps: the vectors must be rotated with the transform, and linear interpolation averages v with −v.

`toolbox_report/`:

| file | content |
|---|---|
| `run_report.txt` | version, git commit, noise (thermal SNR, σ source), Rician truncation, fibers removed by the final threshold, calibration and its source, options, population census, solver diagnostics, detection statistic, uncertainty summary |
| `design_matrix.png/.npz`, `dictionary_columns.csv` | the Stage A dictionary actually used |
| `protocol_calibration.json` | copy of the protocol calibration, when used |
| `uncertainty_maps/` | `NN_<channel>_se.nii.gz` for every continuous channel, `dir1/2_angle_se_deg`, `fisher_log10_condition`, `residual_over_sigma`, `uncertainty_flags`, `README.txt` |

**Uncertainty maps.** The standard error of each continuous map is the Cramér–Rao bound
at the estimate: Fisher information of the reported model (fibers with weight, AD, RD and
direction free; fixed isotropic centroids), delta method for normalised fractions, FA and the
weighted tensor. It is the precision the data and protocol allow, **not the bias**. 
Flags: 1 RD on a bound, 2 AD on a bound, 4 ill-conditioned Fisher matrix, 16 residual larger than the noise explains. For a region of N voxels the error of the mean shrinks by up to √N.

## Protocol calibration for a cohort

Hyperparameters calibrated per acquisition follow that acquisition's noise: in a group comparison
the pipeline would manufacture differences. Calibrate **once per protocol**, on a
group-balanced sample, **with the noise maps**:

```python
from dbsi_toolbox import DBSI_Adaptive, calibrate_protocol, save_protocol_calibration

records = [DBSI_Adaptive().calibrate(data, bvals, bvecs, mask, noise_map=noise)   # no voxel fit
           for data, bvals, bvecs, mask, noise in sample]
cal = calibrate_protocol(records, name='P3', ids=subject_ids)      # rule='geomean'
save_protocol_calibration(cal, 'P3_calibration.json')              # returns the sha256

model = DBSI_Adaptive.from_calibration('P3_calibration.json')      # every acquisition
```

**Step by step: `scripts/calibration.ipynb`.** Point it at the acquisitions (a glob or a list of
directories) and at the file names *as your study names them* (one glob pattern per input,
relative to each acquisition directory); it checks every input without fitting (grids,
orientation of mask and noise map, one protocol fingerprint for the whole sample), calibrates each
acquisition on the toolbox's λ grids (118 log-spaced points), writes `<protocol>_calibration.json` and plots the GCV curves behind the choice.

`calibrate_protocol` refuses records calibrated without a noise map. `from_calibration` imposes
the λ, `n_iso`, the gate and the dictionary options, refuses data with a different protocol
fingerprint (`strict=True`) and refuses overrides of calibrated parameters; the run report states
`calibration_source: protocol` with the file's sha256. Aggregation: `'geomean'` minimises the mean
of the log GCV curves (scale-invariant, so a noisier acquisition does not outvote a cleaner one).


## Notebooks

`notebooks/` holds two self-contained notebooks, each run once to produce all its figures (see
`notebooks/README.md`): `simulazioni_pyDBSI.ipynb` measures accuracy against ground truth on synthetic
phantoms through the production chain (images true / estimated / error, error per tissue scenario and SNR,
estimate versus truth, coverage of the uncertainty maps); `matematica_pyDBSI.ipynb` draws the mathematics of
the fit section by section, from the toolbox's own recorded inputs and calibration steps.


## References

1. Wang Y, et al. (2011). Quantification of increased cellularity during inflammatory
   demyelination. *Brain*, 134(12), 3590–3601.
2. Wang Y, et al. (2015). Differentiation and quantification of inflammation, demyelination and
   axon injury or loss in multiple sclerosis. *Brain*, 138(5), 1223–1238.
3. Vavasour IM, et al. (2022). Characterisation of multiple sclerosis neuroinflammation and
   neurodegeneration with relaxation and diffusion basis spectrum imaging. *Multiple Sclerosis
   Journal*, 28(3), 418–428.
4. Veraart J, et al. (2016). Denoising of diffusion MRI using random matrix theory.
   *NeuroImage*, 142, 394–406.
5. Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Estimation of individual axon
   bundle properties by a Multi-Resolution Discrete-Search method. *Medical Image Analysis*, 42,
   26–43.

