# pyDBSI Toolbox — v1.9.0

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
- [How a voxel is fitted](#how-a-voxel-is-fitted)
- [Outputs](#outputs)
- [Protocol calibration for a cohort](#protocol-calibration-for-a-cohort)
- [Key parameters](#key-parameters)
- [What the numbers mean — known limits](#what-the-numbers-mean--known-limits)
- [Version history](#version-history)

## Inputs

| input | notes |
|---|---|
| preprocessed DWI (4D NIfTI) | preprocessed through topup, eddy and N4 bias correction (recipe below) |
| b-values, b-vectors (FSL format) | the gradient table matching the preprocessed volumes; at least one b=0 (b < 100), used for S0 only |
| brain mask | defines the fitted voxels **and** the calibration sample |
| **thermal noise map** (3D NIfTI) | **required since v1.7.1**, see below |

**3-ISO vs 2-ISO.** With `b_max >= 2000` s/mm² and at least two distinct non-zero shells the
isotropic block is split into RF / HF / WF (3-ISO); otherwise HF and WF are merged into a single
non-restricted fraction NRF (2-ISO).

### The thermal noise map (required)

The Rician correction subtracts 2σ² from the squared magnitude signal, so σ must be the
**thermal** noise of the magnitude images. The temporal spread of the b=0 volumes is not: it also
contains pulsation and motion. On four real acquisitions it was **1.3–2.2× the thermal noise, by a
different factor for every subject**, and correcting with it shifted the restricted fraction by up
to 0.04, a subject-dependent bias a cohort study cannot afford. The toolbox therefore requires the
noise map of MP-PCA denoising (Veraart et al. 2016), brought into the intensity scale of the DWI
it reads:

```bash
dwidenoise dwi.nii.gz dwi_denoised.nii.gz -noise noise.nii.gz            # MP-PCA noise map
# ... mrdegibbs, dwifslpreproc (topup + eddy) ...
dwibiascorrect ants dwi_pre.nii.gz dwi_N4.nii.gz -bias bias.nii.gz ...    # keep the N4 field
mrcalc noise.nii.gz bias.nii.gz -div -finite noise.nii.gz bias.nii.gz -div 0 -if noise_N4.nii.gz
```

N4 divides the signal by the bias field, so the noise is divided by it too. The map is computed
before topup/eddy on the same voxel grid: motion and distortion displace it by a few millimetres,
negligible for a field that varies over centimetres; the intensity modulation that topup/eddy
apply in distorted regions is not followed. The map is used for the Rician correction and for
every σ-normalised quantity (crossing ridge, detection statistic, uncertainty maps, λ
calibration). Mask voxels with no valid value are filled with the median and counted in the run
report; a map with less than 50% valid mask voxels, or on another grid, is rejected.

The run report gives the **thermal SNR** (median of S0 / σ over the mask) and how many mask voxels
had no valid value in the map.

## Quick start

### Command line

```bash
python scripts/run_dbsi.py \
    --dwi dwi_N4.nii.gz --bval dwi.bval --bvec dwi.bvec \
    --mask brain_mask.nii.gz --noise noise_N4.nii.gz \
    --protocol-calibration P3_calibration.json \
    --out results/
```

Without `--protocol-calibration` the hyperparameters are calibrated on the acquisition itself
(fine for a single scan, not for a cohort: see below). Options you do not pass take the
constructor's own default; the CLI gives maps identical bit for bit to the Python call
(`tests/test_cli_matches_production.py`).

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

## How a voxel is fitted

1. **Rician correction** with the thermal noise map, voxel by voxel: `sqrt(max(S² − 2σ², 0))`.
   The fraction of measurements truncated to 0 is reported.
2. **Stage A — direction detection.** Regularised NNLS over a coarse exhaustive dictionary
   (hemisphere directions × a 3×3 (AD, RD) grid with AD/RD ≥ 2, plus an anchored isotropic grid).
   Heavy sparsity on the anisotropic block: Stage A is trusted for *which directions* carry fiber
   signal, not for their diffusivities. Where the anisotropic weight is angularly diffuse
   (isotropic leakage), the anisotropic penalty is boosted and the voxel re-solved
   (concentration modulation).
3. **Populations.** Local maxima on an axial direction graph, basin masses, greedy selection with
   angular non-maximum suppression: 0, 1 or at most 2 candidate populations. The angular-concentration
   gate is off by default (since 1.4.0): it rejected exactly the wide-angle crossings the dictionary can
   resolve. Whether a candidate crossing stays a crossing is decided by the two-vs-one test (step 7).
4. **Single fiber.** The direction is refined inside a cone tied to the dictionary spacing, then
   **Stage C** re-solves the fiber tensor and the compartment fractions jointly (variable
   projection over a 14×12 (AD, RD) grid with local refinement) on the raw normalised signal.
5. **Crossing.** The fiber fraction is **re-solved on the detected support** (the
   detected directions' grid nodes and their neighbours, every admissible (AD, RD) pair, plus the
   isotropic grid) with a light ridge κ·(σ/S0)², then both tensors are fitted jointly by a bounded
   Levenberg–Marquardt with fractions and directions fixed. Stage A's own crossing fraction is
   shrunk by its sparsity penalty: held fixed, the tensors compensated by turning more anisotropic
   and 75% of real crossings had their RD on the floor. Then each population's direction leaves its
   grid node by a **cone search**, one population at a time, the two tensors re-fitted at every
   candidate (`crossing_dir_refine`, since 1.9.0): on randomly oriented crossings at SNR 26 the median
   angular error halves (10 → 5°).
6. **Stage D.** For every fitted voxel the final compartment fractions are re-estimated by NNLS
   over `[fitted fibers | three fixed isotropic centroids (0.15, 1.0, 3.0)×10⁻³ mm²/s]` on the
   Rician-corrected signal (RF/NRF centroids in 2-ISO), crossings included. A crossing whose
   population Stage D gives no weight becomes a single fiber. **`fiber_threshold` is applied to
   this final fiber fraction**: where it is ≤ 0.15 the fiber is removed and the fractions are
   re-solved on the isotropic centroids alone, so a tensor is reported only where the reported
   FF > 0.15 (counted in the run report as `fiber_threshold_final`).
7. **Two-vs-one test** (since 1.9.0). Every voxel with a separated secondary peak is fitted both as a
   crossing and as a single fiber from the same Stage A; the crossing is kept only where
   `(RSS_1pop − RSS_2pop) / σ_voxel² ≥ 12` on the Stage D model, before the fiber threshold. It replaces
   the hand-set basin-mass cut (`min_peak_ratio` 0.35) that gave false second populations and missed
   unbalanced crossings at every SNR. Statistic in `two_vs_one_stat.nii.gz`.
8. **Fiber detection statistic.** `(RSS_iso_only − RSS_fibers+iso) / σ_voxel²` on the Stage D
   model, always computed and written; the rejection threshold is off by default (see
   `fiber_detection_threshold`).
9. **Derived channels, fit quality and uncertainty.** Population-1 fraction, FF-weighted tensor,
   R²/RMSE of the reconstructed signal, and the per-voxel standard errors.

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
interpolation turns NaN into 0), `fiber_detection_stat.nii.gz`, and `toolbox_report/`:

| file | content |
|---|---|
| `run_report.txt` | version, git commit, noise (thermal SNR, σ source), Rician truncation, fibers removed by the final threshold, calibration and its source, options, population census, solver diagnostics, detection statistic, uncertainty summary |
| `design_matrix.png/.npz`, `dictionary_columns.csv` | the Stage A dictionary actually used |
| `protocol_calibration.json` | copy of the protocol calibration, when used |
| `uncertainty_maps/` | `NN_<channel>_se.nii.gz` for every continuous channel, `dir1/2_angle_se_deg`, `fisher_log10_condition`, `residual_over_sigma`, `uncertainty_flags`, `README.txt` |

**Uncertainty maps.** The standard error of each continuous map is the Cramér–Rao bound
at the estimate: Fisher information of the reported model (fibers with weight, AD, RD and
direction free; fixed isotropic centroids), delta method for normalised fractions, FA and the
weighted tensor. It is the precision the data and protocol allow, **not the bias**. On synthetic
data it is calibrated for single fibers and isotropic tissue (SE / empirical SD 0.95–1.35,
coverage 93–100%) and conservative for crossings. Flags: 1 RD on a bound, 2 AD on a bound,
4 ill-conditioned Fisher matrix, 16 residual larger than the noise explains. For a region of N
voxels the error of the mean shrinks by up to √N.

## Protocol calibration for a cohort

Hyperparameters calibrated per acquisition follow that acquisition's noise (on 40 acquisitions of
one protocol, Spearman with SNR −0.75 for `lambda_aniso`, −0.72 for `lambda_iso`): in a group
comparison the pipeline would manufacture differences. Calibrate **once per protocol**, on a
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
acquisition and keeps its record (an interrupted run resumes), aggregates, writes
`<protocol>_calibration.json`, plots the GCV curves behind the choice and prints the command line:

```bash
python scripts/run_dbsi.py --dwi ... --bval ... --bvec ... --mask ... --noise ... --out ... \
    --protocol-calibration P3_calibration.json                  # recommended
python scripts/run_dbsi.py --dwi ... --bval ... --bvec ... --mask ... --noise ... --out ... \
    --skip-calibration --lambda-aniso 5.2233 --lambda-iso 0.1 --n-iso 6 --n-dirs 62   # the same fit
```

The two give identical maps on an acquisition of the protocol; only the file refuses an
acquisition with a different fingerprint and records its sha256 in the run report.

`calibrate_protocol` refuses records calibrated without a noise map. `from_calibration` imposes
the λ, `n_iso`, the gate and the dictionary options, refuses data with a different protocol
fingerprint (`strict=True`) and refuses overrides of calibrated parameters; the run report states
`calibration_source: protocol` with the file's sha256. Aggregation: `'geomean'` minimises the mean
of the log GCV curves (scale-invariant, so a noisier acquisition does not outvote a cleaner one);
`'median_index'` is also reported, with bootstrap stability and each acquisition's distance from
its own optimum. `lambda_iso` is capped by the acquisitions' discrepancy ceilings.

## Key parameters

| parameter | default | |
|---|---|---|
| `two_vs_one_threshold` | 12 | keep a crossing only where it lowers the Stage D residual by ≥ 12 noise variances; `None` (CLI `--no-two-vs-one`) = the pre-1.9 `min_peak_ratio` cut. Synthetic P3 battery: false second populations 36/16/4.5% → 3.3/1.1/0.2%, crossings found 64/51/40% → 64/80/91% at SNR 15/26/40. Real P3 (BR 000148): crossings in GM 35% → 14%, new crossings in the centrum semiovale, midline corpus callosum unchanged. Needs `iso_resolve` (the default switches itself off without it). Fit ×1.3 |
| `crossing_dir_refine` | `True` | cone search on each crossing population's direction (CLI `--no-crossing-dir-refine`) |
| `crossing_ff_kappa` | 30 | ridge of the crossing fraction re-solve, × (σ/S0)²; 0 keeps Stage A's crossing fraction. 65 centres healthy crossing FF on three protocols at SNR 15 and 26; 30 keeps more demyelination contrast at low SNR |
| `fiber_detection_threshold` | `None` (off) | reject fibers whose detection statistic is below it. Off by decision (Oct 2026): adoptable on the synthetic battery (threshold 15), but its real-data control failed and the crossing re-solve changes the statistic |
| `uncertainty` | `True` | write the uncertainty maps |
| `iso_resolve` | `True` | Stage D; the crossing re-solve and the uncertainty maps need it |
| `stagec_refine`, `stagec_dir_refine`, `lambda_aniso_conc_mod` | `True` | Stage C, cone refinement before it, concentration modulation |
| `lambda_aniso`, `lambda_iso` | calibrated | Stage A penalties: GCV (`lambda_aniso_method='gcv'`) and residual GCV with a discrepancy cap; impose them from a protocol calibration in a cohort |
| `n_iso` | 6 | density of the anchored isotropic grid (11 columns); a quadrature parameter, not a model order: Stage D reports fractions on three fixed centroids |
| `min_dominant_concentration` | 0.0 (off) | angular-concentration gate |
| `anisotropy_ratio` | 2.0 | minimum AD/RD of Stage A's fiber atoms (drops the near-isotropic column that leaked into FF) |
| `n_dirs`, `n_ad`, `n_rd` | auto, 3, 3 | Stage A dictionary; directions from the protocol |
| `fiber_threshold` | 0.15 | minimum fiber fraction for a tensor: on Stage A's FF (candidates) and on the final FF (reported) |
| `force_n_iso` | `None` | force 2-ISO or 3-ISO |

## What the numbers mean — known limits

- **Fiber vs hindered water.** As a fiber loses anisotropy it becomes indistinguishable from
  hindered water: at fiber FA 0.4 the Cramér–Rao SD of FF is 0.45 at SNR 26, while FF+HF and RF
  stay determined (0.07 and 0.04). In advanced lesions trust RF, WF and FF+HF before the FF/HF
  split; the uncertainty maps say voxel by voxel when the split is lost.
- **Demyelination contrast is compressed.** RD of a demyelinated fiber is under-estimated
  (≈ −0.1×10⁻³ mm²/s for a single fiber at SNR 26): the sign of a lesion contrast is preserved,
  its magnitude is not. Report contrasts, not absolute RD, and average over regions.
- **Per-population AD in crossings is not quantitative** (Cramér–Rao SD ≈ 0.6×10⁻³): use
  `axial_diffusivity_weighted` there. Per-population RD in crossings is reliable.
- **Perfusion is not modelled.** Fine in white matter (pseudo-diffusion fraction ≈ 0 on real data);
  in grey matter the free-water fraction includes ≈ 0.03 of perfusion signal, more in CSF-adjacent
  voxels.
- **Three populations** are out of scope (`MAX_FIBER_POPULATIONS = 2`).

## Notebooks

`notebooks/` holds two self-contained notebooks, each run once to produce all its figures (see
`notebooks/README.md`): `simulazioni_pyDBSI.ipynb` measures accuracy against ground truth on synthetic
phantoms through the production chain (images true / estimated / error, error per tissue scenario and SNR,
estimate versus truth, coverage of the uncertainty maps); `matematica_pyDBSI.ipynb` draws the mathematics of
the fit section by section, from the toolbox's own recorded inputs and calibration steps.

## Version history

| version | |
|---|---|
| **1.9.0** | **crossings, two defaults changed**: the cone search on each crossing population's direction (`crossing_dir_refine=True`) and the two-vs-one population test (`two_vs_one_threshold=12`) replace grid-node directions and the `min_peak_ratio` cut; the single-fiber alternative is fitted inside the kernel from the same Stage A (fit ×1.3–1.8); `two_vs_one_stat.nii.gz` and a run-report block; CLI `--no-two-vs-one`, `--[no-]crossing-dir-refine`; with both off the fit is 1.8.1's up to 1 float32 ulp in ~1% of crossing voxels. Protocol calibrations made with 1.8 stay valid (the λ do not depend on these options). `scripts/calibration.ipynb`: from the study folders to the fixed-λ command. `notebooks/`: a third simulation arm and two sections (cone, two-vs-one test) in the mathematics |
| 1.8.1 | `fit` checks that bvals/bvecs match the number of volumes (a bvec table one volume short used to run silently, reading past the end of the table in the numba kernels, and return meaningless maps); the run report now prints the `input` and `fiber_threshold_final` blocks (they were in the dict only); `select_lambdas_data_driven` defaults to `lambda_aniso_method='gcv'` like the model; optional paths, off by default, made consistent with the fit: the n_iso SVD counts modes on the fit's grid range [0.1, 5.0]e-3 (was 3.0e-3), the n_iso bootstrap replicates get the fit's Rician correction, the concentration-gate MC null warns that it is computed before the λ_aniso modulation; `experiments/` scripts pass the noise map again; `setup.py` no longer declares a non-existent `model` module; package, calibration, CLI and uncertainty-README texts updated; after the `v1.8.1` tag, still 1.8.1: code the fit never called removed (`calibration/optimizer.py`, `select_n_iso_with_gcv_crosscheck`, `select_n_iso_data_driven_sweep`, `generate_isotropic_grid`, `compute_weighted_centroids`, `generate_fibonacci_sphere`), maps identical bit for bit; `experiments/` moved out of the repository (the tests' synthetic P3 table is now in `tests/data/`); the free calibration's per-voxel NNLS solved in parallel (`core.solvers.nnls_batch`), ~4x faster and bit for bit the same lambdas; `notebooks/`: accuracy simulations against ground truth and the toolbox mathematics in figures |
| 1.8.0 | `fiber_threshold` applied to the final fiber fraction; calibration sigma = RMS of σ/S0 over the sample (the median made the λ_iso ceiling collapse to its grid floor); tSNR removed (one b=0 is enough); single b=0 threshold (100) everywhere, protocol fingerprints included; stale channel map `DBSI_Adaptive.CH` removed |
| 1.7.1 | thermal noise map **required** (`noise_map` / `--noise`); crossings whose population Stage D zeroes become single fibers |
| 1.7.0 | crossing fiber fraction re-solved before the tensor fit; uncertainty maps; `mean_iso_adc` removed (27 channels); detection threshold kept off |
| 1.6.4 | numba kernels no longer use `fastmath` NaN/inf assumptions (they made NaN checks platform-dependent: x86 vs arm64) |
| 1.6.2–1.6.3 | non-finite input voxels excluded; detection statistic computed serially |
| 1.6.1 | fiber detection statistic; SURE cross-check on the fiber-subtracted residual; CLI aligned to production |
| 1.6.0 | restricted-fraction Monte Carlo correction removed (Stages C/D fix RF at the source) |
| 1.5.x | protocol calibration; axial direction graph; fit quality on the Stage D model |
| ≤ 1.4.0 | Stage C/D, concentration modulation, gate off, `toolbox_report/`, unbiased noise estimator |

## Installation

```bash
git clone https://github.com/guarnich/pyDBSI_toolbox
cd pyDBSI_toolbox
pip install -e .
python tests/test_noise_map.py        # every tests/test_*.py runs as a plain script
```

Requirements: Python ≥ 3.8, NumPy ≥ 1.20, Numba ≥ 0.55, NiBabel ≥ 3.2, SciPy ≥ 1.7, tqdm ≥ 4.60.
For the noise map: MRtrix3 (`dwidenoise`, `dwibiascorrect`, `mrcalc`).

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

The two-stage design (direction detection, then diffusivity estimation) followed synthetic
recovery validation showing that a single exhaustive dictionary is not numerically identifiable
for AD/RD; the orientation-space dictionary sampling was grounded in feedback from Alonso
Ramirez-Manzanares.
