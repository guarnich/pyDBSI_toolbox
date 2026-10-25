#!/usr/bin/env python
"""
DBSI Adaptive CLI — the production fit of one session, from the shell.

Same fit as the notebooks, bit for bit (tests/test_cli_matches_production.py):
every default is read from DBSI_Adaptive's constructor, never restated here.
Inputs are the fixed starting point of the toolbox: preprocessed DWI (topup,
eddy, N4), bval/bvec, brain mask and the thermal noise map (--noise).
Outputs adapt to the acquisition scheme (2-ISO or 3-ISO).

- Stage A's (AD,RD) grid (`--n-ad`, `--n-rd`, `--anisotropy-ratio`) is
  deliberately coarse whatever the protocol: Stage A only detects fibers and
  directions; the tensor is estimated afterwards (Stage C for single fibers,
  crossing FF re-solve + MRDS for crossings).
- `--n-dirs` defaults to protocol-derived (autoconfigured) sizing.
- Calibration is data-driven on a sample of this dataset's own voxels:
  lambda_iso by GCV on the fiber-subtracted residual, capped by the
  discrepancy principle; lambda_aniso by ridge-dof GCV; n_iso fixed by
  quadrature (6). `--protocol-calibration` imposes a protocol calibration
  instead. See the dbsi_toolbox.calibration module docstring.

OUTPUT LAYOUT (changed)
-----------------------
Saving now goes through `dbsi_toolbox.save_output_maps`, the single
saving entry point shared with the analysis notebooks, instead of an
inline loop private to this script. Consequences:

- files are named `NN_<channel>.nii.gz` (channel index prefix), NOT
  `dbsi_<channel>.nii.gz` as before;
- `nonrestricted_fraction` is not written in 3-ISO mode, where it is
  hindered + water by construction;
- `fiber_valid.nii.gz`, the validity mask for the fiber-tensor channels,
  and `toolbox_report/` (run_report.txt + the dictionary A as image, .npz
  and column table), recording what produced the maps, are written as part
  of the normal run;
- the R2 / RMSE fit-quality maps are ordinary output channels now, always
  computed by the fit, so `--compute-r2` is gone.

The output layout itself changed too: at most TWO fiber populations, each
with its own fraction/AD/RD/FA/direction, plus the FF-weighted tensor —
plus the R2/RMSE fit-quality maps — 27 channels, no pop3, no
ad_linear/rd_linear. See `DBSI_Adaptive.output_map_names`.
"""

import argparse
import inspect
import os

from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox import load_data
from dbsi_toolbox import save_output_maps

# Defaults are READ from the constructor, never restated here. Until v1.6.1 this
# script passed its own copies (n_ad=3, n_rd=3, ratio 2.0, 0.05, 1.0): equal to
# the constructor's today, but a second source of truth that would have kept the
# CLI on the old values, silently, the day a default changed. Options left at
# None are simply not passed.
_D = {k: v.default for k, v in inspect.signature(DBSI_Adaptive.__init__).parameters.items()}
_DICTIONARY_OPTS = ('n_ad', 'n_rd', 'anisotropy_ratio')


def main():
    parser = argparse.ArgumentParser(description="pyDBSI: DBSI Adaptive fit of one session")
    parser.add_argument("--dwi", required=True)
    parser.add_argument("--bval", required=True)
    parser.add_argument("--bvec", required=True)
    parser.add_argument("--mask", required=True,
                        help="Brain mask. REQUIRED: it defines both the fitted voxels "
                             "and the voxels sampled for calibration, so it changes the "
                             "calibrated lambdas, not just the field of view.")
    parser.add_argument("--noise", required=True,
                        help="Thermal noise map (sigma per voxel) in the SAME scale and grid as --dwi: "
                             "the dwidenoise -noise map divided by the N4 bias field (dwibiascorrect "
                             "-bias). Used for the Rician correction and every sigma-normalised "
                             "quantity. REQUIRED (v1.7.1): the b=0 temporal sigma also contains "
                             "physiological variability and is not a substitute.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--protocol-calibration", dest="protocol_calibration", default=None,
                        help="JSON of a PROTOCOL calibration (calibrate_protocol + "
                             "save_protocol_calibration). Imposes lambda_aniso, lambda_iso, "
                             "n_iso, the gate and the dictionary options, and refuses data "
                             "whose protocol fingerprint differs. Incompatible with --n-iso, "
                             "--lambda-aniso, --lambda-iso, --n-dirs.")
    parser.add_argument("--n-iso", type=int, default=None)
    parser.add_argument("--lambda-aniso", type=float, dest="lambda_aniso", default=None,
                        help="Stage A regularization strength for the anisotropic detection block.")
    parser.add_argument("--lambda-iso", type=float, dest="lambda_iso", default=None,
                        help="Stage A regularization strength for the isotropic spectrum block.")
    parser.add_argument("--n-dirs", type=int, default=None,
                        help="Number of hemisphere directions for Stage A. Default: protocol-derived.")
    parser.add_argument("--n-ad", type=int, default=None,
                        help=f"Number of AD grid steps for Stage A's detection dictionary. Default: {_D['n_ad']} (deliberately coarse).")
    parser.add_argument("--n-rd", type=int, default=None,
                        help=f"Number of RD grid steps for Stage A's detection dictionary. Default: {_D['n_rd']} (deliberately coarse).")
    parser.add_argument("--anisotropy-ratio", type=float, dest="anisotropy_ratio", default=None,
                        help=f"Minimum AD/RD ratio for admissible Stage A pairs. Default: {_D['anisotropy_ratio']} "
                             "(raised from 1.15: drops the near-isotropic ratio-1.83 grid column "
                             "that leaks isotropic signal into fiber_fraction; safe band [2.0, 2.16]).")
    parser.add_argument("--min-weight-fraction", type=float, dest="min_weight_fraction", default=None,
                        help=f"Stage A direction-selection threshold. Default: {_D['min_weight_fraction']}.")
    parser.add_argument("--fiber-detection-threshold", type=float,
                        dest="fiber_detection_threshold", default=None,
                        help="Reject a voxel's fibers when they reduce the Stage D residual "
                             "by less than this many noise variances (dRSS/sigma^2). Default: "
                             "off. The statistic is always written to "
                             "fiber_detection_stat.nii.gz. See DBSI_Adaptive.")
    parser.add_argument("--two-vs-one-threshold", type=float,
                        dest="two_vs_one_threshold", default=None,
                        help="Decide crossing vs single fiber by model selection: keep the crossing "
                             "only where it lowers the Stage D residual by at least this many noise "
                             "variances. Default: off (min_peak_ratio decides). Use together with "
                             "--crossing-dir-refine. See DBSI_Adaptive.")
    parser.add_argument("--crossing-dir-refine", action="store_true", dest="crossing_dir_refine",
                        help="Refine each crossing population's direction with a cone search "
                             "(default: crossing directions stay on Stage A's grid nodes).")
    parser.add_argument("--crossing-ff-kappa", type=float, dest="crossing_ff_kappa", default=None,
                        help=f"Crossing fiber fraction re-solved before the crossing tensors are "
                             f"fitted, with ridge kappa x (sigma/S0)^2. Default: {_D['crossing_ff_kappa']}. "
                             "0 keeps Stage A's crossing fraction (held fixed, it put "
                             "~75%% of real crossings on the RD floor).")
    parser.add_argument("--no-uncertainty", dest="no_uncertainty", action="store_true",
                        help="Skip the uncertainty maps (per-voxel Fisher standard error of every "
                             "continuous map, written to toolbox_report/uncertainty_maps/). ON by "
                             "default; they cost a small fraction of the fit.")
    parser.add_argument("--disable-iso-resolve", dest="disable_iso_resolve", action="store_true",
                        help="Disable the Stage D final constrained iso fraction re-solve. It is "
                             "ON by default: for every voxel it re-estimates the compartment "
                             "fractions by NNLS over [detected fibers | fixed iso centroids] on "
                             "the Rician-corrected signal, replacing the over-complete Stage A "
                             "spectrum (which mis-bins RF/HF/WF). Pass this to keep the raw "
                             "Stage A fractions.")
    parser.add_argument("--disable-stagec", dest="disable_stagec", action="store_true",
                        help="Disable the Stage C joint mono-fiber re-solve (Plan A / bias #1 "
                             "fix). It is ON by default: for single-fiber voxels it re-fits the "
                             "fiber tensor (AD,RD) and all compartment fractions jointly on a "
                             "reduced [fiber|iso] dictionary, correcting the fiber_fraction "
                             "inflation / restricted collapse / AD under-estimation of the raw "
                             "two-stage output. Pass this to keep the raw Stage A/B estimates.")
    parser.add_argument("--disable-conc-modulation", dest="disable_conc_modulation",
                        action="store_true",
                        help="Disable the per-voxel angular-concentration modulation of "
                             "lambda_aniso (Plan A). It is ON by default: it re-solves low-"
                             "concentration (diffuse-leakage) voxels with boosted anisotropic "
                             "regularization to suppress isotropic->fiber_fraction leakage, "
                             "while sparing concentrated (genuine-fiber, incl. crossing) voxels. "
                             "Pass this to fall back to a single unmodulated Stage A solve.")
    parser.add_argument("--n-calibration-voxels", type=int, dest="n_calibration_voxels", default=1000,
                        help="Number of brain-mask voxels sampled for data-driven calibration and "
                             "n_iso selection. Default: 1000 (matches DBSI_Adaptive.fit() default; "
                             "raised from 500 in v1.3.1 — a seed sweep showed lambda_aniso is "
                             "sample-invariant but lambda_iso still jitters by 2-3 grid nodes at "
                             "500 voxels).")
    parser.add_argument("--n-iso-method", choices=["quadrature", "bootstrap", "svd_floor", "fixed"],
                        dest="n_iso_method", default="quadrature",
                        help="How to set n_iso when --n-iso is not given. 'quadrature' (default "
                             "since 1.3.9): fixed at 6 by grid convergence -- n_iso is a "
                             "discretisation parameter, not a model order. 'bootstrap': "
                             "bias-variance selection per dataset (pre-1.3.9 default). "
                             "'svd_floor': SVD information limit + empirical floor. "
                             "'fixed': legacy n_iso=31.")
    parser.add_argument("--n-bootstrap", type=int, dest="n_bootstrap", default=50,
                        help="Noise replicates per voxel for the bootstrap n_iso method. Default: 50.")
    parser.add_argument("--force-n-iso", type=int, choices=[2, 3], default=None,
                        help="Override automatic isotropic-model selection (2 or 3)")
    parser.add_argument("--disable-direction-refinement", action="store_true",
                        dest="disable_direction_refinement",
                        help="Disable MRDS-lite two-level cone refinement of Stage A's dominant "
                             "fiber direction (enabled by default). Refinement closes most of the "
                             "AD accuracy gap caused by the fixed Stage A grid's angular "
                             "discretisation, at negligible added cost — see "
                             "core.solvers module docstring 'MRDS-LITE'. Disable only for "
                             "reproducing pre-refinement results or debugging.")
    parser.add_argument("--target-angular-resolution", type=float, dest="target_angular_resolution_deg",
                        default=None,
                        help="Desired final angular precision (degrees) for direction refinement. "
                             f"Default: {_D['target_angular_resolution_deg']}. Ignored if "
                             "--disable-direction-refinement is set.")
    args = parser.parse_args()
    if args.protocol_calibration:
        _fissati = [k for k in ('n_iso', 'lambda_aniso', 'lambda_iso', 'n_dirs') + _DICTIONARY_OPTS
                    if getattr(args, k) is not None]
        if _fissati:
            parser.error(f"--protocol-calibration fissa gia {_fissati}: non passarli")
    elif args.skip_calibration and (args.lambda_aniso is None or args.lambda_iso is None):
        print("WARNING: --skip-calibration without --lambda-aniso AND --lambda-iso (or "
              "--protocol-calibration): the missing lambda falls back to a fixed built-in "
              "value that was not derived for these data.")
    os.makedirs(args.out, exist_ok=True)

    print("\nDBSI PIPELINE - Adaptive Version (v3, hybrid two-stage)\n")

    data, affine, bvals, bvecs, mask = load_data(
        args.dwi, args.bval, args.bvec, args.mask, verbose=True
    )

    kw = model_kwargs(args)
    if args.protocol_calibration:
        model = DBSI_Adaptive.from_calibration(args.protocol_calibration, **kw)
        print(f"Protocol calibration: {args.protocol_calibration}")
    else:
        model = DBSI_Adaptive(**kw)

    results, model_mode = model.fit(
        data, bvals, bvecs, mask,
        run_calibration=not args.skip_calibration,
        n_calibration_voxels=args.n_calibration_voxels,
        n_iso_method=args.n_iso_method,
        n_bootstrap=args.n_bootstrap,
        noise_map=_load_noise(args.noise, data.shape[:3], affine),
    )

    names = DBSI_Adaptive.output_map_names(model_mode)

    print("\nSaving outputs...")
    saved = save_output_maps(results, names, affine, args.out, model=model)
    skipped = [n for n in names if n not in saved]
    print(f"  {len(saved)} channel maps -> {args.out}/NN_<channel>.nii.gz")
    print(f"  fiber-tensor validity mask -> {args.out}/fiber_valid.nii.gz")
    print(f"  toolbox report (run report + dictionary) -> {args.out}/toolbox_report/")
    if getattr(model, 'uncertainty_', None) is not None:
        print(f"  uncertainty maps (standard errors) -> {args.out}/toolbox_report/uncertainty_maps/")
    print(f"  Skipped {len(skipped)} channels (invalid in {model_mode}-ISO mode "
          f"or exact duplicates): {', '.join(skipped)}")

    print("\nDone!")

def _load_noise(path, shape3d, affine):
    """The --noise map as an array on the data grid, or None."""
    if not path:
        return None
    import nibabel as nib
    import numpy as np
    img = nib.load(path)
    arr = np.asarray(img.dataobj, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.shape != tuple(shape3d):
        raise SystemExit(f"--noise {path}: grid {arr.shape} differs from the DWI {tuple(shape3d)}")
    if not np.allclose(img.affine, affine, atol=1e-3):
        raise SystemExit(f"--noise {path}: orientation matrix differs from the DWI (same grid "
                         f"size, different voxel-to-world mapping): not the map of this DWI")
    return arr


def model_kwargs(args):
    """Constructor kwargs from the parsed CLI: only what the user set, plus the
    on/off switches. Anything left out takes the constructor's own default."""
    kw = dict(force_n_iso=args.force_n_iso,
              enable_direction_refinement=not args.disable_direction_refinement,
              lambda_aniso_conc_mod=not args.disable_conc_modulation,
              stagec_refine=not args.disable_stagec,
              iso_resolve=not args.disable_iso_resolve,
              uncertainty=not getattr(args, 'no_uncertainty', False))
    if getattr(args, 'crossing_dir_refine', False):
        kw['crossing_dir_refine'] = True
    opt = ('min_weight_fraction', 'target_angular_resolution_deg', 'fiber_detection_threshold',
           'crossing_ff_kappa', 'two_vs_one_threshold')
    if not args.protocol_calibration:
        opt = opt + ('n_iso', 'lambda_aniso', 'lambda_iso', 'n_dirs') + _DICTIONARY_OPTS
    for k in opt:
        if getattr(args, k) is not None:
            kw[k] = getattr(args, k)
    return kw


if __name__ == "__main__":
    main()
