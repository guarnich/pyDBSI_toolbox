#!/usr/bin/env python
"""
DBSI Adaptive CLI — v3 (Hybrid Two-Stage Architecture)

Outputs (11 channels) dynamically adapted based on acquisition scheme (2-ISO or 3-ISO).

CHANGES FROM v2
-----------------
- Stage A's (AD,RD) grid (`--n-ad`, `--n-rd`, `--anisotropy-ratio`)
  defaults to a coarse 3x3 grid regardless of protocol, since Stage A
  only needs to detect fiber direction, not estimate diffusivity (Stage
  B does that, via closed-form regression with no free dictionary
  parameters). See dbsi_toolbox.model_Niso_adaptive_ff_thr module
  docstring.
- New: `--min-weight-fraction` controls Stage A's direction-selection
  threshold (see dbsi_toolbox.core.solvers.select_dominant_directions).
- `--n-dirs` still defaults to protocol-derived (autoconfigured) sizing.
- New: `--calibration-method {data_driven,monte_carlo}` selects how
  (lambda_aniso, lambda_iso) are determined. `data_driven` (default)
  uses GCV + the discrepancy principle on a sample of this dataset's
  own voxels (fast, no tissue-fraction priors). `monte_carlo` falls
  back to the legacy grid search over 14 literature-derived tissue
  scenarios. See dbsi_toolbox.calibration module docstring.
- New: `--mc-crosscheck` runs the Monte Carlo tissue-scenario check
  against whichever (lambda_aniso, lambda_iso) was selected, WITHOUT
  changing it — a sanity check, recommended at least once per new
  protocol/dataset type.
"""

import argparse
import os
import sys
import nibabel as nib
import numpy as np

from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox import load_data

def main():
    parser = argparse.ArgumentParser(description="DBSI Adaptive Pipeline (v3, hybrid two-stage)")
    parser.add_argument("--dwi", required=True)
    parser.add_argument("--bval", required=True)
    parser.add_argument("--bvec", required=True)
    parser.add_argument("--mask", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--n-iso", type=int, default=None)
    parser.add_argument("--lambda-aniso", type=float, dest="lambda_aniso", default=None,
                        help="Stage A regularization strength for the anisotropic detection block.")
    parser.add_argument("--lambda-iso", type=float, dest="lambda_iso", default=None,
                        help="Stage A regularization strength for the isotropic spectrum block.")
    parser.add_argument("--n-dirs", type=int, default=None,
                        help="Number of hemisphere directions for Stage A. Default: protocol-derived.")
    parser.add_argument("--n-ad", type=int, default=3,
                        help="Number of AD grid steps for Stage A's detection dictionary. Default: 3 (deliberately coarse).")
    parser.add_argument("--n-rd", type=int, default=3,
                        help="Number of RD grid steps for Stage A's detection dictionary. Default: 3 (deliberately coarse).")
    parser.add_argument("--anisotropy-ratio", type=float, dest="anisotropy_ratio", default=2.0,
                        help="Minimum AD/RD ratio for admissible Stage A pairs. Default: 2.0 "
                             "(raised from 1.15: drops the near-isotropic ratio-1.83 grid column "
                             "that leaks isotropic signal into fiber_fraction; safe band [2.0, 2.16]).")
    parser.add_argument("--min-weight-fraction", type=float, dest="min_weight_fraction", default=0.05,
                        help="Stage A direction-selection threshold. Default: 0.05.")
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
    parser.add_argument("--calibration-method", choices=["data_driven", "monte_carlo"],
                        dest="calibration_method", default="data_driven",
                        help="Method to determine (lambda_aniso, lambda_iso). Default: data_driven "
                             "(GCV + discrepancy principle, no tissue-fraction priors). "
                             "'monte_carlo' falls back to the legacy 14-scenario grid search.")
    parser.add_argument("--n-calibration-voxels", type=int, dest="n_calibration_voxels", default=500,
                        help="Number of brain-mask voxels sampled for data-driven calibration and "
                             "n_iso selection. Default: 500 (matches DBSI_Adaptive.fit() default; "
                             "previously inconsistent at 200 here — fixed).")
    parser.add_argument("--n-iso-method", choices=["bootstrap", "svd_floor", "fixed"],
                        dest="n_iso_method", default="bootstrap",
                        help="How to select n_iso when --n-iso is not given. 'bootstrap' (default): "
                             "bias-variance minimisation on real sampled voxels. 'svd_floor': SVD "
                             "information limit + empirical floor. 'fixed': legacy n_iso=31.")
    parser.add_argument("--n-bootstrap", type=int, dest="n_bootstrap", default=50,
                        help="Noise replicates per voxel for the bootstrap n_iso method. Default: 50.")
    parser.add_argument("--mc-crosscheck", action="store_true", dest="run_mc_crosscheck",
                        help="Run the Monte Carlo tissue-scenario cross-check on the selected "
                             "lambda pair (does not change it; prints a diagnostic report).")
    parser.add_argument("--mc-crosscheck-n-mc", type=int, dest="mc_crosscheck_n_mc", default=200,
                        help="MC samples per scenario for the cross-check report. Default: 200.")
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
                        default=1.0,
                        help="Desired final angular precision (degrees) for direction refinement. "
                             "Default: 1.0. Ignored if --disable-direction-refinement is set.")
    parser.add_argument("--compute-r2", action="store_true",
                        help="Compute fit quality check (R2 and RMSE)")
    parser.add_argument("--compute-transition-confidence", action="store_true",
                        dest="compute_transition_confidence",
                        help="Compute RES/HIN and HIN/WAT transition-zone confidence maps "
                             "(does not correct fractions, only flags proximity to a known "
                             "low-confidence zone — see transition_confidence.py module "
                             "docstring; NOT YET validated on real data).")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("\nDBSI PIPELINE - Adaptive Version (v3, hybrid two-stage)\n")

    data, affine, bvals, bvecs, mask = load_data(
        args.dwi, args.bval, args.bvec, args.mask, verbose=True
    )

    model = DBSI_Adaptive(
        n_iso=args.n_iso,
        lambda_aniso=args.lambda_aniso,
        lambda_iso=args.lambda_iso,
        n_dirs=args.n_dirs,
        n_ad=args.n_ad,
        n_rd=args.n_rd,
        anisotropy_ratio=args.anisotropy_ratio,
        min_weight_fraction=args.min_weight_fraction,
        force_n_iso=args.force_n_iso,
        enable_direction_refinement=not args.disable_direction_refinement,
        target_angular_resolution_deg=args.target_angular_resolution_deg,
        lambda_aniso_conc_mod=not args.disable_conc_modulation,
        stagec_refine=not args.disable_stagec,
        iso_resolve=not args.disable_iso_resolve,
    )

    results, model_mode = model.fit(
        data, bvals, bvecs, mask,
        run_calibration=not args.skip_calibration,
        calibration_method=args.calibration_method,
        n_calibration_voxels=args.n_calibration_voxels,
        n_iso_method=args.n_iso_method,
        n_bootstrap=args.n_bootstrap,
        run_mc_crosscheck=args.run_mc_crosscheck,
        mc_crosscheck_n_mc=args.mc_crosscheck_n_mc,
    )

    names = DBSI_Adaptive.output_map_names(model_mode)

    if model.mc_crosscheck_report_ is not None:
        print(f"\nMonte Carlo cross-check composite loss: "
              f"{model.mc_crosscheck_report_['composite']:.4f}")

    print("\nSaving outputs...")
    for i, name in enumerate(names):
        fname = os.path.join(args.out, f"dbsi_{name}.nii.gz")
        if not name.endswith('_NaN'):
            nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine), fname)
            print(f"  {fname}")
        else:
            print(f"  Skipped {name} (not estimated in {model_mode}-ISO mode)")

    if args.compute_r2:
        print("\nComputing fit quality (R2 and RMSE)...")
        from dbsi_toolbox.fit_quality import compute_fit_quality, save_fit_quality
        r2, rmse = compute_fit_quality(
            data, bvals, bvecs, mask, results, model_mode, n_dirs=model.n_dirs
        )
        save_fit_quality(r2, rmse, affine, args.out)
        print("Fit quality maps saved.")

    if args.compute_transition_confidence:
        print("\nComputing transition-zone confidence maps...")
        print("NOTE: these flag proximity to a known systematic bias zone near the")
        print("compartment thresholds (see project methodological supplement); they")
        print("do not correct fractions and have not yet been validated on real data.")
        from dbsi_toolbox.transition_confidence import compute_transition_confidence, save_transition_confidence
        conf_res, conf_wat = compute_transition_confidence(results, model_mode)
        save_transition_confidence(conf_res, conf_wat, affine, args.out)
        print("Transition-confidence maps saved.")

    print("\nDone!")

if __name__ == "__main__":
    main()
