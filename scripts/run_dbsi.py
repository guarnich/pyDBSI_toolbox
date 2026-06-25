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
    parser.add_argument("--anisotropy-ratio", type=float, dest="anisotropy_ratio", default=1.15,
                        help="Minimum AD/RD ratio for admissible Stage A pairs. Default: 1.15.")
    parser.add_argument("--min-weight-fraction", type=float, dest="min_weight_fraction", default=0.05,
                        help="Stage A direction-selection threshold. Default: 0.05.")
    parser.add_argument("--force-n-iso", type=int, choices=[2, 3], default=None,
                        help="Override automatic isotropic-model selection (2 or 3)")
    parser.add_argument("--compute-r2", action="store_true",
                        help="Compute fit quality check (R2 and RMSE)")

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
        force_n_iso=args.force_n_iso
    )

    results, model_mode = model.fit(
        data, bvals, bvecs, mask,
        run_calibration=not args.skip_calibration
    )

    names = DBSI_Adaptive.output_map_names(model_mode)

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

    print("\nDone!")

if __name__ == "__main__":
    main()
