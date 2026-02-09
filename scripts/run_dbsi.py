#!/usr/bin/env python
"""
DBSI Fusion CLI

Outputs (8 channels):
- fiber_fraction.nii.gz      : Apparent axonal density
- restricted_fraction.nii.gz : Cellularity (ADC ≤ 0.3)
- hindered_fraction.nii.gz   : Edema (0.3 < ADC ≤ 3.0)
- water_fraction.nii.gz      : CSF (ADC > 3.0)
- axial_diffusivity.nii.gz   : Fiber AD
- radial_diffusivity.nii.gz  : Fiber RD
- fiber_fa_intrinsic.nii.gz  : Fiber FA
- mean_iso_adc.nii.gz        : Mean isotropic ADC
"""

import argparse
import os
import sys
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import load_data
from model import DBSI_Fused


def main():
    parser = argparse.ArgumentParser(description="DBSI Pipeline")
    parser.add_argument("--dwi", required=True)
    parser.add_argument("--bval", required=True)
    parser.add_argument("--bvec", required=True)
    parser.add_argument("--mask", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--n-iso", type=int, default=None)
    parser.add_argument("--lambda", type=float, dest="reg_lambda", default=None)
    parser.add_argument("--n-dirs", type=int, default=100)
    parser.add_argument("--no-step2", action="store_true")
    
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    print("\nDBSI PIPELINE - Balanced Final Version\n")
    
    data, affine, bvals, bvecs, mask = load_data(
        args.dwi, args.bval, args.bvec, args.mask, verbose=True
    )
    
    model = DBSI_Fused(
        n_iso=args.n_iso,
        reg_lambda=args.reg_lambda,
        enable_step2=not args.no_step2,
        n_dirs=args.n_dirs
    )
    
    results = model.fit(
        data, bvals, bvecs, mask,
        run_calibration=not args.skip_calibration
    )
    
    names = [
        'fiber_fraction', 
        'restricted_fraction', 
        'hindered_fraction',
        'water_fraction', 
        'axial_diffusivity', 
        'radial_diffusivity',
        'fiber_fa',
        'mean_iso_adc',
        'axial_diffusivity_linear',
        'radial_diffusivity_linear',
        'adaptive_threshold'
    ]
    
    print("\nSaving outputs...")
    for i, name in enumerate(names):
        fname = os.path.join(args.out, f"dbsi_{name}.nii.gz")
        nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine), fname)
        print(f"  {fname}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
