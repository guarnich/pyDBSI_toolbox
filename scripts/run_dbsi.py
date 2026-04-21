#!/usr/bin/env python
"""
DBSI Adaptive CLI

Outputs (11 channels) dynamically adapted based on acquisition scheme (2-ISO or 3-ISO).
"""

import argparse
import os
import sys
import nibabel as nib
import numpy as np

from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox import load_data

def main():
    parser = argparse.ArgumentParser(description="DBSI Adaptive Pipeline")
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
    parser.add_argument("--force-n-iso", type=int, choices=[2, 3], default=None, 
                        help="Override automatic model selection (2 or 3)")
    
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    print("\nDBSI PIPELINE - Adaptive Version\n")
    
    data, affine, bvals, bvecs, mask = load_data(
        args.dwi, args.bval, args.bvec, args.mask, verbose=True
    )
    
    model = DBSI_Adaptive(
        n_iso=args.n_iso,
        reg_lambda=args.reg_lambda,
        enable_step2=not args.no_step2,
        n_dirs=args.n_dirs,
        force_n_iso=args.force_n_iso
    )
    
    # Il nuovo modello restituisce sia i risultati che il model_mode (2 o 3)
    results, model_mode = model.fit(
        data, bvals, bvecs, mask,
        run_calibration=not args.skip_calibration
    )
    
    # Ricaviamo i nomi dinamicamente dal modello in base ai canali calcolati
    names = DBSI_Adaptive.output_map_names(model_mode)
    
    print("\nSaving outputs...")
    for i, name in enumerate(names):
        fname = os.path.join(args.out, f"dbsi_{name}.nii.gz")
        # Salva solo i canali che non terminano esplicitamente per _NaN
        if not name.endswith('_NaN'):
            nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine), fname)
            print(f"  {fname}")
        else:
            print(f"  Skipped {name} (not estimated in {model_mode}-ISO mode)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()