#!/usr/bin/env python
"""
DBSI Fusion CLI: Automated Pipeline

Corrected Version - December 2025

Outputs:
- dbsi_fiber_fraction.nii.gz     : Apparent axonal density
- dbsi_restricted_fraction.nii.gz : Cellularity/inflammation marker (ADC ≤ 0.3)
- dbsi_hindered_fraction.nii.gz  : Edema/tissue loss (0.3 < ADC ≤ 3.0)
- dbsi_water_fraction.nii.gz     : CSF contamination (ADC > 3.0)
- dbsi_axial_diffusivity.nii.gz  : Fiber AD (mm²/s)
- dbsi_radial_diffusivity.nii.gz : Fiber RD (mm²/s)
- dbsi_fiber_fa.nii.gz           : Fiber fractional anisotropy
"""

import argparse
import sys
import os
import nibabel as nib
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import load_data
from model import DBSI_Fused


def main():
    parser = argparse.ArgumentParser(
        description="DBSI Fusion CLI: Automated Pipeline (Corrected Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-calibration
  python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec --out results/
  
  # Skip calibration and use manual parameters
  python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec --out results/ \\
                     --skip-calibration --n-iso 60 --lambda 0.1
  
  # With brain mask
  python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec \\
                     --mask mask.nii.gz --out results/
        """
    )
    
    parser.add_argument("--dwi", required=True, 
                        help="Path to 4D DWI NIfTI file")
    parser.add_argument("--bval", required=True, 
                        help="Path to b-values file")
    parser.add_argument("--bvec", required=True, 
                        help="Path to b-vectors file")
    parser.add_argument("--mask", required=False, 
                        help="Path to brain mask NIfTI (optional)")
    parser.add_argument("--out", required=True, 
                        help="Output directory")
    
    parser.add_argument("--skip-calibration", action="store_true", 
                        help="Skip Monte Carlo calibration and use default/manual parameters")
    parser.add_argument("--n-iso", type=int, default=None, 
                        help="Number of isotropic bases (default: auto-calibrated)")
    parser.add_argument("--lambda", type=float, dest="reg_lambda", default=None, 
                        help="Regularization lambda (default: auto-calibrated)")
    parser.add_argument("--n-dirs", type=int, default=100,
                        help="Number of fiber directions on hemisphere (default: 100)")
    parser.add_argument("--no-step2", action="store_true",
                        help="Disable Step 2 diffusivity refinement")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print("\n" + "="*60)
    print("       DBSI FUSION PIPELINE (Corrected Version)")
    print("="*60)
    print()
    print("Corrections applied vs original:")
    print("  • ADC thresholds: 0.3/3.0 µm²/ms (literature standard)")
    print("  • Isotropic range: Extended to 4.0 µm²/ms")
    print("  • Fiber directions: Hemisphere (antipodal symmetry)")
    print("  • Matrix conditioning: Regularized for stability")
    print("  • Added Fiber FA output")
    print()
    
    # 1. Load Data
    print("[Step 1] Loading Data...")
    try:
        data, affine, bvals, bvecs, mask = load_data(
            args.dwi, args.bval, args.bvec, args.mask, verbose=True
        )
        print(f"   Volume shape: {data.shape}")
        print(f"   Voxels to process: {np.sum(mask):,}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # 2. Initialize Model
    model = DBSI_Fused(
        n_iso=args.n_iso, 
        reg_lambda=args.reg_lambda,
        enable_step2=not args.no_step2,
        n_dirs=args.n_dirs
    )
    
    # 3. Fit
    print("\n[Step 2] Running Analysis...")
    results = model.fit(
        data, bvals, bvecs, mask, 
        run_calibration=not args.skip_calibration
    )
    
    # 4. Save Results
    print("\n[Step 3] Saving Parameter Maps...")
    
    # CORRECTED: 7 output maps including Fiber FA
    map_names = [
        'fiber_fraction', 
        'restricted_fraction', 
        'hindered_fraction', 
        'water_fraction', 
        'axial_diffusivity', 
        'radial_diffusivity',
        'fiber_fa'  # NEW
    ]
    
    map_descriptions = [
        'Apparent axonal density',
        'Cellularity marker (ADC ≤ 0.3 µm²/ms)',
        'Edema/tissue loss (0.3 < ADC ≤ 3.0 µm²/ms)',
        'CSF contamination (ADC > 3.0 µm²/ms)',
        'Fiber axial diffusivity (mm²/s)',
        'Fiber radial diffusivity (mm²/s)',
        'Fiber fractional anisotropy'
    ]
    
    for i, (name, desc) in enumerate(zip(map_names, map_descriptions)):
        fname = os.path.join(args.out, f"dbsi_{name}.nii.gz")
        nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine), fname)
        print(f"   Saved: {fname}")
        print(f"          ({desc})")
        
    print("\n[Complete] All files saved successfully.")
    print()
    print("Output interpretation:")
    print("  • Restricted fraction ↑ → Increased cellularity (inflammation)")
    print("  • Hindered fraction ↑ → Edema or tissue loss")
    print("  • Water fraction ↑ → CSF contamination")
    print("  • Fiber AD ↓ → Axonal injury")
    print("  • Fiber RD ↑ → Demyelination")
    print()


if __name__ == "__main__":
    main()
