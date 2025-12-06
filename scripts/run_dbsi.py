import argparse
import sys
import os
import nibabel as nib
import numpy as np

# Add package to path if not installed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbsi.utils.tools import load_data
from dbsi.model import DBSI_Fused

def main():
    parser = argparse.ArgumentParser(description="DBSI Fusion CLI: Automated Pipeline")
    
    # Input/Output
    parser.add_argument("--dwi", required=True, help="Path to 4D DWI NIfTI file")
    parser.add_argument("--bval", required=True, help="Path to b-values file")
    parser.add_argument("--bvec", required=True, help="Path to b-vectors file")
    parser.add_argument("--mask", required=False, help="Path to brain mask NIfTI (optional)")
    parser.add_argument("--out", required=True, help="Output directory")
    
    # Configuration
    parser.add_argument("--skip-calibration", action="store_true", 
                        help="Skip Monte Carlo calibration and use default parameters")
    parser.add_argument("--n-iso", type=int, default=None, 
                        help="Manual number of isotropic bases")
    parser.add_argument("--lambda", type=float, dest="reg_lambda", default=None, 
                        help="Manual regularization lambda")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print("\n" + "="*50)
    print("       DBSI FUSION PIPELINE")
    print("="*50)
    
    # 1. Load Data (includes protocol summary)
    print("\n[Step 1] Loading Data...")
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
    # Pass manual parameters if provided
    model = DBSI_Fused(n_iso=args.n_iso, reg_lambda=args.reg_lambda)
    
    # 3. Fit (handles SNR, Calibration, Fitting)
    print("\n[Step 2] Running Analysis...")
    results = model.fit(data, bvals, bvecs, mask, calibrate=not args.skip_calibration)
    
    # 4. Save Results
    print("\n[Step 3] Saving Parameter Maps...")
    map_names = ['fiber_fraction', 'restricted_fraction', 'hindered_fraction', 
                 'water_fraction', 'axial_diffusivity', 'radial_diffusivity']
    
    for i, name in enumerate(map_names):
        fname = os.path.join(args.out, f"dbsi_{name}.nii.gz")
        # Ensure float32 for storage
        nib.save(nib.Nifti1Image(results[..., i].astype(np.float32), affine), fname)
        print(f"   Saved: {fname}")
        
    print("\n[Complete] All files saved successfully.\n")

if __name__ == "__main__":
    main()