import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid, 
                        D_ax=1.7e-3, D_rad=0.3e-3):
    """
    Constructs the DBSI Design Matrix A.
    A = [Anisotropic_Basis | Isotropic_Basis]
    """
    N_meas = len(bvals)
    N_dirs = len(fiber_dirs)
    N_iso = len(iso_grid)
    
    A = np.zeros((N_meas, N_dirs + N_iso), dtype=np.float64)
    
    # 1. Anisotropic Basis (Fibers)
    for j in range(N_dirs):
        f_dir = fiber_dirs[j]
        for i in range(N_meas):
            # cos(theta) between gradient and fiber direction
            cos_t = bvecs[i, 0]*f_dir[0] + bvecs[i, 1]*f_dir[1] + bvecs[i, 2]*f_dir[2]
            
            # Cylinder model response
            D_app = D_rad + (D_ax - D_rad) * cos_t * cos_t
            A[i, j] = np.exp(-bvals[i] * D_app)
            
    # 2. Isotropic Basis (Spectrum)
    for k in range(N_iso):
        D_iso = iso_grid[k]
        for i in range(N_meas):
            A[i, N_dirs + k] = np.exp(-bvals[i] * D_iso)
            
    return A

def generate_fibonacci_sphere(samples=150):
    """Generates uniform direction vectors on a unit sphere."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)