"""
DBSI Core: Basis Functions and Design Matrix Construction

Corrected Version:
- Extended isotropic range to 4.0e-3 for proper free water capture
- Hemisphere constraint for fiber directions (antipodal symmetry)
- Improved numerical stability
"""

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid, 
                        D_ax=1.7e-3, D_rad=0.3e-3):
    """
    Constructs the DBSI Design Matrix A.
    
    The design matrix A models the diffusion-weighted signal as:
    S(b,g) = Σ_i f_i * exp(-b * D_app_i(g)) + Σ_j f_j * exp(-b * D_iso_j)
    
    where D_app = D_rad + (D_ax - D_rad) * cos²(θ) for cylinder model.
    
    Parameters
    ----------
    bvals : ndarray (N_meas,)
        B-values in s/mm²
    bvecs : ndarray (N_meas, 3)
        Normalized gradient directions
    fiber_dirs : ndarray (N_dirs, 3)
        Unit vectors for discrete fiber directions (hemisphere)
    iso_grid : ndarray (N_iso,)
        Discrete ADC values for isotropic spectrum in mm²/s
    D_ax : float
        Initial axial diffusivity for fiber basis (mm²/s)
    D_rad : float
        Initial radial diffusivity for fiber basis (mm²/s)
        
    Returns
    -------
    A : ndarray (N_meas, N_dirs + N_iso)
        Design matrix [Anisotropic_Basis | Isotropic_Basis]
    """
    N_meas = len(bvals)
    N_dirs = len(fiber_dirs)
    N_iso = len(iso_grid)
    
    A = np.zeros((N_meas, N_dirs + N_iso), dtype=np.float64)
    
    # 1. Anisotropic Basis (Fibers) - Cylinder Model
    for j in range(N_dirs):
        f_dir = fiber_dirs[j]
        for i in range(N_meas):
            # cos(theta) between gradient and fiber direction
            cos_t = bvecs[i, 0]*f_dir[0] + bvecs[i, 1]*f_dir[1] + bvecs[i, 2]*f_dir[2]
            
            # Cylinder model: D_app = D_rad + (D_ax - D_rad) * cos²(θ)
            # When gradient parallel to fiber: D_app = D_ax (fast diffusion)
            # When gradient perpendicular: D_app = D_rad (slow diffusion)
            D_app = D_rad + (D_ax - D_rad) * cos_t * cos_t
            A[i, j] = np.exp(-bvals[i] * D_app)
            
    # 2. Isotropic Basis (Spectrum of ADC values)
    for k in range(N_iso):
        D_iso = iso_grid[k]
        for i in range(N_meas):
            A[i, N_dirs + k] = np.exp(-bvals[i] * D_iso)
            
    return A


def generate_fibonacci_sphere_hemisphere(samples=150):
    """
    Generates uniform direction vectors on the UPPER HEMISPHERE.
    
    Uses Fibonacci lattice for optimal coverage, then constrains to z >= 0.
    This accounts for antipodal symmetry in diffusion MRI (g ≡ -g).
    
    Parameters
    ----------
    samples : int
        Number of directions to generate (will produce ~samples directions)
        
    Returns
    -------
    points : ndarray (N, 3)
        Unit vectors on the upper hemisphere
    """
    # Generate twice as many points on full sphere, then take hemisphere
    full_samples = samples * 2
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    
    for i in range(full_samples):
        y = 1 - (i / float(full_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        # Only keep upper hemisphere (z >= 0) or flip to upper
        if z < 0:
            # Flip to upper hemisphere
            x, y, z = -x, -y, -z
            
        # Add if not already present (avoid duplicates at equator)
        point = np.array([x, y, z])
        
        # Normalize for safety
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            point = point / norm
            
        points.append(point)
    
    # Remove near-duplicates
    points = np.array(points)
    unique_points = [points[0]]
    
    for i in range(1, len(points)):
        is_duplicate = False
        for up in unique_points:
            if np.abs(np.dot(points[i], up)) > 0.99:  # ~8 degrees tolerance
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(points[i])
            
        if len(unique_points) >= samples:
            break
            
    return np.array(unique_points[:samples])


def generate_fibonacci_sphere(samples=150):
    """
    Generates uniform direction vectors on a unit sphere.
    
    DEPRECATED: Use generate_fibonacci_sphere_hemisphere() for DBSI
    to properly handle antipodal symmetry and improve matrix conditioning.
    
    This function is kept for backward compatibility.
    
    Parameters
    ----------
    samples : int
        Number of points to generate
        
    Returns
    -------
    points : ndarray (samples, 3)
        Unit vectors distributed on the sphere
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
        
    return np.array(points)
