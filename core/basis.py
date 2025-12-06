"""
DBSI Basis Functions

Implements:
- Fibonacci sphere for fiber directions (hemisphere)
- Design matrix construction with cylinder model
"""

import numpy as np
from numba import njit


def generate_fibonacci_sphere_hemisphere(n_points):
    """
    Generate uniformly distributed points on hemisphere using Fibonacci spiral.
    
    Only returns points with z >= 0 to exploit antipodal symmetry.
    """
    # Generate more points and filter to hemisphere
    n_full = n_points * 2 + 10
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_full)
    
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_full)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Keep only hemisphere (z >= 0)
    mask = z >= 0
    x, y, z = x[mask], y[mask], z[mask]
    
    # Take exactly n_points
    dirs = np.column_stack([x[:n_points], y[:n_points], z[:n_points]])
    
    # Normalize
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms
    
    return dirs.astype(np.float64)


def generate_fibonacci_sphere(n_points):
    """Full sphere (deprecated, kept for compatibility)."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.column_stack([x, y, z]).astype(np.float64)


@njit(cache=True, fastmath=True)
def build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid, ad=1.5e-3, rd=0.4e-3):
    """
    Build DBSI design matrix.
    
    Parameters
    ----------
    bvals : array (N,)
        B-values in s/mm²
    bvecs : array (N, 3)
        Normalized gradient directions
    fiber_dirs : array (M, 3)
        Fiber direction candidates
    iso_grid : array (L,)
        Isotropic ADC values
    ad : float
        Axial diffusivity for fiber basis (default 1.5e-3)
    rd : float
        Radial diffusivity for fiber basis (default 0.4e-3)
        
    Returns
    -------
    A : array (N, M+L)
        Design matrix
    """
    n_meas = len(bvals)
    n_dirs = len(fiber_dirs)
    n_iso = len(iso_grid)
    
    A = np.zeros((n_meas, n_dirs + n_iso), dtype=np.float64)
    
    # Anisotropic columns (cylinder model)
    for j in range(n_dirs):
        fdir = fiber_dirs[j]
        for i in range(n_meas):
            b = bvals[i]
            g = bvecs[i]
            
            # cos(theta) = g · fdir
            cos_t = g[0]*fdir[0] + g[1]*fdir[1] + g[2]*fdir[2]
            
            # D_apparent = RD + (AD - RD) * cos²θ
            D_app = rd + (ad - rd) * cos_t * cos_t
            
            A[i, j] = np.exp(-b * D_app)
    
    # Isotropic columns
    for j in range(n_iso):
        D_iso = iso_grid[j]
        for i in range(n_meas):
            A[i, n_dirs + j] = np.exp(-bvals[i] * D_iso)
    
    return A
