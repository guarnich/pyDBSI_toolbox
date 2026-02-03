"""
DBSI Core Module - Basis Functions and Solvers

Contains the mathematical core of the DBSI implementation:
- Design matrix construction (cylinder model)
- Fiber direction generation (hemisphere)
- NNLS solver with coordinate descent
- Step 2 diffusivity refinement
- Fiber FA computation
"""

# OPTION 1: Absolute imports (RECOMMENDED for GitHub/server)
from dbsi_toolbox.core.basis import (
    build_design_matrix,
    generate_fibonacci_sphere_hemisphere,
    generate_fibonacci_sphere, 
)

from dbsi_toolbox.core.solvers import (
    nnls_coordinate_descent,
    step2_refine_diffusivities,
    step2_refine_diffusivities_adaptive,
    compute_weighted_centroids,
    compute_fiber_fa,
)

__all__ = [
    "build_design_matrix",
    "generate_fibonacci_sphere_hemisphere",
    "generate_fibonacci_sphere",
    "nnls_coordinate_descent",
    "step2_refine_diffusivities",
    "step2_refine_diffusivities_adaptive",
    "compute_weighted_centroids",
    "compute_fiber_fa",
]