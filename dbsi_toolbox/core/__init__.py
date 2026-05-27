"""
DBSI Core Module - Basis Functions and Solvers

Contains the mathematical core of the DBSI implementation:
- Design matrix construction (cylinder model)
- Fiber direction generation (hemisphere)
- NNLS solver with coordinate descent
- Fiber FA computation

Note on step2_refine_diffusivities / step2_refine_diffusivities_adaptive
------------------------------------------------------------------------
These functions are deprecated and no longer used by the pipeline.
AD/RD refinement in the fitting kernels is handled by the private
functions _refine_AD_RD_2iso and _refine_AD_RD_3iso defined in
model_Niso_adaptive_ff_thr.py. The legacy functions are kept in
solvers.py for backward compatibility but are NOT exported here to
avoid misleading downstream code into using them.
"""

from .basis import (
    build_design_matrix,
    generate_fibonacci_sphere_hemisphere,
    generate_fibonacci_sphere,
)

from .solvers import (
    nnls_coordinate_descent,
    compute_weighted_centroids,
    compute_fiber_fa,
)

__all__ = [
    "build_design_matrix",
    "generate_fibonacci_sphere_hemisphere",
    "generate_fibonacci_sphere",
    "nnls_coordinate_descent",
    "compute_weighted_centroids",
    "compute_fiber_fa",
]
