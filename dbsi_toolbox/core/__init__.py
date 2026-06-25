"""
DBSI Core Module — Basis Functions and Solvers (v3, Hybrid Two-Stage)

Contains the mathematical core of the v3 DBSI implementation:
- Stage A detection dictionary (exhaustive direction x AD/RD-pair grid,
  used ONLY to identify dominant fiber direction(s) via NNLS + heavy
  sparsity regularization)
- Stage B closed-form (AD, RD) estimation conditioned on Stage A's
  selected direction(s)
- Isotropic spectrum generation and weighted-centroid extraction
  (unchanged from v1/v2)
- Fiber direction generation (hemisphere)
- NNLS solver with coordinate descent
- Decoupled (lambda_aniso, lambda_iso) regularization for Stage A
- Fiber FA computation

Note on v2 single-stage centroid extraction
----------------------------------------------
`compute_aniso_centroids` (v2) is no longer used: synthetic recovery
validation showed it is not numerically identifiable (median AD/RD
relative errors 20%-150%+ across all tested dictionary densities). It is
kept in solvers.py for backward compatibility but NOT exported here.
Use `select_dominant_directions` (Stage A) + `estimate_AD_RD_conditioned`
(Stage B) instead.

Note on v1 single-(AD,RD) design matrix and non-linear Step 2
-----------------------------------------------------------------
`build_design_matrix` (v1, orientation-only) and
`step2_refine_diffusivities[_adaptive]` (v1, non-linear grid search) are
kept for backward compatibility / regression comparison but are NOT used
by `DBSI_Adaptive` in v3 and are NOT exported here.
"""

from .basis import (
    build_design_matrix,                 # v1, deprecated — kept for compatibility
    build_design_matrix_exhaustive,      # v3 Stage A detection dictionary
    build_isotropic_dictionary,
    generate_exhaustive_diffusivity_pairs,  # v3 Stage A
    generate_isotropic_grid,
    generate_fibonacci_sphere_hemisphere,
    generate_fibonacci_sphere,
)

from .solvers import (
    nnls_coordinate_descent,
    compute_regularization_matrix,       # v3 Stage A regularization
    select_dominant_directions,          # v3 Stage A direction detection
    estimate_AD_RD_conditioned,          # v3 Stage B diffusivity estimation
    compute_weighted_centroids,
    compute_fiber_fa,
)

__all__ = [
    "build_design_matrix",
    "build_design_matrix_exhaustive",
    "build_isotropic_dictionary",
    "generate_exhaustive_diffusivity_pairs",
    "generate_isotropic_grid",
    "generate_fibonacci_sphere_hemisphere",
    "generate_fibonacci_sphere",
    "nnls_coordinate_descent",
    "compute_regularization_matrix",
    "select_dominant_directions",
    "estimate_AD_RD_conditioned",
    "compute_weighted_centroids",
    "compute_fiber_fa",
]
