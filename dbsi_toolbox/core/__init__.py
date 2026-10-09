"""
DBSI Core Module — Basis Functions and Solvers

The mathematical core used by `DBSI_Adaptive`:
- Stage A detection dictionary (hemisphere directions x (AD, RD) pairs, plus
  the isotropic block) and its regularized NNLS (coordinate descent, decoupled
  lambda_aniso / lambda_iso); it detects fibers and directions, it does not
  estimate the tensor
- direction selection on the Stage A weights (axial neighbor graph, basins)
- the per-population tensor estimators: cone refinement, Stage C VARPRO for
  single fibers, crossing FF re-solve + MRDS for crossings; the closed-form
  `estimate_AD_RD_conditioned` ("Stage B") only when Stage C is turned off
- Stage D: compartment fractions on fixed isotropic centroids
- isotropic grids: the anchored log-uniform one is the production grid; the
  plain log-uniform one is used only by the optional n_iso SVD selector
- fiber FA

Removed in 1.0.0 — the v1/v2 estimators these superseded
-----------------------------------------------------------------
`compute_aniso_centroids` (v2 single-stage centroid extraction) was not
numerically identifiable: synthetic recovery validation gave median AD/RD
relative errors of 20%-150%+ at every tested dictionary density. Use
`select_dominant_directions` (Stage A) + `estimate_AD_RD_conditioned`
(Stage B) instead.

`build_design_matrix` (v1, orientation-only single-(AD,RD) dictionary) and
`step2_refine_diffusivities[_adaptive]` (v1, non-linear grid search) were
kept for regression comparison long after nothing called them. Use
`build_design_matrix_exhaustive` for Stage A.

Removed in 1.8.1 — never called by the fit
--------------------------------------------
`generate_isotropic_grid` (LINEAR grid: no free-water column at small n, the
v1.3.1 bug), `compute_weighted_centroids` (v1/v2 isotropic centroids, replaced
by Stage D's fixed centroids), `generate_fibonacci_sphere` (full sphere; the
dictionary uses the hemisphere).
"""

from .basis import (
    build_design_matrix_exhaustive,      # v3 Stage A detection dictionary
    build_isotropic_dictionary,
    generate_exhaustive_diffusivity_pairs,  # v3 Stage A
    generate_log_uniform_isotropic_grid,  # adaptive n_iso
    generate_anchored_isotropic_grid,     # threshold-anchored grid (production default)
    generate_fibonacci_sphere_hemisphere,
)

from .solvers import (
    nnls_coordinate_descent,
    compute_regularization_matrix,       # v3 Stage A regularization
    select_dominant_directions,          # v3 Stage A direction detection
    estimate_AD_RD_conditioned,          # v3 Stage B diffusivity estimation
    compute_fiber_fa,
)

__all__ = [
    "build_design_matrix_exhaustive",
    "build_isotropic_dictionary",
    "generate_exhaustive_diffusivity_pairs",
    "generate_log_uniform_isotropic_grid",
    "generate_anchored_isotropic_grid",
    "generate_fibonacci_sphere_hemisphere",
    "nnls_coordinate_descent",
    "compute_regularization_matrix",
    "select_dominant_directions",
    "estimate_AD_RD_conditioned",
    "compute_fiber_fa",
]
