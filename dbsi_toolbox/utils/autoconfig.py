"""
DBSI Autoconfiguration — Protocol-Driven Stage A Sizing (v3)
=================================================================

v3 NOTE
---------
In the v3 hybrid two-stage architecture, this module's output is used
more narrowly than in v2: `DBSI_Adaptive` only takes `M_optimal` (the
hemisphere direction count) from `autoconfigure_dictionary` for Stage
A's detection dictionary. `n_ad`, `n_rd`, and `anisotropy_ratio` are
still computed and returned (useful as protocol diagnostics, and for
any external code doing its own Stage A configuration), but
`DBSI_Adaptive` does NOT use this function's n_ad/n_rd suggestions by
default — Stage A intentionally uses a small, fixed-by-default (AD,RD)
grid (3x3) regardless of protocol richness, because Stage A's job
(direction detection) does not benefit from finer diffusivity
resolution; see `model_Niso_adaptive_ff_thr.py` module docstring for the
full rationale and the synthetic validation that motivated this design.

Angular calibration
--------------------
M (number of fiber directions in Stage A's dictionary) is set
proportional to the maximum number of diffusion-weighted directions
acquired in any single shell:

    M ~= N_max_dirs_per_shell * 1.3

If M is set much higher than the number of actual measurements per
shell, the dictionary is angularly over-complete relative to the data
and the solver cannot distinguish between dictionary directions that are
closer together than the actual angular sampling resolution — this
produces angular leakage (spurious crossing-fiber artifacts).

Spectral (AD/RD) calibration — diagnostic only in v3, not applied by
DBSI_Adaptive's Stage A by default (see note above)
-----------------------------------------------------------------------
Retained from v2 for diagnostic purposes / external use:

    - Single shell or b_max < 1500 s/mm^2: sparse grid (4x4) appropriate.
    - b_max >= 2500 s/mm^2, multi-shell: denser grids supportable (up to
      8x8 or 10x10) IF doing single-stage (AD,RD) recovery — which v3
      does not do in Stage A (see note above).

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601.
Design document: toolbox_v2.md; v3 hybrid redesign notes.
"""

import numpy as np


def autoconfigure_dictionary(bvals, bvecs):
    """
    Analyse the acquisition protocol and suggest the optimal mathematical
    architecture of the exhaustive anisotropic dictionary, to avoid
    ill-conditioning and angular/spectral leakage.

    Parameters
    ----------
    bvals : array-like (N,)
        B-values, s/mm^2.
    bvecs : array-like (N, 3)
        Gradient directions (used only via `bvals` here; accepted for
        API symmetry and forward compatibility with direction-density
        criteria beyond per-shell counts).

    Returns
    -------
    M_optimal : int
        Suggested number of hemisphere directions for the anisotropic
        dictionary.
    n_ad : int
        Suggested number of axial-diffusivity grid steps.
    n_rd : int
        Suggested number of radial-diffusivity grid steps.
    anisotropy_ratio : float
        Suggested minimum AD/RD ratio for admissible anisotropic pairs.
    """
    bvals = np.asarray(bvals, dtype=np.float64)

    # Identify shells (rounded to nearest 100), excluding b=0 volumes.
    unique_bvals, counts = np.unique(np.round(bvals, -2), return_counts=True)
    shell_dirs = [counts[i] for i in range(len(unique_bvals)) if unique_bvals[i] > 50]

    max_dirs_per_shell = max(shell_dirs) if shell_dirs else 30
    b_max = float(max(unique_bvals)) if len(unique_bvals) else 0.0
    n_shells = len(shell_dirs)

    # 1. Angular calibration (avoid angular leakage)
    M_optimal = int(max_dirs_per_shell * 1.3)

    # 2. Spectral calibration (avoid AD/RD collinearity)
    if n_shells <= 1 or b_max < 1500:
        n_ad, n_rd = 4, 4
        anisotropy_ratio = 1.2
    elif n_shells == 2 or b_max < 2500:
        n_ad, n_rd = 6, 6
        anisotropy_ratio = 1.15
    else:  # advanced protocols (e.g. Connectome-style)
        n_ad, n_rd = 8, 8
        anisotropy_ratio = 1.1

    return M_optimal, n_ad, n_rd, anisotropy_ratio
