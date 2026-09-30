"""
DBSI calibration — synthetic tissue-scenario signal generation.

WHAT THIS MODULE IS, AS OF 2026-09-21
-------------------------------------
It used to hold the Monte Carlo calibration path: a grid search over
(lambda_aniso, lambda_iso) scored against 14 literature-derived tissue
scenarios (`optimize_hyperparameters`, exposed as
`calibration_method='monte_carlo'`), plus a cheaper single-pair
cross-check of an already-chosen pair (`evaluate_lambda_pair`, exposed
as `--mc-crosscheck`).

BOTH HAVE BEEN REMOVED. Neither worked on this architecture: their
shared scenario evaluator called `select_dominant_directions` with the
argument list from before `fiber_dirs` became the fifth positional
parameter, so `max_directions` received a float and the numba kernel
refused to compile. Every call raised a TypingError. The failure was
invisible because the default calibration path never goes through here.

Calibration is now data-driven only — see `calibration/data_driven.py`
(GCV + discrepancy principle on the dataset's own voxels).

WHAT IS LEFT
------------
`generate_synthetic_signal` and the `_SCENARIOS` table (14 tissue
scenarios from Wang et al. 2011, Ye et al. 2020, Vavasour et al. 2022).
These work and are kept because the scenario definitions are referenced
content, not just code — but NOTHING in the package calls them any more.
If they are not wanted for synthetic work, this whole module can go.
"""

import numpy as np

# Inspection 2026-09-30: eight imports (among them the legacy LINEAR iso-grid
# constructor, `generate_isotropic_grid`) and the grid-search constants
# (_DEFAULT_AD/RD/N_*, anisotropy ratio, n_dirs, min weight fraction,
# direction peak k, THRESH_RESTRICTED) were left behind by the removed Monte
# Carlo calibration; none was used. Removed. The `weight` / `loss_alpha` fields
# of the scenarios below scored that removed grid search: kept, as part of the
# referenced scenario table, but nothing reads them.


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_D_AX_NOMINAL = 1.60e-3
_D_RAD_NOMINAL = 0.40e-3
_D_CELL = 0.10e-3
_D_FREE = 3.05e-3


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS (unchanged from v1/v2)
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIOS = {

    'WM_normal': dict(
        f_fiber_mu=0.50, f_fiber_sd=0.05,
        f_cell_mu=0.03,  f_cell_sd=0.015,
        f_hin_mu=0.44,   f_free_mu=0.03,
        d_hin_mu=0.80e-3, d_hin_sd=0.03e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'WM_CC': dict(
        f_fiber_mu=0.63, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.32,   f_free_mu=0.03,
        d_hin_mu=0.76e-3, d_hin_sd=0.03e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'WM_subcortical': dict(
        f_fiber_mu=0.37, f_fiber_sd=0.06,
        f_cell_mu=0.04,  f_cell_sd=0.02,
        f_hin_mu=0.52,   f_free_mu=0.07,
        d_hin_mu=0.81e-3, d_hin_sd=0.04e-3,
        weight=0.8, loss_alpha=0.40,
    ),

    'GM_cortex': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.03,  f_cell_sd=0.01,
        f_hin_mu=0.87,   f_free_mu=0.10,
        d_hin_mu=0.88e-3, d_hin_sd=0.05e-3,
        weight=2.0, loss_alpha=1.0,
    ),

    'GM_deep': dict(
        f_fiber_mu=0.08, f_fiber_sd=0.04,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.79,   f_free_mu=0.09,
        d_hin_mu=0.82e-3, d_hin_sd=0.04e-3,
        weight=1.5, loss_alpha=0.80,
    ),

    'GM_cerebellum': dict(
        f_fiber_mu=0.22, f_fiber_sd=0.06,
        f_cell_mu=0.05,  f_cell_sd=0.02,
        f_hin_mu=0.65,   f_free_mu=0.08,
        d_hin_mu=0.80e-3, d_hin_sd=0.04e-3,
        weight=1.0, loss_alpha=0.65,
    ),

    'CSF_pure': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.00,  f_cell_sd=0.00,
        f_hin_mu=0.02,   f_free_mu=0.98,
        d_hin_mu=0.90e-3, d_hin_sd=0.00e-3,
        weight=2.0, loss_alpha=1.0,
    ),

    'NAWM': dict(
        f_fiber_mu=0.44, f_fiber_sd=0.05,
        f_cell_mu=0.09,  f_cell_sd=0.03,
        f_hin_mu=0.41,   f_free_mu=0.06,
        d_hin_mu=0.83e-3, d_hin_sd=0.04e-3,
        weight=0.8, loss_alpha=0.40,
    ),

    'Lesion_active': dict(
        f_fiber_mu=0.17, f_fiber_sd=0.05,
        f_cell_mu=0.40,  f_cell_sd=0.05,
        f_hin_mu=0.30,   f_free_mu=0.13,
        d_hin_mu=1.05e-3, d_hin_sd=0.06e-3,
        weight=1.2, loss_alpha=0.30,
    ),

    'Lesion_chronic': dict(
        f_fiber_mu=0.15, f_fiber_sd=0.04,
        f_cell_mu=0.08,  f_cell_sd=0.03,
        f_hin_mu=0.45,   f_free_mu=0.32,
        d_hin_mu=1.08e-3, d_hin_sd=0.07e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'Lesion_cortical': dict(
        f_fiber_mu=0.04, f_fiber_sd=0.03,
        f_cell_mu=0.17,  f_cell_sd=0.04,
        f_hin_mu=0.65,   f_free_mu=0.14,
        d_hin_mu=0.93e-3, d_hin_sd=0.05e-3,
        weight=1.2, loss_alpha=0.70,
    ),

    'PV_WM_GM': dict(
        f_fiber_mu=0.26, f_fiber_sd=0.05,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.62,   f_free_mu=0.08,
        d_hin_mu=0.85e-3, d_hin_sd=0.04e-3,
        weight=0.75, loss_alpha=0.55,
    ),

    'PV_WM_CSF': dict(
        f_fiber_mu=0.23, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.25,   f_free_mu=0.50,
        d_hin_mu=0.84e-3, d_hin_sd=0.04e-3,
        weight=0.75, loss_alpha=0.50,
    ),

    'PV_GM_CSF': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.47,   f_free_mu=0.51,
        d_hin_mu=0.89e-3, d_hin_sd=0.04e-3,
        weight=1.5, loss_alpha=1.0,
    ),

}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION (unchanged from v1/v2)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_fractions(sc, rng):
    f_fiber = float(np.clip(
        rng.normal(sc['f_fiber_mu'], sc['f_fiber_sd']), 0.0, 1.0
    ))

    remaining_after_fiber = max(0.0, 1.0 - f_fiber)
    f_cell_max = min(sc['f_cell_mu'] + 3.0 * sc['f_cell_sd'], remaining_after_fiber)
    f_cell = float(np.clip(
        rng.normal(sc['f_cell_mu'], sc['f_cell_sd']), 0.0, f_cell_max
    ))

    remaining = max(0.0, 1.0 - f_fiber - f_cell)
    total_iso = sc['f_hin_mu'] + sc['f_free_mu']
    if total_iso > 1e-10:
        f_hin = remaining * sc['f_hin_mu'] / total_iso
        f_free = remaining * sc['f_free_mu'] / total_iso
    else:
        f_hin = remaining
        f_free = 0.0

    return f_fiber, f_cell, f_hin, f_free


def _generate_signal(bvals, bvecs, snr, fiber_dir, f_fiber, f_cell, f_hin, f_free,
                     d_hin, d_ax, d_rad, rng):
    N = len(bvals)
    s = np.zeros(N)
    for i in range(N):
        b = bvals[i]
        cos_t = float(np.dot(bvecs[i], fiber_dir)) if b >= 50 else 0.0
        D_app = d_rad + (d_ax - d_rad) * cos_t**2
        s[i] = (f_fiber * np.exp(-b * D_app)
                + f_cell * np.exp(-b * _D_CELL)
                + f_hin * np.exp(-b * d_hin)
                + f_free * np.exp(-b * _D_FREE))

    sigma = 1.0 / max(float(snr), 1.0)
    n1 = rng.normal(0.0, sigma, N)
    n2 = rng.normal(0.0, sigma, N)
    return np.sqrt((s + n1)**2 + n2**2)


def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3, seed=None):
    """Legacy single-signal generator. Kept for backward compatibility.

    `seed` (added 2026-09-30): without it every call draws a new direction and
    new noise, so no result built on it could be reproduced. None keeps the old
    behaviour."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    f_hin = max(0.0, 1.0 - f_fiber - f_cell)
    return _generate_signal(bvals, bvecs, snr, v, f_fiber, f_cell, f_hin, 0.0,
                            d_hin=0.80e-3, d_ax=_D_AX_NOMINAL,
                            d_rad=_D_RAD_NOMINAL, rng=rng)
