"""
DBSI Calibration: Hyperparameter Optimization — v3 (Hybrid Two-Stage)
========================================================================

ARCHITECTURE CHANGE FROM v2
-------------------------------
v2 calibrated (lambda_aniso, lambda_iso) by evaluating fraction-recovery
error directly from the single-stage NNLS solution. v3 calibrates the
SAME two hyperparameters, but the loss is now evaluated end-to-end
through the full Stage A (direction detection) + Stage B (closed-form
diffusivity estimation) pipeline, because lambda_aniso/lambda_iso affect
Stage B's output indirectly (via the fractions and isotropic centroids
Stage A hands to Stage B, and via which direction Stage A selects) even
though Stage B itself has no free regularization parameter of its own.

This module retains the same 14 physiologically grounded tissue
scenarios from v1/v2 (Wang et al. 2011, Ye et al. 2020, Vavasour et al.
2022) — these encode validated priors about fraction distributions that
are independent of the v2-to-v3 architecture change. The loss function
also gains an AD/RD recovery term (absent in v1/v2's calibration, which
only ever validated fraction recovery): since v3 separates fraction
estimation (Stage A) from diffusivity estimation (Stage B), and the
project's synthetic recovery validation specifically flagged diffusivity
recovery as the weak point, calibration should not be blind to it.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601. doi:10.1093/brain/awr307
Ye Z, et al. (2020). NeuroImage, 221:117228.
Vavasour IM, et al. (2022). Multiple Sclerosis J, 28(3):418-428.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33-47.
Design document: toolbox_v2.md; v3 hybrid redesign motivated by
synthetic recovery validation of the v2 single-stage approach.
"""

import numpy as np
from ..core.basis import (
    build_design_matrix_exhaustive,
    generate_exhaustive_diffusivity_pairs,
    generate_fibonacci_sphere_hemisphere,
    generate_isotropic_grid,
)
from ..core.solvers import (
    nnls_coordinate_descent,
    compute_regularization_matrix,
    select_dominant_directions,
    estimate_AD_RD_conditioned,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

THRESH_RESTRICTED = 0.3e-3
THRESH_FREE = 3.0e-3

_DEFAULT_AD_MIN = 0.5e-3
_DEFAULT_AD_MAX = 2.2e-3
_DEFAULT_RD_MIN = 0.05e-3
_DEFAULT_RD_MAX = 1.2e-3
_DEFAULT_N_AD = 3
_DEFAULT_N_RD = 3
_DEFAULT_ANISOTROPY_RATIO = 1.15
_DEFAULT_N_DIRS = 30
_DEFAULT_MIN_WEIGHT_FRACTION = 0.05

_D_AX_NOMINAL = 1.60e-3
_D_RAD_NOMINAL = 0.40e-3
_D_AX_STD = 0.10e-3
_D_RAD_STD = 0.07e-3

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

_TOTAL_WEIGHT = sum(sc['weight'] for sc in _SCENARIOS.values())


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


def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3):
    """Legacy single-signal generator. Kept for backward compatibility."""
    rng = np.random.default_rng()
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    f_hin = max(0.0, 1.0 - f_fiber - f_cell)
    return _generate_signal(bvals, bvecs, snr, v, f_fiber, f_cell, f_hin, 0.0,
                            d_hin=0.80e-3, d_ax=_D_AX_NOMINAL,
                            d_rad=_D_RAD_NOMINAL, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZATION — v3: end-to-end Stage A + Stage B loss
# ─────────────────────────────────────────────────────────────────────────────

def optimize_hyperparameters(bvals, bvecs, snr, n_mc=200,
                             n_aniso_cols=None, n_iso=None,
                             n_dirs=None, n_ad=None, n_rd=None,
                             anisotropy_ratio=None,
                             ad_range=(_DEFAULT_AD_MIN, _DEFAULT_AD_MAX),
                             rd_range=(_DEFAULT_RD_MIN, _DEFAULT_RD_MAX),
                             iso_range=(0.0, 3.0e-3),
                             min_weight_fraction=_DEFAULT_MIN_WEIGHT_FRACTION):
    """
    v3 multi-tissue MC optimization for (lambda_aniso, lambda_iso),
    evaluated through the full Stage A + Stage B pipeline.

    n_mc defaults to 200 (vs. 1000 in v1/v2) because each MC sample now
    additionally runs Stage A's direction-selection logic and Stage B's
    closed-form regression on top of the NNLS solve, and the grid search
    is anchored on a single well-validated lambda_base region (see
    project synthetic recovery validation notes) rather than searching
    blind — fewer samples per cell are needed for a stable ranking given
    the narrower, pre-informed search grid used here.

    Parameters
    ----------
    bvals, bvecs : arrays
    snr : float
    n_mc : int
        MC samples per (lambda_aniso, lambda_iso, scenario) cell.
    n_aniso_cols : int or None
        Total Stage A dictionary columns (n_dirs * n_pairs), used to set
        the lambda_aniso search grid center.
    n_iso : int or None
        Isotropic grid points. Defaults to 31.
    n_dirs, n_ad, n_rd, anisotropy_ratio : optional
        Stage A dictionary construction parameters. Default to a coarse
        detection-only grid (30 dirs, 3x3 pairs, ratio 1.15) matching
        `DBSI_Adaptive`'s Stage A defaults.
    ad_range, rd_range, iso_range : tuple
    min_weight_fraction : float
        Stage A direction-selection threshold (see
        `core.solvers.select_dominant_directions`).

    Returns
    -------
    best_lambda_aniso : float
    best_lambda_iso : float
    """
    if n_dirs is None:
        n_dirs = _DEFAULT_N_DIRS
    if n_ad is None:
        n_ad = _DEFAULT_N_AD
    if n_rd is None:
        n_rd = _DEFAULT_N_RD
    if anisotropy_ratio is None:
        anisotropy_ratio = _DEFAULT_ANISOTROPY_RATIO
    if n_iso is None:
        n_iso = 31

    diff_pairs = generate_exhaustive_diffusivity_pairs(
        ad_min=ad_range[0], ad_max=ad_range[1], n_ad=n_ad,
        rd_min=rd_range[0], rd_max=rd_range[1], n_rd=n_rd,
        anisotropy_ratio=anisotropy_ratio,
    )
    n_pairs = len(diff_pairs)

    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)
    iso_grid = generate_isotropic_grid(d_min=iso_range[0], d_max=iso_range[1],
                                       n_steps=n_iso)

    n_aniso_cols_actual = n_dirs * n_pairs
    if n_aniso_cols is None:
        n_aniso_cols = n_aniso_cols_actual

    print(f"\n[CALIBRATION v3 — Stage A/B end-to-end, lambda_aniso / lambda_iso]")
    print(f"  Scenarios ({len(_SCENARIOS)}): {', '.join(_SCENARIOS.keys())}")
    print(f"  SNR = {snr:.1f}  |  MC iterations = {n_mc} per cell")
    print(f"  Stage A dictionary: {n_dirs} dirs x {n_pairs} (AD,RD) pairs = "
          f"{n_aniso_cols_actual} anisotropic cols, {n_iso} isotropic cols")

    # Search grid anchored on the validated lambda_base region (see
    # project synthetic recovery validation: lambda_base ~ 0.005 was the
    # empirical sweet spot for this Stage A dictionary size/density).
    lambda_base_grid = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]

    A = build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid)
    AtA = A.T @ A
    At = A.T

    sc_names = list(_SCENARIOS.keys())
    sc_abbr = [n[:8] for n in sc_names]
    col_w = 9
    hdr = (f"{'L_base':<8} | "
           + " | ".join(f"{a:<{col_w}}" for a in sc_abbr)
           + f" | {'Composite':>10}")
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)

    master_rng = np.random.default_rng(seed=42)
    sc_rngs = {
        name: np.random.default_rng(master_rng.integers(0, 2**31))
        for name in sc_names
    }
    sc_rng_states = {
        name: sc_rngs[name].bit_generator.state
        for name in sc_names
    }

    results = []

    for lambda_base in lambda_base_grid:
        lambda_iso = lambda_base
        lambda_aniso = lambda_base * n_aniso_cols

        AtA_reg = compute_regularization_matrix(
            AtA, n_aniso_cols_actual, lambda_aniso, lambda_iso
        )

        sc_losses = {}

        for sc_name in sc_names:
            sc = _SCENARIOS[sc_name]
            rng = sc_rngs[sc_name]
            rng.bit_generator.state = sc_rng_states[sc_name]

            ff_errors = []
            rf_errors = []
            ad_errors = []
            rd_errors = []

            for _ in range(n_mc):
                f_fiber, f_cell, f_hin, f_free = _sample_fractions(sc, rng)

                d_hin = float(np.clip(
                    rng.normal(sc['d_hin_mu'], sc['d_hin_sd']),
                    0.40e-3, 2.50e-3
                ))
                d_ax = float(np.clip(
                    rng.normal(_D_AX_NOMINAL, _D_AX_STD),
                    0.80e-3, 2.50e-3
                ))
                d_rad = float(np.clip(
                    rng.normal(_D_RAD_NOMINAL, _D_RAD_STD),
                    0.05e-3, 0.80e-3
                ))

                fiber_dir = rng.standard_normal(3)
                fiber_dir /= np.linalg.norm(fiber_dir)
                if fiber_dir[2] < 0:
                    fiber_dir = -fiber_dir

                y = _generate_signal(bvals, bvecs, snr, fiber_dir,
                                    f_fiber, f_cell, f_hin, f_free,
                                    d_hin, d_ax, d_rad, rng)
                Aty = At @ y

                w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
                w_aniso = w[:n_aniso_cols_actual]
                w_iso = w[n_aniso_cols_actual:]

                f_tot_hat = float(np.sum(w))
                if f_tot_hat > 1e-10:
                    f_fib_hat = float(np.sum(w_aniso)) / f_tot_hat
                    f_res_hat = float(np.sum(
                        w_iso[iso_grid <= THRESH_RESTRICTED]
                    )) / f_tot_hat
                else:
                    f_fib_hat = 0.0
                    f_res_hat = 0.0

                ff_errors.append(abs(f_fib_hat - f_fiber))
                rf_errors.append(abs(f_res_hat - f_cell))

                # Stage A direction selection + Stage B diffusivity estimate,
                # only evaluated for scenarios with a meaningful fiber
                # fraction (AD/RD are not defined otherwise).
                if f_fiber > 0.15 and f_tot_hat > 1e-10:
                    dir_indices, dir_weights = select_dominant_directions(
                        w_aniso, n_dirs, n_pairs, 1, min_weight_fraction
                    )
                    if dir_indices[0] >= 0:
                        recovered_dir = fiber_dirs[dir_indices[0]]

                        f_nonrf_hat_raw = float(np.sum(w_iso[iso_grid > THRESH_RESTRICTED]))
                        if f_nonrf_hat_raw > 1e-10:
                            D_nonrf_hat = float(np.sum(
                                w_iso[iso_grid > THRESH_RESTRICTED] *
                                iso_grid[iso_grid > THRESH_RESTRICTED]
                            )) / f_nonrf_hat_raw
                        else:
                            D_nonrf_hat = 1.0e-3
                        f_nonrf_hat = f_nonrf_hat_raw / f_tot_hat

                        AD_est, RD_est = estimate_AD_RD_conditioned(
                            bvals, bvecs, y, recovered_dir,
                            f_fib_hat, f_res_hat, f_nonrf_hat, 0.0,
                            0.15e-3, D_nonrf_hat, 0.0, False
                        )
                        if not np.isnan(AD_est):
                            ad_errors.append(abs(AD_est - d_ax) / d_ax)
                            rd_errors.append(abs(RD_est - d_rad) / d_rad)

            mae_ff = float(np.mean(ff_errors))
            mae_rf = float(np.mean(rf_errors))
            mae_ad = float(np.mean(ad_errors)) if ad_errors else 0.0
            mae_rd = float(np.mean(rd_errors)) if rd_errors else 0.0

            alpha = sc['loss_alpha']
            fraction_loss = alpha * mae_ff + (1.0 - alpha) * mae_rf
            # Diffusivity loss is added with modest weight (0.3) relative
            # to fraction loss: fraction recovery remains the primary
            # clinical signal (inflammation/demyelination presence),
            # while AD/RD recovery quality is a secondary but non-trivial
            # consideration given the project's specific concern with RD
            # precision as a demyelination marker.
            sc_losses[sc_name] = fraction_loss + 0.3 * (mae_ad + mae_rd)

        composite = (
            sum(_SCENARIOS[n]['weight'] * sc_losses[n] for n in sc_names)
            / _TOTAL_WEIGHT
        )

        row = (f"{lambda_base:<8.4f} | "
               + " | ".join(f"{sc_losses[n]:<{col_w}.4f}" for n in sc_names)
               + f" | {composite:>10.4f}")
        print(row)

        results.append(dict(
            lambda_base=lambda_base,
            lambda_aniso=lambda_aniso,
            lambda_iso=lambda_iso,
            composite=composite,
            sc_losses=dict(sc_losses),
        ))

    print(sep)

    best_res = min(results, key=lambda r: r['composite'])

    print(f"\n  BEST: lambda_aniso={best_res['lambda_aniso']:.4f}, "
          f"lambda_iso={best_res['lambda_iso']:.4f}  "
          f"(lambda_base={best_res['lambda_base']:.4f}, "
          f"composite={best_res['composite']:.4f})")

    print(f"\n  Per-scenario losses at BEST:")
    for sc_name in sc_names:
        sc = _SCENARIOS[sc_name]
        loss = best_res['sc_losses'][sc_name]
        note = " <- FF suppression only" if sc['loss_alpha'] >= 1.0 else ""
        print(f"    {sc_name:<22}  loss={loss:.4f}  "
              f"(w={sc['weight']:.2f}, alpha={sc['loss_alpha']:.2f}){note}")

    print(f"\n  -> Using: lambda_aniso={best_res['lambda_aniso']:.4f}, "
          f"lambda_iso={best_res['lambda_iso']:.4f}")
    print(sep + "\n")

    return best_res['lambda_aniso'], best_res['lambda_iso']
