"""
DBSI Calibration: Monte Carlo Cross-Check — v3 (Hybrid Two-Stage)
====================================================================

ROLE CHANGE: FROM PRIMARY CALIBRATION TO INDEPENDENT CROSS-CHECK
----------------------------------------------------------------------
In earlier toolbox versions, this module's Monte Carlo scenario-based
optimization WAS the source of (lambda_aniso, lambda_iso). As of this
revision, the PRIMARY calibration path is the data-driven selection in
`calibration/data_driven.py` (GCV for lambda_iso, the discrepancy
principle for lambda_aniso), which derives both hyperparameters purely
from the acquisition protocol and a sample of the dataset's own voxels
— no tissue-fraction priors required.

This module is retained, unchanged in its core methodology, as an
INDEPENDENT CROSS-CHECK: given a candidate (lambda_aniso, lambda_iso)
pair (typically the data-driven module's output), `optimize_hyperparameters`
can still be used to verify that pair's behaviour across 14
physiologically grounded tissue scenarios (Wang et al. 2011, Ye et al.
2020, Vavasour et al. 2022) — e.g. does it adequately suppress spurious
fiber fraction in grey matter, does it preserve restricted-fraction
sensitivity in lesion scenarios. Used this way, the Monte Carlo
scenarios serve as a VALIDATION tool (does a given lambda pair behave
sensibly across known tissue regimes) rather than as the OPTIMIZATION
target (searching for the lambda pair that performs best on those same
scenarios) — a more defensible use of literature-derived priors, since
they are no longer the sole determinant of the toolbox's regularization
strength.

WHY THE ROLE CHANGED
-------------------------
Project validation directly compared data-driven selection against a
freshly-run Monte Carlo calibration on the IDENTICAL protocol/dictionary
and found the two methods select comparable lambda_aniso values and
produce comparable (neither uniformly better) synthetic AD/RD recovery
accuracy across 5 independent test seeds — see `calibration/
data_driven.py` module docstring for the full empirical summary. Given
this comparable performance, the data-driven method's much lower
computational cost (a handful of NNLS solves vs. thousands of Monte
Carlo samples across a 6x14 lambda x scenario grid) and its avoidance of
externally-sourced tissue-fraction priors (see data_driven.py's
discussion of why those priors are a methodological risk when applied
to a specific protocol/population they were not derived from) make it
the more defensible PRIMARY choice. The Monte Carlo scenarios remain
valuable — just in a verification role rather than an optimization role.

This module retains the same 14 physiologically grounded tissue
scenarios from v1/v2/v3 (Wang et al. 2011, Ye et al. 2020, Vavasour et
al. 2022) and the same end-to-end Stage A + Stage B evaluation
introduced in the v3 architecture change (loss now includes an AD/RD
recovery term in addition to fraction recovery, since v3 separates
fraction estimation (Stage A) from diffusivity estimation (Stage B) and
calibration should not be blind to Stage B's quality).

USAGE AS A CROSS-CHECK
---------------------------
    from dbsi_toolbox.calibration.optimizer import evaluate_lambda_pair

    # After obtaining (lambda_aniso, lambda_iso) from the data-driven
    # path (see DBSI_Adaptive.fit() or calibration.data_driven):
    report = evaluate_lambda_pair(
        bvals, bvecs, snr, lambda_aniso, lambda_iso,
        n_dirs=..., n_ad=..., n_rd=..., anisotropy_ratio=...,
    )
    # report contains per-scenario and composite losses for manual
    # inspection / inclusion in a methods supplement table.

The original full grid-search `optimize_hyperparameters` function is
still available for callers who want to fall back to a from-scratch
Monte Carlo search (e.g. if the data-driven method's assumptions are
suspected to fail for a given protocol — see data_driven.py caveats).

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601. doi:10.1093/brain/awr307
Ye Z, et al. (2020). NeuroImage, 221:117228.
Vavasour IM, et al. (2022). Multiple Sclerosis J, 28(3):418-428.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33-47.
Design document: toolbox_v2.md; v3 hybrid redesign motivated by
synthetic recovery validation of the v2 single-stage approach; lambda
selection methodology revised after head-to-head comparison with
GCV/discrepancy-principle data-driven selection (see
calibration/data_driven.py).
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
# SHARED PER-SCENARIO EVALUATION (used by both grid search and single-pair cross-check)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_scenario(sc, rng, bvals, bvecs, fiber_dirs, n_dirs, n_pairs,
                       iso_grid, AtA_reg, At, snr, n_mc, min_weight_fraction):
    """
    Evaluate one tissue scenario's fraction + AD/RD recovery loss for a
    GIVEN (already-regularized) AtA_reg. Shared by both the grid-search
    `optimize_hyperparameters` and the single-pair `evaluate_lambda_pair`
    cross-check, so the two use IDENTICAL scenario logic and cannot
    silently drift apart.

    Returns
    -------
    loss : float
        alpha * mae_ff + (1-alpha) * mae_rf + 0.3 * (mae_ad + mae_rd)
    diagnostics : dict
        {'mae_ff', 'mae_rf', 'mae_ad', 'mae_rd', 'n_ad_rd_valid'}
    """
    n_aniso_cols_actual = n_dirs * n_pairs

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
    # Diffusivity loss is added with modest weight (0.3) relative to
    # fraction loss: fraction recovery remains the primary clinical
    # signal (inflammation/demyelination presence), while AD/RD
    # recovery quality is a secondary but non-trivial consideration
    # given the project's specific concern with RD precision as a
    # demyelination marker.
    loss = fraction_loss + 0.3 * (mae_ad + mae_rd)

    diagnostics = dict(
        mae_ff=mae_ff, mae_rf=mae_rf, mae_ad=mae_ad, mae_rd=mae_rd,
        n_ad_rd_valid=len(ad_errors),
    )

    return loss, diagnostics


def evaluate_lambda_pair(bvals, bvecs, snr, lambda_aniso, lambda_iso, n_mc=200,
                         n_dirs=None, n_ad=None, n_rd=None,
                         anisotropy_ratio=None,
                         ad_range=(_DEFAULT_AD_MIN, _DEFAULT_AD_MAX),
                         rd_range=(_DEFAULT_RD_MIN, _DEFAULT_RD_MAX),
                         iso_range=(0.0, 3.0e-3), n_iso=31,
                         min_weight_fraction=_DEFAULT_MIN_WEIGHT_FRACTION,
                         verbose=True):
    """
    Cross-check a SINGLE candidate (lambda_aniso, lambda_iso) pair —
    typically obtained from `calibration.data_driven.
    select_lambdas_data_driven` — against the 14 physiologically
    grounded tissue scenarios, WITHOUT running a grid search.

    This is the recommended way to use the Monte Carlo scenarios going
    forward: as a verification tool for a candidate lambda pair chosen
    by a primarily data-driven method, not as the mechanism that
    searches for the best pair (see module docstring for the rationale
    behind this role change).

    Parameters
    ----------
    bvals, bvecs : arrays
    snr : float
    lambda_aniso, lambda_iso : float
        The candidate pair to evaluate.
    n_mc : int
        MC samples per scenario. Default 200; increase for a more
        precise cross-check report (e.g. for a methods supplement
        table), decrease for a quick sanity check during development.
    n_dirs, n_ad, n_rd, anisotropy_ratio, ad_range, rd_range, iso_range, n_iso :
        Stage A dictionary parameters — MUST match the dictionary
        `lambda_aniso`/`lambda_iso` were actually selected/will actually
        be used for, or the cross-check is not evaluating the same
        configuration that will be deployed.
    min_weight_fraction : float
        Stage A direction-selection threshold (see
        `core.solvers.select_dominant_directions`).
    verbose : bool
        Print the per-scenario breakdown table.

    Returns
    -------
    report : dict
        {
          'composite': float,
          'sc_losses': {scenario_name: loss},
          'sc_diagnostics': {scenario_name: {'mae_ff', 'mae_rf', 'mae_ad', 'mae_rd', 'n_ad_rd_valid'}},
          'lambda_aniso': float, 'lambda_iso': float,
        }
        Suitable for direct inclusion in a methods supplement table, or
        for programmatic pass/fail checks (e.g. flag if any scenario's
        loss exceeds a tolerance).
    """
    if n_dirs is None:
        n_dirs = _DEFAULT_N_DIRS
    if n_ad is None:
        n_ad = _DEFAULT_N_AD
    if n_rd is None:
        n_rd = _DEFAULT_N_RD
    if anisotropy_ratio is None:
        anisotropy_ratio = _DEFAULT_ANISOTROPY_RATIO

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

    A = build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid)
    AtA = A.T @ A
    At = A.T
    AtA_reg = compute_regularization_matrix(AtA, n_aniso_cols_actual, lambda_aniso, lambda_iso)

    sc_names = list(_SCENARIOS.keys())

    if verbose:
        print(f"\n[CROSS-CHECK — lambda_aniso={lambda_aniso:.4f}, lambda_iso={lambda_iso:.4f}]")
        print(f"  Scenarios ({len(sc_names)}): {', '.join(sc_names)}")
        print(f"  SNR = {snr:.1f}  |  MC iterations = {n_mc} per scenario")
        col_w = 9
        hdr = (f"{'Scenario':<22} | {'MAE_FF':<9} | {'MAE_RF':<9} | "
               f"{'MAE_AD':<9} | {'MAE_RD':<9} | {'Loss':<9} | {'Weight'}")
        sep = "-" * len(hdr)
        print("\n" + sep)
        print(hdr)
        print(sep)

    master_rng = np.random.default_rng(seed=42)

    sc_losses = {}
    sc_diagnostics = {}

    for sc_name in sc_names:
        sc = _SCENARIOS[sc_name]
        rng = np.random.default_rng(master_rng.integers(0, 2**31))

        loss, diag = _evaluate_scenario(
            sc, rng, bvals, bvecs, fiber_dirs, n_dirs, n_pairs, iso_grid,
            AtA_reg, At, snr, n_mc, min_weight_fraction
        )
        sc_losses[sc_name] = loss
        sc_diagnostics[sc_name] = diag

        if verbose:
            print(f"{sc_name:<22} | {diag['mae_ff']:<9.4f} | {diag['mae_rf']:<9.4f} | "
                  f"{diag['mae_ad']:<9.4f} | {diag['mae_rd']:<9.4f} | {loss:<9.4f} | "
                  f"{sc['weight']:.2f}")

    composite = (
        sum(_SCENARIOS[n]['weight'] * sc_losses[n] for n in sc_names)
        / _TOTAL_WEIGHT
    )

    if verbose:
        print(sep)
        print(f"\n  COMPOSITE LOSS: {composite:.4f}")
        print(f"  (Compare against a fresh Monte Carlo grid search's best composite,\n"
              f"  or a previous cross-check run, to assess whether this lambda pair\n"
              f"  behaves reasonably across known tissue regimes.)\n")

    return dict(
        composite=composite,
        sc_losses=sc_losses,
        sc_diagnostics=sc_diagnostics,
        lambda_aniso=lambda_aniso,
        lambda_iso=lambda_iso,
    )


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

            loss, _diag = _evaluate_scenario(
                sc, rng, bvals, bvecs, fiber_dirs, n_dirs, n_pairs, iso_grid,
                AtA_reg, At, snr, n_mc, min_weight_fraction
            )
            sc_losses[sc_name] = loss

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
