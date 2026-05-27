"""
DBSI Calibration: Hyperparameter Optimization — Comprehensive Multi-Tissue Version
===================================================================================

Monte Carlo optimization for (n_iso, reg_lambda) across 14 physiologically grounded
tissue scenarios covering the full range of human brain tissue encountered in
diffusion MRI studies.

─────────────────────────────────────────────────────────────────────────────────────
SCENARIO DESIGN  (values grounded in Wang et al. 2011, Ye et al. 2020,
                  Vavasour et al. 2022, and established dMRI literature)
─────────────────────────────────────────────────────────────────────────────────────

In the DBSI signal model, four biophysical pools contribute to the measured signal:

  fiber  — cylindrically symmetric tensor (D_ax, D_rad), captures organized axons.
           Signal is anisotropic and direction-dependent.
  cell   — isotropic restricted  (ADC ≤ 0.3e-3 mm²/s): inflammatory cells, tightly
           packed axonal segments that appear isotropic at the voxel scale.
  hin    — isotropic hindered    (0.3e-3 < ADC ≤ 3.0e-3 mm²/s): extracellular water,
           interstitial fluid, myelin water. ADC varies by tissue type (~0.7-1.2e-3).
  free   — isotropic free water  (ADC > 3.0e-3 mm²/s): CSF, vasogenic oedema.

Ground-truth fractions are sampled from truncated normal distributions at each Monte
Carlo iteration (not fixed per-call) to reflect intra-class tissue variability and
partial-volume heterogeneity within each class. D_hin is also sampled per iteration.

─────────────────────────────────────────────────────────────────────────────────────
LOSS FUNCTION DESIGN
─────────────────────────────────────────────────────────────────────────────────────

Each scenario contributes two error metrics:
  mae_ff  = |FF_estimated − FF_true|    (fiber fraction error)
  mae_rf  = |RF_estimated − RF_true|    (restricted fraction error)

The per-scenario loss is a weighted combination reflecting clinical priorities:
  • FF-dominant penalty  (FF_true ≈ 0): spurious fiber signal must be suppressed.
    Loss = mae_ff                        (100% fiber suppression, alpha=1.0)
  • RF-dominant penalty  (active lesion, high RF_true): RF is the biomarker.
    Loss = 0.30·mae_ff + 0.70·mae_rf    (alpha=0.30)
  • Balanced penalty (WM, mixed): both fractions matter.
    Loss = 0.40·mae_ff + 0.60·mae_rf    (alpha=0.40)

Composite loss = weighted mean over all scenarios (weights reflect tissue prevalence
and clinical importance in neuroinflammatory studies).

─────────────────────────────────────────────────────────────────────────────────────
DIFFERENTIATED REGULARIZATION (mirrors main fitting kernel exactly)
─────────────────────────────────────────────────────────────────────────────────────
  reg_vec[:n_dirs] = lambda × n_dirs    (fiber columns scaled by dict size)
  reg_vec[n_dirs:] = lambda             (isotropic columns)
  Solver called with reg=0.0 to avoid double-counting the penalty.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590–3601. doi:10.1093/brain/awr307
Ye Z, et al. (2020). NeuroImage, 221:117228.
Vavasour IM, et al. (2022). Multiple Sclerosis J, 28(3):418–428.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33–47.
"""

import numpy as np
from ..core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from ..core.solvers import nnls_coordinate_descent


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match model_Niso_adaptive_ff_thr.py exactly
# ─────────────────────────────────────────────────────────────────────────────

THRESH_RESTRICTED = 0.3e-3   # mm²/s — RF/HF boundary (Wang et al. 2011)
THRESH_FREE       = 3.0e-3   # mm²/s — HF/WF boundary

# Design matrix diffusivities — MUST match model_Niso_adaptive_ff_thr.py
# DESIGN_MATRIX_AD and DESIGN_MATRIX_RD.  Using different values here would
# mean the calibration optimises (n_iso, λ) against a design matrix that
# differs from the one used during fitting, making the selected hyperparameters
# suboptimal for the actual model.
_DESIGN_MATRIX_AD = 1.5e-3   # mm²/s — matches DESIGN_MATRIX_AD in model
_DESIGN_MATRIX_RD = 0.5e-3   # mm²/s — matches DESIGN_MATRIX_RD in model

# Nominal fiber tensor diffusivities (WM in vivo at 3T, Wang 2011)
_D_AX_NOMINAL  = 1.60e-3    # mm²/s
_D_RAD_NOMINAL = 0.40e-3    # mm²/s
_D_AX_STD      = 0.10e-3    # intra-WM variability
_D_RAD_STD     = 0.07e-3

# Restricted pool ADC (inflammatory cells, Wang et al. 2011)
_D_CELL = 0.10e-3            # mm²/s

# Free water ADC (CSF at 37°C)
_D_FREE = 3.05e-3            # mm²/s


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each scenario specifies:
#   f_fiber_mu / _sd   : fiber fraction mean and std for MC sampling
#   f_cell_mu  / _sd   : restricted fraction mean and std for MC sampling
#   f_hin_mu           : nominal hindered fraction (derived after sampling)
#   f_free_mu          : nominal free-water fraction (derived after sampling)
#   d_hin_mu   / _sd   : hindered ADC distribution (tissue-specific)
#   weight             : relative importance in composite loss
#   loss_alpha         : fraction of per-scenario loss assigned to mae_ff
#                        (1-alpha → mae_rf)

_SCENARIOS = {

    # ── Normal white matter ──────────────────────────────────────────────────

    'WM_normal': dict(
        # Typical WM tract. FF ≈ 0.45–0.55 (Wang 2011 normal CC/CST values).
        # RF very low in healthy tissue (< 0.05).
        # D_hin ~ extracellular WM water: 0.76–0.84e-3 mm²/s (Jelescu 2016).
        f_fiber_mu=0.50, f_fiber_sd=0.05,
        f_cell_mu=0.03,  f_cell_sd=0.015,
        f_hin_mu=0.44,   f_free_mu=0.03,
        d_hin_mu=0.80e-3, d_hin_sd=0.03e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'WM_CC': dict(
        # Corpus callosum: highest fiber density in the brain.
        # FF can reach 0.60–0.70 (Wang 2011 Fig. 3; Ye 2020).
        # Very low RF, minimal free water.
        f_fiber_mu=0.63, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.32,   f_free_mu=0.03,
        d_hin_mu=0.76e-3, d_hin_sd=0.03e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'WM_subcortical': dict(
        # Subcortical U-fibers and association tracts: lower FF than deep WM,
        # more interstitial water due to proximity to the cortex.
        f_fiber_mu=0.37, f_fiber_sd=0.06,
        f_cell_mu=0.04,  f_cell_sd=0.02,
        f_hin_mu=0.52,   f_free_mu=0.07,
        d_hin_mu=0.81e-3, d_hin_sd=0.04e-3,
        weight=0.8, loss_alpha=0.40,
    ),

    # ── Grey matter ──────────────────────────────────────────────────────────

    'GM_cortex': dict(
        # Cortical grey matter: no organised fiber tracts → FF must be 0.
        # Mean cortical ADC ~0.80–0.90e-3 mm²/s falls entirely in the hindered
        # range. This is the primary driver of the spurious FF problem.
        # Weight 2.0, loss_alpha=1.0: 100% FF suppression.
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.03,  f_cell_sd=0.01,
        f_hin_mu=0.87,   f_free_mu=0.10,
        d_hin_mu=0.88e-3, d_hin_sd=0.05e-3,
        weight=2.0, loss_alpha=1.0,
    ),

    'GM_deep': dict(
        # Putamen, caudate, thalamus. Small fiber-like structure (thalamic fibres,
        # striatal projections) gives FF 0.05–0.12, but predominantly hindered.
        # ADC slightly lower than cortex (~0.78–0.84e-3 mm²/s).
        f_fiber_mu=0.08, f_fiber_sd=0.04,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.79,   f_free_mu=0.09,
        d_hin_mu=0.82e-3, d_hin_sd=0.04e-3,
        weight=1.5, loss_alpha=0.80,
    ),

    'GM_cerebellum': dict(
        # Cerebellar cortex and subjacent WM intermixed at typical voxel resolution.
        # Moderate FF from cerebellar peduncle/dentate contributions.
        # ADC of cerebellar GM ~0.76–0.84e-3 mm²/s.
        f_fiber_mu=0.22, f_fiber_sd=0.06,
        f_cell_mu=0.05,  f_cell_sd=0.02,
        f_hin_mu=0.65,   f_free_mu=0.08,
        d_hin_mu=0.80e-3, d_hin_sd=0.04e-3,
        weight=1.0, loss_alpha=0.65,
    ),

    # ── CSF / free water ─────────────────────────────────────────────────────

    'CSF_pure': dict(
        # Ventricular and sulcal CSF. FF must be exactly 0; RF negligible.
        # D_free = 3.05e-3 mm²/s at 37°C. Tiny hindered component (<2%).
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.00,  f_cell_sd=0.00,
        f_hin_mu=0.02,   f_free_mu=0.98,
        d_hin_mu=0.90e-3, d_hin_sd=0.00e-3,
        weight=2.0, loss_alpha=1.0,
    ),

    # ── MS white matter pathology ─────────────────────────────────────────────

    'NAWM': dict(
        # Normal-appearing white matter in MS. Subtle microstructural changes:
        # mildly elevated RF (0.06–0.14) reflecting subclinical inflammation,
        # slightly reduced FF (Wang 2011; Vavasour 2022 NAWM values).
        f_fiber_mu=0.44, f_fiber_sd=0.05,
        f_cell_mu=0.09,  f_cell_sd=0.03,
        f_hin_mu=0.41,   f_free_mu=0.06,
        d_hin_mu=0.83e-3, d_hin_sd=0.04e-3,
        weight=0.8, loss_alpha=0.40,
    ),

    'Lesion_active': dict(
        # Active/enhancing MS lesion. Dense inflammatory infiltrate → very high
        # RF (0.30–0.45, Wang 2011 Fig. 6). FF markedly reduced.
        # Elevated D_hin due to vasogenic oedema (~0.95–1.15e-3 mm²/s).
        f_fiber_mu=0.17, f_fiber_sd=0.05,
        f_cell_mu=0.40,  f_cell_sd=0.05,
        f_hin_mu=0.30,   f_free_mu=0.13,
        d_hin_mu=1.05e-3, d_hin_sd=0.06e-3,
        weight=1.2, loss_alpha=0.30,   # RF is the key biomarker here
    ),

    'Lesion_chronic': dict(
        # Chronic/black-hole MS lesion. Severe axon loss (low FF), low RF
        # (burnt-out inflammation), prominent free water (Wallerian degeneration).
        # High WF fraction (0.25–0.40) distinguishes from acute lesion.
        f_fiber_mu=0.15, f_fiber_sd=0.04,
        f_cell_mu=0.08,  f_cell_sd=0.03,
        f_hin_mu=0.45,   f_free_mu=0.32,
        d_hin_mu=1.08e-3, d_hin_sd=0.07e-3,
        weight=1.0, loss_alpha=0.40,
    ),

    'Lesion_cortical': dict(
        # Cortical/leucocortical MS lesion (type I–IV, Bö 2003; Vavasour 2022).
        # Little fiber structure (cortical), elevated RF (cortical demyelination
        # with microglial activation), higher D_hin than normal cortex.
        # FF should remain very low even in the presence of inflammation.
        f_fiber_mu=0.04, f_fiber_sd=0.03,
        f_cell_mu=0.17,  f_cell_sd=0.04,
        f_hin_mu=0.65,   f_free_mu=0.14,
        d_hin_mu=0.93e-3, d_hin_sd=0.05e-3,
        weight=1.2, loss_alpha=0.70,
    ),

    # ── Partial volume ────────────────────────────────────────────────────────

    'PV_WM_GM': dict(
        # 50/50 partial-volume mix at the WM/GM interface.
        # FF intermediate (~0.22–0.30), hindered pool elevated.
        f_fiber_mu=0.26, f_fiber_sd=0.05,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.62,   f_free_mu=0.08,
        d_hin_mu=0.85e-3, d_hin_sd=0.04e-3,
        weight=0.75, loss_alpha=0.55,
    ),

    'PV_WM_CSF': dict(
        # WM voxel adjacent to a sulcus or ventricle.
        # Prominent free-water component (0.45–0.55), low-to-moderate FF.
        f_fiber_mu=0.23, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.25,   f_free_mu=0.50,
        d_hin_mu=0.84e-3, d_hin_sd=0.04e-3,
        weight=0.75, loss_alpha=0.50,
    ),

    'PV_GM_CSF': dict(
        # Cortical GM voxel adjacent to sulcal CSF.
        # FF must be 0; large free-water contamination.
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.47,   f_free_mu=0.51,
        d_hin_mu=0.89e-3, d_hin_sd=0.04e-3,
        weight=1.5, loss_alpha=1.0,
    ),

}

_TOTAL_WEIGHT = sum(sc['weight'] for sc in _SCENARIOS.values())


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_fractions(sc, rng):
    """
    Sample tissue fractions for one MC iteration from truncated normals.

    f_fiber and f_cell are sampled independently. f_hin and f_free fill the
    remaining fraction, preserving their nominal ratio.

    Returns
    -------
    f_fiber, f_cell, f_hin, f_free : float  (all ≥ 0, sum = 1)
    """
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
        f_hin  = remaining * sc['f_hin_mu']  / total_iso
        f_free = remaining * sc['f_free_mu'] / total_iso
    else:
        f_hin  = remaining
        f_free = 0.0

    return f_fiber, f_cell, f_hin, f_free


def _generate_signal(bvals, bvecs, snr, f_fiber, f_cell, f_hin, f_free,
                     d_hin, d_ax, d_rad, rng):
    """
    Generate one noisy DBSI signal for the given tissue fractions.

    Uses a cylindrical fiber tensor + three isotropic pools.
    Rician noise is added at SNR = snr.
    """
    N = len(bvals)
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    if v[2] < 0:
        v = -v

    s = np.zeros(N)
    for i in range(N):
        b     = bvals[i]
        cos_t = float(np.dot(bvecs[i], v)) if b >= 50 else 0.0
        D_app = d_rad + (d_ax - d_rad) * cos_t**2
        s[i]  = (f_fiber * np.exp(-b * D_app)
                 + f_cell  * np.exp(-b * _D_CELL)
                 + f_hin   * np.exp(-b * d_hin)
                 + f_free  * np.exp(-b * _D_FREE))

    sigma = 1.0 / max(float(snr), 1.0)
    n1 = rng.normal(0.0, sigma, N)
    n2 = rng.normal(0.0, sigma, N)
    return np.sqrt((s + n1)**2 + n2**2)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3):
    """
    Legacy single-signal generator. Kept for backward compatibility.
    """
    rng   = np.random.default_rng()
    f_hin = max(0.0, 1.0 - f_fiber - f_cell)
    return _generate_signal(bvals, bvecs, snr, f_fiber, f_cell, f_hin, 0.0,
                            d_hin=0.80e-3, d_ax=_D_AX_NOMINAL,
                            d_rad=_D_RAD_NOMINAL, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_hyperparameters(bvals, bvecs, snr, n_mc=1000):
    """
    Comprehensive multi-tissue MC optimization for (n_iso, reg_lambda).

    Evaluates a weighted composite loss across all 14 tissue scenarios.
    The differentiated regularization scheme (fiber columns penalised n_dirs×
    more than isotropic columns) is applied identically to the main kernel so
    the selected hyperparameters are directly transferable without re-scaling.

    The design matrix is built with the same AD/RD as the fitting model
    (_DESIGN_MATRIX_AD = 1.5e-3, _DESIGN_MATRIX_RD = 0.5e-3) to ensure
    the calibrated hyperparameters apply to the same forward model.

    Parameters
    ----------
    bvals  : ndarray (N,)
    bvecs  : ndarray (N, 3)
    snr    : float  — estimated from data via estimate_snr_robust()
    n_mc   : int    — MC samples per (n_iso, lambda, scenario) cell

    Returns
    -------
    best_n_iso   : int
    best_lambda  : float
    """
    print(f"\n[CALIBRATION — Comprehensive Multi-Tissue]")
    print(f"  Scenarios ({len(_SCENARIOS)}): {', '.join(_SCENARIOS.keys())}")
    print(f"  SNR = {snr:.1f}  |  MC iterations = {n_mc} per cell")
    print(f"  Design matrix: AD={_DESIGN_MATRIX_AD*1e3:.1f}e-3, "
          f"RD={_DESIGN_MATRIX_RD*1e3:.1f}e-3 mm²/s")
    print(f"  Reg scheme: λ_fiber = λ × n_dirs,  λ_iso = λ  (solver reg = 0.0)")

    # ── Hyperparameter grid ───────────────────────────────────────────────────
    bases_grid   = [25, 50, 75, 100, 125, 150, 200]
    lambdas_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]

    n_dirs     = 100
    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)

    # ── Print table header ────────────────────────────────────────────────────
    sc_names = list(_SCENARIOS.keys())
    sc_abbr  = [n[:8] for n in sc_names]
    col_w    = 10
    hdr = (f"{'Bases':<6} | {'Lambda':<7} | "
           + " | ".join(f"{a:<{col_w}}" for a in sc_abbr)
           + f" | {'Composite':>10}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)

    # ── RNG setup: each scenario gets an independent stream ────────────────────
    # States are saved so every (n_iso, lambda) cell reuses identical signals,
    # making the grid comparison fair.
    master_rng = np.random.default_rng(seed=42)
    sc_rngs    = {
        name: np.random.default_rng(master_rng.integers(0, 2**31))
        for name in sc_names
    }
    sc_rng_states = {
        name: sc_rngs[name].bit_generator.state
        for name in sc_names
    }

    results = []

    for n_iso in bases_grid:
        iso_grid = np.linspace(0.0, 3.5e-3, n_iso)

        # Build design matrix with the same AD/RD used in the fitting model.
        # Previously the default rd=0.4e-3 was used here, which differs from
        # DESIGN_MATRIX_RD=0.5e-3 and caused the calibrated hyperparameters
        # to be suboptimal for the actual fitting step.
        A   = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid,
                                  ad=_DESIGN_MATRIX_AD, rd=_DESIGN_MATRIX_RD)
        AtA = A.T @ A
        At  = A.T

        for reg in lambdas_grid:

            # ── Differentiated regularization — identical to main kernel ──────
            reg_vec          = np.ones(AtA.shape[0])
            reg_vec[:n_dirs] = reg * n_dirs
            reg_vec[n_dirs:] = reg
            AtA_reg          = AtA + np.diag(reg_vec)

            sc_losses = {}

            for sc_name in sc_names:
                sc  = _SCENARIOS[sc_name]
                rng = sc_rngs[sc_name]

                # Restore RNG state: every (n_iso, lambda) uses the same signals
                rng.bit_generator.state = sc_rng_states[sc_name]

                ff_errors = []
                rf_errors = []

                for _ in range(n_mc):

                    # Sample fractions and tissue diffusivities
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

                    y   = _generate_signal(bvals, bvecs, snr,
                                           f_fiber, f_cell, f_hin, f_free,
                                           d_hin, d_ax, d_rad, rng)
                    Aty = At @ y

                    # reg=0.0: penalty already in AtA_reg (Fix 2)
                    w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)

                    f_tot_hat = float(np.sum(w))
                    if f_tot_hat > 1e-10:
                        f_fib_hat = float(np.sum(w[:n_dirs])) / f_tot_hat
                        f_res_hat = float(np.sum(
                            w[n_dirs:][iso_grid <= THRESH_RESTRICTED]
                        )) / f_tot_hat
                    else:
                        f_fib_hat = 0.0
                        f_res_hat = 0.0

                    ff_errors.append(abs(f_fib_hat - f_fiber))
                    rf_errors.append(abs(f_res_hat - f_cell))

                mae_ff = float(np.mean(ff_errors))
                mae_rf = float(np.mean(rf_errors))

                alpha = sc['loss_alpha']
                sc_losses[sc_name] = alpha * mae_ff + (1.0 - alpha) * mae_rf

            # ── Composite weighted loss ────────────────────────────────────────
            composite = (
                sum(_SCENARIOS[n]['weight'] * sc_losses[n] for n in sc_names)
                / _TOTAL_WEIGHT
            )

            row = (f"{n_iso:<6} | {reg:<7.2f} | "
                   + " | ".join(f"{sc_losses[n]:<{col_w}.4f}" for n in sc_names)
                   + f" | {composite:>10.4f}")
            print(row)

            results.append(dict(
                n_iso=n_iso,
                lambda_=reg,
                composite=composite,
                sc_losses=dict(sc_losses),
            ))

    print(sep)

    # ── Selection ─────────────────────────────────────────────────────────────
    best_res = min(results, key=lambda r: r['composite'])

    # Efficient best: fewest bases within 5% of the global minimum
    threshold     = best_res['composite'] * 1.05
    efficient_res = best_res
    for r in sorted(results, key=lambda r: r['n_iso']):
        if r['composite'] <= threshold:
            efficient_res = r
            break

    print(f"\n  ABSOLUTE BEST : n_iso={best_res['n_iso']:>3},  "
          f"λ={best_res['lambda_']:.2f}  "
          f"(composite = {best_res['composite']:.4f})")
    print(f"  EFFICIENT BEST: n_iso={efficient_res['n_iso']:>3},  "
          f"λ={efficient_res['lambda_']:.2f}  "
          f"(composite = {efficient_res['composite']:.4f})")

    print(f"\n  Per-scenario losses at EFFICIENT BEST:")
    for sc_name in sc_names:
        sc   = _SCENARIOS[sc_name]
        loss = efficient_res['sc_losses'][sc_name]
        note = " ← FF suppression only" if sc['loss_alpha'] >= 1.0 else ""
        print(f"    {sc_name:<22}  loss={loss:.4f}  "
              f"(w={sc['weight']:.2f}, α={sc['loss_alpha']:.2f}){note}")

    print(f"\n  → Using Efficient Best: n_iso={efficient_res['n_iso']}, "
          f"λ={efficient_res['lambda_']:.2f}")
    print(sep + "\n")

    return efficient_res['n_iso'], efficient_res['lambda_']
