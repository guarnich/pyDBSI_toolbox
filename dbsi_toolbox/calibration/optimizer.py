"""
DBSI Calibration: Hyperparameter Optimization — Multi-Scenario Version
=======================================================================

Monte Carlo optimization to find optimal (n_iso, reg_lambda) hyperparameters
that are robust across the full range of brain tissue types:

    WM       — white matter fibers (high FF, low RF)
    GM       — grey matter / cortex (FF ≈ 0, intermediate ADC ~0.8e-3)
    CSF      — free water (FF = 0, high ADC ~3.0e-3)
    Lesion   — demyelinated white matter (moderate FF, elevated RF)
    Mixed    — partial-volume boundary voxels

Scoring
-------
The composite loss is a weighted sum of per-scenario MAE values:

    L = w_WM  · MAE_RF_WM   +  w_GM  · MAE_FF_GM
      + w_CSF · MAE_FF_CSF  +  w_LES · MAE_RF_LES

where:
    MAE_RF_WM   = |RF_estimated - RF_true|  for WM scenario (target: RF accuracy)
    MAE_FF_GM   = |FF_estimated - 0|        for GM scenario  (target: FF = 0)
    MAE_FF_CSF  = |FF_estimated - 0|        for CSF scenario (target: FF = 0)
    MAE_RF_LES  = |RF_estimated - RF_true|  for lesion scenario

Differentiated Regularization
------------------------------
The calibration applies the SAME differentiated regularization scheme used in
the main fitting kernel:

    reg_vec[:n_dirs] = lambda * n_dirs   (fiber penalty, scaled by dictionary size)
    reg_vec[n_dirs:] = lambda            (isotropic penalty)

The solver is called with reg=0.0 so the penalty is not counted twice.
This ensures that the (n_iso, lambda) pair selected here is directly applicable
to the main fitting loop without re-scaling.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590–3601.
"""

import numpy as np
from ..core.basis import build_design_matrix, generate_fibonacci_sphere_hemisphere
from ..core.solvers import nnls_coordinate_descent


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match model constants exactly
# ─────────────────────────────────────────────────────────────────────────────

THRESH_RESTRICTED = 0.3e-3   # mm²/s — RF/HF boundary
THRESH_FREE       = 3.0e-3   # mm²/s — HF/WF boundary

# Tissue-type ground-truth fractions (approximate physiological values)
_SCENARIOS = {
    # name: (f_fiber, f_cell, f_hindered, f_free, label, weight)
    #   f_hindered + f_free = 1 - f_fiber - f_cell (enforced in signal gen)
    #   weight: relative importance in composite loss
    'WM_typical':   dict(f_fiber=0.50, f_cell=0.05, f_hin=0.40, f_free=0.05, w=1.0),
    'WM_inflamed':  dict(f_fiber=0.35, f_cell=0.30, f_hin=0.30, f_free=0.05, w=1.0),
    'GM_cortex':    dict(f_fiber=0.00, f_cell=0.05, f_hin=0.85, f_free=0.10, w=1.5),
    'GM_deep':      dict(f_fiber=0.10, f_cell=0.05, f_hin=0.75, f_free=0.10, w=1.0),
    'CSF':          dict(f_fiber=0.00, f_cell=0.00, f_hin=0.02, f_free=0.98, w=1.5),
    'Lesion_core':  dict(f_fiber=0.20, f_cell=0.40, f_hin=0.30, f_free=0.10, w=1.0),
    'Mixed_WM_GM':  dict(f_fiber=0.25, f_cell=0.05, f_hin=0.60, f_free=0.10, w=0.5),
}

# ADC values for each tissue pool
_D_CELL = 0.1e-3    # restricted (cells/axons)
_D_HIN  = 0.9e-3    # hindered (extracellular)
_D_FREE = 3.0e-3    # free water (CSF)
_D_AX   = 1.7e-3    # axial diffusivity (fiber)
_D_RAD  = 0.3e-3    # radial diffusivity (fiber)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_signal(bvals, bvecs, snr, f_fiber=0.5, f_cell=0.3):
    """
    Generate synthetic DBSI signal with Rician noise.

    Legacy interface kept for backward compatibility. Internally wraps
    _generate_signal_full() with f_hin = 1 - f_fiber - f_cell, f_free = 0.

    Parameters
    ----------
    bvals : ndarray (N,)
    bvecs : ndarray (N, 3)
    snr   : float
    f_fiber, f_cell : float

    Returns
    -------
    signal : ndarray (N,)
    """
    f_hin  = max(0.0, 1.0 - f_fiber - f_cell)
    f_free = 0.0
    return _generate_signal_full(bvals, bvecs, snr,
                                 f_fiber, f_cell, f_hin, f_free)


def _generate_signal_full(bvals, bvecs, snr,
                           f_fiber, f_cell, f_hin, f_free):
    """
    Generate synthetic DBSI signal for a four-pool tissue model.

    Pools
    -----
    fiber   : cylindrically symmetric tensor (AD=1.7e-3, RD=0.3e-3)
    cell    : restricted isotropic (D=0.1e-3)
    hin     : hindered isotropic   (D=0.9e-3)
    free    : free water isotropic (D=3.0e-3)

    Rician noise is added at the requested SNR (sigma = 1/SNR).
    """
    N = len(bvals)

    # Random fiber direction on the upper hemisphere
    v = np.random.randn(3)
    v /= np.linalg.norm(v)
    if v[2] < 0:
        v = -v

    s = np.zeros(N)
    for i in range(N):
        b = bvals[i]
        cos_t = np.dot(bvecs[i], v) if b >= 50 else 0.0
        D_app = _D_RAD + (_D_AX - _D_RAD) * cos_t**2

        s[i] = (f_fiber * np.exp(-b * D_app)
                + f_cell  * np.exp(-b * _D_CELL)
                + f_hin   * np.exp(-b * _D_HIN)
                + f_free  * np.exp(-b * _D_FREE))

    sigma = 1.0 / max(snr, 1.0)
    n1 = np.random.normal(0, sigma, N)
    n2 = np.random.normal(0, sigma, N)
    return np.sqrt((s + n1)**2 + n2**2)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_hyperparameters(bvals, bvecs, snr, n_mc=1000):
    """
    Multi-scenario Monte Carlo optimization for (n_iso, reg_lambda).

    The optimization minimizes a weighted composite loss across all tissue
    scenarios. For fiber-containing scenarios the target metric is RF accuracy;
    for pure isotropic scenarios (GM, CSF) the target is FF suppression (FF → 0).

    The differentiated regularization scheme (fiber columns penalized n_dirs×
    more than isotropic columns) is applied here exactly as in the main kernel,
    so the selected hyperparameters are directly transferable.

    Parameters
    ----------
    bvals  : ndarray (N,)
    bvecs  : ndarray (N, 3)
    snr    : float
    n_mc   : int — Monte Carlo samples per (n_iso, lambda, scenario) cell

    Returns
    -------
    best_n_iso   : int
    best_lambda  : float
    """
    print(f"\n[CALIBRATION REPORT — Multi-Scenario]")
    print(f"  Scenarios: {', '.join(_SCENARIOS.keys())}")
    print(f"  SNR={snr:.1f}  |  MC_Iterations={n_mc} per cell")
    print(f"  Regularization: differentiated (λ_fiber = λ × n_dirs,  λ_iso = λ)")
    print("=" * 100)

    # Header
    hdr = (f"{'Bases':<6} | {'Lambda':<8} | "
           + " | ".join(f"{k:<14}" for k in _SCENARIOS.keys())
           + " | {'Composite':>10}")
    print(hdr)
    print("-" * len(hdr))

    # Grid
    bases_grid   = [25, 50, 75, 100, 125, 150, 200]
    lambdas_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]

    n_dirs     = 100
    fiber_dirs = generate_fibonacci_sphere_hemisphere(n_dirs)

    # Pre-generate signals for all scenarios (fixed seed → reproducible)
    np.random.seed(42)
    scenario_signals = {}
    for sc_name, sc in _SCENARIOS.items():
        sigs = []
        for _ in range(n_mc):
            sigs.append(_generate_signal_full(
                bvals, bvecs, snr,
                sc['f_fiber'], sc['f_cell'], sc['f_hin'], sc['f_free']
            ))
        scenario_signals[sc_name] = np.array(sigs)   # (n_mc, N)

    results = []

    for n_iso in bases_grid:
        iso_grid = np.linspace(0, 4.5e-3, n_iso)
        A   = build_design_matrix(bvals, bvecs, fiber_dirs, iso_grid)
        AtA = A.T @ A
        At  = A.T

        for reg in lambdas_grid:
            # ── Differentiated regularization — identical to main kernel ──
            reg_vec = np.ones(AtA.shape[0])
            reg_vec[:n_dirs] = reg * n_dirs   # fiber columns
            reg_vec[n_dirs:] = reg            # isotropic columns
            AtA_reg = AtA + np.diag(reg_vec)

            sc_losses = {}

            for sc_name, sc in _SCENARIOS.items():
                sigs = scenario_signals[sc_name]
                gt_ff   = sc['f_fiber']
                gt_cell = sc['f_cell']

                ff_errors  = []
                rf_errors  = []

                for i in range(n_mc):
                    y   = sigs[i]
                    Aty = At @ y

                    # reg=0.0: penalty already embedded in AtA_reg
                    w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)

                    w_fib = w[:n_dirs]
                    w_iso = w[n_dirs:]

                    f_fib_raw = np.sum(w_fib)
                    f_res_raw = np.sum(
                        w_iso[iso_grid <= THRESH_RESTRICTED]
                    )
                    f_tot = np.sum(w)

                    if f_tot > 1e-10:
                        f_fib_hat = f_fib_raw / f_tot
                        f_res_hat = f_res_raw / f_tot
                    else:
                        f_fib_hat = 0.0
                        f_res_hat = 0.0

                    ff_errors.append(abs(f_fib_hat - gt_ff))
                    rf_errors.append(abs(f_res_hat - gt_cell))

                mae_ff = float(np.mean(ff_errors))
                mae_rf = float(np.mean(rf_errors))

                # Per-scenario loss:
                #   fiber voxels  → RF accuracy is primary
                #   isotropic vox → FF suppression is primary
                if gt_ff > 0.15:
                    sc_loss = 0.5 * mae_rf + 0.5 * mae_ff
                else:
                    # GM / CSF: heavily penalize spurious FF
                    sc_loss = 0.8 * mae_ff + 0.2 * mae_rf

                sc_losses[sc_name] = sc_loss

            # Composite weighted loss
            composite = sum(
                _SCENARIOS[sc_name]['w'] * sc_losses[sc_name]
                for sc_name in _SCENARIOS
            )
            total_weight = sum(sc['w'] for sc in _SCENARIOS.values())
            composite /= total_weight

            # Print row
            sc_str = " | ".join(f"{sc_losses[k]:<14.4f}" for k in _SCENARIOS)
            print(f"{n_iso:<6} | {reg:<8.2f} | {sc_str} | {composite:>10.4f}")

            results.append(dict(
                n_iso=n_iso,
                lambda_=reg,
                composite=composite,
                sc_losses=sc_losses,
            ))

    print("=" * 100)

    # ── Selection ─────────────────────────────────────────────────────────
    best_res = min(results, key=lambda r: r['composite'])

    # Efficient best: fewest bases within 5% of best composite loss
    threshold     = best_res['composite'] * 1.05
    efficient_res = best_res
    for r in sorted(results, key=lambda r: r['n_iso']):
        if r['composite'] <= threshold:
            efficient_res = r
            break

    print(f"\n  ABSOLUTE BEST : n_iso={best_res['n_iso']:>3}, "
          f"λ={best_res['lambda_']:.2f}  "
          f"(composite={best_res['composite']:.4f})")
    print(f"  EFFICIENT BEST: n_iso={efficient_res['n_iso']:>3}, "
          f"λ={efficient_res['lambda_']:.2f}  "
          f"(composite={efficient_res['composite']:.4f})")
    print("  → Using Efficient Best configuration.")
    print("=" * 100 + "\n")

    return efficient_res['n_iso'], efficient_res['lambda_']
