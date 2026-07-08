"""
DBSI Adaptive Model v3 — Hybrid Two-Stage Architecture
=========================================================

WHY v2's SINGLE-STAGE EXHAUSTIVE DICTIONARY WAS REPLACED
-------------------------------------------------------------
v2 attempted to estimate fiber orientation AND (AD, RD) simultaneously
from one linear NNLS solve over an exhaustive (direction x AD/RD-pair)
dictionary, taking AD_final/RD_final as NNLS-weighted centroids over the
activated anisotropic columns. Systematic synthetic recovery validation
(55 swept configurations, `recovery_validation.py`) demonstrated this is
NOT numerically identifiable: median AD/RD relative errors ranged from
~20% to >150% across every tested dictionary density, getting WORSE as
the dictionary was made finer (more candidate (AD,RD) pairs -> more
simultaneously-activated columns -> a less informative centroid), and no
regularization strength fixed it. This is a structural collinearity
failure, not a tuning problem.

v3 ARCHITECTURE: STAGE A (detection) + STAGE B (estimation)
-----------------------------------------------------------------
The core idea Alonso Ramirez-Manzanares's feedback motivated — that the
dictionary should "know" pathology changes AD/RD, not just orientation —
is preserved, but the two questions (which direction? what diffusivity?)
are now answered by two separate, appropriately-sized linear problems:

  STAGE A — direction detection (sparse, exhaustive dictionary)
      A small exhaustive (direction x AD/RD-pair) dictionary (deliberately
      coarse on the AD/RD axis — Stage A does not need fine diffusivity
      resolution, only fiber/no-fiber detection per direction) is fit via
      regularized NNLS with heavy sparsity on the anisotropic block
      (lambda_aniso). `core.solvers.select_dominant_directions` collapses
      the per-pair weight breakdown and reports which 1-2 hemisphere
      directions carry meaningful weight. We do NOT trust Stage A's
      (AD, RD) breakdown — only its directional answer.

  STAGE B — diffusivity estimation (closed-form, conditioned)
      Given Stage A's selected direction(s), `core.solvers.
      estimate_AD_RD_conditioned` performs a small closed-form weighted
      least-squares fit (the same analytical construction validated as
      the v1/v2 linear AD/RD initialisation) to obtain the final AD/RD.
      Because direction is now FIXED rather than searched, this problem
      has only 2 free parameters and is well-conditioned regardless of
      how rich Stage A's dictionary was.

There is still no non-linear Step 2 grid search: Stage B's closed-form
estimate is the final value, not an initial guess for further
refinement.

Synthetic validation summary (see project records for full sweep)
-----------------------------------------------------------------------
With a coarse Stage A dictionary (e.g. ~30 directions x 3x3 AD/RD pairs)
and lambda_base ~ 0.005 (lambda_aniso = lambda_base * n_aniso_cols,
lambda_iso = lambda_base): direction recovery cosine similarity ~1.0
across randomized ground truth; median AD relative error ~10-20%, median
RD relative error ~15-25%, with wider (but bounded, unlike v2) upper-tail
errors. This is a substantial improvement over v2 but AD/RD precision —
especially RD, the demyelination marker — should still be treated with
appropriate caution; see project validation notes before reporting
voxel-wise AD/RD as a precise quantitative biomarker without further
protocol-specific validation.

Compartment Definitions (unchanged from v1/v2)
---------------------------------------------------
    RF  : ADC <= THRESH_RES  (0.3 x 10^-3 mm^2/s) — cells, inflammatory infiltrate
    HF  : THRESH_RES < ADC <= THRESH_WAT          — hindered extracellular water
    WF  : ADC > THRESH_WAT  (3.0 x 10^-3 mm^2/s)  — free water, CSF, oedema
    NRF : HF + WF  (Non-Restricted Fraction, 2-ISO mode only)

Model Selection Criterion (unchanged from v1/v2)
------------------------------------------------------
2-ISO vs 3-ISO selection based on b_max / shell count is unaffected by
the v3 architecture change — it governs the isotropic block only.

Output Channels (11 total — unified across both model modes, unchanged
layout from v1/v2)
-------------------------------------------------------------------------
    0  : FF   — Fibre fraction                            (always valid)
    1  : RF   — Restricted fraction  (ADC <= 0.3e-3)      (always valid)
    2  : HF   — Hindered fraction    (0.3e-3 < ADC <= 3.0e-3) (NaN in 2-ISO mode)
    3  : WF   — Free-water fraction  (ADC > 3.0e-3)      (NaN in 2-ISO mode)
    4  : NRF  — Non-Restricted fraction = HF + WF        (always valid)
    5  : AD   — Axial diffusivity   (v3: Stage B closed-form estimate;
                NaN if FF <= fiber_threshold)
    6  : RD   — Radial diffusivity  (v3: Stage B closed-form estimate;
                NaN if FF <= fiber_threshold)
    7  : FA   — Intrinsic fibre FA  (computed from the v3 Stage B AD/RD;
                NaN if FF <= fiber_threshold)
    8  : ADC_iso — Mean isotropic ADC                     (always valid)
    9  : AD_lin  — v3: identical to channel 5 (retained for output-shape
                compatibility; Stage B's estimate is the only diffusivity
                estimate produced, there is no separate "linear" vs.
                "refined" pair any more)
    10 : RD_lin  — v3: identical to channel 6 (see note above)

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601. doi:10.1093/brain/awr307
Shirani A, et al. (2019). Ann Clin Transl Neurol, 6(11):2323-2327.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33-47.
Design document: toolbox_v2.md (Ramirez-Manzanares discussion); v3
hybrid redesign motivated by synthetic recovery validation of the v2
single-stage approach.
"""

import numpy as np
from numba import njit, prange
import time
from tqdm import tqdm

from .core.basis import (
    build_design_matrix_exhaustive,
    generate_exhaustive_diffusivity_pairs,
    generate_fibonacci_sphere_hemisphere,
    generate_isotropic_grid,
    generate_log_uniform_isotropic_grid,
    generate_anchored_isotropic_grid,
)
from .core.solvers import (
    nnls_coordinate_descent,
    compute_regularization_matrix,
    select_dominant_directions,
    estimate_AD_RD_conditioned,
    compute_fiber_fa,
    refine_fiber_direction_cone,
    compute_cone_refinement_schedule,
    measure_hemisphere_spacing,
)
from .calibration.optimizer import optimize_hyperparameters, evaluate_lambda_pair
from .calibration.data_driven import select_lambdas_data_driven, sample_calibration_voxels
from .calibration.adaptive_n_iso import select_n_iso_svd, select_n_iso_bootstrap

from .utils.tools import estimate_snr_robust
from .utils.autoconfig import autoconfigure_dictionary


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FIBER_THRESHOLD = 0.15      # dimensionless — minimum FF for AD/RD estimation

THRESH_RES = 0.3e-3          # mm^2/s — restricted / hindered boundary
THRESH_WAT = 3.0e-3          # mm^2/s — hindered  / free-water boundary

B_THRESH_3ISO = 3000.0       # s/mm^2 — minimum b_max to activate 3-ISO model
MIN_SHELLS_3ISO = 3          # minimum distinct non-zero b-value shells for 3-ISO

# Stage A dictionary defaults. Deliberately coarse on the AD/RD axis —
# Stage A only needs to detect WHICH directions are active, not estimate
# diffusivity precisely (that is Stage B's job). Synthetic validation
# showed direction-recovery cosine similarity ~1.0 essentially
# independent of (n_ad, n_rd) density, so the smallest grid that keeps
# the NNLS solve fast is preferred (solver cost scales worse than
# linearly with total column count under coordinate descent).
_STAGE_A_AD_MIN = 0.5e-3
_STAGE_A_AD_MAX = 2.2e-3
_STAGE_A_RD_MIN = 0.05e-3
_STAGE_A_RD_MAX = 1.2e-3
_STAGE_A_DEFAULT_N_AD = 3
_STAGE_A_DEFAULT_N_RD = 3
_STAGE_A_DEFAULT_ANISOTROPY_RATIO = 1.15
_STAGE_A_DEFAULT_LAMBDA_BASE = 0.005

# Default isotropic spectrum range.
_DEFAULT_ISO_MIN = 0.0
_DEFAULT_ISO_MAX = 3.0e-3
_DEFAULT_N_ISO_STEPS = 31    # Legacy fixed default — NO LONGER USED as the
                             # automatic default; see select_n_iso_svd for the
                             # adaptive default now used when n_iso is left at
                             # None. Retained only for explicit external
                             # reference / backward compatibility.

# Extended upper bound for the ADAPTIVE isotropic grid's d_max, used
# only on the adaptive (n_iso=None) path — see fit()'s grid construction
# block. Project methodological supplement "isotropic_compartment_
# supplement.docx" (Section 4) mapped the HIN/WAT transition zone as
# extending to approximately 5.0e-3 mm^2/s on the high side; extending
# the grid this far ensures at least one column genuinely represents
# free water above the 3.0e-3 threshold rather than relying solely on a
# single anchor column exactly at the boundary.
_ISO_GRID_D_MAX_EXTENDED = 5.0e-3

# Maximum number of fiber populations Stage A will report per voxel.
# 1 = single dominant tract only (matches v1/v2 single-tensor output
# layout, which has no per-population channel structure). Set to 2 to
# allow crossing-fiber detection internally, but the output channels
# still report only the dominant (highest-weight) population, since the
# 11-channel layout has no slot for a second tensor.
_MAX_FIBER_POPULATIONS = 2


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL ANALYSIS UTILITY (unchanged from v1/v2)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_protocol(bvals):
    """
    Analyse the diffusion acquisition scheme and determine which isotropic
    model is appropriate. Unchanged from v1/v2.
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    b_max = float(np.max(bvals))

    rounded = np.round(bvals, -2)
    unique_nonzero = np.unique(rounded[rounded > 50])
    n_nonzero_shells = int(len(unique_nonzero))

    if b_max < B_THRESH_3ISO and n_nonzero_shells < MIN_SHELLS_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: b_max = {b_max:.0f} s/mm^2 < {B_THRESH_3ISO:.0f} s/mm^2 "
            f"AND only {n_nonzero_shells} non-zero shell(s) (minimum {MIN_SHELLS_3ISO} required "
            f"for 3-ISO). Free-water signal below noise floor; NRF = HF + WF merged."
        )
    elif b_max < B_THRESH_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: b_max = {b_max:.0f} s/mm^2 < {B_THRESH_3ISO:.0f} s/mm^2. "
            f"At this b_max, exp(-b_max x D_free) = {np.exp(-b_max * THRESH_WAT):.4f}, "
            f"placing the free-water signal near the noise floor at typical SNR. "
            f"NRF = HF + WF merged for numerical stability."
        )
    elif n_nonzero_shells < MIN_SHELLS_3ISO:
        use_3iso = False
        reason = (
            f"2-ISO model selected: only {n_nonzero_shells} distinct non-zero b-shell(s) "
            f"detected (minimum {MIN_SHELLS_3ISO} required for 3-ISO). "
            f"Insufficient shell diversity to constrain HF/WF separation via NNLS. "
            f"NRF = HF + WF merged."
        )
    else:
        use_3iso = True
        reason = (
            f"3-ISO model selected: b_max = {b_max:.0f} s/mm^2 >= {B_THRESH_3ISO:.0f} s/mm^2 "
            f"with {n_nonzero_shells} distinct non-zero shells. "
            f"Sufficient b-range and shell diversity to resolve RF, HF, and WF separately."
        )

    return b_max, n_nonzero_shells, use_3iso, reason


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL FITTING KERNELS — v3: Stage A (detection) + Stage B (estimation)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_2iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        out):
    """
    v3 parallel fitting kernel — two-compartment isotropic model (2-ISO),
    Stage A direction detection + Stage B closed-form diffusivity
    estimation, optionally followed by MRDS-lite cone refinement of the
    dominant direction (see core.solvers module docstring "MRDS-LITE").

    AtA_reg is Stage A's regularized Gram matrix (decoupled
    lambda_aniso/lambda_iso). Output layout identical to v1/v2 (see
    module docstring for v3 semantics of channels 5-10).
    """
    n_voxels = coords.shape[0]
    n_pairs = len(diff_pairs)
    n_aniso_cols = n_dirs * n_pairs
    n_iso = len(iso_grid)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]

        s0 = 0.0
        cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue

        sig_norm = sig / s0

        # ── STAGE A: regularized NNLS over the exhaustive detection dictionary ──
        Aty = np.zeros(AtA_reg.shape[0])
        for r in range(AtA_reg.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val

        w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
        w_aniso = w[:n_aniso_cols]
        w_iso = w[n_aniso_cols:]

        f_fib_raw = 0.0
        for i in range(n_aniso_cols):
            f_fib_raw += w_aniso[i]

        f_res_raw = 0.0
        f_nonrf_raw = 0.0
        sum_w_iso = 0.0
        sum_wd_iso = 0.0
        sum_res_w = 0.0
        sum_res_wd = 0.0
        sum_nonrf_w = 0.0
        sum_nonrf_wd = 0.0

        for i in range(n_iso):
            adc = iso_grid[i]
            wi = w_iso[i]
            if adc <= THRESH_RES:
                f_res_raw += wi
                sum_res_w += wi
                sum_res_wd += wi * adc
            else:
                f_nonrf_raw += wi
                sum_nonrf_w += wi
                sum_nonrf_wd += wi * adc
            sum_w_iso += wi
            sum_wd_iso += wi * adc

        mean_iso_adc = sum_wd_iso / sum_w_iso if sum_w_iso > 1e-10 else 0.0
        D_res_c = sum_res_wd / sum_res_w if sum_res_w > 1e-10 else 0.15e-3
        D_nonrf_c = sum_nonrf_wd / sum_nonrf_w if sum_nonrf_w > 1e-10 else 1.5e-3

        ftot = f_fib_raw + f_res_raw + f_nonrf_raw
        if ftot < 1e-10:
            continue

        f_fib = f_fib_raw / ftot
        f_res = f_res_raw / ftot
        f_nonrf = f_nonrf_raw / ftot

        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 4] = f_nonrf
        out[x, y, z, 8] = mean_iso_adc

        if f_fib > fiber_threshold:
            # ── STAGE A interpretation: which direction(s) are active? ──
            dir_indices, dir_weights = select_dominant_directions(
                w_aniso, n_dirs, n_pairs, _MAX_FIBER_POPULATIONS, min_weight_fraction
            )

            if dir_indices[0] >= 0:
                dominant_dir = fiber_dirs[dir_indices[0]]

                if enable_direction_refinement:
                    # ── MRDS-LITE: refine the coarse grid direction via
                    # two-level cone search, scored by Stage B's own
                    # closed-form regression (see core.solvers module
                    # docstring "MRDS-LITE"). Never worse than the
                    # unrefined estimate (candidate zero is the coarse
                    # direction itself).
                    _, AD_est, RD_est = refine_fiber_direction_cone(
                        bvals, bvecs, sig_norm, dominant_dir,
                        f_fib, f_res, f_nonrf, 0.0,
                        D_res_c, D_nonrf_c, 0.0, False,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
                    )
                else:
                    # ── STAGE B: closed-form (AD, RD) conditioned on direction ──
                    AD_est, RD_est = estimate_AD_RD_conditioned(
                        bvals, bvecs, sig_norm, dominant_dir,
                        f_fib, f_res, f_nonrf, 0.0,
                        D_res_c, D_nonrf_c, 0.0, False
                    )

                FA = np.nan
                if not np.isnan(AD_est) and not np.isnan(RD_est):
                    FA = compute_fiber_fa(AD_est, RD_est)

                out[x, y, z, 5] = AD_est
                out[x, y, z, 6] = RD_est
                out[x, y, z, 7] = FA
                out[x, y, z, 9] = AD_est
                out[x, y, z, 10] = RD_est


@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_3iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        out):
    """
    v3 parallel fitting kernel — three-compartment isotropic model
    (3-ISO). Same Stage A / Stage B structure as `_fit_voxels_2iso_v3`,
    including the optional MRDS-lite direction refinement (see
    core.solvers module docstring "MRDS-LITE").
    """
    n_voxels = coords.shape[0]
    n_pairs = len(diff_pairs)
    n_aniso_cols = n_dirs * n_pairs
    n_iso = len(iso_grid)

    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data[x, y, z]

        s0 = 0.0
        cnt = 0
        for i in range(len(bvals)):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6:
            continue

        sig_norm = sig / s0

        Aty = np.zeros(AtA_reg.shape[0])
        for r in range(AtA_reg.shape[0]):
            val = 0.0
            for c in range(len(sig_norm)):
                val += At[r, c] * sig_norm[c]
            Aty[r] = val

        w, _ = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
        w_aniso = w[:n_aniso_cols]
        w_iso = w[n_aniso_cols:]

        f_fib_raw = 0.0
        for i in range(n_aniso_cols):
            f_fib_raw += w_aniso[i]

        f_res_raw = 0.0
        f_hin_raw = 0.0
        f_wat_raw = 0.0
        sum_w_iso = 0.0
        sum_wd_iso = 0.0
        sum_res_w = 0.0
        sum_res_wd = 0.0
        sum_hin_w = 0.0
        sum_hin_wd = 0.0
        sum_wat_w = 0.0
        sum_wat_wd = 0.0

        for i in range(n_iso):
            adc = iso_grid[i]
            wi = w_iso[i]
            if adc <= THRESH_RES:
                f_res_raw += wi
                sum_res_w += wi
                sum_res_wd += wi * adc
            elif adc <= THRESH_WAT:
                f_hin_raw += wi
                sum_hin_w += wi
                sum_hin_wd += wi * adc
            else:
                f_wat_raw += wi
                sum_wat_w += wi
                sum_wat_wd += wi * adc
            sum_w_iso += wi
            sum_wd_iso += wi * adc

        mean_iso_adc = sum_wd_iso / sum_w_iso if sum_w_iso > 1e-10 else 0.0
        D_res_c = sum_res_wd / sum_res_w if sum_res_w > 1e-10 else 0.15e-3
        D_hin_c = sum_hin_wd / sum_hin_w if sum_hin_w > 1e-10 else 1.00e-3
        D_wat_c = sum_wat_wd / sum_wat_w if sum_wat_w > 1e-10 else 3.00e-3

        ftot = f_fib_raw + f_res_raw + f_hin_raw + f_wat_raw
        if ftot < 1e-10:
            continue

        f_fib = f_fib_raw / ftot
        f_res = f_res_raw / ftot
        f_hin = f_hin_raw / ftot
        f_wat = f_wat_raw / ftot

        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 2] = f_hin
        out[x, y, z, 3] = f_wat
        out[x, y, z, 4] = f_hin + f_wat
        out[x, y, z, 8] = mean_iso_adc

        if f_fib > fiber_threshold:
            dir_indices, dir_weights = select_dominant_directions(
                w_aniso, n_dirs, n_pairs, _MAX_FIBER_POPULATIONS, min_weight_fraction
            )

            if dir_indices[0] >= 0:
                dominant_dir = fiber_dirs[dir_indices[0]]

                if enable_direction_refinement:
                    _, AD_est, RD_est = refine_fiber_direction_cone(
                        bvals, bvecs, sig_norm, dominant_dir,
                        f_fib, f_res, f_hin, f_wat,
                        D_res_c, D_hin_c, D_wat_c, True,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
                    )
                else:
                    AD_est, RD_est = estimate_AD_RD_conditioned(
                        bvals, bvecs, sig_norm, dominant_dir,
                        f_fib, f_res, f_hin, f_wat,
                        D_res_c, D_hin_c, D_wat_c, True
                    )

                FA = np.nan
                if not np.isnan(AD_est) and not np.isnan(RD_est):
                    FA = compute_fiber_fa(AD_est, RD_est)

                out[x, y, z, 5] = AD_est
                out[x, y, z, 6] = RD_est
                out[x, y, z, 7] = FA
                out[x, y, z, 9] = AD_est
                out[x, y, z, 10] = RD_est


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DBSI_Adaptive:
    """
    Adaptive DBSI model (v3) using a hybrid two-stage anisotropic
    estimation: Stage A detects dominant fiber direction(s) from an
    exhaustive (direction x AD/RD-pair) dictionary under heavy sparsity
    regularization; Stage B estimates AD/RD via closed-form WLS
    conditioned on the detected direction(s).

    Selection rule (isotropic block — unchanged from v1/v2)
    -------------------------------------------------------------
    3-ISO (RF + HF + WF) if:  b_max >= B_THRESH_3ISO  AND  n_shells >= MIN_SHELLS_3ISO
    2-ISO (RF + NRF)     otherwise.

    Stage A dictionary sizing
    ------------------------------
    By default, M (hemisphere directions) is derived automatically from
    the acquisition protocol via `utils.autoconfig.autoconfigure_dictionary`
    (same logic as v2). n_ad, n_rd, anisotropy_ratio default to a coarse
    Stage-A-appropriate grid (3x3, ratio 1.15) rather than the
    protocol-scaled denser grids v2 used, because Stage A's job
    (direction detection) does not benefit from a finer (AD,RD) grid —
    see module docstring.

    Parameters
    ----------
    n_iso : int or None
        Number of isotropic ADC basis points. Defaults to 31 if None.
    lambda_aniso : float or None
        Stage A regularisation strength for the anisotropic block
        (auto-calibrated if None).
    lambda_iso : float or None
        Stage A regularisation strength for the isotropic block
        (auto-calibrated if None).
    n_dirs : int or None
        Number of fibre directions on the Fibonacci hemisphere for
        Stage A. If None, derived automatically from the protocol.
    n_ad, n_rd : int or None
        Number of AD / RD grid steps for Stage A's detection dictionary.
        Default: 3, 3 (deliberately coarse — see module docstring).
    anisotropy_ratio : float or None
        Minimum AD/RD ratio for admissible Stage A pairs. Default: 1.15.
    ad_range, rd_range : tuple (float, float)
        Physical bounds for Stage A's AD / RD grids, mm^2/s.
    iso_range : tuple (float, float)
        ADC range [mm^2/s] of the isotropic basis. Default: (0.0, 3.0e-3).
    fiber_threshold : float
        Minimum fibre fraction for AD/RD/FA estimation. Default: 0.15.
    min_weight_fraction : float
        Minimum fraction of total Stage A anisotropic weight a direction
        must carry to be reported as a fiber population. Default: 0.05.
    force_n_iso : int or None
        Override automatic isotropic-model selection (2 or 3).
    enable_direction_refinement : bool
        Whether to refine Stage A's dominant direction estimate with a
        two-level "MRDS-lite" cone search (Coronado-Leija et al. 2017,
        scoped down to single-direction refinement — see
        `core.solvers` module docstring "MRDS-LITE" for full rationale).
        Default: True. Project validation (synthetic, Verona-protocol
        reconstruction) found this closes ~60-90% of the AD accuracy gap
        attributable to Stage A's fixed grid discretisation (median AD
        relative error dropped from ~6.8-12.6% to ~2.6-7.6% depending on
        test conditions), at negligible added computational cost (a few
        dozen closed-form regressions per fiber voxel, versus the NNLS
        solve's own up to 2000 iterations on a much larger system).
        RD is largely unaffected either way (the closed-form regression's
        RD term is markedly less sensitive to angular offset than AD).
        NOT validated on real (non-synthetic) data or on multi-fiber
        crossing configurations.
    target_angular_resolution_deg : float
        Desired final angular precision (degrees) of the refined
        direction. Default: 1.0. Used to derive the two-level cone
        search schedule (cone angles and candidate counts) directly from
        the ACTUAL measured spacing of the Stage A hemisphere dictionary
        — see `core.solvers.compute_cone_refinement_schedule`. Only used
        if `enable_direction_refinement=True`.

    Notes
    -----
    There is no `enable_step2` parameter: the non-linear Step 2
    refinement stage from v1 has been eliminated since v2 and remains
    eliminated in v3. AD/RD are obtained as Stage B's closed-form
    estimate conditioned on Stage A's detected direction.

    (lambda_aniso, lambda_iso) calibration defaults to the data-driven
    method (GCV + discrepancy principle, see `calibration.data_driven`)
    rather than the legacy Monte Carlo tissue-scenario grid search — see
    `fit()`'s `calibration_method` parameter. After fitting,
    `self.mc_crosscheck_report_` holds the Monte Carlo cross-check
    report (see `fit()`'s `run_mc_crosscheck` parameter) if that was
    requested, else `None`.
    """

    CH = {
        'FF': 0,
        'RF': 1,
        'HF': 2,
        'WF': 3,
        'NRF': 4,
        'AD': 5,
        'RD': 6,
        'FA': 7,
        'ADC_iso': 8,
        'AD_lin': 9,
        'RD_lin': 10,
    }
    N_CHANNELS = 11

    def __init__(self, n_iso=None, lambda_aniso=None, lambda_iso=None,
                 n_dirs=None, n_ad=_STAGE_A_DEFAULT_N_AD, n_rd=_STAGE_A_DEFAULT_N_RD,
                 anisotropy_ratio=_STAGE_A_DEFAULT_ANISOTROPY_RATIO,
                 ad_range=(_STAGE_A_AD_MIN, _STAGE_A_AD_MAX),
                 rd_range=(_STAGE_A_RD_MIN, _STAGE_A_RD_MAX),
                 iso_range=(_DEFAULT_ISO_MIN, _DEFAULT_ISO_MAX),
                 fiber_threshold=FIBER_THRESHOLD,
                 min_weight_fraction=0.05, force_n_iso=None,
                 enable_direction_refinement=True,
                 target_angular_resolution_deg=1.0):
        self.n_iso = n_iso
        self.lambda_aniso = lambda_aniso
        self.lambda_iso = lambda_iso
        self.n_dirs = n_dirs
        self.n_ad = n_ad
        self.n_rd = n_rd
        self.anisotropy_ratio = anisotropy_ratio
        self.ad_range = ad_range
        self.rd_range = rd_range
        self.iso_range = iso_range
        self.fiber_threshold = fiber_threshold
        self.min_weight_fraction = min_weight_fraction
        self.force_n_iso = force_n_iso
        self.enable_direction_refinement = enable_direction_refinement
        self.target_angular_resolution_deg = target_angular_resolution_deg

        self.model_mode_ = None
        self.b_max_ = None
        self.n_shells_ = None
        self.n_aniso_cols_ = None
        self.diff_pairs_ = None
        self.mc_crosscheck_report_ = None
        self.hemisphere_spacing_deg_ = None
        self.cone_refinement_schedule_ = None

    # ------------------------------------------------------------------
    def fit(self, data, bvals, bvecs, mask, run_calibration=True,
           calibration_method='data_driven', n_calibration_voxels=500,
           n_iso_method='bootstrap', n_bootstrap=50,
           run_mc_crosscheck=False, mc_crosscheck_n_mc=200,
           run_n_iso_sweep_diagnostic=False):
        """
        Fit the v3 hybrid two-stage adaptive DBSI model to 4D diffusion
        MRI data.

        Parameters
        ----------
        data : ndarray (X, Y, Z, N)
        bvals : array (N,)
        bvecs : array (N, 3) or (3, N)
        mask : ndarray (X, Y, Z) bool
        run_calibration : bool
            Whether to run calibration for (lambda_aniso, lambda_iso) at
            all (if False, falls back to the fixed default lambda_base
            heuristic; see class docstring).
        calibration_method : {'data_driven', 'monte_carlo'}
            Which method determines (lambda_aniso, lambda_iso). See
            previous docstring entries for full details.
        n_calibration_voxels : int
            Number of brain-mask voxels to sample. Default 500.
        n_iso_method : {'bootstrap', 'svd_floor', 'fixed'}
            How to select n_iso when self.n_iso is None (default None
            means auto):
              'bootstrap' (default) — minimise bias²+variance of RF
                across bootstrapped noise replicates of real sampled
                voxels (see `calibration.adaptive_n_iso.
                select_n_iso_bootstrap`). Validated on Verona protocol
                (4 shells, b_max=2000, SNR=28): consistently selects
                n_iso=8 across 5 independent sampling seeds, with the
                composite score 3-10x lower than all other candidates.
                Preferred over GCV-sweep (systematically flat for dMRI)
                and over fixed floor (not protocol-specific).
              'svd_floor' — SVD-based information limit with empirical
                floor=10 (the previous default). Protocol-agnostic; for
                typical dMRI protocols the floor dominates and the SVD
                component is mostly inactive (raw answer 2-5 for 1-7
                shell protocols). Retained as a fast fallback when
                bootstrap computational cost is a concern.
              'fixed' — use whatever value was passed to n_iso in the
                constructor (or the legacy default 31 if None). Provided
                only for backward compatibility / reproducibility of
                earlier pipeline runs.
        n_bootstrap : int
            Number of noise replicates per voxel for the bootstrap
            method (only used if n_iso_method='bootstrap'). Default 50;
            30 is adequate, >100 gives diminishing returns.
        run_mc_crosscheck : bool
            Monte Carlo tissue-scenario cross-check (see previous docs).
        mc_crosscheck_n_mc : int
            Samples per scenario for cross-check.

        Returns
        -------
        results : ndarray (X, Y, Z, 11)
        model_mode : int
            2 or 3.
        """
        print("\n" + "="*70)
        print("  DBSI ADAPTIVE PIPELINE — v3 (Hybrid Two-Stage Architecture)")
        print("="*70)

        bvecs = np.asarray(bvecs, dtype=np.float64)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms

        # ── Isotropic model selection ──────────────────────────────────────
        b_max, n_shells, use_3iso, reason = analyse_protocol(bvals)
        self.b_max_ = b_max
        self.n_shells_ = n_shells

        if self.force_n_iso is not None:
            if self.force_n_iso == 3:
                use_3iso = True
                print(f"\n  [MODEL OVERRIDE] force_n_iso=3 requested by user.")
            elif self.force_n_iso == 2:
                use_3iso = False
                print(f"\n  [MODEL OVERRIDE] force_n_iso=2 requested by user.")
            else:
                raise ValueError("force_n_iso must be 2, 3, or None.")

        model_mode = 3 if use_3iso else 2
        self.model_mode_ = model_mode

        print(f"\n  Isotropic model selection: {reason}")
        print(f"\n  Active model: {model_mode}-ISO "
              f"({'RF + HF + WF' if use_3iso else 'RF + NRF (HF+WF merged)'})")
        print(f"  b_max detected: {b_max:.0f} s/mm^2  |  "
              f"Non-zero shells: {n_shells}")

        # ── Stage A dictionary autoconfiguration ─────────────────────────
        print("\n1. Autoconfiguring Stage A detection dictionary...")
        M_auto, n_ad_auto, n_rd_auto, ratio_auto = autoconfigure_dictionary(bvals, bvecs)

        if self.n_dirs is None:
            self.n_dirs = M_auto
        # n_ad / n_rd / anisotropy_ratio intentionally do NOT default to
        # the protocol-scaled (potentially dense) autoconfigured values
        # here: Stage A is deliberately kept coarse on the (AD,RD) axis
        # regardless of protocol richness (see module docstring and class
        # docstring). Use the constructor's explicit n_ad/n_rd/
        # anisotropy_ratio arguments (default 3x3, ratio 1.15) unless the
        # caller overrides them.

        print(f"   M (hemisphere directions): {self.n_dirs}")
        print(f"   Stage A n_ad x n_rd grid: {self.n_ad} x {self.n_rd} "
              f"(deliberately coarse — detection only, not diffusivity recovery)")
        print(f"   anisotropy_ratio: {self.anisotropy_ratio:.2f}")

        diff_pairs = generate_exhaustive_diffusivity_pairs(
            ad_min=self.ad_range[0], ad_max=self.ad_range[1], n_ad=self.n_ad,
            rd_min=self.rd_range[0], rd_max=self.rd_range[1], n_rd=self.n_rd,
            anisotropy_ratio=self.anisotropy_ratio,
        )
        self.diff_pairs_ = diff_pairs
        n_pairs = len(diff_pairs)
        n_aniso_cols = self.n_dirs * n_pairs
        self.n_aniso_cols_ = n_aniso_cols

        print(f"   (AD, RD) admissible pairs: {n_pairs} / {self.n_ad * self.n_rd} "
              f"(after anisotropy_ratio filter)")
        print(f"   Stage A dictionary columns: {n_aniso_cols} "
              f"({self.n_dirs} dirs x {n_pairs} pairs)")

        fiber_dirs = generate_fibonacci_sphere_hemisphere(self.n_dirs)

        # ── MRDS-lite direction refinement schedule ─────────────────────────
        # Computed ONCE per protocol (not per voxel) from the ACTUAL Stage A
        # dictionary's own nearest-neighbour spacing — see core.solvers
        # module docstring "MRDS-LITE" for the full rationale and the
        # empirical validation (Verona-protocol reconstruction: closes the
        # AD accuracy gap from ~6.8-9% grid-quantisation bias down to
        # ~3%, matching a true-direction oracle, at n_dirs values ranging
        # from 30 to 62). No independently hardcoded cone angles.
        self.hemisphere_spacing_deg_ = float(np.degrees(measure_hemisphere_spacing(fiber_dirs)))
        if self.enable_direction_refinement:
            _cone1, _n1, _cone2, _n2 = compute_cone_refinement_schedule(
                np.radians(self.hemisphere_spacing_deg_),
                np.radians(self.target_angular_resolution_deg)
            )
            self.cone_refinement_schedule_ = dict(
                cone1_half_angle_deg=float(np.degrees(_cone1)), n1=int(_n1),
                cone2_half_angle_deg=float(np.degrees(_cone2)), n2=int(_n2),
            )
            print(f"\n   MRDS-lite direction refinement: ENABLED "
                  f"(target resolution={self.target_angular_resolution_deg:.2f} deg)")
            print(f"   Dictionary spacing: {self.hemisphere_spacing_deg_:.2f} deg  |  "
                  f"Level 1 cone: +/-{np.degrees(_cone1):.2f} deg ({_n1} candidates)  |  "
                  f"Level 2 cone: +/-{np.degrees(_cone2):.2f} deg ({_n2} candidates)")
        else:
            _cone1, _n1, _cone2, _n2 = 0.0, 1, 0.0, 0
            print(f"\n   MRDS-lite direction refinement: DISABLED "
                  f"(using raw Stage A grid direction, unrefined)")

        # ── SNR estimation ─────────────────────────────────────────────────
        print("\n2. Estimating SNR...")
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)

        # ── Isotropic grid ──────────────────────────────────────────────────
        # n_iso defaults to the SVD-based adaptive estimate (protocol- and
        # SNR-aware, with an empirically-validated floor — see
        # calibration.adaptive_n_iso module docstring) rather than the
        # legacy fixed value of 31.
        # ── Rician bias correction (vectorized, unchanged from v1/v2) ──────
        # Moved before n_iso selection AND calibration: both the bootstrap
        # n_iso method and the data-driven lambda calibration sample real
        # voxel signals from this dataset, and should see the SAME
        # corrected signal the actual Stage A/B fit will use — sampling
        # against uncorrected signal would target a subtly different
        # noise level than what is actually fit.
        print("\n3. Applying Rician Bias Correction...")
        coords = np.argwhere(mask)

        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        data_corr = np.zeros_like(data, dtype=np.float32)
        noise_floor = 2.0 * sigma**2
        masked_sq = data[xs, ys, zs].astype(np.float64)**2
        valid_mask = masked_sq > noise_floor
        corrected = np.where(valid_mask,
                             np.sqrt(np.maximum(masked_sq - noise_floor, 0.0)),
                             0.0).astype(np.float32)
        data_corr[xs, ys, zs] = corrected
        del masked_sq, valid_mask, corrected

        # ── Sample calibration voxels once, reused for n_iso selection
        # AND lambda calibration (avoids sampling twice / inconsistent
        # voxel sets between the two steps) ────────────────────────────
        y_cal, sigma_cal = None, None
        if self.n_iso is None or (run_calibration and
                                  (self.lambda_aniso is None or self.lambda_iso is None) and
                                  calibration_method == 'data_driven'):
            y_cal, sigma_cal = sample_calibration_voxels(
                data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
            )
            print(f"\n   Sampled {len(y_cal)} calibration voxels from the brain mask "
                  f"(sigma_normalised={sigma_cal:.5f})")

        # legacy fixed value of 31. The grid itself is THRESHOLD-ANCHORED
        # log-uniform (Borgia et al. 1998 spectral-resolution rationale,
        # plus exact-boundary anchoring — see
        # core.basis.generate_anchored_isotropic_grid docstring and
        # project methodological supplement "isotropic_compartment_
        # supplement.docx" Section 4) rather than plain linear when
        # n_iso is adaptively determined, since the two were validated
        # together in project synthetic testing.
        #
        # If the caller explicitly set n_iso in the constructor, that
        # exact value is honoured and the LINEAR grid is used instead
        # (matching the original v1/v2/v3 behaviour for explicit
        # overrides) — adaptive selection only engages when n_iso is left
        # at its default (None).
        iso_d_max = max(self.iso_range[1], _ISO_GRID_D_MAX_EXTENDED)

        if self.n_iso is None:
            if n_iso_method == 'bootstrap':
                print(f"\n4. Selecting n_iso — BOOTSTRAP bias-variance "
                      f"({n_bootstrap} replicates/voxel)...")
                self.n_iso, _n_iso_diag = select_n_iso_bootstrap(
                    bvals, y_cal, snr, sigma_cal,
                    d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max,
                    n_bootstrap=n_bootstrap,
                )
                if _n_iso_diag['curve_is_flat'] or _n_iso_diag['sample_looks_homogeneous']:
                    _reason = ("weak separation" if _n_iso_diag['curve_is_flat']
                              else "insufficient voxel tissue diversity")
                    print(f"   [WARNING] Bootstrap result unreliable ({_reason}) "
                          f"for this dataset; falling back to SVD+floor as a "
                          f"safer default.")
                    self.n_iso, _svd_diag = select_n_iso_svd(bvals, snr)
                    print(f"   SVD+floor fallback: n_iso={self.n_iso}")
                else:
                    print(f"   Bootstrap-selected n_iso={self.n_iso}")

            elif n_iso_method == 'svd_floor':
                self.n_iso, _svd_diag = select_n_iso_svd(bvals, snr)
                print(f"\n4. Selecting n_iso — SVD + empirical floor: "
                      f"n_iso={self.n_iso} (raw SVD answer: "
                      f"{_svd_diag['n_iso_raw']}, "
                      f"floor_applied={_svd_diag['floor_applied']})")

            elif n_iso_method == 'fixed':
                self.n_iso = _DEFAULT_N_ISO_STEPS
                print(f"\n4. n_iso fixed at legacy default: n_iso={self.n_iso}")

            else:
                raise ValueError(
                    f"n_iso_method must be 'bootstrap', 'svd_floor', or "
                    f"'fixed', got {n_iso_method!r}."
                )

            iso_grid = generate_anchored_isotropic_grid(
                d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max,
                n_steps=self.n_iso, thresh_res=THRESH_RES, thresh_wat=THRESH_WAT,
            )
        else:
            iso_grid = generate_isotropic_grid(
                d_min=self.iso_range[0], d_max=self.iso_range[1], n_steps=self.n_iso
            )

        # ── Calibration of (lambda_aniso, lambda_iso) ───────────────────────
        if run_calibration and (self.lambda_aniso is None or self.lambda_iso is None):

            if calibration_method == 'data_driven':
                print(f"\n5. Calibrating (lambda_aniso, lambda_iso) — DATA-DRIVEN "
                      f"(GCV + discrepancy principle)...")
                if y_cal is None:
                    y_cal, sigma_cal = sample_calibration_voxels(
                        data_corr, mask, bvals, n_voxels=n_calibration_voxels,
                        seed=0,
                    )
                    print(f"   Sampled {len(y_cal)} calibration voxels from the "
                          f"brain mask (sigma_normalised={sigma_cal:.5f})")
                self.lambda_aniso, self.lambda_iso, _dd_diag = select_lambdas_data_driven(
                    bvals, bvecs, fiber_dirs, diff_pairs, iso_grid, y_cal, sigma_cal,
                )
                print(f"   Data-driven result: lambda_aniso={self.lambda_aniso:.4f}, "
                      f"lambda_iso={self.lambda_iso:.4f}")
                if _dd_diag.get('discrepancy', {}).get('floor_applied'):
                    print(f"   [WARNING] Safety floor was applied to lambda_aniso "
                          f"(raw discrepancy-principle answer was below 10% of "
                          f"lambda_iso, indicating an ill-conditioned/near-zero "
                          f"regularization scenario). Consider increasing "
                          f"n_calibration_voxels and/or running the Monte Carlo "
                          f"cross-check (run_mc_crosscheck=True) before trusting "
                          f"this result.")

            elif calibration_method == 'monte_carlo':
                print(f"\n5. Calibrating (lambda_aniso, lambda_iso) — MONTE CARLO "
                      f"(14 tissue scenarios, full grid search)...")
                self.lambda_aniso, self.lambda_iso = optimize_hyperparameters(
                    bvals, bvecs, snr,
                    n_aniso_cols=n_aniso_cols, n_iso=self.n_iso,
                    n_dirs=self.n_dirs, n_ad=self.n_ad, n_rd=self.n_rd,
                    anisotropy_ratio=self.anisotropy_ratio,
                    ad_range=self.ad_range, rd_range=self.rd_range,
                )

            else:
                raise ValueError(
                    f"calibration_method must be 'data_driven' or 'monte_carlo', "
                    f"got {calibration_method!r}."
                )

        if self.lambda_aniso is None:
            self.lambda_aniso = _STAGE_A_DEFAULT_LAMBDA_BASE * n_aniso_cols
        if self.lambda_iso is None:
            self.lambda_iso = _STAGE_A_DEFAULT_LAMBDA_BASE

        print(f"\n   Hyperparameters: n_iso={self.n_iso}, "
              f"lambda_aniso={self.lambda_aniso:.4f}, lambda_iso={self.lambda_iso:.4f}")
        print(f"   Fibre threshold: {self.fiber_threshold:.2f}  "
              f"(AD/RD/FA valid only where FF > {self.fiber_threshold:.2f})")
        print(f"   Stage A min_weight_fraction: {self.min_weight_fraction:.2f}")
        _thresh_str = (
            f"RF (ADC <= {THRESH_RES*1e3:.1f}x10^-3 mm^2/s) | "
            f"HF ({THRESH_RES*1e3:.1f}-{THRESH_WAT*1e3:.0f}x10^-3 mm^2/s) | "
            f"WF (ADC > {THRESH_WAT*1e3:.0f}x10^-3 mm^2/s)"
            if use_3iso else
            f"RF (ADC <= {THRESH_RES*1e3:.1f}x10^-3 mm^2/s) | "
            f"NRF (ADC > {THRESH_RES*1e3:.1f}x10^-3 mm^2/s)"
        )
        print(f"   Compartments: {_thresh_str}")

        # ── Monte Carlo cross-check (optional, does not change lambda) ─────
        if run_mc_crosscheck:
            print(f"\n   Running Monte Carlo cross-check of the selected lambda pair "
                  f"against 14 tissue scenarios...")
            _crosscheck_report = evaluate_lambda_pair(
                bvals, bvecs, snr, self.lambda_aniso, self.lambda_iso,
                n_mc=mc_crosscheck_n_mc,
                n_dirs=self.n_dirs, n_ad=self.n_ad, n_rd=self.n_rd,
                anisotropy_ratio=self.anisotropy_ratio,
                ad_range=self.ad_range, rd_range=self.rd_range,
                iso_range=self.iso_range, n_iso=self.n_iso,
                min_weight_fraction=self.min_weight_fraction,
                verbose=True,
            )
            self.mc_crosscheck_report_ = _crosscheck_report

        # ── Stage A design matrix ───────────────────────────────────────────
        print("\n6. Building Stage A Detection Dictionary...")

        A = build_design_matrix_exhaustive(bvals, bvecs, fiber_dirs, diff_pairs, iso_grid)
        AtA = A.T @ A
        At = A.T

        AtA_reg = compute_regularization_matrix(
            AtA, n_aniso_cols, self.lambda_aniso, self.lambda_iso
        )

        cond = np.linalg.cond(AtA_reg)
        print(f"   Design matrix: {A.shape}  |  "
              f"Condition number (regularized): {cond:.2e}")
        print(f"   Regularization: lambda_aniso={self.lambda_aniso:.4f}  "
              f"lambda_iso={self.lambda_iso:.4f}")

        # ── Allocate output ────────────────────────────────────────────────
        results = np.zeros(data.shape[:3] + (self.N_CHANNELS,), dtype=np.float32)
        results[..., 5] = np.nan
        results[..., 6] = np.nan
        results[..., 7] = np.nan
        results[..., 9] = np.nan
        results[..., 10] = np.nan
        if not use_3iso:
            results[..., 2] = np.nan
            results[..., 3] = np.nan

        # ── Parallel voxel fitting ─────────────────────────────────────────
        n_voxels = len(coords)
        batch_sz = 10_000
        n_batches = int(np.ceil(n_voxels / batch_sz))

        print(f"\n7. Fitting {n_voxels:,} voxels "
              f"[{model_mode}-ISO model, Stage A + Stage B]...")

        _kernel = _fit_voxels_3iso_v3 if use_3iso else _fit_voxels_2iso_v3

        b0_thr = 100.0

        t0 = time.time()
        with tqdm(total=n_voxels, desc="   Progress", unit="vox") as pbar:
            for i in range(n_batches):
                start = i * batch_sz
                end = min((i + 1) * batch_sz, n_voxels)
                _kernel(
                    data_corr, coords[start:end], AtA_reg, At,
                    bvals, bvecs, fiber_dirs, diff_pairs, self.n_dirs, iso_grid,
                    b0_thr, self.fiber_threshold, self.min_weight_fraction,
                    self.enable_direction_refinement,
                    _cone1, _n1, _cone2, _n2, results
                )
                pbar.update(end - start)

        elapsed = time.time() - t0
        n_fitted = int(np.sum(~np.isnan(results[..., 5]) & mask))
        pct = n_fitted / n_voxels * 100 if n_voxels > 0 else 0.0

        print(f"\n   Completed: {elapsed:.1f}s  "
              f"({n_voxels / elapsed:.0f} vox/s)")
        print(f"   AD/RD estimated: {n_fitted:,} / {n_voxels:,} "
              f"({pct:.1f}%)")
        print(f"\n{'='*70}\n")

        return results, model_mode

    # ------------------------------------------------------------------
    @staticmethod
    def output_map_names(model_mode):
        """
        Return the ordered list of output map file names for the given
        model mode. Unchanged from v1/v2.
        """
        if model_mode == 3:
            return [
                'fiber_fraction',
                'restricted_fraction',
                'hindered_fraction',
                'water_fraction',
                'nonrestricted_fraction',
                'axial_diffusivity',
                'radial_diffusivity',
                'fiber_fa',
                'mean_iso_adc',
                'ad_linear',
                'rd_linear',
            ]
        else:
            return [
                'fiber_fraction',
                'restricted_fraction',
                'hindered_fraction_NaN',
                'water_fraction_NaN',
                'nonrestricted_fraction',
                'axial_diffusivity',
                'radial_diffusivity',
                'fiber_fa',
                'mean_iso_adc',
                'ad_linear',
                'rd_linear',
            ]
