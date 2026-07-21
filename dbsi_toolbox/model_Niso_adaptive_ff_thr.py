"""
DBSI Adaptive Model v3 — Hybrid Two-Stage Architecture + MRDS Multi-Fiber Extension
========================================================================================

WHY v2's SINGLE-STAGE EXHAUSTIVE DICTIONARY WAS REPLACED
-------------------------------------------------------------
v2 attempted to estimate fiber orientation AND (AD, RD) simultaneously
from one linear NNLS solve over an exhaustive (direction x AD/RD-pair)
dictionary, taking AD_final/RD_final as NNLS-weighted centroids over the
activated anisotropic columns. Systematic synthetic recovery validation
(55 swept configurations, `recovery_validation.py`) demonstrated this is
NOT numerically identifiable: median AD/RD relative errors ranged from
~20% to >150% across every tested dictionary density, getting WORSE as
the dictionary was made finer, and no regularization strength fixed it.

v3 ARCHITECTURE: STAGE A (detection) + STAGE B (estimation)
-----------------------------------------------------------------
STAGE A — direction detection (sparse, exhaustive dictionary): a coarse
exhaustive (direction x AD/RD-pair) dictionary fit via regularized NNLS
with heavy sparsity on the anisotropic block. `select_dominant_directions`
reports which 1-3 hemisphere directions carry meaningful weight. Only
the directional answer is trusted, not Stage A's own (AD, RD) breakdown.

STAGE B — diffusivity estimation (conditioned on Stage A's direction(s)):
  - n_pop == 1 (single dominant fiber): unchanged from the original v3
    design -- `estimate_AD_RD_conditioned` (closed-form WLS), optionally
    refined by the MRDS-lite two-level cone search
    (`refine_fiber_direction_cone`).
  - n_pop >= 2 (MRDS multi-fiber extension, NEW): `estimate_AD_RD_mrds`
    (`core.solvers`) -- a short symmetry-breaking alternating pass
    followed by a bounded joint Levenberg-Marquardt fit over all detected
    populations simultaneously. See `core/solvers.py`'s MRDS module
    docstring for the full synthetic validation (2-fiber and one 3-fiber
    configuration) that motivated joint-over-alternating and the
    symmetry-breaking initialisation requirement.

WHAT THE MRDS EXTENSION DOES **NOT** DO — READ BEFORE INTERPRETING OUTPUT
-----------------------------------------------------------------------------
The MRDS extension refines AD/RD/FA/direction reporting for fiber
populations Stage A already detected. It does NOT retroactively correct
Stage A's own fraction estimates (FF/RF/HF/WF/NRF/mean_iso_adc), which
are computed and frozen from Stage A's NNLS weights BEFORE Stage B (single-
or multi-fiber) ever runs. A synthetic validation specifically testing
whether a small unregularized "Stage C" re-fit (using MRDS's refined
tensors to rebuild a tiny custom design matrix and re-derive fractions)
could correct isotropic-compartment leakage in crossing-fiber voxels was
run and REJECTED: collapsing the isotropic block to 1-2 fixed-diffusivity
columns discards the spectral resolution that is this toolbox's core
contribution, and made FF/NRF recovery WORSE in 5 of 6 tested synthetic
crossing conditions. A less destructive "targeted" variant (full isotropic
spectrum retained, only the anisotropic block replaced) was inconsistent
(improved FF in 4/6 conditions, NRF in only 3/6) and was not adopted
either. **Isotropic fraction accuracy in true crossing-fiber voxels is
therefore an OPEN, DOCUMENTED LIMITATION of this release, not something
the MRDS extension claims to fix.** A separate synthetic sweep (matched
isotropic composition, single-fiber vs 2-fiber-crossing ground truth,
same Stage A lambda/dictionary) additionally found NO consistent evidence
that crossing voxels leak MORE than single-fiber voxels in the tested
configuration (median FF error was, if anything, LOWER for crossing than
for single-fiber ground truth at this lambda_aniso) -- so the direction
and magnitude of any true crossing-related isotropic bias remains
unresolved and protocol/lambda-dependent, not a fixed, predictable offset.

MULTI-FIBER SCOPE: WHY 2 DEFAULT, 3 OPTIONAL, NOT MORE
-----------------------------------------------------------
`max_fiber_populations` defaults to 2, matching the practical ceiling
used elsewhere in multi-tensor/MRDS crossing-fiber literature (e.g.
"ball and 2 sticks" as the standard default multi-compartment
configuration) and this toolbox's own validation scope (see
`core/solvers.py`). Direction-estimation error in dedicated crossing-fiber
studies has been reported to grow sharply with population count even at
fixed SNR (on the order of 3 deg for one tensor vs. 7 deg for two vs. 16
deg for three at SNR~25:1) -- resolving 3 populations reliably typically
requires denser angular sampling (HARDI-grade protocols) than this
toolbox's coarse, deliberately protocol-agnostic Stage A dictionary is
designed to assume. 3 is offered as an explicit opt-in
(`max_fiber_populations=3`) for richer protocols, WITHOUT automatic
protocol gating in this release (unlike the 3-ISO isotropic model
selection, which IS auto-gated on b_max/n_shells) -- the person enabling
it is responsible for confirming their protocol's angular sampling
density supports it. Values above 3 are not supported: beyond that point
this toolbox would be duplicating the scope of dedicated multi-fiber
tractography tools (CHARMED, ball-and-sticks, full MRDS) rather than
serving its own stated purpose (isotropic-compartment quantification with
a secondary anisotropic/axonal-integrity readout).

Compartment Definitions (unchanged from v1/v2)
---------------------------------------------------
    RF  : ADC <= THRESH_RES  (0.3 x 10^-3 mm^2/s) — cells, inflammatory infiltrate
    HF  : THRESH_RES < ADC <= THRESH_WAT          — hindered extracellular water
    WF  : ADC > THRESH_WAT  (3.0 x 10^-3 mm^2/s)  — free water, CSF, oedema
    NRF : HF + WF  (Non-Restricted Fraction, 2-ISO mode only)

Model Selection Criterion (unchanged from v1/v2)
------------------------------------------------------
2-ISO vs 3-ISO selection based on b_max / shell count is unaffected by
the MRDS extension -- it governs the isotropic block only.

Output Channels (EXTENDED to 29 total; channels 0-10 UNCHANGED from the
original v3 layout for full backward compatibility with existing
fit_quality.py, transition_confidence.py, and any already-produced
figures/analyses -- channels 11-28 are a pure append)
-------------------------------------------------------------------------
    0  : FF   — Total fibre fraction (summed over ALL detected populations)
    1  : RF   — Restricted fraction  (ADC <= 0.3e-3)      (always valid)
    2  : HF   — Hindered fraction    (0.3e-3 < ADC <= 3.0e-3) (NaN in 2-ISO mode)
    3  : WF   — Free-water fraction  (ADC > 3.0e-3)      (NaN in 2-ISO mode)
    4  : NRF  — Non-Restricted fraction = HF + WF        (always valid)
    5  : AD   — DOMINANT population's axial diffusivity (Stage B/MRDS estimate;
                NaN if FF <= fiber_threshold)
    6  : RD   — DOMINANT population's radial diffusivity
    7  : FA   — DOMINANT population's intrinsic fibre FA
    8  : ADC_iso — Mean isotropic ADC                     (always valid)
    9  : AD_lin  — identical to channel 5 (retained for shape compatibility)
    10 : RD_lin  — identical to channel 6 (retained for shape compatibility)
    ── MRDS extension (NEW, channels 11-28) ──
    11 : N_POP   — number of fiber populations Stage A reported in this
                   voxel (0, 1, 2, or 3). N_POP==1 voxels have channels
                   15-28 as NaN (nothing to report for pop 2/3).
    12-14 : DIR1_XYZ — dominant population's direction (unit vector; NOT
                   previously stored in the 11-channel layout -- lets
                   `fit_quality.py` read the direction directly instead
                   of re-deriving it via grid search, for voxels fit
                   under this schema)
    15 : FF_POP2, 16: AD_POP2, 17: RD_POP2, 18: FA_POP2   (NaN if N_POP<2)
    19-21 : DIR2_XYZ                                       (NaN if N_POP<2)
    22 : FF_POP3, 23: AD_POP3, 24: RD_POP3, 25: FA_POP3   (NaN if N_POP<3,
                   including whenever max_fiber_populations==2)
    26-28 : DIR3_XYZ                                       (NaN if N_POP<3)

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601. doi:10.1093/brain/awr307
Shirani A, et al. (2019). Ann Clin Transl Neurol, 6(11):2323-2327.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33-47.
Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Medical
    Image Analysis, 42, 26-43. (MRDS multi-resolution discrete search,
    the conceptual basis for both the existing single-fiber cone
    refinement and this multi-fiber extension.)
Design document: toolbox_v2.md (Ramirez-Manzanares discussion); v3
hybrid redesign motivated by synthetic recovery validation of the v2
single-stage approach; MRDS multi-fiber extension motivated by project
synthetic validation of joint-vs-alternating Stage B estimation and of
the (rejected) Stage-C fraction re-fit -- see `core/solvers.py`.
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
    build_direction_neighbor_graph,   # NEW — local-maxima peak-finding graph
    estimate_AD_RD_conditioned,
    compute_fiber_fa,
    refine_fiber_direction_cone,
    compute_cone_refinement_schedule,
    measure_hemisphere_spacing,
    estimate_AD_RD_mrds,          # NEW — MRDS multi-fiber Stage B
)
from .calibration.optimizer import optimize_hyperparameters, evaluate_lambda_pair
from .calibration.data_driven import select_lambdas_data_driven, sample_calibration_voxels
from .calibration.adaptive_n_iso import select_n_iso_svd, select_n_iso_bootstrap
from .calibration.mc_sure import crosscheck_lambda_iso_sure, crosscheck_n_iso_sure

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

# Stage A dictionary defaults. Deliberately coarse on the AD/RD axis.
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
_DEFAULT_N_ISO_STEPS = 31    # Legacy fixed default — see select_n_iso_svd.

_ISO_GRID_D_MAX_EXTENDED = 5.0e-3

# Default maximum number of fiber populations Stage A/B will report per
# voxel. Now a per-instance parameter (`DBSI_Adaptive(max_fiber_populations=
# ...)`) rather than a hardcoded module constant -- see module docstring
# "MULTI-FIBER SCOPE" for why the default is 2 and why 3 is offered as an
# explicit, non-protocol-gated opt-in rather than a further-increased
# default.
_DEFAULT_MAX_FIBER_POPULATIONS = 2

# MRDS multi-fiber Stage B defaults (see core.solvers.estimate_AD_RD_mrds).
_MRDS_INIT_N_ITER = 3        # short, deliberately non-converged alternating warm start
_MRDS_LM_MAX_ITER = 25

# Local-maxima peak-finding neighbourhood size for `select_dominant_directions`
# (see core.solvers module notes just above that function). Re-validated
# 2026-07-15 (post sign-bug and detection-fix, k in {4,6,8,10,12}, n_dirs=62,
# 30 seeds/condition, SNR=30): k=6 dominates or ties k=8+ on both axes
# (single-fiber correct-N_POP=1 rate identical at 73.3%; crossing
# sensitivity higher at every tested angle: 30/60/90deg = 46.7/93.3/96.7%
# for k=6 vs 43.3/90.0/93.3% for k=8). k=4 trades single-fiber accuracy
# (63.3%) for a further sensitivity gain that mostly saturates by k=6.
# 30-degree crossing sensitivity (~45-65% across all k) is NOT much
# improved by k alone -- at this dictionary's ~25 deg mean node spacing, a
# 30 deg crossing separates the two true peaks by close to one grid
# spacing, so their neighbourhoods overlap almost by construction; a
# denser Stage A dictionary (larger n_dirs), not a larger/smaller k, is
# the lever for that regime. See project re-validation sweep.
#
# CAVEAT ADDED 2026-07-16: the k=6 default above was only validated at
# n_dirs=62. A separate synthetic check (90 deg crossing, tilted 25 deg
# out of the hemisphere's equatorial boundary, SNR=30, n_iso=4,
# lambda_aniso=1.0/lambda_iso=0.001 fixed) found k=6 gives COMPLETE
# detection failure (0/27 replicates correctly reporting N_POP=2) at
# n_dirs=12 -- the fixed k=6 neighbourhood becomes half the entire
# dictionary at that density, so essentially every candidate suppresses
# every other one. Progressively smaller k restored detection (k=4: 0/27,
# k=3: 7/27, k=2: 27/27). This motivated switching the default from a
# fixed constant to a n_dirs-scaled formula (see
# `_default_direction_peak_k` below): k=max(2, min(6, n_dirs // 5))
# reproduced 26-27/27 correct detection at n_dirs in {12, 20, 30, 40} in
# that same check. This is a MUCH smaller validation than the k-sweep
# above (one crossing angle, one SNR, one phantom) -- treat the scaled
# default as a reasonable interim fix for very coarse dictionaries, not
# as thoroughly validated as the original k=6-at-n_dirs=62 result.
# Explicit `direction_peak_k=` still overrides this scaling entirely.
_DEFAULT_DIRECTION_PEAK_K = None  # None => n_dirs-scaled (see _default_direction_peak_k)

# Minimum angular separation (deg) between two ACCEPTED fiber populations in
# `select_dominant_directions`'s greedy basin-mass selection (non-maximum
# suppression). A candidate peak closer than this to an already-accepted,
# stronger peak is treated as that fiber's grid/regularization smearing skirt,
# not a distinct population. 35 deg comfortably separates true crossings
# (typically resolvable only down to ~30-45 deg at clinical SNR — see the
# module 'MULTI-FIBER SCOPE' note) while absorbing the ~15-30 deg smearing
# halo a single fiber leaves on neighbouring dictionary columns.
_DEFAULT_MIN_SEPARATION_DEG = 35.0

# Minimum ratio of a secondary population's angular-basin mass to the DOMINANT
# population's basin mass for it to be accepted as a distinct fiber (rather
# than a single fiber's smearing skirt that noise shifted past the separation
# angle). Set from an empirical single-fiber-vs-crossing ratio sweep whose
# distributions overlap: 0.35 keeps false crossings on single fibers to ~7%
# while recovering the true 2nd fiber in ~40-80% of crossing voxels at SNR
# 30-20. See `core.solvers.select_dominant_directions`.
_DEFAULT_MIN_PEAK_RATIO = 0.35

# Minimum share of TOTAL anisotropic weight the dominant angular basin must
# hold for ANY fiber population to be reported (concentration gate in
# `select_dominant_directions`). A real fiber concentrates its weight in one
# basin (~0.4-0.6 of total at SNR>=30); diffuse anisotropic weight from
# isotropic (esp. restricted) leakage on a fiber-FREE voxel spreads across many
# near-equal basins (dominant ~0.2-0.24) and would otherwise be mis-read as a
# crossing (npop>=2). Empirical pure-iso-vs-fiber concentration sweep: at SNR30
# a clean gap (pure-iso p95~0.40, real-fiber p05~0.39); at SNR15 the two
# OVERLAP (noise diffuses real fibers down to ~0.25-0.36), so no gate perfectly
# separates there. 0.35 is the chosen balance: cuts the pure-iso false crossing
# by ~89% (npop 2.0->0.22) while KEEPING low-SNR (SNR15) fiber detection
# (npop~1.0 vs 0.78 at 0.38). Raise toward 0.38 for max specificity at the cost
# of low-SNR sensitivity. Below the gate -> npop=0.
_DEFAULT_MIN_DOMINANT_CONCENTRATION = 0.35


def _default_direction_peak_k(n_dirs):
    """
    n_dirs-scaled default for the local-maxima peak-finding neighbourhood
    size `k` (see `_DEFAULT_DIRECTION_PEAK_K` comment above for the
    empirical basis). Matches the validated k=6 default for n_dirs>=30
    (the range the original k-sweep and this scaling agree on) and
    degrades gracefully for coarser dictionaries.
    """
    return max(2, min(6, n_dirs // 5))


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
# PARALLEL FITTING KERNELS — v3 + MRDS multi-fiber extension
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_2iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction, min_separation_cos,
                        min_peak_ratio, min_dominant_concentration,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        neighbor_idx, max_fiber_populations, out):
    """
    v3 parallel fitting kernel — two-compartment isotropic model (2-ISO).

    Stage A direction detection is unchanged. Stage B branches on how many
    populations Stage A reported (n_pop, up to `max_fiber_populations`):
      n_pop == 1 : unchanged single-fiber path (closed-form Stage B,
                   optional MRDS-lite cone refinement).
      n_pop >= 2 : NEW MRDS multi-fiber joint Stage B
                   (`estimate_AD_RD_mrds`), using Stage A's RAW
                   (grid-quantised, NOT cone-refined -- see module
                   docstring scope note) directions.

    Output layout: see module docstring "Output Channels". Channels 0-10
    are written exactly as before (backward compatible); channels 11-28
    are the new MRDS/multi-population block.
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

        # ── Channels 0-10 (fractions): UNCHANGED regardless of n_pop.
        # See module docstring "WHAT THE MRDS EXTENSION DOES NOT DO":
        # these fractions are frozen here and never revisited by Stage B. ──
        out[x, y, z, 0] = f_fib
        out[x, y, z, 1] = f_res
        out[x, y, z, 4] = f_nonrf
        out[x, y, z, 8] = mean_iso_adc

        if f_fib > fiber_threshold:
            dir_indices, dir_weights = select_dominant_directions(
                w_aniso, n_dirs, n_pairs, neighbor_idx, fiber_dirs,
                max_fiber_populations, min_weight_fraction, min_separation_cos,
                min_peak_ratio, min_dominant_concentration
            )

            n_pop = 0
            for k in range(max_fiber_populations):
                if dir_indices[k] >= 0:
                    n_pop += 1
            out[x, y, z, 11] = n_pop

            if n_pop == 1:
                dominant_dir = fiber_dirs[dir_indices[0]]

                if enable_direction_refinement:
                    _, AD_est, RD_est = refine_fiber_direction_cone(
                        bvals, bvecs, sig_norm, dominant_dir,
                        f_fib, f_res, f_nonrf, 0.0,
                        D_res_c, D_nonrf_c, 0.0, False,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
                    )
                else:
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
                out[x, y, z, 12] = dominant_dir[0]
                out[x, y, z, 13] = dominant_dir[1]
                out[x, y, z, 14] = dominant_dir[2]

            elif n_pop >= 2:
                # ── NEW: MRDS multi-fiber joint Stage B ──
                # Directions are Stage A's RAW grid detections (no
                # per-population cone refinement in this release -- see
                # module docstring scope note).
                directions = np.empty((n_pop, 3))
                w_sum = 0.0
                for k in range(n_pop):
                    directions[k] = fiber_dirs[dir_indices[k]]
                    w_sum += dir_weights[k]

                fractions = np.empty(n_pop)
                for k in range(n_pop):
                    fractions[k] = (dir_weights[k] / w_sum) * f_fib if w_sum > 1e-12 else 0.0

                iso_signal = np.empty(len(bvals))
                for i in range(len(bvals)):
                    iso_signal[i] = (f_res * np.exp(-bvals[i] * D_res_c)
                                     + f_nonrf * np.exp(-bvals[i] * D_nonrf_c))

                AD_out, RD_out = estimate_AD_RD_mrds(
                    bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                    init_n_iter=_MRDS_INIT_N_ITER, lm_max_iter=_MRDS_LM_MAX_ITER
                )

                # Dominant population (index 0, highest Stage A weight) ->
                # legacy channels 5/6/7/9/10 + new DIR1, for backward
                # compatibility with single-fiber-era downstream code.
                FA0 = compute_fiber_fa(AD_out[0], RD_out[0])
                out[x, y, z, 5] = AD_out[0]
                out[x, y, z, 6] = RD_out[0]
                out[x, y, z, 7] = FA0
                out[x, y, z, 9] = AD_out[0]
                out[x, y, z, 10] = RD_out[0]
                out[x, y, z, 12] = directions[0, 0]
                out[x, y, z, 13] = directions[0, 1]
                out[x, y, z, 14] = directions[0, 2]

                # Population 2
                FA1 = compute_fiber_fa(AD_out[1], RD_out[1])
                out[x, y, z, 15] = fractions[1]
                out[x, y, z, 16] = AD_out[1]
                out[x, y, z, 17] = RD_out[1]
                out[x, y, z, 18] = FA1
                out[x, y, z, 19] = directions[1, 0]
                out[x, y, z, 20] = directions[1, 1]
                out[x, y, z, 21] = directions[1, 2]

                # Population 3 (only if detected AND max_fiber_populations==3)
                if n_pop >= 3:
                    FA2 = compute_fiber_fa(AD_out[2], RD_out[2])
                    out[x, y, z, 22] = fractions[2]
                    out[x, y, z, 23] = AD_out[2]
                    out[x, y, z, 24] = RD_out[2]
                    out[x, y, z, 25] = FA2
                    out[x, y, z, 26] = directions[2, 0]
                    out[x, y, z, 27] = directions[2, 1]
                    out[x, y, z, 28] = directions[2, 2]


@njit(parallel=True, cache=True, fastmath=True)
def _fit_voxels_3iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction, min_separation_cos,
                        min_peak_ratio, min_dominant_concentration,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        neighbor_idx, max_fiber_populations, out):
    """v3 parallel fitting kernel — three-compartment isotropic model
    (3-ISO). Same Stage A / Stage B (+ MRDS multi-fiber) structure as
    `_fit_voxels_2iso_v3`; see that kernel's docstring for the full
    n_pop branching logic. Here the fixed isotropic signal handed to the
    MRDS joint fit has THREE terms (RES/HIN/WAT) instead of two.
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
                w_aniso, n_dirs, n_pairs, neighbor_idx, fiber_dirs,
                max_fiber_populations, min_weight_fraction, min_separation_cos,
                min_peak_ratio, min_dominant_concentration
            )

            n_pop = 0
            for k in range(max_fiber_populations):
                if dir_indices[k] >= 0:
                    n_pop += 1
            out[x, y, z, 11] = n_pop

            if n_pop == 1:
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
                out[x, y, z, 12] = dominant_dir[0]
                out[x, y, z, 13] = dominant_dir[1]
                out[x, y, z, 14] = dominant_dir[2]

            elif n_pop >= 2:
                directions = np.empty((n_pop, 3))
                w_sum = 0.0
                for k in range(n_pop):
                    directions[k] = fiber_dirs[dir_indices[k]]
                    w_sum += dir_weights[k]

                fractions = np.empty(n_pop)
                for k in range(n_pop):
                    fractions[k] = (dir_weights[k] / w_sum) * f_fib if w_sum > 1e-12 else 0.0

                iso_signal = np.empty(len(bvals))
                for i in range(len(bvals)):
                    iso_signal[i] = (f_res * np.exp(-bvals[i] * D_res_c)
                                     + f_hin * np.exp(-bvals[i] * D_hin_c)
                                     + f_wat * np.exp(-bvals[i] * D_wat_c))

                AD_out, RD_out = estimate_AD_RD_mrds(
                    bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                    init_n_iter=_MRDS_INIT_N_ITER, lm_max_iter=_MRDS_LM_MAX_ITER
                )

                FA0 = compute_fiber_fa(AD_out[0], RD_out[0])
                out[x, y, z, 5] = AD_out[0]
                out[x, y, z, 6] = RD_out[0]
                out[x, y, z, 7] = FA0
                out[x, y, z, 9] = AD_out[0]
                out[x, y, z, 10] = RD_out[0]
                out[x, y, z, 12] = directions[0, 0]
                out[x, y, z, 13] = directions[0, 1]
                out[x, y, z, 14] = directions[0, 2]

                FA1 = compute_fiber_fa(AD_out[1], RD_out[1])
                out[x, y, z, 15] = fractions[1]
                out[x, y, z, 16] = AD_out[1]
                out[x, y, z, 17] = RD_out[1]
                out[x, y, z, 18] = FA1
                out[x, y, z, 19] = directions[1, 0]
                out[x, y, z, 20] = directions[1, 1]
                out[x, y, z, 21] = directions[1, 2]

                if n_pop >= 3:
                    FA2 = compute_fiber_fa(AD_out[2], RD_out[2])
                    out[x, y, z, 22] = fractions[2]
                    out[x, y, z, 23] = AD_out[2]
                    out[x, y, z, 24] = RD_out[2]
                    out[x, y, z, 25] = FA2
                    out[x, y, z, 26] = directions[2, 0]
                    out[x, y, z, 27] = directions[2, 1]
                    out[x, y, z, 28] = directions[2, 2]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DBSI_Adaptive:
    """
    Adaptive DBSI model (v3 + MRDS multi-fiber extension) using a hybrid
    two-stage anisotropic estimation: Stage A detects dominant fiber
    direction(s) from an exhaustive (direction x AD/RD-pair) dictionary
    under heavy sparsity regularization; Stage B estimates AD/RD/FA either
    via closed-form WLS (single fiber) or MRDS joint nonlinear
    least-squares (2-3 crossing fibers) conditioned on the detected
    direction(s).

    NEW: `max_fiber_populations` (default 2, optionally 3) controls how
    many simultaneous fiber populations Stage A/B will detect and report
    per voxel. See module docstring "MULTI-FIBER SCOPE" for why the
    default is 2 and why this toolbox does not support values beyond 3.

    IMPORTANT: the MRDS extension improves AD/RD/FA/direction accuracy
    for detected crossing populations. It does NOT correct the isotropic
    or total-fiber FRACTION estimates (FF/RF/HF/WF/NRF), which are
    computed by Stage A and are unaffected by `max_fiber_populations` --
    see module docstring "WHAT THE MRDS EXTENSION DOES NOT DO" for the
    validation that led to this being an explicit, documented open
    limitation rather than a "fixed" behaviour.

    Selection rule (isotropic block — unchanged from v1/v2)
    -------------------------------------------------------------
    3-ISO (RF + HF + WF) if:  b_max >= B_THRESH_3ISO  AND  n_shells >= MIN_SHELLS_3ISO
    2-ISO (RF + NRF)     otherwise.

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
    max_fiber_populations : int
        Maximum number of simultaneous fiber populations to detect and
        report per voxel. Default: 2. May be set to 3 for protocols with
        sufficiently dense angular sampling (NOT automatically checked --
        see module docstring "MULTI-FIBER SCOPE"). Values other than
        1, 2, or 3 are rejected.
    enable_direction_refinement : bool
        Whether to refine Stage A's dominant direction estimate with a
        two-level "MRDS-lite" cone search for SINGLE-fiber voxels
        (n_pop == 1). Default: True. NOT applied to n_pop >= 2 voxels in
        this release (see module docstring scope note) -- those use
        Stage A's raw grid-quantised directions as MRDS joint Stage B
        input.
    target_angular_resolution_deg : float
        Desired final angular precision (degrees) of the refined
        direction (n_pop == 1 path only). Default: 1.0.

    Notes
    -----
    There is no `enable_step2` parameter: the non-linear Step 2
    refinement stage from v1 has been eliminated since v2 and remains
    eliminated in v3.
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
        'N_POP': 11,
        'DIR1_X': 12, 'DIR1_Y': 13, 'DIR1_Z': 14,
        'FF_POP2': 15, 'AD_POP2': 16, 'RD_POP2': 17, 'FA_POP2': 18,
        'DIR2_X': 19, 'DIR2_Y': 20, 'DIR2_Z': 21,
        'FF_POP3': 22, 'AD_POP3': 23, 'RD_POP3': 24, 'FA_POP3': 25,
        'DIR3_X': 26, 'DIR3_Y': 27, 'DIR3_Z': 28,
    }
    N_CHANNELS = 29
    N_CHANNELS_LEGACY = 11  # for reference / external code checking shape

    def __init__(self, n_iso=None, lambda_aniso=None, lambda_iso=None,
                 n_dirs=None, n_ad=_STAGE_A_DEFAULT_N_AD, n_rd=_STAGE_A_DEFAULT_N_RD,
                 anisotropy_ratio=_STAGE_A_DEFAULT_ANISOTROPY_RATIO,
                 ad_range=(_STAGE_A_AD_MIN, _STAGE_A_AD_MAX),
                 rd_range=(_STAGE_A_RD_MIN, _STAGE_A_RD_MAX),
                 iso_range=(_DEFAULT_ISO_MIN, _DEFAULT_ISO_MAX),
                 fiber_threshold=FIBER_THRESHOLD,
                 min_weight_fraction=0.05, force_n_iso=None,
                 max_fiber_populations=_DEFAULT_MAX_FIBER_POPULATIONS,
                 direction_peak_k=_DEFAULT_DIRECTION_PEAK_K,
                 min_separation_deg=_DEFAULT_MIN_SEPARATION_DEG,
                 min_peak_ratio=_DEFAULT_MIN_PEAK_RATIO,
                 min_dominant_concentration=_DEFAULT_MIN_DOMINANT_CONCENTRATION,
                 enable_direction_refinement=True,
                 target_angular_resolution_deg=1.0):
        if max_fiber_populations not in (1, 2, 3):
            raise ValueError(
                f"max_fiber_populations must be 1, 2, or 3, got "
                f"{max_fiber_populations!r}. See module docstring "
                f"'MULTI-FIBER SCOPE' for why this toolbox does not "
                f"support values beyond 3."
            )

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
        self.max_fiber_populations = max_fiber_populations
        self.direction_peak_k = direction_peak_k
        self.min_separation_deg = min_separation_deg
        self.min_peak_ratio = min_peak_ratio
        self.min_dominant_concentration = min_dominant_concentration
        self.enable_direction_refinement = enable_direction_refinement
        self.target_angular_resolution_deg = target_angular_resolution_deg

        self.model_mode_ = None
        self.b_max_ = None
        self.n_shells_ = None
        self.n_aniso_cols_ = None
        self.diff_pairs_ = None
        self.mc_crosscheck_report_ = None
        self.sure_crosscheck_report_ = None
        self.hemisphere_spacing_deg_ = None
        self.cone_refinement_schedule_ = None

    # ------------------------------------------------------------------
    def fit(self, data, bvals, bvecs, mask, run_calibration=True,
           calibration_method='data_driven', n_calibration_voxels=500,
           n_iso_method='bootstrap', n_bootstrap=50,
           run_mc_crosscheck=False, mc_crosscheck_n_mc=200,
           run_sure_crosscheck=False, sure_crosscheck_n_probes=15,
           run_n_iso_sweep_diagnostic=False):
        """
        Fit the v3 hybrid two-stage adaptive DBSI model (+ MRDS
        multi-fiber extension) to 4D diffusion MRI data.

        Parameters are unchanged from the pre-MRDS release EXCEPT for the
        constructor's new `max_fiber_populations`; see class docstring.

        Returns
        -------
        results : ndarray (X, Y, Z, 29)
            See module docstring "Output Channels".
        model_mode : int
            2 or 3.
        """
        print("\n" + "="*70)
        print("  DBSI ADAPTIVE PIPELINE — v3 + MRDS Multi-Fiber Extension")
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
        print(f"  Max fiber populations: {self.max_fiber_populations} "
              f"{'(default)' if self.max_fiber_populations == _DEFAULT_MAX_FIBER_POPULATIONS else '(OPT-IN — confirm protocol angular density supports this, see module docstring)'}")

        # ── Stage A dictionary autoconfiguration ─────────────────────────
        print("\n1. Autoconfiguring Stage A detection dictionary...")
        M_auto, n_ad_auto, n_rd_auto, ratio_auto = autoconfigure_dictionary(bvals, bvecs)

        if self.n_dirs is None:
            self.n_dirs = M_auto

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

        # ── MRDS-lite direction refinement schedule (single-fiber path only) ──
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
            print(f"\n   MRDS-lite direction refinement (n_pop==1 voxels only): ENABLED "
                  f"(target resolution={self.target_angular_resolution_deg:.2f} deg)")
            print(f"   Dictionary spacing: {self.hemisphere_spacing_deg_:.2f} deg  |  "
                  f"Level 1 cone: +/-{np.degrees(_cone1):.2f} deg ({_n1} candidates)  |  "
                  f"Level 2 cone: +/-{np.degrees(_cone2):.2f} deg ({_n2} candidates)")
            if self.max_fiber_populations >= 2:
                print(f"   NOTE: n_pop>=2 voxels use Stage A's RAW grid directions as "
                      f"MRDS joint Stage B input (no per-population cone refinement in "
                      f"this release — see module docstring scope note).")
        else:
            _cone1, _n1, _cone2, _n2 = 0.0, 1, 0.0, 0
            print(f"\n   MRDS-lite direction refinement: DISABLED "
                  f"(using raw Stage A grid direction, unrefined)")

        # ── Local-maxima peak-finding neighbour graph (Stage A direction
        # selection) — computed ONCE per protocol from the dictionary's own
        # geometry, not per voxel. See core.solvers module notes above
        # `select_dominant_directions` for why global top-K-by-weight alone
        # produces false-positive N_POP>=2 on true single fibers (grid
        # quantisation smearing onto neighbouring columns) and how the
        # local-maxima criterion fixes it. ──
        if self.direction_peak_k is None:
            _effective_k = _default_direction_peak_k(self.n_dirs)
            print(f"\n   direction_peak_k not set explicitly -- using "
                  f"n_dirs-scaled default: k={_effective_k} for n_dirs={self.n_dirs} "
                  f"(see _DEFAULT_DIRECTION_PEAK_K comment for validation caveats)")
        else:
            _effective_k = self.direction_peak_k
        neighbor_idx = build_direction_neighbor_graph(fiber_dirs, k=_effective_k)
        print(f"\n   Direction selection: local-maxima peak-finding "
              f"(k={neighbor_idx.shape[1]} nearest geometric neighbours)")

        # ── SNR estimation ─────────────────────────────────────────────────
        print("\n2. Estimating SNR...")
        snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)

        # ── Rician bias correction (unchanged from v1/v2) ──────────────────
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

        y_cal, sigma_cal = None, None
        if self.n_iso is None or (run_calibration and
                                  (self.lambda_aniso is None or self.lambda_iso is None) and
                                  calibration_method == 'data_driven'):
            y_cal, sigma_cal = sample_calibration_voxels(
                data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
            )
            print(f"\n   Sampled {len(y_cal)} calibration voxels from the brain mask "
                  f"(sigma_normalised={sigma_cal:.5f})")

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
                if _dd_diag.get('lambda_iso_capped'):
                    print(f"   [lambda_iso ceiling] GCV wanted lambda_iso="
                          f"{_dd_diag['lambda_iso_gcv']:.4f}; capped to the "
                          f"noise-referenced discrepancy ceiling "
                          f"{_dd_diag['lambda_iso_cap']:.4f} — prevents the low-SNR "
                          f"isotropic collapse / FF leakage (GCV was railing).")
                if _dd_diag.get('discrepancy', {}).get('floor_applied'):
                    _floor_comp = _dd_diag.get('discrepancy', {}).get('floor_component', 'unknown')
                    print(f"   [WARNING] Safety floor was applied to lambda_aniso "
                          f"(component: {_floor_comp}; raw discrepancy-principle "
                          f"answer was below the floor, indicating an "
                          f"ill-conditioned/near-zero regularization scenario). "
                          f"Consider increasing n_calibration_voxels and/or "
                          f"running the Monte Carlo cross-check "
                          f"(run_mc_crosscheck=True) before trusting this result.")
                    if _floor_comp == 'aniso_floor_fraction':
                        print(f"   [CAVEAT] The aniso_floor_fraction default is "
                              f"UNVALIDATED on this (anchored-isotropic-grid) "
                              f"pipeline as of 2026-07-16 -- it was fitted on a "
                              f"different isotropic grid construction and did NOT "
                              f"fix the FF leakage it targets in a known synthetic "
                              f"crossing-fiber follow-up check. If FF looks close "
                              f"to 1.0 with near-zero isotropic fractions, suspect "
                              f"an isotropic-grid coverage gap (see "
                              f"`select_lambda_aniso_discrepancy` docstring) rather "
                              f"than trusting this floor to have fixed things.")

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
        print(f"   NOTE: isotropic/fiber FRACTIONS above are Stage A's raw NNLS "
              f"output regardless of max_fiber_populations -- the MRDS extension "
              f"does not revise them (see module docstring).")

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

        # ── Monte Carlo SURE cross-check (optional) ─────────────────────────
        if run_sure_crosscheck:
            print(f"\n   Running Monte Carlo SURE cross-check of n_iso and lambda_iso...")
            if y_cal is None:
                y_cal, sigma_cal = sample_calibration_voxels(
                    data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
                )
                print(f"   Sampled {len(y_cal)} calibration voxels for SURE cross-check "
                      f"(sigma_normalised={sigma_cal:.5f})")

            _sure_lambda_agrees, _sure_lambda_report = crosscheck_lambda_iso_sure(
                bvals, iso_grid, y_cal, sigma_cal, self.lambda_iso,
                n_probes=sure_crosscheck_n_probes, verbose=True,
            )
            _sure_n_iso_agrees, _sure_n_iso_report = crosscheck_n_iso_sure(
                bvals, y_cal, sigma_cal, self.n_iso,
                d_min=max(self.iso_range[0], 0.1e-3),
                d_max=max(self.iso_range[1], _ISO_GRID_D_MAX_EXTENDED),
                n_probes=sure_crosscheck_n_probes, verbose=True,
            )
            self.sure_crosscheck_report_ = dict(
                lambda_iso=_sure_lambda_report, n_iso=_sure_n_iso_report,
                lambda_iso_agrees=_sure_lambda_agrees, n_iso_agrees=_sure_n_iso_agrees,
            )
            if not (_sure_lambda_agrees and _sure_n_iso_agrees):
                print(f"\n   [NOTE] Monte Carlo SURE flagged a disagreement above.")

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

        # ── Allocate output (EXTENDED: 29 channels) ─────────────────────────
        results = np.zeros(data.shape[:3] + (self.N_CHANNELS,), dtype=np.float32)
        results[..., 5] = np.nan
        results[..., 6] = np.nan
        results[..., 7] = np.nan
        results[..., 9] = np.nan
        results[..., 10] = np.nan
        results[..., 11] = np.nan  # N_POP: NaN outside fiber_threshold, not 0
        results[..., 12:29] = np.nan  # DIR1 + pop2/pop3 block, all NaN by default
        if not use_3iso:
            results[..., 2] = np.nan
            results[..., 3] = np.nan

        # ── Parallel voxel fitting ─────────────────────────────────────────
        n_voxels = len(coords)
        batch_sz = 10_000
        n_batches = int(np.ceil(n_voxels / batch_sz))

        print(f"\n7. Fitting {n_voxels:,} voxels "
              f"[{model_mode}-ISO model, Stage A + Stage B "
              f"(single-fiber closed-form / MRDS joint up to "
              f"{self.max_fiber_populations} populations)]...")
        if self.max_fiber_populations >= 2:
            print(f"   Population detection: basin-mass local-maxima + angular "
                  f"NMS (min separation {self.min_separation_deg:.0f} deg, "
                  f"min basin mass {self.min_weight_fraction:.2f} of aniso weight, "
                  f"2nd pop >= {self.min_peak_ratio:.2f} x dominant, "
                  f"concentration gate {self.min_dominant_concentration:.2f})")

        _min_separation_cos = float(np.cos(np.radians(self.min_separation_deg)))
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
                    _min_separation_cos, self.min_peak_ratio,
                    self.min_dominant_concentration,
                    self.enable_direction_refinement,
                    _cone1, _n1, _cone2, _n2,
                    neighbor_idx, self.max_fiber_populations, results
                )
                pbar.update(end - start)

        elapsed = time.time() - t0
        n_fitted = int(np.sum(~np.isnan(results[..., 5]) & mask))
        n_multi = int(np.sum((results[..., 11] >= 2) & mask))
        pct = n_fitted / n_voxels * 100 if n_voxels > 0 else 0.0
        pct_multi = n_multi / n_fitted * 100 if n_fitted > 0 else 0.0

        print(f"\n   Completed: {elapsed:.1f}s  "
              f"({n_voxels / elapsed:.0f} vox/s)")
        print(f"   AD/RD estimated: {n_fitted:,} / {n_voxels:,} "
              f"({pct:.1f}%)")
        print(f"   Multi-population (N_POP>=2) voxels: {n_multi:,} "
              f"({pct_multi:.1f}% of fitted voxels)")
        print(f"\n{'='*70}\n")

        return results, model_mode

    # ------------------------------------------------------------------
    @staticmethod
    def output_map_names(model_mode):
        """
        Return the ordered list of output map file names for the given
        model mode. Channels 0-10 unchanged from the pre-MRDS release;
        11-28 are the new MRDS/multi-population block. Names ending in
        '_NaN' mark channels invalid in the given model_mode OR (for the
        pop2/pop3 block) channels that are frequently/always NaN
        depending on max_fiber_populations and per-voxel N_POP -- callers
        should still check for NaN per-voxel rather than assuming a
        channel is entirely absent.
        """
        base_3iso = [
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
        base_2iso = [
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
        mrds_block = [
            'n_fiber_populations',
            'dir1_x', 'dir1_y', 'dir1_z',
            'fiber_fraction_pop2',
            'axial_diffusivity_pop2',
            'radial_diffusivity_pop2',
            'fiber_fa_pop2',
            'dir2_x', 'dir2_y', 'dir2_z',
            'fiber_fraction_pop3',
            'axial_diffusivity_pop3',
            'radial_diffusivity_pop3',
            'fiber_fa_pop3',
            'dir3_x', 'dir3_y', 'dir3_z',
        ]
        base = base_3iso if model_mode == 3 else base_2iso
        return base + mrds_block
