"""
DBSI Adaptive Model — Stage A detection, Stage B/C tensors, Stage D fractions
===============================================================================

INPUT (fixed)
-------------
A preprocessed DWI (topup, eddy, N4 bias correction), its bvals/bvecs, a brain
mask, and the thermal noise map of dwidenoise divided by the N4 bias field
(`noise_map`, required). The noise map is sigma for every step: the Rician
correction (S^2 - 2 sigma^2, per voxel), the calibration of the lambdas, the
crossing ridge, the detection statistic and the uncertainty maps. The b=0
volumes give S0 only (at least one is needed).

PIPELINE
--------
STAGE A — detection. Regularized NNLS over [direction x (AD, RD) dictionary |
anchored isotropic grid], heavy sparsity on the anisotropic block, with a
per-voxel boost of lambda_aniso where the angular concentration is low.
Voxels with Stage A FF > `fiber_threshold` get fiber candidates;
`select_dominant_directions` finds up to two populations. Only the directions
are trusted from Stage A, not its (AD, RD).

STAGE B/C — tensors.
  - n_pop == 1: direction refined by a two-level cone search, then Stage C
    (`stagec_varpro_single_fiber`) re-solves tensor and fractions jointly on the
    raw signal.
  - n_pop == 2: the crossing FF is re-solved on the detected support with a
    light ridge kappa * (sigma/S0)^2 (`crossing_ff_resolve`), then a bounded
    joint Levenberg-Marquardt fits both tensors (`estimate_AD_RD_mrds`). The
    directions stay on the Stage A dictionary grid unless `crossing_dir_refine`
    (v1.9.0) moves each population off its node by a cone search.
  - two_vs_one_threshold (v1.9.0, off by default): instead of `min_peak_ratio`,
    every voxel with a separated secondary peak is fitted both ways and the
    crossing is kept only where it lowers the Stage D residual by at least the
    threshold, in noise variances.

STAGE D — fractions. Every voxel: NNLS over [fitted fibers | fixed iso
centroids] on the Rician-corrected signal (`iso_fraction_resolve`). These are
the reported fractions. `fiber_threshold` is applied again to this FINAL FF:
where it is <= threshold the fiber is removed and the fractions re-solved on
the iso centroids alone; a crossing population given zero weight makes the
voxel a single fiber.

Then: the detection statistic dRSS/sigma^2 (threshold off by default), the
derived channels, R2/RMSE, and the per-voxel Fisher standard errors
(`dbsi_toolbox.uncertainty`).

KNOWN LIMITS
------------
Per-population AD in crossings is not quantitative (fiber/hindered and AD/RD
trade-offs, Cramer-Rao sd ~0.6e-3): report the FF-weighted AD there. The angle
between the two populations includes the grid quantisation of the second.

MULTI-FIBER SCOPE: EXACTLY 2, FIXED
-----------------------------------
`MAX_FIBER_POPULATIONS` is 2 and not configurable ("ball and 2 sticks" is the
standard ceiling; three populations need HARDI-grade sampling, and on real
data a third population never resolved). Beyond two, this toolbox would
duplicate dedicated multi-fiber tools rather than serve its purpose
(isotropic-compartment quantification with an axonal-integrity readout).

Compartment Definitions
-----------------------
    RF  : ADC <= THRESH_RES  (0.3 x 10^-3 mm^2/s) — cells, inflammatory infiltrate
    HF  : THRESH_RES < ADC <= THRESH_WAT          — hindered extracellular water
    WF  : ADC > THRESH_WAT  (3.0 x 10^-3 mm^2/s)  — free water, CSF, oedema
    NRF : HF + WF  (Non-Restricted Fraction, 2-ISO mode only)

3-ISO (RF/HF/WF) when b_max >= 2000 and at least two non-zero shells
(`analyse_protocol`); otherwise 2-ISO (RF/NRF).

Output Channels (27)
--------------------
    0  : FF   — TOTAL fibre fraction (summed over the populations)
    1  : RF   — Restricted fraction  (ADC <= 0.3e-3)      (always valid)
    2  : HF   — Hindered fraction    (0.3e-3 < ADC <= 3.0e-3) (NaN in 2-ISO)
    3  : WF   — Free-water fraction  (ADC > 3.0e-3)       (NaN in 2-ISO)
    4  : NRF  — Non-Restricted fraction = HF + WF         (always valid)
    5  : N_POP   — number of fiber populations resolved in this voxel.
                   THREE-STATE (NaN / 0 / 1-2): see `output_map_names`.
    ── population 1 (dominant), NaN if N_POP < 1 ──
    6  : FF_POP1 — this population's share of FF
    7  : AD_POP1 — axial diffusivity  (Stage C / MRDS joint)
    8  : RD_POP1 — radial diffusivity
    9  : FA_POP1 — intrinsic fibre FA
    10-12 : DIR1_XYZ — unit direction vector
    ── population 2, NaN if N_POP < 2 ──
    13 : FF_POP2, 14: AD_POP2, 15: RD_POP2, 16: FA_POP2
    17-19 : DIR2_XYZ
    ── FF-weighted over the populations present, NaN if no fiber tensor ──
    20 : AD_W, 21: RD_W — fraction-weighted fibre diffusivities
    22 : FA_W  — FA OF the weighted tensor (intrinsic per-fibre anisotropy,
                 NOT the mean of FA_POP1 and FA_POP2)
    ── diagnostics ──
    23 : CONC  — dominant-basin angular concentration
    24 : R2    — goodness of fit of the reconstructed signal (fit_quality.py)
    25 : RMSE  — residual RMSE as a fraction of S0
    26 : NNLS_IT — Stage A solver iterations

Per-voxel standard errors of the continuous channels are written separately,
to toolbox_report/uncertainty_maps/. The `_C_*` module constants are the
single source of truth for these indices -- index through them, never through
a literal.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601. doi:10.1093/brain/awr307
Shirani A, et al. (2019). Ann Clin Transl Neurol, 6(11):2323-2327.
Jelescu IO, et al. (2016). NMR Biomed, 29(1):33-47.
Coronado-Leija R, Ramirez-Manzanares A, Marroquin JL (2017). Medical
    Image Analysis, 42, 26-43. (MRDS multi-resolution discrete search.)
Veraart J, et al. (2016). NeuroImage, 142:394-406. (MP-PCA noise map.)
"""

import numpy as np
from numba import njit, prange
from ._numba_flags import FASTMATH
import time
from tqdm import tqdm

from .core.basis import (
    build_design_matrix_exhaustive,
    generate_exhaustive_diffusivity_pairs,
    generate_fibonacci_sphere_hemisphere,
    generate_anchored_isotropic_grid,
)
from .core.solvers import (
    nnls_coordinate_descent,
    compute_regularization_matrix,
    select_dominant_directions,
    dominant_basin_concentration,  # per-voxel angular concentration
    stagec_varpro_single_fiber,  # Stage C joint tensor+fraction re-solve
    iso_fraction_resolve,  # Stage D final constrained fraction re-solve
    build_direction_neighbor_graph,  # local-maxima peak-finding graph
    estimate_AD_RD_conditioned,
    compute_fiber_fa,
    refine_fiber_direction_cone,
    compute_cone_refinement_schedule,
    measure_hemisphere_spacing,
    estimate_AD_RD_mrds,  # MRDS multi-fiber Stage B
    refine_crossing_directions_cone,  # v1.9.0 — crossing directions off the grid
    crossing_ff_resolve,          # v1.7.0 — crossing FF re-solved before the LM
)
from .calibration.data_driven import (fiber_subtracted_residual,
                                      select_lambdas_data_driven,
                                       sample_calibration_voxels,
                                       calibrate_concentration_gate_mc)
from .calibration.adaptive_n_iso import select_n_iso_svd, select_n_iso_bootstrap
from .calibration.mc_sure import crosscheck_lambda_iso_sure, crosscheck_n_iso_sure

from .utils.tools import estimate_snr_robust, B0_THRESHOLD
from .utils import tools as _tools   # letto a RUNTIME, non per valore:
#   `_SNR_LEGACY_BIASED` e' un interruttore che il chiamante puo' commutare
#   (la batteria sintetica lo fa). Importarlo per valore lo congelerebbe
#   all'import e il run report dichiarerebbe lo stimatore SBAGLIATO mentre
#   gira l'altro -- la stessa classe di bug del ramo della griglia.
from .utils.autoconfig import autoconfigure_dictionary


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FIBER_THRESHOLD = 0.15      # dimensionless — minimum FF for AD/RD estimation

# Passi della griglia isotropa di RIFERIMENTO usata (a) per sottrarre la fibra
# prima del bootstrap di n_iso e (b) come riferimento del bias dentro il
# bootstrap. Deve essere piu' ricca di ogni candidato, altrimenti il bias di
# un candidato fine sarebbe zero per costruzione.
_N_ISO_REF_STEPS = 48

THRESH_RES = 0.3e-3          # mm^2/s — restricted / hindered boundary
THRESH_WAT = 3.0e-3          # mm^2/s — hindered  / free-water boundary

# 3-ISO is now the DEFAULT for any multi-shell protocol (>=2 shells, b_max
# >= 2000). Lowered from (b_max>=3000 AND >=3 shells) on the strength of the
# iso-block identifiability analysis + the Stage D fixed-3 estimator: an SVD/
# CRLB/Monte-Carlo study showed the 3 iso components (RF/HF/WF) are supported
# with low variance down to b_max ~ 2000, and the constrained fixed-centroid
# estimator recovers them near-GT (unlike the old free anchored spectrum). The
# 2-ISO merge (RF + NRF) is now only a single-shell fallback -- and it was shown
# to be LESS accurate on RF than 3-ISO even where it applies. See project
# identifiability analysis.
B_THRESH_3ISO = 2000.0       # s/mm^2 — minimum b_max to activate 3-ISO model
MIN_SHELLS_3ISO = 2          # minimum distinct non-zero b-value shells for 3-ISO

# Stage A dictionary defaults. Deliberately coarse on the AD/RD axis.
_STAGE_A_AD_MIN = 0.5e-3
_STAGE_A_AD_MAX = 2.2e-3
_STAGE_A_RD_MIN = 0.05e-3
_STAGE_A_RD_MAX = 1.2e-3
_STAGE_A_DEFAULT_N_AD = 3
_STAGE_A_DEFAULT_N_RD = 3
# Anisotropy floor for the (AD, RD) detection grid: a pair enters the
# anisotropic block only if AD >= RD * ratio. Raised 1.15 -> 2.0 (2026-07-27)
# as the structural, root-cause complement to the FF-leakage work. On the
# coarse default grid (AD=[0.5,1.35,2.2], RD=[0.05,0.625,1.2] e-3 mm^2/s) the
# admissible AD/RD ratios are {44, 27, 10, 3.52, 2.16, 1.83}. The interval
# [1.15, 1.5) contains NO pair, so raising the ratio through it is a no-op
# (verified end-to-end: identical maps). Raising to 2.0 removes exactly ONE
# column -- the near-isotropic (AD=2.2, RD=1.2) pair (ratio 1.83), which is a
# direct source of isotropic->fiber leakage. On the 8-class heterogeneous
# synthetic brain this lifted tissue discrimination 81.2% -> 85.6%, cut GM FF
# leakage 0.51 -> 0.39 and Edema 0.43 -> 0.34, and did NOT harm the
# demyelinated fiber (its AD/RD ratio 1.5/0.7 = 2.14 is above the floor):
# WM_demyel FF 0.60 -> 0.52 (closer to GT 0.45), detection npop 0.45 -> 0.80.
# SAFE BAND [2.0, 2.16]: the next column (AD=1.35, RD=0.625, ratio 2.16)
# survives up to 2.16 and is needed for demyelinated / low-FA fibers; at 2.2
# it is lost. This is COMPLEMENTARY, not a cure: Tumor's restricted-signal
# leakage (a different mechanism) is untouched (FF ~0.50 either way) and is
# the target of the per-voxel concentration modulation (Plan A). Kept
# constructor-overridable. See Simulations/pyDBSI_pervoxel_prototype for the
# sweep harness.
_STAGE_A_DEFAULT_ANISOTROPY_RATIO = 2.0
_STAGE_A_DEFAULT_LAMBDA_BASE = 0.005

# Default isotropic spectrum range.
_DEFAULT_ISO_MIN = 0.0
_DEFAULT_ISO_MAX = 3.0e-3
_DEFAULT_N_ISO_STEPS = 31    # Legacy fixed default — see select_n_iso_svd.

# n_iso DI QUADRATURA (default dalla 1.3.9, `n_iso_method='quadrature'`).
# n_iso non e' un ordine di modello: Stage D ricalcola le frazioni riportate su
# 3 centroidi FISSI, quindi la griglia le tocca solo per via indiretta (via il
# tensore di Stage C). Si fissa per convergenza, non si seleziona per soggetto.
# Misurato su dati veri (Codes_fixed_20260923/06_convergenza_niso, 3000 voxel,
# lambda congelati, n_iso 4..40). Riferimento = media delle due densita' piu' fini
# con <=5% di NNLS non convergenti (n_iso 12 e 16); scarto peggiore su tutte le
# quantita', in unita' di tolleranza dichiarata (FA 0.01, AD 0.03e-3, frazioni 0.01):
#     n_iso  colonne  rapporto max fra atomi  scarto peggiore  NNLS non conv.
#       4       9           1.92               1.11 (FA)         0%
#       6      11           1.67               0.49 (HF)         0%
#       8      10           1.75               1.12 (FA)         0%
#      10      12           1.54               0.28 (FA)         0%
#      12      14           1.43               riferimento      0.8%
#      40      42           1.11               0.67 (HF)        40%
# Le frazioni restano entro 0.6 tolleranze a OGNI densita'. Il tensore segue il
# rapporto massimo fra atomi adiacenti, non n_iso (Spearman AD -0.96, FA -0.84),
# ed e' per questo che 8 va peggio di 6. Si sceglie 6 perche': entro tolleranza,
# 0% NNLS non convergenti (crescono con le colonne: 40% a n_iso=40), il piu'
# veloce, ed e' la griglia su cui sono stati calibrati i lambda congelati di
# coorte -- la penalita' L2 su lambda_iso non e' invariante per cambio di griglia.
# NON 4: e' cio' che il ripiego SVD restituisce sempre, e fallisce su FA.
_QUADRATURE_N_ISO = 6

_ISO_GRID_D_MAX_EXTENDED = 5.0e-3

# Maximum number of fiber populations Stage A/B reports per voxel. FIXED at 2,
# deliberately: it is the practical ceiling for the angular sampling of the
# clinical protocols this toolbox targets, and a third population was never
# resolvable on real data (the pop3 output channels came back 100% NaN on every
# run). It was briefly a per-instance parameter with an opt-in value of 3; that
# option is gone, along with the pop3 output block it fed. See module docstring
# "MULTI-FIBER SCOPE".
MAX_FIBER_POPULATIONS = 2

# ── OUTPUT CHANNEL LAYOUT ────────────────────────────────────────────────────
# Single source of truth for the channel indices: the njit kernels, the class,
# `output_map_names` and `fit_quality` both index
# through these names, so the layout is defined in exactly one place.
#
# NaN/0 CONVENTION, by block:
#   0-4   compartment fractions   -- 0 means "compartment absent" (0 is the
#                                    correct physical value; they stay summable)
#   5     n_fiber_populations     -- THREE-STATE, see `output_map_names`
#   6-19  per-population block    -- NaN means "this population is absent"
#   20-22 FF-weighted aggregates  -- NaN where no fiber tensor was estimated
#   23-26 diagnostics             -- NaN outside fitted voxels
#
# v1.7.0: `mean_iso_adc` is no longer an output. It was the mean of Stage A's
# over-complete iso spectrum, which Stage D does not use: not coherent with the
# reported fractions and not comparable between branches (inspection
# 2026-09-30). It survives as an INTERNAL scratch channel (_C_ADC_ISO, after the
# public ones) because the 'recovered' fit-quality reconstruction -- used only
# when Stage D is off -- needs it; `fit` drops it before returning.
_C_FF = 0             # fiber_fraction -- TOTAL anisotropic fraction (pop1+pop2)
_C_RF = 1             # restricted_fraction
_C_HF = 2             # hindered_fraction      (NaN in 2-ISO)
_C_WF = 3             # water_fraction         (NaN in 2-ISO)
_C_NRF = 4            # nonrestricted_fraction (== HF+WF in 3-ISO)
_C_NPOP = 5           # n_fiber_populations
_C_FF1 = 6            # ── population 1 (dominant) ──
_C_AD1 = 7
_C_RD1 = 8
_C_FA1 = 9
_C_DIR1 = 10          # dir1_x, dir1_y, dir1_z = 10, 11, 12
_C_FF2 = 13           # ── population 2 ──
_C_AD2 = 14
_C_RD2 = 15
_C_FA2 = 16
_C_DIR2 = 17          # dir2_x, dir2_y, dir2_z = 17, 18, 19
_C_ADW = 20           # ── FF-weighted over the populations present ──
_C_RDW = 21
_C_FAW = 22
_C_CONC = 23          # dominant_basin_concentration (diagnostic)
_C_R2 = 24            # fit_r2   -- goodness of fit of the reconstructed signal
_C_RMSE = 25          # fit_rmse -- residual RMSE, as a fraction of S0
_C_NNLS_IT = 26       # nnls_iterations -- Stage A solver iterations (diagnostic)
_N_CHANNELS = 27      # PUBLIC channels: what `fit` returns and what is saved
_C_ADC_ISO = 27       # INTERNAL scratch: mean of Stage A's iso spectrum (see above)
_N_CHANNELS_INTERNAL = 28

# MRDS multi-fiber Stage B defaults (see core.solvers.estimate_AD_RD_mrds).
_MRDS_INIT_N_ITER = 3        # short, deliberately non-converged alternating warm start
_MRDS_LM_MAX_ITER = 25
# Cone over the crossing directions (v1.9.0): LM iterations per candidate, warm
# started from the current solution, and the default of the option.
_MRDS_CONE_LM_ITER = 10
_DEFAULT_CROSSING_DIR_REFINE = False

# TWO-VS-ONE POPULATION TEST (v1.9.0). How many populations a voxel has was
# decided by `min_peak_ratio` alone: a second fiber is accepted when its angular
# basin holds >= 0.35 of the dominant's mass, a cut set by hand where the
# single-fiber and crossing distributions overlap. Measured on the 1.8.1
# simulations (P3 battery, SNR 15/26/40): 40-55% of single fibers with high RD or
# low FF got a false second population and ~half of unbalanced crossings
# (0.45/0.20, true ratio 0.44) were read as one fiber, at EVERY SNR -- the cut,
# not the noise. The test replaces the cut with the model selection the original
# MRDS uses: every voxel with a separated secondary peak (candidate ratio 0) is
# fitted both as a crossing and as a single fiber, and the crossing is kept only
# if it lowers the Stage D residual by at least `two_vs_one_threshold` noise
# variances:
#     stat21 = (RSS_1pop - RSS_2pop) / sigma_voxel^2
# on the Stage D model (fixed iso centroids, fitted tensors and directions,
# corrected signal), BEFORE the fiber threshold -- otherwise a weak single fiber
# removed by the threshold turns the comparison into 2-vs-0. With the crossing
# directions on the grid (crossing_dir_refine off) the 2-population model starts
# handicapped and stat21 is negative on single fibers; with the cone the null
# behaves (median 1-5, p95 7-18). Emulated at threshold 12 with the cone:
# populations right 57->73.5 / 69->85 / 73->90 % at SNR 15/26/40, FF MAE down,
# RD MAE up at SNR 15-26 (heterogeneous crossings demoted at low SNR).
# Off by default until the cohort-level decision.
_DEFAULT_TWO_VS_ONE_THRESHOLD = None
_TWO_VS_ONE_CANDIDATE_RATIO = 0.0     # any separated secondary peak is a candidate

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
#
# SPENTO DI DEFAULT dalla 1.4.0 (0.0 = il gate non rifiuta mai; resta la soglia
# `fiber_threshold` sulla frazione di fibra). La storia sopra e' stata smentita:
#   - sul fantoccio un crossing a 90 gradi VERO ha concentrazione 0.421 e un
#     voxel tipo tumore 0.414: il gate non puo' distinguerli, perche' un crossing
#     ad angolo largo e' angolarmente diffuso PER COSTRUZIONE. Col nullo MC vero
#     la detezione crolla sopra i 60 gradi: il gate rifiuta i crossing che il
#     dizionario sa risolvere e ammette quelli che non sa risolvere;
#   - sui dati veri (Codes_fixed_20260923/07_gate_su_dati_veri, 4000 voxel,
#     tutto congelato tranne il gate) il gate congelato 0.4535 toglieva la fibra
#     al 19.5% dei voxel; quei voxel, rifiutati, lasciano un residuo di 1.80
#     sigma, e ammessi scendono a 1.03 sigma (il 97% diventa crossing). La
#     riduzione e' 0.57x, contro 0.87-0.94x attese dai soli gradi di liberta'.
#     A OGNI gradino dello sweep (0.30 -> 0.55) i voxel persi passano da ~1.0 a
#     ~1.8 sigma: non esiste un punto di lavoro;
#   - calibrato per sessione, era il parametro piu' legato all'SNR (Spearman
#     -0.86 su 40 soggetti) e la causa radice del divario BR/BT.
# Resta disponibile: `min_dominant_concentration=<valore>` oppure
# `fit(calibrate_concentration_gate=True)` per il nullo MC. Attenzione: se il
# nullo MC e' troppo rado ripiega sul valore del costruttore, che ora e' 0.0 --
# cioe' "nessun gate"; il messaggio "FELL BACK" lo dichiara.
_DEFAULT_MIN_DOMINANT_CONCENTRATION = 0.0
# Ma spegnerlo lascia aperto il problema che il gate doveva risolvere, e lo
# lascia GRANDE a SNR clinico: sul tessuto isotropo sintetico (P3, SNR 26) il
# 38-62% dei voxel riceve una fibra falsa, quasi sempre come crossing. Lo
# strumento giusto non guarda la forma dei pesi di Stage A ma se le fibre
# spiegano piu' del rumore: `fiber_detection_threshold` (v1.6.1, spento finche'
# batteria e dati veri non scelgono la soglia), vedi _fiber_detection_pass.

# Plan A (Point 2) — per-voxel concentration-modulated lambda_aniso defaults.
# ON by default (disable via lambda_aniso_conc_mod=False): when a voxel's
# dominant-basin angular concentration is low (diffuse anisotropic weight, i.e.
# isotropic->fiber leakage) the anisotropic regularization is boosted and the
# voxel re-solved, suppressing the spurious fiber_fraction; concentrated
# (genuine-fiber) voxels are left untouched. Ramp: c>=c_hi -> no boost;
# c<=c_lo -> full boost (xgain on lambda_aniso); linear between. Endpoints
# bracket the observed FIRST-PASS concentration bands on the validation brain
# (leakage Tumor ~0.23 / GM ~0.27; demyelinated fiber ~0.35; healthy ~0.47-0.51)
# so leakage is boosted hard while a demyelinated fiber gets only a mild,
# still-detectable boost. See `_apply_conc_modulation` and
# Simulations/pyDBSI_pervoxel_prototype.
#
# End-to-end ramp sweep (8-class heterogeneous brain, ratio=2.0, SNR30; FF of
# Tumor[GT0]/GM[GT0.10]/demyel[GT0.45], tissue discrimination, leakage metric):
#   config              discr  Tumor  GM   demyel  leak
#   mod OFF (baseline)  0.856  0.50   0.39 0.52    0.267
#   A c_hi0.40 g12      0.906  0.08   0.14 0.30    0.057   <- max leak-suppression
#   B c_hi0.34 g12      0.863  0.10   0.16 0.42    0.068
#   D c_hi0.36 g8       0.887  0.13   0.18 0.40    0.080   <- DEFAULT (balanced)
# The DEFAULT (config D) is the balanced operating point: it keeps ~3x of the
# leakage fix (Tumor 0.50->0.13, leak 0.267->0.080) and +3.1pp discrimination
# while PRESERVING the demyelinated fiber (FF 0.40, vs GT 0.45 -- baseline
# actually OVER-estimated it at 0.52). The gentler gain (8, not 12) is also the
# conservative choice against low-SNR over-correction (where genuine fibers
# diffuse toward the leakage concentration band). Config A (c_hi=0.40, gain=12)
# is the aggressive alternative -- best discrimination and leak suppression, but
# it over-corrects the demyelinated fiber (FF 0.30) -- pass those values
# explicitly if fiber-fraction fidelity on low-FA fibers is not the priority.
# VALIDATED (2026-07-27): SPARES crossings (Sc1 FF0.80 untouched 0.82->0.82 at
# SNR30, 0.73->0.72 at SNR15 -- the exact case a scalar high lambda_aniso
# collapsed to 0.42; a crossing stays concentrated ~0.42 and escapes the boost)
# and suppresses pure-iso/pure-restr leakage at SNR30 AND SNR15. At SNR15 it
# CORRECTS the low-SNR over-estimation instead of collapsing fibers (demyel
# 0.75->0.47~=GT). Weak-fiber edge (FF 0.12-0.20, incl. demyelinated) at SNR15
# and SNR10: NO false negatives -- detection rate stays 1.0, weak fibers are
# pulled toward GT, never zeroed. KNOWN LIMIT (fundamental, not a bug): at low
# SNR a weak/demyelinated fiber's concentration overlaps the leakage band
# (SNR15: weak-demyel ~0.29 vs fiber-free ~0.29), so the modulation cannot
# preferentially preserve a weak fiber's FF -- it suppresses it like leakage.
# mod-OFF cannot separate them either; the lever simply adds no discriminating
# power where the low-SNR signal has none.
# Default flipped ON (2026-07-27) on the strength of this validation (crossings
# spared, leakage suppressed at SNR30/15, low-SNR over-estimation corrected, no
# weak-fiber false negatives). REAL-DATA confirmation is still pending; disable
# per-call with lambda_aniso_conc_mod=False (or --disable-conc-modulation on the
# CLI) if a dataset shows unexpected fiber_fraction suppression.
_DEFAULT_LAMBDA_ANISO_CONC_MOD = True
_DEFAULT_CONC_MOD_C_LO = 0.24
_DEFAULT_CONC_MOD_C_HI = 0.36
_DEFAULT_CONC_MOD_GAIN = 8.0

# Stage C — constrained joint (VARPRO) mono-fiber re-solve of tensor+fractions.
# ON by default (disable via stagec_refine=False). For n_pop==1 voxels, after
# Stage A detection, the fiber tensor (AD, RD) and all compartment fractions are
# re-fitted JOINTLY by minimising the reconstruction residual over a reduced
# dictionary [fiber_col(AD,RD,dir) | iso_grid] -- fixing the decoupled-stage
# mutual bias (fiber_fraction inflation + restricted collapse + AD under-
# estimation). See `core.solvers.stagec_varpro_single_fiber`. (AD, RD) search
# grid + admissibility ratio: coarse grid then a local refine in the helper.
#
# VALIDATED (2026-07-28, 8-class heterogeneous brain, SNR30): decisively fixes
# the HIGH-FA fiber_fraction inflation that was the bias's worst case --
# WM_sano FF 0.72->0.57 (GT 0.55), restricted 0.00->0.09 (GT 0.10), AD
# 1.15->1.65 (GT 1.70). Correct BY CONSTRUCTION on non-inflated fibers (demyel
# FF left at 0.40, it was never inflated) and on crossings (n_pop>=2 untouched).
# KNOWN RESIDUALS (accepted, shipped ON): (1) minor AD overshoot on some
# moderate-FA single fibers; (2) low-FA AD stays under-estimated -- a
# fundamental identifiability limit, not fixable by tuning; (3) the crude
# nearest-centroid discrimination proxy dips ~3pp because an ACCURATE WM_sano
# sits nearer its neighbours in [FF,RF,NRF] space -- the fractions themselves
# are net MORE accurate. Real-data confirmation still pending; opt out with
# stagec_refine=False (or --disable-stagec on the CLI).
_DEFAULT_STAGEC_REFINE = True

# Run the MRDS-lite cone refinement BEFORE Stage C and hand Stage C the refined
# direction. The two used to sit in mutually exclusive branches, so with Stage C
# on by default the cone never ran and a single fiber's direction stayed on
# Stage A's discrete grid. They address orthogonal problems -- the cone fixes
# angular quantisation, Stage C fixes the fraction/tensor bias -- so they
# compose.
#
# VALIDATED 2026-09-19 (`experiments/exp_cone_multiproto.py`, out of the repository): 3 protocols
# (n_dirs 36/62/78, i.e. 22.8/17.2/15.4 deg grid spacing) x 3 SNR (20/30/50),
# 200 synthetic single-fiber voxels each, AD and RD swept per voxel, true
# directions uniform on the hemisphere so they fall between grid nodes. Paired,
# mean absolute error, refined vs unrefined:
#
#   AD  improves in 9/9 conditions, median -18.5% (worst case -5.2%)
#   RD  improves in 9/9,            median -10.1%
#   RF  improves in 9/9,            median  -8.7%
#   FF  improves in 7/9,            median  -8.8% (worst case +2.1%)
#   angular error 7.21-10.82 deg -> 1.78-5.60 deg, always
#
# Runtime cost is within measurement noise. The gain scales with grid spacing
# (P4-like -20.5%, P3-like -16.7%, P1-like -12.9%), which is the mechanism
# working as expected. It does NOT scale the way first predicted with SNR: at
# SNR 20 it drops to -7.8% against -22.8% at SNR 30, because the baseline AD
# error is noise-dominated there (21.2% vs 16.9%) and removing a fixed
# systematic term buys proportionally less.
#
# Single-fiber voxels only: crossings use Stage A's raw grid directions either
# way. Synthetic only -- not yet confirmed on real data.
_DEFAULT_STAGEC_DIR_REFINE = True
_STAGEC_AD_MIN = 0.6e-3
_STAGEC_AD_MAX = 2.6e-3
_STAGEC_N_AD = 14
_STAGEC_RD_MIN = 0.15e-3   # Leva 1: physiological RD floor (was 0.05e-3) — fiber-compartment
                           # perpendicular diffusivity below ~0.1e-3 is non-physiological and lets
                           # noise push RD->0 -> FA->1 on real data. Caps reported FA at ~0.9.
_STAGEC_RD_MAX = 1.1e-3
_STAGEC_N_RD = 12
_STAGEC_ANISO_RATIO = 1.1
# Stage C reuses the model's iso_grid. A dedicated fine geomspace iso grid was
# tried (to give denser restricted-band resolution) and REVERTED: it did not
# improve the low-FA / AD cases and slightly lowered discrimination (0.856 ->
# 0.831 on the validation brain). The residual low-FA AD under-estimation is a
# fundamental identifiability limit (a low-FA fiber is nearly isotropic, so
# (AD,RD) are weakly constrained), not an iso-grid coverage issue.
# Stage C fits the RAW normalised signal, NOT the Rician-corrected data_corr:
# the pipeline's noise-floor subtraction sqrt(max(S^2 - 2 sigma^2, 0)) distorts
# the high-b decay (over-subtraction + the >=0 clamp) and biases the fiber
# tensor low (validated: raw -> AD 1.72/FF 0.56/RF 0.09 vs corrected ->
# AD 1.32/FF 0.73 on a GT-1.70 fiber; both NNLS solvers agree, so it is the
# input signal, not the solver). Stage A detection still uses data_corr.

# Stage D — final constrained compartment-fraction re-solve (ON by default;
# disable via iso_resolve=False). For EVERY fitted voxel, the reported
# fractions are re-estimated by NNLS over [detected fibers | a few FIXED iso
# centroids] on the RICIAN-CORRECTED signal, replacing the over-complete Stage A
# spectrum (whose free anchored grid smears weight and mis-bins RF/HF/WF). The
# identifiability analysis showed 3 iso components are supported down to
# b_max~2000, and the fixed-3 estimator recovers RF/HF/WF near-GT with low
# variance (incl. the high-RF of tumor-like voxels, exactly) where the anchored
# spectrum failed. CORRECTED signal here (opposite to Stage C's RAW): the
# restricted fraction is read from the high-b plateau, which the raw Rician
# noise floor mimics -> corrected de-confounds it. Iso centroids: RF/HF/WF for
# 3-ISO, RF/NRF for the (single-shell) 2-ISO fallback.
_DEFAULT_ISO_RESOLVE = True

# v1.7.0 — crossing FF re-solved before the MRDS LM (see core.solvers,
# crossing_ff_resolve). Ridge = kappa * (sigma / S0_voxel)^2. kappa 65 centres
# healthy crossing FF on P3, HCP-like and a 2-shell protocol at SNR 26 and 15;
# 30 keeps more demyelination contrast at SNR 15 (P3: 0.20 vs 0.15 of a true
# 0.40) for healthy-crossing FF +0.04..+0.06 (1.6.x: -0.11..-0.17) and has the
# best worst case across protocols. 0 = Stage A's FF held fixed (Stage D keeps
# it). Needs Stage D: without it kappa is ignored.
_DEFAULT_CROSSING_FF_KAPPA = 30.0

# v1.7.1 — the thermal noise map is REQUIRED (decision 2026-10-05): the b=0
# temporal sigma also holds pulsation and motion (1.3-2.2x the thermal noise on
# real P3 data, differently per subject), and every sigma-normalised step --
# the Rician correction first -- would inherit a subject-dependent bias. Set to
# True ONLY for synthetic tests/simulations or to reproduce pre-1.7.1 results;
# the run report records it (sigma_source 'b0_temporal').
_ALLOW_B0_SIGMA = False

_NOISE_MAP_HELP = (
    "noise_map is required (v1.7.1): the thermal noise sigma per voxel, on the grid and in the "
    "intensity scale of the DWI passed to fit. With MRtrix: 'dwidenoise in out -noise noise.nii.gz', "
    "'dwibiascorrect ants ... -bias bias.nii.gz', then 'mrcalc noise.nii.gz bias.nii.gz -div -finite "
    "noise.nii.gz bias.nii.gz -div 0 -if noise_N4.nii.gz'. The b=0 temporal sigma is not a substitute: "
    "it also contains physiological variability (README, 'Noise').")
_ISO_RESOLVE_D_3ISO = (0.15e-3, 1.0e-3, 3.0e-3)   # RF, HF, WF centroids
_ISO_RESOLVE_D_2ISO = (0.15e-3, 1.5e-3)            # RF, NRF centroids (single-shell)

# Monte-Carlo null calibration of the concentration gate (see
# calibration.data_driven.calibrate_concentration_gate_mc). When calibration
# runs, the gate is set to this percentile of the dominant-basin concentration
# a fiber-FREE (pure-isotropic) voxel produces under THIS
# protocol/dictionary/lambdas/SNR — data-driven and protocol-general, replacing
# the fixed default above. NOTE: the pure-iso and real-fiber concentration
# distributions OVERLAP, so a too-high percentile enters the signal body and
# suppresses REAL fibers: an empirical percentile sweep (SNR30) found the 99th
# gives gate~0.54 and collapses true detection (T1 npop 1.7->0.22), while
# 85-95th keep full detection AND suppress pure-iso (npop~0.07). 90th (gate
# ~0.39) reproduces the hand-validated 0.35 behaviour, now data-driven.
_CONCENTRATION_GATE_PERCENTILE = 90.0
_CONCENTRATION_GATE_N_MC = 400


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
# PARALLEL FITTING KERNELS — Stage A, selection, Stage C / MRDS, Stage D
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=FASTMATH)
def _apply_conc_modulation(w, AtA_reg, Aty, n_aniso_cols, n_dirs, n_pairs,
                           neighbor_idx, fiber_dirs, lambda_aniso,
                           c_lo, c_hi, gain):
    """
    Plan A (Point 2) — per-voxel angular-concentration modulation of
    lambda_aniso.

    Given a FIRST-PASS Stage-A solution `w` (solved with the shared,
    protocol-level lambda_aniso baked into `AtA_reg`), measure its dominant-
    basin angular concentration `c` (see `core.solvers.
    dominant_basin_concentration`) and, when `c` is low, RE-SOLVE the same
    voxel with a boosted anisotropic regularization. This suppresses the
    diffuse anisotropic weight that fiber-free / leakage voxels absorb from
    the isotropic signal (which inflates fiber_fraction), while leaving genuine
    fibers — which concentrate their weight — essentially untouched.

    Rationale for a CONTINUOUS ramp rather than a binary gate: the concentration
    of a demyelinated / low-FA fiber (~0.36 on the validation brain) sits BELOW
    that of a healthy fiber (~0.48) and only modestly ABOVE diffuse leakage
    (~0.25) — the distributions overlap at the tails. A hard threshold at any
    single value therefore either misses leakage or crushes the demyelinated
    fiber (the npop binary gate was falsified for exactly this reason). A
    continuous ramp instead applies FULL boost only to clearly-diffuse voxels
    (c <= c_lo), NONE to clearly-concentrated ones (c >= c_hi), and a
    proportional, gentle boost to the ambiguous middle — so a demyelinated
    fiber loses only a little anisotropic weight (stays detectable) while
    leakage is strongly suppressed. Voxels with almost no anisotropic weight
    (e.g. CSF: high concentration but negligible FF) sit above c_hi and are
    never modulated, so this cannot manufacture leakage where there was none.

    The boost is applied by adding `(g - 1) * lambda_aniso` to the anisotropic
    block's diagonal of a LOCAL copy of `AtA_reg` (the isotropic block and the
    shared matrix are left untouched), then re-running the same NNLS.

    Parameters
    ----------
    w : array (n_total_cols,)
        First-pass NNLS solution (anisotropic block first).
    AtA_reg : array (n_total_cols, n_total_cols)
        Shared regularized Gram matrix (read-only; copied locally if boosting).
    Aty : array (n_total_cols,)
        A^T y for this voxel (unchanged by the boost).
    n_aniso_cols, n_dirs, n_pairs : int
        Dictionary dimensions (n_aniso_cols == n_dirs * n_pairs).
    neighbor_idx, fiber_dirs : arrays
        Passed through to `dominant_basin_concentration`.
    lambda_aniso : float
        The protocol-level scalar lambda_aniso already baked into AtA_reg's
        anisotropic diagonal; the boost is expressed relative to it.
    c_lo, c_hi : float
        Ramp endpoints. c >= c_hi -> no boost (g=1); c <= c_lo -> full boost
        (g=gain); linear in between.
    gain : float
        Maximum multiplicative boost on lambda_aniso at c <= c_lo.

    Returns
    -------
    w : array (n_total_cols,)
        The re-solved solution if a boost was applied, else the input `w`.
    """
    w_aniso = w[:n_aniso_cols]
    c = dominant_basin_concentration(w_aniso, n_dirs, n_pairs,
                                     neighbor_idx, fiber_dirs)
    if c >= c_hi:
        return w
    frac = (c_hi - c) / (c_hi - c_lo)
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    g = 1.0 + (gain - 1.0) * frac
    if g <= 1.0:
        return w
    delta = (g - 1.0) * lambda_aniso
    AtA_mod = AtA_reg.copy()
    for i in range(n_aniso_cols):
        AtA_mod[i, i] += delta
    w2, _ = nnls_coordinate_descent(AtA_mod, Aty, 0.0)
    return w2


@njit(cache=True, fastmath=FASTMATH)
def _single_fiber_2iso(arr, x, y, z, dominant_dir, sig_norm, bvals, bvecs, b0_thr,
                       f_fib, f_res, f_nonrf, D_res_c, D_nonrf_c,
                       stagec_enabled, stagec_dir_refine, enable_direction_refinement,
                       cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                       data_raw, stagec_iso_grid, iso_forward, iso_gram,
                       stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio):
    """Single-fiber Stage B/C for one voxel, written into `arr` at (x, y, z): cone on
    the direction, then Stage C (or the closed-form Stage B without it). The body of the
    kernels' n_pop == 1 branch, COPIED here (v1.9.0) so that the two-vs-one test can fit
    the single-fiber alternative of a crossing candidate from the same Stage A. A copy, not
    a move: with the kernels calling this helper the default fit moved by 1 float32 ulp
    (~6e-8, 17 channels, same machine): fastmath optimises the helper differently from the
    inlined code, and the default must stay bit-identical to 1.8.1. Keep the two in step
    when editing either.
    2-ISO."""

    if stagec_enabled:
        # Refine the direction first, then hand the refined one to
        # Stage C (see _DEFAULT_STAGEC_DIR_REFINE).
        if stagec_dir_refine and enable_direction_refinement:
            dominant_dir, _, _ = refine_fiber_direction_cone(
                bvals, bvecs, sig_norm, dominant_dir,
                f_fib, f_res, f_nonrf, 0.0,
                D_res_c, D_nonrf_c, 0.0, False,
                cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
            )
        # ── STAGE C: joint (VARPRO) tensor+fraction re-solve ──
        # Replaces the raw over-complete-block fractions AND the
        # decoupled Stage B tensor with a residual-minimising joint
        # fit on [fiber_col(AD,RD,dir) | iso_grid]. Fixes the
        # fiber_fraction inflation + restricted collapse + AD
        # under-estimation (see stagec_varpro_single_fiber).
        # Uses the RAW normalised signal (data_raw), not data_corr:
        # the Rician noise-floor subtraction biases the tensor low.
        sig_raw = data_raw[x, y, z]
        s0_raw = 0.0
        cnt_raw = 0
        for ii in range(len(bvals)):
            if bvals[ii] < b0_thr:
                s0_raw += sig_raw[ii]
                cnt_raw += 1
        if cnt_raw > 0:
            s0_raw /= cnt_raw
        if s0_raw < 1e-6:
            s0_raw = 1e-6
        sig_norm_raw = (sig_raw / s0_raw).astype(np.float64)
        n_iso_sc = stagec_iso_grid.shape[0]
        w_sc = np.zeros(1 + n_iso_sc)
        AD_est, RD_est = stagec_varpro_single_fiber(
            sig_norm_raw, bvals, bvecs, dominant_dir,
            iso_forward, iso_gram, stagec_ad_grid, stagec_rd_grid,
            stagec_aniso_ratio, w_sc
        )
        tot_sc = 0.0
        for i in range(1 + n_iso_sc):
            tot_sc += w_sc[i]
        if tot_sc > 1e-10:
            res_sc = 0.0
            nonrf_sc = 0.0
            wd_sc = 0.0
            for i in range(n_iso_sc):
                wi = w_sc[1 + i]
                adc = stagec_iso_grid[i]
                if adc <= THRESH_RES:
                    res_sc += wi
                else:
                    nonrf_sc += wi
                wd_sc += wi * adc
            f_fib = w_sc[0] / tot_sc
            f_res = res_sc / tot_sc
            f_nonrf = nonrf_sc / tot_sc
            sum_iso_sc = res_sc + nonrf_sc
            arr[x, y, z, _C_FF] = f_fib
            arr[x, y, z, _C_RF] = f_res
            arr[x, y, z, _C_NRF] = f_nonrf
            arr[x, y, z, _C_ADC_ISO] = wd_sc / sum_iso_sc if sum_iso_sc > 1e-10 else 0.0
    elif enable_direction_refinement:
        # the refined direction is what AD/RD were fitted at, so it
        # is also what gets stored (it used to be discarded, and the
        # coarse grid node written to dir1 instead -- leaving the
        # stored direction inconsistent with the stored tensor).
        dominant_dir, AD_est, RD_est = refine_fiber_direction_cone(
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

    arr[x, y, z, _C_AD1] = AD_est
    arr[x, y, z, _C_RD1] = RD_est
    arr[x, y, z, _C_FA1] = FA
    arr[x, y, z, _C_DIR1] = dominant_dir[0]
    arr[x, y, z, _C_DIR1 + 1] = dominant_dir[1]
    arr[x, y, z, _C_DIR1 + 2] = dominant_dir[2]

@njit(cache=True, fastmath=FASTMATH)
def _single_fiber_3iso(arr, x, y, z, dominant_dir, sig_norm, bvals, bvecs, b0_thr,
                       f_fib, f_res, f_hin, f_wat, D_res_c, D_hin_c, D_wat_c,
                       stagec_enabled, stagec_dir_refine, enable_direction_refinement,
                       cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                       data_raw, stagec_iso_grid, iso_forward, iso_gram,
                       stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio):
    """Single-fiber Stage B/C for one voxel, written into `arr` at (x, y, z): cone on
    the direction, then Stage C (or the closed-form Stage B without it). The body of the
    kernels' n_pop == 1 branch, COPIED here (v1.9.0) so that the two-vs-one test can fit
    the single-fiber alternative of a crossing candidate from the same Stage A. A copy, not
    a move: with the kernels calling this helper the default fit moved by 1 float32 ulp
    (~6e-8, 17 channels, same machine): fastmath optimises the helper differently from the
    inlined code, and the default must stay bit-identical to 1.8.1. Keep the two in step
    when editing either.
    3-ISO."""

    if stagec_enabled:
        # Refine the direction first, then hand the refined one to
        # Stage C (see _DEFAULT_STAGEC_DIR_REFINE).
        if stagec_dir_refine and enable_direction_refinement:
            dominant_dir, _, _ = refine_fiber_direction_cone(
                bvals, bvecs, sig_norm, dominant_dir,
                f_fib, f_res, f_hin, f_wat,
                D_res_c, D_hin_c, D_wat_c, True,
                cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
            )
        # ── STAGE C: joint (VARPRO) tensor+fraction re-solve (3-ISO
        # binning RES/HIN/WAT). Uses the RAW normalised signal
        # (data_raw), not data_corr. See stagec_varpro_single_fiber. ──
        sig_raw = data_raw[x, y, z]
        s0_raw = 0.0
        cnt_raw = 0
        for ii in range(len(bvals)):
            if bvals[ii] < b0_thr:
                s0_raw += sig_raw[ii]
                cnt_raw += 1
        if cnt_raw > 0:
            s0_raw /= cnt_raw
        if s0_raw < 1e-6:
            s0_raw = 1e-6
        sig_norm_raw = (sig_raw / s0_raw).astype(np.float64)
        n_iso_sc = stagec_iso_grid.shape[0]
        w_sc = np.zeros(1 + n_iso_sc)
        AD_est, RD_est = stagec_varpro_single_fiber(
            sig_norm_raw, bvals, bvecs, dominant_dir,
            iso_forward, iso_gram, stagec_ad_grid, stagec_rd_grid,
            stagec_aniso_ratio, w_sc
        )
        tot_sc = 0.0
        for i in range(1 + n_iso_sc):
            tot_sc += w_sc[i]
        if tot_sc > 1e-10:
            res_sc = 0.0
            hin_sc = 0.0
            wat_sc = 0.0
            wd_sc = 0.0
            for i in range(n_iso_sc):
                wi = w_sc[1 + i]
                adc = stagec_iso_grid[i]
                if adc <= THRESH_RES:
                    res_sc += wi
                elif adc <= THRESH_WAT:
                    hin_sc += wi
                else:
                    wat_sc += wi
                wd_sc += wi * adc
            f_fib = w_sc[0] / tot_sc
            f_res = res_sc / tot_sc
            f_hin = hin_sc / tot_sc
            f_wat = wat_sc / tot_sc
            sum_iso_sc = res_sc + hin_sc + wat_sc
            arr[x, y, z, _C_FF] = f_fib
            arr[x, y, z, _C_RF] = f_res
            arr[x, y, z, _C_HF] = f_hin
            arr[x, y, z, _C_WF] = f_wat
            arr[x, y, z, _C_NRF] = f_hin + f_wat
            arr[x, y, z, _C_ADC_ISO] = wd_sc / sum_iso_sc if sum_iso_sc > 1e-10 else 0.0
    elif enable_direction_refinement:
        # the refined direction is what AD/RD were fitted at, so it
        # is also what gets stored (it used to be discarded, and the
        # coarse grid node written to dir1 instead -- leaving the
        # stored direction inconsistent with the stored tensor).
        dominant_dir, AD_est, RD_est = refine_fiber_direction_cone(
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

    arr[x, y, z, _C_AD1] = AD_est
    arr[x, y, z, _C_RD1] = RD_est
    arr[x, y, z, _C_FA1] = FA
    arr[x, y, z, _C_DIR1] = dominant_dir[0]
    arr[x, y, z, _C_DIR1 + 1] = dominant_dir[1]
    arr[x, y, z, _C_DIR1 + 2] = dominant_dir[2]

@njit(parallel=True, cache=True, fastmath=FASTMATH)
def _fit_voxels_2iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction, min_separation_cos,
                        min_peak_ratio, min_dominant_concentration,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        neighbor_idx, max_fiber_populations, out,
                        lambda_aniso_scalar, conc_mod_enabled,
                        conc_mod_c_lo, conc_mod_c_hi, conc_mod_gain,
                        stagec_enabled, iso_forward, iso_gram,
                        stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio,
                        data_raw, stagec_iso_grid, stagec_dir_refine,
                        crossing_ff_kappa, sigma_map, crossing_dir_refine,
                        two_vs_one, alt_out):
    """
    v3 parallel fitting kernel — two-compartment isotropic model (2-ISO).

    Stage A direction detection is unchanged. Stage B branches on how many
    populations Stage A reported (n_pop, up to MAX_FIBER_POPULATIONS):
      n_pop == 1 : unchanged single-fiber path (closed-form Stage B,
                   optional MRDS-lite cone refinement).
      n_pop >= 2 : NEW MRDS multi-fiber joint Stage B
                   (`estimate_AD_RD_mrds`), from Stage A's grid-node
                   directions; with `crossing_dir_refine` (v1.9.0) each
                   population is then moved off its node by a cone search.
    How many populations reach this kernel is `min_peak_ratio`'s call. With the
    two-vs-one test (v1.9.0, `two_vs_one`) every crossing candidate is ALSO fitted
    as a single fiber, from the same Stage A, into `alt_out`; the caller keeps the
    better Stage D model (see _DEFAULT_TWO_VS_ONE_THRESHOLD). `alt_out` is never
    touched when `two_vs_one` is False.

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

        # The solver's iteration count is NOT discarded: written out below as
        # `nnls_iterations`. Hitting _NNLS_MAX_ITER means Stage A did not
        # converge in this voxel, which until v1.3.2 was undetectable — every
        # call site dropped the counter. Written here, before any early exit, so
        # the diagnostic exists even for voxels that leave the loop early.
        w, _nnls_it = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
        out[x, y, z, _C_NNLS_IT] = _nnls_it

        # Plan A: per-voxel concentration-modulated lambda_aniso (on by default;
        # skipped when conc_mod_enabled is False).
        if conc_mod_enabled:
            w = _apply_conc_modulation(
                w, AtA_reg, Aty, n_aniso_cols, n_dirs, n_pairs,
                neighbor_idx, fiber_dirs, lambda_aniso_scalar,
                conc_mod_c_lo, conc_mod_c_hi, conc_mod_gain
            )

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
        out[x, y, z, _C_FF] = f_fib
        out[x, y, z, _C_RF] = f_res
        out[x, y, z, _C_NRF] = f_nonrf
        out[x, y, z, _C_ADC_ISO] = mean_iso_adc

        # Per-voxel angular concentration of the anisotropic weight
        # (diagnostic channel; also the modulation lever for Plan A). Computed
        # for EVERY fitted voxel, independent of fiber_threshold, so leakage
        # voxels (diffuse -> low concentration) are characterised too.
        out[x, y, z, _C_CONC] = dominant_basin_concentration(
            w_aniso, n_dirs, n_pairs, neighbor_idx, fiber_dirs
        )

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
            out[x, y, z, _C_NPOP] = n_pop

            if n_pop == 1:
                dominant_dir = fiber_dirs[dir_indices[0]]

                if stagec_enabled:
                    # Refine the direction first, then hand the refined one to
                    # Stage C (see _DEFAULT_STAGEC_DIR_REFINE).
                    if stagec_dir_refine and enable_direction_refinement:
                        dominant_dir, _, _ = refine_fiber_direction_cone(
                            bvals, bvecs, sig_norm, dominant_dir,
                            f_fib, f_res, f_nonrf, 0.0,
                            D_res_c, D_nonrf_c, 0.0, False,
                            cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
                        )
                    # ── STAGE C: joint (VARPRO) tensor+fraction re-solve ──
                    # Replaces the raw over-complete-block fractions AND the
                    # decoupled Stage B tensor with a residual-minimising joint
                    # fit on [fiber_col(AD,RD,dir) | iso_grid]. Fixes the
                    # fiber_fraction inflation + restricted collapse + AD
                    # under-estimation (see stagec_varpro_single_fiber).
                    # Uses the RAW normalised signal (data_raw), not data_corr:
                    # the Rician noise-floor subtraction biases the tensor low.
                    sig_raw = data_raw[x, y, z]
                    s0_raw = 0.0
                    cnt_raw = 0
                    for ii in range(len(bvals)):
                        if bvals[ii] < b0_thr:
                            s0_raw += sig_raw[ii]
                            cnt_raw += 1
                    if cnt_raw > 0:
                        s0_raw /= cnt_raw
                    if s0_raw < 1e-6:
                        s0_raw = 1e-6
                    sig_norm_raw = (sig_raw / s0_raw).astype(np.float64)
                    n_iso_sc = stagec_iso_grid.shape[0]
                    w_sc = np.zeros(1 + n_iso_sc)
                    AD_est, RD_est = stagec_varpro_single_fiber(
                        sig_norm_raw, bvals, bvecs, dominant_dir,
                        iso_forward, iso_gram, stagec_ad_grid, stagec_rd_grid,
                        stagec_aniso_ratio, w_sc
                    )
                    tot_sc = 0.0
                    for i in range(1 + n_iso_sc):
                        tot_sc += w_sc[i]
                    if tot_sc > 1e-10:
                        res_sc = 0.0
                        nonrf_sc = 0.0
                        wd_sc = 0.0
                        for i in range(n_iso_sc):
                            wi = w_sc[1 + i]
                            adc = stagec_iso_grid[i]
                            if adc <= THRESH_RES:
                                res_sc += wi
                            else:
                                nonrf_sc += wi
                            wd_sc += wi * adc
                        f_fib = w_sc[0] / tot_sc
                        f_res = res_sc / tot_sc
                        f_nonrf = nonrf_sc / tot_sc
                        sum_iso_sc = res_sc + nonrf_sc
                        out[x, y, z, _C_FF] = f_fib
                        out[x, y, z, _C_RF] = f_res
                        out[x, y, z, _C_NRF] = f_nonrf
                        out[x, y, z, _C_ADC_ISO] = wd_sc / sum_iso_sc if sum_iso_sc > 1e-10 else 0.0
                elif enable_direction_refinement:
                    # the refined direction is what AD/RD were fitted at, so it
                    # is also what gets stored (it used to be discarded, and the
                    # coarse grid node written to dir1 instead -- leaving the
                    # stored direction inconsistent with the stored tensor).
                    dominant_dir, AD_est, RD_est = refine_fiber_direction_cone(
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

                out[x, y, z, _C_AD1] = AD_est
                out[x, y, z, _C_RD1] = RD_est
                out[x, y, z, _C_FA1] = FA
                out[x, y, z, _C_DIR1] = dominant_dir[0]
                out[x, y, z, _C_DIR1 + 1] = dominant_dir[1]
                out[x, y, z, _C_DIR1 + 2] = dominant_dir[2]

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

                # v1.7.0: crossing FF re-solved on the detected support (light ridge
                # kappa * (sigma/S0)^2) BEFORE the LM, instead of holding Stage A's
                # heavily penalised FF fixed -- see core.solvers.crossing_ff_resolve.
                # The population shares keep Stage A's split; Stage D re-solves the
                # final fractions with the tensors fitted here.
                if crossing_ff_kappa > 0.0 and f_fib > 1e-10:
                    iso_sig_r = np.empty(len(bvals))
                    ff_r = crossing_ff_resolve(sig_norm, bvals, bvecs, fiber_dirs,
                                               directions, n_pop, diff_pairs, iso_grid,
                                               crossing_ff_kappa * (sigma_map[x, y, z] / s0) ** 2,
                                               iso_sig_r)
                    if not np.isnan(ff_r) and ff_r > 1e-6:
                        for k in range(n_pop):
                            fractions[k] = fractions[k] / f_fib * ff_r
                        for i in range(len(bvals)):
                            iso_signal[i] = iso_sig_r[i]
                        out[x, y, z, _C_FF] = ff_r

                AD_out, RD_out = estimate_AD_RD_mrds(
                    bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                    init_n_iter=_MRDS_INIT_N_ITER, lm_max_iter=_MRDS_LM_MAX_ITER
                )
                # v1.9.0: the crossing directions leave the grid nodes (see
                # core.solvers.refine_crossing_directions_cone).
                if crossing_dir_refine and enable_direction_refinement:
                    directions, AD_out, RD_out = refine_crossing_directions_cone(
                        bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                        AD_out, RD_out, min_separation_cos,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        _MRDS_CONE_LM_ITER
                    )

                # Dominant population (index 0, highest Stage A weight) ->
                # legacy channels 5/6/7/9/10 + new DIR1, for backward
                # compatibility with single-fiber-era downstream code.
                FA0 = compute_fiber_fa(AD_out[0], RD_out[0])
                out[x, y, z, _C_AD1] = AD_out[0]
                out[x, y, z, _C_RD1] = RD_out[0]
                out[x, y, z, _C_FA1] = FA0
                out[x, y, z, _C_DIR1] = directions[0, 0]
                out[x, y, z, _C_DIR1 + 1] = directions[0, 1]
                out[x, y, z, _C_DIR1 + 2] = directions[0, 2]

                # Population 2
                FA1 = compute_fiber_fa(AD_out[1], RD_out[1])
                out[x, y, z, _C_FF2] = fractions[1]
                out[x, y, z, _C_AD2] = AD_out[1]
                out[x, y, z, _C_RD2] = RD_out[1]
                out[x, y, z, _C_FA2] = FA1
                out[x, y, z, _C_DIR2] = directions[1, 0]
                out[x, y, z, _C_DIR2 + 1] = directions[1, 1]
                out[x, y, z, _C_DIR2 + 2] = directions[1, 2]
                if two_vs_one:
                    # v1.9.0: the single-fiber alternative of this crossing candidate, from the
                    # SAME Stage A (a second kernel pass would redo Stage A, ~40% of its cost):
                    # Stage A's state (locals: out[_C_FF] now holds the crossing's re-solved FF),
                    # then a copy of the n_pop == 1 branch. Verified: with the test off the fit
                    # is bit-identical to the two-pass version (same machine); with it on the
                    # decisions are identical and 6/4992 crossing voxels move by 1 float32 ulp.
                    # See _two_vs_one_pass.
                    alt_out[x, y, z, _C_FF] = f_fib
                    alt_out[x, y, z, _C_RF] = f_res
                    alt_out[x, y, z, _C_NRF] = f_nonrf
                    alt_out[x, y, z, _C_ADC_ISO] = mean_iso_adc
                    alt_out[x, y, z, _C_CONC] = out[x, y, z, _C_CONC]
                    alt_out[x, y, z, _C_NNLS_IT] = out[x, y, z, _C_NNLS_IT]
                    alt_out[x, y, z, _C_NPOP] = 1
                    _single_fiber_2iso(
                        alt_out, x, y, z, fiber_dirs[dir_indices[0]], sig_norm, bvals, bvecs, b0_thr,
                        f_fib, f_res, f_nonrf, D_res_c, D_nonrf_c,
                        stagec_enabled, stagec_dir_refine, enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        data_raw, stagec_iso_grid, iso_forward, iso_gram,
                        stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio)


@njit(parallel=True, cache=True, fastmath=FASTMATH)
def _fit_voxels_3iso_v3(data, coords, AtA_reg, At, bvals, bvecs,
                        fiber_dirs, diff_pairs, n_dirs, iso_grid, b0_thr,
                        fiber_threshold, min_weight_fraction, min_separation_cos,
                        min_peak_ratio, min_dominant_concentration,
                        enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        neighbor_idx, max_fiber_populations, out,
                        lambda_aniso_scalar, conc_mod_enabled,
                        conc_mod_c_lo, conc_mod_c_hi, conc_mod_gain,
                        stagec_enabled, iso_forward, iso_gram,
                        stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio,
                        data_raw, stagec_iso_grid, stagec_dir_refine,
                        crossing_ff_kappa, sigma_map, crossing_dir_refine,
                        two_vs_one, alt_out):
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

        # The solver's iteration count is NOT discarded: written out below as
        # `nnls_iterations`. Hitting _NNLS_MAX_ITER means Stage A did not
        # converge in this voxel, which until v1.3.2 was undetectable — every
        # call site dropped the counter. Written here, before any early exit, so
        # the diagnostic exists even for voxels that leave the loop early.
        w, _nnls_it = nnls_coordinate_descent(AtA_reg, Aty, 0.0)
        out[x, y, z, _C_NNLS_IT] = _nnls_it

        # Plan A: per-voxel concentration-modulated lambda_aniso (on by default;
        # skipped when conc_mod_enabled is False).
        if conc_mod_enabled:
            w = _apply_conc_modulation(
                w, AtA_reg, Aty, n_aniso_cols, n_dirs, n_pairs,
                neighbor_idx, fiber_dirs, lambda_aniso_scalar,
                conc_mod_c_lo, conc_mod_c_hi, conc_mod_gain
            )

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

        out[x, y, z, _C_FF] = f_fib
        out[x, y, z, _C_RF] = f_res
        out[x, y, z, _C_HF] = f_hin
        out[x, y, z, _C_WF] = f_wat
        out[x, y, z, _C_NRF] = f_hin + f_wat
        out[x, y, z, _C_ADC_ISO] = mean_iso_adc

        # Per-voxel angular concentration (diagnostic + Plan A modulation lever);
        # see the 2-ISO kernel note. Computed for every fitted voxel.
        out[x, y, z, _C_CONC] = dominant_basin_concentration(
            w_aniso, n_dirs, n_pairs, neighbor_idx, fiber_dirs
        )

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
            out[x, y, z, _C_NPOP] = n_pop

            if n_pop == 1:
                dominant_dir = fiber_dirs[dir_indices[0]]

                if stagec_enabled:
                    # Refine the direction first, then hand the refined one to
                    # Stage C (see _DEFAULT_STAGEC_DIR_REFINE).
                    if stagec_dir_refine and enable_direction_refinement:
                        dominant_dir, _, _ = refine_fiber_direction_cone(
                            bvals, bvecs, sig_norm, dominant_dir,
                            f_fib, f_res, f_hin, f_wat,
                            D_res_c, D_hin_c, D_wat_c, True,
                            cone1_half_angle, n1_cone, cone2_half_angle, n2_cone
                        )
                    # ── STAGE C: joint (VARPRO) tensor+fraction re-solve (3-ISO
                    # binning RES/HIN/WAT). Uses the RAW normalised signal
                    # (data_raw), not data_corr. See stagec_varpro_single_fiber. ──
                    sig_raw = data_raw[x, y, z]
                    s0_raw = 0.0
                    cnt_raw = 0
                    for ii in range(len(bvals)):
                        if bvals[ii] < b0_thr:
                            s0_raw += sig_raw[ii]
                            cnt_raw += 1
                    if cnt_raw > 0:
                        s0_raw /= cnt_raw
                    if s0_raw < 1e-6:
                        s0_raw = 1e-6
                    sig_norm_raw = (sig_raw / s0_raw).astype(np.float64)
                    n_iso_sc = stagec_iso_grid.shape[0]
                    w_sc = np.zeros(1 + n_iso_sc)
                    AD_est, RD_est = stagec_varpro_single_fiber(
                        sig_norm_raw, bvals, bvecs, dominant_dir,
                        iso_forward, iso_gram, stagec_ad_grid, stagec_rd_grid,
                        stagec_aniso_ratio, w_sc
                    )
                    tot_sc = 0.0
                    for i in range(1 + n_iso_sc):
                        tot_sc += w_sc[i]
                    if tot_sc > 1e-10:
                        res_sc = 0.0
                        hin_sc = 0.0
                        wat_sc = 0.0
                        wd_sc = 0.0
                        for i in range(n_iso_sc):
                            wi = w_sc[1 + i]
                            adc = stagec_iso_grid[i]
                            if adc <= THRESH_RES:
                                res_sc += wi
                            elif adc <= THRESH_WAT:
                                hin_sc += wi
                            else:
                                wat_sc += wi
                            wd_sc += wi * adc
                        f_fib = w_sc[0] / tot_sc
                        f_res = res_sc / tot_sc
                        f_hin = hin_sc / tot_sc
                        f_wat = wat_sc / tot_sc
                        sum_iso_sc = res_sc + hin_sc + wat_sc
                        out[x, y, z, _C_FF] = f_fib
                        out[x, y, z, _C_RF] = f_res
                        out[x, y, z, _C_HF] = f_hin
                        out[x, y, z, _C_WF] = f_wat
                        out[x, y, z, _C_NRF] = f_hin + f_wat
                        out[x, y, z, _C_ADC_ISO] = wd_sc / sum_iso_sc if sum_iso_sc > 1e-10 else 0.0
                elif enable_direction_refinement:
                    # the refined direction is what AD/RD were fitted at, so it
                    # is also what gets stored (it used to be discarded, and the
                    # coarse grid node written to dir1 instead -- leaving the
                    # stored direction inconsistent with the stored tensor).
                    dominant_dir, AD_est, RD_est = refine_fiber_direction_cone(
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

                out[x, y, z, _C_AD1] = AD_est
                out[x, y, z, _C_RD1] = RD_est
                out[x, y, z, _C_FA1] = FA
                out[x, y, z, _C_DIR1] = dominant_dir[0]
                out[x, y, z, _C_DIR1 + 1] = dominant_dir[1]
                out[x, y, z, _C_DIR1 + 2] = dominant_dir[2]

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

                # v1.7.0: crossing FF re-solved on the detected support (light ridge
                # kappa * (sigma/S0)^2) BEFORE the LM, instead of holding Stage A's
                # heavily penalised FF fixed -- see core.solvers.crossing_ff_resolve.
                # The population shares keep Stage A's split; Stage D re-solves the
                # final fractions with the tensors fitted here.
                if crossing_ff_kappa > 0.0 and f_fib > 1e-10:
                    iso_sig_r = np.empty(len(bvals))
                    ff_r = crossing_ff_resolve(sig_norm, bvals, bvecs, fiber_dirs,
                                               directions, n_pop, diff_pairs, iso_grid,
                                               crossing_ff_kappa * (sigma_map[x, y, z] / s0) ** 2,
                                               iso_sig_r)
                    if not np.isnan(ff_r) and ff_r > 1e-6:
                        for k in range(n_pop):
                            fractions[k] = fractions[k] / f_fib * ff_r
                        for i in range(len(bvals)):
                            iso_signal[i] = iso_sig_r[i]
                        out[x, y, z, _C_FF] = ff_r

                AD_out, RD_out = estimate_AD_RD_mrds(
                    bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                    init_n_iter=_MRDS_INIT_N_ITER, lm_max_iter=_MRDS_LM_MAX_ITER
                )
                # v1.9.0: the crossing directions leave the grid nodes (see
                # core.solvers.refine_crossing_directions_cone).
                if crossing_dir_refine and enable_direction_refinement:
                    directions, AD_out, RD_out = refine_crossing_directions_cone(
                        bvals, bvecs, sig_norm, directions, fractions, iso_signal,
                        AD_out, RD_out, min_separation_cos,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        _MRDS_CONE_LM_ITER
                    )

                FA0 = compute_fiber_fa(AD_out[0], RD_out[0])
                out[x, y, z, _C_AD1] = AD_out[0]
                out[x, y, z, _C_RD1] = RD_out[0]
                out[x, y, z, _C_FA1] = FA0
                out[x, y, z, _C_DIR1] = directions[0, 0]
                out[x, y, z, _C_DIR1 + 1] = directions[0, 1]
                out[x, y, z, _C_DIR1 + 2] = directions[0, 2]

                FA1 = compute_fiber_fa(AD_out[1], RD_out[1])
                out[x, y, z, _C_FF2] = fractions[1]
                out[x, y, z, _C_AD2] = AD_out[1]
                out[x, y, z, _C_RD2] = RD_out[1]
                out[x, y, z, _C_FA2] = FA1
                out[x, y, z, _C_DIR2] = directions[1, 0]
                out[x, y, z, _C_DIR2 + 1] = directions[1, 1]
                out[x, y, z, _C_DIR2 + 2] = directions[1, 2]
                if two_vs_one:
                    # v1.9.0: the single-fiber alternative of this crossing candidate, from the
                    # SAME Stage A (a second kernel pass would redo Stage A, ~40% of its cost):
                    # Stage A's state (locals: out[_C_FF] now holds the crossing's re-solved FF),
                    # then a copy of the n_pop == 1 branch. Verified: with the test off the fit
                    # is bit-identical to the two-pass version (same machine); with it on the
                    # decisions are identical and 6/4992 crossing voxels move by 1 float32 ulp.
                    # See _two_vs_one_pass.
                    alt_out[x, y, z, _C_FF] = f_fib
                    alt_out[x, y, z, _C_RF] = f_res
                    alt_out[x, y, z, _C_HF] = f_hin
                    alt_out[x, y, z, _C_WF] = f_wat
                    alt_out[x, y, z, _C_NRF] = f_hin + f_wat
                    alt_out[x, y, z, _C_ADC_ISO] = mean_iso_adc
                    alt_out[x, y, z, _C_CONC] = out[x, y, z, _C_CONC]
                    alt_out[x, y, z, _C_NNLS_IT] = out[x, y, z, _C_NNLS_IT]
                    alt_out[x, y, z, _C_NPOP] = 1
                    _single_fiber_3iso(
                        alt_out, x, y, z, fiber_dirs[dir_indices[0]], sig_norm, bvals, bvecs, b0_thr,
                        f_fib, f_res, f_hin, f_wat, D_res_c, D_hin_c, D_wat_c,
                        stagec_enabled, stagec_dir_refine, enable_direction_refinement,
                        cone1_half_angle, n1_cone, cone2_half_angle, n2_cone,
                        data_raw, stagec_iso_grid, iso_forward, iso_gram,
                        stagec_ad_grid, stagec_rd_grid, stagec_aniso_ratio)


@njit(parallel=True, cache=True, fastmath=FASTMATH)
def _iso_resolve_pass(data_corr, coords, bvals, bvecs, b0_thr, iso_d, use_3iso, out,
                      crossing_resolve, fiber_threshold):
    """
    STAGE D pass — final constrained compartment-fraction re-solve for EVERY
    fitted voxel, on the RICIAN-CORRECTED signal. Reads the detected structure
    (n_pop + per-population directions/tensors) back from the output array,
    re-solves [fibers | fixed iso centroids] via `iso_fraction_resolve`, and
    OVERWRITES the fraction channels (0,1,2,3,4 and the pop-2 fraction 15).
    The fiber TENSORS/directions are kept as estimated (Stage C on raw / MRDS);
    only the fractions are re-estimated here (on corrected). See
    `core.solvers.iso_fraction_resolve` for the why (raw-tensor / corrected-
    fraction split, fixed-3 vs the over-complete spectrum).

    FIBER THRESHOLD ON THE FINAL STATE (v1.8.0). `fiber_threshold` decides in
    Stage A whether a voxel gets a fiber, on Stage A's FF; the FF reported is
    this pass's. Where the re-solved FF is <= fiber_threshold the fiber is
    REMOVED (n_pop -> NaN, fiber channels -> NaN) and the fractions are re-solved
    on the iso centroids alone, exactly as for a voxel Stage A gave no fiber, so
    "a tensor is reported only where FF > fiber_threshold" holds for the output.
    Measured on 1.7.1 (P3 phantom, SNR 26): 38% of the (false) fibers in
    isotropic tissue and 17% of true weak fibers (FF 0.2) had final FF < 0.15.
    Not applied on the kappa-0 crossing path (crossing_resolve False), where the
    reported crossing FF is Stage A's own.
    """
    n_voxels = coords.shape[0]
    n_iso = iso_d.shape[0]
    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data_corr[x, y, z]
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

        # Reconstruct the fiber set from the output (NaN n_pop => no fiber).
        fdirs = np.zeros((2, 3))
        fad = np.zeros(2)
        frd = np.zeros(2)
        n_fib = 0
        npop = out[x, y, z, _C_NPOP]
        if not np.isnan(npop):
            npi = int(npop)
            if npi >= 1 and not np.isnan(out[x, y, z, _C_AD1]):
                fdirs[0, 0] = out[x, y, z, _C_DIR1]
                fdirs[0, 1] = out[x, y, z, _C_DIR1 + 1]
                fdirs[0, 2] = out[x, y, z, _C_DIR1 + 2]
                fad[0] = out[x, y, z, _C_AD1]
                frd[0] = out[x, y, z, _C_RD1]
                n_fib = 1
                if npi >= 2 and not np.isnan(out[x, y, z, _C_AD2]):
                    fdirs[1, 0] = out[x, y, z, _C_DIR2]
                    fdirs[1, 1] = out[x, y, z, _C_DIR2 + 1]
                    fdirs[1, 2] = out[x, y, z, _C_DIR2 + 2]
                    fad[1] = out[x, y, z, _C_AD2]
                    frd[1] = out[x, y, z, _C_RD2]
                    n_fib = 2

        w_out = np.zeros(n_fib + n_iso)
        tot = 0.0
        f_fib = 0.0
        for _pass in range(2):
            w_out = np.zeros(n_fib + n_iso)
            iso_fraction_resolve(sig_norm, bvals, bvecs, fdirs, fad, frd, n_fib, iso_d, w_out)
            tot = 0.0
            for a in range(n_fib + n_iso):
                tot += w_out[a]
            if tot < 1e-10:
                break
            f_fib = 0.0
            for k in range(n_fib):
                f_fib += w_out[k]
            f_fib /= tot
            if n_fib == 0 or f_fib > fiber_threshold or (n_fib >= 2 and not crossing_resolve):
                break
            # final FF at or below the threshold: no fiber (see docstring)
            out[x, y, z, _C_NPOP] = np.nan
            for c in (_C_AD1, _C_RD1, _C_FA1, _C_DIR1, _C_DIR1 + 1, _C_DIR1 + 2,
                      _C_FF2, _C_AD2, _C_RD2, _C_FA2, _C_DIR2, _C_DIR2 + 1, _C_DIR2 + 2):
                out[x, y, z, c] = np.nan
            n_fib = 0
        if tot < 1e-10:
            continue

        res = 0.0
        hin = 0.0
        wat = 0.0
        for j in range(n_iso):
            wj = w_out[n_fib + j] / tot
            dj = iso_d[j]
            if dj <= THRESH_RES:
                res += wj
            elif dj < THRESH_WAT:   # strict: the WF centroid sits AT THRESH_WAT
                hin += wj
            else:
                wat += wj

        if n_fib >= 2 and crossing_resolve:
            # crossing_ff_kappa > 0: the crossing tensors were fitted with a
            # re-solved FF, so Stage D re-solves the crossing fractions too (total
            # and pop-2), as for single fibers.
            out[x, y, z, _C_FF] = f_fib
            w1n = w_out[0] / tot
            w2n = w_out[1] / tot
            if w1n > 1e-6 and w2n > 1e-6:
                out[x, y, z, _C_FF2] = w2n
            else:
                # The re-solve gave one population NO weight (0.7% of real
                # crossings, notebook 11). Reporting n_pop=2 with a zero-weight
                # population would put a tensor and a direction the data do not
                # support into the pop-2 maps: the voxel becomes a single fiber.
                # If it is population 1 that vanished, population 2 moves into its
                # slot. The surviving tensor is the one fitted jointly with the
                # other population (no Stage C re-fit).
                if w1n <= 1e-6 and w2n > 1e-6:
                    out[x, y, z, _C_AD1] = out[x, y, z, _C_AD2]
                    out[x, y, z, _C_RD1] = out[x, y, z, _C_RD2]
                    out[x, y, z, _C_FA1] = out[x, y, z, _C_FA2]
                    out[x, y, z, _C_DIR1] = out[x, y, z, _C_DIR2]
                    out[x, y, z, _C_DIR1 + 1] = out[x, y, z, _C_DIR2 + 1]
                    out[x, y, z, _C_DIR1 + 2] = out[x, y, z, _C_DIR2 + 2]
                out[x, y, z, _C_NPOP] = 1.0
                for c in (_C_FF2, _C_AD2, _C_RD2, _C_FA2, _C_DIR2, _C_DIR2 + 1, _C_DIR2 + 2):
                    out[x, y, z, c] = np.nan
        elif n_fib >= 2:
            # crossing_ff_kappa = 0: the reported crossing FF is Stage A's (biased
            # low: 0.07-0.21 on synthetic P3 crossings, the cause of crossing RD on
            # the floor, see core/solvers.py MRDS section). Stage D gives only the
            # iso split, rescaled to (1 - FF); FF and the pop-2 fraction untouched.
            ff_keep = out[x, y, z, _C_FF]
            if np.isnan(ff_keep):
                ff_keep = f_fib
            iso_sum = res + hin + wat
            if iso_sum > 1e-10:
                sc = (1.0 - ff_keep) / iso_sum
                res = res * sc
                hin = hin * sc
                wat = wat * sc
        else:
            out[x, y, z, _C_FF] = f_fib

        out[x, y, z, _C_RF] = res
        out[x, y, z, _C_NRF] = hin + wat
        if use_3iso:
            out[x, y, z, _C_HF] = hin
            out[x, y, z, _C_WF] = wat


@njit(cache=True, fastmath=FASTMATH)
def _fiber_detection_pass(data_corr, coords, bvals, bvecs, b0_thr, iso_d, use_3iso,
                          sigma_map, threshold, out, stat_out):
    """
    Detection test for the fiber populations Stage A reported (v1.6.1).

    For every voxel with a fiber, on the Stage D model (fixed iso centroids,
    fitted tensors and directions, Rician-corrected signal):

        stat = (RSS_iso_only - RSS_fibers+iso) / sigma_voxel^2,
        sigma_voxel = sigma_map[x, y, z] / S0_voxel   (thermal sigma map since v1.7.1)

    i.e. how much the fibers reduce the residual, in units of the noise
    variance. Written to `stat_out` for every such voxel. If `threshold` > 0
    and stat < threshold, the fiber is REJECTED: n_pop -> NaN, fiber channels
    -> NaN, FF -> 0 and the iso fractions from the iso-only fit.

    WHY. Measured on synthetic ISOTROPIC tissue (P3 protocol, frozen P3
    lambdas, gate off): at SNR 26, 38-62% of GM-like / hindered / tumour-like
    voxels got a fiber, almost all as a CROSSING with FF 0.16-0.18 (Stage A
    noise spread over two peaks, just above fiber_threshold); 88-90% at SNR 15;
    ~0 at SNR 40. The concentration gate cannot separate them from true
    crossings (concentration anti-correlates with resolvability). This
    statistic can: its null distribution is SNR-invariant (median 4-6, p95
    11-14 at SNR 15, 26 and 40) because it is noise-normalised, while true
    fibers score p05 74 (single fiber FF 0.25) and 17.5 (90-degree crossing,
    FF 0.30) at SNR 26. It is the F-test MRDS uses for model selection, which
    this toolbox had left out. The statistic is always computed and written
    (fiber_detection_stat.nii.gz); the threshold is OFF by default.

    DECIDED OFF for v1.7.0 (2026-10-01). Battery 05 (synthetic, v1.6.2): threshold
    15 adoptable at SNR 30 and 15 (false fibers 0.067 -> 0.044 / 0.22 -> 0.20, no
    true fiber lost, FF MAE 0.053 -> 0.047). Notebook 09 (healthy-tissue mix of a
    real P3 subject, v1.6.4): threshold 15 would remove 15% of fiber voxels (2.9%
    of single fibers, 28.5% of crossings, mostly FF 0.15-0.3), but failed its
    pre-declared control (73% < 80% of the voxels the old gate rejected kept;
    weak premise: that control rested on a residual inflated by the pre-1.5.2
    reconstruction). And the crossing FF re-solve of 1.7.0 changes the crossing
    tensors, hence this statistic: the threshold is to be decided on 1.7.0 output.
    """
    n_voxels = coords.shape[0]
    n_iso = iso_d.shape[0]
    N = len(bvals)
    A_iso = np.empty((N, n_iso))
    for j in range(n_iso):
        for i in range(N):
            A_iso[i, j] = np.exp(-bvals[i] * iso_d[j])
    AtA_iso = A_iso.T @ A_iso
    # SERIAL since v1.6.3. On the workstation (Linux) the parallel version stopped
    # the fit with ZeroDivisionError -> SystemError ("returned a result with an
    # error set"), while the same pass in pure Python on the same data found no
    # zero divisor at all (2026-10-01): an exception state from the parallel
    # region, not a bug in the arithmetic. Serial it costs ~0.15 s per 6000
    # fiber voxels (~4 s on a whole brain), and any real exception surfaces with
    # its own message.
    for idx in range(n_voxels):
        x, y, z = coords[idx]
        npop = out[x, y, z, _C_NPOP]
        if np.isnan(npop) or npop < 1 or np.isnan(out[x, y, z, _C_AD1]):
            continue
        sig = data_corr[x, y, z]
        s0 = 0.0
        cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if not (s0 >= 1e-6) or not np.isfinite(s0):
            continue
        yv = np.empty(N)
        for i in range(N):
            yv[i] = sig[i] / s0
        n_fib = 2 if (npop >= 2 and not np.isnan(out[x, y, z, _C_AD2])) else 1
        A = np.empty((N, n_fib + n_iso))
        for k in range(n_fib):
            c0 = _C_DIR1 if k == 0 else _C_DIR2
            ad = out[x, y, z, _C_AD1] if k == 0 else out[x, y, z, _C_AD2]
            rd = out[x, y, z, _C_RD1] if k == 0 else out[x, y, z, _C_RD2]
            d0 = out[x, y, z, c0]
            d1 = out[x, y, z, c0 + 1]
            d2 = out[x, y, z, c0 + 2]
            for i in range(N):
                c = bvecs[i, 0] * d0 + bvecs[i, 1] * d1 + bvecs[i, 2] * d2
                A[i, k] = np.exp(-bvals[i] * (rd + (ad - rd) * c * c))
        for j in range(n_iso):
            for i in range(N):
                A[i, n_fib + j] = A_iso[i, j]
        w_f, _ = nnls_coordinate_descent(A.T @ A, A.T @ yv, 0.0)
        w_i, _ = nnls_coordinate_descent(AtA_iso, A_iso.T @ yv, 0.0)
        r_f = yv - A @ w_f
        r_i = yv - A_iso @ w_i
        sig2 = (sigma_map[x, y, z] / s0) ** 2
        if not (sig2 > 0.0):      # S0 infinito o sigma nullo: statistica non definita
            continue
        stat = (np.sum(r_i * r_i) - np.sum(r_f * r_f)) / sig2
        stat_out[x, y, z] = stat
        if threshold > 0.0 and stat < threshold:
            tot = 0.0
            for j in range(n_iso):
                tot += w_i[j]
            if tot < 1e-10:
                continue
            res = 0.0
            hin = 0.0
            wat = 0.0
            for j in range(n_iso):
                dj = iso_d[j]
                if dj <= THRESH_RES:
                    res += w_i[j] / tot
                elif dj < THRESH_WAT:
                    hin += w_i[j] / tot
                else:
                    wat += w_i[j] / tot
            out[x, y, z, _C_FF] = 0.0
            out[x, y, z, _C_RF] = res
            out[x, y, z, _C_NRF] = hin + wat
            if use_3iso:
                out[x, y, z, _C_HF] = hin
                out[x, y, z, _C_WF] = wat
            out[x, y, z, _C_NPOP] = np.nan
            for c in (_C_AD1, _C_RD1, _C_FA1, _C_DIR1, _C_DIR1 + 1, _C_DIR1 + 2,
                      _C_FF2, _C_AD2, _C_RD2, _C_FA2, _C_DIR2, _C_DIR2 + 1, _C_DIR2 + 2):
                out[x, y, z, c] = np.nan


@njit(cache=True, fastmath=FASTMATH)
def _stage_d_rss(yv, bvals, bvecs, A_iso, arr, x, y, z, n_fib):
    """Residual of the Stage D model [n_fib fibers of `arr` at (x,y,z) | iso
    centroids] fitted by NNLS to the normalised signal `yv`."""
    N = len(bvals)
    n_iso = A_iso.shape[1]
    A = np.empty((N, n_fib + n_iso))
    for k in range(n_fib):
        c0 = _C_DIR1 if k == 0 else _C_DIR2
        ad = arr[x, y, z, _C_AD1] if k == 0 else arr[x, y, z, _C_AD2]
        rd = arr[x, y, z, _C_RD1] if k == 0 else arr[x, y, z, _C_RD2]
        d0 = arr[x, y, z, c0]
        d1 = arr[x, y, z, c0 + 1]
        d2 = arr[x, y, z, c0 + 2]
        for i in range(N):
            c = bvecs[i, 0] * d0 + bvecs[i, 1] * d1 + bvecs[i, 2] * d2
            A[i, k] = np.exp(-bvals[i] * (rd + (ad - rd) * c * c))
    for j in range(n_iso):
        for i in range(N):
            A[i, n_fib + j] = A_iso[i, j]
    w, _ = nnls_coordinate_descent(A.T @ A, A.T @ yv, 0.0)
    r = yv - A @ w
    return np.sum(r * r)


@njit(cache=True, fastmath=FASTMATH)
def _two_vs_one_pass(data_corr, coords, bvals, bvecs, b0_thr, iso_d, sigma_map,
                     threshold, out, alt, stat_out):
    """
    Two-vs-one population test (v1.9.0, see _DEFAULT_TWO_VS_ONE_THRESHOLD).

    `out` holds the crossing fit of every candidate voxel, `alt` the single-fiber
    fit of the same voxels from the same Stage A (both written by the kernel). On
    the Stage D model,

        stat21 = (RSS_1pop - RSS_2pop) / sigma_voxel^2,

    written to `stat_out`; where stat21 < threshold the voxel's channels are
    replaced by the single-fiber fit. Run BEFORE Stage D, so that neither model
    has been through the fiber threshold yet. Serial, like the detection pass.
    """
    n_voxels = coords.shape[0]
    n_iso = iso_d.shape[0]
    N = len(bvals)
    A_iso = np.empty((N, n_iso))
    for j in range(n_iso):
        for i in range(N):
            A_iso[i, j] = np.exp(-bvals[i] * iso_d[j])
    n_ch = out.shape[3]
    for idx in range(n_voxels):
        x, y, z = coords[idx]
        npop = out[x, y, z, _C_NPOP]
        if np.isnan(npop) or npop < 2 or np.isnan(out[x, y, z, _C_AD1]) \
                or np.isnan(out[x, y, z, _C_AD2]):
            continue
        npop1 = alt[x, y, z, _C_NPOP]
        if np.isnan(npop1) or npop1 < 1 or np.isnan(alt[x, y, z, _C_AD1]):
            continue          # no single-fiber alternative: the crossing stays
        sig = data_corr[x, y, z]
        s0 = 0.0
        cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if not (s0 >= 1e-6) or not np.isfinite(s0):
            continue
        sig2 = (sigma_map[x, y, z] / s0) ** 2
        if not (sig2 > 0.0):
            continue
        yv = np.empty(N)
        for i in range(N):
            yv[i] = sig[i] / s0
        rss2 = _stage_d_rss(yv, bvals, bvecs, A_iso, out, x, y, z, 2)
        rss1 = _stage_d_rss(yv, bvals, bvecs, A_iso, alt, x, y, z, 1)
        stat = (rss1 - rss2) / sig2
        stat_out[x, y, z] = stat
        if stat < threshold:
            for c in range(n_ch):
                out[x, y, z, c] = alt[x, y, z, c]


def _package_provenance():
    """Where this code came from: install path, and the git state if the
    package is being run out of a checkout. Everything is best-effort — an
    installed wheel has no git, and that is not an error, it just means the
    version string is all the provenance there is."""
    import os
    from . import __version__ as _v          # deferred: __init__ imports us
    info = {'toolbox_version': _v,
            'package_path': os.path.dirname(os.path.abspath(__file__))}
    try:
        import subprocess
        repo = os.path.dirname(info['package_path'])
        def _g(*a):
            r = subprocess.run(['git', '-C', repo, *a], capture_output=True,
                               text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ''
        info['git_commit'] = _g('rev-parse', '--short', 'HEAD')
        info['git_branch'] = _g('rev-parse', '--abbrev-ref', 'HEAD')
        info['git_dirty'] = bool(_g('status', '--porcelain'))
    except Exception:
        info['git_commit'] = info['git_branch'] = ''
        info['git_dirty'] = False
    return info


def _solver_diagnostics(results, mask):
    """Did the solver converge, and how often did it stop on a bound?

    Both questions used to be unanswerable from the outputs. Non-convergence
    was invisible because every NNLS call site discarded the iteration counter;
    bound activity was invisible because a clamped value is indistinguishable
    from an estimated one. Neither is rare on real data, and a voxel sitting
    exactly on a bound is a voxel whose tensor the data did not determine — so
    averaging over it reports the bound, not the tissue.

    Percentages are over the voxels where the relevant quantity exists: the
    fitted mask for the solver counters, and the voxels where that population
    has a tensor for the bound counters.
    """
    from .core.solvers import (_TENSOR_AD_FLOOR, _TENSOR_AD_CEIL,
                               _TENSOR_RD_FLOOR, _TENSOR_RD_CEIL,
                               _NNLS_MAX_ITER)
    out = {}
    it = results[..., _C_NNLS_IT][mask]
    it = it[np.isfinite(it)]
    if it.size:
        out.update(nnls_max_iter=int(_NNLS_MAX_ITER),
                   nnls_iter_median=float(np.median(it)),
                   nnls_iter_p95=float(np.percentile(it, 95)),
                   nnls_iter_max=int(it.max()),
                   nnls_not_converged_pct=round(
                       100.0 * float(np.mean(it >= _NNLS_MAX_ITER - 1)), 3))

    # A bound is "active" when the value sits on it to within a relative 1e-6:
    # these are exact assignments, not near misses, so the test is tight.
    def _on(ch, bound):
        v = results[..., ch][mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return round(100.0 * float(np.mean(np.abs(v - bound) <= 1e-6 * bound)), 2)

    for nome, ch, bound in (('rd_pop1_at_floor_pct', _C_RD1, _TENSOR_RD_FLOOR),
                            ('rd_pop2_at_floor_pct', _C_RD2, _TENSOR_RD_FLOOR),
                            ('rd_pop1_at_ceil_pct', _C_RD1, _TENSOR_RD_CEIL),
                            ('rd_pop2_at_ceil_pct', _C_RD2, _TENSOR_RD_CEIL),
                            ('ad_pop1_at_floor_pct', _C_AD1, _TENSOR_AD_FLOOR),
                            ('ad_pop2_at_floor_pct', _C_AD2, _TENSOR_AD_FLOOR),
                            ('ad_pop1_at_ceil_pct', _C_AD1, _TENSOR_AD_CEIL),
                            ('ad_pop2_at_ceil_pct', _C_AD2, _TENSOR_AD_CEIL)):
        val = _on(ch, bound)
        if val is not None:
            out[nome] = val

    # The same floor, split by BRANCH. rd_pop1_at_floor_pct mixes the single
    # fibers with the dominant population of the crossings, two estimators with
    # different failure modes (crossing tensors: re-solved FF + MRDS LM; on a
    # healthy P3 subject ~10-12% per population on the floor, like single fibers).
    npop = results[..., _C_NPOP]
    rd1 = results[..., _C_RD1]
    rd2 = results[..., _C_RD2]
    at = lambda v: np.abs(v - _TENSOR_RD_FLOOR) <= 1e-6 * _TENSOR_RD_FLOOR
    mono = mask & (npop == 1) & np.isfinite(rd1)
    cross = mask & (npop >= 2) & np.isfinite(rd1) & np.isfinite(rd2)
    if mono.any():
        out['rd_at_floor_single_fiber_pct'] = round(100.0 * float(np.mean(at(rd1[mono]))), 2)
    if cross.any():
        out['rd_at_floor_crossing_any_pop_pct'] = round(
            100.0 * float(np.mean(at(rd1[cross]) | at(rd2[cross]))), 2)
    out.update(tensor_ad_floor=_TENSOR_AD_FLOOR, tensor_ad_ceil=_TENSOR_AD_CEIL,
               tensor_rd_floor=_TENSOR_RD_FLOOR, tensor_rd_ceil=_TENSOR_RD_CEIL)
    return out


def _population_census(results, mask):
    """Voxel counts for the three states of `n_fiber_populations`, over the
    fitted mask. The three states are not interchangeable (see
    `output_map_names`), so a run report that collapsed them would hide the
    thing worth noticing: on angularly poor protocols a large share of
    single-fiber voxels never resolves a direction at all."""
    npop = results[..., _C_NPOP]
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return {}
    out = {'n_mask': n,
           'npop_nan': int(np.sum(np.isnan(npop) & m)),
           'npop_0': int(np.sum((npop == 0) & m)),
           'npop_1': int(np.sum((npop == 1) & m)),
           'npop_2': int(np.sum((npop == 2) & m))}
    for k in ('npop_nan', 'npop_0', 'npop_1', 'npop_2'):
        out[k + '_pct'] = round(out[k] / n * 100.0, 2)
    return out


def _fill_derived_channels(out):
    """
    Final vectorised pass: fill the channels that are DERIVED from the fitted
    ones, so that no downstream consumer has to re-derive them (and so they
    cannot silently disagree with the fractions they are built from).

    * `fiber_fraction_pop1` = fiber_fraction (total) - fiber_fraction_pop2.
      Stage A computes the pop-1 share explicitly but the kernels never stored
      it, and Stage D then revises the TOTAL for single-fiber voxels while
      leaving the pop-2 fraction alone -- so the only value consistent with the
      final output is this difference, taken here, after every in-place stage.
    * `*_weighted` = the fiber tensor averaged over the populations present,
      weighted by their fractions. For a single-fiber voxel this is just the
      pop-1 tensor; for a crossing it is the one number that summarises both.
      `fiber_fa_weighted` is the FA OF the weighted tensor (an intrinsic
      per-fiber anisotropy), NOT the average of the two FAs.

    Must run AFTER Stage C, Stage D and the RF bias correction. In-place.
    """
    ff_t = out[..., _C_FF].astype(np.float64)
    ff2 = np.nan_to_num(out[..., _C_FF2].astype(np.float64), nan=0.0)
    ad1 = out[..., _C_AD1].astype(np.float64)
    rd1 = out[..., _C_RD1].astype(np.float64)
    ad2 = np.nan_to_num(out[..., _C_AD2].astype(np.float64), nan=0.0)
    rd2 = np.nan_to_num(out[..., _C_RD2].astype(np.float64), nan=0.0)

    ff_t = np.where(np.isfinite(ff_t), ff_t, 0.0)
    ff1 = np.clip(ff_t - ff2, 0.0, None)

    # A fiber tensor exists exactly where population 1 was estimated; the
    # per-population block is NaN where the population is absent.
    fiber = np.isfinite(ad1) & (ff_t > 0)
    out[..., _C_FF1] = np.where(fiber, ff1, np.nan).astype(np.float32)

    denom = np.where(ff_t > 0, ff_t, 1.0)
    adw = (ff1 * np.nan_to_num(ad1, nan=0.0) + ff2 * ad2) / denom
    rdw = (ff1 * np.nan_to_num(rd1, nan=0.0) + ff2 * rd2) / denom
    adw = np.where(fiber, adw, np.nan)
    rdw = np.where(fiber, rdw, np.nan)

    md = (adw + 2.0 * rdw) / 3.0
    num = np.sqrt((adw - md) ** 2 + 2.0 * (rdw - md) ** 2)
    den = np.sqrt(adw ** 2 + 2.0 * rdw ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        faw = np.sqrt(1.5) * num / den
    faw = np.where(fiber & (den > 0), faw, np.nan)

    out[..., _C_ADW] = adw.astype(np.float32)
    out[..., _C_RDW] = rdw.astype(np.float32)
    out[..., _C_FAW] = faw.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DBSI_Adaptive:
    """
    Adaptive DBSI model: Stage A detection, Stage B/C tensors (single fiber:
    cone refinement + Stage C; crossings: re-solved FF + MRDS joint LM), Stage D
    fractions. See the module docstring for the pipeline and its known limits.

    At most TWO fiber populations per voxel (`MAX_FIBER_POPULATIONS`, fixed).
    3-ISO (RF + HF + WF) if b_max >= B_THRESH_3ISO and n_shells >=
    MIN_SHELLS_3ISO, 2-ISO (RF + NRF) otherwise.

    Parameters
    ----------
    n_iso : int or None
        Density of the anchored isotropic ADC grid (NOT the number of
        columns: n_iso=6 gives 11). If None, set by `fit(n_iso_method=...)`;
        the default method 'quadrature' fixes it at 6.
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
        Minimum AD/RD ratio for admissible Stage A pairs. Default: 2.0.
    ad_range, rd_range : tuple (float, float)
        Physical bounds for Stage A's AD / RD grids, mm^2/s.
    iso_range : tuple (float, float)
        ADC range [mm^2/s] of the isotropic basis, widened by the anchored grid to
        at least [0.1e-3, 5.0e-3] (the free-water columns). Default: (0.0, 3.0e-3).
    fiber_threshold : float
        Minimum fibre fraction for a fiber tensor: checked on Stage A's FF
        (candidates) and on the FINAL FF (reported). Default: 0.15.
    min_weight_fraction : float
        Minimum fraction of total Stage A anisotropic weight a direction
        must carry to be reported as a fiber population. Default: 0.05.
    force_n_iso : int or None
        Override automatic isotropic-model selection (2 or 3).
    enable_direction_refinement : bool
        Whether to refine Stage A's dominant direction estimate with a
        two-level "MRDS-lite" cone search for SINGLE-fiber voxels
        (n_pop == 1). Default: True. Crossings use Stage A's grid
        directions as MRDS input.
    target_angular_resolution_deg : float
        Desired final angular precision (degrees) of the refined
        direction (n_pop == 1 path only). Default: 1.0.

    """

    N_CHANNELS = _N_CHANNELS

    def __init__(self, n_iso=None, lambda_aniso=None, lambda_iso=None,
                 n_dirs=None, n_ad=_STAGE_A_DEFAULT_N_AD, n_rd=_STAGE_A_DEFAULT_N_RD,
                 anisotropy_ratio=_STAGE_A_DEFAULT_ANISOTROPY_RATIO,
                 ad_range=(_STAGE_A_AD_MIN, _STAGE_A_AD_MAX),
                 rd_range=(_STAGE_A_RD_MIN, _STAGE_A_RD_MAX),
                 iso_range=(_DEFAULT_ISO_MIN, _DEFAULT_ISO_MAX),
                 fiber_threshold=FIBER_THRESHOLD,
                 min_weight_fraction=0.05, force_n_iso=None,
                 direction_peak_k=_DEFAULT_DIRECTION_PEAK_K,
                 min_separation_deg=_DEFAULT_MIN_SEPARATION_DEG,
                 min_peak_ratio=_DEFAULT_MIN_PEAK_RATIO,
                 min_dominant_concentration=_DEFAULT_MIN_DOMINANT_CONCENTRATION,
                 fiber_detection_threshold=None,
                 enable_direction_refinement=True,
                 target_angular_resolution_deg=1.0,
                 lambda_aniso_conc_mod=_DEFAULT_LAMBDA_ANISO_CONC_MOD,
                 conc_mod_c_lo=_DEFAULT_CONC_MOD_C_LO,
                 conc_mod_c_hi=_DEFAULT_CONC_MOD_C_HI,
                 conc_mod_gain=_DEFAULT_CONC_MOD_GAIN,
                 stagec_refine=_DEFAULT_STAGEC_REFINE,
                 stagec_dir_refine=_DEFAULT_STAGEC_DIR_REFINE,
                 iso_resolve=_DEFAULT_ISO_RESOLVE,
                 lambda_aniso_method='gcv',
                 uncertainty=True,
                 crossing_ff_kappa=_DEFAULT_CROSSING_FF_KAPPA,
                 crossing_dir_refine=_DEFAULT_CROSSING_DIR_REFINE,
                 two_vs_one_threshold=_DEFAULT_TWO_VS_ONE_THRESHOLD):
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
        self.direction_peak_k = direction_peak_k
        self.min_separation_deg = min_separation_deg
        self.min_peak_ratio = min_peak_ratio
        self.min_dominant_concentration = min_dominant_concentration
        if fiber_detection_threshold is not None and not fiber_detection_threshold > 0:
            raise ValueError('fiber_detection_threshold must be None (off) or > 0')
        self.fiber_detection_threshold = fiber_detection_threshold
        self.enable_direction_refinement = enable_direction_refinement
        self.target_angular_resolution_deg = target_angular_resolution_deg
        self.lambda_aniso_conc_mod = lambda_aniso_conc_mod
        self.conc_mod_c_lo = conc_mod_c_lo
        self.conc_mod_c_hi = conc_mod_c_hi
        self.conc_mod_gain = conc_mod_gain
        self.stagec_refine = stagec_refine
        self.stagec_dir_refine = stagec_dir_refine
        self.iso_resolve = iso_resolve
        if lambda_aniso_method not in ('discrepancy', 'gcv', 'lcurve'):
            raise ValueError("lambda_aniso_method must be 'discrepancy', 'gcv', or 'lcurve', "
                             f"got {lambda_aniso_method!r}.")
        self.lambda_aniso_method = lambda_aniso_method
        self.uncertainty = bool(uncertainty)
        if crossing_ff_kappa is None or not crossing_ff_kappa >= 0:
            raise ValueError('crossing_ff_kappa must be >= 0 (0 = crossing FF from Stage A)')
        self.crossing_ff_kappa = float(crossing_ff_kappa)
        self.crossing_dir_refine = bool(crossing_dir_refine)
        if two_vs_one_threshold is not None and not two_vs_one_threshold > 0:
            raise ValueError('two_vs_one_threshold must be None (off) or > 0')
        if two_vs_one_threshold is not None and not iso_resolve:
            raise ValueError('two_vs_one_threshold needs Stage D (iso_resolve=True): '
                             'the test is defined on the Stage D model')
        self.two_vs_one_threshold = two_vs_one_threshold

        self.model_mode_ = None
        self.b_max_ = None
        self.n_shells_ = None
        self.n_aniso_cols_ = None
        self.diff_pairs_ = None
        self.sure_crosscheck_report_ = None
        self.n_iso_source_ = None
        self.rician_clamp_ = None
        self.fiber_detection_stat_ = None
        self.fiber_detection_ = None
        self.two_vs_one_stat_ = None
        self.two_vs_one_ = None
        self.fiber_threshold_final_ = None
        self.uncertainty_ = None
        self.uncertainty_summary_ = None
        self.nonfinite_voxels_excluded_ = None
        self.noise_ = None
        self.lambda_edges_ = {}
        self.hemisphere_spacing_deg_ = None
        self.cone_refinement_schedule_ = None
        self.run_report_ = None
        self.dictionary_ = None
        # Impostato da `from_calibration`: NON e' un attributo di run, quindi non
        # si azzera in fit() -- descrive come e' stato costruito il modello.
        self.protocol_calibration_ = None

    # ------------------------------------------------------------------
    @classmethod
    def from_calibration(cls, calibration, strict=True, **overrides):
        """Modello con gli iperparametri di una calibrazione di PROTOCOLLO.

        `calibration` e' il percorso del JSON scritto da
        `save_protocol_calibration`, oppure il dict di `calibrate_protocol` /
        `load_protocol_calibration`. Impone lambda_aniso, lambda_iso, n_iso, il
        gate e le opzioni da cui dipende il dizionario (n_dirs, griglie di
        Stage A, anisotropy_ratio, fiber_threshold, iso_range): i lambda hanno
        senso solo sul dizionario su cui sono stati calibrati.

        In `fit()` l'impronta dei bvals viene confrontata con quella del
        protocollo: se differisce, `strict=True` rifiuta di fittare. Il run
        report registra nome, sha256 e impronta della calibrazione, e il JSON
        viene copiato in toolbox_report/.

        `overrides` sono altri argomenti del costruttore (p.es. flag di Stage
        C); ridefinire un parametro calibrato solleva ValueError.
        """
        from .calibration.protocol import load_protocol_calibration
        import os
        if isinstance(calibration, (str, os.PathLike)):
            cal = load_protocol_calibration(calibration)
        else:
            cal = dict(calibration)
            if cal.get('kind') != 'pydbsi_protocol_calibration':
                raise ValueError('non e una calibrazione di protocollo')
        h, mo = cal['hyperparameters'], cal['model']
        fissati = dict(lambda_aniso=float(h['lambda_aniso']),
                       lambda_iso=float(h['lambda_iso']), n_iso=int(h['n_iso']),
                       min_dominant_concentration=float(h['concentration_gate']),
                       lambda_aniso_method=str(h['lambda_aniso_method']),
                       n_dirs=int(mo['n_dirs']), n_ad=int(mo['n_ad']), n_rd=int(mo['n_rd']),
                       anisotropy_ratio=float(mo['anisotropy_ratio']),
                       fiber_threshold=float(mo['fiber_threshold']),
                       iso_range=tuple(float(x) for x in mo['iso_range']))
        conflitti = sorted(set(overrides) & set(fissati))
        if conflitti:
            raise ValueError(f'{conflitti} sono fissati dalla calibrazione di protocollo: '
                             f'ridefinirli romperebbe la riproducibilita. Se serve un '
                             f'valore diverso, serve una calibrazione diversa.')
        m = cls(**fissati, **overrides)
        src = cal.get('_source') or {}
        m.protocol_calibration_ = dict(
            name=cal.get('name'), sha256=src.get('sha256'), path=src.get('path'),
            created_utc=cal.get('created_utc'),
            toolbox_version=cal.get('toolbox_version'),
            n_acquisitions=cal.get('aggregation', {}).get('n_acquisitions'),
            rule=cal.get('aggregation', {}).get('rule'),
            lambda_iso_cap_rule=cal.get('aggregation', {}).get('lambda_iso_cap_rule'),
            protocol=cal['protocol'], strict=bool(strict),
            content={k: v for k, v in cal.items() if k != '_source'})
        return m

    # ------------------------------------------------------------------
    def fit(self, data, bvals, bvecs, mask, run_calibration=True,
           n_calibration_voxels=1000,
           n_iso_method='quadrature', n_bootstrap=50,
           run_sure_crosscheck=False, sure_crosscheck_n_probes=15,
           calibrate_concentration_gate=False,
           concentration_gate_percentile=_CONCENTRATION_GATE_PERCENTILE,
           _calibration_only=False, _lambda_aniso_grid=None, _lambda_iso_grid=None,
           _rician_correction='clamp', noise_map=None):
        """
        Fit the v3 hybrid two-stage adaptive DBSI model (+ MRDS
        multi-fiber extension) to 4D diffusion MRI data.

        n_iso_method : {'quadrature', 'bootstrap', 'svd_floor', 'fixed'}
            Used only when the constructor's n_iso is None. 'quadrature'
            (default since 1.3.9) fixes n_iso=6 by grid convergence — see
            `_QUADRATURE_N_ISO`. 'bootstrap' and 'svd_floor' select it per
            dataset (pre-1.3.9 behaviour); 'fixed' is the legacy n_iso=31.

        Returns
        -------
        results : ndarray (X, Y, Z, 27)
            See module docstring "Output Channels".
        model_mode : int
            2 or 3.
        """
        # ── Attributi di RUN: azzerati QUI, non in __init__ ─────────────────
        # Erano inizializzati SOLO in `__init__`, quindi un secondo `fit()` sullo
        # STESSO oggetto vedeva i valori del primo. Per la maggior parte non
        # importava (vengono riscritti a ogni fit), ma `n_iso_source_` e
        # `sure_crosscheck_report_` si scrivono solo su ALCUNI percorsi: con
        # n_iso passato (calibrazione congelata) `n_iso_source_` restava quello
        # del fit precedente, e il run report dichiarava una provenienza
        # sbagliata. La v1.3.6 ha reso la cosa consequenziale introducendo un
        # ramo che LEGGE `n_iso_source_` per sapere se il ripiego era già
        # scattato: su un oggetto riusato quel ramo scattava a torto, saltando il
        # controllo `curve_is_flat` e lasciando la provenienza mislabellata.
        # Nessun notebook riusa l'oggetto — ognuno costruisce un DBSI_Adaptive
        # per fit — ma è esattamente la classe "il comportamento dipende da come
        # lo invochi" che ha prodotto il bug della griglia isotropa.
        self.model_mode_ = None
        self.b_max_ = None
        self.n_shells_ = None
        self.n_aniso_cols_ = None
        self.diff_pairs_ = None
        self.sure_crosscheck_report_ = None
        self.n_iso_source_ = None
        self.lambda_edges_ = {}
        self.hemisphere_spacing_deg_ = None
        self.cone_refinement_schedule_ = None
        self.run_report_ = None
        self.n_iso_columns_ = None
        self.n_iso_columns_res_ = None
        self.n_iso_columns_wat_ = None
        self.rician_clamp_ = None
        self.fiber_detection_stat_ = None
        self.fiber_detection_ = None
        self.two_vs_one_stat_ = None
        self.two_vs_one_ = None
        self.fiber_threshold_final_ = None
        self.uncertainty_ = None
        self.uncertainty_summary_ = None
        self.nonfinite_voxels_excluded_ = None
        self.noise_ = None
        self.dictionary_ = None
        self.calibration_curves_ = None

        print("\n" + "="*70)
        from . import __version__ as _v       # deferred: __init__ imports us
        print(f"  DBSI ADAPTIVE PIPELINE — pyDBSI {_v}")
        print("="*70)

        bvecs = np.asarray(bvecs, dtype=np.float64)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        # ── Tabella dei gradienti coerente con i dati (v1.8.0) ───────────────
        # I kernel numba non controllano i limiti degli array: con un bvec in
        # meno dei volumi il fit girava fino in fondo leggendo memoria fuori
        # dalla tabella, senza errori e con mappe senza senso (FF = 0 ovunque);
        # con un bval in meno si fermava su un IndexError incomprensibile.
        bvals = np.asarray(bvals, dtype=np.float64).ravel()
        if np.ndim(data) != 4:
            raise ValueError(f"data deve essere 4D (X, Y, Z, volumi), ha forma {np.shape(data)}")
        n_vol = data.shape[3]
        if bvals.size != n_vol or bvecs.ndim != 2 or bvecs.shape != (n_vol, 3):
            raise ValueError(f"tabella dei gradienti incoerente con i dati: {n_vol} volumi, "
                             f"{bvals.size} b-value, bvec di forma {bvecs.shape} (attesa ({n_vol}, 3))")
        _dw0 = (bvals >= B0_THRESHOLD) & (np.linalg.norm(bvecs, axis=1) < 1e-6)
        if _dw0.any():
            raise ValueError(f"{int(_dw0.sum())} volumi pesati (b >= {B0_THRESHOLD:g}) hanno bvec "
                             f"nullo: tabella dei gradienti sbagliata")
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms

        # ── Voxel con valori non finiti: fuori dalla maschera (v1.6.2) ─────────
        # Un NaN o un inf nel segnale (tipico ai bordi dopo una correzione di bias
        # N4 con campo ~0) non si propaga come NaN: la NNLS a discesa coordinata
        # lo trasforma in pesi qualsiasi, e un S0 infinito rende (sigma/S0)^2 = 0 nel
        # test di rilevamento. Il 2026-10-01 il test si e' fermato su un
        # ZeroDivisionError sui dati veri (notebook 09); la causa esatta su quei dati
        # non e' confermata, ma un voxel non finito non ha comunque un fit sensato.
        # Si escludono qui (valgono come fuori maschera), contati nel run report.
        mask = np.asarray(mask, dtype=bool)
        _cm = np.argwhere(mask)
        _bad = ~np.all(np.isfinite(data[mask]), axis=-1)
        self.nonfinite_voxels_excluded_ = int(_bad.sum())
        if _bad.any():
            mask = mask.copy()
            _bx = _cm[_bad]
            mask[_bx[:, 0], _bx[:, 1], _bx[:, 2]] = False
            print(f"\n  [WARNING] {int(_bad.sum()):,} voxel della maschera hanno valori non "
                  f"finiti (NaN/inf) nel segnale: esclusi dal fit, come fuori maschera.")

        # ── Calibrazione di protocollo: questi dati hanno QUEL protocollo? ─────
        self.protocol_fingerprint_match_ = None
        if self.protocol_calibration_ is not None:
            from .calibration.protocol import protocol_fingerprint, fingerprint_mismatches
            _pc = self.protocol_calibration_
            _mm = fingerprint_mismatches(_pc['protocol'], protocol_fingerprint(bvals))
            self.protocol_fingerprint_match_ = not _mm
            if _mm:
                _msg = (f"impronta del protocollo diversa da quella della calibrazione "
                        f"{_pc.get('name')!r}: " + '; '.join(
                            f"{k}: calibrazione {a} / dati {b}" for k, a, b in _mm))
                if _pc.get('strict', True):
                    raise ValueError(_msg + ". Questi iperparametri non valgono per "
                                     "questa acquisizione (from_calibration(strict=False) "
                                     "per forzare, e il run report lo dichiara).")
                print(f"\n  [WARNING] {_msg} -- strict=False, si procede.")
            if calibrate_concentration_gate:
                raise ValueError("calibrate_concentration_gate=True con una calibrazione "
                                 "di protocollo: il gate e' fissato dal protocollo.")

        # ── Isotropic model selection ──────────────────────────────────────
        b_max, n_shells, use_3iso, reason = analyse_protocol(bvals)
        self.b_max_ = b_max
        self.n_shells_ = n_shells

        if self.force_n_iso not in (None, 2, 3):
            raise ValueError("force_n_iso must be 2, 3, or None.")

        # DEFAULT is now 3-ISO (RF + HF + WF) whenever the protocol supports it
        # (>=2 shells, b_max>=2000 per `analyse_protocol`); 2-ISO (RF + NRF,
        # HF+WF merged) is the fallback for single-shell / low-b_max protocols.
        # This flip (from 2-ISO-default + opt-in 3-ISO) rests on the iso-block
        # identifiability analysis + the Stage D fixed-3 estimator: 3 iso
        # components are supported down to b_max~2000 and the constrained
        # estimator recovers RF/HF/WF near-GT with low variance (and 3-ISO is
        # MORE accurate on the restricted fraction than the 2-ISO merge even
        # where the merge applies). `force_n_iso` overrides: 2 forces 2-ISO,
        # 3 forces 3-ISO (with a warning if the protocol can't support it).
        protocol_supports_3iso = use_3iso
        if self.force_n_iso == 2:
            use_3iso = False
            reason = "2-ISO model: force_n_iso=2 requested by user (RF + NRF)."
        elif self.force_n_iso == 3:
            use_3iso = True
            print(f"\n  [MODEL] force_n_iso=3 — 3-ISO (RF + HF + WF).")
            if not protocol_supports_3iso:
                print(f"  [WARNING] analyse_protocol judged this acquisition "
                      f"insufficient for a reliable HF/WF split: {reason} "
                      f"The 3-ISO water fraction may be unstable.")
        else:  # None (DEFAULT) -> follow analyse_protocol
            use_3iso = protocol_supports_3iso
            if not use_3iso:
                reason = ("2-ISO fallback (single-shell / low b_max protocol): "
                          + reason)

        model_mode = 3 if use_3iso else 2
        self.model_mode_ = model_mode

        print(f"\n  Isotropic model selection: {reason}")
        print(f"\n  Active model: {model_mode}-ISO "
              f"({'RF + HF + WF' if use_3iso else 'RF + NRF (HF+WF merged)'})")
        print(f"  b_max detected: {b_max:.0f} s/mm^2  |  "
              f"Non-zero shells: {n_shells}")
        print(f"  Max fiber populations: {MAX_FIBER_POPULATIONS} (fixed)")

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
            print(f"   NOTE: n_pop==2 voxels use Stage A's RAW grid directions as "
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

        # ── Noise: the thermal sigma map ───────────────────────────────────
        print("\n2. Noise (thermal sigma map)...")
        # Il sigma del modello e' il rumore TERMICO, dalla mappa di dwidenoise
        # (MP-PCA) portata nella scala dei dati (divisa per il campo di bias di
        # N4): `noise_map`, parte dell'ingresso fisso del toolbox. La variabilita'
        # temporale dei b=0 non lo sostituisce: contiene anche pulsazione e
        # movimento, 1.3-2.2 volte il termico sui dati veri, diversa per soggetto
        # (notebook 12, 2026-10-02). I b=0 servono solo a S0: ne basta uno.
        if noise_map is None and not _ALLOW_B0_SIGMA:
            raise ValueError(_NOISE_MAP_HELP)
        if not np.any(np.asarray(bvals) < B0_THRESHOLD):
            raise ValueError(f"nessun volume b=0 (b < {B0_THRESHOLD:g}): S0 non e' definito")
        coords = np.argwhere(mask)
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        self.noise_ = {}
        if noise_map is not None:
            noise_map = np.asarray(noise_map, dtype=np.float64)
            if noise_map.ndim == 4 and noise_map.shape[3] == 1:
                noise_map = noise_map[..., 0]
            if noise_map.shape != data.shape[:3]:
                raise ValueError(f"noise_map {noise_map.shape} non ha la griglia dei dati "
                                 f"{data.shape[:3]}")
            _nm = noise_map[xs, ys, zs]
            _ok = np.isfinite(_nm) & (_nm > 0)
            if _ok.mean() < 0.5:
                raise ValueError(f"noise_map: solo {_ok.mean():.0%} dei voxel della maschera "
                                 f"ha un valore finito > 0. E' la mappa giusta, nella scala "
                                 f"dei dati (divisa per il bias di N4)?")
            sigma_map = np.zeros(data.shape[:3], dtype=np.float64)
            sigma_map[xs, ys, zs] = np.where(_ok, _nm, np.median(_nm[_ok]))
            sigma = float(np.median(_nm[_ok]))
            _b0 = np.asarray(bvals) < B0_THRESHOLD
            _s0 = data[xs, ys, zs][:, _b0].astype(np.float64).mean(axis=1)
            _snr_t = float(np.median(_s0[_ok] / _nm[_ok]))
            self.noise_.update(
                sigma_source='noise_map', sigma_thermal_median=sigma,
                snr_thermal=_snr_t,
                noise_map_filled_voxels=int((~_ok).sum()),
                noise_map_filled_fraction=round(float((~_ok).mean()), 6))
            snr = _snr_t
            print(f"   Noise map (thermal sigma): median {sigma:.4f}, SNR (S0/sigma) {_snr_t:.2f}"
                  + (f"; {int((~_ok).sum()):,} mask voxels with no valid value filled with "
                     f"the median" if (~_ok).any() else ""))
        else:
            # Solo test e simulazioni sintetiche (_ALLOW_B0_SIGMA): sigma dalla
            # variabilita' dei b=0, che richiede almeno due volumi b=0.
            snr, sigma = estimate_snr_robust(data, bvals, mask, verbose=True)
            sigma_map = np.full(data.shape[:3], sigma, dtype=np.float64)
            self.noise_.update(sigma_source='b0_temporal', sigma_b0_temporal=float(sigma))
            print("   [WARNING] no noise_map (allowed by _ALLOW_B0_SIGMA, synthetic/legacy use "
                  "only): the model uses the b=0 temporal sigma, which also contains "
                  "physiological variability and overestimates the thermal noise.")

        # ── Rician bias correction, voxel by voxel with sigma_map ──────────
        print("\n3. Applying Rician Bias Correction...")
        data_corr = np.zeros_like(data, dtype=np.float32)
        # per voxel (v1.7.1): 2 sigma^2 col sigma del voxel; costante senza mappa
        noise_floor = (2.0 * sigma_map[xs, ys, zs] ** 2)[:, None]
        masked_sq = data[xs, ys, zs].astype(np.float64)**2
        valid_mask = masked_sq > noise_floor
        corrected = np.where(valid_mask,
                             np.sqrt(np.maximum(masked_sq - noise_floor, 0.0)),
                             0.0).astype(np.float32)
        if _rician_correction == 'signed':
            # EXPERIMENTAL (inspection 2026-09-30): keep the sign instead of
            # truncating at zero, so the noise stays symmetric around low values
            # and the NNLS sees no pile-up at exactly 0.
            _d = masked_sq - noise_floor
            corrected = (np.sign(_d) * np.sqrt(np.abs(_d))).astype(np.float32)
        elif _rician_correction != 'clamp':
            raise ValueError(f"_rician_correction: 'clamp' or 'signed', not {_rician_correction!r}")
        data_corr[xs, ys, zs] = corrected

        # ── Quante misure la correzione AZZERA, e in quanti voxel ───────────
        # `S^2 - 2 sigma^2` e' non distorto per A^2 ma puo' essere NEGATIVO, e qui
        # viene troncato a ESATTAMENTE 0. Non e' cosmetico: Stage D legge la
        # frazione restricted dal plateau ad alto b, e i valori troncati sono
        # proprio i punti ad alto b dei voxel a segnale basso. Una NNLS non puo'
        # riprodurre uno zero esatto se non azzerando i pesi, quindi il fit viene
        # tirato verso le colonne a decadimento rapido -> RF distorta VERSO IL
        # BASSO esattamente dove il segnale e' debole, e senza nessun segnale
        # d'allarme.
        #
        # La v1.3.5, correggendo sigma (che era x0.55 a 2 volumi b=0 e x0.78 a 9),
        # ha reso la correzione piu' forte e quindi ha RADDOPPIATO i troncamenti.
        # Misurato su segnale sintetico con decadimento tipico e 9 volumi b=0:
        #     SNR vero   troncate con sigma vecchio   con sigma 1.3.5
        #        37.2              0.40%                   0.91%
        #        26.3              2.44%                   4.53%
        #        20.0              5.36%                   9.26%
        #        11.7             13.50%                  20.72%
        # La correzione di sigma e' giusta; il difetto e' il TRONCAMENTO, ed era
        # la' da sempre. Qui viene almeno reso osservabile: senza questo conteggio
        # non c'era modo di sapere quanti voxel ne sono toccati.
        _n_clamp = int(np.sum(~valid_mask))
        _tot_meas = int(masked_sq.size)
        _per_vox = np.sum(~valid_mask, axis=1)
        self.rician_clamp_ = dict(
            fraction_of_measurements=round(_n_clamp / max(_tot_meas, 1), 6),
            fraction_of_voxels_any=round(float(np.mean(_per_vox > 0)), 6),
            fraction_of_voxels_over_10pct=round(
                float(np.mean(_per_vox > 0.10 * masked_sq.shape[1])), 6),
            max_clamped_in_one_voxel=int(_per_vox.max()) if _per_vox.size else 0,
            n_volumes=int(masked_sq.shape[1]),
        )
        print(f"   Rician clamp: {100*self.rician_clamp_['fraction_of_measurements']:.2f}% "
              f"of measurements truncated to exactly 0 "
              f"({100*self.rician_clamp_['fraction_of_voxels_any']:.1f}% of voxels "
              f"affected, {100*self.rician_clamp_['fraction_of_voxels_over_10pct']:.1f}% "
              f"with >10% of their volumes truncated)")
        if self.rician_clamp_['fraction_of_voxels_over_10pct'] > 0.05:
            print(f"   [WARNING] more than 5% of voxels have >10% of their measurements "
                  f"truncated to zero. Stage D reads the restricted fraction from the "
                  f"high-b plateau, which is where the truncation bites, so RF is "
                  f"biased DOWN in those voxels. Consider excluding them.")
        del masked_sq, valid_mask, corrected, _per_vox

        # ── Calibration sample: 1000 voxels, fixed seed (v1.3.1) ───────────
        # The default was 500 until v1.3.1. A seed sweep on real data (5P
        # Protocols 1 and 3, 5 seeds x {500, 1000} voxels = 20 calibrations)
        # measured how much of the lambda choice is the SAMPLE rather than the
        # protocol:
        #   lambda_aniso  identical on 20/20 runs (P1 grid node 25, P3 node 24)
        #                 -- the calibration sample does not move it at all;
        #   lambda_iso    at 500 voxels P1 spanned 3 grid nodes (23-25, a 2.03x
        #                 range) and P3 spanned 2 (19-20); at 1000 voxels P1
        #                 narrowed to 2 nodes with 4/5 seeds identical.
        # So 500 is enough for lambda_aniso but not for lambda_iso, and 1000
        # roughly halves the residual lambda_iso jitter for ~2x the calibration
        # cost (a few minutes, negligible against the voxel-wise fit). The seed
        # is fixed at 0 here and at every other sampling site in this method, so
        # a re-run of the same data reproduces the same lambdas exactly.
        if n_iso_method not in ('quadrature', 'bootstrap', 'svd_floor', 'fixed'):
            raise ValueError(
                f"n_iso_method must be 'quadrature', 'bootstrap', 'svd_floor', "
                f"or 'fixed', got {n_iso_method!r}."
            )
        # Per n_iso il campione serve solo al bootstrap: 'svd_floor' guarda solo
        # bvals e SNR, 'quadrature' e 'fixed' sono costanti. Se serve piu' avanti
        # (lambda, SURE) viene campionato li', con lo stesso seme.
        _seleziona_n_iso = (self.n_iso is None and n_iso_method == 'bootstrap')
        y_cal, sigma_cal = None, None
        if _seleziona_n_iso or (run_calibration and
                                (self.lambda_aniso is None or self.lambda_iso is None)):
            y_cal, sigma_cal = sample_calibration_voxels(
                data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
                noise_map=noise_map,
            )
            print(f"\n   Sampled {len(y_cal)} calibration voxels from the brain mask "
                  f"(sigma_normalised={sigma_cal:.5f})")

        iso_d_max = max(self.iso_range[1], _ISO_GRID_D_MAX_EXTENDED)
        # La SVD conta i modi sulla stessa gamma della griglia del fit (1.8.1):
        # col default del modulo (3.0e-3) misurava un problema piu' piccolo.
        _svd_range = dict(d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max)

        if self.n_iso is None:
            if n_iso_method == 'quadrature':
                self.n_iso = _QUADRATURE_N_ISO
                self.n_iso_source_ = 'quadrature'
                print(f"\n4. n_iso fixed by grid convergence (quadrature, not a "
                      f"per-subject selection): n_iso={self.n_iso}")

            elif n_iso_method == 'bootstrap':
                print(f"\n4. Selecting n_iso — BOOTSTRAP bias-variance "
                      f"({n_bootstrap} replicates/voxel)...")
                # Il bootstrap valuta un problema ISOTROPO, quindi va nutrito
                # con cio' che il blocco isotropo deve davvero spiegare: il
                # segnale con la FIBRA SOTTRATTA. Fino alla 1.3.5 riceveva il
                # segnale pieno, e su voxel con fibra il dizionario isotropo la
                # assorbiva: RF di riferimento 0.000 su un fantoccio con RF vera
                # 0.10, e selezione in fuga verso il candidato piu' grande.
                # La sottrazione usa una griglia isotropa FISSA e ricca, quindi
                # non dipende da quale candidato si sta valutando -- e' cosi' che
                # si rompe la circolarita' (serve una griglia per scegliere
                # n_iso, e n_iso sceglie la griglia).
                _ref_grid_pre = generate_anchored_isotropic_grid(
                    d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max,
                    n_steps=_N_ISO_REF_STEPS,
                    thresh_res=THRESH_RES, thresh_wat=THRESH_WAT,
                )
                print(f"   Subtracting the fibre component first "
                      f"({len(_ref_grid_pre)}-column reference iso grid) — the "
                      f"bootstrap scores an isotropic problem, so it must see "
                      f"the isotropic residual.")
                y_cal_iso, _resid_diag = fiber_subtracted_residual(
                    bvals, bvecs, fiber_dirs, diff_pairs, _ref_grid_pre,
                    y_cal, sigma_cal,
                )
                self.n_iso, _n_iso_diag = select_n_iso_bootstrap(
                    bvals, y_cal_iso, snr, sigma_cal,
                    d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max,
                    n_bootstrap=n_bootstrap, n_ref_steps=_N_ISO_REF_STEPS,
                )
                _n_iso_diag['fiber_subtraction'] = _resid_diag
                # Rifiuto esplicito: meglio il ripiego SVD che un n_iso scelto
                # contro un riferimento degenere.
                _rifiutato = (self.n_iso is None)
                if _rifiutato:
                    print(f"   [WARNING] the bootstrap refused to select "
                          f"(degenerate reference). Falling back to SVD+floor.")
                    self.n_iso, _svd_diag = select_n_iso_svd(bvals, snr, **_svd_range)
                    self.n_iso_source_ = 'svd_fallback_degenerate_reference'
                    print(f"   SVD+floor fallback: n_iso={self.n_iso}")
                # ── Fallback: ONLY when the composite curve is flat. ────────
                # `sample_looks_homogeneous` (all per-candidate biases < 1pp)
                # used to veto the bootstrap too. It must not: the criterion
                # is SAMPLE-SIZE DEPENDENT in the wrong direction. The bias
                # proxy is a mean over the sampled voxels, so it shrinks as
                # the sample grows and opposite-signed per-voxel biases
                # cancel — a LARGER sample from the same brain therefore looks
                # "more homogeneous", which is backwards. Measured on real
                # data (5P Protocol 1, 187,658 mask voxels, SNR 30.1): the
                # bootstrap returned n_iso=6 in all 9 configurations
                # (500/1000/2000 voxels x 3 seeds) -- perfectly stable -- yet
                # at 2000 voxels the flag fired on 2 of 3 seeds and replaced
                # that stable, data-driven 6 with the SVD floor of 10. The
                # flag fires MORE the more data you give it, and it vetoed a
                # curve that `curve_is_flat` had already judged informative.
                # It is now a warning only. See `select_n_iso_bootstrap`'s
                # docstring: it was tuned on a near-homogeneous SYNTHETIC
                # sample, where the same threshold means something else.
                # Variabile LOCALE, non l'attributo: l'attributo sopravvive fra
                # due fit sullo stesso oggetto (vedi l'azzeramento in testa a
                # fit()), quindi leggerlo qui faceva scattare questo ramo a torto.
                if _rifiutato:
                    pass                      # gia' deciso e dichiarato sopra
                elif _n_iso_diag['sample_looks_homogeneous']:
                    print(f"   [NOTE] Bias is <1 percentage point for every "
                          f"candidate n_iso. On a large real-data sample this "
                          f"is expected (the bias proxy averages out) and is "
                          f"NOT treated as a failure; it vetoed the bootstrap "
                          f"before v1.3.0.")
                if _n_iso_diag['curve_is_flat']:
                    print(f"   [WARNING] Bootstrap result unreliable (weak "
                          f"separation: the bias-variance curve has no clear "
                          f"minimum) for this dataset; falling back to "
                          f"SVD+floor as a safer default.")
                    self.n_iso, _svd_diag = select_n_iso_svd(bvals, snr, **_svd_range)
                    self.n_iso_source_ = 'svd_fallback'
                    print(f"   SVD+floor fallback: n_iso={self.n_iso}")
                else:
                    self.n_iso_source_ = 'bootstrap'
                    print(f"   Bootstrap-selected n_iso={self.n_iso}")

            elif n_iso_method == 'svd_floor':
                self.n_iso, _svd_diag = select_n_iso_svd(bvals, snr, **_svd_range)
                self.n_iso_source_ = 'svd_floor'
                print(f"\n4. Selecting n_iso — SVD + empirical floor: "
                      f"n_iso={self.n_iso} (raw SVD answer: "
                      f"{_svd_diag['n_iso_raw']}, "
                      f"floor_applied={_svd_diag['floor_applied']})")

            elif n_iso_method == 'fixed':
                self.n_iso = _DEFAULT_N_ISO_STEPS
                self.n_iso_source_ = 'fixed'
                print(f"\n4. n_iso fixed at legacy default: n_iso={self.n_iso}")

            else:
                raise AssertionError('unreachable: n_iso_method validated above')

        # ── Isotropic basis ─────────────────────────────────────────────────
        # The grid depends ONLY on n_iso and iso_range — NOT on whether n_iso was
        # SELECTED or SUPPLIED. Until v1.3.1 this assignment sat inside
        # `if self.n_iso is None:` above, with a `generate_isotropic_grid`
        # fallback in its `else`. The trigger was therefore simply *passing
        # n_iso at all*, for any reason: any caller that pinned the model order
        # — the cohort "frozen calibration" use case, but equally a one-off
        # `DBSI_Adaptive(n_iso=6)` — silently got a LINEAR basis instead of the
        # anchored log-uniform one. With n_iso=6 and the default
        # iso_range that meant 6 linear columns spanning [0, 3.0e-3] — one
        # restricted column (the degenerate d=0 one) and NO free-water column —
        # instead of 11 anchored columns over [0.1e-3, 5.0e-3] split 4/5/2 across
        # the three compartments. Same hyperparameters, different dictionary, no
        # warning. Found by a control fit that imposed a session's own calibrated
        # gate and got the *imposed*-run numbers rather than the free-run ones.
        #
        # `generate_isotropic_grid` was the legacy LINEAR constructor (removed in
        # 1.8.1) and is not
        # interchangeable with the anchored one (see its docstring): the anchoring
        # is what keeps a grid point from landing on the wrong side of THRESH_RES
        # / THRESH_WAT, which once produced an apparent total failure to detect
        # free water.
        iso_grid = generate_anchored_isotropic_grid(
            d_min=max(self.iso_range[0], 0.1e-3), d_max=iso_d_max,
            n_steps=self.n_iso, thresh_res=THRESH_RES, thresh_wat=THRESH_WAT,
        )
        # `n_iso` IS NOT THE NUMBER OF ISOTROPIC COMPONENTS. The anchored
        # constructor returns MORE points than n_steps (it pins a column on each
        # compartment threshold and fills oversized geometric gaps), and the
        # realised length is not even monotone in n_iso: with the production
        # range, n_iso=6 gives 11 columns while n_iso=8 gives 10, and the
        # restricted column count drops from 4 to 3. A reader who sees
        # "n_iso: 6" in the run report will understand six isotropic
        # components; it is eleven. Store the realised length so the report can
        # state the model order that was actually used.
        self.n_iso_columns_ = int(len(iso_grid))
        self.n_iso_columns_res_ = int(np.sum(iso_grid <= THRESH_RES))
        self.n_iso_columns_wat_ = int(np.sum(iso_grid > THRESH_WAT))

        # ── Calibration of (lambda_aniso, lambda_iso) ───────────────────────
        if run_calibration and (self.lambda_aniso is None or self.lambda_iso is None):

            print(f"\n5. Calibrating (lambda_aniso, lambda_iso) — DATA-DRIVEN "
                  f"(GCV + discrepancy principle)...")
            if y_cal is None:
                y_cal, sigma_cal = sample_calibration_voxels(
                    data_corr, mask, bvals, n_voxels=n_calibration_voxels,
                    seed=0, noise_map=noise_map,
                )
                print(f"   Sampled {len(y_cal)} calibration voxels from the "
                      f"brain mask (sigma_normalised={sigma_cal:.5f})")
            self.lambda_aniso, self.lambda_iso, _dd_diag = select_lambdas_data_driven(
                bvals, bvecs, fiber_dirs, diff_pairs, iso_grid, y_cal, sigma_cal,
                lambda_aniso_method=self.lambda_aniso_method,
                lambda_iso_grid=_lambda_iso_grid,
                lambda_aniso_grid=_lambda_aniso_grid,
            )
            # Le curve complete, non solo i minimi: servono ad aggregare la
            # calibrazione di piu' acquisizioni dello stesso protocollo.
            self.calibration_curves_ = _dd_diag
            # Su quale estremo della griglia di ricerca e' finito ciascun lambda?
            # Un lambda al bordo non e' un minimo interno: non si distingue
            # "l'ottimo e' qui" da "l'ottimo sta oltre la griglia". Va visto,
            # non dedotto per caso guardando i numeri di 110 soggetti.
            def _bordo(val, griglia):
                if griglia is None or len(griglia) == 0:
                    return 'unknown'
                g = np.asarray(griglia, dtype=float)
                if abs(val - g.min()) <= 1e-12 * max(abs(val), 1.0):
                    return 'lower'
                if abs(val - g.max()) <= 1e-12 * max(abs(val), 1.0):
                    return 'upper'
                return 'no'
            _g_an = (_dd_diag.get('lambda_aniso_selection') or {}).get('lambda_grid')
            _g_is = (_dd_diag.get('gcv') or {}).get('lambda_')
            self.lambda_edges_ = dict(
                lambda_aniso_at_grid_edge=_bordo(self.lambda_aniso, _g_an),
                lambda_iso_at_grid_edge=_bordo(self.lambda_iso, _g_is),
                lambda_iso_capped=bool(_dd_diag.get('lambda_iso_capped', False)))
            for _k, _v in self.lambda_edges_.items():
                if _v in ('lower', 'upper'):
                    print(f"   [NOTE] {_k}={_v}: il lambda scelto sta sull'ESTREMO "
                          f"della griglia di ricerca, non e' un minimo interno.")
            print(f"   Data-driven result: lambda_aniso={self.lambda_aniso:.4f}, "
                  f"lambda_iso={self.lambda_iso:.4f}  "
                  f"[lambda_aniso via {self.lambda_aniso_method}]")
            _am_diag = _dd_diag.get('lambda_aniso_selection')
            if _am_diag is not None and _am_diag.get('corner_weak'):
                print(f"   [note] {self.lambda_aniso_method} curve has no sharp corner/minimum "
                      f"for this protocol — selection may be soft (see diag).")
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
                      f"Consider increasing n_calibration_voxels "
                      f"before trusting this result.")
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


        if self.lambda_aniso is None:
            self.lambda_aniso = _STAGE_A_DEFAULT_LAMBDA_BASE * n_aniso_cols
        if self.lambda_iso is None:
            self.lambda_iso = _STAGE_A_DEFAULT_LAMBDA_BASE

        print(f"\n   Hyperparameters: n_iso={self.n_iso}, "
              f"lambda_aniso={self.lambda_aniso:.4f}, lambda_iso={self.lambda_iso:.4f}")
        print(f"   Fibre threshold: {self.fiber_threshold:.2f}  "
              f"(on Stage A's FF for candidates, on the final FF for the reported tensor)")
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
        print(f"   NOTE: reported fractions are Stage D's (every voxel); fiber tensors "
              f"only where the FINAL FF > {self.fiber_threshold:.2f}.")

        # ── Monte Carlo SURE cross-check (optional) ─────────────────────────
        if run_sure_crosscheck:
            print(f"\n   Running Monte Carlo SURE cross-check of n_iso and lambda_iso...")
            if y_cal is None:
                y_cal, sigma_cal = sample_calibration_voxels(
                    data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
                    noise_map=noise_map,
                )
                print(f"   Sampled {len(y_cal)} calibration voxels for SURE cross-check "
                      f"(sigma_normalised={sigma_cal:.5f})")

            # Scored on what the iso block must explain -- the fibre-subtracted
            # residual, the signal lambda_iso is SELECTED on -- not the raw
            # signal. Until v1.6.0 the raw signal went in: the fibre mismatch
            # dominated the risk (1.6% variation over a 16x lambda range on a
            # P3-like phantom) and the check could not disagree. Same defect the
            # n_iso bootstrap had until v1.3.6.
            y_sure, _ = fiber_subtracted_residual(
                bvals, bvecs, fiber_dirs, diff_pairs, iso_grid, y_cal, sigma_cal)
            _sure_lambda_agrees, _sure_lambda_report = crosscheck_lambda_iso_sure(
                bvals, iso_grid, y_sure, sigma_cal, self.lambda_iso,
                n_probes=sure_crosscheck_n_probes, verbose=True,
            )
            _sure_n_iso_agrees, _sure_n_iso_report = crosscheck_n_iso_sure(
                bvals, y_sure, sigma_cal, self.n_iso,
                d_min=max(self.iso_range[0], 0.1e-3),
                d_max=max(self.iso_range[1], _ISO_GRID_D_MAX_EXTENDED),
                n_probes=sure_crosscheck_n_probes, verbose=True,
            )
            self.sure_crosscheck_report_ = dict(
                signal='fiber_subtracted_residual',
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
        # Il dizionario EFFETTIVO del fit, conservato per il toolbox_report
        # (immagine + .npz). Piccolo (n_misure x ~300 colonne), e fino alla 1.3.9
        # non era ricostruibile dalle mappe: serviva rifare la configurazione.
        self.dictionary_ = dict(
            A=A, bvals=np.asarray(bvals, dtype=np.float64).ravel(), bvecs=bvecs,
            fiber_dirs=np.asarray(fiber_dirs), diff_pairs=np.asarray(diff_pairs),
            iso_grid=np.asarray(iso_grid), n_dirs=int(self.n_dirs),
            n_pairs=int(n_pairs), n_aniso_cols=int(n_aniso_cols),
            n_iso=int(self.n_iso), lambda_aniso=float(self.lambda_aniso),
            lambda_iso=float(self.lambda_iso),
            condition_number_regularized=float(cond),
            thresh_res=float(THRESH_RES), thresh_wat=float(THRESH_WAT),
        )

        # The MC-null calibration below (concentration gate) simulates
        # unit-S0 signals, so it needs sigma in NORMALISED (S/S0)
        # units — NOT the raw-signal sigma from estimate_snr_robust (which is on
        # the S0 scale, ~thousands on real data). Passing the raw sigma makes the
        # unit-amplitude MC pure noise, saturating the null concentration to ~1.0
        # and forcing the gate to reject every fiber. `sigma_cal` is the
        # calibration sample's normalised sigma and is set whenever any lambda/
        # n_iso calibration ran; sample it here if the caller supplied everything
        # explicitly so it is never None below.
        if sigma_cal is None and run_calibration and (
                calibrate_concentration_gate):
            _yc_tmp, sigma_cal = sample_calibration_voxels(
                data_corr, mask, bvals, n_voxels=n_calibration_voxels, seed=0,
                noise_map=noise_map)
            del _yc_tmp

        # ── Data-driven fiber-detection concentration gate (MC null) ────────
        # Set the concentration gate from a protocol/dictionary/lambda/SNR-
        # specific pure-isotropic NULL rather than the fixed default. Only when
        # asked (calibrate_concentration_gate=True, off since 1.4.0); otherwise
        # the constructor value (0.0 = no gate) is used.
        if run_calibration and calibrate_concentration_gate:
            if self.lambda_aniso_conc_mod:
                print("   [WARNING] calibrate_concentration_gate=True: the MC null is "
                      "computed BEFORE the concentration modulation of lambda_aniso, "
                      "the gate is applied AFTER it (see calibrate_concentration_gate_mc, "
                      "LIMITS).")
            _iso_d_lo = max(self.iso_range[0], 0.1e-3)
            _gate_mc, _gate_diag = calibrate_concentration_gate_mc(
                bvals, At, AtA_reg, n_aniso_cols, self.n_dirs, n_pairs,
                fiber_dirs, neighbor_idx, sigma_cal,
                iso_d_range=(_iso_d_lo, iso_d_max),
                fiber_threshold=self.fiber_threshold,
                default_gate=self.min_dominant_concentration,
                percentile=concentration_gate_percentile,
                n_mc=_CONCENTRATION_GATE_N_MC, b0_thr=B0_THRESHOLD, seed=0,
            )
            _fb = " (FELL BACK to default: sparse null)" if _gate_diag.get('fell_back') else ""
            print(f"   Concentration gate (MC null, {concentration_gate_percentile:.0f}th pct): "
                  f"{_gate_mc:.3f}  [default {self.min_dominant_concentration:.3f}; "
                  f"n_valid={_gate_diag.get('n_valid')}; null p50={_gate_diag['conc_p50']:.3f} "
                  f"p95={_gate_diag['conc_p95']:.3f} max={_gate_diag['conc_max']:.3f}, "
                  f"sigma={_gate_diag['sigma']:.4f}]{_fb}")
            self.min_dominant_concentration_default_ = self.min_dominant_concentration
            self.min_dominant_concentration = _gate_mc
            self.concentration_gate_diag_ = _gate_diag

        # ── Solo calibrazione: ci si ferma qui, prima del fit voxel per voxel ──
        if _calibration_only:
            return self._calibration_record(
                bvals, snr, sigma, sigma_cal, y_cal, iso_grid, run_calibration,
                calibrate_concentration_gate)

        # ── Allocate output (27 channels — see module docstring) ────────────
        # Internal layout: the public channels + the mean_iso_adc scratch channel,
        # dropped before returning (see the channel layout block).
        results = np.zeros(data.shape[:3] + (_N_CHANNELS_INTERNAL,), dtype=np.float32)
        # NaN defaults: N_POP (NaN outside fiber_threshold, NOT 0 -- the two
        # states mean different things, see `output_map_names`), the whole
        # per-population block, the FF-weighted aggregates, and the diagnostic.
        results[..., _C_NPOP:] = np.nan
        if not use_3iso:
            results[..., _C_HF] = np.nan
            results[..., _C_WF] = np.nan

        # ── Parallel voxel fitting ─────────────────────────────────────────
        n_voxels = len(coords)
        batch_sz = 10_000
        n_batches = int(np.ceil(n_voxels / batch_sz))

        print(f"\n7. Fitting {n_voxels:,} voxels "
              f"[{model_mode}-ISO model, Stage A + Stage B "
              f"(single-fiber closed-form / MRDS joint up to "
              f"{MAX_FIBER_POPULATIONS} populations)]...")
        print(f"   Population detection: basin-mass local-maxima + angular "
              f"NMS (min separation {self.min_separation_deg:.0f} deg, "
              f"min basin mass {self.min_weight_fraction:.2f} of aniso weight, "
              f"2nd pop >= {self.min_peak_ratio:.2f} x dominant, "
              f"concentration gate {self.min_dominant_concentration:.2f})")

        if self.lambda_aniso_conc_mod:
            print(f"   Plan A concentration modulation: ENABLED "
                  f"(lambda_aniso boosted up to x{self.conc_mod_gain:.0f} where "
                  f"dominant-basin concentration <= {self.conc_mod_c_lo:.2f}; "
                  f"no boost >= {self.conc_mod_c_hi:.2f}; per-voxel re-solve)")

        # ── Stage C precompute (constant across voxels) ─────────────────────
        # Stage C isotropic grid + forward columns / Gram, and the (AD, RD)
        # search grids for the joint mono-fiber re-solve. Built once per protocol.
        # NOTE: a dedicated fine geomspace iso grid was tried and REVERTED — it
        # did not help the low-FA / AD cases and slightly lowered discrimination
        # (0.856 -> 0.831 on the validation brain). The model's iso_grid is used.
        stagec_iso_grid = iso_grid
        iso_forward = np.exp(-np.outer(bvals, stagec_iso_grid)).astype(np.float64)
        iso_gram = iso_forward.T @ iso_forward
        stagec_ad_grid = np.linspace(_STAGEC_AD_MIN, _STAGEC_AD_MAX, _STAGEC_N_AD)
        stagec_rd_grid = np.linspace(_STAGEC_RD_MIN, _STAGEC_RD_MAX, _STAGEC_N_RD)
        if self.stagec_refine:
            print(f"   Stage C (joint mono-fiber tensor+fraction re-solve): ENABLED "
                  f"(VARPRO over AD[{_STAGEC_N_AD}]xRD[{_STAGEC_N_RD}] grid + local "
                  f"refine; raw signal; n_pop==1 voxels)")

        _min_separation_cos = float(np.cos(np.radians(self.min_separation_deg)))
        _kernel = _fit_voxels_3iso_v3 if use_3iso else _fit_voxels_2iso_v3
        # The crossing FF re-solve needs Stage D to turn the re-solved FF and the
        # new tensors into consistent final fractions; without it, kappa = 0.
        _kappa = self.crossing_ff_kappa if self.iso_resolve else 0.0
        print(f"   Crossing FF: " + (f"RE-SOLVED on the detected support before the LM "
              f"(ridge = {_kappa:g} x (sigma/S0)^2), then Stage D" if _kappa > 0 else
              "Stage A's, held fixed (kappa 0" + (": needs Stage D" if self.crossing_ff_kappa > 0 else "") + ")"))

        b0_thr = B0_THRESHOLD

        _test21 = self.two_vs_one_threshold is not None
        if _test21:
            print(f"   Two-vs-one population test: ENABLED (threshold "
                  f"{float(self.two_vs_one_threshold):g} noise variances; every separated "
                  f"secondary peak is a candidate, also fitted as a single fiber)")
        # The single-fiber alternative of every crossing candidate is written here by
        # the kernel itself, from the same Stage A (see _two_vs_one_pass). Without
        # the test the kernel never touches it: a 1-voxel placeholder.
        if _test21:
            _alt = np.zeros_like(results)
            _alt[..., _C_NPOP:] = np.nan
            if not use_3iso:
                _alt[..., _C_HF] = np.nan
                _alt[..., _C_WF] = np.nan
        else:
            _alt = np.zeros((1, 1, 1, results.shape[3]), dtype=results.dtype)

        def _run_kernel(run_coords, out_arr, peak_ratio, desc):
            with tqdm(total=len(run_coords), desc=desc, unit="vox") as pbar:
                for i in range(int(np.ceil(len(run_coords) / batch_sz))):
                    start = i * batch_sz
                    end = min((i + 1) * batch_sz, len(run_coords))
                    _kernel(
                        data_corr, run_coords[start:end], AtA_reg, At,
                        bvals, bvecs, fiber_dirs, diff_pairs, self.n_dirs, iso_grid,
                        b0_thr, self.fiber_threshold, self.min_weight_fraction,
                        _min_separation_cos, peak_ratio,
                        self.min_dominant_concentration,
                        self.enable_direction_refinement,
                        _cone1, _n1, _cone2, _n2,
                        neighbor_idx, MAX_FIBER_POPULATIONS, out_arr,
                        float(self.lambda_aniso), bool(self.lambda_aniso_conc_mod),
                        float(self.conc_mod_c_lo), float(self.conc_mod_c_hi),
                        float(self.conc_mod_gain),
                        bool(self.stagec_refine), iso_forward, iso_gram,
                        stagec_ad_grid, stagec_rd_grid, float(_STAGEC_ANISO_RATIO),
                        data, stagec_iso_grid, bool(self.stagec_dir_refine),
                        float(_kappa), sigma_map, bool(self.crossing_dir_refine),
                        _test21, _alt
                    )
                    pbar.update(end - start)

        t0 = time.time()
        _run_kernel(coords, results,
                    _TWO_VS_ONE_CANDIDATE_RATIO if _test21 else self.min_peak_ratio,
                    "   Progress")

        # ── Two-vs-one population test (v1.9.0) ─────────────────────────────
        # The kernel has fitted every crossing candidate both ways (crossing in
        # `results`, single fiber in `_alt`); the Stage D residuals decide.
        # See _DEFAULT_TWO_VS_ONE_THRESHOLD.
        self.two_vs_one_stat_ = None
        self.two_vs_one_ = None
        if _test21:
            _cand = coords[results[coords[:, 0], coords[:, 1], coords[:, 2], _C_NPOP] >= 2]
            _t21 = time.time()
            _stat21 = np.full(results.shape[:3], np.nan, dtype=np.float32)
            _iso_d21 = np.array(_ISO_RESOLVE_D_3ISO if use_3iso else _ISO_RESOLVE_D_2ISO,
                                dtype=np.float64)
            _two_vs_one_pass(data_corr, _cand, bvals, bvecs, b0_thr, _iso_d21, sigma_map,
                             float(self.two_vs_one_threshold), results, _alt, _stat21)
            del _alt
            self.two_vs_one_stat_ = _stat21
            _sv21 = _stat21[np.isfinite(_stat21)]
            _q21 = (np.percentile(_sv21, [5, 50, 95]).tolist() if _sv21.size else [None] * 3)
            _kept = int(np.sum(results[_cand[:, 0], _cand[:, 1], _cand[:, 2], _C_NPOP] >= 2))
            self.two_vs_one_ = dict(
                threshold=float(self.two_vs_one_threshold),
                statistic='(RSS_1pop - RSS_2pop) / sigma_voxel^2, Stage D model, before the fiber threshold',
                n_candidates=int(len(_cand)), n_tested=int(_sv21.size),
                n_kept_crossing=_kept,
                frac_kept_crossing=(_kept / len(_cand) if len(_cand) else None),
                stat_p05=_q21[0], stat_p50=_q21[1], stat_p95=_q21[2],
                seconds=round(time.time() - _t21, 1))
            print(f"   Two-vs-one test: {len(_cand):,} candidates, {_kept:,} kept as crossing "
                  f"({(_kept / len(_cand) if len(_cand) else 0):.1%}); stat p05/p50/p95 = "
                  + ('/'.join(f'{v:.1f}' for v in _q21) if _sv21.size else 'n/a')
                  + f"  ({time.time() - _t21:.1f}s)")

        elapsed = time.time() - t0
        n_fitted = int(np.sum(~np.isnan(results[..., _C_AD1]) & mask))
        n_multi = int(np.sum((results[..., _C_NPOP] >= 2) & mask))
        pct = n_fitted / n_voxels * 100 if n_voxels > 0 else 0.0
        pct_multi = n_multi / n_fitted * 100 if n_fitted > 0 else 0.0

        print(f"\n   Completed: {elapsed:.1f}s  "
              f"({n_voxels / elapsed:.0f} vox/s)")
        print(f"   AD/RD estimated: {n_fitted:,} / {n_voxels:,} "
              f"({pct:.1f}%)")
        print(f"   Multi-population (N_POP>=2) voxels: {n_multi:,} "
              f"({pct_multi:.1f}% of fitted voxels)")

        # ── Stage D: final constrained fraction re-solve (fixed iso, corrected) ──
        if self.iso_resolve:
            _iso_d = np.array(_ISO_RESOLVE_D_3ISO if use_3iso else _ISO_RESOLVE_D_2ISO,
                              dtype=np.float64)
            print(f"\n   Stage D (constrained fraction re-solve on CORRECTED signal): "
                  f"ENABLED  [fixed iso centroids {np.round(_iso_d*1e3, 2).tolist()} e-3, "
                  f"{'3-ISO' if use_3iso else '2-ISO'}]")
            _t0d = time.time()
            _npop_pre = results[..., _C_NPOP].copy()
            _iso_resolve_pass(data_corr, coords, bvals, bvecs, b0_thr, _iso_d,
                              use_3iso, results, bool(_kappa > 0),
                              float(self.fiber_threshold))
            print(f"   Stage D completed: {time.time() - _t0d:.1f}s")
            self._record_final_threshold(_npop_pre, results, mask)

            # ── Detection test: do the fibers explain more than noise? ──────
            # Always computed (it is cheap and it is what tells a reader how
            # many of the reported fibers the data support); applied only when
            # fiber_detection_threshold is set. See _fiber_detection_pass.
            _stat = np.full(results.shape[:3], np.nan, dtype=np.float32)
            _npop_before = results[..., _C_NPOP].copy()
            _thr = float(self.fiber_detection_threshold or 0.0)
            _fiber_detection_pass(data_corr, coords, bvals, bvecs, b0_thr, _iso_d,
                                  use_3iso, sigma_map, _thr, results, _stat)
            self.fiber_detection_stat_ = _stat
            _tested = np.isfinite(_stat) & mask
            _rej = _tested & np.isnan(results[..., _C_NPOP]) & ~np.isnan(_npop_before)
            _sv = _stat[_tested]
            _q = (np.percentile(_sv, [5, 50, 95]).tolist() if _sv.size else [None] * 3)
            self.fiber_detection_ = dict(
                threshold=self.fiber_detection_threshold,
                statistic='(RSS_iso_only - RSS_fibers_plus_iso) / sigma_voxel^2, Stage D model',
                n_tested=int(_tested.sum()),
                stat_p05=_q[0], stat_p50=_q[1], stat_p95=_q[2],
                frac_below_15=(float(np.mean(_sv < 15.0)) if _sv.size else None),
                n_rejected=int(_rej.sum()),
                frac_rejected=(float(_rej.sum() / _tested.sum()) if _tested.sum() else 0.0),
                crossings_among_rejected=(float(np.mean(_npop_before[_rej] >= 2))
                                          if _rej.any() else None))
            _fd = self.fiber_detection_
            print(f"   Detection test on {_fd['n_tested']:,} fiber voxels: "
                  f"stat p05/p50/p95 = "
                  + ('/'.join(f'{v:.1f}' for v in _q) if _sv.size else 'n/a')
                  + f"; below 15: {_fd['frac_below_15'] if _fd['frac_below_15'] is not None else 0:.1%}"
                  + (f"; REJECTED {_fd['n_rejected']:,} ({_fd['frac_rejected']:.1%}) at "
                     f"threshold {_thr:g}" if _thr > 0 else " (threshold off: nothing rejected)"))
        elif self.fiber_detection_threshold is not None:
            raise ValueError('fiber_detection_threshold needs Stage D (iso_resolve=True): '
                             'the test is defined on the Stage D model')
        else:
            # Without Stage D the reported fractions are Stage A's / Stage C's: the
            # fiber threshold still applies to the FINAL FF (see _iso_resolve_pass).
            # Fractions stay, as for any voxel Stage A gave no fiber on this path.
            _npop_pre = results[..., _C_NPOP].copy()
            _low = (mask & np.isfinite(_npop_pre) & (_npop_pre >= 1)
                    & ~(results[..., _C_FF] > self.fiber_threshold))
            results[_low, _C_NPOP] = np.nan
            for _c in (_C_AD1, _C_RD1, _C_FA1, _C_DIR1, _C_DIR1 + 1, _C_DIR1 + 2,
                       _C_FF2, _C_AD2, _C_RD2, _C_FA2, _C_DIR2, _C_DIR2 + 1, _C_DIR2 + 2):
                results[_low, _c] = np.nan
            self._record_final_threshold(_npop_pre, results, mask)

        # ── Derived channels (pop-1 fraction + FF-weighted tensor) ──────────
        # Last, because every stage above may still revise the fractions.
        _fill_derived_channels(results)

        # ── Fit quality (R2 / RMSE), as output channels ─────────────────────
        # Not optional: it is a forward evaluation of the model that was just
        # fitted, it costs a small fraction of the fit itself, and a set of
        # output maps nobody can judge the fit of is not a finished result.
        # Imported here rather than at module scope because `fit_quality`
        # imports this module's channel constants -- the deferred import is
        # what keeps that one-directional.
        from .fit_quality import compute_fit_quality
        # NOTE ON WHAT R2 IS MEASURED AGAINST, because it is not obvious and was
        # not declared until v1.3.6: `data` here is the RAW signal, while the
        # compartment fractions come from Stage D fitted on the RICIAN-CORRECTED
        # signal and the mono-fibre tensor from Stage C fitted on the raw one. So
        # R2 is neither a pure fit residual nor a pure prediction error: it
        # answers "how well does the final model explain what was measured",
        # which is the question a reader actually wants, but it carries a
        # systematic penalty at high b, where the raw signal sits above the true
        # decay by the Rician noise floor. The choice is deliberate and is now
        # recorded in the run report as `r2_reference`; it is NOT interchangeable
        # with an R2 computed against data_corr, so the two must never be
        # compared across runs without checking that field.
        # Centroidi della ricostruzione = quelli con cui sono state stimate le
        # frazioni riportate: Stage D se attivo, altrimenti lo spettro di Stage A.
        _fq_centroids = 'stage_d' if self.iso_resolve else 'recovered'
        results[..., _C_R2], results[..., _C_RMSE] = compute_fit_quality(
            data, bvals, bvecs, mask, results, model_mode,
            fiber_threshold=self.fiber_threshold, verbose=True,
            iso_centroids=_fq_centroids,
        )

        # ── Uncertainty maps: Fisher SE of every continuous map (v1.7.0) ────
        # Evaluated at the FINAL reported values, so it runs after every stage
        # that may revise them. Defined on the Stage D model (fixed iso
        # centroids): without Stage D the fractions come from the over-complete
        # spectrum, which has no identifiable Fisher matrix, so nothing is
        # computed and the run report says why. See `dbsi_toolbox.uncertainty`.
        if self.uncertainty and self.iso_resolve:
            from .uncertainty import compute_uncertainty, summarise_uncertainty
            _t0u = time.time()
            self.uncertainty_ = compute_uncertainty(
                data_corr, coords, bvals, bvecs, b0_thr, _iso_d, use_3iso,
                sigma_map, results)
            self.uncertainty_summary_ = summarise_uncertainty(
                self.uncertainty_, results, mask,
                DBSI_Adaptive.output_map_names(model_mode))
            _us = self.uncertainty_summary_
            _m = _us['median_se']
            print(f"   Uncertainty maps (Fisher SE at the estimate): "
                  f"{time.time() - _t0u:.1f}s; median SE "
                  f"FF {_m.get('fiber_fraction', float('nan')):.3f}, "
                  f"RF {_m.get('restricted_fraction', float('nan')):.3f}, "
                  f"RD_w {_m.get('radial_diffusivity_weighted', float('nan')) * 1e3:.3f}e-3; "
                  f"ill-conditioned {(_us['flag_ill_conditioned_pct'] or 0):.1%}")
        else:
            self.uncertainty_summary_ = dict(
                computed=False,
                reason=('disabled (uncertainty=False)' if not self.uncertainty
                        else 'needs Stage D (iso_resolve=True): the SE is defined '
                             'on the Stage D model'))

        # ── Run report: what produced these maps ────────────────────────────
        # Built here because `fit` is the only place that has all of it at once
        # — the calibrated hyperparameters, the protocol, the noise estimate and
        # the flags. `save_output_maps` writes it next to the maps.
        import datetime as _dt
        self.run_report_ = dict(
            run_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
            **_package_provenance(),
            protocol=dict(b_max=float(self.b_max_), n_shells=int(self.n_shells_),
                          n_volumes=int(len(bvals)), model_mode=int(model_mode),
                          n_dirs=int(self.n_dirs),
                          hemisphere_spacing_deg=self.hemisphere_spacing_deg_),
            # `r2_reference` dice contro quale segnale e' calcolato R2. Prima
            # della 1.3.6 non era dichiarato, quindi due R2 non erano
            # confrontabili senza leggere il codice.
            fit_quality_reference=dict(r2_reference='raw_signal',
                                       fractions_fitted_on='rician_corrected',
                                       monofiber_tensor_fitted_on='raw',
                                       iso_centroids=('stage_d_fixed' if self.iso_resolve
                                                      else 'recovered_from_mean_iso_adc')),
            # Quanto la correzione Rician ha TRONCATO. Vedi il blocco della
            # correzione: distorce la RF verso il basso dove morde, e la 1.3.5
            # l'ha raddoppiato correggendo sigma.
            rician_clamp=dict(getattr(self, 'rician_clamp_', {}) or {}),
            fiber_threshold_final=dict(self.fiber_threshold_final_ or {}),
            fiber_detection=dict(self.fiber_detection_ or {}),
            two_vs_one=dict(self.two_vs_one_ or {}),
            uncertainty=dict(self.uncertainty_summary_ or {}),
            protocol_calibration=({} if not self.protocol_calibration_ else dict(
                name=self.protocol_calibration_.get('name'),
                sha256=self.protocol_calibration_.get('sha256') or 'not saved to file',
                created_utc=self.protocol_calibration_.get('created_utc'),
                calibrated_with=self.protocol_calibration_.get('toolbox_version'),
                n_acquisitions=self.protocol_calibration_.get('n_acquisitions'),
                rule=self.protocol_calibration_.get('rule'),
                lambda_iso_cap_rule=self.protocol_calibration_.get('lambda_iso_cap_rule'),
                fingerprint_match=bool(self.protocol_fingerprint_match_))),
            # Il dizionario di Stage A: A = [A_aniso | A_iso]. La matrice stessa e
            # la sua immagine stanno in toolbox_report/ accanto a questo file.
            dictionary=dict(
                n_measurements=int(self.dictionary_['A'].shape[0]),
                n_dirs=self.dictionary_['n_dirs'],
                n_diffusivity_pairs=self.dictionary_['n_pairs'],
                n_aniso_columns=self.dictionary_['n_aniso_cols'],
                n_iso_columns=int(len(self.dictionary_['iso_grid'])),
                n_total_columns=int(self.dictionary_['A'].shape[1]),
                condition_number_regularized=self.dictionary_[
                    'condition_number_regularized'],
            ),
            input=dict(nonfinite_voxels_excluded=int(self.nonfinite_voxels_excluded_ or 0)),
            noise=dict(**(self.noise_ or {}), snr=float(snr), sigma_raw=float(sigma),
                       n_b0=int(np.sum(np.asarray(bvals) < B0_THRESHOLD)),
                       # solo nel percorso sintetico senza mappa (_ALLOW_B0_SIGMA)
                       **({} if (self.noise_ or {}).get('sigma_source') == 'noise_map' else
                          dict(sigma_estimator=('legacy_biased' if _tools._SNR_LEGACY_BIASED
                                                else 'chi_debiased_median')))),
            calibrated=dict(n_iso=int(self.n_iso),
                            # La dimensione VERA del blocco isotropo. n_iso e'
                            # il parametro chiesto, non l'ordine del modello:
                            # vedi il commento accanto a n_iso_columns_.
                            n_iso_columns=int(getattr(self, 'n_iso_columns_', -1)),
                            n_iso_columns_restricted=int(
                                getattr(self, 'n_iso_columns_res_', -1)),
                            n_iso_columns_water=int(
                                getattr(self, 'n_iso_columns_wat_', -1)),
                            lambda_aniso=float(self.lambda_aniso),
                            lambda_iso=float(self.lambda_iso),
                            concentration_gate=float(self.min_dominant_concentration),
                            calibration_run=bool(run_calibration),
                            calibration_method='data_driven',
                            lambda_aniso_method=str(self.lambda_aniso_method),
                            n_iso_method=str(n_iso_method),
                            n_iso_source=str(self.n_iso_source_ or (
                                'protocol' if self.protocol_calibration_ else 'user')),
                            # Da dove vengono lambda e gate: una calibrazione di
                            # PROTOCOLLO (from_calibration), questa acquisizione
                            # (calibrata qui), o valori passati a mano.
                            calibration_source=(
                                'protocol' if self.protocol_calibration_ else
                                'acquisition' if self.calibration_curves_ is not None
                                else 'user'),
                            calibration_seed=0,
                            # Without this the report shows a gate value but not
                            # whether it was CALIBRATED or IMPOSED — the very
                            # distinction a fixed-calibration cohort run rests on.
                            concentration_gate_calibrated=bool(
                                run_calibration and calibrate_concentration_gate),
                            concentration_gate_percentile=float(
                                concentration_gate_percentile),
                            n_calibration_voxels=int(n_calibration_voxels),
                            n_bootstrap=int(n_bootstrap),
                            **{k: (str(v) if isinstance(v, str) else v)
                               for k, v in (self.lambda_edges_ or {}).items()}),
            options=dict(max_fiber_populations=MAX_FIBER_POPULATIONS,
                         fiber_threshold=float(self.fiber_threshold),
                         anisotropy_ratio=float(self.anisotropy_ratio),
                         stagec_refine=bool(self.stagec_refine),
                         stagec_dir_refine=bool(self.stagec_dir_refine),
                         enable_direction_refinement=bool(self.enable_direction_refinement),
                         lambda_aniso_conc_mod=bool(self.lambda_aniso_conc_mod),
                         iso_resolve=bool(self.iso_resolve),
                         uncertainty=bool(self.uncertainty),
                         crossing_ff_kappa=float(_kappa),
                         crossing_dir_refine=bool(self.crossing_dir_refine),
                         two_vs_one_threshold=(None if self.two_vs_one_threshold is None
                                               else float(self.two_vs_one_threshold))),
            populations=_population_census(results, mask),
            solver=_solver_diagnostics(results, mask),
            fit_quality=dict(
                r2_median=float(np.nanmedian(results[..., _C_R2][mask])),
                rmse_median=float(np.nanmedian(results[..., _C_RMSE][mask]))),
            channel_names=list(DBSI_Adaptive.output_map_names(model_mode)),
        )

        print(f"\n{'='*70}\n")

        return np.ascontiguousarray(results[..., :_N_CHANNELS]), model_mode

    def _record_final_threshold(self, npop_pre, results, mask):
        """Count the fibers removed by fiber_threshold on the final FF (v1.8.0)."""
        _had = mask & np.isfinite(npop_pre) & (npop_pre >= 1)
        _gone = _had & np.isnan(results[..., _C_NPOP])
        n_had, n_gone = int(_had.sum()), int(_gone.sum())
        self.fiber_threshold_final_ = dict(
            fiber_threshold=float(self.fiber_threshold), applied_to='final_fiber_fraction',
            n_fiber_voxels_before=n_had, n_removed=n_gone,
            frac_removed=(round(n_gone / n_had, 6) if n_had else 0.0),
            crossings_among_removed=(round(float(np.mean(npop_pre[_gone] >= 2)), 6)
                                     if n_gone else None))
        print(f"   Fiber threshold on the final FF (<= {self.fiber_threshold:.2f}): "
              f"{n_gone:,} of {n_had:,} fiber voxels lose their fiber "
              f"({100 * (n_gone / n_had if n_had else 0):.1f}%)")

    # ------------------------------------------------------------------
    def calibrate(self, data, bvals, bvecs, mask, n_calibration_voxels=1000,
                  lambda_aniso_grid=None, lambda_iso_grid=None,
                  n_iso_method='quadrature', calibrate_concentration_gate=False,
                  noise_map=None):
        """
        Calibra UNA acquisizione senza fittarla: esegue la parte di `fit()` che
        sceglie gli iperparametri e si ferma prima del fit voxel per voxel.

        E' il mattone della calibrazione di PROTOCOLLO: si chiama su un campione
        di acquisizioni con lo stesso protocollo e si aggregano i record. Per
        questo il record contiene le curve di selezione COMPLETE (GCV di
        lambda_aniso e di lambda_iso), non solo i minimi, e l'impronta del
        protocollo (`calibration.protocol.protocol_fingerprint`).

        Il modello su cui si chiama NON viene modificato (si lavora su una
        copia), quindi lo stesso oggetto si puo' riusare su N acquisizioni.

        Parameters
        ----------
        lambda_aniso_grid, lambda_iso_grid : array or None
            Griglie di ricerca. Default: quelle di `fit()` (logspace(-4, 4, 40)
            e logspace(-5, 1, 40)). Una griglia piu' fine ha senso qui, perche'
            la calibrazione di protocollo si paga una volta sola.
        n_iso_method, calibrate_concentration_gate, noise_map
            Come in `fit()`. Il gate e' spento di default dalla 1.4.0.

        Returns
        -------
        record : dict, serializzabile in JSON (liste e float).
        """
        import copy
        if self.lambda_aniso is not None or self.lambda_iso is not None:
            raise ValueError(
                "calibrate() stima i lambda: costruisci il modello SENZA "
                "lambda_aniso / lambda_iso. Con uno solo dei due la calibrazione "
                "riscriverebbe comunque entrambi.")
        m = copy.deepcopy(self)
        return m.fit(data, bvals, bvecs, mask, run_calibration=True,
                     n_calibration_voxels=n_calibration_voxels,
                     n_iso_method=n_iso_method,
                     calibrate_concentration_gate=calibrate_concentration_gate,
                     _calibration_only=True,
                     _lambda_aniso_grid=lambda_aniso_grid,
                     _lambda_iso_grid=lambda_iso_grid,
                     noise_map=noise_map)

    def _calibration_record(self, bvals, snr, sigma, sigma_cal, y_cal, iso_grid,
                            run_calibration, calibrate_concentration_gate):
        """Il record di `calibrate()`: tutto cio' che serve ad aggregare."""
        import datetime as _dt
        from .calibration.protocol import protocol_fingerprint

        def _j(o):
            if isinstance(o, dict):
                return {str(k): _j(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_j(v) for v in o]
            if isinstance(o, np.ndarray):
                return [_j(v) for v in o.tolist()]
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            return o

        dd = self.calibration_curves_ or {}
        an = dd.get('lambda_aniso_selection') or {}
        gi = dd.get('gcv') or {}
        return _j(dict(
            kind='pydbsi_acquisition_calibration', format_version=1,
            run_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
            **_package_provenance(),
            protocol=protocol_fingerprint(bvals),
            noise=dict(**(self.noise_ or {}), snr=snr, sigma_raw=sigma, sigma_normalised=sigma_cal,
                       n_b0=int(np.sum(np.asarray(bvals).ravel() < B0_THRESHOLD))),
            n_calibration_voxels=(0 if y_cal is None else int(len(y_cal))),
            hyperparameters=dict(
                n_iso=self.n_iso, n_iso_source=str(self.n_iso_source_ or 'user'),
                n_iso_columns=int(len(iso_grid)),
                lambda_aniso=self.lambda_aniso, lambda_iso=self.lambda_iso,
                lambda_aniso_method=self.lambda_aniso_method,
                concentration_gate=self.min_dominant_concentration,
                concentration_gate_calibrated=bool(
                    run_calibration and calibrate_concentration_gate)),
            selection=dict(
                lambda_iso_gcv=dd.get('lambda_iso_gcv'),
                lambda_iso_cap=dd.get('lambda_iso_cap'),
                lambda_iso_capped=dd.get('lambda_iso_capped'),
                lambda_iso_pass1=dd.get('lambda_iso_pass1'),
                lambda_aniso_pass1=dd.get('lambda_aniso_pass1'),
                lambda_aniso_at_grid_edge=(self.lambda_edges_ or {}).get(
                    'lambda_aniso_at_grid_edge'),
                lambda_iso_at_grid_edge=(self.lambda_edges_ or {}).get(
                    'lambda_iso_at_grid_edge')),
            curves=dict(
                lambda_aniso={k: v for k, v in an.items()},
                lambda_iso=dict(lambda_grid=gi.get('lambda_'), gcv=gi.get('gcv'))),
            model=dict(n_dirs=self.n_dirs, n_ad=self.n_ad, n_rd=self.n_rd,
                       anisotropy_ratio=self.anisotropy_ratio,
                       fiber_threshold=self.fiber_threshold,
                       iso_range=list(self.iso_range), iso_grid=iso_grid),
        ))

    # ------------------------------------------------------------------
    @staticmethod
    def output_map_names(model_mode):
        """
        Return the ordered list of output channel names for the given model
        mode. The order IS the channel order of the `results` array; the
        `_C_*` module constants are the single source of truth for the indices
        and this function must stay in step with them.

        Names ending in '_NaN' mark channels that are invalid for the given
        model_mode (hindered/water in 2-ISO). Every other channel can still be
        NaN per-voxel -- the population-2 block is NaN wherever the voxel has
        no second fiber, which is most of the brain -- so callers must check
        per-voxel rather than assume a channel is entirely present or absent.

        LAYOUT (27 channels, v1.7.0)::

            0      fiber_fraction              TOTAL anisotropic fraction
            1-4    restricted/hindered/water/nonrestricted fractions
            5      n_fiber_populations         three-state, see below
            6-12   population 1: fraction, AD, RD, FA, dir(x,y,z)
            13-19  population 2: fraction, AD, RD, FA, dir(x,y,z)
            20-22  FF-weighted AD, RD, FA over the populations present
            23-25  diagnostics: dominant_basin_concentration, fit_r2, fit_rmse
            26     nnls_iterations             Stage A solver iterations (v1.3.3)

        Per-population AD in CROSSINGS is not quantitative (Cramer-Rao sd
        ~0.6e-3; it trades with RD and with the hindered compartment): use
        `axial_diffusivity_weighted` there. Per-population RD in crossings is
        reliable (crossing FF re-solved before the LM).

        There is no population 3: the toolbox resolves at most TWO fiber
        populations per voxel (`MAX_FIBER_POPULATIONS`), which is the ceiling
        the targeted clinical angular sampling supports. `ad_linear`/`rd_linear`
        are gone too -- they were byte-identical copies of the population-1
        diffusivities.

        NaN/0 CONVENTION, by block:

        * **Compartment fractions (0-4)** use 0 for "compartment absent". 0 is
          the correct physical value and the fractions stay summable, so these
          need no companion mask.
        * **The per-population block (6-19)** uses NaN for "this population is
          absent" -- including its fraction. This is deliberately NOT the
          fraction convention above: 0 would be indistinguishable from a
          detected population that carries no weight.
        * **The weighted aggregates (20-22)** are NaN wherever no fiber tensor
          was estimated, for the same reason as the tensors themselves: 0 is a
          physically plausible diffusivity and cannot serve as a sentinel.
        * **`fit_r2` / `fit_rmse` (24-25)** are NaN outside the fitted mask.
          They are filled by `fit()` itself, from a forward evaluation of the
          model it just fitted -- so the maps always ship with the means to
          judge them.
        * `fit_quality.save_output_maps` writes the matching validity mask
          (`fiber_valid.nii.gz`). It is needed as soon as the maps are
          resampled: linear interpolation turns NaN into 0 and silently
          depresses the result in proportion to the local NaN density.

        `n_fiber_populations` (channel 5) is THREE-STATE, and the states are
        NOT interchangeable::

            NaN   no fiber: Stage A's FF <= fiber_threshold, or the final FF
                  <= fiber_threshold, or rejected by the detection test;
            0     Stage A found fiber mass above the threshold, but
                  select_dominant_directions rejected every candidate peak
                  (concentration gate, min_weight_fraction, angular
                  separation). No tensor, so Stage D re-solves the fractions
                  without a fiber column: fiber_fraction is 0 in the output;
            1, 2  number of resolved fiber populations, final FF >
                  fiber_threshold.
        """
        base_3iso = [
            'fiber_fraction',
            'restricted_fraction',
            'hindered_fraction',
            'water_fraction',
            'nonrestricted_fraction',
        ]
        base_2iso = [
            'fiber_fraction',
            'restricted_fraction',
            'hindered_fraction_NaN',
            'water_fraction_NaN',
            'nonrestricted_fraction',
        ]
        fiber_block = [
            'n_fiber_populations',
            'fiber_fraction_pop1',
            'axial_diffusivity_pop1',
            'radial_diffusivity_pop1',
            'fiber_fa_pop1',
            'dir1_x', 'dir1_y', 'dir1_z',
            'fiber_fraction_pop2',
            'axial_diffusivity_pop2',
            'radial_diffusivity_pop2',
            'fiber_fa_pop2',
            'dir2_x', 'dir2_y', 'dir2_z',
            'axial_diffusivity_weighted',
            'radial_diffusivity_weighted',
            'fiber_fa_weighted',
            'dominant_basin_concentration',
            'fit_r2',
            'fit_rmse',
            'nnls_iterations',
        ]
        base = base_3iso if model_mode == 3 else base_2iso
        return base + fiber_block
