"""
Per-voxel uncertainty of the reported maps (v1.7.0).

WHAT IS COMPUTED
----------------
For every fitted voxel, the standard error (SE) of each continuous output map,
from the Fisher information of the signal model the maps were reported under:

    S(b, g) / S0 = sum_k w_k exp(-b (RD_k + (AD_k - RD_k) (g . d_k)^2))
                 + sum_j u_j exp(-b D_j)

with D_j the fixed Stage D centroids (RF / HF / WF, or RF / NRF in 2-ISO), up
to two fiber populations k, and Gaussian noise of standard deviation
sigma_voxel = sigma_raw / S0_voxel on the Rician-corrected signal. The free
parameters are, per population, the weight w_k, AD_k, RD_k and two tangent
angles of the direction d_k, plus the iso weights u_j. With J the Jacobian at
the reported estimate,

    Cov = sigma_voxel^2 (J^T J)^-1

and the SE of every reported quantity follows by the delta method: the
normalised fractions (FF, RF, HF, WF, NRF, FF_pop1/2), FA per population, and
the FF-weighted AD / RD / FA. The directions are FREE in J -- the reported
directions are estimates too, and fixing them would make every tensor SE
optimistic -- and their angular SE is reported as its own map.

WHAT THE NUMBER MEANS -- READ BEFORE USING THE MAPS
---------------------------------------------------
* It is the Cramer-Rao bound evaluated at the estimate: the precision an
  unbiased estimator of this model could reach with this protocol and this
  noise. It is NOT the bias. Where the estimator is biased (crossing RD on the
  floor, demyelinated RD under-estimated, see core/solvers.py) the maps can be
  confidently wrong. The coverage check in the dossier measures how far the
  two diverge on synthetic data.
* Crossings report the Stage A total FF and pop-2 fraction, not the fit of
  the model above; their SE is the information the data carry about FF under
  this model, not the spread of the Stage A estimator.
* It is LOCAL: a linearisation at the estimate. It is exact to first order
  where the likelihood is close to Gaussian, which fails near a bound and
  where the model is barely identifiable. Both cases are flagged.
* Boundaries are NOT imposed (non-negative weights, the RD floor): the SE of a
  fraction estimated at 0 is the width of what the data allow around 0, not 0.
  This over-states the spread of a constrained estimator there, which is the
  conservative side.
* Fiber vs hindered degeneracy dominates: a healthy single fiber at SNR 26
  (P3) has FF +/- 0.15 and RD +/- 0.12e-3, a demyelinated one FF +/- 0.66.
  Large SEs there are the finding, not a malfunction. Averaging over a region
  of N voxels shrinks the SE by up to sqrt(N) (less with spatial correlation).

FLAGS (bitmask, `uncertainty_flags`)
------------------------------------
    1  a population's RD is on a bound (the RD floor, in practice): the
       tensor sits where the data did not determine it -- its SE describes the
       unconstrained model, not the clamped value;
    2  a population's AD is on a bound;
    4  ill-conditioned Fisher matrix (log10 condition > _LOG10_COND_FLAG after
       column equilibration): the model is close to non-identifiable in this
       voxel and the linearisation is unreliable. SE is still reported unless
       the matrix is numerically singular (then NaN).
    8  AD of the crossing populations was imposed, not estimated: it is not a
       free parameter and its SE is NaN (reserved for the crossing-AD method
       under evaluation; never set in this release).
   16  the residual is larger than noise explains: residual_over_sigma above
       the 99.9% quantile of its chi-square null. The model does not describe
       the voxel, or the local noise is above the global sigma (g-factor); either
       way the SEs are optimistic by about that ratio.

RESIDUAL / SIGMA (`residual_over_sigma`)
----------------------------------------
sqrt(RSS / (N - P)) / sigma_voxel, on the Rician-corrected signal, for the
reported model (amplitude refitted). ~1 when the model describes the voxel and
sigma is right. It is NOT fit_rmse / sigma: fit_rmse is taken against the RAW
signal (see fit_quality), whose Rician floor alone would push the ratio above 1
at high b, and it has no degrees-of-freedom correction.

WHY THIS AND NOT A BOOTSTRAP
----------------------------
A residual or wild bootstrap would include the estimator's own behaviour
(bias, bound activity), which is attractive, but costs a full refit per
replicate: ~100x the fit. The Fisher SE is a forward evaluation and runs in a
fraction of a second per 10^5 voxels. The price is that it describes the
model, not the estimator; the dossier's coverage section is where the gap
between the two is measured instead of assumed.
"""

import numpy as np
from numba import njit, prange

from .model_Niso_adaptive_ff_thr import (
    _C_FF, _C_RF, _C_HF, _C_WF, _C_NRF, _C_NPOP,
    _C_FF1, _C_AD1, _C_RD1, _C_FA1, _C_DIR1,
    _C_FF2, _C_AD2, _C_RD2, _C_FA2, _C_DIR2,
    _C_ADW, _C_RDW, _C_FAW, _N_CHANNELS,
)

UNCERTAINTY_DIRNAME = 'uncertainty_maps'

# Channels that get an SE map. Everything else is either discrete
# (n_fiber_populations), a unit vector (directions: see the angular map), a
# diagnostic, or `mean_iso_adc`, which no model here produces (it is a mean of
# the Stage A spectrum that Stage D does not use).
SE_CHANNELS = (_C_FF, _C_RF, _C_HF, _C_WF, _C_NRF,
               _C_FF1, _C_AD1, _C_RD1, _C_FA1,
               _C_FF2, _C_AD2, _C_RD2, _C_FA2,
               _C_ADW, _C_RDW, _C_FAW)

FLAG_RD_BOUND = 1
FLAG_AD_BOUND = 2
FLAG_ILL_CONDITIONED = 4
FLAG_AD_IMPOSED = 8
FLAG_RESIDUAL = 16

# Condition number of the column-equilibrated Fisher matrix above which the
# voxel is flagged. Equilibrated, cond ~ 1 / (1 - rho^2) for the worst pair of
# parameters: 1e6 is |rho| > 0.9999995, where a first-order SE is no longer a
# description of the likelihood. Singular (SE = NaN) above 1e14.
_LOG10_COND_FLAG = 6.0
_LOG10_COND_SINGULAR = 14.0

# Flag 16: residual_over_sigma above the 99.9% quantile of its null distribution,
# sqrt(chi2_{N-P}(0.999) / (N-P)) (Wilson-Hilferty), so the threshold follows the
# protocol: 1.22 for P3 (N 91) with one fiber. Measured on synthetic P3 data with
# the correct model: median 1.00 / p99 1.15-1.24 at SNR 26-40; 1.07-1.12 / p99
# 1.23-1.34 at SNR 15, where the Rician-corrected signal is no longer Gaussian.
# Crossings (Stage A FF held fixed): median 1.16 at SNR 26, 1.36 at SNR 40 -- the
# bias shows up as misfit. Fiber dispersion (sd 20 deg) and iso diffusivities off
# the Stage D centroids move the median only to 1.01-1.06: the ratio does not see
# every misfit at these SNRs.
_RESID_FLAG_Z = 3.090

# Diffusivities enter J in units of 1e-3 mm^2/s so the Fisher matrix is not
# scaled by 1e6 between blocks (equilibration handles the rest).
_DSCALE = 1e-3


@njit(cache=True)
def _fa_and_grad(ad, rd):
    """FA of an axially symmetric tensor and its gradient wrt (AD, RD).
    FA = |AD - RD| / sqrt(AD^2 + 2 RD^2)."""
    q = ad * ad + 2.0 * rd * rd
    if q <= 0.0:
        return np.nan, 0.0, 0.0
    sq = np.sqrt(q)
    dif = ad - rd
    s = 1.0 if dif >= 0.0 else -1.0
    fa = abs(dif) / sq
    g_ad = s / sq - abs(dif) * ad / (q * sq)
    g_rd = -s / sq - abs(dif) * 2.0 * rd / (q * sq)
    return fa, g_ad, g_rd


@njit(cache=True)
def _quad(g, C):
    return np.sqrt(max(g @ C @ g, 0.0))


@njit(parallel=True, cache=True)
def _uncertainty_kernel(data_corr, coords, bvals, bvecs, b0_thr, iso_d, use_3iso,
                        sigma_raw, out, ad_bounds, rd_bounds, crossing_ad_fixed,
                        resid_flag, se, dir_se, log10_cond, flags, resid_ratio):
    """Fisher SE for every fitted voxel. Writes se[..., ch] for SE_CHANNELS,
    dir_se[..., 0/1] (degrees), log10_cond and flags. See the module docstring."""
    n_voxels = coords.shape[0]
    n_iso = iso_d.shape[0]
    N = bvals.shape[0]
    for idx in prange(n_voxels):
        x, y, z = coords[idx]
        sig = data_corr[x, y, z]
        s0 = 0.0
        cnt = 0
        for i in range(N):
            if bvals[i] < b0_thr:
                s0 += sig[i]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        if s0 < 1e-6 or np.isnan(out[x, y, z, _C_RF]):
            continue
        sv2 = (sigma_raw / s0) ** 2

        # ── the reported estimate ────────────────────────────────────────
        npop = out[x, y, z, _C_NPOP]
        n_fib = 0
        if not np.isnan(npop) and npop >= 1 and not np.isnan(out[x, y, z, _C_AD1]):
            n_fib = 1
            if npop >= 2 and not np.isnan(out[x, y, z, _C_AD2]):
                n_fib = 2
        w = np.zeros(2)
        ad = np.zeros(2)
        rd = np.zeros(2)
        d = np.zeros((2, 3))
        for k in range(n_fib):
            cf = _C_FF1 if k == 0 else _C_FF2
            ca = _C_AD1 if k == 0 else _C_AD2
            cr = _C_RD1 if k == 0 else _C_RD2
            cd = _C_DIR1 if k == 0 else _C_DIR2
            w[k] = out[x, y, z, cf]
            ad[k] = out[x, y, z, ca]
            rd[k] = out[x, y, z, cr]
            for a in range(3):
                d[k, a] = out[x, y, z, cd + a]
        u = np.zeros(n_iso)
        u[0] = out[x, y, z, _C_RF]
        if use_3iso:
            u[1] = out[x, y, z, _C_HF]
            u[2] = out[x, y, z, _C_WF]
        else:
            u[1] = out[x, y, z, _C_NRF]

        # ── which parameters are free ────────────────────────────────────
        # Per population: w, AD, RD, alpha, beta (AD dropped when imposed).
        fix_ad = crossing_ad_fixed and n_fib == 2
        n_per = 4 if fix_ad else 5
        P = n_fib * n_per + n_iso
        J = np.zeros((N, P))
        fl = 0
        for k in range(n_fib):
            # tangent basis at d_k
            dk0 = d[k, 0]
            dk1 = d[k, 1]
            dk2 = d[k, 2]
            ax0 = abs(dk0)
            ax1 = abs(dk1)
            ax2 = abs(dk2)
            t0 = 0.0
            t1 = 0.0
            t2 = 0.0
            if ax0 <= ax1 and ax0 <= ax2:
                t0 = 1.0
            elif ax1 <= ax2:
                t1 = 1.0
            else:
                t2 = 1.0
            e10 = dk1 * t2 - dk2 * t1
            e11 = dk2 * t0 - dk0 * t2
            e12 = dk0 * t1 - dk1 * t0
            ne = np.sqrt(e10 * e10 + e11 * e11 + e12 * e12)
            e10 /= ne
            e11 /= ne
            e12 /= ne
            e20 = dk1 * e12 - dk2 * e11
            e21 = dk2 * e10 - dk0 * e12
            e22 = dk0 * e11 - dk1 * e10
            base = k * n_per
            for i in range(N):
                gx = bvecs[i, 0]
                gy = bvecs[i, 1]
                gz = bvecs[i, 2]
                c = gx * dk0 + gy * dk1 + gz * dk2
                c2 = c * c
                b = bvals[i]
                E = np.exp(-b * (rd[k] + (ad[k] - rd[k]) * c2))
                col = base
                J[i, col] = E
                col += 1
                if not fix_ad:
                    J[i, col] = -w[k] * b * c2 * E * _DSCALE
                    col += 1
                J[i, col] = -w[k] * b * (1.0 - c2) * E * _DSCALE
                col += 1
                f = -w[k] * b * (ad[k] - rd[k]) * 2.0 * c * E
                J[i, col] = f * (gx * e10 + gy * e11 + gz * e12)
                J[i, col + 1] = f * (gx * e20 + gy * e21 + gz * e22)
            for m in range(rd_bounds.shape[0]):
                if abs(rd[k] - rd_bounds[m]) <= 1e-6 * rd_bounds[m]:
                    fl |= 1
            for m in range(ad_bounds.shape[0]):
                if abs(ad[k] - ad_bounds[m]) <= 1e-6 * ad_bounds[m]:
                    fl |= 2
        if fix_ad:
            fl |= 8
        ib = n_fib * n_per
        for j in range(n_iso):
            for i in range(N):
                J[i, ib + j] = np.exp(-bvals[i] * iso_d[j])

        # ── residual of the reported model, in units of the noise ────────
        # sqrt(RSS / (N - P)) / sigma_voxel on the Rician-corrected signal, the
        # reported model rescaled by its best overall amplitude (the fractions
        # sum to 1, the measured S0 is noisy). ~1 when the model describes the
        # voxel and sigma is right; above 1 the SEs are optimistic by about
        # that factor (misfit, or local noise above the global sigma).
        pred = np.zeros(N)
        for k in range(n_fib):
            for i in range(N):
                pred[i] += w[k] * J[i, k * n_per]
        for j in range(n_iso):
            for i in range(N):
                pred[i] += u[j] * J[i, ib + j]
        num = 0.0
        den = 0.0
        for i in range(N):
            num += pred[i] * sig[i] / s0
            den += pred[i] * pred[i]
        amp = num / den if den > 0.0 else 1.0
        rss = 0.0
        for i in range(N):
            r = sig[i] / s0 - amp * pred[i]
            rss += r * r
        if N > P:
            dof = N - P
            rr = np.sqrt(rss / dof / sv2)
            resid_ratio[x, y, z] = rr
            h = 2.0 / (9.0 * dof)
            q = (1.0 - h + resid_flag * np.sqrt(h)) ** 3   # chi2_dof quantile / dof
            if rr > np.sqrt(q):
                fl |= 16

        # ── Fisher, equilibrated, inverted ───────────────────────────────
        F = J.T @ J
        sc = np.empty(P)
        ok = True
        for p in range(P):
            if F[p, p] <= 0.0:
                ok = False
                break
            sc[p] = np.sqrt(F[p, p])
        if not ok:
            flags[x, y, z] = fl | 4
            log10_cond[x, y, z] = np.inf
            continue
        Fs = np.empty((P, P))
        for p in range(P):
            for q in range(P):
                Fs[p, q] = F[p, q] / (sc[p] * sc[q])
        ev, V = np.linalg.eigh(Fs)
        lmax = ev[P - 1]
        lmin = ev[0]
        if lmin <= lmax * 10.0 ** (-_LOG10_COND_SINGULAR):
            flags[x, y, z] = fl | 4
            log10_cond[x, y, z] = np.inf
            continue
        lc = np.log10(lmax / lmin)
        log10_cond[x, y, z] = lc
        if lc > _LOG10_COND_FLAG:
            fl |= 4
        flags[x, y, z] = fl
        C = np.zeros((P, P))
        for p in range(P):
            for q in range(P):
                acc = 0.0
                for r in range(P):
                    acc += V[p, r] * V[q, r] / ev[r]
                C[p, q] = acc * sv2 / (sc[p] * sc[q])

        # ── delta method ─────────────────────────────────────────────────
        W = 0.0
        for k in range(n_fib):
            W += w[k]
        U = 0.0
        for j in range(n_iso):
            U += u[j]
        tot = W + U
        if tot <= 1e-10:
            continue
        g = np.zeros(P)

        # compartment fractions: FF = W / tot, class c = U_c / tot
        if n_fib > 0:
            g[:] = 0.0
            for k in range(n_fib):
                g[k * n_per] = U / (tot * tot)
            for j in range(n_iso):
                g[ib + j] = -W / (tot * tot)
            se[x, y, z, _C_FF] = _quad(g, C)
        for cls in range(3 if use_3iso else 2):
            # cls 0 = RF, 1 = HF (3-ISO) or NRF (2-ISO), 2 = WF
            Uc = u[cls]
            g[:] = 0.0
            for k in range(n_fib):
                g[k * n_per] = -Uc / (tot * tot)
            for j in range(n_iso):
                g[ib + j] = ((tot if j == cls else 0.0) - Uc) / (tot * tot)
            ch = _C_RF if cls == 0 else (_C_HF if (use_3iso and cls == 1) else
                                         (_C_WF if cls == 2 else _C_NRF))
            se[x, y, z, ch] = _quad(g, C)
        if use_3iso:
            # NRF = HF + WF
            Uc = u[1] + u[2]
            g[:] = 0.0
            for k in range(n_fib):
                g[k * n_per] = -Uc / (tot * tot)
            for j in range(n_iso):
                g[ib + j] = ((tot if j >= 1 else 0.0) - Uc) / (tot * tot)
            se[x, y, z, _C_NRF] = _quad(g, C)

        # per population
        for k in range(n_fib):
            base = k * n_per
            i_ad = -1 if fix_ad else base + 1
            i_rd = base + (1 if fix_ad else 2)
            i_a = i_rd + 1
            cf = _C_FF1 if k == 0 else _C_FF2
            ca = _C_AD1 if k == 0 else _C_AD2
            cr = _C_RD1 if k == 0 else _C_RD2
            cfa = _C_FA1 if k == 0 else _C_FA2
            g[:] = 0.0
            for m in range(n_fib):
                g[m * n_per] = ((tot if m == k else 0.0) - w[k]) / (tot * tot)
            for j in range(n_iso):
                g[ib + j] = -w[k] / (tot * tot)
            se[x, y, z, cf] = _quad(g, C)
            if i_ad >= 0:
                se[x, y, z, ca] = np.sqrt(C[i_ad, i_ad]) * _DSCALE
            se[x, y, z, cr] = np.sqrt(C[i_rd, i_rd]) * _DSCALE
            fa, gad, grd = _fa_and_grad(ad[k], rd[k])
            if i_ad >= 0:
                g[:] = 0.0
                g[i_ad] = gad * _DSCALE
                g[i_rd] = grd * _DSCALE
                se[x, y, z, cfa] = _quad(g, C)
            dir_se[x, y, z, k] = np.degrees(np.sqrt(C[i_a, i_a] + C[i_a + 1, i_a + 1]))

        # FF-weighted tensor: X_w = sum_k w_k X_k / W
        if n_fib > 0 and W > 1e-10:
            adw = 0.0
            rdw = 0.0
            for k in range(n_fib):
                adw += w[k] * ad[k] / W
                rdw += w[k] * rd[k] / W
            g_ad = np.zeros(P)
            g_rd = np.zeros(P)
            for k in range(n_fib):
                base = k * n_per
                i_rd = base + (1 if fix_ad else 2)
                g_ad[base] = (ad[k] - adw) / W
                g_rd[base] = (rd[k] - rdw) / W
                if not fix_ad:
                    g_ad[base + 1] = w[k] / W * _DSCALE
                g_rd[i_rd] = w[k] / W * _DSCALE
            if not fix_ad:
                se[x, y, z, _C_ADW] = _quad(g_ad, C)
            se[x, y, z, _C_RDW] = _quad(g_rd, C)
            if not fix_ad:
                faw, gfa_ad, gfa_rd = _fa_and_grad(adw, rdw)
                g = gfa_ad * g_ad + gfa_rd * g_rd
                se[x, y, z, _C_FAW] = _quad(g, C)


def compute_uncertainty(data_corr, coords, bvals, bvecs, b0_thr, iso_d, use_3iso,
                        sigma_raw, results, crossing_ad_fixed=False):
    """Run the kernel and return a dict of arrays + a summary for the run report.

    `results` must be final (after Stage D, the detection test and
    `_fill_derived_channels`): the SE is evaluated at what is reported.
    """
    from .core.solvers import (_TENSOR_AD_FLOOR, _TENSOR_AD_CEIL,
                               _TENSOR_RD_FLOOR, _TENSOR_RD_CEIL)
    from .model_Niso_adaptive_ff_thr import (_STAGEC_AD_MIN, _STAGEC_AD_MAX,
                                             _STAGEC_RD_MIN, _STAGEC_RD_MAX)
    shp = results.shape[:3]
    se = np.full(shp + (_N_CHANNELS,), np.nan, dtype=np.float64)
    dir_se = np.full(shp + (2,), np.nan, dtype=np.float64)
    lc = np.full(shp, np.nan, dtype=np.float64)
    flags = np.zeros(shp, dtype=np.uint8)
    rr = np.full(shp, np.nan, dtype=np.float64)
    ad_b = np.array([_TENSOR_AD_FLOOR, _TENSOR_AD_CEIL, _STAGEC_AD_MIN, _STAGEC_AD_MAX])
    rd_b = np.array([_TENSOR_RD_FLOOR, _TENSOR_RD_CEIL, _STAGEC_RD_MIN, _STAGEC_RD_MAX])
    _uncertainty_kernel(np.ascontiguousarray(data_corr, dtype=np.float64),
                        np.ascontiguousarray(coords), np.asarray(bvals, np.float64),
                        np.ascontiguousarray(bvecs, dtype=np.float64), float(b0_thr),
                        np.asarray(iso_d, np.float64), bool(use_3iso), float(sigma_raw),
                        np.ascontiguousarray(results, dtype=np.float64), ad_b, rd_b,
                        bool(crossing_ad_fixed), float(_RESID_FLAG_Z), se, dir_se, lc, flags, rr)
    return dict(se=se.astype(np.float32), dir_se_deg=dir_se.astype(np.float32),
                log10_cond=lc.astype(np.float32), flags=flags,
                residual_over_sigma=rr.astype(np.float32))


def _q(v, p):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if v.size else None


def summarise_uncertainty(unc, results, mask, channel_names):
    """Median SE per map over the voxels where it exists, and flag shares.
    Goes into the run report."""
    se, fl = unc['se'], unc['flags']
    fitted = mask & np.isfinite(unc['log10_cond'])
    med = {}
    for ch in SE_CHANNELS:
        nm = channel_names[ch]
        if nm.endswith('_NaN'):
            continue
        v = se[..., ch][mask]
        v = v[np.isfinite(v)]
        if v.size:
            med[nm] = float(np.median(v))
    for k in (0, 1):
        v = unc['dir_se_deg'][..., k][mask]
        v = v[np.isfinite(v)]
        if v.size:
            med[f'dir{k + 1}_angle_deg'] = float(np.median(v))
    n = int(fitted.sum())
    share = (lambda b: float(np.mean((fl[fitted] & b) > 0)) if n else None)
    return dict(method='fisher_crlb_at_estimate_delta_method',
                noise='sigma_raw / S0_voxel, Gaussian, rician-corrected signal',
                directions='free (2 tangent angles per population)',
                bounds='not imposed (unconstrained local information)',
                n_voxels=n,
                median_se=med,
                flag_rd_bound_pct=share(FLAG_RD_BOUND),
                flag_ad_bound_pct=share(FLAG_AD_BOUND),
                flag_ill_conditioned_pct=share(FLAG_ILL_CONDITIONED),
                flag_residual_above_noise_pct=share(FLAG_RESIDUAL),
                residual_over_sigma_median=_q(unc['residual_over_sigma'][mask], 50),
                residual_over_sigma_p95=_q(unc['residual_over_sigma'][mask], 95),
                ill_conditioned_log10_cond=_LOG10_COND_FLAG)
