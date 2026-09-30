"""Oracolo: lo stimatore MRDS con direzioni/frazioni/iso VERI. Da dove viene il bias di RD?"""
import sys, io, contextlib, numpy as np
from dbsi_toolbox.core.solvers import estimate_AD_RD_mrds, _TENSOR_RD_FLOOR
from dbsi_toolbox import DBSI_Adaptive
from pathlib import Path; HERE = Path(__file__).parent; SNR = 26.0
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel()
bvecs = np.loadtxt(HERE / 'p3_like.bvec'); bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
b0 = bvals < 100; rng = np.random.default_rng(5)
ISO = [(0.10, 0.15e-3), (0.20, 1.0e-3), (0.10, 3.0e-3)]
iso_true = sum(f * np.exp(-bvals * d) for f, d in ISO)
def fib(d, AD, RD): return np.exp(-bvals * (RD + (AD - RD) * (bvecs @ d) ** 2))
sd = 1 / SNR; sigma = sd
print(f"{'caso':44s} {'AD bias':>8s} {'RD bias':>8s} {'RD mae':>7s} {'floor':>6s}")
for RD in (0.3e-3, 0.5e-3):
    for lab, sig_kind, ffscale in (('oracolo, segnale CORRETTO', 'corr', 1.0), ('oracolo, segnale GREZZO', 'raw', 1.0),
                                   ('oracolo, senza rumore', 'clean', 1.0),
                                   ('FF x0.70 (come Stage A), corretto', 'corr', 0.7),
                                   ('FF x0.70, iso riscalato a 1-FF', 'corr_iso', 0.7)):
        eA, eR, fl = [], [], []
        for _ in range(300):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            dirs = np.stack([u, t]); fr = np.array([0.3, 0.3])
            S = iso_true + fr[0] * fib(u, 1.7e-3, RD) + fr[1] * fib(t, 1.7e-3, RD)
            if sig_kind == 'clean':
                y = S
            else:
                raw = np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
                raw = raw / raw[b0].mean()
                y = raw if sig_kind == 'raw' else np.sqrt(np.maximum(raw ** 2 - 2 * sigma ** 2, 0)) / np.sqrt(np.maximum(raw[b0].mean() ** 2 - 2 * sigma ** 2, 1e-9))
            frs = fr * ffscale
            iso = iso_true * ((1 - frs.sum()) / 0.4 if sig_kind == 'corr_iso' else 1.0)
            AD, RDo = estimate_AD_RD_mrds(bvals, bvecs, y.astype(np.float64), dirs, frs, iso, 3, 25)
            eA += list(AD - 1.7e-3); eR += list(RDo - RD); fl += list(RDo <= _TENSOR_RD_FLOOR * 1.0001)
        print(f'RD vero {RD*1e3:.1f}  {lab:34s} {np.mean(eA)*1e3:+8.3f} {np.mean(eR)*1e3:+8.3f} {np.mean(np.abs(eR))*1e3:7.3f} {np.mean(fl):6.2f}')

# CONTROLLO mono-fibra nel fit vero, stessi parametri (tutta la FF in una fibra)
N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
for RD in (0.3e-3, 0.5e-3):
    V = 300; data = np.zeros((15, 20, 1, len(bvals)), np.float32)
    for i in range(V):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        S = iso_true + 0.6 * fib(u, 1.7e-3, RD)
        data[i // 20, i % 20, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, np.ones(data.shape[:3], bool), run_calibration=False)
    P = res.reshape(-1, len(N)); one = P[:, IX['n_fiber_populations']] == 1
    print(f'MONO-FIBRA fit vero RD {RD*1e3:.1f}: n_pop=1 {one.mean():.2f}  AD bias {np.nanmean(P[one, IX["axial_diffusivity_pop1"]] - 1.7e-3)*1e3:+.3f}  '
          f'RD bias {np.nanmean(P[one, IX["radial_diffusivity_pop1"]] - RD)*1e3:+.3f}  floor {np.mean(P[one, IX["radial_diffusivity_pop1"]] <= _TENSOR_RD_FLOOR*1.0001):.2f}  '
          f'FF bias {np.nanmean(P[one, IX["fiber_fraction"]] - 0.6):+.3f}  crossing {np.mean(P[:, IX["n_fiber_populations"]] == 2):.2f}')
