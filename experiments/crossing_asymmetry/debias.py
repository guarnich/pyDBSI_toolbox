"""Prototipo: FF dei crossing ristimata SENZA penalita' sul supporto rilevato (direzioni di Stage A del fit vero)."""
import sys, io, contextlib, numpy as np
from scipy.optimize import nnls
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import estimate_AD_RD_mrds, _TENSOR_RD_FLOOR
from pathlib import Path; HERE = Path(__file__).parent; SNR = 26.0; sd = 1 / SNR
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel()
bvecs = np.loadtxt(HERE / 'p3_like.bvec'); bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
b0 = bvals < 100; rng = np.random.default_rng(7)
N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
ISO = [(0.10, 0.15e-3), (0.20, 1.0e-3), (0.10, 3.0e-3)]
fib = lambda d, AD, RD: np.exp(-bvals * (RD + (AD - RD) * (bvecs @ d) ** 2))
for RD in (0.3e-3, 0.5e-3):
    V = 300; data = np.zeros((15, 20, 1, len(bvals)), np.float32); TR = []
    for i in range(V):
        u = rng.normal(size=3); u /= np.linalg.norm(u); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
        S = sum(f * np.exp(-bvals * d) for f, d in ISO) + 0.3 * fib(u, 1.7e-3, RD) + 0.3 * fib(t, 1.7e-3, RD)
        data[i // 20, i % 20, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
        TR.append((u, t))
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, np.ones(data.shape[:3], bool), run_calibration=False)
    D = m.dictionary_; P = res.reshape(-1, len(N)); fd = D['fiber_dirs']; pairs = D['diff_pairs']; iso = D['iso_grid']
    sigma = float(m.run_report_['noise']['sigma_raw'])
    Aiso = np.exp(-np.outer(bvals, iso))
    ffA, ffD, rdA, rdD, flA, flD = [], [], [], [], [], []
    for i in np.where(P[:, IX['n_fiber_populations']] == 2)[0]:
        raw = data[i // 20, i % 20, 0].astype(float)
        corr = np.sqrt(np.maximum(raw ** 2 - 2 * sigma ** 2, 0)); y = corr / corr[b0].mean()
        dirs = np.stack([[P[i, IX[f'dir{k}_{c}']] for c in 'xyz'] for k in (1, 2)])
        # supporto: i nodi di griglia delle due direzioni + i 6 vicini assiali di ciascuno, tutte le coppie (AD,RD)
        cols = []
        for d in dirs:
            near = np.argsort(-np.abs(fd @ d))[:7]
            for j in near:
                for ad, rd in pairs:
                    cols.append(np.exp(-bvals * (rd + (ad - rd) * (bvecs @ fd[j]) ** 2)))
        A = np.column_stack(cols + [Aiso]); w, _ = nnls(A, y)
        na = len(cols); ff_d = w[:na].sum() / w.sum()
        wi = w[na:] / w.sum()
        iso_sig = Aiso @ wi
        ffA.append(P[i, IX['fiber_fraction']]); ffD.append(ff_d)
        shares = np.array([P[i, IX['fiber_fraction_pop1']], P[i, IX['fiber_fraction_pop2']]]); shares = shares / shares.sum()
        AD2, RD2 = estimate_AD_RD_mrds(bvals, bvecs, y, dirs, shares * ff_d, iso_sig, 3, 25)
        rdA += [P[i, IX['radial_diffusivity_pop1']] - RD, P[i, IX['radial_diffusivity_pop2']] - RD]
        rdD += list(RD2 - RD); flA += [P[i, IX['radial_diffusivity_pop1']] <= _TENSOR_RD_FLOOR * 1.0001, P[i, IX['radial_diffusivity_pop2']] <= _TENSOR_RD_FLOOR * 1.0001]
        flD += list(RD2 <= _TENSOR_RD_FLOOR * 1.0001)
    print(f'RD vero {RD*1e3:.1f}  crossing {len(ffA)}:  FF bias Stage A {np.mean(ffA)-0.6:+.3f}  ristimata {np.mean(ffD)-0.6:+.3f} (mae {np.mean(np.abs(np.array(ffD)-0.6)):.3f})   '
          f'RD bias prod {np.mean(rdA)*1e3:+.3f} -> {np.mean(rdD)*1e3:+.3f}   pavimento {np.mean(flA):.2f} -> {np.mean(flD):.2f}')
