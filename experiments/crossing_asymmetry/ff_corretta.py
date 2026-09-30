"""Si puo' correggere la FF dei crossing PRIMA di fissarla? Ri-soluzione della FF sul supporto
rilevato con penalita' ridge lam_c sul blocco anisotropo (lam_c=0: non penalizzata, sovrastima;
Stage A: sottostima). Per ogni lam_c: bias della FF e, fissata quella FF, bias della RD per fascio
dopo l'LM di produzione. Serve un lam_c che vada bene per TUTTI gli scenari, sani e patologici."""
import sys, io, contextlib, numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, '.')
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import estimate_AD_RD_mrds, _TENSOR_RD_FLOOR, iso_fraction_resolve
H = 'experiments/crossing_asymmetry/'
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0; NV = int(sys.argv[2]) if len(sys.argv) > 2 else 150
bv = np.loadtxt(H + 'p3_like.bval').ravel(); bc = np.loadtxt(H + 'p3_like.bvec'); bc = bc.T if bc.shape[0] == 3 else bc
nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb; b0 = bv < 100
NM = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NM)}
ISO = [(0.10, 0.15e-3), (0.20, 1.0e-3), (0.10, 3.0e-3)]
fib = lambda d, AD, RD: np.exp(-bv * (RD + (AD - RD) * (bc @ d) ** 2))
# scenario: (AD_A, RD_A, AD_B, RD_B)
SC = {'sano RD .3': (1.7e-3, .3e-3, 1.7e-3, .3e-3), 'sano RD .4': (1.7e-3, .4e-3, 1.7e-3, .4e-3),
      'sano RD .5': (1.7e-3, .5e-3, 1.7e-3, .5e-3), 'demiel A': (1.5e-3, .8e-3, 1.7e-3, .4e-3),
      'demiel A+B': (1.5e-3, .8e-3, 1.5e-3, .8e-3), 'assonale A': (1.1e-3, .4e-3, 1.7e-3, .4e-3)}
LAMC = [0.0, 1e-2, 3e-2]
ISO_D = np.array([0.15e-3, 1.0e-3, 3.0e-3])
rng = np.random.default_rng(5); sd = 1 / SNR; righe = []
for nome, (adA, rdA, adB, rdB) in SC.items():
    W = 20; Hh = int(np.ceil(NV / W)); data = np.zeros((Hh, W, 1, bv.size), np.float32); DIRV = []
    for i in range(NV):
        u = rng.normal(size=3); u /= np.linalg.norm(u); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
        S = sum(f * np.exp(-bv * d) for f, d in ISO) + 0.3 * fib(u, adA, rdA) + 0.3 * fib(t, adB, rdB)
        data[i // W, i % W, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
        DIRV.append(u)
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bv, bc, np.ones(data.shape[:3], bool), run_calibration=False)
    D = m.dictionary_; P = res.reshape(-1, len(NM))[:NV]; fd = D['fiber_dirs']; pairs = D['diff_pairs']; iso = D['iso_grid']
    sigma = float(m.run_report_['noise']['sigma_raw']); Aiso = np.exp(-np.outer(bv, iso))
    acc = {l: dict(ff=[], ffD=[], rdA=[], rdB=[], fl=[], adA=[], adB=[]) for l in LAMC + ['stage A']}
    for i in np.where(P[:, IX['n_fiber_populations']] == 2)[0]:
        raw = data[i // W, i % W, 0].astype(float)
        corr = np.sqrt(np.maximum(raw ** 2 - 2 * sigma ** 2, 0)); y = corr / corr[b0].mean()
        dirs = np.stack([[P[i, IX[f'dir{k}_{c}']] for c in 'xyz'] for k in (1, 2)])
        # quale popolazione e' il fascio A? quella piu' vicina a u non e' nota qui: si usa l'ordine dei
        # contrasti, confrontando RD stimata e vera a meno della permutazione (min errore)
        cols = []
        for d in dirs:
            for j in np.argsort(-np.abs(fd @ d))[:7]:
                for ad, rd in pairs:
                    cols.append(np.exp(-bv * (rd + (ad - rd) * (bc @ fd[j]) ** 2)))
        A = np.column_stack(cols + [Aiso]); na = len(cols)
        sh = np.array([P[i, IX['fiber_fraction_pop1']], P[i, IX['fiber_fraction_pop2']]]); sh = sh / sh.sum()
        a_first = abs(dirs[0] @ DIRV[i]) >= abs(dirs[1] @ DIRV[i])   # fascio A = direzione piu' vicina a u
        def valuta(key, ff, iso_sig):
            AD2, RD2 = estimate_AD_RD_mrds(bv, bc, y, dirs, sh * ff, iso_sig, 3, 25)
            ra, rb = (RD2[0], RD2[1]) if a_first else (RD2[1], RD2[0])
            aa, ab = (AD2[0], AD2[1]) if a_first else (AD2[1], AD2[0])
            acc[key]['adA'].append(aa); acc[key]['adB'].append(ab)
            # Stage D sui crossing con i tensori appena stimati: [2 fibre | 3 centroidi]
            wo = np.zeros(5); iso_fraction_resolve(y, bv, bc, dirs.copy(), AD2.copy(), RD2.copy(), 2, ISO_D, wo)
            acc[key]['ffD'].append(wo[:2].sum() / wo.sum())
            acc[key]['ff'].append(ff); acc[key]['rdA'].append(ra); acc[key]['rdB'].append(rb)
            acc[key]['fl'].append(np.mean(RD2 <= _TENSOR_RD_FLOOR * 1.0001))
        # Stage A (produzione): FF e iso dello Stage A non sono esposti; si usa la FF riportata
        # e il segnale iso di Stage D riscalato -> approssimazione: re-solve iso con FF fissa
        for l in LAMC:
            Aa = np.vstack([A, np.hstack([np.sqrt(l) * np.eye(na), np.zeros((na, Aiso.shape[1]))])]) if l > 0 else A
            ya = np.concatenate([y, np.zeros(na)]) if l > 0 else y
            w, _ = nnls(Aa, ya, maxiter=20000)
            ff = w[:na].sum() / w.sum(); valuta(l, ff, Aiso @ (w[na:] / w.sum()))
        ffp = P[i, IX['fiber_fraction']]
        acc['stage A']['ff'].append(ffp); acc['stage A']['ffD'].append(np.nan)
        acc['stage A']['rdA'].append(np.nan); acc['stage A']['rdB'].append(np.nan); acc['stage A']['fl'].append(np.nan)
        r1, r2 = P[i, IX['radial_diffusivity_pop1']], P[i, IX['radial_diffusivity_pop2']]
        q1, q2 = P[i, IX['axial_diffusivity_pop1']], P[i, IX['axial_diffusivity_pop2']]
        acc['stage A']['adA'].append(q1 if a_first else q2); acc['stage A']['adB'].append(q2 if a_first else q1)
        acc['stage A']['rdA'][-1], acc['stage A']['rdB'][-1] = (r1, r2) if a_first else (r2, r1)
        acc['stage A']['fl'][-1] = np.mean(np.array([r1, r2]) <= _TENSOR_RD_FLOOR * 1.0001)
    for key, a in acc.items():
        rA, rB = np.array(a['rdA']), np.array(a['rdB'])
        righe.append(dict(scenario=nome, lam_c=key, n=len(a['ff']), ff_bias=np.mean(a['ff']) - 0.6, ffD_bias=np.nanmean(a['ffD']) - 0.6 if a['ffD'] and np.isfinite(a['ffD']).any() else np.nan,
                          ff_sd=np.std(a['ff']), rdA_bias=(np.mean(rA) - rdA) * 1e3, rdB_bias=(np.mean(rB) - rdB) * 1e3,
                          contrasto=(np.mean(rA - rB)) * 1e3, contrasto_vero=(rdA - rdB) * 1e3, pavimento=np.mean(a['fl']),
                          adA_bias=(np.mean(a['adA']) - adA) * 1e3, adB_bias=(np.mean(a['adB']) - adB) * 1e3,
                          adA_sd=np.std(a['adA']) * 1e3, contrasto_AD=np.mean(np.array(a['adA']) - np.array(a['adB'])) * 1e3,
                          contrasto_AD_vero=(adA - adB) * 1e3))
    print(nome, 'fatto', flush=True)
T = pd.DataFrame(righe); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(f'SNR {SNR:g}  (RD in 1e-3)'); print(T.round(3).to_string(index=False))
T.to_csv(f'{sys.argv[3] if len(sys.argv) > 3 else "ff_corretta"}_snr{SNR:g}.csv', index=False)
