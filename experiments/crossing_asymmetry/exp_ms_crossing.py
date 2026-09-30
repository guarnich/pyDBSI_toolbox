"""Punto 12, casi da sclerosi multipla: le varianti del crossing recuperano RD per popolazione anche
quando il tessuto e' patologico e l'AD imposta (dai mono-fibra sani) e' sbagliata?
Modi: 0 produzione, 4 tensore condiviso, 7 AD imposta + RD separate, 8 AD condivisa stimata + RD separate."""
import io, sys, contextlib, numpy as np, pandas as pd
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MODES = tuple(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, 4, 7, 8)
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel(); bvecs = np.loadtxt(HERE / 'p3_like.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
rng = np.random.default_rng(7)
fib = lambda d, ad, rd: np.exp(-bvals * (rd + (ad - rd) * (bvecs @ d) ** 2))
def iso(rf, hf, wf):
    return (rf * np.exp(-bvals * rng.uniform(0.10, 0.25) * 1e-3) + hf * np.exp(-bvals * rng.uniform(0.7, 1.3) * 1e-3)
            + wf * np.exp(-bvals * rng.uniform(2.6, 3.2) * 1e-3))
def rdir():
    u = rng.normal(size=3); return u / np.linalg.norm(u)
S_ = (1.7e-3, 0.4e-3); DEM = (1.5e-3, 0.8e-3); AXO = (1.1e-3, 0.4e-3)
# nome: (tensore A, tensore B, FF, (rf, hf, wf))
SC = {'sano':            (S_, S_, 0.6, (0.10, 0.20, 0.10)),
      'demiel 1 fascio': (DEM, S_, 0.6, (0.10, 0.20, 0.10)),
      'demiel 2 fasci':  (DEM, DEM, 0.6, (0.10, 0.20, 0.10)),
      'assonale 1':      (AXO, S_, 0.6, (0.10, 0.20, 0.10)),
      'lesione':         (DEM, DEM, 0.35, (0.15, 0.25, 0.25))}
sig, lab, TR = [], [], []
G = []
for ang in (90, 60):
    for nome, (ta, tb, ff, isof) in SC.items():
        G.append(f'{nome} {ang}')
        for _ in range(NV):
            u = rdir(); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            v = np.cos(np.radians(ang)) * u + np.sin(np.radians(ang)) * t
            s = iso(*isof) / sum(isof) * (1 - ff) + ff / 2 * (fib(u, *ta) + fib(v, *tb))
            sig.append(s); lab.append(len(G) - 1); TR.append((u, v, ta, tb, ff))
for _ in range(8 * NV):          # sostanza bianca sana mono-fibra: da qui l'AD del soggetto
    rd = rng.uniform(0.3, 0.5) * 1e-3
    sig.append(iso(0.10, 0.25, 0.10) / 0.45 * 0.45 + 0.55 * fib(rdir(), 1.7e-3, rd)); lab.append(-1); TR.append(None)
lab = np.array(lab); V = len(sig); W = 50; H = int(np.ceil(V / W))
data = np.zeros((H, W, 1, len(bvals)), np.float32); mask = np.zeros((H, W, 1), bool); sd = 1 / SNR
for i, S in enumerate(sig):
    data[i // W, i % W, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
    mask[i // W, i % W, 0] = True
OUT, ADF = {}, {}
for mode in MODES:
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, mask, run_calibration=False, _mrds_mode=mode)
    OUT[mode] = res.reshape(-1, len(N))[:V]; ADF[mode] = getattr(m, 'crossing_ad_fixed_', None)
print(f'SNR {SNR}; AD imposta dai mono-fibra sani: ' + ', '.join(f'modo {k}: {v*1e3:.3f}e-3' for k, v in ADF.items() if v))
at = lambda v: np.abs(v - _TENSOR_RD_FLOOR) <= 1e-6 * _TENSOR_RD_FLOOR
righe = []
for gi, g in enumerate(G):
    for mode in MODES:
        P = OUT[mode]; idx = [i for i in np.where(lab == gi)[0] if P[i, IX['n_fiber_populations']] == 2]
        eA, eB, con, rdw, adw, flo, ident = [], [], [], [], [], [], []
        for i in idx:
            u, v, ta, tb, ff = TR[i]
            d1 = np.array([P[i, IX[f'dir1_{c}']] for c in 'xyz']); d2 = np.array([P[i, IX[f'dir2_{c}']] for c in 'xyz'])
            r1, r2 = P[i, IX['radial_diffusivity_pop1']], P[i, IX['radial_diffusivity_pop2']]
            rA, rB = (r1, r2) if abs(d1 @ u) + abs(d2 @ v) >= abs(d1 @ v) + abs(d2 @ u) else (r2, r1)
            eA.append(rA - ta[1]); eB.append(rB - tb[1]); con.append(rA - rB)
            rdw.append(P[i, IX['radial_diffusivity_weighted']] - (ta[1] + tb[1]) / 2)
            adw.append(P[i, IX['axial_diffusivity_weighted']] - (ta[0] + tb[0]) / 2)
            flo.append(at(r1) or at(r2)); ident.append(abs(r1 - r2) < 1e-9)
        ta, tb = SC[g.rsplit(' ', 1)[0]][:2]
        f = lambda x: float(np.mean(x)) * 1e3 if len(x) else np.nan
        righe.append(dict(scenario=g, modo=mode, n=len(idx), RD_A_bias=f(eA), RD_B_bias=f(eB),
                          contrasto_vero=(ta[1] - tb[1]) * 1e3, contrasto_stimato=f(con),
                          RDw_bias=f(rdw), ADw_bias=f(adw), pavimento=float(np.mean(flo)) if flo else np.nan,
                          pop1_uguale_pop2=float(np.mean(ident)) if ident else np.nan))
df = pd.DataFrame(righe)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(df.round(3).to_string(index=False))
df.to_csv(HERE / (f'esito_ms_snr{int(SNR)}' + ('' if MODES == (0, 4, 7, 8) else '_modi' + '-'.join(map(str, MODES))) + '.csv'), index=False)
