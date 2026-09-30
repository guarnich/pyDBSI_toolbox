"""Asimmetria crossing / mono-fibra: tre percorsi del crossing nel fit vero, verita' nota."""
import io, sys, contextlib, json, numpy as np
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
SP = sys.argv[1]; NV = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SNR = float(sys.argv[3]) if len(sys.argv) > 3 else 26.0
MODES = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0, 1, 2)
bvals = np.loadtxt(f'{SP}/nb09/fake_sess/prep/F_corrected.bval').ravel()
bvecs = np.loadtxt(f'{SP}/nb09/fake_sess/prep/F_corrected.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
rng = np.random.default_rng(1)

def rand_pair(ang):
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
    a = np.radians(ang); v = np.cos(a) * u + np.sin(a) * t
    return u, v

CONF = []
for ang in (60, 90):
    for split in ((0.5, 0.5), (0.7, 0.3)):
        for rd in (0.3e-3, 0.5e-3):
            CONF.append(dict(ang=ang, split=split, AD=1.7e-3, RD=rd))
FF, ISO = 0.6, dict(r=(0.10, 0.15e-3), h=(0.20, 1.0e-3), w=(0.10, 3.0e-3))
sig, truth = [], []
for ci, c in enumerate(CONF):
    for _ in range(NV):
        u, v = rand_pair(c['ang'])
        S = sum(f * np.exp(-bvals * d) for f, d in ISO.values())
        for f, d in ((c['split'][0] * FF, u), (c['split'][1] * FF, v)):
            S = S + f * np.exp(-bvals * (c['RD'] + (c['AD'] - c['RD']) * (bvecs @ d) ** 2))
        sd = 1.0 / SNR
        S = np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
        sig.append(1000 * S); truth.append((ci, u, v))
V = len(sig); nx = int(np.ceil(V / 20))
data = np.zeros((nx, 20, 1, len(bvals)), np.float32); mask = np.zeros(data.shape[:3], bool)
for i, s in enumerate(sig):
    data[i // 20, i % 20, 0] = s; mask[i // 20, i % 20, 0] = True

OUT = {}
for mode in MODES:
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, mask, run_calibration=False, _mrds_mode=mode)
    OUT[mode] = np.array([res[i // 20, i % 20, 0] for i in range(V)])

def pops(P, i):
    return [(P[i, IX[f'fiber_fraction_pop{k}']], P[i, IX[f'axial_diffusivity_pop{k}']],
             P[i, IX[f'radial_diffusivity_pop{k}']],
             np.array([P[i, IX[f'dir{k}_{c}']] for c in 'xyz'])) for k in (1, 2)]

righe = []
npop0 = OUT[0][:, IX['n_fiber_populations']]
for ci, c in enumerate(CONF):
    idx = [i for i in range(V) if truth[i][0] == ci and all(OUT[mm][i, IX['n_fiber_populations']] == 2 for mm in OUT)]
    for mode in OUT:
        P = OUT[mode]; e = dict(maj_AD=[], maj_RD=[], min_AD=[], min_RD=[], floor=[], ff=[], share=[], rmse=[])
        for i in idx:
            _, u, v = truth[i]; pp = pops(P, i)
            # abbinamento alla verita' per direzione
            s01 = abs(pp[0][3] @ u) + abs(pp[1][3] @ v); s10 = abs(pp[0][3] @ v) + abs(pp[1][3] @ u)
            maj, mino = (pp[0], pp[1]) if s01 >= s10 else (pp[1], pp[0])
            e['maj_AD'].append(maj[1] - c['AD']); e['maj_RD'].append(maj[2] - c['RD'])
            e['min_AD'].append(mino[1] - c['AD']); e['min_RD'].append(mino[2] - c['RD'])
            e['floor'].append(np.mean([maj[2] <= _TENSOR_RD_FLOOR * 1.0001, mino[2] <= _TENSOR_RD_FLOOR * 1.0001]))
            e['ff'].append(P[i, IX['fiber_fraction']] - FF)
            e['share'].append(maj[0] / max(P[i, IX['fiber_fraction']], 1e-9) - c['split'][0])
            e['rmse'].append(P[i, IX['fit_rmse']] * SNR)
        r = dict(ang=c['ang'], split=f"{c['split'][0]:.1f}", RD=c['RD'] * 1e3, mode=mode, n=len(idx),
                 cross_rate=float(np.mean([OUT[mode][i, IX['n_fiber_populations']] == 2 for i in range(V) if truth[i][0] == ci])))
        for k, v in e.items():
            v = np.array(v)
            if k in ('floor', 'rmse'): r[k] = float(np.mean(v)) if k == 'floor' else float(np.median(v))
            elif k in ('ff', 'share'): r[k + '_bias'] = float(np.mean(v)); r[k + '_mae'] = float(np.mean(np.abs(v)))
            else: r[k + '_bias'] = float(np.mean(v) * 1e3); r[k + '_mae'] = float(np.mean(np.abs(v)) * 1e3)
        righe.append(r)
# controllo: i mono-fibra devono essere identici fra i modi
mono = OUT[0][:, IX['n_fiber_populations']] == 1
ctrl = {mm: float(np.nanmax(np.abs(OUT[mm][mono] - OUT[0][mono]))) for mm in MODES if mm}
json.dump(dict(righe=righe, controllo_mono=ctrl, snr=SNR, nv=NV), open(f'{SP}/ispezione/crossing/esito_snr{int(SNR)}_m{"".join(map(str, MODES))}.json', 'w'), indent=1)
import pandas as pd
df = pd.DataFrame(righe)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.round(3).to_string(index=False))
print('controllo mono-fibra (max |diff| vs modo 0):', ctrl)
