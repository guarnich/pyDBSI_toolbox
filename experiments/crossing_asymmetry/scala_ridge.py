"""Modo 10: la penalita' della FF ri-stimata scala con il numero di misure?

La ridge e' _CROSSING_FF_RIDGE_PER_MEAS x N (0.03 su 91 misure del P3). Se la scala per misura e'
quella giusta, il moltiplicatore migliore deve restare ~1 su protocolli con N diverso.
Protocolli: P3 (9 b0 + 12/20/20/30 a 500-2000, N 91), HCP-like (6 b0 + 3x60 a 1000/2000/3000,
N 186), clinico a 2 gusci (5 b0 + 30/30 a 1000/2000, N 65). Moltiplicatori 1/3, 1, 3 e produzione.
lambda di Stage A congelati al P3 per tutti (isola il modo 10; il rilevamento cambia poco).

    python scala_ridge.py [SNR] [NV]
"""
import io, sys, contextlib, numpy as np, pandas as pd
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NM = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NM)}

def fib_sphere(n, rot):
    i = np.arange(n); z = 1 - 2 * (i + 0.5) / n; r = np.sqrt(1 - z ** 2)
    th = 2 * np.pi * i / ((1 + 5 ** 0.5) / 2) + rot
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])
def prot(n_b0, gusci):
    bv = [0.0] * n_b0; bc = [np.zeros((n_b0, 3))]
    for j, (b, n) in enumerate(gusci):
        bv += [float(b)] * n; bc.append(fib_sphere(n, 0.7 * j))
    return np.array(bv), np.vstack(bc)
bv3 = np.loadtxt(HERE / 'p3_like.bval').ravel(); bc3 = np.loadtxt(HERE / 'p3_like.bvec')
bc3 = bc3.T if bc3.shape[0] == 3 else bc3; n3 = np.linalg.norm(bc3, axis=1, keepdims=True); n3[n3 == 0] = 1
PROT = {'P3 (N 91)': (bv3, bc3 / n3),
        'HCP-like (N 186)': prot(6, [(1000, 60), (2000, 60), (3000, 60)]),
        'clinico 2 gusci (N 65)': prot(5, [(1000, 30), (2000, 30)])}
S_ = (1.7e-3, 0.4e-3)
SC = {'sano RD .3': ((1.7e-3, .3e-3), (1.7e-3, .3e-3)), 'sano RD .4': (S_, S_),
      'sano RD .5': ((1.7e-3, .5e-3), (1.7e-3, .5e-3)), 'demiel A': ((1.5e-3, .8e-3), S_),
      'assonale A': ((1.1e-3, .4e-3), S_)}
K0 = M._CROSSING_FF_RIDGE_PER_MEAS
righe = []
for pn, (bv, bc) in PROT.items():
    rng = np.random.default_rng(9)
    fib = lambda d, t: np.exp(-bv * (t[1] + (t[0] - t[1]) * (bc @ d) ** 2))
    iso = 0.10 * np.exp(-bv * 0.15e-3) + 0.20 * np.exp(-bv * 1.0e-3) + 0.10 * np.exp(-bv * 3.0e-3)
    sig, lab, TR = [], [], []
    for gi, (nome, (ta, tb)) in enumerate(SC.items()):
        for _ in range(NV):
            u = rng.normal(size=3); u /= np.linalg.norm(u); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            sig.append(iso + 0.3 * fib(u, ta) + 0.3 * fib(t, tb)); lab.append(gi); TR.append((u, t, ta, tb))
    lab = np.array(lab); V = len(sig); W = 50; H = int(np.ceil(V / W))
    data = np.zeros((H, W, 1, bv.size), np.float32); mk = np.zeros((H, W, 1), bool)
    for i, S in enumerate(sig):
        data[i // W, i % W, 0] = 1000 * np.sqrt((S + rng.normal(0, 1 / SNR, S.shape)) ** 2 + rng.normal(0, 1 / SNR, S.shape) ** 2)
        mk[i // W, i % W, 0] = True
    for var, mode, mult in [('produzione', 0, 1.0), ('x1/3', 10, 1 / 3), ('x1', 10, 1.0), ('x3', 10, 3.0)]:
        m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = m.fit(data, bv, bc, mk, run_calibration=False, _mrds_mode=mode,
                           _crossing_ff_ridge_per_meas=K0 * mult)
        P = res.reshape(-1, len(NM))[:V]
        for gi, nome in enumerate(SC):
            idx = [i for i in np.where(lab == gi)[0] if P[i, IX['n_fiber_populations']] == 2]
            eA, eB, con, ff, flo = [], [], [], [], []
            for i in idx:
                u, t, ta, tb = TR[i]
                d1 = np.array([P[i, IX[f'dir1_{c}']] for c in 'xyz'])
                r1, r2 = P[i, IX['radial_diffusivity_pop1']], P[i, IX['radial_diffusivity_pop2']]
                rA, rB = (r1, r2) if abs(d1 @ u) >= abs(d1 @ t) else (r2, r1)
                eA.append(rA - ta[1]); eB.append(rB - tb[1]); con.append(rA - rB)
                ff.append(P[i, IX['fiber_fraction']] - 0.6)
                flo.append(min(r1, r2) <= _TENSOR_RD_FLOOR * 1.0001)
            ta, tb = SC[nome]
            f = lambda x: float(np.mean(x)) * 1e3 if len(x) else np.nan
            righe.append(dict(protocollo=pn, variante=var, scenario=nome, n=len(idx),
                              RD_A_bias=f(eA), RD_B_bias=f(eB), contrasto=f(con), contrasto_vero=(ta[1] - tb[1]) * 1e3,
                              FF_bias=float(np.mean(ff)) if ff else np.nan, pavimento=float(np.mean(flo)) if flo else np.nan))
    print(pn, 'fatto', flush=True)
T = pd.DataFrame(righe); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(f'SNR {SNR:g} (RD in 1e-3)'); print(T.round(3).to_string(index=False))
T.to_csv(HERE / f'esito_scala_ridge_snr{SNR:g}.csv', index=False)
