"""Modo 10 (FF dei crossing ri-stimata prima dell'LM) sulla batteria a 13 scenari di luglio.

Protocollo HCP-like (6 b0 + 3x60 a 1000/2000/3000), calibrazione sui dati del fantoccio come in un
uso reale, SNR 30 e 15, produzione contro modo 10, test di rilevamento spento e a soglia 15.
Domanda: il modo 10 migliora i tensori dei crossing senza peggiorare frazioni e scenari non-crossing?
(Il modo 10 tocca solo i voxel con 2 popolazioni: gli altri devono restare identici.)

    python batteria_modo10.py [NEACH]
"""
import io, sys, contextlib, numpy as np, pandas as pd
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
NEACH = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NM = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NM)}

def build_protocol(n_b0=6, dps=(60, 60, 60), bsh=(1000, 2000, 3000)):
    bvals = [0.0] * n_b0; bvecs = [np.zeros(3) for _ in range(n_b0)]; g = (1 + 5 ** 0.5) / 2
    for b, n in zip(bsh, dps):
        idx = np.arange(n); z = 1 - 2 * (idx + 0.5) / n; r = np.sqrt(1 - z ** 2); th = 2 * np.pi * idx / g
        bvecs.extend(np.column_stack([r * np.cos(th), r * np.sin(th), z]).tolist()); bvals.extend([float(b)] * n)
    return np.array(bvals), np.array(bvecs)
bv, bc = build_protocol()
def dp(th, ph):
    t = np.radians(th); p = np.radians(ph); return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
d1, d2, d0 = dp(65, 0), dp(65, 90), dp(90, 0)
I3 = lambda r, h, w: ((r, 0.15e-3), (h, 1.0e-3), (w, 3.0e-3))
BATT = {'WM_sano': ([(0.55, d0, 1.7e-3, 0.30e-3)], I3(0.10, 0.25, 0.10)),
        'WM_demyel': ([(0.45, d0, 1.5e-3, 0.70e-3)], I3(0.15, 0.25, 0.15)),
        'WM_densa': ([(0.70, d0, 1.8e-3, 0.25e-3)], I3(0.10, 0.15, 0.05)),
        'WM_debole': ([(0.20, d0, 1.7e-3, 0.30e-3)], I3(0.10, 0.50, 0.20)),
        'Cross_bil': ([(0.35, d1, 1.6e-3, 0.40e-3), (0.35, d2, 1.6e-3, 0.40e-3)], I3(0.10, 0.15, 0.05)),
        'Cross_sbil': ([(0.45, d1, 1.7e-3, 0.30e-3), (0.20, d2, 1.6e-3, 0.40e-3)], I3(0.10, 0.15, 0.10)),
        'Cross_eterog': ([(0.30, d1, 1.7e-3, 0.30e-3), (0.20, d2, 1.4e-3, 0.65e-3)], I3(0.18, 0.14, 0.18)),
        'GM': ([(0.10, d0, 1.5e-3, 0.50e-3)], I3(0.10, 0.65, 0.15)),
        'CSF': ((), ((0.05, 1.0e-3), (0.95, 3.5e-3))),
        'PureRestr': ((), I3(0.30, 0.50, 0.20)),
        'Tumor': ((), I3(0.55, 0.30, 0.15)),
        'Edema': ([(0.30, d0, 1.7e-3, 0.35e-3)], ((0.15, 1.0e-3), (0.55, 3.0e-3))),
        'Lesione': ([(0.25, d0, 1.6e-3, 0.45e-3)], I3(0.35, 0.30, 0.10))}
sig, lab, VER = [], [], []
for n, (fb, iso) in BATT.items():
    S0 = sum(f * np.exp(-bv * D) for f, D in iso)
    for f, d, ad, rd in fb:
        S0 = S0 + f * np.exp(-bv * (rd + (ad - rd) * (bc @ d) ** 2))
    W = sum(f for f, *_ in fb)
    ver = dict(FF=W, RF=sum(f for f, D in iso if D <= 0.3e-3), WF=sum(f for f, D in iso if D >= 3.0e-3))
    ver['HF'] = sum(f for f, _ in iso) - ver['RF'] - ver['WF']
    ver['RDw'] = sum(f * rd for f, _, _, rd in fb) / W if fb else np.nan
    ver['ADw'] = sum(f * ad for f, _, ad, _ in fb) / W if fb else np.nan
    ver['npop'] = len(fb); ver['fb'] = fb
    for _ in range(NEACH):
        sig.append(S0); lab.append(n); VER.append(ver)
lab = np.array(lab); V = len(sig); Wd = 50; H = int(np.ceil(V / Wd))
righe = []
for snr in (30, 15):
    rng = np.random.default_rng(0)
    data = np.zeros((H, Wd, 1, bv.size), np.float32); mk = np.zeros((H, Wd, 1), bool)
    for i, S in enumerate(sig):
        data[i // Wd, i % Wd, 0] = 1000 * np.sqrt((S + rng.normal(0, 1 / snr, S.shape)) ** 2 + rng.normal(0, 1 / snr, S.shape) ** 2)
        mk[i // Wd, i % Wd, 0] = True
    for mode in (0, 10):
        for soglia in (None, 15.0):
            m = DBSI_Adaptive(n_iso=6, fiber_detection_threshold=soglia)
            with contextlib.redirect_stdout(io.StringIO()):
                res, _ = m.fit(data, bv, bc, mk, run_calibration=True, _mrds_mode=mode)
            P = res.reshape(-1, len(NM))[:V]
            for n in BATT:
                s = lab == n; v = VER[np.where(s)[0][0]]
                r = dict(SNR=snr, modo=mode, soglia=soglia or 0, scenario=n)
                for k, c in (('FF', 'fiber_fraction'), ('RF', 'restricted_fraction'), ('HF', 'hindered_fraction'), ('WF', 'water_fraction')):
                    r['err_' + k] = float(np.mean(np.abs(np.nan_to_num(P[s, IX[c]]) - v[k])))
                npop = P[s, IX['n_fiber_populations']]
                r['npop'] = float(np.mean(np.nan_to_num(npop))); r['npop_vero'] = v['npop']
                if v['npop']:
                    rdw = P[s, IX['radial_diffusivity_weighted']]; adw = P[s, IX['axial_diffusivity_weighted']]
                    r['RDw_bias'] = float(np.nanmean(rdw) - v['RDw']) * 1e3
                    r['ADw_bias'] = float(np.nanmean(adw) - v['ADw']) * 1e3
                    rds = np.concatenate([P[s, IX['radial_diffusivity_pop1']], P[s, IX['radial_diffusivity_pop2']]])
                    rds = rds[np.isfinite(rds)]
                    r['pavimento'] = float(np.mean(rds <= _TENSOR_RD_FLOOR * 1.0001)) if rds.size else np.nan
                righe.append(r)
            print(f'SNR {snr} modo {mode} soglia {soglia}: lambda_aniso {m.lambda_aniso:.3g}', flush=True)
T = pd.DataFrame(righe)
T.to_csv(HERE / 'esito_batteria_modo10.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
cols = ['err_FF', 'err_RF', 'err_HF', 'err_WF']
print('\nMAE su tutti gli scenari'); print(T.groupby(['SNR', 'soglia', 'modo'])[cols].mean().round(4).to_string())
cr = T[T.scenario.str.startswith('Cross')]
print('\nCrossing'); print(cr[['SNR', 'soglia', 'modo', 'scenario', 'err_FF', 'err_RF', 'npop', 'RDw_bias', 'ADw_bias', 'pavimento']].round(3).to_string(index=False))
nc = T[~T.scenario.str.startswith('Cross')]
print('\nNon-crossing (devono coincidere fra modo 0 e 10, salvo voxel classificati come crossing)')
print(nc.groupby(['SNR', 'soglia', 'modo'])[cols + ['RDw_bias']].mean().round(4).to_string())
