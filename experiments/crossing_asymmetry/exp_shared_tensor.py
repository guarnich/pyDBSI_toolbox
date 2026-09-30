"""Punto 12: opzione 2 (tensore condiviso, _mrds_mode=4) e 3 (AD dai mono-fibra, =5) contro produzione.
Soggetto sintetico misto, protocollo P3, SNR 26: crossing veri + mono-fibra + tessuto isotropo (GM)."""
import io, sys, contextlib, json, numpy as np, pandas as pd
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MODES = tuple(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, 4, 5)
RAND = len(sys.argv) > 4 and sys.argv[4] == 'rand'   # diffusivita' iso vere estratte, non sui centroidi
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel(); bvecs = np.loadtxt(HERE / 'p3_like.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
rng = np.random.default_rng(42); AD = 1.7e-3
fib = lambda d, rd: np.exp(-bvals * (rd + (AD - rd) * (bvecs @ d) ** 2))
def ISO(s):
    dr, dh, dw = ((rng.uniform(0.10, 0.25), rng.uniform(0.7, 1.3), rng.uniform(2.6, 3.2)) if RAND else (0.15, 1.0, 3.0))
    return s * (0.10 * np.exp(-bvals * dr * 1e-3) + 0.20 * np.exp(-bvals * dh * 1e-3) + 0.10 * np.exp(-bvals * dw * 1e-3)) / 0.4
GM = lambda: 0.03 * np.exp(-bvals * 0.10e-3) + 0.87 * np.exp(-bvals * 0.88e-3) + 0.10 * np.exp(-bvals * 3.05e-3)
def rdir():
    u = rng.normal(size=3); return u / np.linalg.norm(u)
GRUPPI, sig, lab = [], [], []
for ang in (60, 90):
    for sp in ((0.5, 0.5), (0.7, 0.3)):
        for rd in (0.3e-3, 0.5e-3):
            GRUPPI.append(dict(nome=f'cross{ang} {sp[0]:.1f} RD{rd*1e3:.1f}', tipo='cross', rd=rd))
            for _ in range(NV):
                u = rdir(); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
                v = np.cos(np.radians(ang)) * u + np.sin(np.radians(ang)) * t
                sig.append(ISO(0.4) + 0.6 * (sp[0] * fib(u, rd) + sp[1] * fib(v, rd))); lab.append(len(GRUPPI) - 1)
for rd in (0.3e-3, 0.5e-3):
    GRUPPI.append(dict(nome=f'mono RD{rd*1e3:.1f}', tipo='mono', rd=rd))
    for _ in range(3 * NV):
        sig.append(ISO(0.45) + 0.55 * fib(rdir(), rd)); lab.append(len(GRUPPI) - 1)
GRUPPI.append(dict(nome='GM isotropo', tipo='iso', rd=np.nan))
for _ in range(3 * NV):
    sig.append(GM()); lab.append(len(GRUPPI) - 1)
lab = np.array(lab); V = len(sig); W = 50; H = int(np.ceil(V / W))
data = np.zeros((H, W, 1, len(bvals)), np.float32); mask = np.zeros((H, W, 1), bool)
sd = 1 / SNR
for i, S in enumerate(sig):
    data[i // W, i % W, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
    mask[i // W, i % W, 0] = True
OUT, STAT, ADF = {}, {}, {}
for mode in MODES:
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, mask, run_calibration=False, _mrds_mode=mode)
    OUT[mode] = res.reshape(-1, len(N))[:V]; STAT[mode] = m.fiber_detection_stat_.reshape(-1)[:V]
    ADF[mode] = getattr(m, 'crossing_ad_fixed_', None)
print(f'SNR {SNR}; iso casuali {RAND}; AD imposta (opzione 3): ' + (f'{ADF[5]*1e3:.3f}e-3 (vera 1.700)' if ADF.get(5) else '-'))
at = lambda v: np.abs(v - _TENSOR_RD_FLOOR) <= 1e-6 * _TENSOR_RD_FLOOR
righe = []
for gi, g in enumerate(GRUPPI):
    for mode in OUT:
        P = OUT[mode]; st = STAT[mode]; sel = lab == gi
        npop = np.nan_to_num(P[sel, IX['n_fiber_populations']])
        r = dict(gruppo=g['nome'], modo=mode, fibra=float(np.mean(npop >= 1)), x2=float(np.mean(npop >= 2)),
                 x2_test15=float(np.mean((npop >= 2) & (st[sel] >= 15))))
        if g['tipo'] != 'iso':
            want = 2 if g['tipo'] == 'cross' else 1
            k = sel.copy(); k[sel] = npop == want
            rdw = P[k, IX['radial_diffusivity_weighted']]; adw = P[k, IX['axial_diffusivity_weighted']]
            ff = P[k, IX['fiber_fraction']]
            flo = at(P[k, IX['radial_diffusivity_pop1']]) | (at(P[k, IX['radial_diffusivity_pop2']]) if want == 2 else False)
            r.update(n=int(k.sum()), RDw_bias=np.nanmean(rdw - g['rd']) * 1e3, RDw_mae=np.nanmean(np.abs(rdw - g['rd'])) * 1e3,
                     ADw_bias=np.nanmean(adw - AD) * 1e3, ADw_mae=np.nanmean(np.abs(adw - AD)) * 1e3,
                     pavimento=float(np.mean(flo)), FF_bias=float(np.nanmean(ff - 0.6 if want == 2 else ff - 0.55)))
        righe.append(r)
df = pd.DataFrame(righe)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(df.round(3).to_string(index=False))
c = df[df.gruppo.str.startswith('cross')].groupby('modo')[['RDw_bias', 'RDw_mae', 'ADw_bias', 'ADw_mae', 'pavimento', 'FF_bias']].mean()
print('\nMEDIA SUI CROSSING VERI'); print(c.round(3).to_string())
mono = OUT[0][:, IX['n_fiber_populations']] == 1
print('controllo mono-fibra identici fra i modi:', {mm: bool(np.array_equal(OUT[mm][mono & (OUT[mm][:, IX['n_fiber_populations']] == 1)][:, IX['radial_diffusivity_pop1']],
      OUT[0][mono & (OUT[mm][:, IX['n_fiber_populations']] == 1)][:, IX['radial_diffusivity_pop1']], equal_nan=True)) for mm in MODES if mm})
df.to_csv(HERE / f'esito_shared_snr{int(SNR)}{"_rand" if RAND else ""}.csv', index=False)
