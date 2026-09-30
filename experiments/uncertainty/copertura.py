"""Copertura delle mappe di incertezza: l'errore standard previsto descrive la dispersione vera?

Per ogni scenario, NV voxel con la STESSA verita' e rumore Rician indipendente
(protocollo P3, lambda congelati del P3, fit di produzione). Per ogni mappa:
  sd_emp   deviazione standard empirica della stima fra i voxel
  se_med   mediana dell'errore standard previsto (toolbox_report/uncertainty_maps)
  rapporto se_med / sd_emp       ~1: l'incertezza descrive la varianza
  cop95    quota di voxel con |stima - vero| <= 1.96 SE   (0.95 se niente bias)
  cop95_c  la stessa, centrata sulla media delle stime   (solo varianza, niente bias)
Solo i voxel con il numero di popolazioni giusto (e senza flag 4).

    python copertura.py [SNR] [NV] [SOGLIA_RILEVAMENTO]
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dbsi_toolbox import DBSI_Adaptive

HERE = Path(__file__).parent
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SOGLIA = float(sys.argv[3]) if len(sys.argv) > 3 else None   # fiber_detection_threshold
LAM = dict(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
bv = np.loadtxt(HERE / 'p3_like.bval').ravel(); bc = np.loadtxt(HERE / 'p3_like.bvec')
bc = bc.T if bc.shape[0] == 3 else bc
nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb
NOMI = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NOMI)}
DR, DH, DW = 0.15e-3, 1.0e-3, 3.0e-3

def fib(d, ad, rd): return np.exp(-bv * (rd + (ad - rd) * (bc @ d) ** 2))
def dirz(t, p): return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
u = dirz(1.1, 0.4); v = dirz(1.1, 0.4 + np.pi / 2)
v60 = np.cos(np.pi / 3) * u + np.sin(np.pi / 3) * np.cross(u, np.cross(v, u) / np.linalg.norm(np.cross(v, u)))

# (nome, fibre [(f, AD, RD, dir)], (fr, fh, fw))
SCEN = [
    ('mono sano',        [(0.6, 1.7e-3, 0.4e-3, u)], (0.1, 0.2, 0.1)),
    ('mono demiel',      [(0.6, 1.5e-3, 0.8e-3, u)], (0.1, 0.2, 0.1)),
    ('mono edema',       [(0.3, 1.7e-3, 0.4e-3, u)], (0.05, 0.25, 0.4)),
    ('mono cellulare',   [(0.4, 1.7e-3, 0.4e-3, u)], (0.35, 0.2, 0.05)),
    ('crossing 90 sano', [(0.3, 1.7e-3, 0.4e-3, u), (0.3, 1.7e-3, 0.4e-3, v)], (0.1, 0.2, 0.1)),
    ('crossing 60 sano', [(0.3, 1.7e-3, 0.4e-3, u), (0.3, 1.7e-3, 0.4e-3, v60)], (0.1, 0.2, 0.1)),
    ('isotropo',         [], (0.2, 0.6, 0.2)),
]

def vero(fibre, iso):
    t = {}
    ffs = [f for f, *_ in fibre]; W = sum(ffs)
    t['fiber_fraction'] = W
    t['restricted_fraction'], t['hindered_fraction'], t['water_fraction'] = iso
    if fibre:
        f, a, r, _ = fibre[0]
        t.update(fiber_fraction_pop1=f, axial_diffusivity_pop1=a, radial_diffusivity_pop1=r,
                 fiber_fa_pop1=abs(a - r) / np.sqrt(a * a + 2 * r * r))
        adw = sum(f * a for f, a, r, _ in fibre) / W; rdw = sum(f * r for f, a, r, _ in fibre) / W
        t.update(axial_diffusivity_weighted=adw, radial_diffusivity_weighted=rdw,
                 fiber_fa_weighted=abs(adw - rdw) / np.sqrt(adw ** 2 + 2 * rdw ** 2))
    if len(fibre) == 2:
        f, a, r, _ = fibre[1]
        t.update(fiber_fraction_pop2=f, axial_diffusivity_pop2=a, radial_diffusivity_pop2=r)
    return t

rng = np.random.default_rng(7)
blocchi = []
for nome, fibre, iso in SCEN:
    S = iso[0] * np.exp(-bv * DR) + iso[1] * np.exp(-bv * DH) + iso[2] * np.exp(-bv * DW)
    for f, a, r, d in fibre:
        S = S + f * fib(d, a, r)
    s = 1.0 / SNR
    blocchi.append(1000 * np.sqrt((S + rng.normal(0, s, (NV, len(bv)))) ** 2
                                  + rng.normal(0, s, (NV, len(bv))) ** 2))
data = np.stack(blocchi)[:, :, None, :].astype(np.float32)   # (scen, NV, 1, N)
m = DBSI_Adaptive(**LAM, min_dominant_concentration=0.0, fiber_detection_threshold=SOGLIA)
with contextlib.redirect_stdout(io.StringIO()):
    res, _ = m.fit(data, bv, bc, np.ones(data.shape[:3], bool), run_calibration=False)
se = m.uncertainty_['se']; fl = m.uncertainty_['flags']; lc = m.uncertainty_['log10_cond']
print(f"soglia di rilevamento {SOGLIA}; SNR {SNR:g}, {NV} voxel per scenario, sigma stimata {m.run_report_['noise']['sigma_raw']:.2f} "
      f"(vera {1000 / SNR:.2f})")

righe = []
for si, (nome, fibre, iso) in enumerate(SCEN):
    npop = res[si, :, 0, IX['n_fiber_populations']]
    giusti = (np.isnan(npop) if not fibre else npop == len(fibre)) & ((fl[si, :, 0] & 4) == 0)
    t = vero(fibre, iso)
    for nm, tv in t.items():
        est = res[si, :, 0, IX[nm]][giusti]; e = se[si, :, 0, IX[nm]][giusti]
        ok = np.isfinite(est) & np.isfinite(e)
        if ok.sum() < 20:
            continue
        est, e = est[ok], e[ok]
        sd = est.std(ddof=1)
        righe.append(dict(scenario=nome, mappa=nm, n=int(ok.sum()),
                          quota_npop_giusta=float(giusti.mean()),
                          vero=tv, media=est.mean(), bias=est.mean() - tv, sd_emp=sd,
                          se_med=np.median(e), rapporto=np.median(e) / sd if sd > 0 else np.nan,
                          cop95=np.mean(np.abs(est - tv) <= 1.96 * e),
                          cop95_c=np.mean(np.abs(est - est.mean()) <= 1.96 * e)))
    righe_f = (fl[si, :, 0] & 1) > 0
    righe.append(dict(scenario=nome, mappa='[flag RD al limite]', n=NV, vero=np.nan,
                      media=float(righe_f.mean())))
    righe.append(dict(scenario=nome, mappa='[flag mal condizionata]', n=NV, vero=np.nan,
                      media=float(((fl[si, :, 0] & 4) > 0).mean())))
T = pd.DataFrame(righe)
sc = T['mappa'].str.contains('diffusivity')
for c in ('vero', 'media', 'bias', 'sd_emp', 'se_med'):
    T.loc[sc, c] = T.loc[sc, c] * 1e3
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print('(diffusivita\' in 1e-3 mm^2/s)')
print(T.round(3).to_string(index=False))
T.to_csv(HERE / (f'esito_copertura_snr{SNR:g}' + (f'_ril{SOGLIA:g}' if SOGLIA else '') + '.csv'), index=False)
