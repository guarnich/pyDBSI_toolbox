"""Correzione RF (default ON) su tessuto ETEROGENEO: artefatto o problema vero?

Il sospetto: `build_rf_response_table` si calibra sui voxel del dataset stesso,
quindi su dati omogenei (tutti i voxel con la stessa FF/RF) non ha dinamica da
invertire. Qui le classi coprono RF da 0.02 a 0.50.
"""
import sys, os, io, contextlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbsi_toolbox import DBSI_Adaptive

SNR = 30.0
PER_CLASS = 40
rng = np.random.default_rng(7)

#            nome              FF    RF    HF    WF    AD      RD
CLASSES = [("WM sano",        0.60, 0.08, 0.22, 0.10, 1.80e-3, 0.30e-3),
           ("WM demielin.",   0.50, 0.08, 0.30, 0.12, 1.70e-3, 0.60e-3),
           ("WM infiammato",  0.45, 0.30, 0.15, 0.10, 1.60e-3, 0.40e-3),
           ("GM",             0.15, 0.20, 0.50, 0.15, 1.20e-3, 0.50e-3),
           ("edema",          0.30, 0.05, 0.50, 0.15, 1.60e-3, 0.50e-3),
           ("tumore",         0.20, 0.50, 0.20, 0.10, 1.40e-3, 0.45e-3),
           ("CSF",            0.00, 0.02, 0.08, 0.90, 1.00e-3, 0.50e-3),
           ("WM parz.vol.",   0.35, 0.15, 0.35, 0.15, 1.70e-3, 0.45e-3)]

shells = [(0, 9), (300, 3), (700, 12), (1000, 20), (2000, 48)]
bvals, bvecs = [], []
for b, nd in shells:
    for _ in range(nd):
        bvals.append(float(b))
        if b == 0: bvecs.append([0., 0., 0.])
        else:
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v)
bvals, bvecs = np.array(bvals), np.array(bvecs)
D_RES, D_HIN, D_WAT = 0.15e-3, 1.0e-3, 3.0e-3

N = len(CLASSES) * PER_CLASS
side = int(np.ceil(N ** (1/3))) + 1
idx = [(i,j,k) for i in range(side) for j in range(side) for k in range(side)][:N]
data = np.zeros((side,)*3 + (len(bvals),), np.float32)
mask = np.zeros((side,)*3, bool)
GT = {k: np.zeros(N) for k in ('ff','rf','hf','wf','ad','rd')}
LBL = np.empty(N, object)
for n, (i,j,k) in enumerate(idx):
    name, ff, rf, hf, wf, ad, rd = CLASSES[n // PER_CLASS]
    v = rng.normal(size=3); v /= np.linalg.norm(v); v *= np.sign(v[2]) or 1
    c = bvecs @ v
    sig = rf*np.exp(-bvals*D_RES) + hf*np.exp(-bvals*D_HIN) + wf*np.exp(-bvals*D_WAT)
    if ff > 0:
        sig = sig + ff*np.exp(-bvals*(rd + (ad-rd)*c**2))
    noisy = np.sqrt((sig + rng.normal(0,1/SNR,len(bvals)))**2 + rng.normal(0,1/SNR,len(bvals))**2)
    data[i,j,k] = 1000.0*noisy; mask[i,j,k] = True
    for key, val in zip(('ff','rf','hf','wf','ad','rd'), (ff,rf,hf,wf,ad,rd)): GT[key][n] = val
    LBL[n] = name
print(f"{N} voxel, {len(CLASSES)} classi, SNR {SNR:.0f}")
print(f"RF vera: da {GT['rf'].min():.2f} a {GT['rf'].max():.2f}  (dinamica reale)\n")

def run(corr):
    m = DBSI_Adaptive()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        res, mode = m.fit(data, bvals, bvecs, mask, run_calibration=True,
                          n_calibration_voxels=250,
                          correct_restricted_fraction=corr)
    names = DBSI_Adaptive.output_map_names(mode)
    I = {n_:i for i,n_ in enumerate(names)}
    g = lambda nm: np.array([res[i,j,k,I[nm]] for (i,j,k) in idx])
    log = buf.getvalue()
    rfline = [l for l in log.splitlines() if 'RF bias correction' in l or 'RF response' in l]
    return dict(rf=g('restricted_fraction'), ff=g('fiber_fraction'),
                hf=g('hindered_fraction'), log=rfline)

print("fit con correzione RF OFF...");  OFF = run(False)
print("fit con correzione RF ON...");   ON  = run(True)
for l in ON['log']: print("   ", l.strip())

print(f"\n{'classe':<15}{'RF vera':>9}{'RF (OFF)':>10}{'RF (ON)':>10}{'|err| OFF':>11}{'|err| ON':>10}")
print("-"*66)
for ci,(name,*_ ) in enumerate(CLASSES):
    s = slice(ci*PER_CLASS, (ci+1)*PER_CLASS)
    t = GT['rf'][s]
    a, b = OFF['rf'][s], ON['rf'][s]
    print(f"{name:<15}{t[0]:>9.2f}{np.median(a):>10.3f}{np.median(b):>10.3f}"
          f"{np.mean(np.abs(a-t)):>11.3f}{np.mean(np.abs(b-t)):>10.3f}")
print("-"*66)
eo, en = np.abs(OFF['rf']-GT['rf']), np.abs(ON['rf']-GT['rf'])
print(f"{'TOTALE':<15}{'':>9}{'':>10}{'':>10}{eo.mean():>11.3f}{en.mean():>10.3f}")
print(f"\nla correzione RF {'MIGLIORA' if en.mean()<eo.mean() else 'PEGGIORA'} "
      f"l'errore medio su RF del {abs(en.mean()-eo.mean())/eo.mean()*100:.0f}%")
