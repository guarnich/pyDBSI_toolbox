"""Quanto pesa l'errore di direzione (nodo di griglia, non raffinato) sulla RD dei crossing? Solutore del modo 7."""
import numpy as np
from pathlib import Path
from dbsi_toolbox.core.solvers import crossing_varpro_ad_shared_rd_sep, _TENSOR_RD_FLOOR
from dbsi_toolbox.core.basis import generate_anchored_isotropic_grid
HERE = Path(__file__).parent
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel(); bvecs = np.loadtxt(HERE / 'p3_like.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
b0 = bvals < 100; rng = np.random.default_rng(9)
iso_grid = generate_anchored_isotropic_grid(d_min=0.1e-3, d_max=5.0e-3, n_steps=6, thresh_res=0.3e-3, thresh_wat=3.0e-3)
iso_fw = np.exp(-np.outer(bvals, iso_grid)); adg = np.linspace(0.6e-3, 2.6e-3, 14); rdg = np.linspace(0.15e-3, 1.1e-3, 12)
fib = lambda d, ad, rd: np.exp(-bvals * (rd + (ad - rd) * (bvecs @ d) ** 2))
isoS = lambda: 0.10 * np.exp(-bvals * 0.15e-3) + 0.20 * np.exp(-bvals * 1.0e-3) + 0.10 * np.exp(-bvals * 3.0e-3)
def rdir():
    u = rng.normal(size=3); return u / np.linalg.norm(u)
def tilt(u, deg):
    t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t); a = np.radians(deg)
    return np.cos(a) * u + np.sin(a) * t
for SNR in (26, 40):
    sd = 1 / SNR
    for ad, rd, lab in ((1.7e-3, 0.4e-3, 'sano'), (1.5e-3, 0.8e-3, 'demiel')):
        for err in (0, 4, 8, 12):
            e, fl = [], []
            for _ in range(150):
                u = rdir(); v = tilt(u, 90)
                S = isoS() + 0.3 * (fib(u, ad, rd) + fib(v, ad, rd))
                raw = np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2); raw /= raw[b0].mean()
                dirs = np.stack([tilt(u, err), tilt(v, err)]) if err else np.stack([u, v])
                w = np.zeros(2 + len(iso_grid)); r = np.zeros(2)
                crossing_varpro_ad_shared_rd_sep(raw.astype(np.float64), bvals, bvecs, dirs, iso_fw, adg, rdg, 1.1, ad, w, r)
                e += list(r - rd); fl += list(r <= _TENSOR_RD_FLOOR * 1.0001)
            print(f'SNR {SNR} {lab:6s} errore direzione {err:2d} gradi: RD bias {np.mean(e)*1e3:+.3f}  pavimento {np.mean(fl):.2f}')
