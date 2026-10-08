#!/usr/bin/env python
"""
Sigma di calibrazione dalla mappa di rumore: RMS, non mediana (v1.8.0).

IL PROBLEMA (ispezione 2026-10-08). Il tetto di lambda_iso (e il lambda_aniso di primo passaggio)
confronta la MEDIA sui voxel campionati del residuo^2 con N * sigma^2. La parte di rumore di quella
media e' N * media(sigma_v^2 / S0_v^2). Fino alla 1.7.1 sigma era la MEDIANA di sigma_v / S0_v: con
sigma/S0 eterogeneo (su un DWI vero la media dei quadrati e' 3.5 volte la mediana al quadrato) il
bersaglio era troppo piccolo, nessun lambda lo rispettava e il tetto cadeva al minimo della griglia.

COSA SI PROTEGGE.
  1. sigma_normalised^2 == media di (noise_map / S0)^2 sui voxel campionati.
  2. Su un campione con sigma/S0 eterogeneo ((RMS/mediana)^2 ~ 1.7) il tetto di lambda_iso NON cade al
     minimo della griglia e non sta sotto la scelta della GCV.

    python tests/test_calibration_sigma_rms.py
"""
import sys
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox.calibration.data_driven import (sample_calibration_voxels, select_lambda_iso_gcv,
                                                  lambda_iso_discrepancy_cap)
from dbsi_toolbox.core.basis import generate_anchored_isotropic_grid

EXP = Path(__file__).resolve().parents[1] / 'experiments' / 'uncertainty'


def _volume(nv=1000, seed=0):
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel()
    rng = np.random.default_rng(seed)
    rel = 0.03 * np.exp(rng.normal(0, 0.5, nv))                 # sigma_v / S0_v
    fr = rng.dirichlet([2, 5, 2], nv)
    S = fr[:, :1] * np.exp(-bv * 0.15e-3) + fr[:, 1:2] * np.exp(-bv * 1e-3) + fr[:, 2:] * np.exp(-bv * 3e-3)
    s0 = 1000.0
    d = s0 * S + rng.normal(0, 1, S.shape) * (rel * s0)[:, None]
    return d[:, None, None, :], bv, (rel * s0)[:, None, None]


def test_sigma_e_rms():
    d, bv, nm = _volume()
    y, sig = sample_calibration_voxels(d, np.ones(d.shape[:3], bool), bv, n_voxels=1000, seed=0,
                                       noise_map=nm)
    s0 = d[..., bv < 100].mean(-1)[..., 0, 0]
    rms = np.sqrt(np.mean((nm[:, 0, 0] / s0) ** 2)); med = np.median(nm[:, 0, 0] / s0)
    print(f"  sigma_normalised {sig:.5f}; RMS {rms:.5f}; mediana {med:.5f}")
    assert abs(sig - rms) < 1e-9 * rms, 'sigma_normalised non e la RMS di sigma/S0'


def test_tetto_non_crolla():
    d, bv, nm = _volume()
    y, sig = sample_calibration_voxels(d, np.ones(d.shape[:3], bool), bv, n_voxels=1000, seed=0,
                                       noise_map=nm)
    grid = generate_anchored_isotropic_grid(d_min=0.1e-3, d_max=5e-3, n_steps=6,
                                            thresh_res=0.3e-3, thresh_wat=3e-3)
    lg = np.logspace(-5, 1, 40)
    gcv, _ = select_lambda_iso_gcv(bv, grid, y, lambda_grid=lg)
    cap = lambda_iso_discrepancy_cap(bv, grid, y, sig, lg)
    print(f"  lambda_iso GCV {gcv:.3g}, tetto {cap:.3g} (minimo griglia {lg[0]:g})")
    assert cap > lg[0] and cap >= gcv, 'il tetto di lambda_iso e crollato sotto la GCV'


if __name__ == '__main__':
    print(f"dbsi_toolbox {dbsi_toolbox.__version__}")
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
