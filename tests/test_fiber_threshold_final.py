#!/usr/bin/env python
"""
Soglia di fibra sullo STATO FINALE (v1.8.0).

IL PROBLEMA (ispezione 2026-10-08, 1.7.1). `fiber_threshold` decideva in Stage A, sulla FF di Stage A,
se un voxel ha una fibra; la FF riportata e' invece quella di Stage D. Sul fantoccio P3 a SNR 26 il 38%
delle fibre (false) nel tessuto isotropo e il 17% delle fibre vere deboli (FF 0.2) uscivano con un
tensore e una FF finale < 0.15, mentre il log dichiarava "AD/RD/FA valid only where FF > 0.15".

COSA SI PROTEGGE.
  1. Nessun voxel in uscita ha n_pop >= 1 con FF <= fiber_threshold, con Stage D e senza.
  2. Con Stage D, un voxel che perde la fibra e' trattato come ogni voxel senza fibra: FF = 0, frazioni
     isotrope che sommano a 1, canali del tensore NaN. Il run report conta i voxel tolti (> 0 qui:
     sulla 1.7.1 il test 1 fallisce).
  3. Le fibre robuste (mono-fibra FF 0.6) non sono toccate.

    python tests/test_fiber_threshold_final.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

NOMI = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NOMI)}
LAM = dict(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
EXP = Path(__file__).resolve().parents[1] / 'experiments' / 'uncertainty'
SNR, NV = 26.0, 60
TENSORE = ('n_fiber_populations', 'axial_diffusivity_pop1', 'radial_diffusivity_pop1',
           'fiber_fa_pop1', 'axial_diffusivity_weighted')


def _dati(seed=11):
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel(); bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb
    t, p = 1.1, 0.4
    u = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    iso = lambda fr, fh, fw: fr * np.exp(-bv * 0.15e-3) + fh * np.exp(-bv * 1e-3) + fw * np.exp(-bv * 3e-3)
    fib = lambda f, ad, rd: f * np.exp(-bv * (rd + (ad - rd) * (bc @ u) ** 2))
    scen = [iso(0.15, 0.7, 0.15),                                  # isotropo GM
            iso(0.5, 0.4, 0.1),                                    # tumore
            iso(0.15, 0.5, 0.15) + fib(0.2, 1.7e-3, 0.5e-3),       # fibra debole
            iso(0.1, 0.2, 0.1) + fib(0.6, 1.7e-3, 0.4e-3)]         # mono-fibra sana
    rng = np.random.default_rng(seed); s = 1.0 / SNR
    d = np.stack([1000 * np.sqrt((S + rng.normal(0, s, (NV, bv.size))) ** 2
                                 + rng.normal(0, s, (NV, bv.size)) ** 2) for S in scen])
    return d[:, :, None, :].astype(np.float32), bv, bc


def _fit(**kw):
    d, bv, bc = _dati()
    m = DBSI_Adaptive(**LAM, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        r, _ = m.fit(d, bv, bc, np.ones(d.shape[:3], bool), run_calibration=False,
                     noise_map=np.full(d.shape[:3], 1000.0 / SNR))
    return m, r


def _controlla(m, r, nome):
    npop = r[..., IX['n_fiber_populations']]; ff = r[..., IX['fiber_fraction']]
    sotto = np.isfinite(npop) & (npop >= 1) & ~(ff > m.fiber_threshold)
    print(f"  {nome}: voxel con fibra e FF finale <= {m.fiber_threshold}: {int(sotto.sum())} "
          f"di {int(np.sum(np.isfinite(npop) & (npop >= 1)))}")
    assert not sotto.any(), 'tensore riportato con FF finale sotto la soglia'
    rep = m.run_report_['fiber_threshold_final']
    print(f"  run report: tolti {rep['n_removed']} di {rep['n_fiber_voxels_before']}")
    assert rep['n_removed'] > 0, 'il fantoccio doveva produrre fibre sotto soglia'
    return npop, ff


def test_con_stage_d():
    m, r = _fit()
    npop, ff = _controlla(m, r, 'Stage D')
    nofib = ~np.isfinite(npop)
    for n in TENSORE[1:]:
        assert np.all(np.isnan(r[..., IX[n]][nofib])), f'{n} non NaN dove non c e fibra'
    assert np.all(ff[nofib] == 0), 'voxel senza fibra con FF != 0 (Stage D)'
    tot = sum(r[..., IX[n]] for n in ('fiber_fraction', 'restricted_fraction',
                                      'hindered_fraction', 'water_fraction'))
    assert np.allclose(tot, 1, atol=1e-4), 'frazioni che non sommano a 1'
    sana = npop[3, :, 0]
    print(f"  mono-fibra sana: con fibra {np.mean(np.isfinite(sana)):.2f}")
    assert np.all(np.isfinite(sana)), 'la soglia ha tolto fibre robuste'


def test_senza_stage_d():
    m, r = _fit(iso_resolve=False, uncertainty=False)
    _controlla(m, r, 'senza Stage D')


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
