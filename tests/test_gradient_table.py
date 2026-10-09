#!/usr/bin/env python
"""
Tabella dei gradienti coerente con i dati (v1.8.0).

IL PROBLEMA (ispezione 2026-10-09). fit() non confrontava il numero di bval/bvec con il numero di
volumi. Con un bvec in MENO dei volumi il fit girava fino in fondo senza errori e restituiva mappe
senza senso (FF = 0 in ogni voxel): i kernel numba non controllano i limiti degli array e leggevano
memoria fuori dalla tabella. Con un bval in meno si fermava su un IndexError di numpy che non diceva
quale fosse il problema.

COSA SI PROTEGGE.
  1. bval o bvec con un volume in meno (o in piu'): ValueError che nomina la tabella dei gradienti.
  2. Un volume pesato (b >= 100) con bvec nullo: ValueError.
  3. bvec in formato FSL (3 x N) e bval come riga (1 x N) restano accettati.

    python tests/test_gradient_table.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

EXP = Path(__file__).resolve().parents[1] / 'experiments' / 'uncertainty'
LAM = dict(lambda_aniso=9.8, lambda_iso=0.015, n_iso=6)


def _dati():
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel(); bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    rng = np.random.default_rng(0)
    d = np.abs(1000 * np.exp(-bv * 1e-3) + rng.normal(0, 30, (4, 1, 1, bv.size))).astype(np.float32)
    return d, bv, bc


def _fit(d, bv, bc):
    m = DBSI_Adaptive(**LAM)
    with contextlib.redirect_stdout(io.StringIO()):
        return m.fit(d, bv, bc, np.ones(d.shape[:3], bool), run_calibration=False,
                     noise_map=np.full(d.shape[:3], 30.0))


def _deve_fermarsi(nome, d, bv, bc):
    try:
        _fit(d, bv, bc)
    except ValueError as e:
        print(f"  {nome}: fermato ({str(e)[:70]}...)")
        assert 'gradient' in str(e) or 'bvec' in str(e), 'messaggio che non nomina la tabella'
        return
    raise AssertionError(f'{nome}: il fit non si e fermato')


def test_lunghezze_sbagliate():
    d, bv, bc = _dati()
    _deve_fermarsi('bvec con un volume in meno', d, bv, bc[:-1])
    _deve_fermarsi('bval con un volume in meno', d, bv[:-1], bc)
    _deve_fermarsi('bval e bvec con un volume in meno', d, bv[:-1], bc[:-1])


def test_bvec_nullo_su_volume_pesato():
    d, bv, bc = _dati()
    bc = bc.copy(); bc[np.where(bv >= 100)[0][3]] = 0.0
    _deve_fermarsi('bvec nullo su un volume pesato', d, bv, bc)


def test_formati_fsl_accettati():
    d, bv, bc = _dati()
    r, _ = _fit(d, bv[None, :], bc.T)
    print(f"  bval (1 x N) e bvec (3 x N): fit fatto, {r.shape[-1]} canali")
    assert r.shape[-1] == 27


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
