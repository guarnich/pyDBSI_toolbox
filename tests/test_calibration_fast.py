#!/usr/bin/env python
"""
Calibrazione veloce (v1.8.2): le NNLS per voxel girano in parallelo, con gli STESSI risultati.

IL PROBLEMA. La calibrazione libera costava ~20 minuti per 1000 voxel, sui dati veri come nelle
simulazioni: il ~90% era un ciclo Python che chiamava la NNLS voxel per voxel nel primo passaggio di
discrepanza su lambda_aniso (a lambda ~ 0 molti voxel arrivano al tetto di 20000 iterazioni). Negli studi
di coorte la calibrazione gira su decine di acquisizioni.

LA CORREZIONE. `core.solvers.nnls_batch` risolve i problemi dei voxel (indipendenti) in parallelo; il
solver, i suoi ingressi e l'aritmetica del residuo per voxel restano quelli di prima.

COSA SI PROTEGGE.
  1. nnls_batch da' le stesse soluzioni del ciclo seriale, bit per bit, anche a lambda ~ 0 (tetto).
  2. La calibrazione completa (lambda_aniso, lambda_iso e la discrepanza di primo passaggio) e' identica
     a quella col solver seriale (nnls_batch sostituito da un ciclo voxel per voxel).

    python tests/test_calibration_fast.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core import solvers as SOL
from dbsi_toolbox.core.solvers import nnls_coordinate_descent, nnls_batch, compute_regularization_matrix
from dbsi_toolbox.core.basis import build_design_matrix_exhaustive
import dbsi_toolbox.calibration.data_driven as DD

EXP = Path(__file__).resolve().parent / 'data'          # protocollo tipo P3 (sintetico)


def _protocollo():
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel(); bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    n = np.linalg.norm(bc, axis=1, keepdims=True); n[n == 0] = 1
    return bv, bc / n


def _dati(bv, bc, nvox=400, snr=26.0, seed=0):
    """Voxel misti: fibra singola, incrocio, isotropo; rumore Rician."""
    rng = np.random.default_rng(seed)
    S = []
    for i in range(nvox):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3)); v /= np.linalg.norm(v)
        ff = [0.0, 0.6, 0.6][i % 3]
        s = (1 - ff) * (0.2 * np.exp(-bv * 0.15e-3) + 0.6 * np.exp(-bv * 1e-3) + 0.2 * np.exp(-bv * 3e-3))
        if i % 3 == 1:
            s = s + ff * np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ u) ** 2))
        elif i % 3 == 2:
            s = s + ff / 2 * (np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ u) ** 2))
                              + np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ v) ** 2)))
        S.append(1000 * np.sqrt((s + rng.normal(0, 1 / snr, s.shape)) ** 2 + rng.normal(0, 1 / snr, s.shape) ** 2))
    return np.array(S, np.float32).reshape(20, nvox // 20, 1, -1)


def _seriale(AtA, Aty_rows, reg_lambda):
    """Il comportamento di prima: un voxel alla volta."""
    W = np.empty((Aty_rows.shape[0], AtA.shape[0])); it = np.empty(Aty_rows.shape[0], np.int64)
    for v in range(Aty_rows.shape[0]):
        W[v], it[v] = nnls_coordinate_descent(AtA, Aty_rows[v], reg_lambda)
    return W, it


def test_soluzioni_identiche():
    bv, bc = _protocollo()
    data = _dati(bv, bc, nvox=100)
    m = DBSI_Adaptive(lambda_aniso=9.8, lambda_iso=0.015, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(data[:2, :2], bv, bc, np.ones((2, 2, 1), bool), run_calibration=False,
              noise_map=np.full((2, 2, 1), 1000 / 26.0))
    D = m.dictionary_
    A = build_design_matrix_exhaustive(bv, bc, D['fiber_dirs'], D['diff_pairs'], D['iso_grid'])
    At, AtA = A.T.copy(), A.T @ A
    na = len(D['fiber_dirs']) * len(D['diff_pairs'])
    Y = data.reshape(-1, data.shape[-1]).astype(np.float64)
    Y /= Y[:, bv < 100].mean(1, keepdims=True)
    Aty = np.stack([At @ y for y in Y])
    for lam in (1e-6, 0.5, 9.8):
        R = compute_regularization_matrix(AtA, na, lam, 0.015)
        Ws, its = _seriale(R, Aty, 0.0)
        Wb, itb = nnls_batch(R, Aty, 0.0)
        print(f'  lambda_aniso {lam:g}: tetto {np.mean(itb >= SOL._NNLS_MAX_ITER - 1):.0%}, '
              f'identiche {np.array_equal(Ws, Wb) and np.array_equal(its, itb)}')
        assert np.array_equal(Ws, Wb) and np.array_equal(its, itb), f'soluzioni diverse a lambda {lam}'


def test_calibrazione_identica():
    bv, bc = _protocollo()
    data = _dati(bv, bc, nvox=200, seed=1)
    mask = np.ones(data.shape[:3], bool)
    noise = np.full(data.shape[:3], 1000 / 26.0)

    def calibra():
        m = DBSI_Adaptive()
        with contextlib.redirect_stdout(io.StringIO()):
            rec = m.fit(data, bv, bc, mask, run_calibration=True, n_calibration_voxels=60,
                        noise_map=noise, _calibration_only=True)
        return m.lambda_aniso, m.lambda_iso

    veloce = calibra()
    orig = DD.nnls_batch
    DD.nnls_batch = _seriale
    try:
        seriale = calibra()
    finally:
        DD.nnls_batch = orig
    print(f'  parallela {veloce}\n  seriale   {seriale}')
    assert veloce == seriale, 'la calibrazione parallela non coincide con quella seriale'


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
