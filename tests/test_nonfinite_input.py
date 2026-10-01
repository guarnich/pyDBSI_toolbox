#!/usr/bin/env python
"""
Voxel con valori non finiti nel segnale (v1.6.2).

IL DIFETTO (dati veri, notebook 09, 2026-10-01). Un voxel con inf nel segnale (tipico ai bordi
dopo N4 con campo di bias ~0) non resta NaN: la NNLS a discesa coordinata trasforma i NaN in pesi
qualsiasi, il voxel puo' ricevere una fibra, e il test di rilevamento divide per
(sigma / S0)^2 = (sigma / inf)^2 = 0 -> ZeroDivisionError, e il fit intero si ferma.

COSA SI PROTEGGE.
  1. Con un inf e un NaN in due voxel il fit arriva in fondo.
  2. Quei voxel sono contati nel run report (input.nonfinite_voxels_excluded).
  3. L'uscita e' IDENTICA, voxel per voxel, a quella di un fit con quei due voxel tolti a mano dalla
     maschera: gli esclusi valgono come fuori maschera, e gli altri non cambiano.

    python tests/test_nonfinite_input.py
"""
import io, sys, contextlib
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

LAM = dict(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
NOMI = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NOMI)}


def _dati(seed=2, n=30, snr=25):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * 6; bvecs = [np.zeros(3)] * 6
    for b, k in ((1000., 20), (2000., 30)):
        for _ in range(k):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v); bvals.append(b)
    bvals, bvecs = np.array(bvals), np.vstack(bvecs)
    d = np.zeros((n, 2, 1, len(bvals)), np.float32)
    for i in range(n):
        for j in range(2):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            S = (0.1 * np.exp(-bvals * 0.15e-3) + 0.3 * np.exp(-bvals * 1e-3) + 0.1 * np.exp(-bvals * 3e-3)
                 + 0.5 * np.exp(-bvals * (0.4e-3 + 1.3e-3 * (bvecs @ u) ** 2)))
            d[i, j, 0] = 1000 * np.abs(S + rng.normal(0, 1 / snr, S.shape))
    return d, bvals, bvecs


def _fit(d, bvals, bvecs, mask):
    m = DBSI_Adaptive(**LAM)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(d, bvals, bvecs, mask, run_calibration=False)
    return m, res


def test_voxel_non_finiti_esclusi():
    d, bvals, bvecs = _dati()
    d[3, 0, 0, 0] = np.inf          # un b=0 infinito
    d[7, 1, 0, 40] = np.nan         # un NaN in una misura pesata
    mask = np.ones(d.shape[:3], bool)
    m, res = _fit(d, bvals, bvecs, mask)          # prima della 1.6.2: ZeroDivisionError
    n = m.run_report_['input']['nonfinite_voxels_excluded']
    print(f"  fit completato; voxel esclusi {n}")
    assert n == 2
    mask2 = mask.copy(); mask2[3, 0, 0] = False; mask2[7, 1, 0] = False
    d2 = d.copy(); d2[3, 0, 0, 0] = 1000.0; d2[7, 1, 0, 40] = 1000.0   # valori qualsiasi, fuori maschera
    _, res2 = _fit(d2, bvals, bvecs, mask2)
    assert np.array_equal(np.isnan(res), np.isnan(res2)) and np.allclose(res, res2, equal_nan=True, rtol=0, atol=0), \
        'uscita diversa dal fit con la maschera ridotta a mano'
    print('  uscita identica al fit con la maschera ridotta a mano')


def test_rilevamento_non_divide_per_zero():
    """Il test di rilevamento, chiamato direttamente su un voxel con fibra e S0 infinito: prima della
    1.6.2 ZeroDivisionError (il SystemError visto nel notebook 09), ora statistica NaN."""
    import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
    d, bvals, bvecs = _dati(n=1)
    dc = d[:1, :1].astype(np.float32).copy(); dc[0, 0, 0, 0] = np.inf
    out = np.full((1, 1, 1, M._N_CHANNELS), np.nan, np.float32)
    out[0, 0, 0, M._C_NPOP] = 1; out[0, 0, 0, M._C_AD1] = 1.7e-3; out[0, 0, 0, M._C_RD1] = 0.4e-3
    out[0, 0, 0, M._C_DIR1:M._C_DIR1 + 3] = (1.0, 0.0, 0.0); out[0, 0, 0, M._C_FF] = 0.5
    stat = np.full((1, 1, 1), np.nan, np.float32)
    M._fiber_detection_pass(dc, np.array([[0, 0, 0]]), bvals, bvecs, 100.0,
                            np.array(M._ISO_RESOLVE_D_3ISO), True, 30.0, 0.0, out, stat)
    print(f"  S0 infinito: statistica {stat[0, 0, 0]} (nessuna eccezione)")
    assert np.isnan(stat[0, 0, 0])


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
