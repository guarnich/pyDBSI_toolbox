#!/usr/bin/env python
"""
Test di rilevamento della fibra dopo Stage D (v1.6.1, spento per default).

IL PROBLEMA. Sul tessuto ISOTROPO sintetico (protocollo P3, lambda congelati del
P3, gate spento) a SNR 26 il 38-62% dei voxel tipo GM / hindered / tumore riceve
una fibra, quasi sempre come CROSSING con FF 0.16-0.18: il rumore di Stage A
spalmato su due picchi, appena sopra fiber_threshold. A SNR 15 l'88-90%. Il gate
di concentrazione non li separa dai crossing veri.

LA STATISTICA. dRSS/sigma^2 fra "solo iso" e "fibre + iso" sul modello di Stage D:
nulla invariante con l'SNR (mediana 4-6, p95 11-14), fibre vere p05 74 (mono FF
0.25) e 17.5 (crossing a 90 gradi FF 0.30) a SNR 26.

COSA SI PROTEGGE.
  1. Spento (default): la statistica c'e' su ogni voxel con fibra, nulla e'
     respinto, il report la riassume.
  2. Soglia 15 su tessuto isotropo a SNR 20: le fibre false spariscono; quelle
     vere (mono FF 0.40, crossing FF 0.60) restano.
  3. Un voxel respinto e' coerente: FF 0, n_pop NaN, tensore NaN, frazioni che
     sommano a 1, fit_rmse calcolato.
  4. Rifiuti: soglia <= 0, e soglia senza Stage D.

    python tests/test_fiber_detection.py
"""
import io, sys, contextlib
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

NOMI = DBSI_Adaptive.output_map_names(3)
IX = {n: i for i, n in enumerate(NOMI)}
LAM = dict(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)


def _protocollo(seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * 9; bvecs = [np.zeros(3)] * 9
    for b, n in ((500., 12), (1000., 20), (1500., 20), (2000., 30)):
        for _ in range(n):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v); bvals.append(b)
    return np.array(bvals), np.vstack(bvecs)


def _dati(snr=20, n=100, seed=1):
    """Tre gruppi in fila: isotropo tipo WM (0), mono-fibra FF 0.40 (1), crossing 90 FF 0.60 (2)."""
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(seed)
    iso = lambda s: s * (0.25 * np.exp(-bvals * 0.15e-3) + 0.62 * np.exp(-bvals * 0.8e-3)
                         + 0.13 * np.exp(-bvals * 3.0e-3))
    fib = lambda d: np.exp(-bvals * (0.3e-3 + 1.4e-3 * (bvecs @ d) ** 2))
    d = np.zeros((3, n, 1, len(bvals)), np.float32)
    for g in range(3):
        for i in range(n):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            S = (iso(1.0) if g == 0 else iso(0.60) + 0.40 * fib(u) if g == 1
                 else iso(0.40) + 0.30 * (fib(u) + fib(t)))
            d[g, i, 0] = 1000 * np.sqrt((S + rng.normal(0, 1 / snr, S.shape)) ** 2
                                        + rng.normal(0, 1 / snr, S.shape) ** 2)
    return d, bvals, bvecs


def _fit(d, bvals, bvecs, **kw):
    m = DBSI_Adaptive(**LAM, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(d, bvals, bvecs, np.ones(d.shape[:3], bool), run_calibration=False)
    return m, res


def test_spento_non_toglie_niente():
    d, bvals, bvecs = _dati()
    m, res = _fit(d, bvals, bvecs)
    fib = ~np.isnan(res[..., IX['n_fiber_populations']])
    st = m.fiber_detection_stat_
    fd = m.run_report_['fiber_detection']
    print(f"  voxel con fibra {fib.sum()}, statistica finita {np.isfinite(st).sum()}, "
          f"respinti {fd['n_rejected']}, p50 {fd['stat_p50']:.1f}")
    assert np.array_equal(np.isfinite(st), fib), 'statistica non definita esattamente sui voxel con fibra'
    assert fd['n_rejected'] == 0 and fd['threshold'] is None
    assert np.mean(fib[0]) > 0.2, 'il fantoccio non riproduce le fibre false da togliere'


def test_soglia_15():
    d, bvals, bvecs = _dati()
    _, r0 = _fit(d, bvals, bvecs)
    m, r1 = _fit(d, bvals, bvecs, fiber_detection_threshold=15.0)
    has = lambda r, g: np.mean(~np.isnan(r[g, :, 0, IX['n_fiber_populations']]))
    for g, nome in ((0, 'isotropo'), (1, 'mono FF 0.40'), (2, 'crossing FF 0.60')):
        print(f'  {nome:18s} con fibra: spento {has(r0, g):.2f}  soglia 15 {has(r1, g):.2f}')
    assert has(r1, 0) <= 0.05, 'fibre false non tolte dal tessuto isotropo'
    assert has(r1, 1) == 1.0, 'mono-fibre vere tolte'
    assert has(r1, 2) >= 0.95, 'crossing veri tolti'
    # coerenza dei respinti
    rej = np.isnan(r1[..., IX['n_fiber_populations']]) & ~np.isnan(r0[..., IX['n_fiber_populations']])
    R = r1[rej]
    tot = R[:, IX['fiber_fraction']] + R[:, IX['restricted_fraction']] + R[:, IX['hindered_fraction']] \
        + R[:, IX['water_fraction']]
    print(f'  respinti {rej.sum()}: FF max {np.nanmax(R[:, IX["fiber_fraction"]]):.2f}, '
          f'somma frazioni {tot.min():.4f}-{tot.max():.4f}')
    assert np.all(R[:, IX['fiber_fraction']] == 0)
    assert np.all(np.isnan(R[:, IX['axial_diffusivity_pop1']])) and np.all(np.isnan(R[:, IX['dir2_x']]))
    assert np.allclose(tot, 1.0, atol=1e-5)
    assert np.all(np.isfinite(R[:, IX['fit_rmse']]))


def test_rifiuti():
    for kw in (dict(fiber_detection_threshold=0.0), dict(fiber_detection_threshold=-3.0)):
        try:
            DBSI_Adaptive(**kw)
        except ValueError as e:
            print(f'  rifiutato {kw}: {e}'); continue
        raise AssertionError(f'{kw} accettato')
    d, bvals, bvecs = _dati(n=5)
    try:
        _fit(d, bvals, bvecs, fiber_detection_threshold=15.0, iso_resolve=False)
    except ValueError as e:
        print(f'  rifiutato senza Stage D: {e}'); return
    raise AssertionError('soglia accettata con iso_resolve=False')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_rifiuti, test_spento_non_toglie_niente, test_soglia_15):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
