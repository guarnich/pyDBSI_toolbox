#!/usr/bin/env python
"""
La verifica Monte Carlo SURE giudica lambda_iso sul segnale giusto (v1.6.1).

IL DIFETTO. `fit(run_sure_crosscheck=True)` passava alle due verifiche SURE il
segnale GREZZO dei voxel, con un dizionario solo isotropo. lambda_iso pero' si
sceglie sul residuo con la fibra sottratta, proprio perche' sul grezzo la fibra
(che il blocco isotropo non puo' rappresentare) domina ogni criterio. Misurato su
un fantoccio P3: sul grezzo il rischio variava dell'1.6% su un intervallo di
lambda di 16 volte, quindi con la tolleranza del 15% la verifica non poteva MAI
dare disaccordo; sul residuo varia del 73%. Stesso difetto che il bootstrap di
n_iso aveva fino alla 1.3.6.

Secondo difetto, nel criterio: `candidato <= minimo * 1.15` con un minimo
NEGATIVO (una stima SURE puo' esserlo) chiede al candidato di battere il minimo:
anche il minimo risultava in disaccordo con se' stesso.

COSA SI PROTEGGE.
  1. Nel fit vero, su voxel con fibra, il rischio di entrambe le verifiche sta
     sotto l'energia del rumore (sul grezzo la supera di 7 volte), e il report
     dichiara il segnale usato.
  2. Il criterio di accordo regge con rischi negativi.

    python tests/test_sure_crosscheck_signal.py
"""
import io, sys, contextlib
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive


def _fantoccio(seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * 3; bvecs = [np.zeros(3)] * 3
    for b, n in ((500., 12), (1000., 20), (2000., 30)):
        for _ in range(n):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v); bvals.append(b)
    bvals = np.array(bvals); bvecs = np.vstack(bvecs)
    d = np.zeros((6, 6, 2, len(bvals)), np.float32)
    for idx in np.ndindex(d.shape[:3]):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        S = (0.55 * np.exp(-bvals * (0.3e-3 + 1.4e-3 * (bvecs @ u) ** 2))
             + 0.10 * np.exp(-bvals * 0.15e-3) + 0.25 * np.exp(-bvals * 1.0e-3)
             + 0.10 * np.exp(-bvals * 3.0e-3))
        sd = 1 / 30
        d[idx] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
    return d, bvals, bvecs


def test_rischio_informativo_nel_fit():
    d, bvals, bvecs = _fantoccio()
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(d, bvals, bvecs, np.ones(d.shape[:3], bool), run_calibration=False,
              run_sure_crosscheck=True, sure_crosscheck_n_probes=5, n_calibration_voxels=72)
    rep = m.sure_crosscheck_report_
    # Il rischio SURE stima l'errore di predizione: su cio' che il blocco isotropo
    # puo' spiegare vale ~ gradi di liberta' * sigma^2, ben sotto l'energia del
    # rumore N*sigma^2. Sul grezzo ci si somma la fibra non rappresentabile
    # (misurato: 0.54 contro 0.006, con N*sigma^2 = 0.072).
    rumore = len(bvals) * (1 / 30) ** 2
    for k in ('lambda_iso', 'n_iso'):
        rmin = float(np.min(rep[k]['risks']))
        print(f'  rischio minimo {k:10s} {rmin:.4f}   (N*sigma^2 = {rumore:.4f})')
        assert rmin < rumore, (f'{k}: rischio sopra l energia del rumore -- la verifica sta '
                               f'giudicando il segnale grezzo, dominato dalla fibra')
    print(f'  segnale dichiarato: {rep.get("signal")}')
    assert rep.get('signal') == 'fiber_subtracted_residual'


def test_tolleranza_con_rischi_negativi():
    from dbsi_toolbox.calibration.mc_sure import _within_flat_tolerance
    casi = [((-0.010, -0.010, 0.15), True),     # il minimo stesso
            ((-0.009, -0.010, 0.15), True),     # entro il 15% di |min|
            ((-0.008, -0.010, 0.15), False),    # oltre
            ((0.110, 0.100, 0.15), True),
            ((0.120, 0.100, 0.15), False)]
    for (c, mn, tol), atteso in casi:
        got = _within_flat_tolerance(c, mn, tol)
        print(f'  candidato {c:+.3f} minimo {mn:+.3f}: {got}')
        assert got is atteso, f'({c}, {mn}) -> {got}, atteso {atteso}'


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 6, 1), f'attesa >= 1.6.1, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_tolleranza_con_rischi_negativi, test_rischio_informativo_nel_fit):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
