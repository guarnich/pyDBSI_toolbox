#!/usr/bin/env python
"""
Gate di concentrazione spento di default + calibrate() per una acquisizione (v1.4.0).

COSA SI PROTEGGE.
  1. GATE. Senza argomenti il modello non calibra il gate (il nullo MC non viene
     chiamato) e il gate applicato e' 0.0. Motivo: sui dati veri il gate toglieva
     la fibra al 19.5% dei voxel, che rifiutati lasciavano 1.80 sigma di residuo e
     ammessi 1.03 (notebook 07). Il gate resta disponibile su richiesta.
  2. calibrate(). Si ferma prima del fit voxel per voxel (il kernel NON viene
     chiamato), non modifica il modello su cui e' chiamato, e restituisce un
     record serializzabile in JSON con le curve GCV complete e l'impronta.
  3. Le griglie fini passano davvero ai selettori.
  4. L'impronta ignora i vettori e distingue i protocolli.

    python tests/test_gate_off_and_calibrate.py
"""
import io, json, contextlib, sys
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M


class _Sentinella(Exception):
    pass


def _protocollo(n_b0=2, shells=(500., 1000., 2000., 3000.), per_shell=8, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * n_b0
    bvecs = [np.zeros(3)] * n_b0
    for b in shells:
        for _ in range(per_shell):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            bvecs.append(v); bvals.append(float(b))
    return np.array(bvals), np.vstack(bvecs)


def _dati(bvals, seed=1):
    rng = np.random.default_rng(seed)
    S = 0.6 * np.exp(-bvals * 0.8e-3) + 0.4 * np.exp(-bvals * 2.5e-3)
    data = np.abs(1000 * S + rng.normal(0, 25, (5, 5, 2, len(bvals)))).astype(np.float32)
    return data, np.ones(data.shape[:3], bool)


def _spie(nomi):
    chiamati, orig = [], {n: getattr(M, n) for n in nomi}
    def spia(n, f):
        def _(*a, **k):
            chiamati.append(n); return f(*a, **k)
        return _
    for n, f in orig.items():
        setattr(M, n, spia(n, f))
    return chiamati, orig


def _ripristina(orig):
    for n, f in orig.items():
        setattr(M, n, f)


def test_gate_spento_di_default():
    bvals, bvecs = _protocollo(); data, mask = _dati(bvals)
    m = DBSI_Adaptive()
    assert m.min_dominant_concentration == 0.0, (
        f'gate di default {m.min_dominant_concentration}, atteso 0.0')
    chiamati, orig = _spie(['calibrate_concentration_gate_mc'])
    orig_k = {n: getattr(M, n) for n in ('_fit_voxels_3iso_v3', '_fit_voxels_2iso_v3')}
    def stop(*a, **k): raise _Sentinella()
    try:
        for n in orig_k: setattr(M, n, stop)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(data, bvals, bvecs, mask, n_calibration_voxels=40)
    except _Sentinella:
        pass
    finally:
        _ripristina(orig); _ripristina(orig_k)
    print(f'  gate applicato {m.min_dominant_concentration}  nullo MC chiamato: '
          f'{"si" if chiamati else "no"}')
    assert not chiamati, 'il nullo MC del gate e partito senza essere chiesto'
    assert m.min_dominant_concentration == 0.0


def test_gate_su_richiesta():
    bvals, bvecs = _protocollo(); data, mask = _dati(bvals)
    chiamati, orig = _spie(['calibrate_concentration_gate_mc'])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rec = DBSI_Adaptive().calibrate(data, bvals, bvecs, mask,
                                            n_calibration_voxels=40,
                                            calibrate_concentration_gate=True)
    finally:
        _ripristina(orig)
    print(f'  calibrate_concentration_gate=True -> nullo MC chiamato, gate '
          f'{rec["hyperparameters"]["concentration_gate"]:.3f}')
    assert chiamati, 'con calibrate_concentration_gate=True il nullo MC deve girare'
    assert rec['hyperparameters']['concentration_gate_calibrated'] is True


def test_calibrate_record():
    bvals, bvecs = _protocollo(); data, mask = _dati(bvals)
    m = DBSI_Adaptive()
    chiamati, orig = _spie(['_fit_voxels_3iso_v3', '_fit_voxels_2iso_v3'])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rec = m.calibrate(data, bvals, bvecs, mask, n_calibration_voxels=40)
    finally:
        _ripristina(orig)
    assert not chiamati, f'calibrate() ha lanciato il fit voxel per voxel: {chiamati}'
    assert m.lambda_aniso is None and m.lambda_iso is None and m.n_iso is None, (
        'calibrate() ha modificato il modello su cui e stato chiamato')
    json.dumps(rec)                                   # deve essere serializzabile
    h, c = rec['hyperparameters'], rec['curves']
    ga, gi = c['lambda_aniso']['lambda_grid'], c['lambda_iso']['lambda_grid']
    print(f'  lambda_aniso {h["lambda_aniso"]:.4g} (curva {len(ga)} punti)  '
          f'lambda_iso {h["lambda_iso"]:.4g} (curva {len(gi)} punti)  n_iso {h["n_iso"]} '
          f'({h["n_iso_source"]})  impronta {rec["protocol"]["shells"]}')
    assert h['n_iso'] == 6 and h['n_iso_source'] == 'quadrature'
    assert len(ga) == len(c['lambda_aniso']['gcv']) == 40
    assert len(gi) == len(c['lambda_iso']['gcv']) == 40
    assert h['lambda_aniso'] in ga, 'il lambda_aniso scelto non sta sulla sua curva'
    assert h['lambda_iso'] <= rec['selection']['lambda_iso_gcv'] + 1e-15
    assert rec['protocol']['n_volumes'] == len(bvals)
    # riusabile: seconda chiamata sullo stesso oggetto, stesso risultato
    with contextlib.redirect_stdout(io.StringIO()):
        rec2 = m.calibrate(data, bvals, bvecs, mask, n_calibration_voxels=40)
    assert rec2['hyperparameters']['lambda_aniso'] == h['lambda_aniso']
    assert rec2['curves']['lambda_aniso']['gcv'] == c['lambda_aniso']['gcv']


def test_calibrate_rifiuta_lambda_imposti():
    bvals, bvecs = _protocollo(); data, mask = _dati(bvals)
    try:
        DBSI_Adaptive(lambda_aniso=8.0).calibrate(data, bvals, bvecs, mask)
    except ValueError as e:
        print(f'  rifiutato: {str(e)[:70]}...'); return
    raise AssertionError('calibrate() accetta un modello con lambda gia imposti')


def test_griglie_fini():
    bvals, bvecs = _protocollo(); data, mask = _dati(bvals)
    ga = np.logspace(-4, 4, 121); gi = np.logspace(-5, 1, 97)
    with contextlib.redirect_stdout(io.StringIO()):
        rec = DBSI_Adaptive().calibrate(data, bvals, bvecs, mask, n_calibration_voxels=40,
                                        lambda_aniso_grid=ga, lambda_iso_grid=gi)
    na = len(rec['curves']['lambda_aniso']['lambda_grid'])
    ni = len(rec['curves']['lambda_iso']['lambda_grid'])
    print(f'  griglie fini: lambda_aniso {na} punti, lambda_iso {ni} punti')
    assert (na, ni) == (121, 97), 'le griglie fini non sono arrivate ai selettori'


def test_impronta():
    from dbsi_toolbox import protocol_fingerprint, fingerprint_mismatches
    bvals, bvecs = _protocollo()
    fp = protocol_fingerprint(bvals)
    # stesso protocollo, bvec diversi e b-values con jitter: stessa impronta
    rng = np.random.default_rng(9)
    b2 = bvals + np.where(bvals > 0, rng.uniform(-20, 20, bvals.size), 0.0)
    assert not fingerprint_mismatches(fp, protocol_fingerprint(b2)), (
        'b-values con jitter di +-20 devono dare la stessa impronta')
    # un guscio in meno: impronta diversa
    b3 = bvals[bvals != 3000.]
    diff = fingerprint_mismatches(fp, protocol_fingerprint(b3))
    print(f'  impronta {fp["shells"]}  |  senza b=3000 differisce su '
          f'{[d[0] for d in diff]}')
    assert diff
    # un b=0 in piu': impronta diversa
    assert fingerprint_mismatches(fp, protocol_fingerprint(np.r_[0.0, bvals]))


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 4, 0), f'attesa >= 1.4.0, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_gate_spento_di_default, test_gate_su_richiesta,
               test_calibrate_record, test_calibrate_rifiuta_lambda_imposti,
               test_griglie_fini, test_impronta):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
