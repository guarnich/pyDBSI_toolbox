#!/usr/bin/env python
"""
Calibrazione di PROTOCOLLO (v1.5.0): aggregare i record di calibrate() e usarli.

COSA SI PROTEGGE.
  1. 'median_index' riproduce la regola pre-registrata di WaterMobility, compreso
     il pareggio 20-19 risolto sul caso peggiore (-> indice 24, 8.3768).
  2. 'geomean' e' invariante per la scala della curva di ogni acquisizione: due
     soggetti con la stessa forma di curva e rumore diverso votano uguale.
  3. Acquisizioni con impronta, opzioni del modello o griglie diverse sono
     RIFIUTATE, dicendo cosa differisce.
  4. Catena completa su fantoccio: calibrate() x3 -> calibrate_protocol ->
     save/load (sha256) -> from_calibration -> fit. Il run report dichiara
     calibration_source 'protocol' e lo sha256; toolbox_report/ contiene la copia
     della calibrazione; un protocollo diverso viene rifiutato prima di fittare.
  5. from_calibration rifiuta di ridefinire un parametro calibrato.

    python tests/test_protocol_calibration.py
"""
import io, os, json, copy, contextlib, sys, tempfile
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

GA = np.logspace(-4, 4, 40)
GI = np.logspace(-5, 1, 40)


def _fake(idx_a, idx_i=20, scale=1.0, snr=25.0, fp=None, n_dirs=27, grid_a=GA):
    """Record minimo di calibrate(), con curve a V centrate sugli indici dati."""
    from dbsi_toolbox import protocol_fingerprint
    ka = np.arange(len(grid_a)); ki = np.arange(len(GI))
    return dict(
        kind='pydbsi_acquisition_calibration', format_version=1, toolbox_version='test',
        protocol=fp or protocol_fingerprint(np.r_[np.zeros(2), np.repeat([1000., 2000.], 8)]),
        noise=dict(snr=snr),
        hyperparameters=dict(n_iso=6, lambda_aniso_method='gcv', concentration_gate=0.0,
                             concentration_gate_calibrated=False,
                             lambda_aniso=float(grid_a[idx_a]), lambda_iso=float(GI[idx_i])),
        selection=dict(lambda_iso_cap=None),
        curves=dict(lambda_aniso=dict(lambda_grid=list(grid_a),
                                      gcv=list(scale * (1 + 0.01 * (ka - idx_a) ** 2))),
                    lambda_iso=dict(lambda_grid=list(GI),
                                    gcv=list(scale * (1 + 0.01 * (ki - idx_i) ** 2)))),
        model=dict(n_dirs=n_dirs, n_ad=3, n_rd=3, anisotropy_ratio=2.0,
                   fiber_threshold=0.15, iso_range=[0.0, 0.003]))


def test_median_index_riproduce_watermobility():
    from dbsi_toolbox import calibrate_protocol
    recs = [_fake(25)] * 20 + [_fake(24)] * 19 + [_fake(23)]
    cal = calibrate_protocol(recs, rule='median_index', n_boot=50)
    la = cal['hyperparameters']['lambda_aniso']
    print(f'  20x idx25, 19x idx24, 1x idx23 -> {la:.4f} (atteso 8.3768, idx 24)')
    assert abs(la - GA[24]) < 1e-9
    assert cal['cost']['lambda_aniso_steps_max'] == 1


def test_geomean_invariante_per_scala():
    from dbsi_toolbox import calibrate_protocol
    # stessa forma, scale diversissime: il soggetto "rumoroso" non deve dominare
    recs = [_fake(20, scale=1.0), _fake(20, scale=100.0), _fake(22, scale=1.0),
            _fake(22, scale=100.0)]
    cal = calibrate_protocol(recs, rule='geomean', n_boot=50)
    k = int(np.argmin(np.abs(GA - cal['hyperparameters']['lambda_aniso'])))
    print(f'  due curve a idx20 e due a idx22, scale 1 e 100 -> idx {k} (atteso 21)')
    assert k == 21, f'geomean pesato dalla scala: idx {k}'
    assert cal['aggregation']['alternative']['rule'] == 'median_index'
    json.dumps(cal)


def test_rifiuta_cio_che_non_combacia():
    from dbsi_toolbox import calibrate_protocol, protocol_fingerprint
    casi = {
        'impronta': [_fake(20), _fake(20, fp=protocol_fingerprint(
            np.r_[np.zeros(3), np.repeat([1000., 2000.], 8)]))],
        'n_dirs': [_fake(20), _fake(20, n_dirs=39)],
        'griglia': [_fake(20), _fake(20, grid_a=np.logspace(-4, 4, 40) * 1.01)],
    }
    for nome, recs in casi.items():
        try:
            calibrate_protocol(recs, n_boot=10)
        except ValueError as e:
            print(f'  {nome:<9} rifiutato: {str(e)[:80]}')
            continue
        raise AssertionError(f'{nome} diverso accettato in silenzio')


# ── catena completa su fantoccio ───────────────────────────────────────────
def _protocollo(n_b0=2, shells=(500., 1000., 2000., 3000.), per_shell=8, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * n_b0; bvecs = [np.zeros(3)] * n_b0
    for b in shells:
        for _ in range(per_shell):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            bvecs.append(v); bvals.append(float(b))
    return np.array(bvals), np.vstack(bvecs)


def _dati(bvals, seed, sigma):
    rng = np.random.default_rng(seed)
    S = 0.6 * np.exp(-bvals * 0.8e-3) + 0.4 * np.exp(-bvals * 2.5e-3)
    d = np.abs(1000 * S + rng.normal(0, sigma, (4, 4, 2, len(bvals)))).astype(np.float32)
    return d, np.ones(d.shape[:3], bool)


def test_catena_completa():
    from dbsi_toolbox import (calibrate_protocol, save_protocol_calibration,
                              load_protocol_calibration, save_output_maps)
    bvals, bvecs = _protocollo()
    recs = []
    for seed, sig in ((1, 20), (2, 30), (3, 40)):
        d, m = _dati(bvals, seed, sig)
        with contextlib.redirect_stdout(io.StringIO()):
            recs.append(DBSI_Adaptive().calibrate(d, bvals, bvecs, m, n_calibration_voxels=30))
    cal = calibrate_protocol(recs, name='fantoccio', ids=['a', 'b', 'c'], n_boot=100)
    h = cal['hyperparameters']
    print(f"  protocollo: lambda_aniso {h['lambda_aniso']:.4g}  lambda_iso {h['lambda_iso']:.4g}"
          f"  n_iso {h['n_iso']}  gate {h['concentration_gate']}  stabilita' "
          f"{cal['stability']['lambda_aniso_same']:.2f}")
    tmp = tempfile.mkdtemp(prefix='dbsi_proto_')
    path = os.path.join(tmp, 'fantoccio.json')
    sha = save_protocol_calibration(cal, path)
    cal2 = load_protocol_calibration(path)
    assert cal2['_source']['sha256'] == sha, 'lo sha256 letto non e quello scritto'

    m = DBSI_Adaptive.from_calibration(path)
    assert (m.lambda_aniso, m.lambda_iso, m.n_iso) == (h['lambda_aniso'], h['lambda_iso'], 6)
    d, mk = _dati(bvals, 7, 25)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(d, bvals, bvecs, mk)
        save_output_maps(res, DBSI_Adaptive.output_map_names(mode), np.eye(4), tmp, model=m)
    c = m.run_report_['calibrated']
    pc = m.run_report_['protocol_calibration']
    print(f"  fit: calibration_source={c['calibration_source']}  n_iso_source={c['n_iso_source']}"
          f"  sha={pc['sha256'][:12]}...  impronta ok={pc['fingerprint_match']}")
    assert c['calibration_source'] == 'protocol' and c['n_iso_source'] == 'protocol'
    assert pc['sha256'] == sha and pc['fingerprint_match'] is True
    assert (m.lambda_aniso, m.lambda_iso) == (h['lambda_aniso'], h['lambda_iso']), (
        'il fit ha ricalibrato i lambda del protocollo')
    copia = os.path.join(tmp, 'toolbox_report', 'protocol_calibration.json')
    assert os.path.isfile(copia), 'manca la copia della calibrazione in toolbox_report/'
    txt = open(os.path.join(tmp, 'toolbox_report', 'run_report.txt')).read()
    assert 'Protocol calibration' in txt and sha[:16] in txt

    # protocollo diverso: rifiutato PRIMA di fittare
    b2, v2 = _protocollo(shells=(500., 1000., 2000.))
    d2, mk2 = _dati(b2, 8, 25)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            DBSI_Adaptive.from_calibration(path).fit(d2, b2, v2, mk2)
    except ValueError as e:
        print(f'  protocollo diverso rifiutato: {str(e)[:90]}...')
    else:
        raise AssertionError('fit su un protocollo diverso accettato in silenzio')


def test_override_di_un_parametro_calibrato():
    from dbsi_toolbox import calibrate_protocol
    cal = calibrate_protocol([_fake(20), _fake(21)], n_boot=10)
    try:
        DBSI_Adaptive.from_calibration(cal, lambda_aniso=1.0)
    except ValueError as e:
        print(f'  rifiutato: {str(e)[:80]}...')
    else:
        raise AssertionError('from_calibration ha accettato di ridefinire lambda_aniso')
    m = DBSI_Adaptive.from_calibration(cal, stagec_refine=False)   # non calibrato: ok
    assert m.stagec_refine is False


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 5, 0), f'attesa >= 1.5.0, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_median_index_riproduce_watermobility,
               test_geomean_invariante_per_scala, test_rifiuta_cio_che_non_combacia,
               test_override_di_un_parametro_calibrato, test_catena_completa):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
