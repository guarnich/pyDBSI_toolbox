#!/usr/bin/env python
"""
toolbox_report/: run report + dizionario A accanto alle mappe (v1.3.10).

COSA SI PROTEGGE.
  1. `save_output_maps(..., model=m)` crea `toolbox_report/` con run_report.txt,
     design_matrix.png, design_matrix.npz e dictionary_columns.csv; il run
     report NON sta piu' accanto alle mappe.
  2. La matrice salvata e' QUELLA del fit (non una ricostruita): stessa forma,
     N_aniso = n_dirs x n_pairs, N_iso = colonne della griglia ancorata (11 per
     n_iso=6), e i valori coincidono con `m.dictionary_['A']`.
  3. Il run report descrive il dizionario (sezione Dictionary) e i numeri
     combaciano con la matrice.
  4. `find_run_report` trova il report sia nel layout nuovo sia nel vecchio.

Fit VERO su un fantoccio minuscolo con i parametri congelati: il report nasce
alla fine di fit(), quindi una sentinella prima dei kernel non basterebbe.

    python tests/test_toolbox_report.py
"""
import io, contextlib, csv, os, sys, tempfile
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive, save_output_maps


def _protocollo(n_b0=2, shells=(500., 1000., 2000., 3000.), per_shell=8, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * n_b0
    bvecs = [np.zeros(3)] * n_b0
    for b in shells:
        for _ in range(per_shell):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            bvecs.append(v); bvals.append(float(b))
    return np.array(bvals), np.vstack(bvecs)


FROZEN = dict(lambda_aniso=8.376776400682925,
              lambda_iso=0.017012542798525893,
              min_dominant_concentration=0.45352)

_CACHE = {}


def _fit_e_salva():
    if 'out' in _CACHE:
        return _CACHE['m'], _CACHE['out'], _CACHE['saved']
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(1)
    S = np.exp(-bvals * 1.0e-3)                       # segnale isotropo semplice
    data = (1000 * S + rng.normal(0, 20, (4, 4, 2, len(bvals)))).astype(np.float32)
    data = np.abs(data)
    mask = np.ones(data.shape[:3], bool)
    m = DBSI_Adaptive(**FROZEN)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(data, bvals, bvecs, mask, run_calibration=False,
                          correct_restricted_fraction=False)
    out = tempfile.mkdtemp(prefix='dbsi_report_')
    with contextlib.redirect_stdout(io.StringIO()):
        saved = save_output_maps(res, DBSI_Adaptive.output_map_names(mode),
                                 np.eye(4), out, model=m)
    _CACHE.update(m=m, out=out, saved=saved)
    return m, out, saved


def test_cartella_e_file():
    m, out, _ = _fit_e_salva()
    rep = os.path.join(out, 'toolbox_report')
    attesi = ['run_report.txt', 'design_matrix.png', 'design_matrix.npz',
              'dictionary_columns.csv']
    presenti = sorted(os.listdir(rep)) if os.path.isdir(rep) else []
    print(f'  toolbox_report/: {presenti}')
    for f in attesi:
        assert os.path.isfile(os.path.join(rep, f)), f'manca toolbox_report/{f}'
    assert not os.path.isfile(os.path.join(out, 'run_report.txt')), (
        'run_report.txt ancora accanto alle mappe: deve stare in toolbox_report/')
    with open(os.path.join(rep, 'design_matrix.png'), 'rb') as fh:
        assert fh.read(8) == b'\x89PNG\r\n\x1a\n', 'design_matrix.png non e un PNG'


def test_matrice_e_quella_del_fit():
    m, out, _ = _fit_e_salva()
    z = np.load(os.path.join(out, 'toolbox_report', 'design_matrix.npz'))
    A = z['A']
    n_aniso = int(z['n_aniso_cols']); n_iso_cols = len(z['iso_grid'])
    print(f'  A {A.shape}: N_aniso {n_aniso} = {int(z["n_dirs"])} dir x '
          f'{int(z["n_pairs"])} coppie, N_iso {n_iso_cols} (n_iso={int(z["n_iso"])})')
    assert A.shape == m.dictionary_['A'].shape
    assert np.array_equal(A, m.dictionary_['A']), 'la matrice salvata non e quella del fit'
    assert n_aniso == int(z['n_dirs']) * int(z['n_pairs'])
    assert A.shape[1] == n_aniso + n_iso_cols
    assert A.shape[0] == len(z['bvals'])
    assert (int(z['n_iso']), n_iso_cols) == (6, 11), (
        f'atteso n_iso=6 -> 11 colonne, trovato {int(z["n_iso"])} -> {n_iso_cols}')
    # colonne isotrope: exp(-b D), ricostruibili dalla griglia salvata
    atteso_iso = np.exp(-np.outer(z['bvals'], z['iso_grid']))
    assert np.allclose(A[:, n_aniso:], atteso_iso, atol=1e-10), (
        'il blocco isotropo non e exp(-b D) sulla griglia salvata')
    with open(os.path.join(out, 'toolbox_report', 'dictionary_columns.csv')) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == A.shape[1], 'dictionary_columns.csv: una riga per colonna'
    assert sum(r['block'] == 'iso' for r in rows) == n_iso_cols


def test_report_descrive_il_dizionario():
    m, out, _ = _fit_e_salva()
    txt = open(os.path.join(out, 'toolbox_report', 'run_report.txt')).read()
    assert 'Dictionary' in txt, 'manca la sezione Dictionary nel run report'
    d = m.run_report_['dictionary']
    print(f'  report: {d}')
    assert d['n_total_columns'] == d['n_aniso_columns'] + d['n_iso_columns']
    assert d['n_total_columns'] == m.dictionary_['A'].shape[1]


def test_find_run_report_due_layout():
    from dbsi_toolbox import find_run_report
    _, out, _ = _fit_e_salva()
    assert find_run_report(out) == os.path.join(out, 'toolbox_report', 'run_report.txt')
    vecchio = tempfile.mkdtemp(prefix='dbsi_old_')
    open(os.path.join(vecchio, 'run_report.txt'), 'w').write('x')
    assert find_run_report(vecchio) == os.path.join(vecchio, 'run_report.txt')
    assert find_run_report(tempfile.mkdtemp()) is None
    print('  layout nuovo, vecchio e assente: ok')


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 3, 10), f'attesa >= 1.3.10, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_cartella_e_file, test_matrice_e_quella_del_fit,
               test_report_descrive_il_dizionario, test_find_run_report_due_layout):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1
            print(f'  [FALLITO] {type(e).__name__}: {e}')
    if 'out' in _CACHE:
        print(f'\n(immagine di prova: {_CACHE["out"]}/toolbox_report/design_matrix.png)')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
