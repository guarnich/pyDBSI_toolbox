#!/usr/bin/env python
"""
n_iso si FISSA per quadratura, non si seleziona per soggetto (v1.3.9).

PERCHE'. Stage D ricalcola le frazioni riportate su 3 centroidi fissi, quindi il
bootstrap di n_iso ottimizzava una RF che la pipeline poi sovrascrive. Il notebook
Codes_fixed_20260923/06_convergenza_niso ha misurato la convergenza su dati veri:
n_iso=6 (11 colonne) sta entro tolleranza dal plateau, con 0% di NNLS non
convergenti; n_iso=4 -- cio' che il ripiego SVD restituisce SEMPRE -- fallisce su
FA. Vedi `_QUADRATURE_N_ISO` nel modello.

COSA SI PROTEGGE.
  1. Senza n_iso, il default da' n_iso=6 e 11 colonne, SENZA chiamare ne' il
     bootstrap ne' l'SVD -- sia a calibrazione libera sia coi lambda congelati.
  2. Il bootstrap resta disponibile, ma solo se chiesto.
  3. Un metodo sconosciuto si rifiuta PRIMA di campionare o fittare.

COME SI TESTA. Si strumenta il modulo (non si riproduce la logica) e si ferma il
fit con una sentinella al posto del kernel, come in test_iso_grid_branch.

    python tests/test_niso_quadrature.py
"""
import io, contextlib, sys
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M


class _Sentinella(Exception):
    """Alzata al posto del kernel: la griglia e' costruita, il resto non serve."""


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


def _fino_alla_griglia(kw_costruttore, **kw_fit):
    """Esegue fit() fino ai kernel; torna (modello, selettori chiamati)."""
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(1)
    data = np.abs(rng.normal(1000, 60, (5, 5, 2, len(bvals)))).astype(np.float32)
    mask = np.ones(data.shape[:3], bool)

    chiamati = []
    orig_sel = {n: getattr(M, n) for n in
                ('select_n_iso_bootstrap', 'select_n_iso_svd')}
    orig_kern = {n: getattr(M, n) for n in
                 ('_fit_voxels_3iso_v3', '_fit_voxels_2iso_v3')}

    def spia(nome, f):
        def _(*a, **k):
            chiamati.append(nome); return f(*a, **k)
        return _

    def stop(*a, **k):
        raise _Sentinella()

    m = DBSI_Adaptive(**kw_costruttore)
    congelato = 'min_dominant_concentration' in kw_costruttore
    kw = dict(run_calibration=not congelato, n_calibration_voxels=40,
              calibrate_concentration_gate=not congelato)
    kw.update(kw_fit)
    try:
        for n, f in orig_sel.items():
            setattr(M, n, spia(n, f))
        for n in orig_kern:
            setattr(M, n, stop)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(data, bvals, bvecs, mask, **kw)
    except _Sentinella:
        pass
    finally:
        for n, f in {**orig_sel, **orig_kern}.items():
            setattr(M, n, f)
    return m, chiamati


def test_default_e_quadratura():
    """Senza n_iso: 6 e 11 colonne, nessun selettore chiamato."""
    casi = {'calibrazione libera': {},
            'lambda e gate congelati, n_iso=None': dict(FROZEN)}
    for etichetta, kw in casi.items():
        m, chiamati = _fino_alla_griglia(kw)
        print(f'  {etichetta:40s} -> n_iso={m.n_iso}  colonne={m.n_iso_columns_}  '
              f'fonte={m.n_iso_source_}  selettori={chiamati or "nessuno"}')
        assert not chiamati, (
            f'{etichetta}: il default ha chiamato {chiamati} -- n_iso va FISSATO '
            f'per quadratura, non selezionato per soggetto')
        assert m.n_iso == M._QUADRATURE_N_ISO == 6, (
            f'{etichetta}: n_iso={m.n_iso}, atteso 6')
        assert m.n_iso_source_ == 'quadrature', (
            f'{etichetta}: fonte {m.n_iso_source_!r}, attesa \'quadrature\'')
        assert (m.n_iso_columns_, m.n_iso_columns_res_, m.n_iso_columns_wat_) \
            == (11, 4, 2), (
            f'{etichetta}: griglia {m.n_iso_columns_} colonne '
            f'({m.n_iso_columns_res_} R, {m.n_iso_columns_wat_} W), attesa 11 (4 R, 2 W)')


def test_bootstrap_solo_se_chiesto():
    """Il percorso guidato dai dati esiste ancora, ma va chiesto."""
    m, chiamati = _fino_alla_griglia(dict(FROZEN), n_iso_method='bootstrap')
    print(f'  n_iso_method=\'bootstrap\' -> selettori={chiamati}  '
          f'fonte={m.n_iso_source_}  n_iso={m.n_iso}')
    assert 'select_n_iso_bootstrap' in chiamati, (
        'con n_iso_method=\'bootstrap\' il bootstrap deve girare')


def test_metodo_sconosciuto_rifiutato():
    """Un refuso nel metodo non deve campionare ne' ripiegare in silenzio."""
    try:
        _fino_alla_griglia(dict(FROZEN), n_iso_method='quadratura')
    except ValueError as e:
        print(f'  rifiutato: {e}')
        return
    raise AssertionError('n_iso_method sconosciuto accettato in silenzio')


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 3, 9), f'attesa >= 1.3.9, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_default_e_quadratura,
               test_bootstrap_solo_se_chiesto, test_metodo_sconosciuto_rifiutato):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:                    # es. attributo assente su una versione
            falliti += 1                          # precedente: e un fallimento, non un crash
            print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
