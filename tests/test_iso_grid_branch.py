#!/usr/bin/env python
"""
Regressione per il bug della griglia isotropa (v1.3.1 -> v1.3.2).

IL BUG. `iso_grid` veniva costruita DENTRO il ramo della calibrazione, con un
`generate_isotropic_grid` (LINEARE) nel ramo `else`. Passando n_iso insieme a
entrambe le lambda --- cioe' il caso d'uso della calibrazione congelata di coorte ---
si finiva nell'else e la base ancorata log-uniforme veniva sostituita da una
lineare, in silenzio: con n_iso=6 e iso_range di default, 6 colonne su [0, 3e-3]
con UNA sola colonna restricted (quella degenere a d=0) e NESSUNA colonna per
l'acqua libera, invece di 11 colonne ancorate su [0.1e-3, 5e-3] divise 4/5/2.

COME SI TESTA. Non si riproduce la logica: si **strumenta** il modulo e si guarda
quale generatore viene chiamato davvero, fermando poi il fit con una sentinella
prima del loop sui voxel. E' la stessa disciplina che ha fatto trovare il bug —
ricostruire la chiamata fuori dal modello ha prodotto tre conclusioni sbagliate
in questo progetto.

    python tests/test_iso_grid_branch.py
"""
import io, contextlib, sys
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M


class _Sentinella(Exception):
    """Alzata al posto del kernel: la calibrazione e' finita, il resto non serve."""


def _protocollo(n_b0=2, shells=(500., 1000., 2000., 3000.), per_shell=8, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * n_b0
    bvecs = [np.zeros(3)] * n_b0
    for b in shells:
        for _ in range(per_shell):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            bvecs.append(v); bvals.append(float(b))
    return np.array(bvals), np.vstack(bvecs)


def _griglia_usata(**kw_costruttore):
    """Esegue fit() fino alla costruzione della griglia e torna (nome, n_colonne)."""
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(1)
    data = np.abs(rng.normal(1000, 60, (5, 5, 2, len(bvals)))).astype(np.float32)
    mask = np.ones(data.shape[:3], bool)

    visto = []
    orig_grid = {n: getattr(M, n) for n in
                 ('generate_anchored_isotropic_grid', 'generate_isotropic_grid')}
    orig_kern = {n: getattr(M, n) for n in
                 ('_fit_voxels_3iso_v3', '_fit_voxels_2iso_v3')}

    def spia(nome, f):
        def _(*a, **k):
            g = f(*a, **k); visto.append((nome, int(np.size(g)))); return g
        return _

    def stop(*a, **k):
        raise _Sentinella()

    m = DBSI_Adaptive(**kw_costruttore)
    try:
        for n, f in orig_grid.items():
            setattr(M, n, spia(n, f))
        for n in orig_kern:
            setattr(M, n, stop)
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(data, bvals, bvecs, mask, run_calibration=True,
                  n_calibration_voxels=40,
                  calibrate_concentration_gate=('min_dominant_concentration'
                                                not in kw_costruttore),
                  correct_restricted_fraction=False)
    except _Sentinella:
        pass
    finally:
        for n, f in {**orig_grid, **orig_kern}.items():
            setattr(M, n, f)

    assert len(visto) == 1, f'attesa 1 chiamata alla griglia, viste {visto}'
    return visto[0]


FROZEN = dict(lambda_aniso=8.376776400682925,
              lambda_iso=0.017012542798525893,
              n_iso=6, min_dominant_concentration=0.45352)


def test_griglia_ancorata_sempre():
    """Tutti i percorsi devono usare la griglia ANCORATA, non quella lineare."""
    casi = {
        'calibrazione libera': {},
        'quattro numeri imposti (calibrazione congelata di coorte)': dict(FROZEN),
        'solo le due lambda': {k: FROZEN[k] for k in ('lambda_aniso', 'lambda_iso')},
        'solo n_iso': dict(n_iso=6),
    }
    esiti = {}
    for etichetta, kw in casi.items():
        nome, ncol = _griglia_usata(**kw)
        esiti[etichetta] = (nome, ncol)
        print(f'  {etichetta:58s} -> {nome} ({ncol} colonne)')
    cattivi = {k: v for k, v in esiti.items()
               if v[0] != 'generate_anchored_isotropic_grid'}
    assert not cattivi, (
        'griglia LINEARE su questi percorsi (era il bug di v1.3.1): ' + repr(cattivi))


def test_griglia_lineare_priva_di_acqua_libera():
    """Perche' il ramo sbagliato faceva danno: la lineare non copre i comparti."""
    from dbsi_toolbox.core.basis import (generate_anchored_isotropic_grid,
                                         generate_isotropic_grid)
    def conta(g):
        g = np.ravel(np.asarray(g))
        return (int((g <= M.THRESH_RES).sum()),
                int(((g > M.THRESH_RES) & (g <= M.THRESH_WAT)).sum()),
                int((g > M.THRESH_WAT).sum()))
    anc = generate_anchored_isotropic_grid(
        d_min=1e-4, d_max=max(3.0e-3, M._ISO_GRID_D_MAX_EXTENDED), n_steps=6,
        thresh_res=M.THRESH_RES, thresh_wat=M.THRESH_WAT)
    lin = generate_isotropic_grid(d_min=0.0, d_max=3.0e-3, n_steps=6)
    r_a, h_a, w_a = conta(anc); r_l, h_l, w_l = conta(lin)
    print(f'  ancorata: {np.size(anc):>3} colonne  RES {r_a} HIN {h_a} WAT {w_a}')
    print(f'  lineare : {np.size(lin):>3} colonne  RES {r_l} HIN {h_l} WAT {w_l}')
    assert w_a > 0, 'la griglia ancorata deve avere colonne per l acqua libera'
    assert r_a > 1, 'la griglia ancorata deve avere piu di una colonna restricted'
    assert w_l == 0, ('se la lineare avesse colonne WAT il test perderebbe il suo '
                      'senso: verificare i default')


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 3, 2), f'attesa >= 1.3.2, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_griglia_lineare_priva_di_acqua_libera,
               test_griglia_ancorata_sempre):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)

# ─────────────────────────────────────────────────────────────────────────────
# NOTA sul grilletto, per chi leggera' questo file fra un anno.
#
# La prima diagnosi era "si rompe passando n_iso INSIEME a entrambe le lambda",
# perche' sopra il punto incriminato c'e' un `if` grosso che nomina anche le
# lambda. Sbagliata: sono DUE `if` distinti, e la griglia stava dentro il
# secondo, che guarda solo `self.n_iso is None`. Il grilletto era quindi
# semplicemente **passare n_iso**, per qualunque motivo — anche un innocuo
# `DBSI_Adaptive(n_iso=6)` in un notebook.
#
# L'ha stabilito il test, non la lettura: il caso 'solo n_iso' e' risultato
# LINEARE dove la prima diagnosi prevedeva ancorato. Per questo il caso resta
# nella batteria anche ora che il bug e' chiuso.
