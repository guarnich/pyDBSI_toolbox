#!/usr/bin/env python
"""
Regressione per i bound del tensore e per la convergenza della NNLS (v1.3.3).

I DUE DIFETTI.

1. I bound del tensore erano letterali sparsi su sette siti, e avevano DIVERGITO:
   il pavimento della AD leggeva 0.05e-3 in quattro punti e 0.20e-3 in un quinto.
   0.05e-3 non era un pavimento AD ragionato — e' il valore condiviso PRE-"Leva 1",
   uguale a _STAGE_A_RD_MIN, rimasto indietro quando fu alzato il solo pavimento
   della RD (i commenti accanto dicevano ancora "was 0.05e-3"). Stava anche un
   fattore dieci sotto il range AD del dizionario, che parte da 0.5e-3.

2. `nnls_coordinate_descent` RITORNA il numero di iterazioni, e tutti e dodici i
   punti di chiamata lo scartavano con `w, _ =`. Toccare il tetto era quindi
   inosservabile. Misurato: col vecchio tetto di 2000, ~15% delle risoluzioni sul
   dizionario vero non convergeva, e fra tetto e convergenza il vettore dei pesi
   si muove fino al ~30% relativo.

    python tests/test_tensor_bounds_and_nnls.py
"""
import io, re, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox.core import solvers as S
from dbsi_toolbox.fit_quality import format_run_report

QUI = Path(__file__).resolve().parent.parent


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 3, 4), f'attesa >= 1.3.4, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


def test_bound_hanno_una_sola_definizione():
    """Nessun letterale residuo per i bound del tensore in solvers.py."""
    src = (QUI / 'dbsi_toolbox' / 'core' / 'solvers.py').read_text(encoding='utf-8')
    # solo le righe di CODICE: i commenti possono citare i valori storici
    code = '\n'.join(l for l in src.splitlines()
                     if l.strip() and not l.strip().startswith('#'))
    vietati = {
        '0.05e-3': 'vecchio pavimento AD, sostituito da _TENSOR_AD_FLOOR',
        '0.2e-3':  'pavimento AD letterale, usare _TENSOR_AD_FLOOR',
        '3.5e-3':  'tetto AD letterale, usare _TENSOR_AD_CEIL',
    }
    trovati = {}
    for lit, perche in vietati.items():
        # cerco assegnazioni/clamp, non riferimenti in stringhe di docstring
        hits = [l.strip() for l in code.splitlines()
                if lit in l and ('=' in l or 'max(' in l or 'min(' in l)
                and '_TENSOR_' not in l and '_ISO_NOMINAL' not in l]
        if hits:
            trovati[lit] = (perche, hits[:3])
    assert not trovati, f'letterali residui: {trovati}'
    print('  nessun letterale residuo sui bound del tensore')


def test_pavimento_ad_riconciliato():
    """Un solo pavimento AD, e non e' piu' quello copiato dalla RD."""
    assert S._TENSOR_AD_FLOOR == 0.20e-3, S._TENSOR_AD_FLOOR
    assert S._TENSOR_AD_FLOOR != M._STAGE_A_RD_MIN, (
        'il pavimento AD coincide ancora con _STAGE_A_RD_MIN: era proprio il bug')
    assert S._TENSOR_AD_FLOOR < M._STAGE_A_AD_MIN, (
        'il pavimento non deve stringere il range AD del dizionario')
    assert S._TENSOR_RD_FLOOR < S._TENSOR_AD_FLOOR < S._TENSOR_AD_CEIL
    print(f'  AD [{S._TENSOR_AD_FLOOR*1e3:.2f}, {S._TENSOR_AD_CEIL*1e3:.2f}]  '
          f'RD [{S._TENSOR_RD_FLOOR*1e3:.2f}, {S._TENSOR_RD_CEIL*1e3:.2f}]  (x1e-3)')


def test_tetto_nnls_alzato():
    assert S._NNLS_MAX_ITER >= 20000, S._NNLS_MAX_ITER
    import inspect
    fn = getattr(S.nnls_coordinate_descent, 'py_func', S.nnls_coordinate_descent)
    d = inspect.signature(fn).parameters['max_iter'].default
    assert d == S._NNLS_MAX_ITER, f'default {d} != costante {S._NNLS_MAX_ITER}'
    print(f'  _NNLS_MAX_ITER = {S._NNLS_MAX_ITER} (era 2000)')


def test_nessun_punto_di_chiamata_scarta_il_contatore_nei_kernel():
    """Nei kernel il contatore va USATO, non scartato con `w, _ =`."""
    src = (QUI / 'dbsi_toolbox' / 'model_Niso_adaptive_ff_thr.py').read_text(encoding='utf-8')
    scartati = len(re.findall(r'w,\s*_\s*=\s*nnls_coordinate_descent', src))
    usati = len(re.findall(r'w,\s*_nnls_it\s*=\s*nnls_coordinate_descent', src))
    assert usati == 2, f'attesi 2 kernel strumentati, trovati {usati}'
    scritture = len(re.findall(r'out\[x,\s*y,\s*z,\s*_C_NNLS_IT\]\s*=\s*_nnls_it', src))
    assert scritture == 2, f'attese 2 scritture del canale, trovate {scritture}'
    print(f'  kernel strumentati: {usati}  |  ancora scartato altrove nel modello: {scartati}')


def test_contratto_dei_canali():
    n3 = DBSI_Adaptive.output_map_names(3)
    n2 = DBSI_Adaptive.output_map_names(2)
    assert len(n3) == len(n2) == M._N_CHANNELS == 28, (len(n3), len(n2), M._N_CHANNELS)
    assert n3.index('nnls_iterations') == M._C_NNLS_IT == 27
    assert DBSI_Adaptive.N_CHANNELS == M._N_CHANNELS, (
        'N_CHANNELS della classe e _N_CHANNELS del modulo devono coincidere: '
        'se divergono i kernel scrivono FUORI dall array, in njit, senza errore')
    print(f'  {len(n3)} canali, nnls_iterations all indice {M._C_NNLS_IT}')


def test_diagnostica_del_solver():
    """_solver_diagnostics conta i bound attivi e la non-convergenza."""
    sh = (4, 4, 1)
    res = np.full(sh + (M._N_CHANNELS,), np.nan, np.float32)
    mask = np.ones(sh, bool)
    flat = res.reshape(-1, M._N_CHANNELS)
    n = flat.shape[0]
    flat[:, M._C_NNLS_IT] = 100.0
    flat[:2, M._C_NNLS_IT] = S._NNLS_MAX_ITER - 1        # 2 non convergenti
    flat[:, M._C_RD1] = 0.5e-3
    flat[:4, M._C_RD1] = S._TENSOR_RD_FLOOR              # 4 sul pavimento
    flat[:, M._C_AD1] = 1.5e-3
    flat[:1, M._C_AD1] = S._TENSOR_AD_CEIL               # 1 sul tetto
    sd = M._solver_diagnostics(res, mask)
    assert abs(sd['nnls_not_converged_pct'] - 100.0 * 2 / n) < 1e-6, sd['nnls_not_converged_pct']
    assert abs(sd['rd_pop1_at_floor_pct'] - 100.0 * 4 / n) < 1e-6, sd['rd_pop1_at_floor_pct']
    assert abs(sd['ad_pop1_at_ceil_pct'] - 100.0 * 1 / n) < 1e-6, sd['ad_pop1_at_ceil_pct']
    assert sd['tensor_rd_floor'] == S._TENSOR_RD_FLOOR
    print(f"  non convergenti {sd['nnls_not_converged_pct']}%  "
          f"RD sul pavimento {sd['rd_pop1_at_floor_pct']}%  "
          f"AD sul tetto {sd['ad_pop1_at_ceil_pct']}%")


def test_il_report_mostra_tutto():
    """Le due lacune di provenienza sono chiuse nel testo del report."""
    rep = {
        'toolbox_version': '1.3.3',
        'calibrated': {'concentration_gate': 0.45352,
                       'concentration_gate_calibrated': False,
                       'n_calibration_voxels': 1000, 'n_bootstrap': 50},
        'solver': {'nnls_max_iter': S._NNLS_MAX_ITER, 'nnls_iter_median': 186.0,
                   'nnls_iter_p95': 900.0, 'nnls_iter_max': 1234,
                   'nnls_not_converged_pct': 0.0,
                   'rd_pop1_at_floor_pct': 31.7, 'ad_pop2_at_ceil_pct': 4.4,
                   'tensor_ad_floor': S._TENSOR_AD_FLOOR, 'tensor_ad_ceil': S._TENSOR_AD_CEIL,
                   'tensor_rd_floor': S._TENSOR_RD_FLOOR, 'tensor_rd_ceil': S._TENSOR_RD_CEIL},
    }
    txt = format_run_report(rep)
    for atteso in ('Solver diagnostics', 'NNLS iterations', 'did not converge',
                   'tensor bounds', 'RD pop1 on the floor', 'bound active',
                   'concentration_gate_calibrated', 'n_calibration_voxels'):
        assert atteso in txt, f'manca dal report: {atteso!r}'
    print('  il report dichiara i bound, la convergenza, e se il gate era imposto')


def test_risoluzione_del_raffinamento_stagec():
    """Stage C non deve restituire (AD, RD) su un reticolo a mezzo passo.

    Fino alla v1.3.3 il "local refine" era UN SOLO scan 5x5 a mezzo passo della
    griglia grossa, quindi il risultato cadeva sempre su quel reticolo: misurato
    sui dati veri, il 100.0% dei voxel mono-fibra, con ZERO valori intermedi.
    La quantizzazione arrivava fino ai riassunti: la mediana della RD di un
    cervello scattava su un valore di reticolo, ed e' per questo che la CoV
    cross-protocollo della RD leggeva 0.00% su cinque protocolli — la griglia,
    non la misura.
    """
    assert S._STAGEC_N_REFINE >= 3, S._STAGEC_N_REFINE
    rng = np.random.default_rng(0)
    bv = np.array([0.] * 4 + sum([[float(b)] * 12 for b in (500, 1000, 1500, 2000)], []))
    u = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))
    bc = np.vstack([[0., 0., 1.] if b < 50 else u() for b in bv])
    fdir = np.array([0., 0., 1.])
    iso_d = np.array([0.15e-3, 0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3])
    iso_f = np.exp(-np.outer(bv, iso_d)); iso_g = iso_f.T @ iso_f
    ADg = np.linspace(0.6e-3, 2.6e-3, 14); RDg = np.linspace(0.15e-3, 1.1e-3, 12)
    dr = RDg[1] - RDg[0]

    def synth(ff, ad, rd, snr, seed):
        r = np.random.default_rng(seed); s = np.zeros(len(bv))
        for i, b in enumerate(bv):
            c = float(np.dot(bc[i], fdir))
            s[i] = (ff * np.exp(-b * (rd + (ad - rd) * c * c))
                    + 0.20 * np.exp(-b * 1.0e-3) + 0.10 * np.exp(-b * 3.0e-3)
                    + 0.10 * np.exp(-b * 0.15e-3))
        return np.sqrt((s + r.normal(0, 1 / snr, len(bv))) ** 2
                       + r.normal(0, 1 / snr, len(bv)) ** 2)

    # RD vere su una scala molto piu' fine del mezzo passo (0.043e-3)
    got = []
    for k in range(30):
        y = synth(0.55, 1.7e-3, 0.30e-3 + k * 0.008e-3, 60, k)
        w = np.zeros(1 + len(iso_d))
        _, rd = S.stagec_varpro_single_fiber(y, bv, bc, fdir, iso_f, iso_g,
                                             ADg, RDg, 2.0, w)
        got.append(rd)
    got = np.asarray(got)

    su_mezzo = np.abs((got - RDg[0]) / (dr / 2) - np.round((got - RDg[0]) / (dr / 2))) < 1e-6
    assert su_mezzo.mean() < 0.60, (
        f'{100*su_mezzo.mean():.0f}% dei valori sta ancora sul reticolo a mezzo '
        f'passo: il raffinamento non sta iterando')
    atteso = dr / 2 ** S._STAGEC_N_REFINE
    su_fine = np.abs((got - RDg[0]) / atteso - np.round((got - RDg[0]) / atteso)) < 1e-6
    assert su_fine.mean() > 0.95, f'{100*su_fine.mean():.0f}% sul reticolo fine atteso'
    distinti = len(np.unique(np.round(got, 13)))
    assert distinti >= 15, f'solo {distinti} valori distinti su {len(got)} casi'
    print(f'  sul reticolo a mezzo passo {100*su_mezzo.mean():.0f}% (era 100%), '
          f'valori distinti {distinti}/{len(got)}, risoluzione {atteso*1e3:.4f} (x1e-3)')


def test_un_solo_raffinamento_riproduce_il_vecchio():
    """Con _STAGEC_N_REFINE = 1 gli offset coincidono con quelli pre-1.3.4.

    Non e' una verifica numerica ma un'identita' algebrica, e vale la pena
    scriverla perche' e' la proprieta' che rende bisecabile una regressione:
    il vecchio codice calcolava `(j-2) * 0.5 * da`, il nuovo calcola
    `(j-2) * da'` con `da' = da * 0.5`. La moltiplicazione per 0.5 e' esatta in
    IEEE754 (potenza di due), quindi i due sono bit per bit identici.
    """
    da = 0.15384615384615385e-3
    vecchio = [(j - 2) * 0.5 * da for j in range(5)]
    nuovo = [(j - 2) * (da * 0.5) for j in range(5)]
    assert vecchio == nuovo, list(zip(vecchio, nuovo))
    print('  offset identici bit per bit con N_REFINE = 1')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_bound_hanno_una_sola_definizione,
               test_pavimento_ad_riconciliato, test_tetto_nnls_alzato,
               test_nessun_punto_di_chiamata_scarta_il_contatore_nei_kernel,
               test_contratto_dei_canali, test_diagnostica_del_solver,
               test_il_report_mostra_tutto,
               test_risoluzione_del_raffinamento_stagec,
               test_un_solo_raffinamento_riproduce_il_vecchio):
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
