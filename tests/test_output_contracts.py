#!/usr/bin/env python3
"""
Due contratti di output che erano rotti in silenzio (v1.3.7).

1. LA MASCHERA DI VALIDITA' NON ERA UNA. `compute_fiber_validity_map` produceva
   UNA maschera, costruita da `axial_diffusivity_pop1` e dalla `fiber_fraction`
   totale, e `save_output_maps` la scriveva come unico `fiber_valid.nii.gz`
   descritto come "validity mask for the tensor channels" -- senza qualificare
   QUALI. Ma i canali hanno tre domini distinti: il blocco pop1 e gli aggregati
   pesati sono NaN dove non c'e' fibra, il blocco pop2 e' NaN dove `n_pop < 2`
   (cioe' su ~48% dei voxel di fibra, dato che il censo dei crossing e' ~52%), e
   i canali 0-6/24-27 sono scritti per ogni voxel fittato.

   Usare la maschera di pop1 sulle mappe pop2 dichiara "valido" un voxel dove il
   valore e' NaN: dopo che un resampler trasforma NaN in 0 e la convoluzione
   normalizzata divide per quella validita', le mappe pop2 escono deprese di
   quella frazione. E' esattamente il bug di diluizione che quella funzione
   esiste per prevenire, reintrodotto sui canali pop2.

2. LA RAMPA DI CONFIDENZA ERA INVERTITA. `_confidence_from_distance` restituiva
   confidenza MASSIMA esattamente dove il bias e' peggiore (sulla soglia) e ZERO
   ai bordi della zona dove non ce n'e'. Misurato sulla zona RES ([0.10, 0.50]e-3,
   soglia 0.30e-3): 0.10 -> 0.000, 0.30 -> 1.000, 0.50 -> 0.000, fuori zona
   -> 1.000. Era anche DISCONTINUA al bordo (1.0 fuori, 0.0 appena dentro).
   Nessuno la chiama nella pipeline, quindi nessun risultato prodotto ne e'
   toccato, ma e' esportata in `__all__`.

    python tests/test_output_contracts.py
"""
import sys

import numpy as np

from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.fit_quality import compute_fiber_validity_map as VALID
from dbsi_toolbox.transition_confidence import (
    _confidence_from_distance as CF, _RES_ZONE_LOW, _RES_ZONE_HIGH,
    THRESH_RES, _RES_ZONE_PEAK_BIAS, _WAT_ZONE_LOW, _WAT_ZONE_HIGH,
    THRESH_WAT, _WAT_ZONE_PEAK_BIAS)


def _mappe(n=4):
    """4 voxel: mono-fibra, crossing, mono, e uno con FF azzerata da Stage D."""
    nm = DBSI_Adaptive.output_map_names(3)
    r = np.full((n, 1, 1, 28), np.nan, dtype=np.float32)
    r[..., 0] = np.array([0.5, 0.5, 0.5, 0.0]).reshape(n, 1, 1)
    r[..., nm.index('axial_diffusivity_pop1')] = 1.7e-3
    ad2 = np.array([1.6e-3, np.nan, 1.6e-3, 1.6e-3]).reshape(n, 1, 1)
    r[..., nm.index('axial_diffusivity_pop2')] = ad2
    return r, nm


def test_due_domini_di_validita():
    print("\n=== test_due_domini_di_validita ===")
    r, nm = _mappe()
    v1 = VALID(r, nm, 'pop1').ravel()
    v2 = VALID(r, nm, 'pop2').ravel()
    print(f"  pop1 {v1.tolist()}   (atteso [1, 1, 1, 0])")
    print(f"  pop2 {v2.tolist()}   (atteso [1, 0, 1, 0])")
    assert v1.tolist() == [1, 1, 1, 0], 'dominio pop1 sbagliato'
    assert v2.tolist() == [1, 0, 1, 0], (
        'dominio pop2 sbagliato: il voxel senza pop2 deve valere 0, altrimenti '
        'la correzione di diluizione deprime le mappe pop2')
    assert not np.array_equal(v1, v2), (
        'le due maschere coincidono: una sola maschera non puo servire domini '
        'diversi -- e il caso in cui il bug e invisibile')
    print("  [ok]")


def test_validita_rifiuta_argomento_ignoto():
    print("\n=== test_validita_rifiuta_argomento_ignoto ===")
    r, nm = _mappe()
    try:
        VALID(r, nm, 'pop3')
    except ValueError as e:
        print(f"  alza ValueError: {e}")
        print("  [ok]")
        return
    raise AssertionError("un population ignoto deve alzare ValueError, non "
                         "restituire silenziosamente una maschera")


def test_confidenza_non_invertita():
    """0 sulla soglia (bias massimo), 1 al bordo della zona (nessun bias)."""
    print("\n=== test_confidenza_non_invertita ===")
    ok = True
    for nome, zl, zh, th, pb in (
            ('RES', _RES_ZONE_LOW, _RES_ZONE_HIGH, THRESH_RES, _RES_ZONE_PEAK_BIAS),
            ('WAT', _WAT_ZONE_LOW, _WAT_ZONE_HIGH, THRESH_WAT, _WAT_ZONE_PEAK_BIAS)):
        c_soglia = float(CF(np.array([th]), zl, zh, th, pb)[0])
        c_lo = float(CF(np.array([zl]), zl, zh, th, pb)[0])
        c_hi = float(CF(np.array([zh]), zl, zh, th, pb)[0])
        print(f"  zona {nome}: soglia {c_soglia:.3f}   bordo basso {c_lo:.3f}   "
              f"bordo alto {c_hi:.3f}")
        if not (c_soglia < 0.01 and c_lo > 0.99 and c_hi > 0.99):
            ok = False
    assert ok, ('la rampa e INVERTITA: deve dare 0 sulla soglia (bias massimo) e '
                '1 ai bordi della zona (nessun bias)')
    print("  [ok]")


def test_confidenza_monotona_e_continua():
    print("\n=== test_confidenza_monotona_e_continua ===")
    zl, zh, th, pb = _RES_ZONE_LOW, _RES_ZONE_HIGH, THRESH_RES, _RES_ZONE_PEAK_BIAS
    # monotona: scendendo verso la soglia da sotto, la confidenza scende
    d = np.linspace(zl, th, 25)
    c = CF(d, zl, zh, th, pb)
    assert np.all(np.diff(c) <= 1e-12), f'non monotona sotto soglia: {np.round(c,3)}'
    d2 = np.linspace(th, zh, 25)
    c2 = CF(d2, zl, zh, th, pb)
    assert np.all(np.diff(c2) >= -1e-12), f'non monotona sopra soglia: {np.round(c2,3)}'
    print(f"  monotona su entrambi i lati della soglia")
    # continua al bordo: dentro e fuori devono coincidere
    dentro = float(CF(np.array([zl]), zl, zh, th, pb)[0])
    fuori = float(CF(np.array([zl - 1e-9]), zl, zh, th, pb)[0])
    print(f"  al bordo: dentro {dentro:.4f}   fuori {fuori:.4f}")
    assert abs(dentro - fuori) < 1e-6, (
        f'DISCONTINUA al bordo della zona ({dentro:.3f} dentro contro {fuori:.3f} '
        'fuori): un salto la dove il bias e nullo non ha senso fisico')
    print("  [ok]")


if __name__ == '__main__':
    for f in (test_due_domini_di_validita, test_validita_rifiuta_argomento_ignoto,
              test_confidenza_non_invertita, test_confidenza_monotona_e_continua):
        try:
            f()
        except AssertionError as e:
            print(f"\n!!! {f.__name__}: {e}")
            sys.exit(1)
    print("\ntutti i test passati")
