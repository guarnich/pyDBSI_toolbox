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

2. `transition_confidence` E' STATO RIMOSSO in v1.3.8, e questo test pretende
   che resti rimosso. Aveva due problemi, e il secondo e' quello che decide:

   (a) la rampa era INVERTITA -- confidenza MASSIMA esattamente dove il bias e'
       peggiore (sulla soglia) e ZERO ai bordi della zona dove non ce n'e'.
       Misurato sulla zona RES ([0.10, 0.50]e-3, soglia 0.30e-3): 0.10 -> 0.000,
       0.30 -> 1.000, 0.50 -> 0.000, fuori zona -> 1.000. Anche DISCONTINUA al
       bordo. Corretta in v1.3.7.

   (b) ma la sua PREMESSA non sopravvive a Stage D. Stage D fissa i centroidi
       isotropi a (0.15, 1.0, 3.0)e-3, quindi `mean_iso_adc` e' algebricamente
       determinato dalle frazioni e il "centroide ricostruito" non porta
       informazione su dove stesse la massa spettrale. Il recupero assumeva
       D_wat = 3.05e-3 contro il 3.00e-3 di Stage D, e lo scarto finiva tutto in
       D_hin amplificato da 1/hf: verificato a 6e-15,
           D_hin = 1.0e-3 - 0.05e-3 * (wf/hf).
       Allineando l'ipotesi al valore vero, D_hin diventa ESATTAMENTE 1.0e-3 in
       ogni voxel e le due mappe diventano COSTANTI. Cioe': sbagliata varia per
       il motivo sbagliato, giusta non dice niente. Correggerla l'avrebbe fatta
       sembrare affidabile senza renderla informativa -- peggio che rimuoverla.

    python tests/test_output_contracts.py
"""
import sys

import numpy as np

from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.fit_quality import compute_fiber_validity_map as VALID


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


def test_transition_confidence_resta_rimosso():
    """Il modulo non deve tornare: la sua premessa non sopravvive a Stage D."""
    print("\n=== test_transition_confidence_resta_rimosso ===")
    import dbsi_toolbox
    from pathlib import Path
    pkg = Path(dbsi_toolbox.__file__).parent

    f = pkg / 'transition_confidence.py'
    print(f"  il file esiste ancora? {'SI' if f.exists() else 'no'}")
    assert not f.exists(), (
        'transition_confidence.py e tornato. Se lo si vuole davvero, prima va '
        'risolto il problema che lo ha fatto rimuovere: dopo Stage D il centroide '
        '"ricostruito" e una funzione deterministica di wf/hf, e allineando '
        "l'ipotesi D_wat al valore vero di Stage D le mappe diventano costanti.")

    resti = [n for n in getattr(dbsi_toolbox, '__all__', []) if 'transition' in n]
    print(f"  export residui in __all__: {resti or 'nessuno'}")
    assert not resti, f'export non rimossi: {resti}'

    for nome in ('compute_transition_confidence', 'save_transition_confidence'):
        assert not hasattr(dbsi_toolbox, nome), f'{nome} e ancora importabile'
    print("  [ok]")


def test_stage_d_fissa_i_centroidi():
    """La ragione della rimozione, come misura: Stage D pinna i centroidi.

    Se un giorno Stage D smettesse di usare centroidi fissi, la premessa di
    `transition_confidence` tornerebbe valida e la rimozione andrebbe rivista.
    Questo test rende quel collegamento visibile invece di lasciarlo in un
    messaggio di commit.
    """
    print("\n=== test_stage_d_fissa_i_centroidi ===")
    from dbsi_toolbox.model_Niso_adaptive_ff_thr import (
        _ISO_RESOLVE_D_3ISO as SD, _ISO_RESOLVE_D_2ISO as SD2)
    print(f"  3-ISO: {[f'{d*1e3:.2f}e-3' for d in SD]}")
    print(f"  2-ISO: {[f'{d*1e3:.2f}e-3' for d in SD2]}")
    assert len(SD) == 3 and len(SD2) == 2
    # sono COSTANTI del modulo, non stimate dai dati: e questo il punto
    assert all(isinstance(d, float) for d in SD), (
        'i centroidi di Stage D non sono piu costanti: la premessa di '
        'transition_confidence potrebbe essere tornata valida, rivedere la '
        'rimozione')
    # e il centroide dell'acqua sta ESATTAMENTE su THRESH_WAT
    from dbsi_toolbox.model_Niso_adaptive_ff_thr import THRESH_WAT
    print(f"  centroide acqua {SD[2]*1e3:.2f}e-3 contro THRESH_WAT "
          f"{THRESH_WAT*1e3:.2f}e-3 -> coincidono: {abs(SD[2]-THRESH_WAT) < 1e-12}")
    print("  [ok]")


if __name__ == '__main__':
    for f in (test_due_domini_di_validita, test_validita_rifiuta_argomento_ignoto,
              test_transition_confidence_resta_rimosso,
              test_stage_d_fissa_i_centroidi):
        try:
            f()
        except AssertionError as e:
            print(f"\n!!! {f.__name__}: {e}")
            sys.exit(1)
    print("\ntutti i test passati")
