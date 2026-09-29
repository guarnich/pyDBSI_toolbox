#!/usr/bin/env python
"""
La correzione della frazione restricted e' RIMOSSA (v1.6.0), non solo spenta.

Era una tabella di risposta Monte Carlo invertita per voxel, costruita per la
pipeline con la sola Stage A. Stage C e Stage D correggono la sottostima della RF
alla fonte, e applicarla sopra raddoppiava la correzione: RF 2.45 volte peggio
su 320 voxel sintetici (RF media 0.149 -> 0.274, WM sana 0.076 -> 0.272).
Un'opzione nota per peggiorare il risultato non resta esposta.

    python tests/test_rf_correction_removed.py
"""
import sys
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.calibration.data_driven as DD
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M


def test_argomento_rifiutato():
    import inspect
    par = inspect.signature(DBSI_Adaptive.fit).parameters
    print(f"  correct_restricted_fraction nella firma di fit(): {'correct_restricted_fraction' in par}")
    assert 'correct_restricted_fraction' not in par, 'fit() accetta ancora correct_restricted_fraction'


def test_funzioni_assenti():
    resti = [n for n in ('build_rf_response_table', 'apply_rf_correction', '_mc_signal')
             if hasattr(DD, n)]
    resti += [n for n in dir(M) if n.startswith('_RF_CORRECTION') or n == '_RF_DEADZONE_EST']
    print(f'  resti della correzione RF: {resti or "nessuno"}')
    assert not resti, f'ancora presenti: {resti}'


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 6, 0), f'attesa >= 1.6.0, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_argomento_rifiutato, test_funzioni_assenti):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
