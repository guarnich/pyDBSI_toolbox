#!/usr/bin/env python
"""
Nessun kernel numba puo' assumere che i NaN non esistano (v1.6.4).

IL DIFETTO. fastmath=True accende 'nnan'/'ninf': np.isnan compilato restituisce False anche per NaN,
e i confronti con NaN cambiano segno fra arm64 e x86. Il toolbox usa NaN come sentinella (popolazioni
assenti, tensori non stimati): sulla workstation il test di rilevamento faceva entrare i voxel senza
fibra con parametri NaN e si fermava con ZeroDivisionError (notebook 09, 2026-10-01).

COSA SI PROTEGGE.
  1. Nessuna funzione del pacchetto e' decorata con fastmath=True: si usa `_numba_flags.FASTMATH`.
  2. FASTMATH non contiene 'nnan' ne' 'ninf'.
  3. Con FASTMATH, isnan / isfinite / confronti con NaN compilati rispettano IEEE.

    python tests/test_fastmath_nan.py
"""
import ast, sys, pathlib
import numpy as np
from numba import njit

import dbsi_toolbox
from dbsi_toolbox._numba_flags import FASTMATH

PKG = pathlib.Path(dbsi_toolbox.__file__).parent


def test_nessun_fastmath_true():
    colpevoli = []
    for f in PKG.rglob('*.py'):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for d in node.decorator_list:
                    if isinstance(d, ast.Call):
                        for kw in d.keywords:
                            if kw.arg == 'fastmath' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                colpevoli.append(f'{f.name}:{node.name}')
    print(f"  funzioni con fastmath=True: {colpevoli or 'nessuna'}")
    assert not colpevoli, colpevoli


def test_flag_senza_nnan_ninf():
    print(f"  FASTMATH = {sorted(FASTMATH)}")
    assert 'nnan' not in FASTMATH and 'ninf' not in FASTMATH and 'fast' not in FASTMATH


def test_semantica_ieee():
    @njit(fastmath=FASTMATH)
    def prova(x):
        return np.isnan(x), x < 1.0, np.isfinite(x)
    r_nan, r_inf = prova(np.nan), prova(np.inf)
    print(f"  NaN -> isnan {r_nan[0]}, <1 {r_nan[1]}; inf -> isfinite {r_inf[2]}")
    assert r_nan[0] is True and r_nan[1] is False and r_inf[2] is False


if __name__ == '__main__':
    print(f"dbsi_toolbox {dbsi_toolbox.__version__}")
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
