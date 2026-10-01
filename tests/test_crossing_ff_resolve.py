#!/usr/bin/env python
"""
FF dei crossing ri-stimata prima dell'LM (v1.7.0, default crossing_ff_kappa=30).

IL DIFETTO (1.6.x). La FF dei crossing restava quella di Stage A, ristretta dalla penalita' forte
che serve a rilevare le direzioni, e l'LM la teneva fissa: il tensore compensava diventando piu'
anisotropo, RD sul pavimento. 75% dei crossing sani sintetici a SNR 26, 75% di quelli veri
(notebook 11, controllo sano). Con la FF ri-stimata: 20% (10-12% per popolazione, come le
mono-fibra), rapporto RD crossing / mono vicine 0.63 -> 1.00.

COSA SI PROTEGGE.
  1. Crossing sani sintetici (P3, SNR 26): col default quota sul pavimento < 0.25 e |bias RD| < 0.08e-3;
     con crossing_ff_kappa=0 (percorso 1.6.x) il difetto c'e' ancora (pavimento > 0.4, bias < -0.1e-3).
  2. Le mono-fibra sono IDENTICHE fra i due percorsi (la ri-stima tocca solo i crossing).
  3. Nei crossing le frazioni sommano a 1 e FF_pop1 + FF_pop2 = FF.
  4. kappa < 0 rifiutato; senza Stage D la ri-stima e' spenta e il run report lo dice.

    python tests/test_crossing_ff_resolve.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR

NOMI = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NOMI)}
LAM = dict(lambda_aniso=9.805121767832382, lambda_iso=0.015117750706156615, n_iso=6)
EXP = Path(__file__).resolve().parents[1] / 'experiments' / 'uncertainty'
_CACHE = {}


def _dati(n=80, snr=26.0, seed=4):
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel(); bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb
    rng = np.random.default_rng(seed)
    iso = 0.1 * np.exp(-bv * 0.15e-3) + 0.2 * np.exp(-bv * 1.0e-3) + 0.1 * np.exp(-bv * 3.0e-3)
    f = lambda d: np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ d) ** 2))
    d = np.zeros((2, n, 1, bv.size), np.float32)
    for g in range(2):
        for i in range(n):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            S = iso + (0.6 * f(u) if g == 0 else 0.3 * (f(u) + f(t)))
            d[g, i, 0] = 1000 * np.sqrt((S + rng.normal(0, 1 / snr, S.shape)) ** 2
                                        + rng.normal(0, 1 / snr, S.shape) ** 2)
    return d, bv, bc


def _fit(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        d, bv, bc = _dati()
        m = DBSI_Adaptive(**LAM, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = m.fit(d, bv, bc, np.ones(d.shape[:3], bool), run_calibration=False)
        _CACHE[key] = (m, res)
    return _CACHE[key]


def _crossing_stats(res):
    cro = res[1, :, 0, IX['n_fiber_populations']] == 2
    r = np.concatenate([res[1, cro, 0, IX['radial_diffusivity_pop1']], res[1, cro, 0, IX['radial_diffusivity_pop2']]])
    floor = np.mean((np.abs(res[1, cro, 0, IX['radial_diffusivity_pop1']] - _TENSOR_RD_FLOOR) < 1e-9)
                    | (np.abs(res[1, cro, 0, IX['radial_diffusivity_pop2']] - _TENSOR_RD_FLOOR) < 1e-9))
    return int(cro.sum()), float(floor), float(np.mean(r) - 0.4e-3)


def test_pavimento_e_bias_rd_nei_crossing():
    _, r30 = _fit(); _, r0 = _fit(crossing_ff_kappa=0)
    n30, f30, b30 = _crossing_stats(r30); n0, f0, b0 = _crossing_stats(r0)
    print(f"  default (kappa 30): {n30} crossing, pavimento {f30:.2f}, bias RD {b30 * 1e3:+.3f}e-3")
    print(f"  kappa 0 (1.6.x):    {n0} crossing, pavimento {f0:.2f}, bias RD {b0 * 1e3:+.3f}e-3")
    assert f30 < 0.25 and abs(b30) < 0.08e-3, 'la ri-stima della FF non toglie il pavimento'
    assert f0 > 0.4 and b0 < -0.1e-3, 'il percorso 1.6.x non riproduce piu il difetto documentato'


def test_mono_fibra_identiche():
    _, r30 = _fit(); _, r0 = _fit(crossing_ff_kappa=0)
    mono = (r30[..., IX['n_fiber_populations']] == 1) & (r0[..., IX['n_fiber_populations']] == 1)
    a, b = r30[mono], r0[mono]
    same = np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    print(f"  {int(mono.sum())} mono-fibra: identiche {same}")
    assert same


def test_frazioni_coerenti_nei_crossing():
    _, r = _fit()
    cro = r[..., IX['n_fiber_populations']] == 2
    tot = sum(r[..., IX[c]][cro] for c in ('fiber_fraction', 'restricted_fraction', 'hindered_fraction', 'water_fraction'))
    pops = r[..., IX['fiber_fraction_pop1']][cro] + r[..., IX['fiber_fraction_pop2']][cro]
    e1 = float(np.max(np.abs(tot - 1))); e2 = float(np.max(np.abs(pops - r[..., IX['fiber_fraction']][cro])))
    print(f"  somma frazioni - 1: {e1:.1e};  FF_pop1 + FF_pop2 - FF: {e2:.1e}")
    assert e1 < 1e-5 and e2 < 1e-5


def test_rifiuti_e_senza_stage_d():
    for k in (-1.0, None):
        try:
            DBSI_Adaptive(crossing_ff_kappa=k)
        except ValueError:
            continue
        raise AssertionError(f'crossing_ff_kappa={k} accettato')
    m, _ = _fit(iso_resolve=False)
    k = m.run_report_['options']['crossing_ff_kappa']
    print(f"  senza Stage D: crossing_ff_kappa effettivo {k}")
    assert k == 0.0
    m, _ = _fit()
    assert m.run_report_['options']['crossing_ff_kappa'] == 30.0


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
