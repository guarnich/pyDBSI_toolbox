#!/usr/bin/env python
"""
Soglia del test 2-contro-1 in unita' del rumore VERO dei dati (v1.10.0).

IL DIFETTO. stat21 divide per sigma_mappa^2, quindi la soglia 12 vale "12 varianze di
rumore" solo se la mappa e' il rumore che i dati portano. Sul sintetico lo e'
(residual_over_sigma mediano 1.00); sui fit veri 1.9.0 no: 0.75-0.95 sui cinque
protocolli e su BR/BT, cioe' il 12 agiva come 13-21 varianze vere, e in modo diverso
da protocollo a protocollo. Ora la soglia applicata e' 12 * c, c = min(1, mediana di
residual_over_sigma^2) sui voxel di calibrazione.

COSA SI PROTEGGE.
  1. Mappa giusta: c ~ 1 e la soglia resta ~12. Mappa gonfiata di 1.5x: c ~ 1/2.25 e
     la soglia applicata e' 12 * c; il run report dice fonte, c e soglia.
  2. Lambda dati a mano senza calibrazione: c = 1 dichiarato 'nominal'; c dato
     dall'utente usato cosi' com'e'; c fuori da (0, 1] rifiutato.
  3. calibrate() scrive c nel record; la calibrazione di protocollo prende la mediana
     con il tetto a 1; record senza c -> None e una nota; from_calibration lo impone
     e rifiuta di ridefinirlo.
  4. Un record scritto prima della 1.10.0 si completa misurando solo c
     (add_two_vs_one_noise_ratio), senza rifare i lambda; protocollo diverso rifiutato.

    python tests/test_two_vs_one_noise_ratio.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox import DBSI_Adaptive

DATI = Path(__file__).resolve().parent / 'data'
BV = np.loadtxt(DATI / 'p3_like.bval').ravel().astype(float)
BC = np.loadtxt(DATI / 'p3_like.bvec')
BC = BC.T if BC.shape[0] == 3 and BC.shape[1] != 3 else BC
_n = np.linalg.norm(BC, axis=1, keepdims=True); _n[_n == 0] = 1.0
BC = BC / _n
S0, SIGMA = 1000.0, 1000.0 / 30


def _dir(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _fibra(u, ad, rd):
    c = BC @ u
    return np.exp(-BV * (rd + (ad - rd) * c * c))


def _volume(seed=0, n=8):
    """n x n x 1: fibre singole, crossing e voxel isotropi, rumore Rician vero (SNR 30)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 1, len(BV)), np.float32)
    for k, (i, j) in enumerate(np.ndindex(n, n)):
        iso = 0.1 * np.exp(-BV * 0.2e-3) + 0.15 * np.exp(-BV * 1.0e-3) + 0.1 * np.exp(-BV * 3.0e-3)
        if k % 3 == 0:
            y = 0.65 * _fibra(_dir(rng), 1.7e-3, 0.3e-3) + iso
        elif k % 3 == 1:
            u = _dir(rng); w = np.cross(u, _dir(rng)); w /= np.linalg.norm(w)
            y = 0.33 * _fibra(u, 1.6e-3, 0.4e-3) + 0.32 * _fibra(w, 1.6e-3, 0.4e-3) + iso
        else:
            y = iso / 0.35
        re = S0 * y + rng.normal(0, SIGMA, y.shape)
        im = rng.normal(0, SIGMA, y.shape)
        img[i, j, 0] = np.sqrt(re ** 2 + im ** 2)
    return img, np.ones(img.shape[:3], bool)


def _fit(m, img, mask, fattore=1.0, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return m.fit(img, BV, BC, mask, noise_map=np.full(mask.shape, fattore * SIGMA), **kw)


def test_c_misurato():
    img, mask = _volume()
    out = {}
    for fattore in (1.0, 1.5):
        m = DBSI_Adaptive()
        _fit(m, img, mask, fattore)
        nr, t = m.two_vs_one_noise_ratio_, m.run_report_['two_vs_one']
        assert nr['source'] == 'measured', nr
        assert np.isclose(t['threshold'], 12.0 * nr['applied']), t
        assert t['noise_ratio_source'] == 'measured' and t['threshold_nominal'] == 12.0
        out[fattore] = nr
        print(f"  mappa x{fattore}: residual/sigma {nr['residual_over_sigma_median']:.3f}, "
              f"c misurato {nr['noise_ratio_measured']:.3f}, applicato {nr['applied']:.3f} "
              f"-> soglia {t['threshold']:.2f}")
    assert 0.8 <= out[1.0]['applied'] <= 1.0, out[1.0]
    assert 0.33 <= out[1.5]['applied'] <= 0.55, out[1.5]          # atteso ~1/2.25 = 0.44
    assert np.isclose(out[1.5]['noise_ratio_measured'] / out[1.0]['noise_ratio_measured'],
                      1 / 2.25, rtol=0.15)


def test_c_nominale_e_dato():
    img, mask = _volume(1, n=4)
    kw = dict(lambda_aniso=8.3768, lambda_iso=0.1, n_iso=6, uncertainty=False)
    m = DBSI_Adaptive(**kw)
    _fit(m, img, mask, run_calibration=False)
    assert m.two_vs_one_noise_ratio_['source'] == 'nominal'
    assert m.run_report_['two_vs_one']['threshold'] == 12.0
    m = DBSI_Adaptive(two_vs_one_noise_ratio=0.5, **kw)
    _fit(m, img, mask, run_calibration=False)
    assert m.two_vs_one_noise_ratio_['source'] == 'user'
    assert m.run_report_['two_vs_one']['threshold'] == 6.0
    for bad in (0.0, -0.1, 1.5):
        try:
            DBSI_Adaptive(two_vs_one_noise_ratio=bad)
        except ValueError:
            continue
        raise AssertionError(f'c = {bad} accettato')
    m = DBSI_Adaptive(two_vs_one_threshold=None, **kw)
    _fit(m, img, mask, run_calibration=False)
    assert m.two_vs_one_noise_ratio_['source'] == 'test off'
    print('  nominale 12, dato 0.5 -> 6, valori fuori da (0, 1] rifiutati, test spento dichiarato')


def _record(c):
    from dbsi_toolbox import protocol_fingerprint
    GA, GI = np.logspace(-4, 4, 40), np.logspace(-5, 1, 40)
    k = np.arange(40)
    r = dict(kind='pydbsi_acquisition_calibration', format_version=1, toolbox_version='test',
             protocol=protocol_fingerprint(BV), noise=dict(snr=25.0, sigma_source='noise_map'),
             hyperparameters=dict(n_iso=6, lambda_aniso_method='gcv', concentration_gate=0.0,
                                  concentration_gate_calibrated=False,
                                  lambda_aniso=float(GA[24]), lambda_iso=float(GI[20])),
             selection=dict(lambda_iso_cap=None),
             curves=dict(lambda_aniso=dict(lambda_grid=list(GA), gcv=list(1 + 0.01 * (k - 24) ** 2)),
                         lambda_iso=dict(lambda_grid=list(GI), gcv=list(1 + 0.01 * (k - 20) ** 2))),
             model=dict(n_dirs=27, n_ad=3, n_rd=3, anisotropy_ratio=2.0,
                        fiber_threshold=0.15, iso_range=[0.0, 0.003]))
    if c is not None:
        r['two_vs_one'] = dict(threshold_nominal=12.0, noise_ratio_measured=c)
    return r


def test_protocollo():
    from dbsi_toolbox import calibrate_protocol
    cal = calibrate_protocol([_record(c) for c in (0.5, 0.6, 0.7)], n_boot=10)
    assert np.isclose(cal['hyperparameters']['two_vs_one_noise_ratio'], 0.6)
    assert [a['two_vs_one_noise_ratio'] for a in cal['acquisitions']] == [0.5, 0.6, 0.7]
    m = DBSI_Adaptive.from_calibration(cal)
    assert m.two_vs_one_noise_ratio == 0.6
    try:
        DBSI_Adaptive.from_calibration(cal, two_vs_one_noise_ratio=0.9)
    except ValueError:
        pass
    else:
        raise AssertionError('c ridefinito sopra una calibrazione di protocollo')
    cal = calibrate_protocol([_record(c) for c in (0.9, 1.2, 1.3)], n_boot=10)
    assert cal['hyperparameters']['two_vs_one_noise_ratio'] == 1.0
    assert cal['two_vs_one']['n_acquisitions_above_cap'] == 2
    cal = calibrate_protocol([_record(0.6), _record(None)], n_boot=10)
    assert cal['hyperparameters']['two_vs_one_noise_ratio'] is None and 'note' in cal['two_vs_one']
    assert DBSI_Adaptive.from_calibration(cal).two_vs_one_noise_ratio is None
    print('  mediana 0.6; tetto 1 con 2 acquisizioni sopra; record senza c -> None; imposto e protetto')


def test_record_di_calibrate():
    img, mask = _volume(2, n=6)
    with contextlib.redirect_stdout(io.StringIO()):
        rec = DBSI_Adaptive().calibrate(img, BV, BC, mask, n_calibration_voxels=36,
                                        noise_map=np.full(mask.shape, SIGMA))
    t = rec['two_vs_one']
    assert t['source'] == 'measured' and np.isfinite(t['noise_ratio_measured']), t
    assert t['n_voxels'] > 0 and t['threshold_nominal'] == 12.0
    print(f"  record: c {t['noise_ratio_measured']:.3f} su {t['n_voxels']} voxel")
    vecchio = {k: v for k, v in rec.items() if k != 'two_vs_one'}
    nuovo = DBSI_Adaptive.add_two_vs_one_noise_ratio(vecchio, img, BV, BC, mask,
                                                     np.full(mask.shape, SIGMA), n_calibration_voxels=36)
    assert 'two_vs_one' not in vecchio and nuovo['hyperparameters'] == rec['hyperparameters']
    assert np.isclose(nuovo['two_vs_one']['noise_ratio_measured'], t['noise_ratio_measured'], rtol=0.1)
    try:
        DBSI_Adaptive.add_two_vs_one_noise_ratio(vecchio, img[..., :-1], BV[:-1], BC[:-1], mask,
                                                 np.full(mask.shape, SIGMA))
    except ValueError:
        pass
    else:
        raise AssertionError('record completato su un protocollo diverso')
    print(f"  completato dopo: c {nuovo['two_vs_one']['noise_ratio_measured']:.3f}; protocollo diverso rifiutato")


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(name); fn()
    print('OK')
