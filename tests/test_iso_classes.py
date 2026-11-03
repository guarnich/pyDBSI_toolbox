#!/usr/bin/env python
"""
Soglie dei compartimenti e centroidi di Stage D configurabili (v1.10.0).

PERCHE'. Le soglie 0.3 / 3.0 um^2/ms (Wang 2015, Ye 2020) non sono un consenso: negli
studi sui tumori si usa RF <= 1.0, HF 1.0-1.5, non-restricted > 1.5 (Han 2023, Zhu
2026). In pyDBSI le frazioni riportate sono i pesi dei centroidi di Stage D, non fasce
di uno spettro: soglie e centroidi si danno INSIEME, ogni centroide nella sua classe.

COSA SI PROTEGGE.
  1. Default invariati: valori espliciti uguali ai default danno lo stesso fit, bit per
     bit; i default sono 0.3/3.0 e 0.15/1.0/3.0 (+ 1.5 per il 2-ISO).
  2. Validazione: soglie non di default senza centroidi, centroidi fuori classe,
     soglie non crescenti, valori in um^2/ms passati come mm^2/s -> ValueError; un fit
     2-ISO con soglie non di default chiede il centroide NRF.
  3. Le classi di Han funzionano: un voxel isotropo con componenti sui centroidi
     (0.5, 1.25, 3.0) restituisce le sue frazioni; il run report dichiara soglie e
     centroidi.
  4. Calibrazione di protocollo: le classi sono opzioni del modello (uguali fra le
     acquisizioni, imposte da from_calibration, non ridefinibili); un record senza
     classi (pre-1.10.0) vale come default.
  5. CLI: um^2/ms -> mm^2/s; vietate insieme a --protocol-calibration.

    python tests/test_iso_classes.py
"""
import io, sys, copy, contextlib, subprocess
from pathlib import Path
import numpy as np

import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox import DBSI_Adaptive

DATI = Path(__file__).resolve().parent / 'data'
CLI = Path(__file__).resolve().parents[1] / 'scripts' / 'run_dbsi.py'
BV = np.loadtxt(DATI / 'p3_like.bval').ravel().astype(float)
BC = np.loadtxt(DATI / 'p3_like.bvec')
BC = BC.T if BC.shape[0] == 3 and BC.shape[1] != 3 else BC
_n = np.linalg.norm(BC, axis=1, keepdims=True); _n[_n == 0] = 1.0
BC = BC / _n
HAN = dict(iso_thresholds=(1.0e-3, 1.5e-3), iso_centroids=(0.5e-3, 1.25e-3, 3.0e-3))
LAM = dict(lambda_aniso=8.3768, lambda_iso=0.1, n_iso=6, uncertainty=False)


def _volume(pesi, diff, snr=100.0, n=3, seed=0):
    rng = np.random.default_rng(seed)
    y = sum(w * np.exp(-BV * d) for w, d in zip(pesi, diff))
    sig = 1000.0 / snr
    img = np.empty((n, n, 1, len(BV)), np.float32)
    for i, j in np.ndindex(n, n):
        re = 1000.0 * y + rng.normal(0, sig, y.shape); im = rng.normal(0, sig, y.shape)
        img[i, j, 0] = np.sqrt(re ** 2 + im ** 2)
    return img, np.ones(img.shape[:3], bool), np.full(img.shape[:3], sig)


def _fit(m, img, mask, nm):
    with contextlib.redirect_stdout(io.StringIO()):
        return m.fit(img, BV, BC, mask, run_calibration=False, noise_map=nm)[0]


def test_default_invariati():
    m = DBSI_Adaptive()
    assert m.iso_thresholds == (0.3e-3, 3.0e-3)
    assert m.iso_centroids == (0.15e-3, 1.0e-3, 3.0e-3) and m.iso_centroid_nrf == 1.5e-3
    img, mask, nm = _volume((0.3, 0.5, 0.2), (0.15e-3, 1.0e-3, 3.0e-3), snr=30)
    a = _fit(DBSI_Adaptive(**LAM), img, mask, nm)
    b = _fit(DBSI_Adaptive(iso_thresholds=(0.3e-3, 3.0e-3), iso_centroids=(0.15e-3, 1.0e-3, 3.0e-3),
                           iso_centroid_nrf=1.5e-3, **LAM), img, mask, nm)
    assert np.array_equal(a, b, equal_nan=True)
    print('  default 0.3/3.0 e 0.15/1.0/3.0; valori espliciti = default, bit per bit')


def test_validazione():
    casi = [dict(iso_thresholds=(1.0e-3, 1.5e-3)),                                   # senza centroidi
            dict(iso_thresholds=(0.3, 3.0), iso_centroids=(0.15, 1.0, 3.0)),          # um^2/ms
            dict(iso_centroids=(0.15e-3, 0.2e-3, 3.0e-3)),                            # HF fuori classe
            dict(iso_centroids=(0.4e-3, 1.0e-3, 3.0e-3)),                             # RF fuori classe
            dict(iso_thresholds=(1.5e-3, 1.0e-3), iso_centroids=(0.5e-3, 1.25e-3, 3e-3)),  # non crescenti
            dict(iso_centroid_nrf=0.2e-3)]                                            # NRF sotto la soglia RF
    for kw in casi:
        try:
            DBSI_Adaptive(**kw)
        except ValueError:
            continue
        raise AssertionError(f'accettato: {kw}')
    m = DBSI_Adaptive(**HAN)
    assert m.iso_centroid_nrf is None
    img, mask, nm = _volume((0.4, 0.35, 0.25), (0.5e-3, 1.25e-3, 3.0e-3))
    try:
        _fit(DBSI_Adaptive(force_n_iso=2, **HAN, **LAM), img, mask, nm)
    except ValueError as e:
        assert 'iso_centroid_nrf' in str(e)
    else:
        raise AssertionError('2-ISO con soglie non di default senza centroide NRF')
    print(f'  {len(casi)} configurazioni sbagliate rifiutate; 2-ISO chiede iso_centroid_nrf')


def test_classi_di_han():
    vero = (0.4, 0.35, 0.25)
    img, mask, nm = _volume(vero, (0.5e-3, 1.25e-3, 3.0e-3), snr=100)
    m = DBSI_Adaptive(**HAN, **LAM)
    r = _fit(m, img, mask, nm)
    nomi = DBSI_Adaptive.output_map_names(3)
    stima = [float(np.median(r[..., nomi.index(k)])) for k in
             ('restricted_fraction', 'hindered_fraction', 'water_fraction')]
    ff = float(np.median(r[..., nomi.index('fiber_fraction')]))
    print(f'  Han (0.5/1.25/3.0): vero RF/HF/WF {vero}, stimato {np.round(stima, 3).tolist()}, FF {ff:.3f}')
    assert ff < 0.05 and np.allclose(stima, vero, atol=0.05), (stima, ff)
    o = m.run_report_['options']
    assert o['iso_thresholds'] == [1.0e-3, 1.5e-3] and o['iso_centroids_stage_d'] == [0.5e-3, 1.25e-3, 3.0e-3]
    assert m.dictionary_['thresh_res'] == 1.0e-3 and m.dictionary_['thresh_wat'] == 1.5e-3
    # con le classi di default lo stesso voxel si legge diversamente: le classi contano
    r0 = _fit(DBSI_Adaptive(**LAM), img, mask, nm)
    rf0 = float(np.median(r0[..., nomi.index('restricted_fraction')]))
    print(f'  stesso voxel con le classi di default: RF {rf0:.3f}')
    assert abs(rf0 - vero[0]) > 0.1


def _record(**model):
    from dbsi_toolbox import protocol_fingerprint
    GA, GI = np.logspace(-4, 4, 40), np.logspace(-5, 1, 40); k = np.arange(40)
    mo = dict(n_dirs=27, n_ad=3, n_rd=3, anisotropy_ratio=2.0, fiber_threshold=0.15,
              iso_range=[0.0, 0.003])
    mo.update(model)
    return dict(kind='pydbsi_acquisition_calibration', format_version=1, toolbox_version='test',
                protocol=protocol_fingerprint(BV), noise=dict(snr=25.0, sigma_source='noise_map'),
                hyperparameters=dict(n_iso=6, lambda_aniso_method='gcv', concentration_gate=0.0,
                                     concentration_gate_calibrated=False,
                                     lambda_aniso=float(GA[24]), lambda_iso=float(GI[20])),
                selection=dict(lambda_iso_cap=None),
                curves=dict(lambda_aniso=dict(lambda_grid=list(GA), gcv=list(1 + 0.01 * (k - 24) ** 2)),
                            lambda_iso=dict(lambda_grid=list(GI), gcv=list(1 + 0.01 * (k - 20) ** 2))),
                model=mo)


def test_protocollo():
    from dbsi_toolbox import calibrate_protocol
    han = dict(iso_thresholds=[1.0e-3, 1.5e-3], iso_centroids=[0.5e-3, 1.25e-3, 3.0e-3],
               iso_centroid_nrf=None)
    cal = calibrate_protocol([_record(**han), _record(**han)], n_boot=10)
    m = DBSI_Adaptive.from_calibration(cal)
    assert m.iso_thresholds == (1.0e-3, 1.5e-3) and m.iso_centroids == (0.5e-3, 1.25e-3, 3.0e-3)
    try:
        DBSI_Adaptive.from_calibration(cal, iso_thresholds=(0.3e-3, 3.0e-3))
    except ValueError:
        pass
    else:
        raise AssertionError('classi ridefinite sopra una calibrazione di protocollo')
    try:
        calibrate_protocol([_record(**han), _record()], n_boot=10)
    except ValueError as e:
        assert 'iso_thresholds' in str(e)
    else:
        raise AssertionError('acquisizioni con classi diverse aggregate')
    vecchio = calibrate_protocol([_record(), _record(iso_thresholds=[0.3e-3, 3.0e-3])], n_boot=10)
    m = DBSI_Adaptive.from_calibration(vecchio)
    assert m.iso_thresholds == (0.3e-3, 3.0e-3) and m.iso_centroid_nrf == 1.5e-3
    print('  imposte da from_calibration e protette; classi diverse rifiutate; record pre-1.10 = default')


def test_cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_dbsi', CLI)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = ['--dwi', 'x', '--bval', 'x', '--bvec', 'x', '--mask', 'x', '--noise', 'x', '--out', 'x']
    args = mod.build_parser().parse_args(base + ['--iso-thresholds', '1.0', '1.5',
                                                 '--iso-centroids', '0.5', '1.25', '3.0',
                                                 '--iso-centroid-nrf', '2.0'])
    kw = mod.model_kwargs(args)
    assert np.allclose(kw['iso_thresholds'], (1.0e-3, 1.5e-3))
    assert np.allclose(kw['iso_centroids'], (0.5e-3, 1.25e-3, 3.0e-3))
    assert np.isclose(kw['iso_centroid_nrf'], 2.0e-3)
    DBSI_Adaptive(**kw)
    assert 'iso_thresholds' not in mod.model_kwargs(mod.build_parser().parse_args(base))
    r = subprocess.run([sys.executable, str(CLI)] + base + ['--protocol-calibration', 'x.json',
                                                           '--iso-thresholds', '1.0', '1.5'],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'protocol-calibration' in r.stderr, r.stderr[-300:]
    print('  um^2/ms -> mm^2/s; nessuna classe passata se non data; vietate con --protocol-calibration')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(name); fn()
    print('OK')
