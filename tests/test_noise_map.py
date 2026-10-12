#!/usr/bin/env python
"""
Mappa di rumore termico in ingresso (v1.7.1, `fit(noise_map=...)`, CLI `--noise`).

IL PROBLEMA (notebook 12, 2026-10-02). Il sigma del modello veniva dalla variabilita' temporale dei
b=0, che contiene anche pulsazione e movimento: sui dati veri 1.3-2.2 volte il rumore termico, in
misura diversa per soggetto. La correzione Rician sottrae 2 sigma^2: con sigma gonfiato sottrae
troppo e abbassa RF. Il rumore termico viene dalla mappa di dwidenoise (MP-PCA), divisa per il campo
di bias di N4. Dalla 1.8.0 il tSNR non si calcola piu' e i b=0 servono solo a S0: ne basta uno.

COSA SI PROTEGGE.
  1. Senza mappa, oppure con una mappa COSTANTE pari al sigma dei b=0, le uscite sono identiche.
  2. Su un fantoccio con b=0 piu' rumorosi (sigma_b0 = 1.6 x termico), con la mappa giusta la RF e'
     piu' vicina al vero che senza mappa.
  3. Mappa con griglia sbagliata o quasi tutta non valida: rifiutata. Pochi voxel non validi:
     riempiti con la mediana e contati.
  5. Con la mappa basta UN volume b=0 (v1.8.0; prima ne servivano due per il tSNR); senza b=0 si ferma.
  4. La mappa e' OBBLIGATORIA: senza, fit e calibrate si fermano (salvo _ALLOW_B0_SIGMA, solo per
     test e simulazioni), e calibrate_protocol rifiuta i record calibrati senza mappa.

    python tests/test_noise_map.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M

NOMI = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NOMI)}
LAM = dict(lambda_aniso=9.805121767832382, lambda_iso=0.015117750706156615, n_iso=6)
EXP = Path(__file__).resolve().parent / 'data'          # protocollo tipo P3 (sintetico)
S0, SIG_T, FATT_B0 = 1500.0, 1500.0 / 20, 1.6       # SNR termico 20, b=0 piu' rumorosi di 1.6x


def _dati(seed=5, n=120):
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel(); bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb
    rng = np.random.default_rng(seed); b0 = bv < 100
    d = np.zeros((n, 2, 1, bv.size), np.float32)
    for i in range(n):
        for j in range(2):
            S = 0.25 * np.exp(-bv * 0.15e-3) + 0.55 * np.exp(-bv * 1.0e-3) + 0.20 * np.exp(-bv * 3.0e-3)
            S = S0 * S
            fisio = np.where(b0, rng.normal(0, SIG_T * np.sqrt(FATT_B0 ** 2 - 1), bv.size), 0.0)
            d[i, j, 0] = np.sqrt((S + fisio + rng.normal(0, SIG_T, bv.size)) ** 2 + rng.normal(0, SIG_T, bv.size) ** 2)
    return d, bv, bc


def _fit(d, bv, bc, **kw):
    M._ALLOW_B0_SIGMA = 'noise_map' not in kw      # il confronto senza mappa serve al test 1-2
    m = DBSI_Adaptive(**LAM)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(d, bv, bc, np.ones(d.shape[:3], bool), run_calibration=False, **kw)
    return m, res


def test_mappa_costante_identica():
    d, bv, bc = _dati(n=40)
    m0, r0 = _fit(d, bv, bc)
    sb0 = m0.noise_['sigma_b0_temporal']
    m1, r1 = _fit(d, bv, bc, noise_map=np.full(d.shape[:3], sb0))
    same = np.array_equal(np.isnan(r0), np.isnan(r1)) and np.allclose(r0, r1, rtol=0, atol=0, equal_nan=True)
    st = np.allclose(m0.fiber_detection_stat_, m1.fiber_detection_stat_, rtol=0, atol=0, equal_nan=True)
    print(f"  senza mappa: sorgente {m0.noise_['sigma_source']}; mappa costante = sigma_b0: uscite identiche {same}, "
          f"statistica identica {st}")
    assert m0.noise_['sigma_source'] == 'b0_temporal' and m1.noise_['sigma_source'] == 'noise_map'
    assert same and st


def test_rf_con_sigma_termico():
    d, bv, bc = _dati()
    m0, r0 = _fit(d, bv, bc)
    m1, r1 = _fit(d, bv, bc, noise_map=np.full(d.shape[:3], SIG_T))
    rf0 = np.median(r0[..., IX['restricted_fraction']]); rf1 = np.median(r1[..., IX['restricted_fraction']])
    print(f"  SNR termico {m1.noise_['snr_thermal']:.1f} (vero {S0 / SIG_T:.0f})")
    print(f"  RF mediana (vera 0.25): sigma dei b=0 {rf0:.3f}, mappa termica {rf1:.3f}")
    assert 'tsnr' not in m1.noise_ and 'tsnr' not in m1.run_report_['noise']
    assert abs(rf1 - 0.25) < abs(rf0 - 0.25), 'la mappa termica non avvicina la RF al vero'
    assert m1.run_report_['noise']['sigma_source'] == 'noise_map'


def test_mappa_sbagliata():
    d, bv, bc = _dati(n=10)
    for nome, nm in (('griglia', np.ones((3, 3, 3))), ('quasi tutta nulla', np.zeros(d.shape[:3]))):
        try:
            _fit(d, bv, bc, noise_map=nm)
        except ValueError as e:
            print(f"  {nome}: rifiutata ({str(e)[:60]}...)")
            continue
        raise AssertionError(f'mappa sbagliata ({nome}) accettata')
    nm = np.full(d.shape[:3], SIG_T); nm[0, 0, 0] = np.nan; nm[1, 0, 0] = 0
    m, _ = _fit(d, bv, bc, noise_map=nm)
    print(f"  2 voxel non validi: riempiti {m.noise_['noise_map_filled_voxels']}")
    assert m.noise_['noise_map_filled_voxels'] == 2


def test_un_solo_b0():
    d, bv, bc = _dati(n=10)
    b0 = np.where(bv < 100)[0]
    tieni = np.setdiff1d(np.arange(bv.size), b0[1:])          # un solo b=0
    m, r = _fit(d[..., tieni], bv[tieni], bc[tieni], noise_map=np.full(d.shape[:3], SIG_T))
    print(f"  un solo b=0: fit fatto, n_b0 nel report {m.run_report_['noise']['n_b0']}")
    assert m.run_report_['noise']['n_b0'] == 1 and np.isfinite(r[..., IX['restricted_fraction']]).all()
    senza = np.setdiff1d(np.arange(bv.size), b0)
    try:
        _fit(d[..., senza], bv[senza], bc[senza], noise_map=np.full(d.shape[:3], SIG_T))
    except ValueError as e:
        print(f"  nessun b=0: fermato ({str(e)[:50]}...)")
    else:
        raise AssertionError('fit senza volumi b=0 non si e fermato')


def test_mappa_obbligatoria():
    from dbsi_toolbox.calibration.protocol import calibrate_protocol
    d, bv, bc = _dati(n=10)
    M._ALLOW_B0_SIGMA = False
    for nome, chiama in (('fit', lambda: DBSI_Adaptive(**LAM).fit(d, bv, bc, np.ones(d.shape[:3], bool),
                                                                    run_calibration=False)),
                         ('calibrate', lambda: DBSI_Adaptive().calibrate(d, bv, bc, np.ones(d.shape[:3], bool),
                                                                         n_calibration_voxels=15))):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                chiama()
        except ValueError as e:
            print(f"  {nome} senza mappa: fermato ({str(e)[:55]}...)")
            continue
        raise AssertionError(f'{nome} senza mappa non si e fermato')
    M._ALLOW_B0_SIGMA = True
    with contextlib.redirect_stdout(io.StringIO()):
        recs = [DBSI_Adaptive().calibrate(d, bv, bc, np.ones(d.shape[:3], bool), n_calibration_voxels=15)
                for _ in range(2)]
    M._ALLOW_B0_SIGMA = False
    try:
        calibrate_protocol(recs)
    except ValueError as e:
        print(f"  calibrate_protocol con record senza mappa: rifiutato ({str(e)[:50]}...)")
    else:
        raise AssertionError('calibrate_protocol ha accettato record senza mappa')
    with contextlib.redirect_stdout(io.StringIO()):
        recs = [DBSI_Adaptive().calibrate(d, bv, bc, np.ones(d.shape[:3], bool), n_calibration_voxels=15,
                                          noise_map=np.full(d.shape[:3], SIG_T)) for _ in range(2)]
    cal = calibrate_protocol(recs)
    print(f"  con la mappa: calibrazione di protocollo fatta (lambda_aniso {cal['hyperparameters']['lambda_aniso']:.3g})")


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
