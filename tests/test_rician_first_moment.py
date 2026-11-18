#!/usr/bin/env python
"""
Correzione Rician sul segnale DENOISATO: inversione esatta del primo momento (v1.12.0).

PERCHE'. L'ingresso del toolbox e' il DWI denoisato con MP-PCA: ogni valore e' circa
E[M], la MEDIA della Rice, non una singola estrazione rumorosa. La correzione usata fino
alla 1.11.1, sqrt(max(S^2 - 2 sigma^2, 0)), e' quella del SECONDO momento: giusta sulle
misure grezze, sui dati denoisati toglie troppo (-37% a A = sigma, -11% a 2 sigma). Sul
fantoccio a 13 scenari con dwidenoise vero (SNR 26) spingeva la RF della WM a -0.006/-0.008.

COSA SI PROTEGGE.
  1. L'inversione: E[M]^-1(E[M](A)) = A su tutto l'intervallo (tabella fino a 60 sigma,
     forma asintotica oltre); 0 al pavimento sigma*sqrt(pi/2) e sotto.
  2. Il default e' 'first_moment' e il run report lo dichiara.
  3. Il motivo: un voxel isotropo con S = E[M] esatto e modello esatto restituisce le
     frazioni vere con 'first_moment'; la 'clamp' abbassa la RF (a SNR 10, 0.30 -> ~0.27).
  4. La calibrazione registra la correzione; un protocollo non mescola correzioni; una
     calibrazione scritta prima della 1.12.0 vale come 'clamp' e il fit lo dichiara.

    python tests/test_rician_first_moment.py
"""
import io, sys, contextlib, warnings
import numpy as np

import dbsi_toolbox
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox import DBSI_Adaptive

warnings.filterwarnings('ignore', category=RuntimeWarning)


def _protocollo(seed=0):
    rng = np.random.default_rng(seed)
    bv, bc = [0.0] * 2, [np.zeros(3)] * 2
    for b in (500., 1000., 2000., 3000.):
        for _ in range(8):
            v = rng.normal(size=3)
            bc.append(v / np.linalg.norm(v)); bv.append(b)
    return np.array(bv), np.vstack(bc)


def _voxel_isotropo(bv, frazioni, snr, shape=(3, 3, 2)):
    """S = E[M] esatto (MP-PCA ideale) di un voxel con tre compartimenti sui centroidi di
    Stage D: il modello e' esatto, quindi l'unico errore possibile e' la correzione."""
    rf, hf, wf = frazioni
    sig = 1000.0 / snr
    A = 1000 * (rf * np.exp(-bv * 0.15e-3) + hf * np.exp(-bv * 1.0e-3) + wf * np.exp(-bv * 3.0e-3))
    S = sig * M._rician_mean(A / sig)
    d = np.broadcast_to(S, shape + (len(bv),)).astype(np.float32).copy()
    return d, np.ones(shape, bool), np.full(shape, sig)


def _fit(d, bv, bc, mk, noise, **kw):
    m = DBSI_Adaptive(lambda_aniso=1.0, lambda_iso=0.01, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(d, bv, bc, mk, noise_map=noise, **kw)
    n = DBSI_Adaptive.output_map_names(mode)
    g = lambda c: float(np.nanmedian(res[..., n.index(c)]))
    return m, {k: g(c) for k, c in (('RF', 'restricted_fraction'), ('HF', 'hindered_fraction'),
                                    ('WF', 'water_fraction'))}


def test_inversione_esatta():
    a = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 59.9, 80.0, 500.0])
    err = np.abs(M._rician_mean_inverse(M._rician_mean(a)) - a)
    print(f'  errore massimo di andata e ritorno: {err.max():.2e} sigma')
    assert err.max() < 1e-4, err
    floor = np.sqrt(np.pi / 2)
    assert abs(M._rician_mean(np.array([0.0]))[0] - floor) < 1e-12, 'E[M](0) deve valere sqrt(pi/2)'
    assert np.all(M._rician_mean_inverse(np.array([0.5, 1.0, floor])) == 0.0), 'sotto il pavimento -> 0'


def test_default_e_dichiarato():
    bv, bc = _protocollo()
    d, mk, noise = _voxel_isotropo(bv, (0.3, 0.2, 0.5), snr=15)
    m, _ = _fit(d, bv, bc, mk, noise)
    rc = m.run_report_['rician_clamp']
    print(f"  default: {m.rician_correction_}; run report: correction={rc.get('correction')}, "
          f"misure azzerate {100 * rc['fraction_of_measurements']:.2f}%")
    assert m.rician_correction_ == 'first_moment'
    assert rc.get('correction') == 'first_moment'
    assert 'calibration_correction' not in rc, 'nessuna calibrazione di protocollo: niente avviso'


def test_frazioni_esatte_su_segnale_denoisato():
    bv, bc = _protocollo()
    for snr in (10, 25):
        for vero in ((0.3, 0.2, 0.5), (0.5, 0.3, 0.2)):
            d, mk, noise = _voxel_isotropo(bv, vero, snr)
            _, fm = _fit(d, bv, bc, mk, noise)
            _, cl = _fit(d, bv, bc, mk, noise, _rician_correction='clamp')
            e_fm = max(abs(fm[k] - v) for k, v in zip(('RF', 'HF', 'WF'), vero))
            e_cl = max(abs(cl[k] - v) for k, v in zip(('RF', 'HF', 'WF'), vero))
            print(f'  SNR {snr:2d} vero RF/HF/WF {vero}: first_moment RF {fm["RF"]:.3f} '
                  f'(err max {e_fm:.3f}) | clamp RF {cl["RF"]:.3f} (err max {e_cl:.3f})')
            assert e_fm < 0.005, f'first_moment non recupera le frazioni vere: {fm}'
            assert cl['RF'] < vero[0] - 0.002, 'la clamp dovrebbe abbassare la RF su dati denoisati'
            assert e_cl > e_fm


def test_calibrazione_registra_la_correzione():
    from dbsi_toolbox import calibrate_protocol
    bv, bc = _protocollo()
    rng = np.random.default_rng(3)
    S = 0.6 * np.exp(-bv * 0.8e-3) + 0.4 * np.exp(-bv * 2.5e-3)
    d = np.abs(1000 * S + rng.normal(0, 30, (4, 4, 2, len(bv)))).astype(np.float32)
    mk, noise = np.ones(d.shape[:3], bool), np.full(d.shape[:3], 30.0)
    with contextlib.redirect_stdout(io.StringIO()):
        rec = DBSI_Adaptive().calibrate(d, bv, bc, mk, n_calibration_voxels=30, noise_map=noise)
    print(f"  record: model.rician_correction = {rec['model'].get('rician_correction')}")
    assert rec['model'].get('rician_correction') == 'first_moment'

    # un record senza il campo (scritto prima della 1.12.0) vale 'clamp': non si mescola
    vecchio = {**rec, 'model': {k: v for k, v in rec['model'].items() if k != 'rician_correction'}}
    try:
        calibrate_protocol([rec, vecchio], n_boot=10)
    except ValueError as e:
        print(f'  correzioni mescolate rifiutate: {str(e)[:70]}')
    else:
        raise AssertionError('calibrazione di protocollo con correzioni diverse accettata')

    # calibrazione "vecchia" (solo clamp): si usa, ma il fit lo dichiara
    cal_vecchia = calibrate_protocol([vecchio, vecchio], n_boot=10)
    assert cal_vecchia['model']['rician_correction'] == 'clamp'
    m = DBSI_Adaptive.from_calibration(cal_vecchia)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.fit(d, bv, bc, mk, noise_map=noise)
    rc = m.run_report_['rician_clamp']
    print(f"  calibrazione pre-1.12: correction={rc['correction']}, "
          f"calibration_correction={rc.get('calibration_correction')}")
    assert rc['correction'] == 'first_moment' and rc.get('calibration_correction') == 'clamp'
    assert "computed with the 'clamp' Rician correction" in buf.getvalue()


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')
    assert v >= (1, 12, 0)


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_inversione_esatta, test_default_e_dichiarato,
               test_frazioni_esatte_su_segnale_denoisato, test_calibrazione_registra_la_correzione):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
