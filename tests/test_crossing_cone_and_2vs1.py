#!/usr/bin/env python
"""
Cono sulle direzioni dei crossing e test 2-contro-1 sul numero di popolazioni (v1.9.0).

I DUE DIFETTI (simulazioni 1.8.1, P3, 2026-10-14).

1. Nei crossing le direzioni restavano sui nodi della griglia di Stage A (~16.6 gradi
   sul P3): con orientamenti casuali l'errore mediano era 8-11 gradi, contro 1.4 della
   fibra singola, che passa per il cono. `crossing_dir_refine` applica il cono a una
   popolazione alla volta, rifittando i tensori a ogni candidato.
2. Il numero di popolazioni lo decideva `min_peak_ratio` = 0.35, un taglio a mano su
   distribuzioni sovrapposte: seconde fibre false nel 40-55% delle fibre singole larghe
   o deboli, meta' dei crossing sbilanciati letti come fibra singola, a ogni SNR.
   `two_vs_one_threshold` lo sostituisce con la scelta di modello: il crossing resta
   solo se abbassa il residuo di Stage D di almeno soglia * sigma^2.

COSA SI PROTEGGE.
  1. Il cono non peggiora mai il residuo e avvicina le direzioni vere.
  2. Il cono non porta due popolazioni piu' vicine della separazione minima.
  3. Il passo 2-contro-1 tiene il crossing vero e declassa quello falso.
  4. Con soglia altissima nessun crossing sopravvive al fit intero; opzioni validate.

    python tests/test_crossing_cone_and_2vs1.py
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np

import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core import solvers as S

DATI = Path(__file__).resolve().parent / 'data'
BV = np.loadtxt(DATI / 'p3_like.bval').ravel().astype(float)
BC = np.loadtxt(DATI / 'p3_like.bvec')
BC = BC.T if BC.shape[0] == 3 and BC.shape[1] != 3 else BC
_n = np.linalg.norm(BC, axis=1, keepdims=True); _n[_n == 0] = 1.0
BC = BC / _n
ISO_D = np.array(M._ISO_RESOLVE_D_3ISO)
ISO_F = np.array([0.10, 0.15, 0.05])


def _dir(theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def _ruota(u, gradi, asse=(1.0, 0.0, 0.0)):
    w = np.cross(u, asse); w /= np.linalg.norm(w)
    a = np.radians(gradi)
    return np.cos(a) * u + np.sin(a) * w


def _ang(a, b):
    return np.degrees(np.arccos(min(1.0, abs(float(a @ b)))))


def _segnale(fibre):
    s = sum(f * np.exp(-BV * d) for f, d in zip(ISO_F, ISO_D))
    for f, u, ad, rd in fibre:
        c = BC @ u
        s = s + f * np.exp(-BV * (rd + (ad - rd) * c * c))
    return s


def test_cono_non_peggiora_e_avvicina():
    u, v = _dir(65, 0), _dir(65, 90)
    fr = np.array([0.45, 0.25]); AD = np.array([1.7e-3, 1.6e-3]); RD = np.array([0.3e-3, 0.4e-3])
    y = _segnale([(fr[0], u, AD[0], RD[0]), (fr[1], v, AD[1], RD[1])])
    iso = sum(f * np.exp(-BV * d) for f, d in zip(ISO_F, ISO_D))
    start = np.array([_ruota(u, 8.0), _ruota(v, 7.0, (0.0, 0.0, 1.0))])
    a0, r0 = S.estimate_AD_RD_mrds(BV, BC, y, start, fr, iso)
    sse0 = S._mrds_sse(BV, BC, y, start, fr, iso, a0, r0)
    c1, n1, c2, n2 = S.compute_cone_refinement_schedule(np.radians(16.6), np.radians(1.0))
    d, a, r = S.refine_crossing_directions_cone(BV, BC, y, start, fr, iso, a0, r0,
                                                np.cos(np.radians(35)), c1, n1, c2, n2, 10, 2)
    sse = S._mrds_sse(BV, BC, y, d, fr, iso, a, r)
    assert sse <= sse0, (sse, sse0)
    e0 = (_ang(start[0], u), _ang(start[1], v)); e1 = (_ang(d[0], u), _ang(d[1], v))
    assert e1[0] < e0[0] and e1[1] < e0[1], (e0, e1)
    assert max(e1) < 4.0, e1
    print(f'  errore {e0[0]:.1f}/{e0[1]:.1f} -> {e1[0]:.1f}/{e1[1]:.1f} gradi; SSE {sse0:.2e} -> {sse:.2e}')


def test_cono_rispetta_la_separazione():
    # due fibre vere a 25 gradi, partenza a 40: senza guardia il cono le avvicinerebbe
    u = _dir(90, 0); v = _dir(90, 25)
    fr = np.array([0.35, 0.35]); AD = np.array([1.7e-3, 1.7e-3]); RD = np.array([0.3e-3, 0.3e-3])
    y = _segnale([(fr[0], u, AD[0], RD[0]), (fr[1], v, AD[1], RD[1])])
    iso = sum(f * np.exp(-BV * d) for f, d in zip(ISO_F, ISO_D))
    start = np.array([_dir(90, -8), _dir(90, 32)])
    a0, r0 = S.estimate_AD_RD_mrds(BV, BC, y, start, fr, iso)
    c1, n1, c2, n2 = S.compute_cone_refinement_schedule(np.radians(16.6), np.radians(1.0))
    sep = np.cos(np.radians(35))
    d, _, _ = S.refine_crossing_directions_cone(BV, BC, y, start, fr, iso, a0, r0, sep,
                                                c1, n1, c2, n2, 10, 2)
    ang = _ang(d[0], d[1])
    assert ang >= 35.0 - 1e-6, ang
    print(f'  separazione dopo il cono: {ang:.1f} gradi (minimo 35)')


def _array_voxel(fibre):
    """Una riga di uscita del toolbox (canali interni) con le fibre date."""
    a = np.full((1, 1, 1, M._N_CHANNELS_INTERNAL), np.nan, np.float32)
    a[..., :M._C_NPOP] = 0.0
    a[..., M._C_NPOP] = len(fibre)
    for k, (f, u, ad, rd) in enumerate(fibre):
        c0, cad, crd = (M._C_DIR1, M._C_AD1, M._C_RD1) if k == 0 else (M._C_DIR2, M._C_AD2, M._C_RD2)
        a[0, 0, 0, c0:c0 + 3] = u; a[0, 0, 0, cad] = ad; a[0, 0, 0, crd] = rd
    return a


def test_passo_2vs1_sceglie():
    u, v, w = _dir(65, 0), _dir(65, 90), _dir(20, 45)
    sigma = 1000 / 26
    casi = {
        # fibra singola vera; il crossing aggiunge una seconda popolazione inesistente
        'falso': ([(0.55, u, 1.7e-3, 0.3e-3)],
                  [(0.55, u, 1.7e-3, 0.3e-3), (0.1, w, 1.7e-3, 0.3e-3)],
                  [(0.55, u, 1.7e-3, 0.3e-3)], 1),
        # crossing vero; l'alternativa a fibra singola perde la seconda popolazione
        'vero': ([(0.35, u, 1.6e-3, 0.4e-3), (0.35, v, 1.6e-3, 0.4e-3)],
                 [(0.35, u, 1.6e-3, 0.4e-3), (0.35, v, 1.6e-3, 0.4e-3)],
                 [(0.35, u, 1.6e-3, 0.4e-3)], 2),
    }
    b0 = BV < 100
    for nome, (vere, due, uno, atteso) in casi.items():
        y = _segnale(vere)
        data = (1000 * y / np.mean(y[b0])).astype(np.float32)[None, None, None, :]
        out = _array_voxel(due); alt = _array_voxel(uno)
        stat = np.full((1, 1, 1), np.nan, np.float32)
        M._two_vs_one_pass(data, np.array([[0, 0, 0]]), BV, BC, M.B0_THRESHOLD, ISO_D,
                           np.full((1, 1, 1), sigma), 12.0, out, alt, stat)
        npop = int(out[0, 0, 0, M._C_NPOP])
        assert npop == atteso, (nome, npop, float(stat[0, 0, 0]))
        print(f'  {nome}: stat21 {float(stat[0, 0, 0]):.1f} -> {npop} popolazioni')


def test_opzioni_validate():
    for cattiva in (0.0, -3.0):
        try:
            DBSI_Adaptive(two_vs_one_threshold=cattiva)
        except ValueError:
            pass
        else:
            raise AssertionError(f'two_vs_one_threshold={cattiva} accettato')
    try:
        DBSI_Adaptive(two_vs_one_threshold=12.0, iso_resolve=False)
    except ValueError:
        pass
    else:
        raise AssertionError('two_vs_one_threshold senza Stage D accettato')
    m = DBSI_Adaptive(two_vs_one_threshold=12.0, crossing_dir_refine=True)
    assert m.two_vs_one_threshold == 12.0 and m.crossing_dir_refine
    assert DBSI_Adaptive().two_vs_one_threshold is None and not DBSI_Adaptive().crossing_dir_refine
    print('  soglia <= 0 e test senza Stage D rifiutati; default spenti')


def test_soglia_altissima_niente_crossing():
    """Fit intero: con soglia enorme ogni candidato torna fibra singola, e il report lo dice."""
    rng = np.random.default_rng(3)
    img = np.zeros((4, 4, 1, len(BV)), np.float32)
    for i, j in np.ndindex(4, 4):
        u = _dir(rng.uniform(30, 90), rng.uniform(0, 180)); v = _ruota(u, 75.0)
        y = _segnale([(0.35, u, 1.6e-3, 0.4e-3), (0.35, v, 1.6e-3, 0.4e-3)])
        img[i, j, 0] = 1000 * np.abs(y + rng.normal(0, 1 / 40, y.shape))
    mask = np.ones(img.shape[:3], bool)
    m = DBSI_Adaptive(lambda_aniso=8.3768, lambda_iso=0.1, n_iso=6, crossing_dir_refine=True,
                      two_vs_one_threshold=1e9, uncertainty=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(img, BV, BC, mask, run_calibration=False,
                       noise_map=np.full(mask.shape, 1000 / 40))
    npop = res[..., M._C_NPOP]
    assert not np.any(npop >= 2), np.unique(npop[np.isfinite(npop)])
    t = m.two_vs_one_
    assert t['n_candidates'] > 0 and t['n_kept_crossing'] == 0, t
    assert m.run_report_['two_vs_one']['threshold'] == 1e9
    assert m.run_report_['options']['two_vs_one_threshold'] == 1e9
    print(f"  {t['n_candidates']} candidati, 0 crossing tenuti; stat p50 {t['stat_p50']:.1f}")


if __name__ == '__main__':
    falliti = 0
    for fn in (test_cono_non_peggiora_e_avvicina, test_cono_rispetta_la_separazione,
               test_passo_2vs1_sceglie, test_opzioni_validate,
               test_soglia_altissima_niente_crossing):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1
            print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
