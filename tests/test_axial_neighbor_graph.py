#!/usr/bin/env python
"""
Grafo dei vicini ASSIALE: le fibre vicine all'equatore non si spezzano (v1.5.1).

IL DIFETTO. `build_direction_neighbor_graph` sceglieva i vicini col prodotto
scalare SEMPLICE, sostenendo che l'emisfero risolvesse gia' l'ambiguita' di
segno. Vicino all'equatore non e' cosi': due nodi a z~0 con azimut opposti sono
lo stesso asse ma hanno prodotto ~ -1. Il resto della selezione era gia' assiale
(bacini e NMS usano |dot|), quindi una fibra equatoriale si spezzava in due
massimi locali: bacino dominante 0.55 invece di ~1, falsi crossing fino al 28%
(n_dirs 39, elevazione 20 gradi), ripartizione della FF nei crossing 0.61 invece
di 0.53. In un'acquisizione assiale l'equatore e' il piano delle fibre LR e AP.

COSA SI PROTEGGE.
  1. Struttura: il grafo e' quello per distanza assiale, e ogni coppia di nodi
     "quasi antipodali" (stesso asse) si vede come vicina.
  2. Effetto: con la Stage A vera, una fibra singola all'equatore e a 20 gradi
     ha bacino dominante >= 0.9 e nessun falso crossing.
  3. `measure_hemisphere_spacing` usa la stessa distanza assiale.

    python tests/test_axial_neighbor_graph.py
"""
import sys
import numpy as np

import dbsi_toolbox
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
from dbsi_toolbox.core.basis import (generate_fibonacci_sphere_hemisphere,
                                     generate_exhaustive_diffusivity_pairs,
                                     generate_anchored_isotropic_grid,
                                     build_design_matrix_exhaustive)
from dbsi_toolbox.core.solvers import (build_direction_neighbor_graph, select_dominant_directions,
                                       measure_hemisphere_spacing, nnls_coordinate_descent,
                                       compute_regularization_matrix)


def test_struttura_assiale():
    for n in (39, 62):
        fd = generate_fibonacci_sphere_hemisphere(n)
        k = M._default_direction_peak_k(n)
        nb = build_direction_neighbor_graph(fd, k=k)
        ad = np.abs(fd @ fd.T); np.fill_diagonal(ad, -2.0)
        atteso = np.argsort(-ad, axis=1)[:, :k]
        # confronto per insiemi (i pareggi possono permutare l'ordine)
        ok = all(set(nb[i]) == set(atteso[i]) or
                 np.allclose(np.sort(ad[i, nb[i]]), np.sort(ad[i, atteso[i]]))
                 for i in range(n))
        assert ok, f'n_dirs {n}: il grafo non e quello per distanza assiale'
        # coppie equatoriali "stesso asse": prodotto semplice < -0.9
        raw = fd @ fd.T
        coppie = [(i, j) for i in range(n) for j in range(n) if i < j and raw[i, j] < -0.9]
        viste = sum((j in nb[i]) or (i in nb[j]) for i, j in coppie)
        print(f'  n_dirs {n}: {len(coppie)} coppie equatoriali sullo stesso asse, '
              f'{viste} riconosciute come vicine')
        assert coppie, 'nessuna coppia equatoriale: il test non misura niente'
        assert viste == len(coppie), 'coppie sullo stesso asse ignorate dal grafo'


def test_spaziatura_assiale():
    fd = generate_fibonacci_sphere_hemisphere(62)
    ad = np.abs(fd @ fd.T); np.fill_diagonal(ad, -2.0)
    atteso = float(np.mean(np.arccos(np.clip(ad.max(axis=1), -1, 1))))
    got = measure_hemisphere_spacing(fd)
    print(f'  spaziatura media {np.degrees(got):.2f} gradi (assiale {np.degrees(atteso):.2f})')
    assert abs(got - atteso) < 1e-12


def _protocollo(seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.] * 9; bvecs = [np.zeros(3)] * 9
    for b, n in ((500, 12), (1000, 20), (1500, 21), (2000, 30)):
        for _ in range(n):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvals.append(float(b)); bvecs.append(v)
    return np.array(bvals), np.vstack(bvecs)


def test_fibra_equatoriale_intera():
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(1)
    n_dirs = 39                                   # il caso peggiore misurato
    fd = generate_fibonacci_sphere_hemisphere(n_dirs)
    pairs = generate_exhaustive_diffusivity_pairs()
    iso = generate_anchored_isotropic_grid(d_min=0.1e-3, d_max=5e-3, n_steps=6,
                                           thresh_res=0.3e-3, thresh_wat=3e-3)
    A = build_design_matrix_exhaustive(bvals, bvecs, fd, pairs, iso)
    n_an = n_dirs * len(pairs)
    AtA = compute_regularization_matrix(A.T @ A, n_an, 8.376776400682925, 0.017012542798525893)
    nb = build_direction_neighbor_graph(fd, k=M._default_direction_peak_k(n_dirs))
    sep = float(np.cos(np.radians(M._DEFAULT_MIN_SEPARATION_DEG)))
    for el in (0.0, 20.0):
        falsi, basini = 0, []
        for t in range(60):
            az = rng.uniform(0, 2 * np.pi); e = np.radians(el)
            u = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
            c2 = (bvecs @ u) ** 2
            S = (0.5 * np.exp(-bvals * (0.35e-3 + 1.35e-3 * c2)) + 0.1 * np.exp(-bvals * 0.15e-3)
                 + 0.4 * np.exp(-bvals * 1.0e-3))
            s = 1 / 26.0
            y = np.abs(S + rng.normal(0, s, S.shape) + 1j * rng.normal(0, s, S.shape))
            w, _ = nnls_coordinate_descent(AtA, A.T @ y, 0.0)
            wa = np.ascontiguousarray(w[:n_an])
            di, dw = select_dominant_directions(
                wa, n_dirs, len(pairs), nb, fd, max_directions=2, min_weight_fraction=0.05,
                min_separation_cos=sep, min_peak_ratio=M._DEFAULT_MIN_PEAK_RATIO,
                min_dominant_concentration=0.0)
            falsi += int(np.sum(di >= 0) == 2)
            basini.append(dw[0] / max(wa.sum(), 1e-12))
        print(f'  elevazione {el:>4.0f} gradi: falsi crossing {falsi}/60, '
              f'bacino dominante mediano {np.median(basini):.2f}')
        assert np.median(basini) >= 0.9, (
            f'fibra a {el} gradi spezzata: bacino dominante {np.median(basini):.2f}')
        assert falsi == 0, f'{falsi}/60 falsi crossing su fibra singola a {el} gradi'


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 5, 1), f'attesa >= 1.5.1, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_struttura_assiale, test_spaziatura_assiale,
               test_fibra_equatoriale_intera):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
