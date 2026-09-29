#!/usr/bin/env python
"""
fit_r2 / fit_rmse ricostruiscono il modello FITTATO: centroidi di Stage D (v1.5.2).

IL DIFETTO. Stage D stima le frazioni coi centroidi fissi (0.15, 1.0, 3.0)e-3;
`compute_fit_quality` ricostruiva invece con 0.15e-3, 3.05e-3 e un D_hindered
ricavato da `mean_iso_adc`, canale che Stage D non sovrascrive (viene dallo
spettro di Stage A). R2 e RMSE misuravano quindi un modello mai fittato. Il peso
maggiore cadeva sui voxel senza fibra, dove tutto il segnale e' isotropo.

COSA SI PROTEGGE.
  1. Un segnale generato ESATTAMENTE dal modello di Stage D ha RMSE ~ 0 con la
     ricostruzione di default, qualunque cosa contenga mean_iso_adc; con quella
     vecchia ('recovered') no.
  2. Nel fit vero il canale fit_rmse e' la ricostruzione coi centroidi di Stage D,
     e il run report lo dichiara; con iso_resolve=False resta 'recovered'.
  3. Un modo sconosciuto o un numero sbagliato di centroidi viene rifiutato.
  4. Una fibra RILEVATA a cui Stage D ha portato la FF sotto `fiber_threshold`
     resta nella ricostruzione: fino alla 1.5.1 veniva omessa, gonfiando il
     residuo e rendendolo dipendente dalla soglia.

    python tests/test_fit_quality_centroids.py
"""
import io, sys, contextlib
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive, compute_fit_quality
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M

NOMI = DBSI_Adaptive.output_map_names(3)
IX = {n: i for i, n in enumerate(NOMI)}


def _protocollo(n_b0=2, shells=(500., 1000., 2000., 3000.), per_shell=8, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * n_b0; bvecs = [np.zeros(3)] * n_b0
    for b in shells:
        for _ in range(per_shell):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v); bvals.append(float(b))
    return np.array(bvals), np.vstack(bvecs)


def test_segnale_esatto_di_stage_d():
    bvals, bvecs = _protocollo()
    Dr, Dh, Dw = M._ISO_RESOLVE_D_3ISO
    u = np.array([0.6, 0.0, 0.8]); c2 = (bvecs @ u) ** 2
    casi = [dict(ff=0.0, rf=0.2, hf=0.5, wf=0.3),            # senza fibra
            dict(ff=0.5, rf=0.1, hf=0.3, wf=0.1)]            # mono-fibra
    data = np.zeros((len(casi), 1, 1, len(bvals)), np.float32)
    res = np.full((len(casi), 1, 1, len(NOMI)), np.nan, np.float32)
    for i, c in enumerate(casi):
        S = c['rf'] * np.exp(-bvals * Dr) + c['hf'] * np.exp(-bvals * Dh) + c['wf'] * np.exp(-bvals * Dw)
        r = res[i, 0, 0]
        r[IX['fiber_fraction']] = c['ff']; r[IX['restricted_fraction']] = c['rf']
        r[IX['hindered_fraction']] = c['hf']; r[IX['water_fraction']] = c['wf']
        r[IX['mean_iso_adc']] = 0.6e-3          # spettro di Stage A: NON coerente coi centroidi
        if c['ff'] > 0:
            S = S + c['ff'] * np.exp(-bvals * (0.3e-3 + 1.4e-3 * c2))
            r[IX['n_fiber_populations']] = 1
            r[IX['fiber_fraction_pop1']] = c['ff']
            r[IX['axial_diffusivity_pop1']] = 1.7e-3; r[IX['radial_diffusivity_pop1']] = 0.3e-3
            r[IX['dir1_x']], r[IX['dir1_y']], r[IX['dir1_z']] = u
        data[i, 0, 0] = 1000 * S
    mask = np.ones(data.shape[:3], bool)
    with contextlib.redirect_stdout(io.StringIO()):
        _, rm_new = compute_fit_quality(data, bvals, bvecs, mask, res, 3)
        _, rm_old = compute_fit_quality(data, bvals, bvecs, mask, res, 3, iso_centroids='recovered')
    print(f'  RMSE senza fibra: nuova {rm_new[0,0,0]:.2e}  vecchia {rm_old[0,0,0]:.2e}')
    print(f'  RMSE mono-fibra:  nuova {rm_new[1,0,0]:.2e}  vecchia {rm_old[1,0,0]:.2e}')
    assert np.all(rm_new < 1e-5), 'il modello di Stage D non viene ricostruito esattamente'
    assert rm_old[0, 0, 0] > 1e-3, 'la ricostruzione vecchia dovrebbe lasciare un residuo'


def test_fibra_sotto_soglia_dopo_stage_d():
    """Fibra RILEVATA (n_pop=1, tensore e direzione) ma con FF finale 0.10 < 0.15:
    il modello la fitta, quindi la ricostruzione deve includerla."""
    bvals, bvecs = _protocollo()
    Dr, Dh, Dw = M._ISO_RESOLVE_D_3ISO
    u = np.array([0.0, 0.6, 0.8]); c2 = (bvecs @ u) ** 2
    ff, rf, hf, wf = 0.10, 0.15, 0.50, 0.25
    S = (rf * np.exp(-bvals * Dr) + hf * np.exp(-bvals * Dh) + wf * np.exp(-bvals * Dw)
         + ff * np.exp(-bvals * (0.3e-3 + 1.4e-3 * c2)))
    data = (1000 * S)[None, None, None, :].astype(np.float32)
    res = np.full((1, 1, 1, len(NOMI)), np.nan, np.float32); r = res[0, 0, 0]
    r[IX['fiber_fraction']] = ff; r[IX['restricted_fraction']] = rf
    r[IX['hindered_fraction']] = hf; r[IX['water_fraction']] = wf; r[IX['mean_iso_adc']] = 1e-3
    r[IX['n_fiber_populations']] = 1; r[IX['fiber_fraction_pop1']] = ff
    r[IX['axial_diffusivity_pop1']] = 1.7e-3; r[IX['radial_diffusivity_pop1']] = 0.3e-3
    r[IX['dir1_x']], r[IX['dir1_y']], r[IX['dir1_z']] = u
    mk = np.ones((1, 1, 1), bool)
    out = {}
    for thr in (0.15, 0.05):
        with contextlib.redirect_stdout(io.StringIO()):
            _, rm = compute_fit_quality(data, bvals, bvecs, mk, res, 3, fiber_threshold=thr)
        out[thr] = float(rm[0, 0, 0])
    print(f'  FF finale 0.10: RMSE a soglia 0.15 {out[0.15]:.2e}, a soglia 0.05 {out[0.05]:.2e}')
    assert out[0.15] < 1e-5, 'fibra fittata omessa dalla ricostruzione perche sotto soglia'
    assert out[0.15] == out[0.05], 'la ricostruzione dipende ancora dalla soglia'


def _fit(iso_resolve):
    bvals, bvecs = _protocollo()
    rng = np.random.default_rng(3)
    S = 0.6 * np.exp(-bvals * 0.8e-3) + 0.4 * np.exp(-bvals * 2.5e-3)
    d = np.abs(1000 * S + rng.normal(0, 20, (4, 4, 2, len(bvals)))).astype(np.float32)
    mk = np.ones(d.shape[:3], bool)
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893,
                      n_iso=6, iso_resolve=iso_resolve)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(d, bvals, bvecs, mk, run_calibration=False)
    return m, res, mode, d, bvals, bvecs, mk


def test_fit_vero():
    for iso_resolve, modo, atteso in ((True, M._ISO_RESOLVE_D_3ISO, 'stage_d_fixed'),
                                      (False, 'recovered', 'recovered_from_mean_iso_adc')):
        m, res, mode, d, bvals, bvecs, mk = _fit(iso_resolve)
        with contextlib.redirect_stdout(io.StringIO()):
            _, rm = compute_fit_quality(d, bvals, bvecs, mk, res, mode, iso_centroids=modo)
        diff = np.nanmax(np.abs(rm - res[..., IX['fit_rmse']]))
        rep = m.run_report_['fit_quality_reference']['iso_centroids']
        print(f'  iso_resolve={iso_resolve}: canale vs ricostruzione {modo!r}: max |diff| '
              f'{diff:.1e}   report: {rep}')
        assert diff < 1e-6, 'fit_rmse non e la ricostruzione attesa'
        assert rep == atteso


def test_rifiuti():
    bvals, bvecs = _protocollo()
    d = np.ones((1, 1, 1, len(bvals)), np.float32); res = np.zeros((1, 1, 1, len(NOMI)), np.float32)
    mk = np.ones((1, 1, 1), bool)
    for kw in (dict(iso_centroids='stage-d'), dict(iso_centroids=(0.15e-3, 1e-3))):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                compute_fit_quality(d, bvals, bvecs, mk, res, 3, **kw)
        except ValueError as e:
            print(f'  rifiutato {kw}: {e}')
            continue
        raise AssertionError(f'{kw} accettato')


def test_versione():
    v = tuple(int(x) for x in dbsi_toolbox.__version__.split('.')[:3])
    assert v >= (1, 5, 2), f'attesa >= 1.5.2, trovata {dbsi_toolbox.__version__}'
    print(f'  dbsi_toolbox {dbsi_toolbox.__version__}')


if __name__ == '__main__':
    falliti = 0
    for fn in (test_versione, test_segnale_esatto_di_stage_d, test_fibra_sotto_soglia_dopo_stage_d,
               test_fit_vero, test_rifiuti):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
