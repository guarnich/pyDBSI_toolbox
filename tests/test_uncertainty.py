#!/usr/bin/env python
"""
Mappe di incertezza (v1.7.0): errore standard di Fisher di ogni mappa continua.

COSA SI PROTEGGE.
  1. Il kernel coincide con un calcolo INDIPENDENTE (Jacobiano alle differenze
     finite, direzioni in coordinate sferiche, metodo delta numerico) su una
     mono-fibra e su un crossing: stesse SE per frazioni, AD/RD/FA, tensore
     pesato ed errore angolare.
  2. Fit vero + `save_output_maps(model=m)`: `toolbox_report/uncertainty_maps/`
     contiene un `NN_<canale>_se.nii.gz` per ogni canale continuo salvato (stessa
     numerazione), l'errore angolare, il condizionamento, i flag e il README;
     la SE esiste esattamente dove esiste la mappa, ed e' >= 0; il run report
     ha la sezione.
  3. Spenta (uncertainty=False) o senza Stage D: niente cartella, e il run report
     dice perche'.
  4. Copertura su mono-fibra sana a SNR 26 (P3-like): SE mediana / sd empirica
     fra 0.8 e 1.6 per FF, RD e FA, e verita' entro 1.96 SE in >= 90% dei voxel.
     (Misura completa: experiments/uncertainty/copertura.py.)
  5. Residuo / sigma: ~1 col modello giusto (mediana fra 0.9 e 1.1, flag 16 su
     meno del 3% dei voxel); nei crossing, dove la FF di Stage A resta fissa,
     sopra quello della mono-fibra.

    python tests/test_uncertainty.py
"""
import io, os, sys, contextlib, tempfile
from pathlib import Path
import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive, save_output_maps
from dbsi_toolbox import uncertainty as U
from dbsi_toolbox.model_Niso_adaptive_ff_thr import (
    _C_FF, _C_RF, _C_HF, _C_WF, _C_NRF, _C_NPOP, _C_FF1, _C_AD1, _C_RD1, _C_FA1,
    _C_DIR1, _C_FF2, _C_AD2, _C_RD2, _C_DIR2, _C_ADW, _C_RDW, _C_FAW, _N_CHANNELS)

NOMI = DBSI_Adaptive.output_map_names(3)
IX = {n: i for i, n in enumerate(NOMI)}
LAM = dict(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
EXP = Path(__file__).resolve().parents[1] / 'experiments' / 'uncertainty'
ISO = np.array([0.15e-3, 1.0e-3, 3.0e-3])


def _p3():
    bv = np.loadtxt(EXP / 'p3_like.bval').ravel()
    bc = np.loadtxt(EXP / 'p3_like.bvec')
    bc = bc.T if bc.shape[0] == 3 else bc
    nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1
    return bv, bc / nb


def _sph(t, p):
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def test_kernel_uguale_a_calcolo_indipendente():
    bv, bc = _p3()
    snr = 26.0
    fa = lambda a, r: abs(a - r) / np.sqrt(a * a + 2 * r * r)
    for nome, fib, u in (
            ('mono', [(0.6, 1.7e-3, 0.4e-3, 0.4, 0.3)], [0.1, 0.2, 0.1]),
            ('crossing', [(0.3, 1.7e-3, 0.4e-3, np.pi / 2, 0.0),
                          (0.3, 1.5e-3, 0.8e-3, np.pi / 2, 0.45 * np.pi)], [0.1, 0.2, 0.1])):
        nf = len(fib)

        def model(th):
            S = sum(th[5 * nf + j] * np.exp(-bv * ISO[j]) for j in range(3))
            for k in range(nf):
                w, a, r, t, p = th[5 * k:5 * k + 5]
                S = S + w * np.exp(-bv * (r + (a - r) * (bc @ _sph(t, p)) ** 2))
            return S
        th = np.array(sum([list(f) for f in fib], []) + u, float)
        P = th.size
        J = np.zeros((bv.size, P))
        for i in range(P):
            h = 1e-7 if (i < 5 * nf and i % 5 in (1, 2)) else 1e-5
            e = np.zeros(P); e[i] = h
            J[:, i] = (model(th + e) - model(th - e)) / (2 * h)
        C = np.linalg.inv(J.T @ J) / snr ** 2

        def delta(f):
            g = np.zeros(P)
            for i in range(P):
                h = 1e-7 * max(abs(th[i]), 1e-3); e = np.zeros(P); e[i] = h
                g[i] = (f(th + e) - f(th - e)) / (2 * h)
            return np.sqrt(g @ C @ g)
        T = lambda t: sum(t[5 * k] for k in range(nf)) + sum(t[5 * nf:])
        W = lambda t: sum(t[5 * k] for k in range(nf))
        adw = lambda t: sum(t[5 * k] * t[5 * k + 1] for k in range(nf)) / W(t)
        rdw = lambda t: sum(t[5 * k] * t[5 * k + 2] for k in range(nf)) / W(t)
        ref = {_C_FF: delta(lambda t: W(t) / T(t)), _C_RF: delta(lambda t: t[5 * nf] / T(t)),
               _C_HF: delta(lambda t: t[5 * nf + 1] / T(t)),
               _C_WF: delta(lambda t: t[5 * nf + 2] / T(t)),
               _C_NRF: delta(lambda t: (t[5 * nf + 1] + t[5 * nf + 2]) / T(t)),
               _C_FF1: delta(lambda t: t[0] / T(t)), _C_AD1: np.sqrt(C[1, 1]),
               _C_RD1: np.sqrt(C[2, 2]), _C_FA1: delta(lambda t: fa(t[1], t[2])),
               _C_ADW: delta(adw), _C_RDW: delta(rdw),
               _C_FAW: delta(lambda t: fa(adw(t), rdw(t)))}
        if nf == 2:
            ref[_C_FF2] = delta(lambda t: t[5] / T(t))
            ref[_C_RD2] = np.sqrt(C[7, 7])
        out = np.full((1, 1, 1, _N_CHANNELS), np.nan)
        out[0, 0, 0, _C_NPOP] = nf
        out[0, 0, 0, _C_RF], out[0, 0, 0, _C_HF], out[0, 0, 0, _C_WF] = u
        out[0, 0, 0, _C_NRF] = u[1] + u[2]
        out[0, 0, 0, _C_FF] = sum(f[0] for f in fib)
        for k, (w, a, r, t, p) in enumerate(fib):
            cf, ca, cr, cd = [(_C_FF1, _C_AD1, _C_RD1, _C_DIR1), (_C_FF2, _C_AD2, _C_RD2, _C_DIR2)][k]
            out[0, 0, 0, cf], out[0, 0, 0, ca], out[0, 0, 0, cr] = w, a, r
            out[0, 0, 0, cd:cd + 3] = _sph(t, p)
        data = np.full((1, 1, 1, bv.size), 1000.0)
        r = U.compute_uncertainty(data, np.array([[0, 0, 0]]), bv, bc, 50.0, ISO, True,
                                  1000.0 / snr, out)
        worst = max(abs(r['se'][0, 0, 0, ch] / v - 1) for ch, v in ref.items())
        t0 = fib[0][3]
        ang = np.degrees(np.sqrt(C[3, 3] + np.sin(t0) ** 2 * C[4, 4]))
        worst = max(worst, abs(r['dir_se_deg'][0, 0, 0, 0] / ang - 1))
        print(f"  {nome}: scarto relativo massimo kernel/riferimento {worst:.2e} "
              f"(SE FF {r['se'][0, 0, 0, _C_FF]:.3f}, RD1 {r['se'][0, 0, 0, _C_RD1] * 1e3:.3f}e-3, "
              f"angolo {r['dir_se_deg'][0, 0, 0, 0]:.2f} gradi)")
        assert worst < 1e-3, f'{nome}: kernel diverso dal calcolo indipendente ({worst:.2e})'


def _fantoccio(snr=26.0, n=40, seed=3):
    """Due gruppi: mono-fibra sana e crossing 90, stessa composizione isotropa."""
    bv, bc = _p3()
    rng = np.random.default_rng(seed)
    iso = 0.1 * np.exp(-bv * 0.15e-3) + 0.2 * np.exp(-bv * 1.0e-3) + 0.1 * np.exp(-bv * 3.0e-3)
    f = lambda d: np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ d) ** 2))
    u, v = _sph(1.1, 0.4), _sph(1.1, 0.4 + np.pi / 2)
    d = np.zeros((2, n, 1, bv.size), np.float32)
    for g, S in enumerate((iso + 0.6 * f(u), iso + 0.3 * (f(u) + f(v)))):
        d[g, :, 0] = 1000 * np.sqrt((S + rng.normal(0, 1 / snr, (n, bv.size))) ** 2
                                    + rng.normal(0, 1 / snr, (n, bv.size)) ** 2)
    return d, bv, bc


def _fit(d, bv, bc, **kw):
    m = DBSI_Adaptive(**LAM, min_dominant_concentration=0.0, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(d, bv, bc, np.ones(d.shape[:3], bool), run_calibration=False)
    return m, res, mode


def test_fit_e_salvataggio():
    import nibabel as nib
    d, bv, bc = _fantoccio()
    m, res, mode = _fit(d, bv, bc)
    out = tempfile.mkdtemp(prefix='dbsi_unc_')
    with contextlib.redirect_stdout(io.StringIO()):
        saved = save_output_maps(res, DBSI_Adaptive.output_map_names(mode), np.eye(4), out, model=m)
    ud = os.path.join(out, 'toolbox_report', 'uncertainty_maps')
    files = set(os.listdir(ud))
    attesi = {f'{ch:02d}_{NOMI[ch]}_se.nii.gz' for ch in U.SE_CHANNELS if NOMI[ch] in saved}
    attesi |= {'dir1_angle_se_deg.nii.gz', 'dir2_angle_se_deg.nii.gz',
               'fisher_log10_condition.nii.gz', 'uncertainty_flags.nii.gz', 'README.txt',
               'residual_over_sigma.nii.gz'}
    print(f"  {len(files)} file in uncertainty_maps/ ({len(attesi)} attesi)")
    assert files == attesi, f'file diversi: mancano {attesi - files}, in piu\' {files - attesi}'
    for ch in U.SE_CHANNELS:
        nm = NOMI[ch]
        if nm not in saved:
            continue
        se = nib.load(os.path.join(ud, f'{ch:02d}_{nm}_se.nii.gz')).get_fdata()
        mappa = res[..., ch]
        # la SE esiste esattamente dove esiste la mappa (FF: dove c'e' una fibra)
        dove = np.isfinite(mappa)
        if ch == _C_FF:
            dove &= ~np.isnan(res[..., _C_NPOP]) & np.isfinite(res[..., _C_AD1])
        ok = (m.uncertainty_['flags'] & U.FLAG_ILL_CONDITIONED) == 0
        assert np.array_equal(np.isfinite(se) & ok, dove & ok), f'{nm}: dominio della SE sbagliato'
        assert np.all(se[np.isfinite(se)] >= 0), f'{nm}: SE negativa'
    rr = m.uncertainty_['residual_over_sigma'][..., 0]
    print(f"  residuo/sigma mediano: mono {np.nanmedian(rr[0]):.3f}, crossing {np.nanmedian(rr[1]):.3f}")
    assert np.nanmedian(rr[1]) > np.nanmedian(rr[0]), 'il residuo dei crossing non supera quello della mono-fibra'
    rep = open(os.path.join(out, 'toolbox_report', 'run_report.txt')).read()
    assert 'Uncertainty maps' in rep and 'median standard error' in rep
    assert m.run_report_['uncertainty']['n_voxels'] == int(np.isfinite(m.uncertainty_['log10_cond']).sum())
    med = m.run_report_['uncertainty']['median_se']
    print(f"  SE mediana: FF {med['fiber_fraction']:.3f}, RD_w "
          f"{med['radial_diffusivity_weighted'] * 1e3:.3f}e-3, angolo pop1 {med['dir1_angle_deg']:.2f} gradi")


def test_spenta_o_senza_stage_d():
    d, bv, bc = _fantoccio(n=6)
    for kw, motivo in ((dict(uncertainty=False), 'disabled'), (dict(iso_resolve=False), 'Stage D')):
        m, res, mode = _fit(d, bv, bc, **kw)
        assert m.uncertainty_ is None
        un = m.run_report_['uncertainty']
        assert un['computed'] is False and motivo in un['reason'], un
        out = tempfile.mkdtemp(prefix='dbsi_unc_off_')
        with contextlib.redirect_stdout(io.StringIO()):
            save_output_maps(res, DBSI_Adaptive.output_map_names(mode), np.eye(4), out, model=m)
        assert not os.path.exists(os.path.join(out, 'toolbox_report', 'uncertainty_maps'))
        print(f"  {kw}: nessuna cartella; report: {un['reason']}")


def test_copertura_mono_fibra():
    bv, bc = _p3()
    rng = np.random.default_rng(11)
    n, snr = 200, 26.0
    u = _sph(1.1, 0.4)
    S = (0.1 * np.exp(-bv * 0.15e-3) + 0.2 * np.exp(-bv * 1.0e-3) + 0.1 * np.exp(-bv * 3.0e-3)
         + 0.6 * np.exp(-bv * (0.4e-3 + 1.3e-3 * (bc @ u) ** 2)))
    d = (1000 * np.sqrt((S + rng.normal(0, 1 / snr, (n, bv.size))) ** 2
                        + rng.normal(0, 1 / snr, (n, bv.size)) ** 2))[:, None, None, :]
    m, res, _ = _fit(d.astype(np.float32), bv, bc)
    uno = res[..., _C_NPOP] == 1
    for ch, vero in ((_C_FF, 0.6), (_C_RD1, 0.4e-3), (_C_FA1, 1.3 / np.sqrt(1.7 ** 2 + 2 * 0.4 ** 2))):
        est = res[..., ch][uno]; se = m.uncertainty_['se'][..., ch][uno]
        rap = np.median(se) / est.std(ddof=1)
        cop = np.mean(np.abs(est - vero) <= 1.96 * se)
        print(f"  {NOMI[ch]:24s} SE/sd {rap:.2f}  copertura {cop:.2f}  (n {uno.sum()})")
        assert 0.8 <= rap <= 1.6, f'{NOMI[ch]}: SE/sd {rap:.2f} fuori da [0.8, 1.6]'
        assert cop >= 0.90, f'{NOMI[ch]}: copertura {cop:.2f} < 0.90'
    rr = m.uncertainty_['residual_over_sigma'][uno]
    f16 = np.mean((m.uncertainty_['flags'][uno] & U.FLAG_RESIDUAL) > 0)
    print(f"  residuo/sigma mediano {np.median(rr):.3f}, flag 16 su {f16:.1%}")
    assert 0.9 <= np.median(rr) <= 1.1, f'residuo/sigma {np.median(rr):.3f} col modello giusto'
    assert f16 < 0.03, f'flag 16 su {f16:.1%} col modello giusto'


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
