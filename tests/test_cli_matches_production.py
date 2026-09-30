#!/usr/bin/env python
"""
La CLI riproduce la produzione bit per bit (v1.6.1) — serve a "Data availability".

La produzione costruisce il modello con `pyDBSI_produzione/pydbsi_calibrazione.py`:
    DBSI_Adaptive(lambda_aniso, lambda_iso, n_iso, min_dominant_concentration=gate,
                  lambda_aniso_method='gcv', stagec_dir_refine=True)
    .fit(..., run_calibration=True, calibrate_concentration_gate=False)
Chi legge il paper ha solo `scripts/run_dbsi.py`. Questo test lancia la CLI su un
fantoccio NIfTI con gli stessi quattro numeri e confronta ogni mappa salvata con
il fit della ricetta di produzione.

Fino alla 1.6.0 la CLI RISCRIVEVA cinque default del costruttore (n_ad, n_rd,
anisotropy_ratio, min_weight_fraction, target_angular_resolution_deg): uguali
oggi, ma una seconda fonte di verita' che al primo cambio di default avrebbe
lasciato la CLI sui valori vecchi senza nessun segnale. E `--n-ad/--n-rd/
--anisotropy-ratio` non erano in conflitto con `--protocol-calibration`, che li
impone.

COSA SI PROTEGGE.
  1. Mappe della CLI identiche a quelle della ricetta di produzione.
  2. La CLI non passa opzioni che l'utente non ha dato (niente default copiati).
  3. --protocol-calibration rifiuta le opzioni del dizionario che gia' impone.

    python tests/test_cli_matches_production.py
"""
import io, sys, contextlib, subprocess, tempfile, importlib.util, argparse
from pathlib import Path
import numpy as np
import nibabel as nib

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive, load_data

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts' / 'run_dbsi.py'
LA, LI = 8.376776400682925, 0.017012542798525893


def _scrivi_fantoccio(d, seed=0):
    rng = np.random.default_rng(seed)
    bvals = [0.0] * 3; bvecs = [np.zeros(3)] * 3
    for b, n in ((1000., 20), (2000., 30)):
        for _ in range(n):
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v); bvals.append(b)
    bvals = np.array(bvals); bvecs = np.vstack(bvecs)
    img = np.zeros((5, 5, 2, len(bvals)), np.float32)
    for idx in np.ndindex(img.shape[:3]):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        S = (0.5 * np.exp(-bvals * (0.3e-3 + 1.4e-3 * (bvecs @ u) ** 2))
             + 0.2 * np.exp(-bvals * 1.0e-3) + 0.3 * np.exp(-bvals * 2.5e-3))
        img[idx] = 1000 * np.abs(S + rng.normal(0, 1 / 30, S.shape))
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(img, aff), d / 'dwi.nii.gz')
    nib.save(nib.Nifti1Image(np.ones(img.shape[:3], np.uint8), aff), d / 'mask.nii.gz')
    np.savetxt(d / 'dwi.bval', bvals[None], fmt='%g')
    np.savetxt(d / 'dwi.bvec', bvecs.T, fmt='%.6f')


def _cli_module():
    spec = importlib.util.spec_from_file_location('run_dbsi', CLI)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_cli_uguale_alla_produzione():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _scrivi_fantoccio(d)
        r = subprocess.run([sys.executable, str(CLI), '--dwi', str(d / 'dwi.nii.gz'),
                            '--bval', str(d / 'dwi.bval'), '--bvec', str(d / 'dwi.bvec'),
                            '--mask', str(d / 'mask.nii.gz'), '--out', str(d / 'cli'),
                            '--lambda-aniso', repr(LA), '--lambda-iso', repr(LI), '--n-iso', '6'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        data, aff, bvals, bvecs, mask = load_data(str(d / 'dwi.nii.gz'), str(d / 'dwi.bval'),
                                                  str(d / 'dwi.bvec'), str(d / 'mask.nii.gz'),
                                                  verbose=False)
        m = DBSI_Adaptive(lambda_aniso=LA, lambda_iso=LI, n_iso=6, min_dominant_concentration=0.0,
                          lambda_aniso_method='gcv', stagec_dir_refine=True)
        with contextlib.redirect_stdout(io.StringIO()):
            res, mode = m.fit(data, bvals, bvecs, mask, run_calibration=True,
                              calibrate_concentration_gate=False)
        nomi = DBSI_Adaptive.output_map_names(mode)
        files = sorted((d / 'cli').glob('[0-9][0-9]_*.nii.gz'))
        assert len(files) >= 20, f'solo {len(files)} mappe scritte dalla CLI'
        peggiore = 0.0
        for f in files:
            k = int(f.name[:2]); a = np.asarray(nib.load(f).dataobj, np.float64)
            b = res[..., k].astype(np.float64)
            assert f.name[3:-7] == nomi[k], f'{f.name} non corrisponde al canale {nomi[k]}'
            assert np.array_equal(np.isnan(a), np.isnan(b)), f'{f.name}: NaN in posti diversi'
            diff = np.nanmax(np.abs(a - b)) if np.any(~np.isnan(a)) else 0.0
            peggiore = max(peggiore, float(diff))
        print(f'  {len(files)} mappe, massimo |CLI - produzione| = {peggiore:.2e}')
        assert peggiore == 0.0, 'la CLI non riproduce la produzione'


def test_nessun_default_copiato():
    mod = _cli_module()
    ns = argparse.Namespace(force_n_iso=None, disable_direction_refinement=False,
                            disable_conc_modulation=False, disable_stagec=False,
                            disable_iso_resolve=False, protocol_calibration=None,
                            min_weight_fraction=None, target_angular_resolution_deg=None,
                            n_iso=None, lambda_aniso=None, lambda_iso=None, n_dirs=None,
                            n_ad=None, n_rd=None, anisotropy_ratio=None,
                            fiber_detection_threshold=None)
    kw = mod.model_kwargs(ns)
    copiati = [k for k in ('n_ad', 'n_rd', 'anisotropy_ratio', 'min_weight_fraction',
                           'target_angular_resolution_deg') if k in kw]
    print(f'  opzioni passate senza che l utente le desse: {copiati}')
    assert not copiati


def test_conflitto_con_calibrazione_di_protocollo():
    for opz in (['--n-ad', '4'], ['--n-rd', '4'], ['--anisotropy-ratio', '2.1']):
        r = subprocess.run([sys.executable, str(CLI), '--dwi', 'x', '--bval', 'x', '--bvec', 'x',
                            '--mask', 'x', '--out', 'x', '--protocol-calibration', 'x.json'] + opz,
                           capture_output=True, text=True)
        print(f'  {opz[0]}: exit {r.returncode}')
        assert r.returncode == 2 and 'protocol-calibration' in r.stderr, \
            f'{opz[0]} accettato insieme a --protocol-calibration'


if __name__ == '__main__':
    falliti = 0
    for fn in (test_nessun_default_copiato, test_conflitto_con_calibrazione_di_protocollo,
               test_cli_uguale_alla_produzione):
        print(f'\n=== {fn.__name__} ===')
        try:
            fn(); print('  [ok]')
        except AssertionError as e:
            falliti += 1; print(f'  [FALLITO] {e}')
        except Exception as e:
            falliti += 1; print(f'  [FALLITO] {type(e).__name__}: {e}')
    print(f'\n{"tutti i test passati" if not falliti else f"{falliti} test FALLITI"}')
    sys.exit(1 if falliti else 0)
