#!/usr/bin/env python3
"""
Regressione per il bias dello stimatore di rumore (v1.3.5).

Fino alla 1.3.4 `estimate_snr_robust` restituiva, su un'acquisizione con 2
volumi b=0 (il minimo che il toolbox accetta e quello che fornisce il
protocollo P3):

    SNR   x 1.82   (troppo alto)
    sigma x 0.55   (troppo basso)

per due difetti indipendenti, e questo file li misura entrambi contro un sigma
NOTO invece di fidarsi dell'algebra:

  1. mediana di un rapporto. La std per voxel su k = nb0-1 gradi di liberta'
     e' sigma*sqrt(chi2_k/k); la mediana del RAPPORTO non e' il rapporto delle
     mediane. A nb0=2: x1.483.
  2. la "correzione Rician iterativa" ha punto fisso in forma chiusa,
     snr = sqrt(3/2)*(m/s) = 1.2247*(m/s), quindi le 20 iterazioni
     moltiplicavano per una costante a QUALUNQUE SNR e QUALUNQUE nb0.

Questi test FALLISCONO sulla 1.3.4 (verificato mettendo
`_SNR_LEGACY_BIASED = True`, che riproduce il vecchio percorso bit per bit).

    python tests/test_snr_sigma_bias.py
"""
import sys
import numpy as np

from dbsi_toolbox.utils import tools
from dbsi_toolbox.utils.tools import estimate_snr_robust, _chi_median_factor
from dbsi_toolbox.calibration.data_driven import sample_calibration_voxels

TOLL = 0.05          # 5% sul fattore di scala: il MC ha ~200k voxel
NV = 120_000


def _volume_finto(A, sigma, nb0, seed=0, shape=(60, 60, 30)):
    """Volume con rumore Riciano vero (magnitudine di 2 canali gaussiani)."""
    rng = np.random.default_rng(seed)
    bvals = np.concatenate([np.zeros(nb0), np.full(6, 1000.0)])
    data = np.empty(shape + (len(bvals),))
    for i, b in enumerate(bvals):
        dec = 1.0 if b < 50 else np.exp(-b * 0.8e-3)
        re = A * dec + rng.normal(0, sigma, shape)
        im = rng.normal(0, sigma, shape)
        data[..., i] = np.sqrt(re ** 2 + im ** 2)
    return data, bvals, np.ones(shape, bool)


def test_snr_non_distorto():
    """SNR entro il 5% del vero, a 2/4/8 volumi b=0."""
    print("\n=== test_snr_non_distorto ===")
    ok = True
    for snr_vero in (15.0, 30.0, 45.0):
        A, sigma = 1000.0, 1000.0 / snr_vero
        for nb0 in (2, 4, 8):
            data, bvals, mask = _volume_finto(A, sigma, nb0, seed=nb0)
            snr, _ = estimate_snr_robust(data, bvals, mask, verbose=False)
            r = snr / snr_vero
            flag = "ok" if abs(r - 1.0) < TOLL else "FALLITO"
            if flag != "ok":
                ok = False
            print(f"  SNR vero {snr_vero:5.1f}  nb0={nb0}  stimato {snr:6.2f} "
                  f"(x{r:.3f})  [{flag}]")
    assert ok, "lo SNR stimato non e' entro il 5% del vero"
    print("  [ok]")


def test_sigma_non_distorto():
    """sigma entro il 5% del vero."""
    print("\n=== test_sigma_non_distorto ===")
    ok = True
    for snr_vero in (15.0, 30.0, 45.0):
        A, sigma = 1000.0, 1000.0 / snr_vero
        data, bvals, mask = _volume_finto(A, sigma, 2, seed=11)
        _, sig = estimate_snr_robust(data, bvals, mask, verbose=False)
        r = sig / sigma
        flag = "ok" if abs(r - 1.0) < TOLL else "FALLITO"
        if flag != "ok":
            ok = False
        print(f"  sigma vero {sigma:7.3f}  stimato {sig:7.3f} (x{r:.3f})  [{flag}]")
    assert ok, "il sigma stimato non e' entro il 5% del vero"
    print("  [ok]")


def test_legacy_era_distorto():
    """Il percorso legacy DEVE ancora sbagliare di x1.8 / x0.55.

    Se questo test passa e gli altri due no, il default e' tornato indietro.
    """
    print("\n=== test_legacy_era_distorto ===")
    A, sigma, snr_vero = 1000.0, 1000.0 / 30.0, 30.0
    data, bvals, mask = _volume_finto(A, sigma, 2, seed=7)
    tools._SNR_LEGACY_BIASED = True
    try:
        snr_l, sig_l = estimate_snr_robust(data, bvals, mask, verbose=False)
    finally:
        tools._SNR_LEGACY_BIASED = False
    snr_n, sig_n = estimate_snr_robust(data, bvals, mask, verbose=False)
    print(f"  legacy  SNR {snr_l:6.2f} (x{snr_l/snr_vero:.3f})   "
          f"sigma {sig_l:7.3f} (x{sig_l/sigma:.3f})")
    print(f"  attuale SNR {snr_n:6.2f} (x{snr_n/snr_vero:.3f})   "
          f"sigma {sig_n:7.3f} (x{sig_n/sigma:.3f})")
    assert snr_l / snr_vero > 1.6, "il percorso legacy non riproduce il bias noto"
    assert sig_l / sigma < 0.7, "il percorso legacy non riproduce il bias noto"
    assert abs(snr_l / snr_n - 1.82) < 0.15, (
        f"il rapporto legacy/attuale e' {snr_l/snr_n:.3f}, atteso ~1.82")
    print("  [ok]")


def test_punto_fisso_sqrt_1p5():
    """La 'correzione iterativa' del legacy e' esattamente sqrt(3/2)."""
    print("\n=== test_punto_fisso_sqrt_1p5 ===")
    rng = np.random.default_rng(5)
    m = rng.uniform(500, 1500, 50_000)
    s = rng.uniform(10, 60, 50_000)
    snr = (m / s).copy()
    for _ in range(200):
        var = s ** 2 - m ** 2 / (2 * snr ** 2 + 1e-10)
        var[var < 0] = 1e-10
        snr = m / np.sqrt(var)
    rapporto = float(np.median(snr / (m / s)))
    print(f"  punto fisso / (m/s) = {rapporto:.6f}   sqrt(1.5) = "
          f"{np.sqrt(1.5):.6f}")
    assert abs(rapporto - np.sqrt(1.5)) < 1e-4, (
        "il punto fisso non e' sqrt(3/2): l'analisi nell'intestazione di "
        "tools.py non descrive piu' il codice")
    print("  [ok]")


def test_fattore_chi():
    """Il fattore di de-bias coincide con la mediana campionaria misurata."""
    print("\n=== test_fattore_chi ===")
    rng = np.random.default_rng(9)
    ok = True
    for n in (2, 3, 4, 8, 32):
        x = rng.normal(0, 1.0, (200_000, n))
        emp = float(np.median(np.std(x, axis=1, ddof=1)))
        teo = _chi_median_factor(n)
        flag = "ok" if abs(emp - teo) < 0.01 else "FALLITO"
        if flag != "ok":
            ok = False
        print(f"  n={n:2d}  empirico {emp:.4f}  teorico {teo:.4f}  [{flag}]")
    assert ok, "il fattore chi non corrisponde alla misura"
    assert abs(_chi_median_factor(2) - 0.6745) < 1e-3
    print("  [ok]")


def test_i_due_sigma_concordano():
    """sigma_cal e 1/SNR devono descrivere la stessa acquisizione.

    Prima della 1.3.5 differivano del 22%: x0.674 contro x0.554.
    """
    print("\n=== test_i_due_sigma_concordano ===")
    A, sigma = 1000.0, 1000.0 / 30.0
    data, bvals, mask = _volume_finto(A, sigma, 2, seed=21)
    snr, _ = estimate_snr_robust(data, bvals, mask, verbose=False)
    _, sig_cal = sample_calibration_voxels(data, mask, bvals,
                                           n_voxels=4000, seed=0)
    scarto = abs(1.0 / snr - sig_cal) / sig_cal
    print(f"  1/SNR = {1/snr:.5f}   sigma_cal = {sig_cal:.5f}   "
          f"scarto {scarto*100:.1f}%")
    assert scarto < 0.08, (
        f"i due stimatori di rumore differiscono del {scarto*100:.1f}%: "
        "uno dei due e' distorto")
    print("  [ok]")


if __name__ == '__main__':
    for f in (test_fattore_chi, test_punto_fisso_sqrt_1p5,
              test_snr_non_distorto, test_sigma_non_distorto,
              test_legacy_era_distorto, test_i_due_sigma_concordano):
        try:
            f()
        except AssertionError as e:
            print(f"\n!!! {f.__name__}: {e}")
            sys.exit(1)
    print("\ntutti i test passati")
