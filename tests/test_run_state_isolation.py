#!/usr/bin/env python3
"""
Due fit sullo STESSO oggetto non devono contaminarsi (v1.3.6).

IL DIFETTO. Gli attributi di run (`n_iso_source_`, `sure_crosscheck_report_`,
`lambda_edges_`, ...) erano inizializzati SOLO in `__init__`, mai azzerati in
`fit()`. Per la maggior parte non importava, perche' vengono riscritti a ogni
fit. Ma `n_iso_source_` si scrive **solo dentro il ramo della selezione di
n_iso**: con `n_iso` passato (calibrazione congelata di coorte) non si scrive
affatto, quindi su un oggetto riusato il run report dichiarava la provenienza
del fit PRECEDENTE.

La v1.3.6 ha reso la cosa consequenziale: il rifiuto del bootstrap introduceva
un ramo che LEGGEVA `n_iso_source_` per sapere se il ripiego era gia' scattato.
Su un oggetto riusato quel ramo scattava a torto, saltando il controllo
`curve_is_flat` e lasciando la provenienza mislabellata.

Nessun notebook del progetto riusa l'oggetto — ognuno costruisce un
DBSI_Adaptive per fit — ma e' esattamente la classe di difetto
"il comportamento dipende da come lo invochi" che ha prodotto il bug della
griglia isotropa. Da qui il test.

    python tests/test_run_state_isolation.py
"""
import io
import sys
import contextlib

import numpy as np

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

# gli attributi che `fit()` deve azzerare all'inizio
ATTRIBUTI_DI_RUN = [
    'model_mode_', 'b_max_', 'n_shells_', 'n_aniso_cols_', 'diff_pairs_',
    'sure_crosscheck_report_', 'n_iso_source_', 'lambda_edges_',
    'fiber_detection_stat_', 'fiber_detection_', 'uncertainty_', 'uncertainty_summary_',
    'nonfinite_voxels_excluded_',
    'hemisphere_spacing_deg_', 'cone_refinement_schedule_', 'run_report_',
    'n_iso_columns_', 'n_iso_columns_res_', 'n_iso_columns_wat_',
]


class _Sentinella(Exception):
    """Alzata dentro fit() subito dopo l'azzeramento."""


def _dati(n_vox=6, seed=0):
    rng = np.random.default_rng(seed)
    nb0 = 4
    bvals = np.concatenate([np.zeros(nb0), np.full(8, 1000.0), np.full(8, 2000.0)])
    g = (1 + 5 ** 0.5) / 2
    bvecs = [np.zeros(3)] * nb0
    for k in range(16):
        z = 1 - 2 * (k + 0.5) / 16
        r = np.sqrt(max(0.0, 1 - z ** 2))
        th = 2 * np.pi * k / g
        bvecs.append([r * np.cos(th), r * np.sin(th), z])
    bvecs = np.asarray(bvecs)
    data = np.abs(rng.normal(1000, 60, (n_vox, 1, 1, len(bvals)))).astype(np.float32)
    return data, bvals, bvecs, np.ones(data.shape[:3], bool)


def test_fit_azzera_gli_attributi_di_run():
    """Tutti gli attributi di run devono valere None/{} appena fit() parte."""
    print("\n=== test_fit_azzera_gli_attributi_di_run ===")
    data, bvals, bvecs, mask = _dati()
    m = DBSI_Adaptive(n_iso=6, lambda_aniso=8.0, lambda_iso=0.02,
                      min_dominant_concentration=0.45)

    # sporca gli attributi come li lascerebbe un fit precedente
    sporco = 'VALORE_DEL_FIT_PRECEDENTE'
    for a in ATTRIBUTI_DI_RUN:
        setattr(m, a, sporco)

    # ferma fit() subito dopo l'azzeramento, prima che li riscriva
    import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
    orig = M.estimate_snr_robust
    visti = {}

    def _spia(*a, **k):
        for att in ATTRIBUTI_DI_RUN:
            visti[att] = getattr(m, att, '<assente>')
        raise _Sentinella()

    try:
        M.estimate_snr_robust = _spia
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                m.fit(data, bvals, bvecs, mask, run_calibration=False)
            except _Sentinella:
                pass
    finally:
        M.estimate_snr_robust = orig

    assert visti, 'la sentinella non e mai stata raggiunta: fit() e cambiato'
    # confronto per identita' di stringa: alcuni attributi diventano array o {},
    # e `array == str` non e' un booleano
    def _sporco(v):
        return isinstance(v, str) and v == sporco
    residui = {a: v for a, v in visti.items() if _sporco(v)}
    for a in sorted(visti):
        stato = 'SPORCO' if _sporco(visti[a]) else 'azzerato'
        print(f"  {a:<34} {stato}")
    assert not residui, (
        'questi attributi sopravvivono da un fit al successivo: '
        + repr(sorted(residui)) +
        ' -- vanno azzerati in testa a fit(), non solo in __init__')
    print("  [ok]")


def test_n_iso_source_non_mente_su_calibrazione_congelata():
    """Con n_iso PASSATO, la provenienza deve essere 'user', non quella di prima."""
    print("\n=== test_n_iso_source_non_mente_su_calibrazione_congelata ===")
    data, bvals, bvecs, mask = _dati()
    m = DBSI_Adaptive(n_iso=6, lambda_aniso=8.0, lambda_iso=0.02,
                      min_dominant_concentration=0.45)
    m.n_iso_source_ = 'bootstrap'          # come lo lascerebbe un fit libero
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(data, bvals, bvecs, mask, run_calibration=False)
    src = m.run_report_['calibrated']['n_iso_source']
    print(f"  provenienza dichiarata: {src!r}  (atteso 'user')")
    assert src == 'user', (
        f"il report dichiara n_iso_source={src!r} su una run a n_iso IMPOSTO: "
        "e' il valore rimasto dal fit precedente")
    print("  [ok]")


def test_il_ramo_del_rifiuto_usa_una_locale():
    """Il ramo del rifiuto non deve dipendere da un attributo che sopravvive."""
    print("\n=== test_il_ramo_del_rifiuto_usa_una_locale ===")
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / 'dbsi_toolbox'
           / 'model_Niso_adaptive_ff_thr.py').read_text(encoding='utf-8')
    cattivo = "if self.n_iso_source_ == 'svd_fallback_degenerate_reference':"
    print(f"  legge l'attributo nel ramo del rifiuto? "
          f"{'SI' if cattivo in src else 'no'}")
    assert cattivo not in src, (
        "il ramo del rifiuto legge self.n_iso_source_, che sopravvive fra due "
        "fit: su un oggetto riusato scatta a torto e salta il controllo "
        "curve_is_flat. Usare una variabile locale.")
    assert '_rifiutato' in src, 'la variabile locale attesa non c e'
    print("  [ok]")


if __name__ == '__main__':
    for f in (test_fit_azzera_gli_attributi_di_run,
              test_n_iso_source_non_mente_su_calibrazione_congelata,
              test_il_ramo_del_rifiuto_usa_una_locale):
        try:
            f()
        except AssertionError as e:
            print(f"\n!!! {f.__name__}: {e}")
            sys.exit(1)
    print("\ntutti i test passati")
