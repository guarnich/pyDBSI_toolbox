"""
toolbox_report/ — cosa ha prodotto le mappe, in una cartella sola.

`save_output_maps(..., model=m)` scrive accanto alle mappe una cartella
`toolbox_report/` con:

    run_report.txt           versione, commit, iperparametri, opzioni, censo
    design_matrix.png        l'immagine del dizionario A = [A_aniso | A_iso]
    design_matrix.npz        la matrice stessa + b-values, direzioni, coppie
                             (AD, RD) e griglia isotropa che la generano
    dictionary_columns.csv   una riga per colonna: blocco, direzione, AD/RD o
                             D isotropa, comparto
    protocol_calibration.json  (solo con `from_calibration`) copia della
                             calibrazione di protocollo usata

Perche' anche il dizionario: fino alla 1.3.9 il run report diceva quante colonne
c'erano ma non QUALI. La matrice dipende dal protocollo (b-values, direzioni),
da n_dirs, dalle coppie ammesse da anisotropy_ratio e dalla griglia isotropa;
ricostruirla a posteriori voleva dire rifare la configurazione e sperare di
averla rifatta uguale. Salvarla costa poche centinaia di kB.

La cartella e' pensata per crescere: ogni nuovo artefatto di provenienza va qui,
non sparso fra le mappe.
"""
import csv
import os

import numpy as np

REPORT_DIRNAME = 'toolbox_report'

# colori dei tre comparti isotropi (stessi in tutte le figure del report)
_COL_COMP = {'restricted': '#d95f02', 'hindered': '#7570b3', 'water': '#1b9e77'}


def _compartment(d, thresh_res, thresh_wat):
    """Stessa regola della classificazione del modello: adc <= soglia."""
    if d <= thresh_res:
        return 'restricted'
    if d <= thresh_wat:
        return 'hindered'
    return 'water'


def _shell_groups(bvals, step=100.0):
    """Ordine delle righe per b crescente e confini dei gusci (per le etichette)."""
    b_round = np.round(np.asarray(bvals, float) / step) * step
    order = np.argsort(b_round, kind='stable')
    b_sorted = b_round[order]
    edges = [0] + [i for i in range(1, len(b_sorted)) if b_sorted[i] != b_sorted[i - 1]]
    edges.append(len(b_sorted))
    groups = [(int(b_sorted[edges[k]]), edges[k], edges[k + 1])
              for k in range(len(edges) - 1)]
    return order, groups


def plot_design_matrix(dictionary, path, title_extra=''):
    """Immagine di A = [A_aniso | A_iso], righe ordinate per b-value.

    Il pannello di sinistra e' il blocco anisotropo, organizzato come nel
    codice (pair-major, direction-minor): un riquadro per coppia (AD, RD), con
    dentro le n_dirs direzioni. Quello di destra e' il blocco isotropo, una
    colonna per atomo, etichettata con la sua D e colorata per comparto.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    A = np.asarray(dictionary['A'])
    n_aniso = int(dictionary['n_aniso_cols'])
    n_dirs = int(dictionary['n_dirs'])
    n_pairs = int(dictionary['n_pairs'])
    iso = np.asarray(dictionary['iso_grid'], float)
    pairs = np.asarray(dictionary['diff_pairs'], float)
    tr, tw = float(dictionary['thresh_res']), float(dictionary['thresh_wat'])
    n_meas, n_tot = A.shape
    n_iso_cols = n_tot - n_aniso

    order, groups = _shell_groups(dictionary['bvals'])
    As = A[order]

    fig = Figure(figsize=(15, 7.5), dpi=150)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.2, 1.0, 0.06], wspace=0.08,
                          left=0.07, right=0.95, top=0.89, bottom=0.17)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])

    kw = dict(aspect='auto', interpolation='nearest', cmap='viridis', vmin=0.0, vmax=1.0)
    im = ax_a.imshow(As[:, :n_aniso], **kw)
    ax_i.imshow(As[:, n_aniso:], **kw)

    # confini dei gusci
    for ax in (ax_a, ax_i):
        for _, lo, _hi in groups[1:]:
            ax.axhline(lo - 0.5, color='white', lw=0.8, alpha=0.9)
    ax_a.set_yticks([(lo + hi - 1) / 2 for _, lo, hi in groups])
    ax_a.set_yticklabels([f'b={b}  ({hi - lo})' for b, lo, hi in groups], fontsize=8)
    ax_a.set_ylabel('measurements, sorted by b-value (count per shell)')

    # blocco anisotropo: un riquadro per coppia (AD, RD)
    for p in range(1, n_pairs):
        ax_a.axvline(p * n_dirs - 0.5, color='white', lw=0.8, alpha=0.9)
    ax_a.set_xticks([(p + 0.5) * n_dirs - 0.5 for p in range(n_pairs)])
    ax_a.set_xticklabels([f'AD {ad*1e3:.2f}\nRD {rd*1e3:.2f}' for ad, rd in pairs],
                         fontsize=7)
    ax_a.set_xlabel(f'A_aniso: {n_pairs} (AD, RD) pairs x {n_dirs} directions '
                    f'= {n_aniso} columns   (x10^-3 mm^2/s)')

    # blocco isotropo: una colonna per atomo, colorata per comparto
    ax_i.set_xticks(range(n_iso_cols))
    ax_i.set_xticklabels([f'{d*1e3:.4g}' for d in iso], rotation=90, fontsize=7)
    for lab, d in zip(ax_i.get_xticklabels(), iso):
        lab.set_color(_COL_COMP[_compartment(d, tr, tw)])
    for k in range(1, n_iso_cols):
        if (_compartment(iso[k], tr, tw) != _compartment(iso[k - 1], tr, tw)):
            ax_i.axvline(k - 0.5, color='white', lw=1.4)
    n_r = int(np.sum(iso <= tr)); n_w = int(np.sum(iso > tw)); n_h = n_iso_cols - n_r - n_w
    ax_i.set_xlabel(f'A_iso: {n_iso_cols} columns, D (x10^-3 mm^2/s)\n'
                    f'restricted {n_r} / hindered {n_h} / water {n_w}')
    ax_i.tick_params(axis='y', labelleft=False)

    cb = fig.colorbar(im, cax=ax_c)
    cb.set_label('signal attenuation  S/S0', fontsize=9)

    fig.suptitle(
        f'Stage A dictionary  A = [A_aniso | A_iso]   —   {n_meas} measurements x '
        f'{n_tot} columns\n'
        f'N_aniso = {n_aniso}  ({n_dirs} directions x {n_pairs} (AD, RD) pairs)     '
        f'N_iso = {n_iso_cols}  (n_iso = {dictionary["n_iso"]})',
        fontsize=12)
    foot = (f'lambda_aniso = {dictionary["lambda_aniso"]:.5g}   '
            f'lambda_iso = {dictionary["lambda_iso"]:.5g}   '
            f'cond(A^T A + reg) = {dictionary["condition_number_regularized"]:.3g}   '
            f'thresholds R/H {tr*1e3:.2f}, H/W {tw*1e3:.2f} x10^-3')
    if title_extra:
        foot += f'   |   {title_extra}'
    fig.text(0.5, 0.015, foot, ha='center', fontsize=8, color='#444')
    fig.savefig(path)
    return path


def _column_table(dictionary):
    """Una riga per colonna di A, nell'ordine di A (pair-major, direction-minor)."""
    dirs = np.asarray(dictionary['fiber_dirs'], float)
    pairs = np.asarray(dictionary['diff_pairs'], float)
    iso = np.asarray(dictionary['iso_grid'], float)
    tr, tw = float(dictionary['thresh_res']), float(dictionary['thresh_wat'])
    rows = []
    col = 0
    for p, (ad, rd) in enumerate(pairs):
        for d, v in enumerate(dirs):
            rows.append(dict(column=col, block='aniso', pair_index=p, dir_index=d,
                             dir_x=v[0], dir_y=v[1], dir_z=v[2], ad=ad, rd=rd,
                             d_iso='', compartment='fiber'))
            col += 1
    for k, dval in enumerate(iso):
        rows.append(dict(column=col, block='iso', pair_index='', dir_index='',
                         dir_x='', dir_y='', dir_z='', ad='', rd='', d_iso=dval,
                         compartment=_compartment(dval, tr, tw)))
        col += 1
    return rows


def save_toolbox_report(model, output_dir, saved_channels=None):
    """Scrive `output_dir/toolbox_report/` e ne restituisce i file.

    Richiede un `DBSI_Adaptive` fittato. Se il modello non ha il run report o il
    dizionario (fit non completato), scrive cio' che c'e' e lo dice.
    """
    from .fit_quality import format_run_report

    rep_dir = os.path.join(output_dir, REPORT_DIRNAME)
    os.makedirs(rep_dir, exist_ok=True)
    written = []

    report = getattr(model, 'run_report_', None)
    if report:
        f = os.path.join(rep_dir, 'run_report.txt')
        with open(f, 'w') as fh:
            fh.write(format_run_report(report, saved_channels=saved_channels))
        written.append(f)

    D = getattr(model, 'dictionary_', None)
    if D:
        f = os.path.join(rep_dir, 'design_matrix.npz')
        np.savez_compressed(
            f, A=D['A'], bvals=D['bvals'], bvecs=D['bvecs'],
            fiber_dirs=D['fiber_dirs'], diff_pairs=D['diff_pairs'],
            iso_grid=D['iso_grid'], n_aniso_cols=D['n_aniso_cols'],
            n_dirs=D['n_dirs'], n_pairs=D['n_pairs'], n_iso=D['n_iso'],
            lambda_aniso=D['lambda_aniso'], lambda_iso=D['lambda_iso'],
            condition_number_regularized=D['condition_number_regularized'],
            thresh_res=D['thresh_res'], thresh_wat=D['thresh_wat'])
        written.append(f)

        f = os.path.join(rep_dir, 'dictionary_columns.csv')
        rows = _column_table(D)
        with open(f, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        written.append(f)

        f = os.path.join(rep_dir, 'design_matrix.png')
        ver = (report or {}).get('toolbox_version', '')
        plot_design_matrix(D, f, title_extra=f'pyDBSI {ver}' if ver else '')
        written.append(f)
    # La calibrazione di protocollo usata, COPIATA: il run report ne cita lo
    # sha256, ma il file originale puo' essere spostato o riscritto.
    pc = getattr(model, 'protocol_calibration_', None)
    if pc and pc.get('content'):
        import json
        f = os.path.join(rep_dir, 'protocol_calibration.json')
        with open(f, 'w', encoding='utf-8') as fh:
            json.dump(pc['content'], fh, indent=2, ensure_ascii=False)
        written.append(f)

    mancano = [n for n, v in (('run_report_', report), ('dictionary_', D)) if not v]
    if mancano:
        print(f"   [WARNING] toolbox_report incompleto: il modello non ha "
              f"{', '.join(mancano)} (fit non completato?)")
    return written
