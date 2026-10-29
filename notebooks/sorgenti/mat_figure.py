"""
La matematica di pyDBSI in figure: una funzione per ogni sezione del voxel journey.

Regola: STRUMENTARE, NON RIPRODURRE. Dove possibile le figure usano le funzioni vere del toolbox e i
dati che il fit usa davvero: `intercetta_fit` lancia `DBSI_Adaptive.fit()` e registra gli argomenti del
kernel (segnale corretto, A^T A regolarizzata, dizionario, grafo dei vicini, lambda), e `registra_calibrazione`
registra ingressi e uscite di ogni passo della calibrazione. Dove una figura deve mostrare un passaggio
interno che il toolbox non restituisce (i bacini, i candidati del cono, la traiettoria dell'LM), lo ricalcola
con le stesse funzioni e CONTROLLA che il risultato coincida con quello del toolbox.
"""
from __future__ import annotations

import contextlib
import inspect
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive
import dbsi_toolbox.model_Niso_adaptive_ff_thr as M
import dbsi_toolbox.calibration.data_driven as DD
from dbsi_toolbox.core import solvers as SOL
from dbsi_toolbox.core.basis import build_design_matrix_exhaustive, generate_anchored_isotropic_grid
from sim_accuratezza import segnale, rumore_rician, ruota_verso, errore_angolare

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('seq_blu', ['#f7f9fc', '#cde2fb', '#86b6ef', '#3987e5',
                                                    '#256abf', '#184f95', '#0d366b'])
COMP = dict(R=CAT[7], H=CAT[0], W=CAT[2], F=CAT[1])          # restricted, hindered, acqua, fibra
D_NOM = dict(R=0.15e-3, H=1.0e-3, W=3.0e-3)
NOMI = DBSI_Adaptive.output_map_names(3)
IX = {n: i for i, n in enumerate(NOMI)}


def stile():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight', 'font.size': 9,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def _salva(fig, file):
    if file:
        fig.savefig(file)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# strumenti: intercettare il fit e la calibrazione
# ═════════════════════════════════════════════════════════════════════════════
def intercetta_fit(prot, data, mask, noise_map, lam=None, opzioni=None, **kw_fit):
    """Fit vero; registra gli argomenti del kernel 3-ISO. lam=(lambda_aniso, lambda_iso) per congelarli
    (niente calibrazione); opzioni = altri argomenti di DBSI_Adaptive (es. quelle dei crossing). Ritorna dict
    con gli argomenti per nome + 'res' + 'model'."""
    rec = {}
    orig = M._fit_voxels_3iso_v3
    nomi = list(inspect.signature(orig.py_func).parameters)

    def spia(*a, **k):
        rec.update(dict(zip(nomi, a)))
        rec.update(k)
        return orig(*a, **k)

    kwm = {} if lam is None else dict(lambda_aniso=lam[0], lambda_iso=lam[1], n_iso=6)
    kwm.update(opzioni or {})
    m = DBSI_Adaptive(**kwm)
    M._fit_voxels_3iso_v3 = spia
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res, mode = m.fit(data, prot[0], prot[1], mask, run_calibration=lam is None,
                              noise_map=noise_map, **kw_fit)
    finally:
        M._fit_voxels_3iso_v3 = orig
    assert mode == 3 and 'AtA_reg' in rec, 'kernel 3-ISO non intercettato'
    rec.update(res=res, model=m, prot=prot, n_pairs=len(rec['diff_pairs']),
               n_aniso=len(rec['diff_pairs']) * int(rec['n_dirs']))
    At = np.asarray(rec['At'])
    rec['AtA'] = At @ At.T       # non regolarizzata (A^T A)
    return rec


def registra_calibrazione(prot, data, mask, noise_map, n_cal=1000, seed=None):
    """Calibrazione vera (_calibration_only) con un registratore su ogni passo. seed: se dato, sostituisce
    il seme fisso 0 del campionamento (per studiare la variabilita' del campione)."""
    R = dict(nnls_lambda=[])
    orig = {n: getattr(DD, n) for n in ('select_lambda_iso_gcv', 'select_lambda_aniso_discrepancy',
                                        'lambda_iso_discrepancy_cap', 'select_lambda_aniso_gcv_lcurve',
                                        'nnls_batch')}
    orig_samp = M.sample_calibration_voxels

    def gcv_iso(bvals, iso_grid, y, lambda_grid=None):
        out = orig['select_lambda_iso_gcv'](bvals, iso_grid, y, lambda_grid=lambda_grid)
        R.setdefault('gcv_iso', []).append(dict(y=np.array(y), iso_grid=np.array(iso_grid), lam=out[0], diag=out[1]))
        return out

    def disc(AtA, At, y, n_aniso_cols, sigma, lambda_iso_fixed, **k):
        R['_AtA00'] = float(AtA[0, 0])       # il lambda valutato si legge dalla diagonale di AtA_reg
        R['disc_in'] = dict(AtA=AtA, At=At, y=np.array(y), n_aniso_cols=n_aniso_cols, sigma=sigma,
                            lambda_iso=lambda_iso_fixed, k=k)
        R['nnls_lambda'] = []
        out = orig['select_lambda_aniso_discrepancy'](AtA, At, y, n_aniso_cols, sigma, lambda_iso_fixed, **k)
        R['disc_out'] = out
        R['disc_lambda_valutati'] = list(R['nnls_lambda'])
        return out

    def cap(bvals, iso_grid, y_iso_resid, sigma, lambda_grid):
        out = orig['lambda_iso_discrepancy_cap'](bvals, iso_grid, y_iso_resid, sigma, lambda_grid)
        R['cap'] = dict(y=np.array(y_iso_resid), sigma=sigma, grid=np.array(lambda_grid), out=out, iso_grid=iso_grid)
        return out

    def gcv_aniso(AtA, At, y, n_aniso_cols, lambda_iso_fixed, method='gcv', lambda_grid=None):
        out = orig['select_lambda_aniso_gcv_lcurve'](AtA, At, y, n_aniso_cols, lambda_iso_fixed,
                                                     method=method, lambda_grid=lambda_grid)
        R['aniso'] = dict(lam=out[0], diag=out[1], N=np.asarray(y).shape[1])
        return out

    def batch(AtA_reg, Aty_rows, reg):
        R['nnls_lambda'].append(float(AtA_reg[0, 0] - R['_AtA00']) if '_AtA00' in R else np.nan)
        return orig['nnls_batch'](AtA_reg, Aty_rows, reg)

    def samp(*a, **k):
        if seed is not None:
            k['seed'] = seed
        out = orig_samp(*a, **k)
        R['campione'] = dict(y=np.array(out[0]), sigma_cal=out[1])
        return out

    DD.select_lambda_iso_gcv, DD.select_lambda_aniso_discrepancy = gcv_iso, disc
    DD.lambda_iso_discrepancy_cap, DD.select_lambda_aniso_gcv_lcurve = cap, gcv_aniso
    DD.nnls_batch = batch
    M.sample_calibration_voxels = samp
    try:
        m = DBSI_Adaptive()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(data, prot[0], prot[1], mask, run_calibration=True, n_calibration_voxels=n_cal,
                  noise_map=noise_map, _calibration_only=True)
    finally:
        for n, f in orig.items():
            setattr(DD, n, f)
        M.sample_calibration_voxels = orig_samp
    R.update(lambda_aniso=float(m.lambda_aniso), lambda_iso=float(m.lambda_iso), model=m)
    return R


# ═════════════════════════════════════════════════════════════════════════════
# 3.3  protocollo: 3-ISO o 2-ISO
# ═════════════════════════════════════════════════════════════════════════════
def fig_protocollo(prot, snr=26, protocolli=None, file=None):
    bv = prot[0]
    b = np.linspace(0, 3000, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a = axs[0]
    for k, lab in (('R', 'restricted  D = 0.15'), ('H', 'hindered  D = 1.0'), ('W', 'acqua libera  D = 3.0')):
        a.plot(b, np.exp(-b * D_NOM[k]), color=COMP[k], lw=2, label=lab + ' ×1e-3 mm²/s')
    a.fill_between(b, np.exp(-b * 0.7e-3), np.exp(-b * 1.3e-3), color=COMP['H'], alpha=0.12, lw=0,
                   label='hindered reale (0.7–1.3)')
    a.axhspan(0, 1 / snr, color='#9a988f', alpha=0.25, lw=0)
    a.text(60, 1.15 / snr, f'livello del rumore  σ/S0 = 1/{snr}', color=INK2, fontsize=8)
    gusci = pd.Series(np.round(bv, -2).astype(int)).value_counts().sort_index()
    for g, n in gusci.items():
        if g > 50:
            a.axvline(g, color=INK2, lw=0.8, ls=':')
            a.text(g, 1.02, f'b={g}\n{n} dir', ha='center', va='bottom', fontsize=7, color=INK2)
    for k in 'HW':
        v = np.exp(-2000 * D_NOM[k])
        a.plot(2000, v, 'o', color=COMP[k], ms=6, mec='white')
        a.annotate(f'{v:.3f}', (2000, v), xytext=(8, 4 if k == 'H' else -10), textcoords='offset points',
                   fontsize=8, color=COMP[k])
    a.set_xlabel('b [s/mm²]'); a.set_ylabel('segnale S/S0'); a.set_ylim(0, 1.12)
    a.set_title('Ogni compartimento decade con la sua D')
    a.legend(loc='center right', bbox_to_anchor=(1.0, 0.6), fontsize=7.5)
    a = axs[1]
    contr = np.exp(-b * D_NOM['H']) - np.exp(-b * D_NOM['W'])
    a.plot(b, contr, color=INK, lw=2, label='hindered − acqua')
    a.axhline(2 / snr, color='#9a988f', ls='--', lw=1.2, label=f'2σ (SNR {snr})')
    a.fill_between(b, 0, contr, where=b >= 1500, color=CAT[2], alpha=0.15, lw=0)
    a.set_xlabel('b [s/mm²]'); a.set_ylabel('differenza di segnale')
    a.set_title('Separare hindered e acqua: serve b alto')
    a.text(1550, 0.03, "a b alto l'acqua è già sparita,\nla hindered no: le due curve\nsi separano", fontsize=8,
           color=INK2)
    a.legend(loc='upper right', fontsize=8)
    if protocolli:
        righe = []
        for nome, p in protocolli.items():
            g = sorted(set(np.round(p[0], -2).astype(int)) - {0})
            bmax, nsh, tre, _ = M.analyse_protocol(p[0])
            righe.append(f"{nome}: b_max {bmax:.0f}, {nsh} gusci -> {'3-ISO' if tre else '2-ISO'}")
        fig.text(0.01, -0.06, '   '.join(righe), fontsize=8, color=INK2)
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 4  rumore Rician
# ═════════════════════════════════════════════════════════════════════════════
def _rice_media(A, s):
    from scipy.special import i0e, i1e
    x = A ** 2 / (4 * s ** 2)
    return s * np.sqrt(np.pi / 2) * ((1 + 2 * x) * i0e(x) + 2 * x * i1e(x))


def fig_rician(snr=26, prot=None, file=None, seed=0):
    from scipy.stats import rice
    rng = np.random.default_rng(seed)
    s = 1.0
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    a = axs[0]
    m = np.linspace(0, 8, 400)
    for i, A in enumerate((0, 1, 2, 4)):
        a.plot(m, rice.pdf(m, A / s, scale=s), color=CAT[i], lw=2, label=f'A = {A}σ')
        a.axvline(A, color=CAT[i], lw=0.8, ls=':')
    a.set_xlabel('magnitudine misurata M  [σ]'); a.set_ylabel('densità')
    a.set_title('M = |A + n₁ + i n₂| segue Rice')
    a.legend(fontsize=8)
    a.text(4.6, 0.45, 'a segnale basso la\ndistribuzione è spostata\na destra: M > A in media', fontsize=8,
           color=INK2)
    a = axs[1]
    A = np.linspace(0, 5, 200)
    em = _rice_media(A, s)
    a.plot(A, A, color=INK, lw=1, ls='--', label='vero A')
    a.plot(A, em, color=CAT[7], lw=2, label='E[M] (dato grezzo)')
    a.axhline(np.sqrt(np.pi / 2), color=CAT[7], lw=0.8, ls=':')
    a.text(3.2, np.sqrt(np.pi / 2) + 0.08, 'pavimento σ√(π/2) ≈ 1.25σ', color=CAT[7], fontsize=8)
    # correzione per misura: sqrt(max(M^2 - 2 sigma^2, 0)), media su molte realizzazioni
    corr = []
    for a_ in A:
        Mm = np.sqrt((a_ + rng.normal(0, s, 20000)) ** 2 + rng.normal(0, s, 20000) ** 2)
        c = np.sqrt(np.maximum(Mm ** 2 - 2 * s ** 2, 0))
        corr.append((c.mean(), np.sqrt(np.mean(c ** 2))))
    corr = np.array(corr)
    a.plot(A, corr[:, 1], color=CAT[2], lw=2, label='√E[max(M²−2σ², 0)] (corretto)')
    a.plot(A, corr[:, 0], color=CAT[2], lw=1.2, ls='-.', label='E[√max(M²−2σ², 0)]')
    a.set_xlabel('segnale vero A  [σ]'); a.set_ylabel('[σ]')
    a.set_title('E[M²] = A² + 2σ²: la correzione toglie 2σ²')
    a.legend(fontsize=7.5, loc='upper left')
    a = axs[2]
    if prot is None:
        bv = np.repeat([0, 300, 700, 1000, 2000], [9, 3, 12, 20, 48]).astype(float)
    else:
        bv = prot[0]
    S0, sg = 1000.0, 1000.0 / snr
    vero = S0 * (0.95 * np.exp(-bv * 3.0e-3) + 0.05 * np.exp(-bv * 1.0e-3))      # liquor
    Mm = np.sqrt((vero[None] + rng.normal(0, sg, (400, bv.size))) ** 2 + rng.normal(0, sg, (400, bv.size)) ** 2)
    C = np.sqrt(np.maximum(Mm ** 2 - 2 * sg ** 2, 0))
    o = np.argsort(bv, kind='stable')
    bb = np.unique(bv)
    mean_raw = [Mm[:, bv == x].mean() / S0 for x in bb]
    mean_cor = [C[:, bv == x].mean() / S0 for x in bb]
    vv = [vero[bv == x].mean() / S0 for x in bb]
    a.semilogy(bb, vv, 'o-', color=INK, lw=1.5, label='vero (liquor: 95% acqua)')
    a.semilogy(bb, mean_raw, 's-', color=CAT[7], lw=1.5, label='grezzo: si appoggia al pavimento')
    a.semilogy(bb, mean_cor, '^-', color=CAT[2], lw=1.5, label='corretto')
    a.axhline(sg * np.sqrt(np.pi / 2) / S0, color=CAT[7], lw=0.8, ls=':')
    # cosa direbbe Stage D (NNLS sui tre centroidi) del segnale grezzo e corretto
    rf = []
    for sig in (np.array(mean_raw), np.array(mean_cor)):
        Aiso = np.exp(-np.outer(bb, [D_NOM['R'], D_NOM['H'], D_NOM['W']]))
        from scipy.optimize import nnls
        w, _ = nnls(Aiso, sig)
        rf.append(w / w.sum())
    a.text(0.02, 0.04, f'NNLS su R/H/W (come Stage D):\n  grezzo     RF {rf[0][0]:.2f}  WF {rf[0][2]:.2f}\n'
           f'  corretto   RF {rf[1][0]:.2f}  WF {rf[1][2]:.2f}\n  vero       RF 0.00  WF 0.95',
           transform=a.transAxes, fontsize=8, family='monospace', color=INK)
    a.set_xlabel('b [s/mm²]'); a.set_ylabel('S/S0 (scala log)')
    a.set_title(f'Senza correzione il pavimento sembra RF (SNR {snr})')
    a.legend(fontsize=7.5, loc='upper right')
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 5.3  griglia isotropa ancorata e matrice A
# ═════════════════════════════════════════════════════════════════════════════
def fig_griglia(file=None):
    d_min, d_max, n = 0.1e-3, max(3.0e-3, M._ISO_GRID_D_MAX_EXTENDED), M._QUADRATURE_N_ISO
    tr, tw, eps = M.THRESH_RES, M.THRESH_WAT, 1e-6
    p1 = np.geomspace(d_min, d_max, n)
    finale = np.asarray(generate_anchored_isotropic_grid(d_min=d_min, d_max=d_max, n_steps=n,
                                                        thresh_res=tr, thresh_wat=tw))
    anc = np.array([tr - eps * 1e-3, tw + eps * 1e-3])
    p2 = np.sort(np.concatenate([p1, anc]))
    ins = np.array([x for x in finale if not np.any(np.isclose(x, p2, rtol=1e-6))])
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    righe = [('1. 6 punti log-uniformi su [0.1, 5.0]', p1, CAT[0]),
             ('2. + 2 punti ancorati alle soglie', p2, CAT[1]),
             (f'3. + medie geometriche dove D(j+1)/D(j) > 2  →  {len(finale)} colonne', finale, INK)]
    for r, (lab, pts, col) in enumerate(righe):
        y = 2 - r
        ax.plot(pts * 1e3, np.full(len(pts), y), 'o', color=col, ms=7, mec='white', zorder=3)
        ax.text(0.085, y + 0.22, lab, fontsize=8.5, color=INK)
        if r == 1:
            ax.plot(anc * 1e3, [y, y], 'o', color=CAT[1], ms=11, mfc='none', mew=2)
        if r == 2 and len(ins):
            ax.plot(ins * 1e3, np.full(len(ins), y), 'o', color=CAT[2], ms=11, mfc='none', mew=2)
    for t, lab in ((tr, 'soglia R | H  0.3'), (tw, 'soglia H | W  3.0')):
        ax.axvline(t * 1e3, color=INK2, ls='--', lw=1)
        ax.text(t * 1e3 * 1.03, 2.55, lab, fontsize=8, color=INK2)
    for (lo, hi, k, lab) in ((0.08, tr * 1e3, 'R', 'restricted'), (tr * 1e3, tw * 1e3, 'H', 'hindered'),
                             (tw * 1e3, 6.0, 'W', 'acqua libera')):
        ax.axvspan(lo, hi, color=COMP[k], alpha=0.07, lw=0)
        ax.text(np.sqrt(lo * hi), -0.7, lab, ha='center', fontsize=8.5, color=COMP[k], fontweight='bold')
    ax.set_xscale('log'); ax.set_xlim(0.08, 6.0); ax.set_ylim(-0.9, 2.8)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('D [×1e-3 mm²/s]  (scala logaritmica)')
    nr = int((finale <= tr).sum()); nw = int((finale > tw).sum())
    ax.set_title(f'Griglia isotropa ancorata: {len(finale)} colonne ({nr} R, {len(finale) - nr - nw} H, {nw} W)')
    return _salva(fig, file), finale


def fig_matrice_A(K, file=None):
    """La matrice di Stage A del protocollo: righe = misure ordinate per b, colonne = blocco anisotropo
    (per coppia (AD, RD), poi per direzione) + blocco isotropo."""
    bv, bc = K['prot']
    A = build_design_matrix_exhaustive(bv, bc, K['fiber_dirs'], K['diff_pairs'], K['iso_grid'])
    o = np.argsort(bv, kind='stable')
    A = A[o]
    na, npair, nd = K['n_aniso'], K['n_pairs'], int(K['n_dirs'])
    fig = plt.figure(figsize=(13, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.6])
    a = fig.add_subplot(gs[0])
    im = a.imshow(A, aspect='auto', cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
    for p in range(1, npair + 1):
        a.axvline(p * nd - 0.5, color='white', lw=1.2)
    pos = [p * nd + nd / 2 for p in range(npair)] + [na + len(K['iso_grid']) / 2]
    lab = [f'AD {ad*1e3:.2f}\nRD {rd*1e3:.2f}' for ad, rd in K['diff_pairs']] + ['iso']
    top = a.secondary_xaxis('top')
    top.set_xticks(pos); top.set_xticklabels(lab, fontsize=7)
    bs = bv[o]
    for g in np.unique(np.round(bs, -2)):
        i0 = np.where(np.round(bs, -2) == g)[0][0]
        a.axhline(i0 - 0.5, color='white', lw=0.6)
        a.text(-3, i0, f'b={int(g)}', ha='right', va='top', fontsize=7)
    a.set_yticks([]); a.set_xticks([])
    a.set_xlabel(f'{na} colonne anisotrope ({npair} coppie × {nd} direzioni)  +  {len(K["iso_grid"])} isotrope')
    a.set_title(f'Matrice di Stage A del protocollo: {A.shape[0]} misure × {A.shape[1]} colonne', pad=30)
    fig.colorbar(im, ax=a, shrink=0.8, label='exp(−b D)')
    a2 = fig.add_subplot(gs[1])
    j_dir = 0
    for k, (col, lab, c) in enumerate(((0 * nd + j_dir, 'fibra, coppia 1, direzione 1', CAT[1]),
                                        ((npair - 1) * nd + j_dir, f'fibra, coppia {npair}, direzione 1', CAT[4]),
                                        (na + 2, f'iso D={K["iso_grid"][2]*1e3:.2f}', COMP['R']),
                                        (na + len(K['iso_grid']) - 1, f'iso D={K["iso_grid"][-1]*1e3:.1f}', COMP['W']))):
        a2.plot(A[:, col], color=c, lw=1.2, label=lab)
    a2.set_xlabel('misura (ordinate per b)'); a2.set_ylabel('valore della colonna')
    a2.set_title('Quattro colonne lette per intero')
    a2.legend(fontsize=7)
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 5.5  grafo dei vicini assiale e cono
# ═════════════════════════════════════════════════════════════════════════════
def _proietta(v):
    """Proiezione azimutale equivalente dall'emisfero z >= 0 (antipodi riportati su)."""
    v = np.atleast_2d(v).astype(float).copy()
    v[v[:, 2] < 0] *= -1
    r = np.sqrt(2 * (1 - np.clip(v[:, 2], -1, 1)))
    phi = np.arctan2(v[:, 1], v[:, 0])
    return r * np.cos(phi), r * np.sin(phi)


def _cerchio_equatore(a):
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=INK2, lw=0.8)
    a.set_aspect('equal'); a.axis('off')


def fig_grafo_e_cono(K, file=None):
    fd = np.asarray(K['fiber_dirs']); nb = np.asarray(K['neighbor_idx'])
    nd, k = nb.shape
    x, y = _proietta(fd)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    a = axs[0]
    _cerchio_equatore(a)
    for d in range(nd):
        for j in nb[d]:
            xa, ya = x[d], y[d]
            xb, yb = x[j], y[j]
            if np.hypot(xa - xb, ya - yb) > 1.2:          # vicino oltre l'equatore: e' -v
                continue
            a.plot([xa, xb], [ya, yb], color='#c9c7c0', lw=0.6, zorder=1)
    a.scatter(x, y, s=14, color=CAT[0], zorder=2)
    a.set_title(f'{nd} direzioni, grafo dei vicini (k={k})')
    # una direzione vicina all'equatore: vicini assiali contro prodotto scalare semplice
    eq = int(np.argmin(np.abs(fd[:, 2])))
    a = axs[1]
    _cerchio_equatore(a)
    a.scatter(x, y, s=10, color='#c9c7c0')
    semplice = np.argsort(-(fd @ fd[eq]))[1:k + 1]
    a.scatter(x[semplice], y[semplice], s=70, facecolors='none', edgecolors=CAT[7], lw=1.6,
              label='vicini col prodotto scalare semplice')
    a.scatter(x[nb[eq]], y[nb[eq]], s=28, color=CAT[2], label='vicini assiali |v·v\'| (toolbox)')
    a.scatter(x[eq], y[eq], s=60, color=INK, marker='*', label='direzione vicina all\'equatore')
    a.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.18))
    a.set_title('All\'equatore v e −v sono la stessa fibra')
    # schema del cono
    sp = SOL.measure_hemisphere_spacing(fd)
    a1, n1, a2, n2 = SOL.compute_cone_refinement_schedule(sp, np.radians(1.0))
    a = axs[2]
    v0 = fd[int(np.argmax(fd[:, 2]))]
    ex, ey = _base(v0)
    c1 = np.array([SOL._fibonacci_cone_point(*v0, *ex, *ey, a1, i, int(n1)) for i in range(int(n1))])
    best = c1[int(n1) // 3]
    bx_, by_ = _base(best)
    c2 = np.array([SOL._fibonacci_cone_point(*best, *bx_, *by_, a2, i, int(n2)) for i in range(int(n2))])
    def loc(v):            # coordinate locali in gradi attorno a v0
        return (np.degrees(np.arctan2(v @ ex, v @ v0)), np.degrees(np.arctan2(v @ ey, v @ v0)))
    vic = fd[np.argsort(-np.abs(fd @ v0))[:7]]
    lx, ly = loc(vic * np.sign(vic @ v0)[:, None])
    a.scatter(lx, ly, s=60, color='#c9c7c0', label='nodi del dizionario')
    p1 = loc(c1); p2 = loc(c2)
    a.scatter(*p1, s=26, color=CAT[0], label=f'livello 1: {int(n1)} candidati, ±{np.degrees(a1):.1f}°')
    a.scatter(*p2, s=12, color=CAT[1], label=f'livello 2: {int(n2)} candidati, ±{np.degrees(a2):.1f}°')
    t = np.linspace(0, 2 * np.pi, 100)
    a.plot(np.degrees(a1) * np.cos(t), np.degrees(a1) * np.sin(t), color=CAT[0], lw=0.8)
    bl = loc(best[None])
    a.plot(bl[0] + np.degrees(a2) * np.cos(t), bl[1] + np.degrees(a2) * np.sin(t), color=CAT[1], lw=0.8)
    a.scatter(0, 0, s=70, color=INK, marker='*', label='nodo di partenza (Stage A)')
    a.set_aspect('equal'); a.set_xlabel('gradi'); a.set_ylabel('gradi')
    a.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    a.set_title(f'Il cono: spaziatura {np.degrees(sp):.1f}° → risoluzione 1°')
    return _salva(fig, file), dict(spaziatura=np.degrees(sp), a1=np.degrees(a1), n1=int(n1), a2=np.degrees(a2), n2=int(n2), k=k)


def _base(v0):
    """La stessa base ortonormale del toolbox (refine_fiber_direction_cone)."""
    t = np.array([1.0, 0, 0]) if abs(v0[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(v0, t); x /= np.linalg.norm(x)
    y = np.cross(v0, x)
    return x, y


# ═════════════════════════════════════════════════════════════════════════════
# 6.1  convergenza di n_iso (notebook 06, dati veri, v1.3.6)
# ═════════════════════════════════════════════════════════════════════════════
CONVERGENZA_06 = pd.DataFrame({
    'n_iso': [4, 6, 8, 10, 12, 16, 20, 24, 31, 40], 'colonne': [9, 11, 10, 12, 14, 18, 22, 25, 33, 42],
    'FF': [0.5013, 0.4998, 0.4991, 0.4994, 0.4993, 0.5028, 0.5037, 0.5046, 0.5037, 0.5027],
    'RF': [0.1108, 0.1115, 0.1118, 0.1107, 0.1115, 0.1120, 0.1123, 0.1123, 0.1123, 0.1120],
    'HF': [0.4450, 0.4422, 0.4487, 0.4449, 0.4471, 0.4472, 0.4443, 0.4414, 0.4397, 0.4404],
    'WF': [0.1713, 0.1718, 0.1718, 0.1712, 0.1715, 0.1716, 0.1710, 0.1708, 0.1709, 0.1707],
    'FA': [0.6759, 0.6842, 0.6757, 0.6841, 0.6850, 0.6889, 0.6893, 0.6895, 0.6876, 0.6893],
    'NNLS_non_conv_pct': [0.0, 0.0, 0.0, 0.03, 0.77, 3.7, 7.0, 10.0, 19.5, 40.2],
    'secondi': [126, 121, 140, 133, 153, 203, 254, 303, 409, 614]})


def fig_convergenza(sintetica=None, file=None):
    T = CONVERGENZA_06
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    a = axs[0]
    for i, k in enumerate(('FF', 'RF', 'HF', 'WF')):
        a.plot(T.n_iso, T[k], 'o-', color=[COMP['F'], COMP['R'], COMP['H'], COMP['W']][i], lw=1.8, label=k)
    a.axvline(6, color=INK2, ls='--', lw=1); a.text(6.3, 0.52, 'n_iso = 6', fontsize=8, color=INK2)
    a.set_xlabel('n_iso'); a.set_ylabel('mediana su 3000 voxel')
    a.set_title('Frazioni: non si muovono (Stage D)')
    a.legend(fontsize=8, ncol=4, loc='center right')
    a = axs[1]
    a.plot(T.n_iso, T.FA, 'o-', color=INK, lw=1.8)
    rif = T[T.n_iso.isin([16, 20])].FA.mean()
    a.axhspan(rif - 0.01, rif + 0.01, color=CAT[2], alpha=0.15, lw=0, label='riferimento ±0.01 (media 16, 20)')
    a.axvline(6, color=INK2, ls='--', lw=1)
    a.set_xlabel('n_iso'); a.set_ylabel('FA mediana')
    a.set_title('Tensore: dentro la banda da 6')
    a.legend(fontsize=7.5, loc='lower right')
    a = axs[2]
    a.bar(T.n_iso.astype(str), T.NNLS_non_conv_pct, color=CAT[7])
    a2 = a.twinx()
    a2.plot(T.n_iso.astype(str), T.secondi, 'o-', color=INK2, lw=1)
    a2.set_ylabel('secondi (3000 voxel)', color=INK2)
    a.set_xlabel('n_iso'); a.set_ylabel('NNLS non convergenti [%]', color=CAT[7])
    a.set_title('Più colonne = più costo, niente informazione')
    fig.suptitle('Convergenza di n_iso — notebook 06, un soggetto vero, 3000 voxel (v1.3.6)', fontsize=10)
    if sintetica is not None:
        fig2, b = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        for i, k in enumerate(('FF', 'RF', 'HF', 'WF')):
            b[0].plot(sintetica.n_iso, sintetica[k], 'o-', color=[COMP['F'], COMP['R'], COMP['H'], COMP['W']][i], label=k)
        b[0].set_xlabel('n_iso'); b[0].set_ylabel('MAE contro la verità'); b[0].legend(fontsize=8, ncol=4)
        b[0].set_title(f'Fantoccio, toolbox {dbsi_toolbox.__version__}: errore delle frazioni')
        b[1].plot(sintetica.n_iso, sintetica['RDw'], 'o-', color=INK, label='RD pesata')
        b[1].set_xlabel('n_iso'); b[1].set_ylabel('MAE [e-3 mm²/s]'); b[1].legend(fontsize=8)
        b[1].set_title('ed errore del tensore')
        if file:
            fig2.savefig(str(file).replace('.png', '_sintetica.png'))
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 7  il campione di calibrazione e il suo sigma
# ═════════════════════════════════════════════════════════════════════════════
def rapporti_sigma(data_corr, mask, bvals, noise_map, b0_thr=100.0):
    """sigma_v / S0_v su tutta la maschera, con la stessa esclusione del toolbox (S0 < 5% della mediana)."""
    b0 = bvals < b0_thr
    S0 = data_corr[..., b0].mean(-1)[mask]
    ok = S0 > 0.05 * np.median(S0)
    return (noise_map[mask][ok] / S0[ok]).astype(float)


def fig_sigma_campione(r, n_list=(100, 250, 500, 1000, 2000, 4000), n_rip=200, file=None, seed=0):
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), constrained_layout=True)
    a = axs[0]
    hi = np.percentile(r, 99.5)
    a.hist(np.clip(r, 0, hi), bins=80, color=CAT[0], alpha=0.75)
    med, rms = np.median(r), np.sqrt(np.mean(r ** 2))
    a.axvline(med, color=CAT[1], lw=2, label=f'mediana {med:.4f}')
    a.axvline(rms, color=CAT[2], lw=2, label=f'RMS {rms:.4f}')
    a.set_xlabel('σ_v / S0_v  (rumore normalizzato del voxel)'); a.set_ylabel('voxel')
    a.set_title(f'Eterogeneo: RMS² / mediana² = {(rms / med) ** 2:.2f}')
    a.legend(fontsize=8)
    a = axs[1]
    out = []
    for n in n_list:
        for _ in range(n_rip):
            s = rng.choice(r, size=min(n, len(r)), replace=False)
            out.append(dict(n=n, mediana=np.median(s) / med - 1, RMS=np.sqrt(np.mean(s ** 2)) / rms - 1))
    D = pd.DataFrame(out)
    pos = np.arange(len(n_list))
    for j, (k, c) in enumerate((('mediana', CAT[1]), ('RMS', CAT[2]))):
        q = D.groupby('n')[k].quantile([0.05, 0.5, 0.95]).unstack()
        a.fill_between(pos, 100 * q[0.05], 100 * q[0.95], color=c, alpha=0.2, lw=0)
        a.plot(pos, 100 * q[0.5], 'o-', color=c, lw=1.8, label=f'{k}: 5–95° percentile')
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(pos); a.set_xticklabels([str(n) for n in n_list])
    a.set_xlabel('voxel nel campione'); a.set_ylabel('scarto dal valore su tutto il cervello [%]')
    a.set_title('Quanto cambia σ_cal fra campioni diversi')
    a.legend(fontsize=8)
    return _salva(fig, file), D


def fig_lambda_vs_campione(tab, file=None):
    """tab: colonne n, seme, lambda_aniso, lambda_iso."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ns = sorted(tab.n.unique())
    for a, k in zip(axs, ('lambda_aniso', 'lambda_iso')):
        for i, n in enumerate(ns):
            v = tab[tab.n == n][k]
            a.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=CAT[0], s=24, zorder=3)
            vals, cnt = np.unique(np.round(v, 6), return_counts=True)
            a.text(i, v.max() * 1.15, f'{cnt.max()}/{len(v)} uguali', ha='center', fontsize=7.5, color=INK2)
        a.set_yscale('log'); a.set_xticks(range(len(ns))); a.set_xticklabels([str(n) for n in ns])
        a.set_xlabel('voxel di calibrazione'); a.set_ylabel(k)
        a.set_title(f'{k}: un punto per seme del campione')
        a.grid(axis='y', which='both')
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 8-9  la calibrazione passo per passo
# ═════════════════════════════════════════════════════════════════════════════
def residuo_discrepanza(din, lam_grid):
    """R(lambda) del principio di discrepanza (stessa formula del toolbox, NNLS in parallelo)."""
    AtA, At, y = din['AtA'], din['At'], din['y']
    na, li = din['n_aniso_cols'], din['lambda_iso']
    Aty = np.stack([At @ v for v in y])
    out = []
    for lam in lam_grid:
        reg = np.zeros(AtA.shape[0]); reg[:na] = lam; reg[na:] = li
        W, _ = SOL.nnls_batch(AtA + np.diag(reg), Aty, 0.0)
        r = [max(float(w @ (AtA @ w) - 2 * w @ a + v @ v), 0) for w, a, v in zip(W, Aty, y)]
        out.append(np.mean(r))
    return np.array(out)


def residuo_cap(cap):
    """Residuo medio del blocco isotropo sul residuo senza fibra, per ogni lambda della griglia del tetto."""
    from dbsi_toolbox.core.basis import build_isotropic_dictionary
    bv = cap['bvals']
    A = build_isotropic_dictionary(bv, cap['iso_grid'])
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    y = cap['y']; Uty = U.T @ y.T; yn = np.sum(y ** 2, 1)
    out = []
    for lam in cap['grid']:
        f = s ** 2 / (s ** 2 + lam)
        out.append(np.mean(yn - 2 * np.sum((f[:, None]) * Uty ** 2, 0) + np.sum((f[:, None] ** 2) * Uty ** 2, 0)))
    return np.array(out)


def fig_calibrazione(R, prot, file_prefix=None, n_curva=30):
    """Sei pannelli: passo 1 (GCV iso sul segnale pieno), 2 (discrepanza per lambda_aniso preliminare),
    3 (residuo isotropo su due voxel), 4-5 (GCV sul residuo + tetto), 9 (GCV di lambda_aniso con i dof)."""
    bv = prot[0]
    N = len(bv)
    figs = {}
    g1, g4 = R['gcv_iso'][0], R['gcv_iso'][-1]
    sig = R['disc_in']['sigma']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    # passo 1
    a = axs[0, 0]
    a.loglog(g1['diag']['lambda_'], g1['diag']['gcv'], color=CAT[0], lw=2)
    a.axvline(g1['lam'], color=CAT[1], ls='--'); a.text(g1['lam'] * 1.2, a.get_ylim()[0] * 1.2, f"λ_iso(1) = {g1['lam']:.3g}", color=CAT[1], fontsize=8)
    a.set_xlabel('λ_iso'); a.set_ylabel('GCV'); a.set_title('Passo 1 — GCV del blocco isotropo, segnale pieno')
    # passo 2
    a = axs[0, 1]
    din = R['disc_in']
    lam_g = np.logspace(-6, 4, n_curva)
    Rl = residuo_discrepanza(din, lam_g)
    target = N * sig ** 2
    a.loglog(lam_g, Rl, color=CAT[0], lw=2, label='R(λ): residuo medio')
    a.axhline(target, color=INK2, ls='--', label=f'bersaglio Nσ² = {target:.3g}')
    ev = [l for l in R.get('disc_lambda_valutati', []) if np.isfinite(l) and l > 0]
    if ev:
        Re = residuo_discrepanza(din, np.array(ev))
        a.scatter(ev, Re, c=np.arange(len(ev)), cmap='Oranges', s=26, zorder=3, edgecolors=INK, lw=0.4,
                  label=f'bisezione: {len(ev)} valutazioni')
    la1 = R['disc_out'][0]
    a.axvline(la1, color=CAT[1], ls='--')
    a.text(la1 * 1.5, target * 0.85, f'λ_aniso(1) = {la1:.3g}', color=CAT[1], fontsize=8)
    fl = R['disc_out'][1]
    a.set_xlabel('λ_aniso'); a.set_ylabel('R(λ)')
    a.set_title('Passo 2 — discrepanza: il residuo = rumore')
    a.legend(fontsize=7.5, loc='upper left')
    if fl.get('floor_applied'):
        a.text(0.02, 0.02, f"pavimento applicato ({fl.get('floor_component')})", transform=a.transAxes, fontsize=7.5)
    # passo 3: residuo isotropo per due voxel del campione
    a = axs[0, 2]
    y = din['y']; r = g4['y']
    o = np.argsort(bv, kind='stable')
    an = np.linalg.norm(y - r, axis=1)
    for j, (v, lab) in enumerate(((int(np.argmax(an)), 'voxel con molta fibra'), (int(np.argmin(an)), 'voxel quasi isotropo'))):
        a.plot(y[v][o], color=CAT[j], lw=1.2, label=f'{lab}: segnale y')
        a.plot(r[v][o], color=CAT[j], lw=1.2, ls='--', label='residuo isotropo r = y − A_aniso ŵ')
    a.set_xlabel('misura (ordinate per b)'); a.set_ylabel('S/S0')
    a.set_title('Passo 3 — togliere la fibra')
    a.legend(fontsize=7)
    # passo 4-5
    a = axs[1, 0]
    a.loglog(g4['diag']['lambda_'], g4['diag']['gcv'], color=CAT[0], lw=2, label='GCV sul residuo isotropo')
    a.axvline(g4['lam'], color=CAT[0], ls='--')
    a.set_xlabel('λ_iso'); a.set_ylabel('GCV')
    a2 = a.twinx()
    cp = dict(R['cap']); cp['bvals'] = bv
    rc = residuo_cap(cp)
    a2.loglog(cp['grid'], rc, color=CAT[2], lw=1.6, label='residuo iso (tetto)')
    a2.axhline(N * sig ** 2, color=CAT[2], ls=':', lw=1)
    a2.axvline(cp['out'], color=CAT[2], ls='--')
    a2.set_ylabel('residuo medio', color=CAT[2])
    fin = R['lambda_iso']
    a.set_title(f"Passi 4–5 — GCV {g4['lam']:.3g}, tetto {cp['out']:.3g} → λ_iso {fin:.3g}")
    a.legend(fontsize=7.5, loc='upper left'); a2.legend(fontsize=7.5, loc='upper center')
    # passo 9
    d = R['aniso']['diag']
    a = axs[1, 1]
    a.loglog(d['lambda_grid'], d['gcv'], color=CAT[0], lw=2)
    a.axvline(R['aniso']['lam'], color=CAT[1], ls='--')
    a.text(R['aniso']['lam'] * 1.3, np.nanmin(d['gcv']) * 1.1, f"λ_aniso = {R['aniso']['lam']:.3g}", color=CAT[1], fontsize=8)
    sp = np.nanmax(d['gcv'][np.isfinite(d['gcv'])]) / np.nanmin(d['gcv'])
    a.set_xlabel('λ_aniso'); a.set_ylabel('GCV')
    a.set_title(f"Passo 9 — GCV di λ_aniso (max/min {sp:.1f}{', DEBOLE' if d.get('corner_weak') else ''})")
    a = axs[1, 2]
    a.semilogx(d['lambda_grid'], d['dof'], color=INK, lw=2, label='gradi di libertà ridge')
    a.axhline(N, color=INK2, ls=':', label=f'N = {N} misure')
    i = int(np.argmin(np.abs(d['lambda_grid'] - R['aniso']['lam'])))
    a.plot(d['lambda_grid'][i], d['dof'][i], 'o', color=CAT[1])
    a.text(d['lambda_grid'][i] * 1.3, d['dof'][i], f"{d['dof'][i]:.1f} dof", color=CAT[1], fontsize=8)
    a.set_xlabel('λ_aniso'); a.set_ylabel('dof'); a.set_title('Quanti parametri "efficaci" restano')
    a.legend(fontsize=7.5)
    figs['calibrazione'] = _salva(fig, f'{file_prefix}_passi.png' if file_prefix else None)
    riepilogo = dict(sigma_cal=sig, lambda_iso_passo1=g1['lam'], lambda_aniso_passo2=la1,
                     lambda_iso_gcv=g4['lam'], lambda_iso_tetto=cp['out'], lambda_iso_finale=fin,
                     lambda_aniso_finale=R['aniso']['lam'], dof_al_minimo=float(d['dof'][i]),
                     dof_min=float(np.nanmin(d['dof'])), dof_max=float(np.nanmax(d['dof'])),
                     gcv_max_su_min=float(sp), N=N, voxel=len(din['y']))
    return figs, riepilogo


# ═════════════════════════════════════════════════════════════════════════════
# Stage A di un voxel: dal segnale ai pesi, ai bacini, alla modulazione
# ═════════════════════════════════════════════════════════════════════════════
def stage_a_voxel(K, ijk):
    """Ripete per un voxel quello che il kernel fa in Stage A, con le stesse matrici (intercettate)."""
    x, y, z = ijk
    bv = K['prot'][0]
    sig = np.asarray(K['data'][x, y, z], float)
    s0 = sig[bv < K['b0_thr']].mean()
    sn = sig / s0
    At, AtA_reg = np.asarray(K['At']), np.asarray(K['AtA_reg'])
    Aty = At @ sn
    w1, it = SOL.nnls_coordinate_descent(AtA_reg, Aty, 0.0)
    na, nd, npair = K['n_aniso'], int(K['n_dirs']), K['n_pairs']
    nb, fd = np.asarray(K['neighbor_idx']), np.asarray(K['fiber_dirs'])
    c1 = SOL.dominant_basin_concentration(w1[:na], nd, npair, nb, fd)
    w2 = M._apply_conc_modulation(w1, AtA_reg, Aty, na, nd, npair, nb, fd, K['lambda_aniso_scalar'],
                                  K['conc_mod_c_lo'], K['conc_mod_c_hi'], K['conc_mod_gain'])
    c2 = SOL.dominant_basin_concentration(w2[:na], nd, npair, nb, fd)
    frac = (K['conc_mod_c_hi'] - c1) / (K['conc_mod_c_hi'] - K['conc_mod_c_lo'])
    g = 1 + (K['conc_mod_gain'] - 1) * min(max(frac, 0), 1) if c1 < K['conc_mod_c_hi'] else 1.0
    return dict(sn=sn, s0=s0, w1=w1, w2=w2, c1=c1, c2=c2, g=g,
                ff1=w1[:na].sum() / w1.sum(), ff2=w2[:na].sum() / w2.sum(),
                Wd1=w1[:na].reshape(npair, nd).sum(0), Wd2=w2[:na].reshape(npair, nd).sum(0))


def bacini(Wd, nb, fd):
    """Picchi e bacini come in dominant_basin_concentration (ricalcolati per disegnarli)."""
    nd, k = nb.shape
    picchi = []
    for d in range(nd):
        if Wd[d] <= 0:
            continue
        if all(not (Wd[j] > Wd[d] or (Wd[j] == Wd[d] and j < d)) for j in nb[d]):
            picchi.append(d)
    lab = np.full(nd, -1)
    if picchi:
        cos = np.abs(fd @ fd[picchi].T)
        lab = np.where(Wd > 0, np.argmax(cos, 1), -1)
    massa = np.array([Wd[lab == q].sum() for q in range(len(picchi))])
    c = massa.max() / Wd.sum() if len(massa) and Wd.sum() > 0 else 0.0
    return picchi, lab, massa, c


def fig_concentrazione(K, voxel, file=None):
    """voxel: lista di (nome, (i,j,k), direzioni vere o None)."""
    fd, nb = np.asarray(K['fiber_dirs']), np.asarray(K['neighbor_idx'])
    x, y = _proietta(fd)
    n = len(voxel)
    fig, axs = plt.subplots(2, n, figsize=(3.3 * n, 6.6), constrained_layout=True)
    righe = []
    for j, (nome, ijk, veri) in enumerate(voxel):
        S = stage_a_voxel(K, ijk)
        for r, (Wd, cc, ff, tit) in enumerate(((S['Wd1'], S['c1'], S['ff1'], 'prima soluzione'),
                                               (S['Wd2'], S['c2'], S['ff2'], 'dopo la modulazione'))):
            a = axs[r, j]
            _cerchio_equatore(a)
            pk, lab, massa, c_ric = bacini(Wd, nb, fd)
            col = np.array(['#d9d7d0'] * len(fd), dtype=object)
            for q in range(len(pk)):
                col[lab == q] = CAT[q % len(CAT)]
            sz = 4 + 260 * Wd / max(Wd.max(), 1e-12)
            a.scatter(x, y, s=sz, c=list(col), alpha=0.85, edgecolors='none')
            if veri is not None:
                vx, vy = _proietta(np.asarray(veri))
                a.scatter(vx, vy, s=90, marker='x', color=INK, lw=2)
            assert abs(c_ric - cc) < 1e-9, 'bacini ricalcolati diversi dal toolbox'
            a.set_title(f'{nome}\n{tit}: c = {cc:.2f}, FF {ff:.2f}' if r == 0 else f'{tit}: c = {cc:.2f}, FF {ff:.2f}',
                        fontsize=8.5)
        righe.append(dict(voxel=nome, c_prima=S['c1'], guadagno=S['g'], FF_prima=S['ff1'], FF_dopo=S['ff2'],
                          c_dopo=S['c2']))
    fig.suptitle('Peso anisotropo per direzione (area = peso; colore = bacino; × = fibra vera)', fontsize=10)
    return _salva(fig, file), pd.DataFrame(righe)


def fig_rampa(K, c_fibra=None, c_isotropo=None, file=None):
    clo, chi, G = K['conc_mod_c_lo'], K['conc_mod_c_hi'], K['conc_mod_gain']
    c = np.linspace(0, 1, 400)
    g = np.where(c >= chi, 1, 1 + (G - 1) * np.clip((chi - c) / (chi - clo), 0, 1))
    fig, a = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    a.plot(c, g, color=INK, lw=2.2, label='guadagno g(c) su λ_aniso')
    a.axvline(clo, color=INK2, ls=':'); a.axvline(chi, color=INK2, ls=':')
    a.text(clo, G + 0.2, f'c_lo {clo}', ha='center', fontsize=8); a.text(chi, G + 0.2, f'c_hi {chi}', ha='center', fontsize=8)
    a.set_xlabel('concentrazione c'); a.set_ylabel('g'); a.set_ylim(0.5, G + 1)
    if c_fibra is not None:
        a2 = a.twinx()
        b = np.linspace(0, 1, 41)
        a2.hist(c_isotropo, bins=b, color=CAT[7], alpha=0.35, density=True, label='voxel senza fibra vera')
        a2.hist(c_fibra, bins=b, color=CAT[0], alpha=0.35, density=True, label='voxel con fibra vera')
        a2.set_yticks([]); a2.legend(fontsize=8, loc='center right')
    a.legend(fontsize=8, loc='upper right')
    a.set_title('La rampa: penalità piena dove il peso è diffuso, nessuna dove è concentrato')
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 17.2  il cono su un voxel vero
# ═════════════════════════════════════════════════════════════════════════════
def _frazioni_stage_a(K, w):
    na = K['n_aniso']; iso = np.asarray(K['iso_grid']); wi = w[na:]
    tot = w.sum()
    res = iso <= M.THRESH_RES
    f_res, f_non = wi[res].sum() / tot, wi[~res].sum() / tot
    D_res = (wi[res] * iso[res]).sum() / max(wi[res].sum(), 1e-12) if wi[res].sum() > 1e-10 else 0.15e-3
    D_non = (wi[~res] * iso[~res]).sum() / max(wi[~res].sum(), 1e-12) if wi[~res].sum() > 1e-10 else 1.5e-3
    return w[:na].sum() / tot, f_res, f_non, D_res, D_non


def fig_cono(K, ijk, vera=None, file=None):
    bv, bc = K['prot']
    S = stage_a_voxel(K, ijk)
    fd, nb = np.asarray(K['fiber_dirs']), np.asarray(K['neighbor_idx'])
    na, nd, npair = K['n_aniso'], int(K['n_dirs']), K['n_pairs']
    idx, _ = SOL.select_dominant_directions(S['w2'][:na], nd, npair, nb, fd, 2, K['min_weight_fraction'],
                                            K['min_separation_cos'], K['min_peak_ratio'],
                                            K['min_dominant_concentration'])
    v0 = fd[idx[0]]                      # il nodo dominante, come nel kernel
    f_fib, f_res, f_non, D_res, D_non = _frazioni_stage_a(K, S['w2'])
    a1, n1, a2, n2 = K['cone1_half_angle'], int(K['n1_cone']), K['cone2_half_angle'], int(K['n2_cone'])

    def sse(d):
        AD, RD = SOL.estimate_AD_RD_conditioned(bv, bc, S['sn'], d, f_fib, f_res, f_non, 0.0, D_res, D_non, 0.0, False)
        if np.isnan(AD):
            return np.nan
        cos = bc @ d
        pred = f_fib * np.exp(-bv * (RD + (AD - RD) * cos ** 2)) + f_res * np.exp(-bv * D_res) + f_non * np.exp(-bv * D_non)
        return float(np.sum((S['sn'] - pred) ** 2))
    ex, ey = _base(v0)
    c1 = np.array([SOL._fibonacci_cone_point(*v0, *ex, *ey, a1, i, n1) for i in range(n1)])
    s1 = np.array([sse(c) for c in c1]); s0 = sse(v0)
    b1 = c1[np.nanargmin(s1)] if np.nanmin(s1) < s0 else v0
    bx_, by_ = _base(b1)
    c2 = np.array([SOL._fibonacci_cone_point(*b1, *bx_, *by_, a2, i, n2) for i in range(n2)])
    s2 = np.array([sse(c) for c in c2])
    tutti = np.vstack([v0[None], c1, c2]); ss = np.concatenate([[s0], s1, s2])
    scelta = tutti[np.nanargmin(ss)]
    tb, _, _ = SOL.refine_fiber_direction_cone(bv, bc, S['sn'], v0, f_fib, f_res, f_non, 0.0, D_res, D_non, 0.0,
                                                False, a1, n1, a2, n2)
    assert np.allclose(tb, scelta), 'il cono ricalcolato non sceglie la stessa direzione del toolbox'

    def loc(v):
        v = np.atleast_2d(v)
        v = v * np.sign(v @ v0)[:, None]
        return np.degrees(np.arctan2(v @ ex, v @ v0)), np.degrees(np.arctan2(v @ ey, v @ v0))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    a = axs[0]
    vic = fd[np.argsort(-np.abs(fd @ v0))[:7]]
    a.scatter(*loc(vic), s=70, color='#d9d7d0', label='nodi del dizionario')
    sc = a.scatter(*loc(np.vstack([c1, c2])), c=np.concatenate([s1, s2]), cmap='viridis_r', s=28, zorder=3)
    fig.colorbar(sc, ax=a, label='SSE del candidato')
    a.scatter(*loc(v0), s=110, marker='*', color=INK, label='nodo di partenza', zorder=4)
    a.scatter(*loc(scelta), s=120, facecolors='none', edgecolors=CAT[7], lw=2, label='scelto (= toolbox)', zorder=5)
    if vera is not None:
        a.scatter(*loc(np.asarray(vera)), s=90, marker='x', color=CAT[7], lw=2, label='direzione vera', zorder=5)
    a.set_aspect('equal'); a.set_xlabel('gradi'); a.set_ylabel('gradi')
    a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3)
    ang = lambda u, v: np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1)))
    a.set_title('Candidati dei due livelli colorati per SSE')
    a = axs[1]
    e = [ang(v0, vera), ang(scelta, vera)] if vera is not None else [np.nan, np.nan]
    a.bar(['nodo di Stage A', 'dopo il cono'], e, color=[INK2, CAT[2]], width=0.55)
    for i, v in enumerate(e):
        a.text(i, v, f'{v:.1f}°', ha='center', va='bottom', fontsize=9)
    a.set_ylabel('errore angolare [gradi]'); a.set_title('Quanto si avvicina alla direzione vera')
    return _salva(fig, file), dict(errore_nodo=e[0], errore_cono=e[1], sse_nodo=s0, sse_scelto=float(np.nanmin(ss)))


# ═════════════════════════════════════════════════════════════════════════════
# 18  crossing: FF ri-stimata e MRDS
# ═════════════════════════════════════════════════════════════════════════════
def ff_contro_kappa(K, ijk, kappas):
    """FF del crossing ri-stimata a ogni kappa per un voxel (None se Stage A non vede due fibre)."""
    bv, bc = K['prot']
    S = stage_a_voxel(K, ijk)
    fd, nb, pairs, iso = (np.asarray(K[k]) for k in ('fiber_dirs', 'neighbor_idx', 'diff_pairs', 'iso_grid'))
    na, nd, npair = K['n_aniso'], int(K['n_dirs']), K['n_pairs']
    idx, _ = SOL.select_dominant_directions(S['w2'][:na], nd, npair, nb, fd, 2, K['min_weight_fraction'],
                                            K['min_separation_cos'], K['min_peak_ratio'],
                                            K['min_dominant_concentration'])
    if idx[1] < 0:
        return None
    f_fib = _frazioni_stage_a(K, S['w2'])[0]
    sv = float(np.asarray(K['sigma_map'])[tuple(ijk)])
    out = []
    for kap in kappas:
        io_ = np.empty(len(bv))
        out.append(SOL.crossing_ff_resolve(S['sn'], bv, bc, fd, fd[idx[:2]], 2, pairs, iso,
                                           kap * (sv / S['s0']) ** 2, io_) if kap > 0 else f_fib)
    return np.array(out)


def fig_crossing(K, ijk, verita=None, file=None, kappas=(0, 1, 3, 10, 30, 100, 300, 1000), altri=None):
    bv, bc = K['prot']
    S = stage_a_voxel(K, ijk)
    fd, nb, pairs, iso = (np.asarray(K[k]) for k in ('fiber_dirs', 'neighbor_idx', 'diff_pairs', 'iso_grid'))
    na, nd, npair = K['n_aniso'], int(K['n_dirs']), K['n_pairs']
    idx, wts = SOL.select_dominant_directions(S['w2'][:na], nd, npair, nb, fd, 2, K['min_weight_fraction'],
                                              K['min_separation_cos'], K['min_peak_ratio'],
                                              K['min_dominant_concentration'])
    assert idx[1] >= 0, 'il voxel scelto non e\' un crossing per Stage A'
    dirs = fd[idx[:2]]
    f_fib, f_res, f_non, D_res, D_non = _frazioni_stage_a(K, S['w2'])
    quote = wts[:2] / wts[:2].sum()
    sigma_v = float(np.asarray(K['sigma_map'])[tuple(ijk)])
    ff_k = []
    for kap in kappas:
        iso_out = np.empty(len(bv))
        ff = SOL.crossing_ff_resolve(S['sn'], bv, bc, fd, dirs, 2, pairs, iso, kap * (sigma_v / S['s0']) ** 2, iso_out) \
            if kap > 0 else f_fib
        ff_k.append(ff)
    iso_out = np.empty(len(bv))
    kap0 = K['crossing_ff_kappa']
    ff_r = SOL.crossing_ff_resolve(S['sn'], bv, bc, fd, dirs, 2, pairs, iso, kap0 * (sigma_v / S['s0']) ** 2, iso_out)
    fr = quote * ff_r
    traiett = []
    for n_lm in range(0, 26):
        AD, RD = SOL.estimate_AD_RD_mrds(bv, bc, S['sn'], dirs, fr, iso_out, 3, n_lm)
        E = np.exp(-bv[:, None] * (RD[None] + (AD - RD)[None] * (bc @ dirs.T) ** 2))
        C = float(np.sum((S['sn'] - iso_out - E @ fr) ** 2))
        traiett.append(dict(it=n_lm, C=C, AD1=AD[0], RD1=RD[0], AD2=AD[1], RD2=RD[1]))
    T = pd.DataFrame(traiett)
    fin = K['res'][tuple(ijk)]
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.1), constrained_layout=True)
    a = axs[0]
    a.semilogx([max(k, 0.3) for k in kappas], ff_k, 'o-', color=INK, lw=1.8, label='FF ri-stimata (questo voxel)')
    a.axhline(f_fib, color=INK2, ls=':', label=f'FF di Stage A {f_fib:.2f}')
    if verita is not None:
        a.axhline(verita['FF'], color=CAT[2], ls='--', label=f"FF vera {verita['FF']:.2f}")
    if altri:
        F = np.array([f for f in (ff_contro_kappa(K, v, kappas) for v in altri) if f is not None])
        if len(F):
            xk = [max(k, 0.3) for k in kappas]
            a.fill_between(xk, np.percentile(F, 10, 0), np.percentile(F, 90, 0), color=CAT[0], alpha=0.18, lw=0,
                           label=f'tutti i crossing ({len(F)} voxel): 10–90°')
            a.plot(xk, np.median(F, 0), color=CAT[0], lw=1.6, label='tutti i crossing: mediana')
    a.axvline(kap0, color=CAT[1], lw=1); a.text(kap0 * 1.1, min(ff_k), f'κ = {kap0:g}', color=CAT[1], fontsize=8)
    a.set_xlabel('κ (κ=0 disegnato a 0.3: tiene la FF di Stage A)'); a.set_ylabel('FF del crossing')
    a.set_title('La FF del crossing sul supporto rilevato'); a.legend(fontsize=7.5)
    a = axs[1]
    a.semilogy(T.it, T.C, 'o-', color=INK, lw=1.6, ms=4)
    a.set_xlabel('iterazioni di Levenberg–Marquardt (dopo 3 alternate)'); a.set_ylabel('costo C(p)')
    a.set_title('MRDS: il costo scende e si ferma')
    a = axs[2]
    for k, c, ls in (('AD1', CAT[0], '-'), ('RD1', CAT[0], '--'), ('AD2', CAT[1], '-'), ('RD2', CAT[1], '--')):
        a.plot(T.it, T[k] * 1e3, ls, color=c, lw=1.8, label=k.replace('1', ' pop1').replace('2', ' pop2'))
    if verita is not None:
        for k, c in (('AD', INK2), ('RD', INK2)):
            for v in verita[k]:
                a.axhline(v * 1e3, color=c, lw=0.8, ls=':')
    a.set_xlabel('iterazioni LM'); a.set_ylabel('[e-3 mm²/s]'); a.set_title('I quattro parametri (punteggiato = veri)')
    a.legend(fontsize=7.5, ncol=2)
    return _salva(fig, file), T, dict(FF_stageA=f_fib, FF_ristimata=ff_r, kappa=kap0,
                                      ridge=kap0 * (sigma_v / S['s0']) ** 2, quote=quote.tolist(),
                                      AD_finale=[fin[IX['axial_diffusivity_pop1']], fin[IX['axial_diffusivity_pop2']]],
                                      RD_finale=[fin[IX['radial_diffusivity_pop1']], fin[IX['radial_diffusivity_pop2']]])


# ═════════════════════════════════════════════════════════════════════════════
# 18b  il cono per popolazione nei crossing (opzione crossing_dir_refine)
# ═════════════════════════════════════════════════════════════════════════════
def cono_crossing_voxel(K, ijk):
    """Ripete per un voxel di crossing il percorso del kernel: FF ri-stimata sul supporto, MRDS sulle direzioni
    della griglia, poi il cono per popolazione (refine_crossing_directions_cone). Ritorna prima e dopo."""
    bv, bc = K['prot']
    S = stage_a_voxel(K, ijk)
    fd, nb, pairs, iso = (np.asarray(K[k]) for k in ('fiber_dirs', 'neighbor_idx', 'diff_pairs', 'iso_grid'))
    na, nd, npair = K['n_aniso'], int(K['n_dirs']), K['n_pairs']
    idx, wts = SOL.select_dominant_directions(S['w2'][:na], nd, npair, nb, fd, 2, K['min_weight_fraction'],
                                              K['min_separation_cos'], K['min_peak_ratio'],
                                              K['min_dominant_concentration'])
    assert idx[1] >= 0, "il voxel scelto non e' un crossing per Stage A"
    dirs = fd[idx[:2]].copy()
    sigma_v = float(np.asarray(K['sigma_map'])[tuple(ijk)])
    iso_out = np.empty(len(bv))
    ff_r = SOL.crossing_ff_resolve(S['sn'], bv, bc, fd, dirs, 2, pairs, iso,
                                   K['crossing_ff_kappa'] * (sigma_v / S['s0']) ** 2, iso_out)
    fr = wts[:2] / wts[:2].sum() * ff_r
    AD, RD = SOL.estimate_AD_RD_mrds(bv, bc, S['sn'], dirs, fr, iso_out, M._MRDS_INIT_N_ITER, M._MRDS_LM_MAX_ITER)
    d2, AD2, RD2 = SOL.refine_crossing_directions_cone(
        bv, bc, S['sn'], dirs, fr, iso_out, AD, RD, K['min_separation_cos'], K['cone1_half_angle'],
        int(K['n1_cone']), K['cone2_half_angle'], int(K['n2_cone']), M._MRDS_CONE_LM_ITER)
    return dict(dirs0=dirs, AD0=AD, RD0=RD, dirs1=d2, AD1=AD2, RD1=RD2,
                sse0=float(SOL._mrds_sse(bv, bc, S['sn'], dirs, fr, iso_out, AD, RD)),
                sse1=float(SOL._mrds_sse(bv, bc, S['sn'], d2, fr, iso_out, AD2, RD2)))


def crossing_casuali(prot, n, snr, lam, seed=0, opzioni=None):
    """n crossing bilanciati (0.30 + 0.30, AD 1.7 RD 0.4 e-3, 60-90 gradi) con orientamento CASUALE: errore
    angolare delle due popolazioni dopo il fit a lambda congelati (NaN dove il fit non vede il crossing)."""
    rng = np.random.default_rng(seed)
    bv, bc = prot
    A = np.zeros((n, 1, 1, len(bv)), np.float32)
    vere = np.full((n, 1, 1, 2, 3), np.nan)
    for i in range(n):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v = ruota_verso(u, rng.uniform(60, 90), rng.normal(size=3))
        S = segnale(prot, [(0.3, u, 1.7e-3, 0.4e-3), (0.3, v, 1.7e-3, 0.4e-3)], dict(R=0.1, H=0.2, W=0.1), D_NOM)
        A[i, 0, 0] = S / S[bv < 100].mean()
        vere[i, 0, 0] = u, v
    Mm, sg = rumore_rician(A, snr, seed + 1)
    m = DBSI_Adaptive(lambda_aniso=lam[0], lambda_iso=lam[1], n_iso=6, **(opzioni or {}))
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(Mm, bv, bc, np.ones(A.shape[:3], bool), run_calibration=False,
                       noise_map=np.full(A.shape[:3], sg))
    return errore_angolare(res, vere).reshape(n, 2)


def _ang(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1))))


def fig_cono_crossing(K, C, vere, tab_err, file=None):
    """C = cono_crossing_voxel; vere = (2, 3) direzioni vere; tab_err: scenario, popolazione, braccio, mediana."""
    fd = np.asarray(K['fiber_dirs'])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    for k in (0, 1):
        a = axs[k]
        t = vere[np.argmax([abs(C['dirs0'][k] @ v) for v in vere])]      # la vera piu' vicina a questa popolazione
        ex, ey = _base(t)

        def loc(v):
            v = np.atleast_2d(v); v = v * np.sign(v @ t)[:, None]
            return np.degrees(np.arctan2(v @ ex, v @ t)), np.degrees(np.arctan2(v @ ey, v @ t))
        vic = fd[np.argsort(-np.abs(fd @ t))[:7]]
        a.scatter(*loc(vic), s=70, color='#d9d7d0', label='nodi del dizionario')
        a.scatter(*loc(C['dirs0'][k]), s=130, marker='*', color=INK, zorder=4,
                  label=f"Stage A + MRDS ({_ang(C['dirs0'][k], t):.1f}°)")
        a.scatter(*loc(C['dirs1'][k]), s=120, facecolors='none', edgecolors=CAT[2], lw=2.2, zorder=5,
                  label=f"dopo il cono ({_ang(C['dirs1'][k], t):.1f}°)")
        a.scatter(0, 0, s=90, marker='x', color=CAT[7], lw=2, zorder=5, label='direzione vera')
        a.set_aspect('equal'); a.set_xlabel('gradi'); a.set_ylabel('gradi')
        a.set_title(f'popolazione {k + 1}: AD/RD {C["AD0"][k]*1e3:.2f}/{C["RD0"][k]*1e3:.2f} → '
                    f'{C["AD1"][k]*1e3:.2f}/{C["RD1"][k]*1e3:.2f}', fontsize=9)
        a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    a = axs[2]
    sc = list(dict.fromkeys(tab_err.scenario))
    x = np.arange(len(sc) * 2)
    for j, (br, col) in enumerate((('griglia', INK2), ('cono', CAT[2]))):
        v = [tab_err[(tab_err.scenario == s) & (tab_err.popolazione == q) & (tab_err.braccio == br)].mediana.iloc[0]
             for s in sc for q in (1, 2)]
        a.bar(x + (j - 0.5) * 0.38, v, width=0.38, color=col, label=br)
    a.set_xticks(x, [f'{s}\npop {q}' for s in sc for q in (1, 2)], fontsize=7.5)
    a.set_ylabel('errore angolare mediano [gradi]'); a.legend(fontsize=8)
    a.set_title('Errore mediano: crossing della batteria e orientamenti casuali')
    fig.suptitle(f"Il cono per popolazione: SSE {C['sse0']:.2e} → {C['sse1']:.2e} su questo voxel", fontsize=10)
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 18c  il test 2-contro-1 (opzione two_vs_one_threshold)
# ═════════════════════════════════════════════════════════════════════════════
def fig_test21(stat, npop, T, snr_fig, file=None):
    """stat: DataFrame SNR, vero (popolazioni vere), stat21 dei candidati. npop: SNR, vero, fit ('1.8' o
    'test'), due_pct (quota dei voxel con 2 popolazioni stimate)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    a = axs[0]
    tr = lambda x: np.sign(x) * np.log10(1 + np.abs(x))          # logaritmo con segno: stat21 puo' essere < 0
    d = stat[stat.SNR == snr_fig]
    b = np.linspace(tr(min(-10, d.stat21.min())), tr(max(1e3, d.stat21.max())), 60)
    for v, col, nome in ((1, CAT[0], 'una fibra vera'), (2, CAT[1], 'due fibre vere')):
        q = d[d.vero == v].stat21.to_numpy()
        if q.size:
            a.hist(tr(q), bins=b, color=col, alpha=0.55, weights=np.full(q.size, 100 / q.size),
                   label=f'{nome} (n={q.size})')
    a.axvline(tr(T), color=INK, ls='--', lw=1.2)
    a.text(tr(T) * 1.05, a.get_ylim()[1] * 0.9, f'T = {T:g}', fontsize=8)
    tk = [v for v in (-10, -1, 0, 1, 10, 100, 1000, 10000) if b[0] <= tr(v) <= b[-1]]
    a.set_xticks([tr(v) for v in tk], [str(v) for v in tk])
    a.set_xlabel('stat21 = (RSS_1pop − RSS_2pop) / σ²  (scala log con segno)'); a.set_ylabel('% dei voxel')
    a.set_title(f'I candidati crossing, SNR {snr_fig}'); a.legend(fontsize=7.5)
    for a, v, tit in ((axs[1], 1, 'Falsi crossing (una fibra vera)'), (axs[2], 2, 'Crossing trovati (due fibre vere)')):
        for fit_, ls in (('1.8', '--'), ('test', '-')):
            s = npop[(npop.vero == v) & (npop.fit == fit_)].sort_values('SNR')
            a.plot(s.SNR, s.due_pct, ls, marker='o', color=INK if fit_ == '1.8' else CAT[2], lw=1.8,
                   label='min_peak_ratio 0.35 (fino alla 1.8)' if fit_ == '1.8' else f'test 2-contro-1, T = {T:g} (1.9.0)')
        a.set_xlabel('SNR'); a.set_ylabel('% con 2 popolazioni stimate'); a.set_title(tit); a.set_ylim(0, 100)
        a.grid(alpha=0.5)
    axs[1].legend(fontsize=7.5)
    return _salva(fig, file)


# ═════════════════════════════════════════════════════════════════════════════
# 23  incertezza, spiegata semplice
# ═════════════════════════════════════════════════════════════════════════════
def fig_bersaglio(stime, se, vero, nome_x, nome_y, titoli, file=None):
    """stime: lista di array (n, 2) per gruppo; se: lista di (se_x, se_y) mediane; vero: lista di (x, y)."""
    fig, axs = plt.subplots(1, len(stime), figsize=(5.2 * len(stime), 4.6), constrained_layout=True)
    axs = np.atleast_1d(axs)
    t = np.linspace(0, 2 * np.pi, 200)
    for a, s, e, v, tit in zip(axs, stime, se, vero, titoli):
        s = s[np.all(np.isfinite(s), 1)]
        a.scatter(s[:, 0], s[:, 1], s=10, color=CAT[0], alpha=0.5, label='stime (una per realizzazione del rumore)')
        m = s.mean(0)
        a.plot(v[0] + 1.96 * e[0] * np.cos(t), v[1] + 1.96 * e[1] * np.sin(t), color=CAT[2], lw=2,
               label='±1.96 SE del toolbox attorno al vero')
        a.plot(*v, marker='*', ms=16, color=INK, ls='', label='verità')
        a.plot(*m, marker='o', ms=9, color=CAT[7], ls='', label='media delle stime')
        a.annotate('', xy=m, xytext=v, arrowprops=dict(arrowstyle='->', color=CAT[7], lw=1.5))
        sd = s.std(0)
        a.set_title(f'{tit}\nbias ({m[0]-v[0]:+.3f}, {m[1]-v[1]:+.3f})   sd ({sd[0]:.3f}, {sd[1]:.3f})   SE ({e[0]:.3f}, {e[1]:.3f})',
                    fontsize=8.5)
        a.set_xlabel(nome_x); a.set_ylabel(nome_y)
    axs[0].legend(fontsize=7.5, loc='upper left')
    return _salva(fig, file)


def fig_se_protocolli(tab, file=None):
    """tab: colonne protocollo, SNR, metrica, se (mediana)."""
    met = list(dict.fromkeys(tab.metrica))
    fig, axs = plt.subplots(1, len(met), figsize=(4.0 * len(met), 3.5), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for a, k in zip(axs, met):
        for i, p in enumerate(dict.fromkeys(tab.protocollo)):
            s = tab[(tab.metrica == k) & (tab.protocollo == p)].sort_values('SNR')
            a.plot(s.SNR, s.se, 'o-', color=CAT[i], lw=1.8, label=p)
        a.set_xlabel('SNR'); a.set_ylabel('SE mediana'); a.set_title(k)
        a.grid(alpha=0.5)
    axs[-1].legend(fontsize=8)
    fig.suptitle('La SE dipende da protocollo e SNR, non dal tessuto stimato bene o male', fontsize=10)
    return _salva(fig, file)


def fig_residuo(rr_ok, rr_male, soglia, file=None):
    fig, a = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    b = np.linspace(0.4, max(3, np.nanpercentile(rr_male, 99)), 60)
    a.hist(rr_ok[np.isfinite(rr_ok)], bins=b, color=CAT[0], alpha=0.6, density=True, label='il modello descrive il voxel')
    a.hist(rr_male[np.isfinite(rr_male)], bins=b, color=CAT[7], alpha=0.6, density=True, label='il modello NON lo descrive')
    a.axvline(1, color=INK, lw=1); a.axvline(soglia, color=CAT[7], ls='--')
    a.text(soglia * 1.02, a.get_ylim()[1] * 0.85, 'flag 16', color=CAT[7], fontsize=8)
    a.set_xlabel('residuo / σ  (≈ 1 se modello e σ sono giusti)'); a.set_ylabel('densità')
    a.set_title('residual_over_sigma: il controllo che il modello basti')
    a.legend(fontsize=8)
    return _salva(fig, file)


def soglia_residuo(dof):
    """La soglia di residual_over_sigma del flag 16 (quantile 99.9% di chi2_dof / dof, radice)."""
    from dbsi_toolbox import uncertainty as U
    h = 2.0 / (9.0 * dof)
    return float(np.sqrt((1.0 - h + U._RESID_FLAG_Z * np.sqrt(h)) ** 3))


def fit_voxel_ripetuti(prot, fibre, iso, n, snr, lam, seed=0, D=None):
    """n voxel con la STESSA verita' e rumore Rician indipendente; fit di produzione a lambda congelati.
    Ritorna (risultati (n, canali), se (n, canali), flags (n,), residuo/sigma (n,))."""
    D = D or dict(D_NOM)
    S = segnale(prot, fibre, iso, D)
    S = S / (sum(f for f, *_ in fibre) + sum(iso.values()))
    A = np.repeat(S[None, None, None, :], n, axis=0).astype(np.float32)
    Mm, sg = rumore_rician(A, snr, seed)
    m = DBSI_Adaptive(lambda_aniso=lam[0], lambda_iso=lam[1], n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(Mm, prot[0], prot[1], np.ones(A.shape[:3], bool), run_calibration=False,
                       noise_map=np.full(A.shape[:3], sg))
    u = m.uncertainty_
    return (res.reshape(n, -1), u['se'].reshape(n, -1), u['flags'].reshape(n),
            u['residual_over_sigma'].reshape(n))
