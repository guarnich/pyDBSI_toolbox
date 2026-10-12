"""Figure delle simulazioni di accuratezza (matplotlib, PNG). Colori per ruolo:
- mappe di valore: rampa sequenziale a una tinta (blu), chiaro = 0;
- mappe di errore: divergente blu <-> rosso con grigio neutro a zero, scala simmetrica;
- categorie (SNR, bracci, protocolli): slot categoriali in ordine fisso, mai ciclati.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from sim_accuratezza import METRICHE, FRAZIONI, TENSORE, SCALA, UNITA

# palette di riferimento (validata): slot categoriali, rampa blu, divergente blu-rosso
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('seq_blu', ['#f7f9fc', '#cde2fb', '#86b6ef', '#3987e5',
                                                    '#256abf', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_br', ['#104281', '#3987e5', '#9ec5f4', '#f0efec',
                                                   '#f4a3a2', '#e34948', '#a12a2a'])
SEQ.set_bad('#ffffff'); DIV.set_bad('#ffffff')
NPOP_CMAP = ListedColormap(['#f0efec', CAT[0], CAT[1]])
NPOP_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

ETICHETTA = dict(FF='FF (fibra)', RF='RF (restricted)', HF='HF (hindered)', WF='WF (acqua libera)',
                 ADw='AD pesata', RDw='RD pesata', FAw='FA pesata', npop='n. popolazioni')
INTERVALLO = dict(FF=(0, 0.8), RF=(0, 0.6), HF=(0, 0.9), WF=(0, 1.0), ADw=(1.0, 2.2), RDw=(0.1, 1.0),
                  FAw=(0, 1.0))
ERR_MAX = dict(FF=0.25, RF=0.25, HF=0.25, WF=0.25, ADw=0.6, RDw=0.4, FAw=0.3)


def stile():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight', 'font.size': 9,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def _lab(k):
    u = UNITA.get(k, '')
    return ETICHETTA.get(k, k) + (f' [{u}]' if u else '')


# ─────────────────────────────────────────────────────────────────────────────
# immagini: vero / stimato / errore
# ─────────────────────────────────────────────────────────────────────────────
def mappe_vero_stima_errore(tru, est, z=None, metriche=None, titolo='', file=None):
    """Una riga per metrica: verita', stima, errore (stima - vero). Le diffusivita' in e-3 mm^2/s.
    Il tensore e' bianco dove non c'e' fibra (vera o stimata)."""
    metriche = list(metriche or (*METRICHE, 'npop'))
    z = tru['FF'].shape[2] // 2 if z is None else z
    fig, axs = plt.subplots(len(metriche), 3, figsize=(8.4, 2.45 * len(metriche)), constrained_layout=True)
    for r, k in enumerate(metriche):
        t = tru[k][:, :, z].T
        e = est[k][:, :, z].T
        if k == 'npop':
            for c, (img, nome) in enumerate(((t, 'vero'), (e, 'stimato'))):
                im = axs[r, c].imshow(img, origin='lower', cmap=NPOP_CMAP, norm=NPOP_NORM, interpolation='nearest')
                axs[r, c].set_title(f'{ETICHETTA[k]} — {nome}')
            cb = fig.colorbar(im, ax=axs[r, :2], ticks=[0, 1, 2], shrink=0.85)
            giusto = np.round(e) == np.round(t)
            axs[r, 2].imshow(np.where(giusto, 1.0, 0.0), origin='lower', cmap=ListedColormap(['#e34948', '#f0efec']),
                             vmin=0, vmax=1, interpolation='nearest')
            axs[r, 2].set_title(f'conteggio giusto: {100 * giusto.mean():.0f}% (rosso = sbagliato)')
        else:
            s = SCALA[k]
            lo, hi = INTERVALLO[k]
            for c, (img, nome) in enumerate(((t, 'vero'), (e, 'stimato'))):
                im = axs[r, c].imshow(img * s, origin='lower', cmap=SEQ, vmin=lo, vmax=hi, interpolation='nearest')
                axs[r, c].set_title(f'{ETICHETTA[k]} — {nome}')
            fig.colorbar(im, ax=axs[r, :2], shrink=0.85, label=UNITA[k] or None)
            d = (e - t) * s
            em = ERR_MAX[k]
            im2 = axs[r, 2].imshow(d, origin='lower', cmap=DIV, vmin=-em, vmax=em, interpolation='nearest')
            tit = f'errore: bias {np.nanmean(d):+.3f}, MAE {np.nanmean(np.abs(d)):.3f}'
            if k in TENSORE:
                # fibra vera ma nessun tensore stimato (nero) / tensore stimato senza fibra vera (rosa)
                mancata = np.isfinite(t) & ~np.isfinite(e)
                falsa = ~np.isfinite(t) & np.isfinite(e)
                ov = np.zeros(t.shape + (4,))
                ov[mancata] = (0.10, 0.10, 0.10, 1.0)
                ov[falsa] = (0.91, 0.48, 0.64, 1.0)
                axs[r, 2].imshow(ov, origin='lower', interpolation='nearest')
                nv = max(np.isfinite(t).sum(), 1)
                tit += f'\nmancate (nero) {100 * mancata.sum() / nv:.0f}%, false (rosa) {falsa.sum()} vox'
            axs[r, 2].set_title(tit, fontsize=9)
            fig.colorbar(im2, ax=axs[r, 2], shrink=0.85)
        for c in range(3):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig


def mappa_regioni(pesi, z=None, file=None):
    """Le regioni del fantoccio anatomico: il tessuto dominante in ogni voxel."""
    nomi = list(pesi)
    W = np.stack([pesi[n] for n in nomi], -1)
    z = W.shape[2] // 2 if z is None else z
    dom = W[:, :, z].argmax(-1).T
    pal = ['#d9d7d0', CAT[0], CAT[2], CAT[6], '#9ec5f4', CAT[1], CAT[3], CAT[7], CAT[4]]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    ax.imshow(dom, origin='lower', cmap=ListedColormap(pal[:len(nomi)]), vmin=-0.5, vmax=len(nomi) - 0.5,
              interpolation='nearest')
    for q, n in enumerate(nomi):
        ax.plot([], [], 's', color=pal[q], label=n, markersize=8)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Fantoccio anatomico: tessuto dominante')
    if file:
        fig.savefig(file)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# batteria: errore per scenario, per SNR o per braccio
# ─────────────────────────────────────────────────────────────────────────────
def errori_per_scenario(voxel, metriche, gruppo_col='SNR', titolo='', file=None, ordine=None):
    """voxel: DataFrame con colonne scenario, metrica, errore (stima - vero, gia' scalato), e gruppo_col.
    Punto = errore mediano, barra = 10-90 percentile dei voxel; una colonna per metrica."""
    gruppi = list(dict.fromkeys(voxel[gruppo_col])) if ordine is None else ordine
    scen = list(dict.fromkeys(voxel['scenario']))
    n = len(metriche)
    fig, axs = plt.subplots(1, n, figsize=(2.35 * n + 1.6, 0.36 * len(scen) + 1.4), sharey=True,
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    off = np.linspace(-0.25, 0.25, len(gruppi)) if len(gruppi) > 1 else [0.0]
    for a, k in zip(axs, metriche):
        a.axvline(0, color=INK2, lw=0.8)
        a.grid(axis='x')
        sub = voxel[voxel.metrica == k]
        for gi, g in enumerate(gruppi):
            s = sub[sub[gruppo_col] == g]
            for si, sc in enumerate(scen):
                e = s.loc[s.scenario == sc, 'errore'].to_numpy()
                e = e[np.isfinite(e)]
                if e.size < 3:
                    continue
                y = si + off[gi]
                lo, med, hi = np.percentile(e, [10, 50, 90])
                a.plot([lo, hi], [y, y], color=CAT[gi], lw=2, solid_capstyle='round')
                a.plot(med, y, 'o', color=CAT[gi], ms=5, mec='white', mew=1)
        a.set_title(_lab(k), fontsize=9)
        a.set_xlabel('stima − vero')
    axs[0].set_yticks(range(len(scen)))
    axs[0].set_yticklabels(scen)
    axs[0].invert_yaxis()
    for gi, g in enumerate(gruppi):
        axs[-1].plot([], [], 'o-', color=CAT[gi], label=f'{gruppo_col} {g}' if gruppo_col == 'SNR' else str(g))
    axs[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig


def conteggio_popolazioni(voxel_npop, gruppo_col='SNR', file=None, titolo=''):
    """Quota di voxel con 0 / 1 / 2 popolazioni stimate, per scenario (barra piena = verita')."""
    gruppi = list(dict.fromkeys(voxel_npop[gruppo_col]))
    scen = list(dict.fromkeys(voxel_npop['scenario']))
    fig, axs = plt.subplots(1, len(gruppi), figsize=(3.3 * len(gruppi) + 1.4, 0.36 * len(scen) + 1.4),
                            sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs)
    col = ['#c9c7c0', CAT[0], CAT[1]]
    for a, g in zip(axs, gruppi):
        s = voxel_npop[voxel_npop[gruppo_col] == g]
        for si, sc in enumerate(scen):
            v = s[s.scenario == sc]
            q = [np.mean(np.round(v.stima) == p) for p in (0, 1, 2)]
            left = 0
            for p in range(3):
                a.barh(si, q[p], left=left, color=col[p], height=0.7, edgecolor='white', linewidth=1)
                left += q[p]
            vero = int(np.round(v.vero.median()))
            a.text(1.02, si, f'vero {vero}', va='center', fontsize=7.5, color=INK2)
        a.set_xlim(0, 1.25); a.set_xticks([0, 0.5, 1]); a.set_xticklabels(['0', '50%', '100%'])
        a.set_title(f'{gruppo_col} {g}' if gruppo_col == 'SNR' else str(g), fontsize=9)
    axs[0].set_yticks(range(len(scen))); axs[0].set_yticklabels(scen); axs[0].invert_yaxis()
    from matplotlib.patches import Patch
    axs[-1].legend(handles=[Patch(color=col[p], label=f'{p} popolazioni') for p in range(3)],
                   loc='upper left', bbox_to_anchor=(1.0, 1))
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# campi continui: stima contro verita'
# ─────────────────────────────────────────────────────────────────────────────
def stima_contro_vero(tru, est, metriche=None, titolo='', file=None):
    metriche = list(metriche or METRICHE)
    n = len(metriche)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.0 * nr), constrained_layout=True)
    axs = np.atleast_2d(axs).ravel()
    for a, k in zip(axs, metriche):
        t = tru[k].ravel() * SCALA[k]
        e = est[k].ravel() * SCALA[k]
        ok = np.isfinite(t) & np.isfinite(e)
        lo, hi = INTERVALLO[k]
        a.hexbin(t[ok], e[ok], gridsize=40, extent=(lo, hi, lo, hi), cmap=SEQ, mincnt=1, bins='log',
                 linewidths=0)
        a.plot([lo, hi], [lo, hi], color=INK, lw=1, ls='--')
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
        d = e[ok] - t[ok]
        r = np.corrcoef(t[ok], e[ok])[0, 1] if ok.sum() > 2 else np.nan
        mancanti = np.mean(~np.isfinite(e[np.isfinite(t)])) if k in TENSORE else 0
        txt = f'r {r:.2f}\nbias {d.mean():+.3f}\nMAE {np.abs(d).mean():.3f}'
        if k in TENSORE:
            txt += f'\nfibra mancata {100 * mancanti:.0f}%'
        a.text(0.04, 0.96, txt, transform=a.transAxes, va='top', fontsize=8, color=INK)
        a.set_title(_lab(k), fontsize=9)
        a.set_xlabel('vero'); a.set_ylabel('stimato')
    for a in axs[n:]:
        a.axis('off')
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# incertezza: z = (stima - vero) / SE
# ─────────────────────────────────────────────────────────────────────────────
def istogrammi_z(zdf, metriche, gruppo_col='braccio', titolo='', file=None):
    """zdf: colonne metrica, z, gruppo_col. Curva grigia = N(0,1): se la SE descrive l'errore, gli
    istogrammi la seguono; piu' stretti = SE conservativa, spostati = bias."""
    gruppi = list(dict.fromkeys(zdf[gruppo_col]))
    nc = 4
    nr = int(np.ceil(len(metriche) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc + 1.2, 2.8 * nr), sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for a in axs[len(metriche):]:
        a.axis('off')
    xs = np.linspace(-5, 5, 200)
    for a, k in zip(axs, metriche):
        a.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color='#9a988f', lw=2)
        for gi, g in enumerate(gruppi):
            z = zdf[(zdf.metrica == k) & (zdf[gruppo_col] == g)].z.to_numpy()
            z = z[np.isfinite(z)]
            if z.size < 10:
                continue
            a.hist(np.clip(z, -5, 5), bins=np.linspace(-5, 5, 41), density=True, histtype='step', lw=2,
                   color=CAT[gi])
            cop = np.mean(np.abs(z) <= 1.96)
            a.text(0.03, 0.95 - 0.1 * gi, f'cop. 95%: {100 * cop:.0f}%', transform=a.transAxes,
                   fontsize=7.5, color=INK, va='top')
            a.plot([], [], color=CAT[gi], lw=2, label=str(g))
        a.set_title(_lab(k), fontsize=9)
        a.set_xlabel('z = (stima − vero) / SE')
    axs[len(metriche) - 1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# riepilogo per protocollo / per SNR
# ─────────────────────────────────────────────────────────────────────────────
def mae_per_gruppo(tab, metriche, gruppo_col, titolo='', file=None):
    """tab: colonne metrica, mae, gruppo_col (una riga per scenario). Barre = MAE medio sugli scenari."""
    gruppi = list(dict.fromkeys(tab[gruppo_col]))
    fig, axs = plt.subplots(1, len(metriche), figsize=(2.2 * len(metriche) + 1.0, 2.6), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for a, k in zip(axs, metriche):
        v = [tab[(tab.metrica == k) & (tab[gruppo_col] == g)].mae.mean() for g in gruppi]
        a.bar(range(len(gruppi)), v, color=[CAT[i] for i in range(len(gruppi))], width=0.7)
        for i, x in enumerate(v):
            a.text(i, x, f'{x:.3f}', ha='center', va='bottom', fontsize=7.5, color=INK)
        a.set_xticks(range(len(gruppi))); a.set_xticklabels([str(g) for g in gruppi], fontsize=8)
        a.set_title(_lab(k), fontsize=9)
        a.grid(axis='y')
    axs[0].set_ylabel('MAE medio sugli scenari')
    if titolo:
        fig.suptitle(titolo, fontsize=11, fontweight='bold')
    if file:
        fig.savefig(file)
    return fig
