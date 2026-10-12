"""
Simulazioni di accuratezza per pyDBSI (>= 1.8.1): quanto le mappe del toolbox si avvicinano alla verita'.

Tutto passa per la CATENA DI PRODUZIONE, non per una scorciatoia:

    segnale vero (S0 = 1000)  ->  rumore Rician (sigma = S0 / SNR)  ->  dwidenoise (MRtrix, MP-PCA)
        ->  mappa -noise di dwidenoise come noise_map  ->  DBSI_Adaptive().fit(run_calibration=True)

cioe' cio' che succede ai 5P (calibrazione libera, mappa di rumore di dwidenoise). Il braccio "grezzo"
salta il denoising e passa la sigma vera: misura lo stimatore da solo, e la differenza fra i due bracci
e' l'effetto del denoising. Non c'e' N4: il fantoccio non ha campo di bias.

Tre fantocci, tre domande:
  - BATTERIA     13 scenari omogenei (quelli della validazione di luglio), blocchi 3D con eterogeneita'
                 lieve: per ogni scenario, bias e dispersione di ogni mappa.
  - ANATOMICO    una fetta (48x48x6) "a forma di cervello" (GM, due fasci che si incrociano a 90 gradi, un terzo a
                 60, ventricolo, lesioni demielinizzante / edematosa / infiammatoria, tumore) con volume
                 parziale ai bordi: le immagini vero / stimato / errore.
  - CAMPI        parametri che variano con continuita' nello spazio (campi gaussiani lisci): stima contro
                 verita' voxel per voxel, su tutto l'intervallo dei valori.

La verita' delle diffusivita' isotrope NON coincide coi centroidi fissi di Stage D (0.15 / 1.0 / 3.0e-3)
se D_VARIABILE: R in [0.10, 0.25], H in [0.7, 1.3], W in [2.8, 3.2] e-3 mm^2/s, estratte per voxel. E' il
caso reale (il tessuto non ha diffusivita' esattamente uguali ai centroidi); con D_VARIABILE=False si
misura lo stimatore nel caso in cui il modello e' esatto.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive

S0 = 1000.0
D_NOM = dict(R=0.15e-3, H=1.0e-3, W=3.0e-3)
D_RANGE = dict(R=(0.10e-3, 0.25e-3), H=(0.7e-3, 1.3e-3), W=(2.8e-3, 3.2e-3))

# metriche riportate: nome breve -> canale del toolbox
METRICHE = dict(FF='fiber_fraction', RF='restricted_fraction', HF='hindered_fraction',
                WF='water_fraction', ADw='axial_diffusivity_weighted',
                RDw='radial_diffusivity_weighted', FAw='fiber_fa_weighted')
FRAZIONI = ('FF', 'RF', 'HF', 'WF')
TENSORE = ('ADw', 'RDw', 'FAw')
UNITA = dict(FF='', RF='', HF='', WF='', ADw='e-3 mm²/s', RDw='e-3 mm²/s', FAw='')
SCALA = dict(FF=1, RF=1, HF=1, WF=1, ADw=1e3, RDw=1e3, FAw=1)    # diffusivita' mostrate in e-3

NOMI = DBSI_Adaptive.output_map_names(3)
IX = {n: i for i, n in enumerate(NOMI)}


# ─────────────────────────────────────────────────────────────────────────────
# protocollo
# ─────────────────────────────────────────────────────────────────────────────
def carica_protocollo(bval, bvec):
    """bval/bvec in formato FSL; bvec normalizzati, righe (N, 3)."""
    bv = np.loadtxt(bval).ravel().astype(float)
    bc = np.loadtxt(bvec)
    bc = bc.T if bc.shape[0] == 3 and bc.shape[1] != 3 else bc
    n = np.linalg.norm(bc, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return bv, bc / n


def descrivi_protocollo(prot):
    bv = prot[0]
    gusci = pd.Series(np.round(bv, -2).astype(int)).value_counts().sort_index()
    return ', '.join(f"b={b}: {n}" for b, n in gusci.items())


# ─────────────────────────────────────────────────────────────────────────────
# tessuto -> segnale e verita'
# ─────────────────────────────────────────────────────────────────────────────
def direzione(theta_deg, phi_deg):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    v = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    return v / np.linalg.norm(v)


def ruota_verso(u, gradi, asse_aux=(0.0, 0.0, 1.0)):
    """Direzione a `gradi` gradi da u, nel piano di u e di asse_aux."""
    a = np.asarray(asse_aux, float)
    t = a - (a @ u) * u
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(u, [1.0, 0, 0])
    t /= np.linalg.norm(t)
    g = np.radians(gradi)
    return np.cos(g) * u + np.sin(g) * t


def segnale(prot, fibre, iso, D):
    """fibre: [(f, dir, AD, RD)], iso: {'R': f, 'H': f, 'W': f}, D: {'R': D, ...}. S(b=0) = somma f."""
    bv, bc = prot
    S = np.zeros(len(bv))
    for k, f in iso.items():
        if f > 0:
            S += f * np.exp(-bv * D[k])
    for f, d, ad, rd in fibre:
        if f > 0:
            S += f * np.exp(-bv * (rd + (ad - rd) * (bc @ d) ** 2))
    return S


def fa(ad, rd):
    return np.abs(ad - rd) / np.sqrt(ad ** 2 + 2 * rd ** 2)


SOGLIA_POP = 0.05     # una popolazione esiste, per la verita', se pesa almeno il 5% del voxel


def verita(fibre, iso, soglia_pop=SOGLIA_POP):
    """Valori veri delle metriche riportate (frazioni normalizzate al totale). Il tensore vero e' definito
    solo dalle popolazioni sopra soglia: una coda di volume parziale da 1% non ha un tensore misurabile."""
    tot = sum(f for f, *_ in fibre) + sum(iso.values())
    ff = sum(f for f, *_ in fibre)
    vive = [x for x in fibre if x[0] / tot >= soglia_pop]
    t = dict(FF=ff / tot, RF=iso.get('R', 0) / tot, HF=iso.get('H', 0) / tot, WF=iso.get('W', 0) / tot,
             npop=len(vive))
    fv = sum(f for f, *_ in vive)
    if fv > 0:
        adw = sum(f * ad for f, _, ad, _ in vive) / fv
        rdw = sum(f * rd for f, _, _, rd in vive) / fv
        t.update(ADw=adw, RDw=rdw, FAw=fa(adw, rdw))
    else:
        t.update(ADw=np.nan, RDw=np.nan, FAw=np.nan)
    return t


def estrai_D(rng, variabile):
    if not variabile:
        return dict(D_NOM)
    return {k: rng.uniform(*D_RANGE[k]) for k in D_NOM}


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIA: 13 scenari (validazione di luglio), blocchi 3D omogenei
# ─────────────────────────────────────────────────────────────────────────────
_d0, _d1, _d2 = direzione(90, 0), direzione(65, 0), direzione(65, 90)


def _I(r, h, w):
    return dict(R=r, H=h, W=w)


SCENARI = {
    'WM sana':          ([(0.55, _d0, 1.7e-3, 0.30e-3)], _I(0.10, 0.25, 0.10)),
    'WM demielinizzata': ([(0.45, _d0, 1.5e-3, 0.70e-3)], _I(0.15, 0.25, 0.15)),
    'WM densa':         ([(0.70, _d0, 1.8e-3, 0.25e-3)], _I(0.10, 0.15, 0.05)),
    'WM debole':        ([(0.20, _d0, 1.7e-3, 0.30e-3)], _I(0.10, 0.50, 0.20)),
    'Crossing bil.':    ([(0.35, _d1, 1.6e-3, 0.40e-3), (0.35, _d2, 1.6e-3, 0.40e-3)], _I(0.10, 0.15, 0.05)),
    'Crossing sbil.':   ([(0.45, _d1, 1.7e-3, 0.30e-3), (0.20, _d2, 1.6e-3, 0.40e-3)], _I(0.10, 0.15, 0.10)),
    'Crossing eterog.': ([(0.30, _d1, 1.7e-3, 0.30e-3), (0.20, _d2, 1.4e-3, 0.65e-3)], _I(0.18, 0.14, 0.18)),
    'GM':               ([(0.10, _d0, 1.5e-3, 0.50e-3)], _I(0.10, 0.65, 0.15)),
    'Liquor':           ([], _I(0.0, 0.05, 0.95)),
    'Solo restricted':  ([], _I(0.30, 0.50, 0.20)),
    'Tumore':           ([], _I(0.55, 0.30, 0.15)),
    'Edema':            ([(0.30, _d0, 1.7e-3, 0.35e-3)], _I(0.0, 0.15, 0.55)),
    'Lesione infiamm.': ([(0.25, _d0, 1.6e-3, 0.45e-3)], _I(0.35, 0.30, 0.10)),
}


def fantoccio_batteria(prot, blocco=(8, 8, 6), colonne=7, eterog=0.05, d_variabile=True, seed=0, scenari=None):
    """Un blocco 3D per scenario, su una griglia di `colonne` blocchi. Eterogeneita' lieve nel blocco:
    frazioni x(1 + N(0, eterog)), direzioni che ruotano di qualche grado con la posizione, D isotrope per
    voxel. Ritorna (A pulito in unita' di segnale, verita' {metrica: (X,Y,Z)}, etichette (X,Y,Z) int con -1
    fuori dai blocchi, nomi)."""
    scenari = scenari or SCENARI
    rng = np.random.default_rng(seed)
    bx, by, bz = blocco
    nomi = list(scenari)
    righe = int(np.ceil(len(nomi) / colonne))
    X, Y, N = bx * colonne, by * righe, len(prot[0])
    A = np.zeros((X, Y, bz, N), np.float32)
    lab = np.full((X, Y, bz), -1, np.int16)
    tru = {k: np.full((X, Y, bz), np.nan) for k in (*METRICHE, 'npop')}
    tru['dirs'] = np.full((X, Y, bz, 2, 3), np.nan)
    for s, nome in enumerate(nomi):
        x0, y0 = (s % colonne) * bx, (s // colonne) * by
        fibre0, iso0 = scenari[nome]
        for i in range(bx):
            for j in range(by):
                for k in range(bz):
                    jit = 1 + rng.normal(0, eterog, 3 + len(fibre0))
                    iso = {c: iso0[c] * jit[q] for q, c in enumerate('RHW')}
                    fibre = []
                    for q, (f, d, ad, rd) in enumerate(fibre0):
                        dd = ruota_verso(d, 4.0 * (i - bx / 2) / bx + 3.0 * (j - by / 2) / by)
                        fibre.append((f * jit[3 + q], dd, ad, rd))
                    D = estrai_D(rng, d_variabile)
                    S = segnale(prot, fibre, iso, D)
                    tot = sum(f for f, *_ in fibre) + sum(iso.values())
                    x, y = x0 + i, y0 + j
                    A[x, y, k] = S / tot
                    lab[x, y, k] = s
                    for m, v in verita(fibre, iso).items():
                        tru[m][x, y, k] = v
                    for q, (_, dd, _, _) in enumerate(fibre[:2]):
                        tru['dirs'][x, y, k, q] = dd
    return A, tru, lab, nomi


# ─────────────────────────────────────────────────────────────────────────────
# ANATOMICO: una fetta con fasci, incroci, ventricolo e lesioni, volume parziale ai bordi
# ─────────────────────────────────────────────────────────────────────────────
def _morbida(dist, larghezza=0.8):
    """Appartenenza in [0, 1] da una distanza con segno (negativa dentro)."""
    return 1.0 / (1.0 + np.exp(dist / larghezza))


TESSUTI_ANAT = {
    # nome: (fibre come [(f, chiave_direzione, AD, RD)], iso)
    'GM':              ([], _I(0.10, 0.75, 0.15)),
    'fascio A':        ([(0.60, 'A', 1.7e-3, 0.30e-3)], _I(0.10, 0.22, 0.08)),
    'fascio B':        ([(0.60, 'B', 1.7e-3, 0.30e-3)], _I(0.10, 0.22, 0.08)),
    'fascio C':        ([(0.60, 'C', 1.7e-3, 0.35e-3)], _I(0.10, 0.22, 0.08)),
    'liquor':          ([], _I(0.0, 0.05, 0.95)),
    'demielinizzata':  ([(0.45, 'A', 1.5e-3, 0.75e-3)], _I(0.10, 0.35, 0.10)),
    'edema':           ([(0.30, 'B', 1.7e-3, 0.35e-3)], _I(0.05, 0.20, 0.45)),
    'infiammazione':   ([(0.40, 'A', 1.65e-3, 0.40e-3)], _I(0.30, 0.22, 0.08)),
    'tumore':          ([], _I(0.50, 0.35, 0.15)),
}


def fantoccio_anatomico(prot, nx=64, ny=64, nz=6, d_variabile=True, seed=1):
    """Ritorna (A, verita', pesi {tessuto: (X,Y,Z)}, direzioni vere per voxel {chiave: (X,Y,Z,3)})."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing='ij')
    c = np.array([nx / 2, ny / 2])
    # fascio A: banda orizzontale leggermente curva; B: banda verticale (incrocio a 90 con A);
    # C: banda obliqua a 60 gradi rispetto ad A, nella meta' bassa
    yA = ny * 0.62 + 4 * np.sin(2 * np.pi * x / nx)
    m = {}
    m['fascio A'] = _morbida(np.abs(y - yA) - 6)
    m['fascio B'] = _morbida(np.abs(x - nx * 0.30) - 5)
    ang = np.radians(60)
    dC = np.abs((x - nx * 0.70) * np.sin(ang) - (y - ny * 0.25) * np.cos(ang))
    m['fascio C'] = _morbida(dC - 4.5) * _morbida(y - ny * 0.50, 1.5)
    m['liquor'] = _morbida(np.hypot((x - nx * 0.55) / 7, (y - ny * 0.40) / 4) * 4 - 4)
    def disco(cx, cy, r):
        return _morbida(np.hypot(x - cx, y - cy) - r)
    m['demielinizzata'] = disco(nx * 0.80, ny * 0.62 + 4 * np.sin(2 * np.pi * 0.80), 4.0)    # sul fascio A
    m['infiammazione'] = disco(nx * 0.55, ny * 0.62 + 4 * np.sin(2 * np.pi * 0.55), 3.5)     # sul fascio A
    m['edema'] = disco(nx * 0.30, ny * 0.15, 4.0)                                          # sul fascio B
    m['tumore'] = disco(nx * 0.15, ny * 0.40, 4.5)
    # composizione dall'alto (alpha compositing): liquor > lesioni > fasci > GM. I fasci si dividono
    # la copertura dove si sovrappongono: e' li' che nasce l'incrocio.
    w = {k: np.zeros((nx, ny)) for k in TESSUTI_ANAT}
    resto = np.ones((nx, ny))
    for k in ('liquor', 'tumore', 'demielinizzata', 'infiammazione', 'edema'):
        w[k] = resto * m[k]
        resto = resto - w[k]
    fasci = ('fascio A', 'fascio B', 'fascio C')
    somma = sum(m[k] for k in fasci)
    cop = np.minimum(somma, 1.0)
    for k in fasci:
        w[k] = resto * cop * np.where(somma > 0, m[k] / np.maximum(somma, 1e-12), 0)
    w['GM'] = resto * (1 - cop)
    # direzioni: A segue la curva, B verticale, C a 60 gradi da x; inclinazione fuori piano 15 gradi
    dirs = {}
    slopeA = 4 * 2 * np.pi / nx * np.cos(2 * np.pi * x / nx)
    a = np.arctan(slopeA)
    el = np.radians(15)
    dirs['A'] = np.stack([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.full_like(a, np.sin(el))], -1)
    dirs['B'] = np.broadcast_to(direzione(90 - 15, 90), (nx, ny, 3)).copy()
    dirs['C'] = np.broadcast_to(direzione(90 - 15, 60), (nx, ny, 3)).copy()
    N = len(prot[0])
    A = np.zeros((nx, ny, nz, N), np.float32)
    tru = {k: np.full((nx, ny, nz), np.nan) for k in (*METRICHE, 'npop')}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                D = estrai_D(rng, d_variabile)
                fib_tot, iso_tot = {}, dict(R=0.0, H=0.0, W=0.0)
                for t, wt in w.items():
                    ww = wt[i, j]
                    if ww < 1e-4:
                        continue
                    fl, iso = TESSUTI_ANAT[t]
                    jit = 1 + rng.normal(0, 0.04, 3)
                    for q, cc in enumerate('RHW'):
                        iso_tot[cc] += ww * iso[cc] * jit[q]
                    for f, key, ad, rd in fl:
                        a0 = fib_tot.get(key, [0.0, 0.0, 0.0])
                        fib_tot[key] = [a0[0] + ww * f, a0[1] + ww * f * ad, a0[2] + ww * f * rd]
                fibre = [(v[0], dirs[key][i, j], v[1] / v[0], v[2] / v[0]) for key, v in fib_tot.items()
                         if v[0] > 1e-4]
                S = segnale(prot, fibre, iso_tot, D)
                tot = sum(f for f, *_ in fibre) + sum(iso_tot.values())
                A[i, j, k] = S / tot
                for mm, v in verita(fibre, iso_tot).items():
                    tru[mm][i, j, k] = v
    pesi = {t: np.repeat(wt[:, :, None], nz, 2) for t, wt in w.items()}
    return A, tru, pesi


# ─────────────────────────────────────────────────────────────────────────────
# CAMPI: parametri lisci e continui
# ─────────────────────────────────────────────────────────────────────────────
def _campo(rng, shape, scala):
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(rng.normal(size=shape), scala, mode='wrap')
    g = (g - g.mean()) / (g.std() + 1e-12)
    from scipy.stats import norm
    return norm.cdf(g)              # uniforme in (0, 1), liscio


def fantoccio_campi(prot, shape=(48, 48, 8), scala=3.0, d_variabile=True, seed=2):
    """FF, frazioni iso, AD, RD, angolo di incrocio e quota della seconda fibra variano con continuita'.
    Meta' del volume ha due fibre (incrocio con angolo fra 45 e 90 gradi), meta' una."""
    rng = np.random.default_rng(seed)
    u = {k: _campo(rng, shape, scala) for k in ('ff', 'r', 'h', 'w', 'ad', 'rd', 'th', 'ph', 'ang', 'q2', 'cross')}
    N = len(prot[0])
    A = np.zeros(shape + (N,), np.float32)
    tru = {k: np.full(shape, np.nan) for k in (*METRICHE, 'npop')}
    for idx in np.ndindex(shape):
        g = {k: v[idx] for k, v in u.items()}
        ff = 0.75 * g['ff']                                   # 0 - 0.75
        wiso = np.array([0.05 + g['r'], 0.05 + 1.5 * g['h'], 0.02 + 0.8 * g['w']])
        wiso = wiso / wiso.sum() * (1 - ff)
        ad = 1.3e-3 + 0.7e-3 * g['ad']
        rd = min(0.15e-3 + 0.65e-3 * g['rd'], ad / 1.6)
        d1 = direzione(30 + 120 * g['th'], 360 * g['ph'])
        fibre = []
        if ff > 0.03:
            if g['cross'] > 0.5:
                d2 = ruota_verso(d1, 45 + 45 * g['ang'], asse_aux=direzione(90, 360 * g['ph'] + 90))
                q2 = 0.3 + 0.4 * g['q2']
                fibre = [(ff * (1 - q2), d1, ad, rd), (ff * q2, d2, ad, rd)]
            else:
                fibre = [(ff, d1, ad, rd)]
        iso = dict(R=wiso[0], H=wiso[1], W=wiso[2])
        D = estrai_D(rng, d_variabile)
        S = segnale(prot, fibre, iso, D)
        A[idx] = S / (sum(f for f, *_ in fibre) + wiso.sum())
        for m, v in verita(fibre, iso).items():
            tru[m][idx] = v
    return A, tru


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME UNICO: i tre fantocci affiancati, una calibrazione e un dwidenoise per condizione
# ─────────────────────────────────────────────────────────────────────────────
def volume_composito(prot, d_variabile=True, seed=0, piccolo=False, spazio=4):
    """Anatomico | campi | batteria affiancati lungo x, separati da `spazio` voxel di fondo (segnale
    nullo, fuori maschera). La calibrazione campiona i voxel di tutti e tre, come in un cervello con molti
    tessuti; dwidenoise vede un volume unico. piccolo=True dimezza i fantocci (prova veloce).
    Ritorna (A, maschera, parti) con parti[nome] = dict(sl=slice lungo x, verita'=..., ...)."""
    if piccolo:
        an = fantoccio_anatomico(prot, nx=40, ny=40, nz=6, d_variabile=d_variabile, seed=seed + 1)
        ca = fantoccio_campi(prot, shape=(28, 28, 6), d_variabile=d_variabile, seed=seed + 2)
        ba = fantoccio_batteria(prot, blocco=(6, 6, 6), colonne=7, d_variabile=d_variabile, seed=seed)
    else:
        an = fantoccio_anatomico(prot, nx=48, ny=48, nz=6, d_variabile=d_variabile, seed=seed + 1)
        ca = fantoccio_campi(prot, shape=(32, 32, 6), d_variabile=d_variabile, seed=seed + 2)
        ba = fantoccio_batteria(prot, blocco=(8, 8, 6), colonne=7, d_variabile=d_variabile, seed=seed)
    pezzi = [('anatomico', an[0]), ('campi', ca[0]), ('batteria', ba[0])]
    Y = max(p.shape[1] for _, p in pezzi)
    Z = pezzi[0][1].shape[2]
    X = sum(p.shape[0] for _, p in pezzi) + spazio * (len(pezzi) + 1)
    N = len(prot[0])
    A = np.zeros((X, Y, Z, N), np.float32)
    mask = np.zeros((X, Y, Z), bool)
    parti, x0 = {}, spazio
    for nome, P in pezzi:
        nx, ny = P.shape[:2]
        sl = (slice(x0, x0 + nx), slice(0, ny))
        A[sl] = P
        mask[sl] = P.max(-1) > 0          # il posto vuoto della griglia della batteria resta fuori
        parti[nome] = dict(sl=sl)
        x0 += nx + spazio
    parti['anatomico'].update(verita=an[1], pesi=an[2])
    parti['campi'].update(verita=ca[1])
    parti['batteria'].update(verita=ba[1], etichette=ba[2], nomi=ba[3])
    return A, mask, parti


def ritaglia(mappe, sl):
    """Le mappe stimate ({nome: (X,Y,Z)}) ritagliate su una parte del volume."""
    return {k: v[sl] for k, v in mappe.items()}


# ─────────────────────────────────────────────────────────────────────────────
# catena: rumore -> dwidenoise -> fit
# ─────────────────────────────────────────────────────────────────────────────
def rumore_rician(A, snr, seed=0):
    rng = np.random.default_rng(seed)
    sig = S0 / snr
    X = S0 * A.astype(np.float64)
    M = np.sqrt((X + rng.normal(0, sig, X.shape)) ** 2 + rng.normal(0, sig, X.shape) ** 2)
    return M.astype(np.float32), sig


def trova_dwidenoise(percorso=None):
    for p in (percorso, shutil.which('dwidenoise'), '/usr/local/mrtrix3/dwidenoise',
              '/usr/local/mrtrix3/bin/dwidenoise'):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def denoise(M, eseguibile, voxel_mm=2.0):
    """dwidenoise di MRtrix sui dati (X,Y,Z,N): ritorna (denoisati, mappa -noise)."""
    import nibabel as nib
    aff = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    with tempfile.TemporaryDirectory() as td:
        fi, fo, fn = (os.path.join(td, n) for n in ('in.nii', 'den.nii', 'noise.nii'))
        nib.save(nib.Nifti1Image(M.astype(np.float32), aff), fi)
        subprocess.run([eseguibile, fi, fo, '-noise', fn, '-force', '-quiet'], check=True)
        D = np.asarray(nib.load(fo).dataobj, dtype=np.float32)
        nm = np.asarray(nib.load(fn).dataobj, dtype=np.float64)
    return D, nm


def fit(data, prot, noise_map, mask=None, **kw_fit):
    """Fit di produzione (calibrazione libera salvo kw_fit): ritorna (risultati, modello)."""
    mask = np.ones(data.shape[:3], bool) if mask is None else mask
    m = DBSI_Adaptive()
    kw = dict(run_calibration=True)
    kw.update(kw_fit)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(data, prot[0], prot[1], mask, noise_map=noise_map, **kw)
    assert mode == 3, f'modo {mode}-ISO: il protocollo non regge il 3-ISO'
    return res, m


def esegui(A, prot, snr, braccio, dwidenoise=None, seed=0, mask=None, **kw_fit):
    """Un braccio della catena. braccio = 'produzione' (dwidenoise + sua mappa) o 'grezzo' (sigma vera).
    Il rumore e' aggiunto a TUTTO il volume, fondo compreso (come in uno scanner); il fit solo nella maschera."""
    M, sig = rumore_rician(A, snr, seed)
    if braccio == 'produzione':
        if dwidenoise is None:
            raise RuntimeError('dwidenoise non trovato: braccio di produzione impossibile')
        D, nm = denoise(M, dwidenoise)
        data, noise = D, nm
    elif braccio == 'grezzo':
        data, noise = M, np.full(A.shape[:3], sig)
    else:
        raise ValueError(braccio)
    res, m = fit(data, prot, noise, mask=mask, **kw_fit)
    dentro = np.ones(A.shape[:3], bool) if mask is None else mask
    info = dict(snr=snr, braccio=braccio, sigma_vera=sig, sigma_mappa_mediana=float(np.median(noise[dentro])),
                lambda_aniso=float(m.lambda_aniso), lambda_iso=float(m.lambda_iso), n_iso=int(m.n_iso))
    return res, m, info


def stima(res, m=None):
    """Mappe stimate col nome breve. Le frazioni sono definite in ogni voxel fittato; il tensore e' NaN
    dove il toolbox non ha trovato fibre (lasciato NaN: e' "fibra mancata", contato a parte)."""
    out = {k: res[..., IX[c]].astype(float) for k, c in METRICHE.items()}
    out['npop'] = np.nan_to_num(res[..., IX['n_fiber_populations']].astype(float))
    if m is not None and getattr(m, 'uncertainty_', None) is not None:
        se = m.uncertainty_['se']
        for k, c in METRICHE.items():
            out['SE_' + k] = se[..., IX[c]].astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# misure
# ─────────────────────────────────────────────────────────────────────────────
def interno(lab, margine=1):
    """Maschera dei voxel ad almeno `margine` voxel dal bordo del proprio blocco (lungo x e y): ai
    bordi dwidenoise mescola blocchi diversi, come al confine fra tessuti."""
    ok = lab >= 0
    for ax in (0, 1):
        for s in range(1, margine + 1):
            ok &= np.roll(lab, s, ax) == lab
            ok &= np.roll(lab, -s, ax) == lab
    return ok


def tabella_errori(est, tru, gruppi, nomi_gruppi, sel=None):
    """Per gruppo e metrica: vero (mediana), stima (mediana), bias (media), MAE, sd, n validi, e per le
    metriche del tensore la quota di voxel con fibra vera in cui la stima manca (NaN)."""
    righe = []
    sel = np.ones(gruppi.shape, bool) if sel is None else sel
    for g, nome in enumerate(nomi_gruppi):
        s = (gruppi == g) & sel
        for k in METRICHE:
            t, e = tru[k][s], est[k][s]
            ok = np.isfinite(t) & np.isfinite(e)
            r = dict(gruppo=nome, metrica=k, n=int(s.sum()), n_validi=int(ok.sum()))
            if k in TENSORE:
                vf = np.isfinite(t)
                r['mancanti_pct'] = float(100 * np.mean(~np.isfinite(e[vf]))) if vf.any() else np.nan
            if ok.sum():
                d = (e[ok] - t[ok]) * SCALA[k]
                r.update(vero=float(np.median(t[ok]) * SCALA[k]), stima=float(np.median(e[ok]) * SCALA[k]),
                         bias=float(d.mean()), mae=float(np.abs(d).mean()), sd=float(d.std()))
            righe.append(r)
        tn, en = tru['npop'][s], est['npop'][s]
        righe.append(dict(gruppo=nome, metrica='npop', n=int(s.sum()), n_validi=int(s.sum()),
                          vero=float(np.median(tn)), stima=float(np.mean(en)),
                          giusto_pct=float(100 * np.mean(np.round(en) == np.round(tn)))))
    return pd.DataFrame(righe)


def errore_angolare(res, dirs_vere):
    """Errore angolare (gradi) di ogni popolazione stimata rispetto alla fibra vera piu' vicina.
    dirs_vere: (..., 2, 3) con NaN dove la fibra non c'e'. Ritorna (..., 2), NaN se manca una delle due."""
    out = np.full(res.shape[:-1] + (2,), np.nan)
    V = np.nan_to_num(dirs_vere)
    for p in (0, 1):
        e = np.stack([res[..., IX[f'dir{p + 1}_{c}']] for c in 'xyz'], -1).astype(float)
        cos = np.abs(np.einsum('...c,...kc->...k', np.nan_to_num(e), V))        # (..., 2)
        cos[~np.isfinite(dirs_vere[..., 0])] = -1
        best = np.clip(cos.max(-1), 0, 1)
        ok = np.isfinite(e).all(-1) & np.isfinite(dirs_vere[..., 0, 0])
        out[..., p] = np.where(ok, np.degrees(np.arccos(best)), np.nan)
    return out


def versione():
    return dbsi_toolbox.__version__
