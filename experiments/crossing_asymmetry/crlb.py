"""Limite di Cramer-Rao per i parametri della fibra, protocollo P3: cosa e' identificabile?
Modello (rumore gaussiano, sigma = 1/SNR in S/S0): S = sum_k f_k exp(-b(RD_k + (AD_k-RD_k)cos^2)) + fr e^{-b Dr} + fh e^{-b Dh} + fw e^{-b Dw}.
Direzioni NOTE (limite ottimistico). Si riporta la deviazione standard minima di ogni parametro."""
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent
bv = np.loadtxt(HERE / 'p3_like.bval').ravel(); bc = np.loadtxt(HERE / 'p3_like.bvec'); bc = bc.T if bc.shape[0] == 3 else bc
nb = np.linalg.norm(bc, axis=1, keepdims=True); nb[nb == 0] = 1; bc = bc / nb
Dr, Dh, Dw = 0.15e-3, 1.0e-3, 3.0e-3
u = np.array([1.0, 0, 0]); v = np.array([0, 1.0, 0])

def modello(p, spec):
    """spec: lista dei parametri liberi; p: dizionario completo."""
    S = p['fr'] * np.exp(-bv * Dr) + p['fh'] * np.exp(-bv * Dh) + p['fw'] * np.exp(-bv * Dw)
    for k, d in enumerate(p['dirs']):
        ad = p['AD'] if 'AD' in p else p[f'AD{k}']
        rd = p['RD'] if 'RD' in p else p[f'RD{k}']
        S = S + p[f'f{k}'] * np.exp(-bv * (rd + (ad - rd) * (bc @ d) ** 2))
    return S

def crlb(p, liberi, snr):
    eps = {k: (1e-6 if k.startswith(('AD', 'RD')) else 1e-4) for k in liberi}
    J = []
    for k in liberi:
        q1 = dict(p); q2 = dict(p); q1[k] += eps[k]; q2[k] -= eps[k]
        J.append((modello(q1, liberi) - modello(q2, liberi)) / (2 * eps[k]))
    J = np.array(J).T
    Fi = J.T @ J * snr ** 2
    C = np.linalg.pinv(Fi)
    return {k: float(np.sqrt(C[i, i])) for i, k in enumerate(liberi)}, float(np.linalg.cond(Fi))

CASI = {
 'mono sano':          dict(dirs=[u], f0=0.6, AD0=1.7e-3, RD0=0.4e-3, fr=0.10, fh=0.20, fw=0.10),
 'mono demiel':        dict(dirs=[u], f0=0.6, AD0=1.5e-3, RD0=0.8e-3, fr=0.10, fh=0.20, fw=0.10),
 'crossing 90 sano':   dict(dirs=[u, v], f0=0.3, f1=0.3, AD0=1.7e-3, RD0=0.4e-3, AD1=1.7e-3, RD1=0.4e-3, fr=0.10, fh=0.20, fw=0.10),
 'crossing 90 demiel A': dict(dirs=[u, v], f0=0.3, f1=0.3, AD0=1.5e-3, RD0=0.8e-3, AD1=1.7e-3, RD1=0.4e-3, fr=0.10, fh=0.20, fw=0.10),
}
fraz = ['fr', 'fh', 'fw']
for snr in (26, 40):
    print(f'\n===== SNR {snr}: deviazione standard minima (Cramer-Rao), direzioni note =====')
    for nome, p in CASI.items():
        n = len(p['dirs'])
        modelli = {'libero (AD,RD per pop + frazioni)': [f'f{k}' for k in range(n)] + fraz + sum([[f'AD{k}', f'RD{k}'] for k in range(n)], [])}
        if n == 2:
            modelli['FF fisse (produzione)'] = sum([[f'AD{k}', f'RD{k}'] for k in range(n)], [])
            pp = dict(p); pp['AD'] = p['AD1']
            modelli['AD imposta, RD per pop (3b/3c)'] = ('imposta', [f'f{k}' for k in range(n)] + fraz + [f'RD{k}' for k in range(n)])
        for mn, lib in modelli.items():
            q = dict(p)
            if isinstance(lib, tuple):   # AD imposta = nota: non e' fra i liberi
                lib = lib[1]
            sd, cond = crlb(q, lib, snr)
            txt = '  '.join(f'{k}={sd[k]*1e3:.3f}e-3' if k.startswith(('AD', 'RD')) else f'{k}={sd[k]:.3f}' for k in lib)
            print(f'  {nome:22s} | {mn:34s} | {txt}')


# ── Quantita' derivate: RD pesata e contrasto fra fasci (metodo delta sulla covarianza) ──
def cov(p, liberi, snr):
    eps = {k: (1e-6 if k.startswith(('AD', 'RD')) else 1e-4) for k in liberi}
    J = []
    for k in liberi:
        q1 = dict(p); q2 = dict(p); q1[k] += eps[k]; q2[k] -= eps[k]
        J.append((modello(q1, liberi) - modello(q2, liberi)) / (2 * eps[k]))
    J = np.array(J).T
    return np.linalg.pinv(J.T @ J * snr ** 2)

print('\n===== Crossing 90: RD pesata e contrasto RD_A - RD_B (sd minima, e-3) =====')
for snr in (26, 40):
    for nome in ('crossing 90 sano', 'crossing 90 demiel A'):
        p = CASI[nome]
        # libero per popolazione
        lib = ['f0', 'f1'] + fraz + ['AD0', 'RD0', 'AD1', 'RD1']
        C = cov(p, lib, snr); i = {k: j for j, k in enumerate(lib)}
        g = np.zeros(len(lib)); g[i['RD0']] = g[i['RD1']] = 0.5
        d = np.zeros(len(lib)); d[i['RD0']] = 1; d[i['RD1']] = -1
        sd_w_lib, sd_c_lib = np.sqrt(g @ C @ g), np.sqrt(d @ C @ d)
        # AD imposta, RD per popolazione
        lib2 = ['f0', 'f1'] + fraz + ['RD0', 'RD1']
        C2 = cov(p, lib2, snr); i2 = {k: j for j, k in enumerate(lib2)}
        g2 = np.zeros(len(lib2)); g2[i2['RD0']] = g2[i2['RD1']] = 0.5
        d2 = np.zeros(len(lib2)); d2[i2['RD0']] = 1; d2[i2['RD1']] = -1
        # tensore condiviso (AD, RD comuni) e AD imposta + RD comune
        q = dict(p); q['AD'] = (p['AD0'] + p['AD1']) / 2; q['RD'] = (p['RD0'] + p['RD1']) / 2
        sd_sh, _ = crlb(q, ['f0', 'f1'] + fraz + ['AD', 'RD'], snr)
        sd_sh2, _ = crlb(q, ['f0', 'f1'] + fraz + ['RD'], snr)
        print(f'  SNR {snr} {nome:22s}  RD pesata: libero {sd_w_lib*1e3:.3f} | AD imposta {np.sqrt(g2@C2@g2)*1e3:.3f} | '
              f'tensore condiviso {sd_sh["RD"]*1e3:.3f} | AD imposta + RD comune {sd_sh2["RD"]*1e3:.3f}    '
              f'contrasto RD_A-RD_B: libero {sd_c_lib*1e3:.3f} | AD imposta {np.sqrt(d2@C2@d2)*1e3:.3f}  '
              f'(contrasto vero {(p["RD0"]-p["RD1"])*1e3:.1f})')
