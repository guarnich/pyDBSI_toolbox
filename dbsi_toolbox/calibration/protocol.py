"""
Impronta di un protocollo di acquisizione, per la calibrazione di protocollo.

Gli iperparametri (lambda_aniso, lambda_iso, n_iso) si tarano UNA volta per
protocollo e si applicano a ogni acquisizione con quel protocollo. Serve quindi
un modo di dire "questa acquisizione ha lo stesso protocollo": l'impronta.

COSA C'E' DENTRO: solo conteggi e gusci -- numero di volumi, di b=0, gusci
arrotondati con il conteggio per guscio, b massimo.
COSA NON C'E': i vettori di gradiente esatti. Dopo la correzione di eddy/moto i
bvec vengono ruotati per soggetto e non coincidono mai fra due acquisizioni
dello stesso protocollo; un'impronta sui vettori rifiuterebbe tutto.
"""
import numpy as np

# Soglia sotto la quale un volume e' b=0 (stessa convenzione di tools.py).
B0_THRESHOLD = 50.0
# Arrotondamento dei gusci: b=995 e b=1005 sono lo stesso guscio.
SHELL_ROUNDING = 100.0


def protocol_fingerprint(bvals, b0_threshold=B0_THRESHOLD, shell_rounding=SHELL_ROUNDING):
    """Impronta del protocollo: conteggi e gusci, niente vettori.

    Returns
    -------
    dict con n_volumes, n_b0, n_dw, n_shells, b_max, shells (lista di
    [b arrotondato, conteggio] in ordine crescente), e le due convenzioni usate
    (b0_threshold, shell_rounding) perche' l'impronta si confronta solo a parita'
    di convenzioni.
    """
    b = np.asarray(bvals, dtype=np.float64).ravel()
    dw = b[b >= b0_threshold]
    shells_r = np.round(dw / shell_rounding) * shell_rounding
    vals, counts = np.unique(shells_r, return_counts=True)
    return dict(
        n_volumes=int(b.size),
        n_b0=int(np.sum(b < b0_threshold)),
        n_dw=int(dw.size),
        n_shells=int(vals.size),
        b_max=float(vals.max()) if vals.size else 0.0,
        shells=[[float(v), int(c)] for v, c in zip(vals, counts)],
        b0_threshold=float(b0_threshold),
        shell_rounding=float(shell_rounding),
    )


def fingerprint_mismatches(fp_a, fp_b):
    """Campi in cui due impronte differiscono (lista vuota = stesso protocollo)."""
    keys = ('b0_threshold', 'shell_rounding', 'n_volumes', 'n_b0', 'n_dw',
            'n_shells', 'b_max', 'shells')
    diff = []
    for k in keys:
        a, b = fp_a.get(k), fp_b.get(k)
        if k == 'shells':
            a = [list(map(float, x)) for x in (a or [])]
            b = [list(map(float, x)) for x in (b or [])]
        if a != b:
            diff.append((k, a, b))
    return diff
