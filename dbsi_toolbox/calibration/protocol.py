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


# ═════════════════════════════════════════════════════════════════════════════
# Calibrazione di PROTOCOLLO: aggregare i record di `DBSI_Adaptive.calibrate()`
# ═════════════════════════════════════════════════════════════════════════════
#
# Ogni acquisizione calibrata da sola sceglie i lambda del PROPRIO rumore
# (Spearman con l'SNR -0.56 per lambda_aniso e -0.74 per lambda_iso su 40
# soggetti dello stesso protocollo): in un confronto di gruppo la pipeline
# fabbricherebbe differenze. Qui si sceglie UN valore per protocollo, da un
# campione di acquisizioni (bilanciato fra i gruppi, se ci sono gruppi).
#
# DUE REGOLE, selezionabili:
#
#   'geomean'       il lambda che minimizza la media dei log delle curve GCV,
#                   cioe' la media geometrica delle curve di ciascuna
#                   acquisizione. Ogni curva e' invariante per un fattore di
#                   scala (la GCV di un soggetto piu' rumoroso e' piu' alta di un
#                   fattore ~sigma^2, e il log lo toglie), quindi pesa la FORMA
#                   della curva e non il livello di rumore. E' l'ottimo del
#                   protocollo, non un voto fra ottimi individuali.
#   'median_index'  la regola usata per congelare WaterMobility il 2026-09-23:
#                   l'indice di griglia che minimizza lo scarto assoluto medio
#                   dai minimi individuali, pareggi risolti sul caso peggiore
#                   (L-infinito). Lavora sui soli minimi, quindi con una
#                   griglia grossa produce pareggi (20-19 su 40).
#
# Quale sia il default lo decide la misura (Codes_fixed_20260923/08): finche'
# non c'e', il default e' 'geomean' ma il record riporta SEMPRE anche la
# risposta dell'altra regola.
#
# LIMITE DICHIARATO: la curva di lambda_aniso di ogni acquisizione e' calcolata
# al lambda_iso di QUELLA acquisizione. Aggregarle e' quindi un'approssimazione
# di primo ordine; la misura dice quanto conta.

PROTOCOL_CALIBRATION_KIND = 'pydbsi_protocol_calibration'
PROTOCOL_CALIBRATION_FORMAT = 1
AGGREGATION_RULES = ('geomean', 'median_index')
LAMBDA_ISO_CAP_RULES = ('min', 'median', 'none')

# Opzioni del modello da cui dipende il dizionario, e quindi il significato dei
# lambda: devono coincidere fra le acquisizioni e vengono imposte da
# `DBSI_Adaptive.from_calibration`.
_MODEL_KEYS = ('n_dirs', 'n_ad', 'n_rd', 'anisotropy_ratio', 'fiber_threshold',
               'iso_range')


def _idx_on_grid(value, grid):
    grid = np.asarray(grid, float)
    return int(np.argmin(np.abs(np.log(grid) - np.log(value))))


def _median_index(idxs, n_grid):
    """Indice che minimizza lo scarto medio in passi; pareggi sul caso peggiore."""
    idxs = np.asarray(idxs, int)
    cand = np.arange(n_grid)
    l1 = np.array([np.mean(np.abs(idxs - k)) for k in cand])
    best = cand[np.isclose(l1, l1.min())]
    linf = np.array([np.max(np.abs(idxs - k)) for k in best])
    best = best[linf == linf.min()]
    return int(best[0])          # ancora pari: il piu' piccolo, dichiarato


def _geomean_index(curves):
    """argmin della media dei log delle curve (colonne non finite escluse)."""
    C = np.asarray(curves, float)
    ok = np.all(np.isfinite(C) & (C > 0), axis=0)
    if not ok.any():
        raise ValueError('nessun punto di griglia con curve finite per tutte le acquisizioni')
    L = np.full(C.shape[1], np.inf)
    L[ok] = np.mean(np.log(C[:, ok]), axis=0)
    return int(np.argmin(L)), L


def _pick(rule, curves, selected_idx, n_grid):
    if rule == 'geomean':
        return _geomean_index(curves)[0]
    return _median_index(selected_idx, n_grid)


def _same(values, what):
    first = values[0]
    for i, v in enumerate(values[1:], 1):
        if v != first:
            raise ValueError(f'{what} diverso fra le acquisizioni: #0 {first!r} contro #{i} {v!r}')
    return first


def calibrate_protocol(records, name=None, ids=None, rule='geomean',
                       lambda_iso_cap_rule='min', n_boot=1000, seed=0):
    """Un set di iperparametri per un protocollo, dai record di `calibrate()`.

    Parameters
    ----------
    records : list of dict
        Record di `DBSI_Adaptive.calibrate()`, uno per acquisizione, tutti con
        la STESSA impronta, le stesse opzioni del modello e le stesse griglie
        di lambda. Altrimenti solleva ValueError dicendo cosa differisce.
    name : str
        Nome del protocollo (finisce nel run report di ogni fit).
    ids : list of str or None
        Identificativi delle acquisizioni, per la tabella di provenienza.
    rule : 'geomean' | 'median_index'
        Regola di aggregazione di lambda_aniso e lambda_iso (vedi sopra).
    lambda_iso_cap_rule : 'min' | 'median' | 'none'
        Tetto su lambda_iso. Ogni acquisizione ha un tetto di discrepanza (il
        lambda_iso oltre il quale il suo residuo isotropo esce dal rumore);
        'min' non sovra-regolarizza nessuna acquisizione del campione, 'median'
        ne tollera meta', 'none' lo ignora. Il record riporta quante
        acquisizioni il valore scelto supera.
    n_boot : int
        Ricampionamenti delle acquisizioni per la stabilita' della scelta.

    Returns
    -------
    dict serializzabile in JSON (kind = 'pydbsi_protocol_calibration').
    """
    import datetime as _dt
    if rule not in AGGREGATION_RULES:
        raise ValueError(f'rule deve essere in {AGGREGATION_RULES}, trovato {rule!r}')
    if lambda_iso_cap_rule not in LAMBDA_ISO_CAP_RULES:
        raise ValueError(f'lambda_iso_cap_rule deve essere in {LAMBDA_ISO_CAP_RULES}, '
                         f'trovato {lambda_iso_cap_rule!r}')
    records = list(records)
    n = len(records)
    if n < 2:
        raise ValueError('servono almeno 2 acquisizioni per una calibrazione di protocollo')
    for i, r in enumerate(records):
        if r.get('kind') != 'pydbsi_acquisition_calibration':
            raise ValueError(f'record #{i} non e un record di calibrate()')
    ids = [str(x) for x in ids] if ids is not None else [f'acq{i:03d}' for i in range(n)]
    if len(ids) != n:
        raise ValueError('ids e records hanno lunghezze diverse')

    # ── stesso protocollo, stesso modello, stesse griglie ────────────────────
    fp0 = records[0]['protocol']
    for i, r in enumerate(records[1:], 1):
        mm = fingerprint_mismatches(fp0, r['protocol'])
        if mm:
            raise ValueError(f'acquisizione {ids[i]}: impronta diversa da {ids[0]} su '
                             f'{[m[0] for m in mm]}')
    model = {k: _same([r['model'][k] for r in records], f'opzione del modello {k}')
             for k in _MODEL_KEYS}
    hp = [r['hyperparameters'] for r in records]
    n_iso = _same([h['n_iso'] for h in hp], 'n_iso')
    method = _same([h['lambda_aniso_method'] for h in hp], 'lambda_aniso_method')
    if rule == 'geomean' and method != 'gcv':
        raise ValueError("rule='geomean' richiede lambda_aniso_method='gcv' (serve la "
                         f"curva GCV), trovato {method!r}")
    gate_cal = [bool(h['concentration_gate_calibrated']) for h in hp]
    if any(gate_cal) and not all(gate_cal):
        raise ValueError('gate calibrato in alcune acquisizioni e non in altre')
    gate = (float(np.median([h['concentration_gate'] for h in hp])) if all(gate_cal)
            else float(_same([h['concentration_gate'] for h in hp], 'concentration_gate')))

    ga = np.asarray(records[0]['curves']['lambda_aniso']['lambda_grid'], float)
    gi = np.asarray(records[0]['curves']['lambda_iso']['lambda_grid'], float)
    for i, r in enumerate(records[1:], 1):
        if not (np.allclose(r['curves']['lambda_aniso']['lambda_grid'], ga, rtol=1e-12)
                and np.allclose(r['curves']['lambda_iso']['lambda_grid'], gi, rtol=1e-12)):
            raise ValueError(f'acquisizione {ids[i]}: griglie di lambda diverse da {ids[0]}')

    Ca = np.array([r['curves']['lambda_aniso']['gcv'] for r in records], float) \
        if method == 'gcv' else None
    Ci = np.array([r['curves']['lambda_iso']['gcv'] for r in records], float)
    sel_a = np.array([_idx_on_grid(h['lambda_aniso'], ga) for h in hp])
    sel_i = np.array([_idx_on_grid(h['lambda_iso'], gi) for h in hp])
    caps = np.array([(r['selection'].get('lambda_iso_cap') or np.inf) for r in records], float)

    def _cap(c):
        if lambda_iso_cap_rule == 'none':
            return np.inf
        f = np.min if lambda_iso_cap_rule == 'min' else np.median
        return float(f(c))

    def _scegli(rl, idx=None):
        idx = np.arange(n) if idx is None else idx
        ka = _pick(rl, None if Ca is None else Ca[idx], sel_a[idx], len(ga))
        ki = _pick(rl, Ci[idx], sel_i[idx], len(gi))
        cap = _cap(caps[idx])
        lam_i = min(float(gi[ki]), cap)
        return ka, ki, lam_i, cap

    ka, ki, lam_iso, cap = _scegli(rule)
    lam_aniso = float(ga[ka])
    alt_rule = [x for x in AGGREGATION_RULES if x != rule][0]
    alt = None
    if not (alt_rule == 'geomean' and Ca is None):
        a_ka, _, a_li, _ = _scegli(alt_rule)
        alt = dict(rule=alt_rule, lambda_aniso=float(ga[a_ka]), lambda_iso=a_li)

    # ── stabilita': la scelta sopravvive al ricampionamento delle acquisizioni? ─
    rng = np.random.default_rng(seed)
    boot_a, boot_i = [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        b_ka, _, b_li, _ = _scegli(rule, idx)
        boot_a.append(b_ka - ka)
        boot_i.append(_idx_on_grid(b_li, gi) - _idx_on_grid(lam_iso, gi))
    boot_a, boot_i = np.array(boot_a), np.array(boot_i)

    # ── il prezzo: di quanto ogni acquisizione e' lontana dal proprio ottimo ─
    ki_eff = _idx_on_grid(lam_iso, gi)
    d_a = sel_a - ka
    d_i = sel_i - ki_eff
    snr = np.array([r['noise']['snr'] for r in records], float)
    try:
        import warnings
        from scipy.stats import spearmanr
        with warnings.catch_warnings():            # input costante -> nan, non un avviso
            warnings.simplefilter('ignore')
            rho_a = float(spearmanr(snr, sel_a).correlation)
            rho_i = float(spearmanr(snr, sel_i).correlation)
    except Exception:                                   # scipy e' una dipendenza,
        rho_a = rho_i = float('nan')                    # ma il record non deve morire

    def _edge(k, g):
        return 'lower' if k == 0 else ('upper' if k == len(g) - 1 else 'no')

    from .. import __version__ as _v
    return dict(
        kind=PROTOCOL_CALIBRATION_KIND, format_version=PROTOCOL_CALIBRATION_FORMAT,
        name=name, created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        toolbox_version=_v,
        protocol=fp0,
        hyperparameters=dict(lambda_aniso=lam_aniso, lambda_iso=lam_iso, n_iso=int(n_iso),
                             concentration_gate=gate, lambda_aniso_method=method),
        model=model,
        aggregation=dict(
            rule=rule, lambda_iso_cap_rule=lambda_iso_cap_rule,
            n_acquisitions=n, lambda_aniso_grid=ga.tolist(), lambda_iso_grid=gi.tolist(),
            lambda_iso_before_cap=float(gi[ki]), lambda_iso_cap=(None if not np.isfinite(cap)
                                                                  else cap),
            lambda_iso_capped=bool(np.isfinite(cap) and gi[ki] > cap),
            n_acquisitions_over_own_cap=int(np.sum(lam_iso > caps + 1e-15)),
            lambda_aniso_at_grid_edge=_edge(ka, ga),
            lambda_iso_at_grid_edge=_edge(ki_eff, gi),
            alternative=alt,
            note=("la curva di lambda_aniso di ogni acquisizione e' calcolata al suo "
                  "lambda_iso: aggregarle e' un'approssimazione di primo ordine")),
        stability=(dict(n_boot=0, note='non calcolata (n_boot=0)') if not len(boot_a) else dict(
            n_boot=int(n_boot), seed=int(seed),
            lambda_aniso_same=float(np.mean(boot_a == 0)),
            lambda_aniso_within_1_step=float(np.mean(np.abs(boot_a) <= 1)),
            lambda_iso_same=float(np.mean(boot_i == 0)),
            lambda_iso_within_1_step=float(np.mean(np.abs(boot_i) <= 1)))),
        cost=dict(
            lambda_aniso_steps_mean=float(np.mean(np.abs(d_a))),
            lambda_aniso_steps_max=int(np.max(np.abs(d_a))),
            lambda_aniso_frac_2plus=float(np.mean(np.abs(d_a) >= 2)),
            lambda_iso_steps_mean=float(np.mean(np.abs(d_i))),
            lambda_iso_steps_max=int(np.max(np.abs(d_i))),
            lambda_iso_frac_2plus=float(np.mean(np.abs(d_i) >= 2)),
            spearman_snr_vs_own_lambda_aniso_idx=rho_a,
            spearman_snr_vs_own_lambda_iso_idx=rho_i,
            note='passi della griglia di aggregazione'),
        acquisitions=[dict(id=ids[j], snr=float(snr[j]),
                           toolbox_version=records[j].get('toolbox_version'),
                           lambda_aniso=float(hp[j]['lambda_aniso']), lambda_aniso_idx=int(sel_a[j]),
                           lambda_iso=float(hp[j]['lambda_iso']), lambda_iso_idx=int(sel_i[j]),
                           lambda_iso_cap=(None if not np.isfinite(caps[j]) else float(caps[j])))
                      for j in range(n)],
    )


def save_protocol_calibration(cal, path):
    """Scrive il JSON e ne restituisce lo sha256 (la sua identita' nei run report)."""
    import hashlib, json
    if cal.get('kind') != PROTOCOL_CALIBRATION_KIND:
        raise ValueError('non e una calibrazione di protocollo')
    txt = json.dumps(cal, indent=2, ensure_ascii=False)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def load_protocol_calibration(path):
    """Legge e valida il JSON; aggiunge `_source` = {path, sha256}."""
    import hashlib, json, os
    with open(path, 'rb') as fh:
        raw = fh.read()
    cal = json.loads(raw.decode('utf-8'))
    if cal.get('kind') != PROTOCOL_CALIBRATION_KIND:
        raise ValueError(f'{path}: kind {cal.get("kind")!r}, atteso {PROTOCOL_CALIBRATION_KIND!r}')
    if cal.get('format_version') != PROTOCOL_CALIBRATION_FORMAT:
        raise ValueError(f'{path}: format_version {cal.get("format_version")} non supportato')
    for k in ('lambda_aniso', 'lambda_iso', 'n_iso', 'concentration_gate'):
        if k not in cal.get('hyperparameters', {}):
            raise ValueError(f'{path}: manca hyperparameters.{k}')
    cal['_source'] = dict(path=os.path.abspath(path),
                          sha256=hashlib.sha256(raw).hexdigest())
    return cal
