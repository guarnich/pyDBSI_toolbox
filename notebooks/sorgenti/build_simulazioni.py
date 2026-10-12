"""Genera ../simulazioni_pyDBSI.ipynb: UN notebook, autosufficiente, da lanciare una volta.

Il codice di sim_accuratezza.py e sim_grafici.py viene copiato nel notebook (si modificano quei due file e
si rigenera), e la tabella dei gradienti del P3 vero ci entra come testo: il notebook non dipende da file
accanto a se'. Uso:  python notebooks/sorgenti/build_simulazioni.py
"""
import json
import re
from pathlib import Path

QUI = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))


def code(s):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))


def sorgente(nome):
    s = (QUI / nome).read_text()
    s = re.sub(r'^from __future__ import annotations\n', '', s, flags=re.M)
    s = re.sub(r'^from sim_accuratezza import .*\n', '', s, flags=re.M)
    return s.strip('\n')


P3_BVAL = (QUI / 'protocolli' / 'P3.bval').read_text().strip()
P3_BVEC = (QUI / 'protocolli' / 'P3.bvec').read_text().strip()

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
# pyDBSI — Simulazioni di accuratezza (toolbox ≥ 1.8.1)

**Domanda.** Quanto le mappe del toolbox si avvicinano alla verità? Sui dati veri non c'è una ground
truth (asse 3 del disegno dei 5P: l'accuratezza si misura **solo** in simulazione). Qui la verità è nota
voxel per voxel.

**La catena è quella di produzione**, non una scorciatoia:

> segnale vero (S0 = 1000) → rumore **Rician** (σ = S0/SNR) → **`dwidenoise`** (MRtrix, MP-PCA) →
> mappa `-noise` di dwidenoise come `noise_map` → `DBSI_Adaptive().fit(run_calibration=True)`

cioè quello che succede ai 5P: calibrazione libera, mappa di rumore di dwidenoise. Non c'è N4 (il fantoccio
non ha campo di bias).

**Due bracci, e perché il principale è il grezzo.** Il braccio **grezzo** salta il denoising e passa la σ
vera; il braccio **produzione** passa da dwidenoise. Sul fantoccio MP-PCA è troppo efficace: toglie il
92–96% del rumore (il fantoccio è a basso rango, il tessuto vero no), e la calibrazione libera scende a
λ_aniso ≈ 0.5 contro 8.4 del grezzo e 5–13 dei dati veri, che **sono** denoisati (prova del 2026-10-12, P3,
SNR 26). Il braccio grezzo riproduce il regime di λ dei dati veri, quindi è quello delle figure principali; il
braccio con dwidenoise gira a `SNR_PRODUZIONE` come **analisi di sensibilità su un caso idealizzato**: dice
cosa farebbe il fit se il denoising fosse così efficace, non cosa fa sui dati veri.

**Un volume per condizione.** I tre fantocci stanno affiancati in un unico volume, così ogni condizione
(protocollo × SNR × braccio) ha **una** calibrazione e **un** dwidenoise, come un cervello con molti tessuti:

| fantoccio | che cosa mostra | figure |
|---|---|---|
| **anatomico** | fetta con GM, due fasci che si incrociano a 90° e un terzo a 60°, ventricolo, lesioni demielinizzante / infiammatoria / edematosa, tumore; volume parziale ai bordi | immagini vero / stimato / errore |
| **campi** | tutti i parametri variano con continuità (campi gaussiani lisci), metà voxel con incrocio 45–90° | stima contro verità, voxel per voxel |
| **batteria** | i 13 scenari della validazione di luglio, blocchi omogenei con eterogeneità lieve | errore per scenario, per SNR e per braccio; conteggio delle popolazioni; incertezza |

**Diffusività isotrope vere ≠ centroidi di Stage D.** Con `D_VARIABILE = True` (default) le diffusività
vere sono estratte per voxel (R 0.10–0.25, H 0.7–1.3, W 2.8–3.2 ×1e-3 mm²/s), mentre Stage D usa centroidi
fissi 0.15 / 1.0 / 3.0: è il caso reale, il modello non è esatto. Con `False` si misura lo stimatore quando
il modello è esatto.

**Quanto dura.** La calibrazione libera costa ~20–25 min per condizione (quasi tutto è il primo passaggio di
discrepanza su λ_aniso: NNLS su 1000 voxel a λ quasi nullo). Default: 3 condizioni grezze sul P3 + 1 con
dwidenoise + 1 per ogni altro protocollo trovato. La condizione con dwidenoise è la più lenta (a λ_aniso ≈ 0.5
la NNLS di Stage A arriva al tetto di iterazioni in quasi ogni voxel): togliere `'produzione'` da `BRACCI` per
saltarla. In tutto **3–5 ore**. Ogni condizione
è salvata appena finita in `risultati_simulazioni/v<versione>/`: se il notebook si interrompe, rilanciandolo riparte
da dove era. `RAPIDO = True` fa una prova con fantocci dimezzati (numeri solo indicativi).

**Dove lanciarlo.** Sul fisso, dove ci sono `dwidenoise` e le tabelle dei 5 protocolli sul NAS (senza NAS
gira il solo P3, la cui tabella è dentro il notebook). Da una cartella qualsiasi:
`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 simulazioni_pyDBSI.ipynb`
oppure dal browser, *Run All*.
''')

md('## 0 — Configurazione')
code(r'''
RAPIDO      = False                 # True: fantocci dimezzati, prova veloce (numeri indicativi)
RIFARE      = False                 # True: ignora le condizioni gia' salvate
D_VARIABILE = True                  # diffusivita' isotrope vere estratte per voxel (vedi sopra)
SNR_LISTA   = [15, 26, 40]          # SNR termico (S0 / sigma) delle condizioni sul P3
SNR_RIF     = 26                    # SNR delle figure principali e del confronto fra protocolli
BRACCI      = ['grezzo', 'produzione']  # il primo e' quello delle figure principali (vedi sopra)
SNR_PRODUZIONE = [26]               # SNR a cui gira anche il braccio con dwidenoise (sensibilita')
DWIDENOISE  = None                  # None = cercato da solo (PATH, /usr/local/mrtrix3); oppure il percorso
N_CAL       = 1000                  # voxel di calibrazione (default del toolbox, come nei fit veri)

# Tabelle dei 5P sul NAS: se ci sono, la batteria gira anche su P1, P2, P4, P5 (a SNR_RIF, produzione).
ROOT_5P = "/mnt/msbiobnas/Users/Francesco/DBSI_FG/DATA_5P/Processed"
STEM    = "DWITE106P2sense19nsoppyAPP"
SESSIONE = "20251210"
''')

md(r'''
## 1 — Il codice delle simulazioni

Due celle lunghe: i fantocci, la catena e le misure; poi le figure. Non serve leggerle per usare il
notebook. (Sorgenti in `notebooks/sorgenti/`: `sim_accuratezza.py`, `sim_grafici.py`; il notebook si rigenera con
`python notebooks/sorgenti/build_simulazioni.py`.)
''')
code(sorgente('sim_accuratezza.py'))
code(sorgente('sim_grafici.py'))

code(r'''
if RAPIDO:
    N_CAL = 200                     # prova veloce: calibrazione su meno voxel
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings('ignore', category=RuntimeWarning)
stile()

VER = versione()
_vt = tuple(int(x) for x in VER.split('.')[:3])
assert _vt >= (1, 8, 1), f'serve pyDBSI >= 1.8.1, trovata {VER}'
DN = trova_dwidenoise(DWIDENOISE)
print(f'pyDBSI {VER}\ndwidenoise: {DN or "NON TROVATO -> solo il braccio grezzo"}')
if DN is None:
    BRACCI = ['grezzo']

P3_BVAL = """''' + P3_BVAL + r'''"""
P3_BVEC = """''' + P3_BVEC + r'''"""
def _da_testo(bval_txt, bvec_txt):
    import io as _io
    bv = np.loadtxt(_io.StringIO(bval_txt)).ravel()
    bc = np.loadtxt(_io.StringIO(bvec_txt)); bc = bc.T if bc.shape[0] == 3 else bc
    n = np.linalg.norm(bc, axis=1, keepdims=True); n[n == 0] = 1
    return bv, bc / n

PROTOCOLLI = {'P3': _da_testo(P3_BVAL, P3_BVEC)}
for P in (1, 2, 4, 5):
    base = Path(ROOT_5P) / f'{P:06d}' / SESSIONE / 'DWI_4s' / 'prep'
    fb, fv = base / f'{STEM}_corrected.bval', base / f'{STEM}_corrected.bvec'
    if fb.is_file() and fv.is_file():
        PROTOCOLLI[f'P{P}'] = carica_protocollo(fb, fv)
for k, p in PROTOCOLLI.items():
    print(f'  {k}: {len(p[0])} volumi — {descrivi_protocollo(p)}')

OUT = Path.cwd() / 'risultati_simulazioni' / f'v{VER.replace(".", "")}' / ('rapido' if RAPIDO else 'completo')
FIG = OUT / 'figure'; FIG.mkdir(parents=True, exist_ok=True)
print('risultati in', OUT)
''')

md(r'''
## 2 — Le condizioni

Per ogni condizione: volume pulito → rumore → (dwidenoise) → fit con calibrazione libera. Si salvano le
mappe stimate, le incertezze e la calibrazione scelta. Il volume pulito dipende solo dal protocollo (stesso
seme per tutti): le condizioni differiscono **solo** per rumore e catena, quindi si confrontano direttamente.
''')
code(r'''
CONDIZIONI = [('P3', s, BRACCI[0]) for s in SNR_LISTA]
CONDIZIONI += [('P3', s, b) for b in BRACCI[1:] for s in SNR_PRODUZIONE]
CONDIZIONI += [(p, SNR_RIF, BRACCI[0]) for p in PROTOCOLLI if p != 'P3']

VOLUMI = {p: volume_composito(prot, d_variabile=D_VARIABILE, piccolo=RAPIDO) for p, prot in PROTOCOLLI.items()}
A0, MASK0, PARTI0 = VOLUMI['P3']
print(f'volume {A0.shape[:3]}, {MASK0.sum():,} voxel nella maschera; {len(CONDIZIONI)} condizioni')

def condizione(p, snr, braccio):
    f = OUT / f'{p}_snr{snr}_{braccio}.npz'
    if f.is_file() and not RIFARE:
        z = np.load(f, allow_pickle=True)
        return z['res'], z['se'], json.loads(str(z['info']))
    A, mask, _ = VOLUMI[p]
    t0 = time.time()
    res, m, info = esegui(A, PROTOCOLLI[p], snr, braccio, dwidenoise=DN, seed=1000 + snr, mask=mask,
                          n_calibration_voxels=N_CAL)
    info.update(protocollo=p, secondi=round(time.time() - t0), versione=VER, d_variabile=D_VARIABILE,
                rapido=RAPIDO)
    se = m.uncertainty_['se'].astype(np.float32)
    np.savez_compressed(f, res=res.astype(np.float32), se=se, info=json.dumps(info))
    return res, se, info

RIS, INFO = {}, []
t_tot = time.time()
for i, (p, snr, b) in enumerate(CONDIZIONI, 1):
    t0 = time.time()
    res, se, info = condizione(p, snr, b)
    RIS[(p, snr, b)] = (res, se)
    INFO.append(info)
    print(f'[{i}/{len(CONDIZIONI)}] {p} SNR {snr:>2} {b:<10}  lambda_aniso {info["lambda_aniso"]:.4g}  '
          f'lambda_iso {info["lambda_iso"]:.4g}  sigma mappa/vera {info["sigma_mappa_mediana"] / info["sigma_vera"]:.3f}  '
          f'({time.time() - t0:.0f} s; totale {(time.time() - t_tot) / 60:.0f} min)', flush=True)
CAL = pd.DataFrame(INFO)[['protocollo', 'snr', 'braccio', 'lambda_aniso', 'lambda_iso', 'n_iso',
                          'sigma_vera', 'sigma_mappa_mediana', 'secondi']]
CAL.to_csv(OUT / 'calibrazioni.csv', index=False)
display(CAL)

def mappe(p, snr, b, parte):
    """Mappe stimate (nome breve) e incertezze ritagliate su una parte del volume."""
    res, se = RIS[(p, snr, b)]
    sl = VOLUMI[p][2][parte]['sl']
    out = {k: res[sl][..., IX[c]].astype(float) for k, c in METRICHE.items()}
    out['npop'] = np.nan_to_num(res[sl][..., IX['n_fiber_populations']].astype(float))
    for k, c in METRICHE.items():
        out['SE_' + k] = se[sl][..., IX[c]].astype(float)
    out['_res'] = res[sl]
    return out
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 3 — Il fantoccio anatomico

Il tessuto dominante in ogni voxel e, sotto, le mappe vere. Dove due fasci si sovrappongono nasce un
incrocio (A×B a 90°, A×C a 60°); ai bordi c'è volume parziale. Il tensore vero è definito dove la fibra
pesa almeno il 5% del voxel.
''')
code(r'''
AN = PARTI0['anatomico']
mappa_regioni(AN['pesi'], file=FIG / '01_fantoccio_regioni.png'); plt.show()
''')

md(r'''
## 4 — Vero, stimato, errore (anatomico, catena di produzione)

Una riga per mappa: verità, stima, errore = stima − vero (rosso = sovrastima, blu = sottostima, scala
simmetrica). Nel tensore: **nero** = fibra vera senza tensore stimato (fibra mancata), **rosa** = tensore
stimato dove non c'è fibra vera (fibra falsa). Il conteggio delle popolazioni: rosso dove è sbagliato.
''')
code(r'''
E = mappe('P3', SNR_RIF, BRACCI[0], 'anatomico')
mappe_vero_stima_errore(AN['verita'], E, titolo=f'Anatomico — P3, SNR {SNR_RIF}, {BRACCI[0]}',
                        file=FIG / f'02_anatomico_P3_snr{SNR_RIF}_{BRACCI[0]}.png'); plt.show()
''')
code(r'''
# Per regione (solo i voxel "puri": il tessuto dominante pesa almeno il 90%)
W = np.stack([AN['pesi'][t] for t in AN['pesi']], -1)
puri = W.max(-1) >= 0.9
reg = np.where(puri, W.argmax(-1), -1)
righe = []
for (p, snr, b) in CONDIZIONI:
    if p != 'P3':
        continue
    T = tabella_errori(mappe('P3', snr, b, 'anatomico'), AN['verita'], reg, list(AN['pesi']))
    T['SNR'], T['braccio'] = snr, b
    righe.append(T)
TAB_AN = pd.concat(righe, ignore_index=True)
TAB_AN.to_csv(OUT / 'anatomico_per_regione.csv', index=False)
v = TAB_AN[(TAB_AN.SNR == SNR_RIF) & (TAB_AN.braccio == BRACCI[0]) & TAB_AN.metrica.isin(list(METRICHE))]
print(f'bias per regione (P3, SNR {SNR_RIF}, {BRACCI[0]}; diffusivita in e-3 mm^2/s):')
display(v.pivot(index='gruppo', columns='metrica', values='bias')[list(METRICHE)].round(3))
''')
code(r'''
if ('P3', SNR_RIF, 'produzione') in RIS:
    Ep = mappe('P3', SNR_RIF, 'produzione', 'anatomico')
    mappe_vero_stima_errore(AN['verita'], Ep, metriche=list(METRICHE),
                            titolo=f'Anatomico — P3, SNR {SNR_RIF}, con dwidenoise (sensibilita, fantoccio idealizzato)',
                            file=FIG / f'03_anatomico_P3_snr{SNR_RIF}_produzione.png'); plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 5 — Batteria: errore per scenario

Per ogni scenario e mappa: punto = errore mediano dei voxel, barra = 10°–90° percentile (quanto varia da
voxel a voxel). Lo zero è la verità. Solo i voxel interni ai blocchi (al bordo dwidenoise mescola due
scenari). Le diffusività in 1e-3 mm²/s.

**Come leggere i numeri.** Il *bias* è l'errore medio: dice se lo scenario è stimato sistematicamente troppo
alto o troppo basso. Il *MAE* è la media degli errori **voxel per voxel** in valore assoluto: contiene anche la
dispersione dovuta al rumore, quindi è sempre più grande del bias. (Il riferimento di luglio, MAE FF 0.030, era
l'errore della **media** di ogni scenario: è il bias di qui, non il MAE.) Il tensore dei voxel con poca fibra
(GM, edema, fibra debole) è poco determinato dal dato: lì l'errore del tensore è grande per ragioni fisiche,
non per un difetto del fit (vedi le mappe di incertezza).
''')
code(r'''
BA = PARTI0['batteria']
INN = interno(BA['etichette'])
NOMI_SC = BA['nomi']

def voxel_batteria(p, snr, b):
    """DataFrame lungo: un errore per voxel, scenario e metrica (piu' npop e SE per l'incertezza)."""
    E = mappe(p, snr, b, 'batteria'); tru = VOLUMI[p][2]['batteria']['verita']
    lab = VOLUMI[p][2]['batteria']['etichette']; inn = interno(lab)
    out = []
    for k in METRICHE:
        t, e, se = tru[k][inn], E[k][inn], E['SE_' + k][inn]
        out.append(pd.DataFrame(dict(scenario=np.array(NOMI_SC)[lab[inn]], metrica=k,
                                     vero=t * SCALA[k], stima=e * SCALA[k], errore=(e - t) * SCALA[k],
                                     z=(e - t) / se)))
    out.append(pd.DataFrame(dict(scenario=np.array(NOMI_SC)[lab[inn]], metrica='npop', vero=tru['npop'][inn],
                                 stima=E['npop'][inn], errore=E['npop'][inn] - tru['npop'][inn], z=np.nan)))
    d = pd.concat(out, ignore_index=True); d['protocollo'], d['SNR'], d['braccio'] = p, snr, b
    return d

VB = pd.concat([voxel_batteria(*c) for c in CONDIZIONI], ignore_index=True)
VB.to_csv(OUT / 'batteria_voxel.csv.gz', index=False)

sel = VB[(VB.protocollo == 'P3') & (VB.braccio == BRACCI[0])]
errori_per_scenario(sel, FRAZIONI, gruppo_col='SNR', titolo=f'Batteria — frazioni (P3, {BRACCI[0]})',
                    file=FIG / f'04_batteria_frazioni_P3_{BRACCI[0]}.png'); plt.show()
errori_per_scenario(sel, TENSORE, gruppo_col='SNR', titolo=f'Batteria — tensore pesato (P3, {BRACCI[0]})',
                    file=FIG / f'05_batteria_tensore_P3_{BRACCI[0]}.png'); plt.show()
''')
code(r'''
conteggio_popolazioni(sel[sel.metrica == 'npop'], gruppo_col='SNR',
                      titolo=f'Batteria — popolazioni stimate (P3, {BRACCI[0]})',
                      file=FIG / f'06_batteria_popolazioni_P3_{BRACCI[0]}.png'); plt.show()
''')
code(r'''
# Tabella: per scenario, a SNR_RIF, vero / stima mediana / bias / MAE; e il riepilogo per SNR
def riassunto(d):
    g = d[d.metrica != 'npop'].groupby(['scenario', 'metrica'], sort=False)
    return pd.DataFrame(dict(vero=g.vero.median(), stima=g.stima.median(), bias=g.errore.mean(),
                             mae=g.errore.apply(lambda x: np.nanmean(np.abs(x))),
                             mancanti_pct=g.stima.apply(lambda x: 100 * np.mean(~np.isfinite(x))))).reset_index()
R = riassunto(sel[sel.SNR == SNR_RIF])
R.to_csv(OUT / f'batteria_P3_snr{SNR_RIF}_{BRACCI[0]}.csv', index=False)
print(f'bias per scenario (P3, SNR {SNR_RIF}, {BRACCI[0]}):')
display(R.pivot(index='scenario', columns='metrica', values='bias').reindex(NOMI_SC)[list(METRICHE)].round(3))
M = pd.concat([riassunto(sel[sel.SNR == s]).assign(SNR=s) for s in SNR_LISTA])
print('MAE medio sui 13 scenari, per SNR:')
display(M.groupby(['SNR', 'metrica']).mae.mean().unstack()[list(METRICHE)].round(3))
''')
code(r'''
# Errore angolare della direzione principale, per scenario con fibra (gradi)
righe = []
for snr in SNR_LISTA:
    E = mappe('P3', snr, BRACCI[0], 'batteria')
    ang = errore_angolare(E['_res'], BA['verita']['dirs'])
    for s, n in enumerate(NOMI_SC):
        q = ang[..., 0][INN & (BA['etichette'] == s)]
        if np.isfinite(q).any():
            righe.append(dict(scenario=n, SNR=snr, mediana=np.nanmedian(q), p90=np.nanpercentile(q, 90)))
ANG = pd.DataFrame(righe)
ANG.to_csv(OUT / 'batteria_errore_angolare.csv', index=False)
display(ANG.pivot(index='scenario', columns='SNR', values='mediana').round(1))
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 6 — Effetto del denoising

Stessa batteria, stesso rumore, a SNR_RIF: con `dwidenoise` + sua mappa (produzione) contro dati grezzi +
σ vera. Se i due bracci coincidono, il denoising non sposta le stime; dove differiscono, è il denoising
(non lo stimatore) a cambiare il numero.
''')
code(r'''
if sum(('P3', SNR_RIF, b) in RIS for b in BRACCI) > 1:
    sel2 = VB[(VB.protocollo == 'P3') & (VB.SNR == SNR_RIF)]
    errori_per_scenario(sel2, FRAZIONI, gruppo_col='braccio', ordine=BRACCI,
                        titolo=f'Denoising — frazioni (P3, SNR {SNR_RIF})',
                        file=FIG / f'07_denoising_frazioni_P3_snr{SNR_RIF}.png'); plt.show()
    errori_per_scenario(sel2, TENSORE, gruppo_col='braccio', ordine=BRACCI,
                        titolo=f'Denoising — tensore (P3, SNR {SNR_RIF})',
                        file=FIG / f'08_denoising_tensore_P3_snr{SNR_RIF}.png'); plt.show()
    D = pd.concat([riassunto(sel2[sel2.braccio == b]).assign(braccio=b) for b in BRACCI])
    print('MAE medio sui 13 scenari, per braccio:')
    display(D.groupby(['braccio', 'metrica']).mae.mean().unstack()[list(METRICHE)].round(3))
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 7 — Campi continui: stima contro verità

Ogni punto è un voxel (densità in scala logaritmica); la diagonale tratteggiata è la stima perfetta. Per il
tensore, "fibra mancata" = quota dei voxel con fibra vera in cui il toolbox non stima un tensore.

**Il "gradino" della FF.** Sotto una FF vera di ~0.15 la stima o è 0 o parte da ~0.15: è la soglia di fibra
del toolbox (`fiber_threshold`, applicata alla FF finale dalla 1.8.0). Una fibra più debole della soglia viene
tolta (FF = 0, frazione restituita allo spettro isotropo); una fibra falsa che la supera resta. Sono i due
errori opposti di una soglia, ed è qui che si vedono.
''')
code(r'''
CA = PARTI0['campi']
for b in [b for b in BRACCI if ('P3', SNR_RIF, b) in RIS]:
    E = mappe('P3', SNR_RIF, b, 'campi')
    stima_contro_vero(CA['verita'], E, titolo=f'Campi continui — P3, SNR {SNR_RIF}, {b}',
                      file=FIG / f'09_campi_P3_snr{SNR_RIF}_{b}.png'); plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 8 — Le mappe di incertezza descrivono l'errore?

z = (stima − vero) / SE sui voxel della batteria senza incroci (dove lo stimatore è quasi senza bias). Se
la SE è giusta, gli istogrammi seguono la curva grigia N(0,1) e la copertura al 95% è vicina al 95%.
Attese due deviazioni: con dwidenoise il rumore residuo è più piccolo di σ (SE larga → istogramma più
stretto), e dove c'è bias l'istogramma si sposta.
''')
code(r'''
cross = [n for n in NOMI_SC if n.startswith('Crossing')]
zsel = VB[(VB.protocollo == 'P3') & (VB.SNR == SNR_RIF) & ~VB.scenario.isin(cross)]
istogrammi_z(zsel, list(METRICHE), gruppo_col='braccio', titolo=f'Incertezza — P3, SNR {SNR_RIF}, senza incroci',
             file=FIG / f'10_incertezza_P3_snr{SNR_RIF}.png'); plt.show()
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## 9 — Protocolli dei 5P

La stessa batteria su ogni protocollo trovato, a SNR_RIF e con la catena di produzione: a parità di SNR,
quanto pesa il disegno dell'acquisizione (shell, direzioni) sull'accuratezza. (I 5P veri hanno SNR diversi
fra loro: qui è tenuto fisso apposta.)
''')
code(r'''
if len(PROTOCOLLI) > 1:
    selp = VB[(VB.SNR == SNR_RIF) & (VB.braccio == BRACCI[0])]
    T = pd.concat([riassunto(selp[selp.protocollo == p]).assign(protocollo=p) for p in PROTOCOLLI])
    T.to_csv(OUT / f'protocolli_snr{SNR_RIF}.csv', index=False)
    mae_per_gruppo(T, list(METRICHE), 'protocollo', titolo=f'MAE per protocollo (SNR {SNR_RIF}, {BRACCI[0]})',
                   file=FIG / f'11_protocolli_mae_snr{SNR_RIF}.png'); plt.show()
    errori_per_scenario(selp, FRAZIONI, gruppo_col='protocollo', ordine=list(PROTOCOLLI),
                        titolo=f'Protocolli — frazioni (SNR {SNR_RIF})',
                        file=FIG / f'12_protocolli_frazioni_snr{SNR_RIF}.png'); plt.show()
else:
    print('solo il P3: le tabelle degli altri protocolli non sono state trovate in', ROOT_5P)
''')

# ═════════════════════════════════════════════════════════════════════════════
md('## 10 — Riepilogo')
code(r'''
righe = []
for (p, snr, b) in CONDIZIONI:
    d = VB[(VB.protocollo == p) & (VB.SNR == snr) & (VB.braccio == b)]
    r = riassunto(d)
    riga = dict(protocollo=p, SNR=snr, braccio=b)
    for k in METRICHE:
        riga[f'MAE_{k}'] = r[r.metrica == k].mae.mean()
    n = d[d.metrica == 'npop']
    riga['npop_giusto_pct'] = 100 * np.mean(np.round(n.stima) == np.round(n.vero))
    righe.append(riga)
RIEP = pd.DataFrame(righe)
RIEP.to_csv(OUT / 'riepilogo.csv', index=False)
display(RIEP.round(3))
(OUT / 'leggimi.json').write_text(json.dumps(dict(
    versione=VER, rapido=RAPIDO, d_variabile=D_VARIABILE, snr=SNR_LISTA, snr_rif=SNR_RIF, bracci=BRACCI,
    protocolli={k: descrivi_protocollo(v) for k, v in PROTOCOLLI.items()}, dwidenoise=DN), indent=2))
print('figure in', FIG)
for f in sorted(FIG.glob('*.png')):
    print('  ', f.name)
''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
for c in nb['cells']:
    L = c['source'].split('\n')
    c['source'] = [l + '\n' for l in L[:-1]] + [L[-1]]
out = (QUI.parent if QUI.name == 'sorgenti' else QUI) / 'simulazioni_pyDBSI.ipynb'
json.dump(nb, open(out, 'w'), indent=1, ensure_ascii=False)
print('scritto', out, len(cells), 'celle')
