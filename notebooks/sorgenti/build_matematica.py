"""Genera matematica_pyDBSI.ipynb: le figure della matematica del toolbox, un notebook da lanciare una volta.

Il codice di sim_accuratezza.py (fantocci, rumore) e di mat_figure.py (figure) e' copiato nel notebook:
si modificano quei file e si rigenera.  Uso:  python build_matematica.py
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

md(r'''
# pyDBSI — La matematica del toolbox in figure

Compagno visivo del report *Il viaggio del voxel*: per ogni passaggio che il report scrive in formule, qui
c'è una figura. I numeri delle sezioni (§) sono quelli del report.

**Da dove vengono le figure.** Il più possibile dal toolbox stesso: il notebook lancia il fit vero e ne
**registra** gli ingressi (la matrice A, i λ, il dizionario, il segnale corretto) e ogni passo della
calibrazione. Dove una figura mostra un passaggio interno che il toolbox non restituisce (i bacini, i
candidati del cono, la traiettoria del Levenberg–Marquardt) lo ricalcola con le stesse funzioni e **controlla**
che il risultato coincida con quello del toolbox: se non coincide, la cella si ferma.

**Dati.** Le sezioni sulla calibrazione (§7–9) girano su un'acquisizione vera del P3 se le trovi i file
(configurazione qui sotto: la sessione P3 dei 5P sul NAS), altrimenti sul fantoccio. Le sezioni che hanno
bisogno della verità (§14, §17, §18, §23) girano sul fantoccio della batteria.

**Quanto dura.** ~1–2 ore, quasi tutte nella §7 (stabilità di λ al variare del campione: 4 dimensioni × 5
semi = 20 calibrazioni). `STABILITA_LAMBDA = False` la salta. Le figure finiscono in `risultati_matematica/`.
''')

md('## 0 — Configurazione')
code(r'''
STABILITA_LAMBDA = True          # §7: 20 calibrazioni (la parte lunga)
N_LIST_LAMBDA    = [250, 500, 1000, 2000]
SEMI_LAMBDA      = [0, 1, 2, 3, 4]
SNR_FANTOCCIO    = 26

# Un'acquisizione vera del P3 per la calibrazione (§7-9). Se un file manca si usa il fantoccio.
REALE = dict(
    prep = "/mnt/msbiobnas/Users/Francesco/DBSI_FG/DATA_5P/Processed/000003/20251210/DWI_4s/prep",
    stem = "DWITE106P2sense19nsoppyAPP",
)
''')

md('## 1 — Il codice (fantocci e figure)\n\nSorgenti in `notebooks/sorgenti/` (`sim_accuratezza.py`, `mat_figure.py`); si rigenera con `python notebooks/sorgenti/build_matematica.py`.')
code(sorgente('sim_accuratezza.py'))
code(sorgente('mat_figure.py'))
code(r'''
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings('ignore', category=RuntimeWarning)
stile()
VER = dbsi_toolbox.__version__
assert tuple(int(x) for x in VER.split('.')[:3]) >= (1, 9, 0), f'serve pyDBSI >= 1.9.0, trovata {VER}'
OPZ_18 = dict(crossing_dir_refine=False, two_vs_one_threshold=None)   # i crossing come fino alla 1.8 (§14-18)
FIG = Path.cwd() / 'risultati_matematica' / f'v{VER.replace(".", "")}'
FIG.mkdir(parents=True, exist_ok=True)

P3_BVAL = """''' + P3_BVAL + r'''"""
P3_BVEC = """''' + P3_BVEC + r'''"""
def _da_testo(a, b):
    import io as _io
    bv = np.loadtxt(_io.StringIO(a)).ravel(); bc = np.loadtxt(_io.StringIO(b)); bc = bc.T if bc.shape[0] == 3 else bc
    n = np.linalg.norm(bc, axis=1, keepdims=True); n[n == 0] = 1
    return bv, bc / n
P3 = _da_testo(P3_BVAL, P3_BVEC)
print(f'pyDBSI {VER}; P3: {descrivi_protocollo(P3)}; figure in {FIG}')

# fantoccio della batteria (13 scenari) per le sezioni che hanno bisogno della verita'
A_b, TRU, LAB, NOMI_SC = fantoccio_batteria(P3, blocco=(6, 6, 4), colonne=7, d_variabile=False, seed=0)
MB, SIG_B = rumore_rician(A_b, SNR_FANTOCCIO, seed=1)
MASK_B = LAB >= 0
def voxel_di(nome, q=0):
    s = NOMI_SC.index(nome); c = np.argwhere((LAB == s) & interno(LAB)); return tuple(c[q])
def dir_vere(ijk):
    v = TRU['dirs'][ijk]; return v[np.isfinite(v[:, 0])] if np.isfinite(v[0, 0]) else None
''')

# ═════════════════════════════════════════════════════════════════════════════
md(r'''
## §3.3 — Protocollo: 3-ISO o 2-ISO

**L'idea.** Ogni compartimento isotropo decade con la sua diffusività: `S(b) = exp(−b·D)`. A b = 2000
l'acqua libera (D = 3.0) è già sotto il rumore (e⁻⁶ ≈ 0.0025), mentre la hindered (D ≈ 1.0) vale ancora 0.14.
È solo **a b alto** che le due curve si separano; a b basso si somigliano. Per questo il toolbox chiede
b_max ≥ 2000 e almeno due gusci prima di stimare acqua e hindered separatamente (3-ISO); altrimenti le
fonde in una "non-restricted" (2-ISO).

**Cosa guardare.** A sinistra, dove cadono i gusci del P3 rispetto alle tre curve e al livello del rumore;
a destra, la differenza hindered − acqua: è sopra 2σ in tutta la zona dei b alti del P3.
''')
code(r'''
fig_protocollo(P3, SNR_FANTOCCIO, protocolli={'P3': P3}, file=FIG / '01_protocollo_3iso.png'); plt.show()
print(M.analyse_protocol(P3[0])[3])
''')

md(r'''
## §4 — Il rumore e la correzione Rician

**L'idea.** La magnitudine è il modulo di un segnale complesso con rumore sulle due componenti. Quando il
segnale vero A è grande, il rumore si somma e si sottrae in modo simmetrico. Quando A è piccolo (alto b), il
modulo **non può essere negativo**: la misura resta "sollevata" su un pavimento di circa 1.25σ anche se il
segnale vero è zero. Il secondo momento dice di quanto: E[M²] = A² + 2σ². Il toolbox toglie 2σ² a ogni misura
(con la σ del voxel, dalla mappa di rumore) e tronca a zero.

**Cosa guardare.** Al centro, la curva rossa (grezzo) che non scende sotto il pavimento e la verde
(corretta) che torna sulla diagonale. A destra, il liquor: senza correzione il pavimento ad alto b sembra un
plateau, cioè acqua ristretta; la tabellina dice quanta RF ne inventerebbe Stage D.
''')
code(r'''
fig_rician(SNR_FANTOCCIO, P3, file=FIG / '02_rician.png'); plt.show()
''')

md(r'''
## §5.3 — Lo spettro isotropo: la griglia ancorata, e la matrice A

**L'idea.** Il blocco isotropo è una serie di esponenziali, una per diffusività della griglia. I punti sono
log-uniformi (l'esponenziale è ugualmente sensibile a ogni decade di D), poi se ne aggiunge uno appena dentro
ciascuna soglia di classe (0.3 e 3.0), così nessuna classe resta senza colonna al bordo, e si riempiono i buchi
troppo larghi (rapporto > 2) con la media geometrica. Da n_iso = 6 escono 11 colonne.

**La matrice A del P3.** Ogni colonna è un "atomo": il segnale che darebbe un compartimento puro su tutte le 92
misure. Le colonne anisotrope sono una per coppia (AD, RD) e per direzione; le isotrope una per D. Il fit
cerca la combinazione non negativa di colonne che riproduce il segnale del voxel.

**Cosa guardare.** Nella matrice, le righe a b = 0 tutte a 1; le colonne di fibra "a strisce" (il segnale
dipende dall'angolo fra gradiente e fibra), quelle isotrope lisce (dipendono solo da b).
''')
code(r'''
fig, GRIGLIA = fig_griglia(file=FIG / '03_griglia_isotropa.png'); plt.show()
print('griglia (e-3 mm^2/s):', np.round(GRIGLIA * 1e3, 3))
K = intercetta_fit(P3, MB, MASK_B, np.full(MASK_B.shape, SIG_B), lam=(9.8, 0.015), opzioni=OPZ_18)
print(f"dizionario: {int(K['n_dirs'])} direzioni x {K['n_pairs']} coppie = {K['n_aniso']} colonne di fibra + "
      f"{len(K['iso_grid'])} isotrope; coppie (AD, RD) e-3: {np.round(np.asarray(K['diff_pairs']) * 1e3, 2).tolist()}")
fig_matrice_A(K, file=FIG / '04_matrice_A_P3.png'); plt.show()
''')

md(r'''
## §5.5 — Il grafo dei vicini e lo schema del cono

**Il grafo.** Per trovare i picchi del peso anisotropo serve sapere chi è vicino a chi. Una fibra non ha verso:
v e −v sono la stessa fibra. Vicino all'equatore dell'emisfero, il vicino "vero" di una direzione può stare
dalla parte opposta del cerchio (è il suo −v riportato su). Il toolbox usa |v·v′|: al centro della figura, i
vicini verdi (toolbox) includono quelli oltre l'equatore, i cerchi rossi (prodotto scalare semplice) no.

**Il cono.** Le direzioni del dizionario distano ~16.6°: la fibra vera cade quasi sempre in mezzo. Il cono
prova candidati su una calotta grande quanto la spaziatura, poi su una più piccola attorno al migliore, fino a
una risoluzione di ~1°. Il numero di candidati cresce con l'area della calotta (il quadrato del rapporto degli
angoli).
''')
code(r'''
fig, INFO_CONO = fig_grafo_e_cono(K, file=FIG / '05_grafo_e_cono.png'); plt.show()
print({k: round(float(v), 2) for k, v in INFO_CONO.items()})
''')

md(r'''
## §6.1 — n_iso non si calibra: la convergenza

**L'idea.** n_iso è la densità della griglia isotropa, non il numero di compartimenti: le frazioni riportate
vengono da Stage D, che usa tre centroidi fissi. Il notebook 06 (un soggetto vero, 3000 voxel, toolbox 1.3.6) ha
rifatto il fit a dieci densità. **Risultato:** le frazioni non si muovono; il tensore sta entro la banda di
tolleranza da n_iso = 6; oltre, cresce solo il costo (e le NNLS che non convergono). La tabella è quella del 06.
Sotto, la stessa domanda sul fantoccio col toolbox attuale.
''')
code(r'''
display(CONVERGENZA_06)
righe = []
A_s, TRU_s, LAB_s, _ = fantoccio_batteria(P3, blocco=(4, 4, 2), colonne=7, d_variabile=False, seed=3)
M_s, sg_s = rumore_rician(A_s, SNR_FANTOCCIO, seed=4)
for n_iso in (4, 6, 8, 12):
    m = DBSI_Adaptive(lambda_aniso=9.8, lambda_iso=0.015, n_iso=n_iso)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(M_s, P3[0], P3[1], LAB_s >= 0, run_calibration=False, noise_map=np.full(LAB_s.shape, sg_s))
    sel = (LAB_s >= 0) & interno(LAB_s)
    r = dict(n_iso=n_iso, colonne=len(m.dictionary_['iso_grid']))
    for k, c in (('FF', 'fiber_fraction'), ('RF', 'restricted_fraction'), ('HF', 'hindered_fraction'),
                 ('WF', 'water_fraction'), ('RDw', 'radial_diffusivity_weighted')):
        e = res[..., IX[c]][sel] - TRU_s[k][sel]
        r[k] = float(np.nanmean(np.abs(e))) * (1e3 if k == 'RDw' else 1)
    righe.append(r)
SINT = pd.DataFrame(righe); display(SINT.round(4))
fig_convergenza(SINT, file=FIG / '06_convergenza_niso.png'); plt.show()
''')

md(r'''
## §7 — Il campione di calibrazione e il suo σ

**L'idea.** La calibrazione non usa tutto il cervello: estrae 1000 voxel (seme fisso, quindi riproducibile) e
ne calcola il rumore normalizzato σ/S0. Quel rumore è molto diverso da voxel a voxel (dipende da S0: un voxel
scuro è più rumoroso). I criteri della calibrazione confrontano **la media dei residui al quadrato** con Nσ²:
il σ giusto per quel confronto è la **media quadratica** (RMS) dei σ/S0, non la mediana. Con la mediana il
bersaglio era troppo piccolo (sui dati veri RMS² ≈ 3.5 × mediana²).

**Due domande.** (a) RMS e mediana sono stabili da un campione all'altro? (b) λ cambia se cambia il campione,
e con quanti voxel smette di cambiare? Per la (b) si rifà la calibrazione vera con 5 semi diversi a 4 dimensioni.
''')
code(r'''
def _carica_reale():
    import nibabel as nib
    p = Path(REALE['prep']); s = REALE['stem']
    f = dict(dwi=p / f'{s}_preprocessed_N4.nii.gz', bval=p / f'{s}_corrected.bval', bvec=p / f'{s}_corrected.bvec',
             mask=p / f'{s}_preprocessed_brain_mask.nii.gz', noise=p / f'{s}_noise_N4.nii.gz')
    if not all(v.is_file() for v in f.values()):
        return None
    from dbsi_toolbox import load_data
    data, aff, bv, bc, mask = load_data(str(f['dwi']), str(f['bval']), str(f['bvec']), mask_path=str(f['mask']), verbose=False)
    nm = np.asarray(nib.load(str(f['noise'])).dataobj, dtype=np.float64)
    return (np.asarray(bv, float), np.asarray(bc, float)), np.asarray(data, np.float32), np.asarray(mask, bool), np.squeeze(nm)

REAL = _carica_reale()
if REAL is not None:
    PROT_C, DATA_C, MASK_C, NOISE_C = REAL
    FONTE = f"acquisizione vera ({REALE['prep']})"
else:
    PROT_C = P3
    A_c, MASK_C, _ = volume_composito(P3, piccolo=True)
    DATA_C, s_c = rumore_rician(A_c, SNR_FANTOCCIO, seed=7)
    NOISE_C = np.full(MASK_C.shape, s_c)
    FONTE = 'fantoccio composito (nessuna acquisizione vera trovata)'
print('dati della calibrazione:', FONTE)
dc = np.sqrt(np.maximum(DATA_C.astype(np.float64) ** 2 - 2 * NOISE_C[..., None] ** 2, 0))   # come il toolbox
r = rapporti_sigma(dc, MASK_C, PROT_C[0], NOISE_C)
fig, D_SIG = fig_sigma_campione(r, file=FIG / '07_sigma_campione.png'); plt.show()
print(f'mediana {np.median(r):.4f}, RMS {np.sqrt(np.mean(r**2)):.4f}, RMS^2/mediana^2 = {np.mean(r**2)/np.median(r)**2:.2f}')
''')
code(r'''
if STABILITA_LAMBDA:
    righe = []
    for n in N_LIST_LAMBDA:
        for s in SEMI_LAMBDA:
            t0 = time.time()
            Rs = registra_calibrazione(PROT_C, DATA_C, MASK_C, NOISE_C, n_cal=n, seed=s)
            righe.append(dict(n=n, seme=s, lambda_aniso=Rs['lambda_aniso'], lambda_iso=Rs['lambda_iso'],
                              sigma_cal=Rs['campione']['sigma_cal'], secondi=round(time.time() - t0)))
            print(f"  {n:>5} voxel, seme {s}: lambda_aniso {Rs['lambda_aniso']:.4g}  lambda_iso {Rs['lambda_iso']:.4g}  "
                  f"({righe[-1]['secondi']} s)", flush=True)
    STAB = pd.DataFrame(righe); STAB.to_csv(FIG / '07_lambda_vs_campione.csv', index=False)
    fig_lambda_vs_campione(STAB, file=FIG / '07_lambda_vs_campione.png'); plt.show()
''')

md(r'''
## §8–9 — La calibrazione passo per passo

**Il percorso**, su un'acquisizione (con seme 0, come in produzione):

1. **GCV del blocco isotropo sul segnale pieno** → un λ_iso preliminare. La GCV stima l'errore di
   predizione senza rifare il fit: cerca il λ che bilancia "spiegare i dati" e "non inseguire il rumore".
2. **λ_aniso preliminare per discrepanza**: il residuo medio R(λ) cresce con λ; si cerca per bisezione il λ
   a cui il residuo è grande quanto il rumore (Nσ²). I punti arancioni sono le valutazioni della bisezione.
3. **Il residuo isotropo**: con i due λ preliminari si toglie a ogni voxel la parte di fibra. Resta ciò che il
   blocco isotropo deve spiegare (nel voxel con fibra, la linea tratteggiata è molto più bassa del segnale).
4. **GCV sul residuo isotropo** → λ_iso^GCV.
5. **Tetto di discrepanza**: il λ_iso più grande per cui il blocco isotropo spiega ancora il residuo fino al
   rumore (curva verde contro la linea punteggiata). λ_iso = min(GCV, tetto). Ad alto SNR il tetto è inerte.
9. **λ_aniso finale con la GCV ridge**: la curva GCV e, a destra, i gradi di libertà efficaci: quanti
   parametri "veri" usa il fit a quel λ (da quasi N a λ piccolo a pochi a λ grande).

Il λ_aniso preliminare (passo 2) e quello finale (passo 9) **non devono coincidere**: il primo serve solo a
togliere la fibra al passo 3. La tabella sotto la figura è l'"esempio numerico" del report, rifatto qui.
''')
code(r'''
R0 = registra_calibrazione(PROT_C, DATA_C, MASK_C, NOISE_C, n_cal=1000)
figs, RIEP = fig_calibrazione(R0, PROT_C, file_prefix=str(FIG / '08_calibrazione'))
plt.show()
print('dati:', FONTE)
display(pd.Series(RIEP).to_frame('valore'))
''')

md(r'''
## §14 — Concentrazione angolare e modulazione di λ_aniso

**L'idea.** Dopo la prima soluzione di Stage A si guarda **come** è distribuito il peso anisotropo sulle
direzioni. Una fibra vera concentra il peso attorno alla sua direzione (un bacino grande: c alto). Un voxel
senza fibre "perde" un po' di segnale isotropo nel blocco anisotropo, sparpagliato su tante direzioni (tanti
bacini piccoli: c basso). Se c è basso, il voxel viene risolto di nuovo con λ_aniso moltiplicato per g(c): la
massa sparpagliata torna nello spettro isotropo.

**Cosa guardare.** Una colonna per tipo di voxel: in alto la prima soluzione, in basso dopo la modulazione
(area dei punti = peso della direzione, colore = bacino, × = fibra vera). Nella WM sana c = 1 e non cambia
niente; nella GM e nel "solo restricted" il peso è sparpagliato, la modulazione abbassa la FF. Sotto, la rampa
g(c) con le distribuzioni di c nei voxel con e senza fibra vera. I valori di c coincidono con il canale
`dominant_basin_concentration` del toolbox (controllato nella cella).
''')
code(r'''
SCEL = ['WM sana', 'WM demielinizzata', 'Crossing bil.', 'GM', 'Solo restricted']
V = [(n, voxel_di(n, 3), dir_vere(voxel_di(n, 3))) for n in SCEL]
fig, T14 = fig_concentrazione(K, V, file=FIG / '09_concentrazione.png'); plt.show()
c_tb = [float(K['res'][v[1]][IX['dominant_basin_concentration']]) for v in V]
assert np.allclose(T14.c_dopo, c_tb, atol=1e-6), 'concentrazione ricalcolata diversa dal canale del toolbox'
display(T14.round(3))
cc = K['res'][..., IX['dominant_basin_concentration']]
fib = np.nan_to_num(TRU['npop']) > 0
fig_rampa(K, cc[MASK_B & fib], cc[MASK_B & ~fib], file=FIG / '10_rampa_modulazione.png'); plt.show()
''')

md(r'''
## §17.2 — Il cono: dalla direzione del dizionario a quella vera

**Cosa guardare.** Attorno al nodo di partenza (stella), i candidati dei due livelli colorati per SSE (giallo =
errore più basso). Il cerchio rosso è il candidato scelto, la × la direzione vera. A destra l'errore angolare
prima e dopo. La scelta è controllata contro `refine_fiber_direction_cone` del toolbox.
''')
code(r'''
ij = voxel_di('WM sana', 5)
fig, INFO = fig_cono(K, ij, vera=dir_vere(ij)[0], file=FIG / '11_cono.png'); plt.show()
print({k: round(float(v), 3) for k, v in INFO.items()})
''')

md(r'''
## §18 — Il crossing: la FF ri-stimata e i due tensori (MRDS)

**La FF del crossing.** In Stage A la penalità anisotropa schiaccia la FF dei crossing. Prima di stimare i
tensori, la FF viene ri-stimata con una NNLS sul solo "supporto" (i nodi vicini alle due direzioni rilevate, più
la griglia isotropa) con una ridge leggera κ(σ/S0)². A sinistra: la FF al variare di κ (κ = 0 tiene quella di
Stage A); la banda è su tutti i crossing del fantoccio, la linea nera su un voxel solo.

**I due tensori.** Con direzioni e frazioni fissate, si cercano (D∥, D⊥) delle due popolazioni: 3 iterazioni
alternate (una popolazione alla volta, per rompere la simmetria) e poi Levenberg–Marquardt. Al centro il costo
che scende e si ferma; a destra le quattro diffusività iterazione per iterazione (punteggiate quelle vere). La
traiettoria finisce esattamente sui tensori che il fit del toolbox ha scritto (controllato). Qui il fit ha il
cono dei crossing spento (come nella 1.8): dalla 1.9.0 dopo questo passo le direzioni lasciano la griglia (§18b).
''')
code(r'''
cand = [tuple(c) for c in np.argwhere((LAB == NOMI_SC.index('Crossing bil.')) & interno(LAB))]
err = [abs(K['res'][c][IX['fiber_fraction']] - TRU['FF'][c]) if K['res'][c][IX['n_fiber_populations']] == 2 else np.inf for c in cand]
ok = [c for c, e in zip(cand, err) if np.isfinite(e)]
ij = ok[int(np.argsort([e for e in err if np.isfinite(e)])[len(ok) // 2])]     # il voxel con l'errore mediano
fig, TRAI, INFO18 = fig_crossing(K, ij, verita=dict(FF=TRU['FF'][ij], AD=[1.6e-3, 1.6e-3], RD=[0.4e-3, 0.4e-3]),
                                  altri=ok, file=FIG / '12_crossing.png'); plt.show()
fin = TRAI.iloc[-1]
assert np.allclose([fin.AD1, fin.AD2, fin.RD1, fin.RD2], INFO18['AD_finale'] + INFO18['RD_finale'], rtol=1e-4), \
    'traiettoria MRDS diversa dai tensori del toolbox'
print({k: (np.round(v, 5) if not isinstance(v, list) else np.round(v, 5).tolist()) for k, v in INFO18.items()})
''')

md(r'''
## §18b — Il cono per popolazione nei crossing (opzione `crossing_dir_refine`)

**Il problema.** Nei voxel a una fibra Stage C toglie la direzione dalla griglia del dizionario con il cono
(§17.2). Nei crossing no: le due direzioni restano sui nodi di Stage A, a qualche grado da quelle vere (la
spaziatura della griglia), e l'MRDS compensa l'errore di direzione con i tensori.

**L'opzione.** Dopo l'MRDS, una popolazione alla volta: candidati su due coni attorno alla direzione attuale
(gli stessi angoli del cono di Stage C); per ogni candidato si rifanno i due tensori con qualche passo di LM a
caldo, e si tiene il candidato che abbassa l'errore del modello a due popolazioni. Due giri, perché raffinare la
prima popolazione mentre la seconda è ancora sul suo nodo lascia una parte dell'errore. Un candidato che si
avvicina all'altra popolazione più della separazione minima è scartato.

**Cosa guardare.** Un crossing bilanciato (quello con l'errore mediano della griglia): a sinistra e al centro
le due popolazioni attorno alla direzione vera (×): stella = nodo di Stage A, cerchio = dopo il cono. A destra
l'errore mediano sui crossing della batteria e su 400 crossing orientati a caso. Le direzioni e i tensori ricalcolati sono **controllati** contro
quelli scritti dal fit con l'opzione accesa.

**Perché qui il cono sembra non servire.** Nella batteria le direzioni dei crossing cadono per costruzione
vicino ai nodi della griglia (errore della griglia ~4°, una popolazione dello sbilanciato a 1°): è già il
pavimento del rumore a questo SNR, e il cono non può scendere sotto. Dove la griglia è casualmente quasi
giusta e la popolazione è debole, il cono può anzi allontanarsi un po' (la stima segue il rumore). Il guadagno
vero si vede su orientamenti **casuali**, quando la direzione vera cade fra i nodi: la seconda tabella (crossing
bilanciati a 60–90° orientati a caso, stesso SNR) è il caso che conta per il tessuto.

Dalla 1.9.0 cono e test 2-contro-1 sono il default. Qui il fit di riferimento K (§14–18) li ha spenti, come nella
1.8: la §18 mostra l'MRDS sulle direzioni della griglia, la 18b accende il solo cono (K2), la 18c anche il test.
''')
code(r'''
HA_CROSSING = 'two_vs_one_threshold' in inspect.signature(DBSI_Adaptive).parameters
if not HA_CROSSING:
    print('toolbox senza le opzioni dei crossing (serve il branch crossing-cono-e-test21): 18b e 18c saltate')
else:
    K2 = intercetta_fit(P3, MB, MASK_B, np.full(MASK_B.shape, SIG_B), lam=(9.8, 0.015),
                        opzioni=dict(crossing_dir_refine=True, two_vs_one_threshold=None))   # stessi crossing di K
    CROSS = [n for n in NOMI_SC if n.startswith('Crossing')]
    E0 = errore_angolare(K['res'], TRU['dirs']); E1 = errore_angolare(K2['res'], TRU['dirs'])
    npk = lambda R, c: R[c][IX['n_fiber_populations']]
    cand = [tuple(c) for c in np.argwhere((LAB == NOMI_SC.index('Crossing bil.')) & interno(LAB))
            if npk(K['res'], tuple(c)) == 2 and npk(K2['res'], tuple(c)) == 2]
    ij = cand[int(np.argsort([np.nanmax(E0[c]) for c in cand])[len(cand) // 2])]
    C18 = cono_crossing_voxel(K, ij)
    fin = K2['res'][ij]
    d_tb = np.array([[fin[IX[f'dir{p}_{c}']] for c in 'xyz'] for p in (1, 2)])
    assert all(_ang(C18['dirs1'][k], d_tb[k]) < 0.05 for k in (0, 1)), 'cono ricalcolato diverso dal toolbox'
    assert np.allclose(C18['AD1'], [fin[IX['axial_diffusivity_pop1']], fin[IX['axial_diffusivity_pop2']]], rtol=1e-3), \
        'tensori dopo il cono diversi dal toolbox'
    righe = []
    for sc in CROSS:
        m_ = interno(LAB) & (LAB == NOMI_SC.index(sc))
        for br, E in (('griglia', E0), ('cono', E1)):
            for q in (1, 2):
                righe.append(dict(scenario=sc, popolazione=q, braccio=br, mediana=float(np.nanmedian(E[..., q - 1][m_]))))
    rc = []
    for br, opz in (('griglia', None), ('cono', dict(crossing_dir_refine=True))):
        e = crossing_casuali(P3, 400, SNR_FANTOCCIO, (9.8, 0.015), seed=7, opzioni=opz)
        for q in (0, 1):
            rc.append(dict(scenario='Casuali (bil.)', braccio=br, popolazione=q + 1, mediana=np.nanmedian(e[:, q]),
                           p90=np.nanpercentile(e[:, q], 90), visti_pct=100 * np.mean(np.isfinite(e[:, q]))))
    CAS18B = pd.DataFrame(rc)
    TAB18B = pd.concat([pd.DataFrame(righe), CAS18B[['scenario', 'popolazione', 'braccio', 'mediana']]], ignore_index=True)
    fig_cono_crossing(K, C18, dir_vere(ij), TAB18B, file=FIG / '12b_cono_crossing.png'); plt.show()
    display(TAB18B.pivot_table(index=['scenario', 'popolazione'], columns='braccio', values='mediana', sort=False).round(1))
    print(f'crossing bilanciati a orientamento casuale (400, SNR {SNR_FANTOCCIO}), errore angolare in gradi:')
    display(CAS18B.drop(columns='scenario').set_index(['popolazione', 'braccio']).round(1))
''')

md(r'''
## §18c — Quante popolazioni: il test 2-contro-1 (opzione `two_vs_one_threshold`)

**Il problema.** Fino alla 1.8 il numero di popolazioni lo decideva Stage A: c'è un crossing se il secondo bacino
pesa almeno il 35% del primo (`min_peak_ratio`). È una soglia scelta a mano su distribuzioni che si
sovrappongono: un crossing sbilanciato vero (0.20 contro 0.45, rapporto 0.44) col rumore finisce sotto soglia
metà delle volte, e una fibra sola con un po' di rumore a volte la supera.

**Il test.** Ogni voxel con un secondo picco qualsiasi è un candidato: il fit stima **entrambi** i modelli, a una
e a due popolazioni (stesso Stage A), e li confronta sul modello finale di Stage D, prima della soglia di fibra:
$$\mathrm{stat21} = \frac{\mathrm{RSS}_{\text{1 pop}} - \mathrm{RSS}_{\text{2 pop}}}{\sigma_{\text{voxel}}^2}.$$
Se il voxel ha una fibra sola, il modello a due popolazioni spiega in più solo rumore: stat21 è piccola (circa
una χ² con pochi gradi di libertà, perché la seconda popolazione aggiunge una manciata di parametri). Con due
fibre vere la differenza cresce con l'SNR e con la seconda fibra. Si tiene il crossing se stat21 ≥ T (T = 12).
Il cono di 18b è acceso insieme: senza, il modello a due popolazioni parte svantaggiato (direzioni sulla
griglia) e il nullo diventa negativo.

**Cosa guardare.** A sinistra la statistica dei candidati con una e con due fibre vere. Al centro e a destra,
a tre SNR, la quota di voxel con due popolazioni stimate: deve **scendere** dove la fibra vera è una (falsi
crossing) e **salire** dove sono due (crossing trovati). La cella controlla che il toolbox abbia tenuto come
crossing esattamente i candidati con stat21 ≥ T.
''')
code(r'''
if HA_CROSSING:
    T21 = 12.0
    VERO = np.nan_to_num(TRU['npop']); INN_B = interno(LAB) & MASK_B
    rs, rn = [], []
    for snr in (15, 26, 40):
        if snr == SNR_FANTOCCIO:
            Mn, sg, Kd = MB, SIG_B, K
        else:
            Mn, sg = rumore_rician(A_b, snr, seed=1)
            Kd = intercetta_fit(P3, Mn, MASK_B, np.full(MASK_B.shape, sg), lam=(9.8, 0.015), opzioni=OPZ_18)
        Kt = intercetta_fit(P3, Mn, MASK_B, np.full(MASK_B.shape, sg), lam=(9.8, 0.015),
                            opzioni=dict(crossing_dir_refine=True, two_vs_one_threshold=T21))
        st = Kt['model'].two_vs_one_stat_
        npt = Kt['res'][..., IX['n_fiber_populations']]
        ok = np.isfinite(st) & np.isfinite(npt)
        assert np.array_equal(npt[ok] >= 2, st[ok] >= T21), 'decisione del toolbox diversa da stat21 >= T'
        for v in (1, 2):
            sel = INN_B & np.isfinite(st) & (VERO == v)
            rs += [dict(SNR=snr, vero=v, stat21=float(x)) for x in st[sel]]
            for nome, R in (('1.8', Kd['res']), ('test', Kt['res'])):
                n_ = np.nan_to_num(R[..., IX['n_fiber_populations']])[INN_B & (VERO == v)]
                rn.append(dict(SNR=snr, vero=v, fit=nome, due_pct=100 * float(np.mean(n_ >= 2))))
    STAT21, NPOP21 = pd.DataFrame(rs), pd.DataFrame(rn)
    fig_test21(STAT21, NPOP21, T21, SNR_FANTOCCIO, file=FIG / '12c_test_2contro1.png'); plt.show()
    display(NPOP21.pivot_table(index=['vero', 'fit'], columns='SNR', values='due_pct').round(1))
    display(STAT21.groupby(['SNR', 'vero']).stat21.describe(percentiles=[0.05, 0.5, 0.95]).round(1))
''')

md(r'''
## §23 — Le mappe di incertezza, spiegate semplici

**Due parole: precisione e bias.** Immagina di misurare lo stesso voxel 300 volte, ognuna con un rumore
diverso. Le 300 stime formano una nuvola.
- **Quanto è larga la nuvola** è la *precisione*. L'errore standard (SE) che il toolbox scrive per ogni voxel
  stima proprio questa larghezza, **da una sola misura**: guarda quanto il segnale cambierebbe se il parametro
  cambiasse (se cambia molto, il parametro è ben determinato e la SE è piccola; se cambia poco, è mal
  determinato e la SE è grande) e la confronta con il rumore.
- **Dove sta il centro della nuvola rispetto alla verità** è il *bias*. La SE **non lo vede**: se lo stimatore
  sbaglia sempre nella stessa direzione, la nuvola è stretta ma spostata, e la mappa dice "preciso" anche
  quando è sistematicamente sbagliato.

**Cosa guardare.** Nella prima figura, la fibra sana: nuvola centrata sulla stella, ellisse (±1.96 SE) che la
contiene. Il crossing: la nuvola può essere spostata (freccia = bias). Nella seconda, la SE al variare di SNR e
protocollo: **la SE misura quello che il protocollo permette di sapere**. Per confrontare i protocolli dei 5P è
la parte che serve: a parità di tessuto, un protocollo con SE più grande ha mappe più rumorose.
''')
code(r'''
LAM = (R0['lambda_aniso'], R0['lambda_iso']) if FONTE.startswith('fantoccio') else (9.8, 0.015)
u = direzione(90, 20); v = direzione(90, 110)
I0 = dict(R=0.10, H=0.20, W=0.10)
casi = [('fibra sana', [(0.6, u, 1.7e-3, 0.4e-3)], 0.6, 0.4e-3),
        ('fibra demielinizzata', [(0.6, u, 1.5e-3, 0.8e-3)], 0.6, 0.8e-3),
        ('crossing 90°', [(0.3, u, 1.7e-3, 0.4e-3), (0.3, v, 1.7e-3, 0.4e-3)], 0.6, 0.4e-3)]
stime, ses, veri, tit = [], [], [], []
for nome, fibre, ff, rd in casi:
    res, se, fl, rr = fit_voxel_ripetuti(P3, fibre, I0, 300, SNR_FANTOCCIO, LAM, seed=11)
    stime.append(np.column_stack([res[:, IX['fiber_fraction']], res[:, IX['radial_diffusivity_weighted']] * 1e3]))
    ses.append((np.nanmedian(se[:, IX['fiber_fraction']]), np.nanmedian(se[:, IX['radial_diffusivity_weighted']]) * 1e3))
    veri.append((ff / (ff + 0.4), rd * 1e3)); tit.append(nome)
fig_bersaglio(stime, ses, veri, 'FF', 'RD pesata [e-3 mm²/s]', tit, file=FIG / '13_incertezza_bersaglio.png'); plt.show()
''')
code(r'''
righe = []
protos = {'P3': P3}
_root5p = Path(REALE['prep']).parents[3]          # .../DATA_5P/Processed
for P in (1, 2, 4, 5):
    base = _root5p / f'{P:06d}' / '20251210' / 'DWI_4s' / 'prep'
    fb, fv = base / f"{REALE['stem']}_corrected.bval", base / f"{REALE['stem']}_corrected.bvec"
    if fb.is_file() and fv.is_file():
        protos[f'P{P}'] = carica_protocollo(fb, fv)
print('protocolli:', list(protos))
for snr in (15, 26, 40, 60):
    for p, prot in protos.items():
        res, se, fl, rr = fit_voxel_ripetuti(prot, casi[0][1], I0, 100, snr, LAM, seed=3)
        for k, c, s in (('FF', 'fiber_fraction', 1), ('RD pesata [e-3]', 'radial_diffusivity_weighted', 1e3),
                        ('FA pesata', 'fiber_fa_weighted', 1)):
            righe.append(dict(protocollo=p, SNR=snr, metrica=k, se=float(np.nanmedian(se[:, IX[c]])) * s))
SEP = pd.DataFrame(righe)
fig_se_protocolli(SEP, file=FIG / '14_incertezza_snr.png'); plt.show()
''')

md(r'''
**Il residuo e i flag.** Per ogni voxel il toolbox calcola anche `residual_over_sigma`: quanto il segnale
**non spiegato** dal modello è grande rispetto al rumore. Se modello e σ sono giusti vale ~1. Se è molto più
grande, il modello non descrive il voxel (o la mappa di rumore lo sottostima): il flag 16 si accende e la SE è
ottimista, perché presuppone un modello giusto. Qui: voxel che il modello descrive (diffusività isotrope uguali
ai centroidi) contro voxel con compartimenti che il modello non ha (diffusività isotrope lontane dai centroidi
e una terza fibra).

I flag in breve: **1** = la RD è sul limite inferiore (il dato non la determina, la SE descrive il modello senza
limite); **2** = la AD è sul limite; **4** = il modello è quasi non identificabile in quel voxel (la SE è poco
affidabile); **16** = il residuo è più grande del rumore (modello insufficiente o σ sottostimata).
''')
code(r'''
res_ok, se_ok, fl_ok, rr_ok = fit_voxel_ripetuti(P3, casi[0][1], I0, 300, SNR_FANTOCCIO, LAM, seed=21)
w = direzione(20, 60)
fibre_male = [(0.4, u, 1.7e-3, 0.4e-3), (0.2, v, 1.7e-3, 0.4e-3), (0.2, w, 1.7e-3, 0.4e-3)]
res_m, se_m, fl_m, rr_m = fit_voxel_ripetuti(P3, fibre_male, dict(R=0.05, H=0.10, W=0.05), 300, SNR_FANTOCCIO, LAM,
                                             seed=22, D=dict(R=0.05e-3, H=0.55e-3, W=2.2e-3))
dof = len(P3[0]) - 8
fig_residuo(rr_ok, rr_m, soglia_residuo(dof), file=FIG / '15_residuo_e_flag.png'); plt.show()
for nome, fl in (('modello giusto', fl_ok), ('modello insufficiente', fl_m)):
    print(f"{nome:22s} flag 1 {np.mean(fl & 1 > 0):.0%}  flag 2 {np.mean(fl & 2 > 0):.0%}  "
          f"flag 4 {np.mean(fl & 4 > 0):.0%}  flag 16 {np.mean(fl & 16 > 0):.0%}")
''')

md('## Elenco delle figure')
code(r'''
for f in sorted(FIG.glob('*.png')):
    print(f.name)
''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
for c in nb['cells']:
    L = c['source'].split('\n')
    c['source'] = [l + '\n' for l in L[:-1]] + [L[-1]]
out = (QUI.parent if QUI.name == 'sorgenti' else QUI) / 'matematica_pyDBSI.ipynb'
json.dump(nb, open(out, 'w'), indent=1, ensure_ascii=False)
print('scritto', out, len(cells), 'celle')
