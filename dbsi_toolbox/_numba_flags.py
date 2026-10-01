"""
Flag di fastmath per TUTTI i kernel numba del toolbox (v1.6.4).

`fastmath=True` accende anche 'nnan' e 'ninf': il compilatore puo' assumere che NaN e infiniti
non esistano, e quindi eliminare `np.isnan` / `np.isfinite` e riscrivere i confronti con NaN.
Misurato il 2026-10-01: con fastmath=True `np.isnan(NaN)` compilato restituisce False anche su
questo Mac (arm64); `NaN < 1` vale True su arm64 e False sulla workstation (x86), cosi' lo stesso
codice salta i voxel senza fibra su un'architettura e non sull'altra. Sulla workstation il test
di rilevamento faceva entrare i 740 voxel senza fibra con parametri NaN, e il controllo di zero
della divisione nella NNLS scambiava NaN per zero (ZeroDivisionError, notebook 09). Lo stesso
difetto era gia' stato trovato in `fit_quality` (che per questo non usa fastmath).

Qui si tengono le ottimizzazioni che danno la velocita' (riassociazione, contrazione FMA,
reciproco approssimato, segno dello zero, funzioni approssimate) e si tolgono le due promesse
false: il toolbox usa NaN come sentinella ovunque (popolazioni assenti, tensori non stimati).
"""
FASTMATH = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}   # set: numba non accetta frozenset
