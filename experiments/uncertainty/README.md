# Mappe di incertezza: copertura (2026-09-30)

`copertura.py [SNR] [NV] [SOGLIA]` — per 7 scenari, NV voxel con la stessa verità e rumore Rician
indipendente (protocollo P3, λ congelati del P3, gate spento). Confronta l'errore standard previsto
(`toolbox_report/uncertainty_maps/`) con la dispersione empirica della stima.

Esito a SNR 26 (`esito_copertura_snr26.csv`, `esito_copertura_snr26_ril15.csv`):

| regime | SE / sd empirica | copertura 95% | lettura |
|---|---|---|---|
| mono-fibra (sana, edema, cellulare) | 1.0–1.35 | 94–100% | calibrata, lievemente conservativa (lo stimatore vincolato ha meno varianza del limite) |
| mono-fibra demielinizzata | 1.2–2.0 | 84–100% | SE grande (FF ± 0.40): è la degenerazione fibra/hindered; la copertura cala per il bias (FF −0.085) |
| crossing 90 / 60 | 1.4–9 | ~100% | la stima dei crossing ha poca varianza e molto bias (FF di Stage A fissa): la SE dice quanto poco i dati determinano il valore, e la verità ci cade dentro |
| isotropo, soglia di rilevamento 15 | 0.95–0.97 | 93% | calibrata |
| isotropo, test spento | 0.86–0.91 | 88–95% | solo i 103/300 voxel senza fibre false: effetto di selezione |

La SE è il limite di Cramér-Rao alla stima: non contiene il bias. Dove lo stimatore è corretto
descrive la dispersione vera; dove è distorto (crossing) è la misura onesta di quanto il dato
sostiene, e la precisione apparente della stima è falsa precisione.

**v1.7.0** (`esito_copertura_snr26_v170.csv`), con la FF dei crossing ri-stimata prima dell'LM: nei
crossing a 90 gradi bias FF −0.146 → +0.027, bias RD pesata −0.179 → +0.024, RD sul limite 70% → 5%;
la dispersione empirica della FF sale da 0.037 a 0.117 (la falsa precisione sparisce) e la SE resta
2–3 volte la dispersione (stimatore ancora regolarizzato), copertura 100%. Mono-fibra invariate.
