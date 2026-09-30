"""Punto 12, AD locale (opzione 3c, _mrds_mode=9) su un fantoccio SPAZIALE.

Lastra 40x40x3, protocollo P3. Tratto A lungo x (righe y 10-29), tratto B lungo y (colonne
x 10-29): incrocio a 90 gradi nel quadrato centrale, segmenti a fibra singola lungo i tratti,
sostanza grigia isotropa fuori. Lesione circolare (centro x=12, y=20, raggio 7) sul tratto A:
copre sia un pezzo del segmento mono-fibra di A sia un pezzo dell'incrocio.
Varianti: 0 produzione, 7 AD globale, 9 AD locale per popolazione, 9o oracolo (AD vera per fascio),
10 FF dei crossing ri-stimata sul supporto prima dell LM (+ Stage D sui crossing)."""
import io, sys, contextlib, numpy as np, pandas as pd
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 26.0
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel(); bvecs = np.loadtxt(HERE / 'p3_like.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
NM = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(NM)}
X, Y, Z = 40, 40, 3
SANO = (1.7e-3, 0.4e-3); DEM = (1.5e-3, 0.8e-3); AXO = (1.1e-3, 0.4e-3)
COND = {'sano': dict(A=SANO, B=SANO, lesB=False, lesione_iso=False),
        'demiel A': dict(A=DEM, B=SANO, lesB=False, lesione_iso=False),
        'assonale A': dict(A=AXO, B=SANO, lesB=False, lesione_iso=False),
        'lesione A+B': dict(A=DEM, B=DEM, lesB=True, lesione_iso=True)}
xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
inA = (yy >= 10) & (yy < 30); inB = (xx >= 10) & (xx < 30)
les = (xx - 12) ** 2 + (yy - 20) ** 2 <= 7 ** 2
REG = {'cross in lesione': inA & inB & les, 'cross fuori': inA & inB & ~les,
       'mono A in lesione': inA & ~inB & les, 'mono A fuori': inA & ~inB & ~les}
eA, eB = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

def jitter(d, rng, deg=5):
    t = np.cross(d, rng.normal(size=3)); t /= np.linalg.norm(t); a = np.radians(rng.normal(0, deg))
    return np.cos(a) * d + np.sin(a) * t

def costruisci(cond, rng):
    c = COND[cond]; data = np.zeros((X, Y, Z, len(bvals)), np.float32)
    truth = np.zeros((X, Y, Z, 4)) * np.nan      # AD_A RD_A AD_B RD_B
    fib = lambda d, t: np.exp(-bvals * (t[1] + (t[0] - t[1]) * (bvecs @ d) ** 2))
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                l = les[i, j]
                rf, hf, wf = (0.15, 0.25, 0.25) if (l and c['lesione_iso'] and (inA[i, j] or inB[i, j])) else (0.10, 0.20, 0.10)
                iso = (rf * np.exp(-bvals * rng.uniform(0.10, 0.25) * 1e-3) + hf * np.exp(-bvals * rng.uniform(0.7, 1.3) * 1e-3)
                       + wf * np.exp(-bvals * rng.uniform(2.6, 3.2) * 1e-3))
                tA = c['A'] if l else SANO; tB = c['B'] if (l and c['lesB']) else SANO
                ff = 0.35 if (l and c['lesione_iso'] and (inA[i, j] or inB[i, j])) else 0.6
                if inA[i, j] and inB[i, j]:
                    S = iso / (rf + hf + wf) * (1 - ff) + ff / 2 * (fib(jitter(eA, rng), tA) + fib(jitter(eB, rng), tB))
                    truth[i, j, k] = (*tA, *tB)
                elif inA[i, j]:
                    S = iso / (rf + hf + wf) * (1 - ff * 0.92) + ff * 0.92 * fib(jitter(eA, rng), tA); truth[i, j, k, :2] = tA
                elif inB[i, j]:
                    S = iso / (rf + hf + wf) * (1 - ff * 0.92) + ff * 0.92 * fib(jitter(eB, rng), tB); truth[i, j, k, 2:] = tB
                else:
                    S = 0.03 * np.exp(-bvals * 0.10e-3) + 0.87 * np.exp(-bvals * 0.88e-3) + 0.10 * np.exp(-bvals * 3.05e-3)
                sd = 1 / SNR
                data[i, j, k] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
    return data, truth

def fit(data, mode, admap=None):
    m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = m.fit(data, bvals, bvecs, np.ones(data.shape[:3], bool), run_calibration=False,
                       _mrds_mode=mode, _crossing_ad_map=admap)
    return res, m

def abbina(res):
    """Per ogni voxel crossing: indice (0/1) della popolazione che corrisponde al tratto A."""
    d1 = np.abs(res[..., IX['dir1_x']]); d2 = np.abs(res[..., IX['dir2_x']])
    return np.where(d1 >= d2, 0, 1)

righe = []
for cond in COND:
    rng = np.random.default_rng(100)
    data, truth = costruisci(cond, rng)
    R = {}
    R['0 produzione'], _ = fit(data, 0)
    R['7 AD globale'], _ = fit(data, 7)
    R['9 AD locale'], m9 = fit(data, 9)
    R['10 FF ristimata'], _ = fit(data, 10)
    # oracolo: AD vera per fascio, nell'ordine delle popolazioni di Stage A (identico fra i passaggi)
    r0 = R['0 produzione']; ia = abbina(r0)
    orc = np.zeros(data.shape[:3] + (2,))
    orc[..., 0] = np.where(ia == 0, truth[..., 0], truth[..., 2]); orc[..., 1] = np.where(ia == 0, truth[..., 2], truth[..., 0])
    orc = np.nan_to_num(orc, nan=1.7e-3)
    R['9o oracolo AD'], _ = fit(data, 9, orc)
    loc = m9.crossing_ad_map_
    for nome, res in R.items():
        ia = abbina(res); cross = res[..., IX['n_fiber_populations']] == 2
        mono1 = res[..., IX['n_fiber_populations']] == 1
        rd1, rd2 = res[..., IX['radial_diffusivity_pop1']], res[..., IX['radial_diffusivity_pop2']]
        ad1, ad2 = res[..., IX['axial_diffusivity_pop1']], res[..., IX['axial_diffusivity_pop2']]
        RDA = np.where(ia == 0, rd1, rd2); RDB = np.where(ia == 0, rd2, rd1)
        ADA = np.where(ia == 0, ad1, ad2); ADB = np.where(ia == 0, ad2, ad1)
        for reg, rm in REG.items():
            rmask = np.repeat(rm[:, :, None], Z, axis=2)
            if reg.startswith('cross'):
                s = rmask & cross
                r = dict(condizione=cond, variante=nome, regione=reg, n=int(s.sum()),
                         RD_A=np.nanmean(RDA[s]) * 1e3, RD_A_vera=np.nanmean(truth[..., 1][s]) * 1e3,
                         RD_B=np.nanmean(RDB[s]) * 1e3, RD_B_vera=np.nanmean(truth[..., 3][s]) * 1e3,
                         AD_A=np.nanmean(ADA[s]) * 1e3, AD_A_vera=np.nanmean(truth[..., 0][s]) * 1e3,
                         RDw=np.nanmean(res[..., IX['radial_diffusivity_weighted']][s]) * 1e3,
                         RDw_vera=np.nanmean((truth[..., 1][s] + truth[..., 3][s]) / 2) * 1e3,
                         pav=float(np.mean(((np.abs(rd1 - _TENSOR_RD_FLOOR) < 1e-9) | (np.abs(rd2 - _TENSOR_RD_FLOOR) < 1e-9))[s])))
            else:
                s = rmask & mono1
                r = dict(condizione=cond, variante=nome, regione=reg, n=int(s.sum()),
                         RD_A=np.nanmean(rd1[s]) * 1e3, RD_A_vera=np.nanmean(truth[..., 1][s]) * 1e3,
                         AD_A=np.nanmean(ad1[s]) * 1e3, AD_A_vera=np.nanmean(truth[..., 0][s]) * 1e3)
            righe.append(r)
    s = np.repeat(REG['cross in lesione'][:, :, None], Z, axis=2) & (R['9 AD locale'][..., IX['n_fiber_populations']] == 2)
    ia = abbina(R['9 AD locale'])
    adA_loc = np.where(ia == 0, loc[..., 0], loc[..., 1])
    print(f'[{cond}] AD locale imposta al fascio A nell incrocio in lesione: {np.mean(adA_loc[s])*1e3:.3f}e-3 '
          f'(vera {np.nanmean(truth[..., 0][s])*1e3:.3f}); AD globale modo 7 = mediana mono', flush=True)
df = pd.DataFrame(righe)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(df.round(3).to_string(index=False))
df.to_csv(HERE / f'esito_fantoccio_spaziale_snr{int(SNR)}_con10.csv', index=False)
