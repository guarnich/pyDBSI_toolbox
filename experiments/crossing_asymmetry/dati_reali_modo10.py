"""Modo 10 sui dati reali: produzione contro FF dei crossing ri-stimata, stesso soggetto.

Senza verita', il criterio e' la COERENZA INTERNA in un soggetto sano: nella sostanza bianca i
crossing non hanno ragione di avere una RD molto piu' bassa delle fibre singole dello stesso
soggetto. In produzione ~49% dei crossing sta sul pavimento della RD e la loro RD pesata e' molto
sotto quella delle mono-fibra; se il modo 10 fa sui dati quel che fa sui fantocci, il pavimento
crolla e le due RD si avvicinano, senza toccare le mono-fibra.

    python dati_reali_modo10.py DWI.nii.gz BVAL BVEC MASK.nii.gz OUT_DIR [CALIBRAZIONE.json] [z0:z1]

Con la calibrazione di protocollo i lambda sono quelli fissati (come in produzione); senza, il fit
calibra sui dati. z0:z1 limita le fette assiali (es. 30:50) per fare prima. Scrive un CSV di
confronto e, per ciascun modo, le mappe con save_output_maps in OUT_DIR/modo0 e OUT_DIR/modo10.
"""
import sys, os, numpy as np, pandas as pd
from dbsi_toolbox import DBSI_Adaptive, load_data, save_output_maps
from dbsi_toolbox.core.solvers import _TENSOR_RD_FLOOR

dwi, bval, bvec, maskf, out = sys.argv[1:6]
cal = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].endswith('.json') else None
zr = next((a for a in sys.argv[6:] if ':' in a), None)
data, affine, bvals, bvecs, mask = load_data(dwi, bval, bvec, maskf, verbose=True)
if zr:
    z0, z1 = map(int, zr.split(':'))
    keep = np.zeros_like(mask, bool); keep[:, :, z0:z1] = True; mask = mask & keep
NM = None; righe = []
for mode in (0, 10):
    m = DBSI_Adaptive.from_calibration(cal) if cal else DBSI_Adaptive()
    res, mm = m.fit(data, bvals, bvecs, mask, run_calibration=cal is None, _mrds_mode=mode)
    NM = DBSI_Adaptive.output_map_names(mm); IX = {n: i for i, n in enumerate(NM)}
    save_output_maps(res, NM, affine, os.path.join(out, f'modo{mode}'), model=m)
    npop = res[..., IX['n_fiber_populations']]
    for nome, sel in (('mono-fibra', mask & (npop == 1)), ('crossing', mask & (npop == 2))):
        rdw = res[..., IX['radial_diffusivity_weighted']][sel]; adw = res[..., IX['axial_diffusivity_weighted']][sel]
        faw = res[..., IX['fiber_fa_weighted']][sel]; ff = res[..., IX['fiber_fraction']][sel]
        rds = [res[..., IX['radial_diffusivity_pop1']][sel]]
        if nome == 'crossing':
            rds.append(res[..., IX['radial_diffusivity_pop2']][sel])
        rds = np.concatenate(rds); rds = rds[np.isfinite(rds)]
        righe.append(dict(modo=mode, gruppo=nome, n_voxel=int(sel.sum()),
                          quota_mask=float(sel.sum() / mask.sum()),
                          RDw_mediana=float(np.nanmedian(rdw)) * 1e3, ADw_mediana=float(np.nanmedian(adw)) * 1e3,
                          FAw_mediana=float(np.nanmedian(faw)), FF_mediana=float(np.nanmedian(ff)),
                          RD_sul_pavimento=float(np.mean(rds <= _TENSOR_RD_FLOOR * 1.0001)) if rds.size else np.nan))
T = pd.DataFrame(righe)
pd.set_option('display.width', 200)
print('\n(diffusivita\' in 1e-3 mm^2/s)'); print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(out, 'confronto_modo0_modo10.csv'), index=False)
r = T.set_index(['modo', 'gruppo'])
for mode in (0, 10):
    print(f"modo {mode}: RD pesata crossing / mono-fibra = "
          f"{r.loc[(mode, 'crossing'), 'RDw_mediana'] / r.loc[(mode, 'mono-fibra'), 'RDw_mediana']:.2f}, "
          f"pavimento crossing {r.loc[(mode, 'crossing'), 'RD_sul_pavimento']:.1%}")
