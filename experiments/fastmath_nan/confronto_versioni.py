"""I risultati prodotti sulla WORKSTATION (x86) prima della 1.6.4 sono stati toccati dal difetto di fastmath?

Fino alla 1.6.3 i kernel erano compilati con fastmath=True, che autorizza il compilatore a eliminare i
controlli np.isnan e a riscrivere i confronti con NaN. Su arm64 (Mac) le mappe sono identiche bit per bit
prima e dopo la correzione; su x86 no: il test di rilevamento si fermava (notebook 09). Questo script
misura se ANCHE le mappe di Stage A/B/C cambiano su x86, cioe' se i fit gia' fatti vanno rifatti.

Stage D e il test di rilevamento sono spenti (iso_resolve=False): con la 1.6.3 su x86 il test si ferma,
e cosi' si confrontano le parti che girano in entrambe le versioni.

Uso, sulla workstation. Lo script va COPIATO fuori dal repository prima di tornare alla 1.6.3,
perche' in quella versione non esiste e git lo cancellerebbe:
    cp experiments/fastmath_nan/confronto_versioni.py ~/confronto_versioni.py
    git checkout b77527f            # 1.6.3
    python ~/confronto_versioni.py SESSDIR ~/v163.npz
    git checkout main               # 1.7.0 o successive
    python ~/confronto_versioni.py SESSDIR ~/v17.npz
    python ~/confronto_versioni.py --confronta ~/v163.npz ~/v17.npz

Il confronto e' per NOME di canale (la 1.7.0 ha tolto mean_iso_adc e scalato gli indici) e con
Stage D spento la 1.7.0 non applica la ri-stima della FF dei crossing: le differenze che restano
vengono da fastmath, non dalle novita' della 1.7.0.
"""
import sys, io, contextlib
from pathlib import Path
import numpy as np

if sys.argv[1] == '--confronta':
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    na, nbm = list(a['nomi']), list(b['nomi'])
    comuni = [n for n in na if n in nbm]
    print(f"versioni {a['ver']} contro {b['ver']}, voxel {a['P'].shape[0]}, canali in comune {len(comuni)}"
          + (f" (solo in {a['ver']}: {[n for n in na if n not in nbm]})" if len(comuni) < len(na) else ''))
    tot = 0
    for n in comuni:
        x, y = a['P'][:, na.index(n)], b['P'][:, nbm.index(n)]
        nan = np.isnan(x) != np.isnan(y); both = ~np.isnan(x) & ~np.isnan(y)
        d = np.abs(x[both] - y[both]); nd = int((d > 1e-6 * np.maximum(1, np.abs(x[both]))).sum())
        if nan.any() or nd:
            print(f'  {n:30s} NaN diversi {int(nan.sum()):5d} | valori diversi {nd:5d} | max |diff| {d.max() if d.size else 0:.3g}')
        tot += int(nan.sum()) + nd
    print('NESSUNA differenza: i fit gia\' fatti su questa macchina restano validi' if tot == 0 else
          f'{tot} differenze: le mappe di Stage A/B/C su questa macchina dipendevano dal difetto')
    sys.exit(0)

import dbsi_toolbox
from dbsi_toolbox import DBSI_Adaptive, load_data
sess, out = Path(sys.argv[1]), sys.argv[2]
prep = sess / 'prep'
mk = sorted(prep.glob('*_preprocessed_brain_mask.nii.gz')); assert len(mk) == 1
base = mk[0].name[:-len('_preprocessed_brain_mask.nii.gz')]
data, affine, bvals, bvecs, mask_full = load_data(
    str(prep / f'{base}_preprocessed_N4.nii.gz'), str(prep / f'{base}_corrected.bval'),
    str(prep / f'{base}_corrected.bvec'), mask_path=str(mk[0]), verbose=False)
data = np.asarray(data, np.float32); mask_full = np.asarray(mask_full, bool)
rng = np.random.default_rng(0)                       # gli stessi 4000 voxel del 07/09
coords = np.argwhere(mask_full); sel = rng.choice(len(coords), size=min(4000, len(coords)), replace=False)
mask = np.zeros_like(mask_full)
for x, y, z in coords[sel]:
    mask[x, y, z] = True
m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6, iso_resolve=False)
with contextlib.redirect_stdout(io.StringIO()):
    res, _ = m.fit(data, bvals, bvecs, mask, run_calibration=False)
np.savez_compressed(out, P=res[mask].astype(np.float64), ver=dbsi_toolbox.__version__,
                    nomi=np.array(DBSI_Adaptive.output_map_names(3)))
print('salvato', out, 'versione', dbsi_toolbox.__version__)
