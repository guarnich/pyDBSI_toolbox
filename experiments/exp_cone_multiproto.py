"""stagec_dir_refine: cono->C vs C da solo, su 3 protocolli x 3 SNR.

Estende l'esperimento a una condizione sola (P3, SNR 30, AD fisso) variando
le due cose che dovrebbero contare: la spaziatura angolare del dizionario
(via n_dirs, cioe' via protocollo) e l'SNR. AD e RD sono ora spazzati per
voxel, non fissi.
"""
import sys, os, io, contextlib, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbsi_toolbox import DBSI_Adaptive

N_VOX = 200
SNRS = (20.0, 30.0, 50.0)
# shell (b, n_dir) modellate sui protocolli reali dello studio 5P
PROTOCOLS = {
    "P4-like (grezzo)": [(0,6),(1000,12),(2000,28)],
    "P3-like (medio)":  [(0,9),(300,3),(700,12),(1000,20),(2000,48)],
    "P1-like (fine)":   [(0,12),(500,6),(1000,30),(2000,60)],
}

def build_protocol(shells, rng):
    bv, bc = [], []
    for b, nd in shells:
        for _ in range(nd):
            bv.append(float(b))
            if b == 0: bc.append([0.,0.,0.])
            else:
                v = rng.normal(size=3); v /= np.linalg.norm(v); bc.append(v)
    return np.array(bv), np.array(bc)

def build_volume(bvals, bvecs, snr, rng):
    v = rng.normal(size=(N_VOX,3)); v /= np.linalg.norm(v,axis=1,keepdims=True)
    v[v[:,2] < 0] *= -1
    ad = rng.uniform(1.30e-3, 2.00e-3, N_VOX)          # ora spazzato
    rd = rng.uniform(0.20e-3, 0.70e-3, N_VOX)
    rd = np.minimum(rd, ad/2.2)                         # resta anisotropo
    ff, rf, hf, wf = 0.55, 0.10, 0.20, 0.15
    side = int(np.ceil(N_VOX**(1/3)))+1
    idx = [(i,j,k) for i in range(side) for j in range(side) for k in range(side)][:N_VOX]
    data = np.zeros((side,)*3+(len(bvals),), np.float32)
    mask = np.zeros((side,)*3, bool)
    for n,(i,j,k) in enumerate(idx):
        c = bvecs @ v[n]
        sig = (ff*np.exp(-bvals*(rd[n]+(ad[n]-rd[n])*c**2)) + rf*np.exp(-bvals*0.15e-3)
               + hf*np.exp(-bvals*1.0e-3) + wf*np.exp(-bvals*3.0e-3))
        noisy = np.sqrt((sig+rng.normal(0,1/snr,len(bvals)))**2 + rng.normal(0,1/snr,len(bvals))**2)
        data[i,j,k] = 1000.0*noisy; mask[i,j,k] = True
    return data, mask, idx, dict(dir=v, ad=ad, rd=rd, ff=np.full(N_VOX,ff), rf=np.full(N_VOX,rf))

def fit(data, mask, idx, bvals, bvecs, gt, dir_refine):
    m = DBSI_Adaptive(stagec_dir_refine=dir_refine)
    with contextlib.redirect_stdout(io.StringIO()):
        res, mode = m.fit(data, bvals, bvecs, mask, run_calibration=True,
                          n_calibration_voxels=200, correct_restricted_fraction=False)
    names = DBSI_Adaptive.output_map_names(mode); I={n_:i for i,n_ in enumerate(names)}
    g = lambda nm: np.array([res[i,j,k,I[nm]] for (i,j,k) in idx])
    d = np.stack([g('dir1_x'),g('dir1_y'),g('dir1_z')],1)
    ang = np.degrees(np.arccos(np.abs(np.sum(d*gt['dir'],1)).clip(0,1)))
    return dict(npop=g('n_fiber_populations'), ang=ang, ad=g('axial_diffusivity_pop1'),
                rd=g('radial_diffusivity_pop1'), ff=g('fiber_fraction'),
                rf=g('restricted_fraction'), n_dirs=m.n_dirs, mode=mode)

rows = []
print(f"{N_VOX} voxel/condizione, AD 1.30-2.00e-3, RD 0.20-0.70e-3, "
      f"{len(PROTOCOLS)}x{len(SNRS)} condizioni\n")
for pname, shells in PROTOCOLS.items():
    for snr in SNRS:
        rng = np.random.default_rng(hash((pname,snr)) % 2**31)
        bvals, bvecs = build_protocol(shells, rng)
        data, mask, idx, gt = build_volume(bvals, bvecs, snr, rng)
        t0 = time.time()
        A = fit(data, mask, idx, bvals, bvecs, gt, False)
        B = fit(data, mask, idx, bvals, bvecs, gt, True)
        m_ = (A['npop']==1)&(B['npop']==1)&np.isfinite(A['ad'])&np.isfinite(B['ad'])
        r = dict(proto=pname, snr=snr, n_dirs=int(A['n_dirs']), mode=int(A['mode']),
                 n=int(m_.sum()), secs=round(time.time()-t0))
        for k in ('ang','ad','rd','ff','rf'):
            if k == 'ang':
                r['ang_A'], r['ang_B'] = A['ang'][m_].mean(), B['ang'][m_].mean()
            else:
                den = gt[k][m_] if k in ('ad','rd') else 1.0
                sc  = 100.0 if k in ('ad','rd') else 1.0
                r[k+'_A'] = (np.abs(A[k][m_]-gt[k][m_])/den).mean()*sc
                r[k+'_B'] = (np.abs(B[k][m_]-gt[k][m_])/den).mean()*sc
        rows.append(r)
        print(f"  {pname:<18} SNR {snr:>4.0f}  n_dirs={r['n_dirs']:<4} n={r['n']:<4} "
              f"ang {r['ang_A']:5.2f}->{r['ang_B']:4.2f}°  AD {r['ad_A']:5.2f}->{r['ad_B']:5.2f}%  "
              f"({r['secs']}s)", flush=True)
json.dump(rows, open('/tmp/exp_multiproto.json','w'), indent=1)

print(f"\n{'protocollo':<18}{'SNR':>5}{'n_dirs':>8}{'ang A':>8}{'ang B':>8}"
      f"{'AD A':>8}{'AD B':>8}{'ΔAD':>8}{'ΔRD':>8}{'ΔFF':>8}")
print("-"*90)
for r in rows:
    dad=(r['ad_B']-r['ad_A'])/r['ad_A']*100; drd=(r['rd_B']-r['rd_A'])/r['rd_A']*100
    dff=(r['ff_B']-r['ff_A'])/r['ff_A']*100
    print(f"{r['proto']:<18}{r['snr']:>5.0f}{r['n_dirs']:>8}{r['ang_A']:>7.2f}°{r['ang_B']:>7.2f}°"
          f"{r['ad_A']:>7.2f}%{r['ad_B']:>7.2f}%{dad:>7.1f}%{drd:>7.1f}%{dff:>7.1f}%")
w = sum(1 for r in rows if r['ad_B'] < r['ad_A'])
print(f"\ncono->C migliora AD in {w}/{len(rows)} condizioni")
