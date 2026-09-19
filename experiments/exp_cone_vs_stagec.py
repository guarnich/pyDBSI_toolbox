"""Cono+C vs C da solo: fibra singola, direzione vera fra i nodi di griglia."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.getcwd())
from dbsi_toolbox import DBSI_Adaptive

SNR = 30.0
N_VOX = 240
rng = np.random.default_rng(20260919)

# protocollo tipo P3: b = 0/300/700/1000/2000 con 9/3/12/20/48 direzioni
shells = [(0, 9), (300, 3), (700, 12), (1000, 20), (2000, 48)]
bvals, bvecs = [], []
for b, nd in shells:
    for _ in range(nd):
        bvals.append(float(b))
        if b == 0:
            bvecs.append([0., 0., 0.])
        else:
            v = rng.normal(size=3); v /= np.linalg.norm(v); bvecs.append(v)
bvals = np.array(bvals); bvecs = np.array(bvecs)
print(f"protocollo: {len(bvals)} volumi, b_max={bvals.max():.0f}, "
      f"{len(shells)-1} shell non nulle")

# ── ground truth: direzione UNIFORME sull'emisfero (cade fra i nodi) ──────
def rand_hemi(n):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[v[:, 2] < 0] *= -1
    return v

GT_DIR = rand_hemi(N_VOX)
GT_AD  = np.full(N_VOX, 1.70e-3)
GT_RD  = rng.uniform(0.20e-3, 0.60e-3, N_VOX)     # sano -> demielinizzato
GT_FF  = np.full(N_VOX, 0.55)
GT_RF  = np.full(N_VOX, 0.10)
GT_HF, GT_WF = np.full(N_VOX, 0.20), np.full(N_VOX, 0.15)
D_RES, D_HIN, D_WAT = 0.15e-3, 1.0e-3, 3.0e-3

side = int(np.ceil(N_VOX ** (1/3))) + 1
shape = (side, side, side)
data = np.zeros(shape + (len(bvals),), np.float32)
mask = np.zeros(shape, bool)
idx = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:N_VOX]
for n, (i, j, k) in enumerate(idx):
    c = bvecs @ GT_DIR[n]
    fib = np.exp(-bvals * (GT_RD[n] + (GT_AD[n] - GT_RD[n]) * c**2))
    sig = (GT_FF[n]*fib + GT_RF[n]*np.exp(-bvals*D_RES)
           + GT_HF[n]*np.exp(-bvals*D_HIN) + GT_WF[n]*np.exp(-bvals*D_WAT))
    noisy = np.sqrt((sig + rng.normal(0, 1/SNR, len(bvals)))**2
                    + rng.normal(0, 1/SNR, len(bvals))**2)     # Rician
    data[i, j, k] = 1000.0 * noisy
    mask[i, j, k] = True
print(f"{N_VOX} voxel mono-fibra, SNR {SNR:.0f}, AD=1.70e-3, RD 0.20-0.60e-3\n")

def run(tag, **kw):
    t0 = time.time()
    m = DBSI_Adaptive(**kw)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res, mode = m.fit(data, bvals, bvecs, mask, run_calibration=True,
                          n_calibration_voxels=200,
                          correct_restricted_fraction=False)
    names = DBSI_Adaptive.output_map_names(mode)
    I = {n_: i for i, n_ in enumerate(names)}
    g = lambda nm: np.array([res[i, j, k, I[nm]] for (i, j, k) in idx])
    d = np.stack([g('dir1_x'), g('dir1_y'), g('dir1_z')], 1)
    cos = np.abs(np.sum(d * GT_DIR, 1)).clip(0, 1)
    out = dict(tag=tag, npop=g('n_fiber_populations'),
               ang=np.degrees(np.arccos(cos)), ad=g('axial_diffusivity_pop1'),
               rd=g('radial_diffusivity_pop1'), ff=g('fiber_fraction'),
               rf=g('restricted_fraction'), t=time.time()-t0,
               ndirs=m.n_dirs, mode=mode)
    print(f"  {tag}: {out['t']:.0f}s  (n_dirs={m.n_dirs}, {mode}-ISO)")
    return out

print("fit in corso...")
A = run("C da solo (default)",  stagec_refine=True,  stagec_dir_refine=False)
B = run("cono -> C",            stagec_refine=True,  stagec_dir_refine=True)
C = run("cono da solo (C off)", stagec_refine=False, enable_direction_refinement=True)
np.savez('/tmp/exp_cone_results.npz',
         **{f"{k}_{lab}": o[k] for lab, o in (('A',A),('B',B),('C',C))
            for k in ('npop','ang','ad','rd','ff','rf')},
         gt_ad=GT_AD, gt_rd=GT_RD, gt_ff=GT_FF, gt_rf=GT_RF)

ok = np.isfinite(A['ad']) & np.isfinite(B['ad']) & np.isfinite(C['ad'])
print(f"\nvoxel con fibra risolta in tutte e tre le condizioni: {ok.sum()}/{N_VOX}")
print(f"n_pop==1: A={np.sum(A['npop']==1)}  B={np.sum(B['npop']==1)}  C={np.sum(C['npop']==1)}\n")

hdr = f"{'':<22}{'ang err':>10}{'AD err%':>10}{'RD err%':>10}{'FF err':>9}{'RF err':>9}"
print(hdr); print("-"*len(hdr))
def med(a): return float(np.median(a))
for o in (A, B, C):
    m_ = ok & (o['npop'] == 1)
    ade = np.abs(o['ad'][m_] - GT_AD[m_]) / GT_AD[m_] * 100
    rde = np.abs(o['rd'][m_] - GT_RD[m_]) / GT_RD[m_] * 100
    ffe = np.abs(o['ff'][m_] - GT_FF[m_])
    rfe = np.abs(o['rf'][m_] - GT_RF[m_])
    print(f"{o['tag']:<22}{med(o['ang'][m_]):>9.2f}°{med(ade):>10.2f}{med(rde):>10.2f}"
          f"{med(ffe):>9.3f}{med(rfe):>9.3f}"
          f"   | stimati: AD={med(o['ad'][m_])*1e3:.2f} FF={med(o['ff'][m_]):.3f} RF={med(o['rf'][m_]):.3f}")
