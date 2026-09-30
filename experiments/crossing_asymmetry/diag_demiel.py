"""La RD di un fascio demielinizzato (AD 1.5, RD 0.8, FA ~0.36) e' identificabile? Mono (Stage C) e oracolo crossing."""
import io, contextlib, numpy as np
from pathlib import Path
from dbsi_toolbox import DBSI_Adaptive
from dbsi_toolbox.core.solvers import estimate_AD_RD_mrds, _TENSOR_RD_FLOOR
HERE = Path(__file__).parent
bvals = np.loadtxt(HERE / 'p3_like.bval').ravel(); bvecs = np.loadtxt(HERE / 'p3_like.bvec')
bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
nb = np.linalg.norm(bvecs, axis=1, keepdims=True); nb[nb == 0] = 1; bvecs = bvecs / nb
b0 = bvals < 100; N = DBSI_Adaptive.output_map_names(3); IX = {n: i for i, n in enumerate(N)}
rng = np.random.default_rng(3)
fib = lambda d, ad, rd: np.exp(-bvals * (rd + (ad - rd) * (bvecs @ d) ** 2))
isoS = lambda: 0.10 * np.exp(-bvals * 0.15e-3) + 0.20 * np.exp(-bvals * 1.0e-3) + 0.10 * np.exp(-bvals * 3.0e-3)
def rdir():
    u = rng.normal(size=3); return u / np.linalg.norm(u)
for SNR in (26, 40, 100):
    sd = 1 / SNR
    for ad, rd, lab in ((1.7e-3, 0.4e-3, 'sano'), (1.5e-3, 0.8e-3, 'demiel')):
        V = 300; data = np.zeros((15, 20, 1, len(bvals)), np.float32)
        for i in range(V):
            S = isoS() + 0.6 * fib(rdir(), ad, rd)
            data[i // 20, i % 20, 0] = 1000 * np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2)
        m = DBSI_Adaptive(lambda_aniso=8.376776400682925, lambda_iso=0.017012542798525893, n_iso=6)
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = m.fit(data, bvals, bvecs, np.ones(data.shape[:3], bool), run_calibration=False)
        P = res.reshape(-1, len(N)); one = P[:, IX['n_fiber_populations']] == 1
        # oracolo crossing: direzioni, frazioni e iso VERI, solo la LM del tensore
        eo = []
        for _ in range(200):
            u = rdir(); t = np.cross(u, rng.normal(size=3)); t /= np.linalg.norm(t)
            S = isoS() + 0.3 * (fib(u, ad, rd) + fib(t, ad, rd))
            raw = np.sqrt((S + rng.normal(0, sd, S.shape)) ** 2 + rng.normal(0, sd, S.shape) ** 2); raw /= raw[b0].mean()
            _, RDo = estimate_AD_RD_mrds(bvals, bvecs, raw.astype(np.float64), np.stack([u, t]), np.array([0.3, 0.3]), isoS(), 3, 25)
            eo += list(RDo - rd)
        print(f'SNR {SNR:3d} {lab:6s}: MONO n_pop=1 {one.mean():.2f}  RD bias {np.nanmean(P[one, IX["radial_diffusivity_pop1"]] - rd)*1e3:+.3f}  '
              f'FF bias {np.nanmean(P[one, IX["fiber_fraction"]] - 0.6):+.3f}  pavimento {np.mean(P[one, IX["radial_diffusivity_pop1"]] <= _TENSOR_RD_FLOOR*1.0001):.2f}   '
              f'| ORACOLO crossing RD bias {np.mean(eo)*1e3:+.3f}')
