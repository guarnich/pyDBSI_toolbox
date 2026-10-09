# experiments/

Measurements behind specific decisions in the toolbox. They are a record, not tests:
each folder or script states what it measured, on which version, and where the
result is written down (usually a comment next to the code it changed). Re-running
one on a later version answers a different question.

| script | measured on | decision |
|---|---|---|
| `exp_cone_vs_stagec.py`, `exp_cone_multiproto.py` | 1.1.0 | cone before Stage C (`stagec_dir_refine`) |
| `crossing_asymmetry/` | 1.6.0 | why crossings sit on the RD floor -> crossing FF re-solve (1.7.0) |
| `fastmath_nan/confronto_versioni.py` | 1.6.3 vs 1.7.0 | fits made on x86 before 1.6.4 stay valid |
| `uncertainty/copertura.py` | 1.6.x and 1.7.0 | coverage of the uncertainty maps |

Since 1.7.1 `fit()` requires a noise map: the synthetic scripts pass the true
sigma (`noise_map = 1000 / SNR`, the signal being scaled to S0 = 1000).
`fastmath_nan/confronto_versioni.py` runs on real data without a noise map and
is meant for the two versions it compares (checkout `v1.7.0`, see its docstring).
`uncertainty/copertura.py` writes its CSV next to itself, overwriting the
recorded result for the same SNR and threshold.
