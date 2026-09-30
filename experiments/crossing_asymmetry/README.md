# Crossing / single-fiber asymmetry (inspection 2026-09-30)

Why ~49% of real crossings sit on the RD floor. Protocol: `p3_like.bval/.bvec`
(9 b=0, shells 500/1000/1500/2000). Run from the package root:

    python experiments/crossing_asymmetry/oracolo.py   # MRDS with TRUE inputs + single-fiber control
    python experiments/crossing_asymmetry/debias.py    # unpenalised FF re-solve on the detected support

Result and table: the MRDS section of `dbsi_toolbox/core/solvers.py`
("MEASURED 2026-09-30"). The full-fit comparison of the three alternative
crossing paths (`fit(_mrds_mode=1..3)`, `exp_crossing.py`) lives only on
branch `ispezione-crossing`: none was adopted.
