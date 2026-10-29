# notebooks/

Two notebooks, each meant to be run once, top to bottom, producing all its figures. Both are
self-contained (the code and the real P3 gradient table are pasted in): they need only the installed
toolbox. Run them from this folder; outputs go to `risultati_*/` (ignored by git).

| notebook | what it shows | time |
|---|---|---|
| `simulazioni_pyDBSI.ipynb` | accuracy against ground truth: anatomical phantom (true / estimated / error images), 13-scenario battery (error per scenario at 3 SNR, population count, angular error), continuous fields (estimate vs truth), coverage of the uncertainty maps, the 5P protocols if their tables are on the NAS. Production chain: Rician noise, then free calibration; a second arm with `dwidenoise` (MRtrix) as an explicitly idealised sensitivity analysis; a third arm, `grezzo_18`, with the two crossing options of 1.9.0 switched off (the 1.8 fit), compared with the default in section 6b | 7–10 h (resumable; dropping `grezzo_18` saves its 3 conditions) |
| `matematica_pyDBSI.ipynb` | the mathematics of the toolbox in figures, section by section of the *voxel journey* report: 3-ISO choice, Rician correction, isotropic grid and the design matrix A, neighbour graph and cone, n_iso convergence, calibration sample and sigma, the calibration step by step, angular concentration and lambda modulation, the cone on a voxel, crossing FF and MRDS, the cone per population and the two-vs-one test on crossings (defaults since 1.9.0), the uncertainty maps explained | 1–2 h |

The figures are taken from the toolbox itself wherever possible: the notebooks run the real fit and
record its inputs (design matrix, lambdas, dictionary, corrected signal) and each calibration step.
Where a figure shows an internal step the toolbox does not return, it recomputes it with the same
functions and asserts that the result matches the toolbox.

`sorgenti/` holds the code the notebooks are generated from; edit it, then regenerate:

    python notebooks/sorgenti/build_simulazioni.py
    python notebooks/sorgenti/build_matematica.py
