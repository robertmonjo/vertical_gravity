# Publication pipeline manifest

This file is generated/updated by `scripts/run_publication_from_zero.py`.

## Canonical steps

1. `scripts\build_multiauthor_baryonic_families.py` — Build five-family baryonic source curves
2. `scripts\build_weighted_hybrid_baryon_band.py` — Build weighted-hybrid baryonic band
3. `scripts\build_fig2c_weighted_hybrid_models.py` — Build MC100 common-baryon model fits
4. `scripts\fit_hmg_common_s_mc100.py` — Fit HMG common-s branch on the same MC100 family
5. `scripts\make_fig2b_consolidated.py` — Fit VEG/ReG/f(R)/emergent branches
6. `scripts\reproduce_fig2_table2.py` — Assemble Table/Fig. 2 CSV products

## Policy

- Default route: regenerate from source scripts and correct solvers.
- Generated caches are removed unless `--allow-cache` is passed.
- The simplified radial-template release2 Step 1 is diagnostic only.
- Output snapshots in `release2/outputs` are copies of regenerated
  project outputs, not independent data sources.
