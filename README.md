# vgrav — Vertical Gravity Milky Way benchmark

Reproducibility package for the paper:

> **Vertical gravitational potential of the Milky Way as a multi-model benchmark**
> (Wang et al. 2026, submitted)

This package contains the full analysis pipeline and pre-computed outputs
for testing 11 gravity models against Wang+2026 rotation-curve and
vertical-potential data.

---

## Quick start

### Requirements
- Python 3.9+
- numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7

### Installation

```bash
cd release/
pip install -e .
```

Or without installing:
```bash
pip install numpy scipy matplotlib
export PYTHONPATH=/path/to/release:$PYTHONPATH
```

---

## Two execution modes

### Mode 1 — Fast verification (~seconds)

Reads the pre-computed CSVs in `outputs/` and reproduces Figs. 1–2 + Table 2.
No heavy computation required.

```bash
# Reproduce Fig. 1 (baryonic density + spiral arms)
python scripts/step0_reproduce_fig1.py

# Reproduce Table 2 + Fig. 2 (rotation curve + vertical potential)
python scripts/step3_reproduce_fig2_table2.py

# Reproduce Fig. 2 only
python -c "
from pathlib import Path
from vgrav.figures import plot_fig2
fig = plot_fig2(Path('outputs'), output_path='fig2_reproduced.png', dpi=220)
"
```

Expected output (Table 2, chi2_nu = chi2_total / (N_PRIMARY - k)):

```
Model                              k      p16     p50     p84
--------------------------------------------------------------
Baryonic Newtonian                 0  151.797 194.653 231.409
QUMOND simple                      0    3.911   5.857  11.487
QUMOND standard                    0   29.935  50.615  71.322
QUMOND MLS/RAR *                   0    4.582   7.659  37.000
STVG                               1    4.389   8.222  14.594
CDM NFW                            2    2.571   3.270   4.431
CDM Einasto                        2    2.187   2.768   3.585
HMG anisotropic (k=1)              1    2.583   3.277   4.568
f(R) screened                      2    6.593   8.082  10.024
Refracted Gravity                  2    6.480   8.916  11.233
Emergent Gravity (fixed)           0   12.835  23.637  42.196
--------------------------------------------------------------
* uses old stochastic ensemble
```

The large chi2_nu for Baryonic Newtonian (~195) reflects the well-known
missing-mass problem: baryons alone cannot explain the observed Milky Way
rotation curve.  CDM halos and HMG anisotropic give the best fits (chi2_nu ~2.8-3.3).

### Mode 2 — HMG competitive analysis

```bash
python scripts/step4_analyze_hmg.py
```

Reads `outputs/mc100_chi2_all_models.csv` and reports which of the 100
baryonic realizations are competitive between HMG and CDM Einasto.

---

## Full pipeline (from scratch)

The four step scripts reproduce everything from the raw ingredients:

```bash
# Step 1: Generate 100 MC100 baryonic draws
python scripts/step1_build_baryonic_mc100.py --full

# Step 2: Fit all 11 gravity models
python scripts/step2_fit_all_models.py --full

# Step 3: Chi² summary + Fig. 2
python scripts/step3_reproduce_fig2_table2.py

# Step 4: HMG competitive analysis
python scripts/step4_analyze_hmg.py
```

**Runtime note**: Steps 1 and 2 can take several hours (cylindrical Poisson
solver for QUMOND; direct Yukawa summation for STVG).  Steps 3 and 4 are fast.

---

## Package contents

```
release/
├── pyproject.toml          Package metadata (pip install -e .)
├── requirements.txt        Python dependencies
├── README.md               This file
│
├── data/                   Bundled observational and model data
│   ├── wang2026_rotation_curve.csv      152 radial data points
│   ├── wang2026_vertical_potential.csv  44 vertical potential points
│   ├── baryon_band.csv                  Hybrid baryonic band (5 families, centers)
│   ├── baryonic_target_band.csv         Smoothed target band percentiles (Fig. 2)
│   ├── fig2_observational_catalog.csv   Full radial observation catalog (262 rows)
│   ├── feng2026_cepheid_rotation_curve.csv  Cepheid RC context points
│   ├── fig1_density_grid.npz            Pre-computed baryonic density grid (Fig. 1)
│   └── fig1_spiral_arms.csv             Reid+2019 spiral arm loci (Fig. 1)
│
├── outputs/                Pre-computed results (fast-mode)
│   ├── mc100_baryonic_{radial,vertical}.csv    100 baryonic draws
│   ├── model_{key}_{radial,vertical}.csv       22 files (11 models × 2 grids)
│   ├── mc100_chi2_all_models.csv               1100 chi² rows
│   ├── fig2_observational_data.csv             Observational data catalog
│   └── hmg_competitive_analysis.csv            HMG per-draw comparison
│
├── vgrav/                  Python library
│   ├── __init__.py         Public API
│   ├── _constants.py       Physical constants (G, a0, c, T0, ...)
│   ├── observations.py     Load Wang+2026 data from CSV
│   ├── solver.py           Cylindrical Poisson solver (CylGrid)
│   ├── baryonic.py         Parametric density + MC100 qcopula
│   ├── models.py           All 11 gravity model equations
│   ├── chi2.py             Chi-squared and reduced chi-squared
│   └── figures.py          Fig. 2 renderer + Table 2 printer
│
└── scripts/
    ├── step0_reproduce_fig1.py         Reproduce Fig. 1 (density + spiral arms)
    ├── step1_build_baryonic_mc100.py   MC100 baryonic draws
    ├── step2_fit_all_models.py         Gravity model fitting
    ├── step3_reproduce_fig2_table2.py  Main verification script
    └── step4_analyze_hmg.py            HMG competitive analysis
```

---

## Library API

```python
from vgrav import (
    load_observations,        # Wang+2026 data
    nu_mond,                  # MOND nu(x) interpolation
    hmg_factor,               # HMG f_R = sqrt(1 + beta*extra/g_N)
    nu_fr, nu_rg, nu_eg,      # f(R), Refracted Gravity, VEG
    nfw_mass, einasto_mass,   # CDM halo mass functions
    predict_cdm_nfw,          # CDM NFW: vc + phi from baryonic curves
    predict_cdm_einasto,      # CDM Einasto
    predict_mond_proxy,       # MOND proxy: vc = sqrt(nu)*vc_N
    predict_hmg_proxy,        # HMG anisotropic proxy
    chi2_radial,              # Radial chi-squared
    chi2_vertical,            # Vertical chi-squared (Kz-corrected)
    chi2_nu,                  # Reduced chi-squared
    make_grid,                # Cylindrical solver grid
    solve_axisymmetric,       # Poisson solver
    predict_qumond_solver,    # QUMOND from 3D Poisson
    plot_fig2,                # Fig. 2 renderer
    print_table2,             # Table 2 printer
)
```

### Example: compute MOND proxy chi²

```python
import numpy as np
from vgrav import load_observations, radial_fit_arrays, vertical_arrays
from vgrav import predict_mond_proxy, chi2_radial, chi2_vertical, chi2_nu

rot, vert = load_observations()
rr, vv, ss = radial_fit_arrays(rot=rot)
rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

# Baryonic velocity curve (from pre-computed CSV draw b1)
import csv
from pathlib import Path
with open(Path("outputs/mc100_baryonic_radial.csv")) as f:
    rows = list(csv.reader(f))
header = rows[0]
r_grid = np.array([float(r[0]) for r in rows[1:-1]])
vc_n   = np.array([float(r[1]) for r in rows[1:-1]])  # b1

# ... similarly load phi_n from mc100_baryonic_vertical.csv ...

vc_mond, phi_mond = predict_mond_proxy(r_grid, vc_n, phi_n_at_obs, rv)
c2r = chi2_radial(vc_mond, r_grid, rr, vv, ss)
c2z = chi2_vertical(phi_mond, rv, zv, rv, zv, phi_obs, sig_phi, sig_z)
print(f"MOND chi²_nu = {chi2_nu(c2r + c2z, k=0):.3f}")
```

---

## Baryonic model and MC100 qcopula

The 100 baryonic realizations are drawn from a **mass-constrained
Gaussian-process copula** on the weighted hybrid baryonic band
(five literature estimates combined):

1. McGaugh+2018 / Imig+2025
2. Wang+2026 / Lian+2022
3. McMillan 2017
4. de Salas+2019 B2
5. Barros+2016 MI

Each draw samples a smooth rotation curve consistent with the band's
5th–95th percentile range (clipped to [12%, 88%] at the knot level),
with a GP correlation length λ=0.72 in log(R) and SIGMA_AMP=0.20 for
total mass variation.  Seed: 20260607.

---

## Gravity models

| Key | Model | k | Description |
|-----|-------|---|-------------|
| `baryonic` | Baryonic Newtonian | 0 | Reference: disc+gas+bulge only |
| `qumond_simple` | QUMOND simple | 0 | ν = 0.5 + √(0.25 + 1/x) |
| `qumond_standard` | QUMOND standard | 0 | ν = √(0.5 + √(0.25 + 1/x²)) |
| `qumond_mls` | QUMOND MLS/RAR | 0 | ν = 1/(1 − exp(−√x)) |
| `stvg` | STVG | 1 | Yukawa extra force, α=10.68, μ=0.07/kpc |
| `cdm_nfw` | CDM NFW | 2 | NFW halo, local DM density fit |
| `cdm_einasto` | CDM Einasto | 2 | Einasto halo, α=0.97 |
| `hmg_k1` | HMG anisotropic | 1 | f_R = √(1+β·extra/g_N), anisotropic |
| `fr_screened` | f(R) screened | 2 | ν = 1 + δ·exp(−x/xc) |
| `refracted_gravity` | Refracted Gravity | 2 | ε(x) screening function |
| `emergent_gravity` | Emergent Gravity | 0 | ν = 1 + √(a_EG/g_N), fixed a_EG=cH₀/6 |

Here x = |g_N|/a₀, a₀ = 1.2×10⁻¹⁰ m/s², k = number of free parameters.

---

## Citation

If you use this package, please cite:

```bibtex
@article{wang2026vertical,
  author  = {Wang, ...},
  title   = {Vertical gravitational potential of the Milky Way},
  journal = {...},
  year    = {2026},
}
```

---

## Licence

MIT.  See LICENCE file.
