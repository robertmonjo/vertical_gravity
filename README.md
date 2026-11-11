# vgrav — Radial–vertical gravity decoupling in the Milky Way: a multi-model test

Reproducibility package for the paper:

> **Radial–vertical gravity decoupling in the Milky Way: modified gravity and dark matter tests using Gaia DR3**
> Monjo & Sasi (2026, submitted)

This package contains the full analysis pipeline and pre-computed outputs
for testing 12 gravity model variants (from 7 frameworks) against Gaia DR3
rotation-curve and vertical-potential data from Wang+2026 and Lian+2022.

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
Baryonic Newtonian                 0  132.823 182.953 294.224
QUMOND simple                      0    3.833   7.698  26.286
QUMOND standard                    0   20.363  46.953 106.555
QUMOND MLS/RAR *                   0    4.572   7.652  36.995
VEG original                       0    9.076  26.435  49.909
VEG free a_EG                      1    6.732   7.748   8.844
STVG                               2    2.944   6.997  25.692
CDM NFW                            2    2.641   3.142   4.061
CDM Einasto                        3    1.994   2.516   3.542
HMG (This Work)                    1    2.327   2.898   7.208
f(R) screened                      2    7.323   8.282   9.682
Refracted Gravity                  2    7.353   8.942  11.453
--------------------------------------------------------------
* uses old stochastic ensemble
```

Note: the paper (Monjo & Sasi 2026) reports baryonic chi2_nu ~ 194.7 using radius-dependent
Gaussian weights on the MC100 ensemble; step3 uses unweighted percentiles (p50 ~ 183).
All other models agree with the paper to better than 1%.

The large chi2_nu for Baryonic Newtonian (~183-195) reflects the well-known
missing-mass problem: baryons alone cannot explain the observed Milky Way
rotation curve.  CDM-Einasto and s-HMG give the best fits (chi2_nu ~ 2.5-3.1).

### Table 3 (band-consistent ensembles, statistical errors only)

Table 3 of the earlier version uses the band-consistent (BC) fits stored in
`outputs/bc_nbar{1,2,3,4}/`, one directory per baryonic ensemble (MI, MI+LW,
MI+LW+B2, MI+LW+B2+MM). Each model CSV stores the chi² of every draw in its
last row.

```bash
# Figure with these fits: writes figs/fig2_reproduced_4fam.png
python scripts/step3_bc_fig2.py

# Table 3: p50 (p16/p84) of chi2_nu and number of leads per ensemble
python scripts/bc_leads_table.py
```

To refit the BC models from the baryonic draws (several hours per ensemble):

```bash
python scripts/step2_bc.py --full --no-stvg --baryon-band outputs/nbar1/baryon_band_nbar1.csv --baryonic-dir outputs/nbar1 --outdir outputs/bc_nbar1
```

The s/r_h-HMG curve of that figure (`outputs/bc_nbar4/model_bc_hmg_sk2_*`) is
obtained with the same statistic as its other curves:

```bash
python scripts/step2_bc.py --full --no-stvg --models hmg_sk2 --baryon-band outputs/nbar4/baryon_band_nbar4.csv --baryonic-dir outputs/nbar4 --outdir outputs/bc_nbar4
```

### Full-likelihood model comparison (BIC)

The model comparison uses the likelihood of Wang et al. (2026):

    -2 ln L = sum_rad [ (dv / sigma_v)^2 + ln(2 pi sigma_v^2) ]
            + sum_vert [ (dPhi / sigma_eff)^2 + ln(2 pi sigma_eff^2) ]
            + prior(rho_thin) + prior(rho_thick),

with `sigma_eff^2 = sigma_Phi^2 + (K_z sigma_z)^2`, where `K_z` is the vertical
force of each model. Systematic errors are added in quadrature: the Wang et al.
rotation-curve budget (5–20 per cent, depending on R) to every radial point
outside the Wang curve, whose errors already include it, and 5 per cent of
`Phi` to the vertical points. The local stellar densities have Gaussian priors,
`rho_thin(R0) ~ N(0.045, 0.003)` and `rho_thick(R0) ~ N(0.00525, 0.0006)`
Msun pc^-3. Models are ranked by `BIC = -2 ln L_max + k ln N`.

Five radial data vectors are evaluated (`vgrav.observations.radial_variant_arrays`):

| Variant | Radial points | Content |
|---|---|---|
| `all` | 152 | all model-independent rotation-curve points |
| `no_eilers` | 114 | without the Eilers/McGaugh curve, which overlaps the Wang curve |
| `no_wang` | 118 | without the Wang curve |
| `overlap_correlated` | 152 | all points; Wang and Eilers errors multiplied by sqrt(2) over their common radii |
| `wang_only` | 34 | the Wang et al. curve alone (78 points with the vertical data) |

Two frameworks are provided:

1. **One fit per baryonic reconstruction** (`scripts/likelihood_bic.py`,
   `scripts/likelihood_bic_stvg.py`). MI, LW, B2 and MM are fitted separately;
   the thin- and thick-disc factors are free for every model, under the priors
   above and within +-10 sigma of their centres. QUMOND is solved with its
   Poisson equation, so that v_c and Phi_z derive from one potential: a 9 x 9
   grid of disc factors (steps of 2 sigma) followed by a 5 x 5 refinement
   (steps of 0.5 sigma) around the best point. STVG is solved on the same
   grid: Phi_STVG = Phi_N + alpha (Phi_N - Phi_Y), with (nabla^2 - mu^2) Phi_Y =
   4 pi G rho for the same density (`vgrav.models.stvg_grid_kernel`); the
   correction is linear in the density, so for each mu on a logarithmic grid
   (0.003–3 kpc^-1, including 0.04 kpc^-1) three kernels are computed once and
   (z_t, z_k, alpha) are fitted by linear combination. Refracted gravity solves
   nabla.[eps(rho) nabla phi] = 4 pi G rho with the permittivity of Cesare et al.
   (2020, Eq. 5) on nested grids: the outer grid (1 kpc spacing, 400 kpc) carries
   the far field of an isolated system, phi = -GM/(eps0 r), and gives the
   boundary of the inner grid (`vgrav.models.predict_refracted_gravity`).
   Output: `outputs/likelihood_bic/likelihood_bic_fits.csv`
   (columns: variant, config, model, k, N, m2lnL, chi2, chi2_nu, BIC, zt, zk,
   rho_thin, rho_thick, theta) and `outputs/likelihood_bic/summary.txt`.
2. **Band-consistent MC100 ensembles** (`scripts/step2_bc.py --full-likelihood
   --radial-variant V`). The four cumulative ensembles of Table 3 are refitted
   draw by draw with the same likelihood (discs fixed by each draw), with
   QUMOND solved (`--no-approx`). `scripts/likelihood_bic_mc100_extra.py` adds
   STVG with universal parameters, refracted gravity with the DiskMass
   parameters and the VEG apparent dark matter, computing each prediction once
   per draw and scoring it for the five data vectors.
   `scripts/likelihood_bic_mc100_summary.py` gives the median Delta BIC with
   respect to the best model of each draw and the number of draws in which each
   model has the lowest BIC. Output: `outputs/likelihood_bic_mc100/summary_<variant>.txt`
   and, per ensemble, `bic_per_draw.csv` and `likelihood_setup.json`.

The baryonic realisation can also be treated as a nuisance parameter:
`scripts/likelihood_bic_marginal.py` computes, from the existing BIC values and
without new fits, `BIC_marg = -2 ln[(1/N) sum_i exp(-BIC_i/2)]` over the draws
of each ensemble (and over the four reconstructions in framework 1). Output:
`outputs/likelihood_bic_mc100/marginal_all.txt`.

Reproduce everything from the bundled data and MC100 draws:

```bash
WORKERS=24 bash scripts/run_likelihood_bic.sh     # ~2.5 h on 32 cores
cat outputs/likelihood_bic/summary.txt             # framework 1 tables
cat outputs/likelihood_bic_mc100/summary_all.txt  # framework 2 table
```

The launcher also writes `tables/likelihood_bic_tables.tex`, the rows of the
two full-likelihood tables of the paper (fits per reconstruction; MC100
ensembles with the median and the baryon-marginalised Delta BIC). The tables list only the models with a physical
implementation; the rows of the remaining keys stay in the CSV files.

Fig. 2 of the paper is drawn from the framework-2 fits of the four-source
ensemble, with the legend giving the Delta BIC marginalised over the baryonic
draws (requires the model CSVs written by `run_likelihood_bic.sh`):

```bash
python scripts/step3_bc_fig2_physical.py           # figs/fig2_physical_models.png
```

Each ensemble must be run with its own baryonic band
(`--baryon-band outputs/nbar<n>/baryon_band_nbar<n>.csv`); `run_likelihood_bic.sh`
does so. In the model CSVs written with `--full-likelihood`, the `chi2_vertical`
row holds `-2 ln L_vert`; `likelihood_setup.json` stores N and the radial
`sum ln(2 pi sigma^2)`.

`fr_screened` is the f(R)-type interpolation function with a free amplitude
`delta_sc`; `fr_bounded` is the same function with `delta_sc <= 1/3`, the
largest enhancement of the unscreened force in metric f(R) gravity (4/3).

Model keys:

| Key | Model | Free gravity parameters |
|---|---|---|
| `hmg_k1` (step2_bc), `hmg_sp` (likelihood_bic) | s-HMG | s |
| `hmg_sk2` | s/r_h-HMG: HMG radial field, E-2D vertical potential with r_h | s, r_h |
| `e2d_k2` | sr_h-HMG | s, r_h |
| `mond_*` (step2_bc), `qumond_*` (likelihood_bic) | QUMOND, Simple, Standard and MLS interpolation functions, solved | none |
| `veg_verlinde` | apparent dark matter of Verlinde (2016, Eq. 7.40), `M_D^2 = a0 r^2/(6G) d(M_B r)/dr`, `a0 = c H0`, added as a spherical halo | none |
| `veg_verlinde_free` | the same with `a0` free | a0 |
| `stvg_std` | STVG with `alpha = 8.89`, `mu = 0.04 kpc^-1` (Moffat & Rahvar 2013) | none |
| `fr_bounded` | f(R)-type interpolation function `nu = 1 + delta_sc exp(-x/x_c)` with `delta_sc <= 1/3` | delta_sc, x_c |
| `reg_dms` | refracted gravity solved with its field equation, `eps0 = 0.661`, `Q = 1.79`, `log10 rho_c = -24.54` g cm^-3 (DiskMass Survey, Cesare et al. 2020) | none |
| `reg_free` | the same with `eps0`, `Q` and `rho_c` free within the priors of Cesare et al. (2020) | eps0, Q, rho_c |
| `cdm_nfw`, `cdm_einasto` | CDM haloes | 2 (NFW), 3 (Einasto) |

`veg_fixed`, `veg_free`, `fr_screened` (amplitude unbounded),
`refracted_gravity` (algebraic prescription `nu = 1/eps(x)`), `stvg` (alpha and
mu free) and `hmg_sr` are kept in the CSV files for comparison with the earlier
fits of Table 2.

### Mode 2 — Model comparison analysis

```bash
python scripts/step4_analyze_hmg.py
```

Reads `outputs/mc100_chi2_all_models.csv` and identifies the baryonic realizations in which s-HMG outperforms CDM-Einasto by reduced chi².

---

## Full pipeline (from scratch)

The four step scripts reproduce everything from the raw ingredients:

```bash
# Step 1: Generate 100 MC100 baryonic draws
python scripts/step1_build_baryonic_mc100.py --full

# Step 2: Fit 11 gravity models (STVG runs via step2_stvg_parallel.py)
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
│   ├── model_{key}_{radial,vertical}.csv       24 files (12 models × 2 grids)
│   ├── mc100_chi2_all_models.csv               1100 chi² rows
│   ├── bc_nbar{1,2,3,4}/                       Band-consistent fits (Table 3, Fig. 2)
│   ├── likelihood_bic/                         Full-likelihood fits per reconstruction (BIC)
│   ├── likelihood_bic_mc100/                   Full-likelihood MC100 ensembles (BIC per draw)
│   ├── wang78_table2_joint_disc.csv            Wang 78-point joint-disc fits
│   ├── fig2_observational_data.csv             Observational data catalog
│   └── hmg_competitive_analysis.csv            HMG per-draw comparison
│
├── vgrav/                  Python library
│   ├── __init__.py         Public API
│   ├── _constants.py       Physical constants (G, a0, c, T0, ...)
│   ├── observations.py     Load Wang+2026 data from CSV
│   ├── solver.py           Cylindrical Poisson solver (CylGrid)
│   ├── baryonic.py         Parametric density + MC100 qcopula
│   ├── models.py           All 12 gravity model equations
│   ├── chi2.py             Chi-squared and reduced chi-squared
│   └── figures.py          Fig. 2 renderer + Table 2 printer
│
└── scripts/
    ├── step0_reproduce_fig1.py         Reproduce Fig. 1 (density + spiral arms)
    ├── step1_build_baryonic_mc100.py   MC100 baryonic draws
    ├── step2_fit_all_models.py         Gravity model fitting
    ├── step2_bc.py                     Band-consistent model fitting
    ├── step3_reproduce_fig2_table2.py  Main verification script
    ├── step3_bc_fig2.py                Fig. 2 from the band-consistent fits
    ├── bc_leads_table.py               Table 3 (chi2_nu percentiles and leads)
    ├── likelihood_bic.py               Full-likelihood fits per reconstruction
    ├── likelihood_bic_stvg.py          STVG fits for likelihood_bic.py
    ├── likelihood_bic_mc100_extra.py   STVG (universal), ReG (DiskMass) and VEG fits of the MC100 ensembles
    ├── likelihood_bic_marginal.py      Delta BIC marginalised over the baryonic realisations
    ├── step3_bc_fig2_physical.py       Fig. 2 from the full-likelihood fits
    ├── likelihood_bic_mc100_summary.py BIC summary of the MC100 ensembles
    ├── run_likelihood_bic.sh           Runs the full-likelihood comparison
    ├── likelihood_bic_tables.py        LaTeX rows of the full-likelihood tables
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
    predict_hmg_proxy,        # s-HMG proxy (radial+vertical)
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
| `qumond_mls` | QUMOND MLS/RAR | 0 | ν = 1/(1 − exp(−√x)); old stochastic draws |
| `veg_original` | VEG original | 0 | ν = 1 + √(a_EG/g_N), fixed a_EG=cH₀/6 |
| `emergent_gravity` | VEG free a_EG | 1 | ν = 1 + √(a_EG/g_N), a_EG free |
| `stvg` | STVG | 2 | Yukawa extra force, α, μ free |
| `cdm_nfw` | CDM NFW | 2 | NFW halo, local DM density fit |
| `cdm_einasto` | CDM Einasto | 3 | Einasto halo, α, ρ_s, r_s free |
| `hmg_k1` | s-HMG (This Work) | 1 | single vicinity scale s; radial+vertical |
| `hmg_sk2` | s/r_h-HMG (This Work) | 2 | HMG radial field and E-2D vertical potential, vicinity scale s and radius r_h |
| `e2d_k2` | sr_h-HMG (This Work) | 2 | vicinity scale s and external-regime radius r_h |
| `fr_screened` | f(R) screened | 2 | ν = 1 + δ·exp(−x/xc) |
| `refracted_gravity` | Refracted Gravity | 2 | ε(x) screening function |

Here x = |g_N|/a₀, a₀ = 1.2×10⁻¹⁰ m/s², k = number of free parameters.

---

## Citation

If you use this package, please cite:

```bibtex
@article{monjosasi2026vertical,
  author  = {Monjo, Robert and Sasi, A. E.},
  title   = {Radial--vertical gravity decoupling in the {Milky Way}:
             modified gravity and dark matter tests using {Gaia DR3}},
  year    = {2026},
  note    = {in preparation},
}
```

---

## Licence

MIT.  See LICENCE file.
