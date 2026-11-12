# vgrav — Radial–vertical gravity decoupling in the Milky Way

Reproducibility package for

> **Radial–vertical gravity decoupling in the Milky Way: modified gravity and dark matter tests using Gaia DR3**
> R. Monjo and A. E. Sasi (2026, in preparation)

The package compares cold dark matter haloes and modified-gravity theories on a
common Milky Way data vector: 152 model-independent rotation-curve points and
the 44 vertical-potential measurements of Wang et al. (2026) from the Gaia DR3
phase-space spiral (N = 196). The baryonic distribution is propagated through
four ensembles of 100 Monte Carlo realisations built from up to four published
reconstructions (McGaugh–Imig, Wang–Lian, de Salas, McMillan). Models are fitted
with the full likelihood of Wang et al. (2026) and ranked by the Bayesian
information criterion (BIC).

---

## Installation

```bash
git clone https://github.com/robertmonjo/vertical_gravity
cd vertical_gravity
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=$PWD:$PYTHONPATH
```

Python 3.9+ with NumPy, SciPy, Matplotlib, Astropy and pandas
(`requirements.txt`).

---

## Reproducing the paper

### 1. From the bundled outputs (seconds)

All fits are bundled in `outputs/`. The tables and Fig. 2 of the paper are
rebuilt from them without new fits:

```bash
python scripts/likelihood_bic_tables.py > tables/likelihood_bic_tables.tex   # rows of the two BIC tables
python scripts/likelihood_bic_marginal.py                                    # Delta BIC marginalised over the baryons
python scripts/likelihood_bic.py --summary                                   # all models and radial data vectors, per reconstruction
cat outputs/likelihood_bic_mc100/summary_all.txt                             # MC100 ensembles, all models
python scripts/step3_bc_fig2_physical.py                                     # Fig. 2 -> figs/fig2_physical_models.png
python scripts/step0_reproduce_fig1.py                                       # Fig. 1
```

`tables/likelihood_bic_tables.tex` holds, row by row, the two BIC tables of the
paper: fits per baryonic reconstruction (chi2_nu and Delta BIC), and the MC100
ensembles (median Delta BIC with p16–p84 and number of realisations with the
lowest BIC; Delta BIC marginalised over the realisations).

### 2. Full model comparison (~2.5 h on 32 cores)

```bash
WORKERS=24 bash scripts/run_likelihood_bic.sh
```

This refits every model in both frameworks below, for the five radial data
vectors, and rewrites `outputs/likelihood_bic/`, `outputs/likelihood_bic_mc100/`
and `tables/likelihood_bic_tables.tex`. It requires the baryonic realisations in
`outputs/nbar{1..4}/` (bundled).

### 3. Baryonic ensembles from scratch

```bash
python scripts/run_all_nbar.py --full               # see the script for parallel modes
```

builds, for each ensemble n = 1–4, the weighted baryonic band
(`outputs/nbar<n>/baryon_band_nbar<n>.csv`, `scripts/rebuild_baryon_band.py`) and
its 100 realisations (`outputs/nbar<n>/mc100_baryonic_{radial,vertical}.csv`,
`scripts/step1_build_baryonic_mc100.py`). Ensemble n combines the first n
reconstructions: MI, MI+LW, MI+LW+B2, MI+LW+B2+MM. The weights are
w_i(R) ∝ 1/chi2_local,i(R) between each reconstruction and the radial data,
with a Gaussian kernel of width 0.5 in ln R. The script also refits the models
with the chi2 statistic of Sect. "Statistical errors only" below.

---

## Likelihood and frameworks

    -2 ln L = sum_rad [ (dv / sigma_v)^2 + ln(2 pi sigma_v^2) ]
            + sum_vert [ (dPhi / sigma_eff)^2 + ln(2 pi sigma_eff^2) ]
            + prior(rho_thin) + prior(rho_thick),

with `sigma_eff^2 = sigma_Phi^2 + (K_z sigma_z)^2`, where `K_z` is the vertical
force of each model. Systematic errors are added in quadrature: the Wang et al.
rotation-curve budget (5–20 per cent, depending on R) to every radial point
outside the Wang curve, whose errors already include it, and 5 per cent of
`Phi` to the vertical points. The local stellar densities have Gaussian priors,
`rho_thin(R0) ~ N(0.045, 0.003)` and `rho_thick(R0) ~ N(0.00525, 0.0006)`
Msun pc^-3. `BIC = -2 ln L_max + k ln N`.

Five radial data vectors (`vgrav.observations.radial_variant_arrays`):

| Variant | Radial points | Content |
|---|---|---|
| `all` | 152 | all model-independent rotation-curve points |
| `no_eilers` | 114 | without the Eilers/McGaugh curve, which overlaps the Wang curve |
| `no_wang` | 118 | without the Wang curve |
| `overlap_correlated` | 152 | all points; Wang and Eilers errors multiplied by sqrt(2) over their common radii |
| `wang_only` | 34 | the Wang et al. curve alone (78 points with the vertical data) |

**Framework 1 — one fit per baryonic reconstruction** (`scripts/likelihood_bic.py`,
`scripts/likelihood_bic_stvg.py`). MI, LW, B2 and MM are fitted separately; the
thin- and thick-disc factors are free within ±10 sigma of the prior centres.
Output: `outputs/likelihood_bic/likelihood_bic_fits.csv` (variant, config, model,
k, N, m2lnL, chi2, chi2_nu, BIC, zt, zk, rho_thin, rho_thick, theta).

**Framework 2 — MC100 ensembles** (`scripts/step2_bc.py --full-likelihood
--radial-variant V --no-approx`, `scripts/likelihood_bic_mc100_extra.py`). Each of
the 100 realisations of each ensemble is fitted with its own disc fixed.
`scripts/likelihood_bic_mc100_summary.py --write-csv` writes
`bic_per_draw.csv` per ensemble and the summary `summary_<variant>.txt`.

**Baryonic uncertainty.** `scripts/likelihood_bic_marginal.py` treats the
realisation as a nuisance parameter, `BIC_marg = -2 ln[(1/N) sum_i exp(-BIC_i/2)]`,
using the existing BIC values (no new fit).

### Field equations solved on the grid

QUMOND, STVG and refracted gravity are solved on the axisymmetric grid of the
Newtonian potential (R ≤ 70 kpc, |z| ≤ 20 kpc, 281 × 641 nodes), so that v_c and
Phi_z derive from a single potential:

- QUMOND: Poisson equation with the phantom density (`vgrav.models.predict_qumond_solver`).
- STVG: `Phi_STVG = Phi_N + alpha (Phi_N - Phi_Y)`, with
  `(nabla^2 - mu^2) Phi_Y = 4 pi G rho` for the same density
  (`vgrav.models.stvg_grid_kernel`).
- Refracted gravity: `nabla.[eps(rho) nabla phi] = 4 pi G rho` with the
  permittivity of Cesare et al. (2020, Eq. 5), on nested grids; the outer grid
  (1 kpc spacing, 400 kpc) carries the far field of an isolated system,
  `phi = -GM/(eps0 r)` (`vgrav.models.predict_refracted_gravity`).

The solver is `vgrav.solver.axisymmetric_operator` (conservative form of
`nabla.(eps nabla phi) - mu^2 phi`, factorised sparse matrix).

---

## Gravity models

| Key | Model | Free gravity parameters |
|---|---|---|
| `baryonic` | Newtonian gravity of the baryons | none |
| `mond_*` (framework 2), `qumond_*` (framework 1) | QUMOND with the Simple, Standard and MLS interpolation functions, solved | none |
| `veg_verlinde` | apparent dark matter of Verlinde (2016, Eq. 7.40), `M_D^2 = a0 r^2/(6G) d(M_B r)/dr`, `a0 = c H0`, added as a spherical halo | none |
| `veg_verlinde_free` | the same with `a0` free | a0 |
| `fr_bounded` | f(R)-type interpolation function `nu = 1 + delta_sc exp(-x/x_c)` with `delta_sc <= 1/3` | delta_sc, x_c |
| `reg_dms` | refracted gravity with the DiskMass Survey parameters `eps0 = 0.661`, `Q = 1.79`, `log10 rho_c = -24.54` g cm^-3 (Cesare et al. 2020) | none |
| `reg_free` | refracted gravity with `eps0`, `Q`, `rho_c` free within the priors of Cesare et al. (2020); framework 1 | eps0, Q, rho_c |
| `stvg_std` | STVG with `alpha = 8.89`, `mu = 0.04 kpc^-1` (Moffat & Rahvar 2013) | none |
| `cdm_nfw`, `cdm_einasto` | CDM haloes | 2 (NFW), 3 (Einasto) |
| `hmg_k1` (framework 2), `hmg_sp` (framework 1) | s-HMG | s |
| `e2d_k2` | sr_h-HMG | s, r_h |
| `hmg_sk2` | s/r_h-HMG: HMG radial field, E-2D vertical potential with r_h | s, r_h |

The disc factors (z_t, z_k) add two parameters in framework 1. Other keys in the
CSV files are kept for comparison: `stvg` (alpha and mu free), `fr_screened`
(amplitude unbounded), `refracted_gravity` (algebraic `nu = 1/eps(x)`),
`veg_fixed`, `veg_free` (interpolation-function forms of VEG) and `hmg_sr`.

---

## Statistical errors only

The chi2 statistic with statistical errors only, fixed discs and median chi2_nu
over the realisations is kept for comparison:

```bash
python scripts/bc_leads_table.py          # chi2_nu percentiles and leads per ensemble (outputs/bc_nbar*)
python scripts/step3_bc_fig2.py           # figure with these fits -> figs/fig2_reproduced_4fam.png
python scripts/step3_reproduce_fig2_table2.py
```

---

## Package contents

```
vertical_gravity/
├── pyproject.toml, requirements.txt, CITATION.cff, .zenodo.json, LICENSE
├── data/                     observational data, baryonic bands, Fig. 1 inputs
│   ├── fig2_observational_catalog.csv   radial catalogue (model-independent points used in the fits)
│   ├── wang2026_vertical_potential.csv  44 vertical-potential points
│   ├── baryon_band_nbar{1..4}.csv       weighted baryonic band of each ensemble
│   └── fig1_density_grid.npz, fig1_spiral_arms.csv
├── outputs/
│   ├── nbar{1..4}/                      baryonic realisations of each ensemble
│   ├── likelihood_bic/                  framework 1: likelihood_bic_fits.csv, summary.txt
│   ├── likelihood_bic_mc100/            framework 2: per variant and ensemble, bic_per_draw.csv,
│   │                                    likelihood_setup.json; all/bc_nbar{1,4}/ model CSVs for Fig. 2;
│   │                                    summary_<variant>.txt, marginal_all.txt
│   └── bc_nbar{1..4}/                   fits with statistical errors only
├── tables/likelihood_bic_tables.tex     rows of the BIC tables of the paper
├── figs/                                figures
├── vgrav/                               library
│   ├── observations.py   data vectors and radial variants
│   ├── baryonic.py       baryonic densities, bands, Monte Carlo realisations
│   ├── solver.py         axisymmetric Poisson/Helmholtz/permittivity solver
│   ├── models.py         gravity models
│   ├── chi2.py           likelihood terms
│   └── figures.py        Figs. 1 and 2
└── scripts/
    ├── run_likelihood_bic.sh           full model comparison (both frameworks)
    ├── likelihood_bic.py               framework 1
    ├── likelihood_bic_stvg.py          framework 1, STVG
    ├── step2_bc.py                     framework 2
    ├── likelihood_bic_mc100_extra.py   framework 2, STVG, refracted gravity and VEG
    ├── likelihood_bic_mc100_summary.py framework 2, summaries and bic_per_draw.csv
    ├── likelihood_bic_marginal.py      Delta BIC marginalised over the baryons
    ├── likelihood_bic_tables.py        rows of the BIC tables
    ├── step3_bc_fig2_physical.py       Fig. 2
    ├── step0_reproduce_fig1.py         Fig. 1
    ├── run_all_nbar.py, rebuild_baryon_band.py, step1_build_baryonic_mc100.py
    │                                   baryonic ensembles
    └── bc_leads_table.py, step3_bc_fig2.py, step3_reproduce_fig2_table2.py,
        step2_fit_all_models.py, step4_analyze_hmg.py
                                        statistic with statistical errors only
```

---

## Citation

```bibtex
@article{monjosasi2026vertical,
  author  = {Monjo, Robert and Sasi, A. E.},
  title   = {Radial--vertical gravity decoupling in the {Milky Way}:
             modified gravity and dark matter tests using {Gaia DR3}},
  year    = {2026},
  note    = {in preparation},
}
```

## Licence

MIT (`LICENSE`).
