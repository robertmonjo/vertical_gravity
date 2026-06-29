# vgrav — Milky Way vertical gravity benchmark

Reproducibility package for:

> Monjo & Banik (2026). *Vertical gravitational potential of the Milky Way
> as a multi-model benchmark.* Manuscript in preparation.

Tests 11 model variants from 7 gravity frameworks (QUMOND, STVG,
CDM NFW/Einasto, f(R) screened, Refracted Gravity, VEG, and HMG)
against Wang et al. (2026) rotation-curve and vertical-potential data.

## Installation

```bash
pip install -e .
```

## Publication Pipeline

The pipeline runs in sequential steps; steps 1a and 1b are independent
and can run in parallel.

### Step 0 — Baryonic band

```bash
python scripts/rebuild_baryon_band.py --nbar 4
```

Rebuilds `data/baryon_band.csv` with the four-source kernel-weighted
hybrid band (MI + LW + B2 + MM).

### Step 1a — MC100 stochastic ensemble

```bash
python scripts/step1_build_baryonic_mc100.py
```

Generates 100 mass-constrained stochastic baryonic realizations.
Required by steps 2 and 3.

### Step 1b — Wang 78-point subset with jointly optimised disc scales

```bash
python scripts/step1_table2_wang78_joint_disc.py
```

Fits all gravity models to the Wang 78-point vertical-potential subset
with thin-disc and thick-disc scale factors optimised jointly with the
gravity-model parameters. Produces the chi²_nu summary reported in
Table 2 of the manuscript.

```bash
# Print the LaTeX table body from an existing CSV without recomputing:
python scripts/step1_table2_wang78_joint_disc.py --table-only
```

### Step 2 — Gravity model fits (MC100 ensemble)

```bash
python scripts/step2_fit_all_models.py
```

Fits all gravity models to the 100 stochastic baryonic realizations on
the full 196-point radial–vertical vector. Individual solver scripts are
also available for parallel execution:

```bash
python scripts/step2_qumond_simple.py
python scripts/step2_qumond_standard.py
python scripts/step2_qumond_mls.py
python scripts/step2_stvg.py                # sequential, single process
python scripts/step2_stvg_parallel.py       # recommended: 12 workers, ~83 min/nbar
```

### Step 3 — Chi-squared summary and Fig. 2

```bash
python scripts/step3_reproduce_fig2_table2.py
```

Reads the per-draw model CSVs from `outputs/`, computes the chi²_nu
percentile summary (p16/p50/p84 over the MC100 ensemble), and generates
the reproduced Fig. 2.

### Step 4 — HMG competitive-realization analysis

```bash
python scripts/step4_analyze_hmg.py
```

Characterizes the 100 baryonic realizations by the HMG/Einasto chi²_nu
ratio and identifies the realizations in which HMG is preferred.

### Full multi-nbar run

```bash
python scripts/run_all_nbar.py
```

Runs steps 1a through 3 for all four nbar variants (MI only, MI+LW,
MI+LW+B2, MI+LW+B2+MM) and writes a combined chi²_nu summary to
`outputs/table2_all_nbar.txt`.

## Diagnostic Shortcuts

The following flags are available for testing and must not be used for
the publication table unless explicitly documented as diagnostics.

```bash
# Diagnostic only: algebraic QUMOND proxies
python scripts/step2_fit_all_models.py --full --approx-qumond

# Diagnostic only: skip slow STVG solver
python scripts/step2_fit_all_models.py --full --no-stvg --resume

# Diagnostic only: few-draw smoke test
python scripts/step2_fit_all_models.py --full --n-draws 3 --no-stvg
```

## Acceptance Checks

A publication rerun is acceptable only when:

- the radial fit vector has exactly 152 model-independent points;
- the vertical fit vector has exactly 44 Wang et al. potential points;
- the MC100 baryonic band matches the manuscript consolidated band at
  inner and outer radii;
- the same baryonic realization is used for radial `v_N(R)` and vertical
  `Phi_N(R,z)`;
- all model statistics use `chi2_nu = chi2 / (N - n)` with `N = 196`;
- HMG uses the common neighbourhood scale `s > 1` and `gamma_cen = pi/2`
  by default.

## Main Outputs

After a successful publication rerun, the principal generated files are:

**Step 1b outputs (Table 2 of the manuscript):**

- `outputs/wang78_table2_joint_disc.csv` — chi²_nu per baryonic
  reconstruction and gravity model, Wang 78-point subset, disc scales
  jointly optimised

**Step 2–3 outputs (Fig. 2 and the MC100 chi²_nu summary):**

- `outputs/nbar4/mc100_chi2_all_models.csv`
- `outputs/nbar4/model_*_radial.csv`
- `outputs/nbar4/model_*_vertical.csv`
- `figs/fig2_reproduced.png`
