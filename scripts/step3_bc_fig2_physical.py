#!/usr/bin/env python3
"""Fig. 2 from the full-likelihood band-consistent fits with the physical models.

Same representation as step3_bc_fig2.py, but every curve comes from
outputs/likelihood_bic_mc100/<variant>/bc_nbar4/ (four-source ensemble) and the
s-HMG+Imig overlay from .../bc_nbar1/, so that the models are those of Table 5:

  QUMOND (Simple, Standard, MLS)  solved with its Poisson equation
  VEG                             apparent dark matter of Verlinde (2016), free a0
  STVG                            alpha = 8.89, mu = 0.04 kpc^-1 (Moffat & Rahvar 2013)
  f(R)                            delta_sc <= 1/3
  ReG                             solved on nested grids, DiskMass parameters (Cesare et al. 2020)
  CDM NFW and Einasto, sr_h-HMG, s/r_h-HMG

The legend gives, for each model, Delta BIC with the BIC marginalised over the
100 draws of the ensemble, -2 ln[(1/N) sum_i exp(-BIC_i/2)], so that the
baryonic uncertainty is included (Table 5, marginalised columns). The
s-HMG+Imig overlay comes from the MI ensemble and carries no value, since its
Delta BIC refers to another set of draws.

Requires the model CSVs written by run_likelihood_bic.sh (not bundled; only
bic_per_draw.csv is).

Usage
-----
  python scripts/step3_bc_fig2_physical.py            # figs/fig2_physical_models.png
  python scripts/step3_bc_fig2_physical.py --variant no_eilers
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.figures import plot_fig2
sys.path.insert(0, str(ROOT / "scripts"))
from step3_bc_fig2 import _ensure_baryon_band  # noqa: E402

# figure key -> suffix of model_bc_<suffix>_{radial,vertical}.csv in the likelihood run
KEY_MAP: dict[str, str] = {
    "baryonic":          "baryonic",
    "qumond_simple":     "mond_simple",
    "qumond_standard":   "mond_standard",
    "qumond_mls":        "mond_rar",
    "veg_free":          "veg_verlinde_free",
    "stvg":              "stvg_std",
    "fr_screened":       "fr_bounded",
    "refracted_gravity": "reg_dms",
    "cdm_nfw":           "cdm_nfw",
    "cdm_einasto":       "cdm_einasto",
    "hmg_sp":            "hmg_k1",
    "e2d_k2":            "e2d_k2",
    "hmg_sk2":           "hmg_sk2",
}
FIGURE_KEYS = set(KEY_MAP) - {"hmg_sp"}
LABELS: dict[str, str] = {
    "veg_free":    r"VEG (Verlinde, free $a_0$)",
    "stvg":        "STVG (universal)",
    "refracted_gravity": "ReG (DiskMass)",
    "fr_screened": r"$f(\mathcal{R})$ ($\delta_{\rm sc}\leq 1/3$)",
}


def stage(src: Path, nbar_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for key, suffix in KEY_MAP.items():
        for side in ("radial", "vertical"):
            p = src / f"model_bc_{suffix}_{side}.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            shutil.copy2(p, dest / f"model_{key}_{side}.csv")
    for name in ("mc100_baryonic_radial.csv", "mc100_baryonic_vertical.csv"):
        shutil.copy2(nbar_dir / name, dest / name)


def marginal_delta_bic(directory: Path) -> dict[str, float]:
    """Delta BIC with each BIC marginalised over the draws: -2 ln[(1/N) sum_i exp(-BIC_i/2)]."""
    with open(directory / "bic_per_draw.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keys = [k for k in rows[0] if k != "draw"]
    marg = {}
    for k in keys:
        b = np.array([float(r[k]) for r in rows])
        marg[k] = float(b.min() - 2.0 * np.log(np.mean(np.exp(-0.5 * (b - b.min())))))
    best = min(marg.values())
    inverse = {v: k for k, v in KEY_MAP.items()}
    return {inverse[k]: v - best for k, v in marg.items() if k in inverse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="all")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", default=None, help="default: figs/fig2_physical_models.png")
    args = parser.parse_args()

    run = ROOT / "outputs" / "likelihood_bic_mc100" / args.variant
    dest4 = ROOT / "outputs" / f"fig2_physical_{args.variant}_nbar4"
    dest1 = ROOT / "outputs" / f"fig2_physical_{args.variant}_nbar1"
    stage(run / "bc_nbar4", ROOT / "outputs" / "nbar4", dest4)
    stage(run / "bc_nbar1", ROOT / "outputs" / "nbar1", dest1)

    # baryonic band staged exactly as in step3_bc_fig2.py (same calls, same order)
    _ensure_baryon_band(dest4)
    _ensure_baryon_band(dest1)

    # the s-HMG+Imig overlay belongs to another ensemble: no value in the legend
    values = marginal_delta_bic(run / "bc_nbar4")
    for k, v in sorted(values.items(), key=lambda kv: kv[1]):
        print(f"  {k:<20} marginalised Delta BIC = {v:8.1f}")

    obs = ROOT / "data" / "fig2_observational_data.csv"
    if not obs.exists():
        obs = ROOT / "data" / "fig2_observational_catalog.csv"
    out = Path(args.output) if args.output else ROOT / "figs" / "fig2_physical_models.png"
    out.parent.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_fig2(dest4, obs_path=obs, output_path=out, dpi=args.dpi,
                    hmg_1fam_outputs_dir=dest1, model_keys=FIGURE_KEYS,
                    label_overrides=LABELS, legend_values=values,
                    legend_symbol=r"\Delta{\rm BIC}_{\rm marg}", legend_fmt=".1f",
                    model_legend_spacing=0.5)
    plt.close(fig)


if __name__ == "__main__":
    main()
