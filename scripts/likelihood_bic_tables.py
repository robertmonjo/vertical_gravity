"""LaTeX rows for the full-likelihood tables.

Table A: one fit per baryonic reconstruction (outputs/likelihood_bic/likelihood_bic_fits.csv);
         chi2_nu and Delta BIC with respect to the best model of each reconstruction.
Table B: MC100 ensembles (outputs/likelihood_bic_mc100/<variant>/bc_nbar<n>/bic_per_draw.csv);
         median Delta BIC with respect to the best model of each draw (p16, p84) and
         the number of draws in which the model has the lowest BIC.
Table C: MC100 ensembles; Delta BIC marginalised over the baryonic draws,
         -2 ln[(1/N) sum_i exp(-BIC_i/2)] relative to the best model.

Usage
-----
  python scripts/likelihood_bic_tables.py [--variant all] > tables/likelihood_bic_tables.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["mcgaugh_imig", "lian_wang", "de_salas", "mcmillan"]
ROWS_A = [
    ("baryonic", "Baryonic Newtonian"), ("qumond_simple", "QUMOND Simple"),
    ("qumond_standard", "QUMOND Standard"), ("qumond_mls", "QUMOND MLS"),
    ("veg_verlinde", r"VEG ($a_0=cH_0$)"), ("veg_verlinde_free", r"VEG free $a_0$"),
    ("fr_bounded", r"$f(\mathcal{R})$ ($\delta_{\rm sc}\le1/3$)"),
    ("reg_dms", "ReG (DiskMass)"), ("reg_free", "ReG free"), ("stvg_std", "STVG (universal)"),
    ("cdm_nfw", "CDM-NFW"), ("cdm_einasto", "CDM-Einasto"),
    ("hmg_sp", r"$s\text{-}$HMG (This Work)"), ("hmg_sk2", r"$s/r_\text{h}\text{-}$HMG (This Work)"),
    ("e2d_k2", r"$sr_\text{h}\text{-}$HMG (This Work)"),
]
ROWS_B = [
    ("baryonic", "Baryonic Newtonian", 0), ("mond_simple", "QUMOND Simple", 0),
    ("mond_standard", "QUMOND Standard", 0), ("mond_rar", "QUMOND MLS", 0),
    ("veg_verlinde", r"VEG ($a_0=cH_0$)", 0), ("veg_verlinde_free", r"VEG free $a_0$", 1),
    ("fr_bounded", r"$f(\mathcal{R})$ ($\delta_{\rm sc}\le1/3$)", 2),
    ("reg_dms", "ReG (DiskMass)", 0), ("stvg_std", "STVG (universal)", 0),
    ("cdm_nfw", "CDM-NFW", 2), ("cdm_einasto", "CDM-Einasto", 3),
    ("hmg_k1", r"$s\text{-}$HMG (This Work)", 1), ("hmg_sk2", r"$s/r_\text{h}\text{-}$HMG (This Work)", 2),
    ("e2d_k2", r"$sr_\text{h}\text{-}$HMG (This Work)", 2),
]


def fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}"


def table_a(variant: str) -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "outputs/likelihood_bic/likelihood_bic_fits.csv", encoding="utf-8"))
            if r["variant"] == variant]
    shown = {key for key, _ in ROWS_A}
    best = {c: min(float(r["BIC"]) for r in rows if r["config"] == c and r["model"] in shown) for c in CONFIGS}
    print(f"% Table A, variant {variant}: chi2_nu & Delta BIC per reconstruction (MI, LW, B2, MM)")
    for key, label in ROWS_A:
        cells, k = [], None
        for c in CONFIGS:
            hit = next(r for r in rows if r["config"] == c and r["model"] == key)
            k = hit["k"]
            d = float(hit["BIC"]) - best[c]
            dbic = r"\textbf{0}" if d == 0 else fmt(d)
            cells.append(f"{float(hit['chi2_nu']):.2f} & {dbic}")
        print(f"{label} & {k} & " + " & ".join(cells) + r" \\")
    hmg = {c: next(r for r in rows if r["config"] == c and r["model"] == "hmg_sp") for c in CONFIGS}
    print("% s-HMG best-fit s per reconstruction: "
          + ", ".join(f"{c}: {math.exp(json.loads(hmg[c]['theta'])[0]):.2f}" for c in CONFIGS))


def table_b(variant: str) -> None:
    print(f"% Table B, variant {variant}: median Delta BIC (+p84 -p16) and wins per ensemble")
    data = []
    for n in (1, 2, 3, 4):
        path = ROOT / f"outputs/likelihood_bic_mc100/{variant}/bc_nbar{n}/bic_per_draw.csv"
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        keys = [k for k, _, _ in ROWS_B if k in rows[0]]
        bic = {k: np.array([float(r[k]) for r in rows]) for k in keys}
        stack = np.vstack([bic[k] for k in keys])
        data.append((keys, bic, stack.min(axis=0), np.argmin(stack, axis=0)))
    for key, label, k in ROWS_B:
        cells = []
        for keys, bic, low, arg in data:
            d = bic[key] - low
            p16, p50, p84 = np.percentile(d, [16, 50, 84])
            wins = int(np.sum(arg == keys.index(key)))
            cell = r"\textbf{0}" if p84 == 0 else f"${fmt(p50)}^{{+{fmt(p84 - p50)}}}_{{-{fmt(p50 - p16)}}}$"
            cells.append(cell + (rf"\ ({wins})" if wins else ""))
        print(f"{label} & {k} & " + " & ".join(cells) + r" \\")


def table_c(variant: str) -> None:
    """Delta BIC marginalised over the 100 draws of each ensemble (likelihood_bic_marginal.py)."""
    print(f"% Table C, variant {variant}: Delta BIC marginalised over the baryonic draws, per ensemble")
    marg = []
    for n in (1, 2, 3, 4):
        path = ROOT / f"outputs/likelihood_bic_mc100/{variant}/bc_nbar{n}/bic_per_draw.csv"
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        keys = [k for k, _, _ in ROWS_B if k in rows[0]]
        m = {}
        for key in keys:
            b = np.array([float(r[key]) for r in rows])
            m[key] = float(b.min() - 2.0 * math.log(np.mean(np.exp(-0.5 * (b - b.min())))))
        best = min(m.values())
        marg.append({k: v - best for k, v in m.items()})
    for key, label, k in ROWS_B:
        cells = [r"\textbf{0}" if d[key] == 0 else fmt(d[key]) for d in marg]
        print(f"{label} & {k} & " + " & ".join(cells) + r" \\")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="all")
    args = parser.parse_args()
    table_a(args.variant)
    print()
    table_b(args.variant)
    print()
    table_c(args.variant)


if __name__ == "__main__":
    main()
