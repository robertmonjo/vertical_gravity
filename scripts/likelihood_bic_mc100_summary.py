"""Summarise band-consistent MC100 fits run with step2_bc.py --full-likelihood.

For every draw, -2 ln L = chi2_radial + chi2_vertical (which already holds
-2 ln L_vert) + the model-independent radial log term saved in
likelihood_setup.json. BIC = -2 ln L + k ln N. The table gives, per ensemble,
the median Delta BIC with respect to the best model of each draw (p16, p84)
and the number of draws in which each model has the lowest BIC.

Usage
-----
  python scripts/likelihood_bic_mc100_summary.py outputs/likelihood_bic_mc100/all/bc_nbar{1,2,3,4}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

K = {"baryonic": 0, "mond_simple": 0, "mond_standard": 0, "mond_rar": 0, "veg_fixed": 0,
     "veg_free": 1, "veg_verlinde": 0, "veg_verlinde_free": 1, "hmg_k1": 1, "hmg_sr": 1, "e2d_k2": 2,
     "hmg_sk2": 2, "fr_screened": 2, "fr_bounded": 2, "refracted_gravity": 2, "cdm_nfw": 2,
     "cdm_einasto": 3, "stvg": 2, "stvg_std": 0, "reg_dms": 0}
COMPETITION = ["baryonic", "mond_simple", "mond_standard", "mond_rar", "veg_verlinde", "veg_verlinde_free",
               "fr_bounded", "reg_dms", "stvg_std", "cdm_nfw", "cdm_einasto",
               "hmg_k1", "hmg_sk2", "e2d_k2"]


def last_row(path: Path) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return np.array([float(x) for x in rows[-1] if x not in ("", "chi2_radial", "chi2_vertical")])


def ensemble(directory: Path) -> dict[str, np.ndarray]:
    setup = json.loads((directory / "likelihood_setup.json").read_text())
    n = setup["n_radial"] + setup["n_vertical"]
    bic = {}
    for key in COMPETITION:
        rad, vert = directory / f"model_bc_{key}_radial.csv", directory / f"model_bc_{key}_vertical.csv"
        if rad.exists() and vert.exists():
            m2lnl = last_row(rad) + last_row(vert) + setup["radial_log_term"]
            bic[key] = m2lnl + K[key] * math.log(n)
    return bic


def write_per_draw(directory: Path, bic: dict[str, np.ndarray]) -> None:
    keys = list(bic)
    with open(directory / "bic_per_draw.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["draw"] + keys)
        for j in range(len(bic[keys[0]])):
            w.writerow([f"b{j + 1}"] + [f"{bic[k][j]:.4f}" for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--write-csv", action="store_true",
                        help="also write bic_per_draw.csv (BIC of every model and draw) in each directory")
    args = parser.parse_args()
    data = [ensemble(Path(d)) for d in args.dirs]
    if args.write_csv:
        for d, bic in zip(args.dirs, data):
            write_per_draw(Path(d), bic)
    print("model".ljust(20) + "".join(f"{Path(d).name:>26}" for d in args.dirs))
    for key in COMPETITION:
        cells = []
        for bic in data:
            if key not in bic:
                cells.append(f"{'---':>26}")
                continue
            keys = list(bic)
            stack = np.vstack([bic[k] for k in keys])
            delta = bic[key] - stack.min(axis=0)
            wins = int(np.sum(np.argmin(stack, axis=0) == keys.index(key)))
            p16, p50, p84 = np.percentile(delta, [16, 50, 84])
            cells.append(f"{p50:9.1f} [{p16:6.1f},{p84:6.1f}] ({wins:3d})")
        print(key.ljust(20) + "".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
