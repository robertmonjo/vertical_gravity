"""Reproduce Table 3: BC chi2_nu (p50 with p16/p84) and per-draw leads.

Usage
-----
  python scripts/bc_leads_table.py

For each baryonic ensemble (MI, MI+LW, MI+LW+B2, MI+LW+B2+MM) the per-model
BC outputs in outputs/bc_nbar{1..4}/ are staged with the same routine used by
step3_bc_fig2.py, which writes outputs/bc_fig2_nbar{n}/mc100_chi2_all_models.csv
from the chi2 values stored in each model CSV. A lead is a draw in which the
model has the lowest chi2_nu among the 13 models of Table 3.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

NBAR_LABELS = ["MI", "MI+LW", "MI+LW+B2", "MI+LW+B2+MM"]

MODEL_ORDER = [
    ("baryonic",          "Baryonic Newtonian"),
    ("qumond_simple",     "QUMOND Simple"),
    ("qumond_standard",   "QUMOND Standard"),
    ("qumond_mls",        "QUMOND MLS"),
    ("veg_fixed",         "VEG original"),
    ("veg_free",          "VEG free"),
    ("fr_screened",       r"$f(\mathcal{R})$ screened"),
    ("refracted_gravity", "ReG"),
    ("stvg",              "STVG"),
    ("cdm_nfw",           "CDM-NFW"),
    ("cdm_einasto",       "CDM-Einasto"),
    ("hmg_sp",            r"$s$-HMG"),
    ("e2d_k2",            r"$sr_{\rm h}$-HMG"),
]


def _load_step3():
    spec = importlib.util.spec_from_file_location("step3_bc_fig2", ROOT / "scripts" / "step3_bc_fig2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_chi2(step3, n: int) -> dict[str, list[float]]:
    dest = ROOT / "outputs" / f"bc_fig2_nbar{n}"
    with contextlib.redirect_stdout(io.StringIO()):
        step3._setup_bc_dir(ROOT / "outputs" / f"bc_nbar{n}", ROOT / "outputs" / f"nbar{n}", dest)
        step3._build_chi2_csv(dest)
    by_model: dict[str, list[float]] = defaultdict(list)
    with open(dest / "mc100_chi2_all_models.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_model[row["model_key"]].append(float(row["chi2_nu"]))
    return dict(by_model)


def compute_leads(data: dict[str, list[float]]) -> dict[str, int]:
    keys = [k for k, _ in MODEL_ORDER if k in data]
    leads = {k: 0 for k in keys}
    for i in range(min(len(data[k]) for k in keys)):
        leads[min(keys, key=lambda k: data[k][i])] += 1
    return leads


def fmt_cell(values: list[float], leads: int) -> str:
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    lo, hi = p50 - p16, p84 - p50
    digits = 0 if p50 >= 10 else 2
    cell = f"${p50:.{digits}f}^{{+{hi:.{digits}f}}}_{{-{lo:.{digits}f}}}$"
    return cell + (f"\\ ({leads})" if leads else "")


def main() -> None:
    step3 = _load_step3()
    data = [stage_chi2(step3, n) for n in (1, 2, 3, 4)]
    leads = [compute_leads(d) for d in data]
    print("Model & " + " & ".join(NBAR_LABELS) + r" \\")
    for key, label in MODEL_ORDER:
        cells = [fmt_cell(d[key], l.get(key, 0)) if key in d else "---" for d, l in zip(data, leads)]
        print(f"{label} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
