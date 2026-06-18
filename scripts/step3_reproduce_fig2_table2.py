"""Step 3 of 4 — Reproduce Fig. 2 and Table 2.

Reads all pre-computed CSVs from outputs/, computes the chi^2_nu summary
(Table 2), generates Fig. 2, and writes mc100_chi2_all_models.csv.

Usage
-----
  python scripts/step3_reproduce_fig2_table2.py
  python scripts/step3_reproduce_fig2_table2.py --no-figure
  python scripts/step3_reproduce_fig2_table2.py --output-dir /path/to/outputs

Expected outputs
----------------
  outputs/mc100_chi2_all_models.csv  — 1100 rows (100 draws × 11 models)
  figures/fig2_reproduced.png        — reproduction of Fig. 2

Verification
------------
Expected chi^2_nu values (p50, 100 qcopula draws, seed=20260607):
  Baryonic Newtonian    : ~194.7   (large: baryons alone cannot explain RC)
  QUMOND simple         : ~5.9
  QUMOND standard       : ~50.6
  QUMOND MLS/RAR        : ~7.7
  STVG                  : ~8.2
  CDM NFW               : ~3.3
  CDM Einasto           : ~2.8
  HMG anisotropic (k=1) : ~3.3
  f(R) screened  (k=2)  : ~8.1
  Refracted Gravity(k=2): ~8.9
  Emergent Gravity (k=0): ~23.6

Note: exact values depend on the baryonic ensemble used.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.chi2 import N_PRIMARY
from vgrav.figures import plot_fig2

OUT = ROOT / "outputs"
FIGS = ROOT / "figs"

# Model catalogue — must match the CSV naming from step2
MODEL_SPECS = [
    # key,                  display_name,                              k,  source
    ("baryonic",          "Baryonic Newtonian",                        0, "qcopula"),
    ("qumond_simple",     "QUMOND simple",                             0, "qcopula"),
    ("qumond_standard",   "QUMOND standard",                           0, "qcopula"),
    ("qumond_mls",        "QUMOND MLS/RAR",                            0, "old_stochastic"),
    ("stvg",              "STVG",                                      1, "qcopula"),
    ("cdm_nfw",           "CDM NFW",                                   2, "qcopula"),
    ("cdm_einasto",       "CDM Einasto",                               2, "qcopula"),
    ("hmg_k1",            "HMG anisotropic (k=1)",                     1, "npz"),
    ("fr_screened",       "f(R) screened",                             2, "npz"),
    ("refracted_gravity", "Refracted Gravity",                         2, "npz"),
    ("emergent_gravity",  "Emergent Gravity (fixed)",                   0, "npz"),
]
N_DRAWS = 100


def _read_chi2_from_csv(path: Path) -> np.ndarray:
    """Read chi2_radial or chi2_vertical last-row from a model CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    chi2_row = rows[-1]
    draw_idx = [i for i, h in enumerate(header) if h.startswith("b")]
    return np.array([float(chi2_row[i]) for i in draw_idx])


def _build_chi2_summary(outputs_dir: Path) -> list[dict]:
    """Compute chi2_nu per draw per model.  Returns rows for mc100_chi2_all_models.csv."""
    rows_out = []
    for key, name, k, source in MODEL_SPECS:
        rad_p = outputs_dir / f"model_{key}_radial.csv"
        vert_p = outputs_dir / f"model_{key}_vertical.csv"
        if not rad_p.exists() or not vert_p.exists():
            print(f"  MISSING: model_{key}_*.csv — skipping.")
            continue
        chi2_r = _read_chi2_from_csv(rad_p)
        chi2_z = _read_chi2_from_csv(vert_p)
        if len(chi2_r) != N_DRAWS or len(chi2_z) != N_DRAWS:
            print(f"  WARNING: {key} has {len(chi2_r)} radial / {len(chi2_z)} vertical draws (expected {N_DRAWS})")
        n = min(len(chi2_r), len(chi2_z))
        for i in range(n):
            chi2_tot = chi2_r[i] + chi2_z[i]
            rows_out.append({
                "draw_id":       i + 1,
                "draw_label":    f"b{i+1}",
                "model_key":     key,
                "model_name":    name,
                "k":             k,
                "source":        source,
                "dof":           N_PRIMARY - k,
                "chi2_radial":   f"{chi2_r[i]:.4f}",
                "chi2_vertical": f"{chi2_z[i]:.4f}",
                "chi2_total":    f"{chi2_tot:.4f}",
                "chi2_nu":       f"{chi2_tot / (N_PRIMARY - k):.4f}",
            })
    return rows_out


def _write_chi2_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path.name}  ({len(rows)} rows)")


def _print_table2(rows: list[dict]) -> None:
    model_chi2: dict[str, list] = defaultdict(list)
    model_k: dict[str, int] = {}
    model_name: dict[str, str] = {}
    for row in rows:
        key = row["model_key"]
        model_chi2[key].append(float(row["chi2_nu"]))
        model_k[key] = int(row["k"])
        model_name[key] = row["model_name"]

    print("\n" + "=" * 68)
    print(f"  Table 2 -- chi2_nu (reduced chi-squared) summary")
    print(f"  N_PRIMARY = {N_PRIMARY}  (152 radial + 44 vertical obs points)")
    print("-" * 68)
    print(f"  {'Model':<32} {'k':>3}  {'p16':>7} {'p50':>7} {'p84':>7}")
    print("-" * 68)
    order = [s[0] for s in MODEL_SPECS]
    for key in order:
        if key not in model_chi2:
            continue
        arr = np.array(model_chi2[key])
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        note = " *" if MODEL_SPECS[[s[0] for s in MODEL_SPECS].index(key)][3] == "old_stochastic" else ""
        print(f"  {model_name[key]+note:<32} {model_k[key]:>3}  {p16:7.3f} {p50:7.3f} {p84:7.3f}")
    print("-" * 68)
    print("  * uses old stochastic ensemble (not qcopula)")
    print("=" * 68 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(OUT),
                        help="Directory with model CSVs (default: outputs/)")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip Fig. 2 generation")
    args = parser.parse_args()

    outputs_dir = Path(args.output_dir)
    print("Step 3 — Reproducing Fig. 2 and Table 2")
    print(f"  Reading from: {outputs_dir}")

    rows = _build_chi2_summary(outputs_dir)
    if not rows:
        print("  No model CSVs found.  Run step1 + step2 first.")
        return

    chi2_path = outputs_dir / "mc100_chi2_all_models.csv"
    _write_chi2_summary(rows, chi2_path)
    _print_table2(rows)

    if not args.no_figure:
        FIGS.mkdir(exist_ok=True)
        obs_path = outputs_dir / "fig2_observational_data.csv"
        fig_path = FIGS / "fig2_reproduced.png"
        try:
            fig = plot_fig2(outputs_dir, obs_path=obs_path, output_path=fig_path, dpi=220)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"  Figure generation failed: {e}")

    print("Step 3 complete.  Run step4 for HMG competitive analysis.")


if __name__ == "__main__":
    main()
