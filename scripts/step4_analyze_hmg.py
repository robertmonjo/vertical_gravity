"""Step 4 of 4 — HMG competitive-realization analysis.

Reads mc100_chi2_all_models.csv (produced by step3) and characterises
the 100 qcopula baryonic realizations in terms of the HMG/Einasto chi2_nu
ratio, reporting:

  - Realizations where HMG wins outright (ratio < 1.00)
  - Close realizations (ratio < 1.10): HMG within 10% of CDM Einasto
  - Percentile distributions of s_hmg, ratio, chi2_nu

Outputs
-------
  outputs/hmg_competitive_analysis.csv
      Columns: draw_id, draw_label, s_hmg, chi2_nu_cdm_einasto,
               chi2_nu_hmg_k1, ratio_hmg_einasto, winner, close_10pct

Usage
-----
  python scripts/step4_analyze_hmg.py
  python scripts/step4_analyze_hmg.py --s-detail fig2c_hmg_common_s_mc100_detail.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"


def _load_chi2_summary(chi2_csv: Path) -> tuple[dict, dict]:
    ein_chi2: dict[int, float] = {}
    hmg_chi2: dict[int, float] = {}
    with open(chi2_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            did = int(row["draw_id"])
            key = row["model_key"]
            nu = float(row["chi2_nu"])
            if key == "cdm_einasto":
                ein_chi2[did] = nu
            elif key == "hmg_k1":
                hmg_chi2[did] = nu
    return ein_chi2, hmg_chi2


def _load_s_detail(path: Path) -> dict[int, float]:
    """Load per-draw s values from fig2c_hmg_common_s_mc100_detail.csv."""
    s_map: dict[int, float] = {}
    if not path.exists():
        return s_map
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                # scenario column = "fig2b_NN"
                did = int(row["scenario"].split("_")[-1])
                s_map[did] = float(row["s"])
            except (KeyError, ValueError):
                pass
    return s_map


def _pct_str(arr: np.ndarray, pcts=(5, 16, 50, 84, 95)) -> str:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return "(no data)"
    return "  ".join(f"p{p}={np.percentile(valid, p):.3f}" for p in pcts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s-detail", default=None,
                        help="Path to fig2c_hmg_common_s_mc100_detail.csv "
                             "(default: outputs/fig2c_hmg_common_s_mc100_detail.csv)")
    args = parser.parse_args()

    print("Step 4 — HMG competitive-realization analysis")

    chi2_path = OUT / "mc100_chi2_all_models.csv"
    if not chi2_path.exists():
        print(f"  ERROR: {chi2_path.name} not found.  Run step3 first.")
        return

    ein_chi2, hmg_chi2 = _load_chi2_summary(chi2_path)
    if not ein_chi2 or not hmg_chi2:
        print("  ERROR: cdm_einasto or hmg_k1 not found in chi2 summary.")
        return

    s_detail_path = Path(args.s_detail) if args.s_detail else OUT / "fig2c_hmg_common_s_mc100_detail.csv"
    s_map = _load_s_detail(s_detail_path)
    if s_map:
        print(f"  Loaded s values for {len(s_map)} draws from {s_detail_path.name}")
    else:
        print("  No s-detail file found — s_hmg will be NaN in output.")

    draw_ids = sorted(ein_chi2.keys())
    n_draws = len(draw_ids)
    ratio = np.array([hmg_chi2[d] / ein_chi2[d] for d in draw_ids])
    s_arr = np.array([s_map.get(d, float("nan")) for d in draw_ids])
    winner = ["HMG" if r < 1.0 else "CDM Einasto" for r in ratio]
    close = ratio < 1.10

    # ── Write output CSV ───────────────────────────────────────────────────────
    out_path = OUT / "hmg_competitive_analysis.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["draw_id", "draw_label", "s_hmg",
                    "chi2_nu_cdm_einasto", "chi2_nu_hmg_k1",
                    "ratio_hmg_einasto", "winner", "close_10pct"])
        for i, d in enumerate(draw_ids):
            w.writerow([
                d,
                f"b{d}",
                f"{s_arr[i]:.4f}" if not np.isnan(s_arr[i]) else "",
                f"{ein_chi2[d]:.4f}",
                f"{hmg_chi2[d]:.4f}",
                f"{ratio[i]:.4f}",
                winner[i],
                "yes" if close[i] else "no",
            ])
    print(f"Written: {out_path.name}  ({n_draws} rows)")

    # ── Report ─────────────────────────────────────────────────────────────────
    n_win = int((ratio < 1.0).sum())
    n_close = int(close.sum())
    n_bad = int((ratio > 2.0).sum())
    print(f"\nHMG wins (ratio < 1.00)         : {n_win}/{n_draws}  ({100*n_win/n_draws:.0f}%)")
    print(f"Close (ratio < 1.10)             : {n_close}/{n_draws}  ({100*n_close/n_draws:.0f}%)")
    print(f"HMG >= 2x worse (ratio > 2.0)   : {n_bad}/{n_draws}")

    def _report(label: str, mask: np.ndarray) -> None:
        sv = s_arr[mask]
        rv = ratio[mask]
        n = int(mask.sum())
        print(f"\n  --- {label} (n={n}) ---")
        print(f"  ratio: {_pct_str(rv)}")
        if not np.all(np.isnan(sv)):
            print(f"  s    : {_pct_str(sv)}")
            print(f"  s > 2.5: {int((sv > 2.5).sum())}/{n}  ({100*(sv>2.5).sum()/n:.0f}%)")
            print(f"  s > 2.3: {int((sv > 2.3).sum())}/{n}  ({100*(sv>2.3).sum()/n:.0f}%)")

    all_mask = np.ones(n_draws, dtype=bool)
    _report("All realizations", all_mask)
    _report("Close (ratio < 1.10)", close)
    _report("  HMG wins (ratio < 1.00)", ratio < 1.00)
    _report("  Near-tie (1.00-1.10)", close & (ratio >= 1.00))
    _report("Losing (ratio >= 1.10)", ~close)

    if not np.all(np.isnan(s_arr[close])):
        print("\n  --- s threshold scan for close realizations ---")
        for thresh in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]:
            sv_close = s_arr[close]
            n_above = int((sv_close > thresh).sum())
            print(f"  s > {thresh:.1f}: {n_above}/{n_close}  ({100*n_above/n_close:.0f}%)")


if __name__ == "__main__":
    main()
