"""Monitor progress of run_all_nbar.py sequential run.

Usage (run from release_final/ or any subdirectory):
    python scripts/monitor_chi2.py             # one-shot snapshot
    python scripts/monitor_chi2.py --watch 300 # repeat every 300 s

Reads:
  outputs/run_sequential.log    — live tail of the running process
  outputs/nbarN/mc100_baryonic_*.csv — step1 chi2 (appears after ~2 h)
  outputs/nbarN/model_*.csv     — step2 chi2 per model (~2-8 h later)
  outputs/nbarN/mc100_chi2_all_models.csv — final step3 summary
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs"
LOG_SEQ = OUT / "run_sequential.log"
LOG_ERR = OUT / "run_sequential.err"

N_PRIMARY = 196

MODEL_K: dict[str, int] = {
    "baryonic":          0,
    "qumond_simple":     0,
    "qumond_standard":   0,
    "qumond_mls":        0,
    "veg_fixed":         0,
    "veg_free":          1,
    "stvg":              2,
    "cdm_nfw":           2,
    "cdm_einasto":       3,
    "hmg_k1":            1,
    "fr_screened":       2,
    "refracted_gravity": 2,
}

MODEL_LABEL: dict[str, str] = {
    "baryonic":          "Baryonic Newtonian",
    "qumond_simple":     "QUMOND (simple)",
    "qumond_standard":   "QUMOND (standard)",
    "qumond_mls":        "QUMOND (mls)",
    "veg_fixed":         "VEG fixed",
    "veg_free":          "VEG free",
    "stvg":              "STVG",
    "cdm_nfw":           "CDM NFW",
    "cdm_einasto":       "CDM Einasto",
    "hmg_k1":            "HMG k=1",
    "fr_screened":       "f(R) screened",
    "refracted_gravity": "Refracted gravity",
}


# ── CSV readers ───────────────────────────────────────────────────────────────

def _draw_cols(header: list[str]) -> list[int]:
    return [i for i, h in enumerate(header) if h.startswith("b")]


def read_chi2_last_row(path: Path) -> list[float]:
    """Return per-draw chi2 values from the last row of a step1/step2 CSV."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return []
        header, last = rows[0], rows[-1]
        dc = _draw_cols(header)
        return [float(last[i]) for i in dc if last[i].strip()]
    except Exception:
        return []


def read_combined_chi2(path: Path) -> dict[str, list[float]]:
    """Read mc100_chi2_all_models.csv → {model_key: [chi2_nu per draw]}."""
    result: dict[str, list[float]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                result.setdefault(row["model_key"], []).append(float(row["chi2_nu"]))
    except Exception:
        pass
    return result


def fmt_nu(vals: list[float], k: int) -> str:
    if not vals:
        return "—"
    arr = np.array(vals) / max(N_PRIMARY - k, 1)
    p16, p50, p84 = np.percentile(arr, [16, 50, 84])
    return f"{p50:.3f}  [{p16:.3f} – {p84:.3f}]"


def fmt_chi2_raw(vals: list[float]) -> str:
    if not vals:
        return "—"
    p16, p50, p84 = np.percentile(vals, [16, 50, 84])
    return f"{p50:.1f}  [{p16:.1f} – {p84:.1f}]"


# ── Snapshot logic ────────────────────────────────────────────────────────────

def snapshot() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  vertical_gravity_hmg — chi2 monitor  [{now}]")
    print(f"{'='*70}\n")

    for nbar in [1, 2, 3, 4]:
        outdir = OUT / f"nbar{nbar}"
        print(f"nbar={nbar}  {'-'*60}")
        if not outdir.exists():
            print("  (no output directory yet)\n")
            continue

        # ── Step 1: baryonic Newtonian ────────────────────────────────────
        rad_p  = outdir / "mc100_baryonic_radial.csv"
        vert_p = outdir / "mc100_baryonic_vertical.csv"
        bary_done = rad_p.exists() and vert_p.exists()
        if bary_done:
            cr = read_chi2_last_row(rad_p)
            cv = read_chi2_last_row(vert_p)
            n  = min(len(cr), len(cv))
            if n > 0:
                tot = [cr[i] + cv[i] for i in range(n)]
                print(f"  [step1 DONE  {n} draws]")
                print(f"  Baryonic  chi2_rad  = {fmt_chi2_raw(cr[:n])}")
                print(f"  Baryonic  chi2_vert = {fmt_chi2_raw(cv[:n])}")
                print(f"  Baryonic  chi2_nu   = {fmt_nu(tot, 0)}  (k=0)")
            else:
                print("  [step1 CSVs present — chi2 row not yet written]")
        else:
            print("  [step1 running or not started]")

        # ── Step 2: gravity models ────────────────────────────────────────
        comb_p = outdir / "mc100_chi2_all_models.csv"
        if comb_p.exists():
            # Final combined CSV (written by step3)
            comb = read_combined_chi2(comb_p)
            print(f"  [step2+step3 DONE — {comb_p.name}]")
            for key in MODEL_K:
                if key in comb:
                    vals = comb[key]
                    label = MODEL_LABEL.get(key, key)
                    k = MODEL_K[key]
                    print(f"  {label:<26} k={k}  chi2_nu = "
                          f"{np.median(vals):.3f}  [{np.percentile(vals,16):.3f} – {np.percentile(vals,84):.3f}]")
        else:
            # Try individual model CSVs (step2 partial)
            done_models = []
            for key in MODEL_K:
                if key == "baryonic":
                    continue
                rp = outdir / f"model_{key}_radial.csv"
                vp = outdir / f"model_{key}_vertical.csv"
                if rp.exists() and vp.exists():
                    cr_m = read_chi2_last_row(rp)
                    cv_m = read_chi2_last_row(vp)
                    n = min(len(cr_m), len(cv_m))
                    if n > 0:
                        tot_m = [cr_m[i] + cv_m[i] for i in range(n)]
                        label = MODEL_LABEL.get(key, key)
                        k     = MODEL_K[key]
                        print(f"  {label:<26} k={k}  chi2_nu = {fmt_nu(tot_m, k)}")
                        done_models.append(key)
            if not done_models:
                print("  [step2 not started or running]")

        print()

    # ── Log tail ──────────────────────────────────────────────────────────
    print(f"{'-'*70}")
    print(f"  run_sequential.log — last 30 lines")
    print(f"{'-'*70}")
    if LOG_SEQ.exists():
        lines = LOG_SEQ.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in lines[-30:]:
            print(f"  {ln}")
    else:
        print("  (log not found — run not started yet)")

    if LOG_ERR.exists():
        err = LOG_ERR.read_text(encoding="utf-8", errors="replace").strip()
        if err:
            print(f"\n  [ERRORS in run_sequential.err]:")
            for ln in err.splitlines()[-10:]:
                print(f"  {ln}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", type=int, default=0, metavar="SEC",
                        help="Repeat snapshot every SEC seconds (0 = one-shot, default).")
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Watching every {args.watch} s — Ctrl+C to stop.")
        try:
            while True:
                snapshot()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        snapshot()


if __name__ == "__main__":
    main()
