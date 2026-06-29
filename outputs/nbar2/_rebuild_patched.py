"""Rebuild baryon_band.csv: progressive kernel-weighted baryonic reconstruction ensemble.

Reconstruction progression (cumulative, --nbar N activates the first N baryonic reconstructions):
  1. McGaugh/Imig  (McGaugh2018_Imig2025)
  2. Wang/Lian     (Wang2026_Lian2022)
  3. deSalas B2    (deSalas2019_B2)
  4. McMillan      (McMillan2017)

Weights:
  nbar=1  →  w=1 (single reconstruction, no kernel)
  nbar>=2 →  w_i(R) ∝ 1/chi2_local_i(R) via log(R) kernel, sigma_log=0.35

Stratification: ±1σ (patches vgrav/baryonic.py if needed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BARYON_BAND = DATA / "baryon_band.csv"
OBS_FILE    = DATA / "fig2_observational_catalog.csv"
BARYONIC_PY = ROOT / "vgrav" / "baryonic.py"

# Cumulative reconstruction list — nbar=N activates RECONSTRUCTIONS_PROGRESSIVE[:N]
RECONSTRUCTIONS_PROGRESSIVE = [
    "McGaugh2018_Imig2025",
    "Wang2026_Lian2022",
    "deSalas2019_B2",
    "McMillan2017",
]

# All baryonic reconstructions present in the CSV (for center-column reads and output compatibility)
RECONSTRUCTIONS_ALL = [
    "McGaugh2018_Imig2025",
    "Wang2026_Lian2022",
    "McMillan2017",
    "deSalas2019_B2",
]

RECONSTRUCTION_SIGMA_INNER_OUTER = {
    "McGaugh2018_Imig2025": (0.025, 0.18),
    "Wang2026_Lian2022":    (0.035, 0.22),
    "deSalas2019_B2":       (0.050, 0.28),
    "McMillan2017":         (0.040, 0.24),
}

SIGMA_LOG     = 0.5
SIGMA_V_FLOOR = 3.0


def smoothstep(x):
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def fractional_sigma(reconstruction, radius):
    inner, outer = RECONSTRUCTION_SIGMA_INNER_OUTER[reconstruction]
    x = smoothstep((np.asarray(radius) - 15.0) / 85.0)
    return inner + (outer - inner) * x


def load_band(reconstructions):
    data = np.genfromtxt(BARYON_BAND, delimiter=",", names=True, dtype=None, encoding="utf-8")
    r = data["R_kpc"].astype(float)
    centers = {}
    for f in reconstructions:
        col = f"center_{f}"
        if col not in data.dtype.names:
            raise KeyError(
                f"Column '{col}' not found in {BARYON_BAND}. "
                "Ensure the CSV contains center columns for all requested reconstructions."
            )
        centers[f] = data[col].astype(float)
    centers_all = {
        f: data[f"center_{f}"].astype(float)
        for f in RECONSTRUCTIONS_ALL
        if f"center_{f}" in data.dtype.names
    }
    return r, centers, centers_all


def load_observations():
    import pandas as pd
    df   = pd.read_csv(OBS_FILE)
    mask = (df["kind"] == "direct_rotation") & (df["used_in_fit"] == True)  # noqa: E712
    sub  = df[mask]
    r_obs     = sub["R_kpc"].to_numpy(dtype=float)
    v_obs     = sub["vc_kms"].to_numpy(dtype=float)
    sigma_obs = np.maximum(sub["sigma_v_kms"].to_numpy(dtype=float), SIGMA_V_FLOOR)
    return r_obs, v_obs, sigma_obs


def compute_kernel_weights(reconstructions, r_band, centers, r_obs, v_obs, sigma_obs):
    res2 = {}
    for f in reconstructions:
        v_model = PchipInterpolator(r_band, centers[f])(r_obs)
        res2[f] = ((v_model - v_obs) / sigma_obs) ** 2

    log_r_obs  = np.log(r_obs)
    log_r_band = np.log(r_band)
    weights = {f: np.empty(len(r_band)) for f in reconstructions}

    for i, logR in enumerate(log_r_band):
        K = np.exp(-0.5 * ((log_r_obs - logR) / SIGMA_LOG) ** 2)
        K_sum = max(K.sum(), 1e-15)
        inv = {}
        for f in reconstructions:
            cl = max(float(np.dot(K, res2[f]) / K_sum), 1e-6)
            inv[f] = 1.0 / cl
        total = sum(inv.values())
        for f in reconstructions:
            weights[f][i] = inv[f] / total

    return weights


def build_hybrid(reconstructions, r, centers, weights):
    wstack = np.vstack([
        weights[f] if hasattr(weights[f], "__len__") else np.full(len(r), weights[f])
        for f in reconstructions
    ])
    stack  = np.vstack([centers[f] for f in reconstructions])
    center_mean = np.sum(wstack * stack, axis=0)

    within_var  = np.zeros(len(r))
    between_var = np.zeros(len(r))
    reconstruction_p5   = []
    reconstruction_p95  = []
    for idx, f in enumerate(reconstructions):
        w       = wstack[idx]
        sigma_f = fractional_sigma(f, r) * centers[f]
        within_var  += w * sigma_f ** 2
        between_var += w * (centers[f] - center_mean) ** 2
        reconstruction_p5.append(np.maximum(centers[f] - 1.645 * sigma_f, 0.0))
        reconstruction_p95.append(centers[f] + 1.645 * sigma_f)

    sigma_total = np.sqrt(within_var + between_var)
    p16 = np.maximum(center_mean - sigma_total, 0.0)
    p84 = center_mean + sigma_total
    p5  = np.min(np.vstack(reconstruction_p5),  axis=0)
    p95 = np.max(np.vstack(reconstruction_p95), axis=0)
    return center_mean, p5, p16, p84, p95


def save_band(r, centers_all, center_mean, p5, p16, p84, p95, reconstructions, weights, outfile=None):
    out_path = Path(outfile) if outfile else BARYON_BAND
    all_weights = {
        f: (weights[f] if f in reconstructions else np.zeros(len(r)))
        for f in RECONSTRUCTIONS_ALL
    }
    fields = (
        ["R_kpc", "hybrid_p5", "hybrid_p16", "hybrid_center", "hybrid_p84", "hybrid_p95"]
        + [f"weight_{f}" for f in RECONSTRUCTIONS_ALL]
        + [f"center_{f}" for f in RECONSTRUCTIONS_ALL]
    )
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(fields) + "\n")
        for i in range(len(r)):
            w_row = [
                float(all_weights[f][i]) if hasattr(all_weights[f], "__len__")
                else float(all_weights[f])
                for f in RECONSTRUCTIONS_ALL
            ]
            c_row = [float(centers_all[f][i]) if f in centers_all else 0.0
                     for f in RECONSTRUCTIONS_ALL]
            row = [r[i], p5[i], p16[i], center_mean[i], p84[i], p95[i]] + w_row + c_row
            fh.write(",".join(f"{float(x):.9g}" for x in row) + "\n")


def patch_stratification():
    text  = BARYONIC_PY.read_text(encoding="utf-8")
    old15 = "u_global = _norm.cdf(np.linspace(-1.5, 1.5, n_draws))"
    new10 = "u_global = _norm.cdf(np.linspace(-1.0, 1.0, n_draws))"
    if new10 in text:
        print("  baryonic.py: stratification already ±1σ.")
    elif old15 in text:
        BARYONIC_PY.write_text(text.replace(old15, new10, 1), encoding="utf-8")
        print("  baryonic.py: changed stratification to ±1σ.")
    else:
        print("  WARNING: linspace pattern not found in baryonic.py!")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nbar", type=int, default=3, choices=[1, 2, 3, 4],
        help="Number of baryonic reconstructions (1–4, default 3). "
             "Baryonic reconstructions added progressively: McGaugh → Wang → deSalas → McMillan.",
    )
    parser.add_argument(
        "--outfile", default=None, metavar="PATH",
        help="Write output baryon_band CSV to this path instead of data/baryon_band.csv. "
             "Useful for parallel nbar runs (leaves the main baryon_band.csv unchanged).",
    )
    args = parser.parse_args()

    reconstructions = RECONSTRUCTIONS_PROGRESSIVE[:args.nbar]

    sys.stdout.reconfigure(encoding="utf-8")
    short = {
        "McGaugh2018_Imig2025": "McGaugh/Imig",
        "Wang2026_Lian2022":    "Wang/Lian",
        "deSalas2019_B2":       "deSalas B2",
        "McMillan2017":         "McMillan",
    }
    rec_str = " + ".join(short[f] for f in reconstructions)
    print("=" * 68)
    print(f"rebuild_baryon_band.py  [nbar={args.nbar}: {rec_str}]")
    _wt = "1.0 (single reconstruction)" if args.nbar == 1 else f"kernel (sigma_log={SIGMA_LOG})"
    print(f"  weights: {_wt}")
    print("=" * 68)

    r, centers, centers_all = load_band(reconstructions)
    r_obs, v_obs, sigma_obs = load_observations()
    print(f"  Band: {len(r)} grid pts, R ∈ [{r.min():.1f}, {r.max():.1f}] kpc")
    print(f"  Obs:  {len(r_obs)} radial pts")

    if args.nbar == 1:
        weights = {reconstructions[0]: np.ones(len(r))}
    else:
        print("  Computing kernel weights...")
        weights = compute_kernel_weights(reconstructions, r, centers, r_obs, v_obs, sigma_obs)
        key_r  = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
        col_w  = 13
        header = "  " + f"{'R':>5}  " + "  ".join(f"{short[f]:>{col_w}}" for f in reconstructions)
        print(f"\n{header}")
        print("  " + "-" * (7 + (col_w + 2) * args.nbar))
        for R_check in key_r:
            idx  = np.argmin(np.abs(r - R_check))
            vals = "  ".join(f"{weights[f][idx]:>{col_w}.4f}" for f in reconstructions)
            print(f"  {r[idx]:>5.1f}  {vals}")

    center_mean, p5, p16, p84, p95 = build_hybrid(reconstructions, r, centers, weights)
    out_path = Path(args.outfile) if args.outfile else BARYON_BAND
    save_band(r, centers_all, center_mean, p5, p16, p84, p95, reconstructions, weights, outfile=out_path)
    print(f"\n  Saved: {out_path}")

    patch_stratification()
    print(f"\nDone. Run: python step1_build_baryonic_mc100.py --full")
    print("=" * 68)


if __name__ == "__main__":
    main()
