"""Step 3 (BC variant) — Generate Fig. 2 using Band-Consistent (BC) outputs.

Reads the pre-computed BC per-model CSVs from outputs/bc_nbar{1,4}/, stages
them in transient directories with the naming convention expected by plot_fig2,
builds mc100_chi2_all_models.csv for each, and generates the figure.

Requires:
  outputs/bc_nbar4/   — BC outputs for nbar4 (four-source ensemble)
  outputs/bc_nbar1/   — BC outputs for nbar1 (McGaugh-Imig, for the overlay)

Output:
  figs/fig2_reproduced_4fam.png

Usage
-----
  python scripts/step3_bc_fig2.py              # generate figure
  python scripts/step3_bc_fig2.py --no-figure  # only build chi2 CSV
  python scripts/step3_bc_fig2.py --dpi 300

Notes
-----
The figure shows:
  - "HMG 2p (This Work)"   — e2d_k2 (s + r_hyd free, k=2) from nbar4 BC
  - "s-HMG+Imig (This Work)" — hmg_sr (s only, E-2D vertical, r_hyd→∞) from nbar1 BC (overlay)
  - s_p-HMG (hmg_sp) — point-mass approx., included in chi2 CSV but not shown in figure
  - All other models from nbar4 BC; STVG from old nbar4 pipeline (excluded from BC)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.chi2 import N_PRIMARY
from vgrav.figures import plot_fig2

# ── BC key mapping ─────────────────────────────────────────────────────────────
# step3_key → suffix used in model_bc_{suffix}_{radial|vertical}.csv
BC_KEY_MAP: dict[str, str] = {
    "baryonic":          "baryonic",
    "qumond_simple":     "mond_simple",
    "qumond_standard":   "mond_standard",
    "qumond_mls":        "mond_rar",
    "veg_fixed":         "veg_fixed",
    "veg_free":          "veg_free",
    "cdm_nfw":           "cdm_nfw",
    "cdm_einasto":       "cdm_einasto",
    "hmg_sp":            "hmg_k1",   # s_p-HMG (point-mass approx., not standard)
    "hmg_sr":            "hmg_sr",   # s-HMG (standard)
    "e2d_k2":            "e2d_k2",
    "hmg_sk2":           "hmg_sk2",
    "fr_screened":       "fr_screened",
    "refracted_gravity": "refracted_gravity",
}
# STVG was excluded from the BC run (--no-stvg); use old nbar pipeline values.
FALLBACK_KEYS: set[str] = {"stvg"}

MODEL_SPECS: list[tuple[str, str, int]] = [
    ("baryonic",          "Baryonic Newtonian",   0),
    ("qumond_simple",     "QUMOND simple",         0),
    ("qumond_standard",   "QUMOND standard",       0),
    ("qumond_mls",        "QUMOND MLS/RAR",        0),
    ("veg_fixed",         "VEG original",          0),
    ("veg_free",          "VEG free a_EG",         1),
    ("stvg",              "STVG",                  2),
    ("cdm_nfw",           "CDM NFW",               2),
    ("cdm_einasto",       "CDM Einasto",           3),
    ("hmg_sp",            "s_p-HMG (This Work)",  1),   # point-mass approx., not promoted
    ("hmg_sr",            "s-HMG (This Work)",    1),   # standard (E-2D vertical, r_hyd→∞)
    ("e2d_k2",            "HMG 2p (This Work)",   2),
    ("hmg_sk2",           "s/r_h-HMG (This Work)", 2),
    ("fr_screened",       "f(R) screened",         2),
    ("refracted_gravity", "Refracted Gravity",     2),
]

# Keys shown in Fig. 2 (s-HMG shown only as overlay via hmg_1fam_dir, not as main curve)
FIGURE_MODEL_KEYS: set[str] = {k for k, _, _ in MODEL_SPECS} - {"hmg_sp", "hmg_sr"}

FIGS = ROOT / "figs"


def _setup_bc_dir(bc_dir: Path, fallback_dir: Path, dest_dir: Path) -> None:
    """Populate dest_dir with correctly-named model CSVs from bc_dir."""
    dest_dir.mkdir(exist_ok=True)
    for step3_key, bc_suffix in BC_KEY_MAP.items():
        for side in ("radial", "vertical"):
            src = bc_dir / f"model_bc_{bc_suffix}_{side}.csv"
            dst = dest_dir / f"model_{step3_key}_{side}.csv"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  MISSING BC: {src.name}")
    for key in FALLBACK_KEYS:
        for side in ("radial", "vertical"):
            src = fallback_dir / f"model_{key}_{side}.csv"
            dst = dest_dir / f"model_{key}_{side}.csv"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  MISSING fallback: {src.name}")
    for fname in ("mc100_baryonic_radial.csv", "mc100_baryonic_vertical.csv"):
        src = fallback_dir / fname
        if src.exists():
            shutil.copy2(src, dest_dir / fname)


def _ensure_baryon_band(outputs_dir: Path) -> None:
    """Copy nbar-specific baryon band to outputs/data/ for plot_fig2."""
    data_dir = outputs_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)
    nbar_id = next(
        (part for part in outputs_dir.name.split("_") if part.startswith("nbar")),
        None,
    )
    if nbar_id:
        src = ROOT / "data" / f"baryon_band_{nbar_id}.csv"
        if src.exists():
            shutil.copy2(src, data_dir / "baryon_band.csv")
            shutil.copy2(src, outputs_dir / f"baryon_band_{outputs_dir.name}.csv")
    if not (data_dir / "baryon_band.csv").exists():
        fallback = ROOT / "data" / "baryon_band.csv"
        if fallback.exists():
            shutil.copy2(fallback, data_dir / "baryon_band.csv")
    tband = ROOT / "data" / "baryonic_target_band.csv"
    if tband.exists() and not (data_dir / "baryonic_target_band.csv").exists():
        shutil.copy2(tband, data_dir / "baryonic_target_band.csv")


def _build_chi2_csv(outputs_dir: Path) -> None:
    """Build mc100_chi2_all_models.csv from per-model CSVs in outputs_dir."""
    rows: list[dict] = []
    for key, name, k in MODEL_SPECS:
        rad_p = outputs_dir / f"model_{key}_radial.csv"
        vert_p = outputs_dir / f"model_{key}_vertical.csv"
        if not rad_p.exists() or not vert_p.exists():
            print(f"  skip chi2 {key}: missing CSV")
            continue
        with open(rad_p) as fh:
            rrows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
        with open(vert_p) as fh:
            vrows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
        r_idx = [i for i, h in enumerate(rrows[0]) if h.startswith("b")]
        v_idx = [i for i, h in enumerate(vrows[0]) if h.startswith("b")]
        try:
            chi2_r = np.array([float(rrows[-1][i]) for i in r_idx])
            chi2_z = np.array([float(vrows[-1][i]) for i in v_idx])
        except (ValueError, IndexError) as exc:
            print(f"  ERROR chi2 {key}: {exc}")
            continue
        n = min(len(chi2_r), len(chi2_z))
        dof = N_PRIMARY - k
        for i in range(n):
            rows.append({
                "draw_id": i + 1, "draw_label": f"b{i+1}",
                "model_key": key, "model_name": name,
                "k": k, "dof": dof,
                "chi2_nu": f"{(chi2_r[i] + chi2_z[i]) / dof:.4f}",
            })
    if not rows:
        print("  ERROR: no chi2 rows built")
        return
    out = outputs_dir / "mc100_chi2_all_models.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  chi2 CSV: {out.name}  ({len(rows)} rows)")
    by_key: dict = defaultdict(list)
    for r in rows:
        by_key[r["model_key"]].append(float(r["chi2_nu"]))
    print(f"  {'key':<28} p16    p50    p84")
    for key, _, _ in MODEL_SPECS:
        if key in by_key:
            vals = np.array(sorted(by_key[key]))
            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            print(f"  {key:<28} {p16:.3f}  {p50:.3f}  {p84:.3f}")


def _gen_figure(
    outputs_dir: Path,
    fig_out: Path,
    obs_path: Path,
    dpi: int,
    hmg_1fam_dir: "Path | None",
    model_keys: "set[str] | None",
    label: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n=== Generating {fig_out.name} ===  [{label}]")
    try:
        fig = plot_fig2(
            outputs_dir,
            obs_path=obs_path if obs_path.exists() else None,
            output_path=fig_out,
            dpi=dpi,
            hmg_1fam_outputs_dir=hmg_1fam_dir,
            model_keys=model_keys,
        )
        plt.close(fig)
        print(f"  Saved: {fig_out}")
    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figure", action="store_true",
                        help="Only build chi2 CSV; skip figure generation")
    parser.add_argument("--dpi", type=int, default=220,
                        help="Figure DPI (default: 220)")
    args = parser.parse_args()

    # ── Stage nbar4 (main curves) ────────────────────────────────────────────
    dest4 = ROOT / "outputs" / "bc_fig2_nbar4"
    print("=== Staging bc_fig2_nbar4 ===")
    _setup_bc_dir(ROOT / "outputs" / "bc_nbar4", ROOT / "outputs" / "nbar4", dest4)
    _ensure_baryon_band(dest4)
    _build_chi2_csv(dest4)

    # ── Stage nbar1 (sHMG+Imig overlay and internal figure) ─────────────────
    dest1 = ROOT / "outputs" / "bc_fig2_nbar1"
    print("\n=== Staging bc_fig2_nbar1 ===")
    _setup_bc_dir(ROOT / "outputs" / "bc_nbar1", ROOT / "outputs" / "nbar1", dest1)
    _ensure_baryon_band(dest1)
    _build_chi2_csv(dest1)

    if args.no_figure:
        return

    FIGS.mkdir(exist_ok=True)
    obs_path = ROOT / "data" / "fig2_observational_data.csv"
    if not obs_path.exists():
        obs_path = ROOT / "data" / "fig2_observational_catalog.csv"

    # ── Paper figure: srh-HMG nbar4 main + s-HMG+Imig overlay ─────────────
    # Saved as fig2_bc_nbar4.png (canonical local name) and copied to
    # fig2_reproduced_4fam.png (name referenced in Overleaf main.tex).
    _gen_figure(
        dest4,
        FIGS / "fig2_bc_nbar4.png",
        obs_path,
        args.dpi,
        hmg_1fam_dir=dest1,
        model_keys=FIGURE_MODEL_KEYS,
        label="srh-HMG nbar4 + s-HMG+Imig overlay",
    )
    import shutil as _sh
    _sh.copy2(FIGS / "fig2_bc_nbar4.png", FIGS / "fig2_reproduced_4fam.png")

    # ── Internal figure: ALL models with nbar1 (sHMG + srHMG both shown) ────
    # Shows how the figure looks when restricted to the McGaugh-Imig family.
    # hmg_1fam_dir=None: no overlay (nbar1 IS already the Imig family).
    _gen_figure(
        dest1,
        FIGS / "fig2_internal_nbar1.png",
        obs_path,
        args.dpi,
        hmg_1fam_dir=None,
        model_keys=None,   # show all, including hmg_sp, hmg_sr, and e2d_k2
        label="ALL models nbar1 (internal)",
    )


if __name__ == "__main__":
    main()
