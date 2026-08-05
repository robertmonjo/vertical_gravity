"""Compute corrected HMG s values for nbar1-4 and save to CSV.

nbar1 (MI):          reads existing model_hmg_k1_params.csv (correct)
nbar2 (MI+LW):       recalculates (no params CSV exists)
nbar3 (MI+LW+B2):    reads existing model_hmg_k1_params.csv (correct)
nbar4 (MI+LW+B2+MM): recalculates with correct band (original CSV was contaminated
                      with nfam=1 band instead of the nbar4 band)

Output: outputs/s_hmg_all_nbars.csv
Columns: draw_id, draw_label, s_nbar1, s_nbar2, s_nbar3, s_nbar4
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vgrav.baryonic import build_mc100_draws
from vgrav.models import predict_hmg_common_s
from vgrav.observations import radial_fit_arrays, vertical_arrays

CATALOG_PATH = ROOT / "data" / "fig2_observational_catalog.csv"
N_DRAWS = 100


def load_params_csv(path: Path) -> dict[str, float]:
    """draw_id (str) -> s from model_hmg_k1_params.csv."""
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["draw_id"]] = float(row["s"])
    return result


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def load_baryonic_wide(nbar: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load baryonic outputs for nbar in wide format.

    Returns
    -------
    r_grid    : (Nr,) radii [kpc]
    vc_n_mat  : (Nr, 100) Newtonian vc [km/s] per draw
    rv_vert   : (44,) R [kpc] of vertical obs
    phi_n_mat : (44, 100) Newtonian phi [km²/s²] per draw
    """
    out_dir = ROOT / "outputs" / f"nbar{nbar}"
    draw_cols = [f"b{i}" for i in range(1, N_DRAWS + 1)]

    # Radial: rows = R_kpc, cols = b1..b100
    with open(out_dir / "mc100_baryonic_radial.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if _is_float(r.get("R_kpc", ""))]
    r_grid = np.array([float(r["R_kpc"]) for r in rows])
    vc_n_mat = np.array([[float(r[c]) for c in draw_cols] for r in rows])  # (Nr, 100)

    # Vertical: rows = (R_kpc, z_kpc), cols = b1..b100
    with open(out_dir / "mc100_baryonic_vertical.csv", newline="") as f:
        vrows = list(csv.DictReader(f))
    vrows = [r for r in vrows if _is_float(r.get("R_kpc", ""))]
    rv_vert = np.array([float(r["R_kpc"]) for r in vrows])
    phi_n_mat = np.array([[float(r[c]) for c in draw_cols] for r in vrows])  # (44, 100)

    return r_grid, vc_n_mat, rv_vert, phi_n_mat


def compute_s_for_nbar(nbar: int,
                       rr: np.ndarray, vv: np.ndarray, ss: np.ndarray,
                       rv_obs: np.ndarray, zv_obs: np.ndarray,
                       phi_obs: np.ndarray, sig_phi: np.ndarray,
                       sig_z: np.ndarray) -> dict[str, float]:
    """Compute s for all 100 draws of nbar using the correct baryon_band."""
    band_path = ROOT / "outputs" / f"nbar{nbar}" / f"baryon_band_nbar{nbar}.csv"
    print(f"  nbar{nbar}: band = {band_path.name}", flush=True)

    r_line, band_draws = build_mc100_draws(band_path=band_path)
    r_grid, vc_n_mat, rv_vert, phi_n_mat = load_baryonic_wide(nbar)

    results = {}
    for i, (label, target_v, _) in enumerate(band_draws[:N_DRAWS]):
        draw_id = str(i + 1)

        # v2_n_rad: smooth band target (same as step2 canonical)
        v2_n_rad = target_v ** 2

        # v2_n_vert: Poisson vc_n interpolated at rv_obs
        vc_n_col = vc_n_mat[:, i]  # (Nr,)
        v2_n_vert = np.interp(rv_obs, r_grid, vc_n_col) ** 2

        # phi_n at vertical obs points (already at obs positions)
        phi_n = phi_n_mat[:, i]  # (44,)

        _, _, s_fit = predict_hmg_common_s(
            r_line, v2_n_rad, v2_n_vert,
            phi_n,
            rv_obs, zv_obs,
            rr, vv, ss,
            phi_obs, sig_phi, sig_z,
        )
        results[draw_id] = s_fit

        if (i + 1) % 25 == 0:
            print(f"    [{nbar}] draw {i+1}/{N_DRAWS}: s={s_fit:.4f}", flush=True)

    return results


def summary_stats(s_dict: dict) -> tuple[float, float, float]:
    vals = np.array(sorted(s_dict.values()))
    p16 = float(np.percentile(vals, 16))
    p50 = float(np.percentile(vals, 50))
    p84 = float(np.percentile(vals, 84))
    return p16, p50, p84


def main():
    out_csv = ROOT / "outputs" / "s_hmg_all_nbars.csv"

    # Common observation arrays (same for all nbars)
    cat = CATALOG_PATH if CATALOG_PATH.exists() else None
    rr, vv, ss = radial_fit_arrays(chi2_catalog_path=cat)
    rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays()
    print(f"Radial obs: {len(rr)}  |  Vertical obs: {len(rv_obs)}", flush=True)

    # nbar1 and nbar3: existing correct CSVs
    print("nbar1: reading existing params CSV...", flush=True)
    s_nbar1 = load_params_csv(ROOT / "outputs" / "nbar1" / "model_hmg_k1_params.csv")

    print("nbar3: reading existing params CSV...", flush=True)
    s_nbar3 = load_params_csv(ROOT / "outputs" / "nbar3" / "model_hmg_k1_params.csv")

    # nbar2 and nbar4: recalculate with correct band
    print("nbar2: recalculating...", flush=True)
    s_nbar2 = compute_s_for_nbar(2, rr, vv, ss, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)

    print("nbar4: recalculating (correcting nfam1-band contamination)...", flush=True)
    s_nbar4 = compute_s_for_nbar(4, rr, vv, ss, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)

    # Write combined CSV
    fieldnames = ["draw_id", "draw_label", "s_nbar1", "s_nbar2", "s_nbar3", "s_nbar4"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, N_DRAWS + 1):
            did = str(i)
            writer.writerow({
                "draw_id": did,
                "draw_label": f"b{i}",
                "s_nbar1": f"{s_nbar1.get(did, float('nan')):.6f}",
                "s_nbar2": f"{s_nbar2.get(did, float('nan')):.6f}",
                "s_nbar3": f"{s_nbar3.get(did, float('nan')):.6f}",
                "s_nbar4": f"{s_nbar4.get(did, float('nan')):.6f}",
            })

    print(f"\nEscrit: {out_csv}", flush=True)

    print("\n=== Resum estadístic (p16/p50/p84) ===")
    for label, sd in [("nbar1 (CSV existent)", s_nbar1),
                      ("nbar2 (recalc)", s_nbar2),
                      ("nbar3 (CSV existent)", s_nbar3),
                      ("nbar4 (recalc, corregit)", s_nbar4)]:
        p16, p50, p84 = summary_stats(sd)
        print(f"  {label}: s = {p50:.3f}  +{p84-p50:.3f} / -{p50-p16:.3f}")


if __name__ == "__main__":
    main()
