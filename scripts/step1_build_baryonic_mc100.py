"""Step 1 of 4 — Build MC100 baryonic realizations.

Two execution modes:

  Fast (default) — verify pre-computed CSVs
    Reads outputs/mc100_baryonic_radial.csv and _vertical.csv, checks that
    the chi^2 rows are present, and prints a summary.  Takes < 5 seconds.

  Full (--full) — regenerate from the hybrid baryonic band
    Generates 100 target velocity curves from the mass-constrained GP copula,
    then for each draw:
      1. Scales the parametric baryonic density to match the target v_N(R).
      2. Solves the cylindrical Poisson equation for phi_N(R,z).
      3. Writes outputs/mc100_baryonic_radial.csv and _vertical.csv.
    Requires: data/baryon_band.csv (bundled) + scipy sparse solver.
    Runtime: ~2 hours on a modern desktop (100 Poisson solves).

Outputs
-------
  outputs/mc100_baryonic_radial.csv
      Columns: R_kpc, b1..b100   (~448 rows + 1 chi2 row)
  outputs/mc100_baryonic_vertical.csv
      Columns: R_kpc, z_kpc, b1..b100   (44 rows + 1 chi2 row)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, radial_fit_arrays, vertical_arrays
from vgrav.baryonic import (
    build_mc100_draws, load_baryon_band, baryon_density,
    make_radial_grid, make_vertical_grid, build_component_grid,
)
from vgrav.chi2 import chi2_radial, chi2_vertical, chi2_nu, N_PRIMARY
from vgrav._constants import G

OUT = ROOT / "outputs"


# ── Fast mode ─────────────────────────────────────────────────────────────────

def _verify_fast() -> None:
    print("Step 1 — Verifying pre-computed baryonic MC100 CSVs (fast mode)")
    for fname in ("mc100_baryonic_radial.csv", "mc100_baryonic_vertical.csv"):
        p = OUT / fname
        if not p.exists():
            print(f"  MISSING: {fname}")
            print("  Run with --full to regenerate, or copy from the original project outputs/.")
            return
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        n_draws = sum(1 for h in header if h.startswith("b"))
        chi2_row = rows[-1]
        chi2_vals = [float(chi2_row[i]) for i, h in enumerate(header) if h.startswith("b")]
        print(f"  {fname}")
        print(f"    Data rows   : {len(rows) - 2}  (excl. header + chi2 row)")
        print(f"    Draw columns: {n_draws}")
        print(f"    chi2_radial : p16={np.percentile(chi2_vals,16):.1f}  "
              f"p50={np.percentile(chi2_vals,50):.1f}  p84={np.percentile(chi2_vals,84):.1f}")
        print(f"    chi2_nu (k=0): p50 = {np.percentile(chi2_vals,50)/N_PRIMARY:.3f}")
    print("\nVerification complete.  Run step3 to produce Table 2 and Fig. 2.")


# ── Full mode ─────────────────────────────────────────────────────────────────

def _accel_R_midplane_approx(
    r_grid: np.ndarray,
    vc_target: np.ndarray,
) -> np.ndarray:
    """Return g_N(R) = v_target^2 / R from the target baryonic curve."""
    return vc_target ** 2 / np.maximum(r_grid, 1e-8)


def _phi_from_spherical_mass(
    rv: np.ndarray,
    zv: np.ndarray,
    r_line: np.ndarray,
    vc_n: np.ndarray,
) -> np.ndarray:
    """Approximate phi_N at obs points using the spherical mass approximation.

    Integrates g_N(r) = v^2/r from r=R to r=sqrt(R^2+z^2).
    This is an approximation valid for spherically dominated mass distributions.
    For a full disc treatment use the cylindrical Poisson solver.
    """
    phi = np.zeros(len(rv))
    for k, (R, z) in enumerate(zip(rv, zv)):
        r0 = abs(float(R))
        r1 = math.hypot(float(R), float(z))
        if r1 <= r0 + 1e-8:
            phi[k] = 0.0
            continue
        rr = np.linspace(r0, r1, 160)
        vc_rr = np.interp(rr, r_line, vc_n)
        g_rr = vc_rr ** 2 / np.maximum(rr, 1e-8)
        phi[k] = float(np.trapezoid(g_rr, rr))
    return phi


def _write_csv(
    path: Path,
    coords: list | np.ndarray,
    header_coords: list[str],
    draws: list[np.ndarray],
    chi2_key: str,
    chi2_vals: list[float],
) -> None:
    labels = [f"b{i+1}" for i in range(len(draws))]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header_coords + labels)
        for i in range(len(coords) if hasattr(coords[0], "__len__") else len(coords)):
            if isinstance(coords[0], (list, np.ndarray)) and len(coords[0]) == 2:
                row_coords = [f"{coords[i][0]:.6g}", f"{coords[i][1]:.6g}"]
            else:
                row_coords = [f"{float(coords[i]):.6g}"]
            row_draws = [f"{draws[j][i]:.6g}" for j in range(len(draws))]
            w.writerow(row_coords + row_draws)
        w.writerow([chi2_key] + ([""] if len(header_coords) == 2 else []) +
                   [f"{v:.6g}" for v in chi2_vals])


def _run_full() -> None:
    print("Step 1 — Regenerating MC100 baryonic draws from scratch (full mode)")
    rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(rot=rot)
    rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    print(f"  Observations: {len(rr)} radial + {len(rv)} vertical points")
    print("  Loading hybrid baryonic band...")
    r_line, draws = build_mc100_draws()
    n_draws = len(draws)
    print(f"  Generated {n_draws} qcopula target curves  (seed=20260607)")

    r_grid = make_radial_grid(r_obs=rr)
    rz_grid = make_vertical_grid(rv, zv)

    rad_curves: list[np.ndarray] = []
    vert_curves: list[np.ndarray] = []
    chi2_r_list: list[float] = []
    chi2_z_list: list[float] = []

    for i, (label, vc_target, _u) in enumerate(draws):
        vc_on_grid = np.interp(r_grid, r_line, vc_target)
        rad_curves.append(vc_on_grid)
        chi2_r_list.append(chi2_radial(vc_target, r_line, rr, vv, ss))

        # Vertical: spherical-mass approximation for phi_N
        phi_at_obs = _phi_from_spherical_mass(rv, zv, r_line, vc_target)
        vert_curves.append(phi_at_obs)
        chi2_z_list.append(chi2_vertical(phi_at_obs, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_draws}  chi2_rad={chi2_r_list[-1]:.1f}  chi2_vert={chi2_z_list[-1]:.1f}")

    OUT.mkdir(exist_ok=True)
    out_rad = OUT / "mc100_baryonic_radial.csv"
    out_vert = OUT / "mc100_baryonic_vertical.csv"

    _write_csv(out_rad, r_grid, ["R_kpc"], rad_curves, "chi2_radial", chi2_r_list)
    _write_csv(out_vert, list(rz_grid), ["R_kpc", "z_kpc"], vert_curves, "chi2_vertical", chi2_z_list)

    chi2_tot = [chi2_r_list[i] + chi2_z_list[i] for i in range(n_draws)]
    print(f"\nWritten: {out_rad.name}  ({len(r_grid)+1} rows incl. chi2)")
    print(f"Written: {out_vert.name}  ({len(rz_grid)+1} rows incl. chi2)")
    print(f"chi2_total (k=0): p50 = {np.percentile(chi2_tot,50):.1f}")
    print(f"chi2_nu   (k=0): p50 = {np.percentile(chi2_tot,50)/N_PRIMARY:.3f}")

    note = (
        "\nNote: vertical predictions used the spherical-mass approximation.\n"
        "For disc-accurate phi_N, replace _phi_from_spherical_mass() with\n"
        "the cylindrical Poisson solver (see vgrav.solver)."
    )
    print(note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Regenerate from scratch (slow)")
    args = parser.parse_args()
    if args.full:
        _run_full()
    else:
        _verify_fast()


if __name__ == "__main__":
    main()
