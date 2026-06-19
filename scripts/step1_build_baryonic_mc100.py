"""Step 1 of 4 — Build MC100 baryonic realizations.

Two execution modes:

  Fast (default) — verify pre-computed CSVs
    Reads outputs/mc100_baryonic_radial.csv and _vertical.csv, checks that
    the chi^2 rows are present, and prints a summary.  Takes < 5 seconds.

  Full (--full) — regenerate from the hybrid baryonic band
    Generates 100 target velocity curves from the mass-constrained GP copula,
    then for each draw:
      1. Scales the parametric baryonic density to match the target v_N(R) via
         least-squares fit of the stellar scale factor s.
      2. Solves the cylindrical Poisson equation ∇²φ = 4πGρ(R,z) for φ_N.
      3. Evaluates the rotation curve and vertical potential difference at the
         observed (R, z) points.
      4. Writes outputs/mc100_baryonic_radial.csv and _vertical.csv.
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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, radial_fit_arrays, vertical_arrays
from vgrav.baryonic import (
    build_mc100_draws,
    make_radial_grid,
    basis_potentials,
    calibrate_scale,
)
from vgrav.chi2 import chi2_radial, chi2_vertical, N_PRIMARY
from vgrav.solver import make_grid, phi_difference, radial_speed, blend_outer
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


# ── CSV writer ────────────────────────────────────────────────────────────────

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


# ── Full mode ─────────────────────────────────────────────────────────────────

def _run_full(n_draws_limit: int | None = None) -> None:
    label = f"first {n_draws_limit}" if n_draws_limit else "all"
    print(f"Step 1 — Regenerating baryonic draws from scratch ({label} draws)")
    rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(rot=rot)
    rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    print(f"  Observations: {len(rr)} radial + {len(rv)} vertical points")
    print("  Loading hybrid baryonic band...")
    r_line, all_draws = build_mc100_draws()
    draws = all_draws[:n_draws_limit] if n_draws_limit else all_draws
    n_draws = len(draws)
    print(f"  Processing {n_draws} MC target curves  (seed=20260607)")

    r_grid = make_radial_grid(r_obs=rr)

    # Pre-compute cylindrical grid and basis potentials (once for all draws).
    # By Poisson linearity: rho(s) = s*rho_stellar + rho_fixed
    #                        phi(s) = s*phi_stellar + phi_fixed
    print("  Building cylindrical grid and solving basis potentials "
          "(2 Poisson solves — one-time cost)...")
    cyl_grid = make_grid(r_min=0.1, r_max=40.0, z_max=20.0, nR=121, nz=121)
    _rho_st, phi_st, mass_st, _rho_fx, phi_fx, mass_fx = basis_potentials(cyl_grid)
    print("  Basis potentials ready.")

    rad_curves: list[np.ndarray] = []
    vert_curves: list[np.ndarray] = []
    chi2_r_list: list[float] = []
    chi2_z_list: list[float] = []

    for i, (label, vc_target, _u) in enumerate(draws):
        # Least-squares calibration of stellar scale factor s for this draw.
        s = calibrate_scale(phi_st, phi_fx, mass_st, mass_fx, cyl_grid, vc_target, r_line)

        # Compose the Newtonian potential for this draw.
        phi_N = s * phi_st + phi_fx
        total_mass = s * mass_st + mass_fx

        # Rotation curve via cylindrical solver; monopole at outer boundary.
        vc_N = radial_speed(cyl_grid, phi_N, r_grid)
        outer_vc = np.sqrt(G * total_mass / np.maximum(r_grid, 1e-9))
        vc_N = blend_outer(r_grid, vc_N, outer_vc)
        rad_curves.append(vc_N)
        chi2_r_list.append(chi2_radial(vc_N, r_grid, rr, vv, ss))

        # Vertical potential difference φ(R,z) − φ(R,0) at observation points.
        phi_diff = phi_difference(cyl_grid, phi_N, rv, zv)
        vert_curves.append(phi_diff)
        chi2_z_list.append(
            chi2_vertical(phi_diff, rv, zv, rv, zv, phi_obs, sig_phi, sig_z)
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_draws}  s={s:.3f}  "
                  f"chi2_rad={chi2_r_list[-1]:.1f}  chi2_vert={chi2_z_list[-1]:.1f}")

    OUT.mkdir(exist_ok=True)
    out_rad = OUT / "mc100_baryonic_radial.csv"
    out_vert = OUT / "mc100_baryonic_vertical.csv"

    _write_csv(out_rad, r_grid, ["R_kpc"], rad_curves, "chi2_radial", chi2_r_list)
    _write_csv(out_vert, np.column_stack([rv, zv]), ["R_kpc", "z_kpc"], vert_curves, "chi2_vertical", chi2_z_list)

    chi2_tot = [chi2_r_list[i] + chi2_z_list[i] for i in range(n_draws)]
    print(f"\nWritten: {out_rad.name}  ({len(r_grid)+1} rows incl. chi2)")
    print(f"Written: {out_vert.name}  ({len(rv)+1} rows incl. chi2)")
    print(f"chi2_total (k=0): p50 = {np.percentile(chi2_tot,50):.1f}")
    print(f"chi2_nu   (k=0): p50 = {np.percentile(chi2_tot,50)/N_PRIMARY:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="Regenerate from scratch (slow)")
    parser.add_argument("--n-draws", type=int, default=None, metavar="N",
                        help="Generate only the first N draws (for quick validation)")
    args = parser.parse_args()
    if args.full or args.n_draws:
        _run_full(n_draws_limit=args.n_draws)
    else:
        _verify_fast()


if __name__ == "__main__":
    main()
