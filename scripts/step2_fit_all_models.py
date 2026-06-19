"""Step 2 of 4 — Fit all 12 gravity models to the 100 baryonic draws.

Two execution modes:

  Fast (default) — verify pre-computed model CSVs
    Checks that all 24 model CSVs (12 models × 2 grids) exist and prints
    a chi2_nu summary.  Takes < 10 seconds.

  Full (--full) — regenerate all model predictions
    Applies each gravity model to the 100 baryonic draws from
    mc100_baryonic_radial.csv and _vertical.csv.

    Use --n-draws N to process only the first N draws (for quick validation).

    Model runtimes (approximate, 100 draws):
      Algebraic (f(R), ReG, VEG):           < 5 min each
      HMG (s-parameter fit):                < 10 min
      CDM (NFW, Einasto per-draw):          < 5 min each
      STVG direct summation:                ~ 2 h  (350k cells × 100 draws)
      QUMOND cylindrical solver:            ~ 6 h  (3 variants × 100 Poisson solves)

    Output: outputs/model_{key}_radial.csv and _vertical.csv for each model.

Model keys (12 total including baryonic reference)
---------------------------------------------------
  baryonic, qumond_simple, qumond_standard, qumond_mls,
  veg_fixed, veg_free, stvg,
  cdm_nfw, cdm_einasto,
  hmg_k1, fr_screened, refracted_gravity
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, radial_fit_arrays, vertical_arrays
from vgrav.chi2 import chi2_radial, chi2_vertical, N_PRIMARY
from vgrav._constants import A0_KMS2_PER_KPC, A_EG_FIXED
from vgrav.models import (
    nu_fr, nu_rg, nu_eg,
    predict_qumond_solver,
    predict_stvg,
    predict_hmg_common_s,
    predict_cdm_nfw_per_draw, predict_cdm_einasto_per_draw,
)
from vgrav.baryonic import (
    build_mc100_draws,
    basis_potentials, calibrate_scale, build_component_grid,
)
from vgrav.solver import make_grid, monopole_boundary

OUT = ROOT / "outputs"

# ── k values (free parameters) for reduced chi2 ───────────────────────────────
MODEL_K = {
    "baryonic": 0,
    "qumond_simple": 0, "qumond_standard": 0, "qumond_mls": 0,
    "veg_fixed": 0, "veg_free": 1,
    "stvg": 2,
    "cdm_nfw": 2, "cdm_einasto": 2,
    "hmg_k1": 1,
    "fr_screened": 2, "refracted_gravity": 2,
}


# ── Fast mode ─────────────────────────────────────────────────────────────────

def _verify_fast() -> None:
    print("Step 2 — Verifying pre-computed model CSVs (fast mode)")
    found, missing = [], []
    for key in MODEL_K:
        for suffix in ("radial", "vertical"):
            p = OUT / f"model_{key}_{suffix}.csv"
            (found if p.exists() else missing).append(p.name)

    n_total = len(MODEL_K) * 2
    print(f"  Found  : {len(found)} / {n_total} files")
    if missing:
        print(f"  Missing: {len(missing)} — {', '.join(missing[:4])}{'...' if len(missing)>4 else ''}")
        print("  Run with --full to regenerate.")
    else:
        print("  All model CSVs present.")

    chi2_csv = OUT / "mc100_chi2_all_models.csv"
    if chi2_csv.exists():
        from collections import defaultdict
        model_chi2: dict[str, list] = defaultdict(list)
        with open(chi2_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                model_chi2[row["model_key"]].append(float(row["chi2_nu"]))
        print(f"\n  chi2_nu (p50) from {chi2_csv.name}:")
        for key in MODEL_K:
            if key in model_chi2:
                print(f"    {key:<26} {np.median(model_chi2[key]):.3f}")
    print("\nRun step3 to generate Table 2 and Fig. 2.")


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_baryonic_draws(
    rad_path: Path,
    vert_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Read baryonic draw CSVs.  Returns r_grid, rz_grid, rv, zv, rad_curves, vert_curves."""
    with open(rad_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    draw_idx = [i for i, h in enumerate(hdr) if h.startswith("b")]
    r_idx = hdr.index("R_kpc")
    data_rows = rows[1:-1]
    r_grid = np.array([float(r[r_idx]) for r in data_rows])
    rad_curves = [np.array([float(r[i]) for r in data_rows]) for i in draw_idx]

    with open(vert_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    draw_idx = [i for i, h in enumerate(hdr) if h.startswith("b")]
    R_col, z_col = hdr.index("R_kpc"), hdr.index("z_kpc")
    data_rows = rows[1:-1]
    rv = np.array([float(r[R_col]) for r in data_rows])
    zv = np.array([float(r[z_col]) for r in data_rows])
    rz_grid = np.column_stack([rv, zv])
    vert_curves = [np.array([float(r[i]) for r in data_rows]) for i in draw_idx]

    return r_grid, rz_grid, rv, zv, rad_curves, vert_curves


def _write_model_csvs(
    key: str,
    r_grid: np.ndarray,
    rz_grid: np.ndarray,
    rad_curves: list[np.ndarray],
    vert_curves: list[np.ndarray],
    chi2_r_list: list[float],
    chi2_z_list: list[float],
) -> None:
    n = len(rad_curves)
    labels = [f"b{i+1}" for i in range(n)]

    rad_path = OUT / f"model_{key}_radial.csv"
    with open(rad_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc"] + labels)
        for i, R in enumerate(r_grid):
            w.writerow([f"{R:.6g}"] + [f"{rad_curves[j][i]:.6g}" for j in range(n)])
        w.writerow(["chi2_radial"] + [f"{chi2_r_list[j]:.6g}" for j in range(n)])

    vert_path = OUT / f"model_{key}_vertical.csv"
    with open(vert_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc", "z_kpc"] + labels)
        for i, (R, z) in enumerate(rz_grid):
            w.writerow([f"{R:.6g}", f"{z:.6g}"] + [f"{vert_curves[j][i]:.6g}" for j in range(n)])
        w.writerow(["chi2_vertical", ""] + [f"{chi2_z_list[j]:.6g}" for j in range(n)])

    print(f"  Written: model_{key}_radial.csv + _vertical.csv")


# ── Per-draw fitters ──────────────────────────────────────────────────────────

def _fit_fr(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
):
    """Fit f(R) screened gravity (delta, xc) per draw."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC

    def obj(t):
        d, xc = t
        if d <= 0 or xc <= 1e-3:
            return 1e30
        nu = nu_fr(x_n, d, xc)
        vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
        nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
        phi_m = nu_obs * phi_n
        chi_r = chi2_radial(vc_m, r_grid, rr, vv, ss)
        chi_z = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
        return chi_r + chi_z

    best = None
    for x0 in [(0.33, 1.0), (0.5, 2.0), (0.2, 0.5), (1.0, 3.0), (0.7, 1.5)]:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    d_opt, xc_opt = best.x
    nu = nu_fr(x_n, d_opt, xc_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_obs * phi_n


def _fit_rg(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
):
    """Fit Refracted Gravity (eps_inf, xc) per draw."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC

    def obj(t):
        ei, xc = t
        if ei <= 0 or ei >= 1 or xc <= 1e-3:
            return 1e30
        nu = nu_rg(x_n, ei, xc)
        vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
        nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
        phi_m = nu_obs * phi_n
        chi_r = chi2_radial(vc_m, r_grid, rr, vv, ss)
        chi_z = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
        return chi_r + chi_z

    best = None
    for x0 in [(0.18, 0.7), (0.3, 1.0), (0.1, 0.5), (0.5, 2.0)]:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    ei_opt, xc_opt = best.x
    nu = nu_rg(x_n, ei_opt, xc_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_obs * phi_n


def _fit_veg(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
):
    """Fit VEG free a_EG per draw by minimising chi2_total."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)

    def score(log_a):
        a = np.exp(log_a)
        nu = nu_eg(g_n, a)
        vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
        nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
        phi_m = nu_at_obs * phi_n
        chi_r = chi2_radial(vc_m, r_grid, rr, vv, ss)
        chi_z = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
        return chi_r + chi_z

    res = minimize_scalar(
        score, bounds=(math.log(10.0), math.log(50000.0)), method="bounded",
        options={"xatol": 1e-6},
    )
    a_opt = math.exp(res.x)
    nu = nu_eg(g_n, a_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_at_obs * phi_n


def _fit_stvg(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid, rr, vv, ss, phi_obs, sig_phi, sig_z,
    mu0: float = 0.0678,
):
    """Fit STVG (alpha, mu) per draw by minimising chi2_total.  k=2."""

    def score(t):
        log_alpha, log_mu = t
        alpha = math.exp(log_alpha)
        mu = math.exp(log_mu)
        try:
            vc_m, phi_m = predict_stvg(r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid, alpha, mu)
        except Exception:
            return 1e30
        chi_r = chi2_radial(vc_m, r_grid, rr, vv, ss)
        chi_z = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
        return chi_r + chi_z

    starts = [
        (math.log(10.68), math.log(mu0)),
        (math.log(5.0),   math.log(0.05)),
        (math.log(20.0),  math.log(0.10)),
    ]
    best = None
    for x0 in starts:
        try:
            r = minimize(score, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    alpha_opt = math.exp(best.x[0])
    mu_opt = math.exp(best.x[1])
    return predict_stvg(r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid, alpha_opt, mu_opt)


# ── Full mode ─────────────────────────────────────────────────────────────────

_QUMOND_KIND = {
    "mond_simple":   "simple",
    "mond_standard": "standard",
    "mond_rar":      "rar",
}

_NEEDS_3D = frozenset(["mond_simple", "mond_standard", "mond_rar", "stvg"])


def _run_full(n_draws_limit: int | None = None) -> None:
    label = f"first {n_draws_limit}" if n_draws_limit else "all"
    print(f"Step 2 — Fitting all 12 gravity models ({label} draws)")
    rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(rot=rot)
    rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    rad_p = OUT / "mc100_baryonic_radial.csv"
    vert_p = OUT / "mc100_baryonic_vertical.csv"
    if not rad_p.exists() or not vert_p.exists():
        print("  ERROR: run step1 --full first to produce baryonic MC100 CSVs.")
        return

    r_grid, rz_grid, rv, zv, rad_bary, vert_bary = _read_baryonic_draws(rad_p, vert_p)
    n_draws = min(len(rad_bary), n_draws_limit) if n_draws_limit else len(rad_bary)
    print(f"  Loaded {len(rad_bary)} baryonic draws; processing {n_draws}.")
    print(f"  r_grid={len(r_grid)} pts, rz_grid={len(rz_grid)} pts.")

    # Pre-compute cylindrical basis potentials (used by QUMOND and STVG).
    print("  Computing basis potentials (2 Poisson solves — one-time cost)...")
    cyl_grid = make_grid(r_min=0.1, r_max=40.0, z_max=20.0, nR=121, nz=121)
    rho_st, phi_st, mass_st, rho_fx, phi_fx, mass_fx = basis_potentials(cyl_grid)
    print("  Basis potentials ready.")

    # Recover per-draw stellar scale factor s (deterministic, same seed as step1).
    print("  Recovering scale factors from baryonic band draws...")
    r_line, band_draws = build_mc100_draws()
    s_per_draw = [
        calibrate_scale(phi_st, phi_fx, mass_st, mass_fx, cyl_grid, vc_tgt, r_line)
        for _, vc_tgt, _ in band_draws[:n_draws]
    ]
    print(f"  Scale factors s: p16={np.percentile(s_per_draw,16):.3f}  "
          f"p50={np.percentile(s_per_draw,50):.3f}  p84={np.percentile(s_per_draw,84):.3f}")

    OUT.mkdir(exist_ok=True)

    model_specs = [
        ("baryonic",          0, "bary"),
        ("qumond_simple",     0, "mond_simple"),
        ("qumond_standard",   0, "mond_standard"),
        ("qumond_mls",        0, "mond_rar"),
        ("veg_fixed",         0, "veg_fixed"),
        ("veg_free",          1, "veg_free"),
        ("stvg",              2, "stvg"),
        ("cdm_nfw",           2, "cdm_nfw"),
        ("cdm_einasto",       2, "cdm_einasto"),
        ("hmg_k1",            1, "hmg"),
        ("fr_screened",       2, "fr"),
        ("refracted_gravity", 2, "rg"),
    ]

    for key, k_val, tag in model_specs:
        print(f"\n  --- {key} (k={k_val}) ---")
        rad_out: list[np.ndarray] = []
        vert_out: list[np.ndarray] = []
        chi2_r_list: list[float] = []
        chi2_z_list: list[float] = []

        for i in range(n_draws):
            vc_n = rad_bary[i]
            phi_n = vert_bary[i]
            g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
            s_i = s_per_draw[i]

            if tag == "bary":
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            elif tag in _QUMOND_KIND:
                kind = _QUMOND_KIND[tag]
                rho_3d = s_i * rho_st + rho_fx
                boundary_n = monopole_boundary(cyl_grid, s_i * mass_st + mass_fx)
                phi_n_full = s_i * phi_st + phi_fx
                vc_m, phi_m = predict_qumond_solver(
                    cyl_grid, rho_3d, boundary_n,
                    r_grid, rv_obs, zv_obs, kind,
                    phi_n_precomputed=phi_n_full,
                )

            elif tag == "veg_fixed":
                nu = nu_eg(g_n, A_EG_FIXED)
                vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
                nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
                phi_m = nu_at_obs * phi_n

            elif tag == "veg_free":
                vc_m, phi_m = _fit_veg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "stvg":
                comp_grid = build_component_grid(scale=s_i)
                vc_m, phi_m = _fit_stvg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "cdm_nfw":
                vc_m, phi_m, _ = predict_cdm_nfw_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "cdm_einasto":
                vc_m, phi_m, _ = predict_cdm_einasto_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "hmg":
                v2_n_rad = vc_n ** 2
                v2_n_vert = np.interp(rv_obs, r_grid, vc_n) ** 2
                vc_m, phi_m, _ = predict_hmg_common_s(
                    r_grid, v2_n_rad, v2_n_vert, phi_n,
                    rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "fr":
                vc_m, phi_m = _fit_fr(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "rg":
                vc_m, phi_m = _fit_rg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            else:
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            rad_out.append(vc_m)
            vert_out.append(phi_m)
            cr = chi2_radial(vc_m, r_grid, rr, vv, ss)
            cz = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
            chi2_r_list.append(cr)
            chi2_z_list.append(cz)

        chi2_med = np.median([chi2_r_list[j] + chi2_z_list[j] for j in range(n_draws)])
        print(f"    chi2_nu p50 = {chi2_med / max(N_PRIMARY - k_val, 1):.3f}")
        _write_model_csvs(key, r_grid, rz_grid, rad_out, vert_out, chi2_r_list, chi2_z_list)

    print(f"\nStep 2 complete ({n_draws} draws, {len(model_specs)} models).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true",
                        help="Regenerate model predictions (slow)")
    parser.add_argument("--n-draws", type=int, default=None, metavar="N",
                        help="Process only the first N draws (for quick validation)")
    args = parser.parse_args()
    if args.full or args.n_draws:
        _run_full(n_draws_limit=args.n_draws)
    else:
        _verify_fast()


if __name__ == "__main__":
    main()
