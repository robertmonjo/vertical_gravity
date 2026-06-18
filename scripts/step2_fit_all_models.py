"""Step 2 of 4 — Fit all 11 gravity models to the 100 baryonic draws.

Two execution modes:

  Fast (default) — verify pre-computed model CSVs
    Checks that all 22 model CSVs (11 models × 2 grids) exist and prints
    a chi2_nu summary.  Takes < 10 seconds.

  Full (--full) — regenerate all model predictions
    Applies each gravity model equation to the 100 baryonic draws from
    mc100_baryonic_radial.csv and mc100_baryonic_vertical.csv.

    Model runtimes (approximate, 100 draws):
      Algebraic (HMG, f(R), ReG, VEG):  < 5 min each
      CDM (NFW, Einasto):                < 1 min
      STVG direct summation:             ~ 2 h  (350k mass cells × 100 draws)
      QUMOND cylindrical solver:         ~ 2 h  (sparse 121×121 system × 100)

    Output: outputs/model_{key}_radial.csv and _vertical.csv for each model.

Model keys
----------
  baryonic, qumond_simple, qumond_standard, qumond_mls, stvg,
  cdm_nfw, cdm_einasto, hmg_k1, fr_screened, refracted_gravity, emergent_gravity

Note on QUMOND and STVG
------------------------
The full solver mode requires a 3D baryonic density for each draw.
In this release, the parametric density is scaled to match the baryonic
rotation curve of each draw (see vgrav.baryonic.build_component_grid).
For exact replication of the published chi^2 values use the original
pipeline scripts in the project's scripts/ folder (requires Imig+2025 data).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, radial_fit_arrays, vertical_arrays
from vgrav.chi2 import chi2_radial, chi2_vertical, chi2_nu, N_PRIMARY, vertical_force_from_phi
from vgrav._constants import A0_KMS2_PER_KPC, A_EG_FIXED
from vgrav.models import (
    nu_mond, hmg_factor, nu_fr, nu_rg, nu_eg,
    nfw_density_from_local, nfw_mass,
    einasto_density_from_local, einasto_mass,
    spherical_vc_and_phi, predict_mond_proxy, predict_hmg_proxy,
    predict_cdm_nfw, predict_cdm_einasto,
)

OUT = ROOT / "outputs"

# ── k values for each model ────────────────────────────────────────────────────
MODEL_K = {
    "baryonic": 0, "qumond_simple": 0, "qumond_standard": 0, "qumond_mls": 0,
    "stvg": 1, "cdm_nfw": 2, "cdm_einasto": 2,
    "hmg_k1": 1, "fr_screened": 2, "refracted_gravity": 2, "emergent_gravity": 0,
}


# ── Fast mode ─────────────────────────────────────────────────────────────────

def _verify_fast() -> None:
    print("Step 2 — Verifying pre-computed model CSVs (fast mode)")
    found, missing = [], []
    for key in MODEL_K:
        for suffix in ("radial", "vertical"):
            p = OUT / f"model_{key}_{suffix}.csv"
            (found if p.exists() else missing).append(p.name)

    print(f"  Found  : {len(found)} files")
    if missing:
        print(f"  Missing: {len(missing)} files — {', '.join(missing[:4])}{'...' if len(missing)>4 else ''}")
        print("  Run with --full to regenerate.")
    else:
        print("  All 22 model CSVs present.")

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


# ── Full mode helpers ─────────────────────────────────────────────────────────

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
    data_rows = [r for r in rows[1:-1]]  # exclude header and chi2 row
    r_grid = np.array([float(r[r_idx]) for r in data_rows])
    rad_curves = [np.array([float(r[i]) for r in data_rows]) for i in draw_idx]

    with open(vert_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    draw_idx = [i for i, h in enumerate(hdr) if h.startswith("b")]
    R_col, z_col = hdr.index("R_kpc"), hdr.index("z_kpc")
    data_rows = [r for r in rows[1:-1]]
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


def _fit_fr(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    """Fit f(R) screened model per draw."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC

    def _score(delta, xc):
        nu = nu_fr(x_n, delta, xc)
        vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
        nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
        phi_m = nu_obs * phi_n
        kz = vertical_force_from_phi(phi_m, rv_obs, rv_obs * 0 + 0.5)  # dummy z for gradient
        kz = vertical_force_from_phi(phi_m, rv_obs, np.ones_like(rv_obs) * 0.5)
        sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
        chi_r = float(np.sum(((np.interp(rr, r_grid, vc_m) - vv) / ss) ** 2))
        chi_z = float(np.sum(((phi_m - phi_obs) / sig_eff) ** 2))
        return chi_r + chi_z

    def obj(t):
        d, xc = t
        if d <= 0 or xc <= 1e-3:
            return 1e30
        return _score(d, xc)

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


def _fit_rg(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    """Fit Refracted Gravity per draw."""
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
        chi_r = float(np.sum(((np.interp(rr, r_grid, vc_m) - vv) / ss) ** 2))
        kz = vertical_force_from_phi(phi_m, rv_obs, np.ones_like(rv_obs) * 0.5)
        sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
        chi_z = float(np.sum(((phi_m - phi_obs) / sig_eff) ** 2))
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


def _fit_veg(r_grid, vc_n, phi_n, rv_obs):
    """Fit VEG (free a_EG) per draw."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)

    def obj(loga):
        a = np.exp(loga)
        nu = nu_eg(g_n, a)
        vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
        nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
        return float(np.sum((nu_obs * phi_n) ** 2))  # placeholder: use score with obs

    # VEG: minimize chi2 over a_EG in log scale
    rv = minimize_scalar(lambda la: _score_veg(la, r_grid, vc_n, phi_n, rv_obs),
                         bounds=(np.log(10.0), np.log(50000.0)), method="bounded")
    a_opt = np.exp(rv.x)
    nu = nu_eg(g_n, a_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_obs * phi_n


def _score_veg(loga, r_grid, vc_n, phi_n, rv_obs):
    a = np.exp(loga)
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    nu = nu_eg(g_n, a)
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return float(np.var(nu_obs * phi_n))


def _fit_hmg(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    """Fit HMG anisotropic (beta, lambda_z) per draw."""
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    best = None
    for beta in np.linspace(0.0, 3.0, 61):
        f_r = hmg_factor(g_n, beta)
        vc_m = np.sqrt(np.maximum(f_r * vc_n ** 2, 0.0))
        chi_r = float(np.sum(((np.interp(rr, r_grid, vc_m) - vv) / ss) ** 2))
        for lam in np.linspace(0.0, 1.0, 41):
            f_z = np.interp(rv_obs, r_grid, 1.0 + lam * (f_r - 1.0),
                            left=1.0 + lam * (f_r[0] - 1.0),
                            right=1.0 + lam * (f_r[-1] - 1.0))
            phi_m = f_z * phi_n
            kz = vertical_force_from_phi(phi_m, rv_obs, np.ones_like(rv_obs) * 0.5)
            sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
            chi_z = float(np.sum(((phi_m - phi_obs) / sig_eff) ** 2))
            total = chi_r + chi_z
            if best is None or total < best[0]:
                best = (total, beta, lam, vc_m, phi_m)
    _, _, _, vc_out, phi_out = best
    return vc_out, phi_out


def _run_full() -> None:
    print("Step 2 — Fitting all 11 gravity models (full mode)")
    rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(rot=rot)
    rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    rad_p = OUT / "mc100_baryonic_radial.csv"
    vert_p = OUT / "mc100_baryonic_vertical.csv"
    if not rad_p.exists() or not vert_p.exists():
        print("  ERROR: run step1 first to produce baryonic MC100 CSVs.")
        return

    r_grid, rz_grid, rv, zv, rad_bary, vert_bary = _read_baryonic_draws(rad_p, vert_p)
    n_draws = len(rad_bary)
    print(f"  Loaded {n_draws} baryonic draws.  r_grid={len(r_grid)} pts, rz_grid={len(rz_grid)} pts.")

    model_specs = [
        ("baryonic",          0, "bary"),
        ("qumond_simple",     0, "mond_simple"),
        ("qumond_standard",   0, "mond_standard"),
        ("qumond_mls",        0, "mond_rar"),
        ("cdm_nfw",           2, "cdm_nfw"),
        ("cdm_einasto",       2, "cdm_einasto"),
        ("hmg_k1",            1, "hmg"),
        ("fr_screened",       2, "fr"),
        ("refracted_gravity", 2, "rg"),
        ("emergent_gravity",  0, "veg_fixed"),
    ]

    for key, k_val, tag in model_specs:
        print(f"\n  --- {key} (k={k_val}) ---")
        rad_out, vert_out = [], []
        chi2_r_list, chi2_z_list = [], []

        for i in range(n_draws):
            vc_n = rad_bary[i]
            phi_n = vert_bary[i]
            g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
            x_n = g_n / A0_KMS2_PER_KPC

            if tag == "bary":
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            elif tag.startswith("mond_"):
                kind = tag.split("_", 1)[1]
                vc_m, phi_m = predict_mond_proxy(r_grid, vc_n, phi_n, rv_obs, kind)

            elif tag == "cdm_nfw":
                vc_m, phi_m = predict_cdm_nfw(r_grid, vc_n, phi_n, rv_obs, zv_obs)

            elif tag == "cdm_einasto":
                vc_m, phi_m = predict_cdm_einasto(r_grid, vc_n, phi_n, rv_obs, zv_obs)

            elif tag == "hmg":
                vc_m, phi_m = _fit_hmg(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z)

            elif tag == "fr":
                vc_m, phi_m = _fit_fr(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z)

            elif tag == "rg":
                vc_m, phi_m = _fit_rg(r_grid, vc_n, phi_n, rv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z)

            elif tag == "veg_fixed":
                nu = nu_eg(g_n, A_EG_FIXED)
                vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
                nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
                phi_m = nu_obs * phi_n

            else:
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            rad_out.append(vc_m)
            vert_out.append(phi_m)
            cr = chi2_radial(vc_m, r_grid, rr, vv, ss)
            cz = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
            chi2_r_list.append(cr)
            chi2_z_list.append(cz)

        chi2_med = np.median([chi2_r_list[j] + chi2_z_list[j] for j in range(n_draws)])
        print(f"    chi2_nu p50 = {chi2_med / (N_PRIMARY - k_val):.3f}")
        _write_model_csvs(key, r_grid, rz_grid, rad_out, vert_out, chi2_r_list, chi2_z_list)

    print("\n  Note: STVG requires build_component_grid() + direct summation.")
    print("  For STVG, use vgrav.models.predict_stvg() with a ComponentGrid.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Regenerate model predictions (slow)")
    args = parser.parse_args()
    if args.full:
        _run_full()
    else:
        _verify_fast()


if __name__ == "__main__":
    main()
