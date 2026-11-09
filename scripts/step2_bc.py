"""Step 2 — Band-Consistent variant (BC).

Methodological fix: ALL models use the MC100 band draw j as the baryonic
radial input v_N[j](R), paired with the Poisson phi_N[j](R,z) for the
vertical potential (which is calibrated to reproduce that same band draw,
as documented in step1_build_baryonic_mc100.py lines 138-140).

Motivation
----------
In the original step2_fit_all_models.py, HMG uses the smooth band target
v_N (on r_line) for the radial term, while CDM, VEG, f(R), ReG, STVG use
the Poisson vc_n (on r_grid) for the radial term.  These are two different
estimates of the same physical quantity v_N(R), leading to an inconsistent
chi2 comparison.

Here, modified-gravity models (HMG, MOND variants, VEG, f(R), ReG) use the
band draw j interpolated to r_grid as v_N[j](R) for the radial term.  HMG
additionally uses band draw j for v2_n_vert (the Newtonian circular velocity
at the vertical observation radii), making HMG fully band-consistent.

CDM models (NFW, Einasto) retain the Poisson vc_n for the baryonic radial
contribution, because CDM adds a dark-matter halo ON TOP of the baryonic
Newtonian field: vc_total² = vc_baryon² + vc_halo².  Here, the 3D Poisson
solution IS the baryonic mass distribution; the band is a smooth 1D
observational estimate that is not the natural input for CDM.

STVG retains the Poisson rho_disc for its Yukawa correction integral (the
disc density is required for the 3D volume integral; a 1D velocity curve
is insufficient).

The Poisson phi_N[j] is used for the vertical potential by all models — it
is the best available 3D estimate of phi(R,z) and is already calibrated to
band draw j by step1.

Usage
-----
  # nbar1 (MI only, default band):
  python scripts/step2_bc.py --full --no-stvg \
      --baryonic-dir outputs/nbar1 \
      --outdir outputs/bc_nbar1

  # nbar4 (MI+LW+B2+MM):
  python scripts/step2_bc.py --full --no-stvg \
      --baryon-band outputs/nbar4/baryon_band_nbar4.csv \
      --baryonic-dir outputs/nbar4 \
      --outdir outputs/bc_nbar4

  # Quick validation (5 draws):
  python scripts/step2_bc.py --full --no-stvg --n-draws 5 \
      --baryonic-dir outputs/nbar1 --outdir outputs/bc_test

Outputs
-------
  <outdir>/model_bc_{key}_radial.csv
  <outdir>/model_bc_{key}_vertical.csv
  <outdir>/mc100_chi2_all_models_bc.csv
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
    predict_mond_proxy,
    predict_qumond_solver,
    predict_stvg,
    predict_hmg_common_s,
    hmg_radial, e2d_k2_vertical,
    predict_cdm_nfw_per_draw, predict_cdm_einasto_per_draw,
)
from vgrav.baryonic import (
    build_mc100_draws,
    imig_precompute,
    imig_calibrate_weights,
    disc_rho_from_weights,
    reconstruct_rho_from_weights,
    reconstruct_phi_from_weights,
    build_component_grid_from_rho_cyl,
)
from vgrav.solver import make_grid, monopole_boundary

MODEL_K = {
    "baryonic": 0,
    "veg_fixed": 0, "veg_free": 1,
    "hmg_k1": 1,
    "hmg_sr": 1,
    "e2d_k2": 2,
    "fr_screened": 2, "refracted_gravity": 2,
    "cdm_nfw": 2, "cdm_einasto": 3,
    "stvg": 2,
}

_QUMOND_KIND = {"mond_simple": "simple", "mond_standard": "standard", "mond_rar": "rar"}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_baryonic_draws(rad_path: Path, vert_path: Path):
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


def _write_model_csvs(out_dir: Path, key: str, r_grid, rz_grid, rad_curves,
                      vert_curves, chi2_r_list, chi2_z_list) -> None:
    n = len(rad_curves)
    labels = [f"b{i+1}" for i in range(n)]
    rad_path = out_dir / f"model_bc_{key}_radial.csv"
    with open(rad_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc"] + labels)
        for i, R in enumerate(r_grid):
            w.writerow([f"{R:.6g}"] + [f"{rad_curves[j][i]:.6g}" for j in range(n)])
        w.writerow(["chi2_radial"] + [f"{chi2_r_list[j]:.6g}" for j in range(n)])
    vert_path = out_dir / f"model_bc_{key}_vertical.csv"
    with open(vert_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc", "z_kpc"] + labels)
        for i, (R, z) in enumerate(rz_grid):
            w.writerow([f"{R:.6g}", f"{z:.6g}"] + [f"{vert_curves[j][i]:.6g}" for j in range(n)])
        w.writerow(["chi2_vertical", ""] + [f"{chi2_z_list[j]:.6g}" for j in range(n)])
    print(f"    Written: {rad_path.name} + {vert_path.name}")


# ── Per-draw fitters (identical to step2 but accept v_n_rad, not vc_n) ───────
# v_n_rad = band draw interpolated to r_grid = baryonic v_N for radial term.
# phi_n   = Poisson vertical potential (paired with this draw via step1).

def _fit_fr(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    # v_n_rad: band, for radial chi²; vc_n: Poisson, for vertical nu (consistent with phi_n).
    g_n = v_n_rad ** 2 / np.maximum(r_grid, 1e-8)
    g_n_z = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC
    x_n_z = g_n_z / A0_KMS2_PER_KPC

    def obj(t):
        d, xc = t
        if d <= 0 or xc <= 1e-3:
            return 1e30
        nu = nu_fr(x_n, d, xc)
        vc_m = np.sqrt(np.maximum(nu * v_n_rad ** 2, 0.0))
        nu_z = nu_fr(x_n_z, d, xc)
        nu_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(nu_obs * phi_n, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    best = None
    for x0 in [(0.33, 1.0), (0.5, 2.0), (0.2, 0.5), (1.0, 3.0), (0.7, 1.5)]:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    d_opt, xc_opt = best.x if best is not None else (0.33, 1.0)
    nu_r = nu_fr(x_n, d_opt, xc_opt)
    nu_z = nu_fr(x_n_z, d_opt, xc_opt)
    nu_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
    return np.sqrt(np.maximum(nu_r * v_n_rad ** 2, 0.0)), nu_obs * phi_n


def _fit_rg(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    g_n = v_n_rad ** 2 / np.maximum(r_grid, 1e-8)
    g_n_z = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC
    x_n_z = g_n_z / A0_KMS2_PER_KPC

    def obj(t):
        ei, xc = t
        if ei <= 0 or ei >= 1 or xc <= 1e-3:
            return 1e30
        nu = nu_rg(x_n, ei, xc)
        vc_m = np.sqrt(np.maximum(nu * v_n_rad ** 2, 0.0))
        nu_z = nu_rg(x_n_z, ei, xc)
        nu_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(nu_obs * phi_n, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    best = None
    for x0 in [(0.18, 0.7), (0.3, 1.0), (0.1, 0.5), (0.5, 2.0)]:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    ei_opt, xc_opt = best.x if best is not None else (0.18, 0.7)
    nu_r = nu_rg(x_n, ei_opt, xc_opt)
    nu_z = nu_rg(x_n_z, ei_opt, xc_opt)
    nu_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
    return np.sqrt(np.maximum(nu_r * v_n_rad ** 2, 0.0)), nu_obs * phi_n


def _fit_veg(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    g_n = v_n_rad ** 2 / np.maximum(r_grid, 1e-8)
    g_n_z = vc_n ** 2 / np.maximum(r_grid, 1e-8)

    def score(log_a):
        a = math.exp(log_a)
        nu = nu_eg(g_n, a)
        vc_m = np.sqrt(np.maximum(nu * v_n_rad ** 2, 0.0))
        nu_z = nu_eg(g_n_z, a)
        nu_at_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(nu_at_obs * phi_n, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    res = minimize_scalar(score, bounds=(math.log(10.0), math.log(50000.0)),
                          method="bounded", options={"xatol": 1e-6})
    a_opt = math.exp(res.x)
    nu_r = nu_eg(g_n, a_opt)
    nu_z = nu_eg(g_n_z, a_opt)
    nu_at_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
    return np.sqrt(np.maximum(nu_r * v_n_rad ** 2, 0.0)), nu_at_obs * phi_n


def _fit_hmg_sr(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    """Fit s-HMG: HMG radial (Eq.37) + E-2D vertical with r_hyd→∞ (clip disabled). k=1.

    Uses the E-2D vertical form with r_hyd set to 1e9 kpc so the clipping radius
    never activates (all MW vertical data are at r3d < 11 kpc).
    NOTE: predict_hmg_common_s (s_p-HMG) uses Eq.27 vertical — a different functional
    form that assumes a point-mass density distribution. This version (s-HMG) uses the
    E-2D extended-disc vertical form, which is the standard single-parameter model.
    """
    v2_n_rad = v_n_rad ** 2
    v2_n_vert = np.interp(rv_obs, r_grid, vc_n) ** 2
    v2n_fn = lambda r: np.interp(r, r_grid, v2_n_rad, left=v2_n_rad[0], right=v2_n_rad[-1])
    R_HYD_INF = 1e9

    def score(log_s):
        s = math.exp(float(log_s))
        vc_m = hmg_radial(v2_n_rad, r_grid, s)
        phi_m = e2d_k2_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, v2n_fn, s, R_HYD_INF)
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(score, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    s_opt = math.exp(float(res.x)) if res is not None else 0.43
    return (hmg_radial(v2_n_rad, r_grid, s_opt),
            e2d_k2_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, v2n_fn, s_opt, R_HYD_INF),
            s_opt)


def _fit_e2d_k2(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z):
    """Fit E-2D 2p: neighbourhood scale s and clipping radius r_hyd (both in log space)."""
    from vgrav.models import e2d_k2_radial, e2d_k2_vertical
    v2_n_rad = v_n_rad ** 2
    v2_n_vert = np.interp(rv_obs, r_grid, vc_n) ** 2
    v2n_fn = lambda r: np.interp(r, r_grid, v2_n_rad, left=v2_n_rad[0], right=v2_n_rad[-1])

    def score(t):
        s = math.exp(float(t[0]))
        r_hyd = math.exp(float(t[1]))
        vc_m = e2d_k2_radial(v2_n_rad, r_grid, v2n_fn, s, r_hyd)
        phi_m = e2d_k2_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, v2n_fn, s, r_hyd)
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    starts = [(math.log(s0), math.log(r0))
              for s0 in (0.2, 0.43, 1.0, 2.5)
              for r0 in (12.0, 1e5)]
    best = None
    for x0 in starts:
        try:
            res = minimize(score, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    s_opt = math.exp(float(best.x[0])) if best is not None else 0.43
    rh_opt = math.exp(float(best.x[1])) if best is not None else 18.0
    return (e2d_k2_radial(v2_n_rad, r_grid, v2n_fn, s_opt, rh_opt),
            e2d_k2_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, v2n_fn, s_opt, rh_opt),
            s_opt, rh_opt)


def _fit_stvg(r_grid, v_n_rad, phi_n, rv_obs, zv_obs, comp_grid,
              rr, vv, ss, phi_obs, sig_phi, sig_z, mu0: float = 0.0678):
    def score(t):
        log_alpha, log_mu = t
        try:
            vc_m, phi_m = predict_stvg(r_grid, v_n_rad, phi_n, rv_obs, zv_obs,
                                       comp_grid, math.exp(log_alpha), math.exp(log_mu))
        except Exception:
            return 1e30
        return (chi2_radial(vc_m, r_grid, rr, vv, ss)
                + chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z))

    best = None
    for x0 in [(math.log(10.68), math.log(mu0)),
               (math.log(5.0), math.log(0.05)),
               (math.log(20.0), math.log(0.10))]:
        try:
            r = minimize(score, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    a_opt = math.exp(best.x[0]) if best is not None else 10.68
    m_opt = math.exp(best.x[1]) if best is not None else mu0
    return predict_stvg(r_grid, v_n_rad, phi_n, rv_obs, zv_obs, comp_grid, a_opt, m_opt)


# ── Full mode ─────────────────────────────────────────────────────────────────

def _run_full(out_dir: Path, band_path: Path | None, baryonic_dir: Path,
              n_draws_limit: int | None, no_stvg: bool,
              models_filter: list | None, approx: bool = True,
              full_likelihood: bool = False, radial_variant: str = "all") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _rot, vert = load_observations()
    catalog = ROOT / "data" / "fig2_observational_catalog.csv"
    if full_likelihood:
        import json
        import vgrav.chi2 as chi2_module
        from vgrav.observations import radial_variant_arrays
        chi2_module.LOG_SIGMA_TERM = True
        rr, vv, ss = radial_variant_arrays(radial_variant, catalog)
        rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert, phi_sys_fraction=0.05)
        (out_dir / "likelihood_setup.json").write_text(json.dumps({
            "radial_variant": radial_variant, "n_radial": int(len(rr)), "n_vertical": int(len(rv_obs)),
            "radial_log_term": chi2_module.radial_log_term(ss),
            "note": "chi2_vertical rows hold -2 ln L_vert; add radial_log_term to chi2_radial + chi2_vertical for -2 ln L",
        }, indent=2))
        print(f"  Full likelihood: radial variant {radial_variant} ({len(rr)} pts), "
              f"systematics on, ln sigma term on", flush=True)
    else:
        rr, vv, ss = radial_fit_arrays(chi2_catalog_path=catalog)
        rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    rad_p = baryonic_dir / "mc100_baryonic_radial.csv"
    vert_p = baryonic_dir / "mc100_baryonic_vertical.csv"
    r_grid, rz_grid, rv, zv, rad_bary, vert_bary = _read_baryonic_draws(rad_p, vert_p)
    n_draws = min(len(rad_bary), n_draws_limit) if n_draws_limit else len(rad_bary)
    print(f"  Band-consistent (BC): {n_draws} draws from {rad_p.parent.name}")
    print(f"  r_grid={len(r_grid)} pts; Poisson phi_N from {vert_p.name}")

    r_line, band_draws = build_mc100_draws(band_path=band_path)
    target_v_per_draw = [vc_tgt for _, vc_tgt, _ in band_draws[:n_draws]]
    print(f"  Band r_line={len(r_line)} pts.", flush=True)

    # Precompute 3D Imig basis only if needed (QUMOND 3D or STVG).
    # With --approx --no-stvg all algebraic models run without astropy.
    needs_precomp = (not approx) or (not no_stvg)
    if needs_precomp:
        print("  Computing Imig+2025 calibration basis (13 Poisson solves)...", flush=True)
        cyl_grid_c = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=281, nz=641)
        precomp_c = imig_precompute(cyl_grid_c)
        weights_per_draw = [
            imig_calibrate_weights(precomp_c, vc_tgt, r_line)
            for vc_tgt in target_v_per_draw
        ]
        print("  Imig weights: done.", flush=True)
    else:
        cyl_grid_c = precomp_c = None
        weights_per_draw = [None] * n_draws

    model_specs = [
        ("baryonic",          0, "bary"),
        ("mond_simple",       0, "mond_simple"),
        ("mond_standard",     0, "mond_standard"),
        ("mond_rar",          0, "mond_rar"),
        ("veg_fixed",         0, "veg_fixed"),
        ("veg_free",          1, "veg_free"),
        ("hmg_k1",            1, "hmg_sp"),   # s_p-HMG (point-mass approx., not standard)
        ("hmg_sr",            1, "hmg_sr"),   # s-HMG (standard, E-2D vertical)
        ("e2d_k2",            2, "e2d_k2"),
        ("fr_screened",       2, "fr"),
        ("refracted_gravity", 2, "rg"),
        ("cdm_nfw",           2, "cdm_nfw"),
        ("cdm_einasto",       3, "cdm_einasto"),
        ("stvg",              2, "stvg"),
    ]
    if no_stvg:
        model_specs = [(k, kv, t) for k, kv, t in model_specs if k != "stvg"]
    if models_filter:
        model_specs = [(k, kv, t) for k, kv, t in model_specs if k in models_filter]

    chi2_rows = []

    for key, k_val, tag in model_specs:
        print(f"\n  --- {key} (k={k_val}) ---", flush=True)
        rad_out, vert_out, chi2_r_list, chi2_z_list = [], [], [], []
        params_list: list[dict] = []

        for i in range(n_draws):
            vc_n = rad_bary[i]                                # Poisson vc_n — for vertical only
            phi_n = vert_bary[i]                              # Poisson phi — always kept
            target_v = target_v_per_draw[i]                   # band draw j on r_line
            v_n_rad = np.interp(r_grid, r_line, target_v)     # band → r_grid (radial)
            g_n = v_n_rad ** 2 / np.maximum(r_grid, 1e-8)
            weights_i = weights_per_draw[i]

            if tag == "bary":
                vc_m, phi_m = v_n_rad.copy(), phi_n.copy()

            elif tag in _QUMOND_KIND:
                kind = _QUMOND_KIND[tag]
                if approx:
                    # Algebraic proxy — uses v_n_rad (band) for the MOND boost.
                    vc_m, phi_m = predict_mond_proxy(r_grid, v_n_rad, phi_n, rv_obs, kind)
                else:
                    # 3D QUMOND solver — uses rho_3d from density weights.
                    # Weights are calibrated to the same band draw j (via step1 pairing),
                    # so this is already band-consistent without needing v_n_rad.
                    rho_3d = reconstruct_rho_from_weights(
                        weights_i, precomp_c.stellar, precomp_c.fixed, cyl_grid_c
                    )
                    phi_n_full = reconstruct_phi_from_weights(weights_i, precomp_c)
                    mass_n = float(np.sum(
                        rho_3d * 2.0 * math.pi * cyl_grid_c.R[:, None]
                        * cyl_grid_c.dR * cyl_grid_c.dz
                    ))
                    boundary_n = monopole_boundary(cyl_grid_c, mass_n)
                    vc_m, phi_m = predict_qumond_solver(
                        cyl_grid_c, rho_3d, boundary_n,
                        r_grid, rv_obs, zv_obs, kind,
                        phi_n_precomputed=phi_n_full,
                    )

            elif tag == "veg_fixed":
                # Radial: band g_n; vertical: Poisson g_n (consistent with phi_n).
                nu = nu_eg(g_n, A_EG_FIXED)
                vc_m = np.sqrt(np.maximum(nu * v_n_rad ** 2, 0.0))
                g_n_z = vc_n ** 2 / np.maximum(r_grid, 1e-8)
                nu_z = nu_eg(g_n_z, A_EG_FIXED)
                nu_at_obs = np.interp(rv_obs, r_grid, nu_z, left=nu_z[0], right=nu_z[-1])
                phi_m = nu_at_obs * phi_n

            elif tag == "veg_free":
                vc_m, phi_m = _fit_veg(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs,
                                       rr, vv, ss, phi_obs, sig_phi, sig_z)

            elif tag == "hmg_sp":
                # s_p-HMG (point-mass approx.): Eq.37 radial + Eq.27 vertical via predict_hmg_common_s.
                # Not the standard model — kept for reference. See hmg_sr (s-HMG) for the standard.
                v2_n_rad = target_v ** 2
                v2_n_vert = np.interp(rv_obs, r_grid, vc_n) ** 2  # Poisson at obs
                vc_m_rline, phi_m, _ = predict_hmg_common_s(
                    r_line, v2_n_rad, v2_n_vert, phi_n,
                    rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
                )
                vc_m = np.interp(r_grid, r_line, vc_m_rline)

            elif tag == "hmg_sr":
                # s-HMG (standard, k=1): HMG radial (Eq.37) + E-2D vertical (r_hyd→∞).
                vc_m, phi_m, _s = _fit_hmg_sr(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs,
                                               rr, vv, ss, phi_obs, sig_phi, sig_z)
                params_list.append({"draw_id": i + 1, "draw_label": f"b{i+1}", "s": _s})

            elif tag == "cdm_nfw":
                # CDM: vc_total² = vc_baryon² + vc_halo². Baryonic input IS the
                # Poisson 3D solution (vc_n), not the band — CDM adds a dark halo
                # on top of the actual 3D mass distribution.
                vc_m, phi_m, _ = predict_cdm_nfw_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "cdm_einasto":
                vc_m, phi_m, _ = predict_cdm_einasto_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "stvg":
                rho_disc_i = disc_rho_from_weights(weights_i, precomp_c)
                comp_grid = build_component_grid_from_rho_cyl(rho_disc_i, cyl_grid_c)
                vc_m, phi_m = _fit_stvg(
                    r_grid, v_n_rad, phi_n, rv_obs, zv_obs, comp_grid,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "e2d_k2":
                vc_m, phi_m, _s, _rh = _fit_e2d_k2(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs,
                                                     rr, vv, ss, phi_obs, sig_phi, sig_z)
                params_list.append({"draw_id": i + 1, "draw_label": f"b{i+1}", "s": _s, "r_hyd": _rh})

            elif tag == "fr":
                vc_m, phi_m = _fit_fr(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs,
                                      rr, vv, ss, phi_obs, sig_phi, sig_z)

            elif tag == "rg":
                vc_m, phi_m = _fit_rg(r_grid, v_n_rad, vc_n, phi_n, rv_obs, zv_obs,
                                      rr, vv, ss, phi_obs, sig_phi, sig_z)

            else:
                raise ValueError(f"Unknown model tag: {tag!r} (key={key!r})")

            rad_out.append(vc_m)
            vert_out.append(phi_m)
            cr = chi2_radial(vc_m, r_grid, rr, vv, ss)
            cz = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
            chi2_r_list.append(cr)
            chi2_z_list.append(cz)
            chi2_rows.append({
                "model_key": key, "draw": i + 1, "k": k_val,
                "chi2_r": cr, "chi2_z": cz,
                "chi2_nu": (cr + cz) / max(N_PRIMARY - k_val, 1),
            })

            if (i + 1) % 10 == 0 or i == n_draws - 1:
                nu_now = (cr + cz) / max(N_PRIMARY - k_val, 1)
                print(f"    draw {i+1:3d}/{n_draws}  chi2_nu={nu_now:.3f}", flush=True)

        chi2_med = np.median([chi2_r_list[j] + chi2_z_list[j] for j in range(n_draws)])
        print(f"    >>> {key}: chi2_nu p50={chi2_med / max(N_PRIMARY - k_val, 1):.3f}", flush=True)
        _write_model_csvs(out_dir, key, r_grid, rz_grid, rad_out, vert_out,
                          chi2_r_list, chi2_z_list)
        if params_list:
            params_path = out_dir / f"model_bc_{key}_params.csv"
            with open(params_path, "w", newline="", encoding="utf-8") as _fh:
                _w = csv.DictWriter(_fh, fieldnames=list(params_list[0].keys()))
                _w.writeheader()
                _w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v
                              for k, v in row.items()} for row in params_list)
            p0 = params_list[0]
            if "s" in p0 and "r_hyd" in p0:
                _sv = [r["s"] for r in params_list]
                _rv = [r["r_hyd"] for r in params_list]
                _ps, _pr = np.percentile(_sv, [16, 50, 84]), np.percentile(_rv, [16, 50, 84])
                print(f"    params: s={_ps[1]:.3f} +{_ps[2]-_ps[1]:.3f}/-{_ps[1]-_ps[0]:.3f}"
                      f"  r_hyd={_pr[1]:.2f} +{_pr[2]-_pr[1]:.2f}/-{_pr[1]-_pr[0]:.2f} kpc"
                      f"  -> {params_path.name}", flush=True)
            else:
                param_keys = [k for k in p0 if k not in ("draw_id", "draw_label")]
                parts = []
                for pk in param_keys:
                    vals = [r[pk] for r in params_list]
                    pp = np.percentile(vals, [16, 50, 84])
                    parts.append(f"{pk}={pp[1]:.3f} +{pp[2]-pp[1]:.3f}/-{pp[1]-pp[0]:.3f}")
                print(f"    params: {' '.join(parts)}  -> {params_path.name}", flush=True)

    # Summary chi2 CSV
    chi2_path = out_dir / "mc100_chi2_all_models_bc.csv"
    with open(chi2_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model_key", "draw", "k", "chi2_r", "chi2_z", "chi2_nu"])
        w.writeheader()
        for row in chi2_rows:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\n  Summary: {chi2_path}")

    print("\n=== BC chi2_nu summary (p16/p50/p84) ===")
    import collections
    by_model = collections.defaultdict(list)
    for row in chi2_rows:
        by_model[row["model_key"]].append(row["chi2_nu"])
    for key, k_val, _ in model_specs:
        vals = by_model[key]
        if vals:
            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            print(f"  {key:<26} {p50:.3f} (+{p84-p50:.3f}/-{p50-p16:.3f})  k={k_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Run the fit (required)")
    parser.add_argument("--n-draws", type=int, default=None, metavar="N")
    parser.add_argument("--no-stvg", action="store_true", help="Skip STVG (~2 h)")
    parser.add_argument("--approx", action="store_true", default=True,
                        help="QUMOND: algebraic proxy instead of 3D solver (default: on)")
    parser.add_argument("--no-approx", action="store_false", dest="approx",
                        help="QUMOND: use full 3D solver (slow)")
    parser.add_argument("--models", default=None, metavar="M1,M2",
                        help="Comma-separated model keys to run")
    parser.add_argument("--baryon-band", default=None, metavar="PATH",
                        help="Band CSV (default: data/baryon_band.csv = nbar1)")
    parser.add_argument("--baryonic-dir", default=None, metavar="DIR",
                        help="Directory with mc100_baryonic_*.csv "
                             "(e.g. outputs/nbar1 for the MI reconstruction). "
                             "No usable default — must be specified explicitly.")
    parser.add_argument("--outdir", default=None, metavar="DIR",
                        help="Output directory (default: outputs/bc)")
    parser.add_argument("--full-likelihood", action="store_true",
                        help="Wang et al. (2026) likelihood: ln sigma term and systematic errors")
    parser.add_argument("--radial-variant", default="all",
                        help="all, no_eilers, no_wang, overlap_correlated or wang_only")
    args = parser.parse_args()

    if not args.full:
        print("Use --full to run the band-consistent fits.")
        return

    band_path = Path(args.baryon_band) if args.baryon_band else None
    baryonic_dir = Path(args.baryonic_dir) if args.baryonic_dir else ROOT / "outputs"
    out_dir = Path(args.outdir) if args.outdir else ROOT / "outputs" / "bc"
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    _run_full(out_dir=out_dir, band_path=band_path, baryonic_dir=baryonic_dir,
              n_draws_limit=args.n_draws, no_stvg=args.no_stvg,
              models_filter=models_filter, approx=args.approx,
              full_likelihood=args.full_likelihood, radial_variant=args.radial_variant)


if __name__ == "__main__":
    main()
