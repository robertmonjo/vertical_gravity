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
      Algebraic (f(R), ReG, VEG, HMG):     < 10 min each
      CDM (NFW, Einasto per-draw):          < 10 min each
      QUMOND 3D Poisson solver:             ~ 2 h per variant (3 variants, ~6 h total)
      STVG direct summation:                ~ 2 h  (350k cells × 100 draws)

    Flags for faster reproduction:
      --approx       Use algebraic proxy for QUMOND instead of 3D Poisson.
                     Fast (< 5 min per variant), good approximation, but not
                     identical to the published results.
      --no-stvg      Skip STVG (~2 h savings). Other models are unaffected.
      --models M1,M2 Run only the listed model keys (comma-separated).

    Recommended workflow for full reproduction (split terminal sessions):
      Terminal A:  python step2 --full --no-stvg          # ~1 h, all except STVG
      Terminal B:  python step2 --full --models stvg       # ~2 h, STVG alone

    Output: outputs/model_{key}_radial.csv and _vertical.csv for each model.

Model keys (12 total including baryonic reference)
---------------------------------------------------
  baryonic, qumond_simple, qumond_standard, qumond_mls,
  veg_fixed, veg_free, stvg, cdm_nfw, cdm_einasto,
  hmg_k1 (k=1 common s), fr_screened, refracted_gravity
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
    stvg_disk_accel_and_phi,
    stvg_bulge_yukawa_resolved,
    predict_hmg_common_s,
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

OUT = ROOT / "outputs"

# ── k values (free parameters) for reduced chi2 ───────────────────────────────
MODEL_K = {
    "baryonic": 0,
    "qumond_simple": 0, "qumond_standard": 0, "qumond_mls": 0,
    "veg_fixed": 0, "veg_free": 1,
    "stvg": 2,
    "cdm_nfw": 2, "cdm_einasto": 3,
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


# Parameter column specs per model tag
_PARAM_COLS: dict[str, list[str]] = {
    "hmg":         ["s"],
    "cdm_nfw":     ["log_rho", "log_rs"],
    "cdm_einasto": ["log_rho", "log_rs", "alpha"],
    "veg_free":    ["a_eg"],
    "fr":          ["delta_sc", "xc"],
    "rg":          ["eps_inf", "xc"],
    "stvg":        ["alpha", "mu"],
}


def _write_params_csv(key: str, tag: str, params_out: list) -> None:
    """Write per-draw fitted parameters to model_{key}_params.csv."""
    cols = _PARAM_COLS.get(tag)
    if not cols or not any(p is not None for p in params_out):
        return
    path = OUT / f"model_{key}_params.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["draw_id", "draw_label"] + cols)
        for i, theta in enumerate(params_out):
            label = f"b{i+1}"
            if theta is None:
                row = [i + 1, label] + [""] * len(cols)
            elif len(cols) == 1:
                row = [i + 1, label, f"{theta:.8g}"]
            else:
                row = [i + 1, label] + [f"{v:.8g}" for v in theta]
            w.writerow(row)
    print(f"  Written: model_{key}_params.csv ({len(cols)} param(s): {', '.join(cols)})")


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
    d_opt, xc_opt = best.x if best is not None else (0.33, 1.0)
    nu = nu_fr(x_n, d_opt, xc_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_obs * phi_n, (d_opt, xc_opt)


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
    ei_opt, xc_opt = best.x if best is not None else (0.18, 0.7)
    nu = nu_rg(x_n, ei_opt, xc_opt)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    return vc_out, nu_obs * phi_n, (ei_opt, xc_opt)


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
    return vc_out, nu_at_obs * phi_n, a_opt


def _fit_stvg(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid, rr, vv, ss, phi_obs, sig_phi, sig_z,
    mu0: float = 0.0678,
):
    """Fit STVG (alpha, mu) per draw by minimising chi2_total.  k=2.

    During optimization, STVG accel is evaluated only at the observed radial
    positions (len(rr) ≈ 152) instead of the full r_grid (448), which is ~3×
    cheaper and avoids the interpolation step.  The final curve is evaluated on
    the full r_grid for CSV output.
    """
    # Pre-interpolate Newton quantities at observed radial positions.
    vc_n_at_rr = np.interp(rr, r_grid, vc_n)
    g_n_at_rr  = vc_n_at_rr ** 2 / np.maximum(rr, 1e-8)

    def score(t):
        log_alpha, log_mu = t
        alpha = math.exp(log_alpha)
        mu    = math.exp(log_mu)
        try:
            a_y, ph_y         = stvg_disk_accel_and_phi(rr, rv_obs, zv_obs, comp_grid, alpha, mu)
            a_yb, ph_yb       = stvg_bulge_yukawa_resolved(rr, rv_obs, zv_obs, alpha, mu)
            vc2               = rr * ((1.0 + alpha) * g_n_at_rr - a_y - a_yb)
            vc_m_at_rr        = np.sqrt(np.maximum(vc2, 0.0))
            chi_r             = float(np.sum(((vc_m_at_rr - vv) / ss) ** 2))
            phi_m             = (1.0 + alpha) * phi_n + ph_y + ph_yb
            chi_z             = chi2_vertical(
                phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z
            )
        except Exception:
            return 1e30
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
                         options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    alpha_opt = math.exp(best.x[0]) if best is not None else math.exp(starts[0][0])
    mu_opt    = math.exp(best.x[1]) if best is not None else mu0
    vc_out, phi_out = predict_stvg(r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid, alpha_opt, mu_opt)
    return vc_out, phi_out, (alpha_opt, mu_opt)


# ── Full mode ─────────────────────────────────────────────────────────────────

_QUMOND_KIND = {
    "mond_simple":   "simple",
    "mond_standard": "standard",
    "mond_rar":      "rar",
}

_NEEDS_3D = frozenset(["mond_simple", "mond_standard", "mond_rar", "stvg"])


def _run_full(
    n_draws_limit: int | None = None,
    models_filter: list | None = None,
    approx: bool = False,
    no_stvg: bool = False,
    band_path: Path | None = None,
) -> None:
    label = f"first {n_draws_limit}" if n_draws_limit else "all"
    qumond_mode = "algebraic proxy (--approx)" if approx else "3D Poisson solver"
    print(f"Step 2 — Fitting gravity models ({label} draws)")
    print(f"  QUMOND mode : {qumond_mode}")
    print(f"  STVG        : {'SKIPPED (--no-stvg)' if no_stvg else 'included (~2 h)'}")
    _rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(
        chi2_catalog_path=ROOT / "data" / "fig2_observational_catalog.csv"
    )
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

    # Imig+2025 calibration basis — one-time cost (13 Poisson solves).
    # r_min=0.0 required: axis handled by symmetry condition, not Dirichlet BC.
    print("  Computing Imig+2025 calibration basis (r_min=0.0)...", flush=True)
    cyl_grid_c = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=281, nz=641)
    precomp_c = imig_precompute(cyl_grid_c)
    print("  Imig+2025 basis ready (r_min=0.0).", flush=True)

    # Per-draw Imig calibration weights (same seed as step1, deterministic).
    # lsq_linear only — no Poisson per draw here; phi_N comes from step1 CSV.
    print("  Calibrating Imig density weights for all draws...", flush=True)
    r_line, band_draws = build_mc100_draws(band_path=band_path)
    weights_per_draw = [
        imig_calibrate_weights(precomp_c, vc_tgt, r_line)
        for _, vc_tgt, _ in band_draws[:n_draws]
    ]

    # QUMOND 3D Poisson grid (r_min=0.1): inner Dirichlet BC where nu≈1 → no divergence.
    # Only computed when 3D solver is requested (not --approx).
    needs_qumond_solver = (not approx) and (
        models_filter is None or any(m in (models_filter or []) for m in
                                     ["qumond_simple", "qumond_standard", "qumond_mls"])
    )
    if needs_qumond_solver:
        cyl_grid_q = cyl_grid_c
        precomp_q = precomp_c
        weights_q_per_draw = weights_per_draw
    else:
        cyl_grid_q = precomp_q = weights_q_per_draw = None  # type: ignore[assignment]
    # Smooth MC100 band targets on r_line — used as v_N input for HMG radial,
    # matching the default fit_hmg_common_s_mc100.py which uses target_v (not Poisson).
    target_v_per_draw = [vc_tgt for _, vc_tgt, _ in band_draws[:n_draws]]
    print(f"  Imig weights: done ({n_draws} draws).", flush=True)

    OUT.mkdir(exist_ok=True)

    # Ordered fastest → slowest so --no-stvg produces useful output quickly.
    model_specs = [
        ("baryonic",          0, "bary"),          # < 1 min
        ("veg_fixed",         0, "veg_fixed"),      # < 1 min
        ("veg_free",          1, "veg_free"),        # < 2 min
        ("hmg_k1",            1, "hmg"),             # < 5 min (one common s)
        ("fr_screened",       2, "fr"),              # < 5 min
        ("refracted_gravity", 2, "rg"),              # < 5 min
        ("cdm_nfw",           2, "cdm_nfw"),         # < 10 min
        ("cdm_einasto",       3, "cdm_einasto"),     # < 10 min
        ("stvg",              2, "stvg"),            # ~2 h  (run separately: --models stvg)
        ("qumond_simple",     0, "mond_simple"),     # ~2 h  (3D Poisson)
        ("qumond_standard",   0, "mond_standard"),   # ~2 h  (3D Poisson)
        ("qumond_mls",        0, "mond_rar"),        # ~2 h  (3D Poisson)
    ]

    if no_stvg:
        model_specs = [(k, kv, t) for k, kv, t in model_specs if k != "stvg"]
    if models_filter:
        model_specs = [(k, kv, t) for k, kv, t in model_specs if k in models_filter]

    for key, k_val, tag in model_specs:
        print(f"\n  --- {key} (k={k_val}) ---", flush=True)
        rad_out: list[np.ndarray] = []
        vert_out: list[np.ndarray] = []
        chi2_r_list: list[float] = []
        chi2_z_list: list[float] = []
        params_out: list = []

        for i in range(n_draws):
            vc_n = rad_bary[i]
            phi_n = vert_bary[i]
            g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
            weights_i = weights_per_draw[i]
            theta = None  # fitted parameter(s) for this draw; None if model has no free params

            if tag == "bary":
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            elif tag in _QUMOND_KIND:
                kind = _QUMOND_KIND[tag]
                if approx:
                    vc_m, phi_m = predict_mond_proxy(r_grid, vc_n, phi_n, rv_obs, kind)
                else:
                    weights_q_i = weights_q_per_draw[i]
                    rho_3d = reconstruct_rho_from_weights(
                        weights_q_i, precomp_q.stellar, precomp_q.fixed, cyl_grid_q
                    )
                    phi_n_full = reconstruct_phi_from_weights(weights_q_i, precomp_q)
                    mass_n = float(np.sum(
                        rho_3d * 2.0 * math.pi * cyl_grid_q.R[:, None]
                        * cyl_grid_q.dR * cyl_grid_q.dz
                    ))
                    boundary_n = monopole_boundary(cyl_grid_q, mass_n)
                    vc_m, phi_m = predict_qumond_solver(
                        cyl_grid_q, rho_3d, boundary_n,
                        r_grid, rv_obs, zv_obs, kind,
                        phi_n_precomputed=phi_n_full,
                    )

            elif tag == "veg_fixed":
                nu = nu_eg(g_n, A_EG_FIXED)
                vc_m = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
                nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
                phi_m = nu_at_obs * phi_n

            elif tag == "veg_free":
                vc_m, phi_m, theta = _fit_veg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "stvg":
                # Full grid (2.7M cells) for accuracy; ~25 min/draw sequentially.
                # For faster parallel execution use step2_stvg_parallel.py instead.
                rho_disc_i = disc_rho_from_weights(weights_i, precomp_c)
                comp_grid = build_component_grid_from_rho_cyl(
                    rho_disc_i, cyl_grid_c, nr=190, nz=90, nphi=80
                )
                vc_m, phi_m, theta = _fit_stvg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "cdm_nfw":
                vc_m, phi_m, theta = predict_cdm_nfw_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "cdm_einasto":
                vc_m, phi_m, theta = predict_cdm_einasto_per_draw(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "hmg":
                # Canonical (fit_hmg_common_s_mc100.py) uses smooth band target_v
                # for hmg_radial, and 3D Poisson vc_n for hmg_eq27_vertical.
                v2_n_rad = target_v_per_draw[i] ** 2     # smooth target on r_line
                v2_n_vert = np.interp(rv_obs, r_grid, vc_n) ** 2  # Poisson at obs
                vc_m_rline, phi_m, theta = predict_hmg_common_s(
                    r_line, v2_n_rad, v2_n_vert, phi_n,
                    rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
                )
                # Interpolate HMG curve from r_line back to r_grid for CSV output.
                vc_m = np.interp(r_grid, r_line, vc_m_rline)

            elif tag == "fr":
                vc_m, phi_m, theta = _fit_fr(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            elif tag == "rg":
                vc_m, phi_m, theta = _fit_rg(
                    r_grid, vc_n, phi_n, rv_obs, zv_obs,
                    rr, vv, ss, phi_obs, sig_phi, sig_z,
                )

            else:
                vc_m, phi_m = vc_n.copy(), phi_n.copy()

            params_out.append(theta)
            rad_out.append(vc_m)
            vert_out.append(phi_m)
            cr = chi2_radial(vc_m, r_grid, rr, vv, ss)
            cz = chi2_vertical(phi_m, rv_obs, zv_obs, rv_obs, zv_obs, phi_obs, sig_phi, sig_z)
            chi2_r_list.append(cr)
            chi2_z_list.append(cz)

            if (i + 1) % 10 == 0 or i == n_draws - 1:
                nu_now = (cr + cz) / max(N_PRIMARY - k_val, 1)
                print(f"    draw {i+1:3d}/{n_draws}  chi2_nu={nu_now:.2f}", flush=True)

        chi2_med = np.median([chi2_r_list[j] + chi2_z_list[j] for j in range(n_draws)])
        print(f"    >>> {key} done: chi2_nu p50={chi2_med / max(N_PRIMARY - k_val, 1):.3f}",
              flush=True)
        _write_model_csvs(key, r_grid, rz_grid, rad_out, vert_out, chi2_r_list, chi2_z_list)
        _write_params_csv(key, tag, params_out)
        print(f"    Written: model_{key}_radial/vertical.csv", flush=True)

    print(f"\nStep 2 complete ({n_draws} draws, {len(model_specs)} models).", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true",
                        help="Regenerate model predictions (slow)")
    parser.add_argument("--n-draws", type=int, default=None, metavar="N",
                        help="Process only the first N draws (for quick validation)")
    parser.add_argument("--models", type=str, default=None, metavar="M1,M2",
                        help="Comma-separated subset of model keys (e.g. veg_fixed,veg_free)")
    parser.add_argument("--approx", action="store_true",
                        help="QUMOND: algebraic proxy instead of 3D Poisson solver. "
                             "Fast (<5 min/variant) but not identical to published results.")
    parser.add_argument("--no-stvg", action="store_true",
                        help="Skip STVG (~2 h savings). Run separately with --models stvg "
                             "in a parallel terminal for full reproduction.")
    parser.add_argument(
        "--outdir", default=None, metavar="DIR",
        help="Directory for model CSVs (default: outputs/). "
             "Must contain mc100_baryonic_*.csv from step1.",
    )
    parser.add_argument(
        "--baryon-band", default=None, metavar="PATH",
        help="Path to baryon_band.csv (for parallel nbar mode; overrides data/baryon_band.csv).",
    )
    args = parser.parse_args()

    global OUT
    if args.outdir:
        OUT = Path(args.outdir)

    band_path = Path(args.baryon_band) if args.baryon_band else None
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    if args.full or args.n_draws or models_filter:
        _run_full(
            n_draws_limit=args.n_draws,
            models_filter=models_filter,
            approx=args.approx,
            no_stvg=args.no_stvg,
            band_path=band_path,
        )
    else:
        _verify_fast()


if __name__ == "__main__":
    main()
