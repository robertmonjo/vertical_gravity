"""Model comparison with the full likelihood of Wang et al. (2026) and the BIC.

For every baryonic reconstruction (MI, LW, B2, MM, fitted separately) and every
radial data-vector variant, each gravity model is fitted by minimising

    -2 ln L = sum_rad (dv/sigma_v)^2 + ln(2 pi sigma_v^2)
            + sum_vert (dPhi/sigma_eff)^2 + ln(2 pi sigma_eff^2)
            + prior(rho_thin) + prior(rho_thick),

with sigma_eff^2 = sigma_Phi^2 + (K_z sigma_z)^2, a radius-dependent systematic
error (5-20 per cent, from the Wang et al. budget) on the radial points outside
the Wang curve, 5 per cent of Phi on the vertical points, and Gaussian priors
rho_thin(R0) ~ N(0.045, 0.003) and rho_thick(R0) ~ N(0.00525, 0.0006) Msun/pc^3
on the local stellar densities. The thin- and thick-disc scale factors are free
for every model. BIC = -2 ln L_max + k ln N, with k including the two disc factors.

Usage
-----
  python scripts/likelihood_bic.py --workers 24
  python scripts/likelihood_bic_stvg.py --workers 24   # STVG, appended to the same CSV
  python scripts/likelihood_bic.py --summary        # tables from the existing CSV

Output
------
  outputs/likelihood_bic/likelihood_bic_fits.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav._constants import A0_KMS2_PER_KPC, A_EG_FIXED, G
from vgrav.baryonic import imig_precompute, _WEIGHT_LOW, _WEIGHT_HIGH
from vgrav.chi2 import vertical_force_from_phi
from vgrav.models import (
    einasto_density_from_local, einasto_mass, nfw_density_from_local, nfw_mass,
    spherical_vc_and_phi, nu_eg, nu_fr, nu_rg, hmg_radial, hmg_eq27_vertical,
    e2d_k2_radial, e2d_k2_vertical, predict_qumond_solver, predict_refracted_gravity,
)
from vgrav.observations import RADIAL_VARIANTS, radial_variant_arrays, vertical_arrays
from vgrav.solver import make_grid, monopole_boundary

_spec = importlib.util.spec_from_file_location("table2", ROOT / "scripts" / "step1_table2_wang78_joint_disc.py")
table2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table2)
disc_fractions, _sc = table2.disc_fractions, table2._sc

CATALOG = ROOT / "data" / "fig2_observational_catalog.csv"
OUT_DIR = ROOT / "outputs" / "likelihood_bic"
OUT_CSV = OUT_DIR / "likelihood_bic_fits.csv"
CONFIGS = table2.CONFIGS
CONFIG_LABEL = table2.CONFIG_LABEL

RHO_THIN_PRIOR = (0.045, 0.003)
RHO_THICK_PRIOR = (0.00525, 0.0006)
PHI_SYS_FRACTION = 0.05
BOUND_SIGMAS = 10.0
COARSE_STEPS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
FINE_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)
A0_VERLINDE = 6.0 * A_EG_FIXED
FIELDS = ["variant", "config", "model", "k", "N", "m2lnL", "chi2", "chi2_nu", "BIC",
          "zt", "zk", "rho_thin", "rho_thick", "theta"]

LN = math.log
ALGEBRAIC_MODELS = {
    # key: (gravity-parameter bounds, gravity-parameter starts)
    "baryonic": ([], [[]]),
    "veg_fixed": ([], [[]]),
    "veg_free": ([(LN(10.0), LN(5.0e4))], [[LN(A_EG_FIXED)], [LN(A_EG_FIXED) + 1], [LN(A_EG_FIXED) - 1]]),
    "fr_screened": ([(1e-3, 20.0), (1e-3, 50.0)], [[0.33, 1.0], [0.5, 2.0], [0.2, 0.5], [1.0, 3.0]]),
    "fr_bounded": ([(1e-3, 1.0 / 3.0), (1e-3, 50.0)], [[0.3, 1.0], [0.3, 2.0], [0.2, 0.5], [0.3, 3.0]]),
    "veg_verlinde": ([], [[]]),
    "veg_verlinde_free": ([(LN(A0_VERLINDE / 30.0), LN(A0_VERLINDE * 30.0))],
                          [[LN(A0_VERLINDE)], [LN(A0_VERLINDE) + 1], [LN(A0_VERLINDE) - 1]]),
    "refracted_gravity": ([(1e-4, 1 - 1e-4), (1e-3, 50.0)], [[0.18, 0.7], [0.3, 1.0], [0.1, 0.5], [0.5, 2.0]]),
    "cdm_nfw": ([(-3.0, 1.0), (0.0, 2.3)], [[-0.46, 1.24], [-0.5, 1.0], [-0.3, 1.5], [-0.6, 0.8]]),
    "cdm_einasto": ([(-3.0, 1.0), (0.0, 2.3), (0.1, 15.0)],
                    [[-0.27, 0.99, 0.97], [-0.14, 0.92, 2.0], [-0.30, 1.20, 1.5], [-0.50, 1.50, 0.8]]),
    "hmg_sp": ([(LN(0.05), LN(300.0))], [[LN(0.5)], [LN(2.0)], [LN(5.0)]]),
    "hmg_sr": ([(LN(0.05), LN(300.0))], [[LN(0.5)], [LN(2.0)], [LN(5.0)]]),
    "hmg_sk2": ([(LN(0.05), LN(300.0)), (LN(1.0), LN(1.0e6))],
                [[LN(0.43), LN(3.0)], [LN(0.43), LN(17.0)], [LN(2.5), LN(7.0)], [LN(2.5), LN(100.0)]]),
    "e2d_k2": ([(LN(0.05), LN(300.0)), (LN(2.0), LN(1.0e6))],
               [[LN(0.43), LN(18.0)], [LN(2.5), LN(18.0)], [LN(0.43), LN(1e5)], [LN(2.5), LN(1e5)]]),
}
QUMOND_KINDS = {"qumond_simple": "simple", "qumond_standard": "standard", "qumond_mls": "rar"}
# Refracted gravity, nabla.[eps(rho) nabla phi] = 4 pi G rho. Universal parameters: joint fit of the
# 30 DiskMass Survey galaxies (Cesare et al. 2020, Sect. 5): eps0, Q, log10 rho_c [g cm^-3].
# Free versions: bounds = flat priors of Cesare et al. (2020, Sect. 3).
REG_DMS = (0.661, 1.79, -24.54)
REG_BOUNDS = [(0.10, 1.0), (0.01, 2.0), (-27.0, -23.0)]
REG_MODELS = {"reg_dms": False, "reg_free": True}   # key: free parameters

STATE: dict = {}


# ── Data and priors ───────────────────────────────────────────────────────────

def load_data():
    rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(phi_sys_fraction=PHI_SYS_FRACTION)
    vert = dict(rv=rv, zv=zv, phi_obs=phi_obs, sig_phi=sig_phi, sig_z=sig_z)
    radial = {v: radial_variant_arrays(v, CATALOG) for v in RADIAL_VARIANTS}
    return vert, radial


def disc_prior_centre(weights):
    c_sun = float(np.clip(weights[4], _WEIGHT_LOW, _WEIGHT_HIGH))
    return 1.0 / c_sun, c_sun


def local_densities(zt, zk, c_sun):
    return zt * c_sun * RHO_THIN_PRIOR[0], zk * c_sun * RHO_THICK_PRIOR[0]


def prior_term(zt, zk, c_sun):
    rt, rk = local_densities(zt, zk, c_sun)
    return ((rt - RHO_THIN_PRIOR[0]) / RHO_THIN_PRIOR[1]) ** 2 + ((rk - RHO_THICK_PRIOR[0]) / RHO_THICK_PRIOR[1]) ** 2


def disc_bounds(zt0, zk0):
    ft, fk = RHO_THIN_PRIOR[1] / RHO_THIN_PRIOR[0], RHO_THICK_PRIOR[1] / RHO_THICK_PRIOR[0]
    return ((zt0 * (1 - BOUND_SIGMAS * ft), zt0 * (1 + BOUND_SIGMAS * ft)),
            (max(zk0 * (1 - BOUND_SIGMAS * fk), 0.05), zk0 * (1 + BOUND_SIGMAS * fk)))


def disc_point(zt0, zk0, i, j):
    """Disc factors displaced by i and j prior standard deviations of the thin and thick discs."""
    ft, fk = RHO_THIN_PRIOR[1] / RHO_THIN_PRIOR[0], RHO_THICK_PRIOR[1] / RHO_THICK_PRIOR[0]
    return zt0 * (1 + i * ft), max(zk0 * (1 + j * fk), 0.05)


def disc_grid(zt0, zk0):
    return [disc_point(zt0, zk0, i, j) for i in COARSE_STEPS for j in COARSE_STEPS]


def score(vc_line, r_line, phi_m, radial, vert):
    """Return (-2 ln L without priors, chi2) for one data vector."""
    rr, vv, ss = radial
    vc_at = np.interp(rr, r_line, vc_line)
    chi_r = float(np.sum(((vc_at - vv) / ss) ** 2))
    log_r = float(np.sum(np.log(2.0 * np.pi * ss ** 2)))
    kz = vertical_force_from_phi(phi_m, vert["rv"], vert["zv"])
    var = vert["sig_phi"] ** 2 + (kz * vert["sig_z"]) ** 2
    chi_z = float(np.sum((phi_m - vert["phi_obs"]) ** 2 / var))
    log_z = float(np.sum(np.log(2.0 * np.pi * var)))
    return chi_r + log_r + chi_z + log_z, chi_r + chi_z


# ── Model predictions ─────────────────────────────────────────────────────────

def scaled_inputs(config, zt, zk):
    b = STATE["bases"][config]
    r_line, vc_n, phi_n, rv = b["r_line"], b["vc_n"], b["phi_n"], STATE["vert"]["rv"]
    fT_r, fK_r = disc_fractions(r_line)
    fT_v, fK_v = disc_fractions(rv)
    sc_r, sc_v = _sc(fT_r, fK_r, zt, zk), _sc(fT_v, fK_v, zt, zk)
    v2_rad = np.maximum(sc_r, 0.0) * vc_n ** 2
    v2_vert = np.maximum(sc_v, 0.0) * np.interp(rv, r_line, vc_n) ** 2
    return r_line, v2_rad, v2_vert, sc_v * phi_n


def verlinde_mass_function(config, zt, zk, a0):
    """Apparent dark matter of Verlinde (2016, Eq. 7.40): M_D^2 = a0 r^2/(6G) d(M_B r)/dr."""
    sph = STATE["spherical"][config]
    m_cells = sph["w"] * np.maximum(_sc(sph["fT"], sph["fK"], zt, zk), 0.0)
    m_enc = np.interp(sph["r_edges"], sph["r_sorted"], np.cumsum(m_cells[sph["order"]]))
    d_mr = np.gradient(m_enc * sph["r_edges"], sph["r_edges"])
    m_d = np.sqrt(np.maximum(a0 * sph["r_edges"] ** 2 / (6.0 * G) * d_mr, 0.0))
    return lambda r: np.interp(r, sph["r_edges"], m_d)


def predict_algebraic(model, config, theta, zt, zk):
    r_line, v2_rad, v2_vert, phi_sc = scaled_inputs(config, zt, zk)
    rv, zv = STATE["vert"]["rv"], STATE["vert"]["zv"]
    vc_sc = np.sqrt(v2_rad)
    if model == "baryonic":
        return vc_sc, phi_sc
    if model in ("veg_fixed", "veg_free", "fr_screened", "fr_bounded", "refracted_gravity"):
        g_n = v2_rad / np.maximum(r_line, 1e-8)
        if model == "veg_fixed":
            nu = nu_eg(g_n, A_EG_FIXED)
        elif model == "veg_free":
            nu = nu_eg(g_n, math.exp(theta[0]))
        elif model in ("fr_screened", "fr_bounded"):
            nu = nu_fr(g_n / A0_KMS2_PER_KPC, theta[0], theta[1])
        else:
            nu = nu_rg(g_n / A0_KMS2_PER_KPC, theta[0], theta[1])
        return np.sqrt(nu) * vc_sc, np.interp(rv, r_line, nu, left=nu[0], right=nu[-1]) * phi_sc
    if model in ("veg_verlinde", "veg_verlinde_free"):
        a0 = A0_VERLINDE if model == "veg_verlinde" else math.exp(theta[0])
        vc_h, phi_h = spherical_vc_and_phi(r_line, rv, zv, verlinde_mass_function(config, zt, zk, a0))
        return np.sqrt(v2_rad + vc_h ** 2), phi_sc + phi_h
    if model in ("cdm_nfw", "cdm_einasto"):
        if model == "cdm_nfw":
            rho_s, rs = nfw_density_from_local(theta[0], theta[1])
            mass = lambda r: nfw_mass(r, rho_s, rs)
        else:
            rho_s, rs, al = einasto_density_from_local(theta[0], theta[1], theta[2])
            mass = lambda r: einasto_mass(r, rho_s, rs, al)
        vc_h, phi_h = spherical_vc_and_phi(r_line, rv, zv, mass)
        return np.sqrt(v2_rad + vc_h ** 2), phi_sc + phi_h
    s = math.exp(theta[0])
    v2_fn = lambda r: np.interp(r, r_line, v2_rad, left=v2_rad[0], right=v2_rad[-1])
    if model == "hmg_sp":
        return hmg_radial(v2_rad, r_line, s), hmg_eq27_vertical(rv, zv, v2_vert, phi_sc, s)
    if model == "hmg_sr":
        return hmg_radial(v2_rad, r_line, s), e2d_k2_vertical(rv, zv, v2_vert, phi_sc, v2_fn, s, 1e9)
    r_h = math.exp(theta[1])
    if model == "hmg_sk2":
        return hmg_radial(v2_rad, r_line, s), e2d_k2_vertical(rv, zv, v2_vert, phi_sc, v2_fn, s, r_h)
    return (e2d_k2_radial(v2_rad, r_line, v2_fn, s, r_h),
            e2d_k2_vertical(rv, zv, v2_vert, phi_sc, v2_fn, s, r_h))


# ── Fits ──────────────────────────────────────────────────────────────────────

def row(variant, config, model, k, best_m2lnl, best_chi2, zt, zk, theta, c_sun):
    n = len(STATE["radial"][variant][0]) + len(STATE["vert"]["rv"])
    rt, rk = local_densities(zt, zk, c_sun)
    return dict(variant=variant, config=config, model=model, k=k, N=n,
                m2lnL=f"{best_m2lnl:.4f}", chi2=f"{best_chi2:.4f}", chi2_nu=f"{best_chi2 / (n - k):.4f}",
                BIC=f"{best_m2lnl + k * math.log(n):.4f}", zt=f"{zt:.4f}", zk=f"{zk:.4f}",
                rho_thin=f"{rt:.5f}", rho_thick=f"{rk:.6f}", theta=json.dumps([round(float(t), 5) for t in theta]))


def fit_algebraic(task):
    config, model = task
    zt0, zk0, c_sun = STATE["prior"][config]
    bounds_g, starts_g = ALGEBRAIC_MODELS[model]
    bounds = list(disc_bounds(zt0, zk0)) + bounds_g
    r_line = STATE["bases"][config]["r_line"]
    rows = []
    for variant in RADIAL_VARIANTS:
        radial = STATE["radial"][variant]

        def objective(p):
            try:
                vc, phi = predict_algebraic(model, config, p[2:], p[0], p[1])
                m2, _ = score(vc, r_line, phi, radial, STATE["vert"])
                value = m2 + prior_term(p[0], p[1], c_sun)
                return value if np.isfinite(value) else 1e12
            except Exception:
                return 1e12

        best = None
        for g0 in starts_g:
            for dz in ((0, 0), (1, 1), (-1, -1)):
                x0 = [zt0 * (1 + 0.0667 * dz[0]), zk0 * (1 + 0.114 * dz[1])] + list(g0)
                try:
                    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 1500, "ftol": 1e-12})
                except Exception:
                    continue
                if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                    best = res
        p = best.x
        vc, phi = predict_algebraic(model, config, p[2:], p[0], p[1])
        m2, chi2 = score(vc, r_line, phi, radial, STATE["vert"])
        rows.append(row(variant, config, model, 2 + len(bounds_g), m2 + prior_term(p[0], p[1], c_sun),
                        chi2, p[0], p[1], p[2:], c_sun))
    return rows


def scaled_density(config, zt, zk):
    grid = STATE["cyl_grid"]
    fT_c, fK_c = disc_fractions(grid.R)
    return _sc(fT_c, fK_c, zt, zk)[:, None] * STATE["bases"][config]["rho_3d"]


def solved_prediction(config, model, zt, zk, theta=()):
    """v_c on r_line and Phi_z at the vertical points for the models solved on the grid."""
    b, grid = STATE["bases"][config], STATE["cyl_grid"]
    rho = scaled_density(config, zt, zk)
    rv, zv = STATE["vert"]["rv"], STATE["vert"]["zv"]
    if model in QUMOND_KINDS:
        mass = float(np.sum(rho * 2.0 * math.pi * grid.R[:, None] * grid.dR * grid.dz))
        return predict_qumond_solver(grid, rho, monopole_boundary(grid, mass), b["r_line"], rv, zv,
                                     QUMOND_KINDS[model])
    eps0, q, log_rho_c = tuple(theta) if REG_MODELS[model] else REG_DMS
    return predict_refracted_gravity(grid, rho, b["r_line"], rv, zv, eps0, q, log_rho_c)


def fit_qumond(task):
    """Models solved on the grid with no free gravity parameter: QUMOND and refracted gravity
    with the DiskMass parameters. Disc factors on a coarse grid of prior widths, refined
    around the best point of each radial variant."""
    config, model = task
    zt0, zk0, c_sun = STATE["prior"][config]
    b = STATE["bases"][config]
    best = {v: None for v in RADIAL_VARIANTS}
    cache = {}

    def evaluate(i, j):
        if (i, j) in cache:
            return
        zt, zk = disc_point(zt0, zk0, i, j)
        try:
            cache[(i, j)] = (zt, zk) + tuple(solved_prediction(config, model, zt, zk))
        except Exception:
            cache[(i, j)] = None

    def penalised(entry, v):
        zt, zk, vc, phi = entry
        m2, chi2 = score(vc, b["r_line"], phi, STATE["radial"][v], STATE["vert"])
        return m2 + prior_term(zt, zk, c_sun), chi2

    for i in COARSE_STEPS:
        for j in COARSE_STEPS:
            evaluate(i, j)
    for v in RADIAL_VARIANTS:
        ib, jb = min((k for k in cache if cache[k] is not None), key=lambda k: penalised(cache[k], v)[0])
        for di in FINE_STEPS:
            for dj in FINE_STEPS:
                evaluate(ib + di, jb + dj)
    for (i, j), entry in cache.items():
        if entry is None:
            continue
        zt, zk, vc, phi = entry
        for v in RADIAL_VARIANTS:
            m2, chi2 = score(vc, b["r_line"], phi, STATE["radial"][v], STATE["vert"])
            total = m2 + prior_term(zt, zk, c_sun)
            if best[v] is None or total < best[v][0]:
                best[v] = (total, chi2, zt, zk)
    return [row(v, config, model, 2, t, c, zt, zk, [], c_sun) for v, (t, c, zt, zk) in best.items()]


def fit_reg_free(task):
    """Refracted gravity with eps0, Q and log10 rho_c free (k = 5), one radial variant."""
    config, model, variant = task
    zt0, zk0, c_sun = STATE["prior"][config]
    r_line = STATE["bases"][config]["r_line"]
    radial = STATE["radial"][variant]
    bounds = list(disc_bounds(zt0, zk0)) + REG_BOUNDS

    def objective(p):
        try:
            vc, phi = solved_prediction(config, model, p[0], p[1], p[2:])
            m2, _ = score(vc, r_line, phi, radial, STATE["vert"])
            value = m2 + prior_term(p[0], p[1], c_sun)
            return value if np.isfinite(value) else 1e12
        except Exception:
            return 1e12

    best = None
    for g0 in (REG_DMS, (0.3, 1.0, -25.5), (0.9, 0.5, -24.0)):
        res = minimize(objective, [zt0, zk0, *g0], method="Nelder-Mead", bounds=bounds,
                       options={"maxfev": 400, "xatol": 1e-3, "fatol": 1e-3})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    p = best.x
    vc, phi = solved_prediction(config, model, p[0], p[1], p[2:])
    m2, chi2 = score(vc, r_line, phi, radial, STATE["vert"])
    return [row(variant, config, model, 5, m2 + prior_term(p[0], p[1], c_sun), chi2, p[0], p[1], p[2:], c_sun)]


# ── Driver ────────────────────────────────────────────────────────────────────

def build_state(configs):
    t0 = time.time()
    vert, radial = load_data()
    grid = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=table2.CYL_NR, nz=table2.CYL_NZ)
    precomp = imig_precompute(grid)
    bases, prior = {}, {}
    for config in configs:
        bases[config] = table2.build_pure_basis(config, grid, precomp, vert["rv"], vert["zv"])
        zt0, c_sun = disc_prior_centre(bases[config]["weights"])
        prior[config] = (zt0, zt0, c_sun)
    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    r_cells = np.sqrt(RR ** 2 + ZZ ** 2).ravel()
    order = np.argsort(r_cells)
    fT_cells, fK_cells = disc_fractions(RR.ravel())
    spherical = {c: dict(w=(bases[c]["rho_3d"] * 2.0 * math.pi * RR * grid.dR * grid.dz).ravel(),
                         fT=fT_cells, fK=fK_cells, order=order, r_sorted=r_cells[order],
                         r_edges=np.geomspace(0.05, 400.0, 800)) for c in configs}
    STATE.update(vert=vert, radial=radial, cyl_grid=grid, bases=bases, prior=prior, spherical=spherical)
    print(f"State built in {time.time() - t0:.1f}s; prior centres "
          + ", ".join(f"{CONFIG_LABEL[c]}: c_sun={prior[c][2]:.3f}" for c in configs), flush=True)


def write_rows(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def summary():
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
    for variant in RADIAL_VARIANTS:
        print(f"\n=== {variant} ===")
        sub = [r for r in rows if r["variant"] == variant]
        models = sorted({r["model"] for r in sub})
        print(f"{'model':<20}{'k':>3} " + "".join(f"{CONFIG_LABEL[c] + ' chi2nu':>12}{'dBIC':>8}" for c in CONFIGS))
        best_bic = {c: min(float(r["BIC"]) for r in sub if r["config"] == c) for c in CONFIGS
                    if any(r["config"] == c for r in sub)}
        for m in sorted(models, key=lambda m: float(next(r["BIC"] for r in sub if r["model"] == m))):
            cells = ""
            for c in CONFIGS:
                hit = next((r for r in sub if r["model"] == m and r["config"] == c), None)
                cells += (f"{float(hit['chi2_nu']):12.2f}{float(hit['BIC']) - best_bic[c]:8.1f}"
                          if hit else f"{'---':>12}{'':>8}")
            k = next(r["k"] for r in sub if r["model"] == m)
            print(f"{m:<20}{k:>3} {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--models", default=None, help="comma-separated subset; default all")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    configs = args.configs.split(",")
    wanted = set(args.models.split(",")) if args.models else None
    build_state(configs)
    ctx = get_context("fork")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        tasks = [(c, m) for c in configs for m in ALGEBRAIC_MODELS if not wanted or m in wanted]
        for rows in pool.imap_unordered(fit_algebraic, tasks):
            write_rows(rows)
            print(f"  [{time.time() - t0:7.1f}s] {rows[0]['config']:<13} {rows[0]['model']:<18} "
                  f"chi2_nu(all)={rows[0]['chi2_nu']}", flush=True)
        solved = list(QUMOND_KINDS) + [m for m, free in REG_MODELS.items() if not free]
        tasks = [(c, m) for c in configs for m in solved if not wanted or m in wanted]
        for rows in pool.imap_unordered(fit_qumond, tasks):
            write_rows(rows)
            print(f"  [{time.time() - t0:7.1f}s] {rows[0]['config']:<13} {rows[0]['model']:<18} "
                  f"chi2_nu(all)={rows[0]['chi2_nu']}", flush=True)
        tasks = [(c, m, v) for c in configs for m, free in REG_MODELS.items() if free
                 and (not wanted or m in wanted) for v in RADIAL_VARIANTS]
        for rows in pool.imap_unordered(fit_reg_free, tasks):
            write_rows(rows)
            print(f"  [{time.time() - t0:7.1f}s] {rows[0]['config']:<13} {rows[0]['model']:<18} "
                  f"{rows[0]['variant']:<18} chi2_nu={rows[0]['chi2_nu']}", flush=True)
    print(f"Total {(time.time() - t0) / 60:.1f} min. Output: {OUT_CSV}")
    summary()


if __name__ == "__main__":
    main()
