"""Wang 78-point vertical-potential subset — all gravity models — jointly optimised disc scales.

Fits each gravity model to the Wang 78-point vertical-potential subset with
the thin-disc (z_t) and thick-disc (z_k) normalisation factors optimised
jointly with the model parameters.  Uses the pure centre curve of each
baryonic reconstruction (not MC100 stochastic draws).

Computation is organised in three phases by solver cost:
  Phase 1 (algebraic):  baryonic, VEG, f(R), ReG, CDM, HMG  — 4 baryonic reconstructions
  Phase 2 (3D Poisson): QUMOND simple / standard / MLS        — 4 baryonic reconstructions
  Phase 3 (3D Yukawa):  STVG                                  — 4 baryonic reconstructions

Usage
-----
  python scripts/step_table2_wang78_joint_disc.py             # full computation
  python scripts/step_table2_wang78_joint_disc.py --table-only  # print LaTeX from existing CSV

Outputs
-------
  outputs/wang78_table2_joint_disc.csv   (chi2_nu per baryonic reconstruction and model)
  outputs/wang78_table2_joint_disc.log   (redirect stdout)

Acceptance check (four baryonic reconstructions: MI / LW / B2 / MM)
--------------------------------------------------------------------
Expected chi2_nu (MI / LW / B2 / MM):
  Baryonic Newtonian:  5.00 / 10.25 / 11.42 /  6.10
  QUMOND simple:       1.82 /  4.23 /  1.59 /  2.48
  QUMOND standard:     1.68 /  3.88 /  2.29 /  2.20
  QUMOND (RAR):        1.69 /  4.12 /  1.53 /  2.37
  VEG (fixed):         5.72 /  6.44 /  1.31 /  6.06
  VEG (free):          1.56 /  3.92 /  1.32 /  2.15
  f(R) screened:       1.00 /  2.76 /  0.83 /  1.51
  Refracted gravity:   0.99 /  2.80 /  0.83 /  1.50
  STVG:                1.05 /  2.64 /  1.00 /  1.42
  CDM-NFW:             1.44 /  1.66 /  2.83 /  1.30
  CDM-Einasto:         1.26 /  1.49 /  2.36 /  1.14
  HMG (This Work):     1.30 /  1.41 /  2.08 /  1.16
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT    = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, vertical_arrays
from vgrav.baryonic import (
    stellar_density, gas_density,
    imig_precompute, calibrate_imig_draw,
    disc_rho_from_weights,
    reconstruct_rho_from_weights,
    build_component_grid_from_rho_cyl,
)
from vgrav.solver import (
    make_grid, phi_difference, radial_speed, blend_outer, monopole_boundary,
)
from vgrav._constants import R_SUN, A0_KMS2_PER_KPC, A_EG_FIXED, G
from vgrav.models import (
    predict_hmg_common_s, _cdm_score,
    einasto_density_from_local, einasto_mass,
    nfw_density_from_local, nfw_mass,
    nu_fr, nu_rg, nu_eg,
    predict_qumond_solver,
    predict_stvg,
)
from vgrav.chi2 import chi2_radial, chi2_vertical

# ── Paths and configuration ────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_CSV  = OUT_DIR / "wang78_table2_joint_disc.csv"

# baryon_band_nbar4.csv has centre columns for all four baryonic reconstructions.
BAND_FILE = DATA_DIR / "baryon_band_nbar4.csv"

CONFIGS = ["mcgaugh_imig", "lian_wang", "de_salas", "mcmillan"]
CONFIG_LABEL = {
    "mcgaugh_imig": "MI",
    "lian_wang":    "LW",
    "de_salas":     "B2",
    "mcmillan":     "MM",
}
CONFIG_CENTER = {
    "mcgaugh_imig": "center_McGaugh2018_Imig2025",
    "lian_wang":    "center_Wang2026_Lian2022",
    "de_salas":     "center_deSalas2019_B2",
    "mcmillan":     "center_McMillan2017",
}

FAST_MODELS = [
    ("baryonic",          2),
    ("veg_fixed",         2),
    ("veg_free",          3),
    ("fr_screened",       4),
    ("refracted_gravity", 4),
    ("cdm_nfw",           4),
    ("cdm_einasto",       5),
    ("hmg",               3),
]
QUMOND_MODELS = [
    ("qumond_simple",   2, "simple"),
    ("qumond_standard", 2, "standard"),
    ("qumond_mls",      2, "rar"),
]
STVG_MODEL = ("stvg", 4)

# Table 2 display order and LaTeX row labels
TABLE_ROWS = [
    # (model_key,          LaTeX label,           k,  notes)
    ("baryonic",          "Baryonic Newtonian",   0,  "star"),
    ("qumond_simple",     "QUMOND simple",         0,  ""),
    ("qumond_standard",   "QUMOND standard",       0,  ""),
    ("qumond_mls",        "QUMOND (RAR)",          0,  ""),
    ("veg_fixed",         "VEG (fixed)",           0,  ""),
    ("veg_free",          "VEG (free)",            1,  ""),
    ("fr_screened",       "$f(R)$ screened",       2,  ""),
    ("refracted_gravity", "Refracted gravity",     2,  ""),
    ("stvg",              "STVG",                  2,  ""),
    ("cdm_nfw",           "CDM-NFW",               2,  ""),
    ("cdm_einasto",       "CDM-Einasto",           3,  ""),
    ("hmg",               "HMG (This Work)",       1,  ""),
]

DISC_BOUNDS = (0.70, 1.30)
_ZT_VALS    = np.linspace(0.70, 1.30, 5)
_ZK_VALS    = np.linspace(0.70, 1.30, 5)
CYL_NR, CYL_NZ = 281, 641
CSV_FIELDS  = ["config", "model", "n_D", "dof", "chi2_nu", "zt", "zk"]


# ── CSV incremental write ──────────────────────────────────────────────────────

def _append_row(row: dict) -> None:
    exists = OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _save(config, model_key, n_D, N, chi2_nu, zt, zk) -> None:
    _append_row({"config": config, "model": model_key,
                 "n_D": str(n_D), "dof": str(N - n_D),
                 "chi2_nu": f"{chi2_nu:.4f}",
                 "zt": f"{zt:.4f}", "zk": f"{zk:.4f}"})


def _already_done(config, model_key) -> bool:
    if not OUT_CSV.exists():
        return False
    with open(OUT_CSV, newline="", encoding="utf-8") as fh:
        return any(r["config"] == config and r["model"] == model_key
                   for r in csv.DictReader(fh))


# ── Helpers ────────────────────────────────────────────────────────────────────

def disc_fractions(r_arr):
    R   = np.asarray(r_arr, dtype=float)
    z0  = np.zeros_like(R)
    hz_t = 0.39 * np.exp(0.027 * (R - R_SUN))
    hz_k = 0.85 * np.exp(0.057 * (R - R_SUN))
    st   = stellar_density(R, z0, thin=True)  * 2.0 * hz_t
    sk   = stellar_density(R, z0, thin=False) * 2.0 * hz_k
    shi  = gas_density(R, z0, hi=True)  * 4.0 * 0.085
    sh2  = gas_density(R, z0, hi=False) * 4.0 * 0.045
    tot  = np.maximum(st + sk + shi + sh2, 1e-30)
    return st / tot, sk / tot


def _sc(fT, fK, zt, zk):
    return zt * fT + zk * fK + (1.0 - fT - fK)


def _safe(v):
    f = float(v)
    return f if np.isfinite(f) else 1e10


def _multistart(obj, starts, bounds):
    best = None
    for x0 in starts:
        try:
            r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 800, "ftol": 1e-12})
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    return best


def scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, zt, zk):
    sc_r = _sc(fT_r,  fK_r,  zt, zk)
    sc_v = _sc(fT_rv, fK_rv, zt, zk)
    return (np.sqrt(np.maximum(sc_r, 0.0)) * basis["vc_n"],
            sc_v * basis["phi_n"])


def _nu_chi2(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
             fT_r, fK_r, fT_rv, fK_rv, zt, zk, nu_func):
    r_line = basis["r_line"]
    vc_sc, phi_sc = scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, zt, zk)
    g_n  = vc_sc**2 / np.maximum(r_line, 1e-8)
    nu   = nu_func(g_n)
    vc_m = np.sqrt(np.maximum(nu * vc_sc**2, 0.0))
    nu_v = np.interp(rv, r_line, nu, left=nu[0], right=nu[-1])
    phi_m = nu_v * phi_sc
    return (chi2_radial(vc_m, r_line, rr, vv, ss) +
            chi2_vertical(phi_m, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_wang_radial():
    rr, vv, ss = [], [], []
    with open(DATA_DIR / "wang2026_rotation_curve.csv", newline="") as f:
        for row in csv.DictReader(f):
            rr.append(float(row["R_kpc"]))
            vv.append(float(row["vc_kms"]))
            ss.append(float(row["sigma_total_kms"]))
    return np.array(rr), np.array(vv), np.array(ss)


# ── Baryonic basis ─────────────────────────────────────────────────────────────

def build_pure_basis(config, cyl_grid, precomp, rv, zv):
    center_col = CONFIG_CENTER[config]
    rows   = list(csv.DictReader(open(BAND_FILE)))
    r_band = np.array([float(r["R_kpc"])     for r in rows])
    vc_band = np.array([float(r[center_col]) for r in rows])
    r_line = r_band  # radial grid from the band file

    t1 = time.time()
    print(f"  calibrate_imig_draw [{config}]...", end=" ", flush=True)
    rho_N, total_mass, phi_N, weights = calibrate_imig_draw(precomp, vc_band, r_line)
    print(f"{time.time()-t1:.1f}s", flush=True)

    vc_n      = radial_speed(cyl_grid, phi_N, r_line)
    outer_vc  = np.sqrt(G * total_mass / np.maximum(r_line, 1e-9))
    vc_n      = blend_outer(r_line, vc_n, outer_vc)
    phi_n     = phi_difference(cyl_grid, phi_N, rv, zv)
    rho_3d    = reconstruct_rho_from_weights(weights, precomp.stellar, precomp.fixed, cyl_grid)
    rho_disc  = disc_rho_from_weights(weights, precomp)
    mass_n    = float(np.sum(rho_3d * 2.0 * math.pi
                              * cyl_grid.R[:, None] * cyl_grid.dR * cyl_grid.dz))
    boundary  = monopole_boundary(cyl_grid, mass_n)
    return dict(r_line=r_line, vc_n=vc_n, phi_n=phi_n,
                rho_3d=rho_3d, mass_n=mass_n, boundary_n=boundary,
                rho_disc=rho_disc, weights=weights)


# ── Fast model fits ────────────────────────────────────────────────────────────

def fit_baryonic(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                 fT_r, fK_r, fT_rv, fK_rv):
    r_line = basis["r_line"]; N = len(rr) + len(rv)
    def obj(p):
        vc_sc, phi_sc = scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, p[0], p[1])
        return _safe(chi2_radial(vc_sc, r_line, rr, vv, ss) +
                     chi2_vertical(phi_sc, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))
    starts = [[1.0,1.0],[0.9,1.0],[1.0,0.9],[1.1,1.0],[1.0,1.1],[0.8,1.2],
              [1.2,0.8],[0.9,0.9],[1.3,1.0],[1.0,1.3],[0.7,1.0],[1.0,0.7]]
    best = _multistart(obj, starts, [DISC_BOUNDS, DISC_BOUNDS])
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[0], *DISC_BOUNDS))
    zk = float(np.clip(best.x[1], *DISC_BOUNDS))
    return best.fun / (N - 2), zt, zk


def fit_nu_proxy(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                 fT_r, fK_r, fT_rv, fK_rv, n_D, nu_func):
    N = len(rr) + len(rv)
    def obj(p):
        return _safe(_nu_chi2(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                               fT_r, fK_r, fT_rv, fK_rv, float(p[0]), float(p[1]), nu_func))
    starts = [[1.0,1.0],[0.9,1.0],[1.0,0.9],[1.1,1.0],[1.0,1.1],
              [0.8,1.2],[1.2,0.8],[0.9,0.9],[0.7,1.0],[1.0,0.7]]
    best = _multistart(obj, starts, [DISC_BOUNDS, DISC_BOUNDS])
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[0], *DISC_BOUNDS))
    zk = float(np.clip(best.x[1], *DISC_BOUNDS))
    return best.fun / (N - n_D), zt, zk


def fit_veg_free(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                 fT_r, fK_r, fT_rv, fK_rv):
    N = len(rr) + len(rv); log_a0 = math.log(A_EG_FIXED)
    def obj(p):
        a_eg = math.exp(float(p[0])); zt, zk = float(p[1]), float(p[2])
        return _safe(_nu_chi2(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                               fT_r, fK_r, fT_rv, fK_rv, zt, zk, lambda g: nu_eg(g, a_eg)))
    b3 = [(math.log(10.0), math.log(50000.0)), DISC_BOUNDS, DISC_BOUNDS]
    starts = [[log_a0,1.0,1.0],[log_a0,0.9,1.0],[log_a0,1.0,0.9],
              [log_a0+1,1.0,1.0],[log_a0-1,1.0,1.0],[log_a0,1.1,1.0],[log_a0,0.8,1.2]]
    best = _multistart(obj, starts, b3)
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[1], *DISC_BOUNDS))
    zk = float(np.clip(best.x[2], *DISC_BOUNDS))
    return best.fun / (N - 3), zt, zk


def fit_fr(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
           fT_r, fK_r, fT_rv, fK_rv):
    N = len(rr) + len(rv)
    def obj(p):
        d, xc, zt, zk = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if d <= 0 or xc <= 1e-3:
            return 1e10
        return _safe(_nu_chi2(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                               fT_r, fK_r, fT_rv, fK_rv, zt, zk,
                               lambda g: nu_fr(g / A0_KMS2_PER_KPC, d, xc)))
    b4 = [(1e-3, 20.0), (1e-3, 50.0), DISC_BOUNDS, DISC_BOUNDS]
    starts = [[0.33,1.0,1.0,1.0],[0.5,2.0,1.0,1.0],[0.2,0.5,1.0,1.0],[1.0,3.0,1.0,1.0],
              [0.33,1.0,0.9,1.0],[0.33,1.0,1.0,0.9],[0.5,2.0,0.9,0.9]]
    best = _multistart(obj, starts, b4)
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[2], *DISC_BOUNDS))
    zk = float(np.clip(best.x[3], *DISC_BOUNDS))
    return best.fun / (N - 4), zt, zk


def fit_rg(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
           fT_r, fK_r, fT_rv, fK_rv):
    N = len(rr) + len(rv)
    def obj(p):
        ei, xc, zt, zk = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if not (0 < ei < 1) or xc <= 1e-3:
            return 1e10
        return _safe(_nu_chi2(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                               fT_r, fK_r, fT_rv, fK_rv, zt, zk,
                               lambda g: nu_rg(g / A0_KMS2_PER_KPC, ei, xc)))
    b4 = [(1e-4, 1.0-1e-4), (1e-3, 50.0), DISC_BOUNDS, DISC_BOUNDS]
    starts = [[0.18,0.7,1.0,1.0],[0.3,1.0,1.0,1.0],[0.1,0.5,1.0,1.0],[0.5,2.0,1.0,1.0],
              [0.18,0.7,0.9,1.0],[0.18,0.7,1.0,0.9],[0.3,1.0,0.9,0.9]]
    best = _multistart(obj, starts, b4)
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[2], *DISC_BOUNDS))
    zk = float(np.clip(best.x[3], *DISC_BOUNDS))
    return best.fun / (N - 4), zt, zk


def fit_cdm_nfw(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                fT_r, fK_r, fT_rv, fK_rv):
    r_line = basis["r_line"]; N = len(rr) + len(rv)
    def obj(p):
        lr, ls, zt, zk = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        vc_sc, phi_sc = scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, zt, zk)
        try:
            rho_s, rs = nfw_density_from_local(lr, ls)
            if not (np.isfinite(rho_s) and 0 < rho_s < 1e20):
                return 1e10
            chi2, _, _ = _cdm_score(r_line, vc_sc, phi_sc, rv, zv,
                                     rr, vv, ss, phi_obs, sig_phi, sig_z,
                                     lambda r: nfw_mass(r, rho_s, rs))
            return _safe(chi2)
        except Exception:
            return 1e10
    b2 = [(-3.0, 1.0), (0.0, 2.3)]
    best2 = _multistart(lambda p: obj([p[0],p[1],1.0,1.0]),
                        [[-0.46,1.24],[-0.50,1.0],[-0.30,1.5],[-0.60,0.8]], b2)
    t0 = best2.x if best2 is not None else np.array([-0.46, 1.24])
    b4 = [(-3.0,1.0),(0.0,2.3),DISC_BOUNDS,DISC_BOUNDS]
    starts = [[t0[0],t0[1],z,k] for z in [1.0,0.9,1.1,0.8,1.2] for k in [1.0,0.9,1.1]]
    best = _multistart(obj, starts, b4)
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[2], *DISC_BOUNDS))
    zk = float(np.clip(best.x[3], *DISC_BOUNDS))
    return best.fun / (N - 4), zt, zk


def fit_cdm_einasto(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv):
    r_line = basis["r_line"]; N = len(rr) + len(rv)
    def obj5(p):
        zt, zk, lr, ls, al = (float(p[0]), float(p[1]), float(p[2]),
                               float(p[3]), float(p[4]))
        vc_sc, phi_sc = scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, zt, zk)
        try:
            rho_s, rs, al_ = einasto_density_from_local(lr, ls, al)
            if not (np.isfinite(rho_s) and 0 < rho_s < 1e20):
                return 1e10
            chi2, _, _ = _cdm_score(r_line, vc_sc, phi_sc, rv, zv,
                                     rr, vv, ss, phi_obs, sig_phi, sig_z,
                                     lambda r: einasto_mass(r, rho_s, rs, al_))
            return _safe(chi2)
        except Exception:
            return 1e10
    b3 = [(-3.0,1.0),(0.0,2.3),(0.1,15.0)]
    best3 = _multistart(lambda p: obj5([1.0,1.0]+list(p)),
                        [[-0.27,0.99,0.97],[-0.14,0.92,2.0],[-0.30,1.20,1.5],
                         [-0.06,0.94,2.0],[-0.50,1.50,0.8]], b3)
    lr0, ls0, al0 = (best3.x if best3 is not None else np.array([-0.27,0.99,0.97]))
    b5 = [DISC_BOUNDS,DISC_BOUNDS,(-3.0,1.0),(0.0,2.3),(0.1,15.0)]
    starts5 = [[z,k,lr0,ls0,al0] for z in [1.0,0.9,1.1,0.8,1.2,0.7,1.3]
               for k in [1.0,0.9,1.1]]
    best = _multistart(obj5, starts5, b5)
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[0], *DISC_BOUNDS))
    zk = float(np.clip(best.x[1], *DISC_BOUNDS))
    return best.fun / (N - 5), zt, zk


def fit_hmg(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
            fT_r, fK_r, fT_rv, fK_rv):
    r_line = basis["r_line"]; vc_n = basis["vc_n"]; phi_n = basis["phi_n"]
    N = len(rr) + len(rv)
    fT_rv2, fK_rv2 = disc_fractions(rv)
    vc_n_rv = np.interp(rv, r_line, vc_n)
    def _call(zt, zk):
        sc_r = _sc(fT_r,   fK_r,   zt, zk)
        sc_v = _sc(fT_rv2, fK_rv2, zt, zk)
        vc_m, phi_m, _ = predict_hmg_common_s(
            r_line, sc_r * vc_n**2, sc_v * vc_n_rv**2, sc_v * phi_n,
            rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z)
        return (chi2_radial(vc_m, r_line, rr, vv, ss) +
                chi2_vertical(phi_m, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))
    def obj(p):
        try:
            return _safe(_call(float(p[0]), float(p[1])))
        except Exception:
            return 1e10
    starts = [[1.0,1.0],[0.9,1.0],[1.0,0.9],[1.1,1.0],[1.0,1.1],[0.8,1.2],[1.2,0.8],
              [0.9,0.9],[1.1,1.1],[0.7,1.3],[1.3,0.7],[1.3,1.0],[1.0,1.3],[0.7,1.0],[1.0,0.7]]
    best = _multistart(obj, starts, [DISC_BOUNDS, DISC_BOUNDS])
    if best is None:
        return float("nan"), float("nan"), float("nan")
    zt = float(np.clip(best.x[0], *DISC_BOUNDS))
    zk = float(np.clip(best.x[1], *DISC_BOUNDS))
    return _call(zt, zk) / (N - 3), zt, zk


# ── Slow model fits ────────────────────────────────────────────────────────────

def fit_qumond(config, kind, basis, cyl_grid,
               rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
               fT_cyl, fK_cyl):
    r_line = basis["r_line"]; N = len(rr) + len(rv); dof = N - 2
    best_chi2nu = float("inf"); best_zt = best_zk = float("nan")
    n_pts = len(_ZT_VALS) * len(_ZK_VALS)
    print(f"    QUMOND({kind}): {n_pts} grid points ...", flush=True)
    t0 = time.time()
    for zt in _ZT_VALS:
        for zk in _ZK_VALS:
            sc_cyl  = _sc(fT_cyl, fK_cyl, zt, zk)
            rho_sc  = sc_cyl[:, None] * basis["rho_3d"]
            mass_sc = float(np.sum(rho_sc * 2.0 * math.pi
                                    * cyl_grid.R[:, None] * cyl_grid.dR * cyl_grid.dz))
            bnd_sc  = monopole_boundary(cyl_grid, mass_sc)
            try:
                vc_m, phi_m = predict_qumond_solver(
                    cyl_grid, rho_sc, bnd_sc, r_line, rv, zv, kind)
                c = (chi2_radial(vc_m, r_line, rr, vv, ss) +
                     chi2_vertical(phi_m, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))
                c_nu = c / dof
            except Exception as e:
                print(f"      err zt={zt:.2f} zk={zk:.2f}: {e}", flush=True)
                c_nu = float("inf")
            if c_nu < best_chi2nu:
                best_chi2nu = c_nu; best_zt, best_zk = float(zt), float(zk)
    print(f"    Done {time.time()-t0:.1f}s  chi2_nu={best_chi2nu:.4f}"
          f"  zt={best_zt:.2f} zk={best_zk:.2f}", flush=True)
    return best_chi2nu, best_zt, best_zk


def fit_stvg(config, basis, cyl_grid,
             rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
             fT_cyl, fK_cyl, fT_r, fK_r, fT_rv, fK_rv):
    r_line = basis["r_line"]; N = len(rr) + len(rv); dof = N - 4
    best_chi2nu = float("inf"); best_zt = best_zk = float("nan")
    n_pts = len(_ZT_VALS) * len(_ZK_VALS)
    print(f"    STVG: {n_pts} grid points ...", flush=True)
    t0 = time.time()
    for zt in _ZT_VALS:
        for zk in _ZK_VALS:
            sc_cyl       = _sc(fT_cyl, fK_cyl, zt, zk)
            rho_disc_sc  = sc_cyl[:, None] * basis["rho_disc"]
            try:
                comp_grid = build_component_grid_from_rho_cyl(
                    rho_disc_sc, cyl_grid, nr=50, nz=25, nphi=36)
            except Exception as e:
                print(f"      comp_grid err zt={zt:.2f} zk={zk:.2f}: {e}", flush=True)
                continue
            vc_sc, phi_sc = scaled_vc_phi(basis, fT_r, fK_r, fT_rv, fK_rv, zt, zk)
            def obj_s(p, _v=vc_sc, _p=phi_sc, _cg=comp_grid):
                la, lm = float(p[0]), float(p[1])
                try:
                    vc_m, phi_m = predict_stvg(r_line, _v, _p, rv, zv,
                                                _cg, math.exp(la), math.exp(lm))
                    c = (chi2_radial(vc_m, r_line, rr, vv, ss) +
                         chi2_vertical(phi_m, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))
                    return _safe(c)
                except Exception:
                    return 1e10
            bstvg    = [(math.log(0.001), math.log(1000.0)),
                        (math.log(0.001), math.log(100.0))]
            starts_s = [[math.log(8.0), math.log(0.04)],
                        [math.log(1.0), math.log(0.1)],
                        [math.log(15.0), math.log(0.02)],
                        [math.log(5.0), math.log(0.05)]]
            best_s = _multistart(obj_s, starts_s, bstvg)
            if best_s is None:
                continue
            c_nu = best_s.fun / dof
            if c_nu < best_chi2nu:
                best_chi2nu = c_nu; best_zt, best_zk = float(zt), float(zk)
    print(f"    STVG done {time.time()-t0:.1f}s  chi2_nu={best_chi2nu:.4f}"
          f"  zt={best_zt:.2f} zk={best_zk:.2f}", flush=True)
    return best_chi2nu, best_zt, best_zk


# ── LaTeX table output ─────────────────────────────────────────────────────────

def print_latex_table() -> None:
    if not OUT_CSV.exists():
        print("CSV not found — run without --table-only first.")
        return

    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))

    def get(config, model):
        hit = next((r for r in rows if r["config"] == config
                    and r["model"] == model), None)
        return float(hit["chi2_nu"]) if hit else float("nan")

    # chi2_nu matrix: {model_key: {config: value}}
    matrix = {mk: {c: get(c, mk) for c in CONFIGS} for mk, *_ in TABLE_ROWS}

    # Top-2 per column (lowest chi2_nu bolded)
    bold = {c: set() for c in CONFIGS}
    for c in CONFIGS:
        vals = [(mk, matrix[mk][c]) for mk, *_ in TABLE_ROWS
                if np.isfinite(matrix[mk][c])]
        vals.sort(key=lambda x: x[1])
        for mk, _ in vals[:2]:
            bold[c].add(mk)

    def fmt(val, mk, c, notes):
        if not np.isfinite(val):
            return "---"
        s = f"{val:.2f}"
        if notes == "star":
            s = s + r" [$\boldsymbol{*}$]"
        if mk in bold[c]:
            s = r"\textbf{" + s + "}"
        return s

    print("\n% chi2_nu summary — Wang 78-point subset, disc scales jointly optimised")
    print(r"% Columns: Model & $n_\mathcal{D}$ & MI & LW & B2 & MM")
    print()
    for mk, label, k, notes in TABLE_ROWS:
        col_vals = " & ".join(fmt(matrix[mk][c], mk, c, notes) for c in CONFIGS)
        print(f"{label} & {k} & {col_vals} \\\\")
    print()

    # Plain-text summary
    print("\nPlain summary (config order: MI  LW  B2  MM):")
    print(f"  {'Model':<22} {'k':>3}  {'MI':>8} {'LW':>8} {'B2':>8} {'MM':>8}")
    print("  " + "-"*58)
    for mk, label, k, _ in TABLE_ROWS:
        vals = "  ".join(f"{matrix[mk][c]:8.4f}"
                         if np.isfinite(matrix[mk][c]) else "     nan"
                         for c in CONFIGS)
        print(f"  {label:<22} {k:>3}  {vals}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table-only", action="store_true",
                        help="Skip fitting; print LaTeX table from existing CSV.")
    args = parser.parse_args()

    if args.table_only:
        print_latex_table()
        return

    t_total = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    rr_w, vv_w, ss_w = load_wang_radial()
    _rot, vert = load_observations()
    rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)
    N = len(rr_w) + len(rv)
    print(f"Wang 78-point subset — jointly optimised disc scale factors")
    print(f"  {len(rr_w)} radial + {len(rv)} vertical = {N} total data points")
    print(f"  Pure baryonic centre curves; QUMOND 3D solver; STVG grid search")
    print(f"  Output: {OUT_CSV}\n", flush=True)

    t0 = time.time()
    print("Building cylindrical grid + baryonic precomputation ...", flush=True)
    cyl_grid = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=CYL_NR, nz=CYL_NZ)
    precomp  = imig_precompute(cyl_grid)
    print(f"  Done in {time.time()-t0:.1f}s\n", flush=True)

    fT_cyl, fK_cyl = disc_fractions(cyl_grid.R)

    print("-- Building baryonic bases for all four reconstructions --", flush=True)
    bases = {}
    for config in CONFIGS:
        bases[config] = build_pure_basis(config, cyl_grid, precomp, rv, zv)

    # ── Phase 1: fast models ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Phase 1 — fast models (baryonic, VEG, f(R), ReG, CDM, HMG)")
    print("="*70, flush=True)

    for config in CONFIGS:
        print(f"\n[{CONFIG_LABEL[config]} — {config}]", flush=True)
        basis  = bases[config]
        r_line = basis["r_line"]
        fT_r,  fK_r  = disc_fractions(r_line)
        fT_rv, fK_rv = disc_fractions(rv)

        for model_key, n_D in FAST_MODELS:
            if _already_done(config, model_key):
                print(f"  {model_key:<22} already in CSV — skipped.", flush=True)
                continue
            t_mod = time.time()
            print(f"  {model_key:<22} n_D={n_D} ...", end="  ", flush=True)

            if model_key == "baryonic":
                chi2_nu, zt, zk = fit_baryonic(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "veg_fixed":
                chi2_nu, zt, zk = fit_nu_proxy(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv, n_D=2,
                    nu_func=lambda g: nu_eg(g, A_EG_FIXED))
            elif model_key == "veg_free":
                chi2_nu, zt, zk = fit_veg_free(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "fr_screened":
                chi2_nu, zt, zk = fit_fr(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "refracted_gravity":
                chi2_nu, zt, zk = fit_rg(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "cdm_nfw":
                chi2_nu, zt, zk = fit_cdm_nfw(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "cdm_einasto":
                chi2_nu, zt, zk = fit_cdm_einasto(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            elif model_key == "hmg":
                chi2_nu, zt, zk = fit_hmg(
                    basis, rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                    fT_r, fK_r, fT_rv, fK_rv)
            else:
                chi2_nu, zt, zk = float("nan"), float("nan"), float("nan")

            print(f"chi2_nu={chi2_nu:.4f}  zt={zt:.4f} zk={zk:.4f}"
                  f"  ({time.time()-t_mod:.1f}s)", flush=True)
            _save(config, model_key, n_D, N, chi2_nu, zt, zk)

    # ── Phase 2: QUMOND solver ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Phase 2 — QUMOND 3D Poisson solver (3 kinds x 4 reconstructions x 25 pts)")
    print("="*70, flush=True)

    for config in CONFIGS:
        print(f"\n[{CONFIG_LABEL[config]} — {config}]", flush=True)
        basis = bases[config]
        for model_key, n_D, kind in QUMOND_MODELS:
            if _already_done(config, model_key):
                print(f"  {model_key:<22} already in CSV — skipped.", flush=True)
                continue
            print(f"  {model_key:<22} n_D={n_D}", flush=True)
            chi2_nu, zt, zk = fit_qumond(
                config, kind, basis, cyl_grid,
                rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
                fT_cyl, fK_cyl)
            _save(config, model_key, n_D, N, chi2_nu, zt, zk)

    # ── Phase 3: STVG ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Phase 3 — STVG (4 reconstructions x 25 pts)")
    print("="*70, flush=True)

    for config in CONFIGS:
        print(f"\n[{CONFIG_LABEL[config]} — {config}]", flush=True)
        if _already_done(config, STVG_MODEL[0]):
            print(f"  stvg already in CSV — skipped.", flush=True)
            continue
        basis  = bases[config]
        r_line = basis["r_line"]
        fT_r,  fK_r  = disc_fractions(r_line)
        fT_rv, fK_rv = disc_fractions(rv)
        model_key, n_D = STVG_MODEL
        print(f"  {model_key:<22} n_D={n_D}", flush=True)
        chi2_nu, zt, zk = fit_stvg(
            config, basis, cyl_grid,
            rv, zv, rr_w, vv_w, ss_w, phi_obs, sig_phi, sig_z,
            fT_cyl, fK_cyl, fT_r, fK_r, fT_rv, fK_rv)
        _save(config, model_key, n_D, N, chi2_nu, zt, zk)

    # ── Table output ───────────────────────────────────────────────────────────
    print_latex_table()
    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
