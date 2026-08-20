"""s/r_h-HMG hybrid — extra Table 2 values (not shown in paper).

Fits the s/r_h-HMG hybrid model to the Wang 78-point subset for all four
baryonic reconstructions.  The model uses the one-parameter radial formula
(hmg_radial, same as s-HMG) together with the two-parameter vertical formula
(e2d_k2_vertical, same as sr_h-HMG E-2D), so it has k=2 free parameters
(s, r_hyd) plus the jointly-optimised disc scales (zt, zk).

These values are mentioned in the paper (Monjo 2026) as:
  "...yields chi2_nu = 1.02--1.69 across the four baryonic ensembles (not shown)."

Acceptance check (MI / LW / B2 / MM):
  s/r_h-HMG:  1.086 / 1.329 / 1.687 / 1.020

Note: the paper text rounds to "1.02--1.69"; the minimum is MM=1.020, not MI=1.086.

Runtime: ~5--10 min (dominated by the Poisson-solver baryonic precomputation,
same as step1_table2_wang78_joint_disc.py Phase 1).

Usage
-----
  python scripts/extras/hmg_hybrid_table2_extra.py

Outputs
-------
  outputs/extras/hmg_hybrid_table2_extra.csv
"""
from __future__ import annotations

import csv
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT   = Path(__file__).resolve().parents[2]   # release_final/
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, vertical_arrays
from vgrav.baryonic import (
    imig_precompute, calibrate_imig_draw,
    disc_rho_from_weights, reconstruct_rho_from_weights,
    stellar_density, gas_density,
)
from vgrav.solver import (
    make_grid, phi_difference, radial_speed, blend_outer, monopole_boundary,
)
from vgrav._constants import R_SUN, G
from vgrav.models import hmg_radial, e2d_k2_vertical
from vgrav.chi2 import chi2_radial, chi2_vertical

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs" / "extras"
OUT_CSV  = OUT_DIR / "hmg_hybrid_table2_extra.csv"

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

DISC_BOUNDS = (0.70, 1.30)
CYL_NR, CYL_NZ = 281, 641

# Expected values for self-check
EXPECTED = {
    "mcgaugh_imig": 1.086,
    "lian_wang":    1.329,
    "de_salas":     1.687,
    "mcmillan":     1.020,
}

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_wang_radial():
    rr, vv, ss = [], [], []
    with open(DATA_DIR / "wang2026_rotation_curve.csv", newline="") as f:
        for row in csv.DictReader(f):
            rr.append(float(row["R_kpc"]))
            vv.append(float(row["vc_kms"]))
            ss.append(float(row["sigma_total_kms"]))
    return np.array(rr), np.array(vv), np.array(ss)


def disc_fractions(r_arr):
    R    = np.asarray(r_arr, dtype=float)
    z0   = np.zeros_like(R)
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


def build_basis(config, cyl_grid, precomp, rv, zv):
    center_col = CONFIG_CENTER[config]
    rows    = list(csv.DictReader(open(BAND_FILE)))
    r_line  = np.array([float(r["R_kpc"])      for r in rows])
    vc_band = np.array([float(r[center_col])    for r in rows])
    t0 = time.time()
    print(f"  calibrate [{config}] ...", end=" ", flush=True)
    rho_N, total_mass, phi_N, weights = calibrate_imig_draw(precomp, vc_band, r_line)
    print(f"{time.time()-t0:.1f}s", flush=True)
    vc_n     = radial_speed(cyl_grid, phi_N, r_line)
    outer_vc = np.sqrt(G * total_mass / np.maximum(r_line, 1e-9))
    vc_n     = blend_outer(r_line, vc_n, outer_vc)
    phi_n    = phi_difference(cyl_grid, phi_N, rv, zv)
    return dict(r_line=r_line, vc_n=vc_n, phi_n=phi_n)


# ── Model fit ──────────────────────────────────────────────────────────────────

def fit_hmg_hybrid(basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
                   fT_r, fK_r, fT_rv, fK_rv):
    """s/r_h-HMG: hmg_radial radially + e2d_k2_vertical vertically.

    Free parameters: s (neighbourhood scale), r_hyd (hydro radius), zt, zk.
    DOF = N - 4.
    """
    r_line   = basis["r_line"]
    vc_n     = basis["vc_n"]
    phi_n    = basis["phi_n"]
    N        = len(rr) + len(rv)
    fT_rv2, fK_rv2 = disc_fractions(rv)
    vc_n_rv  = np.interp(rv, r_line, vc_n)

    def score(p):
        zt   = float(np.clip(p[0], *DISC_BOUNDS))
        zk   = float(np.clip(p[1], *DISC_BOUNDS))
        s    = math.exp(float(p[2]))
        r_hyd = math.exp(float(p[3]))
        sc_r  = _sc(fT_r,   fK_r,   zt, zk)
        sc_v  = _sc(fT_rv2, fK_rv2, zt, zk)
        v2_n_rad  = (sc_r * vc_n)    ** 2
        v2_n_vert = (sc_v * vc_n_rv) ** 2
        phi_n_sc  = sc_v * phi_n
        v2n_fn    = lambda r: np.interp(r, r_line, v2_n_rad,
                                        left=v2_n_rad[0], right=v2_n_rad[-1])
        try:
            vc_m  = hmg_radial(v2_n_rad, r_line, s)
            phi_m = e2d_k2_vertical(rv, zv, v2_n_vert, phi_n_sc, v2n_fn, s, r_hyd)
            return (chi2_radial(vc_m, r_line, rr, vv, ss) +
                    chi2_vertical(phi_m, rv, zv, rv, zv, phi_obs, sig_phi, sig_z))
        except Exception:
            return 1e10

    starts = [
        [zt0, zk0, math.log(s0), math.log(rh0)]
        for zt0 in (0.9, 1.0, 1.1)
        for zk0 in (0.9, 1.0, 1.1)
        for s0  in (0.43, 1.0, 2.5)
        for rh0 in (5.0, 15.0, 30.0)
    ]
    best = None
    for x0 in starts:
        try:
            res = minimize(score, x0, method="Nelder-Mead",
                           options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-4})
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass
    if best is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    zt    = float(np.clip(best.x[0], *DISC_BOUNDS))
    zk    = float(np.clip(best.x[1], *DISC_BOUNDS))
    s_opt = math.exp(float(best.x[2]))
    rh_opt = math.exp(float(best.x[3]))
    chi2_nu = best.fun / (N - 4)
    print(f"    best: s={s_opt:.4f}  r_hyd={rh_opt:.2f} kpc"
          f"  zt={zt:.3f} zk={zk:.3f}  chi2_nu={chi2_nu:.4f}", flush=True)
    return chi2_nu, zt, zk, s_opt, rh_opt


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rr, vv, ss = load_wang_radial()
    _rot, vert = load_observations()
    rv, zv, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)
    N = len(rr) + len(rv)
    print(f"s/r_h-HMG hybrid — Wang 78-point subset  (N={N}, DOF={N-4})")
    print(f"  {len(rr)} radial + {len(rv)} vertical points\n", flush=True)

    fT_rv, fK_rv = disc_fractions(rv)   # rv-based; same for all configs

    print("Building cylindrical grid + Imig precompute ...", flush=True)
    t0 = time.time()
    cyl_grid = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=CYL_NR, nz=CYL_NZ)
    precomp  = imig_precompute(cyl_grid)
    print(f"  Done in {time.time()-t0:.1f}s\n", flush=True)

    fields = ["config", "label", "chi2_nu", "zt", "zk", "s_opt", "r_hyd_opt"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    results = {}
    for config in CONFIGS:
        label = CONFIG_LABEL[config]
        print(f"-- {label} ({config}) --", flush=True)
        basis = build_basis(config, cyl_grid, precomp, rv, zv)
        fT_r, fK_r = disc_fractions(basis["r_line"])   # must be on the baryonic r_line grid
        chi2_nu, zt, zk, s_opt, rh_opt = fit_hmg_hybrid(
            basis, rv, zv, rr, vv, ss, phi_obs, sig_phi, sig_z,
            fT_r, fK_r, fT_rv, fK_rv)
        results[config] = chi2_nu
        with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow({
                "config": config, "label": label,
                "chi2_nu": f"{chi2_nu:.4f}",
                "zt": f"{zt:.4f}", "zk": f"{zk:.4f}",
                "s_opt": f"{s_opt:.4f}", "r_hyd_opt": f"{rh_opt:.4f}",
            })
        print(flush=True)

    print("=" * 55)
    print(f"  {'Config':<6}  {'chi2_nu':>8}  {'expected':>8}  {'diff':>7}")
    print("  " + "-" * 36)
    ok = True
    for cfg in CONFIGS:
        lbl = CONFIG_LABEL[cfg]
        got = results[cfg]
        exp = EXPECTED[cfg]
        diff = got - exp
        flag = "" if abs(diff) < 0.02 else "  <-- MISMATCH"
        if flag:
            ok = False
        print(f"  {lbl:<6}  {got:8.4f}  {exp:8.4f}  {diff:+7.4f}{flag}")
    print("=" * 55)
    print(f"{'PASS' if ok else 'FAIL — check optimizer or data paths'}")
    print(f"\nTotal time: {time.time()-t_total:.0f}s")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
