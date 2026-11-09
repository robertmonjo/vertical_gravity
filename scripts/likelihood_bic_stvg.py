"""STVG fits for likelihood_bic.py, using the linearity of the Yukawa term.

The STVG prediction is
    v_c^2 = R [(1 + alpha) g_N - alpha (a_Y,disc + a_Y,bulge)]
    Phi   = (1 + alpha) Phi_N + alpha (phi_Y,disc + phi_Y,bulge),
where the Yukawa terms (per unit alpha) depend only on mu and are linear in
the density. The disc density scaled by the thin- and thick-disc factors is
z_t rho_T + z_k rho_K + rho_rest, so for each mu three disc kernels (thin,
thick, rest) and one bulge kernel are computed once. The fit over
(z_t, z_k, alpha) at fixed mu then costs only linear combinations, and mu is
profiled on a logarithmic grid.

Usage
-----
  python scripts/likelihood_bic_stvg.py --workers 24

Rows are appended to outputs/likelihood_bic/likelihood_bic_fits.csv
(model "stvg", k = 4: z_t, z_k, alpha, mu).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import likelihood_bic as L
from vgrav.models import stvg_disk_accel_and_phi, stvg_bulge_yukawa_resolved

MU_GRID = np.geomspace(0.003, 3.0, 19)
ALPHA_BOUNDS = (math.log(1e-3), math.log(1e3))
COMPONENTS = ("thin", "thick", "rest")


def disc_component(config, which):
    b, grid = L.STATE["bases"][config], L.STATE["cyl_grid"]
    fT, fK = L.disc_fractions(grid.R)
    weight = {"thin": fT, "thick": fK, "rest": 1.0 - fT - fK}[which]
    return weight[:, None] * b["rho_disc"]


def kernel(task):
    config, which, mu = task
    comp = L.build_component_grid_from_rho_cyl(disc_component(config, which), L.STATE["cyl_grid"],
                                               nr=50, nz=25, nphi=36)
    a, p = stvg_disk_accel_and_phi(L.STATE["bases"][config]["r_line"], L.STATE["vert"]["rv"],
                                   L.STATE["vert"]["zv"], comp, 1.0, float(mu))
    return task, a, p


def fit(config, variant, kernels):
    zt0, zk0, c_sun = L.STATE["prior"][config]
    r_line = L.STATE["bases"][config]["r_line"]
    radial = L.STATE["radial"][variant]
    bounds = list(L.disc_bounds(zt0, zk0)) + [ALPHA_BOUNDS]
    best = None
    for mu in MU_GRID:
        kT, kK, kR, kB = (kernels[(config, w, mu)] for w in ("thin", "thick", "rest", "bulge"))

        def predict(p):
            zt, zk, alpha = p[0], p[1], math.exp(p[2])
            _, v2_rad, _, phi_sc = L.scaled_inputs(config, zt, zk)
            g_n = v2_rad / np.maximum(r_line, 1e-8)
            a_y = zt * kT[0] + zk * kK[0] + kR[0] + kB[0]
            p_y = zt * kT[1] + zk * kK[1] + kR[1] + kB[1]
            vc = np.sqrt(np.maximum(r_line * ((1.0 + alpha) * g_n - alpha * a_y), 0.0))
            return vc, (1.0 + alpha) * phi_sc + alpha * p_y

        def objective(p):
            vc, phi = predict(p)
            m2, _ = L.score(vc, r_line, phi, radial, L.STATE["vert"])
            value = m2 + L.prior_term(p[0], p[1], c_sun)
            return value if np.isfinite(value) else 1e12

        for la in (math.log(0.1), math.log(1.0), math.log(10.0)):
            res = minimize(objective, [zt0, zk0, la], method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1500, "ftol": 1e-12})
            if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
                vc, phi = predict(res.x)
                _, chi2 = L.score(vc, r_line, phi, radial, L.STATE["vert"])
                best = (res.fun, chi2, res.x[0], res.x[1], [res.x[2], math.log(mu)])
    return L.row(variant, config, "stvg", 4, best[0], best[1], best[2], best[3], best[4], c_sun)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=",".join(L.CONFIGS))
    args = parser.parse_args()
    configs = args.configs.split(",")
    L.build_state(configs)
    t0 = time.time()
    kernels = {}
    rv, zv = L.STATE["vert"]["rv"], L.STATE["vert"]["zv"]
    for config in configs:
        for mu in MU_GRID:
            kernels[(config, "bulge", mu)] = stvg_bulge_yukawa_resolved(
                L.STATE["bases"][config]["r_line"], rv, zv, 1.0, float(mu))
    tasks = [(c, w, mu) for c in configs for w in COMPONENTS for mu in MU_GRID]
    with get_context("fork").Pool(args.workers) as pool:
        for i, (task, a, p) in enumerate(pool.imap_unordered(kernel, tasks), 1):
            kernels[task] = (a, p)
            if i % 12 == 0 or i == len(tasks):
                print(f"  [{time.time() - t0:7.1f}s] kernels {i}/{len(tasks)}", flush=True)
    rows = [fit(c, v, kernels) for c in configs for v in L.RADIAL_VARIANTS]
    L.write_rows(rows)
    print(f"STVG done in {(time.time() - t0) / 60:.1f} min", flush=True)
    L.summary()


if __name__ == "__main__":
    main()
