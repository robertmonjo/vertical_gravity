"""STVG fits for likelihood_bic.py, solved on the axisymmetric grid.

In the weak-field limit of STVG (Moffat & Rahvar 2013),
    Phi_STVG = Phi_N + alpha D,   D = Phi_N - Phi_Y,
    nabla^2 Phi_N = 4 pi G rho,   (nabla^2 - mu^2) Phi_Y = 4 pi G rho,
so that v_c^2 = v_N^2 + alpha R dD/dR at z = 0 (vgrav.models.stvg_grid_kernel).
Both potentials are solved on the grid of the Newtonian potential and for the
same density. The density scaled by the disc factors is
    rho(z_t, z_k) = [1 + f_T (z_t - 1) + f_K (z_k - 1)] rho_3d,
as for QUMOND in likelihood_bic.py, and D is linear in the density; for each mu
three kernels (rho_3d, f_T rho_3d, f_K rho_3d) are computed once, and the fit over
(z_t, z_k, alpha) at fixed mu costs only linear combinations. mu is profiled on
a logarithmic grid.

Usage
-----
  python scripts/likelihood_bic_stvg.py --workers 24

Rows are appended to outputs/likelihood_bic/likelihood_bic_fits.csv:
  stvg_std  alpha = 8.89, mu = 0.04 kpc^-1 (Moffat & Rahvar 2013); k = 2 (z_t, z_k)
  stvg      alpha and mu free; k = 4 (z_t, z_k, alpha, mu)
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
from vgrav.models import stvg_grid_kernel
from vgrav.solver import axisymmetric_operator, monopole_boundary

MU_UNIVERSAL, ALPHA_UNIVERSAL = 0.04, 8.89   # Moffat & Rahvar (2013)
MU_GRID = np.unique(np.append(np.geomspace(0.003, 3.0, 19), MU_UNIVERSAL))
ALPHA_BOUNDS = (math.log(1e-3), math.log(1e3))
COMPONENTS = ("all", "thin", "thick")


def component_densities(config):
    b, grid = L.STATE["bases"][config], L.STATE["cyl_grid"]
    fT, fK = L.disc_fractions(grid.R)
    return {"all": b["rho_3d"], "thin": fT[:, None] * b["rho_3d"], "thick": fK[:, None] * b["rho_3d"]}


def kernel(task):
    """Kernels (R dD/dR on r_line, D(R,z) - D(R,0) at the vertical points) for one config and mu."""
    config, mu = task
    grid = L.STATE["cyl_grid"]
    poisson, yukawa = axisymmetric_operator(grid), axisymmetric_operator(grid, mu=float(mu))
    rv, zv = L.STATE["vert"]["rv"], L.STATE["vert"]["zv"]
    out = {}
    for which, rho in component_densities(config).items():
        mass = float(np.sum(rho * 2.0 * math.pi * grid.R[:, None] * grid.dR * grid.dz))
        phi_n = poisson(4.0 * math.pi * L.G * rho, monopole_boundary(grid, mass))
        out[which] = stvg_grid_kernel(grid, rho, float(mu), L.STATE["bases"][config]["r_line"], rv, zv,
                                      phi_n_grid=phi_n, yukawa_solve=yukawa)
    return task, out


def combine(ker, zt, zk):
    k0, kT, kK = ker["all"], ker["thin"], ker["thick"]
    return (k0[0] + (zt - 1.0) * kT[0] + (zk - 1.0) * kK[0],
            k0[1] + (zt - 1.0) * kT[1] + (zk - 1.0) * kK[1])


def predict(config, ker, zt, zk, alpha):
    r_line, v2_rad, _, phi_sc = L.scaled_inputs(config, zt, zk)
    dv2, dphi = combine(ker, zt, zk)
    return np.sqrt(np.maximum(v2_rad + alpha * dv2, 0.0)), phi_sc + alpha * dphi


def fit_universal(config, variant, kernels):
    """STVG with the universal parameters of Moffat & Rahvar (2013); only the discs are fitted."""
    zt0, zk0, c_sun = L.STATE["prior"][config]
    r_line = L.STATE["bases"][config]["r_line"]
    radial = L.STATE["radial"][variant]
    ker = kernels[(config, MU_UNIVERSAL)]

    def objective(p):
        vc, phi = predict(config, ker, p[0], p[1], ALPHA_UNIVERSAL)
        m2, _ = L.score(vc, r_line, phi, radial, L.STATE["vert"])
        value = m2 + L.prior_term(p[0], p[1], c_sun)
        return value if np.isfinite(value) else 1e12

    best = None
    for dz in ((0, 0), (2, 2), (-2, -2), (4, 0), (0, 4)):
        x0 = list(L.disc_point(zt0, zk0, *dz))
        res = minimize(objective, x0, method="L-BFGS-B", bounds=list(L.disc_bounds(zt0, zk0)),
                       options={"maxiter": 1500, "ftol": 1e-12})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    vc, phi = predict(config, ker, best.x[0], best.x[1], ALPHA_UNIVERSAL)
    _, chi2 = L.score(vc, r_line, phi, radial, L.STATE["vert"])
    return L.row(variant, config, "stvg_std", 2, best.fun, chi2, best.x[0], best.x[1],
                 [math.log(ALPHA_UNIVERSAL), math.log(MU_UNIVERSAL)], c_sun)


def fit(config, variant, kernels):
    zt0, zk0, c_sun = L.STATE["prior"][config]
    r_line = L.STATE["bases"][config]["r_line"]
    radial = L.STATE["radial"][variant]
    bounds = list(L.disc_bounds(zt0, zk0)) + [ALPHA_BOUNDS]
    best = None
    for mu in MU_GRID:
        ker = kernels[(config, mu)]

        def objective(p):
            vc, phi = predict(config, ker, p[0], p[1], math.exp(p[2]))
            m2, _ = L.score(vc, r_line, phi, radial, L.STATE["vert"])
            value = m2 + L.prior_term(p[0], p[1], c_sun)
            return value if np.isfinite(value) else 1e12

        for la in (math.log(0.1), math.log(1.0), math.log(10.0)):
            res = minimize(objective, [zt0, zk0, la], method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1500, "ftol": 1e-12})
            if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
                vc, phi = predict(config, ker, res.x[0], res.x[1], math.exp(res.x[2]))
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
    tasks = [(c, mu) for c in configs for mu in MU_GRID]
    kernels = {}
    with get_context("fork").Pool(args.workers) as pool:
        for i, (task, ker) in enumerate(pool.imap_unordered(kernel, tasks), 1):
            kernels[task] = ker
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{time.time() - t0:7.1f}s] kernels {i}/{len(tasks)}", flush=True)
    rows = [fit(c, v, kernels) for c in configs for v in L.RADIAL_VARIANTS]
    rows += [fit_universal(c, v, kernels) for c in configs for v in L.RADIAL_VARIANTS]
    L.write_rows(rows)
    print(f"STVG done in {(time.time() - t0) / 60:.1f} min", flush=True)
    L.summary()


if __name__ == "__main__":
    main()
