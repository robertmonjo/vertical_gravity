"""MC100 band-consistent fits under the full likelihood for STVG with universal
parameters, refracted gravity with the DiskMass parameters, and the apparent dark
matter of Verlinde (2016).

  stvg_std            STVG with alpha = 8.89, mu = 0.04 kpc^-1 (Moffat & Rahvar 2013); k = 0.
                      Phi_STVG = Phi_N + alpha (Phi_N - Phi_Y), with (nabla^2 - mu^2) Phi_Y = 4 pi G rho
                      solved on the grid of Phi_N for the density of the draw
                      (vgrav.models.stvg_grid_kernel), added to the Newtonian baseline of the draw
  reg_dms             refracted gravity, nabla.[eps(rho) nabla phi] = 4 pi G rho, with the
                      parameters of the 30 DiskMass galaxies (Cesare et al. 2020), solved on
                      nested grids (vgrav.models.predict_refracted_gravity); k = 0
  veg_verlinde        M_D^2(r) = a0 r^2/(6G) d(M_B r)/dr with a0 = c H0 (Verlinde 2016, Eq. 7.40),
                      added as a spherical halo; k = 0
  veg_verlinde_free   the same with a0 free; k = 1

For each ensemble and draw the predictions are computed once (they do not depend on
the radial data vector) and then scored for every radial variant. Output files follow
step2_bc.py: model_bc_<key>_{radial,vertical}.csv in
outputs/likelihood_bic_mc100/<variant>/bc_nbar<n>/, whose last rows hold chi2_radial
and -2 ln L_vert.

Usage
-----
  python scripts/likelihood_bic_mc100_extra.py --workers 28
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vgrav.chi2 as chi2_module
from vgrav._constants import A_EG_FIXED, G
from vgrav.baryonic import (build_mc100_draws, imig_calibrate_weights, imig_precompute,
                            reconstruct_rho_from_weights)
from vgrav.models import predict_refracted_gravity, spherical_vc_and_phi, stvg_grid_kernel
from vgrav.solver import axisymmetric_operator, monopole_boundary
from vgrav.observations import RADIAL_VARIANTS, load_observations, radial_variant_arrays, vertical_arrays
from vgrav.solver import make_grid

_spec = importlib.util.spec_from_file_location("step2_bc", ROOT / "scripts" / "step2_bc.py")
step2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step2)

CATALOG = ROOT / "data" / "fig2_observational_catalog.csv"
OUT = ROOT / "outputs" / "likelihood_bic_mc100"
ALPHA_UNIVERSAL, MU_UNIVERSAL = 8.89, 0.04
A0_VERLINDE = 6.0 * A_EG_FIXED
SOLVED_KEYS = ("stvg_std", "reg_dms")
R_EDGES = np.geomspace(0.05, 400.0, 800)
REG_DMS = (0.661, 1.79, -24.54)   # Cesare et al. (2020), joint fit of the DiskMass galaxies
STATE: dict = {}


def load_ensemble(n, n_draws=None):
    base = ROOT / "outputs" / f"nbar{n}"
    r_grid, rz_grid, rv, zv, rad_bary, vert_bary = step2._read_baryonic_draws(
        base / "mc100_baryonic_radial.csv", base / "mc100_baryonic_vertical.csv")
    r_line, band_draws = build_mc100_draws(band_path=base / f"baryon_band_nbar{n}.csv")
    band_draws = band_draws[:n_draws] if n_draws else band_draws
    weights = [imig_calibrate_weights(STATE["precomp"], vc_tgt, r_line) for _, vc_tgt, _ in band_draws]
    v_n_rad = [np.interp(r_grid, r_line, vc_tgt) for _, vc_tgt, _ in band_draws]
    return dict(r_grid=r_grid, rz_grid=rz_grid, rv=rv, zv=zv, phi_n=vert_bary, v_n_rad=v_n_rad, weights=weights)


def enclosed_mass(rho):
    grid = STATE["grid"]
    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    w = (rho * 2.0 * math.pi * RR * grid.dR * grid.dz).ravel()
    r = np.sqrt(RR ** 2 + ZZ ** 2).ravel()
    order = np.argsort(r)
    return np.interp(R_EDGES, r[order], np.cumsum(w[order]))


def verlinde_halo(m_enc, a0, r_grid, rv, zv):
    d_mr = np.gradient(m_enc * R_EDGES, R_EDGES)
    m_d = np.sqrt(np.maximum(a0 * R_EDGES ** 2 / (6.0 * G) * d_mr, 0.0))
    return spherical_vc_and_phi(r_grid, rv, zv, lambda r: np.interp(r, R_EDGES, m_d))


def predict_draw(task):
    n, i = task
    ens = STATE["ensembles"][n]
    precomp, grid = STATE["precomp"], STATE["grid"]
    w = ens["weights"][i]
    rho_3d = reconstruct_rho_from_weights(w, precomp.stellar, precomp.fixed, grid)
    if "yukawa" not in STATE:   # factorised once per worker
        STATE["poisson"] = axisymmetric_operator(grid)
        STATE["yukawa"] = axisymmetric_operator(grid, mu=MU_UNIVERSAL)
    mass = float(np.sum(rho_3d * 2.0 * math.pi * grid.R[:, None] * grid.dR * grid.dz))
    phi_n = STATE["poisson"](4.0 * math.pi * G * rho_3d, monopole_boundary(grid, mass))
    dv2, dphi = stvg_grid_kernel(grid, rho_3d, MU_UNIVERSAL, ens["r_grid"], ens["rv"], ens["zv"],
                                 phi_n_grid=phi_n, yukawa_solve=STATE["yukawa"])
    preds = {"stvg_std": (np.sqrt(np.maximum(ens["v_n_rad"][i] ** 2 + ALPHA_UNIVERSAL * dv2, 0.0)),
                          ens["phi_n"][i] + ALPHA_UNIVERSAL * dphi)}
    preds["reg_dms"] = predict_refracted_gravity(grid, rho_3d, ens["r_grid"], ens["rv"], ens["zv"], *REG_DMS)
    return n, i, preds, enclosed_mass(rho_3d)


def scores(vc, phi, ens, radial, vert):
    cr = chi2_module.chi2_radial(vc, ens["r_grid"], *radial)
    cz = chi2_module.chi2_vertical(phi, ens["rv"], ens["zv"], vert[0], vert[1], vert[2], vert[3], vert[4])
    return cr, cz


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-draws", type=int, default=None, help="limit draws per ensemble (testing)")
    parser.add_argument("--ensembles", default="1,2,3,4")
    parser.add_argument("--outdir", default=str(OUT))
    args = parser.parse_args()
    ensembles = [int(x) for x in args.ensembles.split(",")]
    out_root = Path(args.outdir)
    chi2_module.LOG_SIGMA_TERM = True
    t0 = time.time()
    STATE["grid"] = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=281, nz=641)
    STATE["precomp"] = imig_precompute(STATE["grid"])
    STATE["ensembles"] = {n: load_ensemble(n, args.n_draws) for n in ensembles}
    print(f"State ready in {time.time() - t0:.0f}s", flush=True)

    preds = {}
    tasks = [(n, i) for n in ensembles for i in range(len(STATE["ensembles"][n]["weights"]))]
    with get_context("fork").Pool(args.workers) as pool:
        for k, (n, i, solved, m_enc) in enumerate(pool.imap_unordered(predict_draw, tasks), 1):
            preds[(n, i)] = (solved, m_enc)
            if k % 40 == 0 or k == len(tasks):
                print(f"  [{time.time() - t0:7.0f}s] draws {k}/{len(tasks)}", flush=True)

    _rot, vert_rows = load_observations()
    vert = vertical_arrays(vert=vert_rows, phi_sys_fraction=0.05)
    for variant in RADIAL_VARIANTS:
        radial = radial_variant_arrays(variant, CATALOG)
        for n in ensembles:
            ens = STATE["ensembles"][n]
            out_dir = out_root / variant / f"bc_nbar{n}"
            out_dir.mkdir(parents=True, exist_ok=True)
            results = {key: ([], [], [], []) for key in SOLVED_KEYS + ("veg_verlinde", "veg_verlinde_free")}
            for i in range(len(ens["weights"])):
                solved, m_enc = preds[(n, i)]
                vn, phin = ens["v_n_rad"][i], ens["phi_n"][i]

                def verlinde(a0):
                    vc_h, phi_h = verlinde_halo(m_enc, a0, ens["r_grid"], ens["rv"], ens["zv"])
                    return np.sqrt(vn ** 2 + vc_h ** 2), phin + phi_h

                res = minimize_scalar(lambda la: sum(scores(*verlinde(math.exp(la)), ens, radial, vert)),
                                      bounds=(math.log(A0_VERLINDE / 30), math.log(A0_VERLINDE * 30)),
                                      method="bounded", options={"xatol": 1e-4})
                for key, (vc, phi) in [*solved.items(),
                                       ("veg_verlinde", verlinde(A0_VERLINDE)),
                                       ("veg_verlinde_free", verlinde(math.exp(res.x)))]:
                    cr, cz = scores(vc, phi, ens, radial, vert)
                    for lst, val in zip(results[key], (vc, phi, cr, cz)):
                        lst.append(val)
            for key, (vcs, phis, crs, czs) in results.items():
                step2._write_model_csvs(out_dir, key, ens["r_grid"], ens["rz_grid"], vcs, phis, crs, czs)
        print(f"  [{time.time() - t0:7.0f}s] variant {variant} written", flush=True)


if __name__ == "__main__":
    main()
