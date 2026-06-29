"""step2_stvg_parallel.py — Parallel STVG fitting with configurable grid and workers.

Standalone replacement for the STVG section of step2_fit_all_models.py.
Supports parallel draws (multiprocessing) and configurable grid resolution,
so it can run efficiently on Granizo (many cores, full grid) or on a normal
laptop (fewer cores, reduced grid, but correct algorithm).

Usage
-----
  # Granizo — full accuracy, auto-detect cores:
  python scripts/step2_stvg_parallel.py --outdir outputs/nbar4

  # Laptop — balanced speed/accuracy (~3 min/draw):
  python scripts/step2_stvg_parallel.py --outdir outputs/nbar4 \\
      --nr 60 --nz 40 --nphi 40 --n-workers 4

  # Quick test (5 draws, coarse grid):
  python scripts/step2_stvg_parallel.py --outdir outputs/nbar4 \\
      --nr 15 --nz 10 --nphi 10 --n-workers 4 --n-draws 5

  # Run all 4 nbars on Granizo (in 4 separate terminals):
  for nbar in nbar1 nbar2 nbar3 nbar4; do
    python scripts/step2_stvg_parallel.py --outdir outputs/$nbar &
  done

Two-level grid strategy (default)
----------------------------------
  The optimizer (Nelder-Mead) runs on a COARSE grid (--nr-opt × --nz-opt × --nphi-opt,
  default 30×20×20 = 12 k cells).  All ~200 optimizer obs fit in one chunk → no mmap
  allocation per score() call → very fast convergence (~4 s per draw).

  The final rotation-curve prediction uses the FINE grid (--nr × --nz × --nphi,
  default 190×90×80 = 1.37 M cells) with pre-allocated workspace buffers (--ws-fine)
  to eliminate mmap overhead.  Only 1 fine-grid call per draw → ~2 s per draw.

  Total wall-clock: ~6–10 s/draw on Granizo with 6 workers → ~2 min for 100 draws.

Fine-grid resolution guide (--nr/--nz/--nphi)
---------------------------------------------
  nr=190, nz=90,  nphi=80  [default] : 1.37 M cells — accurate prediction
  nr=90,  nz=50,  nphi=48            :  648 k cells — good
  nr=60,  nz=40,  nphi=40            :  192 k cells — moderate (laptop)

Estimated RAM per worker (5 buffers × obs_chunk × n_cells)
-----------------------------------------------------------
  Default (opt 12k, fine 1.37M, obs_chunk=2) : ~200 MB/worker — safe for any system
  n_workers × peak_RAM must fit within available system RAM.

Resume after interruption
-------------------------
  Completed draws are checkpointed in --outdir/stvg_checkpoints/.
  Rerun the same command to skip already-finished draws automatically.
  Use --no-resume to force recomputation of all draws.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.observations import load_observations, radial_fit_arrays, vertical_arrays
from vgrav.chi2 import chi2_radial, chi2_vertical, N_PRIMARY
from vgrav.models import (
    predict_stvg,
    stvg_disk_accel_and_phi,
    stvg_bulge_yukawa_resolved,
    make_stvg_workspace,
)
from vgrav.baryonic import (
    build_mc100_draws,
    imig_precompute,
    imig_calibrate_weights,
    disc_rho_from_weights,
    build_component_grid_from_rho_cyl,
)
from vgrav.solver import make_grid

# ── Worker global state ────────────────────────────────────────────────────────
# On Linux (fork): populated in the parent before Pool creation → inherited
#                  by all workers for free (no pickling of large arrays).
# On Windows (spawn): populated by _worker_init which runs once per worker.
_G: dict = {}


def _worker_init(state: dict) -> None:
    """Populate worker state. No-op on fork (already inherited); runs on spawn."""
    if not _G:
        _G.update(state)


# ── STVG optimizer (runs inside each worker) ──────────────────────────────────

def _fit_stvg_worker(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    comp_grid_opt,
    comp_grid_fine,
    rr: np.ndarray,
    vv: np.ndarray,
    ss: np.ndarray,
    phi_obs: np.ndarray,
    sig_phi: np.ndarray,
    sig_z: np.ndarray,
    maxiter: int,
    mu0: float = 0.0678,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Fit STVG (alpha, mu) for one draw by minimising chi2_total (k=2).

    Two-level grid strategy:
      • Optimizer score() uses comp_grid_opt (coarse, non-zero cells only) with
        a per-draw workspace → no mmap per score() call.
      • Final curve prediction uses comp_grid_fine (full res) with ws_fine.

    Workspaces are built here (once per draw) using actual non-zero cell counts,
    because build_component_grid_from_rho_cyl filters zero-mass cells so the
    real cell count differs from nr*nz*nphi.
    """
    n_obs_score = len(rr) + len(rv_obs)
    ws_opt  = make_stvg_workspace(n_obs_score, comp_grid_opt.mass.size)
    obs_chunk_fine = max(1, 300_000_000 // (comp_grid_fine.mass.size * 8))
    ws_fine = make_stvg_workspace(obs_chunk_fine, comp_grid_fine.mass.size)

    vc_n_at_rr = np.interp(rr, r_grid, vc_n)
    g_n_at_rr = vc_n_at_rr ** 2 / np.maximum(rr, 1e-8)

    def score(t: tuple) -> float:
        log_alpha, log_mu = t
        alpha = math.exp(log_alpha)
        mu = math.exp(log_mu)
        try:
            a_y, ph_y = stvg_disk_accel_and_phi(
                rr, rv_obs, zv_obs, comp_grid_opt, alpha, mu, workspace=ws_opt
            )
            a_yb, ph_yb = stvg_bulge_yukawa_resolved(rr, rv_obs, zv_obs, alpha, mu)
            vc2 = rr * ((1.0 + alpha) * g_n_at_rr - a_y - a_yb)
            vc_m = np.sqrt(np.maximum(vc2, 0.0))
            chi_r = float(np.sum(((vc_m - vv) / ss) ** 2))
            phi_m = (1.0 + alpha) * phi_n + ph_y + ph_yb
            chi_z = chi2_vertical(
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
            r = minimize(
                score, x0, method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3},
            )
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass

    alpha_opt = math.exp(best.x[0]) if best is not None else 10.68
    mu_opt = math.exp(best.x[1]) if best is not None else mu0
    vc_out, phi_out = predict_stvg(
        r_grid, vc_n, phi_n, rv_obs, zv_obs, comp_grid_fine, alpha_opt, mu_opt,
        workspace=ws_fine,
    )
    return vc_out, phi_out, (alpha_opt, mu_opt)


def _process_draw(
    task: tuple,
) -> tuple:
    """Worker entry point: build ComponentGrids + fit STVG for one draw.

    Returns (draw_i, 'ok', vc_m, phi_m, theta, chi2_r, chi2_z)
         or (draw_i, error_str, None, None, None, None, None).
    """
    draw_i, weights_i, vc_n, phi_n = task
    g = _G
    try:
        rho_disc = disc_rho_from_weights(weights_i, g["precomp_c"])
        # Coarse grid for optimizer (fast score calls)
        comp_grid_opt = build_component_grid_from_rho_cyl(
            rho_disc, g["cyl_grid_c"], g["nr_opt"], g["nz_opt"], g["nphi_opt"]
        )
        # Fine grid for final prediction (accurate curves)
        comp_grid_fine = build_component_grid_from_rho_cyl(
            rho_disc, g["cyl_grid_c"], g["nr"], g["nz"], g["nphi"]
        )
        vc_m, phi_m, theta = _fit_stvg_worker(
            g["r_grid"], vc_n, phi_n,
            g["rv_obs"], g["zv_obs"],
            comp_grid_opt, comp_grid_fine,
            g["rr"], g["vv"], g["ss"],
            g["phi_obs"], g["sig_phi"], g["sig_z"],
            g["maxiter"],
        )
        cr = chi2_radial(vc_m, g["r_grid"], g["rr"], g["vv"], g["ss"])
        cz = chi2_vertical(
            phi_m, g["rv_obs"], g["zv_obs"],
            g["rv_obs"], g["zv_obs"], g["phi_obs"], g["sig_phi"], g["sig_z"],
        )
        return (draw_i, "ok", vc_m, phi_m, theta, cr, cz)
    except Exception as exc:
        return (draw_i, str(exc), None, None, None, None, None)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _read_baryonic_csvs(
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    rad_path = outdir / "mc100_baryonic_radial.csv"
    vert_path = outdir / "mc100_baryonic_vertical.csv"
    for p in (rad_path, vert_path):
        if not p.exists():
            print(f"  ERROR: {p} not found. Run step1 --full first.", flush=True)
            sys.exit(1)

    with open(rad_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    b_idx = [i for i, h in enumerate(hdr) if h.startswith("b")]
    r_idx = hdr.index("R_kpc")
    data = rows[1:-1]
    r_grid = np.array([float(r[r_idx]) for r in data])
    rad_curves = [np.array([float(r[i]) for r in data]) for i in b_idx]

    with open(vert_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    b_idx2 = [i for i, h in enumerate(hdr) if h.startswith("b")]
    R_col = hdr.index("R_kpc")
    z_col = hdr.index("z_kpc")
    data = rows[1:-1]
    rv = np.array([float(r[R_col]) for r in data])
    zv = np.array([float(r[z_col]) for r in data])
    rz_grid = np.column_stack([rv, zv])
    vert_curves = [np.array([float(r[i]) for r in data]) for i in b_idx2]

    return r_grid, rz_grid, rv, zv, rad_curves, vert_curves


def _write_results(
    outdir: Path,
    r_grid: np.ndarray,
    rz_grid: np.ndarray,
    results: dict,
    n_draws: int,
) -> None:
    rad_out = [results[i]["vc_m"] for i in range(n_draws)]
    vert_out = [results[i]["phi_m"] for i in range(n_draws)]
    chi2_r = [results[i]["chi2_r"] for i in range(n_draws)]
    chi2_z = [results[i]["chi2_z"] for i in range(n_draws)]
    params = [results[i]["theta"] for i in range(n_draws)]
    labels = [f"b{i+1}" for i in range(n_draws)]

    rad_path = outdir / "model_stvg_radial.csv"
    with open(rad_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc"] + labels)
        for j, R in enumerate(r_grid):
            w.writerow([f"{R:.6g}"] + [f"{rad_out[i][j]:.6g}" for i in range(n_draws)])
        w.writerow(["chi2_radial"] + [f"{chi2_r[i]:.6g}" for i in range(n_draws)])

    vert_path = outdir / "model_stvg_vertical.csv"
    with open(vert_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["R_kpc", "z_kpc"] + labels)
        for j, (R, z) in enumerate(rz_grid):
            w.writerow([f"{R:.6g}", f"{z:.6g}"] + [f"{vert_out[i][j]:.6g}" for i in range(n_draws)])
        w.writerow(["chi2_vertical", ""] + [f"{chi2_z[i]:.6g}" for i in range(n_draws)])

    params_path = outdir / "model_stvg_params.csv"
    with open(params_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["draw_id", "draw_label", "alpha", "mu"])
        for i, theta in enumerate(params):
            label = f"b{i+1}"
            if theta is None:
                w.writerow([i + 1, label, "", ""])
            else:
                w.writerow([i + 1, label, f"{theta[0]:.8g}", f"{theta[1]:.8g}"])

    k = 2
    chi2_med = np.median([chi2_r[i] + chi2_z[i] for i in range(n_draws)])
    chi2_nu_p50 = chi2_med / max(N_PRIMARY - k, 1)
    print(f"\n  Written: model_stvg_radial.csv + _vertical.csv + _params.csv")
    print(f"  >>> stvg done: chi2_nu p50 = {chi2_nu_p50:.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outdir", default="outputs/nbar4", metavar="DIR",
        help="Directory with mc100_baryonic_*.csv (step1 output). Default: outputs/nbar4",
    )
    parser.add_argument(
        "--baryon-band", default=None, metavar="PATH",
        help="Path to baryon_band.csv for this nbar. "
             "Default: --outdir/baryon_band.csv (falls back to data/baryon_band.csv).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None, metavar="N",
        help="Parallel worker processes. Default: min(cpu_count-1, n_draws). "
             "Set to 1 for sequential execution on a laptop.",
    )
    parser.add_argument(
        "--n-draws", type=int, default=None, metavar="N",
        help="Process only the first N draws (default: all 100). Use for testing.",
    )
    parser.add_argument(
        "--nr", type=int, default=190, metavar="INT",
        help="Fine-grid radial cells for final prediction (default: 190).",
    )
    parser.add_argument(
        "--nz", type=int, default=90, metavar="INT",
        help="Fine-grid vertical half-cells for final prediction (default: 90).",
    )
    parser.add_argument(
        "--nphi", type=int, default=80, metavar="INT",
        help="Fine-grid azimuthal cells for final prediction (default: 80).",
    )
    parser.add_argument(
        "--nr-opt", type=int, default=30, metavar="INT",
        help="Coarse-grid radial cells for Nelder-Mead optimizer (default: 30). "
             "Smaller = faster score() calls; increase if optimizer diverges.",
    )
    parser.add_argument(
        "--nz-opt", type=int, default=20, metavar="INT",
        help="Coarse-grid vertical half-cells for optimizer (default: 20).",
    )
    parser.add_argument(
        "--nphi-opt", type=int, default=20, metavar="INT",
        help="Coarse-grid azimuthal cells for optimizer (default: 20).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=500, metavar="N",
        help="Nelder-Mead max iterations per optimizer start (default: 500).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing checkpoints and recompute all draws from scratch.",
    )
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.baryon_band:
        band_path = Path(args.baryon_band)
    elif (outdir / "baryon_band.csv").exists():
        band_path = outdir / "baryon_band.csv"
    else:
        band_path = ROOT / "data" / "baryon_band.csv"

    ckpt_dir = outdir / "stvg_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Resource estimation ────────────────────────────────────────────────────
    n_cells_fine = args.nr * args.nz * args.nphi
    n_cells_opt  = args.nr_opt * args.nz_opt * args.nphi_opt
    # Each worker holds 2 ComponentGrids + 2 workspaces (5 buffers each).
    # Workspaces are allocated per-draw using actual non-zero cell counts.
    # These estimates use the upper-bound cell counts for RAM planning only.
    n_obs_total = 200  # conservative upper bound for rr + rv_obs
    ws_opt_mb  = 5 * n_obs_total * n_cells_opt  * 8 / 1e6
    ws_fine_mb = 5 * 2           * n_cells_fine * 8 / 1e6
    mem_mb_est = max(50, ws_opt_mb + ws_fine_mb + n_cells_fine * 4 * 8 / 1e6 + 50)

    cpu_avail = max(1, (mp.cpu_count() or 2) - 1)
    n_workers_req = args.n_workers if args.n_workers is not None else cpu_avail
    n_workers = max(1, n_workers_req)

    print(f"\nstep2_stvg_parallel.py")
    print(f"  outdir      : {outdir}")
    print(f"  baryon-band : {band_path}")
    print(f"  fine grid   : nr={args.nr}, nz={args.nz}, nphi={args.nphi}"
          f"  ({n_cells_fine:,} cells)  — prediction")
    print(f"  opt  grid   : nr={args.nr_opt}, nz={args.nz_opt}, nphi={args.nphi_opt}"
          f"  ({n_cells_opt:,} cells) — optimizer")
    print(f"  n_workers   : {n_workers}  (CPUs available: {cpu_avail})")
    print(f"  RAM/worker  : ~{mem_mb_est:.0f} MB  "
          f"(peak ≈ {n_workers * mem_mb_est / 1024:.1f} GB total)")
    print(f"  maxiter     : {args.maxiter}")
    print(f"  resume      : {'disabled (--no-resume)' if args.no_resume else 'enabled'}")
    print()

    # ── Observational data ─────────────────────────────────────────────────────
    _rot, vert = load_observations()
    rr, vv, ss = radial_fit_arrays(
        chi2_catalog_path=ROOT / "data" / "fig2_observational_catalog.csv"
    )
    rv_obs, zv_obs, phi_obs, sig_phi, sig_z = vertical_arrays(vert=vert)

    # ── Baryonic draws ─────────────────────────────────────────────────────────
    r_grid, rz_grid, rv, zv, rad_bary, vert_bary = _read_baryonic_csvs(outdir)
    n_draws_avail = len(rad_bary)
    n_draws = min(n_draws_avail, args.n_draws or n_draws_avail)
    print(f"  Baryonic draws: {n_draws_avail} in CSV, processing {n_draws}.", flush=True)

    # ── Imig+2025 precompute ───────────────────────────────────────────────────
    print("  Imig+2025 precompute (13 Poisson solves) ...", flush=True)
    t0 = time.perf_counter()
    cyl_grid_c = make_grid(r_min=0.0, r_max=70.0, z_max=20.0, nR=281, nz=641)
    precomp_c = imig_precompute(cyl_grid_c)
    print(f"  Precompute done in {time.perf_counter() - t0:.0f}s.", flush=True)

    # ── Calibrate per-draw Imig weights ───────────────────────────────────────
    print("  Calibrating Imig weights per draw ...", flush=True)
    r_line, band_draws = build_mc100_draws(band_path=band_path)
    weights_per_draw = [
        imig_calibrate_weights(precomp_c, vc_tgt, r_line)
        for _, vc_tgt, _ in band_draws[:n_draws]
    ]
    print(f"  Weights done ({n_draws} draws).", flush=True)

    # ── Checkpoint scan ────────────────────────────────────────────────────────
    results: dict = {}
    pending: list = []

    for i in range(n_draws):
        ckpt = ckpt_dir / f"draw_{i:04d}.npz"
        if not args.no_resume and ckpt.exists():
            try:
                d = np.load(ckpt, allow_pickle=True)
                results[i] = {
                    "vc_m": d["vc_m"],
                    "phi_m": d["phi_m"],
                    "theta": (float(d["alpha"]), float(d["mu"])),
                    "chi2_r": float(d["chi2_r"]),
                    "chi2_z": float(d["chi2_z"]),
                }
                continue
            except Exception:
                pass  # checkpoint corrupt → recompute
        pending.append((i, weights_per_draw[i], rad_bary[i], vert_bary[i]))

    n_skip = n_draws - len(pending)
    if n_skip > 0:
        print(f"  Resuming: {n_skip}/{n_draws} draws already done (skipped).")
    print(f"  Pending : {len(pending)} draws.", flush=True)

    if not pending:
        print("  All draws already done. Writing final CSVs.")
        _write_results(outdir, r_grid, rz_grid, results, n_draws)
        return

    # ── Shared worker state ────────────────────────────────────────────────────
    # Workspaces are created per-draw in _fit_stvg_worker using actual non-zero
    # cell counts (build_component_grid filters zero-mass cells, so mass.size !=
    # nr_opt*nz_opt*nphi_opt and pre-allocated workspaces would never match).
    state = dict(
        precomp_c=precomp_c,
        cyl_grid_c=cyl_grid_c,
        r_grid=r_grid,
        rv_obs=rv_obs,
        zv_obs=zv_obs,
        rr=rr,
        vv=vv,
        ss=ss,
        phi_obs=phi_obs,
        sig_phi=sig_phi,
        sig_z=sig_z,
        nr=args.nr,
        nz=args.nz,
        nphi=args.nphi,
        nr_opt=args.nr_opt,
        nz_opt=args.nz_opt,
        nphi_opt=args.nphi_opt,
        maxiter=args.maxiter,
    )

    # Populate global BEFORE Pool creation so fork-children inherit it with
    # zero pickling overhead (Linux).  On Windows/spawn, _worker_init fills it.
    _G.update(state)

    # ── Multiprocessing pool ───────────────────────────────────────────────────
    # Use fork on Linux (shared memory, fast) and spawn on Windows (safe).
    start_method = "fork" if sys.platform != "win32" else "spawn"
    ctx = mp.get_context(start_method)

    t_start = time.perf_counter()
    n_completed = n_skip
    errors: list = []

    print(f"\n  Starting pool ({n_workers} workers, {start_method})...\n", flush=True)

    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(state,),
    ) as pool:
        for result in pool.imap_unordered(_process_draw, pending):
            draw_i, status, vc_m, phi_m, theta, chi2_r, chi2_z = result
            n_completed += 1
            elapsed = time.perf_counter() - t_start
            n_done_so_far = n_completed - n_skip
            rate = n_done_so_far / elapsed if elapsed > 0 else 0.0
            remaining = n_draws - n_completed
            eta_s = remaining / rate if rate > 0 else float("inf")
            if eta_s < 3600:
                eta_str = f"{eta_s / 60:.0f}min"
            elif eta_s < 86400:
                eta_str = f"{eta_s / 3600:.1f}h"
            else:
                eta_str = f"{eta_s / 86400:.1f}d"

            if status == "ok":
                results[draw_i] = {
                    "vc_m": vc_m,
                    "phi_m": phi_m,
                    "theta": theta,
                    "chi2_r": chi2_r,
                    "chi2_z": chi2_z,
                }
                k = 2
                chi2_nu = (chi2_r + chi2_z) / max(N_PRIMARY - k, 1)
                print(
                    f"  draw {draw_i+1:3d}/{n_draws}"
                    f"  chi2_nu={chi2_nu:.2f}"
                    f"  alpha={theta[0]:.2f}  mu={theta[1]:.4f}"
                    f"  [{n_completed}/{n_draws} done, ETA {eta_str}]",
                    flush=True,
                )
                # Save checkpoint so partial runs can be resumed
                ckpt = ckpt_dir / f"draw_{draw_i:04d}.npz"
                np.savez_compressed(
                    ckpt,
                    vc_m=vc_m,
                    phi_m=phi_m,
                    alpha=np.float64(theta[0]),
                    mu=np.float64(theta[1]),
                    chi2_r=np.float64(chi2_r),
                    chi2_z=np.float64(chi2_z),
                )
            else:
                errors.append((draw_i + 1, status))
                print(f"  draw {draw_i+1:3d} ERROR: {status}", flush=True)

    # ── Final output ───────────────────────────────────────────────────────────
    if errors:
        print(f"\n  WARNING: {len(errors)} draw(s) failed: "
              f"{[e[0] for e in errors]}. Rerun to retry.")

    if len(results) == n_draws:
        _write_results(outdir, r_grid, rz_grid, results, n_draws)
    else:
        print(f"\n  Only {len(results)}/{n_draws} draws succeeded."
              f" Rerun to complete missing draws (resume is automatic).")


if __name__ == "__main__":
    main()
