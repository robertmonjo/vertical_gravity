"""Parametric baryonic density model and MC100 qcopula generation.

Baryonic component model
------------------------
Five components (McMillan 2011 / Bland-Hawthorn & Gerhard 2016 style):
  - Thin stellar disc   (rho_sun = 0.045 Msun/pc^3)
  - Thick stellar disc  (rho_sun = 0.00525 Msun/pc^3)
  - Molecular gas disc  (HI)
  - Atomic gas disc     (H2)
  - Bulge               (two-component Gaussian)

These parametric densities provide the 3D mass distribution used for
the cylindrical Poisson solver (QUMOND, STVG) and direct summation (STVG).

MC100 qcopula
-------------
Draws 100 baryonic target rotation curves from the weighted hybrid
baryonic band using a mass-constrained Gaussian-process copula:
  1. GP draw on 14 knot radii with log-R correlation length lambda=0.72
  2. Centre the shape (zero net amplitude shift) + small global amplitude
  3. Map knot quantiles to velocities from the band's 5 percentile levels
  4. Interpolate (Pchip in log-R) to the full radial grid
  5. Clip to [u_low=0.12, u_high=0.88] of the band

Reference: build_fig2_consolidated_mc100.py (this repository)
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

from vgrav._constants import G, R_SUN

_DATA_DIR = Path(__file__).parent.parent / "data"

# ── MC100 hyper-parameters ────────────────────────────────────────────────────
N_DRAWS = 100
RANDOM_SEED = 20260607
KNOT_R = np.array([1.0, 1.6, 2.5, 4.0, 6.5, 10.0, 16.0, 25.0, 40.0, 65.0, 100.0, 160.0, 250.0, 400.0])
LOGR_CORRELATION_LENGTH = 0.72
SIGMA_AMP = 0.20
U_KNOT_LOW = 0.12
U_KNOT_HIGH = 0.88


# ═══════════════════════════════════════════════════════════════════════════════
# Parametric density components
# ═══════════════════════════════════════════════════════════════════════════════

def _sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x) ** 2


def stellar_density(
    R: np.ndarray,
    z: np.ndarray,
    thin: bool,
    scale: float = 1.0,
) -> np.ndarray:
    """Exponential-sech stellar disc density [Msun/kpc^3].

    Parameters
    ----------
    thin  : True for thin disc, False for thick disc.
    scale : overall density normalisation factor.
    """
    if thin:
        rho_sun_pc3, rb, houter, hz_sun, aflare = 0.045, 7.46, 2.08, 0.39, 0.027
    else:
        rho_sun_pc3, rb, houter, hz_sun, aflare = 0.00525, 7.31, 1.47, 0.85, 0.057
    rho_sun = scale * rho_sun_pc3 * 1e9  # Msun/kpc^3
    hz = hz_sun * np.exp(aflare * (R - R_SUN))
    sigma_sun = 2.0 * hz_sun * rho_sun
    sigma_b = sigma_sun * math.exp((R_SUN - rb) / houter)
    sigma = np.where(R <= rb, sigma_b, sigma_b * np.exp(-(R - rb) / houter))
    return sigma / (2.0 * hz) * np.exp(-np.abs(z) / hz)


def gas_density(R: np.ndarray, z: np.ndarray, hi: bool) -> np.ndarray:
    """Gas disc density (HI or H2) [Msun/kpc^3].

    Parameters
    ----------
    hi : True for HI, False for H2 (molecular).
    """
    if hi:
        sigma0, rd, rm, zd = 53.1e6, 7.0, 4.0, 0.085
    else:
        sigma0, rd, rm, zd = 2180.0e6, 1.5, 12.0, 0.045
    safe_R = np.maximum(R, 1e-4)
    return sigma0 / (4.0 * zd) * np.exp(-rm / safe_R - safe_R / rd) * _sech2(z / (2.0 * zd))


def bulge_density(R: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Two-Gaussian spherical bulge density [Msun/kpc^3]."""
    r = np.sqrt(R * R + z * z)
    rho = np.zeros_like(r, dtype=float)
    for mass, sigma in ((6.5e9, 0.5), (1.48e10, 1.4)):
        rho += mass / ((2.0 * math.pi) ** 1.5 * sigma ** 3) * np.exp(-(r * r) / (2.0 * sigma * sigma))
    return rho


def baryon_density(R: np.ndarray, z: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Total baryonic density = thin + thick + HI + H2 + bulge [Msun/kpc^3]."""
    return (
        stellar_density(R, z, True, scale)
        + stellar_density(R, z, False, scale)
        + gas_density(R, z, True)
        + gas_density(R, z, False)
        + bulge_density(R, z)
    )


@dataclass
class ComponentGrid:
    """3D mass grid for direct potential/force summation."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mass: np.ndarray


def build_component_grid(
    nr: int = 190,
    nz: int = 90,
    nphi: int = 80,
    scale: float = 1.0,
) -> ComponentGrid:
    """Build the 3D disc mass grid (excludes bulge — added analytically).

    Parameters
    ----------
    nr, nz, nphi : grid resolution.
    scale        : baryonic density normalisation.

    Returns
    -------
    ComponentGrid with (x, y, z, mass) flattened arrays.
    """
    r_edges = np.linspace(0.02, 35.0, nr + 1)
    u_edges = np.linspace(0.0, 1.0, nz + 1)
    zpos_edges = 7.0 * u_edges ** 1.8
    phi_edges = np.linspace(0.0, 2.0 * math.pi, nphi + 1)

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    z_mid_pos = 0.5 * (zpos_edges[:-1] + zpos_edges[1:])
    dz_pos = np.diff(zpos_edges)
    z_mid = np.concatenate((-z_mid_pos[::-1], z_mid_pos))
    dz = np.concatenate((dz_pos[::-1], dz_pos))
    phi_mid = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dphi = np.diff(phi_edges)

    R_m, Z_m = np.meshgrid(r_mid, z_mid, indexing="ij")
    DR_m, DZ_m = np.meshgrid(dr, dz, indexing="ij")
    rho = (
        stellar_density(R_m, Z_m, True, scale)
        + stellar_density(R_m, Z_m, False, scale)
        + gas_density(R_m, Z_m, True)
        + gas_density(R_m, Z_m, False)
    )
    wedge_mass = rho[:, :, None] * R_m[:, :, None] * DR_m[:, :, None] * DZ_m[:, :, None] * dphi[None, None, :]
    phi_a = phi_mid[None, None, :]
    x = (R_m[:, :, None] * np.cos(phi_a)).ravel()
    y = (R_m[:, :, None] * np.sin(phi_a)).ravel()
    z = np.broadcast_to(Z_m[:, :, None], wedge_mass.shape).ravel()
    mass = wedge_mass.ravel()
    keep = mass > 0
    return ComponentGrid(x[keep], y[keep], z[keep], mass[keep])


# ═══════════════════════════════════════════════════════════════════════════════
# MC100 qcopula generation
# ═══════════════════════════════════════════════════════════════════════════════

def load_baryon_band(path: Optional[Path] = None) -> dict[str, np.ndarray]:
    """Load the weighted hybrid baryonic band from a CSV.

    Returns a dict with keys: R_kpc, hybrid_center, hybrid_p5, hybrid_p16,
    hybrid_p50, hybrid_p84, hybrid_p95 (and family center columns).
    """
    p = Path(path) if path is not None else _DATA_DIR / "baryon_band.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Copy fig2c_weighted_hybrid_baryon_band.csv from "
            "the project's outputs/ folder to release/data/baryon_band.csv."
        )
    data: dict[str, list] = {}
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def _build_cov() -> tuple[np.ndarray, np.ndarray]:
    logr = np.log(KNOT_R)
    dist = logr[:, None] - logr[None, :]
    cov = np.exp(-0.5 * (dist / LOGR_CORRELATION_LENGTH) ** 2)
    cov += 1.0e-8 * np.eye(len(KNOT_R))
    return logr, cov


def _quantile_map_1d(
    u_line: np.ndarray,
    q05: np.ndarray,
    q16: np.ndarray,
    q50: np.ndarray,
    q84: np.ndarray,
    q95: np.ndarray,
) -> np.ndarray:
    probs = np.array([0.05, 0.16, 0.50, 0.84, 0.95])
    out = np.empty_like(u_line)
    for j in range(len(u_line)):
        out[j] = np.interp(u_line[j], probs, [q05[j], q16[j], q50[j], q84[j], q95[j]])
    return out


def build_mc100_draws(
    band_path: Optional[Path] = None,
    n_draws: int = N_DRAWS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Generate *n_draws* baryonic velocity curves using the qcopula.

    Parameters
    ----------
    band_path : path to baryon_band.csv; defaults to release/data/baryon_band.csv.
    n_draws   : number of Monte Carlo draws (default 100).
    seed      : NumPy random seed (default 20260607).

    Returns
    -------
    r_line : 1-D radii array [kpc]
    draws  : list of (label, v_curve [km/s], u_line) tuples, length n_draws.
             label = "fig2b_NN" (NN = 1..n_draws).
    """
    from scipy.stats import norm as _norm

    band = load_baryon_band(band_path)
    r_line = band["R_kpc"]

    q05 = np.interp(r_line, band["R_kpc"], band["hybrid_p5"])
    q16 = np.interp(r_line, band["R_kpc"], band["hybrid_p16"])
    q50 = np.interp(r_line, band["R_kpc"], band["hybrid_center"])
    q84 = np.interp(r_line, band["R_kpc"], band["hybrid_p84"])
    q95 = np.interp(r_line, band["R_kpc"], band["hybrid_p95"])

    knot_logr, cov = _build_cov()
    line_logr = np.log(r_line)
    u_global = _norm.cdf(np.linspace(-1.5, 1.5, n_draws))
    rng = np.random.default_rng(seed)

    draws = []
    for i in range(n_draws):
        z_raw = rng.multivariate_normal(np.zeros(len(KNOT_R)), cov)
        z_shape = z_raw - z_raw.mean()
        u_shape_knots = ndtr(z_shape) - 0.5

        u_knots = np.clip(
            u_global[i] + SIGMA_AMP * u_shape_knots,
            U_KNOT_LOW,
            U_KNOT_HIGH,
        )
        u_line = np.clip(
            PchipInterpolator(knot_logr, u_knots)(line_logr),
            U_KNOT_LOW,
            U_KNOT_HIGH,
        )
        v = _quantile_map_1d(u_line, q05, q16, q50, q84, q95)
        draws.append((f"fig2b_{i + 1:02d}", v, u_line))

    return r_line, draws


# ═══════════════════════════════════════════════════════════════════════════════
# Grid helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_radial_grid(
    n_log: int = 300,
    r_min: float = 1.0,
    r_max: float = 800.0,
    r_obs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ~448-point R grid: log-regular merged with observational radii."""
    r_log = np.logspace(np.log10(r_min), np.log10(r_max), n_log)
    if r_obs is not None:
        return np.unique(np.concatenate([r_log, r_obs]))
    return r_log


def make_vertical_grid(rv: np.ndarray, zv: np.ndarray) -> np.ndarray:
    """Return N_vert × 2 array of (R, z) obs points sorted by (R, z)."""
    idx = np.lexsort((zv, rv))
    return np.column_stack([rv[idx], zv[idx]])
