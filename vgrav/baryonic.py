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
  2. Global amplitude shift (stratified ±1σ) + shape perturbation (SIGMA_AMP=0.20)
  3. Map knot quantiles to velocities from the band's 5 percentile levels
  4. Pchip-interpolate to r_line; clip to [u_low=0.12, u_high=0.88] of the band

Seed: 20260607.  Alternative ±1σ stratification (narrower global range than release/).

Reference: step1_build_baryonic_mc100.py (this repository)
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
    hybrid_p50, hybrid_p84, hybrid_p95 (and reconstruction center columns).
    """
    p = Path(path) if path is not None else _DATA_DIR / "baryon_band.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Place baryon_band.csv in release/data/."
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
             label = "draw_NN" (NN = 1..n_draws).
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
    u_global = _norm.cdf(np.linspace(-1.0, 1.0, n_draws))
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
        draws.append((f"draw_{i + 1:02d}", v, u_line))

    return r_line, draws


# ═══════════════════════════════════════════════════════════════════════════════
# Grid helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_radial_weights(
    radial_baryons: np.ndarray,
    band_path: Optional[Path] = None,
    sigma_w_base: float = 0.15,
    r_min: float = 2.0,
    r_max: float = 60.0,
) -> np.ndarray:
    """Per-draw radius-dependent Gaussian weights for the MC100 ensemble.

    Each draw is scored by how much its baryonic velocity deviates from the
    band median (u=0.5), normalised by a radius-dependent tolerance sigma_w(R)
    that is wider where the band itself is wide (outer R) and narrower where
    the band is tight (inner R).  Score integrated log-uniformly over R.

    Parameters
    ----------
    radial_baryons : (N_draws, n_r) baryonic rotation curve array [km/s]
    band_path      : path to baryon_band.csv; defaults to release/data/baryon_band.csv
    sigma_w_base   : tolerance at the median-width radius (default 0.15)
    r_min, r_max   : radial integration range [kpc] (default 2-60 kpc)

    Returns
    -------
    w : (N_draws,) normalised weights summing to 1
    """
    band = load_baryon_band(band_path)
    r = band["R_kpc"]
    mask = (r >= r_min) & (r <= r_max)
    r_m = r[mask]

    q05 = band["hybrid_p5"][mask]
    q16 = band["hybrid_p16"][mask]
    q50 = band["hybrid_center"][mask]
    q84 = band["hybrid_p84"][mask]
    q95 = band["hybrid_p95"][mask]

    frac_width = (q84 - q16) / (2.0 * np.maximum(q50, 1.0))
    ref_width = np.median(frac_width)
    sigma_w = sigma_w_base * frac_width / ref_width  # (n_masked,)

    # Interpolate each draw to the band radii
    n_draws = radial_baryons.shape[0]
    V = np.empty((n_draws, r_m.size))
    for i in range(n_draws):
        V[i] = np.interp(r_m, band["R_kpc"], radial_baryons[i])

    # Invert quantile map: velocity → quantile level
    bp_u = np.array([0.05, 0.16, 0.50, 0.84, 0.95])
    bp_v = np.vstack([q05, q16, q50, q84, q95])  # (5, n_masked)
    u = np.empty_like(V)
    for j in range(V.shape[1]):
        u[:, j] = np.interp(V[:, j], bp_v[:, j], bp_u, left=0.0, right=1.0)

    # Log-R weighted score: ⟨(u − 0.5)² / σ_w²⟩_{log R}
    dlogR = np.gradient(np.log(r_m))
    dlogR /= dlogR.sum()
    dev2 = (u - 0.5) ** 2 / sigma_w[np.newaxis, :] ** 2
    scores = np.average(dev2, axis=1, weights=dlogR)

    w = np.exp(-0.5 * scores)
    w /= w.sum()
    return w


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


# ═══════════════════════════════════════════════════════════════════════════════
# Cylindrical-solver support: basis potentials + per-draw scale calibration
# ═══════════════════════════════════════════════════════════════════════════════

def basis_potentials(grid):
    """Pre-compute the two Newtonian basis potentials on a cylindrical grid.

    The total baryonic density decomposes linearly as:
        rho(R,z; s) = s * rho_stellar(R,z) + rho_fixed(R,z)

    where rho_stellar = thin + thick stellar discs (scalable) and
    rho_fixed = HI + H2 gas + bulge (held fixed).

    Because Poisson's equation is linear, the total potential is:
        phi_N(R,z; s) = s * phi_stellar(R,z) + phi_fixed(R,z)

    This function solves for phi_stellar and phi_fixed once so that per-draw
    scaling requires only interpolation (no further Poisson solves).

    Parameters
    ----------
    grid : CylGrid from vgrav.solver.make_grid

    Returns
    -------
    rho_stellar : stellar density on grid [Msun/kpc³]
    phi_stellar : Newtonian potential of stellar component [(km/s)²]
    mass_stellar : total stellar mass [Msun]
    rho_fixed   : gas + bulge density on grid [Msun/kpc³]
    phi_fixed   : Newtonian potential of fixed components [(km/s)²]
    mass_fixed  : total fixed mass [Msun]
    """
    from vgrav.solver import solve_newtonian

    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")

    rho_stellar = stellar_density(RR, ZZ, True) + stellar_density(RR, ZZ, False)
    rho_fixed = (
        gas_density(RR, ZZ, True) + gas_density(RR, ZZ, False) + bulge_density(RR, ZZ)
    )

    mass_stellar, phi_stellar = solve_newtonian(grid, rho_stellar)
    mass_fixed, phi_fixed = solve_newtonian(grid, rho_fixed)

    return rho_stellar, phi_stellar, mass_stellar, rho_fixed, phi_fixed, mass_fixed


def calibrate_scale(
    phi_stellar,
    phi_fixed,
    mass_stellar: float,
    mass_fixed: float,
    grid,
    vc_target: np.ndarray,
    r_line: np.ndarray,
) -> float:
    """Find the stellar scale factor s that best matches a target rotation curve.

    Solves the least-squares problem:
        min_s  ||s * v²_stellar(R) + v²_fixed(R) - v²_target(R)||²

    where v²(R) = R * ∂φ/∂R|_{z=0} from the pre-computed basis potentials.
    The fit is restricted to 1 ≤ R ≤ 25 kpc where the data are most constraining.

    Parameters
    ----------
    phi_stellar, phi_fixed : basis potentials on grid [(km/s)²]
    mass_stellar, mass_fixed : basis masses [Msun]
    grid    : CylGrid
    vc_target : target rotation curve at r_line [km/s]
    r_line    : radii for vc_target [kpc]

    Returns
    -------
    s : stellar scale factor (clipped to [0.3, 4.0])
    """
    from vgrav.solver import radial_v2, blend_outer
    import math as _math
    from vgrav._constants import G as _G

    v2_st = radial_v2(grid, phi_stellar, r_line)
    v2_fx = radial_v2(grid, phi_fixed, r_line)

    # Blend to monopole beyond 55 kpc (solver boundary inaccuracy)
    outer_st = _G * mass_stellar / np.maximum(r_line, 1e-9)
    outer_fx = _G * mass_fixed / np.maximum(r_line, 1e-9)
    v2_st = blend_outer(r_line, v2_st, outer_st)
    v2_fx = blend_outer(r_line, v2_fx, outer_fx)

    mask = (r_line >= 1.0) & (r_line <= 25.0)
    v2_t = vc_target[mask] ** 2
    A = v2_st[mask]
    b = v2_t - v2_fx[mask]

    s = float(np.dot(A, b) / np.maximum(np.dot(A, A), 1e-30))
    return float(np.clip(s, 0.3, 4.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Imig+2025 population-resolved stellar density + radial calibration
# ═══════════════════════════════════════════════════════════════════════════════

_R_SUN_IMIG = 8.122      # kpc — Imig+2025 solar radius for profile normalisation only
_R_OBS_MAX = 20.0        # kpc — outer radius of APOGEE constraint domain
_R_TAPER_MAX = 30.0      # kpc — outer taper fully zero here
_IMIG_FITS = _DATA_DIR / "mw_density_params.fits"

# Calibration hyperparameters — default build_fig2_consolidated_radialcal_hybrid.py
_CONTROL_R = np.array([0.0, 2.0, 4.0, 6.0, R_SUN, 10.0, 14.0, 20.0, 30.0, 45.0, 60.0, 70.0])
_FIT_R = np.geomspace(1.0, 60.0, 140)
_SIGMA_TARGET_KMS = 2.5
_SIGMA_SOLAR_WEIGHT = 0.25
_SIGMA_OUTER_WEIGHT = 0.45
_SMOOTH_STRENGTH = 2.0
_WEIGHT_LOW = 0.15
_WEIGHT_HIGH = 5.0


@dataclass
class ImigPrecomp:
    """Pre-computed Imig+2025 calibration basis (one-time 13 Poisson solves).

    Call imig_precompute(grid) once before the MC loop; pass the result to
    calibrate_imig_draw() for each draw (1 Poisson solve per draw).
    """
    grid: object
    stellar: np.ndarray    # (nR, nz) Imig+2025 stellar density [Msun/kpc³]
    gas: np.ndarray        # (nR, nz) HI + H2 gas density [Msun/kpc³]
    bulge: np.ndarray      # (nR, nz) bulge density [Msun/kpc³]
    basis_rho: list        # 12 × (nR, nz): hat_k(R) × stellar
    phi_basis: list        # 12 × (nR, nz): Newtonian potentials [(km/s)²]
    mass_basis: list       # 12 floats [Msun]
    fixed: np.ndarray      # (nR, nz) gas + bulge [Msun/kpc³]
    phi_fixed: np.ndarray  # (nR, nz) Newtonian potential of fixed [(km/s)²]
    mass_fixed: float      # total fixed mass [Msun]
    v2_basis: np.ndarray   # (140, 12) radial v² design matrix [(km/s)²]
    v2_fixed: np.ndarray   # (140,) radial v² of fixed [(km/s)²]
    A_bottom: np.ndarray   # (13, 12) constant constraint rows
    b_bottom: np.ndarray   # (13,) constant constraint RHS


def _radial_broken_exp(
    h_r_in: float,
    h_r_out: float,
    r_peak: float,
    radius: np.ndarray,
) -> np.ndarray:
    """Imig+2025 broken-exponential radial profile, normalised at R_SUN_IMIG."""
    radius = np.abs(np.asarray(radius, dtype=float))
    rp_in = -h_r_in * (r_peak - _R_SUN_IMIG)
    rp_out = -h_r_out * (r_peak - _R_SUN_IMIG)
    norm_constant = rp_in - rp_out
    values = np.where(
        radius <= r_peak,
        np.exp(-h_r_in * (radius - _R_SUN_IMIG) - norm_constant),
        np.exp(-h_r_out * (radius - _R_SUN_IMIG)),
    )
    if r_peak <= _R_SUN_IMIG:
        values = values * math.exp(norm_constant)
    return values


def _population_shape(RR: np.ndarray, ZZ: np.ndarray, params: np.ndarray) -> np.ndarray:
    h_r_in, h_r_out, h_z0, r_peak, a_flare = params
    h_z = h_z0 + a_flare * (np.abs(RR) - _R_SUN_IMIG)
    h_z = np.maximum(h_z, 0.01)  # guard against non-positive scale heights at small R
    return _radial_broken_exp(h_r_in, h_r_out, r_peak, RR) * np.exp(-np.abs(ZZ) / h_z)


def _outer_taper(radius: np.ndarray) -> np.ndarray:
    """Unity inside 20 kpc; smooth cosine to zero at 30 kpc."""
    radius = np.asarray(radius, dtype=float)
    weight = np.ones_like(radius)
    weight[radius >= _R_TAPER_MAX] = 0.0
    mid = (radius > _R_OBS_MAX) & (radius < _R_TAPER_MAX)
    t = (radius[mid] - _R_OBS_MAX) / (_R_TAPER_MAX - _R_OBS_MAX)
    weight[mid] = 0.5 * (1.0 + np.cos(math.pi * t))
    return weight


def imig_stellar_density(grid, tapered: bool = True) -> np.ndarray:
    """Imig+2025 APOGEE population-resolved stellar density [Msun/kpc³].

    Reads mw_density_params.fits (FITS columns PARAMS_MED, LOG_STELLAR_MASS).
    186 populations; each normalised to its published mass via cylindrical volume
    element 2π R dR dz.

    Parameters
    ----------
    grid    : CylGrid from vgrav.solver.make_grid
    tapered : apply smooth outer taper 20–30 kpc (recommended)
    """
    from astropy.io import fits as _fits

    if not _IMIG_FITS.exists():
        raise FileNotFoundError(
            f"Imig+2025 FITS not found: {_IMIG_FITS}\n"
            "Place mw_density_params.fits in release_alt/data/."
        )

    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    table = _fits.getdata(str(_IMIG_FITS))
    valid = np.isfinite(table["LOG_STELLAR_MASS"]) & np.isfinite(table["PARAMS_MED"][:, 0])

    taper = _outer_taper(RR) if tapered else np.ones_like(RR)
    volume = 2.0 * math.pi * RR * grid.dR * grid.dz
    density = np.zeros_like(RR)

    for params, log_mass in zip(table["PARAMS_MED"][valid], table["LOG_STELLAR_MASS"][valid]):
        shape = _population_shape(RR, ZZ, np.asarray(params, dtype=float)) * taper
        norm = float(np.sum(shape * volume))
        if norm > 0.0:
            density += float(10.0 ** log_mass) * shape / norm

    return density


def _hat_weights(rr: np.ndarray) -> list:
    """Piecewise-linear hat basis functions at _CONTROL_R knots."""
    weights = []
    for index in range(_CONTROL_R.size):
        values = np.zeros_like(_CONTROL_R)
        values[index] = 1.0
        weights.append(np.interp(rr, _CONTROL_R, values))
    return weights


def _interpolation_row(radius: float) -> np.ndarray:
    """Piecewise-linear interpolation weights at a single radius."""
    return np.array(
        [np.interp(radius, _CONTROL_R, column) for column in np.eye(_CONTROL_R.size)]
    )


def _build_imig_constraint_rows() -> tuple:
    """Pre-build the constant lower rows of the lsq_linear system."""
    n = _CONTROL_R.size  # 12

    smooth = np.zeros((n - 2, n))
    for row in range(n - 2):
        smooth[row, row:row + 3] = _SMOOTH_STRENGTH * np.array([1.0, -2.0, 1.0])

    solar = _interpolation_row(R_SUN)
    outer_60 = _interpolation_row(60.0)
    outer_70 = _interpolation_row(70.0)

    A_bottom = np.vstack([
        smooth,
        solar[np.newaxis, :] / _SIGMA_SOLAR_WEIGHT,
        outer_60[np.newaxis, :] / _SIGMA_OUTER_WEIGHT,
        outer_70[np.newaxis, :] / _SIGMA_OUTER_WEIGHT,
    ])
    b_bottom = np.concatenate([
        np.zeros(n - 2),
        np.array([1.0 / _SIGMA_SOLAR_WEIGHT]),
        np.array([1.0 / _SIGMA_OUTER_WEIGHT, 1.0 / _SIGMA_OUTER_WEIGHT]),
    ])
    return A_bottom, b_bottom


def imig_precompute(grid, tapered: bool = True) -> ImigPrecomp:
    """Pre-compute Imig+2025 calibration basis: 13 Poisson solves (one-time).

    Parameters
    ----------
    grid    : CylGrid (r_min=0.0, r_max=70.0, nR=281, nz=641 recommended)
    tapered : outer taper on stellar density

    Returns
    -------
    ImigPrecomp for use in imig_calibrate_weights() and calibrate_imig_draw()
    """
    from vgrav.solver import solve_newtonian, radial_v2

    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    n_ctrl = _CONTROL_R.size

    print("  [imig_precompute] Loading Imig+2025 stellar density...", flush=True)
    stellar = imig_stellar_density(grid, tapered=tapered)

    print("  [imig_precompute] Computing gas + bulge density...", flush=True)
    gas = gas_density(RR, ZZ, True) + gas_density(RR, ZZ, False)
    bulge = bulge_density(RR, ZZ)
    fixed = gas + bulge

    print("  [imig_precompute] Solving phi_fixed (1/13 Poisson solves)...", flush=True)
    mass_fixed, phi_fixed = solve_newtonian(grid, fixed)
    v2_fixed = radial_v2(grid, phi_fixed, _FIT_R)

    print(f"  [imig_precompute] Solving {n_ctrl} basis potentials...", flush=True)
    basis_rho, phi_basis, mass_basis, v2_cols = [], [], [], []
    for k, hat_k in enumerate(_hat_weights(RR)):
        rho_k = hat_k * stellar
        basis_rho.append(rho_k)
        mass_k, phi_k = solve_newtonian(grid, rho_k)
        phi_basis.append(phi_k)
        mass_basis.append(mass_k)
        v2_cols.append(radial_v2(grid, phi_k, _FIT_R))
        print(f"    basis {k + 1}/{n_ctrl} done.", flush=True)

    v2_basis = np.column_stack(v2_cols)  # (140, 12)
    A_bottom, b_bottom = _build_imig_constraint_rows()

    print("  [imig_precompute] Complete (13 Poisson solves done).", flush=True)
    return ImigPrecomp(
        grid=grid,
        stellar=stellar,
        gas=gas,
        bulge=bulge,
        basis_rho=basis_rho,
        phi_basis=phi_basis,
        mass_basis=mass_basis,
        fixed=fixed,
        phi_fixed=phi_fixed,
        mass_fixed=mass_fixed,
        v2_basis=v2_basis,
        v2_fixed=v2_fixed,
        A_bottom=A_bottom,
        b_bottom=b_bottom,
    )


def imig_calibrate_weights(
    precomp: ImigPrecomp,
    target_v: np.ndarray,
    target_r: np.ndarray,
) -> np.ndarray:
    """lsq_linear calibration only — returns 12 correction weights (no Poisson solve).

    Use in step2 where phi_N comes from the step1 CSV (avoids redundant Poisson).

    Parameters
    ----------
    precomp  : result of imig_precompute()
    target_v : target rotation curve [km/s] at target_r
    target_r : radii [kpc]

    Returns
    -------
    weights : (12,) correction weights in [0.15, 5.0]
    """
    from scipy.optimize import lsq_linear as _lsq

    target_fit = PchipInterpolator(target_r, target_v, extrapolate=True)(_FIT_R)
    sigma_v2 = np.maximum(2.0 * target_fit * _SIGMA_TARGET_KMS, 1.0)

    A = np.vstack([precomp.v2_basis / sigma_v2[:, np.newaxis], precomp.A_bottom])
    b = np.concatenate([(target_fit ** 2 - precomp.v2_fixed) / sigma_v2, precomp.b_bottom])

    result = _lsq(
        A, b,
        bounds=(np.full(_CONTROL_R.size, _WEIGHT_LOW), np.full(_CONTROL_R.size, _WEIGHT_HIGH)),
        lsmr_tol="auto",
    )
    return result.x


def calibrate_imig_draw(
    precomp: ImigPrecomp,
    target_v: np.ndarray,
    target_r: np.ndarray,
) -> tuple:
    """Calibrate Imig+2025 density for one draw: lsq_linear + 1 Poisson solve.

    Matches default calibrate_imig_density() exactly.

    Parameters
    ----------
    precomp  : result of imig_precompute()
    target_v : target rotation curve [km/s] at target_r
    target_r : radii [kpc]

    Returns
    -------
    rho_N   : (nR, nz) calibrated baryonic density [Msun/kpc³]
    mass_N  : total baryonic mass [Msun]
    phi_N   : (nR, nz) consistent Newtonian potential [(km/s)²]
    weights : (12,) lsq_linear correction weights
    """
    from vgrav.solver import solve_newtonian as _solve

    weights = imig_calibrate_weights(precomp, target_v, target_r)

    correction = PchipInterpolator(_CONTROL_R, weights, extrapolate=True)(precomp.grid.R)
    correction_2d = np.clip(correction, _WEIGHT_LOW, _WEIGHT_HIGH)[:, np.newaxis]
    rho_N = correction_2d * precomp.stellar + precomp.fixed

    mass_N, phi_N = _solve(precomp.grid, rho_N)
    return rho_N, mass_N, phi_N, weights


def reconstruct_rho_from_weights(
    weights: np.ndarray,
    stellar: np.ndarray,
    fixed: np.ndarray,
    grid,
) -> np.ndarray:
    """Reconstruct calibrated density from weights (no Poisson; for secondary grids).

    Uses the same Pchip correction as calibrate_imig_draw.

    Parameters
    ----------
    weights : (12,) from imig_calibrate_weights or calibrate_imig_draw
    stellar : (nR, nz) Imig+2025 stellar density on the target grid
    fixed   : (nR, nz) gas + bulge on the target grid
    grid    : CylGrid with the target R array
    """
    correction = PchipInterpolator(_CONTROL_R, weights, extrapolate=True)(grid.R)
    correction_2d = np.clip(correction, _WEIGHT_LOW, _WEIGHT_HIGH)[:, np.newaxis]
    return correction_2d * stellar + fixed


def disc_rho_from_weights(weights: np.ndarray, precomp: ImigPrecomp) -> np.ndarray:
    """Calibrated disc density (stellar + gas, no bulge) from weights.

    Use for STVG ComponentGrid: bulge is added analytically in predict_stvg().

    Parameters
    ----------
    weights : (12,) from imig_calibrate_weights or calibrate_imig_draw
    precomp : ImigPrecomp with stellar and gas arrays
    """
    correction = PchipInterpolator(_CONTROL_R, weights, extrapolate=True)(precomp.grid.R)
    correction_2d = np.clip(correction, _WEIGHT_LOW, _WEIGHT_HIGH)[:, np.newaxis]
    return correction_2d * precomp.stellar + precomp.gas


def reconstruct_phi_from_weights(weights: np.ndarray, precomp: ImigPrecomp) -> np.ndarray:
    """Approximate Newtonian potential via Poisson linearity (no new Poisson solve).

    phi ≈ Σ_k w_k × phi_basis_k + phi_fixed.  Accurate when the Pchip correction
    stays close to the piecewise-linear hat combination (small clipping region).
    Use for QUMOND boundary condition on the secondary grid (cyl_grid_q).

    Parameters
    ----------
    weights : (12,) from imig_calibrate_weights or calibrate_imig_draw
    precomp : ImigPrecomp with phi_basis and phi_fixed
    """
    phi = precomp.phi_fixed.copy()
    for w, phi_k in zip(weights, precomp.phi_basis):
        phi = phi + w * phi_k
    return phi


def build_component_grid_from_rho_cyl(
    rho_disc: np.ndarray,
    cyl_grid,
    nr: int = 190,
    nz: int = 90,
    nphi: int = 80,
) -> ComponentGrid:
    """Build 3D ComponentGrid from Imig+2025-calibrated disc density (for STVG).

    Replaces build_component_grid(scale=s).  Samples the calibrated
    disc density (stellar + gas, NO bulge — bulge is added analytically in
    predict_stvg via stvg_bulge_yukawa) via bilinear interpolation.

    Parameters
    ----------
    rho_disc : (nR, nz_cyl) disc density = disc_rho_from_weights(weights, precomp_c)
    cyl_grid : CylGrid used for calibration (r_min=0.0)
    nr, nz, nphi : 3D grid resolution (matches existing build_component_grid)
    """
    from vgrav.solver import interp2

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

    rho_flat = interp2(cyl_grid, rho_disc, R_m.ravel(), Z_m.ravel())
    rho_3d = rho_flat.reshape(R_m.shape)

    wedge_mass = (
        rho_3d[:, :, np.newaxis]
        * R_m[:, :, np.newaxis]
        * DR_m[:, :, np.newaxis]
        * DZ_m[:, :, np.newaxis]
        * dphi[np.newaxis, np.newaxis, :]
    )
    phi_a = phi_mid[np.newaxis, np.newaxis, :]
    x = (R_m[:, :, np.newaxis] * np.cos(phi_a)).ravel()
    y = (R_m[:, :, np.newaxis] * np.sin(phi_a)).ravel()
    z_arr = np.broadcast_to(Z_m[:, :, np.newaxis], wedge_mass.shape).ravel()
    mass_arr = wedge_mass.ravel()
    keep = mass_arr > 0
    return ComponentGrid(x[keep], y[keep], z_arr[keep], mass_arr[keep])
