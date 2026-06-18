"""Gravity model equations for 11 models tested against Wang+2026 data.

All functions work in kpc / (km/s)^2 units throughout.

Models
------
k=0 (no free parameters):
  Baryonic Newtonian         — reference prediction
  QUMOND simple / standard   — proxy (vc = sqrt(nu) * vc_N)
  QUMOND MLS/RAR             — proxy
  HMG k=1                    — see predict_hmg()
  Emergent Gravity (fixed)   — nu = 1 + sqrt(A_EG / g_N)

k=1 (one free parameter):
  STVG                       — Yukawa disk + bulge, (alpha, mu) fitted
  VEG free a_EG              — nu = 1 + sqrt(a_EG / g_N), a_EG fitted
  f(R) screened              — nu = 1 + delta * exp(-x/xc), (delta, xc) fitted
  Refracted Gravity          — eps(x), (eps_inf, xc) fitted

k=2 (two free parameters):
  CDM NFW                    — (rho_s, r_s) from local DM density + scale radius
  CDM Einasto                — (rho_s, r_s, alpha) fitted; k=2 in reduced chi2

Proxy vs solver
---------------
For MOND the proxy applies vc = sqrt(nu(g_N/a0)) * vc_N pointwise (k=0).
The full QUMOND solver (in solver.py) adds the phantom density to the Poisson
equation; the proxy is used for the main publication figures.
For STVG and QUMOND from scratch, use predict_stvg_solver() and
predict_qumond_solver() which call the cylindrical Poisson solver.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from vgrav._constants import (
    G, A0_KMS2_PER_KPC, A_EG_FIXED, HMG_EXTRA, R_SUN, GEV_CM3_TO_MSUN_KPC3,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MOND interpolation functions
# ═══════════════════════════════════════════════════════════════════════════════

def nu_mond(x: np.ndarray, kind: str = "standard") -> np.ndarray:
    """MOND nu(x) interpolation function.

    Parameters
    ----------
    x    : |g_N| / a0  (dimensionless)
    kind : 'simple', 'standard', or 'rar' (RAR/MLS)

    Returns
    -------
    nu : enhancement factor such that g_total = nu * g_N
    """
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    if kind == "simple":
        return 0.5 + np.sqrt(0.25 + 1.0 / x)
    if kind == "standard":
        return np.sqrt(0.5 + np.sqrt(0.25 + 1.0 / (x * x)))
    if kind == "rar":
        return 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    raise ValueError(f"Unknown MOND kind: {kind!r}.  Use 'simple', 'standard', or 'rar'.")


# ═══════════════════════════════════════════════════════════════════════════════
# HMG (Hubble-scale Modified Gravity)
# ═══════════════════════════════════════════════════════════════════════════════

def hmg_factor(gN: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """HMG enhancement factor f_R = sqrt(1 + beta * extra / |g_N|).

    Parameters
    ----------
    gN   : Newtonian gravitational acceleration magnitude [kpc/(km/s)^2]
    beta : coupling strength (dimensionless; beta=1 is isotropic, parameter-free)

    Returns
    -------
    f_R : factor such that g_R_total = f_R * g_N_R
          and v_c_total = sqrt(f_R) * v_c_N
    """
    return np.sqrt(1.0 + beta * HMG_EXTRA / np.maximum(np.abs(np.asarray(gN, dtype=float)), 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# Algebraic nu-proxy functions (f(R), Refracted Gravity, VEG)
# ═══════════════════════════════════════════════════════════════════════════════

def nu_fr(x: np.ndarray, delta: float, xc: float) -> np.ndarray:
    """f(R) screened gravity: nu(x) = 1 + delta * exp(-x / xc).

    Parameters
    ----------
    x     : |g_N| / a0
    delta : amplitude of the screening correction (>0)
    xc    : screening scale in units of a0 (>0)
    """
    return 1.0 + delta * np.exp(-np.asarray(x, dtype=float) / xc)


def nu_rg(
    x: np.ndarray,
    eps_inf: float,
    xc: float,
    n: float = 2.0,
) -> np.ndarray:
    """Refracted Gravity: nu(x) = 1 / eps(x).

    eps(x) = eps_inf + (1 - eps_inf) * u^n / (u^n + 1),  u = x / xc

    Parameters
    ----------
    x       : |g_N| / a0
    eps_inf : permittivity at infinity (0 < eps_inf < 1)
    xc      : transition scale in units of a0
    n       : power-law index (default 2)
    """
    u = np.asarray(x, dtype=float) / xc
    un = np.power(np.maximum(u, 0.0), n)
    eps = eps_inf + (1.0 - eps_inf) * un / (un + 1.0)
    return 1.0 / np.maximum(eps, 1e-6)


def nu_eg(g_n: np.ndarray, a_eg: float = A_EG_FIXED) -> np.ndarray:
    """Verlinde Emergent Gravity: nu = 1 + sqrt(a_EG / |g_N|).

    Parameters
    ----------
    g_n  : Newtonian acceleration magnitude [kpc/(km/s)^2 / kpc] — same units as a_EG
    a_eg : EG scale acceleration (default: fixed Verlinde = c*H0/6)
    """
    return 1.0 + np.sqrt(np.maximum(a_eg / np.maximum(np.abs(np.asarray(g_n, dtype=float)), 1e-12), 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# CDM halo mass functions
# ═══════════════════════════════════════════════════════════════════════════════

def nfw_density_from_local(
    log10_rho_local_gev: float = -0.38,
    log10_rs: float = 0.80,
) -> tuple[float, float]:
    """Compute NFW (rho_s, r_s) from local DM density + scale radius.

    Parameters
    ----------
    log10_rho_local_gev : log10(rho_DM at R_Sun) in GeV/cm^3 (paper MAP: -0.38)
    log10_rs            : log10(r_s) in kpc (paper MAP: 0.80, i.e. r_s~6.3 kpc)

    Returns
    -------
    rho_s [Msun/kpc^3], r_s [kpc]
    """
    rs = 10.0 ** log10_rs
    rho_local = 10.0 ** log10_rho_local_gev * GEV_CM3_TO_MSUN_KPC3
    x = R_SUN / rs
    rho_s = rho_local * x * (1.0 + x) ** 2
    return rho_s, rs


def nfw_mass(r: np.ndarray, rho_s: float, rs: float) -> np.ndarray:
    """NFW enclosed mass M(<r) [Msun]."""
    x = np.maximum(np.asarray(r, dtype=float) / rs, 1e-10)
    return 4.0 * math.pi * rho_s * rs ** 3 * (np.log1p(x) - x / (1.0 + x))


def einasto_density_from_local(
    log10_rho_local_gev: float = -0.27,
    log10_rs: float = 0.99,
    alpha: float = 0.97,
) -> tuple[float, float, float]:
    """Compute Einasto (rho_s, r_s, alpha) from local DM density + shape.

    Paper MAP: log10_rho=-0.27, log10_rs=0.99, alpha=0.97
    """
    rs = 10.0 ** log10_rs
    rho_local = 10.0 ** log10_rho_local_gev * GEV_CM3_TO_MSUN_KPC3
    rho_s = rho_local / math.exp(-(2.0 / alpha) * ((R_SUN / rs) ** alpha - 1.0))
    return rho_s, rs, alpha


def einasto_mass(
    r_eval: np.ndarray,
    rho_s: float,
    rs: float,
    alpha: float,
    n_grid: int = 6000,
) -> np.ndarray:
    """Einasto enclosed mass M(<r) [Msun] by numerical integration."""
    rmax = max(float(np.max(r_eval)) * 1.02, 30.0)
    r = np.linspace(0.0, rmax, n_grid)
    rr = np.maximum(r, 1e-8)
    rho = rho_s * np.exp(-(2.0 / alpha) * ((rr / rs) ** alpha - 1.0))
    shell = 4.0 * math.pi * rr * rr * rho
    dr = np.diff(r)
    cum = np.zeros_like(r)
    cum[1:] = np.cumsum(0.5 * (shell[:-1] + shell[1:]) * dr)
    return np.interp(r_eval, r, cum)


# ═══════════════════════════════════════════════════════════════════════════════
# Bulge (shared with CDM and STVG)
# ═══════════════════════════════════════════════════════════════════════════════

def bulge_potential(r: np.ndarray) -> np.ndarray:
    """Spherical bulge gravitational potential [kpc^2/(km/s)^2 per kpc?? → (km/s)^2]."""
    from scipy.special import erf as _erf
    r = np.asarray(r, dtype=float)
    phi = np.zeros_like(r)
    for mass, sigma in ((6.5e9, 0.5), (1.48e10, 1.4)):
        x = r / (math.sqrt(2.0) * sigma)
        phi -= G * mass * _erf(x) / np.maximum(r, 1e-8)
    return phi


def bulge_accel_components(
    R: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical bulge acceleration (a_R, a_z) [(km/s)^2/kpc]."""
    from scipy.special import erf as _erf
    R = np.asarray(R, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.sqrt(R * R + z * z)
    menc = np.zeros_like(r)
    for mass, sigma in ((6.5e9, 0.5), (1.48e10, 1.4)):
        x = r / (math.sqrt(2.0) * sigma)
        term = math.sqrt(2.0 / math.pi) * (r / sigma) * np.exp(-x * x)
        menc += mass * (_erf(x) - term)
    fac = -G * menc / np.maximum(r, 1e-8) ** 3
    return fac * R, fac * z


# ═══════════════════════════════════════════════════════════════════════════════
# Spherical mass → v_c and phi  (CDM halos, analytic)
# ═══════════════════════════════════════════════════════════════════════════════

def spherical_vc_and_phi(
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    mass_func: Callable[[np.ndarray], np.ndarray],
    n_phi: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular velocity + vertical potential from a spherical mass function.

    Parameters
    ----------
    Rrot     : radii for vc evaluation [kpc]
    Rv, zv   : (R, z) pairs for phi evaluation [kpc]
    mass_func: M(<r) function [Msun]
    n_phi    : quadrature points for phi integral

    Returns
    -------
    vc  : [km/s], shape (len(Rrot),)
    phi : Phi(R,z) - Phi(R,0)  [(km/s)^2], shape (len(Rv),)
    """
    m_rot = mass_func(np.asarray(Rrot, dtype=float))
    vc = np.sqrt(np.maximum(G * m_rot / np.maximum(Rrot, 1e-8), 0.0))

    phi = []
    for R, z in zip(Rv, zv):
        r0 = abs(float(R))
        r1 = math.hypot(float(R), float(z))
        if r1 <= r0:
            phi.append(0.0)
            continue
        rr = np.linspace(r0, r1, n_phi)
        mm = mass_func(rr)
        integrand = G * mm / np.maximum(rr, 1e-8) ** 2
        phi.append(float(np.trapezoid(integrand, rr)))
    return vc, np.array(phi)


# ═══════════════════════════════════════════════════════════════════════════════
# High-level proxy predictors
# (take baryonic curves as input, return total model curves)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_mond_proxy(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    kind: str = "standard",
    a0_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """MOND proxy: v_c = sqrt(nu(g_N/a0)) * v_N, phi = nu(g_N/a0_R) * phi_N.

    Parameters
    ----------
    r_grid       : radii for v_c grid [kpc]
    vc_n         : baryonic rotation curve on r_grid [km/s]
    phi_n_at_obs : baryonic Phi at obs (R,z) points [(km/s)^2]
    rv_obs       : R coordinates of vertical obs points [kpc]
    kind         : MOND interpolation function
    a0_scale     : free a0 multiplier (1.0 = parameter-free k=0)

    Returns
    -------
    vc   : [km/s] on r_grid
    phi  : [(km/s)^2] at obs points
    """
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / (A0_KMS2_PER_KPC * a0_scale)
    nu = nu_mond(x_n, kind)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))

    nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    phi_out = nu_at_obs * phi_n_at_obs
    return vc_out, phi_out


def predict_hmg_proxy(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    beta: float = 1.0,
    lambda_z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """HMG proxy (anisotropic coupling).

    v_c = sqrt(f_R) * v_N,  phi = (1 + lambda_z * (f_R(R) - 1)) * phi_N

    Parameters
    ----------
    beta     : HMG coupling (k=1 if beta != 1.0, else k=0)
    lambda_z : vertical anisotropy fraction (1.0 = isotropic)
    """
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    f_r = hmg_factor(g_n, beta)
    vc_out = np.sqrt(np.maximum(f_r * vc_n ** 2, 0.0))

    f_r_at_obs = np.interp(rv_obs, r_grid, f_r, left=f_r[0], right=f_r[-1])
    phi_out = (1.0 + lambda_z * (f_r_at_obs - 1.0)) * phi_n_at_obs
    return vc_out, phi_out


def predict_nu_proxy(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    nu_func,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic nu-proxy: v_c = sqrt(nu * v_N^2), phi = nu(R) * phi_N.

    Parameters
    ----------
    nu_func : callable(x_grid) -> nu array, where x = g_N / a0
    """
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    x_n = g_n / A0_KMS2_PER_KPC
    nu = nu_func(x_n)
    vc_out = np.sqrt(np.maximum(nu * vc_n ** 2, 0.0))
    nu_at_obs = np.interp(rv_obs, r_grid, nu, left=nu[0], right=nu[-1])
    phi_out = nu_at_obs * phi_n_at_obs
    return vc_out, phi_out


def predict_cdm_nfw(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    log10_rho_gev: float = -0.38,
    log10_rs: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """CDM NFW total prediction."""
    rho_s, rs = nfw_density_from_local(log10_rho_gev, log10_rs)
    vc_h, phi_h = spherical_vc_and_phi(
        r_grid, rv_obs, zv_obs,
        lambda r: nfw_mass(r, rho_s, rs),
    )
    return np.sqrt(vc_n ** 2 + vc_h ** 2), phi_n_at_obs + phi_h


def predict_cdm_einasto(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    log10_rho_gev: float = -0.27,
    log10_rs: float = 0.99,
    alpha: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """CDM Einasto total prediction."""
    rho_s, rs, al = einasto_density_from_local(log10_rho_gev, log10_rs, alpha)
    vc_h, phi_h = spherical_vc_and_phi(
        r_grid, rv_obs, zv_obs,
        lambda r: einasto_mass(r, rho_s, rs, al),
    )
    return np.sqrt(vc_n ** 2 + vc_h ** 2), phi_n_at_obs + phi_h


# ═══════════════════════════════════════════════════════════════════════════════
# STVG direct summation (midplane acceleration + potential)
# ═══════════════════════════════════════════════════════════════════════════════

def stvg_disk_accel_and_phi(
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    component_grid,
    alpha: float = 10.68,
    mu: float = 0.07,
    chunk: int = 350_000,
    eps: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG Yukawa extra force for disc component.

    Parameters
    ----------
    component_grid : ComponentGrid from baryonic.build_component_grid()
    alpha, mu      : STVG coupling and screening length [1/kpc]
    eps            : gravitational softening [kpc]
    """
    g = component_grid
    accel = np.zeros(len(Rrot))
    for k_i, rr in enumerate(Rrot):
        total = 0.0
        for s in range(0, g.mass.size, chunk):
            dx = rr - g.x[s: s + chunk]
            dy = -g.y[s: s + chunk]
            dz = -g.z[s: s + chunk]
            dist2 = dx * dx + dy * dy + dz * dz + eps ** 2
            dist = np.sqrt(dist2)
            total += float(np.sum(
                G * alpha * g.mass[s: s + chunk] * dx * np.exp(-mu * dist) * (1.0 + mu * dist) / (dist2 * dist)
            ))
        accel[k_i] = total

    phi = np.zeros(len(Rv))
    for k_i, (R, z) in enumerate(zip(Rv, zv)):
        vals = []
        for zz in (z, 0.0):
            total = 0.0
            for s in range(0, g.mass.size, chunk):
                dx = R - g.x[s: s + chunk]
                dy = -g.y[s: s + chunk]
                dz = zz - g.z[s: s + chunk]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz + eps ** 2)
                total += float(np.sum(G * alpha * g.mass[s: s + chunk] * np.exp(-mu * dist) / dist))
            vals.append(total)
        phi[k_i] = vals[0] - vals[1]
    return accel, phi


def stvg_bulge_yukawa(
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    alpha: float = 10.68,
    mu: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG Yukawa extra force for spherical bulge (point-mass approximation)."""
    m_bulge = 6.5e9 + 1.48e10
    r = np.maximum(Rrot, 1e-8)
    accel = G * alpha * m_bulge * np.exp(-mu * r) * (1.0 + mu * r) / (r * r)

    phi = np.zeros(len(Rv))
    for k_i, (R, z) in enumerate(zip(Rv, zv)):
        r1 = math.hypot(float(R), float(z))
        r0 = abs(float(R))
        phi[k_i] = G * alpha * m_bulge * (math.exp(-mu * r1) / r1 - math.exp(-mu * r0) / r0)
    return accel, phi


def predict_stvg(
    r_grid: np.ndarray,
    vc_n: np.ndarray,
    phi_n_at_obs: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    component_grid,
    alpha: float = 10.68,
    mu: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG total prediction (Newton + Yukawa extra).

    v_c^2 = (1 + alpha) * v_N^2 - R * (a_Y_disk + a_Y_bulge)
    phi   = (1 + alpha) * phi_N + phi_Y_disk + phi_Y_bulge
    """
    a_y_disk, ph_y_disk = stvg_disk_accel_and_phi(r_grid, rv_obs, zv_obs, component_grid, alpha, mu)
    a_y_bulge, ph_y_bulge = stvg_bulge_yukawa(r_grid, rv_obs, zv_obs, alpha, mu)
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    vc_out = np.sqrt(np.maximum(r_grid * ((1.0 + alpha) * g_n - a_y_disk - a_y_bulge), 0.0))
    phi_out = (1.0 + alpha) * phi_n_at_obs + ph_y_disk + ph_y_bulge
    return vc_out, phi_out


# ═══════════════════════════════════════════════════════════════════════════════
# QUMOND full Poisson solver
# ═══════════════════════════════════════════════════════════════════════════════

def predict_qumond_solver(
    cyl_grid,
    rho_3d: np.ndarray,
    boundary_n: np.ndarray,
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    kind: str = "standard",
    a0_scale: float = 1.0,
):
    """QUMOND predictions from the full 3D Poisson solver.

    Parameters
    ----------
    cyl_grid  : CylGrid (from solver.make_grid)
    rho_3d    : density on cyl_grid [Msun/kpc^3]
    boundary_n: Newtonian boundary values on cyl_grid
    Rrot      : radii for v_c [kpc]
    Rv, zv    : obs (R, z) for phi [kpc]
    kind      : MOND interpolation function
    a0_scale  : free a0 multiplier

    Returns
    -------
    vc   : [km/s] at Rrot
    phi  : [(km/s)^2] Phi(R,z)-Phi(R,0) at (Rv, zv)
    """
    import math as _math
    from vgrav.solver import (
        solve_axisymmetric, gradients, interp2, cylindrical_divergence,
    )

    a0 = A0_KMS2_PER_KPC * a0_scale
    phi_n = solve_axisymmetric(cyl_grid, 4.0 * _math.pi * G * rho_3d, boundary_n)
    dphi_n_dR, dphi_n_dz = gradients(cyl_grid, phi_n)
    gabs = np.sqrt(dphi_n_dR ** 2 + dphi_n_dz ** 2)
    nu = nu_mond(gabs / a0, kind)
    phantom_div = cylindrical_divergence(cyl_grid, (nu - 1.0) * dphi_n_dR, (nu - 1.0) * dphi_n_dz)
    bq = boundary_n.copy()
    edge = np.zeros(cyl_grid.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    bq[edge] = boundary_n[edge] * nu[edge]
    phi_q = solve_axisymmetric(cyl_grid, 4.0 * _math.pi * G * rho_3d + phantom_div, bq)
    dphiq_dR, _ = gradients(cyl_grid, phi_q)

    z0_rot = np.zeros_like(Rrot)
    z0_vert = np.zeros_like(Rv)
    vc_out = np.sqrt(np.maximum(Rrot * interp2(cyl_grid, dphiq_dR, Rrot, z0_rot), 0.0))
    phi_out = interp2(cyl_grid, phi_q, Rv, zv) - interp2(cyl_grid, phi_q, Rv, z0_vert)
    return vc_out, phi_out
