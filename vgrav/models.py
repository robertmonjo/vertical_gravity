"""Gravity model equations for 11 models tested against Wang+2026 data.

All functions work in kpc / (km/s)^2 units throughout.

Models
------
k=0 (no free parameters):
  Baryonic Newtonian         — reference prediction
  QUMOND simple / standard   — proxy (vc = sqrt(nu) * vc_N)
  QUMOND MLS/RAR             — proxy
  VEG fixed a_EG             — nu = 1 + sqrt(A_EG / g_N), A_EG fixed

k=1 (one free parameter):
  VEG free a_EG              — nu = 1 + sqrt(a_EG / g_N), a_EG fitted
  HMG k=1                    — see predict_hmg()

k=2 (two free parameters):
  CDM NFW                    — (rho_s, r_s) from local DM density + scale radius
  STVG                       — Yukawa disk + bulge, (alpha, mu) fitted
  f(R) screened              — nu = 1 + delta * exp(-x/xc), (delta, xc) fitted
  Refracted Gravity          — eps(x), (eps_inf, xc) fitted

k=3 (three free parameters):
  CDM Einasto                — (rho_s, r_s, alpha) fitted

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


# ── Real HMG (neighbourhood scale s) ─────────────────────────────────────────
# These implement the published HMG formulae exactly as used in the paper.
# Reference: Monjo (2023) Eqs. 27 and 37; fit_hmg_common_s_mc100.py

from vgrav._constants import C_KMS, T0_KPC_PER_KMS


def hmg_angular_q_from_x(x: np.ndarray) -> np.ndarray:
    """Projected angle factor q(x) from the HMG angular geometry.

    Parameters
    ----------
    x : dimensionless speed ratio (sqrt(2)*v_N / (eps * v_H))
    """
    phi_u = math.pi / 3.0
    phi_cen = math.pi / 2.0
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    delta = np.abs(x * x - 1.0) / (x * x + 1.0)
    s2 = math.sin(phi_u) ** 2 + (math.sin(phi_cen) ** 2 - math.sin(phi_u) ** 2) * delta
    gamma = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gamma) / gamma


def epsilon_from_s(v2_n: np.ndarray, radius: np.ndarray, s: float) -> np.ndarray:
    """HMG confinement parameter ε(s).

    ε² = 2 v²_N / (s³ v²_H) + 1/6,  v_H = R / T₀
    """
    v_h = np.asarray(radius, dtype=float) / T0_KPC_PER_KMS
    return np.sqrt(2.0 * np.asarray(v2_n, dtype=float) / (s ** 3 * v_h * v_h) + 1.0 / 6.0)


def hmg_radial(v2_n: np.ndarray, radius: np.ndarray, s: float) -> np.ndarray:
    """HMG circular speed using Eq. 37 (centripetal form).

    v²_tot = (1 + extra/g_N) * v²_N,  extra = 2c/T₀ * q(x)

    Parameters
    ----------
    v2_n   : Newtonian v² on the radial grid [(km/s)²]
    radius : corresponding radii [kpc]
    s      : neighbourhood scale (free parameter, k=1)

    Returns
    -------
    vc : [km/s]
    """
    v2_n = np.asarray(v2_n, dtype=float)
    radius = np.asarray(radius, dtype=float)
    g_n = np.maximum(v2_n / np.maximum(radius, 1e-12), 1e-12)
    v_h = radius / T0_KPC_PER_KMS
    eps = epsilon_from_s(v2_n, radius, s)
    x = np.sqrt(2.0 * np.maximum(v2_n, 1e-12)) / (eps * v_h)
    q = hmg_angular_q_from_x(x)
    extra = 2.0 * C_KMS / T0_KPC_PER_KMS * q
    # Canonical: f_R = sqrt(1 + extra/g_N), vc = sqrt(f_R * v_N^2)  [fit_hmg_common_s_mc100.py]
    factor = np.sqrt(1.0 + extra / g_n)
    return np.sqrt(np.maximum(factor * v2_n, 0.0))


def integrate_delta_force(
    delta_k: np.ndarray,
    rv: np.ndarray,
    zv: np.ndarray,
) -> np.ndarray:
    """Cumulative integral of Δk_z(R, z) over z from 0 to z_obs.

    Used to compute the HMG vertical potential correction:
    Δφ(R,z) = ∫₀ᶻ Δk_z(R,z') dz'

    Parameters
    ----------
    delta_k : extra vertical force Δk_z at each (R,z) obs point [(km/s)²/kpc]
    rv, zv  : (R, z) observation coordinates [kpc]
    """
    out = np.zeros_like(delta_k)
    for radius in sorted(set(rv)):
        idx = np.where(np.abs(rv - radius) < 1e-8)[0]
        order = idx[np.argsort(zv[idx])]
        z = zv[order]
        dk = delta_k[order]
        if len(z) and z[0] <= 1e-12:
            z0, dk0 = z, dk
        else:
            z0 = np.r_[0.0, z]
            dk0 = np.r_[0.0, dk]
        cumulative = np.zeros_like(z0)
        for i in range(1, len(z0)):
            cumulative[i] = cumulative[i - 1] + 0.5 * (dk0[i] + dk0[i - 1]) * (z0[i] - z0[i - 1])
        out[order] = cumulative if len(z) and z[0] <= 1e-12 else cumulative[1:]
    return out


def hmg_eq27_vertical(
    rv: np.ndarray,
    zv: np.ndarray,
    v2_n_vert: np.ndarray,
    phi_n: np.ndarray,
    s: float,
) -> np.ndarray:
    """HMG vertical potential Δφ using Eq. 27 (spatial projection).

    Computes the HMG extra vertical force and integrates it over z.

    Parameters
    ----------
    rv, zv     : (R, z) observation coordinates [kpc]
    v2_n_vert  : Newtonian v²_N interpolated at obs R values [(km/s)²]
    phi_n      : Newtonian Φ(R,z)−Φ(R,0) at obs points [(km/s)²]
    s          : neighbourhood scale

    Returns
    -------
    phi_hmg : HMG Φ(R,z)−Φ(R,0) [(km/s)²]
    """
    from vgrav.chi2 import vertical_force_from_phi
    rv = np.asarray(rv, dtype=float)
    zv = np.asarray(zv, dtype=float)
    r = np.sqrt(rv * rv + zv * zv)
    k_z_n = vertical_force_from_phi(phi_n, rv, zv)
    g_r_n = np.maximum(v2_n_vert / np.maximum(rv, 1e-9), 1e-12)
    g_n = np.sqrt(g_r_n * g_r_n + k_z_n * k_z_n)
    v2_space = np.maximum(r * g_n, 1e-12)
    v_h = r / T0_KPC_PER_KMS
    eps_z = epsilon_from_s(v2_space, r, s)
    x_z = np.sqrt(2.0 * v2_space) / (eps_z * v_h)
    q_z = hmg_angular_q_from_x(x_z)
    extra_space = C_KMS / T0_KPC_PER_KMS * q_z
    delta_k = extra_space * zv / np.maximum(r, 1e-12)
    return phi_n + integrate_delta_force(delta_k, rv, zv)


def predict_hmg_common_s(
    r_grid: np.ndarray,
    v2_n_radial: np.ndarray,
    v2_n_vert: np.ndarray,
    phi_n: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    rr: np.ndarray,
    vv: np.ndarray,
    ss: np.ndarray,
    phi_obs: np.ndarray,
    sig_phi: np.ndarray,
    sig_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit HMG with a single common neighbourhood scale s (k=1).

    Replicates the default fit_hmg_common_s_mc100.py pipeline:
    one scalar s governs both Eq. 37 (radial) and Eq. 27 (vertical).
    Optimised via bounded 1-D search (minimize_scalar).

    Parameters
    ----------
    r_grid       : radii for vc evaluation [kpc]
    v2_n_radial  : Newtonian v²_N on r_grid [(km/s)²]
    v2_n_vert    : Newtonian v²_N interpolated at rv_obs [(km/s)²]
    phi_n        : Newtonian Φ(R,z)−Φ(R,0) at obs points [(km/s)²]
    rv_obs, zv_obs : obs (R,z) [kpc]
    rr, vv, ss   : radial fit points
    phi_obs, sig_phi, sig_z : vertical fit arrays

    Returns
    -------
    vc    : [km/s] on r_grid
    phi   : [(km/s)²] at obs points
    s_fit : fitted neighbourhood scale
    """
    from vgrav.chi2 import vertical_force_from_phi
    from scipy.optimize import minimize_scalar

    def score(log_s):
        s = math.exp(float(log_s))
        vc_line = hmg_radial(v2_n_radial, r_grid, s)
        phi = hmg_eq27_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, s)
        vc_at = np.interp(rr, r_grid, vc_line)
        kz = vertical_force_from_phi(phi, rv_obs, zv_obs)
        sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
        chi_r = float(np.sum(((vc_at - vv) / ss) ** 2))
        chi_z = float(np.sum(((phi - phi_obs) / sig_eff) ** 2))
        return chi_r + chi_z

    res = minimize_scalar(
        score,
        bounds=(0.0, math.log(300.0)),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 500},
    )
    s_fit = math.exp(float(res.x))
    vc_out = hmg_radial(v2_n_radial, r_grid, s_fit)
    phi_out = hmg_eq27_vertical(rv_obs, zv_obs, v2_n_vert, phi_n, s_fit)
    return vc_out, phi_out, s_fit


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

def make_stvg_workspace(obs_per_chunk: int, n_cells: int) -> dict:
    """Pre-allocate 5 reusable buffers for stvg_disk_accel_and_phi.

    Pass the returned dict as `workspace=` to stvg_disk_accel_and_phi and
    predict_stvg to eliminate repeated large-array allocations (mmap overhead)
    that otherwise dominate runtime for grids with many cells.

    obs_per_chunk : rows per iteration (e.g. 2 for fine grid, all_obs for coarse grid)
    n_cells       : number of ComponentGrid cells (nr × nz × nphi)
    """
    shape = (obs_per_chunk, n_cells)
    return {k: np.empty(shape, dtype=np.float64) for k in "abcde"}


def stvg_disk_accel_and_phi(
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    component_grid,
    alpha: float = 10.68,
    mu: float = 0.07,
    chunk: int = 350_000,
    eps: float = 0.03,
    workspace: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG Yukawa extra force for disc component.

    Parameters
    ----------
    component_grid : ComponentGrid from baryonic.build_component_grid()
    alpha, mu      : STVG coupling and screening length [1/kpc]
    eps            : gravitational softening [kpc]
    workspace      : optional pre-allocated buffer dict from make_stvg_workspace().
                     Eliminates repeated large-array (mmap) allocations on repeated
                     calls with the same grid.  Ignored if shape mismatches.
    """
    g = component_grid
    n_cells = g.mass.size

    # Determine obs_chunk and whether to use pre-allocated workspace
    _ws_ok = (
        workspace is not None
        and all(k in workspace for k in "abcde")
        and workspace["a"].ndim == 2
        and workspace["a"].shape[1] == n_cells
    )
    if _ws_ok:
        obs_chunk = workspace["a"].shape[0]
        ba, bb, bc, bd, be = (workspace[k] for k in "abcde")
    else:
        obs_chunk = max(1, 300_000_000 // (n_cells * 8))

    # Static 1-D terms (no obs dependence)
    _static_a = g.y * g.y + g.z * g.z + eps * eps  # accel: dy²+dz²+eps²
    _gy2      = g.y * g.y                           # phi  : gy²
    _gz2_eps2 = g.z * g.z + eps * eps               # phi  : gz²+eps²
    _mcst     = g.mass * (G * alpha)                # G·α·m per cell

    # ── Accel (midplane obs points at z=0) ────────────────────────────────
    Rrot = np.asarray(Rrot, dtype=np.float64)
    accel = np.empty(len(Rrot))

    if _ws_ok:
        for i0 in range(0, len(Rrot), obs_chunk):
            i1 = min(i0 + obs_chunk, len(Rrot))
            b = i1 - i0
            np.subtract(Rrot[i0:i1, None], g.x, out=ba[:b])   # dx
            np.multiply(ba[:b], ba[:b], out=bb[:b])            # dx²
            bb[:b] += _static_a                                  # dist²
            np.sqrt(bb[:b], out=bc[:b])                         # r
            np.multiply(bb[:b], bc[:b], out=bb[:b])             # r³ = dist²·r
            np.multiply(mu, bc[:b], out=bd[:b])                 # μr
            bd[:b] += 1.0                                        # 1+μr
            bc[:b] *= -mu                                        # −μr (overwrite r)
            np.exp(bc[:b], out=bc[:b])                          # e^{−μr}
            np.multiply(ba[:b], bc[:b], out=ba[:b])             # dx·e^{−μr}
            np.multiply(ba[:b], bd[:b], out=ba[:b])             # ·(1+μr)
            np.divide(ba[:b], bb[:b], out=ba[:b])               # /r³
            ba[:b] *= _mcst                                      # ·G·α·m
            np.sum(ba[:b], axis=1, out=accel[i0:i1])
    else:
        for i0 in range(0, len(Rrot), obs_chunk):
            rr = Rrot[i0: i0 + obs_chunk, None]
            dx = rr - g.x
            dist2 = dx * dx + _static_a
            dist = np.sqrt(dist2)
            accel[i0: i0 + obs_chunk] = np.sum(
                _mcst * dx * np.exp(-mu * dist) * (1.0 + mu * dist) / (dist2 * dist),
                axis=1,
            )

    # ── Phi (vertical obs points: Φ(R,z) − Φ(R,0)) ───────────────────────
    Rv = np.asarray(Rv, dtype=np.float64)
    zv = np.asarray(zv, dtype=np.float64)
    phi = np.empty(len(Rv))

    if _ws_ok:
        for i0 in range(0, len(Rv), obs_chunk):
            i1 = min(i0 + obs_chunk, len(Rv))
            b = i1 - i0
            np.subtract(Rv[i0:i1, None], g.x, out=ba[:b])     # dx_v
            np.multiply(ba[:b], ba[:b], out=bb[:b])            # dx_v²
            bb[:b] += _gy2                                       # dx_v²+gy²
            # (z−gz)² → be[:b]
            np.subtract(zv[i0:i1, None], g.z, out=be[:b])
            np.multiply(be[:b], be[:b], out=be[:b])
            # dist_zv² = bb + be + eps² → bc[:b]
            np.add(bb[:b], be[:b], out=bc[:b])
            bc[:b] += eps * eps
            np.sqrt(bc[:b], out=bc[:b])                         # dist_zv
            # dist_z0² = bb + gz²+eps² → bd[:b]
            np.add(bb[:b], _gz2_eps2, out=bd[:b])
            np.sqrt(bd[:b], out=bd[:b])                         # dist_z0
            # bb[:b] = e^{−μ·dist_zv}/dist_zv
            np.multiply(-mu, bc[:b], out=bb[:b])
            np.exp(bb[:b], out=bb[:b])
            bb[:b] /= bc[:b]
            # be[:b] = e^{−μ·dist_z0}/dist_z0
            np.multiply(-mu, bd[:b], out=be[:b])
            np.exp(be[:b], out=be[:b])
            be[:b] /= bd[:b]
            bb[:b] -= be[:b]                                     # term1−term2
            bb[:b] *= _mcst
            np.sum(bb[:b], axis=1, out=phi[i0:i1])
    else:
        for i0 in range(0, len(Rv), obs_chunk):
            R = Rv[i0: i0 + obs_chunk, None]
            z = zv[i0: i0 + obs_chunk, None]
            dx_v = R - g.x
            dist_zv = np.sqrt(dx_v * dx_v + _gy2 + (z - g.z) ** 2 + eps * eps)
            dist_z0 = np.sqrt(dx_v * dx_v + _gz2_eps2)
            phi[i0: i0 + obs_chunk] = np.sum(
                _mcst * (np.exp(-mu * dist_zv) / dist_zv
                         - np.exp(-mu * dist_z0) / dist_z0),
                axis=1,
            )
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


def stvg_bulge_yukawa_resolved(
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    alpha: float = 10.68,
    mu: float = 0.07,
    n_grid: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG Yukawa extra force for spherical bulge — exact spherical-shell integrals.

    Replaces the point-mass approximation with shell-by-shell integration over the
    two-Gaussian bulge profile, consistent with bulge_accel_components (Newtonian).

    The Yukawa potential/force from a thin shell at r'' at field point r is:
        exterior (r > r'): phi  = -G*alpha*dM * exp(-mu*r)  * sinh(mu*r') / (mu*r'*r)
                           force =  G*alpha*dM * exp(-mu*r)  * (1+mu*r) * sinh(mu*r') / (mu*r'*r^2)
        interior (r < r'): phi  = -G*alpha*dM * exp(-mu*r') * sinh(mu*r)  / (mu*r'*r)
                           force =  G*alpha*dM * exp(-mu*r') * (mu*r*cosh(mu*r)-sinh(mu*r)) / (mu*r'*r^2)

    Error of old point-mass: O((mu*sigma)^2) ~ 1 %% for mu=0.07 kpc^-1, sigma=1.4 kpc.
    """
    r_s = np.geomspace(1e-4, 25.0, n_grid)
    r_mid = 0.5 * (r_s[:-1] + r_s[1:])
    dr_s = np.diff(r_s)

    rho_r = np.zeros_like(r_mid)
    for mass, sigma in ((6.5e9, 0.5), (1.48e10, 1.4)):
        rho_r += mass / (math.sqrt(2.0 * math.pi) * sigma) ** 3 * np.exp(-r_mid ** 2 / (2.0 * sigma ** 2))
    dM = 4.0 * math.pi * r_mid ** 2 * rho_r * dr_s

    # Kernel weights: w_sinh[k] = dM[k]*sinh(mu*r'[k])/r'[k]  (enclosed shells)
    #                 w_exp [k] = dM[k]*exp(-mu*r'[k])/r'[k]  (outer shells)
    w_sinh = dM * np.sinh(mu * r_mid) / r_mid
    w_exp  = dM * np.exp(-mu * r_mid)  / r_mid

    # Cumulative integrals on r_s grid edges (length = n_grid)
    I1 = np.concatenate([[0.0], np.cumsum(w_sinh)])                           # I1[k] = sum_{j<k} w_sinh[j]
    I2 = np.concatenate([np.flip(np.cumsum(np.flip(w_exp))), [0.0]])          # I2[k] = sum_{j>=k} w_exp[j]

    def _phi(r_eval: np.ndarray) -> np.ndarray:
        # STVG convention: repulsive Yukawa correction potential is positive (+G*alpha*...)
        # consistent with stvg_disk_accel_and_phi and the original stvg_bulge_yukawa.
        r_e = np.maximum(np.asarray(r_eval, dtype=float), 1e-6)
        i1 = np.interp(r_e, r_s, I1, left=0.0, right=float(I1[-1]))
        i2 = np.interp(r_e, r_s, I2, left=float(I2[0]), right=0.0)
        return +G * alpha * (np.exp(-mu * r_e) * i1 + np.sinh(mu * r_e) * i2) / (mu * r_e)

    def _force(r_eval: np.ndarray) -> np.ndarray:
        # Radial STVG repulsive force.  Outer shells contribute -I2 term (inward in repulsive conv.)
        r_e = np.maximum(np.asarray(r_eval, dtype=float), 1e-6)
        i1 = np.interp(r_e, r_s, I1, left=0.0, right=float(I1[-1]))
        i2 = np.interp(r_e, r_s, I2, left=float(I2[0]), right=0.0)
        return G * alpha * (
            np.exp(-mu * r_e) * (1.0 + mu * r_e) * i1
            - (mu * r_e * np.cosh(mu * r_e) - np.sinh(mu * r_e)) * i2
        ) / (mu * r_e ** 2)

    accel = _force(np.maximum(np.asarray(Rrot, dtype=float), 1e-8))
    r1 = np.sqrt(np.asarray(Rv, dtype=float) ** 2 + np.asarray(zv, dtype=float) ** 2)
    r0 = np.abs(np.asarray(Rv, dtype=float))
    phi = _phi(r1) - _phi(r0)
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
    workspace: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """STVG total prediction (Newton + Yukawa extra).

    v_c^2 = (1 + alpha) * v_N^2 - R * (a_Y_disk + a_Y_bulge)
    phi   = (1 + alpha) * phi_N + phi_Y_disk + phi_Y_bulge

    workspace : optional pre-allocated buffer dict from make_stvg_workspace().
                Passed through to stvg_disk_accel_and_phi to avoid mmap overhead.
    """
    a_y_disk, ph_y_disk = stvg_disk_accel_and_phi(
        r_grid, rv_obs, zv_obs, component_grid, alpha, mu, workspace=workspace
    )
    a_y_bulge, ph_y_bulge = stvg_bulge_yukawa_resolved(r_grid, rv_obs, zv_obs, alpha, mu)
    g_n = vc_n ** 2 / np.maximum(r_grid, 1e-8)
    vc_out = np.sqrt(np.maximum(r_grid * ((1.0 + alpha) * g_n - a_y_disk - a_y_bulge), 0.0))
    phi_out = (1.0 + alpha) * phi_n_at_obs + ph_y_disk + ph_y_bulge
    return vc_out, phi_out


# ═══════════════════════════════════════════════════════════════════════════════
# QUMOND full Poisson solver
# ═══════════════════════════════════════════════════════════════════════════════

_QUMOND_R_BLEND_START = 48.0   # kpc: start blending solver → spherical MOND
_QUMOND_R_SOLVE_MAX   = 60.0   # kpc: fully use spherical MOND beyond this radius


def _mond_monopole_boundary(cyl_grid, mass: float, kind: str, a0: float) -> np.ndarray:
    """Correct QUMOND boundary: numerically integrate g_MOND(r) on a 1-D radial grid.

    Replaces the first-order approximation nu*phi_N, which becomes constant in the
    deep-MOND regime (g/a0 << 1) and produces the wrong radial shape at the grid
    boundary (R=70 kpc, g/a0 ~ 0.15 for the MW).
    """
    RR, ZZ = np.meshgrid(cyl_grid.R, cyl_grid.z, indexing="ij")
    rr = np.sqrt(RR * RR + ZZ * ZZ)
    r_eval = np.maximum(rr, 1e-5)
    r_grid = np.geomspace(1e-5, float(np.max(r_eval)) * 1.001, 4000)
    g_n = G * mass / np.maximum(r_grid * r_grid, 1e-12)
    g_mond = nu_mond(g_n / a0, kind) * g_n
    dr = np.diff(r_grid)
    primitive = np.zeros_like(r_grid)
    primitive[1:] = np.cumsum(0.5 * (g_mond[:-1] + g_mond[1:]) * dr)
    return np.interp(r_eval, r_grid, primitive)


def _mond_outer_vc(Rrot: np.ndarray, mass: float, kind: str, a0: float) -> np.ndarray:
    """Spherical MOND circular speed used for outer-region blending (R > 48 kpc)."""
    g_n = G * mass / np.maximum(Rrot * Rrot, 1e-30)
    return np.sqrt(np.maximum(Rrot * nu_mond(g_n / a0, kind) * g_n, 0.0))


def predict_qumond_solver(
    cyl_grid,
    rho_3d: np.ndarray,
    boundary_n: np.ndarray,
    Rrot: np.ndarray,
    Rv: np.ndarray,
    zv: np.ndarray,
    kind: str = "standard",
    a0_scale: float = 1.0,
    phi_n_precomputed: Optional[np.ndarray] = None,
):
    """QUMOND predictions from the full 3D Poisson solver.

    Parameters
    ----------
    cyl_grid          : CylGrid (from solver.make_grid)
    rho_3d            : density on cyl_grid [Msun/kpc^3]
    boundary_n        : Newtonian boundary values on cyl_grid (used if phi_n_precomputed is None)
    Rrot              : radii for v_c [kpc]
    Rv, zv            : obs (R, z) for phi [kpc]
    kind              : MOND interpolation function
    a0_scale          : free a0 multiplier
    phi_n_precomputed : pre-computed Newtonian phi on cyl_grid (skips first solve)

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
    if phi_n_precomputed is not None:
        phi_n = phi_n_precomputed
    else:
        phi_n = solve_axisymmetric(cyl_grid, 4.0 * _math.pi * G * rho_3d, boundary_n)
    dphi_n_dR, dphi_n_dz = gradients(cyl_grid, phi_n)
    gabs = np.sqrt(dphi_n_dR ** 2 + dphi_n_dz ** 2)
    nu = nu_mond(gabs / a0, kind)
    phantom_div = cylindrical_divergence(
        cyl_grid, (nu - 1.0) * dphi_n_dR, (nu - 1.0) * dphi_n_dz
    )
    mass_tot = float(np.sum(
        rho_3d * (2.0 * _math.pi * cyl_grid.R[:, None] * cyl_grid.dR * cyl_grid.dz)
    ))
    bq = _mond_monopole_boundary(cyl_grid, mass_tot, kind, a0)
    phi_q = solve_axisymmetric(cyl_grid, 4.0 * _math.pi * G * rho_3d + phantom_div, bq)
    dphiq_dR, _ = gradients(cyl_grid, phi_q)

    Rrot = np.asarray(Rrot, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    zv = np.asarray(zv, dtype=float)
    z0_rot = np.zeros_like(Rrot)
    z0_vert = np.zeros_like(Rv)
    vc_inner = np.sqrt(np.maximum(Rrot * interp2(cyl_grid, dphiq_dR, Rrot, z0_rot), 0.0))
    vc_outer = _mond_outer_vc(Rrot, mass_tot, kind, a0)
    vc_out = vc_inner.copy()
    _mid = (Rrot > _QUMOND_R_BLEND_START) & (Rrot < _QUMOND_R_SOLVE_MAX)
    _t = (Rrot[_mid] - _QUMOND_R_BLEND_START) / (_QUMOND_R_SOLVE_MAX - _QUMOND_R_BLEND_START)
    vc_out[_mid] = (1.0 - _t) * vc_inner[_mid] + _t * vc_outer[_mid]
    vc_out[Rrot >= _QUMOND_R_SOLVE_MAX] = vc_outer[Rrot >= _QUMOND_R_SOLVE_MAX]
    phi_out = interp2(cyl_grid, phi_q, Rv, zv) - interp2(cyl_grid, phi_q, Rv, z0_vert)
    return vc_out, phi_out


# ═══════════════════════════════════════════════════════════════════════════════
# Per-draw CDM parameter fitting
# ═══════════════════════════════════════════════════════════════════════════════

def _cdm_score(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
    mass_func,
):
    """Evaluate CDM chi2_total given a spherical halo mass function."""
    from vgrav.chi2 import vertical_force_from_phi
    vc_h, phi_h = spherical_vc_and_phi(r_grid, rv_obs, zv_obs, mass_func)
    vc_tot = np.sqrt(np.maximum(vc_n ** 2 + vc_h ** 2, 0.0))
    phi_tot = phi_n + phi_h
    vc_at = np.interp(rr, r_grid, vc_tot)
    kz = vertical_force_from_phi(phi_tot, rv_obs, zv_obs)
    sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
    chi_r = float(np.sum(((vc_at - vv) / ss) ** 2))
    chi_z = float(np.sum(((phi_tot - phi_obs) / sig_eff) ** 2))
    return chi_r + chi_z, vc_tot, phi_tot


def predict_cdm_nfw_per_draw(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
    x0=(-0.46, 1.27),
    bounds=((-3.0, 1.0), (0.0, 2.3)),
):
    """Fit CDM-NFW parameters per baryonic draw.

    Optimizes (log10_rho_local [GeV/cm³], log10_rs [kpc]) to minimise
    chi2_total for this draw, exactly as in the paper pipeline.

    Returns
    -------
    vc    : [km/s] on r_grid
    phi   : [(km/s)²] at obs points
    theta : (log10_rho_local, log10_rs) best-fit
    """
    from scipy.optimize import minimize

    def obj(t):
        rho_s, rs = nfw_density_from_local(float(t[0]), float(t[1]))
        val, _, _ = _cdm_score(
            r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
            lambda r: nfw_mass(r, rho_s, rs),
        )
        return val

    starts = [(-0.46, 1.27), (-0.3, 1.5), (-0.6, 1.0), (-0.2, 1.8), (-0.7, 0.8)]
    best = None
    for s in starts:
        try:
            r = minimize(obj, s, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 300, "ftol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    res = best
    rho_s, rs = nfw_density_from_local(float(res.x[0]), float(res.x[1]))
    _, vc_out, phi_out = _cdm_score(
        r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
        lambda r: nfw_mass(r, rho_s, rs),
    )
    return vc_out, phi_out, tuple(res.x)


def predict_cdm_einasto_per_draw(
    r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
    x0=(-0.27, 0.99, 0.97),
    bounds=((-3.0, 1.0), (0.0, 2.3), (0.1, 3.0)),
):
    """Fit CDM-Einasto parameters per baryonic draw.

    Optimizes (log10_rho_local, log10_rs, alpha) to minimise chi2_total.
    k=3 free parameters as in the paper.

    Returns
    -------
    vc    : [km/s] on r_grid
    phi   : [(km/s)²] at obs points
    theta : (log10_rho_local, log10_rs, alpha) best-fit
    """
    from scipy.optimize import minimize

    def obj(t):
        rho_s, rs, al = einasto_density_from_local(float(t[0]), float(t[1]), float(t[2]))
        val, _, _ = _cdm_score(
            r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
            lambda r: einasto_mass(r, rho_s, rs, al),
        )
        return val

    starts = [
        (-0.27, 0.99, 0.97),   # default MAP
        (-0.41, 0.98, 0.63),   # alternative
        (-0.5,  1.3,  0.5),
        (-0.1,  1.5,  1.5),
        (-0.7,  0.7,  0.3),
    ]
    best = None
    for s in starts:
        try:
            r = minimize(obj, s, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 300, "ftol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    res = best
    rho_s, rs, al = einasto_density_from_local(float(res.x[0]), float(res.x[1]), float(res.x[2]))
    _, vc_out, phi_out = _cdm_score(
        r_grid, vc_n, phi_n, rv_obs, zv_obs, rr, vv, ss, phi_obs, sig_phi, sig_z,
        lambda r: einasto_mass(r, rho_s, rs, al),
    )
    return vc_out, phi_out, tuple(res.x)
