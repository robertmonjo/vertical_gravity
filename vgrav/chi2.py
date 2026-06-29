"""Chi-squared statistics for the combined radial + vertical fit.

Definitions
-----------
chi2_radial  = sum_i [(vc_model(R_i) - vc_obs_i) / sigma_i]^2
chi2_vertical = sum_j [(phi_model(R_j,z_j) - phi_obs_j) / sigma_eff_j]^2

where sigma_eff_j^2 = sigma_Phi_j^2 + (Kz_j * sigma_z_j)^2
and Kz_j = dPhi/dz|(R_j, z_j) is the vertical force computed numerically
from phi_model at each observation radius R.

chi2_nu (reduced) = chi2_total / (N_PRIMARY - k)
  N_PRIMARY = 196 = 152 radial + 44 vertical Wang+2026 obs points
  k = number of free model parameters (0 for parameter-free predictions)
"""
from __future__ import annotations

import numpy as np

N_OBS_RAD: int = 152   # direct_rotation / used_in_fit=true rows in fig2_observational_catalog.csv
N_OBS_VERT: int = 44
N_PRIMARY: int = N_OBS_RAD + N_OBS_VERT  # 196


def vertical_force_from_phi(
    phi: np.ndarray,
    rv: np.ndarray,
    zv: np.ndarray,
) -> np.ndarray:
    """Return Kz = dΦ/dz at each (R, z) obs point.

    Points are grouped by R and a second-order central-difference gradient
    is computed along z within each group.

    Parameters
    ----------
    phi : potential differences Φ(R,z) - Φ(R,0)  [length N_vert]
    rv  : galactocentric radii of obs points [kpc]
    zv  : vertical heights of obs points [kpc]

    Returns
    -------
    kz : dΦ/dz at each point  [(km/s)^2 kpc^-1]
    """
    phi = np.asarray(phi, dtype=float)
    rv = np.asarray(rv, dtype=float)
    zv = np.asarray(zv, dtype=float)
    kz = np.zeros_like(phi)
    for R_unique in np.unique(rv):
        mask = np.abs(rv - R_unique) < 1e-8
        z_grp = zv[mask]
        phi_grp = phi[mask]
        order = np.argsort(z_grp)
        rev = np.argsort(order)
        kz_grp = np.gradient(phi_grp[order], z_grp[order])
        where = np.where(mask)[0]
        kz[where] = kz_grp[rev]
    return kz


def chi2_radial(
    vc_model: np.ndarray,
    r_model: np.ndarray,
    rr: np.ndarray,
    vv: np.ndarray,
    ss: np.ndarray,
) -> float:
    """Radial chi-squared.

    Parameters
    ----------
    vc_model : model circular velocity on r_model grid  [km/s]
    r_model  : radii for vc_model  [kpc]
    rr       : observed radii  [kpc]
    vv       : observed circular velocities  [km/s]
    ss       : total velocity uncertainties  [km/s]
    """
    vc_at_obs = np.interp(rr, r_model, vc_model)
    return float(np.sum(((vc_at_obs - vv) / ss) ** 2))


def chi2_vertical(
    phi_model: np.ndarray,
    rv_model: np.ndarray,
    zv_model: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
    phi_obs: np.ndarray,
    sig_phi: np.ndarray,
    sig_z: np.ndarray,
) -> float:
    """Vertical chi-squared with Kz-corrected sigma_eff.

    Parameters
    ----------
    phi_model         : predicted Φ(R,z) - Φ(R,0)  [(km/s)^2]
    rv_model, zv_model: (R,z) coordinates for phi_model
    rv_obs, zv_obs    : observed (R,z) points
    phi_obs           : observed Φ  [(km/s)^2]
    sig_phi           : uncertainty on Φ  [(km/s)^2]
    sig_z             : uncertainty on z height  [kpc]
    """
    phi_at_obs = _match_rz(phi_model, rv_model, zv_model, rv_obs, zv_obs)
    kz = vertical_force_from_phi(phi_at_obs, rv_obs, zv_obs)
    sig_eff = np.sqrt(sig_phi ** 2 + (kz * sig_z) ** 2)
    return float(np.sum(((phi_at_obs - phi_obs) / sig_eff) ** 2))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles,
) -> np.ndarray:
    """Weighted quantiles via sorted-CDF interpolation.

    Parameters
    ----------
    values    : 1-D array of values
    weights   : 1-D non-negative weights (need not sum to 1)
    quantiles : scalar or sequence of percentile levels in [0, 100]

    Returns
    -------
    np.ndarray of same length as quantiles
    """
    sorter = np.argsort(values)
    sv = values[sorter]
    sw = weights[sorter]
    cdf = np.cumsum(sw)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles) / 100.0, cdf, sv)


def chi2_nu(
    chi2_total: float,
    k: int,
    n_primary: int = N_PRIMARY,
) -> float:
    """Reduced chi-squared: chi2_total / (n_primary - k)."""
    return chi2_total / max(n_primary - k, 1)


def _match_rz(
    phi_model: np.ndarray,
    rv_model: np.ndarray,
    zv_model: np.ndarray,
    rv_obs: np.ndarray,
    zv_obs: np.ndarray,
) -> np.ndarray:
    """Match model phi to obs (R,z) points by exact float lookup or nearest."""
    phi_out = np.empty(len(rv_obs), dtype=float)
    for k, (R, z) in enumerate(zip(rv_obs, zv_obs)):
        mask = (np.abs(rv_model - R) < 1e-8) & (np.abs(zv_model - z) < 1e-8)
        if mask.any():
            phi_out[k] = phi_model[mask][0]
        else:
            dist = (rv_model - R) ** 2 + (zv_model - z) ** 2
            phi_out[k] = phi_model[np.argmin(dist)]
    return phi_out
