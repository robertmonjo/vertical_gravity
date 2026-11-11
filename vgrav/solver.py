"""Cylindrical axisymmetric Poisson solver.

Solves (∇² − μ²) φ = rhs on a uniform (R, z) grid with Dirichlet boundary
conditions.  The operator is ∂²/∂R² + (1/R)∂/∂R + ∂²/∂z².

Typical usage
-------------
>>> grid = make_grid(r_min=0.0, r_max=40.0, z_max=20.0, nR=121, nz=121)
>>> phi  = solve_axisymmetric(grid, 4*pi*G*rho, boundary, mu=0.0)
>>> gR, gz = gradients(grid, phi)
>>> phi_obs = interp2(grid, phi, R_obs, z_obs)

High-level helpers
------------------
>>> boundary = monopole_boundary(grid, mass)
>>> mass, phi = solve_newtonian(grid, rho)
>>> vc = radial_speed(grid, phi, radii)
>>> phi_diff = phi_difference(grid, phi, rv, zv)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from vgrav._constants import G


@dataclass
class CylGrid:
    R: np.ndarray
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.R.size, self.z.size)

    @property
    def dR(self) -> float:
        return float(self.R[1] - self.R[0])

    @property
    def dz(self) -> float:
        return float(self.z[1] - self.z[0])


def make_grid(
    r_min: float = 0.0,
    r_max: float = 40.0,
    z_max: float = 20.0,
    nR: int = 121,
    nz: int = 121,
) -> CylGrid:
    """Create a uniform cylindrical grid."""
    return CylGrid(
        R=np.linspace(r_min, r_max, nR),
        z=np.linspace(-z_max, z_max, nz),
    )


def solve_axisymmetric(
    grid: CylGrid,
    rhs: np.ndarray,
    boundary: np.ndarray,
    mu: float = 0.0,
) -> np.ndarray:
    """Solve (∇² − μ²) φ = rhs on the cylindrical grid.

    Parameters
    ----------
    grid     : CylGrid with uniform R and z spacing.
    rhs      : source term array of shape grid.shape.
    boundary : Dirichlet boundary values, shape grid.shape.
    mu       : Helmholtz screening length^-1 (0 for Poisson).

    Returns
    -------
    phi : solution array of shape grid.shape.
    """
    nR, nz = grid.shape
    dR2 = grid.dR * grid.dR
    dz2 = grid.dz * grid.dz
    n = nR * nz
    mat = lil_matrix((n, n), dtype=float)
    vec = rhs.reshape(n).astype(float).copy()

    def idx(i: int, j: int) -> int:
        return i * nz + j

    for i, R in enumerate(grid.R):
        for j, _z in enumerate(grid.z):
            row = idx(i, j)
            if i == nR - 1 or j == 0 or j == nz - 1 or (i == 0 and R > 0.0):
                mat[row, row] = 1.0
                vec[row] = boundary[i, j]
                continue

            if i == 0:
                # Symmetry axis: ∂²φ/∂R² + (1/R)∂φ/∂R → 2∂²φ/∂R²
                mat[row, idx(i + 1, j)] = 4.0 / dR2
                mat[row, idx(i, j + 1)] = 1.0 / dz2
                mat[row, idx(i, j - 1)] = 1.0 / dz2
                mat[row, row] = -4.0 / dR2 - 2.0 / dz2 - mu * mu
                continue

            rp = 1.0 / dR2 + 1.0 / (2.0 * R * grid.dR)
            rm = 1.0 / dR2 - 1.0 / (2.0 * R * grid.dR)
            zp = 1.0 / dz2
            zm = 1.0 / dz2
            cc = -2.0 / dR2 - 2.0 / dz2 - mu * mu

            mat[row, idx(i + 1, j)] = rp
            mat[row, idx(i - 1, j)] = rm
            mat[row, idx(i, j + 1)] = zp
            mat[row, idx(i, j - 1)] = zm
            mat[row, row] = cc

    sol = spsolve(mat.tocsr(), vec)
    return sol.reshape((nR, nz))


def axisymmetric_operator(
    grid: CylGrid,
    eps: np.ndarray | None = None,
    mu: float = 0.0,
):
    """Factorised operator for ∇·(ε∇φ) − μ²φ = rhs with Dirichlet boundaries.

    The divergence is written in conservative form,
        (1/R) ∂_R(R ε ∂_R φ) + ∂_z(ε ∂_z φ),
    with ε at the cell faces taken as the mean of the two adjacent nodes, so that
    the flux of ε∇φ is continuous across the variations of ε. For ε ≡ 1 the
    stencil is that of solve_axisymmetric. The grid must start at R = 0.

    Parameters
    ----------
    grid : CylGrid with uniform spacing and R[0] = 0.
    eps  : coefficient on the grid nodes (default 1).
    mu   : screening constant [1/kpc] (0 for Poisson).

    Returns
    -------
    solve(rhs, boundary) -> phi, both of shape grid.shape. The boundary rows are
    the outer radius, the axis-free edges z = ±z_max, as in solve_axisymmetric.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    nR, nz = grid.shape
    if grid.R[0] != 0.0:
        raise ValueError("axisymmetric_operator requires a grid starting at R = 0")
    dR, dz = grid.dR, grid.dz
    e = np.ones((nR, nz)) if eps is None else np.asarray(eps, dtype=float)
    e_r = 0.5 * (e[1:, :] + e[:-1, :])          # faces (i+1/2, j), shape (nR-1, nz)
    e_z = 0.5 * (e[:, 1:] + e[:, :-1])          # faces (i, j+1/2), shape (nR, nz-1)
    R = grid.R
    idx = np.arange(nR * nz).reshape(nR, nz)

    boundary = np.zeros((nR, nz), dtype=bool)
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    interior = ~boundary

    rows, cols, vals = [], [], []

    def add(mask, col_idx, values):
        rows.append(idx[mask])
        cols.append(col_idx[mask])
        vals.append(values[mask])

    # vertical fluxes (all interior nodes, including the axis)
    cz_p = np.zeros((nR, nz))
    cz_m = np.zeros((nR, nz))
    cz_p[:, :-1] = e_z / dz ** 2
    cz_m[:, 1:] = e_z / dz ** 2
    # radial fluxes
    cr_p = np.zeros((nR, nz))
    cr_m = np.zeros((nR, nz))
    Ri = R[1:-1, None]
    cr_p[1:-1, :] = (Ri + 0.5 * dR) * e_r[1:, :] / (Ri * dR ** 2)
    cr_m[1:-1, :] = (Ri - 0.5 * dR) * e_r[:-1, :] / (Ri * dR ** 2)
    cr_p[0, :] = 4.0 * e_r[0, :] / dR ** 2      # axis: flux through the cylinder of radius dR/2

    shift_rp = np.zeros_like(idx); shift_rp[:-1] = idx[1:]
    shift_rm = np.zeros_like(idx); shift_rm[1:] = idx[:-1]
    shift_zp = np.zeros_like(idx); shift_zp[:, :-1] = idx[:, 1:]
    shift_zm = np.zeros_like(idx); shift_zm[:, 1:] = idx[:, :-1]

    diag = -(cr_p + cr_m + cz_p + cz_m) - mu * mu
    add(interior, idx, diag)
    add(interior, shift_rp, cr_p)
    add(interior & (np.arange(nR)[:, None] > 0), shift_rm, cr_m)
    add(interior, shift_zp, cz_p)
    add(interior, shift_zm, cz_m)
    add(boundary, idx, np.ones((nR, nz)))

    n = nR * nz
    mat = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    lu = splu(mat)
    bmask = boundary.ravel()

    def solve(rhs: np.ndarray, boundary_values: np.ndarray) -> np.ndarray:
        vec = np.asarray(rhs, dtype=float).ravel().copy()
        vec[bmask] = np.asarray(boundary_values, dtype=float).ravel()[bmask]
        return lu.solve(vec).reshape((nR, nz))

    return solve


def yukawa_boundary(grid: CylGrid, mass: float, mu: float) -> np.ndarray:
    """Dirichlet boundary values φ = −GM e^{−μr}/r (screened monopole)."""
    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    rr = np.maximum(np.sqrt(RR * RR + ZZ * ZZ), 1e-6)
    return -G * mass * np.exp(-mu * rr) / rr


def gradients(grid: CylGrid, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (∂φ/∂R, ∂φ/∂z) on the grid using second-order finite differences."""
    dR = np.gradient(phi, grid.dR, axis=0, edge_order=2)
    dz = np.gradient(phi, grid.dz, axis=1, edge_order=2)
    return dR, dz


def cylindrical_divergence(
    grid: CylGrid,
    FR: np.ndarray,
    Fz: np.ndarray,
) -> np.ndarray:
    """Compute (1/R) ∂(R FR)/∂R + ∂Fz/∂z on the grid."""
    R = grid.R[:, None]
    div_R = np.empty_like(FR, dtype=float)
    if grid.R[0] == 0.0:
        div_R[1:] = np.gradient(R * FR, grid.dR, axis=0, edge_order=2)[1:] / R[1:]
        div_R[0] = 2.0 * (FR[1] - FR[0]) / grid.dR
    else:
        div_R[:] = np.gradient(R * FR, grid.dR, axis=0, edge_order=2) / R
    div_z = np.gradient(Fz, grid.dz, axis=1, edge_order=2)
    return div_R + div_z


def interp2(
    grid: CylGrid,
    values: np.ndarray,
    Rq: np.ndarray,
    zq: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of grid *values* at query points (Rq, zq)."""
    Rq = np.asarray(Rq, dtype=float)
    zq = np.asarray(zq, dtype=float)
    out = np.empty_like(Rq, dtype=float)
    for k, (R, z) in enumerate(zip(Rq, zq)):
        i = int(np.searchsorted(grid.R, R) - 1)
        j = int(np.searchsorted(grid.z, z) - 1)
        i = max(0, min(i, grid.R.size - 2))
        j = max(0, min(j, grid.z.size - 2))
        R0, R1 = grid.R[i], grid.R[i + 1]
        z0, z1 = grid.z[j], grid.z[j + 1]
        t = (R - R0) / (R1 - R0)
        u = (z - z0) / (z1 - z0)
        out[k] = (
            (1 - t) * (1 - u) * values[i, j]
            + t * (1 - u) * values[i + 1, j]
            + (1 - t) * u * values[i, j + 1]
            + t * u * values[i + 1, j + 1]
        )
    return out


# ── High-level Newtonian helpers ──────────────────────────────────────────────

def monopole_boundary(grid: CylGrid, mass: float) -> np.ndarray:
    """Dirichlet boundary values φ = −GM/r (Newtonian monopole)."""
    RR, ZZ = np.meshgrid(grid.R, grid.z, indexing="ij")
    rr = np.sqrt(RR * RR + ZZ * ZZ)
    return -G * mass / np.maximum(rr, 1e-6)


def solve_newtonian(grid: CylGrid, rho: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve the Newtonian Poisson equation ∇²φ = 4πG ρ.

    Parameters
    ----------
    grid : CylGrid
    rho  : density [Msun/kpc^3], shape grid.shape

    Returns
    -------
    mass : total enclosed mass [Msun]
    phi  : gravitational potential [kpc^2/(km/s)^2 ≡ (km/s)^2], shape grid.shape
    """
    mass = float(np.sum(rho * (2.0 * math.pi * grid.R[:, None] * grid.dR * grid.dz)))
    boundary = monopole_boundary(grid, mass)
    phi = solve_axisymmetric(grid, 4.0 * math.pi * G * rho, boundary)
    return mass, phi


def radial_speed(grid: CylGrid, phi: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Circular speed v_c(R) = sqrt(R * ∂φ/∂R) at z=0.

    Parameters
    ----------
    grid   : CylGrid
    phi    : potential on grid, shape grid.shape
    radii  : evaluation radii [kpc]

    Returns
    -------
    vc : circular speed [km/s]
    """
    dphi_dR, _ = gradients(grid, phi)
    z0 = np.zeros_like(np.asarray(radii, dtype=float))
    dphidR_mid = interp2(grid, dphi_dR, np.asarray(radii, dtype=float), z0)
    return np.sqrt(np.maximum(np.asarray(radii, dtype=float) * dphidR_mid, 0.0))


def radial_v2(grid: CylGrid, phi: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return R * ∂φ/∂R at z=0 (= v_c²) without the sqrt."""
    dphi_dR, _ = gradients(grid, phi)
    z0 = np.zeros_like(np.asarray(radii, dtype=float))
    return np.asarray(radii, dtype=float) * interp2(grid, dphi_dR, np.asarray(radii, dtype=float), z0)


def phi_difference(
    grid: CylGrid,
    phi: np.ndarray,
    rv: np.ndarray,
    zv: np.ndarray,
) -> np.ndarray:
    """Return φ(R,z) − φ(R,0) at observation points [(km/s)²].

    Parameters
    ----------
    grid : CylGrid
    phi  : potential on grid, shape grid.shape
    rv   : R coordinates of obs points [kpc]
    zv   : z coordinates of obs points [kpc]
    """
    rv = np.asarray(rv, dtype=float)
    zv = np.asarray(zv, dtype=float)
    return interp2(grid, phi, rv, zv) - interp2(grid, phi, rv, np.zeros_like(zv))


def blend_outer(
    radii: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    r_blend_start: float = 55.0,
    r_solve_max: float = 60.0,
) -> np.ndarray:
    """Smoothly transition from inner (solver) to outer (monopole) solution.

    Beyond r_solve_max the cylindrical Poisson solver is inaccurate because
    the boundary condition dominates.  This blends between the two solutions
    over [r_blend_start, r_solve_max] using a smooth cubic.
    """
    result = np.asarray(inner, dtype=float).copy()
    outer_mask = radii >= r_solve_max
    result[outer_mask] = np.asarray(outer, dtype=float)[outer_mask]
    mid = (radii > r_blend_start) & (radii < r_solve_max)
    t = (radii[mid] - r_blend_start) / (r_solve_max - r_blend_start)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    result[mid] = (1.0 - smooth) * result[mid] + smooth * np.asarray(outer, dtype=float)[mid]
    return result
