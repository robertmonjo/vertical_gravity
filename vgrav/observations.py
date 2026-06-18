"""Load Wang+2026 rotation curve and vertical potential data.

Data files are bundled in the package's ../data/ directory:
  wang2026_rotation_curve.csv
      columns: R_kpc, vc_kms, sigma_obs_kms, sigma_total_kms

  wang2026_vertical_potential.csv
      columns: R_kpc, z_kpc, sigma_z_kpc, Phi_kms2, sigma_Phi_kms2

The fit-constraint subset (chi^2 fitting) excludes model-dependent
outer-halo points and uses sigma_total_kms for the uncertainty.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"


def _csv_path(filename: str, override: Optional[Path]) -> Path:
    if override is not None:
        return Path(override)
    p = _DATA_DIR / filename
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Copy it from the project's data/ folder "
            "or pass an explicit path."
        )
    return p


def load_rotation_curve(path: Optional[Path] = None) -> list[dict]:
    """Return Wang+2026 rotation curve as a list of dicts.

    Keys: R_kpc, vc_kms, sigma_obs_kms, sigma_total_kms
    """
    with open(_csv_path("wang2026_rotation_curve.csv", path), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def load_vertical_potential(path: Optional[Path] = None) -> list[dict]:
    """Return Wang+2026 vertical potential data as a list of dicts.

    Keys: R_kpc, z_kpc, sigma_z_kpc, Phi_kms2, sigma_Phi_kms2
    Sorted by (R_kpc, z_kpc).
    """
    with open(_csv_path("wang2026_vertical_potential.csv", path), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    data = [{k: float(v) for k, v in row.items()} for row in rows]
    return sorted(data, key=lambda r: (r["R_kpc"], r["z_kpc"]))


def load_observations(
    rot_path: Optional[Path] = None,
    vert_path: Optional[Path] = None,
) -> tuple[list[dict], list[dict]]:
    """Return (rotation_data, vertical_data) as lists of dicts."""
    return load_rotation_curve(rot_path), load_vertical_potential(vert_path)


def radial_fit_arrays(
    rot: Optional[list[dict]] = None,
    rot_path: Optional[Path] = None,
    chi2_catalog_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_kpc, vc_kms, sigma_total_kms) arrays for the chi^2 fitting subset.

    The fitting subset excludes model-dependent outer-disc points flagged in
    the chi^2 catalog (fig2_observational_data.csv).  Falls back to all
    rotation-curve rows when no catalog is available.

    Returns
    -------
    rr : [kpc], vv : [km/s], ss : [km/s]
    """
    if rot is None:
        rot = load_rotation_curve(rot_path)

    if chi2_catalog_path is not None and Path(chi2_catalog_path).exists():
        fit_radii: set[float] = set()
        with open(chi2_catalog_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("type", "") == "radial" and row.get("in_chi2_fit", "").lower() == "true":
                    fit_radii.add(float(row["R_kpc"]))
        selected = [r for r in rot if r["R_kpc"] in fit_radii]
        if selected:
            rot = selected

    rr = np.array([r["R_kpc"] for r in rot])
    vv = np.array([r["vc_kms"] for r in rot])
    ss = np.array([r["sigma_total_kms"] for r in rot])
    return rr, vv, ss


def vertical_arrays(
    vert: Optional[list[dict]] = None,
    vert_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rv, zv, phi_obs, sig_phi, sig_z) arrays for all 44 vertical points."""
    if vert is None:
        vert = load_vertical_potential(vert_path)
    rv      = np.array([r["R_kpc"]          for r in vert])
    zv      = np.array([r["z_kpc"]          for r in vert])
    phi_obs = np.array([r["Phi_kms2"]        for r in vert])
    sig_phi = np.array([r["sigma_Phi_kms2"]  for r in vert])
    sig_z   = np.array([r["sigma_z_kpc"]     for r in vert])
    return rv, zv, phi_obs, sig_phi, sig_z
