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
    sigma_floor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_kpc, vc_kms, sigma_kms) arrays for the chi^2 fitting subset.

    When chi2_catalog_path is supplied (recommended), loads all 152
    kind=direct_rotation / used_in_fit=true rows from the observational
    catalog and uses their sigma_v_kms (floored at sigma_floor=3 km/s).
    This matches the analysis in Monjo & Banik 2026.

    Without chi2_catalog_path, falls back to the Wang+2026 34-row CSV
    using sigma_total_kms (suitable for quick checks only).

    Returns
    -------
    rr : [kpc], vv : [km/s], ss : [km/s]
    """
    if chi2_catalog_path is not None and Path(chi2_catalog_path).exists():
        rr_list, vv_list, ss_list = [], [], []
        with open(chi2_catalog_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("kind", "") == "direct_rotation"
                        and row.get("used_in_fit", "").lower() == "true"
                        and row.get("R_kpc", "").strip()
                        and row.get("vc_kms", "").strip()
                        and row.get("sigma_v_kms", "").strip()):
                    rr_list.append(float(row["R_kpc"]))
                    vv_list.append(float(row["vc_kms"]))
                    ss_list.append(max(float(row["sigma_v_kms"]), sigma_floor))
        if rr_list:
            order = np.argsort(rr_list)
            return (np.array(rr_list)[order],
                    np.array(vv_list)[order],
                    np.array(ss_list)[order])

    # Fallback: Wang+2026 34-row CSV
    if rot is None:
        rot = load_rotation_curve(rot_path)
    rr = np.array([r["R_kpc"] for r in rot])
    vv = np.array([r["vc_kms"] for r in rot])
    ss = np.maximum(np.array([r["sigma_total_kms"] for r in rot]), sigma_floor)
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
