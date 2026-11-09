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
    phi_sys_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rv, zv, phi_obs, sig_phi, sig_z) arrays for all 44 vertical points.

    phi_sys_fraction adds a systematic error phi_sys_fraction * Phi in quadrature
    (Wang et al. 2026 adopt 0.05).
    """
    if vert is None:
        vert = load_vertical_potential(vert_path)
    rv      = np.array([r["R_kpc"]          for r in vert])
    zv      = np.array([r["z_kpc"]          for r in vert])
    phi_obs = np.array([r["Phi_kms2"]        for r in vert])
    sig_phi = np.array([r["sigma_Phi_kms2"]  for r in vert])
    sig_z   = np.array([r["sigma_z_kpc"]     for r in vert])
    sig_phi = np.sqrt(sig_phi ** 2 + (phi_sys_fraction * phi_obs) ** 2)
    return rv, zv, phi_obs, sig_phi, sig_z


WANG_DATASET = "Wang2026_rotation_curve"
EILERS_DATASET = "Eilers2019_McGaugh2019_adjusted"
RADIAL_VARIANTS = ("all", "no_eilers", "no_wang", "overlap_correlated", "wang_only")


def wang_systematic_fraction(R: np.ndarray, rot_path: Optional[Path] = None) -> np.ndarray:
    """Fractional systematic error of the Wang et al. (2026) rotation curve at R.

    Recovered from the published budget as sqrt(sigma_total^2 - sigma_obs^2)/v_c,
    interpolated in R and clipped to the 5-20 per cent range quoted by Wang et al.
    """
    rot = load_rotation_curve(rot_path)
    r_w = np.array([r["R_kpc"] for r in rot])
    frac = np.array([np.sqrt(max(r["sigma_total_kms"] ** 2 - r["sigma_obs_kms"] ** 2, 0.0)) / r["vc_kms"]
                     for r in rot])
    order = np.argsort(r_w)
    return np.clip(np.interp(R, r_w[order], frac[order]), 0.05, 0.20)


def radial_variant_arrays(
    variant: str,
    chi2_catalog_path: Path,
    add_systematics: bool = True,
    sigma_floor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial fit arrays for one data-vector variant.

    all                 the 152 points of the catalogue
    no_eilers           without the Eilers/McGaugh curve (overlaps the Wang curve)
    no_wang             without the Wang curve
    overlap_correlated  all points, with the Wang and Eilers errors multiplied by
                        sqrt(2) where both curves cover the same radii, which is
                        the weight of two fully correlated measurements
    wang_only           the 34 Wang points

    With add_systematics, points outside the Wang curve receive the Wang
    systematic fraction at their radius in quadrature; the Wang errors already
    include it.
    """
    if variant not in RADIAL_VARIANTS:
        raise ValueError(f"unknown radial variant {variant!r}")
    rows = []
    with open(chi2_catalog_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("kind", "") == "direct_rotation"
                    and row.get("used_in_fit", "").lower() == "true"):
                rows.append(row)
    if variant == "no_eilers":
        rows = [r for r in rows if r["dataset"] != EILERS_DATASET]
    elif variant == "no_wang":
        rows = [r for r in rows if r["dataset"] != WANG_DATASET]
    elif variant == "wang_only":
        rows = [r for r in rows if r["dataset"] == WANG_DATASET]
    rr = np.array([float(r["R_kpc"]) for r in rows])
    vv = np.array([float(r["vc_kms"]) for r in rows])
    ss = np.maximum(np.array([float(r["sigma_v_kms"]) for r in rows]), sigma_floor)
    dataset = np.array([r["dataset"] for r in rows])
    if add_systematics:
        external = dataset != WANG_DATASET
        ss[external] = np.hypot(ss[external], wang_systematic_fraction(rr[external]) * vv[external])
    if variant == "overlap_correlated":
        in_w, in_e = dataset == WANG_DATASET, dataset == EILERS_DATASET
        lo = max(rr[in_w].min(), rr[in_e].min())
        hi = min(rr[in_w].max(), rr[in_e].max())
        overlap = (in_w | in_e) & (rr >= lo) & (rr <= hi)
        ss[overlap] *= np.sqrt(2.0)
    order = np.argsort(rr)
    return rr[order], vv[order], ss[order]
