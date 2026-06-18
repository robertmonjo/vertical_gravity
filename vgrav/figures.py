"""Figure renderers for the vertical gravity benchmark.

All renderers read from the pre-computed CSV files in outputs/.
They reproduce the published figures without re-running any heavy computation.

Functions
---------
plot_fig2(outputs_dir, obs_path=None, output_path=None, dpi=220)
    Reproduce Fig. 2: radial rotation curve (top) + 9 vertical panels (bottom).

print_table2(outputs_dir)
    Print Table 2 (chi^2_nu summary) to stdout.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# ── Model display config ───────────────────────────────────────────────────────
# key → (display_name, color, linestyle, linewidth)
_MODEL_STYLES: dict[str, tuple] = {
    "baryonic":           ("Baryonic (Newtonian)",   "#6baed6", "--", 1.8),
    "qumond_simple":      ("QUMOND simple",           "#1a9850", "-",  1.8),
    "qumond_standard":    ("QUMOND standard",         "#91cf60", "-",  1.8),
    "qumond_mls":         ("QUMOND MLS/RAR",          "#d9ef8b", "-",  1.8),
    "stvg":               ("STVG",                    "#9467bd", "-",  1.8),
    "cdm_nfw":            ("CDM NFW",                 "#ff7f0e", "--", 2.2),
    "cdm_einasto":        ("CDM Einasto",             "#d62728", "--", 2.2),
    "hmg_k1":             ("HMG anisotropic",         "#111111", "-",  2.2),
    "fr_screened":        ("f(R) screened",           "#4393c3", "-",  1.8),
    "refracted_gravity":  ("Refracted Gravity",       "#74c476", "-",  1.8),
    "emergent_gravity":   ("Emergent Gravity (fixed)","#fdae61", ":",  1.8),
}
_BARY_COLOR = "#6baed6"


def _read_draw_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a model CSV with columns [R_kpc or (R_kpc,z_kpc), b1..b100, chi2_* last row].

    Returns (coords, draw_matrix, chi2_array).
    coords is a 1-D R array for radial CSVs, a (N,2) array for vertical.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            rows.append(row)

    has_z = "z_kpc" in header
    draw_cols = [i for i, h in enumerate(header) if h.startswith("b")]
    chi2_row = rows[-1]
    data_rows = rows[:-1]

    if has_z:
        R_col = header.index("R_kpc")
        z_col = header.index("z_kpc")
        coords = np.array([[float(r[R_col]), float(r[z_col])] for r in data_rows])
    else:
        R_col = header.index("R_kpc")
        coords = np.array([float(r[R_col]) for r in data_rows])

    draws = np.array([[float(r[c]) for c in draw_cols] for r in data_rows])
    chi2 = np.array([float(chi2_row[c]) for c in draw_cols])
    return coords, draws, chi2


def _pct(arr: np.ndarray, p: float, axis: int = 1) -> np.ndarray:
    return np.percentile(arr, p, axis=axis)


def _to_float(v: str):
    """Convert string to float; return None for empty strings."""
    s = v.strip()
    return float(s) if s else None


def _load_obs(obs_path: Optional[Path]) -> tuple[list[dict], list[dict]]:
    """Load observational data from fig2_observational_data.csv.

    Returns (rot_rows, vert_rows). Each row is a dict with numeric values
    where available; empty cells become None. The 'type', 'dataset',
    'source_paper', 'tracer', 'method', and 'in_chi2_fit' keys stay as str.
    """
    _STR_KEYS = {"type", "dataset", "source_paper", "tracer", "method", "in_chi2_fit"}
    if obs_path is None or not Path(obs_path).exists():
        return [], []
    rot_rows, vert_rows = [], []
    with open(obs_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed = {k: (v if k in _STR_KEYS else _to_float(v)) for k, v in row.items()}
            if row.get("type") == "radial":
                rot_rows.append(parsed)
            elif row.get("type") == "vertical":
                vert_rows.append(parsed)
    return rot_rows, vert_rows


def plot_fig2(
    outputs_dir: Path,
    obs_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    dpi: int = 220,
    model_keys: Optional[list[str]] = None,
) -> plt.Figure:
    """Reproduce Fig. 2: rotation curve + 9 vertical potential panels.

    Parameters
    ----------
    outputs_dir : directory containing model_*_radial.csv and _vertical.csv,
                  mc100_baryonic_*.csv, and optionally fig2_observational_data.csv.
    obs_path    : explicit path to fig2_observational_data.csv (else auto-detected).
    output_path : save figure here if given (e.g. "fig2.png").
    dpi         : figure resolution.
    model_keys  : subset of model keys to plot (default: all available).

    Returns
    -------
    matplotlib Figure object.
    """
    outputs_dir = Path(outputs_dir)
    if obs_path is None:
        obs_path = outputs_dir / "fig2_observational_data.csv"

    rot_obs, vert_obs = _load_obs(obs_path)

    # ── Load all model CSVs ────────────────────────────────────────────────────
    available: dict[str, dict] = {}
    for key in _MODEL_STYLES:
        rad_p = outputs_dir / f"model_{key}_radial.csv"
        vert_p = outputs_dir / f"model_{key}_vertical.csv"
        if rad_p.exists() and vert_p.exists():
            r_coords, r_draws, r_chi2 = _read_draw_csv(rad_p)
            v_coords, v_draws, v_chi2 = _read_draw_csv(vert_p)
            available[key] = {
                "r_coords": r_coords, "r_draws": r_draws, "r_chi2": r_chi2,
                "v_coords": v_coords, "v_draws": v_draws, "v_chi2": v_chi2,
            }

    # Baryonic band from mc100 CSVs
    bary_rad_p = outputs_dir / "mc100_baryonic_radial.csv"
    bary_vert_p = outputs_dir / "mc100_baryonic_vertical.csv"
    bary_r, bary_r_draws = None, None
    bary_v_coords, bary_v_draws = None, None
    if bary_rad_p.exists():
        bary_r, bary_r_draws, _ = _read_draw_csv(bary_rad_p)
    if bary_vert_p.exists():
        bary_v_coords, bary_v_draws, _ = _read_draw_csv(bary_vert_p)

    if model_keys is not None:
        available = {k: v for k, v in available.items() if k in model_keys}

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 11.0, "axes.labelsize": 12.0,
        "xtick.labelsize": 10.0, "ytick.labelsize": 10.0,
        "legend.fontsize": 8.5,
    })
    fig = plt.figure(figsize=(12.8, 12.1))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.26, 0.10, 1.0],
        hspace=0.035, left=0.07, right=0.985, bottom=0.055, top=0.87,
    )
    gs_vert = gs[2].subgridspec(3, 3, hspace=0.32, wspace=0.12)
    ax_top = fig.add_subplot(gs[0])

    # ── Radial panel ──────────────────────────────────────────────────────────
    if bary_r is not None and bary_r_draws is not None:
        sm = lambda v: gaussian_filter1d(v, sigma=6)
        ax_top.fill_between(
            bary_r, sm(_pct(bary_r_draws, 5)), sm(_pct(bary_r_draws, 95)),
            color=_BARY_COLOR, alpha=0.08, lw=0, zorder=0,
        )
        ax_top.fill_between(
            bary_r, sm(_pct(bary_r_draws, 16)), sm(_pct(bary_r_draws, 84)),
            color=_BARY_COLOR, alpha=0.20, lw=0, zorder=0.05,
            label=r"Baryonic MC100 $p_{16}$–$p_{84}$",
        )

    for key, dat in available.items():
        label, color, ls, lw = _MODEL_STYLES[key]
        med = gaussian_filter1d(_pct(dat["r_draws"], 50), sigma=4)
        chi2_med = float(np.median(dat["r_chi2"] + available.get("baryonic", dat)["r_chi2"] * 0))
        ax_top.fill_between(
            dat["r_coords"],
            gaussian_filter1d(_pct(dat["r_draws"], 16), sigma=4),
            gaussian_filter1d(_pct(dat["r_draws"], 84), sigma=4),
            color=color, alpha=0.08, lw=0,
        )
        ax_top.plot(dat["r_coords"], med, color=color, ls=ls, lw=lw,
                    label=label, zorder=3)

    if rot_obs:
        # model-dependent context points (grey, background, not used in chi2)
        dep = [r for r in rot_obs if r["in_chi2_fit"].strip().lower() != "true"
               and r.get("vc_kms") is not None and r.get("sigma_vc_kms") is not None]
        if dep:
            ax_top.errorbar(
                [r["R_kpc"] for r in dep], [r["vc_kms"] for r in dep],
                yerr=[r["sigma_vc_kms"] for r in dep],
                fmt="o", ms=2.2, color="#aaaaaa", ecolor="#cccccc",
                lw=0.5, zorder=2, alpha=0.55, label="Other surveys (context)",
            )
        # model-independent fit points (black, foreground, enter chi2)
        fit = [r for r in rot_obs if r["in_chi2_fit"].strip().lower() == "true"
               and r.get("vc_kms") is not None and r.get("sigma_vc_kms") is not None]
        if fit:
            ax_top.errorbar(
                [r["R_kpc"] for r in fit], [r["vc_kms"] for r in fit],
                yerr=[r["sigma_vc_kms"] for r in fit],
                fmt="o", ms=3.5, color="black", ecolor="black",
                lw=0.8, zorder=10, label="Observations (fit)",
            )

    ax_top.set_xscale("symlog", linthresh=2.0)
    ax_top.set_xlim(0.5, 250.0)
    ax_top.set_ylim(100, 310)
    ax_top.set_xlabel(r"$R$ [kpc]")
    ax_top.set_ylabel(r"$v_c$ [km s$^{-1}$]")
    ax_top.legend(loc="upper right", ncol=2, fontsize=7.5, frameon=True)
    fig.add_subplot(gs[1]).axis("off")

    # ── Vertical panels ────────────────────────────────────────────────────────
    if bary_v_coords is not None:
        all_R_vert = sorted(set(bary_v_coords[:, 0]))
        radii_9 = all_R_vert[:9]
    elif available:
        first_key = next(iter(available))
        all_R_vert = sorted(set(available[first_key]["v_coords"][:, 0]))
        radii_9 = all_R_vert[:9]
    else:
        radii_9 = []

    for idx_p, R in enumerate(radii_9):
        axp = fig.add_subplot(gs_vert[idx_p // 3, idx_p % 3])

        # Observational data
        if vert_obs:
            obs_R = [v for v in vert_obs if abs(v["R_kpc"] - R) < 0.05]
            if obs_R:
                zobs = np.array([v["z_kpc"] for v in obs_R])
                phio = np.array([v["Phi_kms2"] for v in obs_R])
                sigo = np.array([v["sigma_Phi_kms2"] for v in obs_R])
                sig_z = np.array([v.get("sigma_z_kpc", 0.0) for v in obs_R])
                order_obs = np.argsort(zobs)
                axp.fill_between(zobs[order_obs], (phio - sigo)[order_obs],
                                 (phio + sigo)[order_obs], color="#ff9999", alpha=0.28, lw=0)
                axp.errorbar(zobs, phio, yerr=sigo, fmt="o", ms=3.0,
                             color="black", ecolor="black", elinewidth=0.9,
                             capsize=2.4, capthick=0.9, zorder=8)

        # Baryonic band
        if bary_v_coords is not None:
            mask_b = np.abs(bary_v_coords[:, 0] - R) < 0.05
            if mask_b.any():
                zb = bary_v_coords[mask_b, 1]
                order_b = np.argsort(zb)
                zb_s, b16, b50, b84 = (
                    zb[order_b],
                    _pct(bary_v_draws[mask_b], 16)[order_b],
                    _pct(bary_v_draws[mask_b], 50)[order_b],
                    _pct(bary_v_draws[mask_b], 84)[order_b],
                )
                def _dense(zv, phi):
                    zd = np.linspace(0.0, 1.1, 80)
                    return zd, PchipInterpolator(np.r_[0.0, zv], np.r_[0.0, phi], extrapolate=True)(zd)
                zd, b16d = _dense(zb_s, b16)
                _,  b50d = _dense(zb_s, b50)
                _,  b84d = _dense(zb_s, b84)
                axp.fill_between(zd, b16d, b84d, color=_BARY_COLOR, alpha=0.18, lw=0)
                axp.plot(zd, b50d, color=_BARY_COLOR, lw=2.8, alpha=0.35)

        # Model curves
        for key, dat in available.items():
            label, color, ls, lw = _MODEL_STYLES[key]
            mask_m = np.abs(dat["v_coords"][:, 0] - R) < 0.05
            if not mask_m.any():
                continue
            zm = dat["v_coords"][mask_m, 1]
            order_m = np.argsort(zm)
            zm_s = zm[order_m]
            phi50 = _pct(dat["v_draws"][mask_m], 50)[order_m]
            phi16 = _pct(dat["v_draws"][mask_m], 16)[order_m]
            phi84 = _pct(dat["v_draws"][mask_m], 84)[order_m]
            if len(zm_s) >= 2:
                zd = np.linspace(0.0, 1.1, 80)
                def _interp(phi_vals):
                    return PchipInterpolator(
                        np.r_[0.0, zm_s], np.r_[0.0, phi_vals], extrapolate=True
                    )(zd)
                axp.fill_between(zd, _interp(phi16), _interp(phi84),
                                 color=color, alpha=0.08, lw=0)
                axp.plot(zd, _interp(phi50), color=color, ls=ls, lw=lw, alpha=0.88)

        axp.set_title(rf"$R={R:.2f}$ kpc", pad=2, fontsize=11.5)
        axp.set_xlim(0, 1.12)
        axp.set_ylim(0, 1450)
        if idx_p % 3 == 0:
            axp.set_ylabel(r"$\Phi_z$ [km$^2$s$^{-2}$]")
        else:
            axp.tick_params(labelleft=False)
        if idx_p // 3 == 2:
            axp.set_xlabel(r"$z$ [kpc]")
        else:
            axp.tick_params(labelbottom=False)
        axp.grid(alpha=0.14)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {output_path}")
    return fig


def print_table2(outputs_dir: Path, chi2_csv: Optional[Path] = None) -> None:
    """Print Table 2 chi2_nu summary to stdout.

    Parameters
    ----------
    outputs_dir : directory containing mc100_chi2_all_models.csv.
    chi2_csv    : explicit path to the CSV (overrides auto-detect).
    """
    if chi2_csv is None:
        chi2_csv = Path(outputs_dir) / "mc100_chi2_all_models.csv"
    if not chi2_csv.exists():
        print(f"  {chi2_csv.name} not found in {outputs_dir}")
        return

    from collections import defaultdict
    model_chi2: dict[str, list] = defaultdict(list)
    model_name_map: dict[str, str] = {}
    with open(chi2_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["model_key"]
            model_chi2[key].append(float(row["chi2_nu"]))
            model_name_map[key] = row["model_name"]

    print(f"\n{'Model':<32} {'p16':>7} {'p50':>7} {'p84':>7}")
    print("-" * 58)
    order = [
        "baryonic", "qumond_simple", "qumond_standard", "qumond_mls",
        "stvg", "cdm_nfw", "cdm_einasto",
        "hmg_k1", "fr_screened", "refracted_gravity", "emergent_gravity",
    ]
    for key in order:
        if key not in model_chi2:
            continue
        arr = np.array(model_chi2[key])
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        name = model_name_map.get(key, key)
        print(f"  {name:<30} {p16:7.3f} {p50:7.3f} {p84:7.3f}")
    print()
