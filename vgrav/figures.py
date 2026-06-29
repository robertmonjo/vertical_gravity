"""Figure renderers for the vertical gravity benchmark.

All renderers read from the pre-computed CSV files in outputs/ and data/.
They reproduce the published figures without re-running any heavy computation.

Functions
---------
plot_fig1(data_dir=None, output_path=None, dpi=260)
    Reproduce Fig. 1: baryonic density face-on (a) + meridional (b).

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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# ── Model display config ───────────────────────────────────────────────────────
# key → (display_name, color, linestyle, linewidth)
_MODEL_STYLES: dict[str, tuple] = {
    "baryonic":           ("Baryonic (Newtonian)",      "#6baed6", "--", 1.5),
    "qumond_standard":    ("QUMOND standard",            "#28cae0", "-",  1.000),
    "qumond_simple":      ("QUMOND simple",              "#285ee0", "-",  1.288),
    "qumond_mls":         ("QUMOND MLS/RAR",             "#2800e0", "-",  1.500),
    "veg_fixed":          (r"VEG (fixed $a_{\rm EG}$)",  "#5d28e0", "-",  1.575),
    "veg_free":           (r"VEG (free $a_{\rm EG}$)",   "#7b28e0", "-",  1.431),
    "refracted_gravity":  ("ReG",                         "#c928e0", "-",  1.862),
    "fr_screened":        ("f(R) screened",               "#e0288b", "-",  2.150),
    "stvg":               ("STVG",                        "#e03028", "-",  2.438),
    "cdm_nfw":            ("CDM NFW",                     "#e09c28", "-",  2.725),
    "cdm_einasto":        ("CDM Einasto",                 "#b8e028", "-",  3.012),
    "hmg_k1":             ("HMG (This Work)",             "#4de028", "-",  3.300),
}
_BARY_COLOR = "#6baed6"

# Baryonic reconstruction colours
_FAMILY_COLORS: dict[str, str] = {
    "McGaugh2018_Imig2025": "#d62728",
    "Wang2026_Lian2022":    "#ff7f0e",
    "McMillan2017":         "#2ca02c",
    "deSalas2019_B2":       "#1f77b4",
    "Barros2016_MI":        "#9467bd",
}
_FAMILY_LABELS: dict[str, str] = {
    "McGaugh2018_Imig2025": "McGaugh/Imig",
    "Wang2026_Lian2022":    "Wang/Lian",
    "McMillan2017":         "McMillan 2017",
    "deSalas2019_B2":       "de Salas 2019 B2",
    "Barros2016_MI":        "Barros 2016 MI",
}


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
    hmg_1fam_outputs_dir: Optional[Path] = None,
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

    # ── Baryonic target band ──────────────────────────────────────────────────
    # Prefer baryonic_target_band.csv (processed band); fall back to baryon_band.csv
    _data_dir = outputs_dir.parent / "data"
    _tband_path = _data_dir / "baryonic_target_band.csv"
    _bband_path = _data_dir / "baryon_band.csv"
    for _p in (_data_dir.parent / "data" / "baryonic_target_band.csv",
               outputs_dir / "baryonic_target_band.csv"):
        if not _tband_path.exists():
            _tband_path = _p
    bband_R: Optional[np.ndarray] = None
    bband_p5 = bband_p16 = bband_p84 = bband_p95 = None
    bband_reconstructions: dict[str, np.ndarray] = {}
    if _tband_path.exists():
        with open(_tband_path, newline="", encoding="utf-8") as fh:
            _tb = list(csv.DictReader(fh))
        bband_R   = np.array([float(r["R_kpc"])              for r in _tb])
        bband_p5  = np.array([float(r["Baryonic_target_p5"]) for r in _tb])
        bband_p16 = np.array([float(r["Baryonic_target_p16"])for r in _tb])
        bband_p84 = np.array([float(r["Baryonic_target_p84"])for r in _tb])
        bband_p95 = np.array([float(r["Baryonic_target_p95"])for r in _tb])
    elif _bband_path.exists():
        with open(_bband_path, newline="", encoding="utf-8") as fh:
            _tb = list(csv.DictReader(fh))
        bband_R   = np.array([float(r["R_kpc"])      for r in _tb])
        bband_p5  = np.array([float(r["hybrid_p5"])  for r in _tb])
        bband_p16 = np.array([float(r["hybrid_p16"]) for r in _tb])
        bband_p84 = np.array([float(r["hybrid_p84"]) for r in _tb])
        bband_p95 = np.array([float(r["hybrid_p95"]) for r in _tb])
    # Family dotted lines always come from baryon_band.csv
    if _bband_path.exists():
        with open(_bband_path, newline="", encoding="utf-8") as fh:
            _bb = list(csv.DictReader(fh))
        for _fam in _FAMILY_COLORS:
            if f"center_{_fam}" in _bb[0]:
                bband_reconstructions[_fam] = np.array([float(r[f"center_{_fam}"]) for r in _bb])

    # Baryonic draws — only needed for vertical panels
    bary_vert_p = outputs_dir / "mc100_baryonic_vertical.csv"
    bary_v_coords, bary_v_draws = None, None
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
    sm = lambda v: gaussian_filter1d(v, sigma=6)
    if bband_R is not None:
        ax_top.fill_between(
            bband_R, sm(bband_p5), sm(bband_p95),
            color=_BARY_COLOR, alpha=0.075, lw=0, zorder=0,
        )
        ax_top.fill_between(
            bband_R, sm(bband_p16), sm(bband_p84),
            color=_BARY_COLOR, alpha=0.18, lw=0, zorder=0.05,
            label=r"Baryonic band $p_{16}$–$p_{84}$",
        )
        for _fam, _fcol in _FAMILY_COLORS.items():
            if _fam in bband_reconstructions:
                ax_top.plot(bband_R, bband_reconstructions[_fam],
                            color=_fcol, lw=1.4, linestyle=":", alpha=0.65, zorder=0.08)

    # Load chi²_ν medians from mc100_chi2_all_models.csv if available
    _chi2_med: dict[str, float] = {}
    _chi2_csv = outputs_dir / "mc100_chi2_all_models.csv"
    if _chi2_csv.exists():
        _chi2_by_key: dict[str, list] = {}
        with open(_chi2_csv, newline="", encoding="utf-8") as _fh:
            for _row in csv.DictReader(_fh):
                _chi2_by_key.setdefault(_row["model_key"], []).append(float(_row["chi2_nu"]))
        _chi2_med = {k: float(np.median(v)) for k, v in _chi2_by_key.items()}

    # VEG fixed is in Table 2 but not shown in the figure
    _FIG2_SKIP = {"veg_fixed"}

    # Load HMG 1-fam overlay (release_alt / self-consistent Imig) if provided
    hmg_1fam_data: Optional[dict] = None
    if hmg_1fam_outputs_dir is not None:
        _h1r = Path(hmg_1fam_outputs_dir) / "model_hmg_k1_radial.csv"
        _h1v = Path(hmg_1fam_outputs_dir) / "model_hmg_k1_vertical.csv"
        if _h1r.exists() and _h1v.exists():
            _r1c, _r1d, _r1chi2 = _read_draw_csv(_h1r)
            _v1c, _v1d, _v1chi2 = _read_draw_csv(_h1v)
            hmg_1fam_data = {
                "r_coords": _r1c, "r_draws": _r1d,
                "v_coords": _v1c, "v_draws": _v1d,
            }

    model_handles = []
    _model_curves: dict = {}
    for key, dat in available.items():
        if key in _FIG2_SKIP:
            continue
        label_base, color, ls, lw = _MODEL_STYLES[key]
        med = gaussian_filter1d(_pct(dat["r_draws"], 50), sigma=4)
        p16 = gaussian_filter1d(_pct(dat["r_draws"], 16), sigma=4)
        p84 = gaussian_filter1d(_pct(dat["r_draws"], 84), sigma=4)
        ax_top.fill_between(dat["r_coords"], p16, p84, color=color, alpha=0.08, lw=0)
        chi2_sfx = (f" ($\\chi^2_{{\\nu}}={_chi2_med[key]:.2f}$)"
                    if key in _chi2_med and key != "baryonic" else "")
        h, = ax_top.plot(dat["r_coords"], med, color=color, ls=ls, lw=lw,
                         label=label_base + chi2_sfx, zorder=3, alpha=0.82)
        if key != "baryonic":
            model_handles.append(h)
            _model_curves[key] = (dat["r_coords"], med, p16, p84, color, ls, lw)

    # HMG 1-fam overlay (self-consistent Imig, dashed green)
    if hmg_1fam_data is not None:
        _h1_med = gaussian_filter1d(_pct(hmg_1fam_data["r_draws"], 50), sigma=4)
        _h1_p16 = gaussian_filter1d(_pct(hmg_1fam_data["r_draws"], 16), sigma=4)
        _h1_p84 = gaussian_filter1d(_pct(hmg_1fam_data["r_draws"], 84), sigma=4)
        _h1_color = "#1a7a10"  # darker green to contrast with nbar4 HMG solid curve
        ax_top.fill_between(hmg_1fam_data["r_coords"], _h1_p16, _h1_p84,
                            color=_h1_color, alpha=0.08, lw=0)
        _h1_chi2_sfx = ""
        if "hmg_k1" in _chi2_med:
            _h1_chi2_csv = Path(hmg_1fam_outputs_dir) / "mc100_chi2_all_models.csv"
            if _h1_chi2_csv.exists():
                _h1_vals = []
                with open(_h1_chi2_csv, newline="", encoding="utf-8") as _fh:
                    for _row in csv.DictReader(_fh):
                        if _row["model_key"] == "hmg_k1":
                            _h1_vals.append(float(_row["chi2_nu"]))
                if _h1_vals:
                    _h1_chi2_sfx = f" ($\\chi^2_{{\\nu}}={float(np.median(_h1_vals)):.2f}$)"
        h1, = ax_top.plot(hmg_1fam_data["r_coords"], _h1_med,
                          color=_h1_color, ls="-", lw=1.6,
                          label="HMG+Imig (This Work)" + _h1_chi2_sfx, zorder=4, alpha=0.92)
        model_handles.append(h1)

    # Observational data — per-survey styled (default group_style)
    _group_style = [
        ("Feng2026",                     "h", "#f781bf", "Feng et al. 2026 Cepheids",      "independent", False),
        ("McClureDickey2016",            "o", "#009e73", "McClure-Dickey 2016 Q1 H I",     "independent", False),
        ("McClureDickey2007",            "o", "#b54f12", "McClure-Dickey 2007 Q4 H I",     "independent", False),
        ("Eilers2019_McGaugh2019",       "s", "#1b9e9a", "Eilers/McGaugh 2019 Gaia",       "independent", False),
        ("Watkins2019",                  "v", "#888888", "Watkins 2019 TME",                "dependent",   True),
        ("Deason2021",                   "*", "#888888", "Deason 2021 DF",                  "dependent",   True),
        ("Wang2020_Deason2012",          "^", "#888888", "Deason 2012 DF",                  "dependent",   True),
        ("Wang2020_",                    "x", "#888888", "Wang 2020 compilation",           "dependent",   False),
        ("Bird2022_digitized_KG_Jeans",  "o", "#888888", "Bird 2022 KG Jeans",             "dependent",   False),
        ("Bird2022_digitized_KG_TME",    "o", "#888888", "Bird 2022 KG TME",               "dependent",   True),
        ("Bird2022_digitized_BHB_Jeans", "s", "#888888", "Bird 2022 BHB Jeans",            "dependent",   False),
        ("Bird2022_digitized_BHB_TME",   "s", "#888888", "Bird 2022 BHB TME",              "dependent",   True),
    ]
    independent_handles: list = []
    dependent_handles: list = []
    wang_rr = wang_vv = wang_ss_obs = wang_ss_total = wang_order = None
    if rot_obs:
        wang_rows = [r for r in rot_obs
                     if str(r.get("dataset", "")).startswith("Wang2026_rotation_curve")
                     and r.get("vc_kms") is not None]
        if wang_rows:
            wang_rr = np.array([r["R_kpc"] for r in wang_rows])
            wang_vv = np.array([r["vc_kms"] for r in wang_rows])
            wang_ss_total = np.array([r.get("sigma_vc_kms") or 0.0 for r in wang_rows])
            wang_ss_obs = np.array([r.get("sigma_obs_kms") or r.get("sigma_vc_kms") or 0.0
                                     for r in wang_rows])
            wang_order = np.argsort(wang_rr)
            band = ax_top.fill_between(
                wang_rr[wang_order],
                wang_vv[wang_order] - wang_ss_obs[wang_order],
                wang_vv[wang_order] + wang_ss_obs[wang_order],
                color="#ff9999", alpha=0.32, lw=0, label="Wang et al. 2026 (obs. err.)",
            )
            independent_handles.append(band)
            h = ax_top.errorbar(wang_rr, wang_vv, yerr=wang_ss_total, fmt="o", ms=5.0,
                                color="black", ecolor="black",
                                lw=0.8, label="Wang et al. 2026 RC", zorder=8)
            independent_handles.append(h)

        for key, marker, color, label, legend_group, filled in _group_style:
            sub = [r for r in rot_obs
                   if key in str(r.get("dataset", "")) and r.get("vc_kms") is not None]
            if key == "Wang2020_":
                sub = [r for r in sub if "Deason2012" not in str(r.get("dataset", ""))]
            if not sub:
                continue
            rr = np.array([r["R_kpc"] for r in sub])
            vv = np.array([r["vc_kms"] for r in sub])
            ss = np.array([r.get("sigma_vc_kms") or 0.0 for r in sub])
            h = ax_top.errorbar(
                rr, vv, yerr=ss if np.any(ss > 0) else None,
                fmt=marker, ms=5.5 if marker != "*" else 9.0,
                mfc=color if filled else "white", mec=color, ecolor=color,
                alpha=0.72, lw=0.7, label=label, zorder=8,
            )
            (independent_handles if legend_group == "independent" else dependent_handles).append(h)

    ax_top.set_xscale("log")
    ax_top.set_xlim(2.0, 400.0)
    ax_top.set_ylim(0.0, 300.0)
    ax_top.set_xlabel(r"$R$ [kpc]")
    ax_top.set_ylabel(r"$v_c$ [km s$^{-1}$]")
    ax_top.set_xticks([2, 5, 10, 20, 50, 100, 200, 400])
    ax_top.set_xticklabels(["2", "5", "10", "20", "50", "100", "200", "400"])
    ax_top.grid(True, which="major", color="0.88", lw=0.8)

    _leg_kw = dict(frameon=False, fontsize=9.0, title_fontsize=10.8,
                   handlelength=1.2, handletextpad=0.3, columnspacing=0.6, labelspacing=0.08)
    if independent_handles:
        fig.legend(handles=independent_handles, loc="upper left",
                   bbox_to_anchor=(0.015, 0.995), ncol=1,
                   title="Model-independent obs.", **_leg_kw)
    if dependent_handles:
        fig.legend(handles=dependent_handles, loc="upper left",
                   bbox_to_anchor=(0.205, 0.995), ncol=1,
                   title="Model-dependent obs.", **_leg_kw)
    _bary_handles = [
        Patch(facecolor="#6baed6", alpha=0.18, label=r"MC $p_{16}$–$p_{84}$"),
        Patch(facecolor="#6baed6", alpha=0.075, label=r"MC $p_5$–$p_{95}$"),
    ] + [
        Line2D([], [], color=_fc, lw=1.4, linestyle=":", alpha=0.65,
               label=_FAMILY_LABELS[_fam])
        for _fam, _fc in _FAMILY_COLORS.items()
    ]
    fig.legend(handles=_bary_handles, loc="upper left",
               bbox_to_anchor=(0.405, 0.995), ncol=1,
               title=r"Baryonic ($v_N$)", **_leg_kw)
    if model_handles:
        fig.legend(handles=model_handles, loc="upper right",
                   bbox_to_anchor=(0.985, 0.995), ncol=2, frameon=False,
                   title="Gravity models", fontsize=9.0, title_fontsize=10.8,
                   handlelength=2.0, handletextpad=0.3, labelspacing=0.08,
                   columnspacing=0.5)

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
                sig_z_arr = np.array([v.get("sigma_z_kpc") or 0.0 for v in obs_R])
                order_obs = np.argsort(zobs)
                axp.fill_between(zobs[order_obs], (phio - sigo)[order_obs],
                                 (phio + sigo)[order_obs], color="#ff9999", alpha=0.28, lw=0)
                axp.errorbar(zobs, phio, yerr=sigo, xerr=sig_z_arr,
                             fmt="o", ms=3.0, color="black", ecolor="black",
                             elinewidth=0.9, capsize=2.4, capthick=0.9, zorder=8)

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

        # Model curves (skip baryonic — shown as band; skip veg_fixed — Table 2 only)
        _vert_skip = {"baryonic"} | _FIG2_SKIP
        for key, dat in available.items():
            if key in _vert_skip:
                continue
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

        # HMG 1-fam vertical overlay
        if hmg_1fam_data is not None:
            mask_h1 = np.abs(hmg_1fam_data["v_coords"][:, 0] - R) < 0.05
            if mask_h1.any():
                zh1 = hmg_1fam_data["v_coords"][mask_h1, 1]
                order_h1 = np.argsort(zh1)
                zh1_s = zh1[order_h1]
                phi50_h1 = _pct(hmg_1fam_data["v_draws"][mask_h1], 50)[order_h1]
                if len(zh1_s) >= 2:
                    zd = np.linspace(0.0, 1.1, 80)
                    phi_h1_interp = PchipInterpolator(
                        np.r_[0.0, zh1_s], np.r_[0.0, phi50_h1], extrapolate=True
                    )(zd)
                    axp.plot(zd, phi_h1_interp, color="#1a7a10", ls="-", lw=1.6, alpha=0.92)

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
        "hmg_k1", "fr_screened", "refracted_gravity", "veg_fixed", "veg_free",
    ]
    for key in order:
        if key not in model_chi2:
            continue
        arr = np.array(model_chi2[key])
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        name = model_name_map.get(key, key)
        print(f"  {name:<30} {p16:7.3f} {p50:7.3f} {p84:7.3f}")
    print()


# ── Figure 1 ──────────────────────────────────────────────────────────────────

_R_SUN = 8.277          # kpc (Reid+2019)
_X_SUN_GC = -_R_SUN    # GC at (-R_SUN, 0) in Sun-centred display coords


def plot_fig1(
    data_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    dpi: int = 260,
) -> plt.Figure:
    """Reproduce Fig. 1: baryonic density + spiral arm tracers.

    Parameters
    ----------
    data_dir    : directory containing fig1_density_grid.npz and
                  fig1_spiral_arms.csv.  Defaults to ../data/ relative to
                  this file.
    output_path : save figure here if given.
    dpi         : figure resolution.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir = Path(data_dir)

    # ── Load density grid ─────────────────────────────────────────────────
    npz_path = data_dir / "fig1_density_grid.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Density grid not found: {npz_path}")
    d = np.load(npz_path)
    R_grid = d["R"]
    z_grid = d["z"]
    rho_pc3 = d["rho_pc3"]
    sigma = d["sigma"]

    # ── Load spiral arm loci ──────────────────────────────────────────────
    arms_path = data_dir / "fig1_spiral_arms.csv"
    arm_display: dict[str, dict] = {}
    arm_confirm: dict[str, dict] = {}
    if arms_path.exists():
        with open(arms_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row["arm"]
                seg = row["segment"]
                x = float(row["x_disp"])
                y = float(row["y_disp"])
                meta = {k: row[k] for k in ("color", "label", "lw")}
                meta["lw"] = float(meta["lw"])
                target = arm_display if seg == "display" else arm_confirm
                if name not in target:
                    target[name] = {**meta, "xs": [], "ys": []}
                target[name]["xs"].append(x)
                target[name]["ys"].append(y)

    # ── Figure layout (matches default exactly) ─────────────────────────
    _AH  = 4.00
    _AW  = _AH * 52 / 48
    _CBW = 0.20
    _CBP = 0.07
    _LM  = 0.85
    _BOT = 0.62
    _GAP = 1.35
    _RM  = 0.90
    FH   = 4.75
    FW   = _LM + _AW + _CBP + _CBW + _GAP + _AW + _CBP + _CBW + _RM

    AH  = _AH / FH;  AW  = _AW / FW
    CBW = _CBW / FW; CBP = _CBP / FW
    GAP = _GAP / FW; L0  = _LM / FW;  BOT = _BOT / FH

    X1 = L0 + AW + CBP
    X2 = X1 + CBW + GAP
    X3 = X2 + AW + CBP

    fig  = plt.figure(figsize=(FW, FH))
    ax0  = fig.add_axes([L0, BOT, AW,  AH])
    cax0 = fig.add_axes([X1, BOT, CBW, AH])
    ax1  = fig.add_axes([X2, BOT, AW,  AH])
    cax1 = fig.add_axes([X3, BOT, CBW, AH])

    # ── Panel (a): face-on map ────────────────────────────────────────────
    from matplotlib.colors import LogNorm
    x  = np.linspace(-26.0, 26.0, 460)
    y  = np.linspace(-32.0, 16.0, 460)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    x_gc = _X_SUN_GC - yy
    y_gc = xx
    rr = np.sqrt(x_gc**2 + y_gc**2)
    sigma_xy = np.interp(rr, R_grid, sigma, left=sigma[0], right=sigma[-1])

    im0 = ax0.pcolormesh(
        x, y, np.maximum(sigma_xy, 1e-4),
        shading="auto", cmap="Greys",
        norm=LogNorm(vmin=float(np.nanmin(sigma_xy[sigma_xy > 0])), vmax=1.0e3),
        alpha=0.72,
    )

    for name, data in arm_display.items():
        ax0.plot(data["xs"], data["ys"],
                 color=data["color"], lw=data["lw"] * 0.65, alpha=0.50, zorder=5)
    for name, data in arm_confirm.items():
        ax0.plot(data["xs"], data["ys"],
                 color=data["color"], lw=data["lw"], alpha=0.90, zorder=6,
                 label=data["label"])

    ax0.axhline(0.0, color="white", lw=0.8, alpha=0.42, zorder=4)
    ax0.axvline(0.0, color="white", lw=0.8, alpha=0.42, zorder=4)
    ax0.scatter([0], [_X_SUN_GC], marker="*",       s=95,  color="black",   zorder=8,
                label="Galactic centre")
    ax0.scatter([0], [0],         marker=r"$\odot$", s=160, color="#f4df3a", zorder=9,
                linewidth=0.0, label="Sun")

    ax0.set_aspect("equal")
    ax0.set_xlim(-26, 26)
    ax0.set_ylim(-32, 16)
    ax0.set_xlabel(r"$x_\odot$ [kpc]", fontsize=12)
    ax0.set_ylabel(r"$y_\odot$ [kpc]", fontsize=12)
    ax0.text(0.025, 0.965, "(a)", transform=ax0.transAxes,
             ha="left", va="top", fontsize=12)

    cb0 = fig.colorbar(im0, cax=cax0)
    cb0.set_label(r"$\Sigma_b$ [$M_\odot$ pc$^{-2}$]", fontsize=10)
    cb0.ax.tick_params(labelsize=9)

    leg = ax0.legend(frameon=True, fontsize=7.8, loc="lower left",
                     ncol=2, handlelength=2.0, columnspacing=0.6)
    leg.set_zorder(30)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.58)
    leg.get_frame().set_linewidth(0.0)
    for sp in ax0.spines.values():
        sp.set_linewidth(0.85)
    ax0.tick_params(direction="in", top=True, right=True, labelsize=10, length=4)

    # ── Panel (b): meridional cross-section ──────────────────────────────
    mask_R = (R_grid >= 0.0) & (R_grid <= 35.0)
    z_pos = z_grid[z_grid >= 0]
    mask_z = z_pos <= 8.0
    rho_pos = rho_pc3[:, z_grid >= 0]

    im1 = ax1.pcolormesh(
        R_grid[mask_R], z_pos[mask_z],
        np.maximum(rho_pos[np.ix_(mask_R, mask_z)].T, 1.0e-8),
        shading="auto", cmap="Greys",
        norm=LogNorm(
            vmin=1.0e-7,
            vmax=max(1.0e1, float(np.nanpercentile(rho_pos[np.ix_(mask_R, mask_z)], 99.8))),
        ),
    )
    levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cs = ax1.contour(
        R_grid[mask_R], z_pos[mask_z],
        rho_pos[np.ix_(mask_R, mask_z)].T,
        levels=levels, colors="white", linewidths=0.65, alpha=0.82,
    )
    ax1.clabel(cs, inline=True, fontsize=7,
               fmt=lambda v: rf"$10^{{{int(np.log10(v))}}}$")
    ax1.set_xlabel(r"$R$ [kpc]", fontsize=12)
    ax1.set_ylabel(r"$z$ [kpc]", fontsize=12)
    ax1.text(0.025, 0.965, "(b)", transform=ax1.transAxes,
             ha="left", va="top", fontsize=12)

    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r"$\rho_b$ [$M_\odot$ pc$^{-3}$]", fontsize=10)
    cb1.ax.tick_params(labelsize=9)
    for sp in ax1.spines.values():
        sp.set_linewidth(0.85)
    ax1.tick_params(direction="in", top=True, right=True, labelsize=10, length=4)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {output_path}")
    return fig
