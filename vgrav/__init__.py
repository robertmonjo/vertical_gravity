"""vgrav — Vertical Gravity Milky Way benchmark.

Package structure
-----------------
observations   Load Wang+2026 rotation curve and vertical potential data.
solver         Cylindrical axisymmetric Poisson solver (for QUMOND / STVG).
baryonic       Parametric density model + MC100 qcopula generation.
models         Gravity model equations for 11 models (CDM, MOND, STVG, HMG,
               f(R), Refracted Gravity, Emergent Gravity).
chi2           Chi-squared and reduced chi-squared computation.
figures        Standalone Figure 2 renderer and Table 2 printer.

Quick start — fast verification mode
-------------------------------------
Requires pre-computed CSVs in the outputs/ directory (included in this release).

    from pathlib import Path
    from vgrav.figures import print_table2, plot_fig2

    out = Path("outputs")
    print_table2(out)                       # reproduces Table 2
    fig = plot_fig2(out, dpi=220)           # reproduces Fig. 2
    fig.savefig("fig2_reproduced.png", dpi=220, bbox_inches="tight")

Full pipeline (from scratch)
-----------------------------
See scripts/step1_build_baryonic_mc100.py  through  step4_analyze_hmg.py.

    python scripts/step1_build_baryonic_mc100.py   # build 100 baryonic draws
    python scripts/step2_fit_all_models.py          # fit 11 gravity models
    python scripts/step3_reproduce_fig2_table2.py   # chi^2 summary + figure
    python scripts/step4_analyze_hmg.py             # HMG competitive analysis

Reference
---------
Wang et al. (2026), vertical gravitational potential of the Milky Way.
This package version: 1.0.0.
"""

from vgrav.observations import (
    load_rotation_curve,
    load_vertical_potential,
    load_observations,
    radial_fit_arrays,
    vertical_arrays,
)
from vgrav.solver import (
    CylGrid,
    make_grid,
    solve_axisymmetric,
    gradients,
    cylindrical_divergence,
    interp2,
)
from vgrav.chi2 import (
    chi2_radial,
    chi2_vertical,
    chi2_nu,
    vertical_force_from_phi,
    N_PRIMARY,
)
from vgrav.baryonic import (
    baryon_density,
    build_component_grid,
    build_mc100_draws,
    make_radial_grid,
    make_vertical_grid,
    ComponentGrid,
)
from vgrav.models import (
    nu_mond,
    hmg_factor,
    nu_fr,
    nu_rg,
    nu_eg,
    nfw_mass,
    nfw_density_from_local,
    einasto_mass,
    einasto_density_from_local,
    predict_mond_proxy,
    predict_hmg_proxy,
    predict_nu_proxy,
    predict_cdm_nfw,
    predict_cdm_einasto,
    predict_stvg,
    predict_qumond_solver,
)
from vgrav.figures import plot_fig2, print_table2

__version__ = "1.0.0"
__all__ = [
    # observations
    "load_rotation_curve", "load_vertical_potential", "load_observations",
    "radial_fit_arrays", "vertical_arrays",
    # solver
    "CylGrid", "make_grid", "solve_axisymmetric", "gradients",
    "cylindrical_divergence", "interp2",
    # chi2
    "chi2_radial", "chi2_vertical", "chi2_nu", "vertical_force_from_phi", "N_PRIMARY",
    # baryonic
    "baryon_density", "build_component_grid", "build_mc100_draws",
    "make_radial_grid", "make_vertical_grid", "ComponentGrid",
    # models
    "nu_mond", "hmg_factor", "nu_fr", "nu_rg", "nu_eg",
    "nfw_mass", "nfw_density_from_local", "einasto_mass", "einasto_density_from_local",
    "predict_mond_proxy", "predict_hmg_proxy", "predict_nu_proxy",
    "predict_cdm_nfw", "predict_cdm_einasto",
    "predict_stvg", "predict_qumond_solver",
    # figures
    "plot_fig2", "print_table2",
]
