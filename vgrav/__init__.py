"""vgrav — Radial-vertical gravity decoupling in the Milky Way.

Package structure
-----------------
observations   Rotation-curve and vertical-potential data, radial data vectors.
solver         Axisymmetric Poisson, Helmholtz and permittivity solver.
baryonic       Baryonic densities, weighted bands and Monte Carlo realisations.
models         Gravity models (CDM, QUMOND, VEG, STVG, f(R), refracted gravity, HMG).
chi2           Likelihood terms.
figures        Figs. 1 and 2.

The scripts that reproduce the tables and figures of the paper are listed in
README.md.
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
    predict_hmg_common_s,
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
    "predict_mond_proxy", "predict_hmg_common_s", "predict_hmg_proxy", "predict_nu_proxy",
    "predict_cdm_nfw", "predict_cdm_einasto",
    "predict_stvg", "predict_qumond_solver",
    # figures
    "plot_fig2", "print_table2",
]
