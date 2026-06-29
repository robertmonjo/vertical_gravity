# Final reproduction criteria for the vertical-gravity HMG paper

This document is the default checklist for `release2`.  Its purpose is to
prevent the release pipeline from drifting away from the manuscript analysis.

## Primary rule

The release must reproduce the paper as described in `../main.tex`.

If the code differs from the manuscript description, that difference is a bug
unless it is explicitly documented and justified as scientifically more
rigorous.  Computational convenience is not a sufficient justification.

## Publication-default requirements

The default reproduction path must:

1. Start from the documented observational and baryonic input sources, not
   from final model CSVs.
2. Regenerate the four baryonic reconstructions:
   - McGaugh/Imig (MI);
   - Wang/Lian (LW);
   - de Salas (B2);
   - McMillan (MM).
3. Apply the manuscript conditioning of the baryonic families:
   - common inner-radial baryonic information;
   - mass-constrained MC100 stochastic realizations;
   - smooth correlated variations in `log R`;
   - no forced same-sign error at all radii.
4. Reconstruct, for every MC realization, both:
   - the radial Newtonian baryonic speed `v_N(R)`;
   - the corresponding three-dimensional Newtonian potential `Phi_N(R,z)`.
5. Fit all gravity prescriptions on the same primary data vector:
   - 152 model-independent radial kinematic constraints;
   - 44 Wang et al. vertical-potential measurements;
   - total `N = 196` data points.
6. Use the same chi-square definition for every model:
   - radial velocity residuals;
   - vertical potential residuals;
   - vertical effective uncertainty
     `sigma_eff^2 = sigma_Phi^2 + (K_z sigma_z)^2`;
   - reduced statistic `chi2_nu = chi2 / (N - n)`.
7. Use the physically adopted HMG closure:
   - a single neighbourhood scale `s`, with `s > 1`;
   - total/effective HMG radial relation for `v_c`;
   - spatial HMG projection for the vertical potential;
   - `gamma_cen = pi/2` by default.
8. Use the correct solvers by default:
   - full 3D/cylindrical Newtonian Poisson reconstruction for baryons;
   - QUMOND 3D solver for QUMOND branches;
   - direct 3D Yukawa calculation for STVG;
   - fitted CDM, VEG, ReG, screened-gravity and HMG branches on the same
     MC100 baryonic realizations.

## Things that are allowed only as explicit diagnostics

The following are not publication-default operations:

- using final precomputed model CSVs instead of regenerating them;
- using algebraic QUMOND proxies instead of the QUMOND 3D solver;
- loading QUMOND MLS/RAR from an older stochastic ensemble instead of the
  same MC100 qcopula realization family used by QUMOND simple/standard;
- using a single-template radial-weighted baryonic reconstruction in place of
  the MC100 construction from all four baryonic reconstructions (MI + LW + B2 + MM);
- skipping STVG;
- reducing the number of MC draws;
- changing the fit vector;
- changing the HMG gamma convention;
- changing parameter bounds or degrees of freedom.

Each diagnostic shortcut must require an explicit command-line flag and must
write output names or metadata identifying it as diagnostic.

## Acceptance tests

A regenerated release is acceptable only if all the following checks pass:

1. The radial fit vector contains exactly 152 points.
2. The vertical fit vector contains exactly 44 points.
3. The MC100 baryonic radial percentiles reproduce the accepted manuscript
   band within numerical tolerance at representative radii, especially:
   - inner disc: `R ~ 5--10 kpc`;
   - outer transition: `R ~ 20--60 kpc`.
4. The baryonic median near `R ~ 50 kpc` remains close to the manuscript
   consolidated value, not the erroneous high-outer-mass diagnostic branch.
5. The final table reproduces the manuscript-level model ordering:
   - CDM Einasto lowest median statistic;
   - HMG competitive and close to CDM;
   - CDM NFW next;
   - other modified-gravity prescriptions less successful on the joint vector.
6. HMG competitive statistics are recomputed from the same MC100 realization
   family used for the displayed figure and table.

## Documentation rule

Every generated CSV, figure, table, and log must be traceable to the script
and options that produced it.  If a result is diagnostic, the filename,
metadata, or documentation must say so.  Publication-default outputs must not
silently mix diagnostic approximations with the manuscript pipeline.
