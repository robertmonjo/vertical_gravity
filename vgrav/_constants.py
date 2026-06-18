"""Physical constants in kpc / (km/s)^2 units throughout."""
import math

G = 4.300917270e-6          # kpc (km/s)^2 Msun^-1
C_KMS = 299_792.458         # speed of light [km/s]
YR_S = 365.25 * 24 * 3600  # year in seconds
T0_S = 13.8e9 * YR_S       # age of universe [s]
KPC_KM = 3.085677581e16     # km per kpc
T0_KPC_PER_KMS = T0_S / KPC_KM  # T0 in kpc / (km/s) = T0 / c  [kpc s km^-1]
A0_KMS2_PER_KPC = 1.2e-10 * 3.085677581e19 / 1e6  # MOND a0 [kpc (km/s)^-2] -> [(km/s)^2 kpc^-1]
R_SUN = 8.178               # Solar galactocentric radius [kpc]
GEV_CM3_TO_MSUN_KPC3 = 2.63366e7  # unit conversion

# Verlinde EG: a_EG = c * H0 / 6, H0 = 70 km/s/Mpc = 0.070 km/s/kpc
A_EG_FIXED = C_KMS * (70.0 / 1000.0) / 6.0  # [(km/s)^2 kpc^-1]

# HMG angular factor (isotropic, gamma_u = pi/3)
_GAMMA_U = math.pi / 3.0
HMG_EXTRA = 2.0 * C_KMS / T0_KPC_PER_KMS * math.cos(_GAMMA_U) / _GAMMA_U
