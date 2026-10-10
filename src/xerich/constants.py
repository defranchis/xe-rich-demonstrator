"""Physical constants and fixed parameters, each defined once.

Units: lengths in mm, angles in rad, masses in GeV, pressures in bar, unless the
name says otherwise.
"""
import numpy as np

# Physical constants
R_GAS = 8.314462618          # molar gas constant [J mol^-1 K^-1]
T0 = 273.15                  # 0 degC [K]
P0 = 101325.                 # 1 atm [Pa]
T_ROOM = 293.15              # default gas temperature [K]
ALPHA = 1/137.035999         # fine-structure constant
HC = 1239.84198              # h c [eV nm]
HBARC_EV_CM = 1.97326980e-5  # hbar c [eV cm]

# Particle masses [GeV]
M_PI = 0.13957
M_K = 0.493677
M_P = 0.938272

# Cosmic muons
COSMIC_I0 = 70.0                 # vertical intensity [m^-2 s^-1 sr^-1], cos^2 zenith law
MUON_THMAX = np.radians(75)      # zenith cut of the 3D toys [rad]

# Phase-1 mirror (Edmund #43-584)
MIRROR_R = 406.4             # radius of curvature [mm]
MIRROR_D = 76.2              # diameter [mm]
MIRROR_CLEAR_FRAC = 0.9      # clear aperture diameter / mirror diameter

# SiPM
SENSOR_HALF_2IN = 25.4       # half side of the 2" x 2" active area [mm]
SENSOR_HALF_PROTO = 25.6     # half side of 2x2 matrices of 8x8 pixels at 3.2 mm pitch [mm]
PIXEL_PITCH = 3.2            # [mm]

# Aerogel tile
AEROGEL_HALF = 25.0          # half side of the 5 x 5 cm tile [mm]
AEROGEL_T = 10.0             # thickness [mm]
AEROGEL_N_AIR = 1.03         # refractive index as quoted in air
AEROGEL_POROSITY = 0.97      # pore volume fraction used for the gas-filled index
AEROGEL_LAMBDA_REF = 400.0   # wavelength at which the dispersive aerogel models equal AEROGEL_N_AIR [nm]
AEROGEL_POLE_NM = 83.22      # single-pole aerogel dispersion lambda_0 [nm] (Bellunato et al., NIM A 595 (2008) 183)
RAYLEIGH_C_AEROGEL = 0.005   # aerogel clarity C in Lambda = lambda^4/C [um^4/cm] (k4geo RAYLEIGH__Aerogel_BelleII)
AEROGEL_SURFACE_A = 1.0      # aerogel surface factor A in T = A exp(-C t/lambda^4) (k4geo applies none)

# SiPM protective window (part unknown) and the muon's direct hit
WINDOW_N = 1.5               # refractive index
WINDOW_T = 0.3               # thickness [mm]
MIP_DIRECT_PE = 1.0          # fired cells from the muon crossing the silicon (one microcell at normal incidence)

# Fixed values used in place of computed ones
THC_GAS_3BAR = 0.0636        # Xe Cherenkov angle, 3 bar, 320-450 nm, beta = 1 [rad] (Bideau-Mehu 1981 index; kept for the legacy regression)
THC_GAS_3BAR_400 = 0.0632    # Xe Cherenkov angle, 3 bar, 400 nm, beta = 1, drawn in the sketches [rad] (Bideau-Mehu 1981 index; kept for the legacy regression)
THETA_AERO_GAS = 0.249       # aerogel Cherenkov angle taken as the ring angle in the gas [rad]
AERO_YIELD_PER_CM = 24.7     # emitted aerogel photons per cm, 320-450 nm
GAS_YIELD_REF = 33.5         # emitted Xe photons over GAS_YIELD_REF_LEN, 3 bar, 320-450 nm
GAS_YIELD_REF_LEN = 193.0    # [mm]
SENSOR_MUON_RATE = 1033.     # muons per hour through a 51.2 x 51.2 mm sensor (zenith < 75 deg)
