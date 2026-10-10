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
M_MU = 0.10565839         # muon mass (PDG muon tables)

# Cosmic muons
COSMIC_I0 = 70.0                 # vertical intensity [m^-2 s^-1 sr^-1], cos^2 zenith law
MUON_THMAX = np.radians(75)      # zenith cut of the 3D toys [rad]
COSMIC_I0_PMIN = 1.0             # momentum above which COSMIC_I0 counts the vertical muons [GeV/c]

# f = 203 mm mirror of configurations 1 and 5 (Edmund #43-584)
MIRROR_R = 406.4             # radius of curvature [mm]
MIRROR_D = 76.2              # diameter [mm]
MIRROR_CLEAR_FRAC = 0.9      # clear aperture diameter / mirror diameter

# f/1 short-focal-length mirrors of configurations 2 and 3: Edmund stock number -> (focal length, diameter) [mm]
MIRROR_SHORT_F = {'43-543': (50.8, 50.8), '43-553': (76.2, 76.2)}

# SiPM
SENSOR_HALF_2IN = 25.4       # half side of the 2" x 2" active area [mm]
SENSOR_HALF_PROTO = 25.6     # half side of 2x2 matrices of 8x8 pixels at 3.2 mm pitch [mm]
PIXEL_PITCH = 3.2            # [mm]
PIXEL_PITCH_FINE = 0.8       # finer alternative pixel pitch [mm]

# Aerogel tile
AEROGEL_HALF = 25.0          # half side of the 5 x 5 cm tile [mm]
AEROGEL_T = 10.0             # thickness [mm]
AEROGEL_N_AIR = 1.03         # refractive index as quoted in air
AEROGEL_POROSITY = 0.97      # pore volume fraction used for the gas-filled index
# pore fraction for the shift relative to air, 0.935 as in docs/design_options.md section 6.3 (1 - 0.14/2.2 = 0.936)
AEROGEL_POROSITY_AIRREF = 0.935
AEROGEL_LAMBDA_REF = 400.0   # wavelength at which the dispersive aerogel models equal AEROGEL_N_AIR [nm]
AEROGEL_POLE_NM = 83.22      # single-pole aerogel dispersion lambda_0 [nm] (Bellunato et al., NIM A 595 (2008) 183)
RAYLEIGH_C_AEROGEL = 0.005   # aerogel clarity C in Lambda = lambda^4/C [um^4/cm] (k4geo RAYLEIGH__Aerogel_BelleII)
AEROGEL_SURFACE_A = 1.0      # aerogel surface factor A in T = A exp(-C t/lambda^4) (k4geo applies none)

# SiPM protective window (part unknown) and the muon's direct hit
WINDOW_N = 1.5               # refractive index
WINDOW_T = 0.3               # thickness [mm]
MIP_DIRECT_PE = 1.0          # fired cells from the muon crossing the silicon (one microcell at normal incidence)

# Fixed values used in place of computed ones
THETA_AERO_GAS = 0.249       # aerogel Cherenkov angle taken as the ring angle in the gas [rad]
AERO_YIELD_PER_CM = 24.7     # emitted aerogel photons per cm, 320-450 nm
GAS_YIELD_REF = 33.5         # emitted Xe photons over GAS_YIELD_REF_LEN, 3 bar, 320-450 nm
GAS_YIELD_REF_LEN = 193.0    # [mm]
SENSOR_MUON_RATE = 1033.     # muons per hour through a 51.2 x 51.2 mm sensor (zenith < 75 deg)
