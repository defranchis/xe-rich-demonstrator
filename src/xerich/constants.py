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
T_PORE = T_ROOM              # default temperature of the gas in the aerogel pores [K] (box thermal state not known yet)
P_REF = 2.0                  # reference Xe pressure [bar abs] of the defaults, headline numbers and figures
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
MUON_THMAX = np.radians(75)      # zenith cut of the cosmic-muon models [rad]
COSMIC_I0_PMIN = 1.0             # momentum above which COSMIC_I0 counts the vertical muons [GeV/c]

# f = 203 mm mirror of configurations 1 and 5 (Edmund #43-584)
MIRROR_R = 406.4             # radius of curvature [mm]
MIRROR_D = 76.2              # diameter [mm]
MIRROR_CLEAR_FRAC = 0.9      # clear aperture diameter / mirror diameter
# Nominal reflectivity of the Edmund enhanced-Al mirrors (#43-584, #43-543, #43-553) vs wavelength [nm], linear between the
# nodes: assumed curve from the vendor spec (Ravg > 89 % at 250-450 nm, > 85 % at 250-700 nm, nothing above), not a measured curve
MIRROR_REFL_NOMINAL_NM = (250.0, 450.0, 700.0, 800.0, 850.0, 950.0)
MIRROR_REFL_NOMINAL = (0.89, 0.89, 0.87, 0.82, 0.82, 0.85)

# f/1 short-focal-length mirrors of configurations 2 and 3: Edmund stock number -> (focal length, diameter) [mm]
MIRROR_SHORT_F = {'43-543': (50.8, 50.8), '43-553': (76.2, 76.2)}

# SiPM
SENSOR_HALF_2IN = 25.4       # half side of the 2" x 2" active area [mm]
SENSOR_HALF_PROTO = 25.6     # half side of 2x2 matrices of 8x8 pixels at 3.2 mm pitch [mm]
PIXEL_PITCH = 3.2            # [mm]
PIXEL_PITCH_FINE = 0.8       # finer alternative pixel pitch [mm]
# ARC prototype board: 2 x 2 abutting S13361-3050NE-08 packages of 8 x 8 pads (datasheet KAPD1054E07, dimensional outline)
BOARD_PACKAGE = 25.8         # package side [mm]
BOARD_EDGE = 0.2             # package edge to the first active pad [mm]
BOARD_PAD = 3.0              # active pad side [mm] (pitch PIXEL_PITCH, 0.2 mm between pads)
BOARD_NPAD = 8               # pads per package side
PAD_KMAX = 40                # fired-pad counts tabulated per ring: P(N >= k) for k = 0..PAD_KMAX
# Board reflectance, flat in wavelength (Congedo et al., arXiv:2602.20139, Sec. 3.1, Table 1: S13361-3050/-2050 8 x 8 arrays,
# 50 um cells, epoxy, integrating sphere at 8 deg; total 18-24 % above 380 nm, diffuse about 10 %, specular = total - diffuse;
# lower below about 380 nm, not modelled). Literature measurement of the same part, to be confirmed on our board.
BOARD_R_SPEC = 0.11          # specular reflectance (range 0.08-0.14)
BOARD_R_DIFF = 0.10          # diffuse (Lambertian) reflectance

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

# SiPM protective window and the muon's direct hit
WINDOW_N = 1.55              # refractive index: S13361 datasheet epoxy value, to be confirmed for the board's arrays
WINDOW_T = 0.1               # thickness [mm]: S13361 datasheet value (0.05-0.15 mm band), to be confirmed for the board's arrays
MIP_DIRECT_PE = 1.0          # fired cells from the muon crossing the silicon (one microcell at normal incidence)

# Fixed values used in place of computed ones
THETA_AERO_GAS = 0.249       # aerogel Cherenkov angle taken as the ring angle in the gas [rad]
YIELD_BAND_NM = (320.0, 450.0)   # wavelength band of the fixed yields below [nm]
AERO_YIELD_PER_CM = 24.7     # emitted aerogel photons per cm, in YIELD_BAND_NM
# emitted Xe photons over GAS_YIELD_REF_LEN at P_REF, beta = 1, in YIELD_BAND_NM (Frank-Tamm): N = 2 pi alpha L int (1 - 1/n^2) dlambda/lambda^2,
# n = media.n_xe(lambda, P_REF, T_ROOM), Boerzsoenyi (21.72; 32.73 at 3 bar, where the legacy toy used 33.5)
GAS_YIELD_REF = 21.72
GAS_YIELD_REF_LEN = 193.0    # [mm]
SENSOR_MUON_RATE = 1033.     # muons per hour through a 51.2 x 51.2 mm sensor (zenith < 75 deg)
