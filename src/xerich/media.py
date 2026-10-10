"""Refractive-index models: Xe gas n(lambda, P, T), C4F10 and air tables, aerogel, the Medium wrapper; Rayleigh scattering."""
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from .constants import (AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_POLE_NM, AEROGEL_POROSITY, AEROGEL_POROSITY_AIRREF,
                        AEROGEL_SURFACE_A, HC, P0, R_GAS, RAYLEIGH_C_AEROGEL, T0, T_ROOM)

# Xe second virial coefficient B(T) = B_XE_293 + B_XE_SLOPE (T - B_XE_TREF), approximate (Dymond & Smith)
B_XE_293 = -130.0            # [cm^3/mol]
B_XE_SLOPE = 0.9             # [cm^3 mol^-1 K^-1]
B_XE_TREF = 293.15           # reference temperature of the B(T) parametrisation [K]

# C4F10 index vs photon energy [eV], RINDEX__C4F10_PFRICH of the ARC simulation:
# k4geo e1ba7bc, FCCee/CLD/compact/CLD_o3_v01/materials_arc_o1_v01.xml:40
C4F10_E_EV = (1.87855, 1.92389, 2.10539, 2.3247, 2.59502, 2.93647, 3.38139, 3.98521, 4.85156, 6.19921)
C4F10_N = (1.0013, 1.0013, 1.00131, 1.00131, 1.00132, 1.00133, 1.00134, 1.00136, 1.0014, 1.00149)

# Air index, RINDEX__Air (constant over 1.0-5.1 eV): k4geo e1ba7bc, same file, line 35
N_AIR = 1.00029

# Aerogel index vs photon energy [eV], RINDEX__Aerogel_PFRICH of the ARC simulation: k4geo e1ba7bc, same file, line 64
AEROGEL_K4GEO_E_EV = (1.87855, 1.96673, 2.05490, 2.14308, 2.23126, 2.31943, 2.40761, 2.49579, 2.58396, 2.67214, 2.76032,
                      2.84849, 2.93667, 3.02485, 3.11302, 3.20120, 3.28938, 3.37755, 3.46573, 3.55391, 3.64208, 3.73026,
                      3.81844, 3.90661, 3.99479, 4.08297, 4.17114, 4.25932, 4.34750, 4.43567, 4.52385, 4.61203, 4.70020,
                      4.78838, 4.87656, 4.96473, 5.05291, 5.14109, 5.22927, 5.31744, 5.40562, 5.49380, 5.58197, 5.67015,
                      5.75833, 5.84650, 5.93468, 6.02286, 6.11103, 6.19921)
AEROGEL_K4GEO_N = (1.01852, 1.01856, 1.01861, 1.01866, 1.01871, 1.01876, 1.01881, 1.01887, 1.01893, 1.01899, 1.01905,
                   1.01912, 1.01919, 1.01926, 1.01933, 1.01941, 1.01948, 1.01956, 1.01965, 1.01973, 1.01982, 1.01991,
                   1.02001, 1.02010, 1.02020, 1.02030, 1.02041, 1.02052, 1.02063, 1.02074, 1.02086, 1.02098, 1.02111,
                   1.02123, 1.02136, 1.02150, 1.02164, 1.02178, 1.02193, 1.02208, 1.02223, 1.02239, 1.02255, 1.02271,
                   1.02288, 1.02306, 1.02324, 1.02342, 1.02361, 1.02381)

# Bideau-Mehu et al. (1981) Sellmeier terms for Xe at 0 degC, 1 atm: (A_i, B_i) with lambda in um
SELLMEIER_XE = ((0.00322869, 46.301), (0.00355393, 59.578), (0.0606764, 112.74))

# Boerzsoenyi et al., Appl. Opt. 47, 4856 (2008), Xe at 273 K, 1000 mbar: n^2 - 1 = sum B_i l^2/(l^2 - C_i), (B_i, C_i [um^2]);
# C_1 = 12.75e-3 um^2 as corrected on refractiveindex.info (the paper prints 12.75e-6)
BORZSONYI_XE = ((1.0370161e-3, 12.75e-3), (3.122861e-4, 0.561e-3))
BORZSONYI_P = 1e5            # reference pressure [Pa]; the reference temperature is taken as T0

XE_INDEX_MODELS = ('borzsonyi', 'bm1981')
XE_INDEX_DEFAULT = 'borzsonyi'


def B_virial(T):
    """Xe second virial coefficient [m^3/mol] at temperature T [K]."""
    return (B_XE_293 + B_XE_SLOPE*(T-B_XE_TREF))*1e-6


def molar_density(P, T):
    """Molar density [mol/m^3] solving P = rho R T (1 + B rho); P in Pa, T in K."""
    B = B_virial(T); a = B*R_GAS*T; b = R_GAS*T
    return (-b+np.sqrt(b*b+4*a*P))/(2*a) if a != 0 else P/b


RHO0 = P0/(R_GAS*T0)*(1/(1+B_virial(T0)*P0/(R_GAS*T0)))   # density of the Sellmeier reference state [mol/m^3]
RHO_BORZSONYI = molar_density(BORZSONYI_P, T0)              # density of the Boerzsoenyi reference state [mol/m^3]


def nm1_ref(lam_nm):
    """(n - 1) of Xe at 0 degC, 1 atm; lam_nm in nm."""
    (a1, b1), (a2, b2), (a3, b3) = SELLMEIER_XE
    l = lam_nm*1e-3; x = l**-2
    return a1/(b1-x)+a2/(b2-x)+a3/(b3-x)


def n2m1_borzsonyi(lam_nm):
    """(n^2 - 1) of Xe at the Boerzsoenyi reference state (273 K, 1000 mbar); lam_nm in nm."""
    l2 = (np.asarray(lam_nm, float)*1e-3)**2
    return sum(b*l2/(l2-c) for b, c in BORZSONYI_XE)


def n_xe(lam_nm, P_bar, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """Xe refractive index at wavelength lam_nm [nm], pressure P_bar [bar], temperature T [K].

    Lorentz-Lorenz scaling of the reference index with the virial molar density. model: 'borzsonyi'
    (Boerzsoenyi 2008, reference density RHO_BORZSONYI) or 'bm1981' (Bideau-Mehu 1981, RHO0).
    """
    if model == 'bm1981':
        n0 = 1+nm1_ref(lam_nm); LL0 = (n0**2-1)/(n0**2+2)
        LL = LL0*molar_density(P_bar*1e5, T)/RHO0
    elif model == 'borzsonyi':
        s = n2m1_borzsonyi(lam_nm)
        LL = s/(s+3)*molar_density(P_bar*1e5, T)/RHO_BORZSONYI          # (n^2 - 1)/(n^2 + 2) = s/(s + 3)
    else:
        raise ValueError('unknown Xe index model %r (choose from %s)' % (model, ', '.join(XE_INDEX_MODELS)))
    return np.sqrt((1+2*LL)/(1-LL))


def aerogel_index(n_gas, n_air=AEROGEL_N_AIR, porosity=AEROGEL_POROSITY):
    """Aerogel index with gas of index n_gas in the pores: n_air + porosity (n_gas - 1)."""
    return n_air+porosity*(n_gas-1)


def aerogel_index_air(n_gas, n_air=AEROGEL_N_AIR, porosity=AEROGEL_POROSITY_AIRREF, n_gas_ref=N_AIR):
    """Aerogel index with gas of index n_gas in the pores, shifted from the in-air value: n_air + porosity (n_gas - n_gas_ref)."""
    return n_air+porosity*(n_gas-n_gas_ref)


def pore_index_shift(n_gas, porosity=AEROGEL_POROSITY):
    """Aerogel index shift from gas of index n_gas in the pores: porosity (n_gas - 1)."""
    return porosity*(n_gas-1)


def n_c4f10(lam_nm):
    """C4F10 index at wavelength lam_nm [nm]: ARC simulation table, linear in photon energy.

    Outside the table (above 660 nm or below 200 nm) the end value is held constant.
    """
    return np.interp(HC/np.asarray(lam_nm, float), C4F10_E_EV, C4F10_N)


def n_aerogel_single_pole(lam_nm, n_ref=AEROGEL_N_AIR, lam_ref=AEROGEL_LAMBDA_REF, lam0=AEROGEL_POLE_NM):
    """Aerogel index n^2 - 1 = a0 l^2/(l^2 - l0^2) with a0 fixed by n(lam_ref) = n_ref (wavelengths in nm)."""
    l2 = np.asarray(lam_nm, float)**2; a0 = (n_ref**2-1)*(1-lam0**2/lam_ref**2)
    return np.sqrt(1+a0*l2/(l2-lam0**2))


def n_aerogel_k4geo(lam_nm, n_ref=AEROGEL_N_AIR, lam_ref=AEROGEL_LAMBDA_REF):
    """ARC simulation aerogel table (linear in photon energy, end values held outside) with (n - 1) scaled to n(lam_ref) = n_ref."""
    def table(lam):
        return np.interp(HC/np.asarray(lam, float), AEROGEL_K4GEO_E_EV, AEROGEL_K4GEO_N)
    return 1+(table(lam_nm)-1)*(n_ref-1)/(table(lam_ref)-1)


def n_air(lam_nm):
    """Air index of the ARC simulation (wavelength independent)."""
    return np.full(np.shape(lam_nm), N_AIR)


class ConstantIndex:
    """Wavelength-independent refractive index, callable like the other index functions."""

    def __init__(self, n):
        self.n = float(n)

    def __call__(self, lam_nm):
        return np.full(np.shape(lam_nm), self.n)


@dataclass(frozen=True)
class Medium:
    """Optical medium: a label and its refractive index as a function of wavelength [nm]."""
    label: str
    index: Callable

    def n(self, lam_nm):
        return self.index(lam_nm)


def xenon(P_bar, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """Xe gas at P_bar [bar] and T [K] (n_xe with the given index model)."""
    return Medium('Xe %g bar' % P_bar, partial(n_xe, P_bar=P_bar, T=T, model=model))


def c4f10():
    """C4F10 with the ARC simulation table (n_c4f10)."""
    return Medium('C4F10 1 bar', n_c4f10)


def air():
    """Air with the ARC simulation index (n_air)."""
    return Medium('air', n_air)


def aerogel_constant(n=AEROGEL_N_AIR):
    """Aerogel with a constant index n: no dispersion, no pore-gas shift."""
    return Medium('aerogel n = %g' % n, ConstantIndex(n))


AEROGEL_MODELS = ('constant', 'single-pole', 'k4geo')


def aerogel_medium(model='constant', n=AEROGEL_N_AIR, lam_ref=AEROGEL_LAMBDA_REF):
    """Aerogel with index n at lam_ref [nm]: 'constant' (aerogel_constant), 'single-pole' or 'k4geo' dispersion."""
    if model == 'constant':
        return aerogel_constant(n)
    if model == 'single-pole':
        return Medium('aerogel single pole, n(%g nm) = %g' % (lam_ref, n), partial(n_aerogel_single_pole, n_ref=n, lam_ref=lam_ref))
    if model == 'k4geo':
        return Medium('aerogel k4geo shape, n(%g nm) = %g' % (lam_ref, n), partial(n_aerogel_k4geo, n_ref=n, lam_ref=lam_ref))
    raise ValueError('unknown aerogel model %r (choose from %s)' % (model, ', '.join(AEROGEL_MODELS)))


@dataclass(frozen=True)
class Rayleigh:
    """Rayleigh scattering in a slab: mean free path lambda^4/C and surface factor A.

    Hunt parametrisation of the tile transmission, T = A exp(-C t/lambda^4); A enters as a factor sqrt(A)
    on every transmission through a slab face (A for a straight pass), the rest is lost at the surface.
    C in um^4/cm; the default is the k4geo RAYLEIGH__Aerogel_BelleII table (same file, line 168), which
    follows lambda^4/C with C = 0.005 um^4/cm.
    """
    C: float = RAYLEIGH_C_AEROGEL
    A: float = AEROGEL_SURFACE_A

    def length(self, lam_nm):
        """Mean free path [mm] at wavelength lam_nm [nm]."""
        return 10*(np.asarray(lam_nm, float)*1e-3)**4/self.C

    @property
    def face_transmission(self):
        """Transmitted fraction sqrt(A) at each slab face crossing."""
        return float(np.sqrt(self.A))
