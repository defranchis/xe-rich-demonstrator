"""Refractive-index models: Xe gas n(lambda, P, T), C4F10 and air tables, aerogel, and the Medium wrapper."""
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from .constants import AEROGEL_N_AIR, AEROGEL_POROSITY, HC, P0, R_GAS, T0, T_ROOM

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

# Bideau-Mehu et al. (1981) Sellmeier terms for Xe at 0 degC, 1 atm: (A_i, B_i) with lambda in um
SELLMEIER_XE = ((0.00322869, 46.301), (0.00355393, 59.578), (0.0606764, 112.74))


def B_virial(T):
    """Xe second virial coefficient [m^3/mol] at temperature T [K]."""
    return (B_XE_293 + B_XE_SLOPE*(T-B_XE_TREF))*1e-6


def molar_density(P, T):
    """Molar density [mol/m^3] solving P = rho R T (1 + B rho); P in Pa, T in K."""
    B = B_virial(T); a = B*R_GAS*T; b = R_GAS*T
    return (-b+np.sqrt(b*b+4*a*P))/(2*a) if a != 0 else P/b


RHO0 = P0/(R_GAS*T0)*(1/(1+B_virial(T0)*P0/(R_GAS*T0)))   # density of the Sellmeier reference state [mol/m^3]


def nm1_ref(lam_nm):
    """(n - 1) of Xe at 0 degC, 1 atm; lam_nm in nm."""
    (a1, b1), (a2, b2), (a3, b3) = SELLMEIER_XE
    l = lam_nm*1e-3; x = l**-2
    return a1/(b1-x)+a2/(b2-x)+a3/(b3-x)


def n_xe(lam_nm, P_bar, T=T_ROOM):
    """Xe refractive index at wavelength lam_nm [nm], pressure P_bar [bar], temperature T [K].

    Lorentz-Lorenz scaling of the reference index with the virial molar density.
    """
    n0 = 1+nm1_ref(lam_nm); LL0 = (n0**2-1)/(n0**2+2)
    LL = LL0*molar_density(P_bar*1e5, T)/RHO0
    return np.sqrt((1+2*LL)/(1-LL))


def aerogel_index(n_gas, n_air=AEROGEL_N_AIR, porosity=AEROGEL_POROSITY):
    """Aerogel index with gas of index n_gas in the pores: n_air + porosity (n_gas - 1)."""
    return n_air+porosity*(n_gas-1)


def pore_index_shift(n_gas, porosity=AEROGEL_POROSITY):
    """Aerogel index shift from gas of index n_gas in the pores: porosity (n_gas - 1)."""
    return porosity*(n_gas-1)


def n_c4f10(lam_nm):
    """C4F10 index at wavelength lam_nm [nm]: ARC simulation table, linear in photon energy.

    Outside the table (above 660 nm or below 200 nm) the end value is held constant.
    """
    return np.interp(HC/np.asarray(lam_nm, float), C4F10_E_EV, C4F10_N)


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


def xenon(P_bar, T=T_ROOM):
    """Xe gas at P_bar [bar] and T [K] (n_xe)."""
    return Medium('Xe %g bar' % P_bar, partial(n_xe, P_bar=P_bar, T=T))


def c4f10():
    """C4F10 with the ARC simulation table (n_c4f10)."""
    return Medium('C4F10 1 bar', n_c4f10)


def air():
    """Air with the ARC simulation index (n_air)."""
    return Medium('air', n_air)


def aerogel_constant(n=AEROGEL_N_AIR):
    """Aerogel with a constant index n: no dispersion, no pore-gas shift."""
    return Medium('aerogel n = %g' % n, ConstantIndex(n))
