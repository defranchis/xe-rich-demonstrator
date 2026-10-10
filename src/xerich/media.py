"""Refractive-index models: Xe gas n(lambda, P, T) and aerogel with gas-filled pores."""
import numpy as np

from .constants import AEROGEL_N_AIR, AEROGEL_POROSITY, P0, R_GAS, T0, T_ROOM

# Xe second virial coefficient B(T) = B_XE_293 + B_XE_SLOPE (T - B_XE_TREF), approximate (Dymond & Smith)
B_XE_293 = -130.0            # [cm^3/mol]
B_XE_SLOPE = 0.9             # [cm^3 mol^-1 K^-1]
B_XE_TREF = 293.15           # reference temperature of the B(T) parametrisation [K]

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
