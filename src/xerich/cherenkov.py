"""Cherenkov angle, threshold and Frank-Tamm photon yield."""
import numpy as np

from .constants import ALPHA, HBARC_EV_CM, HC, T_ROOM
from .media import n_xe


def beta(p, m):
    """Velocity of a particle with momentum p and mass m [GeV]."""
    return p/np.hypot(p, m)


def theta_c(n):
    """Cherenkov angle [rad] for beta = 1 in a medium of index n."""
    return np.arccos(1/n)


def theta_c_track(n, p, m):
    """Cherenkov angle [rad] for momentum p and mass m [GeV]; 0 below threshold."""
    return np.arccos(np.minimum(1, 1/(n*p/np.hypot(p, m))))


def p_threshold(m, n):
    """Cherenkov threshold momentum [GeV] for mass m [GeV] in a medium of index n."""
    return m/np.sqrt(n*n-1)


def frank_tamm(n, b):
    """Photon yield d2N/dx dE [eV^-1 cm^-1] for index n and velocity b."""
    s2 = 1-1/(b*n)**2
    return np.where(s2 > 0, ALPHA/HBARC_EV_CM*s2, 0.0)


def dN_dxdE(E_eV, p, m, P_bar, T=T_ROOM):
    """Frank-Tamm spectrum in Xe [eV^-1 cm^-1] at photon energy E_eV [eV]."""
    return frank_tamm(n_xe(HC/E_eV, P_bar, T), beta(p, m))


def dN_dxdlam(lam_nm, p, m, P_bar, T=T_ROOM):
    """Frank-Tamm spectrum in Xe [nm^-1 cm^-1] at wavelength lam_nm [nm]."""
    return dN_dxdE(HC/lam_nm, p, m, P_bar, T)*(HC/lam_nm**2)


def energy_grid(lam_lo, lam_hi, n):
    """n photon energies [eV] spanning the wavelength band lam_lo-lam_hi [nm]."""
    return np.linspace(HC/lam_hi, HC/lam_lo, n)


def photon_yield(E, p, m, P_bar, T=T_ROOM):
    """Emitted photons per cm in Xe integrated over the energy grid E [eV] (trapezoid rule)."""
    return np.trapezoid(dN_dxdE(E, p, m, P_bar, T), E)
