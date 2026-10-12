"""Cherenkov angle, threshold, Frank-Tamm photon yield and spectrum sampling, SiPM quantum efficiency."""
import numpy as np

from .constants import AEROGEL_N_AIR, ALPHA, HBARC_EV_CM, HC, T_ROOM
from .media import N_AIR, XE_INDEX_DEFAULT, aerogel_index, aerogel_index_air, n_xe

# ARC SiPM quantum efficiency vs photon energy [eV], ARC_SiPM_QuantumEfficiency of the ARC simulation:
# k4geo e1ba7bc, FCCee/CLD/compact/CLD_o3_v01/materials_arc_o1_v01.xml:6
QE_ARC_E_EV = (4.9, 4.7, 4.5, 4.32, 4.147, 3.85, 3.647, 3.378, 3.163, 3.084, 3.002, 2.938, 2.837, 2.743, 2.655,
               2.465, 2.101, 1.771, 1.55, 1.3, 1.1)
QE_ARC = (0.0, 0.0, 0.1, 0.2, 0.41, 0.46, 0.47, 0.52, 0.59, 0.6, 0.6, 0.59, 0.57, 0.56, 0.53, 0.51, 0.4, 0.26, 0.13,
          0.0, 0.0)


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


def dN_dxdE(E_eV, p, m, P_bar, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """Frank-Tamm spectrum in Xe [eV^-1 cm^-1] at photon energy E_eV [eV]; model is the n_xe index model."""
    return frank_tamm(n_xe(HC/E_eV, P_bar, T, model), beta(p, m))


def dN_dxdlam(lam_nm, p, m, P_bar, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """Frank-Tamm spectrum in Xe [nm^-1 cm^-1] at wavelength lam_nm [nm]; model is the n_xe index model."""
    return dN_dxdE(HC/lam_nm, p, m, P_bar, T, model)*(HC/lam_nm**2)


def energy_grid(lam_lo, lam_hi, n):
    """n photon energies [eV] spanning the wavelength band lam_lo-lam_hi [nm]."""
    return np.linspace(HC/lam_hi, HC/lam_lo, n)


def photon_yield(E, p, m, P_bar, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """Emitted photons per cm in Xe integrated over the energy grid E [eV] (trapezoid rule); model is the n_xe index model."""
    return np.trapezoid(dN_dxdE(E, p, m, P_bar, T, model), E)


def qe_arc(E_eV):
    """ARC SiPM quantum efficiency at photon energy E_eV [eV], linear in energy, 0 outside the table."""
    return np.interp(E_eV, QE_ARC_E_EV[::-1], QE_ARC[::-1], left=0.0, right=0.0)


def qe_band():
    """Photon-energy band (E_lo, E_hi) [eV] where the ARC SiPM QE is non-zero, bounded by the adjacent zero entries."""
    E = np.array(QE_ARC_E_EV[::-1]); i = np.flatnonzero(np.array(QE_ARC[::-1]) > 0)
    return float(E[i[0]-1]), float(E[i[-1]+1])


def chromatic_rms(index, weight=qe_arc, band=None, ngrid=4001):
    """rms [rad] of the beta = 1 Cherenkov angle over the Frank-Tamm x weight spectrum (index n(lambda [nm]), band
    (E_lo, E_hi) [eV] default qe_band(), weight(E) default the ARC SiPM QE, None for pure FT)."""
    E = np.linspace(*(qe_band() if band is None else band), ngrid); ct = 1/index(HC/E)
    w = ALPHA/HBARC_EV_CM*(1-ct**2)
    if weight is not None:
        w = w*weight(E)
    th = np.arccos(ct)
    m = np.trapezoid(w*th, E)/np.trapezoid(w, E)
    return float(np.sqrt(np.trapezoid(w*(th-m)**2, E)/np.trapezoid(w, E)))


class PhotonSpectrum:
    """Frank-Tamm spectrum d2N/dx dE of a particle of velocity beta in a medium, over photon energies E_lo-E_hi [eV].

    index is n(lambda [nm]); an optional weight(E) (e.g. qe_arc) multiplies the spectrum. The spectrum is
    tabulated on ngrid energies: yield_per_mm is its trapezoid integral, sample() draws energies by inverse CDF.
    """

    def __init__(self, index, E_lo, E_hi, beta=1.0, weight=None, ngrid=20001):
        self.index = index; self.beta = beta
        self.E = np.linspace(E_lo, E_hi, ngrid)
        f = frank_tamm(index(HC/self.E), beta)
        self.pdf = f*weight(self.E) if weight is not None else f
        self.cdf = np.concatenate([[0.0], np.cumsum(0.5*(self.pdf[1:]+self.pdf[:-1])*np.diff(self.E))])
        self.yield_per_mm = self.cdf[-1]/10

    def sample(self, rng, n):
        """n photon energies [eV] drawn from the spectrum."""
        return np.interp(rng.random(n)*self.cdf[-1], self.cdf, self.E)


def ftqe_angle_yield(index, beta, weight=qe_arc, band=None, ngrid=4001):
    """Mean Cherenkov angle [rad] and photon yield [1/mm] over the Frank-Tamm x weight spectrum, per velocity in beta.

    index is n(lambda [nm]); band (E_lo, E_hi) [eV] defaults to qe_band(); weight(E) (default the ARC SiPM QE; None for
    pure FT) multiplies the spectrum. Below threshold the angle and the yield are 0.
    """
    E_lo, E_hi = qe_band() if band is None else band
    E = np.linspace(E_lo, E_hi, ngrid); n = index(HC/E)
    b = np.atleast_1d(np.asarray(beta, float))[:, None]
    ct = 1/(b*n[None, :]); w = np.where(ct < 1, ALPHA/HBARC_EV_CM*(1-ct**2), 0.0)
    if weight is not None:
        w = w*weight(E)[None, :]
    th = np.arccos(np.minimum(ct, 1)); Y = np.trapezoid(w, E, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(Y > 0, np.trapezoid(w*th, E, axis=1)/Y, 0.0)
    return mean, Y/10


def theta_xe(P_bar, beta=1.0, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """FT x QE-weighted mean Cherenkov angle [rad] in Xe at P_bar [bar] over the ARC SiPM QE band (264-954 nm)."""
    return float(ftqe_angle_yield(lambda lam: n_xe(lam, P_bar, T, model), beta)[0][0])


def pore_signal(P_bar, lam=400.0, n_tile=AEROGEL_N_AIR, T=T_ROOM, model=XE_INDEX_DEFAULT):
    """beta = 1 aerogel-angle shift [rad] from Xe at P_bar in the pores, relative to air in the pores, at lam [nm]:
    (n_tile = index in air, media.aerogel_index_air; n_tile = index with empty pores, media.aerogel_index)."""
    ng = n_xe(lam, P_bar, T, model)
    return tuple(float(theta_c(f(ng, n_tile))-theta_c(f(N_AIR, n_tile))) for f in (aerogel_index_air, aerogel_index))
