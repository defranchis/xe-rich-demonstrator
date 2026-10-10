"""Containment fractions, Monte Carlo rates, resolution terms and binning helpers."""
import numpy as np

from .constants import COSMIC_I0
from .tracer import REASONS

SCATTER_LAM_EDGES = np.arange(260.0, 961.0, 10.0)   # wavelength bins of the scattered-fraction histograms [nm]


def azimuths(n):
    """n equally spaced azimuths in [0, 2 pi)."""
    return np.arange(n)/n*2*np.pi


def ring_in_square(cx, cy, r, half):
    """True where the ring of radius r centred at (cx, cy) lies inside |x|,|y| < half."""
    return (np.abs(cx)+r < half)&(np.abs(cy)+r < half)


def ring_fraction_in_square(cx, cy, r, psi, half):
    """Fraction of the ring points at azimuths psi that fall inside |x|,|y| < half."""
    qx = cx[:, None]+r*np.cos(psi); qy = cy[:, None]+r*np.sin(psi)
    return ((np.abs(qx) < half)&(np.abs(qy) < half)).mean(1)


def mc_rate(w, acc, box, thmax, I0=COSMIC_I0):
    """Accepted muon rate [Hz] for sample_box muons: I0 x area x solid angle x <w acc>.

    box is the half side of the sampled square [mm], thmax the zenith cut [rad].
    """
    area = (2*box*1e-3)**2; dO = 2*np.pi*(1-np.cos(thmax))
    return I0*area*dO*np.mean(w*acc)


def pixel_sigma(pitch, f):
    """Per-photon angular resolution [rad] from the pixel size: pitch/sqrt(12)/f."""
    return pitch/np.sqrt(12)/f


def binned_mean(v, idx, nbins):
    """Mean of v in each bin, for bin indices idx (e.g. np.digitize(x, edges) - 1)."""
    return np.array([v[idx == b].mean() for b in range(nbins)])


def scattered_fraction(lam_end, oi):
    """Per wavelength bin: weight of origin oi with >= 1 Rayleigh scatter over the weight reaching the tile.

    lam_end: one weighting, shape (origin, reason, scattered, wavelength bin). Gas photons (oi = 0) lost at the
    mirror aperture without scattering never reach the tile and are left out of the denominator.
    """
    le = lam_end[oi]; tot = le.sum((0, 1))
    if oi == 0:
        tot = tot-le[REASONS.index('mirror aperture'), 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        return le[:, 1].sum(0)/tot
