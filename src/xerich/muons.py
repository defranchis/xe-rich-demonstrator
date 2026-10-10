"""Cosmic-muon sampling for a cos^2(zenith) intensity.

Each sampler draws from the generator in a fixed order, so a given seed always
produces the same muons.
"""
import numpy as np

from .constants import COSMIC_I0


def sample_cos3(rng, N, half, z0, thmax):
    """N downward muons crossing the square |x|,|y| < half at height z0.

    The zenith angle follows cos^3 (cos^2 intensity times projected area) up to thmax,
    by accept-reject in batches of 4N. Returns directions (N,3), entry points (N,3)
    and zenith angles [deg].
    """
    out = []
    while sum(len(o) for o in out) < N:
        ct = rng.uniform(np.cos(thmax), 1, 4*N); keep = rng.uniform(0, 1, 4*N) < ct**3
        out.append(ct[keep])
    ct = np.concatenate(out)[:N]; st = np.sqrt(1-ct**2); ph = rng.uniform(0, 2*np.pi, N)
    u = np.stack([st*np.cos(ph), st*np.sin(ph), -ct], 1)
    x0 = rng.uniform(-half, half, N); y0 = rng.uniform(-half, half, N)
    return u, np.stack([x0, y0, np.full(N, z0)], 1), np.degrees(np.arccos(ct))


def redraw_xy(rng, p0, half):
    """Redraw the entry points p0 (N,3) in place, uniform over |x|,|y| < half (x first)."""
    n = len(p0)
    p0[:, 0] = rng.uniform(-half, half, n); p0[:, 1] = rng.uniform(-half, half, n)
    return p0


def sample_box(rng, N, box, thmax):
    """N muons with impact points uniform over |x|,|y| < box and directions uniform in solid angle up to thmax.

    Returns x, y, the transverse slopes sx, sy and the weight cos^3(zenith) that
    turns the uniform solid-angle sampling into the flux through a horizontal plane.
    """
    x = rng.uniform(-box, box, N); y = rng.uniform(-box, box, N)
    ct = rng.uniform(np.cos(thmax), 1, N); st = np.sqrt(1-ct**2); ph = rng.uniform(0, 2*np.pi, N)
    sx, sy = st*np.cos(ph)/ct, st*np.sin(ph)/ct
    w = ct**3
    return x, y, sx, sy, w


def flux_through_square(side_m, thmax, I0=COSMIC_I0):
    """Muon rate [Hz] through a horizontal square of side side_m [m] with zenith < thmax.

    Integral of I0 cos^2 x cos over the solid angle: I0 pi/2 A (1 - cos^4 thmax).
    """
    return I0*np.pi/2*side_m**2*(1-np.cos(thmax)**4)
