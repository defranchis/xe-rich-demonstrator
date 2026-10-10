"""Geometric Monte Carlo models of the demonstrator setups, built on tracer, muons and analysis.

- aberration: exact-sphere imaging of a Cherenkov cone (spherical aberration).
- ring_rate_full, ring_rate_partial: cosmic gas-ring rates with a flat-mirror footprint.
- MirrorToy: 3D toy of the mirror setup with an aerogel tile under the sensor.
- proximity: proximity-focusing aerogel toy.
No photodetection efficiency, reflectivity, scattering or absorption is applied.
"""
import numpy as np

from . import tracer
from .analysis import azimuths, mc_rate, ring_fraction_in_square, ring_in_square
from .cherenkov import theta_c
from .constants import AERO_YIELD_PER_CM, GAS_YIELD_REF, GAS_YIELD_REF_LEN, MUON_THMAX
from .muons import flux_through_square, sample_box

UP = np.array([0, 0, 1.])
DOWN = np.array([0, 0, -1.])


def aberration(h0, tilt, rng, mirror, L, th, n=4000):
    """Reconstructed Cherenkov angles for one track imaged by an exact spherical mirror.

    Frame: centre of curvature at the origin, mirror vertex at z = +R, detector plane
    at z = R - f. The track starts on the detector plane at x = h0 [mm] with angle tilt
    [rad] to the axis; n photons at angle th are emitted at uniform path lengths in
    [0, L]. The angle is reconstructed as atan(r/f) with r measured from the ideal ring
    centre. Returns (angles [rad], radial distance of each mirror hit from the axis [mm]).
    """
    R = mirror.R; f = mirror.f
    d0 = np.array([np.sin(tilt), 0, np.cos(tilt)])
    rng.random(n)                         # unused draws keep the random sequence of the original toy
    s = rng.uniform(0, L, n)
    p0 = np.array([h0, 0, R-f])+s[:, None]*d0
    phi = rng.uniform(0, 2*np.pi, n)
    e1 = np.array([np.cos(tilt), 0, -np.sin(tilt)]); e2 = np.array([0, 1, 0])
    u = tracer.cone(d0[None], th, phi, e1[None], e2[None])[0]
    q, _ = tracer.sphere_far(p0, u, tracer.ORIGIN, R)
    rho = np.hypot(q[:, 0], q[:, 1])
    v = tracer.reflect(u, tracer.sphere_normal(q, tracer.ORIGIN, R))
    hit = tracer.to_plane(q, v, R-f)
    c0 = (R-f)/d0[2]*d0
    r = np.hypot(hit[:, 0]-c0[0], hit[:, 1]-c0[1])
    return np.arctan(r/f), rho


def footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi):
    """Fraction of Cherenkov photons landing within radius a of the mirror axis (flat mirror).

    Tracks hit the mirror plane at (x, y) with slopes (sx, sy); photons are emitted at
    ns heights over L and npsi azimuths. The footprint offset includes the term
    S (sx, sy) from the track slope.
    """
    s = (np.arange(ns)+0.5)/ns*L; psi = azimuths(npsi)
    S, P = np.meshgrid(s, psi, indexing='ij')
    hx = x[:, None, None]+S*sx[:, None, None]+S*np.tan(thc)*np.cos(P)
    hy = y[:, None, None]+S*sy[:, None, None]+S*np.tan(thc)*np.sin(P)
    return (hx**2+hy**2 < a*a).mean(axis=(1, 2))


def ring_rate_full(rng, mirror, sensor, L, thc, frac_req=1.0, N=400000, box_margin=0.6,
                   thmax=np.radians(35), ns=8, npsi=16):
    """Rate [1/h] of rings fully inside the sensor with >= frac_req of the photons on the mirror.

    The ring sits at f (sx, sy) on the sensor with radius f tan(thc). Also returns the
    mean collected photon fraction of the accepted muons.
    """
    a = mirror.clear_radius; f = mirror.f
    box = a+L*box_margin
    x, y, sx, sy, w = sample_box(rng, N, box, thmax)
    inside = footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi)
    rr = f*np.tan(thc); cx, cy = f*sx, f*sy
    acc = (inside >= frac_req-1e-9)&ring_in_square(cx, cy, rr, sensor.half)
    rate = mc_rate(w, acc, box, thmax)
    eff = np.sum(w*acc*inside)/max(np.sum(w*acc), 1e-12)
    return rate*3600, eff


def ring_rate_partial(rng, mirror, sensor, L, thc, req_m=0.8, req_s=0.8, N=300000, box_margin=0.4,
                      thmax=np.radians(25), ns=8, npsi=24):
    """Rates [1/h] of rings with >= req_m of the photons on the mirror and >= req_s of the ring on the sensor.

    Returns (rate with the requirements, rate with full containment on both).
    """
    a = mirror.clear_radius; f = mirror.f
    box = a+L*box_margin
    x, y, sx, sy, w = sample_box(rng, N, box, thmax)
    fm = footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi)
    rr = f*np.tan(thc); cx, cy = f*sx, f*sy
    fs = ring_fraction_in_square(cx, cy, rr, azimuths(npsi), sensor.half)
    acc = (fm >= req_m)&(fs >= req_s)
    return mc_rate(w, acc, box, thmax)*3600, mc_rate(w, (fm >= 0.999)&(fs >= 0.999), box, thmax)*3600


class MirrorToy:
    """3D toy of the mirror setup with an aerogel tile directly below the sensor.

    Frame: mirror vertex at z = 0, centre of curvature at z = R, sensor plane at the
    focal plane z = f, aerogel tile between z = f - T and z = f, Xe elsewhere. Muons
    enter at z = f going down. Photons are emitted at ns depths per radiator and nphi
    azimuths; aerogel photons leaving through the tile sides are lost.
    """

    def __init__(self, mirror, sensor, aerogel, n_gas, ns=8, nphi=90):
        self.R = mirror.R; self.f = mirror.f; self.amir = mirror.clear_radius
        self.C = np.array([0, 0, self.R])
        self.T = aerogel.thickness; self.tile = aerogel.half; self.sens = sensor.half
        self.na = aerogel.n; self.ng = n_gas
        self.tha = theta_c(self.na); self.thg = theta_c(self.ng)
        self.ns = ns; self.phi = azimuths(nphi)

    def image(self, p, d):
        """Trace downward photons (p, d) via the mirror to the sensor plane.

        Photons crossing the tile on the way up are refracted into it. Returns the
        sensor coordinates (..., 2) and the mask of photons that hit the clear aperture
        and the sensor.
        """
        q, hit = tracer.sphere_far(p, d, self.C, self.R)
        ok = hit&(np.hypot(q[..., 0], q[..., 1]) < self.amir)&(d[..., 2] < 0)
        d2 = tracer.reflect(d, tracer.sphere_normal(q, self.C, self.R))
        ok &= d2[..., 2] > 0
        h = tracer.to_plane(q, d2, self.f-self.T, valid=d2[..., 2] > 0)
        intile = (np.abs(h[..., 0]) < self.tile)&(np.abs(h[..., 1]) < self.tile)
        d3, _ = tracer.refract(d2, DOWN, self.ng/self.na)
        d3 = np.where(intile[..., None], d3, d2)
        e = tracer.advance(h, d3, self.T)
        ok &= (np.abs(e[..., 0]) < self.sens)&(np.abs(e[..., 1]) < self.sens)
        return e[..., :2], ok

    def aerogel_exit(self, pe, u, phi):
        """Aerogel photons from points pe (N,3) on tracks u (N,3) to the tile bottom face and into the gas.

        Returns exit points and gas directions (N,M,3) and the mask of photons going down,
        leaving through the bottom face and not totally reflected.
        """
        d = tracer.cone(u, self.tha, phi)
        down = d[..., 2] < 0
        pb = tracer.to_plane(pe[:, None, :], d, self.f-self.T, valid=down)
        dg, okr = tracer.refract(d, UP, self.na/self.ng)
        ok = down&(np.abs(pb[..., 0]) < self.tile)&(np.abs(pb[..., 1]) < self.tile)&okr
        return pb, dg, ok

    def gas_image(self, pe, u, phi):
        """Xe photons from points pe (N,3) on tracks u (N,3) imaged on the sensor."""
        d = tracer.cone(u, self.thg, phi)
        return self.image(np.broadcast_to(pe[:, None, :], d.shape), d)

    def run(self, N, sampler, chunk=2000):
        """Trace N muons from sampler(n) -> (u, p0, zenith [deg]) in chunks.

        Returns per-muon arrays: th (zenith, deg); fa, fg (fraction of the aerogel / gas
        photons reaching the sensor); arc_a, arc_g (fraction of the azimuths with at
        least one photon on the sensor); na, ng (emitted photons reaching the sensor,
        scaled from the fixed yields).
        """
        T = self.T; f = self.f; NS = self.ns; phi = self.phi
        res = {'th': [], 'fa': [], 'arc_a': [], 'na': [], 'fg': [], 'arc_g': [], 'ng': []}
        for i0 in range(0, N, chunk):
            u, p0, th = sampler(chunk)
            tt = ((np.arange(NS)+0.5)/NS)[None, :]*T/(-u[:, 2])[:, None]
            acc = np.zeros((chunk, NS, len(phi)), bool)
            for k in range(NS):
                pe = p0+tt[:, k:k+1]*u
                inside = (np.abs(pe[:, 0]) < self.tile)&(np.abs(pe[:, 1]) < self.tile)
                pb, dg, okb = self.aerogel_exit(pe, u, phi)
                xy, ok = self.image(pb, dg)
                acc[:, k, :] = okb&inside[:, None]&ok
            L = T/(-u[:, 2])
            nemit = AERO_YIELD_PER_CM*L/10.0
            fa = acc.mean((1, 2)); arc = acc.any(1).mean(1)
            zm = 0.0; Lg = ((f-T)-zm)/(-u[:, 2]); tg = (T/(-u[:, 2]))[:, None]+((np.arange(NS)+0.5)/NS)[None, :]*Lg[:, None]
            accg = np.zeros((chunk, NS, len(phi)), bool)
            for k in range(NS):
                pe = p0+tg[:, k:k+1]*u
                xy, ok = self.gas_image(pe, u, phi)
                accg[:, k, :] = ok
            fg = accg.mean((1, 2)); arcg = accg.any(1).mean(1); nemg = GAS_YIELD_REF*Lg/GAS_YIELD_REF_LEN
            for key, v in zip(res, [th, fa, arc, fa*nemit, fg, arcg, fg*nemg]):
                res[key].append(v)
        return {k: np.concatenate(v) for k, v in res.items()}


def proximity(aerogel, sensor, gap, n_gas, sampler, N=40000, ns=6, nphi=72, thmax=MUON_THMAX):
    """Proximity-focusing toy: aerogel tile (top face z = 0) above a sensor at z = -(T + gap).

    Muons from sampler(N) -> (u, p0, zenith) enter the top face. Returns the rates [1/h]
    of muons through the tile with >= 80 % of the photons on the sensor and with
    >= 50 % of the azimuths hit.
    """
    u, p0, th = sampler(N)
    p0[:, 2] = 0.0
    T = aerogel.thickness; tile = aerogel.half; sens = sensor.half
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False); acc = np.zeros((N, ns, nphi), bool)
    d = tracer.cone(u, theta_c(aerogel.n), phi)
    dg, okr = tracer.refract(d, UP, aerogel.n/n_gas)
    for k, fr in enumerate((np.arange(ns)+0.5)/ns):
        pe = p0+(fr*T/(-u[:, 2]))[:, None]*u
        pb = tracer.to_plane(pe[:, None, :], d, -T)
        okb = (np.abs(pb[..., 0]) < tile)&(np.abs(pb[..., 1]) < tile)&(np.abs(pe[:, None, 0]) < tile)&(np.abs(pe[:, None, 1]) < tile)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = tracer.advance(pb, dg, -gap)
        acc[:, k] = okb&okr&(np.abs(e[..., 0]) < sens)&(np.abs(e[..., 1]) < sens)
    rate = flux_through_square(2*tile*1e-3, thmax)*3600
    fa = acc.mean((1, 2)); arc = acc.any(1).mean(1)
    return rate*np.mean(fa >= .8), rate*np.mean(arc >= .5)
