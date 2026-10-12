"""Geometric Monte Carlo models of the demonstrator setups, built on tracer, muons and analysis.

- aberration: exact-sphere imaging of a Cherenkov cone (spherical aberration).
- ring_rate_full, ring_rate_partial: cosmic gas-ring rates with a flat-mirror footprint.
- MirrorToy: 3D toy of the mirror setup with an aerogel tile under the sensor.
- proximity: proximity-focusing aerogel toy; proximity_fractions: its containment per muon at any Cherenkov angle.
- track_segments, emit_photons, muon_photons, onaxis_muon: Cherenkov photons of one muon
  in a Phase1Layout, sampled in wavelength and traced with Fresnel optics and optional Rayleigh
  scattering in the tile (tracer.trace); window_cluster: the muon's own hit on the SiPM.
No photodetection efficiency, reflectivity or absorption is applied.
"""
from dataclasses import replace

import numpy as np

from . import tracer
from .analysis import azimuths, mc_rate, ring_fraction_in_square, ring_in_square
from .cherenkov import theta_c
from .constants import AERO_YIELD_PER_CM, GAS_YIELD_REF, GAS_YIELD_REF_LEN, HC, MUON_THMAX
from .muons import flux_through_square, sample_box

UP = np.array([0, 0, 1.])
DOWN = np.array([0, 0, -1.])
CHUNK = 20000                # muons per vectorised block of the footprint model


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


def footprint_fraction_tracks(x, y, u, L, thc, a, ns, npsi):
    """Fraction of Cherenkov photons landing within radius a of the axis on the flat mirror plane z = 0.

    Tracks with unit directions u (N,3, downward) hit z = 0 at (x, y); photons at angle thc (scalar or per track) are
    emitted at ns heights over L and npsi azimuths and follow straight lines from their emission point. Photons not
    going down are counted as lost.
    """
    u = np.asarray(u, float); N = len(u); thc = np.broadcast_to(np.asarray(thc, float), (N,))
    S = (np.arange(ns)+0.5)/ns*L; psi = azimuths(npsi)
    e1, e2 = tracer.basis(u)
    out = np.empty(N)
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, i0+CHUNK)
        d = (np.cos(thc[sl])[:, None, None]*u[sl, None, :]
             + np.sin(thc[sl])[:, None, None]*(np.cos(psi)[None, :, None]*e1[sl, None, :]+np.sin(psi)[None, :, None]*e2[sl, None, :]))
        down = d[..., 2] < 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ox = d[..., 0]/-d[..., 2]-u[sl, None, 0]/-u[sl, None, 2]; oy = d[..., 1]/-d[..., 2]-u[sl, None, 1]/-u[sl, None, 2]
        hx = x[sl, None, None]+S[None, :, None]*ox[:, None, :]; hy = y[sl, None, None]+S[None, :, None]*oy[:, None, :]
        out[sl] = ((hx**2+hy**2 < a*a)&down[:, None, :]).mean(axis=(1, 2))
    return out


def footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi, slope_term=False):
    """Fraction of Cherenkov photons landing within radius a of the mirror axis (flat mirror).

    Tracks hit the mirror plane at (x, y) with slopes (sx, sy); photons are emitted at
    ns heights over L and npsi azimuths along straight tracks (footprint_fraction_tracks).
    slope_term=True: footprint of the legacy beta = 1 toys, which adds S (sx, sy) to the
    offset and so counts the track displacement twice.
    """
    if not slope_term:
        c = 1/np.sqrt(1+sx*sx+sy*sy)
        return footprint_fraction_tracks(x, y, np.stack([sx*c, sy*c, -c], 1), L, thc, a, ns, npsi)
    s = (np.arange(ns)+0.5)/ns*L; psi = azimuths(npsi)
    S, P = np.meshgrid(s, psi, indexing='ij')
    hx = x[:, None, None]+S*sx[:, None, None]+S*np.tan(thc)*np.cos(P)
    hy = y[:, None, None]+S*sy[:, None, None]+S*np.tan(thc)*np.sin(P)
    return (hx**2+hy**2 < a*a).mean(axis=(1, 2))


def ring_rate_full(rng, mirror, sensor, L, thc, frac_req=1.0, N=400000, box_margin=0.6,
                   thmax=np.radians(35), ns=8, npsi=16, slope_term=False):
    """Rate [1/h] of rings fully inside the sensor with >= frac_req of the photons on the mirror.

    The ring sits at f (sx, sy) on the sensor with radius f tan(thc). Also returns the
    mean collected photon fraction of the accepted muons. slope_term: see footprint_fraction.
    """
    a = mirror.clear_radius; f = mirror.f
    box = a+L*box_margin
    x, y, sx, sy, w = sample_box(rng, N, box, thmax)
    inside = footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi, slope_term)
    rr = f*np.tan(thc); cx, cy = f*sx, f*sy
    acc = (inside >= frac_req-1e-9)&ring_in_square(cx, cy, rr, sensor.half)
    rate = mc_rate(w, acc, box, thmax)
    eff = np.sum(w*acc*inside)/max(np.sum(w*acc), 1e-12)
    return rate*3600, eff


def ring_rate_partial(rng, mirror, sensor, L, thc, req_m=0.8, req_s=0.8, N=300000, box_margin=0.4,
                      thmax=np.radians(25), ns=8, npsi=24, slope_term=False):
    """Rates [1/h] of rings with >= req_m of the photons on the mirror and >= req_s of the ring on the sensor.

    Returns (rate with the requirements, rate with full containment on both). slope_term: see footprint_fraction.
    """
    a = mirror.clear_radius; f = mirror.f
    box = a+L*box_margin
    x, y, sx, sy, w = sample_box(rng, N, box, thmax)
    fm = footprint_fraction(x, y, sx, sy, L, thc, a, ns, npsi, slope_term)
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


def proximity_fractions(aerogel, sensor, gap, n_gas, u, p0, th, ns=6, nphi=72):
    """Per muon: fraction of the aerogel photons on the sensor and of the azimuths hit, in the geometry of proximity.

    Muons (directions u, entry points p0 (N,3) on the tile top face z = 0) radiate at angle th in the tile, scalar or per
    muon (0: no light). Returns (fa, arc).
    """
    N = len(u); th = np.broadcast_to(np.asarray(th, float), (N,))
    p0 = np.array(p0, float); p0[:, 2] = 0.0
    T = aerogel.thickness; tile = aerogel.half; sens = sensor.half
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False); acc = np.zeros((N, ns, nphi), bool)
    e1, e2 = tracer.basis(u)
    d = (np.cos(th)[:, None, None]*u[:, None, :]+np.sin(th)[:, None, None]*(np.cos(phi)[None, :, None]*e1[:, None, :]
                                                                          + np.sin(phi)[None, :, None]*e2[:, None, :]))
    dg, okr = tracer.refract(d, UP, aerogel.n/n_gas)
    for k, fr in enumerate((np.arange(ns)+0.5)/ns):
        pe = p0+(fr*T/(-u[:, 2]))[:, None]*u
        pb = tracer.to_plane(pe[:, None, :], d, -T, valid=d[..., 2] < 0)
        okb = (np.abs(pb[..., 0]) < tile)&(np.abs(pb[..., 1]) < tile)&(np.abs(pe[:, None, 0]) < tile)&(np.abs(pe[:, None, 1]) < tile)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = tracer.advance(pb, dg, -gap)
        acc[:, k] = okb&okr&(d[..., 2] < 0)&(th[:, None] > 0)&(np.abs(e[..., 0]) < sens)&(np.abs(e[..., 1]) < sens)
    return acc.mean((1, 2)), acc.any(1).mean(1)


def _box_interval(p0, u, lo, hi, s_end):
    """[(s0, s1)] where the track p0 + s u, 0 <= s <= s_end, lies in the box [lo, hi], or []."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    with np.errstate(divide='ignore'):
        t1 = (lo-p0)/u; t2 = (hi-p0)/u
    par = u == 0; within = (p0 >= lo)&(p0 <= hi)
    tn = np.max(np.where(par, np.where(within, -np.inf, np.inf), np.minimum(t1, t2)))
    tf = np.min(np.where(par, np.where(within, np.inf, -np.inf), np.maximum(t1, t2)))
    tn, tf = max(tn, 0.0), min(tf, s_end)
    return [(tn, tf)] if tf > tn else []


def track_segments(layout, p0, u):
    """Path-length intervals of the track p0 + s u (s >= 0) in the gas, the tile and the window, up to the mirror sphere.

    Returns {'gas': [(s0, s1), ...], 'tile': [(s0, s1)] or [], 'window': [(s0, s1)] or []}.
    """
    p0 = np.asarray(p0, float); u = np.asarray(u, float)
    m = p0-np.array([0, 0, layout.mirror.R]); b = m@u; c = m@m-layout.mirror.R**2
    s_end = -b+np.sqrt(b*b-c)
    tile, window = [], []
    if layout.tile is not None:
        h = layout.tile.half
        tile = _box_interval(p0, u, (-h, -h, layout.z_tile_bottom), (h, h, layout.z_tile_top), s_end)
    if layout.window is not None:
        h = layout.sensor.half
        window = _box_interval(p0, u, (-h, -h, layout.z_window_bottom), (h, h, layout.z_sensor), s_end)
    gas, s = [], 0.0
    for a, b_ in sorted(tile+window)+[(s_end, s_end)]:
        if a > s:
            gas.append((s, a))
        s = b_
    return {'gas': gas, 'tile': tile, 'window': window}


def emit_photons(rng, p0, u, segments, n, spectrum):
    """n Cherenkov photons from the track p0 + s u, with s uniform over the intervals segments in one medium.

    Photon energies come from spectrum (cherenkov.PhotonSpectrum), the emission angle from
    cos(theta) = 1/(beta n(lambda)) per photon, the azimuth is uniform and the polarisation lies
    in the plane of the track and the photon. Returns points, directions, wavelengths [nm] and
    polarisation vectors.
    """
    u = np.asarray(u, float)
    L = np.array([b-a for a, b in segments]); a0 = np.array([a for a, _ in segments])
    k = rng.choice(len(L), n, p=L/L.sum())
    s = a0[k]+rng.random(n)*L[k]
    lam = HC/spectrum.sample(rng, n)
    phi = rng.uniform(0, 2*np.pi, n)
    ct = 1/(spectrum.beta*spectrum.index(lam)); st = np.sqrt(1-ct**2)
    e1, e2 = tracer.basis(u[None])
    d = ct[:, None]*u+st[:, None]*(np.cos(phi)[:, None]*e1+np.sin(phi)[:, None]*e2)
    return np.asarray(p0, float)+s[:, None]*u, d, lam, tracer.cherenkov_field(u[None], d)


def muon_photons(layout, rng, p0, u, spectra, n_gas, n_tile, max_reflections=2, polarised=True, n_window=0,
                 record=False):
    """Trace the Cherenkov photons of one muon (track p0 + s u, s >= 0) through the layout.

    spectra = {'gas': PhotonSpectrum in the gas, 'tile': PhotonSpectrum in the tile, 'window': PhotonSpectrum
    in the window (needed only if n_window > 0)}; n_gas, n_tile and n_window photons are sampled from each.
    Returns (photons, ends): photons holds per sampled photon the wavelength lam [nm], origin (0 gas,
    1 tile, 2 window) and weight (emitted photons per muon that each sampled photon stands for); ends is
    the tracer.trace output, whose index i refers to photons. max_reflections, polarised (False:
    unpolarised Fresnel) and record are passed to tracer.trace, with rng for the Rayleigh scattering.
    """
    seg = track_segments(layout, p0, u)
    slab = {'tile': 0, 'window': int(layout.tile is not None)}
    parts, norm = [], []
    for org, key, n in ((0, 'gas', n_gas), (1, 'tile', n_tile), (2, 'window', n_window)):
        if n == 0 or not seg[key]:
            continue
        p, d, lam, E = emit_photons(rng, p0, u, seg[key], n, spectra[key])
        L = sum(b-a for a, b in seg[key])
        parts.append((p, d, lam, E, np.full(n, org, np.int8), np.full(n, -1 if org == 0 else slab[key], np.int16)))
        norm.append(np.full(n, spectra[key].yield_per_mm*L/n))
    p, d, lam, E, org, reg = (np.concatenate(x) for x in zip(*parts))
    ends = tracer.trace(layout.scene(), p, d, lam, reg, E if polarised else None, max_reflections, rng=rng, record=record)
    return {'lam': lam, 'origin': org, 'weight': np.concatenate(norm)}, ends


def onaxis_muon(layout, rng, spectra, n_gas, n_tile, max_reflections=2, polarised=True, n_window=0, record=False):
    """muon_photons for a vertical muon on the mirror axis entering at the sensor plane and moving towards the mirror."""
    return muon_photons(layout, rng, (0.0, 0.0, layout.z_sensor), (0.0, 0.0, -1.0), spectra, n_gas, n_tile,
                        max_reflections, polarised, n_window, record)


def window_cluster(layout, rng, spectrum, n, window=None, polarised=True, record=False):
    """The muon's own SiPM hit: Cherenkov light of a vertical on-axis muon in the SiPM window.

    The muon crosses the silicon, then the window (layout.window, else window, a layout.Window) on its way
    down; its light reaches the silicon after total internal reflection at the window bottom face.
    spectrum: PhotonSpectrum in the window; n sampled photons. Returns (photons, ends) as onaxis_muon,
    with only window photons (origin 2); the direct hit of the muon (constants.MIP_DIRECT_PE) is not included.
    """
    if layout.window is None:
        layout = replace(layout, window=window)
    return onaxis_muon(layout, rng, {'window': spectrum}, 0, 0, polarised=polarised, n_window=n, record=record)


def cluster_stats(ph, ends, weight=None):
    """Photons per muon reaching the sensor and their radial spread around the impact point (0, 0).

    weight(lam) (e.g. lambda lam: qe_arc(HC/lam)) multiplies each photon; returns the dict n (photons
    per muon on the sensor), r_mean, r_rms and r_max [mm] (weighted, about the axis).
    """
    det = ends['reason'] == 0; i = ends['i'][det]
    w = ph['weight'][i]*ends['w'][det]*(1.0 if weight is None else weight(ph['lam'][i]))
    r = np.hypot(ends['p'][det, 0], ends['p'][det, 1])
    return {'n': w.sum(), 'r_mean': np.average(r, weights=w), 'r_rms': np.sqrt(np.average(r**2, weights=w)),
            'r_max': r.max()}
