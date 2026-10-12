"""Momentum-aware cosmic-muon models: per-muon velocity in the fast gas-ring footprint model and in the 3D tracer.

- gas_ring_fractions: flat-mirror footprint and sensor containment of the gas ring with a per-muon Cherenkov angle;
  slope_term=True keeps the slope-term footprint of the legacy beta = 1 toys (sim.footprint_fraction), the default traces
  straight lines.
- emit_batch, trace_batch: Cherenkov photons of many muons at their own velocity in a Phase1Layout, traced with
  tracer.trace and summed per muon and origin (gas, tile).
No photodetection efficiency other than the ARC SiPM QE weighting, no mirror reflectivity, no absorption. The tile side
faces are black: tracer.trace ends every photon that reaches a slab side face (reason 'tile sides'), also on its way
back from the mirror (a black holder frame). Polished side faces would add light, mostly off the ring, by an amount
that depends on the face model.
"""
import numpy as np

from . import tracer
from .analysis import azimuths
from .cherenkov import qe_arc, qe_band
from .constants import ALPHA, HBARC_EV_CM, HC
from .sim import footprint_fraction, footprint_fraction_tracks, track_segments

ORIGINS = ('gas', 'tile')
N_ARC = 36                   # azimuth bins of the arc coverage
MAX_SEGMENTS = 3             # track segments per origin handled by emit_batch


def gas_ring_fractions(x, y, sx, sy, thc, mirror, sensor, L, ns=8, npsi=24, slope_term=False):
    """Per muon: fraction fm of the gas photons on the mirror and fraction fs of the ring on the sensor.

    Muons hit the mirror plane at (x, y) with slopes (sx, sy) = tan(zenith)(cos, sin)(azimuth); thc is the Cherenkov
    angle, scalar or per muon. The ring of radius f tan(thc) sits at f (sx, sy) on the sensor. slope_term=True uses
    sim.footprint_fraction with its slope term (footprint shifted by S (sx, sy), the legacy beta = 1 toys; scalar or per-muon thc).
    """
    a = mirror.clear_radius; f = mirror.f
    if slope_term:
        th = np.asarray(thc, float)
        fm = footprint_fraction(x, y, sx, sy, L, th if th.ndim == 0 else th[:, None, None], a, ns, npsi, slope_term=True)
    else:
        c = 1/np.sqrt(1+sx*sx+sy*sy)
        fm = footprint_fraction_tracks(x, y, np.stack([sx*c, sy*c, -c], 1), L, thc, a, ns, npsi)
    rr = f*np.tan(thc); psi = azimuths(npsi)
    rr = rr if np.ndim(rr) == 0 else rr[:, None]
    qx = f*sx[:, None]+rr*np.cos(psi); qy = f*sy[:, None]+rr*np.sin(psi)
    fs = ((np.abs(qx) < sensor.half)&(np.abs(qy) < sensor.half)).mean(1)
    return fm, fs


def emit_batch(layout, rng, P0, U, beta, n_per, origins=ORIGINS, band=None):
    """Cherenkov photons of muons (entry points P0, directions U (M,3), velocities beta (M,)) in layout.

    n_per photons per muon and origin (int or {origin: int}), uniform in photon energy over band (default the ARC SiPM
    QE band), along the track and in azimuth; emission angle cos = 1/(beta n(lambda)). Each photon carries w_ft, the Frank-Tamm photons per
    muon it stands for (0 below threshold), and qe, the ARC SiPM QE at its energy. Returns a dict of per-photon
    arrays p, d, lam, reg (start region), muon, origin (index into ORIGINS), phi (azimuth about the track), w_ft, qe.
    Raises ValueError for a track with more than MAX_SEGMENTS segments in one origin.
    """
    E_lo, E_hi = qe_band() if band is None else band
    index = {'gas': layout.gas.n, 'tile': layout.tile_index.n if layout.tile is not None else None}
    region = {'gas': -1, 'tile': 0}
    segs = [track_segments(layout, p, u) for p, u in zip(P0, U)]
    out = []
    for oi, org in enumerate(ORIGINS):
        if org not in origins or index[org] is None:
            continue
        n_org = n_per[org] if isinstance(n_per, dict) else n_per
        a0 = np.zeros((len(P0), MAX_SEGMENTS)); ln = np.zeros((len(P0), MAX_SEGMENTS))
        for m, sg in enumerate(segs):
            if len(sg[org]) > MAX_SEGMENTS:
                raise ValueError('more than %d track segments in the %s' % (MAX_SEGMENTS, org))
            for k, (a, b) in enumerate(sg[org]):
                a0[m, k] = a; ln[m, k] = b-a
        Lt = ln.sum(1); mu = np.flatnonzero(Lt > 0)
        if not len(mu):
            continue
        idx = np.repeat(mu, n_org); n = len(idx)
        cum = np.cumsum(ln[idx], 1)/Lt[idx, None]; r = rng.random(n)
        k = np.minimum((r[:, None] > cum).sum(1), MAX_SEGMENTS-1)
        s = a0[idx, k]+rng.random(n)*ln[idx, k]
        E = rng.uniform(E_lo, E_hi, n); lam = HC/E; nn = index[org](lam)
        ct = 1/(beta[idx]*nn); ok = ct < 1; ct = np.minimum(ct, 1.0); st = np.sqrt(1-ct*ct)
        phi = rng.uniform(0, 2*np.pi, n)
        e1, e2 = tracer.basis(U[idx])
        d = ct[:, None]*U[idx]+st[:, None]*(np.cos(phi)[:, None]*e1+np.sin(phi)[:, None]*e2)
        w = np.where(ok, (E_hi-E_lo)*ALPHA/HBARC_EV_CM*(1-ct*ct)*Lt[idx]/10/n_org, 0.0)
        out.append(dict(p=P0[idx]+s[:, None]*U[idx], d=d, lam=lam, reg=np.full(n, region[org], np.int16), muon=idx,
                        origin=np.full(n, oi, np.int8), phi=phi, w_ft=w, qe=qe_arc(E)))
    if not out:
        return None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def trace_batch(layout, rng, P0, U, beta, n_per, origins=ORIGINS, band=None, max_reflections=2):
    """Trace the photons of emit_batch and sum them per muon and origin (arrays of shape (M, len(ORIGINS))).

    Returns emit_ft, emit_qe (emitted FT and FT x QE photons), det_ft, det_qe (detected on the sensor), det_qe_clean
    (detected without Rayleigh scatter), out_ft (reaching the sensor plane outside the sensor), mirror_ft (lost at the
    mirror aperture; photons absorbed on a tile side face count in neither), arc (fraction of the N_ARC azimuth bins where
    a photon keeps half its weight on the sensor) and arc_half (fraction of the bins with at least half of their emitted
    weight on the sensor). Fractions: fm = 1 - mirror_ft/emit_ft (photons on the mirror), fs = det_ft/(det_ft + out_ft)
    (ring on the sensor), fd = det_ft/emit_ft.
    """
    M = len(P0); no = len(ORIGINS)
    res = {k: np.zeros((M, no)) for k in ('emit_ft', 'emit_qe', 'det_ft', 'det_qe', 'det_qe_clean', 'out_ft', 'mirror_ft', 'arc', 'arc_half')}
    ph = emit_batch(layout, rng, P0, U, beta, n_per, origins, band)
    if ph is None:
        return res
    key = ph['muon']*no+ph['origin']
    res['emit_ft'] += np.bincount(key, ph['w_ft'], M*no).reshape(M, no)
    res['emit_qe'] += np.bincount(key, ph['w_ft']*ph['qe'], M*no).reshape(M, no)
    live = ph['w_ft'] > 0
    sel = {k: v[live] for k, v in ph.items()}
    if not live.any():
        return res
    ends = tracer.trace(layout.scene(), sel['p'], sel['d'], sel['lam'], sel['reg'], None, max_reflections, rng=rng)
    i = ends['i']; w = ends['w']*sel['w_ft'][i]; kk = key[live][i]; reason = ends['reason']
    det = reason == tracer.REASONS.index('detected')
    res['det_ft'] += np.bincount(kk[det], w[det], M*no).reshape(M, no)
    res['det_qe'] += np.bincount(kk[det], (w*sel['qe'][i])[det], M*no).reshape(M, no)
    cl = det&(ends['cls'] != tracer.CLASSES.index('scattered'))
    res['det_qe_clean'] += np.bincount(kk[cl], (w*sel['qe'][i])[cl], M*no).reshape(M, no)
    for name, rs in (('out_ft', tracer.REASONS.index('outside sensor')), ('mirror_ft', tracer.REASONS.index('mirror aperture'))):
        m = reason == rs
        res[name] += np.bincount(kk[m], w[m], M*no).reshape(M, no)
    b = np.minimum((sel['phi'][i]/(2*np.pi)*N_ARC).astype(int), N_ARC-1)
    hitbin = np.zeros(M*no*N_ARC, bool); hitbin[(kk*N_ARC+b)[det&(ends['w'] >= 0.5)]] = True   # main branch only, not Fresnel ghosts
    res['arc'] = hitbin.reshape(M, no, N_ARC).mean(2)
    b0 = np.minimum((sel['phi']/(2*np.pi)*N_ARC).astype(int), N_ARC-1)
    eb = np.bincount(key[live]*N_ARC+b0, sel['w_ft'], M*no*N_ARC)
    db = np.bincount((kk*N_ARC+b)[det], w[det], M*no*N_ARC)
    res['arc_half'] = ((eb > 0)&(db >= 0.5*eb)).reshape(M, no, N_ARC).mean(2)
    return res


def fractions(res, origin):
    """(fm, fs, fd) per muon for origin ('gas' or 'tile') from trace_batch output; 0 where nothing was emitted."""
    o = ORIGINS.index(origin)
    e = res['emit_ft'][:, o]; d = res['det_ft'][:, o]; out = res['out_ft'][:, o]; mi = res['mirror_ft'][:, o]
    with np.errstate(invalid='ignore', divide='ignore'):
        fm = np.where(e > 0, 1-mi/e, 0.0); fs = np.where(d+out > 0, d/(d+out), 0.0); fd = np.where(e > 0, d/e, 0.0)
    return fm, fs, fd
