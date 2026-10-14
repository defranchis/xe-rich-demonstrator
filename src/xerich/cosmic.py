"""Momentum-aware cosmic-muon models: per-muon velocity in the fast gas-ring footprint model and in the 3D tracer.

- gas_ring_fractions: flat-mirror footprint and sensor containment of the gas ring with a per-muon Cherenkov angle;
  slope_term=True keeps the slope-term footprint of the legacy beta = 1 toys (sim.footprint_fraction), the default traces
  straight lines.
- emit_batch, trace_batch: Cherenkov photons of many muons at their own velocity in a Phase1Layout, traced with
  tracer.trace and summed per muon and origin (gas, tile).
No photodetection efficiency other than the QE weighting (ARC SiPM QE by default, S13361 PDE as an option), mirror
reflectivity 1 unless layout.Mirror.reflectivity sets one, no absorption. The tile side faces are black: tracer.trace
ends every photon that reaches a slab side face (reason 'tile sides'), also on its way back from the mirror (a black
holder frame). Polished side faces would add light, mostly off the ring, by an amount that depends on the face model.
"""
import numpy as np

from . import tracer
from .analysis import azimuths, pad_hits
from .cherenkov import qe_arc, qe_band
from .constants import ALPHA, HBARC_EV_CM, HC, PAD_KMAX
from .sim import footprint_fraction, footprint_fraction_tracks, track_segments

ORIGINS = ('gas', 'tile')
N_ARC = 36                   # azimuth bins of the arc coverage
MAX_SEGMENTS = 3             # track segments per origin handled by emit_batch
BOARD_KEYS = ('det_qe_plane', 'det_qe_ring', 'pads', 'pads_all')   # extra trace_batch sums of the board sensor model
REFL_KEYS = ('refl_qe', 'pads_refl')   # extra trace_batch sums with a board reflectance
REFL_GENS = 2                # board reflections followed
REFL_TH_CONE = 0.40          # diffuse sampling split [rad]: every board-to-mirror direction of configurations 1 and 5 lies inside
REFL_BLOCK = 200000          # incident photons reflected per block (memory)
# partial Fresnel reflections followed for the reflected light: none (negligible for the specular light; about -10 % of the
# diffuse part in configuration 5, i.e. about 0.004 photons per ring, at a large saving of branches)
REFL_MAX_REFLECTIONS = 0


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


def emit_batch(layout, rng, P0, U, beta, n_per, origins=ORIGINS, band=None, response=qe_arc):
    """Cherenkov photons of muons (entry points P0, directions U (M,3), velocities beta (M,)) in layout.

    n_per photons per muon and origin (int or {origin: int}), uniform in photon energy over band (default the ARC SiPM
    QE band), along the track and in azimuth, one photon per azimuth sector of 2 pi / n_per (every one of the N_ARC
    bins holds photons when n_per is a multiple of N_ARC or at least 2 N_ARC); emission angle cos = 1/(beta n(lambda)).
    Each photon carries w_ft, the Frank-Tamm photons per muon it stands for (0 below threshold), and qe, the photodetection
    response at its energy (response(E), default the ARC SiPM QE; cherenkov.response gives the options). Returns a dict
    of per-photon arrays p, d, lam, reg (start region), muon, origin (index into ORIGINS), phi (azimuth about the track),
    w_ft, qe.
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
        phi = 2*np.pi*(np.tile(np.arange(n_org), len(mu))+rng.random(n))/n_org   # one photon per sector of 2 pi / n_org
        e1, e2 = tracer.basis(U[idx])
        d = ct[:, None]*U[idx]+st[:, None]*(np.cos(phi)[:, None]*e1+np.sin(phi)[:, None]*e2)
        w = np.where(ok, (E_hi-E_lo)*ALPHA/HBARC_EV_CM*(1-ct*ct)*Lt[idx]/10/n_org, 0.0)
        out.append(dict(p=P0[idx]+s[:, None]*U[idx], d=d, lam=lam, reg=np.full(n, region[org], np.int16), muon=idx,
                        origin=np.full(n, oi, np.int8), phi=phi, w_ft=w, qe=response(E)))
    if not out:
        return None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def board_reflections(scene, rng, p, d, lam, w, spec, diff, gens=REFL_GENS, max_reflections=REFL_MAX_REFLECTIONS):
    """Light reflected by the board: photons landing at p (K, 3) on the sensor plane with directions d, wavelengths lam
    [nm] and weights w are re-launched from the plane through scene, specularly (d_z -> -d_z, reflectance spec) and
    diffusely (cosine-weighted, reflectance diff: one direction inside and one outside REFL_TH_CONE from the normal per
    photon, weighted by their shares sin^2 and cos^2 of the cone), for gens reflections; light landing beyond the
    sensor plane is not reflected; partial Fresnel reflections beyond max_reflections are dropped. Returns (index into
    the input photons, landing point, wavelength, weight) of every branch landing on the sensor plane, all generations
    together; rng draws the diffuse directions and the scattering.
    """
    out = []; src = np.arange(len(p)); s2 = np.sin(REFL_TH_CONE)**2
    for _ in range(gens):
        if not len(p):
            break
        sign = -np.sign(d[:, 2]); k = len(p); parts = []
        if spec > 0:
            ds = d.copy(); ds[:, 2] = -ds[:, 2]
            parts.append((np.arange(k), ds, w*spec))
        if diff > 0:
            u = np.r_[rng.random(k)*s2, s2+rng.random(k)*(1-s2)]    # sin^2 of the polar angle, cosine-weighted per part
            st = np.sqrt(u); ph = 2*np.pi*rng.random(2*k); idx = np.r_[np.arange(k), np.arange(k)]
            dd = np.stack([st*np.cos(ph), st*np.sin(ph), np.r_[sign, sign]*np.sqrt(1-u)], 1)
            parts.append((idx, dd, w[idx]*diff*np.r_[np.full(k, s2), np.full(k, 1-s2)]))
        idx, D, W = (np.concatenate([q[j] for q in parts]) for j in range(3))
        ends = tracer.trace(scene, p[idx], D, lam[idx], np.full(len(idx), -1, np.int16), None, max_reflections, rng=rng)
        det = ends['reason'] == tracer.REASONS.index('detected'); j = idx[ends['i'][det]]
        p, d, lam, w, src = ends['p'][det], ends['d'][det], lam[j], W[ends['i'][det]]*ends['w'][det], src[j]
        out.append((src, p, lam, w))
    if not out:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    return tuple(np.concatenate(x) for x in zip(*out))


def trace_batch(layout, rng, P0, U, beta, n_per, origins=ORIGINS, band=None, max_reflections=2, response=qe_arc, keep_muon_pad=False,
                refl=(0.0, 0.0), refl_rng=None):
    """Trace the photons of emit_batch and sum them per muon and origin (arrays of shape (M, len(ORIGINS))).

    Returns emit_ft, emit_qe (emitted FT and FT x QE photons), det_ft, det_qe (detected on the sensor), det_qe_clean
    (detected without Rayleigh scatter), out_ft (reaching the sensor plane outside the sensor), mirror_ft (lost at the
    mirror aperture; photons absorbed on a tile side face count in neither), arc (fraction of the N_ARC azimuth bins
    holding photons above threshold where a photon keeps half its weight on the sensor) and arc_half (fraction of those
    bins with at least half of their emitted weight on the sensor). Fractions: fm = 1 - mirror_ft/emit_ft (photons on
    the mirror), fs = det_ft/(det_ft + out_ft) (ring on the sensor), fd = det_ft/emit_ft. response(E) is the QE weight
    of emit_batch. A mirror reflectivity (layout.Mirror.reflectivity) lowers det_qe and det_qe_clean only: the FT sums,
    the fractions and the arcs are geometric. With the 'board' sensor model (layout.Sensor) the sensor is the board
    footprint for the FT sums, the fractions and the arcs, det_qe and det_qe_clean count the photons on active pads
    only (the muon's pad included), and BOARD_KEYS are added: det_qe_plane (photons without Rayleigh scatter on the
    footprint, active or not), det_qe_ring (those on the active pads, the pad crossed by the muon left out), pads and
    pads_all (mean fired pads, binary hits from the photons without Rayleigh scatter and from all photons, analysis.pad_hits) and
    pads_pge (shape (M, len(ORIGINS), PAD_KMAX + 1): P(N >= k) of the pads fired by photons without Rayleigh scatter).
    The pad hits are ring hits: the pad crossed by the muon on the sensor plane is left out (keep_muon_pad=True keeps it).
    The pad hits use the sampled photons as the expected counts per pad, so few photons per muon bias them low (concave
    1 - exp(-mu)); cosmic_rates.py samples cli.NPHOT_DEFAULT tile photons per muon by default.
    refl = (specular, diffuse) board reflectance (board model only): the light reaching the footprint is shared into
    detected (response), reflected (refl) and absorbed, so every other output is unchanged, and board_reflections
    re-launches the reflected part with its own generator refl_rng. REFL_KEYS are added: refl_qe (reflected photons
    detected on the active pads, response-weighted) and pads_refl (mean fired pads from the ring-hit photons and the
    reflected light together: a pad fired by both counts once); both leave out the muon's pad like the ring hits.
    """
    M = len(P0); no = len(ORIGINS); board = layout.sensor.board
    res = {k: np.zeros((M, no)) for k in ('emit_ft', 'emit_qe', 'det_ft', 'det_qe', 'det_qe_clean', 'out_ft', 'mirror_ft', 'arc',
                                          'arc_half')
           + (BOARD_KEYS if board is not None else ())}
    reflect = board is not None and (refl[0] > 0 or refl[1] > 0)
    if board is not None:
        res['pads_pge'] = np.zeros((M, no, PAD_KMAX+1)); res['pads_pge'][..., 0] = 1
    if reflect:
        if refl_rng is None:
            raise ValueError('a board reflectance needs its own random generator refl_rng')
        if layout.window is not None:   # the reflected light starts in the gas, and the measured reflectance includes the window
            raise ValueError('the board reflectance is not modelled with a window slab in the layout')
        res.update({k: np.zeros((M, no)) for k in REFL_KEYS})
    ph = emit_batch(layout, rng, P0, U, beta, n_per, origins, band, response)
    if ph is None:
        return res
    key = ph['muon']*no+ph['origin']
    res['emit_ft'] += np.bincount(key, ph['w_ft'], M*no).reshape(M, no)
    res['emit_qe'] += np.bincount(key, ph['w_ft']*ph['qe'], M*no).reshape(M, no)
    live = ph['w_ft'] > 0
    sel = {k: v[live] for k, v in ph.items() if k not in ('muon', 'origin')}; kl = key[live]
    del ph, key                                                 # only the live photons are traced (memory)
    if not live.any():
        return res
    ends = tracer.trace(layout.scene(), sel['p'], sel['d'], sel['lam'], sel['reg'], None, max_reflections, rng=rng)
    i = ends['i']; w = ends['w']*sel['w_ft'][i]; kk = kl[i]; reason = ends['reason']
    wt = ends['w']/ends['wr'] if 'wr' in ends else ends['w']   # branch weight without the mirror losses (geometry)
    wg = wt*sel['w_ft'][i]
    det = reason == tracer.REASONS.index('detected')
    res['det_ft'] += np.bincount(kk[det], wg[det], M*no).reshape(M, no)
    clean = ends['cls'] != tracer.CLASSES.index('scattered'); wq = w*sel['qe'][i]
    if board is not None:
        pad = board.pad_index(ends['p'][:, 0], ends['p'][:, 1])
        res['det_qe_plane'] += np.bincount(kk[det&clean], wq[det&clean], M*no).reshape(M, no)
        act = det&(pad >= 0)                                    # photons on active silicon
        sc = (layout.z_sensor-P0[:, 2])/U[:, 2]                # the muon on the sensor plane, its pad (-1: none)
        mpad = board.pad_index(P0[:, 0]+sc*U[:, 0], P0[:, 1]+sc*U[:, 1])
        ring = act if keep_muon_pad else act&(pad != mpad[kk//no])
        res['det_qe_ring'] += np.bincount(kk[ring&clean], wq[ring&clean], M*no).reshape(M, no)
        for name, m in (('pads_all', ring), ('pads', ring&clean)):
            mean, pge = pad_hits(kk[m], pad[m], wq[m], M*no)
            res[name] += mean.reshape(M, no)
            if name == 'pads':
                res['pads_pge'] = pge.reshape(M, no, -1)
        if reflect:
            inc = np.flatnonzero(det); scene = layout.scene(); kr, pr, qr = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0)]
            for b0 in range(0, len(inc), REFL_BLOCK):
                blk = inc[b0:b0+REFL_BLOCK]
                src, pe, le, we = board_reflections(scene, refl_rng, ends['p'][blk], ends['d'][blk], sel['lam'][i[blk]], w[blk], *refl)
                k_ = kk[blk][src]; pd = board.pad_index(pe[:, 0], pe[:, 1])
                on = (pd >= 0)&((pd != mpad[k_//no]) if not keep_muon_pad else True)
                kr.append(k_[on]); pr.append(pd[on]); qr.append(we[on]*response(HC/le[on]))
            kr, pr, qr = np.concatenate(kr), np.concatenate(pr), np.concatenate(qr)
            res['refl_qe'] += np.bincount(kr, qr, M*no).reshape(M, no)
            m = ring&clean
            res['pads_refl'] += pad_hits(np.r_[kk[m], kr], np.r_[pad[m], pr], np.r_[wq[m], qr], M*no)[0].reshape(M, no)
    else:
        act = det
    res['det_qe'] += np.bincount(kk[act], wq[act], M*no).reshape(M, no)
    cl = act&clean
    res['det_qe_clean'] += np.bincount(kk[cl], wq[cl], M*no).reshape(M, no)
    for name, rs in (('out_ft', tracer.REASONS.index('outside sensor')), ('mirror_ft', tracer.REASONS.index('mirror aperture'))):
        m = reason == rs
        res[name] += np.bincount(kk[m], wg[m], M*no).reshape(M, no)
    b = np.minimum((sel['phi'][i]/(2*np.pi)*N_ARC).astype(int), N_ARC-1)
    hitbin = np.zeros(M*no*N_ARC, bool); hitbin[(kk*N_ARC+b)[det&(wt >= 0.5)]] = True   # main branch only, not Fresnel ghosts
    b0 = np.minimum((sel['phi']/(2*np.pi)*N_ARC).astype(int), N_ARC-1)
    eb = np.bincount(kl*N_ARC+b0, sel['w_ft'], M*no*N_ARC)
    db = np.bincount((kk*N_ARC+b)[det], wg[det], M*no*N_ARC)
    nb = np.maximum((eb > 0).reshape(M, no, N_ARC).sum(2), 1)   # bins holding photons above threshold
    res['arc'] = hitbin.reshape(M, no, N_ARC).sum(2)/nb
    res['arc_half'] = ((eb > 0)&(db >= 0.5*eb)).reshape(M, no, N_ARC).sum(2)/nb
    return res


def fractions(res, origin):
    """(fm, fs, fd) per muon for origin ('gas' or 'tile') from trace_batch output; 0 where nothing was emitted."""
    o = ORIGINS.index(origin)
    e = res['emit_ft'][:, o]; d = res['det_ft'][:, o]; out = res['out_ft'][:, o]; mi = res['mirror_ft'][:, o]
    with np.errstate(invalid='ignore', divide='ignore'):
        fm = np.where(e > 0, 1-mi/e, 0.0); fs = np.where(d+out > 0, d/(d+out), 0.0); fd = np.where(e > 0, d/e, 0.0)
    return fm, fs, fd
