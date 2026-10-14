"""Aerogel ring imaged by a short-focal-length f/1 spherical mirror (configurations 2 and 3 of docs/design_options.md).

The 1 cm aerogel tile (constant n = 1.03, no pore-gas shift, Rayleigh scattering) sits 1 mm in front of the 2" x 2"
SiPM plane at the paraxial focal plane z = f of a spherical mirror, Edmund #43-543 (f = D = 50.8 mm) or #43-553
(f = D = 76.2 mm), in air at 1 atm or Xe at 1, 2 and 3 bar. Photons are sampled from the Frank-Tamm spectrum over the
band where the ARC SiPM QE is non-zero and counted with the FT x QE and the pure FT weighting (as hitmap_onaxis.py);
beta = 1, mirror reflectivity 1. Parts (--parts):
  onaxis   vertical muon on the mirror axis: aerogel and gas rings, path classes, loss reasons, the muon's own window
           cluster, pixel maps at 3.2 and 0.8 mm pitch, --nseeds master seeds (matched across mirrors and gases);
  terms    ring-width terms switched one at a time at matched seeds, plus a deterministic meridional trace with an
           ideal-focusing reference;
  zscan    ring radius and width vs the sensor position (tile moving with the sensor);
  angles   containment vs muon zenith angle, azimuth and impact offset at the tile mid-plane;
  tol      focal-length tolerance, aerogel index step (dr/dn), clear-aperture fraction.
Outputs in --outdir: short_f_<part>.npz, short_f_summary.json, short_f_*.png, short_f_*.csv; --replot redraws
from the npz files.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, to_rgb  # noqa: E402

from xerich import cli, provenance, sim, tracer  # noqa: E402
from xerich.analysis import pixel_sigma  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band, theta_c  # noqa: E402
from xerich.constants import (AEROGEL_HALF, AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_T, HC, MIP_DIRECT_PE,  # noqa: E402
                              MIRROR_CLEAR_FRAC, MIRROR_SHORT_F, P_REF, PIXEL_PITCH, PIXEL_PITCH_FINE, SENSOR_HALF_2IN)
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window  # noqa: E402
from xerich.media import Medium, ConstantIndex, Rayleigh, aerogel_medium, air, xenon  # noqa: E402
from xerich.plotting import AEROGEL, GAS, INK, MUTED, SCINT_EDGE, plt, save  # noqa: E402
from xerich.tracer import CLASSES, REASONS  # noqa: E402

MIRRORS = MIRROR_SHORT_F                                      # Edmund stock number: (focal length, diameter) [mm]
MIRROR_KEYS = tuple(MIRRORS)
GAS_NAMES = ('air', 'xe1', 'xe2', 'xe3')
GAS_LABEL = {'air': 'air 1 atm', 'xe1': 'Xe 1 bar', 'xe2': 'Xe 2 bar', 'xe3': 'Xe 3 bar'}
GAS_COLOUR = {'air': '#2a9d8f', 'xe1': '#8ab17d', 'xe2': '#1f6fb4', 'xe3': '#d1495b'}
REF_GAS = 'xe%g' % P_REF             # Xe at the reference pressure: on-axis figures, sensor-position and tolerance scans
MIRROR_STYLE = {'43-543': '-', '43-553': '--'}
ORIGINS = ('gas', 'aerogel', 'window')
WEIGHTINGS = ('ftqe', 'ft')
PITCHES = (PIXEL_PITCH, PIXEL_PITCH_FINE)   # pixel pitches [mm]: the prototype matrices and a finer alternative
R_DET, R_OUT = REASONS.index('detected'), REASONS.index('outside sensor')
NPH = 8                            # pixel-grid phases per axis (grid offsets k/NPH of a pitch)
PIX_WIN = 8.0                      # half side of the central region of the pixel maps [mm]
R_BIN = 0.01                       # radial histogram bin [mm]
R_MAX = 36.0                       # beyond the sensor half diagonal [mm]
CORE_HALF = 0.25                   # half window of the fine aerogel-ring histogram [mm]
CORE_BIN = 0.00025                 # its bin [mm]
GCORE_HALF = 1.0                   # half window of the fine gas-ring histogram [mm]
GCORE_BIN = 0.001                  # its bin [mm]
ZCORE_HALF = 1.5                   # half window of the fine aerogel-ring histogram in the sensor-position scan [mm]
ZCORE_BIN = 0.0015                 # its bin [mm]
MAP_BIN = 0.2                      # hit-map bin [mm]
ANG_BIN = 0.1                      # incidence-angle bin [deg]
ANG_MAX = 40.0                     # [deg]
NPHI = 36                          # emission-azimuth bins of the containment study
NARROW_TILE_HALF = 5.0             # tile half side that removes the return pass through the tile [mm]
DEPTH_MID = 0.5                    # single-depth emission: fraction of the track length in the tile
GROUPS = ((0,), (1, 2, 3, 4), (5,))  # class groups of the hit maps: direct, reflected at tile faces, scattered
GROUP_NAMES = ('direct', 'reflected', 'scattered')
ZENITH = np.arange(0.0, 25.5, 1.0)          # muon zenith angles of the containment study [deg]
OFFSET = np.arange(-20.0, 20.01, 2.5)       # impact offsets at the tile mid-plane [mm]
AZIMUTH = (0.0, 45.0)                        # muon azimuth: along a sensor edge, along the diagonal [deg]
OFFSET_DIRS = ('parallel', 'perpendicular')  # offset direction relative to the muon azimuth
CLEAR_SCAN = (0.8, 0.9, 1.0)                 # clear-aperture fractions of the zero-offset curves
DZ_SCAN = np.arange(-8.0, 8.01, 0.5)         # sensor (and tile) displacement from z = f [mm]
F_SCALE = (0.98, 1.0, 1.02)                  # focal-length tolerance +-2 %
DN_TILE = 0.001                              # aerogel index step for dr/dn
TERMS = ('full', 'norayl', 'mono_depth', 'mono_depth_noreturn', 'gasdisp', 'mono_point',
         'sp_point', 'k4_point', 'sp_full', 'k4_full')
_CFG = {}


# ---------------------------------------------------------------- setup

def gas_medium(name, fixed=False):
    """Medium of a gas name; fixed=True freezes its index at AEROGEL_LAMBDA_REF (no dispersion)."""
    g = air() if name == 'air' else xenon(float(name[2:]))
    if fixed:
        return Medium(g.label+' (n fixed)', ConstantIndex(float(g.n(AEROGEL_LAMBDA_REF))))
    return g


def make_layout(mkey, gname, *, rayleigh=True, gas_fixed=False, aerogel='constant', n_tile=AEROGEL_N_AIR,
                tile_half=AEROGEL_HALF, f_scale=1.0, clear=MIRROR_CLEAR_FRAC, dz=0.0, z_abs=None):
    """Configuration 2/3 layout: mirror mkey, gas gname, tile 1 mm below the 2" SiPM at z = f + dz (or z_abs)."""
    f, D = MIRRORS[mkey]
    mir = Mirror.from_focal(f*f_scale, D, clear_frac=clear)
    zs = z_abs if z_abs is not None else (None if dz == 0 else mir.f+dz)
    return Phase1Layout(mir, Sensor(SENSOR_HALF_2IN), gas_medium(gname, gas_fixed), Aerogel(n_tile, half=tile_half),
                        tile_medium=aerogel_medium(aerogel, n=n_tile), z_sensor_=zs,
                        tile_scatter=Rayleigh() if rayleigh else None)


def spectra_for(lay, band):
    lo, hi = band
    return {'gas': PhotonSpectrum(lay.gas.index, lo, hi), 'tile': PhotonSpectrum(lay.tile_index.index, lo, hi)}


def trace_track(lay, rng, p0, u, spectra, n_gas, n_tile, depth=None, max_refl=2):
    """sim.muon_photons (same random sequence) that also returns the emission directions; depth (fraction of the
    track length in the tile) puts all tile photons at one point, keeping the full-tile normalisation."""
    seg = sim.track_segments(lay, p0, u)
    parts, norm = [], []
    for org, key, n in ((0, 'gas', n_gas), (1, 'tile', n_tile)):
        if n == 0 or not seg[key]:
            continue
        sg = seg[key]; L = sum(b-a for a, b in sg)
        if depth is not None and key == 'tile':
            a, b = sg[0]; s = a+depth*(b-a); sg = [(s, s+1e-9)]
        p, d, lam, E = sim.emit_photons(rng, p0, u, sg, n, spectra[key])
        parts.append((p, d, lam, E, np.full(n, org, np.int8), np.full(n, -1 if org == 0 else 0, np.int16)))
        norm.append(np.full(n, spectra[key].yield_per_mm*L/n))
    if not parts:
        return None, None
    p, d, lam, E, org, reg = (np.concatenate(x) for x in zip(*parts))
    ends = tracer.trace(lay.scene(), p, d, lam, reg, E, max_refl, rng=rng)
    return {'lam': lam, 'origin': org, 'weight': np.concatenate(norm), 'd': d}, ends


# ---------------------------------------------------------------- deterministic meridional trace

def meridional(lay, depth, lam=AEROGEL_LAMBDA_REF, mirror='sphere', ret=True, n_tile=None):
    """Landing radius on the sensor plane of aerogel photons of a vertical on-axis muon, emitted at depth fractions
    depth (0 = tile top) at one wavelength, in the meridional plane (exact for the on-axis muon by symmetry).

    mirror 'sphere' reflects on the spherical cap; 'ideal' sends every ray from its mirror hit point to
    (f tan(theta), f), the image of an ideal focusing mirror. ret=False removes the return pass through the tile.
    Returns (radius on the sensor plane, height of the mirror hit) [mm].
    """
    na = lay.tile_index.n(np.atleast_1d(lam))[0] if n_tile is None else n_tile
    ng = float(lay.gas.n(np.atleast_1d(lam))[0])
    R = lay.mirror.R; f = lay.mirror.f; zt = lay.z_tile_top; zb = lay.z_tile_bottom; T = lay.tile.thickness
    ta = np.arccos(1/na); d = np.asarray(depth, float)*T
    x = (T-d)*np.tan(ta)                                          # radius on the tile bottom face
    sg = na*np.sin(ta)/ng; ux, uz = sg, -np.sqrt(1-sg*sg)
    qz = zb-R; b = x*ux+qz*uz; c = x*x+qz*qz-R*R; s = -b+np.sqrt(b*b-c)
    mx, mz = x+s*ux, zb+s*uz
    if mirror == 'sphere':
        nx, nz = mx/R, (mz-R)/R; dot = ux*nx+uz*nz
        vx, vz = ux-2*dot*nx, uz-2*dot*nz
    else:
        tx = f*sg/np.sqrt(1-sg*sg); vx, vz = tx-mx, f-mz
        nv = np.hypot(vx, vz); vx, vz = vx/nv, vz/nv
    px = mx+(zb-mz)/vz*vx
    if ret:
        sa = vx*ng/na; ca = np.sqrt(1-sa*sa)
        px = px+T*sa/ca
        sg2 = sa*na/ng; px = px+(lay.z_sensor-zt)*sg2/np.sqrt(1-sg2*sg2)
    else:
        px = px+(lay.z_sensor-zb)*vx/vz
    return px, mx


# ---------------------------------------------------------------- per-chunk tallies

def weights_of(ph):
    """(2, photons): FT x QE and FT weights."""
    w0 = ph['weight']
    return np.stack([w0*qe_arc(HC/ph['lam']), w0])


def tally(ph, e, ref_a, ref_g, core_half=CORE_HALF, core_bin=CORE_BIN, onaxis=False):
    """Weight sums of one traced chunk: emitted, loss reasons, classes, radial moments and histograms, fine
    histograms around the aerogel (ref_a) and gas (ref_g) ring radii; onaxis adds maps, pixel and angle tallies."""
    no, nr_, nc_ = len(ORIGINS), len(REASONS), len(CLASSES)
    ws = weights_of(ph); org = ph['origin']
    emit = np.array([[w[org == o].sum() for o in range(no)] for w in ws])
    o = org[e['i']].astype(np.intp); we = ws[:, e['i']]*e['w']
    assert e['reason'].max(initial=0) < nr_                               # no 'mirror absorption' (reflectivity 1)
    key = o*nr_+e['reason']
    reason = np.stack([np.bincount(key, we[k], minlength=no*nr_).reshape(no, nr_) for k in range(2)])
    det = e['reason'] == R_DET
    x, y = e['p'][det, 0], e['p'][det, 1]; r = np.hypot(x, y)
    od = o[det]; cl = e['cls'][det]; oc = od*nc_+cl; wd = we[:, det]
    mom = np.stack([np.stack([np.bincount(oc, wd[k]*r**p, minlength=no*nc_).reshape(no, nc_) for p in range(3)], -1)
                    for k in range(2)])
    nr = int(round(R_MAX/R_BIN)); ir = np.minimum((r/R_BIN).astype(int), nr-1)
    radial = np.stack([np.bincount(oc*nr+ir, wd[k], minlength=no*nc_*nr).reshape(no, nc_, nr) for k in range(2)])
    out = dict(emit=emit, reason=reason, mom=mom, radial=radial, nbranch=np.array(len(e['i'])))
    for name, oi, ref, half, bw in (('core_a', 1, ref_a, core_half, core_bin), ('core_g', 0, ref_g, GCORE_HALF, GCORE_BIN)):
        nb = int(round(2*half/bw)); m = od == oi
        ib = np.floor((r[m]-ref+half)/bw).astype(int); ok = (ib >= 0)&(ib < nb)
        dr = (cl[m] == 0)[ok]; ib = ib[ok]
        out[name] = np.stack([np.stack([np.bincount(ib[dr], wd[k, m][ok][dr], minlength=nb),
                                        np.bincount(ib, wd[k, m][ok], minlength=nb)]) for k in range(2)])
    if not onaxis:
        return out
    wq = wd[0]
    nb = int(round(2*SENSOR_HALF_2IN/MAP_BIN))
    ix = np.clip(((x+SENSOR_HALF_2IN)/MAP_BIN).astype(int), 0, nb-1); iy = np.clip(((y+SENSOR_HALF_2IN)/MAP_BIN).astype(int), 0, nb-1)
    grp = np.zeros(len(cl), np.intp)
    for gi, cs in enumerate(GROUPS):
        grp[np.isin(cl, cs)] = gi
    out['map'] = np.bincount(((od*len(GROUPS)+grp)*nb+ix)*nb+iy, wq, minlength=no*len(GROUPS)*nb*nb).reshape(no, len(GROUPS), nb, nb)
    ph_ = (np.arange(NPH)/NPH)
    for pi, p in enumerate(PITCHES):
        W = p*np.ceil(PIX_WIN/p); npx = int(round(2*W/p))+1
        cen = np.abs(x) < W+p
        cen &= np.abs(y) < W+p
        hx = np.floor((x[cen, None]+W+ph_[None, :]*p)/p).astype(int); hy = np.floor((y[cen, None]+W+ph_[None, :]*p)/p).astype(int)
        hist = np.zeros((no, NPH, NPH, npx, npx))
        for a in range(NPH):
            for b in range(NPH):
                okp = (hx[:, a] >= 0)&(hx[:, a] < npx)&(hy[:, b] >= 0)&(hy[:, b] < npx)
                idx = (od[cen][okp]*npx+hx[okp, a])*npx+hy[okp, b]
                hist[:, a, b] = np.bincount(idx, wq[cen][okp], minlength=no*npx*npx).reshape(no, npx, npx)
        out['pix%d' % pi] = hist
        pm = np.zeros((2, NPH*NPH, 3))
        for oi in (0, 1):
            m = (od == oi)&(cl == 0)
            cx = (np.floor((x[m, None]+ph_[None, :]*p)/p)+0.5)*p-ph_[None, :]*p
            cy = (np.floor((y[m, None]+ph_[None, :]*p)/p)+0.5)*p-ph_[None, :]*p
            rp = np.hypot(cx[:, :, None], cy[:, None, :]).reshape(m.sum(), -1)
            pm[oi] = np.stack([wq[m]@rp**k for k in range(3)], -1)
        out['pixmom%d' % pi] = pm
    inc = np.degrees(np.arccos(np.clip(np.abs(e['d'][det, 2]), 0, 1)))
    na_ = int(round(ANG_MAX/ANG_BIN)); ia = np.minimum((inc/ANG_BIN).astype(int), na_-1)
    out['inc'] = np.stack([np.bincount(od*na_+ia, wq*(cl == 0), minlength=no*na_).reshape(no, na_),
                           np.bincount(od*na_+ia, wq, minlength=no*na_).reshape(no, na_)], 1)
    return out


def run_onaxis_like(task):
    """Chunk of a vertical on-axis muon; task = (part, key, chunk index)."""
    part, key, ci = task
    c = _CFG; spec = c['jobs'][(part, key)]
    lay = spec['layout']; rng = np.random.default_rng(spec['seeds'][ci])
    p0 = (0.0, 0.0, lay.z_sensor); u = (0.0, 0.0, -1.0)
    ph, e = trace_track(lay, rng, p0, u, spec['spectra'], spec['n_gas'], spec['n_tile'], spec.get('depth'))
    if spec.get('cluster'):
        phw, ew = sim.window_cluster(lay, rng, spec['wspec'], spec['n_tile'], spec['cluster'])
        ew['i'] = ew['i']+len(ph['lam'])
        ph = {k: np.concatenate([ph[k], phw[k]]) for k in ('lam', 'origin', 'weight')}
        e = {k: np.concatenate([e[k], ew[k]]) for k in e}
    return tally(ph, e, spec['ref_a'], spec['ref_g'], spec.get('core_half', CORE_HALF), spec.get('core_bin', CORE_BIN),
                 onaxis=part == 'onaxis')


def run_angle(task):
    """One muon direction and impact point: aerogel-ring containment numbers; task = (key, )."""
    key = task
    c = _CFG; spec = c['jobs'][('angles', key)]
    lay = spec['layout']; mk, gn, cf, az, od, zi, oi = key
    th = np.radians(ZENITH[zi]); phi = np.radians(AZIMUTH[az])
    u = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), -np.cos(th)])
    ed = np.array([np.cos(phi), np.sin(phi), 0.0]) if od == 0 else np.array([-np.sin(phi), np.cos(phi), 0.0])
    zmid = 0.5*(lay.z_tile_top+lay.z_tile_bottom)
    pm = OFFSET[oi]*ed+np.array([0, 0, zmid])
    p0 = pm-(lay.z_sensor-zmid)/(-u[2])*u
    rng = np.random.default_rng(spec['seed'])
    res = dict(p0=p0[:2].copy(), stats=np.full(24, np.nan), phi=np.zeros((4, NPHI)), reason=np.zeros(len(REASONS)))
    ph, e = trace_track(lay, rng, p0, u, spec['spectra'], 0, spec['n_tile'])
    if ph is None:
        res['stats'][0] = 0.0
        return key, res
    ws = weights_of(ph); wq = ws[0]
    e1, e2 = tracer.basis(u[None])
    ang = np.mod(np.arctan2(ph['d']@e2[0], ph['d']@e1[0]), 2*np.pi); ib = np.minimum((ang/(2*np.pi)*NPHI).astype(int), NPHI-1)
    i = e['i']; we = wq[i]*e['w']; wf = ws[1, i]*e['w']
    det = e['reason'] == R_DET; dr = e['cls'] == 0
    plane = dr&((e['reason'] == R_DET)|(e['reason'] == R_OUT))
    assert e['reason'].max(initial=0) < len(REASONS)                      # no 'mirror absorption' (reflectivity 1)
    res['reason'] = np.bincount(e['reason'], we, minlength=len(REASONS))
    res['phi'][0] = np.bincount(ib, wq, minlength=NPHI)
    res['phi'][1] = np.bincount(ib[i[det&dr]], we[det&dr], minlength=NPHI)
    res['phi'][2] = np.bincount(ib[i[plane]], we[plane], minlength=NPHI)
    res['phi'][3] = np.bincount(ib[i[det]], we[det], minlength=NPHI)
    s = res['stats']
    s[0] = wq.sum(); s[1] = ws[1].sum(); s[2] = we[det].sum(); s[3] = we[det&dr].sum(); s[4] = wf[det].sum(); s[5] = wf[det&dr].sum()
    m = plane
    if we[m].sum() > 0 and m.sum() > 10:
        X, Y = e['p'][m, 0], e['p'][m, 1]; w = we[m]
        A = np.stack([X, Y, np.ones_like(X)], 1)*np.sqrt(w)[:, None]; bb = -(X*X+Y*Y)*np.sqrt(w)
        D, E_, F = np.linalg.lstsq(A, bb, rcond=None)[0]
        cx, cy = -D/2, -E_/2; s[6], s[7] = cx, cy; s[8] = np.sqrt(max(cx*cx+cy*cy-F, 0))
        rr = np.hypot(X-cx, Y-cy); s[9] = np.sqrt(np.average((rr-s[8])**2, weights=w))
    m = det&dr
    if we[m].sum() > 0:
        inc = np.degrees(np.arccos(np.clip(np.abs(e['d'][m, 2]), 0, 1))); w = we[m]; o = np.argsort(inc)
        cw = np.cumsum(w[o])/w.sum()
        s[10] = np.average(inc, weights=w); s[11] = np.sqrt(np.average((inc-s[10])**2, weights=w))
        s[12] = inc[o][np.searchsorted(cw, 0.5)]; s[13] = inc[o][min(np.searchsorted(cw, 0.95), len(o)-1)]; s[14] = inc.max()
    seg = sim.track_segments(lay, p0, u)
    s[15] = sum(b-a for a, b in seg['tile'])
    s[16] = float((np.abs(p0[0]) < SENSOR_HALF_2IN)&(np.abs(p0[1]) < SENSOR_HALF_2IN))
    return key, res


def run_task(task):
    if task[0] == 'angles':
        return run_angle(task[1])
    return run_onaxis_like(task)


def reduce_into(acc, res):
    for k, v in res.items():
        acc[k] = acc[k]+v if k in acc else np.array(v, float)
    return acc


# ---------------------------------------------------------------- job definitions

def ref_radii(lay):
    """Deterministic aerogel-ring radius (mean over depth) and paraxial gas-ring radius f tan(theta_gas) at 400 nm."""
    ra = meridional(lay, (np.arange(200)+0.5)/200)[0].mean()
    rg = lay.mirror.f*np.tan(theta_c(float(lay.gas.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])))
    return float(ra), float(rg)


def build_jobs(args, parts, band):
    jobs, tasks = {}, []
    win = Window(); wspec = PhotonSpectrum(win.medium.index, *band)
    nchunk = max(1, int(np.ceil(args.nphot/args.chunk)))

    def add(part, key, lay, nphot, seed, n_gas=None, **kw):
        nc = max(1, int(np.ceil(nphot/args.chunk)))
        ra, rg = ref_radii(lay)
        jobs[(part, key)] = dict(layout=lay, spectra=spectra_for(lay, band), seeds=np.random.SeedSequence(seed).spawn(nc),
                                 n_gas=args.chunk if n_gas is None else n_gas, n_tile=args.chunk, ref_a=ra, ref_g=rg,
                                 wspec=wspec, **kw)
        tasks.extend((part, key, ci) for ci in range(nc))

    if 'onaxis' in parts:
        for s in range(args.nseeds):
            for mk in MIRROR_KEYS:
                for gn in GAS_NAMES:
                    add('onaxis', (mk, gn, s), make_layout(mk, gn), args.nphot, args.seed+s, cluster=win)
    if 'terms' in parts:
        for mk in MIRROR_KEYS:
            for gn in GAS_NAMES:
                for t in TERMS:
                    kw = dict(rayleigh=t in ('full', 'sp_full', 'k4_full'),
                              gas_fixed=t in ('mono_depth', 'mono_depth_noreturn', 'mono_point', 'sp_point', 'k4_point'),
                              aerogel={'sp': 'single-pole', 'k4': 'k4geo'}.get(t[:2], 'constant'),
                              tile_half=NARROW_TILE_HALF if t == 'mono_depth_noreturn' else AEROGEL_HALF)
                    depth = DEPTH_MID if t in ('gasdisp', 'mono_point', 'sp_point', 'k4_point') else None
                    add('terms', (mk, gn, t), make_layout(mk, gn, **kw), args.nphot_terms, args.seed, depth=depth,
                        core_half=ZCORE_HALF if t[:2] in ('sp', 'k4') else CORE_HALF,
                        core_bin=ZCORE_BIN if t[:2] in ('sp', 'k4') else CORE_BIN)
    if 'zscan' in parts:
        for mk in MIRROR_KEYS:
            for gn in ('air', REF_GAS):
                for zi, dz in enumerate(DZ_SCAN):
                    add('zscan', (mk, gn, zi), make_layout(mk, gn, rayleigh=False, dz=dz), args.nphot_zscan, args.seed,
                        core_half=ZCORE_HALF, core_bin=ZCORE_BIN)
    if 'tol' in parts:
        for mk in MIRROR_KEYS:
            f0 = MIRRORS[mk][0]
            for gn in ('air', REF_GAS):
                for fi, fs in enumerate(F_SCALE):
                    add('tol', (mk, gn, 'focus', fi), make_layout(mk, gn, rayleigh=False, f_scale=fs), args.nphot_zscan, args.seed,
                        core_half=ZCORE_HALF, core_bin=ZCORE_BIN)
                    add('tol', (mk, gn, 'nominal', fi), make_layout(mk, gn, rayleigh=False, f_scale=fs, z_abs=f0), args.nphot_zscan,
                        args.seed, core_half=ZCORE_HALF, core_bin=ZCORE_BIN)
                add('tol', (mk, gn, 'dn', 0), make_layout(mk, gn, rayleigh=False, n_tile=AEROGEL_N_AIR+DN_TILE), args.nphot_zscan,
                    args.seed, core_half=ZCORE_HALF, core_bin=ZCORE_BIN)
    if 'angles' in parts:
        cache = {}
        for mk in MIRROR_KEYS:
            for gn in GAS_NAMES:
                for cfi, cf in enumerate(CLEAR_SCAN):
                    lay = make_layout(mk, gn, clear=cf); sp = spectra_for(lay, band)
                    cache[(mk, gn, cfi)] = (lay, sp)
        for (mk, gn, cfi), (lay, sp) in cache.items():
            for az in range(len(AZIMUTH)):
                for od in range(len(OFFSET_DIRS)):
                    for zi in range(len(ZENITH)):
                        for oi in range(len(OFFSET)):
                            if CLEAR_SCAN[cfi] != MIRROR_CLEAR_FRAC and (od != 0 or OFFSET[oi] != 0):
                                continue
                            key = (mk, gn, cfi, az, od, zi, oi)
                            jobs[('angles', key)] = dict(layout=lay, spectra=sp, n_tile=args.nphot_angle, seed=args.seed)
                            tasks.append(('angles', key))
    return jobs, tasks, nchunk


# ---------------------------------------------------------------- statistics helpers

def moments(mom, k, oi, classes):
    """(sum w, mean r, rms r) of origin oi over the class list, weighting k."""
    s = mom[k, oi, list(classes)].sum(0)
    if s[0] <= 0:
        return 0.0, np.nan, np.nan
    m = s[1]/s[0]
    return float(s[0]), float(m), float(np.sqrt(max(s[2]/s[0]-m*m, 0)))


def fwhm(h, bw):
    """FWHM of a histogram (linear interpolation of the outermost half-maximum crossings around the peak)."""
    if h.max() <= 0:
        return np.nan
    i = int(np.argmax(h)); half = h[i]/2
    lo = i
    while lo > 0 and h[lo-1] >= half:
        lo -= 1
    hi = i
    while hi < len(h)-1 and h[hi+1] >= half:
        hi += 1
    xl = lo-(h[lo]-half)/(h[lo]-h[lo-1]) if lo > 0 else lo
    xh = hi+(h[hi]-half)/(h[hi]-h[hi+1]) if hi < len(h)-1 else hi
    return float((xh-xl)*bw)


def central_halfwidth(h, bw, q=0.6827):
    """Half width of the central interval holding the fraction q of the histogram."""
    c = np.cumsum(h)/h.sum(); x = (np.arange(len(h))+1)*bw
    return float(0.5*(np.interp(0.5+q/2, c, x)-np.interp(0.5-q/2, c, x)))


def rebin(h, k):
    n = h.shape[-1]//k*k
    return h[..., :n].reshape(*h.shape[:-1], -1, k).sum(-1)


def ring_numbers(a, ref_a, ref_g, k=0, core_half=CORE_HALF, core_bin=CORE_BIN):
    """Aerogel and gas ring numbers from reduced (per-muon) tallies, weighting k."""
    out = {}
    na, ma, sa = moments(a['mom'], k, 1, [0])
    out['aero_direct_n'], out['aero_r_mean'], out['aero_r_rms'] = na, ma, sa
    ca = a['core_a'][k]; rb = max(1, int(round(0.001/core_bin)))
    out['aero_fwhm'] = fwhm(rebin(ca[0], rb), core_bin*rb)
    out['aero_fwhm_all'] = fwhm(rebin(ca[1], rb), core_bin*rb)
    out['aero_hw68'] = central_halfwidth(ca[0], core_bin)
    rad = a['radial'][k, 1].sum(0); rc = (np.arange(len(rad))+0.5)*R_BIN; tot = rad.sum()
    out['aero_sipm_all'] = float(tot)
    for w in (0.05, 0.4, 1.6):
        out['aero_tail_out_%g' % w] = float(rad[np.abs(rc-ma) > w].sum()/tot) if tot > 0 else np.nan
    ng_, mg, sg = moments(a['mom'], k, 0, [0])
    out['gas_direct_n'], out['gas_r_mean'], out['gas_r_rms'] = ng_, mg, sg
    cg = a['core_g'][k]
    out['gas_fwhm'] = fwhm(rebin(cg[0], 5), GCORE_BIN*5) if cg[0].sum() > 0 else np.nan
    out['gas_hw68'] = central_halfwidth(cg[0], GCORE_BIN) if cg[0].sum() > 0 else np.nan
    out['gas_sipm_all'] = float(a['radial'][k, 0].sum())
    out['emit'] = a['emit'][k].tolist()
    out['cls'] = (a['radial'][k].sum(-1)).tolist()
    out['reason_frac'] = (a['reason'][k]/np.where(a['emit'][k][:, None] > 0, a['emit'][k][:, None], 1)).tolist()
    return out


# ---------------------------------------------------------------- analysis and figures per part

def analyse_onaxis(out, acc, nseeds, nchunk, args, summary):
    """On-axis ring numbers and figures."""
    S = summary.setdefault('onaxis', {}); S['N'] = dict(photons_per_origin_per_seed=nchunk*args.chunk, seeds=nseeds)
    comb, per_seed = {}, {}
    for (mk, gn, s), a in acc.items():
        comb[(mk, gn)] = reduce_into(comb.get((mk, gn), {}), a)
        per_seed[(mk, gn, s)] = a
    for k in comb:
        for q in comb[k]:
            if q != 'nbranch':
                comb[k][q] = comb[k][q]/nseeds
    rows = []
    for mk in MIRROR_KEYS:
        for gn in GAS_NAMES:
            a = comb[(mk, gn)]; lay = make_layout(mk, gn); ra, rg = ref_radii(lay)
            d = {}
            for k, wn in enumerate(WEIGHTINGS):
                d[wn] = ring_numbers(a, ra, rg, k)
            # per-seed ring radii for the spread
            d['seed_aero_r'] = [moments(per_seed[(mk, gn, s)]['mom'], 0, 1, [0])[1] for s in range(nseeds)]
            d['seed_gas_r'] = [moments(per_seed[(mk, gn, s)]['mom'], 0, 0, [0])[1] for s in range(nseeds)]
            d['seed_aero_n'] = [per_seed[(mk, gn, s)]['radial'][0, 1].sum() for s in range(nseeds)]
            d['trace_r'] = ra; f = lay.mirror.f
            na_ = AEROGEL_N_AIR; ng = float(lay.gas.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])
            ta = theta_c(na_); tg = np.arcsin(na_*np.sin(ta)/ng)
            d['theta_aero'] = float(ta); d['theta_gas'] = float(tg); d['paraxial_r'] = float(f*np.tan(tg))
            d['paraxial_gas_ring'] = rg
            # window cluster and incidence angles
            win = a['radial'][0, 2]; rc = (np.arange(win.shape[-1])+0.5)*R_BIN
            d['window_n'] = float(win.sum()); wsum = win.sum(0)
            d['window_r_max'] = float(rc[np.flatnonzero(wsum > 0)].max()+R_BIN/2) if wsum.sum() > 0 else np.nan
            d['window_r_mean'] = float((wsum*rc).sum()/wsum.sum())
            inc = a['inc']; ac = (np.arange(inc.shape[-1])+0.5)*ANG_BIN
            for oi, nm in ((1, 'aero'), (0, 'gas')):
                h = inc[oi, 0]
                if h.sum() > 0:
                    cw = np.cumsum(h)/h.sum()
                    d['inc_%s_mean' % nm] = float((h*ac).sum()/h.sum()); d['inc_%s_p95' % nm] = float(np.interp(0.95, cw, ac))
                    d['inc_%s_max' % nm] = float(ac[np.flatnonzero(h > 0).max()]+ANG_BIN/2)
            # ghosts: aerogel photons reflected at a tile face, and their radii
            rad = a['radial'][0, 1]
            for ci, cn in enumerate(CLASSES):
                h = rad[ci]
                d['aero_cls_%s' % cn] = float(h.sum())
                if h.sum() > 0 and cn != 'direct':
                    cw = np.cumsum(h)/h.sum()
                    d['aero_cls_%s_r' % cn] = [float(np.interp(q, cw, rc)) for q in (0.05, 0.5, 0.95)]
            # pixel analysis of the centre region: gas ring vs window cluster
            for pi, p in enumerate(PITCHES):
                hist = a['pix%d' % pi]                                       # (origin, phx, phy, px, py)
                wpx = hist[2] > 0.02*hist[2].sum((-1, -2), keepdims=True)    # window pixels: >= 2 % of the cluster light
                W = p*np.ceil(PIX_WIN/p); npx = hist.shape[-1]
                fr, ngp, gpx, bg = [], [], [], []
                for ia in range(NPH):
                    for ib in range(NPH):
                        cxp = int(np.floor((W+ia/NPH*p)/p)); cyp = int(np.floor((W+ib/NPH*p)/p))
                        occ = wpx[ia, ib].copy(); occ[cxp, cyp] = True               # the muon's direct hit
                        g = hist[0, ia, ib]; free = ~occ&(g > 0.01*g.sum())
                        fr.append(g[~occ].sum()/g.sum() if g.sum() > 0 else np.nan); ngp.append(free.sum())
                        gpx.append(g[free].sum()); bg.append(hist[1, ia, ib][free].sum())
                d['pix%g' % p] = dict(gas_frac_clear=float(np.nanmean(fr)), gas_frac_clear_min=float(np.nanmin(fr)),
                                      gas_frac_clear_max=float(np.nanmax(fr)), n_gas_pixels=float(np.mean(ngp)),
                                      gas_photons_clear=float(np.mean(gpx)), aero_bkg_in_gas_pixels=float(np.mean(bg)),
                                      window_pixels=float(wpx.sum((-1, -2)).mean()+1))
                pm = a['pixmom%d' % pi]
                for oi, nm in ((1, 'aero'), (0, 'gas')):
                    s0, s1, s2 = pm[oi, :, 0], pm[oi, :, 1], pm[oi, :, 2]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mphase = s1/s0; rphase = np.sqrt(np.maximum(s2/s0-mphase**2, 0))
                    d['pix%g' % p]['%s_r_pix_mean' % nm] = [float(np.nanmin(mphase)), float(np.nanmax(mphase))]
                    d['pix%g' % p]['%s_r_pix_rms' % nm] = [float(np.nanmin(rphase)), float(np.nanmean(rphase)), float(np.nanmax(rphase))]
            # ring area on the SiPM: 3.2 mm pixels with >= 0.5 % of the aerogel ring light
            mp_ = a['map'][1].sum(0); nb = mp_.shape[0]; k = int(round(PIXEL_PITCH/MAP_BIN))
            nb2 = nb//k*k; off = (nb-nb2)//2
            px = mp_[off:off+nb2, off:off+nb2].reshape(nb2//k, k, nb2//k, k).sum((1, 3))
            d['aero_pixels_3.2'] = int((px > 0.005*px.sum()).sum()); d['pixels_3.2_total'] = int(px.size)
            S['%s_%s' % (mk, gn)] = d
            q = d['ftqe']
            rows.append([mk, gn, q['aero_r_mean'], q['aero_r_rms'], q['aero_fwhm'], q['aero_hw68'], d['trace_r'], d['paraxial_r'],
                         q['aero_direct_n'], q['aero_sipm_all'], q['gas_r_mean'], q['gas_r_rms'], q['gas_direct_n'], q['gas_sipm_all'],
                         d['window_n'], np.std(d['seed_aero_r'], ddof=1) if nseeds > 1 else np.nan])
    head = ['mirror', 'gas', 'aero_r_mean_mm', 'aero_r_rms_mm', 'aero_fwhm_mm', 'aero_hw68_mm', 'trace_r_mm', 'paraxial_r_mm',
            'aero_direct_sipm', 'aero_all_sipm', 'gas_r_mean_mm', 'gas_r_rms_mm', 'gas_direct_sipm', 'gas_all_sipm', 'window_sipm',
            'aero_r_seed_std_mm']
    with open(out/'short_f_onaxis.csv', 'w') as fh:
        fh.write(','.join(head)+'\n')
        for r in rows:
            fh.write(','.join(r[:2]+['%.6g' % v for v in r[2:]])+'\n')
    # air vs Xe radius differences per seed (matched seeds)
    S['delta_r_xe_minus_air'] = {}
    for mk in MIRROR_KEYS:
        ra = np.array(S['%s_air' % mk]['seed_aero_r'])
        for gn in GAS_NAMES[1:]:
            dx = np.array(S['%s_%s' % (mk, gn)]['seed_aero_r'])-ra
            S['delta_r_xe_minus_air']['%s_%s' % (mk, gn)] = dict(mean=float(dx.mean()), std=float(dx.std(ddof=1)) if nseeds > 1 else np.nan,
                                                                   per_seed=dx.tolist())
    fig_onaxis_radial(out, comb, S)
    fig_onaxis_maps(out, comb, S)
    fig_gas_ring(out, comb, S)
    return comb


def fig_onaxis_radial(out, comb, S):
    """Radial profiles: full range (log) by origin and class group, and the aerogel-ring core (linear, um scale)."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    rc = None
    for j, mk in enumerate(MIRROR_KEYS):
        a = ax[0, j]; rb = 10
        rad = rebin(comb[(mk, REF_GAS)]['radial'][0], rb)/(R_BIN*rb); rc = (np.arange(rad.shape[-1])+0.5)*R_BIN*rb
        for oi, nm, col in ((1, 'aerogel', AEROGEL), (0, 'gas', GAS), (2, 'window', SCINT_EDGE)):
            for gi, (cs, gname) in enumerate(zip(GROUPS, GROUP_NAMES)):
                y = rad[oi, list(cs)].sum(0)
                if y.sum() > 0:
                    a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=col, lw=1.4 if gi == 0 else 0.9,
                           ls=('-', ':', '-.')[gi], label='%s, %s (%.3g / muon)' % (nm, gname, y.sum()*R_BIN*rb))
        a.set_yscale('log'); a.set_xlim(0, R_MAX); a.set_ylim(1e-6, None); a.grid(alpha=.3)
        a.axvline(SENSOR_HALF_2IN, color=MUTED, lw=.8, ls=':')
        a.set_xlabel('radius r on the SiPM plane [mm]'); a.set_ylabel('dN/dr [photons / mm / muon], FT x QE')
        a.set_title('#%s (f = %.1f mm), %s: all photons on the SiPM, %.1f mm bins' % (mk, MIRRORS[mk][0], GAS_LABEL[REF_GAS], R_BIN*rb), fontsize=10)
        a.legend(fontsize=7.5, loc='upper right')
        a = ax[1, j]
        for gn in GAS_NAMES:
            ca = comb[(mk, gn)]['core_a'][0, 1]; rb = 4
            h = rebin(ca, rb); bw = CORE_BIN*rb
            x0 = S['%s_%s' % (mk, gn)]['trace_r']
            xc = x0-CORE_HALF+(np.arange(len(h))+0.5)*bw
            a.step(1e3*(xc-S['%s_air' % mk]['trace_r']), h/bw, where='mid', color=GAS_COLOUR[gn], lw=1.3,
                   label='%s: <r> %.3f mm, rms %.1f um, FWHM %.1f um' % (
                       GAS_LABEL[gn], S['%s_%s' % (mk, gn)]['ftqe']['aero_r_mean'], 1e3*S['%s_%s' % (mk, gn)]['ftqe']['aero_r_rms'],
                       1e3*S['%s_%s' % (mk, gn)]['ftqe']['aero_fwhm_all']))
        a.set_xlim(-60, 25); a.grid(alpha=.3)
        a.set_xlabel('r - r(air, deterministic trace) [um]   (r(air) = %.3f mm)' % S['%s_air' % mk]['trace_r'])
        a.set_ylabel('dN/dr [photons / mm / muon], FT x QE')
        a.set_title('#%s: aerogel-ring core, all aerogel photons, %.0f um bins (axis clipped to -60..+25 um)' % (mk, 1e3*CORE_BIN*4), fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle('Vertical muon on the mirror axis: radial hit profiles (FT x QE)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save(out, 'short_f_onaxis_radial.png', dpi=120, fig=fig); plt.close(fig)


def fig_onaxis_maps(out, comb, S):
    """SiPM hit maps (log), Xe at the reference pressure, both mirrors, with the 3.2 mm pixel grid."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    cmap = plt.get_cmap('magma_r').copy(); cmap.set_bad('white')
    for a, mk in zip(ax, MIRROR_KEYS):
        m = comb[(mk, REF_GAS)]['map'].sum((0, 1))/MAP_BIN**2
        im = a.imshow(np.ma.masked_less_equal(m.T, 0), origin='lower', extent=(-SENSOR_HALF_2IN, SENSOR_HALF_2IN)*2,
                      norm=LogNorm(m.max()*1e-6, m.max()), cmap=cmap, interpolation='nearest')
        for g in np.arange(-8, 9)*PIXEL_PITCH:
            a.axvline(g, color=MUTED, lw=0.3, alpha=0.5); a.axhline(g, color=MUTED, lw=0.3, alpha=0.5)
        a.set_xlim(-SENSOR_HALF_2IN, SENSOR_HALF_2IN); a.set_ylim(-SENSOR_HALF_2IN, SENSOR_HALF_2IN)
        a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]'); a.set_aspect('equal')
        d = S['%s_%s' % (mk, REF_GAS)]
        a.set_title('#%s, %s: aerogel ring r = %.2f mm, gas ring r = %.2f mm,\nwindow cluster r < %.2f mm; grid = 3.2 mm pixels'
                    % (mk, GAS_LABEL[REF_GAS], d['ftqe']['aero_r_mean'], d['ftqe']['gas_r_mean'], d['window_r_max']), fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label='photons / mm² / muon (FT x QE)')
    fig.suptitle('SiPM hit density, vertical on-axis muon, all origins and paths (log scale, %.1f mm bins)' % MAP_BIN, fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(out, 'short_f_onaxis_maps.png', dpi=120, fig=fig); plt.close(fig)


def fig_gas_ring(out, comb, S):
    """Centre region: gas ring per pressure, window cluster, aerogel light; plus the 0.8 mm / 3.2 mm pixel views."""
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for j, mk in enumerate(MIRROR_KEYS):
        a = ax[j, 0]
        for gn in GAS_NAMES:
            rad = comb[(mk, gn)]['radial'][0]/R_BIN; rc = (np.arange(rad.shape[-1])+0.5)*R_BIN
            y = rad[0].sum(0)
            a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=GAS_COLOUR[gn], lw=1.3,
                   label='%s gas photons (%.2f / muon, <r> %.2f mm)' % (GAS_LABEL[gn], S['%s_%s' % (mk, gn)]['ftqe']['gas_sipm_all'],
                                                                     S['%s_%s' % (mk, gn)]['ftqe']['gas_r_mean']))
        rad = comb[(mk, REF_GAS)]['radial'][0]/R_BIN
        y = rad[2].sum(0); a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=SCINT_EDGE, lw=1.3,
                                  label="window cluster (%.1f / muon + %g cell)" % (S['%s_%s' % (mk, REF_GAS)]['window_n'], MIP_DIRECT_PE))
        y = rad[1].sum(0); a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=AEROGEL, lw=1.0,
                                  label='aerogel photons, all paths (%s)' % GAS_LABEL[REF_GAS])
        for k in range(1, 5):
            a.axvline(k*PIXEL_PITCH, color=MUTED, lw=.6, ls=':')
        a.set_yscale('log'); a.set_xlim(0, 8); a.set_ylim(1e-5, None); a.grid(alpha=.3)
        a.set_xlabel('r [mm] (dotted: multiples of 3.2 mm)'); a.set_ylabel('dN/dr [photons / mm / muon], FT x QE')
        a.set_title('#%s: centre region, %.2f mm bins' % (mk, R_BIN), fontsize=10); a.legend(fontsize=7)
        for pi, p in enumerate(PITCHES):
            a = ax[j, 1+pi]
            hist = comb[(mk, REF_GAS)]['pix%d' % pi][:, 0, 0]               # phase 0: pixel corner at the impact point
            W = p*np.ceil(PIX_WIN/p); npx = hist.shape[-1]
            tot = hist.sum(0); tmax = tot.max()
            rgb = np.array([to_rgb(GAS), to_rgb(AEROGEL), to_rgb(SCINT_EDGE)])
            for ix, iy in zip(*np.nonzero(tot > 1e-4)):
                col = (hist[:, ix, iy]/tot[ix, iy])@rgb
                x0 = -W+ix*p; y0 = -W+iy*p
                a.add_patch(plt.Rectangle((x0, y0), p, p, facecolor=col, edgecolor='none', alpha=0.15+0.85*np.sqrt(tot[ix, iy]/tmax)))
                if p > 1 and abs(x0+p/2) < PIX_WIN and abs(y0+p/2) < PIX_WIN:
                    a.text(x0+p/2, y0+p/2, '%.2f' % tot[ix, iy], ha='center', va='center', fontsize=6.5, color=INK)
            for k in range(npx+1):
                a.axvline(-W+k*p, color=MUTED, lw=0.3); a.axhline(-W+k*p, color=MUTED, lw=0.3)
            d = S['%s_%s' % (mk, REF_GAS)]['pix%g' % p]
            a.set_xlim(-PIX_WIN, PIX_WIN); a.set_ylim(-PIX_WIN, PIX_WIN); a.set_aspect('equal')
            a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]')
            a.set_title('#%s, %s, %.1f mm pixels, corner at the impact point\ngas light outside window pixels: %.0f %% (mean over grid phases)'
                        % (mk, GAS_LABEL[REF_GAS], p, 100*d['gas_frac_clear']), fontsize=9)
    fig.suptitle('Small Xe gas ring vs the muon\'s own window cluster (vertical on-axis muon, FT x QE; pixel colour: blue gas, green window, '
                 'orange aerogel; numbers: photons per muon per 3.2 mm pixel)',
                 fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(out, 'short_f_gas_ring.png', dpi=120, fig=fig); plt.close(fig)


def analyse_terms(out, acc, args, summary):
    """Itemised ring-width terms per mirror and gas, deterministic r(depth) curves."""
    S = summary.setdefault('terms', {}); S['N'] = dict(photons_per_origin=int(np.ceil(args.nphot_terms/args.chunk))*args.chunk)
    dep = (np.arange(400)+0.5)/400
    for mk in MIRROR_KEYS:
        for gn in GAS_NAMES:
            d = {}
            lay = make_layout(mk, gn, rayleigh=False, gas_fixed=True)
            for t in TERMS:
                a = acc[(mk, gn, t)]
                lt = make_layout(mk, gn)
                ra, rg = ref_radii(lt)
                ch = ZCORE_HALF if t[:2] in ('sp', 'k4') else CORE_HALF; cb = ZCORE_BIN if t[:2] in ('sp', 'k4') else CORE_BIN
                d[t] = ring_numbers(a, ra, rg, 0, ch, cb)
                sc = a['radial'][0, 1, CLASSES.index('scattered')]; rc = (np.arange(len(sc))+0.5)*R_BIN; mr = d[t]['aero_r_mean']
                d[t]['aero_scattered'] = float(sc.sum())
                for w in (0.4, 1.6):
                    d[t]['aero_scattered_within_%g' % w] = float(sc[np.abs(rc-mr) < w].sum())
                    d[t]['aero_direct_within_%g' % w] = float(a['radial'][0, 1, 0][np.abs(rc-mr) < w].sum())
            det = {}
            for mir in ('sphere', 'ideal'):
                for ret in (True, False):
                    r, h = meridional(lay, dep, mirror=mir, ret=ret)
                    det['%s_%s' % (mir, 'return' if ret else 'noreturn')] = dict(mean=float(r.mean()), rms=float(r.std()), min=float(r.min()),
                                                                                 max=float(r.max()), h_min=float(h.min()), h_max=float(h.max()))
            d['deterministic'] = det
            f = lay.mirror.f; ta = theta_c(AEROGEL_N_AIR); ng = float(lay.gas.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])
            tg = np.arcsin(AEROGEL_N_AIR*np.sin(ta)/ng)
            d['paraxial_r'] = float(f*np.tan(tg)); d['no_refraction_r'] = float(f*np.tan(ta))
            # dr/dtheta_aero and dr/dn by index steps
            rp = meridional(lay, dep, n_tile=AEROGEL_N_AIR+1e-4)[0].mean(); rm = meridional(lay, dep, n_tile=AEROGEL_N_AIR-1e-4)[0].mean()
            dth = theta_c(AEROGEL_N_AIR+1e-4)-theta_c(AEROGEL_N_AIR-1e-4)
            d['dr_dtheta_aero_mm_per_mrad'] = float((rp-rm)/dth*1e-3); d['dr_dn_mm_per_1e-3'] = float((rp-rm)/2e-4*1e-3)
            dg = 1e-4; tgp = np.arcsin(AEROGEL_N_AIR*np.sin(ta)/(ng+dg)); tgm = np.arcsin(AEROGEL_N_AIR*np.sin(ta)/(ng-dg))
            lp = make_layout(mk, gn, rayleigh=False); lp = replace(lp, gas=Medium('g+', ConstantIndex(ng+dg)))
            lm = replace(lp, gas=Medium('g-', ConstantIndex(ng-dg)))
            d['dr_dtheta_gas_mm_per_mrad'] = float((meridional(lp, dep)[0].mean()-meridional(lm, dep)[0].mean())/(tgp-tgm)*1e-3)
            for p in PITCHES:
                d['pixel_term_mm_%g' % p] = float(pixel_sigma(p, f)*f)
                d['pixel_term_mrad_%g' % p] = float(pixel_sigma(p, f)*f/d['dr_dtheta_aero_mm_per_mrad'])
            S['%s_%s' % (mk, gn)] = d
    fig_terms(out, S, dep)


def fig_terms(out, S, dep):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for j, mk in enumerate(MIRROR_KEYS):
        a = ax[j]
        lay = make_layout(mk, 'air', rayleigh=False, gas_fixed=True)
        base = S['%s_air' % mk]['paraxial_r']
        for mir, ret, col, lab in (('sphere', True, INK, 'sphere + return pass (as built)'), ('sphere', False, AEROGEL, 'sphere, no return pass'),
                                   ('ideal', True, GAS, 'ideal focusing + return pass'), ('ideal', False, MUTED, 'ideal focusing, no return pass')):
            r, h = meridional(lay, dep, mirror=mir, ret=ret)
            a.plot(10*dep, 1e3*(r-base), color=col, lw=1.5, label='%s: rms %.1f um' % (lab, 1e3*r.std()))
        a.set_xlabel('emission depth below the tile top face [mm]'); a.set_ylabel('r - f tan(theta_gas) [um]')
        a.set_title('#%s, air, 400 nm: landing radius vs emission depth\n(f tan theta_gas = %.3f mm)' % (mk, base), fontsize=10)
        a.grid(alpha=.3); a.legend(fontsize=8)
    a = ax[2]
    terms = (('mono_depth', 'depth'), ('mono_depth_noreturn', 'depth,\nno return'),
             ('gasdisp', 'gas\ndisp.'), ('mono_point', 'none\n(check)'), ('sp_point', 'aero disp.\n1-pole'),
             ('k4_point', 'aero disp.\nk4geo'), ('norayl', 'all but\nRayleigh'), ('full', 'full'))
    xs = np.arange(len(terms)); wbar = 0.1
    k = 0
    for mk in MIRROR_KEYS:
        for gn in ('air', REF_GAS):
            v = [1e3*S['%s_%s' % (mk, gn)][t]['aero_r_rms'] for t, _ in terms]
            a.bar(xs+(k-1.5)*wbar*1.1, np.maximum(v, 1e-3), wbar, color=GAS_COLOUR[gn], alpha=1 if mk == '43-543' else 0.5,
                  edgecolor=INK, lw=0.3, hatch='' if mk == '43-543' else '//', label='#%s, %s' % (mk, GAS_LABEL[gn]))
            k += 1
    for p, ls in zip(PITCHES, ('--', ':')):
        a.axhline(1e3*p/np.sqrt(12), color=INK, ls=ls, lw=1, label='pixel term %.1f mm / sqrt(12)' % p)
    a.set_yscale('log'); a.set_xticks(xs); a.set_xticklabels([t[1] for t in terms], fontsize=7.5)
    a.set_ylabel('rms of the aerogel direct-ring radius [um] (MC; 0 drawn at 1e-3)'); a.grid(alpha=.3, axis='y')
    a.set_title('Ring-width terms switched one at a time (direct photons, FT x QE)', fontsize=10); a.legend(fontsize=7, ncol=2)
    fig.suptitle('Aberration budget of the aerogel ring (vertical on-axis muon)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(out, 'short_f_terms.png', dpi=120, fig=fig); plt.close(fig)


def analyse_zscan(out, acc, args, summary):
    S = summary.setdefault('zscan', {}); S['N'] = dict(photons_per_origin=int(np.ceil(args.nphot_zscan/args.chunk))*args.chunk)
    S['dz_mm'] = DZ_SCAN.tolist()
    dep = (np.arange(400)+0.5)/400
    for mk in MIRROR_KEYS:
        for gn in ('air', REF_GAS):
            res = {q: [] for q in ('aero_r_mean', 'aero_r_rms', 'aero_fwhm', 'aero_hw68', 'gas_r_mean', 'gas_r_rms', 'gas_hw68',
                                   'det_r_mean', 'det_r_rms', 'aero_direct_n')}
            for zi, dz in enumerate(DZ_SCAN):
                a = acc[(mk, gn, zi)]; lay = make_layout(mk, gn, rayleigh=False, dz=dz); ra, rg = ref_radii(lay)
                q = ring_numbers(a, ra, rg, 0, ZCORE_HALF, ZCORE_BIN)
                for k_ in ('aero_r_mean', 'aero_r_rms', 'aero_fwhm', 'aero_hw68', 'gas_r_mean', 'gas_r_rms', 'gas_hw68', 'aero_direct_n'):
                    res[k_].append(q[k_])
                r = meridional(lay, dep)[0]
                res['det_r_mean'].append(float(r.mean())); res['det_r_rms'].append(float(r.std()))
            S['%s_%s' % (mk, gn)] = res
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    for mk in MIRROR_KEYS:
        for gn in ('air', REF_GAS):
            r = S['%s_%s' % (mk, gn)]; ls = MIRROR_STYLE[mk]; col = GAS_COLOUR[gn]
            ax[0].plot(DZ_SCAN, 1e3*np.array(r['aero_r_rms']), ls=ls, color=col, marker='o', ms=3, label='#%s, %s (MC rms)' % (mk, GAS_LABEL[gn]))
            ax[0].plot(DZ_SCAN, 1e3*np.array(r['aero_fwhm'])/2.355, ls=ls, color=col, lw=0.7, alpha=0.6)
            ax[1].plot(DZ_SCAN, np.array(r['aero_r_mean'])-r['aero_r_mean'][len(DZ_SCAN)//2], ls=ls, color=col, marker='o', ms=3,
                       label='#%s, %s' % (mk, GAS_LABEL[gn]))
            if gn == REF_GAS:
                ax[2].plot(DZ_SCAN, 1e3*np.array(r['gas_r_rms']), ls=ls, color=col, marker='o', ms=3, label='#%s, %s gas ring rms' % (mk, GAS_LABEL[REF_GAS]))
    ax[0].set_ylabel('aerogel direct-ring width [um] (markers: rms; thin: FWHM/2.355)'); ax[0].set_yscale('log')
    ax[1].set_ylabel('<r>(dz) - <r>(0) [mm]'); ax[2].set_ylabel('gas direct-ring rms [um]')
    for a in ax:
        a.set_xlabel('sensor and tile displacement from z = f [mm] (+: away from the mirror)'); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_title('Aerogel ring width vs sensor position (Rayleigh off)', fontsize=10)
    ax[1].set_title('Aerogel ring radius vs sensor position', fontsize=10); ax[2].set_title('%s gas ring width' % GAS_LABEL[REF_GAS], fontsize=10)
    fig.suptitle('Sensor-position scan (tile kept 1 mm below the SiPM)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(out, 'short_f_zscan.png', dpi=120, fig=fig); plt.close(fig)


def analyse_tol(out, acc, args, summary):
    S = summary.setdefault('tol', {}); S['N'] = dict(photons_per_origin=int(np.ceil(args.nphot_zscan/args.chunk))*args.chunk)
    S['f_scale'] = list(F_SCALE)
    for mk in MIRROR_KEYS:
        for gn in ('air', REF_GAS):
            d = {}
            for mode in ('focus', 'nominal'):
                d[mode] = []
                for fi, fs in enumerate(F_SCALE):
                    a = acc[(mk, gn, mode, fi)]
                    lay = make_layout(mk, gn, rayleigh=False, f_scale=fs, z_abs=None if mode == 'focus' else MIRRORS[mk][0])
                    ra, rg = ref_radii(lay); q = ring_numbers(a, ra, rg, 0, ZCORE_HALF, ZCORE_BIN)
                    d[mode].append(dict(aero_r_mean=q['aero_r_mean'], aero_r_rms=q['aero_r_rms'], gas_r_mean=q['gas_r_mean'],
                                        gas_r_rms=q['gas_r_rms'], det_r=ra, aero_direct_n=q['aero_direct_n']))
            lay = make_layout(mk, gn, rayleigh=False, n_tile=AEROGEL_N_AIR+DN_TILE); ra, rg = ref_radii(lay)
            q = ring_numbers(acc[(mk, gn, 'dn', 0)], ra, rg, 0, ZCORE_HALF, ZCORE_BIN)
            d['dn'] = dict(aero_r_mean=q['aero_r_mean'], det_r=ra, dn=DN_TILE,
                           dtheta_mrad=float(1e3*(theta_c(AEROGEL_N_AIR+DN_TILE)-theta_c(AEROGEL_N_AIR))))
            S['%s_%s' % (mk, gn)] = d


def analyse_angles(out, res, args, summary):
    """Containment maps and curves; arrays saved with explicit axes (read by cosmic_rates_plots.py --short-f)."""
    S = summary.setdefault('angles', {}); S['N'] = dict(photons_per_config=args.nphot_angle)
    shape = (len(MIRROR_KEYS), len(GAS_NAMES), len(CLEAR_SCAN), len(AZIMUTH), len(OFFSET_DIRS), len(ZENITH), len(OFFSET))
    st = np.full(shape+(24,), np.nan); phi = np.full(shape+(4, NPHI), np.nan); rs = np.full(shape+(len(REASONS),), np.nan)
    p0 = np.full(shape+(2,), np.nan)
    for key, r in res.items():
        mk, gn, cfi, az, od, zi, oi = key
        idx = (MIRROR_KEYS.index(mk), GAS_NAMES.index(gn), cfi, az, od, zi, oi)
        st[idx] = r['stats']; phi[idx] = r['phi']; rs[idx] = r['reason']; p0[idx] = r['p0']
    emit = st[..., 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        f_det = st[..., 2]/emit; f_dir = st[..., 3]/emit
        f_mirror = rs[..., 1]/emit; f_out = rs[..., 3]/emit; f_sides = rs[..., 2]/emit
        on_b = np.where(phi[..., 2, :] > 0, phi[..., 1, :]/phi[..., 2, :], 0.0)  # per azimuth: on the SiPM / reaching its plane
    ic = CLEAR_SCAN.index(MIRROR_CLEAR_FRAC); i0 = list(OFFSET).index(0.0)
    ref_eff = f_dir[:, :, ic, 0, 0, 0, i0]                                # on-axis direct efficiency per mirror, gas
    arc = on_b.mean(-1)
    arc[np.isnan(emit)|(emit <= 0)] = np.nan
    rel = f_det/f_det[:, :, ic, 0, 0, 0, i0][:, :, None, None, None, None, None]
    np.savez(out/'short_f_angles.npz', **provenance.npz(), mirror=np.array(MIRROR_KEYS), mirror_f_mm=np.array([MIRRORS[k][0] for k in MIRROR_KEYS]),
             gas=np.array(GAS_NAMES), clear_frac=np.array(CLEAR_SCAN), azimuth_deg=np.array(AZIMUTH), offset_dir=np.array(OFFSET_DIRS),
             zenith_deg=ZENITH, offset_mm=OFFSET, stats=st, phi=phi, reasons_w=rs, reasons=np.array(REASONS), entry_xy_mm=p0,
             f_det=f_det, f_direct=f_dir, f_mirror_aperture=f_mirror, f_outside_sensor=f_out, f_tile_sides=f_sides, arc_frac=arc,
             rel_det=rel, axes=np.array(['mirror', 'gas', 'clear_frac', 'azimuth_deg', 'offset_dir', 'zenith_deg', 'offset_mm']),
             stats_fields=np.array(STATS_FIELDS))
    # containment half-angles at zero offset (parallel), nominal clear aperture
    S['axes'] = 'arrays [mirror, gas, clear_frac, azimuth, offset_dir, zenith, offset]'
    for mi, mk in enumerate(MIRROR_KEYS):
        for gi, gn in enumerate(GAS_NAMES):
            d = {}
            for az in range(len(AZIMUTH)):
                for cfi, cf in enumerate(CLEAR_SCAN):
                    a1 = arc[mi, gi, cfi, az, 0, :, i0]; r1 = rel[mi, gi, cfi, az, 0, :, i0]
                    tag = 'az%g_clear%g' % (AZIMUTH[az], cf)
                    for thr in (1.0, 0.95, 0.8, 0.5):
                        d['arc_%g_%s' % (thr, tag)] = max_zenith(a1, thr); d['rel_%g_%s' % (thr, tag)] = max_zenith(r1, thr)
                    d['f_mirror_%s' % tag] = f_mirror[mi, gi, cfi, az, 0, :, i0].tolist()
                    d['rel_%s' % tag] = r1.tolist(); d['arc_%s' % tag] = a1.tolist()
                d['centre_x_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 6].tolist()
                d['centre_y_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 7].tolist()
                d['fit_r_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 8].tolist()
                d['inc_mean_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 10].tolist()
                d['inc_p95_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 13].tolist()
                d['inc_max_az%g' % AZIMUTH[az]] = st[mi, gi, ic, az, 0, :, i0, 14].tolist()
            d['f_det_onaxis'] = float(f_det[mi, gi, ic, 0, 0, 0, i0]); d['f_direct_onaxis'] = float(ref_eff[mi, gi])
            d['emit_onaxis'] = float(emit[mi, gi, ic, 0, 0, 0, i0])
            # 2D: fraction of the (zenith, offset) cells with >= 80 % relative photons, per azimuth / offset direction
            for az in range(len(AZIMUTH)):
                for od in range(len(OFFSET_DIRS)):
                    m = rel[mi, gi, ic, az, od]
                    d['cells_rel80_az%g_%s' % (AZIMUTH[az], OFFSET_DIRS[od])] = float(np.nanmean(m >= 0.8))
            S['%s_%s' % (mk, gn)] = d
    gi0 = GAS_NAMES.index('air')
    S['max_abs_rel_diff_xe_vs_air'] = float(np.nanmax(np.abs(rel[:, 1:, ic]-rel[:, gi0:gi0+1, ic])))
    S['max_abs_arc_diff_xe_vs_air'] = float(np.nanmax(np.abs(arc[:, 1:, ic]-arc[:, gi0:gi0+1, ic])))
    fig_angles_maps(out, rel, arc, S)
    fig_angles_curves(out, rel, arc, f_mirror, st, S)
    return dict(rel=rel, arc=arc, st=st)


def max_zenith(curve, thr):
    """Largest scanned zenith [deg] up to which curve >= thr at every scanned angle (nan if not even at 0)."""
    bad = np.flatnonzero(~(np.asarray(curve) >= thr-1e-9))
    if not len(bad):
        return float(ZENITH[-1])
    return float(ZENITH[bad[0]-1]) if bad[0] > 0 else np.nan


STATS_FIELDS = ('emit_ftqe', 'emit_ft', 'det_ftqe', 'det_direct_ftqe', 'det_ft', 'det_direct_ft', 'centre_x_mm', 'centre_y_mm',
                'fit_r_mm', 'fit_r_rms_mm', 'inc_mean_deg', 'inc_rms_deg', 'inc_p50_deg', 'inc_p95_deg', 'inc_max_deg',
                'tile_path_mm', 'entry_on_sipm', 'unused17', 'unused18', 'unused19', 'unused20', 'unused21', 'unused22', 'unused23')


def fig_angles_maps(out, rel, arc, S):
    ic = CLEAR_SCAN.index(MIRROR_CLEAR_FRAC); gi = GAS_NAMES.index('air')
    fig, ax = plt.subplots(4, 4, figsize=(18, 17))
    ext = (OFFSET[0]-1.25, OFFSET[-1]+1.25, ZENITH[0]-0.5, ZENITH[-1]+0.5)
    for mi, mk in enumerate(MIRROR_KEYS):
        for az in range(len(AZIMUTH)):
            for od in range(len(OFFSET_DIRS)):
                for q, (arr, nm) in enumerate(((rel, 'photons on the SiPM / on-axis value'), (arc, 'ring azimuth fraction imaged on the SiPM'))):
                    a = ax[2*mi+q, 2*az+od]
                    m = arr[mi, gi, ic, az, od]
                    im = a.imshow(m, origin='lower', extent=ext, aspect='auto', vmin=0, vmax=1.05 if q == 0 else 1, cmap='viridis')
                    cs = a.contour(OFFSET, ZENITH, m, levels=[0.5, 0.8, 0.95], colors=['w', 'w', 'k'], linewidths=0.8)
                    a.clabel(cs, fontsize=7, fmt='%.2f')
                    a.set_title('#%s, %s\nazimuth %g deg, offset %s' % (mk, nm, AZIMUTH[az], OFFSET_DIRS[od]), fontsize=9)
                    a.set_xlabel('impact offset at the tile mid-plane [mm]'); a.set_ylabel('zenith [deg]')
                    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle('Aerogel-ring containment vs muon zenith and impact offset, air, clear aperture %.0f %% (Xe 1-3 bar maps: max |difference| %.3f '
                 '(photons), %.3f (azimuth))' % (100*MIRROR_CLEAR_FRAC, S['max_abs_rel_diff_xe_vs_air'], S['max_abs_arc_diff_xe_vs_air']),
                 fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save(out, 'short_f_angles_maps.png', dpi=100, fig=fig); plt.close(fig)


def fig_angles_curves(out, rel, arc, f_mirror, st, S):
    gi = GAS_NAMES.index('air'); i0 = list(OFFSET).index(0.0); ic = CLEAR_SCAN.index(MIRROR_CLEAR_FRAC)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for mi, mk in enumerate(MIRROR_KEYS):
        f = MIRRORS[mk][0]
        for az, col in zip(range(len(AZIMUTH)), (GAS, AEROGEL)):
            for cfi, cf in enumerate(CLEAR_SCAN):
                lw = 1.6 if cf == MIRROR_CLEAR_FRAC else 0.8; ls = MIRROR_STYLE[mk]
                al = 1 if cf == MIRROR_CLEAR_FRAC else 0.6
                lab = '#%s, az %g deg, clear %.0f %%' % (mk, AZIMUTH[az], 100*cf)
                ax[0, 0].plot(ZENITH, rel[mi, gi, cfi, az, 0, :, i0], ls=ls, color=col, lw=lw, alpha=al, label=lab)
                ax[0, 1].plot(ZENITH, arc[mi, gi, cfi, az, 0, :, i0], ls=ls, color=col, lw=lw, alpha=al, label=lab)
                ax[0, 2].plot(ZENITH, f_mirror[mi, gi, cfi, az, 0, :, i0], ls=ls, color=col, lw=lw, alpha=al, label=lab)
            s = st[mi, gi, ic, az, 0, :, i0]
            cr = np.hypot(s[:, 6], s[:, 7])
            ax[1, 0].plot(ZENITH, cr, ls=MIRROR_STYLE[mk], color=col, marker='o', ms=3, label='#%s, az %g deg: fitted centre' % (mk, AZIMUTH[az]))
            if az == 0:
                ax[1, 0].plot(ZENITH, f*np.tan(np.radians(ZENITH)), ls=MIRROR_STYLE[mk], color=MUTED, lw=0.8, label='#%s: f tan(zenith)' % mk)
            ax[1, 1].plot(ZENITH, s[:, 8], ls=MIRROR_STYLE[mk], color=col, marker='o', ms=3, label='#%s, az %g deg' % (mk, AZIMUTH[az]))
            ax[1, 2].plot(ZENITH, s[:, 10], ls=MIRROR_STYLE[mk], color=col, lw=1.5, label='#%s, az %g deg: mean' % (mk, AZIMUTH[az]))
            ax[1, 2].plot(ZENITH, s[:, 14], ls=MIRROR_STYLE[mk], color=col, lw=0.8, alpha=0.6, label='#%s, az %g deg: max' % (mk, AZIMUTH[az]))
    ax[0, 0].set_ylabel('aerogel photons on the SiPM / on-axis value'); ax[0, 1].set_ylabel('ring azimuth fraction on the SiPM')
    ax[0, 2].set_ylabel('aerogel photons lost at the mirror aperture / emitted')
    ax[1, 0].set_ylabel('ring-centre distance from the sensor centre [mm]'); ax[1, 1].set_ylabel('fitted ring radius [mm]')
    ax[1, 2].set_ylabel('angle of incidence on the SiPM plane [deg] (detected direct photons)')
    for a in ax.ravel():
        a.set_xlabel('muon zenith angle [deg] (zero offset)'); a.grid(alpha=.3); a.legend(fontsize=6.5)
    fig.suptitle('Aerogel ring vs muon zenith angle, zero impact offset, air (FT x QE)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save(out, 'short_f_angles_curves.png', dpi=110, fig=fig); plt.close(fig)


# ---------------------------------------------------------------- main

def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    return x


def save_acc(out, part, acc):
    flat = {}
    for key, a in acc.items():
        tag = '|'.join(str(k) for k in key)
        for q, v in a.items():
            flat['%s|%s' % (tag, q)] = v
    np.savez_compressed(out/('short_f_%s.npz' % part), **flat, **provenance.npz())


def load_acc(path, nkey):
    z = np.load(path, allow_pickle=False); acc = {}
    for name in [n for n in z.files if n != provenance.KEY]:
        parts = name.split('|'); key = tuple(int(p) if p.lstrip('-').isdigit() else p for p in parts[:nkey]); q = parts[nkey]
        acc.setdefault(key, {})[q] = z[name]
    return acc


def main():
    p = cli.parser(__doc__)
    p.add_argument('--parts', default='onaxis,terms,zscan,tol,angles', help='comma-separated parts to run (default all)')
    p.add_argument('--nphot', type=float, default=2e6, help='sampled photons per origin, mirror, gas and seed in the on-axis part (default 2e6)')
    p.add_argument('--nseeds', type=int, default=4, help='master seeds of the on-axis part, --seed + k (default 4)')
    p.add_argument('--nphot-terms', type=float, default=1e6, help='sampled photons per origin and variant in the terms part (default 1e6)')
    p.add_argument('--nphot-zscan', type=float, default=2e5, help='sampled photons per origin and point in the zscan and tol parts (default 2e5)')
    p.add_argument('--nphot-angle', type=int, default=10000, help='sampled aerogel photons per muon configuration in the angles part (default 10000)')
    p.add_argument('--chunk', type=int, default=50000, help='sampled photons per origin and chunk (default 50000)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed; mirrors, gases and variants share the chunk seeds')
    p.add_argument('--nproc', type=int, default=8, help='worker processes (default 8)')
    p.add_argument('--replot', action='store_true', help='redo the analysis and figures from the npz files in --outdir')
    args = p.parse_args(); out = cli.outdir(args)
    parts = [x for x in args.parts.split(',') if x]
    t0 = time.time(); band = qe_band()
    summary = dict(band_eV=list(band), mirrors={k: dict(f=v[0], D=v[1]) for k, v in MIRRORS.items()}, clear_frac=MIRROR_CLEAR_FRAC,
                   sensor_half=SENSOR_HALF_2IN, aerogel_n=AEROGEL_N_AIR, tile_t=AEROGEL_T, tile_half=AEROGEL_HALF,
                   gas_n_400nm={gn: float(gas_medium(gn).n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0]) for gn in GAS_NAMES},
                   args={k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in vars(args).items()})
    sp = out/'short_f_summary.json'
    if sp.exists():
        old = json.loads(sp.read_text()); old.update({k: v for k, v in summary.items()}); summary = old
    if not args.replot:
        jobs, tasks, nchunk = build_jobs(args, parts, band)
        _CFG.update(jobs=jobs)
        print('band %.2f-%.2f eV; parts %s; %d tasks on %d processes' % (*band, ','.join(parts), len(tasks), args.nproc), flush=True)
        accs = {pt: {} for pt in parts}; nch = {}
        with mp.get_context('fork').Pool(args.nproc) as pool:
            for t, res in zip(tasks, pool.imap(run_task, tasks, chunksize=4 if 'angles' in parts else 1)):
                if t[0] == 'angles':
                    accs['angles'][res[0]] = res[1]
                else:
                    reduce_into(accs[t[0]].setdefault(t[1], {}), res); nch[(t[0], t[1])] = nch.get((t[0], t[1]), 0)+1
        print('Monte Carlo wall time %.1f s' % (time.time()-t0), flush=True)
        for pt in parts:
            if pt == 'angles':
                continue
            for key, a in accs[pt].items():
                n = nch[(pt, key)]
                for q in a:
                    if q != 'nbranch':
                        a[q] = a[q]/n
                a['nchunk'] = np.array(n)
            save_acc(out, pt, accs[pt])
        if 'angles' in parts:
            flat = {}
            for key, r in accs['angles'].items():
                for q, v in r.items():
                    flat['%s|%s' % ('|'.join(str(k) for k in key), q)] = v
            np.savez_compressed(out/'short_f_angles_raw.npz', **flat, **provenance.npz())
    else:
        accs = {}
        for pt in parts:
            if pt == 'angles':
                accs[pt] = {k: v for k, v in load_acc(out/'short_f_angles_raw.npz', 7).items()}
            else:
                accs[pt] = load_acc(out/('short_f_%s.npz' % pt), 3 if pt != 'tol' else 4)
    nchunk = max(1, int(np.ceil(args.nphot/args.chunk)))
    if 'onaxis' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['onaxis'].items()}
        nseeds = len({k[2] for k in acc})
        analyse_onaxis(out, acc, nseeds, nchunk, args, summary)
    if 'terms' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['terms'].items()}
        analyse_terms(out, acc, args, summary)
    if 'zscan' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['zscan'].items()}
        analyse_zscan(out, acc, args, summary)
    if 'tol' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['tol'].items()}
        analyse_tol(out, acc, args, summary)
    if 'angles' in parts:
        analyse_angles(out, accs['angles'], args, summary)
    sp.write_text(json.dumps(to_jsonable(provenance.with_record(summary)), indent=1))
    print('summary: %s; total wall time %.1f s' % (sp, time.time()-t0))


if __name__ == '__main__':
    main()
