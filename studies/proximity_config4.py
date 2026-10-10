"""Proximity focusing of the aerogel ring (configuration 4 of docs/design_options.md) with the 3D tracer.

The 1 cm aerogel tile (5 x 5 cm, constant n = 1.03, Rayleigh scattering) and, after a gap g = 30 / 50 / 70 mm from its
exit face to the silicon, the 2" x 2" SiPM plane facing the tile, covered by its protective window (n = 1.5, 0.3 mm;
the window lies inside g). No mirror. Air at 1 atm or Xe at 1, 2, 3 bar fills the gap and the pores. The tile index
is run with the pore-gas shift off and with both shift conventions: 'vac' = 1.03 + 0.97 (n_gas - 1) (media.aerogel_index)
and 'air' = 1.03 + 0.935 (n_gas - n_air) (media.aerogel_index_air), with n_gas(lambda). A beta = 1 muon crosses tile, gap,
window and silicon; tile, gap-gas and window light are three origins. Photons are sampled from Frank-Tamm over the
band where the ARC SiPM QE is non-zero and counted with FT x QE and pure FT.

The scene is the mirror image (z -> -z) of the real set-up: tile at 0 < z < T, muon moving up, sensor plane at
z = T + g on top. tracer.trace counts photons that reach the sensor plane from below through a window whose top face
lies on that plane; a window below a downward-facing sensor is not handled (its photons escape), hence the mirror image.

Parts (--parts):
  onaxis   vertical muon through the tile centre, --nseeds matched master seeds, every gas, gap and pore variant;
           plus vertical muons at a few impact points (variant off);
  terms    ring-width terms switched one at a time at matched seeds, deterministic r(depth), dr/dtheta, dr/dn;
  angles   containment vs zenith, muon azimuth and impact offset at the tile mid-plane (all gases, gaps, variants).
Outputs in --outdir: prox_<part>.npz, prox_angles.npz (explicit axes), prox_summary.json, prox_*.png, prox_*.csv;
--replot redraws from the npz files.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, to_rgb  # noqa: E402

from xerich import cli, sim, tracer  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band, theta_c  # noqa: E402
from xerich.constants import (AEROGEL_HALF, AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_POROSITY, AEROGEL_POROSITY_AIRREF,  # noqa: E402
                              AEROGEL_T, HC, MIP_DIRECT_PE, PIXEL_PITCH, PIXEL_PITCH_FINE, RAYLEIGH_C_AEROGEL, SENSOR_HALF_2IN,
                              WINDOW_N, WINDOW_T)
from xerich.layout import Aerogel, Sensor, Window  # noqa: E402
from xerich.media import (N_AIR, ConstantIndex, Medium, Rayleigh, aerogel_index, aerogel_index_air, aerogel_medium,  # noqa: E402
                          air, xenon)
from xerich.plotting import AEROGEL, GAS, INK, MUTED, SCINT_EDGE, plt, save  # noqa: E402
from xerich.tracer import CLASSES, REASONS, Scene, SensorPlane, Slab  # noqa: E402

GAPS = (30.0, 50.0, 70.0)                 # tile exit face to silicon [mm]
GAS_NAMES = ('air', 'xe1', 'xe2', 'xe3')
GAS_LABEL = {'air': 'air 1 atm', 'xe1': 'Xe 1 bar', 'xe2': 'Xe 2 bar', 'xe3': 'Xe 3 bar'}
GAS_COLOUR = {'air': '#2a9d8f', 'xe1': '#8ab17d', 'xe2': '#1f6fb4', 'xe3': '#d1495b'}
GAP_STYLE = {30.0: '-', 50.0: '--', 70.0: ':'}
GAP_COLOUR = {30.0: '#1f6fb4', 50.0: '#d97706', 70.0: '#6d597a'}
VARIANTS = ('off', 'vac', 'air')
VAR_LABEL = {'off': 'pore shift off', 'vac': 'pore shift 0.97 (n_gas - 1)', 'air': 'pore shift 0.935 (n_gas - n_air)'}
ORIGINS = ('gas', 'aerogel', 'window')
XREASONS = REASONS+('window sides',)    # tracer reasons; 'tile sides' split into the tile and the window side faces
PITCHES = (PIXEL_PITCH, PIXEL_PITCH_FINE)
R_DET, R_SIDE, R_OUT = REASONS.index('detected'), REASONS.index('tile sides'), REASONS.index('outside sensor')
NPH = 8                                 # pixel-grid phases per axis
PIX_WIN = 8.0                           # half side of the central pixel region [mm]
R_BIN = 0.01; R_MAX = 36.0              # radial histogram [mm]
CORE_HALF = 3.0; CORE_BIN = 0.002       # fine histogram around the deterministic ring radius [mm]
MAP_BIN = 0.2
ANG_BIN = 0.1; ANG_MAX = 60.0           # incidence-angle histogram [deg]
NPHI = 36                               # emission-azimuth bins of the containment study
NPIX_FULL = 16                          # 3.2 mm pixels per side of the full-sensor pixel map
GROUPS = ((0,), (1, 2, 3, 4), (5,)); GROUP_NAMES = ('direct', 'reflected', 'scattered')
DEPTH_MID = 0.5
POINTS = ((12.5, 0.0), (20.0, 0.0), (12.5, 12.5), (20.0, 20.0))   # extra vertical impact points [mm]
ZENITH = np.arange(0.0, 35.5, 1.0)
OFFSET = np.arange(-20.0, 20.01, 2.5)
AZIMUTH = (0.0, 45.0)
OFFSET_DIRS = ('parallel', 'perpendicular')
DN_TILE = 0.001
TERMS = ('full', 'norayl', 'depth', 'point', 'gasdisp', 'gasdisp_vac', 'gasdisp_air', 'sp_point', 'k4_point', 'sp_full', 'k4_full',
         'norefr', 'nowindow', 'dn')
STATS_FIELDS = ('emit_ftqe', 'emit_ft', 'det_ftqe', 'det_direct_ftqe', 'det_ft', 'det_direct_ft', 'centre_x_mm', 'centre_y_mm',
                'fit_r_mm', 'fit_r_rms_mm', 'inc_win_mean_deg', 'inc_win_max_deg', 'inc_gas_mean_deg', 'inc_gas_p95_deg',
                'inc_gas_max_deg', 'tile_path_mm', 'impact_x_mm', 'impact_y_mm', 'impact_on_sipm', 'mean_r_from_impact_mm',
                'n_fit_photons')
_CFG = {}


# ---------------------------------------------------------------- layout (driver-side helper)

@dataclass(frozen=True)
class ProxLayout:
    """Configuration 4 in the mirror-image frame: tile [-h, h]^2 x [0, T], gap g from the tile exit face z = T to the
    silicon at z = T + g, window of the sensor size directly below the silicon (inside the gap), gas everywhere else."""
    gas: Medium
    gap: float
    tile_medium: Medium
    tile_scatter: Rayleigh | None = Rayleigh()
    window: Window | None = Window()
    tile_half: float = AEROGEL_HALF
    tile_t: float = AEROGEL_T
    sensor_half: float = SENSOR_HALF_2IN

    @property
    def z_sensor(self):
        return self.tile_t+self.gap

    @property
    def z_window_bottom(self):
        return self.z_sensor-(self.window.thickness if self.window is not None else 0.0)

    def scene(self):
        h = self.tile_half; hs = self.sensor_half
        slabs = (Slab((-h, -h, 0.0), (h, h, self.tile_t), self.tile_medium, self.tile_scatter),)
        if self.window is not None:
            slabs += (Slab((-hs, -hs, self.z_window_bottom), (hs, hs, self.z_sensor), self.window.medium),)
        return Scene(self.gas, slabs, None, SensorPlane(self.z_sensor, hs))


def box_interval(p0, u, lo, hi, s_end):
    """[(s0, s1)] where the track p0 + s u, 0 <= s <= s_end, lies inside the box [lo, hi], or []."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo-p0)/u; t2 = (hi-p0)/u
    par = u == 0; within = (p0 >= lo)&(p0 <= hi)
    tn = np.max(np.where(par, np.where(within, -np.inf, np.inf), np.minimum(t1, t2)))
    tf = np.min(np.where(par, np.where(within, np.inf, -np.inf), np.maximum(t1, t2)))
    tn, tf = max(tn, 0.0), min(tf, s_end)
    return [(float(tn), float(tf))] if tf > tn else []


def prox_segments(lay, p0, u):
    """Path-length intervals of the track p0 + s u (p0 on the plane z = 0, u_z > 0) in the tile, the window and the gas,
    up to the silicon plane z = T + g."""
    p0 = np.asarray(p0, float); u = np.asarray(u, float)
    s_end = (lay.z_sensor-p0[2])/u[2]
    h = lay.tile_half; hs = lay.sensor_half
    tile = box_interval(p0, u, (-h, -h, 0.0), (h, h, lay.tile_t), s_end)
    window = box_interval(p0, u, (-hs, -hs, lay.z_window_bottom), (hs, hs, lay.z_sensor), s_end) if lay.window is not None else []
    gas, s = [], 0.0
    for a, b in sorted(tile+window)+[(s_end, s_end)]:
        if a > s:
            gas.append((s, a))
        s = b
    return {'gas': gas, 'tile': tile, 'window': window, 's_end': float(s_end)}


def check_segments():
    """Segments against hand calculations; raises on mismatch. Returns the cases."""
    lay = make_layout('air', 50.0)
    out = []
    seg = prox_segments(lay, (0, 0, 0), (0, 0, 1.0))                      # vertical, centre
    exp = dict(tile=[(0.0, 10.0)], gas=[(10.0, 59.7)], window=[(59.7, 60.0)])
    out.append(('vertical centre, g = 50', seg, exp))
    c1 = np.cos(np.radians(10)); s1 = np.sin(np.radians(10))              # 10 deg along x: window entry at x = 59.7 tan10 = 10.5
    seg2 = prox_segments(lay, (0, 0, 0), (s1, 0, c1))
    exp2 = dict(tile=[(0.0, 10/c1)], gas=[(10/c1, 59.7/c1)], window=[(59.7/c1, 60/c1)])
    out.append(('10 deg along x from the centre', seg2, exp2))
    c = np.cos(np.radians(30)); s_ = np.sin(np.radians(30))               # 30 deg: window plane reached at x = 34.5, off the sensor
    seg2b = prox_segments(lay, (0, 0, 0), (s_, 0, c))
    exp2b = dict(tile=[(0.0, 10/c)], gas=[(10/c, 60/c)], window=[])
    out.append(('30 deg along x from the centre (misses the window)', seg2b, exp2b))
    seg3 = prox_segments(lay, (20.0, 0, 0), (s_, 0, c))                    # exits the tile side at x = 25: s = 5/sin30 = 10
    exp3 = dict(tile=[(0.0, 10.0)], gas=[(10.0, 60/c)], window=[])        # window entry at x = 20 + 59.7 tan30 = 54.5: outside
    out.append(('30 deg along x from x = 20 (leaves through the tile side)', seg3, exp3))
    for name, seg, exp in out:
        for k in ('tile', 'gas', 'window'):
            got = seg[k]
            if len(got) != len(exp[k]) or any(abs(a-b) > 1e-9 or abs(c_-d) > 1e-9 for (a, c_), (b, d) in zip(got, exp[k])):
                raise AssertionError('%s: %s %s != %s' % (name, k, got, exp[k]))
    return [(n, {k: s[k] for k in ('tile', 'gas', 'window')}) for n, s, _ in out]


def gas_medium(name, fixed=False):
    """Medium of a gas name; fixed=True freezes its index at AEROGEL_LAMBDA_REF."""
    g = air() if name == 'air' else xenon(float(name[2:]))
    if fixed:
        return Medium(g.label+' (n fixed)', ConstantIndex(float(g.n(AEROGEL_LAMBDA_REF))))
    return g


def tile_medium(variant, gas, model='constant', n=AEROGEL_N_AIR):
    """Tile index: the aerogel model (constant n by default) plus the pore-gas shift of the variant, with n_gas(lambda)."""
    base = aerogel_medium(model, n=n)
    if variant == 'off':
        return base
    if variant == 'vac':
        return Medium(base.label+' + 0.97 (n_gas - 1)', lambda lam: aerogel_index(gas.n(lam), n_air=base.n(lam), porosity=AEROGEL_POROSITY))
    if variant == 'air':
        return Medium(base.label+' + 0.935 (n_gas - n_air)',
                      lambda lam: aerogel_index_air(gas.n(lam), n_air=base.n(lam), porosity=AEROGEL_POROSITY_AIRREF, n_gas_ref=N_AIR))
    raise ValueError(variant)


def make_layout(gname, gap, variant='off', *, rayleigh=True, gas_fixed=False, aerogel='constant', n_tile=AEROGEL_N_AIR,
                window=True, norefr=False):
    gas = gas_medium(gname, gas_fixed)
    tm = tile_medium(variant, gas, aerogel, n_tile)
    if norefr:
        gas = Medium('gap with the tile index', ConstantIndex(float(tm.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])))
    return ProxLayout(gas, gap, tm, Rayleigh() if rayleigh else None, Window() if window else None)


def spectra_for(lay, band):
    lo, hi = band
    sp = {'gas': PhotonSpectrum(lay.gas.index, lo, hi), 'tile': PhotonSpectrum(lay.tile_medium.index, lo, hi)}
    if lay.window is not None:
        sp['window'] = PhotonSpectrum(lay.window.medium.index, lo, hi)
    return sp


def trace_track(lay, rng, p0, u, spectra, n_gas, n_tile, n_win, depth=None, max_refl=2):
    """Cherenkov photons of the track p0 + s u from the gap gas, the tile and the window, traced through the scene
    (as sim.muon_photons, also returning the emission directions). depth puts all tile photons at that fraction of the
    tile path, keeping the full-tile normalisation."""
    seg = prox_segments(lay, p0, u)
    parts, norm = [], []
    for org, key, n, reg in ((0, 'gas', n_gas, -1), (1, 'tile', n_tile, 0), (2, 'window', n_win, 1)):
        if n == 0 or not seg[key]:
            continue
        sg = seg[key]; L = sum(b-a for a, b in sg)
        if depth is not None and key == 'tile':
            a, b = sg[0]; s = a+depth*(b-a); sg = [(s, s+1e-9)]
        p, d, lam, E = sim.emit_photons(rng, p0, u, sg, n, spectra[key])
        parts.append((p, d, lam, E, np.full(n, org, np.int8), np.full(n, reg, np.int16)))
        norm.append(np.full(n, spectra[key].yield_per_mm*L/n))
    if not parts:
        return None, None, seg
    p, d, lam, E, org, reg = (np.concatenate(x) for x in zip(*parts))
    ends = tracer.trace(lay.scene(), p, d, lam, reg, E, max_refl, rng=rng)
    xr = ends['reason'].astype(np.int8).copy()
    xr[(xr == R_SIDE)&(ends['p'][:, 2] > lay.tile_t+1e-6)] = len(REASONS)        # side face of the window, not of the tile
    ends['xreason'] = xr
    return {'lam': lam, 'origin': org, 'weight': np.concatenate(norm), 'd': d}, ends, seg


# ---------------------------------------------------------------- deterministic trace

def det_radius(lay, depth, lam=AEROGEL_LAMBDA_REF, n_tile=None, n_gas=None):
    """Landing radius of aerogel photons of a vertical muon emitted at depth fractions depth (0 = tile entry face) at one
    wavelength: (T - z) tan(theta_a) + (g - t_w) tan(theta_gas) + t_w tan(theta_w). Exact on the axis by symmetry."""
    la = np.atleast_1d(lam)
    na = float(lay.tile_medium.n(la)[0]) if n_tile is None else n_tile
    ng = float(lay.gas.n(la)[0]) if n_gas is None else n_gas
    ta = np.arccos(1/na); k = na*np.sin(ta)
    tw = lay.window.thickness if lay.window is not None else 0.0
    nw = lay.window.n if lay.window is not None else ng
    tg = np.arcsin(k/ng); tww = np.arcsin(k/nw)
    z = np.asarray(depth, float)*lay.tile_t
    return (lay.tile_t-z)*np.tan(ta)+(lay.gap-tw)*np.tan(tg)+tw*np.tan(tww)


def formula_radius(lay, depth, lam=AEROGEL_LAMBDA_REF):
    """Proximity ring radius r = g tan(theta_gas) + d tan(theta_a), d = distance of the emission point from the exit face."""
    la = np.atleast_1d(lam); na = float(lay.tile_medium.n(la)[0]); ng = float(lay.gas.n(la)[0])
    ta = np.arccos(1/na); tg = np.arcsin(na*np.sin(ta)/ng)
    return lay.gap*np.tan(tg)+(1-np.asarray(depth, float))*lay.tile_t*np.tan(ta)


DEP = (np.arange(400)+0.5)/400


def ref_radius(lay):
    return float(det_radius(lay, DEP).mean())


def ring_distortion(lay, zen_deg, depth=DEPTH_MID, nphi=720, refract=True):
    """Deterministic ring of one emission point (depth fraction along the tile path) of a muon at zenith zen_deg along x
    through the tile centre, 400 nm, no window: circle-fit radius, rms of the residuals, centre (x, y) relative to the
    muon's crossing of the sensor plane, and the mean angle to the normal in the gas [deg]."""
    th = np.radians(zen_deg); u = np.array([np.sin(th), 0, np.cos(th)])
    na = float(lay.tile_medium.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0]); ng = float(lay.gas.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])
    if not refract:
        ng = na
    pm = np.array([0, 0, lay.tile_t/2]); p0 = pm-(lay.tile_t/2)/u[2]*u
    pe = p0+depth*lay.tile_t/u[2]*u
    phi = np.arange(nphi)/nphi*2*np.pi
    d = tracer.cone(u[None], np.arccos(1/na), phi)[0]
    pb = tracer.to_plane(np.broadcast_to(pe, d.shape), d, lay.tile_t)
    dg, ok = tracer.refract(d, np.array([0, 0, -1.0]), na/ng)
    e = tracer.advance(pb, dg, lay.gap)
    X, Y = e[ok, 0], e[ok, 1]
    A = np.stack([X, Y, np.ones_like(X)], 1); bb = -(X*X+Y*Y)
    D, E_, F = np.linalg.lstsq(A, bb, rcond=None)[0]
    cx, cy = -D/2, -E_/2; r = np.sqrt(cx*cx+cy*cy-F)
    imp = p0+(lay.z_sensor-p0[2])/u[2]*u
    res = np.hypot(X-cx, Y-cy)-r
    return dict(r=float(r), rms=float(res.std()), cx=float(cx-imp[0]), cy=float(cy-imp[1]),
                inc_gas_mean=float(np.degrees(np.arccos(dg[ok, 2])).mean()), inc_gas_max=float(np.degrees(np.arccos(dg[ok, 2])).max()),
                rmin=float(np.hypot(X-cx, Y-cy).min()), rmax=float(np.hypot(X-cx, Y-cy).max()))


# ---------------------------------------------------------------- per-chunk tallies

def weights_of(ph):
    w0 = ph['weight']
    return np.stack([w0*qe_arc(HC/ph['lam']), w0])


def gas_angle(e, ph, lay):
    """Angle [deg] of each detected photon's direction to the sensor normal in the gas before the window (Snell back
    through the parallel window faces) and in the window (its end direction)."""
    cw = np.clip(np.abs(e['d'][:, 2]), 0, 1); sw = np.sqrt(1-cw*cw)
    lam = ph['lam'][e['i']]
    nw = lay.window.n if lay.window is not None else lay.gas.n(lam)
    sg = np.clip(nw*sw/lay.gas.n(lam), 0, 1)
    return np.degrees(np.arcsin(sg)), np.degrees(np.arccos(cw))


def tally(ph, e, lay, ref_a, c0, onaxis=False, maps=False):
    """Weight sums of one traced chunk: emitted, loss reasons, path classes, radial moments and histograms about the
    impact point c0 (x, y), the fine ring histogram around ref_a; onaxis adds pixel and incidence tallies, maps the hit maps."""
    no, nr_, nc_ = len(ORIGINS), len(XREASONS), len(CLASSES)
    ws = weights_of(ph); org = ph['origin']
    emit = np.array([[w[org == o].sum() for o in range(no)] for w in ws])
    o = org[e['i']].astype(np.intp); we = ws[:, e['i']]*e['w']
    key = o*nr_+e['xreason']
    reason = np.stack([np.bincount(key, we[k], minlength=no*nr_).reshape(no, nr_) for k in range(2)])
    det = e['reason'] == R_DET
    X, Y = e['p'][det, 0], e['p'][det, 1]; x, y = X-c0[0], Y-c0[1]; r = np.hypot(x, y)
    od = o[det]; cl = e['cls'][det]; oc = od*nc_+cl; wd = we[:, det]
    mom = np.stack([np.stack([np.bincount(oc, wd[k]*r**p, minlength=no*nc_).reshape(no, nc_) for p in range(3)], -1) for k in range(2)])
    nr = int(round(R_MAX/R_BIN)); ir = np.minimum((r/R_BIN).astype(int), nr-1)
    radial = np.stack([np.bincount(oc*nr+ir, wd[k], minlength=no*nc_*nr).reshape(no, nc_, nr) for k in range(2)])
    out = dict(emit=emit, reason=reason, mom=mom, radial=radial, nbranch=np.array(len(e['i'])))
    nb = int(round(2*CORE_HALF/CORE_BIN)); m = od == 1
    ib = np.floor((r[m]-ref_a+CORE_HALF)/CORE_BIN).astype(int); ok = (ib >= 0)&(ib < nb)
    dr = (cl[m] == 0)[ok]; ib = ib[ok]
    out['core_a'] = np.stack([np.stack([np.bincount(ib[dr], wd[k, m][ok][dr], minlength=nb),
                                        np.bincount(ib, wd[k, m][ok], minlength=nb)]) for k in range(2)])
    wq = wd[0]
    if maps:
        nbm = int(round(2*SENSOR_HALF_2IN/MAP_BIN))
        ix = np.clip(((X+SENSOR_HALF_2IN)/MAP_BIN).astype(int), 0, nbm-1); iy = np.clip(((Y+SENSOR_HALF_2IN)/MAP_BIN).astype(int), 0, nbm-1)
        grp = np.zeros(len(cl), np.intp)
        for gi, cs in enumerate(GROUPS):
            grp[np.isin(cl, cs)] = gi
        out['map'] = np.bincount(((od*len(GROUPS)+grp)*nbm+ix)*nbm+iy, wq, minlength=no*len(GROUPS)*nbm*nbm).reshape(
            no, len(GROUPS), nbm, nbm).astype(np.float32)
    if not onaxis:
        return out
    ph_ = np.arange(NPH)/NPH
    for pi, p in enumerate(PITCHES):
        W = p*np.ceil(PIX_WIN/p); npx = int(round(2*W/p))+1
        cen = (np.abs(x) < W+p)&(np.abs(y) < W+p)
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
            mm = (od == oi)&(cl == 0)
            cx = (np.floor((x[mm, None]+ph_[None, :]*p)/p)+0.5)*p-ph_[None, :]*p
            cy = (np.floor((y[mm, None]+ph_[None, :]*p)/p)+0.5)*p-ph_[None, :]*p
            rp = np.hypot(cx[:, :, None], cy[:, None, :]).reshape(mm.sum(), -1)
            pm[oi] = np.stack([wq[mm]@rp**k for k in range(3)], -1)
        out['pixmom%d' % pi] = pm
    # full-sensor 3.2 mm pixel map, pixel corner (phase 0) or pixel centre (phase 0.5) at the impact point
    fm = np.zeros((2, no, NPIX_FULL, NPIX_FULL))
    for k, phs in enumerate((0.0, 0.5)):
        jx = np.floor(x/PIXEL_PITCH+phs).astype(int)+NPIX_FULL//2; jy = np.floor(y/PIXEL_PITCH+phs).astype(int)+NPIX_FULL//2
        okf = (jx >= 0)&(jx < NPIX_FULL)&(jy >= 0)&(jy < NPIX_FULL)
        fm[k] = np.bincount((od[okf]*NPIX_FULL+jx[okf])*NPIX_FULL+jy[okf], wq[okf], minlength=no*NPIX_FULL**2).reshape(no, NPIX_FULL, NPIX_FULL)
    out['pixfull'] = fm
    ag, aw = gas_angle(dict(d=e['d'][det], i=e['i'][det]), ph, lay)
    na_ = int(round(ANG_MAX/ANG_BIN))
    out['inc'] = np.stack([np.bincount(od*na_+np.minimum((a_/ANG_BIN).astype(int), na_-1), wq*(cl == 0), minlength=no*na_).reshape(no, na_)
                           for a_ in (ag, aw)])
    return out


def run_onaxis_like(task):
    part, key, ci = task
    spec = _CFG['jobs'][(part, key)]
    lay = spec['layout']; rng = np.random.default_rng(spec['seeds'][ci])
    x0, y0 = spec.get('xy', (0.0, 0.0))
    p0 = (x0, y0, 0.0); u = (0.0, 0.0, 1.0)
    ph, e, _ = trace_track(lay, rng, p0, u, spec['spectra'], spec['n_gas'], spec['n_tile'], spec['n_win'], spec.get('depth'))
    return tally(ph, e, lay, spec['ref_a'], (x0, y0), onaxis=spec.get('onaxis', False), maps=spec.get('maps', False))


def circle_fit(X, Y, w):
    A = np.stack([X, Y, np.ones_like(X)], 1)*np.sqrt(w)[:, None]; bb = -(X*X+Y*Y)*np.sqrt(w)
    D, E_, F = np.linalg.lstsq(A, bb, rcond=None)[0]
    cx, cy = -D/2, -E_/2; r = np.sqrt(max(cx*cx+cy*cy-F, 0))
    return cx, cy, r, float(np.sqrt(np.average((np.hypot(X-cx, Y-cy)-r)**2, weights=w)))


def muon_geometry(lay, zen_deg, az_deg, od, off):
    """Direction, entry point on z = 0 and the crossing of the silicon plane of a muon at zenith/azimuth whose track
    crosses the tile mid-plane at offset off along (od = 0) or across (od = 1) its azimuth."""
    th = np.radians(zen_deg); phi = np.radians(az_deg)
    u = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)])
    ed = np.array([np.cos(phi), np.sin(phi), 0.0]) if od == 0 else np.array([-np.sin(phi), np.cos(phi), 0.0])
    pm = off*ed+np.array([0, 0, lay.tile_t/2])
    p0 = pm-(lay.tile_t/2)/u[2]*u
    imp = p0+(lay.z_sensor-p0[2])/u[2]*u
    return u, p0, imp


def run_angle(task):
    key = task
    spec = _CFG['jobs'][('angles', key)]
    lay = spec['layout']; vn, gn, gi, az, od, zi, oi = key
    u, p0, imp = muon_geometry(lay, ZENITH[zi], AZIMUTH[az], od, OFFSET[oi])
    rng = np.random.default_rng(spec['seed'])
    res = dict(p0=p0[:2].copy(), stats=np.full(len(STATS_FIELDS), np.nan), phi=np.zeros((4, NPHI)), reason=np.zeros(len(XREASONS)))
    s = res['stats']; s[16], s[17] = imp[0], imp[1]; s[18] = float((np.abs(imp[0]) < lay.sensor_half)&(np.abs(imp[1]) < lay.sensor_half))
    ph, e, seg = trace_track(lay, rng, p0, u, spec['spectra'], 0, spec['n_tile'], 0)
    s[15] = sum(b-a for a, b in seg['tile'])
    if ph is None:
        s[0] = 0.0
        return key, res
    ws = weights_of(ph); wq = ws[0]
    e1, e2 = tracer.basis(u[None])
    ang = np.mod(np.arctan2(ph['d']@e2[0], ph['d']@e1[0]), 2*np.pi); ib = np.minimum((ang/(2*np.pi)*NPHI).astype(int), NPHI-1)
    i = e['i']; we = wq[i]*e['w']; wf = ws[1, i]*e['w']
    det = e['reason'] == R_DET; dr = e['cls'] == 0
    plane = dr&((e['reason'] == R_DET)|(e['reason'] == R_OUT))
    geo = dr&np.isin(e['reason'], (R_DET, R_SIDE, R_OUT))          # direct photons ending on the SiPM, off it, or on a tile / window side face
    res['reason'] = np.bincount(e['xreason'], we, minlength=len(XREASONS))
    res['phi'][0] = np.bincount(ib, wq, minlength=NPHI)
    res['phi'][1] = np.bincount(ib[i[det&dr]], we[det&dr], minlength=NPHI)
    res['phi'][2] = np.bincount(ib[i[geo]], we[geo], minlength=NPHI)
    res['phi'][3] = np.bincount(ib[i[det]], we[det], minlength=NPHI)
    s[0] = wq.sum(); s[1] = ws[1].sum(); s[2] = we[det].sum(); s[3] = we[det&dr].sum(); s[4] = wf[det].sum(); s[5] = wf[det&dr].sum()
    m = plane
    if we[m].sum() > 0 and m.sum() > 10:
        cx, cy, r, rr = circle_fit(e['p'][m, 0], e['p'][m, 1], we[m])
        s[6], s[7], s[8], s[9] = cx-imp[0], cy-imp[1], r, rr; s[20] = m.sum()
    m = det&dr
    if we[m].sum() > 0:
        ag, aw = gas_angle(dict(d=e['d'][m], i=i[m]), ph, lay); w = we[m]
        s[10] = np.average(aw, weights=w); s[11] = aw.max()
        o_ = np.argsort(ag); cw = np.cumsum(w[o_])/w.sum()
        s[12] = np.average(ag, weights=w); s[13] = ag[o_][min(np.searchsorted(cw, 0.95), len(o_)-1)]; s[14] = ag.max()
        s[19] = np.average(np.hypot(e['p'][m, 0]-imp[0], e['p'][m, 1]-imp[1]), weights=w)
    return key, res


def run_task(task):
    if task[0] == 'angles':
        return run_angle(task[1])
    return run_onaxis_like(task)


def reduce_into(acc, res):
    for k, v in res.items():
        acc[k] = acc[k]+v if k in acc else np.array(v, np.float64 if k != 'map' else np.float32)
    return acc


# ---------------------------------------------------------------- job definitions

def build_jobs(args, parts, band):
    jobs, tasks = {}, []

    def add(part, key, lay, nphot, seed, n_gas=None, n_win=None, **kw):
        nc = max(1, int(np.ceil(nphot/args.chunk)))
        jobs[(part, key)] = dict(layout=lay, spectra=spectra_for(lay, band), seeds=np.random.SeedSequence(seed).spawn(nc),
                                 n_gas=args.chunk if n_gas is None else n_gas, n_tile=args.chunk,
                                 n_win=(args.chunk if lay.window is not None else 0) if n_win is None else n_win, ref_a=ref_radius(lay), **kw)
        tasks.extend((part, key, ci) for ci in range(nc))

    if 'onaxis' in parts:
        for s in range(args.nseeds):
            for vn in VARIANTS:
                for gn in GAS_NAMES:
                    for g in GAPS:
                        add('onaxis', (vn, gn, g, s), make_layout(gn, g, vn), args.nphot, args.seed+s, onaxis=True,
                            maps=(s == 0 and vn == 'off'))
        for pi, xy in enumerate(POINTS):
            for gn in ('air', 'xe3'):
                for g in GAPS:
                    add('points', ('off', gn, g, pi), make_layout(gn, g), args.nphot_points, args.seed, xy=xy, maps=True)
    if 'terms' in parts:
        for gn in GAS_NAMES:
            for g in GAPS:
                for t in TERMS:
                    kw = dict(rayleigh=t in ('full', 'sp_full', 'k4_full'),
                              gas_fixed=t in ('depth', 'point', 'sp_point', 'k4_point', 'norefr', 'nowindow', 'dn'),
                              aerogel={'sp': 'single-pole', 'k4': 'k4geo'}.get(t[:2], 'constant'),
                              n_tile=AEROGEL_N_AIR+DN_TILE if t == 'dn' else AEROGEL_N_AIR,
                              window=t != 'nowindow', norefr=t == 'norefr')
                    vn = {'gasdisp_vac': 'vac', 'gasdisp_air': 'air'}.get(t, 'off')
                    depth = DEPTH_MID if t in ('point', 'gasdisp', 'gasdisp_vac', 'gasdisp_air', 'sp_point', 'k4_point') else None
                    add('terms', (gn, g, t), make_layout(gn, g, vn, **kw), args.nphot_terms, args.seed, n_gas=0, n_win=0, depth=depth)
    if 'angles' in parts:
        for vn in VARIANTS:
            for gn in GAS_NAMES:
                for g in GAPS:
                    lay = make_layout(gn, g, vn); sp = spectra_for(lay, band)
                    for az in range(len(AZIMUTH)):
                        for od in range(len(OFFSET_DIRS)):
                            for zi in range(len(ZENITH)):
                                for oi in range(len(OFFSET)):
                                    if vn != 'off' and (OFFSET[oi] != 0 or od != 0):
                                        continue
                                    key = (vn, gn, g, az, od, zi, oi)
                                    jobs[('angles', key)] = dict(layout=lay, spectra=sp, n_tile=args.nphot_angle, seed=args.seed)
                                    tasks.append(('angles', key))
    return jobs, tasks


# ---------------------------------------------------------------- statistics helpers

def moments(mom, k, oi, classes):
    s = mom[k, oi, list(classes)].sum(0)
    if s[0] <= 0:
        return 0.0, np.nan, np.nan
    m = s[1]/s[0]
    return float(s[0]), float(m), float(np.sqrt(max(s[2]/s[0]-m*m, 0)))


def central_halfwidth(h, bw, q=0.6827):
    c = np.cumsum(h)/h.sum(); x = (np.arange(len(h))+1)*bw
    return float(0.5*(np.interp(0.5+q/2, c, x)-np.interp(0.5-q/2, c, x)))


def quantile_hist(h, x0, bw, q):
    c = np.cumsum(h)/h.sum(); x = x0+(np.arange(len(h))+1)*bw
    return float(np.interp(q, c, x))


def rebin(h, k):
    n = h.shape[-1]//k*k
    return h[..., :n].reshape(*h.shape[:-1], -1, k).sum(-1)


def ring_numbers(a, lay, ref_a, k=0):
    """Ring numbers from reduced (per-muon) tallies, weighting k (0: FT x QE, 1: FT)."""
    out = {}
    na, ma, sa = moments(a['mom'], k, 1, [0])
    out['aero_direct_n'], out['aero_r_mean'], out['aero_r_rms'] = na, ma, sa
    ca = a['core_a'][k]; x0 = ref_a-CORE_HALF
    if ca[0].sum() > 0:
        out['aero_hw68'] = central_halfwidth(ca[0], CORE_BIN)
        out['aero_q025'], out['aero_q975'] = quantile_hist(ca[0], x0, CORE_BIN, 0.025), quantile_hist(ca[0], x0, CORE_BIN, 0.975)
        out['aero_core_frac'] = float(ca[0].sum()/na) if na > 0 else np.nan
    rad = a['radial'][k, 1].sum(0); rc = (np.arange(len(rad))+0.5)*R_BIN; tot = rad.sum()
    lo, hi = det_radius(lay, 1.0), det_radius(lay, 0.0)          # geometric band edges at 400 nm (exit face, entry face)
    out['band_lo'], out['band_hi'] = float(lo), float(hi)
    out['aero_sipm_all'] = float(tot)
    for w in (0.1, 0.5, 2.0):
        out['aero_tail_out_%g' % w] = float(rad[(rc < lo-w)|(rc > hi+w)].sum()/tot) if tot > 0 else np.nan
    # windowed mean of all aerogel photons in the band +- 1 mm (what a reconstruction without truth classes sees)
    mw = (rc > lo-1.0)&(rc < hi+1.0)
    out['aero_win_n'] = float(rad[mw].sum()); out['aero_win_mean'] = float((rad[mw]*rc[mw]).sum()/rad[mw].sum()) if rad[mw].sum() > 0 else np.nan
    ng_, mg, sg = moments(a['mom'], k, 0, [0])
    out['gas_direct_n'], out['gas_r_mean'], out['gas_r_rms'] = ng_, mg, sg
    gr = a['radial'][k, 0, 0]
    out['gas_sipm_all'] = float(a['radial'][k, 0].sum())
    out['gas_r_max'] = float(rc[np.flatnonzero(gr > 1e-6*gr.sum())].max()+R_BIN/2) if gr.sum() > 0 else np.nan
    out['emit'] = a['emit'][k].tolist()
    out['cls'] = (a['radial'][k].sum(-1)).tolist()
    out['reason'] = a['reason'][k].tolist()
    out['reason_frac'] = (a['reason'][k]/np.where(a['emit'][k][:, None] > 0, a['emit'][k][:, None], 1)).tolist()
    return out


def deterministic(lay):
    """Deterministic numbers of a layout at 400 nm: mean / rms / band of r(depth), formula mean, derivatives."""
    r = det_radius(lay, DEP); rf = formula_radius(lay, DEP)
    na = float(lay.tile_medium.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0]); ng = float(lay.gas.n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])
    ta = np.arccos(1/na); tg = np.arcsin(na*np.sin(ta)/ng)
    dn = 1e-5
    rp = det_radius(lay, DEP, n_tile=na+dn).mean(); rm = det_radius(lay, DEP, n_tile=na-dn).mean()
    dth = np.arccos(1/(na+dn))-np.arccos(1/(na-dn))
    gp = det_radius(lay, DEP, n_gas=ng+dn).mean(); gm = det_radius(lay, DEP, n_gas=ng-dn).mean()
    return dict(n_tile=na, n_gas=ng, theta_a=float(ta), theta_gas=float(tg), r_mean=float(r.mean()), r_rms=float(r.std()),
                r_min=float(r.min()), r_max=float(r.max()), formula_mean=float(rf.mean()), formula_min=float(rf.min()), formula_max=float(rf.max()),
                depth_rms_analytic=float(lay.tile_t*np.tan(ta)/np.sqrt(12)),
                dr_dtheta_aero_mm_per_mrad=float((rp-rm)/dth*1e-3), dr_dgap=float(np.tan(tg)), dr_dT=float(np.tan(ta)/2),
                no_refraction_mean=float(((lay.gap+lay.tile_t*(1-DEP))*np.tan(ta)).mean()),
                **{'dr_dn_mm_per_1e-3': float((rp-rm)/(2*dn)*1e-3), 'dr_dngas_mm_per_1e-3': float((gp-gm)/(2*dn)*1e-3)})


# ---------------------------------------------------------------- analysis: on-axis

def analyse_onaxis(out, acc, nseeds, args, summary, accp):
    S = summary.setdefault('onaxis', {})
    nchunk = int(np.ceil(args.nphot/args.chunk))
    S['N'] = dict(photons_per_origin_per_seed=nchunk*args.chunk, seeds=nseeds)
    comb = {}
    for (vn, gn, g, s), a in acc.items():
        comb[(vn, gn, g)] = reduce_into(comb.get((vn, gn, g), {}), {q: v for q, v in a.items() if q != 'map'})
        if 'map' in a:
            comb[(vn, gn, g)]['map'] = a['map']
    for k in comb:
        for q in comb[k]:
            if q not in ('nbranch', 'map'):
                comb[k][q] = comb[k][q]/nseeds
    rows = []
    for vn in VARIANTS:
        for gn in GAS_NAMES:
            for g in GAPS:
                a = comb[(vn, gn, g)]; lay = make_layout(gn, g, vn); ra = ref_radius(lay)
                d = dict(det=deterministic(lay))
                for k, wn in enumerate(('ftqe', 'ft')):
                    d[wn] = ring_numbers(a, lay, ra, k)
                ps = [acc[(vn, gn, g, s)] for s in range(nseeds)]
                d['seed_aero_r'] = [moments(p['mom'], 0, 1, [0])[1] for p in ps]
                d['seed_aero_win'] = [ring_numbers(p, lay, ra, 0)['aero_win_mean'] for p in ps]
                d['seed_aero_n'] = [float(p['radial'][0, 1].sum()) for p in ps]
                rc = (np.arange(int(round(R_MAX/R_BIN)))+0.5)*R_BIN
                win = a['radial'][0, 2].sum(0)
                d['window_n'] = float(win.sum())
                if win.sum() > 0:
                    d['window_r_max'] = float(rc[np.flatnonzero(win > 0)].max()+R_BIN/2)
                    d['window_r_mean'] = float((win*rc).sum()/win.sum())
                inc = a['inc']; ac = (np.arange(inc.shape[-1])+0.5)*ANG_BIN
                for j, nm in ((0, 'gas'), (1, 'win')):
                    h = inc[j, 1]
                    if h.sum() > 0:
                        cw = np.cumsum(h)/h.sum()
                        d['inc_aero_%s_mean' % nm] = float((h*ac).sum()/h.sum()); d['inc_aero_%s_p95' % nm] = float(np.interp(0.95, cw, ac))
                        d['inc_aero_%s_max' % nm] = float(ac[np.flatnonzero(h > 0).max()]+ANG_BIN/2)
                rad = a['radial'][0, 1]
                for ci, cn in enumerate(CLASSES):
                    h = rad[ci]; d['aero_cls_%s' % cn] = float(h.sum())
                    if h.sum() > 0 and cn != 'direct':
                        cw = np.cumsum(h)/h.sum(); d['aero_cls_%s_r' % cn] = [float(np.interp(q, cw, rc)) for q in (0.05, 0.5, 0.95)]
                sc = rad[CLASSES.index('scattered')]
                for w in (0.5, 1.6):
                    d['aero_scattered_within_%g' % w] = float(sc[(rc > d['ftqe']['band_lo']-w)&(rc < d['ftqe']['band_hi']+w)].sum())
                # the muon's own pixels (window-cluster pixels with >= 2 % of its light, plus the impact pixel)
                for pi, p in enumerate(PITCHES):
                    hist = a['pix%d' % pi]
                    wpx = hist[2] > 0.02*hist[2].sum((-1, -2), keepdims=True)
                    W = p*np.ceil(PIX_WIN/p)
                    fa, fg, nwp, ga = [], [], [], []
                    for ia in range(NPH):
                        for ib_ in range(NPH):
                            cxp = int(np.floor((W+ia/NPH*p)/p)); cyp = int(np.floor((W+ib_/NPH*p)/p))
                            occ = wpx[ia, ib_].copy(); occ[cxp, cyp] = True
                            fa.append(hist[1, ia, ib_][occ].sum()); fg.append(hist[0, ia, ib_][occ].sum()); nwp.append(occ.sum())
                            ga.append(hist[0, ia, ib_][~occ].sum())
                    gtot = a['radial'][0, 0].sum()
                    d['pix%g' % p] = dict(aero_in_muon_pixels=float(np.mean(fa)), aero_in_muon_pixels_max=float(np.max(fa)),
                                          gas_in_muon_pixels=float(np.mean(fg)), gas_frac_outside=float(np.mean(ga)/gtot) if gtot > 0 else np.nan,
                                          gas_frac_outside_min=float(np.min(ga)/gtot) if gtot > 0 else np.nan, muon_pixels=float(np.mean(nwp)))
                    pm = a['pixmom%d' % pi]
                    for oi, nm in ((1, 'aero'), (0, 'gas')):
                        s0, s1, s2 = pm[oi, :, 0], pm[oi, :, 1], pm[oi, :, 2]
                        with np.errstate(invalid='ignore', divide='ignore'):
                            mph = s1/s0; rph = np.sqrt(np.maximum(s2/s0-mph**2, 0))
                        d['pix%g' % p]['%s_r_pix_mean' % nm] = [float(np.nanmin(mph)), float(np.nanmean(mph)), float(np.nanmax(mph)), float(np.nanstd(mph))]
                        d['pix%g' % p]['%s_r_pix_rms' % nm] = [float(np.nanmin(rph)), float(np.nanmean(rph)), float(np.nanmax(rph))]
                pf = a['pixfull'][:, 1]
                d['aero_pixels_3.2'] = [int((pf[k_] > 0.005*pf[k_].sum()).sum()) for k_ in range(2)]
                S['%s_%s_%g' % (vn, gn, g)] = d
                q = d['ftqe']
                rows.append([vn, gn, '%g' % g, q['aero_r_mean'], q['aero_r_rms'], q['aero_hw68'], d['det']['r_mean'], d['det']['formula_mean'],
                             q['aero_direct_n'], q['aero_sipm_all'], d['aero_cls_scattered'], q['gas_sipm_all'], q['gas_r_mean'], d['window_n'],
                             np.std(d['seed_aero_r'], ddof=1) if nseeds > 1 else np.nan])
    head = ['variant', 'gas', 'gap_mm', 'aero_r_mean_mm', 'aero_r_rms_mm', 'aero_hw68_mm', 'det_r_mm', 'formula_r_mm', 'aero_direct_sipm',
            'aero_all_sipm', 'aero_scattered_sipm', 'gas_sipm', 'gas_r_mean_mm', 'window_sipm', 'aero_r_seed_std_mm']
    with open(out/'prox_onaxis.csv', 'w') as fh:
        fh.write(','.join(head)+'\n')
        for r in rows:
            fh.write(','.join(r[:3]+['%.6g' % v for v in r[3:]])+'\n')
    # gas - air radius differences per seed (matched seeds), per variant and gap; and rings needed
    D = S['delta_r_vs_air'] = {}
    for vn in VARIANTS:
        for g in GAPS:
            ra = np.array(S['%s_air_%g' % (vn, g)]['seed_aero_r']); rw = np.array(S['%s_air_%g' % (vn, g)]['seed_aero_win'])
            for gn in GAS_NAMES[1:]:
                dx = np.array(S['%s_%s_%g' % (vn, gn, g)]['seed_aero_r'])-ra
                dw = np.array(S['%s_%s_%g' % (vn, gn, g)]['seed_aero_win'])-rw
                det_ = S['%s_%s_%g' % (vn, gn, g)]['det']['r_mean']-S['%s_air_%g' % (vn, g)]['det']['r_mean']
                D['%s_%s_%g' % (vn, gn, g)] = dict(mean=float(dx.mean()), std=float(dx.std(ddof=1)) if nseeds > 1 else np.nan, per_seed=dx.tolist(),
                                                   win_mean=float(dw.mean()), win_std=float(dw.std(ddof=1)) if nseeds > 1 else np.nan, det_400nm=float(det_))
    R = S['rings_needed'] = {}
    for g in GAPS:
        for p in PITCHES:
            d = S['off_air_%g' % g]; px = d['pix%g' % p]
            n = d['ftqe']['aero_direct_n']
            var1 = px['aero_r_pix_rms'][1]**2/n+px['aero_r_pix_mean'][3]**2       # per ring: photon spread / N + grid-phase spread
            var1_true = d['ftqe']['aero_r_rms']**2/n
            row = dict(sigma_ring_mm=float(np.sqrt(var1)), sigma_ring_nopix_mm=float(np.sqrt(var1_true)), n_photons=n,
                       pix_rms_mm=px['aero_r_pix_rms'][1], pix_phase_std_mm=px['aero_r_pix_mean'][3])
            row['rings_for_10um'] = float(var1/0.010**2)
            for vn in VARIANTS:
                dlt = D['%s_xe3_%g' % (vn, g)]['mean']
                row['rings_5sigma_xe3_%s' % vn] = float(2*var1*(5/dlt)**2) if dlt != 0 else np.inf
            R['%g_%g' % (g, p)] = row
    fig_onaxis_radial(out, comb, S)
    fig_maps(out, comb, accp, S)
    fig_centre(out, comb, S)
    fig_signal(out, S)
    return comb


def analyse_points(out, acc, args, summary):
    S = summary.setdefault('points', {}); S['N'] = dict(photons_per_origin=int(np.ceil(args.nphot_points/args.chunk))*args.chunk)
    S['points_mm'] = [list(p) for p in POINTS]
    for (vn, gn, g, pi), a in acc.items():
        lay = make_layout(gn, g, vn); q = ring_numbers(a, lay, ref_radius(lay), 0)
        S['%s_%s_%g_%d' % (vn, gn, g, pi)] = dict(xy=list(POINTS[pi]), aero_direct_n=q['aero_direct_n'], aero_sipm_all=q['aero_sipm_all'],
                                                  aero_r_mean=q['aero_r_mean'], emit=q['emit'], reason_frac=q['reason_frac'][1], cls=q['cls'][1],
                                                  gas_sipm_all=q['gas_sipm_all'], window_n=float(a['radial'][0, 2].sum()))


def fig_onaxis_radial(out, comb, S):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for j, g in enumerate(GAPS):
        a = ax[0, j]; rb = 10
        rad = rebin(comb[('off', 'xe3', g)]['radial'][0], rb)/(R_BIN*rb); rc = (np.arange(rad.shape[-1])+0.5)*R_BIN*rb
        for oi, nm, col in ((1, 'aerogel', AEROGEL), (0, 'gap gas', GAS), (2, 'window', SCINT_EDGE)):
            for gi, (cs, gname) in enumerate(zip(GROUPS, GROUP_NAMES)):
                y = rad[oi, list(cs)].sum(0)
                if y.sum()*R_BIN*rb > 1e-5:
                    a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=col, lw=1.4 if gi == 0 else 0.9, ls=('-', ':', '-.')[gi],
                           label='%s, %s (%.3g / muon)' % (nm, gname, y.sum()*R_BIN*rb))
        a.set_yscale('log'); a.set_xlim(0, R_MAX); a.set_ylim(1e-4, None); a.grid(alpha=.3)
        a.axvline(SENSOR_HALF_2IN, color=MUTED, lw=.8, ls=':')
        a.set_xlabel('distance r from the muon impact point [mm] (dotted: sensor half side)'); a.set_ylabel('dN/dr [photons / mm / muon], FT x QE')
        a.set_title('gap %g mm, Xe 3 bar, pore shift off: all photons on the SiPM, %.1f mm bins' % (g, R_BIN*rb), fontsize=10)
        a.legend(fontsize=7, loc='upper right')
        a = ax[1, j]; rb = 25; bw = CORE_BIN*rb
        for vn, ls in (('off', '-'), ('vac', '--'), ('air', ':')):
            for gn in (GAS_NAMES if vn == 'off' else ('xe3',)):
                d = S['%s_%s_%g' % (vn, gn, g)]; ca = comb[(vn, gn, g)]['core_a'][0, 1]
                h = rebin(ca, rb); xc = d['det']['r_mean']-CORE_HALF+(np.arange(len(h))+0.5)*bw
                a.step(xc, h/bw, where='mid', color=GAS_COLOUR[gn], ls=ls, lw=1.2,
                       label='%s, %s: <r> %.3f mm, rms %.0f um' % (GAS_LABEL[gn], VAR_LABEL[vn], d['ftqe']['aero_r_mean'], 1e3*d['ftqe']['aero_r_rms']))
        d = S['off_air_%g' % g]['det']
        a.axvspan(d['formula_min'], d['formula_max'], color=MUTED, alpha=0.12, label='formula g tan(th_gas) + d tan(th_a), air')
        a.set_xlim(d['r_min']-1.2, d['r_max']+1.6); a.grid(alpha=.3)
        a.set_xlabel('r [mm]'); a.set_ylabel('dN/dr [photons / mm / muon], FT x QE')
        a.set_ylim(0, None)
        a.set_title('gap %g mm: aerogel ring, all aerogel paths, %.0f um bins' % (g, 1e3*bw), fontsize=10); a.legend(fontsize=6, loc='lower center')
    fig.suptitle('Vertical muon through the tile centre: radial hit profiles (FT x QE)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save(out, 'prox_onaxis_radial.png', dpi=110, fig=fig); plt.close(fig)


def fig_maps(out, comb, accp, S):
    fig, ax = plt.subplots(2, 3, figsize=(17, 11))
    cmap = plt.get_cmap('magma_r').copy(); cmap.set_bad('white')
    panels = [(comb[('off', 'xe3', g)]['map'], 'gap %g mm, impact (0, 0)' % g, (0, 0)) for g in GAPS]
    for pi in (0, 1, 3):
        a_ = accp[('off', 'xe3', 50.0, pi)]
        panels.append((a_['map']/a_['nchunk'], 'gap 50 mm, impact (%g, %g) mm' % POINTS[pi], POINTS[pi]))
    for a, (m, title, xy) in zip(ax.ravel(), panels):
        m = m.sum((0, 1))/MAP_BIN**2
        im = a.imshow(np.ma.masked_less(m.T, m.max()*1e-5), origin='lower', extent=(-SENSOR_HALF_2IN, SENSOR_HALF_2IN)*2,
                      norm=LogNorm(m.max()*1e-5, m.max()), cmap=cmap, interpolation='nearest')
        for gl in np.arange(-8, 9)*PIXEL_PITCH:
            a.axvline(gl, color=MUTED, lw=0.3, alpha=0.5); a.axhline(gl, color=MUTED, lw=0.3, alpha=0.5)
        a.plot([-AEROGEL_HALF, AEROGEL_HALF, AEROGEL_HALF, -AEROGEL_HALF, -AEROGEL_HALF], [-AEROGEL_HALF, -AEROGEL_HALF, AEROGEL_HALF, AEROGEL_HALF, -AEROGEL_HALF],
               color=AEROGEL, lw=0.8, ls='--')
        a.plot(*xy, '+', color=INK, ms=8)
        a.set_xlim(-SENSOR_HALF_2IN, SENSOR_HALF_2IN); a.set_ylim(-SENSOR_HALF_2IN, SENSOR_HALF_2IN); a.set_aspect('equal')
        a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]'); a.set_title('Xe 3 bar, %s' % title, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label='photons / mm² / muon')
    fig.suptitle('SiPM hit density, vertical muons, all origins and paths, FT x QE (log; grid: 3.2 mm pixels; dashed: tile outline; +: impact point)',
                 fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(out, 'prox_maps.png', dpi=110, fig=fig); plt.close(fig)


def fig_centre(out, comb, S):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))
    rc = (np.arange(int(round(R_MAX/R_BIN)))+0.5)*R_BIN
    for j, g in enumerate(GAPS):
        a = ax[0, j]; rb = 5
        for gn in GAS_NAMES:
            rad = rebin(comb[('off', gn, g)]['radial'][0], rb)/(R_BIN*rb); rr = rebin(rc[None], rb)[0]/rb
            y = rad[0].sum(0)
            a.step(rr, np.where(y > 0, y, np.nan), where='mid', color=GAS_COLOUR[gn], lw=1.2,
                   label='%s gap-gas light (%.2f / muon, r < %.2f mm)' % (GAS_LABEL[gn], S['off_%s_%g' % (gn, g)]['ftqe']['gas_sipm_all'],
                                                                         S['off_%s_%g' % (gn, g)]['ftqe']['gas_r_max']))
        rad = rebin(comb[('off', 'xe3', g)]['radial'][0], rb)/(R_BIN*rb)
        y = rad[2].sum(0); a.step(rr, np.where(y > 0, y, np.nan), where='mid', color=SCINT_EDGE, lw=1.4,
                                  label='window cluster (%.1f / muon + %g cell)' % (S['off_xe3_%g' % g]['window_n'], MIP_DIRECT_PE))
        y = rad[1].sum(0); a.step(rr, np.where(y > 0, y, np.nan), where='mid', color=AEROGEL, lw=1.0, label='aerogel, all paths (Xe 3 bar)')
        for k in range(1, 8):
            a.axvline(k*PIXEL_PITCH, color=MUTED, lw=.5, ls=':')
        a.set_yscale('log'); a.set_xlim(0, 25); a.set_ylim(1e-3, None); a.grid(alpha=.3)
        a.set_xlabel('r from the impact point [mm] (dotted: multiples of 3.2 mm)'); a.set_ylabel('dN/dr [photons / mm / muon]')
        a.set_title('gap %g mm: centre region, %.2f mm bins' % (g, R_BIN*rb), fontsize=10); a.legend(fontsize=6.5)
        a = ax[1, j]
        pf = comb[('off', 'xe3', g)]['pixfull'][0]        # pixel corner at the impact point
        tot = pf.sum(0); tmax = tot.max(); rgb = np.array([to_rgb(GAS), to_rgb(AEROGEL), to_rgb(SCINT_EDGE)])
        h0 = NPIX_FULL//2*PIXEL_PITCH
        for ix, iy in zip(*np.nonzero(tot > 1e-3)):
            col = (pf[:, ix, iy]/tot[ix, iy])@rgb
            x0 = -h0+ix*PIXEL_PITCH; y0 = -h0+iy*PIXEL_PITCH
            a.add_patch(plt.Rectangle((x0, y0), PIXEL_PITCH, PIXEL_PITCH, facecolor=col, edgecolor='none', alpha=0.15+0.85*np.sqrt(tot[ix, iy]/tmax)))
            if tot[ix, iy] > 0.05:
                a.text(x0+PIXEL_PITCH/2, y0+PIXEL_PITCH/2, '%.1f' % tot[ix, iy], ha='center', va='center', fontsize=5.5, color=INK)
        for k in range(NPIX_FULL+1):
            a.axvline(-h0+k*PIXEL_PITCH, color=MUTED, lw=0.3); a.axhline(-h0+k*PIXEL_PITCH, color=MUTED, lw=0.3)
        a.set_xlim(-h0, h0); a.set_ylim(-h0, h0); a.set_aspect('equal'); a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]')
        a.set_title('gap %g mm, Xe 3 bar, 3.2 mm pixels, corner at the impact point\n(photons per muon per pixel, shown if > 0.05)' % g, fontsize=9)
    fig.suptitle("The muon's own hit (window light), the gap-gas light and the aerogel ring (vertical muon, FT x QE; pixel colour: blue gap gas, "
                 "green window, orange aerogel)", fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(out, 'prox_centre.png', dpi=110, fig=fig); plt.close(fig)


def fig_signal(out, S):
    D = S['delta_r_vs_air']; R = S['rings_needed']
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    P = np.array([1, 2, 3])
    for g in GAPS:
        for vn, mk in (('off', 'o'), ('vac', 's'), ('air', '^')):
            y = np.array([D['%s_%s_%g' % (vn, gn, g)]['mean'] for gn in GAS_NAMES[1:]])*1e3
            e = np.array([D['%s_%s_%g' % (vn, gn, g)]['std'] for gn in GAS_NAMES[1:]])*1e3
            a = ax[0] if vn != 'off' else ax[1]
            a.errorbar(P+(GAPS.index(g)-1)*0.06, y, e, color=GAP_COLOUR[g], marker=mk, ls=GAP_STYLE[g], ms=5, capsize=2,
                       label='gap %g mm, %s' % (g, VAR_LABEL[vn]))
    ax[0].set_ylabel('<r>(Xe) - <r>(air) [um], direct aerogel photons'); ax[1].set_ylabel('<r>(Xe) - <r>(air) [um] (refraction only)')
    ax[0].set_title('Ring radius shift with the pore-gas shift on (two conventions)', fontsize=10)
    ax[1].set_title('Ring radius shift with the pore-gas shift off', fontsize=10)
    for a in ax[:2]:
        a.set_xlabel('Xe pressure [bar]'); a.set_xticks(P); a.grid(alpha=.3); a.legend(fontsize=7)
    a = ax[2]
    for g in GAPS:
        for p, ls in zip(PITCHES, ('-', '--')):
            r = R['%g_%g' % (g, p)]
            a.bar(GAPS.index(g)*4+PITCHES.index(p)*2+np.array([0, 0.9]), [r['rings_5sigma_xe3_vac'], r['rings_5sigma_xe3_air']], 0.85,
                  color=[GAS_COLOUR['xe2'], GAS_COLOUR['xe3']], edgecolor=INK, lw=0.4, alpha=1 if p == PIXEL_PITCH else 0.55)
    a.set_xticks([GAPS.index(g)*4+PITCHES.index(p)*2+0.45 for g in GAPS for p in PITCHES])
    a.set_xticklabels(['%g mm\n%.1f mm px' % (g, p) for g in GAPS for p in PITCHES], fontsize=8)
    a.set_yscale('log'); a.grid(alpha=.3, axis='y'); a.set_ylabel('rings per gas for a 5 sigma Xe 3 bar - air shift')
    a.set_title('Rings needed (blue: 0.97 (n_gas - 1); red: 0.935 (n_gas - n_air))\nper-photon spread from the pixel-centre radii, impact point known',
                fontsize=9)
    fig.suptitle('Air vs Xe ring radius (vertical muon, FT x QE, matched seeds; error bars: std over seeds)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(out, 'prox_signal.png', dpi=110, fig=fig); plt.close(fig)


# ---------------------------------------------------------------- analysis: terms

def analyse_terms(out, acc, args, summary):
    S = summary.setdefault('terms', {}); S['N'] = dict(photons_per_variant=int(np.ceil(args.nphot_terms/args.chunk))*args.chunk)
    for gn in GAS_NAMES:
        for g in GAPS:
            d = {}
            base = make_layout(gn, g, rayleigh=False, gas_fixed=True)
            for t in TERMS:
                a = acc[(gn, g, t)]
                kw = dict(n_tile=AEROGEL_N_AIR+DN_TILE if t == 'dn' else AEROGEL_N_AIR, window=t != 'nowindow', norefr=t == 'norefr',
                          gas_fixed=t in ('depth', 'point', 'sp_point', 'k4_point', 'norefr', 'nowindow', 'dn'))
                lt = make_layout(gn, g, {'gasdisp_vac': 'vac', 'gasdisp_air': 'air'}.get(t, 'off'), **kw)
                d[t] = ring_numbers(a, lt, ref_radius(lt), 0)
                d[t]['det_r_mean'] = ref_radius(lt)
                sc = a['radial'][0, 1, CLASSES.index('scattered')]; rc = (np.arange(len(sc))+0.5)*R_BIN
                d[t]['aero_scattered'] = float(sc.sum())
                for w in (0.5, 1.6):
                    d[t]['aero_scattered_within_%g' % w] = float(sc[(rc > d[t]['band_lo']-w)&(rc < d[t]['band_hi']+w)].sum())
            d['det'] = deterministic(base)
            for p in PITCHES:
                d['pixel_term_mm_%g' % p] = float(p/np.sqrt(12))
                d['pixel_term_mrad_%g' % p] = float(p/np.sqrt(12)/d['det']['dr_dtheta_aero_mm_per_mrad'])
            d['depth_term_mrad'] = float(d['depth']['aero_r_rms']/d['det']['dr_dtheta_aero_mm_per_mrad'])
            d['distortion'] = {('%g' % z): {('refr' if rf else 'norefr'): ring_distortion(base, z, refract=rf) for rf in (True, False)}
                               for z in (0, 10, 20, 30)}
            S['%s_%g' % (gn, g)] = d
    fig_terms(out, S)


def fig_terms(out, S):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    a = ax[0]
    for g in GAPS:
        lay = make_layout('air', g, rayleigh=False, gas_fixed=True)
        r = det_radius(lay, DEP); rf = formula_radius(lay, DEP)
        a.plot(10*DEP, r, color=GAP_COLOUR[g], lw=1.6, label='gap %g mm: trace with window, rms %.0f um, <r> %.3f mm' % (g, 1e3*r.std(), r.mean()))
        a.plot(10*DEP, rf, color=GAP_COLOUR[g], lw=0.8, ls='--', label='gap %g mm: formula g tan th_gas + d tan th_a (%+.0f um)' % (g, 1e3*(rf-r).mean()))
    a.set_xlabel('emission depth below the tile entry face [mm]'); a.set_ylabel('landing radius r [mm]')
    a.set_title('Air, 400 nm: landing radius vs emission depth (vertical muon)', fontsize=10); a.grid(alpha=.3); a.legend(fontsize=7)
    a = ax[1]
    terms = (('depth', 'depth'), ('gasdisp', 'gas\ndisp.'), ('gasdisp_vac', 'gas disp.\n+pore 0.97'), ('point', 'none\n(check)'),
             ('sp_point', 'aero disp.\n1-pole'), ('k4_point', 'aero disp.\nk4geo'), ('norayl', 'all but\nRayleigh'), ('full', 'full'))
    xs = np.arange(len(terms)); wbar = 0.13; k = 0
    for g in GAPS:
        for gn in ('air', 'xe3'):
            v = [1e3*S['%s_%g' % (gn, g)][t]['aero_r_rms'] for t, _ in terms]
            a.bar(xs+(k-2.5)*wbar, np.maximum(v, 1e-3), wbar, color=GAP_COLOUR[g], alpha=1 if gn == 'air' else 0.5, edgecolor=INK, lw=0.3,
                  hatch='' if gn == 'air' else '//', label='gap %g mm, %s' % (g, GAS_LABEL[gn]))
            k += 1
    for p, ls in zip(PITCHES, ('--', ':')):
        a.axhline(1e3*p/np.sqrt(12), color=INK, ls=ls, lw=1, label='pixel %.1f mm / sqrt(12)' % p)
    a.set_yscale('log'); a.set_xticks(xs); a.set_xticklabels([t[1] for t in terms], fontsize=7.5)
    a.set_ylabel('rms of the direct aerogel-ring radius [um] (0 drawn at 1e-3)'); a.grid(alpha=.3, axis='y')
    a.set_title('Ring-width terms, one at a time (direct photons, FT x QE)', fontsize=10); a.legend(fontsize=6.5, ncol=2)
    a = ax[2]
    Z = (0, 10, 20, 30)
    for g in GAPS:
        d = S['air_%g' % g]['distortion']
        a.plot(Z, [1e3*d['%g' % z]['refr']['rms'] for z in Z], color=GAP_COLOUR[g], marker='o', label='gap %g mm, with refraction' % g)
        a.plot(Z, [1e3*d['%g' % z]['norefr']['rms'] for z in Z], color=GAP_COLOUR[g], marker='x', ls='--', lw=0.8, label='gap %g mm, no refraction' % g)
    a.set_yscale('symlog', linthresh=1); a.set_xlabel('muon zenith angle [deg] (single emission point, 400 nm)')
    a.set_ylabel('rms of the ring about the fitted circle [um]'); a.grid(alpha=.3); a.legend(fontsize=7)
    a.set_title('Ring distortion from inclination, air: refraction at the exit face vs none', fontsize=10)
    fig.suptitle('Ring-width budget of proximity focusing (itemised, not combined)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(out, 'prox_terms.png', dpi=110, fig=fig); plt.close(fig)


# ---------------------------------------------------------------- analysis: angles

def max_zenith(curve, thr):
    bad = np.flatnonzero(~(np.asarray(curve) >= thr-1e-9))
    if not len(bad):
        return float(ZENITH[-1])
    return float(ZENITH[bad[0]-1]) if bad[0] > 0 else np.nan


def prox_reference(gn, g, u, p0):
    """sim.proximity_fractions for the same muons (its frame: tile top z = 0, muons going down): (fa, arc)."""
    n_g = float(gas_medium(gn).n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0])
    uu = u.copy(); uu[:, 2] = -uu[:, 2]
    return sim.proximity_fractions(Aerogel(AEROGEL_N_AIR), Sensor(SENSOR_HALF_2IN), g, n_g, uu, p0, theta_c(AEROGEL_N_AIR))


def analyse_angles(out, res, args, summary):
    S = summary.setdefault('angles', {}); S['N'] = dict(photons_per_config=args.nphot_angle)
    shape = (len(VARIANTS), len(GAS_NAMES), len(GAPS), len(AZIMUTH), len(OFFSET_DIRS), len(ZENITH), len(OFFSET))
    st = np.full(shape+(len(STATS_FIELDS),), np.nan); phi = np.full(shape+(4, NPHI), np.nan); rs = np.full(shape+(len(XREASONS),), np.nan)
    p0 = np.full(shape+(2,), np.nan)
    for key, r in res.items():
        vn, gn, g, az, od, zi, oi = key
        idx = (VARIANTS.index(vn), GAS_NAMES.index(gn), GAPS.index(g), az, od, zi, oi)
        st[idx] = r['stats']; phi[idx] = r['phi']; rs[idx] = r['reason']; p0[idx] = r['p0']
    emit = st[..., 0]
    i0 = list(OFFSET).index(0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        f_det = st[..., 2]/emit; f_dir = st[..., 3]/emit
        f_out = (rs[..., 3]+rs[..., len(REASONS)])/emit; f_sides = rs[..., 2]/emit
        # per emission-azimuth bin: direct photons on the SiPM / direct photons ending on the SiPM, off it or on a side face
        perbin = np.where(phi[..., 2, :] > 0, phi[..., 1, :]/np.where(phi[..., 2, :] > 0, phi[..., 2, :], 1), 0.0)
        arc = perbin.mean(-1)                         # geometric azimuth coverage of the ring on the SiPM
        arc_hit = np.mean(perbin >= 0.5, -1)          # azimuth bins with >= half of their direct light on the SiPM
        rel = f_det/f_det[:, :, :, 0, 0, 0, i0][:, :, :, None, None, None, None]
    nolight = emit <= 0; notrun = np.isnan(emit)
    for x_ in (arc, arc_hit, rel):
        x_[nolight] = 0.0; x_[notrun] = np.nan
    # geometric reference: sim.proximity_fractions on the same muons (variant off)
    pf = np.full(shape[1:], np.nan); parc = np.full(shape[1:], np.nan)
    for gi, gn in enumerate(GAS_NAMES):
        for ji, g in enumerate(GAPS):
            lay = make_layout(gn, g)
            U, P0, IDX = [], [], []
            for az in range(len(AZIMUTH)):
                for od in range(len(OFFSET_DIRS)):
                    for zi in range(len(ZENITH)):
                        for oi in range(len(OFFSET)):
                            u, pp, _ = muon_geometry(lay, ZENITH[zi], AZIMUTH[az], od, OFFSET[oi])
                            U.append(u); P0.append(pp); IDX.append((az, od, zi, oi))
            fa, ar = prox_reference(gn, g, np.array(U), np.array(P0))
            for (az, od, zi, oi), x, y in zip(IDX, fa, ar):
                pf[gi, ji, az, od, zi, oi] = x; parc[gi, ji, az, od, zi, oi] = y
    np.savez(out/'prox_angles.npz', variant=np.array(VARIANTS), gas=np.array(GAS_NAMES), gap_mm=np.array(GAPS), azimuth_deg=np.array(AZIMUTH),
             offset_dir=np.array(OFFSET_DIRS), zenith_deg=ZENITH, offset_mm=OFFSET, stats=st, stats_fields=np.array(STATS_FIELDS), phi=phi,
             phi_fields=np.array(['emitted', 'direct_on_sipm', 'direct_geometric_end', 'all_on_sipm']), reasons_w=rs, reasons=np.array(XREASONS),
             entry_xy_mm=p0, f_det=f_det, f_direct=f_dir, f_outside_sensor=f_out, f_tile_sides=f_sides, arc_frac=arc, arc_hit=arc_hit, rel_det=rel,
             prox_fa=pf, prox_arc=parc,
             axes=np.array(['variant', 'gas', 'gap_mm', 'azimuth_deg', 'offset_dir', 'zenith_deg', 'offset_mm']),
             note=np.array('beta = 1, FT x QE weights; offsets of variants vac/air computed at 0 only (nan elsewhere); entry_xy at z = 0 of the '
                           'mirror-image frame = tile entry face; impact = crossing of the silicon plane; prox_* = sim.proximity_fractions '
                           '(variant off) on the same muons'))
    S['axes'] = 'arrays [variant, gas, gap, azimuth, offset_dir, zenith, offset]'
    for vi, vn in enumerate(VARIANTS):
        for gi, gn in enumerate(GAS_NAMES):
            for ji, g in enumerate(GAPS):
                d = {}
                for az in range(len(AZIMUTH)):
                    a1 = arc[vi, gi, ji, az, 0, :, i0]; r1 = rel[vi, gi, ji, az, 0, :, i0]; tag = 'az%g' % AZIMUTH[az]
                    for thr in (0.999, 0.99, 0.95, 0.8, 0.5):
                        d['arc_%g_%s' % (thr, tag)] = max_zenith(a1, thr); d['rel_%g_%s' % (thr, tag)] = max_zenith(r1, thr)
                    d['rel_%s' % tag] = r1.tolist(); d['arc_%s' % tag] = a1.tolist()
                    if vi == 0:
                        d['prox_fa_%s' % tag] = pf[gi, ji, az, 0, :, i0].tolist()
                        d['prox_fa80_%s' % tag] = max_zenith(pf[gi, ji, az, 0, :, i0], 0.8)
                        d['prox_fa100_%s' % tag] = max_zenith(pf[gi, ji, az, 0, :, i0], 0.999)
                    for fi, fn in enumerate(STATS_FIELDS):
                        if fn in ('centre_x_mm', 'centre_y_mm', 'fit_r_mm', 'fit_r_rms_mm', 'inc_win_mean_deg', 'inc_gas_mean_deg', 'inc_gas_max_deg',
                                  'inc_gas_p95_deg', 'mean_r_from_impact_mm'):
                            d['%s_%s' % (fn, tag)] = st[vi, gi, ji, az, 0, :, i0, fi].tolist()
                d['f_det_onaxis'] = float(f_det[vi, gi, ji, 0, 0, 0, i0]); d['f_direct_onaxis'] = float(f_dir[vi, gi, ji, 0, 0, 0, i0])
                d['emit_onaxis'] = float(emit[vi, gi, ji, 0, 0, 0, i0])
                if vi == 0:
                    for az in range(len(AZIMUTH)):
                        for od in range(len(OFFSET_DIRS)):
                            m = rel[vi, gi, ji, az, od]
                            d['cells_rel80_az%g_%s' % (AZIMUTH[az], OFFSET_DIRS[od])] = float(np.mean(m >= 0.8))
                            d['cells_rel95_az%g_%s' % (AZIMUTH[az], OFFSET_DIRS[od])] = float(np.mean(m >= 0.95))
                    # vertical muons: containment vs offset (zenith 0, offset parallel to the edge)
                    d['rel_vs_offset_az0'] = rel[vi, gi, ji, 0, 0, 0].tolist(); d['rel_vs_offset_az45'] = rel[vi, gi, ji, 1, 0, 0].tolist()
                S['%s_%s_%g' % (vn, gn, g)] = d
    gi0 = GAS_NAMES.index('air')
    S['max_abs_rel_diff_xe_vs_air'] = float(np.nanmax(np.abs(rel[0, 1:]-rel[0, gi0:gi0+1])))
    S['max_abs_arc_diff_xe_vs_air'] = float(np.nanmax(np.abs(arc[0, 1:]-arc[0, gi0:gi0+1])))
    S['max_abs_rel_diff_variants'] = float(np.nanmax(np.abs(rel[1:, :, :, :, :, :, i0]-rel[0:1, :, :, :, :, :, i0])))
    # MC vs the geometric toy: relative containment difference (variant off, all cells)
    with np.errstate(invalid='ignore'):
        dd = rel[0]-pf
    S['mc_minus_prox_rel'] = dict(mean=float(np.nanmean(dd)), max_abs=float(np.nanmax(np.abs(dd))), p95_abs=float(np.nanpercentile(np.abs(dd), 95)))
    fig_angles_maps(out, rel, arc, S)
    fig_angles_curves(out, rel, arc, st, pf, S)
    return dict(rel=rel, arc=arc, st=st)


def fig_angles_maps(out, rel, arc, S):
    gi = GAS_NAMES.index('air')
    ext = (OFFSET[0]-1.25, OFFSET[-1]+1.25, ZENITH[0]-0.5, ZENITH[-1]+0.5)
    for arr, nm, fn in ((rel, 'aerogel photons on the SiPM / vertical central muon', 'prox_angles_rel.png'),
                        (arc, 'ring azimuth on the SiPM (fraction of the direct light reaching the SiPM, averaged over 36 emission azimuths)', 'prox_angles_arc.png')):
        fig, ax = plt.subplots(4, 3, figsize=(16, 18))
        for ji, g in enumerate(GAPS):
            for az in range(len(AZIMUTH)):
                for od in range(len(OFFSET_DIRS)):
                    a = ax[2*az+od, ji]; m = arr[0, gi, ji, az, od]
                    im = a.imshow(m, origin='lower', extent=ext, aspect='auto', vmin=0, vmax=1.05, cmap='viridis')
                    cs = a.contour(OFFSET, ZENITH, m, levels=[0.5, 0.8, 0.95], colors=['w', 'w', 'k'], linewidths=0.8)
                    a.clabel(cs, fontsize=7, fmt='%.2f')
                    a.set_title('gap %g mm; muon azimuth %g deg, offset %s' % (g, AZIMUTH[az], OFFSET_DIRS[od]), fontsize=9)
                    a.set_xlabel('impact offset at the tile mid-plane [mm]'); a.set_ylabel('zenith [deg]')
                    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        fig.suptitle('%s, air, pore shift off\n(Xe 1-3 bar: max |difference| from air %.3f (photons), %.3f (azimuth))'
                     % (nm, S['max_abs_rel_diff_xe_vs_air'], S['max_abs_arc_diff_xe_vs_air']), fontsize=12, color=INK)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        save(out, fn, dpi=95, fig=fig); plt.close(fig)


def fig_angles_curves(out, rel, arc, st, pf, S):
    gi = GAS_NAMES.index('air'); i0 = list(OFFSET).index(0.0)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    F = {f: k for k, f in enumerate(STATS_FIELDS)}
    for ji, g in enumerate(GAPS):
        col = GAP_COLOUR[g]
        for az, ls in zip(range(len(AZIMUTH)), ('-', '--')):
            lab = 'gap %g mm, az %g deg' % (g, AZIMUTH[az])
            ax[0, 0].plot(ZENITH, rel[0, gi, ji, az, 0, :, i0], color=col, ls=ls, lw=1.5, label=lab)
            ax[0, 0].plot(ZENITH, pf[gi, ji, az, 0, :, i0], color=col, ls=ls, lw=0.7, alpha=0.6)
            ax[0, 1].plot(ZENITH, arc[0, gi, ji, az, 0, :, i0], color=col, ls=ls, lw=1.5, label=lab)
            s = st[0, gi, ji, az, 0, :, i0]
            ax[0, 2].plot(ZENITH, np.hypot(s[:, F['centre_x_mm']], s[:, F['centre_y_mm']]), color=col, ls=ls, marker='o', ms=2.5, label=lab)
            ax[1, 0].plot(ZENITH, s[:, F['fit_r_mm']], color=col, ls=ls, marker='o', ms=2.5, label=lab)
            ax[1, 1].plot(ZENITH, s[:, F['fit_r_rms_mm']], color=col, ls=ls, marker='o', ms=2.5, label=lab)
            ax[1, 2].plot(ZENITH, s[:, F['inc_gas_mean_deg']], color=col, ls=ls, lw=1.5, label=lab+': mean, gas')
            if az == 0:
                ax[1, 2].plot(ZENITH, s[:, F['inc_gas_max_deg']], color=col, ls=':', lw=0.8, label=lab+': max, gas')
                ax[1, 2].plot(ZENITH, s[:, F['inc_win_mean_deg']], color=col, ls='-.', lw=0.8, label=lab+': mean, in the window')
    ax[0, 0].set_ylabel('aerogel photons on the SiPM / vertical central muon\n(thin: sim.proximity_fractions, no Fresnel/Rayleigh)')
    ax[0, 1].set_ylabel('ring azimuth on the SiPM'); ax[0, 2].set_ylabel('fitted ring centre - muon impact on the SiPM plane [mm]')
    ax[1, 0].set_ylabel('fitted ring radius [mm]'); ax[1, 1].set_ylabel('rms about the fitted circle [mm]')
    ax[1, 2].set_ylabel('angle of incidence on the SiPM [deg] (detected direct photons)')
    for a in ax.ravel():
        a.set_xlabel('muon zenith angle [deg] (track through the tile centre)'); a.grid(alpha=.3); a.legend(fontsize=6.5)
    fig.suptitle('Aerogel ring vs muon zenith, zero offset, air, pore shift off (FT x QE)', fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save(out, 'prox_angles_curves.png', dpi=110, fig=fig); plt.close(fig)


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
    np.savez_compressed(out/('prox_%s.npz' % part), **flat)


def _key(p):
    if p.lstrip('-').isdigit():
        return int(p)
    try:
        return float(p)
    except ValueError:
        return p


def load_acc(path, nkey):
    z = np.load(path, allow_pickle=False); acc = {}
    for name in z.files:
        parts = name.split('|'); key = tuple(_key(p) for p in parts[:nkey]); q = parts[nkey]
        acc.setdefault(key, {})[q] = z[name]
    return acc


def main():
    p = cli.parser(__doc__)
    p.add_argument('--parts', default='onaxis,terms,angles', help='comma-separated parts (default all)')
    p.add_argument('--nphot', type=float, default=1e6, help='sampled photons per origin, configuration and seed, on-axis part (default 1e6)')
    p.add_argument('--nseeds', type=int, default=4, help='master seeds of the on-axis part, --seed + k (default 4)')
    p.add_argument('--nphot-points', type=float, default=2e5, help='sampled photons per origin at the extra impact points (default 2e5)')
    p.add_argument('--nphot-terms', type=float, default=1e6, help='sampled tile photons per variant in the terms part (default 1e6)')
    p.add_argument('--nphot-angle', type=int, default=10000, help='sampled tile photons per muon configuration, angles part (default 10000)')
    p.add_argument('--chunk', type=int, default=50000, help='sampled photons per origin and chunk (default 50000)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed; gases, gaps and variants share the chunk seeds')
    p.add_argument('--nproc', type=int, default=8, help='worker processes (default 8)')
    p.add_argument('--replot', action='store_true', help='redo the analysis and figures from the npz files in --outdir')
    args = p.parse_args(); out = cli.outdir(args)
    parts = [x for x in args.parts.split(',') if x]
    t0 = time.time(); band = qe_band()
    summary = dict(band_eV=list(band), gaps_mm=list(GAPS), sensor_half=SENSOR_HALF_2IN, aerogel_n=AEROGEL_N_AIR, tile_t=AEROGEL_T,
                   tile_half=AEROGEL_HALF, window=dict(n=WINDOW_N, t=WINDOW_T), rayleigh_C=RAYLEIGH_C_AEROGEL,
                   porosity_vac=AEROGEL_POROSITY, porosity_air=AEROGEL_POROSITY_AIRREF, n_air=N_AIR,
                   gas_n_400nm={gn: float(gas_medium(gn).n(np.atleast_1d(AEROGEL_LAMBDA_REF))[0]) for gn in GAS_NAMES},
                   segments_check=check_segments(),
                   args={k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in vars(args).items()})
    sp = out/'prox_summary.json'
    if sp.exists():
        old = json.loads(sp.read_text()); old.update(summary); summary = old
    accs = {}
    if not args.replot:
        jobs, tasks = build_jobs(args, parts, band)
        _CFG.update(jobs=jobs)
        print('band %.2f-%.2f eV; parts %s; %d tasks on %d processes' % (*band, ','.join(parts), len(tasks), args.nproc), flush=True)
        accs = {pt: {} for pt in parts+(['points'] if 'onaxis' in parts else [])}; nch = {}
        with mp.get_context('fork').Pool(args.nproc) as pool:
            for t, res in zip(tasks, pool.imap(run_task, tasks, chunksize=8 if t_is_angles(tasks) else 1)):
                if t[0] == 'angles':
                    accs['angles'][res[0]] = res[1]
                else:
                    reduce_into(accs[t[0]].setdefault(t[1], {}), res); nch[(t[0], t[1])] = nch.get((t[0], t[1]), 0)+1
        print('Monte Carlo wall time %.1f s' % (time.time()-t0), flush=True)
        for pt in accs:
            if pt == 'angles':
                continue
            for key, a in accs[pt].items():
                n = nch[(pt, key)]
                for q in a:
                    if q not in ('nbranch', 'map'):
                        a[q] = a[q]/n
                a['nchunk'] = np.array(n)
            save_acc(out, pt, accs[pt])
        if 'angles' in parts:
            flat = {}
            for key, r in accs['angles'].items():
                for q, v in r.items():
                    flat['%s|%s' % ('|'.join(str(k) for k in key), q)] = v
            np.savez_compressed(out/'prox_angles_raw.npz', **flat)
    else:
        for pt in parts:
            if pt == 'angles':
                accs[pt] = load_acc(out/'prox_angles_raw.npz', 7)
            else:
                accs[pt] = load_acc(out/('prox_%s.npz' % pt), 4 if pt == 'onaxis' else 3)
        if 'onaxis' in parts:
            accs['points'] = load_acc(out/'prox_points.npz', 4)
    if 'onaxis' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['onaxis'].items()}
        nseeds = len({k[3] for k in acc})
        # maps are per-chunk sums: normalise per muon
        for k, a in accs['onaxis'].items():
            if 'map' in a:
                acc[k]['map'] = a['map']/a['nchunk']
        analyse_points(out, {k: {q: v for q, v in a.items() if q not in ('nchunk', 'map')} for k, a in accs['points'].items()}, args, summary)
        analyse_onaxis(out, acc, nseeds, args, summary, accs['points'])
    if 'terms' in parts:
        acc = {k: {q: v for q, v in a.items() if q != 'nchunk'} for k, a in accs['terms'].items()}
        analyse_terms(out, acc, args, summary)
    if 'angles' in parts:
        if args.replot:
            res = {k: v for k, v in accs['angles'].items()}
        else:
            res = accs['angles']
        analyse_angles(out, res, args, summary)
    sp.write_text(json.dumps(to_jsonable(summary), indent=1))
    print('summary: %s; total wall time %.1f s' % (sp, time.time()-t0))


def t_is_angles(tasks):
    return any(t[0] == 'angles' for t in tasks)


if __name__ == '__main__':
    main()
