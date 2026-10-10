"""Sketches of the optical effects in configuration 5 (f = 203 mm mirror with the tile), from rays traced with xerich.

Layout (xerich.layout.Phase1Layout with the xerich.constants values): spherical mirror, flat SiPM at the
focal plane, gas gap, aerogel tile, Xe 3 bar unless stated; vertical beta = 1 muon on the mirror axis entering at the
sensor plane. Rays are followed branch by branch with the tracer's own geometry and Fresnel calls (follow();
terminal points and weights are checked against tracer.trace in several scenes, and a mismatch raises), and the
Monte Carlo profiles come from sim.onaxis_muon. Every number in a label comes from the layout objects, the xerich
constants, the traced rays or the sample parameters (emission heights, offsets, fractions, bin widths and photon counts
are named below; tabulated reference wavelengths are inline), and the computed ones are printed to the numbers file.

Sketches 1-8 are ray optics without Rayleigh scattering; sketch 9 adds the Rayleigh scattering in the tile (halo),
sketch 10 is the muon's own hit from its light in the SiPM window.

Figures: sketch_effects_{1_overview, 2_tile_refraction, 3_aerogel_reflection, 4_gas_ghosts, 5_chromatic,
6_spherical_aberration, 7_tir, 8_polarisation, 9_rayleigh_halo, 10_muon_window}.png and
sketch_effects_numbers.txt (stdout copy).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import dataclasses  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from xerich import cli, sim, tracer  # noqa: E402
from xerich.analysis import SCATTER_LAM_EDGES, scattered_fraction  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band, theta_xe  # noqa: E402
from xerich.constants import (AEROGEL_N_AIR, HC, MIP_DIRECT_PE, MIRROR_D, MIRROR_R, SENSOR_HALF_2IN, T0,  # noqa: E402
                              T_ROOM, WINDOW_N, WINDOW_T)
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window  # noqa: E402
from xerich.media import ConstantIndex, Rayleigh, c4f10, xenon  # noqa: E402
from xerich.plotting import AEROGEL, GAS, INK, LIGHT, MIRROR, MUTED, RADIATOR, plt, save  # noqa: E402

GHOST = '#a21caf'                  # gas Fresnel ghosts (purple)
SENSOR_C = '#111827'
TILE_FILL = '#fde7c4'
GAS_FILL = '#eef4ff'
LAM0 = 400.0                       # wavelength of the drawn rays [nm]
DOWN = np.array([0.0, 0.0, -1.0])
EX = np.array([1.0, 0.0, 0.0])
GASES = {'xe2': xenon(2), 'xe3': xenon(3), 'c4f10': c4f10()}
R_BIN = 0.01                       # radial bin of the Monte Carlo profiles [mm]
R_MAX = float(np.ceil(SENSOR_HALF_2IN*np.sqrt(2)))   # radial range of the profiles: sensor half-diagonal, no overflow bin [mm]
R_REBIN = 5                        # profile bins merged in figs 3 and 4 (R_BIN x R_REBIN wide)
R_REBIN_RING = 2                   # profile bins merged in the ring panel of fig 5
ABERR_OFFSETS = (0.0, 15.0, 25.0, 50.0)   # track offsets of the aberration study and its check [mm]
FOLLOW_TOL_MM = 1e-9               # allowed end point difference between follow() and tracer.trace [mm]
FOLLOW_TOL_W = 1e-12               # allowed weight difference
TILT_DEG = 70.0                    # zenith angle of the tilted muon of figs 7 and check_follow [deg]
SENSOR_HALF_WIDE = 80.0            # larger sensor half side of the aberration scenes [mm]
TRACK_LEN = 200.0                  # track length of the aberration photons from the sensor plane [mm]
N_ABERR = 200000                   # photons per track in the aberration study
N_SHOWN = 4000                     # aberration photons drawn per track
N_TIR = 4000                       # photons per zenith angle in the TIR fraction
N_TILE_MC = 400000                 # photons of the with/without-tile comparison
ZOOM_Z0 = 180.0                    # lower z edge of the zoom in fig 4b [mm]
ZOOM_R1 = 1.2                      # upper r edge of the zoom in fig 4c [mm]
Z_POL_GAS = 5.0                    # emission height of the gas photon in fig 8a [mm]
Z_FAN = (12.0, 50.0, 90.0, 130.0, 170.0)   # emission heights of the gas fan in fig 1, plus the mid-gap point [mm]
Z_TILE_SHIFT = (5.0, 50.0, 100.0, 150.0, 190.0)   # emission heights of the gas photons in fig 2 [mm]
Z_GHOST = 100.0                    # emission height of the gas photon in fig 4 [mm]
Z_CHROMA = (10.0, 100.0, 185.0)    # emission heights of the gas photons in fig 5 [mm]
Z_ABERR = (20.0, 100.0, 180.0)     # emission heights of the drawn rays in fig 6a [mm]
ABERR_FAN_OFFSETS = (25.0, 50.0)   # track offsets of the drawn rays in fig 6a [mm]
ABERR_LEGACY_OFFSETS = (0.0, 25.0, 50.0)   # track offsets of the sim.aberration comparison in fig 6d [mm]
TIR_FRACS = (0.15, 0.4, 0.65, 0.9)   # emission points along the tilted track in the tile (fractions of the segment), fig 7a
N_WIN_PICK = 20000                 # window photons sampled to pick the drawn paths of fig 10
CHECK_N = dict(gas_low=6, gas_gap=2, aero=4, tilted=4, phi=5, phi_tilted=9)   # emission points and azimuths in check_follow
# seed offsets from --seed (aberration_real and tir are in different figures and share no draws)
SEED = dict(tile_shift=1, aberration=6, aberration_real=7, tir=7, rayleigh=11, window=12, window_mc=13, check=99)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5,
                     'axes.labelsize': 9.5, 'axes.edgecolor': '#4b5563', 'axes.labelcolor': INK,
                     'xtick.color': '#374151', 'ytick.color': '#374151', 'legend.frameon': False})

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LINES.append(s)


def layout(gas, scatter=None):
    return Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), gas, Aerogel(AEROGEL_N_AIR), tile_scatter=scatter)


def gas_label(gas):
    """Medium label with the chemical subscripts."""
    return gas.label.replace('C4F10', 'C₄F₁₀')


def unbounded(scene, sensor_half=None, mirror=True):
    """Scene with the mirror sphere usable at any radius (if mirror) and optionally a larger sensor."""
    m = dataclasses.replace(scene.mirror, clear_radius=1e9) if mirror else scene.mirror
    s = scene.sensor if sensor_half is None else dataclasses.replace(scene.sensor, half=sensor_half)
    return dataclasses.replace(scene, mirror=m, sensor=s)


# Branch-by-branch ray follower (same calls and order as tracer.trace, plus the vertex record)

_NONE, _SENSOR, _MIRROR, _EXIT, _ENTRY = -1, 0, 1, 2, 3


def _next_event(scene, P, D, reg, eps):
    best, ev, face, slab = np.inf, _NONE, -1, -1
    P1, D1 = P[None], D[None]
    for j, sl in enumerate(scene.slabs):
        if reg == j:
            s, f = tracer.box_exit(P1, D1, sl.lo, sl.hi)
            if s[0] < best:
                best, ev, face, slab = s[0], _EXIT, int(f[0]), j
        if reg < 0:
            s, f = tracer.box_entry(P1, D1, sl.lo, sl.hi, eps)
            if s[0] < best:
                best, ev, face, slab = s[0], _ENTRY, int(f[0]), j
    mir = scene.mirror
    if mir is not None and reg < 0:
        q = P-np.asarray(mir.C); b = q@D; c = q@q-mir.R**2; disc = b*b-c
        if disc > 0:
            r = np.sqrt(disc); s1, s2 = -b-r, -b+r
            s = s1 if s1 > eps else (s2 if s2 > eps else np.inf)
            if s < best:
                best, ev = s, _MIRROR
    if reg < 0 and D[2] != 0:
        s = (scene.sensor.z-P[2])/D[2]
        if s > eps and s < best:
            best, ev = s, _SENSOR
    return best, ev, face, slab


def follow(scene, p, d, lam, reg, E=None, max_reflections=2, max_interactions=64, eps=1e-7):
    """Follow photons through the scene branch by branch, recording the vertices of every branch.

    Same arguments and physics as tracer.trace (weight splitting at the slab z faces, mirror reflection,
    terminal reasons and path classes), without Rayleigh scattering, surface loss or a window.
    Returns a list of branches, each a dict with the photon index i,
    the vertex array pts (K,3), the field at each vertex Es (K,3) (None if unpolarised), the weight after
    each vertex ws (K,), the final reason and class, and the interface events (kind, face, z, incidence
    angle, n1, n2, R, T, s-fraction of the incident field).
    """
    p = np.asarray(p, float); d = np.asarray(d, float); lam = np.asarray(lam, float)
    stack = [dict(i=k, pts=[p[k]], Es=[None if E is None else np.asarray(E[k], complex)], ws=[1.0], p=p[k], d=d[k],
                  lam=lam[k], reg=int(reg[k]), w=1.0, nint=0, nb=0, nt=0, ntir=0,
                  E=None if E is None else np.asarray(E[k], complex), ev=[]) for k in range(len(p))]
    out = []

    def end(b, reason):
        out.append(dict(i=b['i'], pts=np.array(b['pts']), Es=None if E is None else np.array(b['Es']),
                        ws=np.array(b['ws']), w=b['w'], reason=reason,
                        cls=int(tracer.path_class(np.array([b['nb']]), np.array([b['nt']]), np.array([b['ntir']]))[0]),
                        events=b['ev'], d=b['d'], E=b['E']))

    def push(b, q, dnew, Enew, wnew, **kw):
        nb_ = dict(b); nb_.update(kw)
        nb_['pts'] = b['pts']+[q]; nb_['Es'] = b['Es']+[Enew]; nb_['ws'] = b['ws']+[wnew]
        nb_['p'] = q; nb_['d'] = dnew; nb_['E'] = Enew; nb_['w'] = wnew; nb_['ev'] = list(b['ev'])
        return nb_

    while stack:
        b = stack.pop()
        best, ev, face, j = _next_event(scene, b['p'], b['d'], b['reg'], eps)
        q = b['p']+(best if np.isfinite(best) else 0)*b['d']
        if ev == _NONE:
            end(push(b, q, b['d'], b['E'], b['w']), 5); continue
        if ev == _SENSOR:
            inside = abs(q[0]-scene.sensor.cx) < scene.sensor.half and abs(q[1]-scene.sensor.cy) < scene.sensor.half
            end(push(b, q, b['d'], b['E'], b['w']), 0 if inside else 3); continue
        if ev in (_EXIT, _ENTRY) and face < 4:
            end(push(b, q, b['d'], b['E'], b['w']), 2); continue
        mir = scene.mirror
        if ev == _MIRROR:
            a = np.asarray(mir.axis, float); qq = q-np.asarray(mir.C); ax = qq@a
            if not (ax < 0 and np.linalg.norm(qq-ax*a) < mir.clear_radius):
                end(push(b, q, b['d'], b['E'], b['w']), 1); continue
        if b['nint'] >= max_interactions:
            end(push(b, q, b['d'], b['E'], b['w']), 4); continue
        if ev == _MIRROR:
            nrm = (q-np.asarray(mir.C))/mir.R
            dn = tracer.reflect(b['d'][None], nrm[None])[0]
            En = None if b['E'] is None else tracer.mirror_field(b['E'][None], nrm[None])[0]
            nb_ = push(b, q, dn, En, b['w'], nint=b['nint']+1)
            nb_['ev'].append(dict(kind='mirror', z=q[2], rho=np.hypot(q[0], q[1])))
            stack.append(nb_); continue
        sl = scene.slabs[j]
        q = q.copy(); q[2] = sl.lo[2] if face == 4 else sl.hi[2]
        beyond = q+np.sign(b['d'][2])*np.array([0, 0, 1e-6])
        other = j if ev == _ENTRY else int(tracer._region_at(scene, beyond[None])[0])
        n1 = float(tracer._region_index(scene, np.array([b['reg']]), np.array([b['lam']]))[0])
        n2 = float(tracer._region_index(scene, np.array([other]), np.array([b['lam']]))[0])
        N = np.array([0.0, 0.0, -np.sign(b['d'][2])])
        Ein = None if b['E'] is None else b['E'][None]
        (d_r, E_r, R), (d_t, E_t, T), tir = tracer.interface_split(b['d'][None], Ein, N[None], np.array([n1]),
                                                                   np.array([n2]))
        R, T, tir = float(R[0]), float(T[0]), bool(tir[0])
        cosi = -b['d']@N
        sfrac = np.nan
        if b['E'] is not None:
            sv = np.cross(b['d'], N); ns = np.linalg.norm(sv)
            sfrac = float(np.abs(b['E']@(sv/ns))**2) if ns > 1e-12 else np.nan
        evd = dict(face='bottom' if face == 4 else 'top', z=q[2], inc=float(np.arccos(np.clip(cosi, -1, 1))),
                   n1=n1, n2=n2, R=R, T=T, tir=tir, sfrac=sfrac, x=q[0], y=q[1])
        if not tir:
            t = push(b, q, d_t[0], None if E_t is None else E_t[0], b['w']*T, reg=other, nint=b['nint']+1)
            t['ev'].append(dict(evd, kind='trans'))
            stack.append(t)
        nbv = b['nb']+(not tir and face == 4); ntv = b['nt']+(not tir and face == 5)
        r = push(b, q, d_r[0], None if E_r is None else E_r[0], b['w']*R, nint=b['nint']+1, nb=nbv, nt=ntv,
                 ntir=b['ntir']+tir)
        r['ev'].append(dict(evd, kind='tir' if tir else 'refl'))
        if not tir and nbv+ntv > max_reflections:
            end(r, 4)
        else:
            stack.append(r)
    return out


def validate(scene, p, d, lam, reg, E, max_reflections=2):
    """Largest differences in end point and weight between follow() and tracer.trace for the same photons."""
    br = follow(scene, p, d, lam, reg, E, max_reflections)
    tr = tracer.trace(scene, p, d, lam, reg, E, max_reflections)
    a = np.array([[b['i'], b['reason'], b['cls'], b['w'], *b['pts'][-1]] for b in br])
    c = np.column_stack([tr['i'], tr['reason'], tr['cls'], tr['w'], tr['p']])
    if len(a) != len(c):
        return len(a), len(c), np.inf, np.inf
    a = a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))]; c = c[np.lexsort((c[:, 3], c[:, 2], c[:, 1], c[:, 0]))]
    if not np.array_equal(a[:, :3], c[:, :3]):
        return len(a), len(c), np.inf, np.inf
    return len(a), len(c), np.abs(a[:, 4:]-c[:, 4:]).max(), np.abs(a[:, 3]-c[:, 3]).max()


# Emission helpers

def th_c(n):
    return np.arccos(1/np.asarray(n, float))


def emit(scene, medium_n, pts, u, lam, phis, e1=EX):
    """Cherenkov photons (beta = 1) emitted at the points pts (N,3) of a track with direction u.

    The azimuths phis are measured from e1 (made perpendicular to u) towards u x e1. Returns p, d, lam, region and E
    for every (point, azimuth) pair.
    """
    u = np.asarray(u, float); e1 = np.asarray(e1, float); e1 = e1-(e1@u)*u; e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    pts = np.atleast_2d(np.asarray(pts, float)); phis = np.atleast_1d(np.asarray(phis, float))
    lam = np.broadcast_to(np.asarray(lam, float), (len(pts),)).copy()
    th = th_c(medium_n(lam))
    d = np.concatenate([tracer.cone(u[None], th[k], phis, e1[None], e2[None])[0] for k in range(len(pts))])
    P = np.repeat(pts, len(phis), axis=0); L = np.repeat(lam, len(phis))
    reg = tracer._region_at(scene, P)
    E = tracer.cherenkov_field(np.broadcast_to(u, d.shape), d)
    return P, d, L, reg, E


def pick(branches, cls=None, reason=None, i=None):
    out = [b for b in branches if (cls is None or b['cls'] == cls) and (reason is None or b['reason'] == reason)
           and (i is None or b['i'] == i)]
    return out


def ray(ax, pts, color, ls='-', lw=1.2, alpha=1.0, arrows=(), zorder=3, plane=(0, 2), label=None):
    """Polyline of a branch in the chosen projection with arrowheads on the listed segments."""
    x, z = pts[:, plane[0]], pts[:, plane[1]]
    ax.plot(x, z, color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder, label=label, solid_capstyle='round')
    for k in arrows:
        if k < len(pts)-1:
            m = 0.55*pts[k]+0.45*pts[k+1]; m2 = 0.5*pts[k]+0.5*pts[k+1]
            ax.annotate('', xy=(m2[plane[0]], m2[plane[1]]), xytext=(m[plane[0]], m[plane[1]]),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.5', color=color, lw=0,
                                        alpha=alpha, shrinkA=0, shrinkB=0), zorder=zorder)


def draw_layout(ax, lay, xlim, mirror=True, ext=60.0, gas_fill=True, sensor_t=1.2, lw_m=2.6):
    """Tile, gap, SiPM and mirror of the layout in the x-z plane (to scale where the axes are equal)."""
    zs = lay.z_sensor
    h = lay.tile.half if lay.tile is not None else None
    if h is not None:
        zt0, zt1 = lay.z_tile_bottom, lay.z_tile_top
    if gas_fill:
        ax.axvspan(*xlim, color=GAS_FILL, zorder=0, lw=0)
    if h is not None:
        ax.add_patch(Rectangle((-h, zt0), 2*h, zt1-zt0, fc=TILE_FILL, ec=AEROGEL, lw=1.0, zorder=1))
    hs = lay.sensor.half
    ax.add_patch(Rectangle((-hs, zs), 2*hs, sensor_t, fc=SENSOR_C, ec=SENSOR_C, lw=0.5, zorder=4))
    if mirror:
        R = lay.mirror.R; a = lay.mirror.D/2; ac = lay.mirror.clear_radius
        xs = np.linspace(-ext, ext, 600); zsph = R-np.sqrt(R*R-xs**2)
        ax.plot(xs, zsph, color=MIRROR, lw=0.9, ls=(0, (4, 3)), alpha=.7, zorder=2)
        xm = np.linspace(-a, a, 400); zm = R-np.sqrt(R*R-xm**2)
        ax.plot(xm, zm, color='#a3aebd', lw=lw_m, zorder=2, solid_capstyle='butt')
        xc = np.linspace(-ac, ac, 300)
        ax.plot(xc, R-np.sqrt(R*R-xc**2), color=MIRROR, lw=lw_m, zorder=2, solid_capstyle='butt')


def px_per_mm(fig, ax):
    fig.canvas.draw()
    bb = ax.get_window_extent(); x0, x1 = ax.get_xlim(); z0, z1 = ax.get_ylim()
    return bb.width/abs(x1-x0), bb.height/abs(z1-z0)


def zoom_box(ax_main, ax_in, color=MUTED, text=None):
    """Rectangle on the main axes marking the region shown in the inset."""
    x0, x1 = ax_in.get_xlim(); z0, z1 = ax_in.get_ylim()
    ax_main.add_patch(Rectangle((x0, z0), x1-x0, z1-z0, fill=False, ec=color, lw=0.9, zorder=6))
    if text:
        ax_main.text(x1, z1, ' '+text, color=color, fontsize=8.5, va='bottom', ha='left', zorder=6)


def note(ax, x, y, s, color=INK, fs=8.5, ha='left', va='center', **kw):
    return ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va, zorder=8,
                   bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2), **kw)


# Monte Carlo (sim.onaxis_muon) for the profiles

_CFG = {}


def mc_chunk(task):
    gi, ci = task
    c = _CFG
    rng = np.random.default_rng(c['seeds'][ci])
    ph, e = sim.onaxis_muon(c['layouts'][gi], rng, c['spectra'][gi], c['n'], c['n'])
    w0 = ph['weight']; qe = qe_arc(HC/ph['lam']); org = ph['origin']
    ws = np.stack([w0*qe, w0])                                     # (FT x QE, FT), photons per muon
    det = e['reason'] == 0
    r = np.hypot(e['p'][det, 0], e['p'][det, 1]); o = org[e['i'][det]].astype(int); cl = e['cls'][det]
    wd = ws[:, e['i'][det]]*e['w'][det]
    nr = int(round(R_MAX/R_BIN)); ir = (r/R_BIN).astype(int)
    key = (o*len(tracer.CLASSES)+cl)*nr+ir
    radial = np.stack([np.bincount(key, wd[k], minlength=2*len(tracer.CLASSES)*nr).reshape(2, len(tracer.CLASSES), nr)
                       for k in range(2)])
    g = (o == 0)&(cl == 0)
    ring = np.array([[wd[k, g].sum(), (wd[k, g]*r[g]).sum(), (wd[k, g]*r[g]**2).sum()] for k in range(2)])
    emit_w = np.array([[ws[k, org == oo].sum() for oo in (0, 1)] for k in range(2)])
    nl = len(SCATTER_LAM_EDGES)-1; il = np.clip(np.searchsorted(SCATTER_LAM_EDGES, ph['lam'][e['i']])-1, 0, nl-1)
    lkey = (((org[e['i']].astype(int)*len(tracer.REASONS)+e['reason'])*2+(e['nsc'] > 0))*nl+il)
    lam = np.stack([np.bincount(lkey, ws[k][e['i']]*e['w'], minlength=2*len(tracer.REASONS)*2*nl).reshape(2, len(tracer.REASONS), 2, nl)
                    for k in range(2)])
    return dict(radial=radial, ring=ring, emit=emit_w, lam=lam)


def run_mc(gases, n, nchunk, nproc, seed):
    """sim.onaxis_muon for each (name, gas, tile scatter): radial profiles by origin and class, ring mean and rms
    (FT x QE and FT), end weights by wavelength."""
    lo, hi = qe_band()
    lays = [layout(g, sc) for _, g, sc in gases]
    spectra = [{'gas': PhotonSpectrum(l.gas.index, lo, hi), 'tile': PhotonSpectrum(l.tile_index.index, lo, hi)} for l in lays]
    _CFG.update(layouts=lays, spectra=spectra, n=n, seeds=np.random.SeedSequence(seed).spawn(nchunk))
    tasks = [(gi, ci) for gi in range(len(gases)) for ci in range(nchunk)]
    acc = {}
    with mp.get_context('fork').Pool(nproc) as pool:
        for t, r in zip(tasks, pool.imap(mc_chunk, tasks)):
            k = t[0]
            if k in acc:
                for kk in r:
                    acc[k][kk] = acc[k][kk]+r[kk]
            else:
                acc[k] = r
    out = {}
    for gi, (name, _, _) in enumerate(gases):
        a = acc[gi]
        s = {'radial': a['radial']/nchunk/R_BIN, 'emit': a['emit']/nchunk, 'spectra': spectra[gi], 'layout': lays[gi],
             'lam_end': a['lam']/nchunk}
        for k, wn in enumerate(('ftqe', 'ft')):
            sw, s1, s2 = a['ring'][k]; s['r_'+wn] = (s1/sw, np.sqrt(s2/sw-(s1/sw)**2))
            s['cls_'+wn] = a['radial'][k].sum(-1)/nchunk
        out[name] = s
    return out


def x_at(pts, z):
    """x and y where the last upward leg of a branch polyline crosses height z (nan outside)."""
    up = np.flatnonzero(np.diff(pts[:, 2]) > 0)
    k0 = up[0] if len(up) else 0
    seg = pts[k0:]
    out = np.full((len(np.atleast_1d(z)), 2), np.nan)
    for j, zz in enumerate(np.atleast_1d(z)):
        for a, b in zip(seg[:-1], seg[1:]):
            if (a[2]-zz)*(b[2]-zz) <= 0 and b[2] != a[2]:
                t = (zz-a[2])/(b[2]-a[2]); out[j] = (a+t*(b-a))[:2]; break
    return out


def mark_mag(fig, ax_main, ax_in, title):
    """Append the magnification of the inset relative to the main axes to its title."""
    m = px_per_mm(fig, ax_in)[0]/px_per_mm(fig, ax_main)[0]
    ax_in.set_title('%s, magnified ×%.0f, to scale' % (title, m), loc='left')
    return m


# Figure 1: overview

def fig_overview(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); f = lay.mirror.f
    zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    fig = plt.figure(figsize=(14, 11.2))
    ax = fig.add_axes([0.035, 0.045, 0.46, 0.90])
    XL, ZL = (-64, 64), (-12, 218)
    draw_layout(ax, lay, XL, ext=58)
    # gas fan
    zg = np.array([*Z_FAN, 0.5*(zt1+zs)])
    P, D, L, Rg, E = emit(sc, lay.gas.n, np.c_[np.zeros_like(zg), np.zeros_like(zg), zg], DOWN, LAM0, [0.0, np.pi])
    br = follow(sc, P, D, L, Rg, E)
    direct = pick(br, cls=0, reason=0)
    ends = np.array([b['pts'][-1] for b in direct])
    for b in direct:
        ray(ax, b['pts'], GAS, lw=0.9, arrows=(0, 1))
    # aerogel fan (transmitted branch: to the mirror sphere outside the clear aperture)
    za = np.array([zt0+0.05, 0.5*(zt0+zt1), zt1-0.05])
    Pa, Da, La, Ra, Ea = emit(sc, lay.tile_index.n, np.c_[np.zeros(3), np.zeros(3), za], DOWN, LAM0, [0.0, np.pi])
    bra = follow(sc, Pa, Da, La, Ra, Ea)
    lost = pick(bra, cls=0, reason=1)
    rho_a = np.array([np.hypot(*b['pts'][-1][:2]) for b in lost])
    th_gas_a = np.array([np.arccos(-b['d'][2]) for b in lost])
    for b in lost:
        ray(ax, b['pts'], AEROGEL, lw=1.0, arrows=(1,))
        ax.plot(*b['pts'][-1][[0, 2]], marker='x', color=AEROGEL, ms=6, mew=1.6, zorder=5)
    # muon
    ax.annotate('', xy=(0, 1.0), xytext=(0, 216), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.8), zorder=7)
    note(ax, 1.5, 213, 'μ (β = 1), vertical, on the mirror axis', fs=9)
    # dimension and labels
    ax.annotate('', xy=(-59, 0), xytext=(-59, zs), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.9))
    ax.text(-60.5, 100, 'f = R/2 = %.1f mm' % f, rotation=90, ha='right', va='center', color=MUTED, fontsize=9)
    r_mc = ctx['mc']['xe3']['r_ftqe'][0]
    ax.annotate('', xy=(0, zs+6.5), xytext=(r_mc, zs+6.5), arrowprops=dict(arrowstyle='<->', color=GAS, lw=0.9))
    note(ax, -1, zs+6.5, 'ring r = %.2f mm ' % r_mc, color=GAS, ha='right', fs=8.5)
    note(ax, 27, zs+3.8, 'SiPM %.1f mm (z = f)' % (2*lay.sensor.half), fs=8.5)
    note(ax, 27.5, 0.5*(zt0+zt1), 'aerogel tile\n%.0f×%.0f×%.0f mm, n = %.2f' % (2*lay.tile.half, 2*lay.tile.half, lay.tile.thickness, lay.tile.n),
         color='#92400e', fs=8.5)
    note(ax, -36, 120, '%s (%.0f °C)\nfills the vessel' % (lay.gas.label, T_ROOM-T0), color='#1e40af', fs=9, ha='center')
    note(ax, 42, 18, 'mirror sphere beyond\nthe %.1f mm mirror\n(dashed)' % lay.mirror.D, color=MIRROR, fs=8, ha='center')
    note(ax, -14, 145, 'gas photons\nθc = %.1f mrad\n(%.0f nm)' % (1e3*ctx['thg400'], LAM0), color=GAS, fs=8.5, ha='center')
    note(ax, 30, 120, 'aerogel photons\n%.1f mrad in the tile,\n%.1f mrad in the gas' % (1e3*ctx['tha'], 1e3*th_gas_a.mean()),
         color='#b45309', fs=8.5, ha='center')
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z above the mirror vertex [mm]')
    ax.set_title('(a) side view through the axis, to scale', loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    # inset b: focal region
    bx = fig.add_axes([0.515, 0.47, 0.20, 0.48])
    draw_layout(bx, lay, (6, 17), mirror=False, sensor_t=0.6)
    for b in direct:
        if b['pts'][-1][0] > 0:
            ray(bx, b['pts'], GAS, lw=1.0, arrows=(2,))
    bx.set_xlim(6.5, 16.5); bx.set_ylim(189.0, 205.0); bx.set_aspect('equal')
    spread = ends[ends[:, 0] > 0, 0]
    bx.axvline(r_mc, color=GAS, lw=0.6, ls=':', zorder=2)
    note(bx, 15.9, zs+1.0, 'SiPM', color='white', fs=8, ha='right').set_bbox(dict(fc=SENSOR_C, ec='none'))
    note(bx, 7.0, 0.5*(zt1+zs), 'gap %.0f mm' % lay.gap, fs=8)
    note(bx, 7.0, 196.5, 'tile', color='#92400e', fs=8)
    note(bx, 7.0, 189.4, 'rays from %d heights land\nat x = %.3f–%.3f mm\n(%.0f nm)' % (len(zg), spread.min(), spread.max(), LAM0),
         color=GAS, fs=8, va='bottom')
    bx.set_xlabel('x [mm]'); bx.set_ylabel('z [mm]')
    zoom_box(ax, bx, text='b')
    # inset c: mirror rim
    cx = fig.add_axes([0.53, 0.17, 0.45, 0.20])
    draw_layout(cx, lay, (25, 57), ext=58, lw_m=4.5)
    for b in lost:
        if b['pts'][-1][0] > 0:
            ray(cx, b['pts'], AEROGEL, lw=1.1)
            cx.plot(*b['pts'][-1][[0, 2]], marker='x', color=AEROGEL, ms=7, mew=1.8, zorder=5)
    a_clear, a_edge = lay.mirror.clear_radius, lay.mirror.D/2
    for xv, lab, dy in ((a_clear, 'clear aperture\nρ = %.1f mm' % a_clear, 2.6), (a_edge, 'mirror edge\nρ = %.1f mm' % a_edge, 4.4)):
        zv = lay.mirror.R-np.sqrt(lay.mirror.R**2-xv**2)
        cx.plot([xv, xv], [zv-0.6, zv+dy-0.6], color=MIRROR, lw=0.8)
        note(cx, xv, zv+dy, lab, color=MIRROR, fs=8, ha='center', va='bottom')
    note(cx, rho_a.mean(), 6.6, 'aerogel photons reach the sphere at\nρ = %.1f–%.1f mm: lost' % (rho_a.min(), rho_a.max()),
         color='#b45309', fs=8, ha='center', va='bottom')
    cx.set_xlim(25, 57); cx.set_ylim(-1.0, 9.5); cx.set_aspect('equal')
    cx.set_xlabel('x [mm]'); cx.set_ylabel('z [mm]')
    zoom_box(ax, cx, text='c')
    # top view d: hit pattern on the SiPM (to scale), from traced endpoints
    dx = fig.add_axes([0.775, 0.50, 0.215, 0.42])
    hs = lay.sensor.half
    dx.add_patch(Rectangle((-hs, -hs), 2*hs, 2*hs, fc='#f8fafc', ec=INK, lw=1.0))
    t = np.linspace(0, 2*np.pi, 721)
    rg = ctx['ghost_r']
    dx.add_patch(plt.Circle((0, 0), rg[1], fc=GHOST, alpha=0.13, ec='none'))
    dx.add_patch(plt.Circle((0, 0), rg[0], fc='#f8fafc', ec=GHOST, lw=0.6, ls=':'))
    ra = ctx['aero_r']
    mcx = ctx['mc']['xe3']; ghost_frac = mcx['cls_ftqe'][0, 1:3].sum()/mcx['emit'][0, 0]    # both faces, FT x QE
    th2 = np.r_[t, t[::-1]]; rr = np.r_[np.full_like(t, ra[0]), np.full_like(t, ra[1])]
    dx.fill(rr*np.cos(th2), rr*np.sin(th2), color=AEROGEL, alpha=0.45, lw=0)
    dx.plot(r_mc*np.cos(t), r_mc*np.sin(t), color=GAS, lw=1.6)
    dx.plot(0, 0, marker='o', ms=3, color=INK)
    note(dx, 0, -r_mc-1.2, 'gas ring r = %.2f mm' % r_mc, color=GAS, fs=8, ha='center', va='top')
    note(dx, 0, ra[1]+0.6, 'aerogel, reflected at the\ntile bottom: %.2f–%.2f mm' % tuple(ra), color='#b45309', fs=7.5,
         ha='center', va='bottom')
    note(dx, -hs+1, hs-1.5, 'gas ghosts: disc %.2f–%.2f mm\n(purple, %s of the gas light)' % (*rg, sci(ghost_frac, 2)), color=GHOST, fs=7.5, va='top')
    dx.set_xlim(-hs-1, hs+1); dx.set_ylim(-hs-1, hs+1); dx.set_aspect('equal')
    dx.set_xlabel('x [mm]'); dx.set_ylabel('y [mm]')
    dx.set_title('(d) SiPM, top view, to scale\n(no Rayleigh scattering, see sketch 9)', loc='left')
    fig.canvas.draw()
    mb = mark_mag(fig, ax, bx, '(b) focal region'); mc_ = mark_mag(fig, ax, cx, '(c) mirror rim')
    leg = [Line2D([], [], color=INK, lw=1.8, label='muon'), Line2D([], [], color=GAS, lw=1.2, label='gas Cherenkov photons (direct)'),
           Line2D([], [], color=AEROGEL, lw=1.2, label='aerogel Cherenkov photons (transmitted)'),
           Line2D([], [], color=MIRROR, lw=2.6, label='mirror: dark = %.0f %% clear aperture, light = rim' % (100*lay.mirror.clear_frac))]
    fig.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.53, 0.025), fontsize=8.5, ncol=2)
    p = save(out, 'sketch_effects_1_overview.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig1] gas rays %.0f nm land at x = %.4f-%.4f mm (%d heights); MC <r> FTxQE %.3f mm' % (LAM0, spread.min(), spread.max(), len(zg), r_mc))
    say('[fig1] aerogel photons hit the sphere at rho = %.2f-%.2f mm; angle in the gas %.1f mrad; insets x%.0f, x%.0f'
        % (rho_a.min(), rho_a.max(), 1e3*th_gas_a.mean(), mb, mc_))
    say('[fig1] gas ghosts (bottom + top tile face) %.3e of the emitted gas light, bottom %.3e top %.3e (FTxQE, Xe 3 bar)'
        % (ghost_frac, *(mcx['cls_ftqe'][0, 1:3]/mcx['emit'][0, 0])))
    return p


# Shared numbers

def numbers(ctx):
    lay = ctx['lay']; sc = lay.scene(); f = lay.mirror.f
    for g in GASES.values():
        n = g.n(np.array([300., 400., 600.]))
        say('[n] %-11s n-1 at 300/400/600 nm = %s; theta_c = %s mrad; f tan theta_c = %s mm' % (
            g.label, ' '.join('%.4e' % v for v in n-1), ' '.join('%.2f' % v for v in 1e3*th_c(n)),
            ' '.join('%.3f' % v for v in f*np.tan(th_c(n)))))
    wl = np.array([264., 300., 350., 400., 500., 954.])
    say('[n] %s paraxial f tan theta_c / ARC QE at %s nm: %s' % (lay.gas.label, '/'.join('%.0f' % v for v in wl),
        ', '.join('%.3f mm / %.3f' % (r_, q_) for r_, q_ in zip(f*np.tan(th_c(lay.gas.n(wl))), qe_arc(HC/wl)))))
    ctx['thg400'] = float(th_c(lay.gas.n(LAM0))); ctx['tha'] = float(th_c(lay.tile.n))
    say('[n] aerogel theta_c (beta = 1, n = %.2f) = %.2f mrad = %.3f deg' % (lay.tile.n, 1e3*ctx['tha'], np.degrees(ctx['tha'])))
    # aerogel bottom-face reflection annulus and gas ghost disc from traced rays
    za = np.array([lay.z_tile_bottom+1e-4, lay.z_tile_top-1e-4])
    P, D, L, Rg, E = emit(sc, lay.tile_index.n, np.c_[np.zeros(2), np.zeros(2), za], DOWN, LAM0, [0.0])
    br = pick(follow(sc, P, D, L, Rg, E), cls=1, reason=0)
    ctx['aero_r'] = sorted(np.hypot(*b['pts'][-1][:2]) for b in br)
    zg = np.array([1e-3, lay.z_tile_bottom-1e-3])
    P, D, L, Rg, E = emit(sc, lay.gas.n, np.c_[np.zeros(2), np.zeros(2), zg], DOWN, LAM0, [0.0])
    bra = follow(sc, P, D, L, Rg, E); br = pick(bra, cls=1, reason=0)
    ctx['ghost_r'] = sorted(np.hypot(*b['pts'][-1][:2]) for b in br)
    ctx['alpha_max'] = max(b['events'][1]['inc'] for b in pick(bra, cls=0, reason=0))
    say('[n] largest angle of a returning gas ray at the tile bottom face (emitted at the mirror): %.2f mrad' % (1e3*ctx['alpha_max']))
    P, D, L, Rg, E = emit(sc, lay.tile_index.n, np.c_[np.zeros(3), np.zeros(3), [za[0], za.mean(), za[1]]], DOWN, LAM0, [0.0])
    lost = pick(follow(sc, P, D, L, Rg, E), cls=0, reason=1)
    rho = np.array([np.hypot(*b['pts'][-1][:2]) for b in lost])
    ctx['rho_a'] = (rho.min(), rho.max()); ctx['th_gas_a'] = float(np.mean([np.arccos(-b['d'][2]) for b in lost]))
    say('[n] aerogel photons (transmitted, %.0f nm) reach the mirror sphere at rho = %.2f-%.2f mm, %.1f mrad in the gas'
        % (LAM0, *ctx['rho_a'], 1e3*ctx['th_gas_a']))
    say('[n] aerogel bottom-face reflection lands at r = %.3f-%.3f mm; gas bottom-face ghosts at r = %.3f-%.3f mm (%.0f nm)'
        % (*ctx['aero_r'], *ctx['ghost_r'], LAM0))


SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(x, nd=2):
    """x as a×10ⁿ with unicode superscripts."""
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x)))); m = x/10**e
    return ('%.*f×10%s' % (nd-1, m, str(e).translate(SUP))) if e not in (0, 1, -1) else ('%.*g' % (nd+1, x))


def direct_and_ghosts(br, i):
    d = pick(br, cls=0, reason=0, i=i); b = pick(br, cls=1, reason=0, i=i); t = pick(br, cls=2, reason=0, i=i)
    return (d[0] if d else None), (b[0] if b else None), (t[0] if t else None)


def tile_shift_mc(lay, n, seed):
    """Same gas photons traced with and without the tile: per-photon radius shifts of the direct ring photons."""
    sc = lay.scene(); sc_nt = dataclasses.replace(sc, slabs=()); f = lay.mirror.f; zs = lay.z_sensor
    rng = np.random.default_rng(seed); lo, hi = qe_band(); spec = PhotonSpectrum(lay.gas.index, lo, hi)
    seg = sim.track_segments(lay, (0, 0, zs), DOWN)
    p, d, lam, E = sim.emit_photons(rng, (0, 0, zs), DOWN, seg['gas'], n, spec)
    reg = np.full(n, -1, np.int16)
    r = []
    for scene in (sc, sc_nt):
        e = tracer.trace(scene, p, d, lam, reg, E)
        m = (e['reason'] == 0)&(e['cls'] == 0)
        rr = np.full(n, np.nan); rr[e['i'][m]] = np.hypot(e['p'][m, 0], e['p'][m, 1]); r.append(rr)
    rpar = f*np.tan(th_c(lay.gas.n(lam)))
    w = qe_arc(HC/lam)
    res = {}
    for wn, ww in (('ftqe', w), ('ft', np.ones(n))):
        res[wn] = dict(tile=np.average(r[0]-r[1], weights=ww), sph=np.average(r[1]-rpar, weights=ww),
                       tot=np.average(r[0]-rpar, weights=ww))
    return res


# Figure 2: refraction through the tile and the gap

def fig_tile_refraction(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); sc_nt = dataclasses.replace(sc, slabs=())
    zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor; t = lay.tile.thickness
    zes = np.array(Z_TILE_SHIFT)
    pts = np.c_[np.zeros_like(zes), np.zeros_like(zes), zes]
    P, D, L, Rg, E = emit(sc, lay.gas.n, pts, DOWN, LAM0, [0.0])
    br = follow(sc, P, D, L, Rg, E); brn = follow(sc_nt, P, D, L, Rg, E)
    ng = float(lay.gas.n(LAM0)); na = lay.tile.n
    fig = plt.figure(figsize=(14, 7.4))
    ax = fig.add_axes([0.045, 0.09, 0.27, 0.83])
    # (a) one photon to scale
    i0 = 0
    dct, gb, gt = direct_and_ghosts(br, i0); nt = pick(brn, cls=0, reason=0, i=i0)[0]
    xl = dct['pts'][-1][0]
    XL = (xl-6.6, xl+1.4); ZL = (189.3, 205.0)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.6)
    ray(ax, nt['pts'], '#6b7280', ls=(0, (1, 2)), lw=1.3, zorder=2)
    ray(ax, dct['pts'], GAS, lw=1.5, arrows=(1, 2, 3))
    kb = 2; kt = 3                                           # vertex index of the bottom and top face crossings
    ray(ax, gb['pts'][kb:kb+2], GHOST, ls=(0, (5, 3)), lw=1.3, arrows=(0,))
    ray(ax, gt['pts'][kt:kt+3], GHOST, ls=(0, (5, 3)), lw=1.3, arrows=(0, 1))
    evb, evt = dct['events'][1], dct['events'][2]
    alpha = evb['inc']
    note(ax, XL[0]+0.1, zt0-0.9, 'bottom face, Xe → aerogel\nincidence %.1f mrad\nT = %.5f   R = %s'
         % (1e3*evb['inc'], evb['T'], sci(evb['R'])), fs=8, va='top')
    note(ax, XL[0]+0.1, zt1-0.35, 'top face, aerogel → Xe\nincidence %.1f mrad\nT = %.5f   R = %s'
         % (1e3*evt['inc'], evt['T'], sci(evt['R'])), fs=8, va='top')
    note(ax, XL[0]+0.1, 0.5*(zt0+zt1)-1, 'aerogel tile\n%.0f mm, n = %.2f' % (t, na), color='#92400e', fs=8.5)
    note(ax, XL[0]+0.1, 0.5*(zt1+zs), 'gap %.0f mm, Xe' % lay.gap, fs=8)
    note(ax, XL[1]-0.1, zs+0.3, 'SiPM (absorbs)', color='white', fs=8, ha='right').set_bbox(dict(fc=SENSOR_C, ec='none'))
    note(ax, gb['pts'][kb+1][0]-0.35, 189.8, 'reflected at\nthe bottom face', color=GHOST, fs=7.5, ha='right', va='bottom')
    note(ax, gt['pts'][kt+2][0]+0.1, 190.4, 'reflected at\nthe top face', color=GHOST, fs=7.5, ha='left', va='bottom')
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) a returning gas photon, to scale\n(emitted at z = %.0f mm, %.0f nm)' % (zes[i0], LAM0), loc='left')
    # (b) deviation from the no-tile path
    bx = fig.add_axes([0.38, 0.09, 0.27, 0.83])
    zz = np.linspace(184, zs, 600)
    cols = plt.get_cmap('Blues')(np.linspace(0.95, 0.45, len(zes)))
    shifts, alphas = [], []
    for i in range(len(zes)):
        d_, _, _ = direct_and_ghosts(br, i); n_ = pick(brn, cls=0, reason=0, i=i)[0]
        dx_ = (x_at(d_['pts'], zz)[:, 0]-x_at(n_['pts'], zz)[:, 0])*1e3
        a_ = d_['events'][1]['inc']; alphas.append(a_); shifts.append((d_['pts'][-1][0]-n_['pts'][-1][0])*1e3)
        bx.plot(dx_, zz, color=cols[i], lw=1.6, label='emitted at z = %3.0f mm: α = %4.1f mrad, shift %5.1f µm'
                % (zes[i], 1e3*a_, shifts[-1]))
    bx.axhspan(zt0, zt1, color=TILE_FILL, zorder=0); bx.axhspan(zs, zs+0.6, color=SENSOR_C, zorder=0)
    bx.axvline(0, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    bx.text(0.35, 186, 'no-tile path', color=MUTED, fontsize=8.5, ha='left', rotation=90)
    bx.text(min(shifts)*0.97, 197.2, 'inside the tile the ray is\nbent towards the normal:\nthe deviation grows linearly',
            fontsize=8.5, color=INK, va='center')
    bx.text(min(shifts)*0.97, 202.7, 'gap: parallel to the original ray', fontsize=8.5, color=INK, va='center')
    bx.set_ylim(184, zs+0.6); bx.set_xlim(min(shifts)*1.08, 2.5)
    bx.set_xlabel('x(with tile) − x(without tile) [µm]')
    bx.set_ylabel('z [mm]')
    bx.legend(loc='lower left', bbox_to_anchor=(0.0, 0.1), fontsize=7.8, title='angle α of the returning ray to the axis',
              title_fontsize=8)
    bx.set_title('(b) lateral shift caused by the tile (x in µm, z in mm)', loc='left')
    # (c) shift at the SiPM vs angle
    cx = fig.add_axes([0.72, 0.09, 0.26, 0.83])
    zfine = np.linspace(2, zt0-2, 25)
    P2, D2, L2, R2, E2 = emit(sc, lay.gas.n, np.c_[np.zeros_like(zfine), np.zeros_like(zfine), zfine], DOWN, LAM0, [0.0])
    b2 = follow(sc, P2, D2, L2, R2, E2); b2n = follow(sc_nt, P2, D2, L2, R2, E2)
    al, sh = [], []
    for i in range(len(zfine)):
        d_, _, _ = direct_and_ghosts(b2, i); n_ = pick(b2n, cls=0, reason=0, i=i)[0]
        al.append(d_['events'][1]['inc']); sh.append((np.hypot(*d_['pts'][-1][:2])-np.hypot(*n_['pts'][-1][:2]))*1e3)
    al = np.array(al); sh = np.array(sh)
    aa = np.linspace(0, al.max()*1.05, 200); at = np.arcsin(ng*np.sin(aa)/na)
    cx.plot(1e3*aa, -t*(np.tan(aa)-np.tan(at))*1e3, color=MUTED, lw=1.0, label='t (tan α − tan α_tile), t = %.0f mm' % t)
    cx.plot(1e3*al, sh, 'o', color=GAS, ms=4.5, label='tracer (%d emission heights, %.0f nm)' % (len(zfine), LAM0))
    ms = ctx['tile_mc']
    cx.axhline(1e3*ms['ftqe']['tile'], color=GAS, lw=1.0, ls='--',
               label='mean over the FT×QE spectrum and track: %.1f µm' % (1e3*ms['ftqe']['tile']))
    cx.set_xlabel('angle α of the returning ray to the axis [mrad]'); cx.set_ylabel('radius shift at the SiPM [µm]')
    cx.legend(loc='lower left', fontsize=8)
    cx.text(0.97, 0.97, 'Ring radius budget, %s, FT×QE:\n⟨r⟩ − f·tan θc(λ) = %.1f µm\n  spherical aberration %.1f µm\n  tile refraction %.1f µm'
            % (lay.gas.label, 1e3*ms['ftqe']['tot'], 1e3*ms['ftqe']['sph'], 1e3*ms['ftqe']['tile']), transform=cx.transAxes, va='top',
            ha='right', fontsize=8.5, color=INK)
    cx.set_title('(c) radius shift at the SiPM\n(no Rayleigh scattering, see sketch 9)', loc='left'); cx.grid(alpha=.3)
    p = save(out, 'sketch_effects_2_tile_refraction.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig2] photon z_e = %.0f mm: alpha = %.2f mrad; bottom face T = %.6f R = %.4e (sfrac %.2g); top face T = %.6f R = %.4e'
        % (zes[i0], 1e3*alpha, evb['T'], evb['R'], evb['sfrac'], evt['T'], evt['R']))
    say('[fig2] shifts at the SiPM [um] for z_e = %s: %s' % (zes.tolist(), ', '.join('%.2f' % v for v in shifts)))
    say('[fig2] max |shift| %.2f um at alpha %.1f mrad; MC mean (FTxQE) tile %.4f mm, spherical aberration %.4f mm, total %.4f mm; FT: %.4f / %.4f / %.4f'
        % (abs(sh).max(), 1e3*al[np.argmax(abs(sh))], ms['ftqe']['tile'], ms['ftqe']['sph'], ms['ftqe']['tot'],
           ms['ft']['tile'], ms['ft']['sph'], ms['ft']['tot']))
    return p


# Figure 3: aerogel Fresnel reflection at the tile bottom face

def fig_aerogel_reflection(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    za = np.array([zt1-0.02, 0.5*(zt0+zt1), zt0+0.02])
    P, D, L, Rg, E = emit(sc, lay.tile_index.n, np.c_[np.zeros(3), np.zeros(3), za], DOWN, LAM0, [0.0])
    br = follow(sc, P, D, L, Rg, E)
    bru = follow(sc, P, D, L, Rg, None)                     # unpolarised Fresnel, same photons
    fig = plt.figure(figsize=(14, 7.2))
    ax = fig.add_axes([0.05, 0.09, 0.40, 0.83])
    XL = (-2.2, 9.8); ZL = (189.2, 204.8)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.6)
    ax.plot([0, 0], [ZL[0], zs], color=INK, lw=1.6, zorder=5)
    ax.annotate('', xy=(0, ZL[0]+0.2), xytext=(0, ZL[0]+1.5), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6), zorder=5)
    note(ax, -0.15, 190.0, 'μ', fs=10, ha='right')
    shades = ['#b45309', AEROGEL, '#f59e0b']
    ends = []
    for i in range(3):
        tr = pick(br, cls=0, reason=1, i=i)[0]; rf = pick(br, cls=1, reason=0, i=i)[0]
        ray(ax, tr['pts'], shades[i], lw=1.5, arrows=(0, 1))
        ray(ax, rf['pts'][1:], shades[i], ls=(0, (5, 3)), lw=1.4, arrows=(0, 1, 2))
        ax.plot(*tr['pts'][0][[0, 2]], 'o', color=shades[i], ms=4.5, zorder=6)
        ends.append(rf['pts'][-1][0])
        if i == 0:
            mu = pick(br, cls=4, reason=1, i=i)[0]
            ray(ax, mu['pts'][2:4], shades[i], ls=(0, (1, 2)), lw=1.0, alpha=.8, arrows=(0,))
            ev2 = mu['events'][1]
    evb, evt = pick(br, cls=1, reason=0, i=1)[0]['events'][:2]
    evu = pick(bru, cls=1, reason=0, i=1)[0]['events'][0]
    ax.annotate('', xy=(ends[2], zs-0.25), xytext=(ends[0], zs-0.25), arrowprops=dict(arrowstyle='<->', color='#b45309', lw=1.0))
    note(ax, 0.5*(ends[0]+ends[2]), zs-0.5, 'lands at r = %.2f–%.2f mm' % tuple(ctx['aero_r']), color='#b45309', fs=8.5,
         ha='center', va='top')
    note(ax, 3.7, zt0-0.25, 'bottom face, aerogel → Xe\nincidence %.2f°, p-polarised\nR_p = %s (unpolarised %s)\ntransmitted (1 − R_p): to the mirror\nsphere at ρ = %.1f–%.1f mm, lost'
         % (np.degrees(evb['inc']), sci(evb['R'], 3), sci(evu['R'], 3), *ctx['rho_a']), fs=8, va='top')
    note(ax, XL[1]-0.1, zs+0.3, 'SiPM', color='white', fs=8, ha='right').set_bbox(dict(fc=SENSOR_C, ec='none'))
    note(ax, 6.0, zt1-0.6, 'top face, aerogel → Xe\nT = %.5f\nreflected again (dotted):\nweight %s, then lost\nat the mirror aperture' % (evt['T'], sci(ev2['R']*evb['R'])),
         fs=8, va='top')
    note(ax, -2.0, 0.5*(zt0+zt1)+2.5, 'emitted at the\ntop, middle and\nbottom of the tile', color='#b45309', fs=8.5)
    note(ax, -2.0, 0.5*(zt1+zs), 'gap', fs=8)
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) aerogel photons (%.0f nm) in the tile and gap, to scale' % LAM0, loc='left')
    # (b) MC radial profile of the reflected aerogel light
    bx = fig.add_axes([0.53, 0.09, 0.44, 0.83])
    rc = (np.arange(int(round(R_MAX/R_BIN)))+0.5)*R_BIN
    for name, wn, ls in (('xe3', 'ftqe', '-'), ('xe3', 'ft', '--'), ('xe2', 'ftqe', '-.')):
        s = ctx['mc'][name]; k = 0 if wn == 'ftqe' else 1; lab = '%s, %s' % (gas_label(s['layout'].gas), 'FT×QE' if wn == 'ftqe' else 'FT')
        y = rebin(s['radial'][k, 1, 1], R_REBIN); tot = s['cls_'+wn][1, 1]
        bx.step(rebin(rc, R_REBIN), y, where='mid', color=AEROGEL if name == 'xe3' else '#92400e', ls=ls, lw=1.4,
                label='%s: %s photons per muon' % (lab, sci(tot, 3)))
    for r_ in ctx['aero_r']:
        bx.axvline(r_, color=MUTED, lw=0.7, ls=':')
    bx.set_xlim(0, 9); bx.set_ylim(0, bx.get_ylim()[1]*1.3)
    bx.set_xlabel('radius r on the SiPM [mm]'); bx.set_ylabel('dN/dr [photons / mm / muon]')
    bx.legend(loc='upper right', fontsize=8.5)
    tot = ctx['mc']['xe3']['cls_ftqe'][1, 1]; em = ctx['mc']['xe3']['emit'][0, 1]
    bx.text(0.02, 0.55, 'The only aerogel light\non the SiPM: %s of\nthe emitted aerogel light\n(FT×QE).\n\nFlat in r: the landing\nradius is linear in the\nemission depth, which\nis uniform.'
            % sci(tot/em, 2), transform=bx.transAxes, va='center', fontsize=8.5, color=INK)
    bx.set_title('(b) Monte Carlo radial profile (%.2f mm bins),\nwithout Rayleigh scattering (see sketch 9)' % (R_BIN*R_REBIN), loc='left'); bx.grid(alpha=.3)
    p = save(out, 'sketch_effects_3_aerogel_reflection.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig3] aerogel %.0f nm: incidence %.3f deg, Rp %.4e, unpolarised R %.4e, ratio %.4f; top-face T %.6f; landing r %s mm'
        % (LAM0, np.degrees(evb['inc']), evb['R'], evu['R'], evb['R']/evu['R'], evt['T'], ', '.join('%.3f' % e for e in ends)))
    say('[fig3] MC %s aerogel bottom reflection: %.4e (FTxQE) %.4e (FT) photons/muon; fraction of emitted %.3e'
        % (lay.gas.label, ctx['mc']['xe3']['cls_ftqe'][1, 1], ctx['mc']['xe3']['cls_ft'][1, 1], tot/em))
    return p


def stretch_label(fig, ax, where=(0.02, 0.01)):
    """State the horizontal-to-vertical scale ratio of a deliberately anisotropic sketch."""
    sx, sz = px_per_mm(fig, ax)
    ax.text(*where, 'NOT TO SCALE: horizontal ×%.1f relative to vertical' % (sx/sz), transform=ax.transAxes, fontsize=8,
            color='#b91c1c', ha='left', va='bottom', zorder=9, bbox=dict(fc='white', ec='#b91c1c', lw=0.6, pad=2))
    return sx/sz


def rebin(y, k):
    return y.reshape(-1, k).mean(1)


# Figure 4: gas Fresnel ghosts

def ghost_kind(b):
    """'bottom'/'top' face of the single partial reflection of a ghost branch and whether it came after the mirror."""
    k = [e for e in b['events'] if e.get('kind') == 'refl'][0]
    first_m = next((n for n, e in enumerate(b['events']) if e['kind'] == 'mirror'), None)
    after = first_m is not None and first_m < b['events'].index(k)
    return k['face'], after


def fig_ghosts(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); f = lay.mirror.f
    zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    ze = Z_GHOST
    P, D, L, Rg, E = emit(sc, lay.gas.n, np.array([[0, 0, ze]]), DOWN, LAM0, [0.0])
    br = follow(sc, P, D, L, Rg, E)
    dct, gb, gt = direct_and_ghosts(br, 0)
    alpha = dct['events'][1]['inc']
    fig = plt.figure(figsize=(14.5, 7.8))
    ax = fig.add_axes([0.045, 0.08, 0.32, 0.84])
    XL = (-8.5, 31); ZL = (-5, 222)
    draw_layout(ax, lay, XL, ext=40, lw_m=2.2)
    ax.plot([0, 0], [0, zs], color=INK, lw=1.2, zorder=3)
    note(ax, 0.4, 20, 'μ', fs=9)
    ray(ax, dct['pts'], GAS, lw=1.4, arrows=(0, 1))
    ray(ax, gb['pts'][2:], GHOST, ls=(0, (5, 3)), lw=1.4, arrows=(0, 1))
    ray(ax, gt['pts'][3:], GHOST, ls=(0, (6, 2, 1, 2)), lw=1.2, arrows=(1, 2))
    rg_b = np.hypot(*gb['pts'][-1][:2]); rg_t = np.hypot(*gt['pts'][-1][:2]); rd = np.hypot(*dct['pts'][-1][:2])
    # auxiliary ray parallel to the ghost's downward leg, reflected at the mirror vertex
    v = gb['pts'][3]-gb['pts'][2]; v /= np.linalg.norm(v)
    a0 = -v*(zt0/(-v[2]))
    a1 = np.array([-v[0], 0, -v[2]]); a1 = a1*(f/a1[2])
    ax.plot([a0[0], 0, a1[0]], [a0[2], 0, a1[2]], color=MUTED, lw=0.9, ls=(0, (1, 1.5)), zorder=2)
    note(ax, -8.2, 150, 'auxiliary ray parallel\nto the ghost, reflected\nat the vertex: it meets\nthe focal plane at the\nsame point, f·tan α', color=MUTED,
         fs=7.5, ha='left')
    note(ax, 0.6, ze, 'emitted at\nz = %.0f mm' % ze, fs=8, va='center', ha='left')
    note(ax, 0.5, 178, 'returning ray:\nα = %.1f mrad' % (1e3*alpha), color=GAS, fs=8, ha='left')
    note(ax, 20.5, 45, 'reflected at the\ntile bottom (dashed)\nor top (dash-dot) face,\nit goes down at the\nsame angle α', color=GHOST, fs=8, ha='left')
    ax.annotate('ghost lands at\nr = %.2f mm = f·tan α' % rg_b, xy=(rg_b, zs+1.2), xytext=(rg_b-0.5, 213), color=GHOST, fontsize=8,
                ha='right', va='center', arrowprops=dict(arrowstyle='-', color=GHOST, lw=0.7))
    ax.annotate('ring\nr = %.2f mm' % rd, xy=(rd, zs+1.2), xytext=(rd+1.0, 213), color=GAS, fontsize=8,
                ha='left', va='center', arrowprops=dict(arrowstyle='-', color=GAS, lw=0.7))
    ax.set_xlim(*XL); ax.set_ylim(*ZL)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) ghost paths of one gas photon (%.0f nm)' % LAM0, loc='left')
    fig.canvas.draw()
    sr = stretch_label(fig, ax)
    ax.legend(handles=[Line2D([], [], color=GAS, lw=1.4, label='direct'),
                       Line2D([], [], color=GHOST, ls=(0, (5, 3)), lw=1.4, label='bottom-face ghost'),
                       Line2D([], [], color=GHOST, ls=(0, (6, 2, 1, 2)), lw=1.2, label='top-face ghost')],
              loc='upper right', bbox_to_anchor=(1.0, 0.64), fontsize=8, facecolor='white', framealpha=0.9, frameon=True,
              edgecolor='none')
    # (b) landing radius vs emission height
    bx = fig.add_axes([0.43, 0.08, 0.26, 0.84])
    zg = np.r_[np.linspace(0.5, zt0-0.05, 40), np.linspace(zt1+0.05, zs-0.05, 5)]
    P2, D2, L2, R2, E2 = emit(sc, lay.gas.n, np.c_[np.zeros_like(zg), np.zeros_like(zg), zg], DOWN, LAM0, [0.0])
    b2 = follow(sc, P2, D2, L2, R2, E2)
    pts = {k: [] for k in ('direct', ('bottom', True), ('top', True), ('bottom', False), ('top', False))}
    for b in b2:
        if b['reason'] != 0:
            continue
        if b['cls'] == 0:
            pts['direct'].append((zg[b['i']], np.hypot(*b['pts'][-1][:2])))
        elif b['cls'] in (1, 2):
            pts[ghost_kind(b)].append((zg[b['i']], np.hypot(*b['pts'][-1][:2])))
    pts = {k: np.array(v) for k, v in pts.items()}
    th = ctx['thg400']
    zz = np.linspace(0, zs, 300)
    bx.axvspan(zt0, zt1, color=TILE_FILL, zorder=0)
    bx.plot(zz, f*np.tan(th*(1-zz/f)), color=MUTED, lw=1.0, label='paraxial f·tan α, α = θc (1 − z/f)')
    bx.plot(*pts['direct'].T, 'o', color=GAS, ms=3.5, label='direct (ring)')
    bx.plot(*pts[('bottom', True)].T, 's', color=GHOST, ms=4.5, mfc='none', label='ghost: bottom-face reflection')
    bx.plot(*pts[('top', True)].T, 'x', color=GHOST, ms=4.5, label='ghost: top-face reflection')
    bx.plot(*pts[('bottom', False)].T, '^', color='#701a75', ms=4.5, label='gap light reflected on the way down,\nbottom face (r ≈ %.2f–%.2f mm)'
            % (pts[('bottom', False)][:, 1].min(), pts[('bottom', False)][:, 1].max()))
    bx.plot(*pts[('top', False)].T, 'v', color='#701a75', ms=4.5, label='… top face (r < %.2f mm)' % pts[('top', False)][:, 1].max())
    low = pts[('bottom', True)][pts[('bottom', True)][:, 0] < zt0, 1]
    low = np.r_[low, pts[('top', True)][pts[('top', True)][:, 0] < zt0, 1]]
    gap_after = np.r_[pts[('bottom', True)][pts[('bottom', True)][:, 0] > zt1, 1], pts[('top', True)][pts[('top', True)][:, 0] > zt1, 1]]
    hole = (max(gap_after.max(), pts[('top', False)][:, 1].max()), low.min())
    bx.set_xlim(0, zs+2); bx.set_ylim(0, 14.2)
    bx.set_xlabel('emission height z on the track [mm]'); bx.set_ylabel('landing radius r on the SiPM [mm]')
    bx.legend(loc='upper right', bbox_to_anchor=(1.0, 0.875), fontsize=7.6); bx.grid(alpha=.3)
    bx.set_title('(b) landing radius vs emission height (%.0f nm)' % LAM0, loc='left')
    bx.text(0.98, 0.585, 'r linear in z, z uniform along the track\n→ flat dN/dr for %.2f < r < %.2f mm' % (low.min(), low.max()),
            transform=bx.transAxes, fontsize=8.2, color=INK, va='top', ha='right')
    # zoom on the top end of the track
    kx = bx.inset_axes([0.07, 0.07, 0.40, 0.33])
    kx.axvspan(zt0, zt1, color=TILE_FILL, zorder=0)
    for k, mk, c in ((('bottom', True), 's', GHOST), (('top', True), 'x', GHOST), (('bottom', False), '^', '#701a75'), (('top', False), 'v', '#701a75')):
        kx.plot(*pts[k].T, mk, color=c, ms=4, mfc='none' if mk == 's' else c)
    kx.plot(zz, f*np.tan(th*(1-zz/f)), color=MUTED, lw=0.8)
    kx.axhspan(*hole, color=GHOST, alpha=0.08)
    kx.set_xlim(ZOOM_Z0, zs+0.5); kx.set_ylim(0, 1.6); kx.tick_params(labelsize=7)
    kx.text(ZOOM_Z0+1, 0.5*(hole[0]+hole[1]), 'empty: %.2f–%.2f mm' % hole, fontsize=7, color=GHOST, va='center')
    kx.text(0.5*(zt0+zt1), 1.45, 'tile', fontsize=7, color='#92400e', ha='center', va='top')
    kx.set_title('zoom: z > %.0f mm' % ZOOM_Z0, fontsize=7.5, loc='left')
    # (c) MC radial profile of the ghosts
    cx = fig.add_axes([0.745, 0.08, 0.235, 0.84])
    s3 = ctx['mc']['xe3']; rc = (np.arange(s3['radial'].shape[-1])+0.5)*R_BIN
    for ci, ls, lab in ((1, (0, (5, 3)), 'bottom-face reflections'), (2, (0, (6, 2, 1, 2)), 'top-face reflections')):
        cx.step(rebin(rc, R_REBIN), rebin(s3['radial'][0, 0, ci], R_REBIN)*1e4, where='mid', color=GHOST, ls=ls, lw=1.3,
                label='%s: %s per muon' % (lab, sci(s3['cls_ftqe'][0, ci], 3)))
    cx.axvline(s3['r_ftqe'][0], color=GAS, lw=1.0)
    cx.text(s3['r_ftqe'][0]-0.2, 11.6, 'gas ring (off scale)', color=GAS, fontsize=8, rotation=90, ha='right', va='top')
    xg = 0.5*(pts[('bottom', False)][:, 1].min()+pts[('bottom', False)][:, 1].max())
    yg = rebin(s3['radial'][0, 0, 1], R_REBIN)[int(xg/(R_BIN*R_REBIN))]*1e4                # profile height at the gap-light peak
    cx.annotate('gap light reflected at\nthe bottom face on the\nway down', xy=(xg, yg), xytext=(xg+1.3, yg-0.6), fontsize=7.5,
                color='#701a75', arrowprops=dict(arrowstyle='-', color='#701a75', lw=0.7), va='center')
    cx.set_xlim(0, 14.5); cx.set_ylim(0, 12.5)
    cx.set_xlabel('radius r on the SiPM [mm]'); cx.set_ylabel('dN/dr [10⁻⁴ photons / mm / muon]')
    cx.legend(loc='upper left', fontsize=7.6, title='%s, FT×QE, %.2f mm bins' % (s3['layout'].gas.label, R_BIN*R_REBIN), title_fontsize=8); cx.grid(alpha=.3)
    cx.set_title('(c) Monte Carlo ghost profile,\nwithout Rayleigh scattering (see sketch 9)', loc='left')
    jx = fig.add_axes([0.80, 0.135, 0.15, 0.24])
    jx.step(rc, s3['radial'][0, 0, 1]*1e4, where='mid', color=GHOST, lw=1.0, ls=(0, (5, 3)))
    jx.step(rc, s3['radial'][0, 0, 2]*1e4, where='mid', color=GHOST, lw=0.8, ls=(0, (6, 2, 1, 2)))
    jx.axvspan(*hole, color=GHOST, alpha=0.08)
    jx.set_xlim(0, ZOOM_R1); jx.set_ylim(0, 14); jx.tick_params(labelsize=7)
    jx.set_title('zoom r < %.1f mm, %.2f mm bins' % (ZOOM_R1, R_BIN), fontsize=7.5, loc='left'); jx.grid(alpha=.3)
    jx.text(0.01, 12.8, 'gap light', fontsize=7, color='#701a75', va='top')
    p = save(out, 'sketch_effects_4_gas_ghosts.png', dpi=150, fig=fig); plt.close(fig)
    plateau = s3['radial'][0, 0, 1][int(2/R_BIN):int(12/R_BIN)].mean()
    small = s3['radial'][0, 0, 1:3, :int(hole[0]/R_BIN)+2].sum()*R_BIN
    say('[fig4] z_e = %.0f mm: alpha = %.2f mrad; ghost bottom lands at %.3f, top at %.3f, direct %.3f mm; f tan alpha = %.3f mm; stretch x%.1f'
        % (ze, 1e3*alpha, rg_b, rg_t, rd, f*np.tan(alpha), sr))
    say('[fig4] after-mirror ghosts from gas below the tile: r = %.3f-%.3f mm; from the gap: %.3f-%.3f mm; gap light reflected on the way down: bottom %.3f-%.3f, top %.3f-%.3f mm; empty band %.3f-%.3f mm'
        % (low.min(), low.max(), gap_after.min(), gap_after.max(), pts[('bottom', False)][:, 1].min(), pts[('bottom', False)][:, 1].max(), pts[('top', False)][:, 1].min(),
           pts[('top', False)][:, 1].max(), *hole))
    say('[fig4] MC ghosts per muon (FTxQE): bottom %.4e top %.4e; plateau dN/dr (bottom) %.3e /mm; photons at r < %.2f mm (both classes) %.3e = %.2f%% of all single-reflection ghosts'
        % (s3['cls_ftqe'][0, 1], s3['cls_ftqe'][0, 2], plateau, hole[0], small, 100*small/s3['cls_ftqe'][0, 1:3].sum()))
    return p


# Figure 5: chromatic dispersion

WL_COL = {300: '#7c3aed', 400: '#0e7490', 600: '#b45309'}


def fig_chromatic(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); f = lay.mirror.f; zs = lay.z_sensor
    fig = plt.figure(figsize=(14.5, 8.6))
    ax = fig.add_axes([0.05, 0.08, 0.22, 0.85])
    zes = np.array(Z_CHROMA)
    XL = (12.2, 13.55); ZL = (183, 206.8)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.4)
    land = {}
    for wl, c in WL_COL.items():
        P, D, L, Rg, E = emit(sc, lay.gas.n, np.c_[np.zeros(3), np.zeros(3), zes], DOWN, float(wl), [0.0])
        br = follow(sc, P, D, L, Rg, E)
        dd = [direct_and_ghosts(br, i)[0] for i in range(3)]
        land[wl] = np.array([b['pts'][-1][0] for b in dd])
        for b in dd:
            ray(ax, b['pts'][1:], c, lw=1.2, arrows=(1,))
        ax.plot(land[wl].mean(), zs, 'v', color=c, ms=7, zorder=6)
        note(ax, land[wl].mean(), zs+0.75, '%d nm\n%.2f' % (wl, land[wl].mean()), color=c, fs=8, ha='center', va='bottom')
    ax.set_xlim(*XL); ax.set_ylim(*ZL)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) rays at %s nm\nnear the SiPM (%s)' % (', '.join(str(w) for w in WL_COL), lay.gas.label), loc='left')
    note(ax, XL[0]+0.03, 195.5, 'tile', color='#92400e', fs=8)
    note(ax, XL[0]+0.03, 186.5, 'emitted at\nz = %s mm' % ', '.join('%.0f' % z for z in zes), fs=7.5)
    fig.canvas.draw(); sr = stretch_label(fig, ax, where=(0.02, 0.005))
    # (b) ring radius vs wavelength
    lam = np.linspace(260, 960, 500)
    lo, hi = qe_band()
    bx = fig.add_axes([0.35, 0.56, 0.28, 0.37])
    for key, ls in (('xe3', '-'), ('xe2', '--'), ('c4f10', '-.')):
        g = GASES[key]
        bx.plot(lam, f*np.tan(th_c(g.n(lam))), color=RADIATOR[key], ls=ls, lw=1.5, label=gas_label(g))
    for wl, c in WL_COL.items():
        bx.plot(wl, land[wl].mean(), 'o', mfc=c, mec=INK, ms=7, zorder=5)
        bx.annotate('%d nm' % wl, (wl, land[wl].mean()), xytext=(6, 4), textcoords='offset points', fontsize=8, color=c)
    bx.set_xlim(255, 960); bx.set_xlabel('wavelength λ [nm]'); bx.set_ylabel('ring radius f·tan θc(λ) [mm]')
    bx.legend(loc='upper right', fontsize=8.5, title='dots: traced %s rays of (a)' % lay.gas.label, title_fontsize=8); bx.grid(alpha=.3)
    bx.set_title('(b) ring radius vs wavelength (paraxial)', loc='left')
    bx.text(0.98, 0.42, 'the ARC SiPM QE is non-zero over\n%.0f–%.0f nm, the range shown' % (HC/hi, HC/lo), transform=bx.transAxes,
            fontsize=7.5, color=MUTED, ha='right')
    # (c) spectra
    cx = fig.add_axes([0.35, 0.08, 0.28, 0.37])
    for key, ls in (('xe3', '-'), ('c4f10', '-.')):
        g = GASES[key]; name = gas_label(g)
        sp = PhotonSpectrum(g.index, lo, hi); l_ = HC/sp.E
        dl = sp.pdf*sp.E**2/HC/10                                   # photons per nm per mm
        col = RADIATOR[key]
        cx.plot(l_, dl*1e3, color=col, ls=ls, lw=1.0, alpha=0.55, label='%s, FT' % name)
        cx.plot(l_, dl*qe_arc(sp.E)*1e3, color=col, ls=ls, lw=1.8, label='%s, FT×QE' % name)
        say('[fig5] %s yield per mm in %.0f-%.0f nm: FT %.4f, FT x QE %.4f' % (key, HC/hi, HC/lo, sp.yield_per_mm,
            PhotonSpectrum(g.index, lo, hi, weight=qe_arc).yield_per_mm))
    cx.set_xlim(255, 960); cx.set_ylim(0, None)
    cx.set_xlabel('wavelength λ [nm]'); cx.set_ylabel('dN/dλ [10⁻³ photons / nm / mm]')
    cx.legend(loc='upper right', fontsize=8.5); cx.grid(alpha=.3)
    cx.set_title('(c) Frank–Tamm spectrum with and without the SiPM QE', loc='left')
    # (d) ring profiles
    dx = fig.add_axes([0.70, 0.08, 0.28, 0.85])
    rc = (np.arange(int(round(R_MAX/R_BIN)))+0.5)*R_BIN
    for key, wn, lw, al in (('xe3', 'ftqe', 1.6, 1), ('xe3', 'ft', 1.0, 0.55), ('xe2', 'ftqe', 1.4, 1), ('c4f10', 'ftqe', 1.6, 1)):
        s = ctx['mc'][key]; k = 0 if wn == 'ftqe' else 1; lab = '%s, %s' % (gas_label(s['layout'].gas), 'FT×QE' if wn == 'ftqe' else 'FT')
        y = s['radial'][k, 0, 0]
        m, r_ = s['r_'+wn]
        dx.step(rebin(rc, R_REBIN_RING), rebin(y, R_REBIN_RING), where='mid', color=RADIATOR[key], lw=lw, alpha=al,
                ls='-' if key != 'c4f10' else '-.', label='%s: ⟨r⟩ = %.3f, rms %.3f mm (%.2f mrad)' % (lab, m, r_, 1e3*r_/f))
    for wl, c in WL_COL.items():
        dx.axvline(land[wl].mean(), color=c, lw=0.9, ls=':')
        dx.text(land[wl].mean(), 0.6, '%d nm ' % wl, color=c, fontsize=8, rotation=90, va='top', ha='right',
                transform=dx.get_xaxis_transform())
    dx.set_xlim(9.6, 14.0); dx.set_ylim(0, dx.get_ylim()[1]*1.45)
    dx.set_xlabel('radius r on the SiPM [mm]'); dx.set_ylabel('dN/dr [photons / mm / muon]')
    dx.legend(loc='upper left', fontsize=7.8, title='direct gas photons, %.2f mm bins' % (R_BIN*R_REBIN_RING), title_fontsize=8); dx.grid(alpha=.3)
    dx.set_title('(d) Monte Carlo ring profiles,\nwithout Rayleigh scattering (see sketch 9)', loc='left')
    p = save(out, 'sketch_effects_5_chromatic.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig5] traced landing x (%s, mean of z_e = %s): ' % (lay.gas.label, '/'.join('%.0f' % z for z in zes)) + ', '.join('%d nm %.3f mm (spread %.4f)' % (wl, v.mean(), np.ptp(v)) for wl, v in land.items()) + '; stretch x%.1f' % sr)
    for key in ('xe3', 'xe2', 'c4f10'):
        s = ctx['mc'][key]
        say('[fig5] %s ring: FTxQE <r> %.4f rms %.4f mm (%.3f mrad); FT <r> %.4f rms %.4f mm' % (key, *s['r_ftqe'], 1e3*s['r_ftqe'][1]/f, *s['r_ft']))
    return p


# Figure 6: spherical aberration for tracks parallel to the axis

def aberr_trace(lay, h0, n, rng, th, L=TRACK_LEN, real=False):
    """Monochromatic photons at angle th from a vertical track at x = h0 (path [0, L] from the sensor plane), traced
    without the tile; mirror unbounded unless real. Returns rho on the mirror, reconstructed angle atan(r/f), detected mask."""
    sc = unbounded(dataclasses.replace(lay.scene(), slabs=()), SENSOR_HALF_WIDE, mirror=not real)
    zs = lay.z_sensor
    s = rng.uniform(0, L, n); phi = rng.uniform(0, 2*np.pi, n)
    p = np.c_[np.full(n, h0), np.zeros(n), zs-s]
    d = np.c_[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), -np.full(n, np.cos(th))]
    lamv = np.full(n, LAM0); reg = np.full(n, -1, np.int16)
    e = tracer.trace(sc, p, d, lamv, reg, None)
    det = np.zeros(n, bool); r = np.full(n, np.nan)
    m = e['reason'] == 0
    det[e['i'][m]] = True; r[e['i'][m]] = np.hypot(e['p'][m, 0], e['p'][m, 1])
    q, _ = tracer.sphere_far(p, d, np.array([0, 0, lay.mirror.R]), lay.mirror.R)
    return np.hypot(q[:, 0], q[:, 1]), np.arctan(r/lay.mirror.f), det


def fig_aberration(out, ctx):
    lay = ctx['lay']; f = lay.mirror.f; zs = lay.z_sensor
    sc = unbounded(dataclasses.replace(lay.scene(), slabs=()), SENSOR_HALF_WIDE)
    fig = plt.figure(figsize=(14.5, 8.4))
    ax = fig.add_axes([0.04, 0.08, 0.33, 0.85])
    XL = (-24, 82); ZL = (-6, 212)
    draw_layout(ax, dataclasses.replace(lay, tile=None), XL, ext=80, lw_m=2.4)
    cols = dict(zip(ABERR_FAN_OFFSETS, (GAS, '#1e3a8a')))
    th = ctx['thg400']; zes = np.array(Z_ABERR)
    for h0, c in cols.items():
        P, D, L, Rg, E = emit(sc, lay.gas.n, np.c_[np.full(3, h0), np.zeros(3), zes], DOWN, LAM0, [0.0, np.pi])
        D = np.c_[np.sin(th)*np.sign(D[:, 0]), np.zeros(len(D)), -np.full(len(D), np.cos(th))]
        for b in follow(sc, P, D, L, Rg, None):
            ray(ax, b['pts'], c, lw=0.9, arrows=(0,))
        ax.plot([h0, h0], [0, zs], color=INK, lw=1.3, zorder=4)
        ax.annotate('', xy=(h0, 2), xytext=(h0, 30), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3), zorder=4)
        note(ax, h0+1.5, 150 if h0 == 25 else 130, 'μ at\nx = %.0f mm' % h0, fs=8.5, ha='left', va='center', color=c)
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) tracks parallel to the axis, to scale\n(tile removed; θ = %.1f mrad at %.0f nm; mirror sphere extended)' % (1e3*th, LAM0), loc='left')
    h_far = max(cols); r_ring = f*np.tan(th)
    note(ax, 55, 30, 'the %.0f mm track uses the\nsphere at ρ ≈ %.0f–%.0f mm,\nbeyond the %.1f mm mirror' % (h_far, h_far-r_ring, h_far+r_ring, lay.mirror.D/2),
         color='#1e3a8a', fs=8, ha='center')
    note(ax, -1, zs+3, 'rings stay centred on the axis (r ≈ ±%.1f mm)' % r_ring, fs=8, ha='center', va='bottom')
    # (b) landing deviation vs mirror radius, full azimuth
    rng = np.random.default_rng(ctx['seed']+SEED['aberration'])
    bx = fig.add_axes([0.44, 0.56, 0.25, 0.37])
    stats = {}
    offs = ABERR_OFFSETS; oc = {0.0: '#93c5fd', 15.0: '#3b82f6', 25.0: GAS, 50.0: '#1e3a8a'}
    for h0 in offs:
        rho, ang, det = aberr_trace(lay, h0, N_ABERR, rng, th)
        rho_r, ang_r, det_r = aberr_trace(lay, h0, N_ABERR, np.random.default_rng(ctx['seed']+SEED['aberration_real']), th, real=True)   # same draws per offset
        dr = f*np.tan(ang)-f*np.tan(th)
        k = slice(0, N_SHOWN)
        bx.plot(rho[k], 1e3*dr[k], '.', ms=1.5, color=oc[h0], alpha=0.6, rasterized=True)
        stats[h0] = dict(mean=np.nanmean(ang), rms=np.nanstd(ang), frac_real=det_r.mean(), rms_real=np.nanstd(ang_r[det_r]) if det_r.any() else np.nan,
                         ang=ang, rho=(np.nanmin(rho), np.nanmax(rho)))
    bx.axvline(lay.mirror.clear_radius, color=MIRROR, lw=0.9, ls='--'); bx.axvline(lay.mirror.D/2, color=MIRROR, lw=0.9, ls=':')
    bx.text(lay.mirror.clear_radius-0.8, 0.97, 'clear aperture', rotation=90, fontsize=7.5, color=MIRROR, ha='right', va='top',
            transform=bx.get_xaxis_transform())
    bx.text(lay.mirror.D/2+0.5, 0.97, 'mirror edge', rotation=90, fontsize=7.5, color=MIRROR, ha='left', va='top',
            transform=bx.get_xaxis_transform())
    bx.set_xlabel('radius ρ of the reflection point on the mirror [mm]')
    bx.set_ylabel('r − f·tan θ on the SiPM [µm]')
    bx.set_title('(b) landing error vs mirror zone', loc='left'); bx.grid(alpha=.3)
    bx.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=oc[h], label='track at x = %.0f mm' % h) for h in offs],
              loc='upper left', fontsize=8, title='%d photons per track' % N_SHOWN, title_fontsize=8)
    # (c) reconstructed angle
    cx = fig.add_axes([0.44, 0.08, 0.25, 0.37])
    bins = np.linspace(1e3*th-0.8, 1e3*th+2.4, 161)
    for h0 in offs:
        cx.hist(1e3*stats[h0]['ang'], bins=bins, histtype='step', color=oc[h0], lw=1.3, density=True,
                label='x = %.0f mm: rms %.2f mrad' % (h0, 1e3*stats[h0]['rms']))
    cx.axvline(1e3*th, color=MUTED, lw=0.8, ls=':')
    cx.set_yscale('log'); cx.set_xlabel('reconstructed angle atan(r/f) [mrad]'); cx.set_ylabel('density [1/mrad]')
    cx.legend(loc='upper right', fontsize=8); cx.grid(alpha=.3)
    cx.set_title('(c) reconstructed Cherenkov angle (mirror unbounded)', loc='left')
    # (d) numbers
    legacy = {}; th_rt = theta_xe(3.0)
    for h0 in ABERR_LEGACY_OFFSETS:
        t_, _ = sim.aberration(h0, 0.0, np.random.default_rng(1), Mirror(MIRROR_R), TRACK_LEN, th_rt, n=N_ABERR)   # same seed value and angle as raytrace.py, for comparison
        legacy[h0] = t_.std()
    tx = fig.add_axes([0.735, 0.08, 0.26, 0.85]); tx.axis('off')
    lines = ['Track parallel to the axis at offset x, vertical,', 'β = 1, θ = %.1f mrad (%.0f nm), photons uniform over' % (1e3*th, LAM0),
             '%.0f mm of track, full azimuth, tile removed.' % TRACK_LEN, '',
             'offset  mirror zone   rms angle  on the real', ' x [mm]   ρ [mm]       [mrad]   mirror (ρ<%.1f)' % lay.mirror.clear_radius]
    for h0 in offs:
        st = stats[h0]
        lines.append('  %3.0f   %4.1f–%4.1f     %5.2f      %5.1f %%' % (h0, *st['rho'], 1e3*st['rms'], 100*st['frac_real']))
    lines += ['', 'sim.aberration (FT x QE angle %.1f mrad, seed 1):' % (1e3*th_rt)] + \
        ['  x = %2.0f mm: rms %.2f mrad' % (h0, 1e3*v) for h0, v in legacy.items()] + \
        ['', 'For the vertical on-axis muon of the hit-map', 'study the mirror zone is ρ < %.1f mm and the' % stats[0.0]['rho'][1],
         'effect is a radius shift of %.1f µm (Fig. 2c)' % (1e3*ctx['tile_mc']['ftqe']['sph']), 'and an angle rms of %.3f mrad.' % (1e3*stats[0.0]['rms']), '',
         'The light of a track at %.0f mm offset lands on' % h_far, 'the sphere beyond the %.1f mm mirror (lost).' % lay.mirror.D]
    tx.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', fontsize=8.6, family='DejaVu Sans Mono', color=INK, transform=tx.transAxes)
    tx.set_title('(d) numbers', loc='left')
    p = save(out, 'sketch_effects_6_spherical_aberration.png', dpi=150, fig=fig); plt.close(fig)
    for h0 in offs:
        st = stats[h0]
        say('[fig6] offset %4.1f mm: rho %.1f-%.1f mm, <theta_rec> %.3f mrad, rms %.3f mrad; real mirror: %.3f of photons, rms %.3f mrad'
            % (h0, *st['rho'], 1e3*st['mean'], 1e3*st['rms'], st['frac_real'], 1e3*st['rms_real']))
    say('[fig6] legacy sim.aberration rms: ' + ', '.join('%.0f mm %.3f mrad' % (h, 1e3*v) for h, v in legacy.items()))
    return p


# Figure 7: total internal reflection

def tilted_track(lay, theta, through=None):
    """Start point on the sensor plane and direction of a track at zenith angle theta in the x-z plane through a point."""
    c = np.array([0.0, 0.0, 0.5*(lay.z_tile_bottom+lay.z_tile_top)]) if through is None else np.asarray(through, float)
    u = np.array([np.sin(theta), 0.0, -np.cos(theta)])
    return c-u*((lay.z_sensor-c[2])/np.cos(theta)), u


def first_face_dz(theta, tha, phi):
    """z component of aerogel photon directions around a track at zenith theta (azimuth phi from the tilt plane)."""
    return -np.cos(tha)*np.cos(theta)+np.sin(tha)*np.cos(phi)*np.sin(theta)


def fig_tir(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    tha = ctx['tha']; ng = float(lay.gas.n(LAM0)); na = lay.tile.n
    thcrit = np.arcsin(ng/na); thmu_min = thcrit-tha
    fig = plt.figure(figsize=(14, 9.6))
    ax = fig.add_axes([0.05, 0.56, 0.92, 0.37])
    th70 = np.radians(TILT_DEG)
    p0, u = tilted_track(lay, th70)
    seg = sim.track_segments(lay, p0, u)['tile'][0]
    XL = (-27, 27); ZL = (188.6, 205.0)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.5)
    e_end = p0+u*((zs-188.6)/np.cos(th70))
    ax.plot([p0[0], e_end[0]], [p0[2], e_end[2]], color=INK, lw=1.6, zorder=5)
    ax.annotate('', xy=(e_end[0], e_end[2]), xytext=(p0[0]+0.85*(e_end[0]-p0[0]), p0[2]+0.85*(e_end[2]-p0[2])),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6), zorder=5)
    fr = np.array(TIR_FRACS)
    pts = p0+np.outer(seg[0]+fr*(seg[1]-seg[0]), u)
    P, D, L, Rg, E = emit(sc, lay.tile_index.n, pts, u, LAM0, [0.0, np.pi])
    br = follow(sc, P, D, L, Rg, E)
    tirs = pick(br, cls=3)
    for b in tirs:
        ray(ax, b['pts'], '#b45309', lw=1.4, arrows=(0, 1, 2))
        for ev, q in zip(b['events'], b['pts'][1:]):
            if ev['kind'] == 'tir':
                ax.plot(q[0], q[2], 'o', ms=5, mfc='white', mec='#b45309', zorder=6)
    side = pick(br, cls=0, reason=2)
    for b in side:
        ray(ax, b['pts'], '#b45309', lw=1.4, arrows=(0,))
        ax.plot(b['pts'][-1][0], b['pts'][-1][2], marker='|', ms=9, mew=2, color='#7c2d12', zorder=6)
    trans = pick(br, cls=0, reason=1)
    for b in trans:
        ray(ax, b['pts'], AEROGEL, lw=1.3, arrows=(0,))
    refl = [b for b in br if b['cls'] == 1 and b['reason'] in (0, 3)]
    for b in refl:
        ray(ax, b['pts'][1:], AEROGEL, ls=(0, (5, 3)), lw=1.0, arrows=(0,))
    ev_t = trans[0]['events'][0]; ev_tir = tirs[0]['events'][0]
    note(ax, -26.5, 190.1, 'muon at %.0f° from the vertical' % TILT_DEG, fs=8.5)
    note(ax, -20.0, 197.4, 'photons at φ = 0 (most horizontal), %.1f° from the vertical:\nincidence %.1f° > critical %.2f° → total internal reflection\n(circle) where they meet a face; most reach the side face\nfirst (bars). The toy absorbs light at the side faces.'
         % (np.degrees(ev_tir['inc']), np.degrees(ev_tir['inc']), np.degrees(thcrit)), color='#92400e', fs=8, ha='left')
    note(ax, -9.2, 190.0, 'photons at φ = π: incidence %.1f°, transmitted\n(T = %.4f; reflected branch dashed, R = %s)' % (np.degrees(ev_t['inc']), ev_t['T'], sci(ev_t['R'])),
         color='#b45309', fs=8, ha='left')
    note(ax, 25.6, zs+0.25, 'SiPM', color='white', fs=8, ha='right').set_bbox(dict(fc=SENSOR_C, ec='none'))
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) aerogel photons (%.0f nm) of a muon at %.0f° zenith angle, in the plane of the track: tile, gap and SiPM to scale' % (LAM0, TILT_DEG), loc='left')
    # (b) incidence angle vs azimuth
    bx = fig.add_axes([0.06, 0.07, 0.40, 0.38])
    phi = np.linspace(0, 2*np.pi, 721)
    cmap = plt.get_cmap('Oranges')
    ths = (0, 20, 40, np.degrees(thmu_min), TILT_DEG)
    for k, tdeg in enumerate(ths):
        dz = first_face_dz(np.radians(tdeg), tha, phi)
        bx.plot(np.degrees(phi), np.degrees(np.arccos(np.abs(dz))), color=cmap(0.35+0.6*k/(len(ths)-1)), lw=1.6,
                label='θμ = %.1f°' % tdeg if k == 3 else 'θμ = %.0f°' % tdeg)
    bx.axhline(np.degrees(thcrit), color=INK, lw=1.0, ls='--')
    bx.axhspan(np.degrees(thcrit), 90, color='#fee2e2', zorder=0)
    bx.text(185, np.degrees(thcrit)+1.5, 'TIR: incidence > critical angle %.2f° (aerogel → %s, %.0f nm)' % (np.degrees(thcrit), lay.gas.label, LAM0),
            fontsize=8, color='#991b1b', ha='center', va='bottom')
    bx.set_xlim(0, 360); bx.set_ylim(0, 90); bx.set_xticks(range(0, 361, 60))
    bx.set_xlabel('photon azimuth φ around the track [deg] (φ = 0: towards the tilt)')
    bx.set_ylabel('incidence angle at the tile face [deg]')
    bx.legend(loc='center right', bbox_to_anchor=(1.0, 0.42), fontsize=8.5, title='muon zenith angle', title_fontsize=8)
    bx.grid(alpha=.3); bx.set_title('(b) incidence angle of aerogel photons at the first tile face', loc='left')
    # (c) TIR fraction vs zenith angle
    cx = fig.add_axes([0.56, 0.07, 0.41, 0.38])
    tg = np.linspace(0, 89, 400); phig = np.linspace(0, 2*np.pi, 3600, endpoint=False)
    frac_geom = [np.mean(np.abs(first_face_dz(np.radians(t), tha, phig)) < np.cos(thcrit)) for t in tg]
    cx.plot(tg, frac_geom, color='#b45309', lw=1.8, label='geometry: cone beyond the critical angle')
    rng = np.random.default_rng(ctx['seed']+SEED['tir'])
    wide = dataclasses.replace(sc, slabs=(tracer.Slab((-1e4, -1e4, zt0), (1e4, 1e4, zt1), lay.tile_index),))
    tt = np.arange(0, 86, 5.0); ftr, fwide, fside = [], [], []
    for t in tt:
        p0_, u_ = tilted_track(lay, np.radians(t))
        sg = sim.track_segments(lay, p0_, u_)['tile']
        n = N_TIR
        sv = rng.uniform(sg[0][0], sg[0][1], n)
        P_ = p0_+np.outer(sv, u_); ph_ = rng.uniform(0, 2*np.pi, n)
        e1 = np.array([np.cos(np.radians(t)), 0, np.sin(np.radians(t))]); e2 = np.cross(u_, e1)
        D_ = tracer.cone(u_[None], tha, ph_, e1[None], e2[None])[0]
        Ef = tracer.cherenkov_field(np.broadcast_to(u_, D_.shape), D_)
        e = tracer.trace(sc, P_, D_, np.full(n, LAM0), np.zeros(n, np.int16), Ef)
        ftr.append(e['w'][e['cls'] == 3].sum()/n); fside.append(e['w'][(e['cls'] == 0)&(e['reason'] == 2)].sum()/n)
        e = tracer.trace(wide, P_, D_, np.full(n, LAM0), np.zeros(n, np.int16), Ef)
        fwide.append(e['w'][e['cls'] == 3].sum()/n)
    ftr = np.array(ftr); fwide = np.array(fwide); fside = np.array(fside)
    cx.plot(tt, fwide, 'o', color=AEROGEL, ms=5.5, mec='#92400e', label='tracer, unbounded tile: TIR')
    cx.plot(tt, ftr, 's', color='#92400e', ms=5, mfc='white', label='tracer, %.0f mm tile: TIR' % (2*lay.tile.half))
    cx.plot(tt, fside, '^', color=MUTED, ms=5, mfc='white', label='tracer, %.0f mm tile: ends on a side face, no TIR' % (2*lay.tile.half))
    cx.plot(tg, np.cos(np.radians(tg))**2, color=MUTED, lw=1.0, ls='--', label='cos²θμ (relative muon intensity)')
    th_acc = np.degrees(np.arctan((lay.sensor.half-ctx['mc']['xe3']['r_ftqe'][0])/lay.mirror.f))
    cx.axvspan(0, th_acc, color=GAS, alpha=0.18, lw=0)
    cx.axvline(np.degrees(thmu_min), color=INK, lw=0.9, ls=':')
    cx.text(np.degrees(thmu_min)-1, 0.55, 'TIR starts at\nθμ = θcrit − θ_aerogel\n= %.1f°' % np.degrees(thmu_min), ha='right', fontsize=8.5, color=INK)
    cx.text(th_acc+1, 0.86, '← ring fully on the SiPM: |θμ| < %.1f°' % th_acc, fontsize=8.5, color=GAS)
    cx.set_xlim(0, 90); cx.set_ylim(0, 1.0)
    cx.set_xlabel('muon zenith angle θμ [deg]'); cx.set_ylabel('fraction')
    cx.legend(loc='center left', bbox_to_anchor=(0.0, 0.33), fontsize=7.8, title='track through the tile centre, %.0f nm' % LAM0, title_fontsize=7.8)
    cx.grid(alpha=.3)
    cx.set_title('(c) fraction of aerogel photons trapped by TIR', loc='left')
    p = save(out, 'sketch_effects_7_tir.png', dpi=150, fig=fig); plt.close(fig)
    # side faces (absorbing in the toy) for a vertical track
    inc_side = np.pi/2-tha
    Rs_, Rp_, _, _ = tracer.fresnel_powers(np.cos(inc_side), na, ng)
    crit = {}
    for g in GASES.values():
        name = g.label; nn = float(g.n(LAM0)); crit[name] = (np.degrees(np.arcsin(nn/na)), np.degrees(np.arcsin(nn/na)-tha))
    say('[fig7] critical angle aerogel->gas (%.0f nm): ' % LAM0 + ', '.join('%s %.2f deg (muon threshold %.1f deg)' % (k, *v) for k, v in crit.items()))
    say('[fig7] %.0f deg muon: phi=0 incidence %.2f deg (TIR); phi=pi incidence %.2f deg, T %.4f R %.4e; geometric TIR fraction at %.0f deg %.3f, tracer %.3f'
        % (TILT_DEG, np.degrees(ev_tir['inc']), np.degrees(ev_t['inc']), ev_t['T'], ev_t['R'], TILT_DEG, frac_geom[np.argmin(abs(tg-TILT_DEG))], ftr[tt == TILT_DEG][0]))
    say('[fig7] tracer TIR fractions (%.0f mm tile / unbounded / geometry / side first): ' % (2*lay.tile.half) + ', '.join('%.0f:%.3f/%.3f/%.3f/%.3f' % (a, b, c, np.mean(np.abs(first_face_dz(np.radians(a), tha, phig)) < np.cos(thcrit)), d) for a, b, c, d in zip(tt, ftr, fwide, fside)))
    say('[fig7] ring-on-SiPM acceptance |theta_mu| < %.2f deg; cos^2 at threshold %.3f' % (th_acc, np.cos(thmu_min)**2))
    say('[fig7] vertical track: aerogel photons would meet a vertical side face at %.2f deg (critical %.2f deg): Rs %.3f Rp %.3f; lateral travel in the tile <= %.2f mm vs half width %.0f mm'
        % (np.degrees(inc_side), np.degrees(thcrit), Rs_, Rp_, lay.tile.thickness*np.tan(tha), lay.tile.half))
    return p


# Figure 8: polarisation

def efield_marks(ax, b, every, L=1.3, color=INK, plane=(0, 2), lw=1.1):
    """Double-headed arrows along a branch showing the real part of the field (projected on the drawing plane)."""
    for k in range(len(b['pts'])-1):
        a, c = b['pts'][k], b['pts'][k+1]; Ev = np.real(b['Es'][k])
        seg = np.linalg.norm(c-a); nmk = max(1, int(seg/every))
        e2 = np.array([Ev[plane[0]], Ev[plane[1]]]); ne = np.linalg.norm(e2)
        if ne < 1e-6:
            continue
        e2 = e2/ne*L/2
        for t in (np.arange(nmk)+0.5)/nmk:
            m = a+t*(c-a)
            ax.annotate('', xy=(m[plane[0]]+e2[0], m[plane[1]]+e2[1]), xytext=(m[plane[0]]-e2[0], m[plane[1]]-e2[1]),
                        arrowprops=dict(arrowstyle='<|-|>,head_width=0.18,head_length=0.35', color=color, lw=lw,
                                        shrinkA=0, shrinkB=0), zorder=7)


def fig_polarisation(out, ctx):
    lay = ctx['lay']; sc = lay.scene(); zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    ng = float(lay.gas.n(LAM0)); na = lay.tile.n; tha = ctx['tha']
    fig = plt.figure(figsize=(14.5, 8.8))
    # (a) side view with E-field marks
    ax = fig.add_axes([0.04, 0.08, 0.30, 0.84])
    XL = (-1.0, 14.0); ZL = (187.5, 204.6)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.4)
    P, D, L, Rg, E = emit(sc, lay.tile_index.n, np.array([[0, 0, 0.5*(zt0+zt1)]]), DOWN, LAM0, [0.0])
    br = follow(sc, P, D, L, Rg, E)
    ta = pick(br, cls=0, reason=1)[0]; ra = pick(br, cls=1, reason=0)[0]
    ray(ax, ta['pts'], AEROGEL, lw=1.4); ray(ax, ra['pts'][1:], AEROGEL, ls=(0, (5, 3)), lw=1.3)
    efield_marks(ax, dict(pts=ta['pts'], Es=ta['Es']), 2.0, color='#7c2d12')
    efield_marks(ax, dict(pts=ra['pts'][1:], Es=ra['Es'][1:]), 2.5, color='#7c2d12')
    Pg, Dg, Lg, Rgg, Eg = emit(sc, lay.gas.n, np.array([[0, 0, Z_POL_GAS]]), DOWN, LAM0, [0.0])
    bg = follow(sc, Pg, Dg, Lg, Rgg, Eg)
    dg, gbg, _ = direct_and_ghosts(bg, 0)
    seg_pts = dg['pts'][1:]; seg_E = dg['Es'][1:]
    ray(ax, seg_pts, GAS, lw=1.4, arrows=(1, 2))
    efield_marks(ax, dict(pts=seg_pts, Es=seg_E), 2.5, color='#1e3a8a')
    ray(ax, gbg['pts'][2:4], GHOST, ls=(0, (5, 3)), lw=1.2)
    efield_marks(ax, dict(pts=gbg['pts'][2:4], Es=gbg['Es'][2:4]), 2.5, color=GHOST)
    sf = [e['sfrac'] for b in (ta, ra, dg, gbg) for e in b['events'] if e['kind'] != 'mirror']
    note(ax, -0.8, 199.5, 'μ', fs=9)
    ax.plot([0, 0], [ZL[0], zs], color=INK, lw=1.4, zorder=5)
    note(ax, 1.6, 196.5, 'aerogel photon\nE in the plane of\nthe track and photon', color='#7c2d12', fs=8)
    note(ax, 8.6, 190.2, 'returning gas photon\n(emitted at z = %.0f mm)' % Z_POL_GAS, color='#1e3a8a', fs=8, ha='left')
    note(ax, 2.6, 187.8, 'E (double arrows) lies in the drawing plane\n= plane of incidence: pure p-polarisation\n(largest s-component at any face: |E_s|² = %.2g)' % max(sf), fs=8, ha='left', va='bottom')
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) field direction on the traced rays (on axis), to scale', loc='left')
    # (b) top view of the aerogel photon directions at the bottom face: on axis vs tilted
    bx = fig.add_axes([0.385, 0.53, 0.30, 0.39])
    phis = np.linspace(0, 2*np.pi, 16, endpoint=False)
    S = 4.0                                                     # drawing scale of the horizontal direction components
    for cxo, tdeg in ((-2.4, 0.0), (0.6, 20.0)):
        th = np.radians(tdeg); u = np.array([np.sin(th), 0, -np.cos(th)])
        e1 = np.array([np.cos(th), 0, np.sin(th)]); e2 = np.cross(u, e1)
        d = tracer.cone(u[None], tha, phis, e1[None], e2[None])[0]
        Ev = np.real(tracer.cherenkov_field(np.broadcast_to(u, d.shape), d))
        for k in range(len(phis)):
            c0 = np.array([cxo, 0])+S*d[k, :2]
            bx.plot([cxo, c0[0]], [0, c0[1]], color=LIGHT, lw=0.8, zorder=1)
            exy = Ev[k, :2]; ln = np.linalg.norm(exy)
            if ln > 1e-9:
                h = exy/ln*0.17
                bx.annotate('', xy=c0+h, xytext=c0-h, arrowprops=dict(arrowstyle='<|-|>,head_width=0.15,head_length=0.3',
                            color='#7c2d12', lw=1.0, shrinkA=0, shrinkB=0), zorder=3)
        bx.plot(cxo, 0, 'o', ms=4, color=INK, zorder=4)
        bx.text(cxo+(0 if tdeg == 0 else 1.2), -1.55, 'muon at %.0f°' % tdeg + (' (on axis)' if tdeg == 0 else ', tilted towards +x'),
                ha='center', fontsize=8.5)
    bx.set_xlim(-3.7, 3.4); bx.set_ylim(-1.75, 1.4); bx.set_aspect('equal'); bx.axis('off')
    bx.set_title('(b) top view of the aerogel photon directions (grey: plane of\nincidence at the tile face) and their E projection (arrows)', loc='left')
    # (c) s-fraction vs azimuth for tilted tracks (tracer events at the first face)
    cx = fig.add_axes([0.40, 0.08, 0.27, 0.36])
    phi = np.linspace(0, 2*np.pi, 181)
    sfr_max = {}
    for k, tdeg in enumerate((0, 5, 20, 45)):
        p0, u = tilted_track(lay, np.radians(tdeg))
        Pt, Dt, Lt, Rt, Et = emit(sc, lay.tile_index.n, p0[None]+u[None]*((zs-0.5*(zt0+zt1))/u[2]*-1), u, LAM0, phi,
                                  e1=np.array([np.cos(np.radians(tdeg)), 0, np.sin(np.radians(tdeg))]))
        bt = follow(sc, Pt, Dt, Lt, Rt, Et, max_reflections=0)
        first = {}
        for b in bt:
            evs = [e for e in b['events'] if e['kind'] in ('trans', 'refl', 'tir')]
            if evs:
                first[b['i']] = evs[0]['sfrac']
        y = np.array([first.get(i, np.nan) for i in range(len(phi))])
        sfr_max[tdeg] = np.nanmax(y)
        cx.plot(np.degrees(phi), y, color=plt.get_cmap('Oranges')(0.4+0.18*k), lw=1.6, label='θμ = %d°: max %.2f' % (tdeg, sfr_max[tdeg]))
    cx.set_xlim(0, 360); cx.set_xticks(range(0, 361, 60)); cx.set_ylim(-0.02, 1.0)
    cx.set_xlabel('photon azimuth φ around the track [deg]'); cx.set_ylabel('s fraction |E_s|² at the first tile face')
    cx.legend(loc='upper right', fontsize=8); cx.grid(alpha=.3)
    cx.set_title('(c) s-polarised fraction, aerogel photons (tracer)', loc='left')
    # (d) Fresnel reflectances
    dx = fig.add_axes([0.73, 0.08, 0.255, 0.84])
    ang = np.radians(np.linspace(0.01, 89.99, 3000))
    for n1, n2, ls, lab in ((na, ng, '-', 'aerogel → %s' % lay.gas.label), (ng, na, '--', '%s → aerogel' % lay.gas.label)):
        Rs, Rp, _, _ = tracer.fresnel_powers(np.cos(ang), n1, n2)
        dx.plot(np.degrees(ang), Rs, color=AEROGEL if n1 > n2 else GAS, ls=ls, lw=1.6, label='R_s, %s' % lab)
        dx.plot(np.degrees(ang), np.maximum(Rp, 1e-12), color=AEROGEL if n1 > n2 else GAS, ls=ls, lw=0.9, alpha=0.75, label='R_p, %s' % lab)
    thB_in = np.degrees(np.arctan(ng/na)); thB_out = np.degrees(np.arctan(na/ng)); thc = np.degrees(np.arcsin(ng/na))
    for xv, lab, c, ha in ((thB_in, 'Brewster %.1f°' % thB_in, AEROGEL, 'right'), (thB_out, 'Brewster %.1f°' % thB_out, GAS, 'left'),
                           (thc, 'critical %.1f°' % thc, INK, 'right')):
        dx.axvline(xv, color=c, lw=0.7, ls=':')
        dx.text(xv, 2e-9, lab, rotation=90, fontsize=7.5, color=c, va='bottom', ha=ha)
    Rs_a, Rp_a, _, _ = tracer.fresnel_powers(np.cos(tha), na, ng)
    dx.plot(np.degrees(tha), Rp_a, 'o', color=AEROGEL, ms=6, mec=INK, zorder=6)
    dx.annotate('aerogel photon at the bottom face\n%.2f°: R_p = %s, R_s = %s\nR_p / unpolarised = %.3f' % (np.degrees(tha), sci(Rp_a, 3), sci(Rs_a, 3), 2*Rp_a/(Rs_a+Rp_a)),
                (np.degrees(tha), Rp_a), xytext=(8, 3e-6), fontsize=7.8, color='#92400e', arrowprops=dict(arrowstyle='-', color='#92400e', lw=0.7))
    amax = np.degrees(ctx['alpha_max'])
    dx.axvspan(0, amax, color=GAS, alpha=0.18, lw=0)
    Rs_g, Rp_g, _, _ = tracer.fresnel_powers(np.cos(ctx['alpha_max']), ng, na)
    dx.annotate('gas photons at the tile: 0–%.1f°\nR ≈ %s; R_p / unpolarised ≥ %.4f' % (amax, sci(Rp_g, 2), 2*Rp_g/(Rs_g+Rp_g)),
                (amax, Rp_g), xytext=(8, 2e-3), fontsize=7.8, color='#1e3a8a', arrowprops=dict(arrowstyle='-', color='#1e3a8a', lw=0.7))
    dx.set_yscale('log'); dx.set_ylim(1e-9, 1.5); dx.set_xlim(0, 90)
    dx.set_xlabel('incidence angle [deg]'); dx.set_ylabel('reflectance R')
    dx.legend(loc='upper left', bbox_to_anchor=(0.0, 0.83), fontsize=7.6); dx.grid(alpha=.3, which='both')
    dx.set_title('(d) Fresnel reflectances (tracer.fresnel_powers)', loc='left')
    p = save(out, 'sketch_effects_8_polarisation.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig8] on-axis s fractions at the faces (aerogel, gas, ghost): max %.3g' % max(sf))
    say('[fig8] s fraction max at the first face for tilted tracks: ' + ', '.join('%d deg %.3f' % kv for kv in sfr_max.items()))
    say('[fig8] Brewster aerogel->Xe %.2f deg, Xe->aerogel %.2f deg, critical %.2f deg' % (thB_in, thB_out, thc))
    say('[fig8] aerogel %.2f deg: Rs %.4e Rp %.4e ratio Rp/unpol %.4f; gas at %.2f deg: Rs %.4e Rp %.4e ratio %.5f; normal incidence R %.4e'
        % (np.degrees(tha), Rs_a, Rp_a, 2*Rp_a/(Rs_a+Rp_a), amax, Rs_g, Rp_g, 2*Rp_g/(Rs_g+Rp_g), ((na-ng)/(na+ng))**2))
    return p


# Figure 9: Rayleigh halo

SC_COL = '#be123c'                 # scattered paths and halo (crimson)
Y_PLANE = 0.3                      # drawn paths stay within this |y| at their scatter vertex [mm]
N_REC = 40000                      # photons per origin traced with recorded paths in fig 9a
N_GAS_PATHS, N_GAP_PATHS, N_AERO_PATHS, N_DIRECT_PATHS = 3, 1, 3, 3
R_REBIN_HALO = 20                  # profile bins merged in fig 9b (R_BIN x R_REBIN_HALO wide)


def scatter_vertices(path, zt0, zt1, skip_first):
    """Vertices strictly inside the tile slab (Rayleigh scatter points; the emission point is skipped if asked)."""
    q = np.asarray(path)[int(skip_first):]
    return q[(q[:, 2] > zt0+1e-6)&(q[:, 2] < zt1-1e-6)]


def pick_halo_paths(lay, ph, e):
    """Indices of recorded branches to draw: unscattered gas photons, scattered gas photons, scattered aerogel photons."""
    zt0, zt1 = lay.z_tile_bottom, lay.z_tile_top
    isc = tracer.CLASSES.index('scattered'); org = ph['origin'][e['i']]
    out = {'direct': [], 'gas': [], 'gap': [], 'aero': []}
    for k in range(len(e['i'])):
        if e['reason'][k] != 0:
            continue
        end = e['p'][k]
        if org[k] == 0 and e['cls'][k] == 0 and len(out['direct']) < N_DIRECT_PATHS and 10.0 < end[0] < 15.0 and abs(end[1]) < Y_PLANE:
            out['direct'].append(k)
        elif e['cls'][k] == isc:
            key = ('gap' if e['path'][k][0][2] > zt1 else 'gas') if org[k] == 0 else 'aero' if org[k] == 1 else None
            if key is None or len(out[key]) >= dict(gas=N_GAS_PATHS, gap=N_GAP_PATHS, aero=N_AERO_PATHS)[key]:
                continue
            sv = scatter_vertices(e['path'][k], zt0, zt1, org[k] == 1)
            if len(sv) == 1 and abs(sv[0, 1]) < Y_PLANE and abs(end[0]) < 15.0 and (key != 'gas' or 8.0 < sv[0, 0] < 15.0):
                out[key].append(k)
    return out


def fig_rayleigh(out, ctx):
    lay = ctx['mc']['xe3_ray']['layout']; zt0, zt1, zs = lay.z_tile_bottom, lay.z_tile_top, lay.z_sensor
    ray_ = lay.tile_scatter; t = lay.tile.thickness; tha = ctx['tha']
    lo, hi = qe_band()
    rng = np.random.default_rng(ctx['seed']+SEED['rayleigh'])
    spectra = {'gas': PhotonSpectrum(lay.gas.index, lo, hi), 'tile': PhotonSpectrum(lay.tile_index.index, lo, hi)}
    ph, e = sim.onaxis_muon(lay, rng, spectra, N_REC, N_REC, record=True)
    sel = pick_halo_paths(lay, ph, e)
    fig = plt.figure(figsize=(14.5, 9.4))
    ax = fig.add_axes([0.045, 0.33, 0.34, 0.59])
    XL = (-2.0, 16.0); ZL = (zt0-1.8, zs+1.0)
    draw_layout(ax, lay, XL, mirror=False, sensor_t=0.5)
    ax.plot([0, 0], [ZL[0], zs], color=INK, lw=1.4, zorder=5)
    ax.annotate('', xy=(0, ZL[0]+0.1), xytext=(0, ZL[0]+1.2), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4), zorder=5)
    note(ax, -0.2, ZL[0]+0.6, 'μ', fs=10, ha='right')

    def draw(k, color, ls='-', lw=1.0, marker=True, skip_first=False):
        pts = np.asarray(e['path'][k])
        ray(ax, pts, color, ls=ls, lw=lw, arrows=tuple(range(len(pts)-1)))
        if marker:
            sv = scatter_vertices(pts, zt0, zt1, skip_first)
            ax.plot(sv[:, 0], sv[:, 2], '*', color=color, mec=INK, mew=0.5, ms=11, zorder=6)
        return ph['lam'][e['i'][k]]
    for k in sel['direct']:
        draw(k, GAS, marker=False)
    lam_g = [draw(k, SC_COL, ls='-', lw=1.1) for k in sel['gas']]
    lam_p = [draw(k, SC_COL, ls='--', lw=1.1) for k in sel['gap']]
    lam_a = [draw(k, AEROGEL, ls='-', lw=1.4, skip_first=True) for k in sel['aero']]
    for k in sel['aero']:
        ax.plot(*e['path'][k][0][[0, 2]], 'o', color=AEROGEL, ms=4, zorder=6)
    note(ax, XL[0]+0.1, zt0-0.9, 'Xe', fs=8, va='top')
    note(ax, XL[0]+0.1, 0.5*(zt0+zt1), 'aerogel tile\n%.0f mm, n = %.2f\nΛ = %.0f mm at %.0f nm' % (t, lay.tile.n, ray_.length(LAM0), LAM0),
         color='#92400e', fs=8.5)
    note(ax, XL[1]-0.1, zs+0.3, 'SiPM', color='white', fs=8, ha='right').set_bbox(dict(fc=SENSOR_C, ec='none'))
    note(ax, XL[1]-0.1, zs-0.5*lay.gap-0.05, 'gap %.0f mm' % lay.gap, fs=7.5, ha='right', va='top')
    ax.legend(handles=[Line2D([], [], color=GAS, lw=1.0, label='gas photon from the mirror, not scattered'),
                       Line2D([], [], color=SC_COL, lw=1.1, label='gas photon from the mirror, scattered in the tile (λ = %s nm)' % ', '.join('%.0f' % v for v in lam_g)),
                       Line2D([], [], color=SC_COL, lw=1.1, ls='--', label='gas photon emitted in the gap, scattered while crossing the tile (λ = %s nm)' % ', '.join('%.0f' % v for v in lam_p)),
                       Line2D([], [], color=AEROGEL, lw=1.4, label='aerogel photon scattered in the tile (λ = %s nm)' % ', '.join('%.0f' % v for v in lam_a)),
                       Line2D([], [], ls='', marker='*', color='white', mec=INK, ms=10, label='scattering point')],
              loc='upper left', bbox_to_anchor=(0.0, -0.12), fontsize=7.6, frameon=False)
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) recorded photon paths, projected on the x-z plane, to scale\n(%s, %d sampled photons per origin)' % (lay.gas.label, N_REC), loc='left')
    # (b) radial profile
    s0 = ctx['mc']['xe3']; s1 = ctx['mc']['xe3_ray']; isc = tracer.CLASSES.index('scattered')
    rc = (np.arange(s1['radial'].shape[-1])+0.5)*R_BIN
    bx = fig.add_axes([0.43, 0.50, 0.55, 0.42])
    rad = s1['radial'][0]                                              # FT x QE: (origin, class, r)
    ring = rad[0, 0].sum()*R_BIN; halo_g = rad[0, isc].sum()*R_BIN; halo_a = rad[1, isc].sum()*R_BIN
    gas_other = (rad[0].sum(0)-rad[0, 0]-rad[0, isc]).sum()*R_BIN
    aero_other = (rad[1].sum(0)-rad[1, isc]).sum()*R_BIN
    gas_all = rad[0].sum()*R_BIN; aero_all = rad[1].sum()*R_BIN
    curves = ((rad[0, 0], GAS, '-', 'gas, direct (ring): %.2f photons' % ring),
              (rad[0, isc], SC_COL, '-', 'gas, scattered in the tile (halo): %.2f photons' % halo_g),
              (rad[1, isc], AEROGEL, ':', 'aerogel, scattered: %.2f photons' % halo_a),
              (rad[1].sum(0)-rad[1, isc], AEROGEL, '--', 'aerogel, other paths: %.3f photons' % aero_other))
    for y, c, ls, lab in curves:
        bx.step(rebin(rc, R_REBIN_HALO), np.maximum(rebin(y, R_REBIN_HALO), 1e-9), where='mid', color=c, ls=ls, lw=1.4, label=lab)
    bx.axvline(lay.sensor.half, color=MUTED, lw=0.8, ls=':')
    bx.text(lay.sensor.half+0.2, 0.04, 'SiPM half side\n%.1f mm' % lay.sensor.half, transform=bx.get_xaxis_transform(), fontsize=7.5, color=MUTED)
    bx.set_yscale('log'); bx.set_xlim(0, R_MAX); bx.set_ylim(1e-4, 3*rebin(rad[0, 0], R_REBIN_HALO).max())
    bx.set_xlabel('radius r on the SiPM [mm]'); bx.set_ylabel('dN/dr [photons / mm / muon]')
    bx.legend(loc='upper right', bbox_to_anchor=(1.0, 0.8), fontsize=8.2, title='%s, FT×QE, %.1f mm bins, per muon' % (lay.gas.label, R_BIN*R_REBIN_HALO), title_fontsize=8)
    bx.grid(alpha=.3, which='both')
    frac_halo = halo_g/gas_all; frac_halo_ring = halo_g/(ring+halo_g)
    ring0 = s0['cls_ftqe'][0, 0]
    bx.text(0.02, 0.97, 'halo fraction of the gas light on the SiPM: %.1f %%\n(%.2f of %.2f photons per muon)\nring without scattering: %.2f, with: %.2f photons (%.1f %%)'
            % (100*frac_halo, halo_g, gas_all, ring0, ring, 100*(ring/ring0-1)), transform=bx.transAxes, fontsize=8.5, va='top', color=INK,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    bx.set_title('(b) where the light lands: ring and Rayleigh halo (Monte Carlo, Λ = λ⁴/C, C = %g µm⁴/cm)' % ray_.C, loc='left')
    # (c) scattered fraction vs wavelength
    cx = fig.add_axes([0.43, 0.08, 0.55, 0.32])
    lc = 0.5*(SCATTER_LAM_EDGES[1:]+SCATTER_LAM_EDGES[:-1]); le = s1['lam_end']
    fg = scattered_fraction(le[1], 0); fa = scattered_fraction(le[1], 1)
    Lam = ray_.length(lc); cosa = np.cos(tha); La = Lam*cosa
    cx.step(lc, fg, where='mid', color=GAS, lw=1.5, label='gas photons crossing the tile (Monte Carlo)')
    cx.step(lc, fa, where='mid', color=AEROGEL, lw=1.5, label='aerogel photons (Monte Carlo)')
    cx.plot(lc, 1-np.exp(-t/Lam), color=GAS, ls='--', lw=1.0, label='1 − exp(−t/Λ), t = %.0f mm: one pass through the tile' % t)
    cx.plot(lc, 1-(La/t)*(1-np.exp(-t/La)), color=AEROGEL, ls='--', lw=1.0,
            label='aerogel, single pass from a uniform emission depth,\nΛ cos θ at θ = %.1f mrad' % (1e3*tha))
    cx.set_xlim(260, 960); cx.set_ylim(0, 1.0); cx.grid(alpha=.3)
    cx.set_xlabel('wavelength λ [nm]'); cx.set_ylabel('fraction with ≥ 1 Rayleigh scatter')
    cx.legend(loc='upper right', fontsize=7.8)
    cx.set_title('(c) scattered fraction vs wavelength (per bin, Frank–Tamm weighting)', loc='left')
    p = save(out, 'sketch_effects_9_rayleigh_halo.png', dpi=150, fig=fig); plt.close(fig)
    # numbers
    sg0 = s1['emit'][0]
    isc_ = tracer.CLASSES.index('scattered'); org_ = ph['origin'][e['i']]
    wgt = ph['weight'][e['i']]*qe_arc(HC/ph['lam'][e['i']])*e['w']
    z0 = np.array([q[0][2] for q in e['path']])
    mh = (org_ == 0)&(e['cls'] == isc_)&(e['reason'] == 0)
    gap_share = wgt[mh&(z0 > zt1)].sum()/wgt[mh].sum()
    say('[fig9] share of the gas halo from photons emitted in the %.0f mm gap above the tile (they cross it going down): %.1f %% (sampled paths, FTxQE)' % (lay.gap, 100*gap_share))
    say('[fig9] Rayleigh C = %g um^4/cm, A = %g; mean free path %.1f mm (300 nm), %.1f (400 nm), %.1f (600 nm), %.0f (900 nm); tile %.0f mm: 1-exp(-t/Lambda) = %.3f / %.3f / %.3f / %.3f'
        % (ray_.C, ray_.A, *ray_.length(np.array([300., 400., 600., 900.])), t, *(1-np.exp(-t/ray_.length(np.array([300., 400., 600., 900.]))))))
    say('[fig9] recorded paths drawn: %d direct gas, %d scattered gas from the mirror (lambda %s nm), %d from the gap (%s nm), %d scattered aerogel (lambda %s nm)'
        % (len(sel['direct']), len(sel['gas']), '/'.join('%.0f' % v for v in lam_g), len(sel['gap']), '/'.join('%.0f' % v for v in lam_p),
           len(sel['aero']), '/'.join('%.0f' % v for v in lam_a)))
    say('[fig9] %s with Rayleigh, per muon on the SiPM (FTxQE): gas ring %.3f, gas halo %.3f, gas other paths %.3f, aerogel scattered %.3f, aerogel other %.3f; emitted gas %.2f aerogel %.2f'
        % (lay.gas.label, ring, halo_g, gas_other, halo_a, aero_other, *sg0))
    say('[fig9] halo fraction of the gas light on the SiPM %.2f %% (halo/(ring+halo) %.2f %%); aerogel scattered share of the aerogel light on the SiPM %.1f %%'
        % (100*frac_halo, 100*frac_halo_ring, 100*halo_a/aero_all))
    say('[fig9] ring photons per muon: %.3f without Rayleigh, %.3f with (%+.2f %%); FT: %.3f -> %.3f'
        % (ring0, ring, 100*(ring/ring0-1), s0['cls_ft'][0, 0], s1['cls_ft'][0, 0]))
    def overall(k, oi):
        a = le[k, oi]; den = a.sum()-(a[tracer.REASONS.index('mirror aperture'), 0].sum() if oi == 0 else 0.0)
        return a[:, 1].sum()/den
    say('[fig9] scattered fraction (>= 1 Rayleigh scatter, of the weight reaching the tile): gas FTxQE %.4f (FT %.4f), aerogel FTxQE %.4f (FT %.4f)'
        % (overall(0, 0), overall(1, 0), overall(0, 1), overall(1, 1)))
    say('[fig9] ring change with Rayleigh: FTxQE %+.2f %%, FT %+.2f %%' % (100*(ring/ring0-1), 100*(s1['cls_ft'][0, 0]/s0['cls_ft'][0, 0]-1)))
    say('[fig9] per-bin scattered fraction (FT) at 300/400/600 nm: gas %.3f/%.3f/%.3f aerogel %.3f/%.3f/%.3f'
        % (*(fg[np.argmin(abs(lc-v))] for v in (300, 400, 600)), *(fa[np.argmin(abs(lc-v))] for v in (300, 400, 600))))
    return p


# Figure 10: the muon's own hit in the SiPM window

N_WIN_PATHS = 3                    # drawn photons per side in fig 10a
N_WIN_PTS = 4000                   # landing points drawn in fig 10b
R_BIN_WIN = 0.005                  # radial bin of the window spot profile [mm]
WIN_FILL = '#cffafe'
WIN_EDGE = '#0e7490'


def window_chunk(task):
    seed, n = task
    c = _CFG
    ph, e = sim.window_cluster(c['wlay'], np.random.default_rng(seed), c['wspec'], n, c['window'])
    q = sim.cluster_stats(ph, e, weight=lambda lam: qe_arc(HC/lam)); f = sim.cluster_stats(ph, e)
    det = e['reason'] == 0; i = e['i'][det]
    pts = np.c_[e['p'][det, :2], ph['weight'][i]*e['w'][det]]
    emit = [(ph['weight']*qe_arc(HC/ph['lam'])).sum(), ph['weight'].sum()]
    nb = int(round(1.0/R_BIN_WIN)); r = np.hypot(e['p'][det, 0], e['p'][det, 1]); ir = (r/R_BIN_WIN).astype(int)
    w = ph['weight'][i]*e['w'][det]
    hist = np.stack([np.bincount(ir, w*qe_arc(HC/ph['lam'][i]), minlength=nb), np.bincount(ir, w, minlength=nb)])
    return dict(q=q, f=f, emit=emit, hist=hist, pts=pts[:N_WIN_PTS])


def run_window(lay, n, nchunk, nproc, seed):
    """Window photons per muon on the silicon (FT x QE and FT), spot moments, radial histogram, sample of landing points."""
    lo, hi = qe_band()
    _CFG.update(wlay=lay, wspec=PhotonSpectrum(ConstantIndex(WINDOW_N), lo, hi), window=Window(WINDOW_N, WINDOW_T))
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(nchunk)]
    with mp.get_context('fork').Pool(nproc) as pool:
        res = pool.map(window_chunk, [(s, n) for s in seeds])
    out = {}
    for k in ('q', 'f'):
        nn = np.mean([r[k]['n'] for r in res])
        out[k] = dict(n=nn, r_mean=np.average([r[k]['r_mean'] for r in res], weights=[r[k]['n'] for r in res]),
                      r_rms=np.sqrt(np.average([r[k]['r_rms']**2 for r in res], weights=[r[k]['n'] for r in res])),
                      r_max=max(r[k]['r_max'] for r in res))
    out['emit'] = np.mean([r['emit'] for r in res], axis=0); out['hist'] = np.mean([r['hist'] for r in res], axis=0)/R_BIN_WIN
    out['pts'] = res[0]['pts']
    return out


def fig_window(out, ctx):
    lay = ctx['lay']; zs = lay.z_sensor; tw = WINDOW_T
    lo, hi = qe_band(); spec = PhotonSpectrum(ConstantIndex(WINDOW_N), lo, hi)
    thw = float(th_c(WINDOW_N)); ng = float(lay.gas.n(LAM0)); tcrit = np.arcsin(ng/WINDOW_N)
    r_in, r_out = tw*np.tan(thw), 2*tw*np.tan(thw)
    rng = np.random.default_rng(ctx['seed']+SEED['window'])
    ph, e = sim.window_cluster(lay, rng, spec, N_WIN_PICK, Window(WINDOW_N, WINDOW_T), record=True)
    cand = [k for k in range(len(e['i'])) if e['reason'][k] == 0 and len(e['path'][k]) == 3 and abs(e['path'][k][-1][1]) < 0.02]
    zdepth = np.array([zs-e['path'][k][0][2] for k in cand]); side = np.array([e['path'][k][-1][0] > 0 for k in cand])
    chosen = []
    for sd in (True, False):
        for fr in np.linspace(0.12, 0.88, N_WIN_PATHS):
            idx = np.flatnonzero(side == sd)
            chosen.append(cand[idx[np.argmin(abs(zdepth[idx]-fr*tw))]])
    run = ctx['window']
    fig = plt.figure(figsize=(14.5, 7.8))
    ax = fig.add_axes([0.045, 0.09, 0.34, 0.80])
    XL = (-0.95, 0.95); ZL = (zs-1.2, zs+0.22)
    ax.axvspan(*XL, color=GAS_FILL, zorder=0, lw=0)
    ax.add_patch(Rectangle((XL[0], zs-lay.gap-0.6), XL[1]-XL[0], 0.6, fc=TILE_FILL, ec=AEROGEL, lw=1.0, zorder=1))
    ax.add_patch(Rectangle((XL[0], zs-tw), XL[1]-XL[0], tw, fc=WIN_FILL, ec=WIN_EDGE, lw=1.0, zorder=1))
    ax.add_patch(Rectangle((XL[0], zs), XL[1]-XL[0], 0.22, fc=SENSOR_C, ec=SENSOR_C, lw=0.5, zorder=4))
    ax.plot([0, 0], [ZL[0], ZL[1]], color=INK, lw=1.6, zorder=5)
    ax.annotate('', xy=(0, zs-0.85), xytext=(0, zs-0.45), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6), zorder=5)
    for k in chosen:
        pts = np.asarray(e['path'][k]); ray(ax, pts, '#0e7490', lw=1.2, arrows=(0, 1))
        ax.plot(*pts[0][[0, 2]], 'o', color='#0e7490', ms=3.5, zorder=6)
        ax.plot(*pts[1][[0, 2]], 's', mfc='white', mec='#0e7490', ms=5, zorder=6)
        ax.plot(*pts[2][[0, 2]], 'v', color='#be123c', ms=6, mec=INK, mew=0.5, zorder=6)
    note(ax, XL[0]+0.03, zs+0.11, 'SiPM silicon (sensor plane, absorbs)', color='white', fs=8).set_bbox(dict(fc=SENSOR_C, ec='none'))
    note(ax, XL[0]+0.03, zs-0.5*tw, 'window, n = %g, t = %g mm' % (WINDOW_N, WINDOW_T), color=WIN_EDGE, fs=8.5)
    note(ax, XL[0]+0.03, zs-0.5*(tw+lay.gap), 'Xe gap, %.1f mm' % (lay.gap-tw), fs=8.5)
    note(ax, XL[0]+0.03, zs-lay.gap-0.1, 'aerogel tile (top part)', color='#92400e', fs=8.5)
    note(ax, 0.04, zs-0.65, 'μ', fs=10)
    note(ax, XL[1]-0.03, zs-0.5*tw-0.17, 'θc = acos(1/n) = %.1f°\ncritical angle to Xe %.1f°\n→ total internal reflection' % (np.degrees(thw), np.degrees(tcrit)),
         fs=8, ha='right', color=WIN_EDGE)
    ax.legend(handles=[Line2D([], [], color='#0e7490', lw=1.2, marker='o', ms=3.5, label='Cherenkov photon, emission point'),
                       Line2D([], [], ls='', marker='s', mfc='white', mec='#0e7490', ms=5, label='total internal reflection'),
                       Line2D([], [], ls='', marker='v', color='#be123c', mec=INK, ms=6, label='landing on the silicon')],
              loc='lower right', bbox_to_anchor=(1.0, 0.02), fontsize=7.8, facecolor='white', framealpha=0.9, frameon=True, edgecolor='none')
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title('(a) side view, to scale: light of the muon in the window\n(photons in the x-z plane, vertical μ at x = 0)', loc='left')
    # (b) top view
    bx = fig.add_axes([0.43, 0.09, 0.27, 0.80])
    pts = run['pts']; wsel = pts[:, 2] > 0.5*np.median(pts[:, 2])
    bx.plot(pts[wsel, 0], pts[wsel, 1], '.', ms=2.2, color='#0e7490', alpha=0.5, rasterized=True)
    t_ = np.linspace(0, 2*np.pi, 361)
    for rr, ls_, lab in ((r_in, '--', 't tan θc = %.3f mm' % r_in), (r_out, ':', '2t tan θc = %.3f mm' % r_out)):
        bx.plot(rr*np.cos(t_), rr*np.sin(t_), color='#be123c', ls=ls_, lw=1.2, label=lab)
    bx.plot(0, 0, '+', color=INK, ms=12, mew=1.5, label='impact point of the muon')
    L = 1.3*r_out; bx.set_xlim(-L, L); bx.set_ylim(-L, L); bx.set_aspect('equal')
    bx.set_xlabel('x [mm]'); bx.set_ylabel('y [mm]'); bx.legend(loc='upper right', fontsize=7.8, framealpha=0.9, frameon=True, edgecolor='none')
    bx.set_title('(b) landing points on the silicon, top view\n(%d photons shown, Monte Carlo)' % wsel.sum(), loc='left'); bx.grid(alpha=.3)
    # (c) photons per muon
    cx = fig.add_axes([0.795, 0.09, 0.19, 0.80])
    qn, fn = run['q']['n'], run['f']['n']
    labels = ['window light,\nFT×QE', 'window light,\nFT (no QE)', 'direct hit\n(silicon crossing)', 'total,\nFT×QE + direct']
    vals = [qn, fn, MIP_DIRECT_PE, qn+MIP_DIRECT_PE]
    cols = ['#0e7490', '#67c5d6', INK, '#be123c']
    cx.barh(range(4), vals, color=cols)
    for i_, v in enumerate(vals):
        cx.text(v+0.15, i_, '%.2f' % v, va='center', fontsize=9)
    cx.set_yticks(range(4)); cx.set_yticklabels(labels, fontsize=8.5); cx.invert_yaxis(); cx.set_xlim(0, max(vals)*1.25)
    cx.set_xlabel('photons (fired cells) per muon on the silicon'); cx.grid(alpha=.3, axis='x')
    cx.set_title('(c) size of the muon cluster\nspot (FT×QE): ⟨r⟩ = %.3f mm,\nrms %.3f mm, max %.3f mm' % (run['q']['r_mean'], run['q']['r_rms'], run['q']['r_max']), loc='left')
    p = save(out, 'sketch_effects_10_muon_window.png', dpi=150, fig=fig); plt.close(fig)
    say('[fig10] window n = %g, t = %g mm: theta_c = %.2f deg, critical angle to %s (%.0f nm) %.2f deg; annulus t tan(theta_c) = %.4f to 2 t tan(theta_c) = %.4f mm'
        % (WINDOW_N, WINDOW_T, np.degrees(thw), lay.gas.label, LAM0, np.degrees(tcrit), r_in, r_out))
    say('[fig10] window photons on the silicon per muon: FTxQE %.3f, FT %.3f (emitted %.3f / %.3f: %.1f %% / %.1f %% reach the silicon); direct hit %.1f fired cell; total with direct %.3f'
        % (qn, fn, *run['emit'], 100*qn/run['emit'][0], 100*fn/run['emit'][1], MIP_DIRECT_PE, qn+MIP_DIRECT_PE))
    say('[fig10] spot (FTxQE): <r> %.4f mm, rms %.4f mm, max %.4f mm; FT: <r> %.4f, max %.4f mm; drawn paths: %d of %d candidates'
        % (run['q']['r_mean'], run['q']['r_rms'], run['q']['r_max'], run['f']['r_mean'], run['f']['r_max'], len(chosen), len(cand)))
    return p


def check_follow(ctx):
    """follow() against tracer.trace in the scenes the figures use, polarised and unpolarised; raises on a mismatch."""
    lay = ctx['lay']; sc = lay.scene(); zt0, zt1 = lay.z_tile_bottom, lay.z_tile_top
    sc_nt = dataclasses.replace(sc, slabs=())
    sc_far = unbounded(sc_nt, SENSOR_HALF_WIDE)
    rng = np.random.default_rng(ctx['seed']+SEED['check'])
    zg = np.r_[rng.uniform(0.5, zt0, CHECK_N['gas_low']), rng.uniform(zt1, lay.z_sensor, CHECK_N['gas_gap'])]
    za = rng.uniform(zt0, zt1, CHECK_N['aero'])
    p0, u = tilted_track(lay, np.radians(TILT_DEG))
    seg = sim.track_segments(lay, p0, u)['tile'][0]

    def gas(scene):
        return emit(scene, lay.gas.n, np.c_[np.zeros_like(zg), np.zeros_like(zg), zg], DOWN, rng.uniform(270, 900, len(zg)),
                    rng.uniform(0, 2*np.pi, CHECK_N['phi']))

    gas_sc = gas(sc)
    aero = emit(sc, lay.tile_index.n, np.c_[np.zeros_like(za), np.zeros_like(za), za], DOWN, LAM0, rng.uniform(0, 2*np.pi, CHECK_N['phi']))
    tilted = emit(sc, lay.tile_index.n, p0+np.outer(rng.uniform(*seg, CHECK_N['tilted']), u), u, LAM0,
                  rng.uniform(0, 2*np.pi, CHECK_N['phi_tilted']))
    far = [emit(sc_far, lay.gas.n, np.c_[np.full_like(zg, h0), np.zeros_like(zg), zg], DOWN, LAM0, rng.uniform(0, 2*np.pi, CHECK_N['phi']))
           for h0 in ABERR_OFFSETS]
    cases = [('layout scene', sc, 2, [gas_sc, aero, tilted]), ('no tile', sc_nt, 2, [gas(sc_nt)]),
             ('unbounded mirror, larger sensor, no tile', sc_far, 2, far), ('layout scene, max_reflections = 0', sc, 0, [aero, tilted])]
    for name, scene, nrefl, sets in cases:
        worst = [0, 0.0, 0.0]
        for P, D, L, Rg, E in sets:
            for EE in (E, None):
                nb, nt, dp, dw = validate(scene, P, D, L, Rg, EE, nrefl)
                if nb != nt or not (dp <= FOLLOW_TOL_MM and dw <= FOLLOW_TOL_W):
                    raise RuntimeError('follow() and tracer.trace disagree (%s): %d vs %d branches, end point %.3g mm, weight %.3g'
                                       % (name, nb, nt, dp, dw))
                worst = [worst[0]+nb, max(worst[1], dp), max(worst[2], dw)]
        say('[check] follow() vs tracer.trace, %s: %d terminal branches, max |end point difference| %.3g mm, max |weight difference| %.3g'
            % (name, *worst))


FIGS = {1: fig_overview, 2: fig_tile_refraction, 3: fig_aerogel_reflection, 4: fig_ghosts, 5: fig_chromatic,
        6: fig_aberration, 7: fig_tir, 8: fig_polarisation, 9: fig_rayleigh, 10: fig_window}


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nphot', type=float, default=2e6, help='Monte Carlo photons per origin and gas (default 2e6)')
    p.add_argument('--chunk', type=int, default=50000, help='photons per origin and chunk (default 50000)')
    p.add_argument('--nproc', type=int, default=24, help='worker processes (default 24)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed; each sample adds its offset from SEED (default 20261010)')
    p.add_argument('--only', type=int, nargs='*', default=None, help='figure numbers to draw (default all)')
    args = p.parse_args(); out = cli.outdir(args)
    ctx = {'lay': layout(GASES['xe3']), 'seed': args.seed}
    say('sketch_effects: f = %.0f mm layout, %s unless stated, %d worker processes' % (ctx['lay'].mirror.f, ctx['lay'].gas.label, args.nproc))
    nchunk = int(np.ceil(args.nphot/args.chunk))
    t0 = time.time()
    ctx['mc'] = run_mc(tuple((k, g, None) for k, g in GASES.items()) + (('xe3_ray', GASES['xe3'], Rayleigh()),),
                       args.chunk, nchunk, args.nproc, args.seed)
    for k, s in ctx['mc'].items():
        say('[mc] %-6s %d x %d photons per origin: ring <r> FTxQE %.4f rms %.4f | FT %.4f rms %.4f mm; emitted FTxQE gas %.3f aerogel %.3f'
            % (k, nchunk, args.chunk, *s['r_ftqe'], *s['r_ft'], *s['emit'][0]))
    ctx['window'] = run_window(ctx['lay'], args.chunk, nchunk, args.nproc, args.seed+SEED['window_mc'])
    say('[mc] window photons: %d x %d per muon sample, wall time %.1f s' % (nchunk, args.chunk, time.time()-t0))
    numbers(ctx)
    check_follow(ctx)
    ctx['tile_mc'] = tile_shift_mc(ctx['lay'], N_TILE_MC, args.seed+SEED['tile_shift'])
    for k in sorted(FIGS):
        if args.only is None or k in args.only:
            say('wrote', FIGS[k](out, ctx))
    (out/'sketch_effects_numbers.txt').write_text('\n'.join(LINES)+'\n')


if __name__ == '__main__':
    main()
