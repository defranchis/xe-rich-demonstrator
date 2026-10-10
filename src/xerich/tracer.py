"""Vectorised 3D ray geometry and optics.

Points and directions are arrays of shape (..., 3) with z as the optical axis;
directions are unit vectors. Polarisation is carried as a complex unit field
vector E transverse to the direction.

- Geometry: cones, refraction, reflection, sphere and plane intersections, boxes.
- Fresnel: s/p amplitude and power coefficients, field splitting at a plane interface.
- trace: weight-splitting tracer through a Scene (ambient medium, Fresnel slabs,
  spherical mirror, sensor plane).
"""
from dataclasses import dataclass

import numpy as np

ORIGIN = np.zeros(3)


def basis(u):
    """Two unit vectors e1, e2 orthogonal to each direction in u (N,3), with e2 = u x e1."""
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.]]), np.array([[1., 0, 0]]))
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(u, e1)
    return e1, e2


def cone(u, th, phi, e1=None, e2=None):
    """Directions at angle th around each u (N,3) for azimuths phi (M,), shape (N,M,3).

    The azimuth is measured from e1 towards e2 (default: basis(u)).
    """
    if e1 is None:
        e1, e2 = basis(u)
    return (np.cos(th)*u[:, None, :]+np.sin(th)*(np.cos(phi)[None, :, None]*e1[:, None, :]
                                                 + np.sin(phi)[None, :, None]*e2[:, None, :]))


def refract(d, N, eta):
    """Snell refraction of directions d at a surface with unit normal N facing the incoming ray.

    eta = n_in/n_out. Returns the refracted unit directions and a mask that is False
    for total internal reflection.
    """
    cosi = -(d*N).sum(-1, keepdims=True); k = 1-eta**2*(1-cosi**2)
    ok = (k[..., 0] > 0); t = eta*d+(eta*cosi-np.sqrt(np.clip(k, 0, None)))*N
    return t/np.linalg.norm(t, axis=-1, keepdims=True), ok


def reflect(d, n):
    """Specular reflection of directions d at surfaces with unit normal n."""
    return d-2*(d*n).sum(-1, keepdims=True)*n


def sphere_far(p, d, C, R):
    """Far intersection of the rays p + s d with the sphere |x - C| = R.

    Returns the points and a mask of the rays that meet the sphere.
    """
    m = p-C; b = (m*d).sum(-1); c = (m*m).sum(-1)-R*R; disc = b*b-c
    s = -b+np.sqrt(np.clip(disc, 0, None))
    return p+s[..., None]*d, disc > 0


def sphere_normal(q, C, R):
    """Outward unit normal of the sphere |x - C| = R at the points q."""
    return (q-C)/R


def to_plane(p, d, z, valid=None):
    """Points where the rays p + t d cross the plane at height z.

    Rays outside the optional mask `valid` use d_z = 1 to avoid dividing by zero.
    """
    dz = d[..., 2] if valid is None else np.where(valid, d[..., 2], 1)
    t = (z-p[..., 2])/dz
    return p+t[..., None]*d


def advance(p, d, dz):
    """Move the points p along d by the height difference dz."""
    t = dz/d[..., 2]
    return p+t[..., None]*d


def snell_angle(th, n1, n2):
    """Angle to the normal after crossing a plane interface from index n1 to n2."""
    return np.arcsin(n1*np.sin(th)/n2)


# Fresnel optics and polarisation

def fresnel_amplitudes(cos_i, n1, n2):
    """Fresnel amplitude coefficients (rs, rp, ts, tp) and cos_t for light from index n1 into n2.

    cos_i is the cosine of the incidence angle. The p vectors follow p = k x s for each wave, so
    rp = (n2 cos_i - n1 cos_t)/(n2 cos_i + n1 cos_t). Under total internal reflection cos_t is
    imaginary and |rs| = |rp| = 1. All outputs are complex.
    """
    cos_i = np.asarray(cos_i, float); eta = np.asarray(n1, float)/np.asarray(n2, float)
    cos_t = np.sqrt((1-eta**2*(1-cos_i**2)).astype(complex))
    a, b = n1*cos_i, n2*cos_t; c, e = n2*cos_i, n1*cos_t
    return (a-b)/(a+b), (c-e)/(c+e), 2*a/(a+b), 2*a/(c+e), cos_t


def fresnel_powers(cos_i, n1, n2):
    """Fresnel reflectances and transmittances (Rs, Rp, Ts, Tp) from index n1 into n2; T = 0 under TIR."""
    rs, rp, ts, tp, cos_t = fresnel_amplitudes(cos_i, n1, n2)
    g = np.real(n2*cos_t)/(n1*np.asarray(cos_i, float))
    return np.abs(rs)**2, np.abs(rp)**2, g*np.abs(ts)**2, g*np.abs(tp)**2


def cherenkov_field(u, k):
    """Cherenkov polarisation: unit E in the plane of the track u and the photon k, transverse to k (complex)."""
    e = (u*k).sum(-1, keepdims=True)*k-u
    e = e-(e*k).sum(-1, keepdims=True)*k               # second projection removes rounding for small angles
    return (e/np.linalg.norm(e, axis=-1, keepdims=True)).astype(complex)


def mirror_field(E, n):
    """Field after reflection on a perfect conductor with unit normal n: 2 (E.n) n - E."""
    return 2*(E*n).sum(-1, keepdims=True)*n-E


def interface_fields(d, E, N, n1, n2):
    """Reflected and transmitted field amplitudes at a plane interface (unit normal N facing the incoming ray).

    d (M,3) unit directions, E (M,3) complex incident fields, n1 and n2 (M,) indices before and after.
    Returns d_r, E_r, d_t, E_t and the TIR mask; E_r and E_t are amplitudes (not normalised), d_t is
    undefined where TIR.
    """
    cos_i = -(d*N).sum(-1)
    rs, rp, ts, tp, cos_t = fresnel_amplitudes(cos_i, n1, n2)
    eta = (n1/n2)[:, None]
    tir = np.imag(cos_t) > 0
    d_r = reflect(d, N)
    d_t = eta*d+(eta*cos_i[:, None]-np.real(cos_t)[:, None])*N
    d_t /= np.linalg.norm(d_t, axis=-1, keepdims=True)
    sv = np.cross(d, N); ns = np.linalg.norm(sv, axis=-1, keepdims=True)
    sv = np.where(ns > 1e-12, sv/np.where(ns > 1e-12, ns, 1), basis(d)[0])
    Es = (E*sv).sum(-1); Ep = (E*np.cross(d, sv)).sum(-1)
    E_r = (rs*Es)[:, None]*sv+(rp*Ep)[:, None]*np.cross(d_r, sv)
    E_t = (ts*Es)[:, None]*sv+(tp*Ep)[:, None]*np.cross(d_t, sv)
    return d_r, E_r, d_t, E_t, tir


def interface_split(d, E, N, n1, n2):
    """Split rays at a plane interface into reflected and transmitted branches.

    Same arguments as interface_fields; E = None means unpolarised light, R = (Rs + Rp)/2.
    Returns (d_r, E_r, R), (d_t, E_t, T) and the TIR mask, with unit fields (None if E is None)
    and R + T = 1.
    """
    if E is None:
        cos_i = -(d*N).sum(-1)
        Rs, Rp, _, _ = fresnel_powers(cos_i, n1, n2)
        eta = (n1/n2)[:, None]; k = 1-eta**2*(1-cos_i[:, None]**2)
        tir = k[:, 0] <= 0
        d_t = eta*d+(eta*cos_i[:, None]-np.sqrt(np.clip(k, 0, None)))*N
        d_t /= np.linalg.norm(d_t, axis=-1, keepdims=True)
        R = np.where(tir, 1.0, 0.5*(Rs+Rp))
        return (reflect(d, N), None, R), (d_t, None, 1-R), tir
    d_r, E_r, d_t, E_t, tir = interface_fields(d, E, N, n1, n2)
    R = np.where(tir, 1.0, (np.abs(E_r)**2).sum(-1)); T = 1-R
    return (d_r, _unit(E_r), R), (d_t, _unit(E_t), T), tir


def _unit(E):
    """Complex field vectors scaled to unit norm (zero vectors left unchanged)."""
    a = np.sqrt((np.abs(E)**2).sum(-1, keepdims=True))
    return E/np.where(a > 0, a, 1)


# Axis-aligned boxes

def box_exit(p, d, lo, hi):
    """Distance from points p inside the box [lo, hi] along d to its boundary, and the exit face.

    Faces are numbered 2k (low side) and 2k+1 (high side) for axis k = 0, 1, 2.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(d > 0, (np.asarray(hi)-p)/d, np.where(d < 0, (np.asarray(lo)-p)/d, np.inf))
    k = np.argmin(t, axis=-1)
    s = np.take_along_axis(t, k[:, None], -1)[:, 0]
    return s, 2*k+(np.take_along_axis(d, k[:, None], -1)[:, 0] > 0)


def box_entry(p, d, lo, hi, eps=1e-7):
    """Distance from points p outside the box [lo, hi] along d to the entry face (inf if missed), and that face."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo-p)/d; t2 = (hi-p)/d
    par = d == 0; within = (p > lo)&(p < hi)
    tmin = np.where(par, np.where(within, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(par, np.where(within, np.inf, -np.inf), np.maximum(t1, t2))
    k = np.argmax(tmin, axis=-1)
    tn = np.take_along_axis(tmin, k[:, None], -1)[:, 0]; tf = tmax.min(-1)
    ok = (tn < tf)&(tn > eps)
    return np.where(ok, tn, np.inf), 2*k+(np.take_along_axis(d, k[:, None], -1)[:, 0] < 0)


# Weight-splitting tracer

@dataclass(frozen=True)
class Slab:
    """Box [lo, hi] (mm) of a medium: Fresnel interfaces on its z faces, absorbing side faces."""
    lo: tuple
    hi: tuple
    medium: object


@dataclass(frozen=True)
class SphereMirror:
    """Spherical mirror cap: centre C, radius R, unit axis from the vertex to C, clear-aperture radius, reflectivity 1."""
    C: tuple
    R: float
    axis: tuple
    clear_radius: float


@dataclass(frozen=True)
class SensorPlane:
    """Perfectly absorbing plane z = z; a hit inside |x - cx|, |y - cy| < half counts as detected."""
    z: float
    half: float
    cx: float = 0.0
    cy: float = 0.0


@dataclass(frozen=True)
class Scene:
    """Ambient medium with non-overlapping slabs, an optional spherical mirror and a sensor plane."""
    ambient: object
    slabs: tuple
    mirror: SphereMirror | None
    sensor: SensorPlane


REASONS = ('detected', 'mirror aperture', 'tile sides', 'outside sensor', 'interaction cap', 'escaped')
CLASSES = ('direct', 'bottom', 'top', 'TIR', 'multiple')
_EV_NONE, _EV_SENSOR, _EV_MIRROR, _EV_EXIT, _EV_ENTRY = -1, 0, 1, 2, 3


def path_class(nb, nt, ntir):
    """Path class index into CLASSES from the counts of partial reflections on bottom and top faces and of TIRs."""
    c = np.full(np.shape(nb), 4, np.int8)
    c[(nb == 0)&(nt == 0)&(ntir == 0)] = 0
    c[(nb == 1)&(nt == 0)&(ntir == 0)] = 1
    c[(nb == 0)&(nt == 1)&(ntir == 0)] = 2
    c[(nb == 0)&(nt == 0)&(ntir > 0)] = 3
    return c


def _take(st, m):
    return {k: (v[m] if v is not None else None) for k, v in st.items()}


def _cat(parts):
    return {k: (np.concatenate([q[k] for q in parts]) if parts[0][k] is not None else None) for k in parts[0]}


def _region_index(scene, reg, lam):
    """Refractive index of each ray's region (-1: ambient, j: slab j) at its wavelength."""
    n = np.asarray(scene.ambient.n(lam), float).copy()
    for j, sl in enumerate(scene.slabs):
        m = reg == j
        if m.any():
            n[m] = sl.medium.n(lam[m])
    return n


def _region_at(scene, q):
    """Region index (-1 ambient, j slab) containing the points q."""
    reg = np.full(len(q), -1, np.int16)
    for j, sl in enumerate(scene.slabs):
        reg[np.all((q > np.asarray(sl.lo))&(q < np.asarray(sl.hi)), axis=-1)] = j
    return reg


def trace(scene, p, d, lam, region, E=None, max_reflections=2, max_interactions=64, eps=1e-7):
    """Trace photons through the scene, splitting the weight at every Fresnel interface.

    p, d (M,3) start points and unit directions, lam (M,) wavelengths [nm], region (M,) start
    region (-1 ambient, j slab j), E (M,3) unit polarisation vectors or None for unpolarised light.
    At a slab z face a branch splits into a transmitted branch (weight x T) and a reflected one
    (weight x R); under TIR only the reflected one exists. Each branch ends on the sensor plane
    (detected inside the sensor, else lost), on the mirror sphere outside its clear aperture, on a
    slab side face, or when nothing lies ahead (escaped). The interaction cap drops a reflected
    branch that would exceed max_reflections partial reflections, and stops a branch that would
    exceed max_interactions surface interactions (mirror reflections, slab-face crossings and
    reflections). Returns a dict of terminal branches: photon index i, reason (index into REASONS),
    weight w (initial weight 1), end point p, direction d, path class cls (index into CLASSES) and
    interaction count nint.
    """
    M = len(p); zi = np.zeros(M, np.int16)
    st = dict(i=np.arange(M), p=np.asarray(p, float), d=np.asarray(d, float), lam=np.asarray(lam, float),
              reg=np.asarray(region, np.int16), w=np.ones(M), nint=zi, nb=zi.copy(), nt=zi.copy(), ntir=zi.copy(),
              E=None if E is None else np.asarray(E, complex))
    ends = []
    sen = scene.sensor; mir = scene.mirror

    def finish(sub, reason):
        if len(sub['i']):
            ends.append(dict(i=sub['i'], reason=np.full(len(sub['i']), reason, np.int8), w=sub['w'], p=sub['p'],
                             d=sub['d'], cls=path_class(sub['nb'], sub['nt'], sub['ntir']), nint=sub['nint']))

    while len(st['i']):
        P, D, reg = st['p'], st['d'], st['reg']; n = len(P)
        best = np.full(n, np.inf); ev = np.full(n, _EV_NONE, np.int8)
        face = np.full(n, -1, np.int8); slab = np.full(n, -1, np.int16)
        amb = reg < 0
        for j, sl in enumerate(scene.slabs):
            m = np.flatnonzero(reg == j)
            if len(m):
                s, f = box_exit(P[m], D[m], sl.lo, sl.hi)
                u = s < best[m]; mu = m[u]
                best[mu] = s[u]; ev[mu] = _EV_EXIT; face[mu] = f[u]; slab[mu] = j
            m = np.flatnonzero(amb)
            if len(m):
                s, f = box_entry(P[m], D[m], sl.lo, sl.hi, eps)
                u = s < best[m]; mu = m[u]
                best[mu] = s[u]; ev[mu] = _EV_ENTRY; face[mu] = f[u]; slab[mu] = j
        if mir is not None:
            m = np.flatnonzero(amb)
            if len(m):
                q = P[m]-np.asarray(mir.C); b = (q*D[m]).sum(-1); c = (q*q).sum(-1)-mir.R**2; disc = b*b-c
                r = np.sqrt(np.clip(disc, 0, None)); s1 = -b-r; s2 = -b+r
                s = np.where(s1 > eps, s1, np.where(s2 > eps, s2, np.inf)); s[disc <= 0] = np.inf
                u = s < best[m]; mu = m[u]
                best[mu] = s[u]; ev[mu] = _EV_MIRROR
        m = np.flatnonzero(amb&(D[:, 2] != 0))
        if len(m):
            s = (sen.z-P[m, 2])/D[m, 2]; s[s <= eps] = np.inf
            u = s < best[m]; mu = m[u]
            best[mu] = s[u]; ev[mu] = _EV_SENSOR
        hit = P+np.where(np.isfinite(best), best, 0)[:, None]*D
        st['p'] = hit
        finish(_take(st, ev == _EV_NONE), 5)
        m = ev == _EV_SENSOR
        inside = (np.abs(hit[:, 0]-sen.cx) < sen.half)&(np.abs(hit[:, 1]-sen.cy) < sen.half)
        finish(_take(st, m&inside), 0); finish(_take(st, m&~inside), 3)
        side = ((ev == _EV_EXIT)|(ev == _EV_ENTRY))&(face < 4)
        finish(_take(st, side), 2)
        mm = ev == _EV_MIRROR
        if mir is not None and mm.any():
            a = np.asarray(mir.axis, float); q = hit-np.asarray(mir.C)
            ax = (q*a).sum(-1); rad = np.linalg.norm(q-ax[:, None]*a, axis=-1)
            onmir = mm&(ax < 0)&(rad < mir.clear_radius)
            finish(_take(st, mm&~onmir), 1)
            mm = onmir
        zf = ((ev == _EV_EXIT)|(ev == _EV_ENTRY))&(face >= 4)
        capped = (mm|zf)&(st['nint'] >= max_interactions)
        finish(_take(st, capped), 4)
        mm &= ~capped; zf &= ~capped
        nxt = []
        if mm.any():
            r = _take(st, mm); nrm = (r['p']-np.asarray(mir.C))/mir.R
            r['d'] = reflect(r['d'], nrm)
            if r['E'] is not None:
                r['E'] = mirror_field(r['E'], nrm)
            r['nint'] = r['nint']+1
            nxt.append(r)
        if zf.any():
            r = _take(st, zf); f = face[zf]; j = slab[zf]; e = ev[zf]
            zface = np.array([sl.lo[2] for sl in scene.slabs]+[0.0])[j]
            zface = np.where(f == 5, np.array([sl.hi[2] for sl in scene.slabs]+[0.0])[j], zface)
            r['p'][:, 2] = zface
            beyond = r['p']+np.sign(r['d'][:, 2])[:, None]*np.array([0, 0, 1e-6])
            other = np.where(e == _EV_ENTRY, j, _region_at(scene, beyond)).astype(np.int16)
            n1 = _region_index(scene, r['reg'], r['lam']); n2 = _region_index(scene, other, r['lam'])
            N = np.zeros_like(r['d']); N[:, 2] = -np.sign(r['d'][:, 2])
            (d_r, E_r, R), (d_t, E_t, T), tir = interface_split(r['d'], r['E'], N, n1, n2)
            r['nint'] = r['nint']+1
            t = _take(r, ~tir)
            t['d'] = d_t[~tir]; t['w'] = t['w']*T[~tir]; t['reg'] = other[~tir]
            if E_t is not None:
                t['E'] = E_t[~tir]
            rr = dict(r); rr['d'] = d_r; rr['w'] = r['w']*R
            if E_r is not None:
                rr['E'] = E_r
            rr['ntir'] = r['ntir']+tir
            rr['nb'] = r['nb']+(~tir&(f == 4)); rr['nt'] = r['nt']+(~tir&(f == 5))
            drop = ~tir&(rr['nb']+rr['nt'] > max_reflections)
            finish(_take(rr, drop), 4)
            nxt += [t, _take(rr, ~drop)]
        st = _cat(nxt) if nxt else _take(st, np.zeros(n, bool))
    return _cat(ends) if ends else None
