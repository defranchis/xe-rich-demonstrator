"""Fresnel coefficients, polarisation handling and slab transport in xerich.tracer."""
import numpy as np
import pytest

from xerich import tracer
from xerich.media import ConstantIndex, Medium, n_xe

Z = np.array([0, 0, 1.])
N_XE3 = float(n_xe(400, 3))


def rand_dirs(rng, n, thmax, up=True):
    th = rng.uniform(0, thmax, n); ph = rng.uniform(0, 2*np.pi, n)
    return np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), (1 if up else -1)*np.cos(th)], 1), th


def transverse_fields(rng, d):
    """Random complex unit fields transverse to the directions d."""
    e1, e2 = tracer.basis(d)
    a = rng.normal(size=(len(d), 2))+1j*rng.normal(size=(len(d), 2))
    E = a[:, :1]*e1+a[:, 1:]*e2
    return E/np.sqrt((np.abs(E)**2).sum(-1, keepdims=True))


@pytest.mark.parametrize('n1,n2', [(1.0, 1.5), (1.5, 1.0), (N_XE3, 1.03), (1.03, N_XE3)])
def test_normal_incidence(n1, n2):
    Rs, Rp, Ts, Tp = tracer.fresnel_powers(np.array([1.0]), n1, n2)
    R0 = ((n1-n2)/(n1+n2))**2
    assert Rs[0] == pytest.approx(R0, rel=1e-12) and Rp[0] == pytest.approx(R0, rel=1e-12)
    assert Ts[0] == pytest.approx(1-R0, rel=1e-12) and Tp[0] == pytest.approx(1-R0, rel=1e-12)


def test_known_angle_45deg_air_glass():
    i = np.radians(45); t = np.arcsin(np.sin(i)/1.5)
    Rs, Rp, _, _ = tracer.fresnel_powers(np.array([np.cos(i)]), 1.0, 1.5)
    assert Rs[0] == pytest.approx((np.sin(i-t)/np.sin(i+t))**2, rel=1e-12)
    assert Rp[0] == pytest.approx((np.tan(i-t)/np.tan(i+t))**2, rel=1e-12)
    assert Rs[0] == pytest.approx(0.0920, abs=1e-4) and Rp[0] == pytest.approx(0.00847, abs=1e-5)


@pytest.mark.parametrize('n1,n2', [(1.0, 1.5), (1.5, 1.0), (N_XE3, 1.03), (1.03, N_XE3)])
def test_energy_conservation(n1, n2):
    ci = np.linspace(1e-4, 1, 5001)
    Rs, Rp, Ts, Tp = tracer.fresnel_powers(ci, n1, n2)
    assert np.allclose(Rs+Ts, 1, rtol=0, atol=1e-12) and np.allclose(Rp+Tp, 1, rtol=0, atol=1e-12)
    assert np.all((Rs >= 0)&(Rs <= 1+1e-15)&(Rp >= 0)&(Rp <= 1+1e-15))


@pytest.mark.parametrize('n1,n2', [(1.0, 1.5), (1.5, 1.0), (N_XE3, 1.03), (1.03, N_XE3)])
def test_brewster(n1, n2):
    tb = np.arctan(n2/n1)
    Rs, Rp, _, _ = tracer.fresnel_powers(np.array([np.cos(tb)]), n1, n2)
    assert Rp[0] < 1e-25 and Rs[0] > 1e-6


def test_tir_onset_aerogel_to_xe():
    n1, n2 = 1.03, N_XE3
    tc = np.arcsin(n2/n1)
    ci = np.cos(np.array([tc-1e-6, tc+1e-6]))
    Rs, Rp, Ts, Tp = tracer.fresnel_powers(ci, n1, n2)
    assert Rs[0] < 1 and Rp[0] < 1 and Ts[0] > 0 and Tp[0] > 0
    assert Rs[1] == pytest.approx(1, abs=1e-14) and Rp[1] == pytest.approx(1, abs=1e-14) and Ts[1] == 0 and Tp[1] == 0
    d = np.stack([np.sin([tc-1e-6, tc+1e-6]), [0, 0], np.cos([tc-1e-6, tc+1e-6])], 1)
    N = -np.tile(Z, (2, 1)); n1a, n2a = np.full(2, n1), np.full(2, n2)
    for E in (None, tracer.cherenkov_field(-np.tile(Z, (2, 1)), d)):
        (_, _, R), (_, _, T), tir = tracer.interface_split(d, E, N, n1a, n2a)
        assert list(tir) == [False, True] and R[1] == 1 and T[1] == 0 and R[0] < 1


@pytest.mark.parametrize('n1,n2', [(1.0, 1.5), (1.5, 1.0), (N_XE3, 1.03), (1.03, N_XE3)])
def test_interface_fields_boundary_conditions(n1, n2):
    """Tangential E and H = n k x E are continuous, energy is conserved, fields stay transverse."""
    rng = np.random.default_rng(4)
    d, _ = rand_dirs(rng, 4000, np.radians(89)); d[0] = Z
    E = transverse_fields(rng, d)
    N = -np.tile(Z, (len(d), 1)); a1, a2 = np.full(len(d), n1), np.full(len(d), n2)
    d_r, E_r, d_t, E_t, tir = tracer.interface_fields(d, E, N, a1, a2)
    ok = ~tir
    tan = lambda v: np.cross(v, Z)
    assert np.allclose(tan(E+E_r)[ok], tan(E_t)[ok], atol=1e-12)
    assert np.allclose(tan(n1*np.cross(d, E)+n1*np.cross(d_r, E_r))[ok], tan(n2*np.cross(d_t, E_t))[ok], atol=1e-12)
    R = (np.abs(E_r)**2).sum(-1); T = n2*d_t[:, 2]/(n1*d[:, 2])*(np.abs(E_t)**2).sum(-1)
    assert np.allclose((R+T)[ok], 1, atol=1e-12) and np.allclose(R[tir], 1, atol=1e-12)
    (dr, Er, R2), (dt, Et, T2), tir2 = tracer.interface_split(d, E, N, a1, a2)
    assert np.array_equal(tir, tir2) and np.allclose(R2, R, atol=1e-14) and np.allclose(R2+T2, 1, atol=1e-15)
    for v, k, m in ((Er, dr, slice(None)), (Et, dt, ok)):
        assert np.allclose((np.abs(v[m])**2).sum(-1), 1, atol=1e-12)
        assert np.allclose(np.abs((v[m]*k[m]).sum(-1)), 0, atol=1e-12)


def test_split_pure_s_and_p_and_unpolarised():
    rng = np.random.default_rng(5)
    d, th = rand_dirs(rng, 500, np.radians(80)); d = d[th > 1e-3]; th = th[th > 1e-3]
    N = -np.tile(Z, (len(d), 1)); n1, n2 = np.full(len(d), 1.0), np.full(len(d), 1.5)
    s = np.cross(d, N); s /= np.linalg.norm(s, axis=-1, keepdims=True); p = np.cross(d, s)
    Rs, Rp, _, _ = tracer.fresnel_powers(np.cos(th), 1.0, 1.5)
    assert np.allclose(tracer.interface_split(d, s.astype(complex), N, n1, n2)[0][2], Rs, atol=1e-14)
    assert np.allclose(tracer.interface_split(d, p.astype(complex), N, n1, n2)[0][2], Rp, atol=1e-14)
    assert np.allclose(tracer.interface_split(d, None, N, n1, n2)[0][2], 0.5*(Rs+Rp), atol=1e-14)


def test_cherenkov_and_mirror_fields():
    rng = np.random.default_rng(6)
    u = np.tile(-Z, (300, 1)); k, _ = rand_dirs(rng, 300, 0.3, up=False)
    E = tracer.cherenkov_field(u, k)
    assert np.allclose(np.abs((E*k).sum(-1)), 0, atol=1e-14) and np.allclose((np.abs(E)**2).sum(-1), 1)
    assert np.allclose(np.abs((E*np.cross(u, k)).sum(-1)), 0, atol=1e-14)
    n = rng.normal(size=(300, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Em = tracer.mirror_field(E, n); km = tracer.reflect(k, n)
    assert np.allclose((np.abs(Em)**2).sum(-1), 1) and np.allclose(np.abs((Em*km).sum(-1)), 0, atol=1e-14)


def test_snell_round_trip_through_slab():
    """Direct path through a slab: direction restored, lateral shift t (tan th1 - tan th2),
    unpolarised weight = product of the two interface transmissions 1-(Rs+Rp)/2."""
    n_out, n_in, t = 1.002, 1.03, 10.0
    gas = Medium('gas', ConstantIndex(n_out)); aer = Medium('aerogel', ConstantIndex(n_in))
    scene = tracer.Scene(gas, (tracer.Slab((-1e3, -1e3, 0.0), (1e3, 1e3, t), aer),), None, tracer.SensorPlane(20.0, 1e4))
    rng = np.random.default_rng(7)
    d, th1 = rand_dirs(rng, 3000, 1.2)
    p0 = np.zeros((len(d), 3)); p0[:, 2] = -5.0
    e = tracer.trace(scene, p0, d, np.full(len(d), 400.0), np.full(len(d), -1), None, max_reflections=0)
    m = (e['reason'] == 0)&(e['cls'] == 0)
    assert m.sum() == len(d)
    i = e['i'][m]
    assert np.allclose(e['d'][m], d[i], atol=1e-12)
    th2 = np.arcsin(n_out*np.sin(th1[i])/n_in)
    straight = p0[i]+(25.0/d[i, 2])[:, None]*d[i]
    shift = np.hypot(*(straight-e['p'][m])[:, :2].T)
    assert np.allclose(shift, t*(np.tan(th1[i])-np.tan(th2)), rtol=0, atol=1e-9)
    Rs1, Rp1, _, _ = tracer.fresnel_powers(np.cos(th1[i]), n_out, n_in)
    Rs2, Rp2, _, _ = tracer.fresnel_powers(np.cos(th2), n_in, n_out)
    assert np.allclose(e['w'][m], (1-0.5*(Rs1+Rp1))*(1-0.5*(Rs2+Rp2)), rtol=1e-12)
    W = np.bincount(e['i'], e['w'], minlength=len(d))
    assert np.allclose(W, 1, rtol=0, atol=1e-12)
