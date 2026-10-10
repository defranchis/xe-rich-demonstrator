import numpy as np
import pytest

from xerich import tracer
from xerich.constants import MIRROR_R
from xerich.layout import Mirror
from xerich.sim import aberration

UP = np.array([0, 0, 1.])


def directions(rng, n, thmax):
    """Downward unit vectors with angle to -z uniform in [0, thmax]."""
    th = rng.uniform(0, thmax, n); ph = rng.uniform(0, 2*np.pi, n)
    return np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), -np.cos(th)], 1), th


@pytest.mark.parametrize('n1,n2', [(1.0319, 1.002), (1.002, 1.0319), (1.5, 1.0)])
def test_refract_snell_and_tir(n1, n2):
    rng = np.random.default_rng(0)
    d, th1 = directions(rng, 20000, np.radians(89))
    t, ok = tracer.refract(d, UP, n1/n2)
    tir = n1*np.sin(th1) > n2
    assert np.array_equal(ok, ~tir)
    assert np.allclose(np.linalg.norm(t, axis=-1), 1)
    sin2 = np.linalg.norm(np.cross(t[ok], UP), axis=-1)
    assert np.allclose(n1*np.sin(th1[ok]), n2*sin2, rtol=0, atol=1e-12)
    assert np.all(t[ok, 2] < 0)
    assert np.allclose(np.einsum('ij,ij->i', np.cross(d[ok], UP), t[ok]), 0, atol=1e-12)


def test_reflect_mirror_symmetry():
    rng = np.random.default_rng(1)
    d, _ = directions(rng, 1000, np.radians(80))
    v = tracer.reflect(d, UP)
    assert np.allclose(v[:, :2], d[:, :2]) and np.allclose(v[:, 2], -d[:, 2])


def test_cone_angle_and_norm():
    rng = np.random.default_rng(2)
    u, _ = directions(rng, 50, np.radians(80))
    phi = np.linspace(0, 2*np.pi, 36, endpoint=False)
    c = tracer.cone(u, 0.25, phi)
    assert c.shape == (50, 36, 3)
    assert np.allclose(np.linalg.norm(c, axis=-1), 1)
    assert np.allclose(np.einsum('nmk,nk->nm', c, u), np.cos(0.25))


def test_to_plane_and_advance():
    rng = np.random.default_rng(3)
    d, _ = directions(rng, 100, np.radians(60)); p = rng.uniform(-10, 10, (100, 3))
    assert np.allclose(tracer.to_plane(p, d, -50.0)[:, 2], -50.0)
    assert np.allclose(tracer.advance(p, d, -7.0)[:, 2], p[:, 2]-7.0)


def test_sphere_reflection_exact_focus():
    """A ray parallel to the axis at height h crosses the axis R/(2 cos a) from the centre, sin a = h/R."""
    R = MIRROR_R
    h = np.array([1.0, 10.0, 38.1, 100.0])
    p = np.stack([h, 0*h, 0*h], 1); d = np.tile(UP, (len(h), 1))
    q, hit = tracer.sphere_far(p, d, tracer.ORIGIN, R)
    assert hit.all() and np.allclose(np.linalg.norm(q, axis=-1), R)
    v = tracer.reflect(d, tracer.sphere_normal(q, tracer.ORIGIN, R))
    z_axis = q[:, 2]-q[:, 0]/v[:, 0]*v[:, 2]
    assert np.allclose(z_axis, R/(2*np.cos(np.arcsin(h/R))), rtol=0, atol=1e-9)


def test_on_axis_ring_radius():
    """On-axis track: ring radius f tan(theta_c) up to the small spherical aberration of the exact sphere."""
    mirror, th = Mirror(MIRROR_R), 0.0636
    t, rho = aberration(0, 0.0, np.random.default_rng(1), mirror, 200.0, th)
    r = mirror.f*np.tan(t)
    assert abs(r.mean()/(mirror.f*np.tan(th))-1) < 5e-3
    assert 1e3*t.std() < 0.1
    t75, _ = aberration(75, 0.0, np.random.default_rng(1), mirror, 200.0, th)
    assert t75.std() > 10*t.std()
