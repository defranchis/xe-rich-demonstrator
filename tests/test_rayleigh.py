"""Rayleigh scattering: mean free path, dipole angular distribution with polarisation, tracer weights and paths."""
import numpy as np
import pytest

from xerich import sim, tracer
from xerich.cherenkov import PhotonSpectrum
from xerich.constants import AEROGEL_N_AIR, MIRROR_D, MIRROR_R, SENSOR_HALF_2IN
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor
from xerich.media import ConstantIndex, Medium, Rayleigh, xenon

N = 400000


def test_mean_free_path():
    r = Rayleigh()
    assert r.length(400.0) == pytest.approx(51.2, rel=1e-12)            # (0.4 um)^4/0.005 um^4/cm = 5.12 cm
    assert np.allclose(r.length(np.array([300.0, 600.0])), [16.2, 259.2], rtol=1e-12)
    assert Rayleigh(A=0.968).face_transmission == pytest.approx(np.sqrt(0.968))


def test_unpolarised_angular_distribution():
    """1 + cos^2: <cos> = 0, <cos^2> = 2/5."""
    d = np.tile([0, 0, 1.0], (N, 1))
    k, E = tracer.rayleigh_scatter(np.random.default_rng(1), d)
    c = k[:, 2]
    assert E is None and np.allclose(np.linalg.norm(k, axis=1), 1)
    assert abs(c.mean()) < 4e-3 and c.var() == pytest.approx(0.4, abs=3e-3)


def test_polarised_angular_distribution_and_field():
    """Field along x: dP/dOmega ~ 1 - k_x^2, so <k_x^2> = 1/5 and <k_y^2> = <k_z^2> = 2/5; new field transverse, unit."""
    d = np.tile([0, 0, 1.0], (N, 1)); E = np.tile([1.0+0j, 0, 0], (N, 1))
    k, E2 = tracer.rayleigh_scatter(np.random.default_rng(2), d, E)
    m2 = (k**2).mean(0)
    assert m2 == pytest.approx([0.2, 0.4, 0.4], abs=3e-3)
    assert np.allclose(np.abs((k*E2).sum(1)), 0, atol=1e-12) and np.allclose((np.abs(E2)**2).sum(1), 1)
    assert np.allclose(E2, (E-k[:, :1]*k)/np.linalg.norm(E-k[:, :1]*k, axis=1, keepdims=True))


def test_polarisation_average_is_unpolarised():
    """Random linear fields transverse to d: the 1 + cos^2 law with respect to d."""
    rng = np.random.default_rng(3); a = rng.uniform(0, 2*np.pi, N)
    d = np.tile([0, 0, 1.0], (N, 1)); E = np.stack([np.cos(a), np.sin(a), 0*a], 1).astype(complex)
    k, _ = tracer.rayleigh_scatter(rng, d, E)
    assert abs(k[:, 2].mean()) < 4e-3 and (k[:, 2]**2).mean() == pytest.approx(0.4, abs=3e-3)


def matched_slab(C, A=1.0, t=10.0):
    """Index-matched slab (no Fresnel) of thickness t between z = 0 and t in a medium n = 1.03, sensor at z = t + 1."""
    m = Medium('n', ConstantIndex(AEROGEL_N_AIR))
    return tracer.Scene(m, (tracer.Slab((-1e3, -1e3, 0.0), (1e3, 1e3, t), m, Rayleigh(C, A)),), None,
                        tracer.SensorPlane(t+1.0, 1e3))


def test_unscattered_fraction_and_surface_factor():
    """Normal incidence: exp(-t/Lambda) unscattered; A < 1 removes 1 - A of a straight pass, weights add up to 1."""
    lam = 400.0; n = 200000
    p = np.zeros((n, 3)); p[:, 2] = -1.0; d = np.tile([0, 0, 1.0], (n, 1)); E = np.tile([1.0+0j, 0, 0], (n, 1))
    for A in (1.0, 0.9):
        e = tracer.trace(matched_slab(0.005, A), p, d, np.full(n, lam), np.full(n, -1), E, rng=np.random.default_rng(4))
        assert np.allclose(np.bincount(e['i'], e['w'], minlength=n), 1, rtol=0, atol=1e-12)
        direct = (e['reason'] == 0)&(e['cls'] == 0)
        assert np.all(e['nsc'][e['cls'] == 5] > 0) and np.all(e['nsc'][e['cls'] != 5] == 0)
        assert direct.sum()/n == pytest.approx(np.exp(-10/Rayleigh().length(lam)), abs=3e-3)
        assert np.allclose(e['w'][direct], A)
        if A < 1:
            assert e['w'][e['reason'] == 6].sum()/n == pytest.approx(1-A, abs=3e-3)
    e = tracer.trace(matched_slab(1e-12, 0.9), p[:100], d[:100], np.full(100, lam), np.full(100, -1), E[:100],
                     rng=np.random.default_rng(5))
    assert np.allclose(e['w'][(e['reason'] == 0)&(e['w'] > 0)], 0.9)          # zero-weight branches: reflections at R = 0


def test_trace_needs_rng_with_scattering():
    with pytest.raises(ValueError):
        tracer.trace(matched_slab(0.005), np.zeros((1, 3)), np.array([[0, 0, 1.0]]), np.array([400.0]), np.array([-1]))


@pytest.mark.parametrize('polarised', [True, False])
def test_onaxis_weights_and_paths_with_scattering(polarised):
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(3), Aerogel(AEROGEL_N_AIR),
                       tile_scatter=Rayleigh(0.05, 0.95))
    sp = {'gas': PhotonSpectrum(lay.gas.index, 1.3, 4.7), 'tile': PhotonSpectrum(lay.tile_index.index, 1.3, 4.7)}
    ph, e = sim.onaxis_muon(lay, np.random.default_rng(6), sp, 2000, 2000, polarised=polarised, record=True)
    assert np.allclose(np.bincount(e['i'], e['w'], minlength=len(ph['lam'])), 1, rtol=0, atol=1e-12)
    assert (e['cls'] == 5).sum() > 100 and (e['nsc'] > 1).sum() > 10
    p0 = np.array([0, 0, lay.z_sensor])
    for k in np.flatnonzero(e['reason'] != 5)[::97]:
        path = e['path'][k]
        assert np.allclose(path[-1], e['p'][k]) and np.allclose(path[0, :2], 0) and path[0, 2] <= p0[2]
        assert len(path) >= 2+e['nsc'][k]


def test_record_does_not_change_the_trace():
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(3), Aerogel(AEROGEL_N_AIR), tile_scatter=Rayleigh())
    sp = {'gas': PhotonSpectrum(lay.gas.index, 1.3, 4.7), 'tile': PhotonSpectrum(lay.tile_index.index, 1.3, 4.7)}
    _, a = sim.onaxis_muon(lay, np.random.default_rng(8), sp, 1000, 1000)
    _, b = sim.onaxis_muon(lay, np.random.default_rng(8), sp, 1000, 1000, record=True)
    for k in a:
        assert np.array_equal(a[k], b[k])
