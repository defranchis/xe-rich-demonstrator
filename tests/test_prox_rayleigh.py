"""Configuration 4: Rayleigh transmission of each aerogel photon along its path in the tile (cosmic_rates.py --prox-rayleigh)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from xerich import cli, muons, sim, tracer
from xerich.cherenkov import pde_s13361, theta_c
from xerich.constants import AEROGEL_T
from xerich.media import aerogel_medium


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'studies'/('%s.py' % name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_vertical_muon_is_proximity_rayleigh():
    """With a constant index a vertical muon gets proximity_rayleigh; the single-pole default index reweights the spectrum
    by 1 - 1/n(lambda)^2 and gives 0.93343 (0.93402 constant)."""
    s = AEROGEL_T/np.cos(theta_c(1.03))*(np.arange(200)+0.5)/200
    assert sim.rayleigh_transmission(s, pde_s13361).mean() == pytest.approx(sim.proximity_rayleigh(1.03, pde_s13361), rel=1e-12)
    idx = aerogel_medium('single-pole', 1.03, 400.0).index
    assert sim.rayleigh_transmission(s, pde_s13361, idx).mean() == pytest.approx(0.93343, abs=1e-5)
    f = sim.rayleigh_transmission_table(pde_s13361)
    assert np.max(np.abs(f(s)-sim.rayleigh_transmission(s, pde_s13361))) < 1e-6 and f(np.inf) == 0


@pytest.mark.parametrize('zenith, value', [(0, 0.9340), (20, 0.9299), (30, 0.9240), (40, 0.9141), (50, 0.8975)])
def test_transmission_vs_zenith(zenith, value):
    """Mean over emission depth and azimuth (unbounded tile, n = 1.03, S13361 PDE); the reference values come from an
    independent integral (4001 energies, 360 azimuths, 200 depths) and agree within 5e-5."""
    f = sim.rayleigh_transmission_table(pde_s13361); th = theta_c(1.03); z = np.radians(zenith)
    phi = np.linspace(0, 2*np.pi, 360, endpoint=False); fr = (np.arange(200)+0.5)/200
    u = np.array([[np.sin(z), 0.0, -np.cos(z)]]); e1, e2 = tracer.basis(u)
    d = np.cos(th)*u[0]+np.sin(th)*(np.cos(phi)[:, None]*e1[0]+np.sin(phi)[:, None]*e2[0])
    assert f(AEROGEL_T*(1-fr)[:, None]/np.abs(d[None, :, 2])).mean() == pytest.approx(value, abs=1e-3)


def test_paths_of_proximity_fractions():
    rng = np.random.default_rng(3); u, p0, _ = muons.sample_cos3(rng, 50, 25.0, 0.0, 1.2)
    from xerich.layout import Aerogel, Sensor
    a = sim.proximity_fractions(Aerogel(1.03), Sensor(25.8), 30.0, 1.0007, u, p0, np.full(50, 0.24), points=True)
    b = sim.proximity_fractions(Aerogel(1.03), Sensor(25.8), 30.0, 1.0007, u, p0, np.full(50, 0.24), points=True, paths=True)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)) and b[4].shape == b[3].shape
    assert np.all(b[4][b[3]] > 0) and np.all(b[4][b[3]] < AEROGEL_T/np.cos(1.2+0.24))      # zenith < 1.2 rad, cone 0.24 rad


def test_gap30_shift():
    """Per-photon vs vertical-muon transmission on the same muons, default inputs, Xe 2 bar, gap 30 mm, >= 80 % rings in the
    telescope: P(>= 10 hits) about -2 %, ring hits about -0.55 %, photons on active pads about -0.57 % (two independent 20000-muon samples: -2.02 / -1.98 % and -0.555 / -0.538 %)."""
    cr = load_driver('cosmic_rates'); sp = muons.MuonSpectrum()
    run = [r for r in cr.runs(1) if r['name'] == 'c4_xe2_g30_plain'][0]; R = {}
    for mode in ('vertical', 'per-photon'):
        cr._G.update(inputs={**cli.INPUT_DEFAULTS, 'prox_rayleigh': mode}, spec=sp)
        o = cr.work_prox(run, np.random.default_rng(20261014), 12000)
        sel = (o['br_tile_fd'] >= .8)&(o['br_tile_th'] > 0)&o['bot']; w = sp.weight_vs_cos2(o['c'])[sel]
        R[mode] = [np.average(o['br_tile_pads_pge'][sel, 10], weights=w), np.average(o['br_tile_pads'][sel], weights=w)]
        if mode == 'per-photon':
            assert np.all(o['br_tile_tr'] <= 1) and np.all(o['br_tile_act_ring'] <= o['br_tile_act']+1e-12) and np.all(o['br_tile_act'] <= 1+1e-12)
    d10, dh = (R['per-photon'][i]/R['vertical'][i]-1 for i in range(2))
    assert -0.03 < d10 < -0.01 and -0.0075 < dh < -0.0035
