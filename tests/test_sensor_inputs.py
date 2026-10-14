"""Baseline-sensor inputs as options: S13361 PDE, mirror reflectivity, board pixel map, pore-gas temperature, preset."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from xerich import cli, cosmic, muons, sim, tracer
from xerich.analysis import pad_hits
from xerich.cherenkov import (PDE_S13361, PDE_S13361_NM, chromatic_rms, ftqe_angle_yield, pde_s13361, pore_signal, qe_arc,
                              qe_band, response)
from xerich.constants import HC, MIRROR_D, MIRROR_R, MUON_THMAX, PAD_KMAX, R_GAS, SENSOR_HALF_2IN, T_PORE
from xerich.layout import Aerogel, Board, Mirror, Phase1Layout, Sensor, mirror_reflectivity
from xerich.media import B_XE_293, B_XE_SLOPE, B_XE_TREF, N_AIR, aerogel_medium, molar_density, n_xe, xenon


def xe3(lam):
    return n_xe(lam, 3.0)


def test_s13361_table_and_interpolation():
    assert len(PDE_S13361_NM) == len(PDE_S13361) == 117 and PDE_S13361_NM[-2:] == (895, 898)
    assert pde_s13361(HC/450.0) == pytest.approx(0.3995, abs=1e-12)          # table node at 450 nm
    assert pde_s13361(HC/452.5) == pytest.approx(0.5*(0.3995+0.4016), abs=1e-12)
    assert pde_s13361(HC/319.0) == 0 and pde_s13361(HC/899.0) == 0
    assert response('arc_qe') is qe_arc and response('s13361') is pde_s13361
    with pytest.raises(ValueError):
        response('unknown')
    lo, hi = qe_band()
    assert HC/hi < 320 and HC/lo > 898                     # the S13361 band lies inside the sampled photon band


def test_s13361_yield_angle_and_chromatic_rms():
    # Xe 3 bar, beta = 1, Boerzsoenyi index (independent re-implementation: 0.480-0.490, 63.21-63.31 mrad, 0.64-0.68 mrad)
    th_q, y_q = ftqe_angle_yield(xe3, 1.0)
    th_p, y_p = ftqe_angle_yield(xe3, 1.0, weight=pde_s13361)
    assert 0.480 < y_p[0]/y_q[0] < 0.490
    assert 63.21e-3 < th_p[0] < 63.31e-3
    assert 0.64e-3 < chromatic_rms(xe3, weight=pde_s13361) < 0.68e-3


def test_response_option_default_and_trace_batch():
    p = cli.detector_inputs(argparse.ArgumentParser())
    assert cli.inputs(p.parse_args([]))['response'] == 's13361'
    assert cli.inputs(p.parse_args(['--response', 'arc_qe']))['response'] == 'arc_qe'
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(3.0)); M = 50
    P0 = np.tile([0.0, 0.0, lay.z_sensor], (M, 1)); U = np.tile([0.0, 0.0, -1.0], (M, 1)); b = np.ones(M)
    a = cosmic.trace_batch(lay, np.random.default_rng(4), P0, U, b, 72, ('gas',))
    d = cosmic.trace_batch(lay, np.random.default_rng(4), P0, U, b, 72, ('gas',), response=qe_arc)
    s = cosmic.trace_batch(lay, np.random.default_rng(4), P0, U, b, 72, ('gas',), response=pde_s13361)
    assert all(np.array_equal(a[k], d[k]) for k in a)                         # the default is the ARC QE, bit for bit
    assert np.array_equal(a['det_ft'], s['det_ft'])                           # same photons, only the weight changes
    r = s['det_qe'][:, 0].sum()/a['det_qe'][:, 0].sum()
    assert 0.46 < r < 0.51


def spectrum_mean_R(weight, P=3.0):
    """FT x weight average of the nominal mirror reflectivity, Xe at P bar, beta = 1."""
    R = mirror_reflectivity('nominal'); idx = lambda lam: n_xe(lam, P)
    return ftqe_angle_yield(idx, 1.0, weight=lambda E: weight(E)*R(HC/E))[1][0]/ftqe_angle_yield(idx, 1.0, weight=weight)[1][0]


def test_mirror_reflectivity_options():
    assert mirror_reflectivity() is None and mirror_reflectivity(1) is None and mirror_reflectivity('1') is None
    R = mirror_reflectivity('nominal')
    assert np.allclose(R(np.array([250.0, 450.0, 575.0, 800.0, 830.0, 900.0, 954.0])), [0.89, 0.89, 0.88, 0.82, 0.82, 0.835, 0.85])
    assert np.all(mirror_reflectivity('0.886')(np.array([300.0, 700.0])) == 0.886)
    with pytest.raises(ValueError):
        mirror_reflectivity(1.2)
    assert Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0)).scene().mirror.reflectivity is None
    # spectrum-averaged reflectivity of the gas ring (independent value: 0.886 with the ARC QE, 0.884 with the S13361 PDE)
    assert spectrum_mean_R(qe_arc) == pytest.approx(0.886, abs=0.001)
    assert spectrum_mean_R(pde_s13361) == pytest.approx(0.884, abs=0.001)


def test_mirror_reflectivity_in_tracer():
    rng = np.random.default_rng(3); n = 4000
    lam = rng.uniform(270, 950, n); phi = rng.uniform(0, 2*np.pi, n); th = np.arccos(1/n_xe(lam, 3.0))
    d = np.stack([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), -np.cos(th)], 1); p = np.tile([0.0, 0.0, 100.0], (n, 1))
    tiled = Phase1Layout(Mirror(MIRROR_R, MIRROR_D, reflectivity='nominal'), Sensor(25.4), xenon(3.0), Aerogel(1.03))
    et = tracer.trace(tiled.scene(), p, d, lam, np.full(n, -1))                  # tile ghosts: several mirror reflections
    assert np.allclose(np.bincount(et['i'], et['w'], n), 1, atol=1e-12)          # weights still sum to 1 per photon
    assert np.any(et['wr'] < 0.82**2)
    lay0 = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0))
    lay1 = Phase1Layout(Mirror(MIRROR_R, MIRROR_D, reflectivity='nominal'), Sensor(25.4), xenon(3.0))
    e0 = tracer.trace(lay0.scene(), p, d, lam, np.full(n, -1))
    e1 = tracer.trace(lay1.scene(), p, d, lam, np.full(n, -1))
    assert 'wr' not in e0 and tracer.REASONS_ALL[:-1] == tracer.REASONS
    assert np.allclose(np.bincount(e1['i'], e1['w'], n), 1, atol=1e-12)
    lost = e1['reason'] == len(tracer.REASONS)
    assert tracer.REASONS_ALL[len(tracer.REASONS)] == 'mirror absorption'
    R = mirror_reflectivity('nominal')(lam)
    absorbed = np.bincount(e1['i'][lost], e1['w'][lost], n)
    assert np.allclose(absorbed, 1-R, atol=1e-12)                                 # one reflection each: 1 - R absorbed
    det0 = np.bincount(e0['i'][e0['reason'] == 0], e0['w'][e0['reason'] == 0], n)
    m1 = e1['reason'] == 0
    assert np.allclose(np.bincount(e1['i'][m1], e1['w'][m1], n), det0*R, atol=1e-12)
    assert np.allclose(np.bincount(e1['i'][m1], (e1['w']/e1['wr'])[m1], n), det0, atol=1e-12)


def test_mirror_reflectivity_scales_onaxis_ring():
    """On axis the detected FT x response of the gas ring scales by the spectrum-averaged R within MC error."""
    M = 400; base = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0))
    P0 = np.tile([0.0, 0.0, base.z_sensor], (M, 1)); U = np.tile([0.0, 0.0, -1.0], (M, 1)); b = np.ones(M)
    for w in (qe_arc, pde_s13361):
        a = cosmic.trace_batch(base, np.random.default_rng(5), P0, U, b, 72, ('gas',), response=w)
        lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D, reflectivity='nominal'), Sensor(25.4), xenon(3.0))
        r = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), response=w)
        ratio = r['det_qe'].sum()/a['det_qe'].sum()
        assert ratio == pytest.approx(spectrum_mean_R(w), abs=0.002)              # MC error of the ratio < 0.001
        assert np.allclose(r['det_ft'], a['det_ft'], rtol=1e-12) and np.array_equal(r['arc'], a['arc'])


def test_board_geometry():
    b = Board()
    assert b.fill(25.6) == pytest.approx(0.879, abs=5e-4) and b.fill() == pytest.approx(0.865, abs=5e-4)
    assert b.half == 25.8 and b.npads == 256 and b.pad_lo[8] == pytest.approx(0.2) and b.pad_lo[-1]+b.pad == pytest.approx(25.6)
    x = np.random.default_rng(1).uniform(-25.8, 25.8, (2, 400000))
    assert (b.pad_index(*x) >= 0).mean() == pytest.approx(b.fill(), abs=0.002)
    assert list(b.pad_index(np.array([0.1, 0.3, 3.15, 3.25, 25.5, 25.7]), np.full(6, 1.0))) == [-1, 136, 136, -1, 248, -1]
    assert Sensor(25.4).board is None and Sensor(25.4).plane_half == 25.4
    assert Sensor(25.4, model='board').plane_half == 25.8
    with pytest.raises(ValueError):
        Sensor(25.4, model='hexagonal')


def test_pad_hits_poisson_binomial():
    rng = np.random.default_rng(2); group = rng.integers(0, 3, 60); pad = rng.integers(0, 12, 60); mu = rng.uniform(0, 0.6, 60)
    mean, pge = pad_hits(group, pad, mu, 4)
    assert mean[3] == 0 and pge[3, 0] == 1 and pge[3, 1] == 0
    for g in range(3):
        m = group == g; mu_pad = np.bincount(pad[m], mu[m], 12)
        assert mean[g] == pytest.approx(np.sum(1-np.exp(-mu_pad)), rel=1e-12)
        n = (rng.poisson(mu_pad, (200000, 12)) > 0).sum(1)                       # Monte Carlo of the fired pads
        assert np.allclose([np.mean(n >= k) for k in range(8)], pge[g, :8], atol=0.004)
        assert np.sum(pge[g, 1:]) == pytest.approx(mean[g], rel=1e-9)            # E[N] = sum_k P(N >= k)


@pytest.fixture(scope='module')
def thin_ring_counts():
    """Thin ring of r = 13 mm on the board, 20000 centres uniform within +-6 mm (seed 7), 4000 points uniform in azimuth:
    (points per pad and centre (20000, npads), pad index and centre of the points of the first 50 centres)."""
    b = Board(); n = 4000; phi = 2*np.pi*np.arange(n)/n; c = np.random.default_rng(7).uniform(-6, 6, (20000, 2))
    cnt = np.zeros((len(c), b.npads))
    for i0 in range(0, len(c), 2000):
        cc = c[i0:i0+2000]; pad = b.pad_index(cc[:, :1]+13*np.cos(phi), cc[:, 1:]+13*np.sin(phi)); m = pad >= 0
        g = np.broadcast_to(np.arange(len(cc))[:, None], pad.shape)
        cnt[i0:i0+2000] = np.bincount(g[m]*b.npads+pad[m], minlength=len(cc)*b.npads).reshape(len(cc), -1)
        if i0 == 0:
            first = (pad[:50][m[:50]], g[:50][m[:50]])
    return cnt, first


@pytest.mark.parametrize('n_sensor, fired', [(5, 4.006), (10, 7.380), (20, 12.635), (40, 19.159)])
def test_thin_ring_fired_pads(thin_ring_counts, n_sensor, fired):
    """n_sensor photons on the board footprint per ring: mean fired pads = sum_pads (1 - exp(-mu_pad)) over the 20000
    centres; pad_hits gives the same per ring."""
    cnt, (pad, g) = thin_ring_counts; mu = cnt*n_sensor/4000
    assert np.mean((1-np.exp(-mu)).sum(1)) == pytest.approx(fired, rel=2e-4)  # a 1 % error is 50 times the tolerance
    assert np.allclose(pad_hits(g, pad, np.full(len(pad), n_sensor/4000), 50)[0], (1-np.exp(-mu[:50])).sum(1), rtol=1e-12)


def test_board_in_trace_batch():
    sq = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0))
    bd = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4, model='board'), xenon(3.0)); M = 300
    rng = np.random.default_rng(8); s = rng.uniform(-0.03, 0.03, (M, 2)); f = sq.z_sensor   # ring centres f s within +-6 mm
    U = np.c_[s, -np.ones(M)]/np.sqrt(1+(s*s).sum(1))[:, None]; P0 = np.c_[-f*s, np.full(M, f)]; b = np.ones(M)
    a = cosmic.trace_batch(sq, np.random.default_rng(5), P0, U, b, 72, ('gas',))
    r = cosmic.trace_batch(bd, np.random.default_rng(5), P0, U, b, 72, ('gas',))
    assert not set(cosmic.BOARD_KEYS) & set(a) and 'pads_pge' not in a
    assert np.allclose(r['det_qe_plane'], a['det_qe_clean'])                  # contained rings: same photons on the footprint
    f = r['det_qe'][:, 0].sum()/r['det_qe_plane'][:, 0].sum()
    assert f == pytest.approx(Board().fill(25.6), abs=0.03)                   # active fraction along the rings
    assert np.all(r['pads'] <= r['det_qe_clean']+1e-12) and np.all(r['pads_all'] >= r['pads'])
    assert np.allclose(r['pads_pge'][:, 0, 1:].sum(1), r['pads'][:, 0])
    # ring hits leave out the pad the muon crosses (here at its entry point P0 on the sensor plane)
    k = cosmic.trace_batch(bd, np.random.default_rng(5), P0, U, b, 72, ('gas',), keep_muon_pad=True)
    d = k['pads'][:, 0]-r['pads'][:, 0]; mpad = Board().pad_index(P0[:, 0], P0[:, 1])
    assert np.all(d >= -1e-12) and np.any(d > 0) and np.all(d[mpad < 0] == 0) and np.all(k['pads_all'] >= r['pads_all']-1e-12)
    assert np.array_equal(k['det_qe'], r['det_qe'])
    # photons on the active pads without the muon's pad: <= those with it, equal where the muon misses the pads
    assert np.all(r['det_qe_ring'] <= r['det_qe_clean']+1e-12) and np.allclose(r['det_qe_ring'][mpad < 0], r['det_qe_clean'][mpad < 0], rtol=1e-12)
    assert np.any(r['det_qe_ring'] < r['det_qe_clean']-1e-9) and np.all(r['pads'] <= r['det_qe_ring']+1e-12)


def test_array_temperature_and_pore_gas():
    T = np.array([253.15, 293.15])
    n = n_xe(400.0, 3.0, T)
    assert n.shape == (2,) and n[0] == n_xe(400.0, 3.0, 253.15) and n[1] == n_xe(400.0, 3.0, 293.15)
    T_B0 = B_XE_TREF-B_XE_293/B_XE_SLOPE                                  # B(T) = 0
    assert molar_density(3e5, T_B0) == pytest.approx(3e5/(R_GAS*T_B0), rel=1e-13)
    assert np.allclose(molar_density(3e5, np.array([T_B0, 293.15])), [3e5/(R_GAS*T_B0), molar_density(3e5, 293.15)], rtol=1e-13)
    assert molar_density(3e5, 293.15) == molar_density(3e5, np.array([293.15]))[0]
    assert cli.inputs(cli.detector_inputs(argparse.ArgumentParser()).parse_args([]))['pore_t'] == T_PORE == 293.15
    # pore shift of the aerogel angle, Xe 3 bar at 400 nm, relative to the tile in air (porosity 0.935): 6.32 mrad at 293.15 K
    air_ref = [pore_signal(3.0, T=t)[0] for t in (253.15, 293.15)]
    assert air_ref[1] == pytest.approx(6.32e-3, abs=0.01e-3) and air_ref[0] == pytest.approx(7.53e-3, abs=0.01e-3)


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'studies'/('%s.py' % name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_dispersive_aerogel_and_porosity_options():
    sp = aerogel_medium('single-pole', 1.03, 400.0)
    assert sp.n(400.0) == pytest.approx(1.03, abs=1e-12) and sp.n(300.0) > sp.n(400.0) > sp.n(600.0)
    assert 1.5e-3 < chromatic_rms(sp.index, weight=pde_s13361) < 1.7e-3       # independent value 1.6 +- 0.1 mrad
    cr = load_driver('cosmic_rates')
    p = cli.detector_inputs(argparse.ArgumentParser())
    run = dict(P=3.0, variant='pore-vac', tile=True, mirror=(MIRROR_R, MIRROR_D), sensor=cr.SIPM_HALF, rayleigh=False)
    cr._G['inputs'] = cli.inputs(p.parse_args(['--preset', 'published']))
    lay = cr.layout_of(run)
    assert lay.tile_medium is None and lay.tile.n == 1.03+0.97*(n_xe(400.0, 3.0)-1)
    cr._G['inputs'] = cli.inputs(p.parse_args(['--aerogel-index', 'single-pole', '--porosity-vac', '0.935']))
    lay = cr.layout_of(run); n400 = 1.03+0.935*(n_xe(400.0, 3.0)-1)
    assert lay.tile.n == pytest.approx(n400, rel=1e-15) and lay.tile_index.n(400.0) == pytest.approx(n400, abs=1e-12)
    assert lay.tile_index.n(320.0) > n400


def test_defaults_and_presets():
    p = cli.detector_inputs(argparse.ArgumentParser())
    b = cli.inputs(p.parse_args([]))
    assert b == cli.INPUT_DEFAULTS == cli.BASELINE == cli.preset('baseline-board') == cli.inputs(p.parse_args(['--preset', 'baseline-board']))
    assert b == dict(response='s13361', mirror_r='nominal', sensor_model='board', pore_t=293.15, aerogel_index='single-pole',
                     aerogel_lambda_ref=400.0, porosity_vac=0.935, board_r_spec=0.11, board_r_diff=0.10, prox_rayleigh='per-photon')
    assert cli.inputs(p.parse_args(['--preset', 'published'])) == cli.preset('published') == dict(
        response='arc_qe', mirror_r='1', sensor_model='square', pore_t=293.15, aerogel_index='constant', aerogel_lambda_ref=400.0,
        porosity_vac=0.97, board_r_spec=0.0, board_r_diff=0.0, prox_rayleigh='vertical')
    # the preset minus one input at a time
    assert cli.inputs(p.parse_args(['--preset', 'baseline-board', '--response', 'arc_qe'])) == {**b, 'response': 'arc_qe'}
    assert cli.inputs(p.parse_args(['--preset', 'baseline-board', '--mirror-r', '1']))['mirror_r'] == '1'
    assert cli.inputs(p.parse_args(['--preset', 'baseline-board', '--sensor-model', 'square']))['sensor_model'] == 'square'
    with pytest.raises(ValueError):
        cli.preset('unknown')


def test_proximity_points_and_board_hits():
    rng = np.random.default_rng(9); N = 300
    u, p0, _ = muons.sample_cos3(rng, N, 25.0, 0.0, MUON_THMAX); aero = Aerogel(1.03); ng = n_xe(400.0, 3.0)
    th = np.full(N, np.arccos(1/1.03))
    fa, arc = sim.proximity_fractions(aero, Sensor(25.8), 30.0, ng, u, p0, th)
    fa2, arc2, pts, acc = sim.proximity_fractions(aero, Sensor(25.8), 30.0, ng, u, p0, th, points=True)
    assert np.array_equal(fa, fa2) and np.array_equal(arc, arc2) and np.array_equal(acc.mean((1, 2)), fa)
    assert np.all(np.abs(pts[acc][:, :2]) < 25.8)
    cr = load_driver('cosmic_rates')
    cr._G['inputs'] = cli.preset('baseline-board'); cr._G['spec'] = muons.MuonSpectrum()
    run = dict(P=3.0, variant='plain', gap=30.0, sensor=cr.SIPM_HALF, top_half=cr.TOP_HALF, zmax=float(np.degrees(MUON_THMAX)))
    out = cr.work_prox(run, np.random.default_rng(1), 400)
    assert out['b1_tile_pads_pge'].shape == (400, PAD_KMAX+1) and np.all(out['b1_tile_act'] <= 1)
    lit = out['b1_tile_fd'] > 0.8
    assert 0.8 < out['b1_tile_act'][lit].mean() < 0.95                        # active fraction of the aerogel rings
    assert np.allclose(out['b1_tile_pads_pge'][:, 1:].sum(1), out['b1_tile_pads'])
    assert np.all(out['b1_tile_act_ring'] <= out['b1_tile_act'])
    cr._G['inputs'] = cli.preset('published')
    assert 'b1_tile_pads' not in cr.work_prox(run, np.random.default_rng(1), 50)


def test_board_arcs_are_geometric():
    """Board arcs equal those of a continuous square of the board footprint (same photons); only det_qe sees the pads."""
    sq = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(Board().half), xenon(2.0))
    bd = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4, model='board'), xenon(2.0)); M = 200
    rng = np.random.default_rng(6); s = rng.uniform(-0.12, 0.12, (M, 2)); f = sq.z_sensor     # rings partly off the board
    U = np.c_[s, -np.ones(M)]/np.sqrt(1+(s*s).sum(1))[:, None]; P0 = np.c_[-f*s, np.full(M, f)]; b = np.ones(M)
    a = cosmic.trace_batch(sq, np.random.default_rng(5), P0, U, b, 72, ('gas',))
    r = cosmic.trace_batch(bd, np.random.default_rng(5), P0, U, b, 72, ('gas',))
    for k in ('arc', 'arc_half', 'det_ft', 'out_ft', 'mirror_ft', 'emit_qe'):
        assert np.array_equal(r[k], a[k]), k
    assert 0 < np.mean(r['arc'][:, 0]) < 1 and np.all(r['det_qe'] <= a['det_qe']+1e-12) and r['det_qe'].sum() < 0.9*a['det_qe'].sum()


def test_ring_definitions_are_geometric():
    """A ring counts from its light whatever the response: emit_qe = 0 (light only outside the response band) keeps it."""
    crp = load_driver('cosmic_rates_plots')
    one = np.ones(3); zero = np.zeros(3)
    z = {'r/br_gas_%s' % k: v for k, v in (('fm', one), ('fs', one), ('fd', one), ('arc', one), ('arc_half', one), ('emit_qe', zero),
                                            ('det_qe', zero))}
    A = crp.acceptances(z, 'r', 'c1', 'gas', 'br')
    assert all(A[k].all() for k in ('full', 'r80', 'fd80', 'arc50', 'arc90', 'arc50_any', 'arc90_any')) and not A['any'].any()
    z.update({'r/br_gas_fm': zero, 'r/br_gas_fd': zero, 'r/br_gas_arc': zero, 'r/br_gas_arc_half': zero})
    A = crp.acceptances(z, 'r', 'c1', 'gas', 'br')                           # no light above threshold: no ring
    assert not any(A[k].any() for k in A)
    zt = {'r/b1_tile_%s' % k: v for k, v in (('fm', one), ('fs', one), ('fd', 0.85*one), ('arc', one), ('arc_half', one),
                                              ('emit_qe', zero), ('det_qe', zero))}
    A = crp.acceptances(zt, 'r', 'c23', 'tile', 'b1')
    assert A['r80'].all() and not A['full'].any()


def test_preset_nphot_pore_t_and_mirror_option():
    cr = load_driver('cosmic_rates')
    ns = argparse.Namespace
    assert cr.nphot_of(ns(nphot=None, preset=None)) == cr.NPHOT == cli.NPHOT_DEFAULT == 7200
    assert cr.nphot_of(ns(nphot=None, preset='baseline-board')) == 7200 and cr.nphot_of(ns(nphot=None, preset='published')) == 180
    assert cr.nphot_of(ns(nphot=450, preset='baseline-board')) == 450
    assert cli.preset('baseline-board')['pore_t'] == T_PORE
    p = cli.detector_inputs(argparse.ArgumentParser())
    for spec in ('1', 'nominal', '0.9'):
        assert cli.inputs(p.parse_args(['--mirror-r', spec]))['mirror_r'] == spec
    for spec in ('2', '0', 'shiny'):
        with pytest.raises(SystemExit):
            p.parse_args(['--mirror-r', spec])
    assert '--nphot 7200' in cli.preset_help() and '--porosity-vac 0.935' in cli.preset_help()


def test_chunk_of():
    """Muons per work_mirror chunk: CHUNK up to --nphot 1071 (gas + tile), fewer at the default 7200 (memory bound)."""
    cr = load_driver('cosmic_rates')
    run = {r['name']: r for r in cr.runs(1)}
    for nphot, want in ((180, (2000, 2000, 2000)), (1071, (2000, 2000, 2000)), (1072, (2000, 1998, 2000)), (7200, (1041, 297, 416))):
        cr._G.update(ngas=int(round(0.4*nphot)), ntile=nphot)
        assert tuple(cr.chunk_of(run[k]) for k in ('c1_xe2', 'c5_xe2_plain', 'c5_xe2_pore-vac')) == want
    assert [run[k]['origins'] for k in ('c1_xe2', 'c5_xe2_plain', 'c5_xe2_pore-vac')] == [('gas',), ('gas', 'tile'), ('tile',)]


def test_board_table_states_nphot(tmp_path):
    crp = load_driver('cosmic_rates_plots')
    crp.board_table({}, tmp_path, 7200)
    assert 'cosmic_rates.py --nphot 7200' in (tmp_path/'cosmic_rates_board.md').read_text()


def test_pad_arrays_only_for_consistent_photons():
    """Aerogel light of mirror runs without Rayleigh scattering has a transmission-scaled photon count: no pad arrays."""
    crp = load_driver('cosmic_rates_plots')
    npz = type('Npz', (dict,), {'files': property(list)})                   # dict with the .files of np.load
    z = npz({'r/br_tile_pads': np.ones(2), 'r/br_tile_pads_pge': np.ones((2, 3)), 'r/br_tile_det_qe_plane': np.ones(2),
             'r/br_tile_det_qe_ring': np.ones(2)})
    assert crp.pad_arrays(z, 'r', dict(group='c23', rayleigh=False), 'tile', 'br') is None
    assert crp.pad_arrays(z, 'r', dict(group='c23', rayleigh=True), 'tile', 'br') is not None


def test_pore_signal_porosity():
    ng = n_xe(400.0, 3.0)
    for por in (0.97, 0.935):
        vac = pore_signal(3.0, porosity=por)[1]
        assert vac == pytest.approx(float(np.arccos(1/(1.03+por*(ng-1)))-np.arccos(1/(1.03+por*(N_AIR-1)))), abs=1e-15)
    assert pore_signal(3.0) == pore_signal(3.0, porosity=0.97)
    assert pore_signal(3.0, porosity=0.935)[0] == pore_signal(3.0)[0]       # the air-referenced element has its own porosity


def test_proximity_photons():
    """Configuration-4 photons of both drivers: FT x weight yield over T/c; the constant index has no beta < 1 / n light."""
    b = np.array([1.0, 0.99, 0.975, 0.97]); c = np.array([1.0, 0.8, 1.0, 1.0])
    y = sim.proximity_photons(1.03, b, c)
    assert y[0] == pytest.approx(ftqe_angle_yield(lambda lam: np.full(np.shape(lam), 1.03), 1.0)[1][0]*10.0, rel=1e-6)
    assert y[3] == 0 and y[2] > 0
    assert sim.proximity_photons(1.03, 1.0, 0.8) == pytest.approx(y[0]/0.8, rel=1e-12)
    sp = sim.proximity_photons(1.03, b, c, pde_s13361, 'single-pole')
    assert sp[0] == pytest.approx(ftqe_angle_yield(aerogel_medium('single-pole', 1.03).index, 1.0, pde_s13361)[1][0]*10.0, rel=1e-6)


def test_prox_ring_points_skip_the_muon_pad():
    """Configuration 4: no ring point on the pad the muon crosses (beta = 1, board footprint)."""
    cr = load_driver('cosmic_rates'); rng = np.random.default_rng(12); N = 400; b = Board(); aero = Aerogel(1.03)
    u, p0, _ = muons.sample_cos3(rng, N, 25.0, 0.0, MUON_THMAX); th = np.full(N, np.arccos(1/1.03))
    _, _, pts, acc = sim.proximity_fractions(aero, Sensor(b.half), 30.0, n_xe(400.0, 2.0), u, p0, th, points=True)
    bot = p0+((-aero.thickness-30.0)/u[:, 2])[:, None]*u
    pad, m, ring = cr.ring_points(b, pts, acc, bot)
    assert not np.any(ring&(pad == b.pad_index(bot[:, 0], bot[:, 1])[:, None, None])) and not np.any(ring&~m)


def test_prox_ring_points_remove_the_muon_pad():
    """Synthetic points: those on the pad the muon crosses are removed from the ring, the others kept."""
    cr = load_driver('cosmic_rates'); b = Board()
    xy = np.array([[1.7, 1.7], [1.0, 2.5], [8.1, 1.7], [-5.0, -5.0], [0.1, 1.7]])     # muon pad, muon pad, other pad, other pad, gap
    pts = np.zeros((1, 1, len(xy), 3)); pts[0, 0, :, :2] = xy; acc = np.ones((1, 1, len(xy)), bool)
    pad, m, ring = cr.ring_points(b, pts, acc, np.array([[1.7, 1.7, -40.0]]))
    assert list(m[0, 0]) == [True, True, True, True, False] and list(ring[0, 0]) == [False, False, True, True, False]
