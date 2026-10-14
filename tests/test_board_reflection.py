"""Light reflected by the SiPM board (cosmic.board_reflections, trace_batch(refl=...), cosmic_rates.py --board-r-*)."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from xerich import cli, cosmic, muons, tracer
from xerich.cherenkov import pde_s13361, qe_arc
from xerich.constants import BOARD_R_DIFF, BOARD_R_SPEC, MIRROR_D, MIRROR_R
from xerich.layout import Mirror, Phase1Layout, Sensor
from xerich.media import xenon


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'studies'/('%s.py' % name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rings(M=300, seed=8):
    """Board layout (Xe 2 bar, nominal mirror) and muons whose ring centres lie within +-6 mm."""
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D, reflectivity='nominal'), Sensor(25.4, model='board'), xenon(2.0))
    s = np.random.default_rng(seed).uniform(-0.03, 0.03, (M, 2)); f = lay.z_sensor
    U = np.c_[s, -np.ones(M)]/np.sqrt(1+(s*s).sum(1))[:, None]; P0 = np.c_[-f*s, np.full(M, f)]
    return lay, P0, U, np.ones(M)


def test_options_and_preset():
    p = cli.detector_inputs(argparse.ArgumentParser())
    d = cli.inputs(p.parse_args(['--preset', 'published']))
    assert d['board_r_spec'] == d['board_r_diff'] == 0.0
    b = cli.inputs(p.parse_args([]))
    assert (b['board_r_spec'], b['board_r_diff']) == (BOARD_R_SPEC, BOARD_R_DIFF) == (0.11, 0.10)
    assert cli.inputs(p.parse_args(['--preset', 'baseline-board', '--board-r-spec', '0']))['board_r_spec'] == 0.0
    for bad in ('1', '-0.1', 'x'):
        with pytest.raises(SystemExit):
            p.parse_args(['--board-r-diff', bad])


def test_zero_reflectance_and_unchanged_arrays():
    """R = 0 is the default bit for bit; with R > 0 every other output is unchanged and only REFL_KEYS are added."""
    lay, P0, U, b = rings()
    a = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), response=pde_s13361)
    z = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), response=pde_s13361, refl=(0.0, 0.0),
                           refl_rng=np.random.default_rng(9))
    r = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), response=pde_s13361, refl=(BOARD_R_SPEC, BOARD_R_DIFF),
                           refl_rng=np.random.default_rng(9))
    assert set(z) == set(a) and all(np.array_equal(a[k], z[k]) for k in a)
    assert set(r)-set(a) == set(cosmic.REFL_KEYS) and all(np.array_equal(a[k], r[k]) for k in a)
    assert np.all(r['refl_qe'] >= 0) and r['refl_qe'].sum() > 0
    assert np.all(r['pads_refl'] >= r['pads']-1e-12) and np.all(r['pads_refl']-r['pads'] <= r['refl_qe']+1e-12)
    with pytest.raises(ValueError):
        cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), refl=(0.1, 0.0))
    sq = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(2.0))      # square sensor: no reflection modelled
    assert not set(cosmic.REFL_KEYS) & set(cosmic.trace_batch(sq, np.random.default_rng(5), P0, U, b, 72, ('gas',), refl=(0.1, 0.1),
                                                               refl_rng=np.random.default_rng(9)))


def test_driver_streams():
    """cosmic_rates.py: the reflection has its own generator, so the board outputs of the preset do not depend on it."""
    cr = load_driver('cosmic_rates')
    cr._G.update(inputs=cli.preset('baseline-board'), spec=muons.MuonSpectrum(), ngas=72, ntile=180, seed=3)
    run = [r for r in cr.runs(1) if r['name'] == 'c1_xe2'][0]
    o = cr.work_mirror(run, np.random.default_rng(2), 300, np.random.default_rng(7))
    cr._G['inputs'] = {**cr._G['inputs'], 'board_r_spec': 0.0, 'board_r_diff': 0.0}
    o0 = cr.work_mirror(run, np.random.default_rng(2), 300, np.random.default_rng(7))
    assert set(o)-set(o0) == {'%s_gas_%s' % (t, k) for t in ('b1', 'br') for k in cosmic.REFL_KEYS}
    assert all(np.array_equal(o[k], o0[k]) for k in o0)


def axis_layout():
    """Configuration 1 geometry, Xe 2 bar, mirror reflectivity 1, continuous 2" sensor (geometry only)."""
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(2.0))
    return lay, lay.scene(), lay.mirror.f


def test_specular_ghost_is_a_uniform_disc():
    """Vertical muon on the axis: the specular reflection lands as a disc of the ring radius, uniform in radius (mean
    half the ring radius), all of it back on the sensor (the paraxial case of the stand-alone board-reflection study)."""
    lay, scene, f = axis_layout(); rng = np.random.default_rng(20261014)
    ph = cosmic.emit_batch(lay, rng, np.array([[0, 0, f]]), np.array([[0, 0, -1.0]]), np.ones(1), 100000, ('gas',), response=qe_arc)
    e = tracer.trace(scene, ph['p'], ph['d'], ph['lam'], ph['reg']); det = e['reason'] == 0; w = ph['w_ft'][e['i'][det]]
    r_ring = np.average(np.hypot(*e['p'][det, :2].T), weights=w)
    src, pe, _, we = cosmic.board_reflections(scene, rng, e['p'][det], e['d'][det], ph['lam'][e['i'][det]], w, 1.0, 0.0, gens=1)
    rg = np.hypot(*pe[:, :2].T)
    assert we.sum()/w.sum() == pytest.approx(1, abs=0.01)
    assert np.average(rg, weights=we) == pytest.approx(r_ring/2, rel=0.01)
    h = np.histogram(rg/r_ring, np.linspace(0, 1, 6), weights=we)[0]/we.sum()
    assert np.allclose(h, 0.2, atol=0.01)


def test_diffuse_reaches_the_mirror_as_sin2_alpha():
    """Lambertian light from the axis point of the focal plane: sin^2(alpha) of it reaches the clear aperture, and the
    collimated return puts the cos^4-weighted share of the clear disc inside the sensor back on it (same study)."""
    lay, scene, f = axis_layout(); a = lay.mirror.clear_radius; n = 400000
    src, pe, _, we = cosmic.board_reflections(scene, np.random.default_rng(20261014), np.tile([0, 0, f], (n, 1)), np.tile([0, 0, 1.0], (n, 1)),
                                              np.full(n, 400.0), np.ones(n), 0.0, 1.0, gens=1)
    sag = MIRROR_R-np.sqrt(MIRROR_R**2-a**2); alpha = np.arctan(a/(f-sag))
    g = np.linspace(-a, a, 1001); X, Y = np.meshgrid(g, g); c4 = (f/np.sqrt(f*f+X**2+Y**2))**4*(X**2+Y**2 < a*a)
    share = (c4*(np.abs(X) < 25.4)*(np.abs(Y) < 25.4)).sum()/c4.sum()
    assert we.sum()/n == pytest.approx(np.sin(alpha)**2*share, rel=0.02)


@pytest.mark.parametrize('run, R, nmu, refl, other, tol_refl, tol_other', [
    ('c1_xe2', (0.10, 0.0), 12000, 0.416, 0.395, 0.04, 0.04),
    ('c1_xe2', (0.0, 0.10), 12000, 0.0088, 0.0086, 0.0012, 0.0012),
    ('c5_xe2_plain_ray', (0.10, 0.0), 6000, 0.303, 0.288, 0.06, 0.06),
    ('c5_xe2_plain_ray', (0.0, 0.10), 6000, 0.0402, 0.0371, 0.006, 0.006)])   # study 0.0397-0.0407: partial Fresnel reflections left out
def test_study_numbers(run, R, nmu, refl, other, tol_refl, tol_other):
    """A stand-alone board-reflection study, written separately from cosmic.board_reflections and not kept in the
    repository (100000 / 20000 muons): reflected photons on active
    pads and extra fired pads per >= 80 % ring through the telescope, preset inputs, Xe 2 bar, --nphot 180, the muon's
    pad kept as in the study; tolerances about 3 standard deviations of this sample."""
    cr = load_driver('cosmic_rates')
    cr._G.update(inputs={**cli.preset('baseline-board'), 'board_r_spec': R[0], 'board_r_diff': R[1]}, spec=muons.MuonSpectrum(),
                 ngas=72, ntile=180, seed=20261014)
    rd = [r for r in cr.runs(1) if r['name'] == run][0]; orig = cosmic.trace_batch
    cosmic.trace_batch = lambda *a, **k: orig(*a, keep_muon_pad=True, **k)
    try:
        o = [cr.work_mirror(rd, np.random.default_rng([20261014, c]), 2000, np.random.default_rng([20261014, c, 1])) for c in range(nmu//2000)]
    finally:
        cosmic.trace_batch = orig
    o = {k: np.concatenate([x[k] for x in o]) for k in o[0]}; g = lambda k: o['br_gas_%s' % k]
    sel = (g('fm') >= .8)&(g('fs') >= .8)&o['top']&o['bot']; w = cr._G['spec'].weight_vs_cos2(o['c'])[sel]
    assert np.average(g('refl_qe')[sel], weights=w) == pytest.approx(refl, abs=tol_refl)
    assert np.average((g('pads_refl')-g('pads'))[sel], weights=w) == pytest.approx(other, abs=tol_other)


def test_board_table_columns(tmp_path):
    """The board table adds the reflected photons and other hits; the ring-hit columns are those of the ring alone."""
    crp = load_driver('cosmic_rates_plots')
    x = dict(r80_guan=[9.0, 0.5], pads_rate_ge_r80={'3': [6.0, 0.4], '10': [0.5, 0.1]}, plane_photons_mean_r80=7.0, ring_photons_mean_r80=6.0,
             pads_mean_r80=4.8, pads_frac_ge_r80=[1.0]*4+[0.6]*7, refl_photons_mean_r80=0.451, other_hits_mean_r80=0.428)
    crp.board_table({'c1_xe2': {'gas_br': {'tel': x}}, 'c1_xe3': {'gas_br': {'tel': {k: v for k, v in x.items() if 'refl' not in k and 'other' not in k}}}},
                    tmp_path, 7200)
    t = (tmp_path/'cosmic_rates_board.md').read_text().splitlines()[2:]   # after the provenance line and a blank line
    assert t[2].endswith('| reflected photons on active pads | other hits from board reflection |')
    assert t[4].startswith('| 1, Xe 2 bar |') and t[4].endswith('| 0.451 | 0.428 |') and '| 4.80 |' in t[4]
    assert t[5].startswith('| 1, Xe 3 bar |') and t[5].endswith('|  |  |')


def test_muon_pad_left_out_of_the_reflected_light():
    """refl_qe and pads_refl leave out the pad crossed by the muon: keeping it only adds, and only where the muon is on a pad."""
    lay, P0, U, b = rings(M=200, seed=4)
    kw = dict(response=pde_s13361, refl=(0.5, 0.0))
    r = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), refl_rng=np.random.default_rng(9), **kw)
    k = cosmic.trace_batch(lay, np.random.default_rng(5), P0, U, b, 72, ('gas',), refl_rng=np.random.default_rng(9), keep_muon_pad=True, **kw)
    from xerich.layout import Board
    off = Board().pad_index(P0[:, 0], P0[:, 1]) < 0
    assert np.all(k['refl_qe'] >= r['refl_qe']-1e-12) and np.any(k['refl_qe'] > r['refl_qe']+1e-9)
    assert np.allclose(k['refl_qe'][off], r['refl_qe'][off], rtol=1e-12, atol=0)
    assert np.all(k['pads_refl']-k['pads'] >= -1e-12)


def test_reflection_only_where_used_and_no_window():
    cr = load_driver('cosmic_rates')
    runs = {r['name']: r for r in cr.runs(1)}
    assert cr.board_reflection_used(runs['c1_xe2']) and cr.board_reflection_used(runs['c5_xe2_plain_ray'])
    assert not any(cr.board_reflection_used(runs[k]) for k in ('c5_xe2_plain_geo', 'c5_xe2_plain', 'c3_f50_xe2_plain', 'c2_f50_plain_ray'))
    cr._G.update(inputs=cli.preset('baseline-board'), spec=muons.MuonSpectrum(), ngas=72, ntile=180, seed=3)
    o = cr.work_mirror(runs['c5_xe2_plain_geo'], np.random.default_rng(2), 100, np.random.default_rng(7))
    assert not any(k.endswith(cosmic.REFL_KEYS) for k in o)
    from xerich.layout import Window
    lay, P0, U, b = rings(M=10)
    win = Phase1Layout(lay.mirror, lay.sensor, lay.gas, window=Window())
    with pytest.raises(ValueError):
        cosmic.trace_batch(win, np.random.default_rng(5), P0, U, b, 72, ('gas',), refl=(0.1, 0.0), refl_rng=np.random.default_rng(9))
