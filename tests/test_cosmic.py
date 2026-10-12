"""Muon spectrum parametrisations, range tables, and the momentum-aware cosmic models against the beta = 1 functions."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from xerich import cli, cosmic, muons, sim
from xerich.analysis import mc_rate, ring_in_square
from xerich.cherenkov import PhotonSpectrum, chromatic_rms, ftqe_angle_yield, pore_signal, qe_arc, qe_band, theta_c, theta_xe
from xerich.constants import COSMIC_I0, MIRROR_D, MIRROR_R, MUON_THMAX
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor
from xerich.media import N_AIR, n_xe, xenon
from xerich.muon_data import BESS_TSUKUBA, BESS_TSUKUBA_COS


def test_spectrum_flux_matches_direct_integral():
    sp = muons.MuonSpectrum('guan', thmax=MUON_THMAX)
    u = np.linspace(np.log(0.2), np.log(1000), 4001); c = np.linspace(np.cos(MUON_THMAX), 1, 801)
    I = muons.intensity(np.exp(u)[:, None], c[None, :], 'guan')*np.exp(u)[:, None]*c[None, :]
    direct = 2*np.pi*np.trapezoid(np.trapezoid(I, c, axis=1), u)
    assert sp.flux_per_m2 == pytest.approx(direct, rel=2e-3)


def test_spectrum_norm_I0():
    sp = muons.MuonSpectrum('reyna', norm='I0')
    assert sp.scale*muons.vertical_integral('reyna') == pytest.approx(COSMIC_I0, rel=1e-12)


def test_reweighting_cos3_sample_reproduces_spectrum_rate():
    sp = muons.MuonSpectrum('guan'); rng = np.random.default_rng(1)
    u, p0, _ = muons.sample_cos3(rng, 200000, 25.0, 0.0, MUON_THMAX)
    w = sp.weight_vs_cos2(-u[:, 2])
    assert muons.flux_through_square(0.05, MUON_THMAX)*w.mean() == pytest.approx(sp.rate_through_square(25.0), rel=5e-3)


def test_shukla_quoted_vertical_flux():
    # Shukla & Sankrith Table 1 (Durham): I0 = 72.5 m^-2 s^-1 sr^-1 above 0.5 GeV; the (1 + E/eps)^-1 factor lowers it slightly
    assert muons.vertical_integral('shukla', 0.5) == pytest.approx(72.5, rel=0.01)


def test_guan_reduces_to_gaisser_at_high_energy():
    # PDG Eq. (30.4) times the low-energy factor (1 + 3.64/E)^-2.7 of Guan et al. Eq. (3), vertical
    E = 1000.0; p = np.sqrt(E**2-muons.M_MU**2)
    gaisser = 1e4*0.14*E**-2.7*(1/(1+1.1*E/115)+0.054/(1+1.1*E/850))
    assert muons.intensity_guan(p, 1.0)*E/p/gaisser == pytest.approx((1+3.64/E)**-2.7, rel=1e-9)
    assert muons.cos_theta_star(1.0) == pytest.approx(1.0, abs=1e-12)


def test_reyna_scaling_and_c1():
    assert muons.intensity_reyna(1.0, 1.0) == pytest.approx(1e4*muons.REYNA_C[0], rel=1e-12)
    assert muons.intensity_reyna(4.0, 0.5)/muons.intensity_reyna(2.0, 1.0) == pytest.approx(0.125, rel=1e-12)


def test_guan_vs_bess_tsukuba():
    t = np.array(BESS_TSUKUBA); m = (t[:, 2] > 1)&(t[:, 2] < 5)
    r = muons.intensity(t[m, 2], BESS_TSUKUBA_COS, 'guan')/t[m, 3]
    assert np.all(np.abs(r-1) < 0.05)


def test_factorised_law_is_cos2():
    p = np.array([0.5, 2.0, 20.0]); c = 0.6
    assert np.allclose(muons.intensity(p, c, 'guan', factorise=True)/muons.intensity(p, 1.0, 'guan'), c*c, rtol=1e-12)
    assert np.allclose(muons.angular_exponent(p, 'shukla-eq10'), muons.angular_exponent(p, 'shukla-eq10')[0])


def test_altitude_hook():
    with pytest.raises(NotImplementedError):
        muons.altitude_factor(450.0)


def test_range_roundtrip_and_pdg_value():
    p = np.geomspace(0.1, 50, 20)
    assert np.allclose(muons.p_from_range(muons.csda_range(p, 'lead'), 'lead'), p, rtol=1e-10)
    assert muons.csda_range(1.0, 'iron') == pytest.approx(576.7, rel=2e-3)   # PDG muE_iron_Fe: T = 900 MeV, p = 1.000 GeV/c


def _flux_above(sp, pmin):
    u = 0.5*(sp.u_edges[1:]+sp.u_edges[:-1]); cell = np.diff(np.r_[0.0, sp.cdf]).reshape(len(u), -1).sum(1)*sp.flux_per_m2
    return cell[np.exp(u) > pmin].sum()


def test_range_above_table_and_overburden():
    p = np.array([99.0, 100.1, 101.0, 300.0, 1000.0]); R = muons.csda_range(p, 'concrete')
    assert np.all(np.diff(R) > 0) and np.allclose(muons.p_from_range(R, 'concrete'), p, rtol=1e-10)
    # 50 g/cm^2 of concrete removes about 0.1 GeV/c: the flux above 101 GeV/c must survive
    assert _flux_above(muons.MuonSpectrum(overburden=50), 101.0) == pytest.approx(_flux_above(muons.MuonSpectrum(), 101.0), rel=0.02)


def test_pore_signal_and_chromatic_rms():
    ng = n_xe(400, 3.0); air_ref, vac_ref = pore_signal(3.0)
    assert air_ref == pytest.approx(theta_c(1.03+0.935*(ng-N_AIR))-theta_c(1.03), rel=1e-12)
    assert vac_ref == pytest.approx(theta_c(1.03+0.97*(ng-1))-theta_c(1.03+0.97*(N_AIR-1)), rel=1e-12)
    assert 0.9e-3 < chromatic_rms(lambda lam: n_xe(lam, 3.0)) < 1.2e-3


def test_xe_air_table_uses_air_run_of_same_variant(tmp_path):
    spec = importlib.util.spec_from_file_location('cosmic_rates_plots', Path(__file__).parents[1]/'studies'/'cosmic_rates_plots.py')
    plots = importlib.util.module_from_spec(spec); spec.loader.exec_module(plots)

    def run(variant, th):
        t = dict(theta_tile_seed=[th, th], theta_tile_win_seed=[th, th], theta_tile_ownwin_seed=[th, th], theta_tile_edge_seed=[th, th],
                 win_fraction=1.0, ownwin_fraction=1.0, theta_tile_mean_p2=th)
        return dict(group='c23', variant=variant, rayleigh=False, tile_b1=t, tile_br=t)
    plots.tables({'c2_f50_pore-vac': run('pore-vac', 243.0), 'c3_f50_xe3_pore-vac': run('pore-vac', 249.5)}, {}, {}, tmp_path)
    assert '| mirror f50 | 3 bar | pore-vac | 6.50 +- 0.00 |' in (tmp_path/'cosmic_rates_tables.md').read_text()


def test_ftqe_beta1_matches_photon_spectrum():
    idx = lambda lam: n_xe(lam, 3.0)
    th, Y = ftqe_angle_yield(idx, 1.0)
    ps = PhotonSpectrum(idx, *qe_band(), beta=1.0, weight=qe_arc)
    assert Y[0] == pytest.approx(ps.yield_per_mm, rel=1e-4)
    assert 0.0635 < theta_xe(3.0) < 0.0642
    th_low, Y_low = ftqe_angle_yield(idx, 0.99)
    assert th_low[0] == 0.0 and Y_low[0] == 0.0


def test_thc_option_reproduces_legacy_angle():
    p = cli.thc(cli.xe_index(argparse.ArgumentParser()))
    assert cli.thc_value(p.parse_args(['--thc-mrad', '63.6']), 3.0) == 0.0636
    assert cli.thc_value(p.parse_args([]), 3.0) == theta_xe(3.0)


def test_beta1_limit_bit_identical_partial_and_full():
    M = Mirror(MIRROR_R, MIRROR_D); S = Sensor(25.6); L = 190; thc = 0.0636
    old = sim.ring_rate_partial(np.random.default_rng(3), M, S, L, thc, N=40000)
    rng = np.random.default_rng(3); box = M.clear_radius+L*0.4; thmax = np.radians(25)
    x, y, sx, sy, w = muons.sample_box(rng, 40000, box, thmax)
    fm, fs = cosmic.gas_ring_fractions(x, y, sx, sy, np.full(len(x), thc), M, S, L, 8, 24)
    new = (mc_rate(w, (fm >= .8)&(fs >= .8), box, thmax)*3600, mc_rate(w, (fm >= .999)&(fs >= .999), box, thmax)*3600)
    assert old == new


def test_ring_rate_full_inclined_track():
    """ring_rate_full takes the straight-track footprint, which keeps all the light of an inclined track as the tracer does."""
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(40), xenon(3.0)); M = lay.mirror; f = lay.z_sensor; th = theta_xe(3.0)
    s = 0.05; c = 1/np.sqrt(1+s*s)
    r = cosmic.trace_batch(lay, np.random.default_rng(1), np.array([[20.0-f*s, 0, f]]), np.array([[s*c, 0, -c]]), np.array([1.0]), 20000, ('gas',))
    one = (np.array([20.0]), np.array([0.0]), np.array([s]), np.array([0.0]), f, th, M.clear_radius, 8, 16)
    assert cosmic.fractions(r, 'gas')[0][0] == sim.footprint_fraction(*one)[0] == 1.0
    assert sim.footprint_fraction(*one, slope_term=True)[0] < 0.9
    rng = np.random.default_rng(2); box = M.clear_radius+f*0.6; thmax = np.radians(35)
    x, y, sx, sy, w = muons.sample_box(rng, 40000, box, thmax); cz = 1/np.sqrt(1+sx*sx+sy*sy)
    fm = cosmic.footprint_fraction_tracks(x, y, np.stack([sx*cz, sy*cz, -cz], 1), f, th, M.clear_radius, 8, 16)
    acc = (fm >= 1-1e-9)&ring_in_square(M.f*sx, M.f*sy, M.f*np.tan(th), lay.sensor.half)
    assert sim.ring_rate_full(np.random.default_rng(2), M, lay.sensor, f, th, N=40000)[0] == mc_rate(w, acc, box, thmax)*3600


def test_beta1_limit_bit_identical_proximity():
    ng = n_xe(400, 3.0); aero = Aerogel(1.0316); sens = Sensor(25.6); N = 5000
    r1 = np.random.default_rng(7)
    u, p0, _ = muons.sample_cos3(r1, N, 25.4, 203.2, MUON_THMAX); p0 = muons.redraw_xy(r1, p0, aero.half)
    a, b = sim.proximity(aero, sens, 30, ng, lambda n: (u, p0.copy(), None), N=N)
    fa, arc = sim.proximity_fractions(aero, sens, 30, ng, u, p0, np.full(N, theta_c(aero.n)))
    rate = muons.flux_through_square(2*aero.half*1e-3, MUON_THMAX)*3600
    assert (a, b) == (rate*np.mean(fa >= .8), rate*np.mean(arc >= .5))


def test_footprint_slope_term_ab():
    """The slope term double-counts the track displacement: corrected model and tracer agree, the old one does not."""
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0)); f = lay.z_sensor
    th = theta_xe(3.0); s = np.tan(np.radians(3.0)); c = 1/np.sqrt(1+s*s)
    args = (np.array([25.0]), np.array([0.0]), np.array([s]), np.array([0.0]), th, lay.mirror, lay.sensor, f, 32, 144)
    new = cosmic.gas_ring_fractions(*args)[0][0]; old = cosmic.gas_ring_fractions(*args, slope_term=True)[0][0]
    r = cosmic.trace_batch(lay, np.random.default_rng(1), np.array([[25.0-f*s, 0, f]]), np.array([[s*c, 0, -c]]), np.array([1.0]), 20000, ('gas',))
    trc = cosmic.fractions(r, 'gas')[0][0]
    assert abs(new-trc) < 0.01 and old < trc-0.15
    vert = (np.array([30.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), th, lay.mirror, lay.sensor, f, 16, 72)
    assert cosmic.gas_ring_fractions(*vert)[0][0] == pytest.approx(cosmic.gas_ring_fractions(*vert, slope_term=True)[0][0], abs=1e-12)


def test_arc_of_contained_ring_is_full():
    """A gas ring fully on the sensor covers every azimuth bin, also with few photons per muon and near threshold."""
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(25.4), xenon(3.0)); M = 200
    P0 = np.tile([0.0, 0.0, lay.z_sensor], (M, 1)); U = np.tile([0.0, 0.0, -1.0], (M, 1))
    b_mid = 1/n_xe(400.0, 3.0)   # about half of the QE band below threshold
    for n in (cosmic.N_ARC, 2*cosmic.N_ARC):
        for b in (1.0, b_mid):
            r = cosmic.trace_batch(lay, np.random.default_rng(n), P0, U, np.full(M, b), n, ('gas',))
            lit = r['emit_ft'][:, 0] > 0
            assert lit.sum() > M/2 and np.all(r['arc'][lit, 0] == 1) and np.all(r['arc_half'][lit, 0] == 1)
