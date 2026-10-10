"""Optics tables, photon spectra and the on-axis muon tracer (xerich.media, cherenkov, layout, sim)."""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from xerich import sim, tracer
from xerich.cherenkov import QE_ARC, QE_ARC_E_EV, PhotonSpectrum, frank_tamm, qe_arc, qe_band, theta_c
from xerich.constants import AEROGEL_N_AIR, ALPHA, HBARC_EV_CM, HC, MIRROR_D, MIRROR_R, SENSOR_HALF_2IN
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window
from xerich.media import (AEROGEL_K4GEO_E_EV, AEROGEL_K4GEO_N, C4F10_E_EV, C4F10_N, N_AIR, ConstantIndex, Medium, Rayleigh,
                          aerogel_constant, n_c4f10, xenon)

K4GEO = os.environ.get('XERICH_K4GEO')
MATERIALS = 'FCCee/CLD/compact/CLD_o3_v01/materials_arc_o1_v01.xml'
UNIT = {'eV': 1.0, 'mm': 1.0, 'cm': 10.0, 'm': 1e3}


def phase1(gas, tile=True):
    return Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), gas, Aerogel(AEROGEL_N_AIR) if tile else None)


def spectra(lay, lo=1.3, hi=4.7):
    sp = {'gas': PhotonSpectrum(lay.gas.index, lo, hi)}
    if lay.tile is not None:
        sp['tile'] = PhotonSpectrum(lay.tile_index.index, lo, hi)
    return sp


def k4geo_matrix(text, name):
    """(line of the <matrix> tag, energies [eV], values) of the named matrix; values scaled by their unit."""
    m = re.search(r'<matrix\s+name="%s"[^>]*values="([^"]*)"' % name, text)
    assert m, name
    rows = re.findall(r'([-+0-9.eE]+)\s*\*\s*(\w+)\s+([-+0-9.eE]+)(?:\s*\*\s*(\w+))?', m.group(1))
    return (text.count('\n', 0, m.start())+1, [float(e)*UNIT[u] for e, u, _, _ in rows],
            [float(v)*(UNIT[vu] if vu else 1.0) for _, _, v, vu in rows])


@pytest.mark.skipif(not K4GEO or not (Path(K4GEO)/MATERIALS).is_file(),
                    reason='$XERICH_K4GEO (k4geo tree root) not set or without '+MATERIALS)
def test_k4geo_tables_embedded():
    text = (Path(K4GEO)/MATERIALS).read_text()
    src = Path(__file__).resolve().parents[1]/'src'/'xerich'
    cites = (src/'cherenkov.py').read_text()+(src/'media.py').read_text()
    cited = {int(a or b) for a, b in re.findall(r'materials_arc_o1_v01\.xml:(\d+)|same file, line (\d+)', cites)}
    lines = {}
    for name, E, v in (('RINDEX__C4F10_PFRICH', C4F10_E_EV, C4F10_N), ('ARC_SiPM_QuantumEfficiency', QE_ARC_E_EV, QE_ARC),
                       ('RINDEX__Aerogel_PFRICH', AEROGEL_K4GEO_E_EV, AEROGEL_K4GEO_N)):
        lines[name], Ex, vx = k4geo_matrix(text, name)
        assert list(E) == Ex and list(v) == vx
    lines['RINDEX__Air'], Ea, va = k4geo_matrix(text, 'RINDEX__Air')
    assert set(va) == {N_AIR} and Ea[0] <= 1.0 and Ea[-1] >= 5.1
    lines['RAYLEIGH__Aerogel_BelleII'], Er, Lr = k4geo_matrix(text, 'RAYLEIGH__Aerogel_BelleII')
    assert np.allclose(Rayleigh().length(HC/np.array(Er)), Lr, rtol=0.02)   # lambda^4/C within the rounding of the energies
    assert cited == set(lines.values())            # source comments cite the line of each <matrix> tag


def test_c4f10_interpolation():
    assert np.allclose(n_c4f10(HC/np.array(C4F10_E_EV)), C4F10_N, rtol=0, atol=1e-15)
    E = 0.5*(C4F10_E_EV[5]+C4F10_E_EV[6])
    assert n_c4f10(HC/E) == pytest.approx(0.5*(C4F10_N[5]+C4F10_N[6]), abs=1e-15)
    assert n_c4f10(950.0) == C4F10_N[0] and n_c4f10(150.0) == C4F10_N[-1]


def test_qe_table_and_band():
    assert qe_band() == (1.3, 4.7)
    assert np.allclose(qe_arc(np.array(QE_ARC_E_EV)), QE_ARC) and qe_arc(5.5) == 0 and qe_arc(1.0) == 0
    assert qe_arc(0.5*(3.084+3.002)) == pytest.approx(0.6)


def test_spectrum_yield_and_sampling():
    lo, hi = qe_band()
    sp = PhotonSpectrum(ConstantIndex(AEROGEL_N_AIR), lo, hi)
    assert sp.yield_per_mm == pytest.approx(ALPHA/HBARC_EV_CM*(1-1/AEROGEL_N_AIR**2)*(hi-lo)/10, rel=1e-12)
    E = sp.sample(np.random.default_rng(1), 400000)
    assert E.min() >= lo and E.max() <= hi and E.mean() == pytest.approx(0.5*(lo+hi), abs=0.005)
    sq = PhotonSpectrum(ConstantIndex(AEROGEL_N_AIR), lo, hi, weight=qe_arc)
    Eg = np.linspace(lo, hi, 200001); q = qe_arc(Eg)
    assert sq.sample(np.random.default_rng(2), 400000).mean() == pytest.approx(np.trapezoid(Eg*q, Eg)/np.trapezoid(q, Eg), abs=0.004)
    gas = PhotonSpectrum(xenon(3).index, lo, hi)
    assert np.allclose(gas.pdf, frank_tamm(xenon(3).n(HC/gas.E), 1.0))


def test_track_segments_onaxis():
    lay = phase1(xenon(3)); f = lay.mirror.f
    seg = sim.track_segments(lay, (0, 0, f), (0, 0, -1))
    assert np.allclose(seg['tile'], [(1.0, 11.0)]) and np.allclose(seg['gas'], [(0.0, 1.0), (11.0, f)])
    seg = sim.track_segments(phase1(xenon(3), tile=False), (0, 0, f), (0, 0, -1))
    assert seg['tile'] == [] and np.allclose(seg['gas'], [(0.0, f)])


@pytest.mark.parametrize('polarised', [True, False])
def test_weight_conservation(polarised):
    lay = phase1(xenon(3)); sp = spectra(lay)
    for cap in (0, 2):
        ph, e = sim.onaxis_muon(lay, np.random.default_rng(3), sp, 3000, 3000, max_reflections=cap, polarised=polarised)
        W = np.bincount(e['i'], e['w'], minlength=len(ph['lam']))
        assert np.allclose(W, 1, rtol=0, atol=1e-12)
        if cap == 0:
            assert e['w'][e['reason'] == 4].sum() > 0
    ph, e = sim.muon_photons(lay, np.random.default_rng(4), (7.0, -3.0, lay.z_sensor), (np.sin(0.2), 0, -np.cos(0.2)),
                             sp, 2000, 2000, polarised=polarised)
    assert np.allclose(np.bincount(e['i'], e['w'], minlength=len(ph['lam'])), 1, rtol=0, atol=1e-12)


def test_closure_ring_radius_without_tile():
    """No tile, constant gas index: ring radius f tan(theta_c) within the on-axis spherical-aberration level."""
    n = 1.002
    lay = phase1(Medium('gas', ConstantIndex(n)), tile=False)
    ph, e = sim.onaxis_muon(lay, np.random.default_rng(5), spectra(lay), 20000, 0)
    det = e['reason'] == 0
    assert det.sum() == 20000 and np.allclose(e['w'][det], 1) and np.all(e['cls'][det] == 0)
    r = np.hypot(e['p'][det, 0], e['p'][det, 1]); r0 = lay.mirror.f*np.tan(theta_c(n))
    assert abs(r.mean()/r0-1) < 5e-3
    assert 1e3*np.arctan(r/lay.mirror.f).std() < 0.1


def test_onaxis_aerogel_light_is_fresnel_only():
    """On axis all aerogel light misses the mirror; on the SiPM only the bottom-face reflection, weight Rp x Tp."""
    lay = phase1(xenon(3)); sp = spectra(lay)
    ph, e = sim.onaxis_muon(lay, np.random.default_rng(6), sp, 0, 5000, max_reflections=1)
    lost = (e['reason'] == 1)&(e['cls'] == 0)
    assert lost.sum() == 5000
    det = e['reason'] == 0
    assert np.all(e['cls'][det] == 1) and det.sum() == 5000
    ta = theta_c(AEROGEL_N_AIR); ng = xenon(3).n(ph['lam'][e['i'][det]])
    _, Rp, _, Tp = tracer.fresnel_powers(np.cos(ta), AEROGEL_N_AIR, ng)
    assert np.allclose(e['w'][det], Rp*Tp, rtol=1e-9)
    r = np.hypot(e['p'][det, 0], e['p'][det, 1])
    gap = np.tan(np.arcsin(AEROGEL_N_AIR*np.sin(ta)/ng))
    assert np.all(r > 10*np.tan(ta)+gap-1e-9) and np.all(r < 20*np.tan(ta)+gap+1e-9)


def test_track_segments_with_window():
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(3), Aerogel(AEROGEL_N_AIR), window=Window(1.5, 0.3))
    f = lay.mirror.f
    seg = sim.track_segments(lay, (0, 0, f), (0, 0, -1))
    assert np.allclose(seg['window'], [(0.0, 0.3)]) and np.allclose(seg['tile'], [(1.0, 11.0)])
    assert np.allclose(seg['gas'], [(0.3, 1.0), (11.0, f)])
    assert len(lay.scene().slabs) == 2 and len(phase1(xenon(3)).scene().slabs) == 1


@pytest.mark.parametrize('n_w,t', [(1.5, 0.3), (1.45, 0.1), (1.55, 0.5)])
def test_window_cluster(n_w, t):
    """Window light is totally reflected at its bottom face and lands at (2t - d) tan(theta_c), d = depth below the silicon."""
    lay = phase1(xenon(3)); lo, hi = qe_band()
    sp = PhotonSpectrum(ConstantIndex(n_w), lo, hi)
    ph, e = sim.window_cluster(lay, np.random.default_rng(7), sp, 20000, Window(n_w, t))
    det = e['reason'] == 0
    assert det.all() and np.all(e['cls'] == 3) and np.allclose(e['w'], 1)
    st = sim.cluster_stats(ph, e)
    assert st['n'] == pytest.approx(ALPHA/HBARC_EV_CM*(1-1/n_w**2)*(hi-lo)*t/10, rel=1e-9)
    tt = np.tan(theta_c(n_w)); r = np.hypot(e['p'][:, 0], e['p'][:, 1])
    assert np.all(r > t*tt-1e-9) and np.all(r < 2*t*tt+1e-9) and st['r_mean'] == pytest.approx(1.5*t*tt, rel=5e-3)
    assert np.allclose(e['p'][:, 2], lay.z_sensor)
    sq = sim.cluster_stats(ph, e, weight=lambda lam: qe_arc(HC/lam))
    assert sq['n'] == pytest.approx(PhotonSpectrum(ConstantIndex(n_w), lo, hi, weight=qe_arc).yield_per_mm*t, rel=0.01)
