"""Reference values of the Xe index models at 20 degC (Bideau-Mehu 1981 at the precision of the bm1981 design values), Boerzsoenyi 2008, aerogel."""
import numpy as np
import pytest

from xerich.cherenkov import energy_grid, p_threshold, photon_yield, theta_c
from xerich.constants import AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_POLE_NM, HC, M_K, T0
from xerich.media import (AEROGEL_K4GEO_E_EV, AEROGEL_K4GEO_N, BORZSONYI_XE, RHO_BORZSONYI, aerogel_index, aerogel_medium,
                          molar_density, n2m1_borzsonyi, n_aerogel_k4geo, n_aerogel_single_pole, n_xe)


@pytest.mark.parametrize('P,nm1', [(2, 1.41e-3), (3, 2.12e-3)])
def test_bm1981_index_300nm(P, nm1):
    assert n_xe(300, P, model='bm1981')-1 == pytest.approx(nm1, abs=0.005e-3)


@pytest.mark.parametrize('P,th', [(2, 51.4), (3, 63.2)])
def test_bm1981_theta_c_400nm(P, th):
    assert 1e3*theta_c(n_xe(400, P, model='bm1981')) == pytest.approx(th, abs=0.05)


@pytest.mark.parametrize('P,pk', [(2, 9.3), (3, 7.6)])
def test_bm1981_kaon_threshold_300nm(P, pk):
    assert p_threshold(M_K, n_xe(300, P, model='bm1981')) == pytest.approx(pk, abs=0.05)


@pytest.mark.parametrize('P,nph', [(2, 4.4), (3, 6.6)])
def test_bm1981_saturated_yield(P, nph):
    assert photon_yield(energy_grid(200, 600, 400), 1e4, 0.13957, P, model='bm1981') == pytest.approx(nph, abs=0.05)


def test_index_scales_with_pressure():
    lam = np.linspace(200, 600, 5)
    assert np.all(n_xe(lam, 3) > n_xe(lam, 2)) and np.all(np.diff(n_xe(lam, 2)) < 0)


def test_aerogel_index_without_gas():
    assert aerogel_index(1.0) == 1.03


def test_borzsonyi_coefficients():
    """Coefficients of refractiveindex.info main/Xe/Borzsonyi (formula 2, C_1 typo of the paper corrected)."""
    assert BORZSONYI_XE == ((103701.61e-8, 12.75e-3), (31228.61e-8, 0.561e-3))


@pytest.mark.parametrize('lam,nm1', [(400, 7.198377e-4), (800, 6.850930e-4), (1000, 6.812031e-4)])
def test_borzsonyi_reference_state(lam, nm1):
    """At 273.15 K and 1000 mbar n_xe reproduces the formula over the fitted 400-1000 nm range (800 nm: measured)."""
    l2 = (lam*1e-3)**2
    n_formula = np.sqrt(1+1.0370161e-3*l2/(l2-0.01275)+3.122861e-4*l2/(l2-0.000561))
    assert n_xe(lam, 1.0, T0)-1 == pytest.approx(n_formula-1, rel=1e-12)
    assert n_formula-1 == pytest.approx(nm1, rel=1e-6)
    assert n_xe(lam, 1.0, T0)**2-1 == pytest.approx(n2m1_borzsonyi(lam), rel=1e-12)


def test_borzsonyi_lorentz_lorenz_scaling():
    lam = np.array([264.0, 400.0, 900.0])
    for P, T in ((2, 293.15), (3, 293.15), (3, 273.15)):
        n = n_xe(lam, P, T); LL = (n**2-1)/(n**2+2)
        s = n2m1_borzsonyi(lam)
        assert np.allclose(LL, s/(s+3)*molar_density(P*1e5, T)/RHO_BORZSONYI, rtol=1e-12, atol=0)
    assert RHO_BORZSONYI == pytest.approx(44.3311, rel=5e-4)      # NIST Webbook, Xe at 273.15 K and 0.1 MPa; literal kept as an independent check


@pytest.mark.parametrize('P,nm1,th,pk,nph', [(2, 1.42e-3, 51.9, 9.2, 4.4), (3, 2.15e-3, 63.7, 7.5, 6.7)])
def test_borzsonyi_design_values(P, nm1, th, pk, nph):
    """Design-doc values with the default (Boerzsoenyi) index at 20 degC."""
    assert n_xe(300, P)-1 == pytest.approx(nm1, abs=0.005e-3)
    assert 1e3*theta_c(n_xe(400, P)) == pytest.approx(th, abs=0.05)
    assert p_threshold(M_K, n_xe(300, P)) == pytest.approx(pk, abs=0.05)
    assert photon_yield(energy_grid(200, 600, 400), 1e4, 0.13957, P) == pytest.approx(nph, abs=0.05)


def test_unknown_index_model():
    with pytest.raises(ValueError):
        n_xe(400, 3, model='koch')


def test_aerogel_dispersion_models():
    lam = np.array([300.0, AEROGEL_LAMBDA_REF, 600.0])
    for f in (n_aerogel_single_pole, n_aerogel_k4geo):
        n = f(lam)
        assert n[1] == pytest.approx(AEROGEL_N_AIR, abs=1e-14) and n[0] > n[1] > n[2]
    assert aerogel_medium('single-pole', lam_ref=350).n(350.0) == pytest.approx(AEROGEL_N_AIR, abs=1e-14)
    assert np.all(aerogel_medium('constant').n(lam) == AEROGEL_N_AIR)
    E = np.array(AEROGEL_K4GEO_E_EV); N = np.array(AEROGEL_K4GEO_N)
    t400 = np.interp(HC/AEROGEL_LAMBDA_REF, E, N)
    assert np.allclose(n_aerogel_k4geo(HC/E)-1, (N-1)*(AEROGEL_N_AIR-1)/(t400-1), rtol=1e-12)
    l0 = AEROGEL_POLE_NM; a0 = (AEROGEL_N_AIR**2-1)*(1-l0**2/AEROGEL_LAMBDA_REF**2)
    assert n_aerogel_single_pole(250.0)**2-1 == pytest.approx(a0*250**2/(250**2-l0**2), rel=1e-12)
