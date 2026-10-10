"""Reference values of the Xe model at 20 degC, at the precision quoted in docs/design_options.md."""
import numpy as np
import pytest

from xerich.cherenkov import energy_grid, p_threshold, photon_yield, theta_c
from xerich.constants import M_K
from xerich.media import aerogel_index, n_xe


@pytest.mark.parametrize('P,nm1', [(2, 1.41e-3), (3, 2.12e-3)])
def test_index_300nm(P, nm1):
    assert n_xe(300, P)-1 == pytest.approx(nm1, abs=0.005e-3)


@pytest.mark.parametrize('P,th', [(2, 51.4), (3, 63.2)])
def test_theta_c_400nm(P, th):
    assert 1e3*theta_c(n_xe(400, P)) == pytest.approx(th, abs=0.05)


@pytest.mark.parametrize('P,pk', [(2, 9.3), (3, 7.6)])
def test_kaon_threshold_300nm(P, pk):
    assert p_threshold(M_K, n_xe(300, P)) == pytest.approx(pk, abs=0.05)


@pytest.mark.parametrize('P,nph', [(2, 4.4), (3, 6.6)])
def test_saturated_yield(P, nph):
    assert photon_yield(energy_grid(200, 600, 400), 1e4, 0.13957, P) == pytest.approx(nph, abs=0.05)


def test_index_scales_with_pressure():
    lam = np.linspace(200, 600, 5)
    assert np.all(n_xe(lam, 3) > n_xe(lam, 2)) and np.all(np.diff(n_xe(lam, 2)) < 0)


def test_aerogel_index_without_gas():
    assert aerogel_index(1.0) == 1.03
