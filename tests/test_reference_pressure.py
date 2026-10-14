"""The reference Xe pressure (constants.P_REF) of the figures and defaults."""
import importlib.util
from pathlib import Path

from xerich.constants import P_REF


def load_driver(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'studies'/('%s.py' % name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_reference_pressure_value():
    assert P_REF == 2.0                                                     # Xe 2 bar, the reference pressure


def test_figure_drivers_use_the_reference_gas():
    sf = load_driver('short_f_mirrors'); px = load_driver('proximity_config4'); sk = load_driver('sketch_effects')
    assert sf.REF_GAS == px.REF_GAS == sk.REF_GAS == 'xe2'
    assert sf.REF_GAS in sf.GAS_NAMES and px.REF_GAS in px.GAS_NAMES and sk.REF_GAS in sk.GASES
    assert px.POINT_GASES == ('air', 'xe2')
    assert len(px.check_segments()) == 4                                    # segment check with the layout's window thickness


def test_gas_yield_reference():
    """GAS_YIELD_REF from its definition, N = 2 pi alpha L int_320^450 (1 - 1/n^2) dlambda/lambda^2 at P_REF, beta = 1."""
    import argparse
    import numpy as np
    from xerich import cli, sim
    from xerich.constants import ALPHA, GAS_YIELD_REF, GAS_YIELD_REF_LEN
    from xerich.media import n_xe
    lam = np.linspace(320.0, 450.0, 20001)
    N = 2*np.pi*ALPHA*GAS_YIELD_REF_LEN*1e6*np.trapezoid((1-1/n_xe(lam, P_REF)**2)/lam**2, lam)
    assert GAS_YIELD_REF == round(N, 2) == 21.72
    assert sim.gas_yield(P_REF) == GAS_YIELD_REF and abs(sim.gas_yield(3.0)-32.73) < 0.01
    args = cli.gas_yield(cli.pressure(argparse.ArgumentParser())).parse_args([])
    assert args.pressure == P_REF and args.gas_yield is None


def test_reference_defaults():
    from xerich import sketch
    assert sketch.P_GAS == P_REF and sketch.P_ALT == 3.0
    cr = load_driver('cosmic_rates')
    r = {x['name']: x for x in cr.runs(1)}
    assert r['c3_f50_xe2_plain_ray']['P'] == P_REF and 'c3_f50_xe3_plain_ray' not in r
    assert all(x['P'] == P_REF for x in r.values() if x['group'] == 's61')
