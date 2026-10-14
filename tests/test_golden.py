"""Golden outputs of the default drivers and of the published model (tests/golden.py): every frozen number and array must
be reproduced bit for bit on a CPU type with its own reference, every number within golden.TOL on another."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import golden  # noqa: E402


def test_golden_outputs():
    msg, note = golden.check(4)
    print(note)
    assert not msg, 'golden outputs moved (%s; regenerate only with an input-card change: python tests/golden.py regen):\n%s' % (
        note, '\n'.join(msg))


def test_tolerance_comparison():
    """The tolerance mode accepts last-bit changes and names a real one."""
    assert golden.close(1.0, 1.0+1e-14) and not golden.close(1.0, 1.0+1e-9) and golden.close(0.0, 0.0)
    assert golden.close('4.0 +- 1.9', '4.0 +- 1.9') and not golden.close('4.0 +- 1.9', '4.1 +- 1.9') and not golden.close('a 1', 'b 1')
    ref = {'models': {'default': {'rates_table': {'x': [[1.0, 2.0], 3.0]}, 'npz_runs': {'r': 'abc'}}}, 'stdout': {'s': ['t 1.5']}}
    new = {'models': {'default': {'rates_table': {'x': [[1.0, 2.0*(1+1e-14)], 3.0]}, 'npz_runs': {'r': 'abd'}}}, 'stdout': {'s': ['t 1.5']}}
    assert golden.compare(ref, new, exact=False) == []
    assert golden.compare(ref, new) and any('npz_runs' in m for m in golden.compare(ref, new))
    new['models']['default']['rates_table']['x'][1] = 3.1
    assert golden.compare(ref, new, exact=False) == ['default / rates_table / x / [1]: 3.0 -> 3.1']


def test_chunked_model_spans_chunks():
    """The 'chunked' golden model traces more muons per task than one work_mirror chunk holds at the default --nphot."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cosmic_rates', golden.REPO/'studies'/'cosmic_rates.py')
    cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
    a = golden.RATES+golden.MODELS['chunked']; opt = lambda k: a[len(a)-1-a[::-1].index(k)+1]   # last occurrence wins
    cr._G.update(ngas=int(round(0.4*cr.NPHOT)), ntile=cr.NPHOT)
    for r in (r for r in cr.runs(float(opt('--nmu-scale'))) if r['group'] == 'c1'):
        assert r['n_mu']//int(opt('--nseed')) > cr.chunk_of(r) == 1041


def test_moved_array_and_dict_are_named():
    ref = {'models': {'m': {'npz_arrays': {'c1_xe2/b1_gas_fd': 'a', 'c1_xe2/p': 'b'}, 'provenance': {'x': {'r': 0.1}}}}, 'stdout': {}}
    new = {'models': {'m': {'npz_arrays': {'c1_xe2/b1_gas_fd': 'c', 'c1_xe2/p': 'b'}}}, 'stdout': {}}
    assert golden.compare(ref, new) == ['m / npz_arrays / c1_xe2/b1_gas_fd: "a" -> "c"', 'm / provenance: dict of 1 keys ["x"] -> absent']
