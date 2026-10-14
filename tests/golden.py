"""Golden outputs: small fixed-seed runs of the default drivers and of the published model, frozen bit for bit.

    python tests/golden.py check [--jobs N]     compare with the reference of this CPU type (exit 1 on any change)
    python tests/golden.py regen [--jobs N]     rewrite the reference of this CPU type

A regeneration is allowed only with a change of the input card (docs/design_options.md): a code change must leave every
golden output unchanged. Runs: cosmic_rates.py --groups c1 c5 c4 and cosmic_rates_plots.py at the defaults and with
--preset published (the model of the published tables), configuration 1 alone with more muons per task than one
work_mirror chunk holds (model 'chunked'), plus the stdout of raytrace.py, cosmic_acc.py and toy_prox.py.
Frozen: every npz array (sha256 of dtype, shape and bytes, one digest per array, and per run and per field), the summary
json (digest per run, and per run and beta tag the count, sum, sum of absolute values and a weighted sum of its numbers;
the configuration-4 Rayleigh record verbatim), the rates table (telescope, >= 80 % rings), the board table and the stdout
texts, numbers verbatim; the provenance records without their volatile fields (commit, versions, command line, input and
output directories, workers). The reference also records the commit it was generated at (not compared).
References: tests/golden_reference_<key>.json, key from the CPU features numpy dispatches on (platform_key): float
results can differ in the last bit between CPU types. On a CPU type without its own reference, check skips the digests
(and says so) and compares every number of the rates and board tables, the stdout and the summary sums with a relative
tolerance TOL against another reference.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from xerich import provenance  # noqa: E402

GROUPS = ('c1', 'c5', 'c4')
RATES = ['--nmu-scale', '0.002', '--nseed', '2', '--seed', '20261010', '--groups', *GROUPS]
MODELS = {'default': [], 'published': ['--preset', 'published'],   # cosmic_rates.py inputs of each frozen model (after RATES)
          'chunked': ['--groups', 'c1', '--nmu-scale', '0.011', '--nseed', '1']}   # 1100 muons per task: chunks of 1041 + 59
TEXT = {'raytrace': [], 'cosmic_acc': [], 'toy_prox': ['--nmu', '2000', '--seed', '7']}   # stdout frozen as text
TOL = 1e-10                  # relative tolerance of the numbers on a CPU type without its own reference
NUMERIC = ('summary_sums', 'rates_table', 'board_table', 'rayleigh_c4')   # per-model sections compared with TOL there (and the stdout)
NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def platform_key():
    """CPU type of the reference: 'avx512' (AVX512F), 'avx2' or 'baseline', from numpy's dispatched CPU features."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as f
    except ImportError:
        from numpy.core._multiarray_umath import __cpu_features__ as f
    return 'avx512' if f.get('AVX512F') else 'avx2' if f.get('AVX2') else 'baseline'


def reference(key):
    return REPO/'tests'/('golden_reference_%s.json' % key)


def run(script, out, *a, nproc=None):
    cmd = [sys.executable, str(REPO/'studies'/('%s.py' % script)), '--outdir', str(out), *a] + (['--nproc', str(nproc)] if nproc else [])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=out)
    if r.returncode:
        raise RuntimeError('%s failed:\n%s' % (' '.join(cmd), r.stderr[-3000:]))
    return r.stdout


def digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:20]


def array_digest(a):
    return digest(a.dtype.str, a.shape, np.ascontiguousarray(a).tobytes())


def stable(rec):
    """provenance.stable without the worker count and the input directory (the outputs do not depend on them)."""
    if isinstance(rec, dict):
        return {k: stable(v) for k, v in provenance.stable(rec).items() if k not in ('nproc', 'input')}
    return [stable(v) for v in rec] if isinstance(rec, list) else rec


def body(path):
    """Text of a markdown output without its provenance line."""
    return '\n'.join(l for l in path.read_text().splitlines() if not l.startswith('<!-- %s:' % provenance.KEY))


def leaves(x):
    """Numbers of a JSON value in a fixed order (keys sorted)."""
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in leaves(x[k])]
    if isinstance(x, list):
        return [v for e in x for v in leaves(e)]
    return [float(x)] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def sums(x):
    """[count, non-finite count, sum, sum of |x|, sum of w_i x_i] of the numbers of x (sums over the finite ones),
    w_i = 1 + frac(0.618 i) by position: a moved number cannot be compensated by another."""
    v = np.array(leaves(x)); w = 1+np.modf(0.6180339887*np.arange(len(v)))[0]; f = np.isfinite(v)
    return [len(v), int((~f).sum()), float(v[f].sum()), float(np.abs(v[f]).sum()), float((w*v)[f].sum())]


def rates_outputs(out, args, jobs):
    """Frozen quantities of cosmic_rates.py + cosmic_rates_plots.py with the extra arguments args, run in out."""
    G = {}
    run('cosmic_rates', out, *RATES, *args, nproc=jobs)
    run('cosmic_rates_plots', out, '--input', str(out), nproc=jobs)
    runs, fields, arrays, prov = {}, {}, {}, {}
    for g in GROUPS:
        if not (out/('cosmic_rates_%s.npz' % g)).exists():
            continue
        z = np.load(out/('cosmic_rates_%s.npz' % g))
        prov['npz_%s' % g] = stable(json.loads(str(z[provenance.KEY])))
        for k in sorted(z.files):
            if k == provenance.KEY:
                continue
            r, f = k.split('/', 1); d = array_digest(z[k]); arrays[k] = d
            runs.setdefault(r, []).append(f+d); fields.setdefault(f, []).append(r+d)
    G['npz_runs'] = {r: digest(*v) for r, v in sorted(runs.items())}
    G['npz_fields'] = {f: digest(*v) for f, v in sorted(fields.items())}
    G['npz_arrays'] = dict(sorted(arrays.items()))
    meta = json.loads((out/'cosmic_rates_meta.json').read_text())
    prov['meta'] = stable(meta.pop(provenance.KEY)); meta = stable({k: v for k, v in meta.items() if k != 'wall_s'})
    G['meta'] = digest(json.dumps(meta, sort_keys=True))
    S = json.loads((out/'cosmic_rates_summary.json').read_text())
    prov['summary'] = stable(S.pop(provenance.KEY)); R = S.pop('runs')
    if 'rayleigh_c4' in S:
        G['rayleigh_c4'] = S.pop('rayleigh_c4')
    G['summary_runs'] = {r: digest(json.dumps(x, sort_keys=True)) for r, x in sorted(R.items())}
    G['summary_rest'] = {k: digest(json.dumps(v, sort_keys=True)) for k, v in sorted(S.items())}
    G['summary_sums'] = {'%s %s' % (r, t): sums(x[t]) for r, x in sorted(R.items()) for t in sorted(x) if t.startswith(('gas_', 'tile_'))}
    G['summary_sums'].update({'rest': sums(S)})
    T = json.loads((out/'cosmic_rates_table.json').read_text())
    prov['table'] = stable(T[provenance.KEY])
    G['rates_table'] = {'%s | %s' % (r['config'], r['origin']): [r['beta1'], r['real'], r['band'], r['photons'], r['precision']]
                        for r in T['rows'] if r['norm'] == 'tel' and r['definition'] == 'r80'}
    G['tables_md'] = digest(body(out/'cosmic_rates_tables.md'))
    board = out/'cosmic_rates_board.md'
    G['board_table'] = {l.split('|')[1].strip(): [c.strip() for c in l.split('|')[2:-1]]
                        for l in (body(board).splitlines() if board.exists() else []) if l.startswith('| ') and not l.startswith('| Configuration')}
    G['provenance'] = prov
    return G


def collect(jobs):
    """Run the golden drivers in temporary directories and return the frozen quantities."""
    G = {'models': {}}
    for name, args in MODELS.items():
        with tempfile.TemporaryDirectory() as td:
            G['models'][name] = rates_outputs(Path(td), args, jobs)
    with tempfile.TemporaryDirectory() as td:
        G['stdout'] = {s: run(s, Path(td), *a).splitlines() for s, a in TEXT.items()}
    return G


def close(a, b):
    """a and b equal within TOL: numbers relatively, strings with the same text and close numbers, containers per item."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a == b or abs(a-b) <= TOL*max(abs(a), abs(b)) or (a != a and b != b)   # NaN == NaN
    if isinstance(a, str) and isinstance(b, str):
        na, nb = NUM.findall(a), NUM.findall(b)
        return NUM.sub('#', a) == NUM.sub('#', b) and len(na) == len(nb) and all(close(float(x), float(y)) for x, y in zip(na, nb))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    return a == b


def show(x):
    """Short text of a frozen value: a dict by its keys, anything else by its first 200 characters."""
    if isinstance(x, dict):
        return 'dict of %d keys %s' % (len(x), json.dumps(sorted(x)[:20])[:-1]+(', ...]' if len(x) > 20 else ']'))
    return 'absent' if x is None else json.dumps(x)[:200]


def diff(path, a, b, same, msg):
    """Append to msg a line for every item of a and b (nested) for which same() fails, named by its path."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            diff(path+[str(k)], a.get(k), b.get(k), same, msg)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and any(isinstance(x, (list, dict, str)) for x in a+b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(path+['[%d]' % i], x, y, same, msg)
    elif not same(a, b):
        msg.append('%s: %s -> %s' % (' / '.join(path), show(a), show(b)))


def compare(ref, new, exact=True):
    """Lines naming every frozen quantity that moved (empty: identical); exact=False compares only the numbers with TOL."""
    msg = []
    for name in sorted(set(ref['models']) | set(new['models'])):
        a, b = ref['models'].get(name, {}), new['models'].get(name, {})
        for sec in sorted(set(a) | set(b)):
            if exact:
                diff([name, sec], a.get(sec), b.get(sec), lambda x, y: x == y, msg)
            elif sec in NUMERIC:
                diff([name, sec], a.get(sec), b.get(sec), close, msg)
    diff(['stdout'], ref['stdout'], new['stdout'], (lambda x, y: x == y) if exact else close, msg)
    return msg


def check(jobs):
    """(lines naming what moved, note on the reference used) for this CPU type."""
    key = platform_key(); ref = reference(key)
    if ref.exists():
        return compare(json.loads(ref.read_text()), collect(jobs)), 'reference %s: exact' % ref.name
    others = sorted(REPO.glob('tests/golden_reference_*.json'))
    if not others:
        raise FileNotFoundError('no golden reference (python tests/golden.py regen)')
    note = ('no reference for CPU type %r: digests skipped, numbers compared with relative tolerance %g against %s'
            % (key, TOL, others[0].name))
    return compare(json.loads(others[0].read_text()), collect(jobs), exact=False), note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('check', 'regen'))
    ap.add_argument('--jobs', type=int, default=4, help='worker processes of the drivers (the outputs do not depend on it)')
    a = ap.parse_args()
    if a.mode == 'regen':
        G = {'commands': ['cosmic_rates.py %s %s' % (' '.join(RATES), ' '.join(x)) for x in MODELS.values()]
             + ['cosmic_rates_plots.py --input <same>'] + ['%s.py %s' % (s, ' '.join(x)) for s, x in TEXT.items()],
             'platform': platform_key(), 'reference_numpy': np.__version__, **collect(a.jobs)}
        G['generated'] = dict(zip(('commit', 'dirty'), provenance.git_state()))   # not compared
        ref = reference(G['platform'])
        ref.write_text(json.dumps(G, indent=0, sort_keys=True)+'\n')
        print('wrote %s (%d bytes)' % (ref, ref.stat().st_size))
        return 0
    msg, note = check(a.jobs)
    print(note)
    print('\n'.join(msg) if msg else 'golden outputs identical')
    return 1 if msg else 0


if __name__ == '__main__':
    sys.exit(main())
