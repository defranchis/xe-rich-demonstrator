"""Regression gate between the legacy scripts and the studies drivers.

    python tests/regression.py run legacy OUTDIR [--jobs N] [--nevt 200]
    python tests/regression.py run new    OUTDIR [--jobs N] [--nevt 200]
    python tests/regression.py compare BASEDIR NEWDIR

`run` executes every script with OUTDIR as working directory (independent ones in
parallel), writing <job>.stdout, <job>.stderr and timing.tsv. `compare` checks that
every stdout log is identical and every npz array matches bit for bit; it prints a
markdown table and exits non-zero on any difference. PNG pixel differences are
reported for information only.
"""
import argparse, importlib.util, os, re, subprocess, sys, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from matplotlib import image as plt_image

REPO = Path(__file__).resolve().parents[1]
KEYS = ('91', '160', '240', '365')
SIMPLE = ('raytrace', 'cosmic_acc', 'cosmic_acc2', 'toy_aerogel', 'toy_scan', 'toy_prox',
          'plot_spectra', 'sketch', 'sketch_phase1')
LEGACY_THC_MRAD = '63.6'   # fixed Xe Cherenkov angle of the legacy raytrace and cosmic_acc scripts [mrad]
XE_INDEX = ('cosmic_acc2', 'toy_aerogel', 'toy_scan', 'toy_prox', 'plot_spectra', 'toy_plots', 'plot_lambda', 'plot_fcc')
LEGACY_ARGS = {'raytrace': ('--thc-mrad', LEGACY_THC_MRAD), 'cosmic_acc': ('--thc-mrad', LEGACY_THC_MRAD, '--slope-term'),
               'cosmic_acc2': ('--slope-term',), 'toy_prox': ('--legacy',)}
NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def job_table(mode, out, nevt):
    """{name: (argv, dependencies)} for the legacy scripts or the studies drivers."""
    if mode == 'legacy':
        cmd = lambda s, *a: [sys.executable, str(REPO / 'legacy' / f'{s}.py'), *a]
    else:
        cmd = lambda s, *a: [sys.executable, str(REPO / 'studies' / f'{s}.py'), '--outdir', str(out), *a,
                             *(('--xe-index', 'bm1981') if s in XE_INDEX else ()),   # the legacy scripts use Bideau-Mehu 1981
                             *LEGACY_ARGS.get(s, ())]
    jobs = {s: (cmd(s), ()) for s in SIMPLE}
    jobs['toy_plots'] = (cmd('toy_plots'), ('toy_aerogel',))
    if importlib.util.find_spec('pythia8mc') is not None:
        gen = tuple(f'gen_pk_{k}' for k in KEYS)
        for k in KEYS:
            jobs[f'gen_pk_{k}'] = (cmd('gen_pk', k, str(nevt)), ())
        jobs['plot_lambda'] = (cmd('plot_lambda'), gen)
        jobs['plot_fcc'] = (cmd('plot_fcc'), gen)
    return jobs


def run_one(name, argv, out, env):
    t0 = time.time()
    with open(out / f'{name}.stdout', 'w') as fo, open(out / f'{name}.stderr', 'w') as fe:
        rc = subprocess.call(argv, cwd=out, stdout=fo, stderr=fe, env=env)
    return name, rc, time.time() - t0


def run(mode, out, njobs, nevt):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f'refusing to write into non-empty {out}')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    jobs = job_table(mode, out, nevt)
    done, failed, running, rows = set(), set(), {}, []
    t0 = time.time()
    with ThreadPoolExecutor(njobs) as ex:
        while len(done) + len(failed) < len(jobs):
            for name, (argv, deps) in jobs.items():
                if name in done or name in failed or name in running.values():
                    continue
                if any(d in failed for d in deps):
                    failed.add(name); rows.append((name, 'skipped', 0.0)); continue
                if all(d in done for d in deps):
                    running[ex.submit(run_one, name, argv, out, env)] = name
            if not running:
                continue
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for fu in fin:
                name, rc, dt = fu.result(); del running[fu]
                (done if rc == 0 else failed).add(name)
                rows.append((name, rc, dt))
                print(f'{name:14s} rc={rc} {dt:7.1f} s', flush=True)
    with open(out / 'timing.tsv', 'w') as f:
        f.write('job\trc\twall_s\n')
        for name, rc, dt in rows:
            f.write(f'{name}\t{rc}\t{dt:.1f}\n')
        f.write(f'TOTAL\t-\t{time.time() - t0:.1f}\n')
    print(f'{mode}: {len(done)} ok, {len(failed)} failed, {time.time() - t0:.1f} s total')
    return 1 if failed else 0


def maxdiff(a, b):
    if a.shape != b.shape:
        return float('nan')
    if a.size == 0:
        return 0.0
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        d = np.abs(a.astype(float) - b.astype(float))
        d[np.isnan(a) & np.isnan(b)] = 0.0
        return float(np.max(d))
    return float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))


def compare(base, new):
    base, new = Path(base), Path(new)
    rows, ok = [], True
    for f in sorted(base.glob('*.stdout')):
        g = new / f.name
        if not g.exists():
            rows.append((f.name, 'stdout text', 'MISSING')); ok = False; continue
        a, b = f.read_text(), g.read_text()
        na, nb = [float(x) for x in NUM.findall(a)], [float(x) for x in NUM.findall(b)]
        md = max((abs(x - y) for x, y in zip(na, nb)), default=0.0) if len(na) == len(nb) else float('nan')
        same = a == b
        ok &= same
        rows.append((f.name, f'stdout text ({len(na)} numbers)', f'{md:g}' + ('' if same else ' TEXT DIFFERS')))
        if not same:
            for la, lb in zip(a.splitlines(), b.splitlines()):
                if la != lb:
                    print(f'  {f.name}\n   - {la}\n   + {lb}')
    for f in sorted(base.glob('*.npz')):
        g = new / f.name
        if not g.exists():
            rows.append((f.name, 'npz', 'MISSING')); ok = False; continue
        A, B = np.load(f), np.load(g)
        if sorted(A.files) != sorted(B.files):
            rows.append((f.name, 'npz keys', f'{A.files} vs {B.files}')); ok = False; continue
        for k in A.files:
            a, b = A[k], B[k]
            same = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
            ok &= same
            rows.append((f.name, f'array {k} {a.dtype} {a.shape}', f'{maxdiff(a, b):g}' + ('' if same else ' DIFFERS')))
    for p in sorted(base.glob('*.png')):
        q = new / p.name
        if not q.exists():
            rows.append((p.name, 'png', 'MISSING')); ok = False; continue
        a, b = plt_image.imread(p), plt_image.imread(q)
        same_bytes = p.read_bytes() == q.read_bytes()
        rows.append((p.name, f'png pixels {a.shape}, informational (bytes {"identical" if same_bytes else "differ"})', f'{maxdiff(a, b):g}'))
    print('| output | compared quantity | maxdiff |\n|---|---|---|')
    for r in rows:
        print('| %s | %s | %s |' % r)
    print('GATE:', 'PASS (identical stdout text and npz arrays)' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('mode', choices=('legacy', 'new')); r.add_argument('outdir')
    r.add_argument('--jobs', type=int, default=16); r.add_argument('--nevt', type=int, default=200)
    c = sub.add_parser('compare'); c.add_argument('base'); c.add_argument('new')
    a = ap.parse_args()
    sys.exit(run(a.mode, a.outdir, a.jobs, a.nevt) if a.cmd == 'run' else compare(a.base, a.new))
