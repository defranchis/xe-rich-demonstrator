"""Provenance of the driver outputs: git commit, resolved inputs, python and numpy versions.

cli.outdir(args) registers the parsed arguments of a driver (set_args); record() then returns, at write time, the
commit hash of the checkout (with a dirty-tree flag; 'unknown' outside a git checkout), the python and numpy versions,
the command line, every argument, the resolved detector inputs of drivers that take them (cli.inputs), the reference
constants and any extra fields. Writers: npz() (npz entry KEY, a JSON string), with_record() (JSON key KEY),
md_line() (first line of markdown tables, an HTML comment), text_line() (first line of text logs) and the PNG metadata
of plotting.save. VOLATILE names the fields that depend on the checkout, the platform or the call, not on the inputs.
"""
import functools
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from .constants import P_REF, T_PORE

KEY = 'provenance'
VOLATILE = ('commit', 'dirty', 'python', 'numpy', 'argv')   # stable() also drops 'outdir'
REPO = Path(__file__).resolve().parents[2]
_CTX = {}


@functools.lru_cache(maxsize=None)
def git_state(repo=REPO):
    """(commit hash, dirty flag) of the checkout at repo; ('unknown', None) without git or outside a checkout."""
    try:
        run = lambda *a: subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True, timeout=20)
        head = run('rev-parse', 'HEAD')
        if head.returncode != 0:
            return 'unknown', None
        st = run('status', '--porcelain', '--untracked-files=no')
        return head.stdout.strip(), (bool(st.stdout.strip()) if st.returncode == 0 else None)
    except (OSError, subprocess.SubprocessError):
        return 'unknown', None


def jsonable(v):
    """v with paths, numpy scalars and arrays and tuples turned into JSON types (anything else: its str)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def set_args(args):
    """Register the parsed arguments of the running driver (read at write time, so later resolutions are kept)."""
    _CTX['args'] = args


def record(**extra):
    """Provenance record of the running driver, with extra fields."""
    commit, dirty = git_state()
    args = _CTX.get('args')
    rec = dict(commit=commit, dirty=dirty, python=platform.python_version(), numpy=np.__version__, argv=list(sys.argv),
               script=Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
               args=jsonable(vars(args)) if args is not None else {}, constants=dict(P_REF=P_REF, T_PORE=T_PORE))
    if args is not None and hasattr(args, 'preset'):
        from .cli import inputs
        rec['inputs'] = jsonable(inputs(args))
    rec.update(jsonable(extra))
    return rec


def npz(**extra):
    """{KEY: record as a JSON string} for np.savez."""
    return {KEY: np.array(json.dumps(record(**extra), sort_keys=True))}


def with_record(obj, **extra):
    """dict obj with the record under KEY first (a new dict); a record already in obj (a merged older output) is replaced."""
    return {KEY: record(**extra), **{k: v for k, v in obj.items() if k != KEY}}


def md_line(**extra):
    """First line of a markdown output: the record as an HTML comment (not rendered)."""
    return '<!-- %s: %s -->' % (KEY, json.dumps(record(**extra), sort_keys=True))


def text_line(**extra):
    """First line of a text output."""
    return '# %s: %s' % (KEY, json.dumps(record(**extra), sort_keys=True))


def stable(rec):
    """The record without the VOLATILE fields and the output directories, at any depth (what the inputs alone determine)."""
    if isinstance(rec, dict):
        return {k: stable(v) for k, v in rec.items() if k not in VOLATILE+('outdir',)}
    return [stable(v) for v in rec] if isinstance(rec, list) else rec
