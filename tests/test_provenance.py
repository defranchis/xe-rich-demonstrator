"""Provenance records of the driver outputs (xerich.provenance)."""
import argparse
import json

import numpy as np

from xerich import cli, provenance
from xerich.constants import P_REF


def test_git_state_and_record(tmp_path):
    commit, dirty = provenance.git_state()
    assert commit == 'unknown' or (len(commit) == 40 and dirty in (True, False))
    assert provenance.git_state(tmp_path) == ('unknown', None)                 # outside a checkout: no crash
    p = cli.detector_inputs(cli.parser('x')); p.add_argument('--nphot', type=int, default=None)
    args = p.parse_args(['--outdir', str(tmp_path), '--preset', 'published'])
    cli.outdir(args); args.nphot = 180                                          # resolved after outdir(): still recorded
    rec = provenance.record(pressures=(1.0, 2.0))
    assert set(provenance.VOLATILE) <= set(rec) and rec['numpy'] == np.__version__
    assert rec['args']['nphot'] == 180 and rec['args']['preset'] == 'published' and rec['pressures'] == [1.0, 2.0]
    assert rec['inputs'] == cli.preset('published') and rec['constants']['P_REF'] == P_REF
    s = provenance.stable(dict(rec, source=dict(provenance=rec)))
    assert not set(provenance.VOLATILE) & set(s) and 'outdir' not in s['args'] and 'commit' not in s['source']['provenance']
    z = provenance.npz(); assert json.loads(str(z[provenance.KEY]))['args']['nphot'] == 180
    assert provenance.md_line().startswith('<!-- provenance: {') and provenance.text_line().startswith('# provenance: {')
    assert provenance.with_record(dict(a=1))['a'] == 1
    old = provenance.with_record({provenance.KEY: dict(commit='OLD'), 'a': 1}, pressures=(3.0,))   # merged summary of an older run
    assert old[provenance.KEY]['pressures'] == [3.0] and old[provenance.KEY]['commit'] != 'OLD' and list(old) == [provenance.KEY, 'a']
    np.savez(tmp_path/'x.npz', a=np.ones(2), **z); assert json.loads(str(np.load(tmp_path/'x.npz')[provenance.KEY]))['script']
    provenance.set_args(argparse.Namespace(outdir=None))
