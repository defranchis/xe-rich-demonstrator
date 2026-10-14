"""Legacy scripts vs studies drivers.

The raytrace check always runs (about 1 s). XERICH_FULL_REGRESSION=1 runs every script
and driver into a temporary directory and compares them (about 2 min);
XERICH_BASELINE and XERICH_NEW compare two existing output directories instead.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import regression  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_raytrace_stdout_identical(tmp_path):
    old = subprocess.run([sys.executable, str(REPO/'legacy'/'raytrace.py')], cwd=tmp_path, capture_output=True, text=True, check=True)
    new = subprocess.run([sys.executable, str(REPO/'studies'/'raytrace.py'), *regression.LEGACY_ARGS['raytrace']], cwd=tmp_path,
                         capture_output=True, text=True, check=True)
    assert old.stdout == new.stdout


def test_provenance_key():
    """regression.py imports no xerich on purpose: its copy of the provenance key must match."""
    from xerich import provenance
    assert regression.PROVENANCE_KEY == provenance.KEY


@pytest.mark.skipif(os.environ.get('XERICH_FULL_REGRESSION') != '1', reason='set XERICH_FULL_REGRESSION=1')
def test_full_gate(tmp_path):
    assert regression.run('legacy', tmp_path/'baseline', 16, 200) == 0
    assert regression.run('new', tmp_path/'new', 16, 200) == 0
    assert regression.compare(tmp_path/'baseline', tmp_path/'new') == 0


@pytest.mark.skipif(not (os.environ.get('XERICH_BASELINE') and os.environ.get('XERICH_NEW')),
                    reason='set XERICH_BASELINE and XERICH_NEW')
def test_existing_outputs():
    assert regression.compare(os.environ['XERICH_BASELINE'], os.environ['XERICH_NEW']) == 0
