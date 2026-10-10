"""Shared matplotlib setup (non-interactive Agg backend), colours and axis helpers."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import HC  # noqa: E402

# Setup drawings
INK = '#1f2933'; MUTED = '#6b7280'; GAS = '#2563eb'; AEROGEL = '#d97706'; MIRROR = '#475569'
LIGHT = '#9ca3af'; GRID = '#d1d5db'; SCINT = '#10b981'; SCINT_EDGE = '#047857'

# Physics plots
PRESSURE = {2: '#1f6fb4', 3: '#d1495b'}
SPECIES = {'pi': '#2a9d8f', 'K': '#e76f51', 'p': '#6d597a'}
SQRTS_KEYS = ['91', '160', '240', '365']
SQRTS = {'91': '#264653', '160': '#2a9d8f', '240': '#e9c46a', '365': '#e76f51'}
SQRTS_SHORT = {'91': 'Z', '160': 'WW', '240': 'ZH', '365': 'tt̄'}
SQRTS_LONG = {'91': 'Z (91 GeV)', '160': 'WW (160 GeV)', '240': 'ZH (240 GeV)', '365': 'tt̄ (365 GeV)'}


def lambda_axis(ax, ticks=(200, 250, 300, 400, 600)):
    """Secondary top axis in wavelength [nm] for an axis in photon energy [eV]."""
    top = ax.secondary_xaxis('top', functions=(lambda e: HC/np.maximum(e, 1e-3), lambda l: HC/np.maximum(l, 1e-3)))
    top.set_xticks(list(ticks)); top.set_xlabel('λ [nm]')
    return top


def save(outdir, name, dpi=140, fig=None):
    """Save the figure (default: current) as outdir/name; returns the path."""
    path = Path(outdir)/name
    (fig or plt.gcf()).savefig(path, dpi=dpi)
    return path
