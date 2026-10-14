"""Size of the board-reflected light in configuration 4 (proximity focusing, no mirror), which cosmic_rates.py does not model.

The light reflected by the board goes back up to the tile and returns to the board only through Rayleigh back-scatter in
the tile and Fresnel reflections at its faces. Geometry of studies/proximity_config4.py (its mirror-image frame, no
window), board footprint, Xe at P_REF, constant n = 1.03 tile with Rayleigh scattering, the default response and board
reflectances (cli.BASELINE), two board reflections with their partial Fresnel reflections; vertical beta = 1 muons
uniform within +-12 mm of the tile centre. Prints, per gap, the photons on active pads of the ring and those returned
by the reflection, and the ring hits and the extra fired pads, without the muon's pad (stdout).
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from xerich import cli, cosmic
from xerich.analysis import pad_hits
from xerich.cherenkov import qe_band, response
from xerich.constants import HC, P_REF
from xerich.layout import Board

spec = importlib.util.spec_from_file_location('px', Path(__file__).resolve().parents[3]/'studies'/'proximity_config4.py')
px = importlib.util.module_from_spec(spec); spec.loader.exec_module(px)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=int, default=40, help='vertical muons per gap (default 40)')
    p.add_argument('--nphot', type=int, default=20000, help='tile photons per muon (default 20000)')
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    b = Board(); rng = np.random.default_rng(a.seed); qe = response(cli.BASELINE['response'])
    refl = (cli.BASELINE['board_r_spec'], cli.BASELINE['board_r_diff'])
    for gap in px.GAPS:
        base = px.make_layout('xe%g' % P_REF, gap, window=False)
        lay = px.ProxLayout(base.gas, gap, base.tile_medium, base.tile_scatter, None, sensor_half=b.half); sc = lay.scene()
        sp = px.spectra_for(lay, qe_band()); R = []
        for _ in range(a.tracks):
            p0 = np.r_[rng.uniform(-12, 12, 2), 0.0]
            ph, e, _ = px.trace_track(lay, rng, p0, np.array([0, 0, 1.0]), sp, 0, a.nphot, 0)
            det = e['reason'] == 0; w = ph['weight'][e['i'][det]]*e['w'][det]; lam = ph['lam'][e['i'][det]]
            mp = b.pad_index(p0[0], p0[1]); pad = b.pad_index(e['p'][det, 0], e['p'][det, 1]); on = (pad >= 0)&(pad != mp)
            q = w*qe(HC/lam)
            _, pe, le, we = cosmic.board_reflections(sc, rng, e['p'][det], e['d'][det], lam, w, *refl, max_reflections=2)
            pr = b.pad_index(pe[:, 0], pe[:, 1]); onr = (pr >= 0)&(pr != mp); qr = we*qe(HC/le)
            h0 = pad_hits(np.zeros(on.sum(), int), pad[on], q[on], 1)[0][0]
            h1 = pad_hits(np.zeros(on.sum()+onr.sum(), int), np.r_[pad[on], pr[onr]], np.r_[q[on], qr[onr]], 1)[0][0]
            R.append((q[on].sum(), qr[onr].sum(), h0, h1-h0))
        R = np.array(R).mean(0)
        print('gap %g mm (%d muons x %d photons, seed %d): ring photons on active pads %.2f, returned by the board reflection %.4f '
              '(%.2f %%); ring hits %.2f, extra fired pads %.4f' % (gap, a.tracks, a.nphot, a.seed, R[0], R[1], 100*R[1]/R[0], R[2], R[3]))


if __name__ == '__main__':
    main()
