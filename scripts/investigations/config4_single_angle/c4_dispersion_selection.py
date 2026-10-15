"""Configuration 4: change of the >= 80 % ring rate when the containment takes the aerogel dispersion into account.

cosmic_rates.py selects the configuration-4 (proximity focusing) >= 80 % rings with sim.proximity_fractions at one Cherenkov
angle per muon, that of the tile index n = 1.03 (AEROGEL_N_AIR, the index at 400 nm) and the Xe index at 400 nm; the
aerogel dispersion (--aerogel-index single-pole) enters only the photon yield. On the telescope muons of a cosmic_rates.py
run (cosmic_rates_c4.npz: c, phi, x0, y0, p of the runs c4_xe2_g30/50/70_plain, Xe 2 bar, plain tile, board footprint of
half side 25.8 mm), the containment fd is recomputed
  (a) at n = 1.03 (must reproduce the >= 80 % selection of the npz, br_tile_fd),
  (b) with the dispersion: fd = sum_j W_j fa(n_a(E_j), n_Xe(E_j)) / sum_j W_j over NE energies uniform on the QE band, with
      W_j = (1 - 1/(n_a(E_j) beta)^2) x S13361 response(E_j) (Frank-Tamm x response at the muon's velocity),
and the >= 80 % ring rate of (b) is compared with that of (a), paired on the same muons (Guan spectrum weights). The error
is the seed-to-seed standard deviation of the relative change over sqrt(number of seeds). Only muons with fd > 0.6 at
n = 1.03 are recomputed: rings far below 0.8 at n = 1.03 cannot reach it. Writes OUT and prints the same lines.
"""
import argparse
import multiprocessing as mp
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np  # noqa: E402

from xerich import muons, sim  # noqa: E402
from xerich.cherenkov import beta as beta_of, qe_band, response  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, BOARD_PACKAGE, COSMIC_I0, HC, M_MU, MUON_THMAX  # noqa: E402
from xerich.layout import Aerogel, Sensor  # noqa: E402
from xerich.media import aerogel_medium, n_xe  # noqa: E402

HALF = BOARD_PACKAGE; P = 2.0        # board footprint half side (one package side) [mm], Xe pressure [bar] of the runs read
NE = 12                              # energies on the QE band (set by --ne)
na = ngs = R = None                  # tile and Xe indices and S13361 response at the energies (set by setup)


def setup(ne):
    """Tile index (single pole, n = 1.03 at 400 nm), Xe index and S13361 response at ne energies on the QE band."""
    global NE, na, ngs, R
    NE = ne
    idx = aerogel_medium('single-pole', AEROGEL_N_AIR, 400.0).index
    E = np.linspace(*qe_band(), NE+2)[1:-1]; lam = HC/E
    na = np.array([float(idx(l)) for l in lam]); ngs = np.array([float(n_xe(l, P)) for l in lam]); R = response('s13361')(E)


def block(arg):
    gap, u, p0, b = arg
    th0 = np.arccos(np.minimum(1, 1/(AEROGEL_N_AIR*b)))
    f0 = sim.proximity_fractions(Aerogel(AEROGEL_N_AIR), Sensor(HALF), gap, float(n_xe(400.0, P)), u, p0, th0)[0]
    num = np.zeros(len(b)); den = np.zeros(len(b))
    for j in range(NE):
        th = np.arccos(np.minimum(1, 1/(na[j]*b)))
        W = np.where(na[j]*b > 1, 1-1/(na[j]*b)**2, 0.0)*R[j]
        fa = sim.proximity_fractions(Aerogel(na[j]), Sensor(HALF), gap, ngs[j], u, p0, th)[0]
        num += W*fa; den += W
    return f0, np.where(den > 0, num/np.maximum(den, 1e-300), 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='output directory of cosmic_rates.py with cosmic_rates_c4.npz (default inputs, group c4)')
    ap.add_argument('out', help='text file written with the results')
    ap.add_argument('--ne', type=int, default=12, help='energies on the QE band (default 12)')
    ap.add_argument('--nproc', type=int, default=4, help='worker processes (default 4)')
    args = ap.parse_args()
    setup(args.ne)
    z = np.load(os.path.join(args.run_dir, 'cosmic_rates_c4.npz')); sp = muons.MuonSpectrum('guan')
    F = muons.flux_through_square(2*25.0e-3, MUON_THMAX)*3600
    lines = ['energies %d on the QE band %.3f-%.3f eV; single-pole n_a: %s; FT x S13361 weighted mean n_a (beta = 1) %.5f'
             % (NE, *qe_band(), ' '.join('%.4f' % x for x in na), np.sum(na*(1-1/na**2)*R)/np.sum((1-1/na**2)*R))]
    for g in (30, 50, 70):
        nm = 'c4_xe2_g%d_plain' % g
        c = z[nm+'/c']; phi = z[nm+'/phi']; x0 = z[nm+'/x0']; y0 = z[nm+'/y0']; p = z[nm+'/p']; n = len(c)
        tel = z[nm+'/top']&z[nm+'/bot']; fd_npz = z[nm+'/br_tile_fd']
        cand = tel&(fd_npz > 0.6)                                      # rings far below 0.8 at n = 1.03 cannot reach it
        k = np.flatnonzero(cand); s = np.sqrt(1-c[k]**2)
        u = np.stack([s*np.cos(phi[k]), s*np.sin(phi[k]), -c[k]], 1); p0 = np.stack([x0[k], y0[k], np.zeros(len(k))], 1); b = beta_of(p[k], M_MU)
        blocks = [(float(g), u[i:i+5000], p0[i:i+5000], b[i:i+5000]) for i in range(0, len(k), 5000)]
        with mp.get_context('fork').Pool(args.nproc) as pool:
            res = pool.map(block, blocks)
        f0 = np.concatenate([r[0] for r in res]); fdisp = np.concatenate([r[1] for r in res])
        w = sp.intensity_above(c)/(COSMIC_I0*c*c); seed = z[nm+'/seed']; ns = int(seed.max())+1
        sel0 = np.zeros(n, bool); sel0[k] = (f0 >= 0.8)&(b*AEROGEL_N_AIR > 1)
        seld = np.zeros(n, bool); seld[k] = fdisp >= 0.8
        selnpz = tel&(fd_npz >= 0.8)&(z[nm+'/br_tile_th'] > 0)
        r0 = np.array([F*np.sum((w*sel0)[seed == q])/np.sum(seed == q) for q in range(ns)])
        rd = np.array([F*np.sum((w*seld)[seed == q])/np.sum(seed == q) for q in range(ns)])
        d = 100*(rd-r0)/r0.mean(); err = d.std(ddof=1)/np.sqrt(ns)
        lines.append('gap %d: %d telescope muons, %d with fd > 0.6; n = 1.03 selection reproduces the npz: %s (%d rings); '
                     'rings /h %.4f (n = 1.03) -> %.4f (dispersive, FT x S13361): %+.3f +- %.3f %% (paired over %d seeds); rings %d -> %d'
                     % (g, tel.sum(), len(k), bool(np.array_equal(sel0, selnpz)), sel0.sum(), F*np.sum(w*sel0)/n, F*np.sum(w*seld)/n, d.mean(), err, ns,
                        sel0.sum(), seld.sum()))
        print(lines[-1], flush=True)
    open(args.out, 'w').write('\n'.join(lines)+'\n')
    print(lines[0])


if __name__ == '__main__':
    main()
