"""The muon's own SiPM hit vs the protective-window thickness and index (stdout, muon_window_cluster.png/.csv).

A vertical beta = 1 muon crosses the silicon, then the window below it; its Cherenkov light is totally reflected
at the window bottom face and lands on the silicon in an annulus around the impact point. Photons are counted with
the FT x QE (ARC SiPM QE) and the pure FT weighting; the direct hit of the muon adds constants.MIP_DIRECT_PE.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, sim  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band, theta_c  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, HC, MIP_DIRECT_PE, MIRROR_D, MIRROR_R, SENSOR_HALF_2IN, WINDOW_N, WINDOW_T  # noqa: E402
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window  # noqa: E402
from xerich.media import ConstantIndex, xenon  # noqa: E402
from xerich.plotting import INK, MUTED, plt, save  # noqa: E402

THICKNESS = np.round(np.arange(0.1, 0.501, 0.05), 3)   # window thickness grid [mm]
INDICES = (1.45, 1.5, WINDOW_N)
_CFG = {}


def run_config(task):
    """Cluster photons (FT x QE, FT) and spot radii for one (index, thickness, seed)."""
    n_w, t, seed = task
    lo, hi = qe_band()
    ph, e = sim.window_cluster(_CFG['layout'], np.random.default_rng(seed), PhotonSpectrum(ConstantIndex(n_w), lo, hi),
                               _CFG['n'], Window(n_w, t))
    q = sim.cluster_stats(ph, e, weight=lambda lam: qe_arc(HC/lam)); f = sim.cluster_stats(ph, e)
    return n_w, t, q, f


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nphot', type=float, default=2e5, help='sampled photons per configuration (default 2e5)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed')
    p.add_argument('--nproc', type=int, default=8, help='worker processes (default 8)')
    args = cli.pressure(p).parse_args(); out = cli.outdir(args)
    _CFG.update(layout=Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(args.pressure), Aerogel(AEROGEL_N_AIR)),
                n=int(args.nphot))
    cfg = [(n_w, t) for n_w in INDICES for t in THICKNESS]
    seeds = np.random.SeedSequence(args.seed).spawn(len(cfg))
    with mp.get_context('fork').Pool(args.nproc) as pool:
        res = pool.map(run_config, [(n_w, t, s) for (n_w, t), s in zip(cfg, seeds)])
    rows = [(n_w, t, q['n'], f['n'], q['n']+MIP_DIRECT_PE, q['r_mean'], q['r_rms'], f['r_max'],
             t*np.tan(theta_c(n_w)), 2*t*np.tan(theta_c(n_w))) for n_w, t, q, f in res]
    head = ('window_n', 'window_t_mm', 'photons_ftqe', 'photons_ft', 'pe_with_direct_hit', 'spot_r_mean_mm', 'spot_r_rms_mm',
            'spot_r_max_mm', 'annulus_r_in_mm', 'annulus_r_out_mm')
    with open(out/'muon_window_cluster.csv', 'w') as fo:
        fo.write(','.join(head)+'\n')
        for r in rows:
            fo.write(','.join('%.6g' % v for v in r)+'\n')
    print("muon's own SiPM hit: window photons on the silicon per muon (plus %g fired cell from the silicon crossing)" % MIP_DIRECT_PE)
    print(' n     t [mm]  FT x QE   FT      p.e. incl. direct   spot <r>  rms r  max r [mm]  annulus (t, 2t) tan(theta_c) [mm]')
    for r in rows:
        print(' %.2f  %.2f    %6.2f  %6.2f   %6.2f              %.3f     %.3f  %.3f       %.3f-%.3f' % r)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    styles = {1.45: ':', 1.5: '--', WINDOW_N: '-'}
    R = np.array(rows)
    for n_w in INDICES:
        m = R[:, 0] == n_w
        ax[0].plot(R[m, 1], R[m, 4], color=INK, ls=styles[n_w], marker='o', ms=3, label='n = %.2f, FT x QE + direct hit' % n_w)
        ax[0].plot(R[m, 1], R[m, 3], color=MUTED, ls=styles[n_w], lw=1, label='n = %.2f, FT (no QE)' % n_w)
        ax[1].plot(R[m, 1], R[m, 5], color=INK, ls=styles[n_w], marker='o', ms=3, label='n = %.2f, mean r' % n_w)
        ax[1].fill_between(R[m, 1], R[m, 8], R[m, 9], color=MUTED, alpha=0.12)
    ax[0].axvline(WINDOW_T, color=MUTED, lw=.8, ls=':'); ax[1].axvline(WINDOW_T, color=MUTED, lw=.8, ls=':')
    ax[0].set_ylabel('photons (p.e.) per muon on the silicon'); ax[0].set_title('Cluster size vs window thickness', fontsize=10)
    ax[1].set_ylabel('distance from the impact point [mm]')
    ax[1].set_title('Spot: mean radius; band: annulus t tan θc to 2t tan θc', fontsize=10)
    for a in ax:
        a.set_xlabel('window thickness t [mm]'); a.grid(alpha=.3); a.legend(fontsize=8); a.set_ylim(0, None)
    fig.suptitle("The muon's own SiPM hit: Cherenkov light in the protective window, trapped by total internal reflection",
                 fontsize=11.5, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    print('figure:', save(out, 'muon_window_cluster.png', dpi=130, fig=fig).name)


if __name__ == '__main__':
    main()
