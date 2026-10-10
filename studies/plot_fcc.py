"""FCC-ee pi/K momentum spectra vs the Xe thresholds and track-averaged photon spectra.

Reads mom_<key>.npz from gen_pk.py (stdout table, xe_fccee_pik.png).
"""
from pathlib import Path

import numpy as np

from xerich import cli
from xerich.cherenkov import dN_dxdE, energy_grid, p_threshold
from xerich.constants import M_K, M_PI
from xerich.media import n_xe
from xerich.plotting import SQRTS, SQRTS_KEYS, SQRTS_LONG, lambda_axis, plt, save

NSAMPLE = 20000              # tracks averaged per species and energy
LAM_TH = 300                 # wavelength for the thresholds [nm]


def pth(m, P, lam=LAM_TH):
    return p_threshold(m, n_xe(lam, P))


def main():
    p = cli.parser(__doc__)
    p.add_argument('--momdir', type=Path, default=None, help='directory with mom_<key>.npz (default: outdir)')
    args = p.parse_args(); out = cli.outdir(args); momdir = args.momdir if args.momdir is not None else out
    keys = SQRTS_KEYS
    D = {k: np.load(momdir/f'mom_{k}.npz') for k in keys}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    bins = np.logspace(np.log10(0.3), np.log10(150), 70); bc = np.sqrt(bins[1:]*bins[:-1]); dl = np.diff(np.log10(bins))
    for j, (sp, m, nm) in enumerate((('pi', M_PI, 'π±'), ('K', M_K, 'K±'))):
        a = ax[0, j]
        for k in keys:
            h, _ = np.histogram(D[k][sp], bins); a.step(bc, h/D[k]['N']/dl, where='mid', color=SQRTS[k], lw=1.8, label=SQRTS_LONG[k])
        for P, ls in ((2, '--'), (3, '-')):
            a.axvline(pth(m, P), color='k', ls=ls, lw=1)
        a.set_xscale('log'); a.set_xlabel('p [GeV]'); a.set_ylabel(f'{nm} per event / dlog₁₀p')
        a.set_title(f'{nm} momentum, |cosθ|<0.9  (lines: Xe threshold, -- 2 bar, — 3 bar)', fontsize=9.5)
        a.legend(fontsize=8); a.grid(alpha=.3, which='both')
    E = energy_grid(200, 600, 300)
    for j, (sp, m, nm) in enumerate((('pi', M_PI, 'π±'), ('K', M_K, 'K±'))):
        a = ax[1, j]
        for k in keys:
            pp = D[k][sp]; pp = pp[pp > 1.0]
            pp = np.random.default_rng(1).choice(pp, size=min(len(pp), NSAMPLE), replace=False)
            for P, ls in ((2, '--'), (3, '-')):
                s = np.mean([dN_dxdE(E, x, m, P) for x in pp], axis=0)
                a.plot(E, s, color=SQRTS[k], ls=ls, lw=1.6, label=f'{SQRTS_LONG[k]}, {P} bar')
        a.set_xlabel('photon energy [eV]'); a.set_ylabel('⟨d²N/dx dE⟩ per track [cm⁻¹ eV⁻¹]')
        a.set_title(f'Photon spectrum averaged over {nm} with p>1 GeV', fontsize=10)
        lambda_axis(a)
        a.legend(fontsize=6.5, ncol=2); a.grid(alpha=.3)
    plt.tight_layout(); save(out, 'xe_fccee_pik.png', dpi=140)
    rows = []
    for k in keys:
        for sp, m in (('pi', M_PI), ('K', M_K)):
            pp = D[k][sp]; N = D[k]['N']
            r = [k, sp, len(pp)/N, np.median(pp), np.mean(pp)]
            for P in (2, 3):
                r += [np.mean(pp > pth(m, P)), np.mean((pp > pth(M_PI, P))&(pp < pth(M_K, P))) if sp == 'K' else np.nan]
            r += [np.mean(pp > 20), np.mean(pp > 40)]
            rows.append(r)
    print('sqrt_s sp  n/ev  median mean | 2bar: f>th  f(thr-mode) | 3bar: f>th f(thr-mode) | f>20 f>40')
    for r in rows:
        print('%4s %3s %6.2f %6.2f %6.2f | %.3f %.3f | %.3f %.3f | %.3f %.3f' % tuple(r))


if __name__ == '__main__':
    main()
