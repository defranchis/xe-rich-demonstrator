"""Frank-Tamm spectra vs wavelength at fixed momentum and averaged over FCC-ee pi/K tracks.

Reads mom_<key>.npz from gen_pk.py (stdout, xe_cherenkov_lambda.png).
"""
from pathlib import Path

import numpy as np

from xerich import cli
from xerich.cherenkov import dN_dxdlam
from xerich.constants import M_K, M_PI
from xerich.plotting import SPECIES, SQRTS, SQRTS_KEYS, SQRTS_SHORT, plt, save

NSAMPLE = 20000              # tracks averaged per species and energy


def main():
    p = cli.xe_index(cli.parser(__doc__))
    p.add_argument('--momdir', type=Path, default=None, help='directory with mom_<key>.npz (default: outdir)')
    args = p.parse_args(); out = cli.outdir(args); momdir = args.momdir if args.momdir is not None else out
    model = args.xe_index
    L = np.linspace(165, 650, 600)
    D = {k: np.load(momdir/f'mom_{k}.npz') for k in SQRTS_KEYS}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for i, P in enumerate((2, 3)):
        a = ax[0, i]
        for pm, ls in ((5, ':'), (10, '--'), (20, '-'), (45, '-.')):
            a.plot(L, dN_dxdlam(L, pm, M_PI, P, model=model), color=SPECIES['pi'], ls=ls, lw=1.6, label=f'π {pm} GeV')
            a.plot(L, dN_dxdlam(L, pm, M_K, P, model=model), color=SPECIES['K'], ls=ls, lw=1.6, label=f'K {pm} GeV')
        a.set_title(f'Xe {P} bar, fixed momentum'); a.set_ylabel('d²N/dx dλ [cm⁻¹ nm⁻¹]')
        a.set_yscale('log'); a.set_ylim(1e-4, 2e-1); a.legend(ncol=2, fontsize=7.5); a.grid(alpha=.3, which='both')
    rng = np.random.default_rng(1)
    for j, (sp, m, nm) in enumerate((('pi', M_PI, 'π±'), ('K', M_K, 'K±'))):
        a = ax[1, j]
        for k in SQRTS_KEYS:
            pp = D[k][sp]; pp = rng.choice(pp, size=min(len(pp), NSAMPLE), replace=False)
            for P, ls in ((2, '--'), (3, '-')):
                s = np.mean([dN_dxdlam(L, x, m, P, model=model) for x in pp], axis=0)
                a.plot(L, s, color=SQRTS[k], ls=ls, lw=1.6, label=f'{SQRTS_SHORT[k]}, {P} bar')
        a.set_title(f'Averaged over FCC-ee {nm} (p>0.3 GeV, |cosθ|<0.9)', fontsize=10)
        a.set_ylabel('⟨d²N/dx dλ⟩ per track [cm⁻¹ nm⁻¹]'); a.set_yscale('log'); a.legend(ncol=2, fontsize=7.5); a.grid(alpha=.3, which='both')
    for a in ax[1]:
        a.set_xlabel('λ [nm]')
    for a in ax.flat:
        a.set_xlim(165, 650)
    plt.tight_layout(); save(out, 'xe_cherenkov_lambda.png', dpi=140)
    for P in (2, 3):
        s = dN_dxdlam(L, 45, M_PI, P, model=model)
        print(P, ['%d:%.4f' % (l, s[np.argmin(abs(L-l))]) for l in (180, 200, 250, 300, 400, 500, 600)])
        band, full = (L >= 200)&(L <= 300), (L >= 200)&(L <= 600)
        print('  frac(200-300)/(200-600)=%.2f' % (np.trapezoid(s[band], L[band])/np.trapezoid(s[full], L[full])))


if __name__ == '__main__':
    main()
