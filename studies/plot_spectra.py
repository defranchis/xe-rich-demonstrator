"""Xe dispersion, Frank-Tamm spectra for pi/K, integrated yield vs momentum, chromatic range (stdout, xe_cherenkov_spectra.png)."""
import numpy as np

from xerich import cli
from xerich.cherenkov import dN_dxdE, energy_grid, photon_yield, theta_c, theta_c_track
from xerich.constants import M_K, M_P, M_PI
from xerich.media import n_xe
from xerich.plotting import PRESSURE, SPECIES, lambda_axis, plt, save


def main():
    out = cli.outdir(cli.parser(__doc__).parse_args())
    E = energy_grid(165, 650, 800)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) dispersion
    a = ax[0, 0]
    L = np.linspace(160, 650, 500)
    for P in (2, 3):
        a.plot(L, (n_xe(L, P)-1)*1e3, color=PRESSURE[P], lw=2, label=f'Xe {P} bar, 20°C')
    a.axvline(147, color='gray', ls=':'); a.text(150, 3.9, '147 nm\nresonance', fontsize=8, color='gray')
    a.set_xlabel('λ [nm]'); a.set_ylabel('(n−1) × 10³'); a.set_ylim(1, 4.3); a.set_xlim(140, 650)
    a.legend(); a.set_title('Refractive index (Bideau-Mehu 1981, density-scaled)')
    # (b), (c) spectra at fixed momentum
    for P in (2, 3):
        a = ax[0, 1] if P == 2 else ax[1, 0]
        for p, ls in ((5, ':'), (10, '--'), (20, '-'), (40, '-.')):
            a.plot(E, dN_dxdE(E, p, M_PI, P), color=SPECIES['pi'], ls=ls, lw=1.6, label=f'π {p} GeV')
            a.plot(E, dN_dxdE(E, p, M_K, P), color=SPECIES['K'], ls=ls, lw=1.6, label=f'K {p} GeV')
        a.set_xlabel('photon energy [eV]'); a.set_ylabel('d²N/dx dE  [cm⁻¹ eV⁻¹]')
        a.set_title(f'Frank–Tamm spectrum, Xe {P} bar'); a.set_xlim(E[0], E[-1])
        lambda_axis(a)
        a.legend(ncol=2, fontsize=7.5); a.grid(alpha=.3)
    # (d) integrated yield vs momentum
    a = ax[1, 1]; p = np.logspace(np.log10(1.5), np.log10(100), 400)
    for P in (2, 3):
        for m, c, nm in ((M_PI, SPECIES['pi'], 'π'), (M_K, SPECIES['K'], 'K'), (M_P, SPECIES['p'], 'p')):
            Ey = energy_grid(200, 600, 400)
            N = [photon_yield(Ey, pp, m, P) for pp in p]
            a.plot(p, N, color=c, ls='-' if P == 3 else '--', lw=1.8, label=f'{nm}, {P} bar')
    a.set_xscale('log'); a.set_xlabel('p [GeV]'); a.set_ylabel('N_γ / cm  (200–600 nm, emitted)')
    a.set_title('Integrated yield (before window / QE)'); a.legend(fontsize=7.5, ncol=2); a.grid(alpha=.3, which='both')
    plt.tight_layout(); save(out, 'xe_cherenkov_spectra.png', dpi=140)
    # numbers
    Ey = energy_grid(200, 600, 400)
    for P in (2, 3):
        for pp in (5, 10, 20, 45):
            n300 = n_xe(300, P)
            print(P, pp, 'pi %.3f' % photon_yield(Ey, pp, M_PI, P), 'K %.3f' % photon_yield(Ey, pp, M_K, P),
                  'dtheta(pi-K) mrad@300nm %.2f' % (1e3*(theta_c_track(n300, pp, M_PI)-theta_c_track(n300, pp, M_K))))
        for lo, hi in ((200, 600), (250, 500), (300, 600)):
            print('  chromatic theta_c range beta=1, %d-%d nm: %.1f -> %.1f mrad' % (lo, hi, 1e3*theta_c(n_xe(hi, P)), 1e3*theta_c(n_xe(lo, P))))


if __name__ == '__main__':
    main()
