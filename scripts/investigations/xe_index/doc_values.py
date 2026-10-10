"""Design-doc values that depend on the Xe index, for each index model (stdout).

Sections 1.1-1.3, 2.2 and 6.3 of docs/design_options.md, the Xe density against NIST, the Boerzsoenyi index against Koch, the aerogel
theta_c spread of the dispersion options, and the single-pass Rayleigh fractions of the tile.
"""
import argparse

import numpy as np

from xerich.cherenkov import (PhotonSpectrum, dN_dxdE, energy_grid, frank_tamm, p_threshold, qe_arc, qe_band,
                              theta_c, theta_c_track)
from xerich.constants import AEROGEL_N_AIR, AEROGEL_T, HC, M_K, M_P, M_PI, MIRROR_R, P0, R_GAS
from xerich.media import (XE_INDEX_MODELS, Rayleigh, c4f10, molar_density, n_aerogel_k4geo, n_aerogel_single_pole, n_air,
                          n_xe, xenon)

F_ROUND = round(MIRROR_R/2)      # focal length of the printed ring radii [mm]
POROSITY_DOC = 0.935             # section 6.3 pore fraction of an n = 1.03 tile, relative to air (design-doc convention, not AEROGEL_POROSITY)
# NIST Chemistry Webbook, Xe isotherms (fluid.cgi, C7440633): (T [K], P [MPa], density [mol/m^3])
NIST_XE = ((273.15, 0.1, 44.3311), (273.0, 0.1, 44.3559), (293.15, 0.1, 41.2539), (293.15, 0.2, 82.9694),
           (293.15, 0.3, 125.160))
# Koch (1949) as tabulated by Leonard, At. Data Nucl. Data Tables 14, 21 (1974), Table II(e), ref. 13: (air wavelength at 15 degC [nm], (n-1) 1e6 at 273.16 K, 1 atm)
KOCH_XE = ((230.209, 855.19), (234.555, 846.64), (237.999, 840.25), (244.6905, 829.07), (246.4068, 826.40), (257.6295, 810.78),
           (267.4990, 799.23), (275.2775, 791.24), (275.9712, 790.60), (285.6969, 781.86), (289.3595, 778.85), (292.5406, 776.37),
           (334.1478, 751.43), (380.1658, 734.30), (410.8066, 726.23), (491.6036, 712.41), (567.717, 704.77), (607.262, 701.88),
           (612.327, 701.57), (623.437, 700.91))
BANDS = ((320, 450), (300, 600), (250, 600), (200, 600))


def band_stats(P, model, lo, hi, n=4000):
    """Saturated N/cm and the per-photon rms of theta_c (beta = 1) over lo-hi nm."""
    E = energy_grid(lo, hi, n); w = dN_dxdE(E, 1e4, M_PI, P, model=model); th = theta_c(n_xe(HC/E, P, model=model))
    N = np.trapezoid(w, E); m = np.trapezoid(w*th, E)/N
    return N, np.sqrt(np.trapezoid(w*(th-m)**2, E)/N)


def n_air_edlen(lam_nm):
    """Index of standard air (15 degC, 101325 Pa, Edlen, Metrologia 2, 71 (1966)); wavenumber taken from lam_nm."""
    s2 = (1e3/np.asarray(lam_nm, float))**2
    return 1+1e-8*(8342.13+2406030/(130-s2)+15997/(38.9-s2))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for model in XE_INDEX_MODELS:
        print('== Xe index model %s (20 degC)' % model)
        for P in (2, 3):
            n3, n4 = n_xe(300, P, model=model), n_xe(400, P, model=model)
            print(' %d bar: n-1(300) %.4e  theta_c(400) %.2f mrad  ring f tan(theta_c(400)) at f = %d: %.2f mm' % (
                P, n3-1, 1e3*theta_c(n4), F_ROUND, F_ROUND*np.tan(theta_c(n4))))
            print('        thresholds at 300 nm: pi %.3f  K %.3f  p %.3f GeV' % tuple(p_threshold(m, n3) for m in (M_PI, M_K, M_P)))
            for p in (20, 45):
                print('        pi-K separation at %d GeV, 300 nm: %.2f mrad' % (
                    p, 1e3*(theta_c_track(n3, p, M_PI)-theta_c_track(n3, p, M_K))))
            for lo, hi in BANDS:
                N, rms = band_stats(P, model, lo, hi)
                print('        %d-%d nm: N/cm %.3f  chromatic rms %.3f mrad  rms/sqrt(N per cm) %.3f mrad' % (lo, hi, N, 1e3*rms, 1e3*rms/np.sqrt(N)))
            L = np.linspace(165, 650, 600); s = dN_dxdE(HC/L, 45, M_PI, P, model=model)*HC/L**2
            band, full = (L >= 200)&(L <= 300), (L >= 200)&(L <= 600)
            print('        fraction of the 200-600 nm photons below 300 nm (45 GeV pi): %.3f' % (
                np.trapezoid(s[band], L[band])/np.trapezoid(s[full], L[full])))
            ng = n_xe(400, P, model=model); na = n_air(400.0)
            for ref, nref in (('air', na), ('vacuum', 1.0)):
                dn = POROSITY_DOC*(ng-nref)
                print('        aerogel pore shift at 400 nm vs %-6s: dn %+.5f  dtheta_c %+.2f mrad' % (
                    ref, dn, 1e3*(theta_c(AEROGEL_N_AIR+dn)-theta_c(AEROGEL_N_AIR))))
        n = n_xe(400, 3, model=model)
        print(' theta_c(400 nm, 3 bar) / bm1981: %.5f' % (theta_c(n)/theta_c(n_xe(400, 3, model='bm1981'))))
    print('\n== bm1981 relative to borzsonyi (20 degC)')
    L = np.array([264.0, 400.0, 954.0])
    print(' (n-1) ratio at 264 / 400 / 954 nm: %.4f / %.4f / %.4f' % tuple((n_xe(L, 3, model='bm1981')-1)/(n_xe(L, 3)-1)))
    for P in (2, 3):
        print(' %d bar chromatic rms ratio, %s' % (P, '  '.join('%d-%d nm %.3f' % (lo, hi, band_stats(P, 'bm1981', lo, hi)[1]/band_stats(P, 'borzsonyi', lo, hi)[1])
                                                         for lo, hi in BANDS)))
    print('\n== Boerzsoenyi vs Koch at 273.16 K, 1 atm (Leonard 1974); formula in vacuum wavelengths, Koch air wavelengths converted')
    la, nk = np.array(KOCH_XE).T; lv = la*n_air_edlen(la); nk = nk*1e-6
    r = (n_xe(lv, P0/1e5, 273.16)-1)/nk; r_air = (n_xe(la, P0/1e5, 273.16)-1)/nk
    print(' ratio (n-1)_Boerzsoenyi/(n-1)_Koch: ' + ' '.join('%.1f:%.5f' % x for x in zip(lv, r)))
    print(' air-to-vacuum conversion: lambda +%.3f to +%.3f nm, ratio changed by %+.4f %% (230 nm) to %+.4f %% (623 nm)' % (
        (lv-la).min(), (lv-la).max(), 100*(r[0]/r_air[0]-1), 100*(r[-1]/r_air[-1]-1)))
    for lmin in (290, 330):
        print(' max |ratio - 1| above %d nm: %.3f %%' % (lmin, 100*np.abs(r[lv > lmin]-1).max()))
    r264 = np.interp(264.0, lv, r); n3 = n_xe(264.0, 3)
    print(' ratio - 1 at 264 nm (interpolated): %+.3f %%; at %.1f nm: %+.3f %%' % (100*(r264-1), lv[0], 100*(r[0]-1)))
    print(' theta_c(264 nm, 3 bar, 20 degC), Koch-scaled minus Boerzsoenyi: %+.3f mrad' % (1e3*(theta_c(1+(n3-1)/r264)-theta_c(n3))))
    print('\n== Xe density: code (B = -130 cm^3/mol at 293 K) vs NIST')
    for T, P, rho in NIST_XE:
        r = molar_density(P*1e6, T); B = (P*1e6/(rho*R_GAS*T)-1)/rho*1e6
        print(' %.2f K %.1f MPa: code %.4f NIST %.4f mol/m^3 (%+.3f %%), NIST effective B %.1f cm^3/mol' % (
            T, P, r, rho, 100*(r/rho-1), B))
    print('\n== Aerogel theta_c (beta = 1) over 300-600 nm for n(400 nm) = 1.03')
    for name, f in (('single pole', n_aerogel_single_pole), ('k4geo shape', n_aerogel_k4geo)):
        t = 1e3*theta_c(f(np.array([300.0, 400.0, 600.0])))
        print(' %-12s theta_c(300/400/600) = %.2f / %.2f / %.2f mrad, spread 300-600 %.2f mrad' % (name, *t, t[0]-t[2]))
    print('\n== Rayleigh, Lambda = lambda^4/C, C = %g um^4/cm' % Rayleigh().C)
    print(' Lambda at 300/400/600 nm: %.2f / %.2f / %.2f cm' % tuple(Rayleigh().length(np.array([300.0, 400.0, 600.0]))/10))
    print(' scattered in one normal %g mm pass at 350 / 400 nm: %.3f / %.3f' % (
        AEROGEL_T, *(1-np.exp(-AEROGEL_T/Rayleigh().length(np.array([350.0, 400.0]))))))
    lo, hi = qe_band()
    for med in (xenon(2), xenon(3), c4f10()):
        for wn, wt in (('FT x QE', qe_arc), ('FT', None)):
            sp = PhotonSpectrum(med.index, lo, hi, weight=wt); L = Rayleigh().length(HC/sp.E)
            fr = np.trapezoid(sp.pdf*(1-np.exp(-AEROGEL_T/L)), sp.E)/sp.cdf[-1]
            print(' %-12s %-7s: scattered fraction in one normal %g mm pass %.4f' % (med.label, wn, AEROGEL_T, fr))
    for wn, wt in (('FT x QE', qe_arc), ('FT', None)):
        E = np.linspace(lo, hi, 20001); pdf = frank_tamm(np.full_like(E, AEROGEL_N_AIR), 1.0)*(wt(E) if wt else 1)
        Lc = Rayleigh().length(HC/E)*np.cos(theta_c(AEROGEL_N_AIR))
        fr = np.trapezoid(pdf*(1-Lc/AEROGEL_T*(1-np.exp(-AEROGEL_T/Lc))), E)/np.trapezoid(pdf, E)
        print(' aerogel photons %-7s: scattered before the bottom face (uniform depth, straight path) %.4f' % (wn, fr))


if __name__ == '__main__':
    main()
