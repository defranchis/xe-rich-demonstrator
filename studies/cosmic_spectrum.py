"""Surface muon momentum-zenith spectrum, thresholds, range, and the momentum-aware gas-ring footprint model.

stdout, cosmic_spectrum_*.png, cosmic_spectrum.json:
- the parametrisations of muons.SPECTRUM_MODELS checked against the numbers their papers quote and against the BESS
  vertical spectra; effective zenith exponent n(p) of I ~ cos^n;
- fractions of the triggered muons above the Xe and aerogel thresholds vs pressure, overburden sensitivity;
- aerogel Cherenkov angle vs momentum; momentum cut of an absorber from the PDG CSDA range;
- gas-ring rates of the table of docs/design_options.md section 5 (muons over a box at the mirror plane) with the
  footprint slope term on (legacy beta = 1 toys) and off, at beta = 1 and with the spectrum, and the fast model against the
  3D tracer.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, cosmic, muons  # noqa: E402
from xerich.analysis import mc_rate  # noqa: E402
from xerich.cherenkov import beta as beta_of, ftqe_angle_yield, p_threshold, pore_signal, qe_band, theta_c, theta_xe  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, COSMIC_I0, COSMIC_I0_PMIN, HC, M_MU, MIRROR_D, MIRROR_R, MUON_THMAX, SENSOR_HALF_2IN  # noqa: E402
from xerich.layout import Mirror, Phase1Layout, Sensor  # noqa: E402
from xerich.media import n_xe, xenon  # noqa: E402
from xerich.muon_data import BESS_LYNN_LAKE, BESS_LYNN_LAKE_COS, BESS_TSUKUBA, BESS_TSUKUBA_COS  # noqa: E402
from xerich.muons import MuonSpectrum, flux_through_square, intensity, sample_box, sample_cos3  # noqa: E402
from xerich.plotting import INK, MUTED, plt, save  # noqa: E402

PRESSURES = (1.0, 1.5, 2.0, 2.5, 3.0)
JOINT = ('guan', 'reyna', 'shukla')
COL = {'guan': '#2a78d6', 'reyna': '#eb6834', 'shukla': '#1baf7a', 'shukla-eq10': '#eda100', 'cos2': MUTED}
LAB = {'guan': 'Guan et al. 2015', 'reyna': 'Reyna 2006', 'shukla': 'Shukla-Sankrith (energy-loss form)',
       'shukla-eq10': 'Shukla-Sankrith Eq. 10 (factorised)'}
SEC5_ROWS = [(50.8, 25.6), (76.2, 25.6), (76.2, 20.0), (76.2, 40.0)]   # docs/design_options.md section 5: mirror D, SiPM half [mm]
SEC5_L = 190.0               # gas length of the section 5 estimate [mm]
TEL_TOP, TEL_BOT = 25.0, 40.0                # trigger counters: half sides at the SiPM plane and the mirror vertex plane [mm]
P_AEROGEL = float(p_threshold(M_MU, AEROGEL_N_AIR))   # muon threshold in the n = 1.03 tile [GeV/c]
_G = {}


def theta_table(P):
    """(beta grid, FT x QE mean angle [rad], FT x QE yield [1/mm]) in Xe at P bar."""
    b = 1-np.geomspace(1e-6, 0.2, 4000)[::-1]
    b = np.r_[b, 1.0]
    th, Y = ftqe_angle_yield(lambda lam: n_xe(lam, P), b)
    return b, th, Y


def task_sec5(arg):
    """Section 5 gas-ring rates for one (pressure, row, seed): beta = 1 with and without the slope term, and with the spectrum."""
    P, D, half, seed = arg
    rng = np.random.default_rng([_G['seed'], int(10*P), int(10*D), int(10*half), seed])
    mirror = Mirror.from_focal(MIRROR_R/2, D); sensor = Sensor(half); thmax = np.radians(25)
    box = mirror.clear_radius+SEC5_L*0.4
    x, y, sx, sy, w = sample_box(rng, 300000, box, thmax)
    c = 1/np.sqrt(1+sx*sx+sy*sy)
    sp = _G['spec']; p = sp.sample_p(rng, c)
    b, th, _ = _G['tab'][P]
    th1 = theta_xe(P); thr = np.interp(beta_of(p, M_MU), b, th)
    out = {}; fr = {}
    for tag, thc, slope in (('legacy', th1, True), ('fixed', th1, False), ('real', thr, False)):
        fm, fs = cosmic.gas_ring_fractions(x, y, sx, sy, thc, mirror, sensor, SEC5_L, 8, 24, slope_term=slope)
        live = np.asarray(thc) > 0
        fr[tag] = ((fm >= .8)&(fs >= .8)&live, (fm >= .999)&(fs >= .999)&live)
    for tag in ('legacy', 'fixed'):
        out[tag] = tuple(mc_rate(w, a, box, thmax)*3600 for a in fr[tag])
    wg = sp.intensity_above(c)/(COSMIC_I0*c*c)
    out['beta1_spectrum'] = tuple(mc_rate(w*wg, a, box, thmax)*3600 for a in fr['fixed'])
    out['beta1_spectrum_p1'] = tuple(mc_rate(w*wg*(p > COSMIC_I0_PMIN), a, box, thmax)*3600 for a in fr['fixed'])   # beta = 1, muons above 1 GeV/c
    for m in muons.SPECTRUM_MODELS:
        wm = intensity(p, c, m)/intensity(p, c, sp.model)*wg
        out['real_%s' % m] = tuple(mc_rate(w*wm, a, box, thmax)*3600 for a in fr['real'])
    wf = intensity(p, 1.0, sp.model)*c*c/intensity(p, c, sp.model)*wg
    out['real_cos2'] = tuple(mc_rate(w*wf, a, box, thmax)*3600 for a in fr['real'])
    return (P, D, half, seed), out


def task_val(arg):
    """Fast footprint model vs the 3D tracer for one muon (zenith [deg], mirror-plane offset x [mm], momentum, pressure)."""
    k, zen, xm, pm, P = arg
    lay = Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), xenon(P)); f = lay.z_sensor
    b = float(beta_of(pm, M_MU)); bt, th, _ = _G['tab'][P]; thc = float(np.interp(b, bt, th))
    s = np.tan(np.radians(zen)); c = 1/np.sqrt(1+s*s); u = np.array([[s*c, 0, -c]])
    fast = cosmic.gas_ring_fractions(np.array([xm]), np.array([0.0]), np.array([s]), np.array([0.0]), thc, lay.mirror, lay.sensor, f, 32, 144)
    leg = cosmic.gas_ring_fractions(np.array([xm]), np.array([0.0]), np.array([s]), np.array([0.0]), thc, lay.mirror, lay.sensor, f, 32, 144,
                                    slope_term=True)
    r = cosmic.trace_batch(lay, np.random.default_rng([_G['seed'], k]), np.array([[xm-f*s, 0, f]]), u, np.array([b]), 40000, ('gas',))
    fm, fs, fd = cosmic.fractions(r, 'gas')
    return dict(zen=zen, x=xm, p=pm, P=P, thc_mrad=1e3*thc, fast_fm=float(fast[0][0]), fast_fs=float(fast[1][0]),
                legacy_fm=float(leg[0][0]), legacy_fs=float(leg[1][0]), tracer_fm=float(fm[0]), tracer_fs=float(fs[0]),
                tracer_det_qe=float(r['det_qe'][0, 0]))


def triggered(rng, n, sp):
    """Muons through the section 5 telescope (top 5 x 5 cm at the SiPM plane, bottom 8 x 8 cm at the mirror vertex)."""
    f = MIRROR_R/2
    u, p0, zen = sample_cos3(rng, n, TEL_TOP, f, MUON_THMAX)
    bot = p0+((0-f)/u[:, 2])[:, None]*u
    tel = (np.abs(bot[:, 0]) < TEL_BOT)&(np.abs(bot[:, 1]) < TEL_BOT)
    c = -u[:, 2]; p = sp.sample_p(rng, c)
    return c, p, tel


def model_weights(sp, p, c):
    """Per-muon weights of every parametrisation (and the factorised cos^2 law) for cos^3 muons with momenta from sp."""
    base = sp.intensity_above(c)/(COSMIC_I0*c*c)
    w = {m: intensity(p, c, m)/intensity(p, c, sp.model)*base for m in muons.SPECTRUM_MODELS}
    w['cos2'] = intensity(p, 1.0, sp.model)*c*c/intensity(p, c, sp.model)*base
    return w


def main():
    ap = cli.parser(__doc__)
    ap.add_argument('--nmu', type=float, default=2e6, help='muons for the trigger and threshold fractions (default 2e6)')
    ap.add_argument('--nseed', type=int, default=5, help='seeds of the section 5 rates (default 5)')
    ap.add_argument('--seed', type=int, default=20261010, help='master seed')
    ap.add_argument('--nproc', type=int, default=20, help='worker processes (default 20)')
    args = ap.parse_args(); out = cli.outdir(args)
    sp = MuonSpectrum(); _G['spec'] = sp; _G['seed'] = args.seed
    with mp.get_context('fork').Pool(min(args.nproc, len(PRESSURES))) as pool:
        _G['tab'] = dict(zip(PRESSURES, pool.map(theta_table, PRESSURES)))
    R = {}
    # 1. parametrisations: own numbers, PDG, BESS
    print('== spectrum parametrisations (vertical intensity [m^-2 s^-1 sr^-1]) ==')
    R['vertical'] = {}
    for m in muons.SPECTRUM_MODELS:
        v = {k: muons.vertical_integral(m, plo) for k, plo in (('p>1', 1.0), ('p>0.5', 0.5), ('p>0.2', 0.2))}
        R['vertical'][m] = v
        print('%-11s above 1 GeV/c %.1f (%.2f of COSMIC_I0 = %g), above 0.5 %.1f, above 0.2 %.1f' % (m, v['p>1'], v['p>1']/COSMIC_I0, COSMIC_I0, v['p>0.5'], v['p>0.2']))
    par = muons.SHUKLA_DURHAM
    print('Shukla-Sankrith Table 1 (Durham): I0 = %g above %g GeV; this code %.2f (the (1 + E/eps) factor makes it lower)' % (par['I0'], par['Ec'], R['vertical']['shukla']['p>0.5']))
    E = 1000.0; pdg = 1e4*0.14*E**-2.7*(1/(1+1.1*E/115)+0.054/(1+1.1*E/850))
    g = muons.intensity_guan(np.sqrt(E**2-M_MU**2), 1.0)*E/np.sqrt(E**2-M_MU**2)
    print('Guan vs PDG Eq. (30.4) (Gaisser) at E = 1 TeV, vertical: ratio %.4f (expected (1 + 3.64/1000)^-2.7 = %.4f)' % (g/pdg, (1+3.64/1000)**-2.7))
    print('Reyna I_v(1 GeV/c) = %.5g cm^-2 s^-1 sr^-1 GeV^-1 (c1 = %g); I(p, 60 deg)/I_v(p/2) = %.4f (cos^3 = 0.125)' % (
        muons.intensity_reyna(1.0, 1.0)/1e4, muons.REYNA_C[0], muons.intensity_reyna(4.0, 0.5)/muons.intensity_reyna(2.0, 1.0)))
    D75 = muons.path_ratio_shukla(np.cos(np.radians(75)))
    print('Shukla D(75 deg) = %.3f: energy-loss form E0 D = %.1f GeV vs their 75 deg fit E0 = 23.78 GeV (Table 1, Hamburg)' % (D75, par['E0']*D75))
    R['bess'] = {}
    for name, T, cs in (('Tsukuba', BESS_TSUKUBA, BESS_TSUKUBA_COS), ('Lynn Lake', BESS_LYNN_LAKE, BESS_LYNN_LAKE_COS)):
        t = np.array(T); R['bess'][name] = {}
        for m in muons.SPECTRUM_MODELS:
            r = intensity(t[:, 2], cs, m)/t[:, 3]
            lo = t[:, 2] < 1; mid = (t[:, 2] >= 1)&(t[:, 2] < 5); hi = t[:, 2] >= 5
            R['bess'][name][m] = dict(below1=float(r[lo].mean()), p1to5=float(r[mid].mean()), above5=float(r[hi].mean()), max_dev=float(np.abs(r-1).max()))
            print('BESS %-9s / %-10s model/data: p<1 %.3f, 1-5 GeV/c %.3f, >5 %.3f' % (name, m, r[lo].mean(), r[mid].mean(), r[hi].mean()))
        pv = t[:, 2]; ints = np.sum(t[:, 3]*(t[:, 1]-t[:, 0])*(t[:, 0] >= 0.97))
        tail = t[-1, 3]*t[-1, 2]/1.7   # integral of a p^-2.7 power law above the last bin
        print('BESS %-9s vertical intensity 0.97-20.6 GeV/c %.1f, + power-law tail above 20 GeV/c ~%.1f' % (name, ints, tail))
        R['bess'][name]['int_1_20'] = float(ints); R['bess'][name]['tail'] = float(tail)
    pgrid = np.array([0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100])
    R['n_eff'] = {m: dict(zip(map(str, pgrid), np.round(muons.angular_exponent(pgrid, m), 3).tolist())) for m in muons.SPECTRUM_MODELS}
    print('effective exponent n(p), I ~ cos^n over 0-75 deg (beta = 1 toys: n = 2):')
    print('  p [GeV/c] ' + ' '.join('%6g' % x for x in pgrid))
    for m in muons.SPECTRUM_MODELS:
        print('  %-10s' % m + ' '.join('%6.2f' % R['n_eff'][m][str(x)] for x in pgrid))
    for thm in (20, 45):
        print('  over 0-%d deg, guan: ' % thm + ' '.join('%6.2f' % x for x in muons.angular_exponent(pgrid, 'guan', np.radians(thm))))
    # 2. thresholds vs pressure for triggered muons
    rng = np.random.default_rng(args.seed)
    c, p, tel = triggered(rng, int(args.nmu), sp)
    W = model_weights(sp, p, c)
    b = beta_of(p, M_MU)
    R['thresholds'] = {}
    ftop = flux_through_square(2*TEL_TOP*1e-3, MUON_THMAX)*3600
    print('\n== triggered muons: section 5 telescope (5 x 5 cm at the SiPM plane, 8 x 8 cm at the mirror vertex plane) ==')
    for m in list(muons.SPECTRUM_MODELS)+['cos2']:
        w = W[m]
        R['thresholds'][m] = dict(rate_top_h=float(ftop*w.mean()), rate_tel_h=float(ftop*(w*tel).mean()), legacy_top_h=float(ftop),
                                  legacy_tel_h=float(ftop*tel.mean()))
        for P in PRESSURES:
            bt, th, Y = _G['tab'][P]; y = np.interp(b, bt, Y)/Y[-1]
            R['thresholds'][m][str(P)] = dict(above=float(np.sum(w*tel*(y > 0))/np.sum(w*tel)),
                                              half_yield=float(np.sum(w*tel*(y > 0.5))/np.sum(w*tel)),
                                              mean_yield=float(np.sum(w*tel*y)/np.sum(w*tel)))
        R['thresholds'][m]['aerogel'] = float(np.sum(w*tel*(p > P_AEROGEL))/np.sum(w*tel))
        print('%-10s top counter %.0f /h (legacy %.0f), telescope %.0f /h (legacy %.0f); above Xe threshold (any FT x QE light) at %s bar: %s; >= half the beta = 1 yield: %s; above the aerogel threshold %.3f' % (
            m, ftop*w.mean(), ftop, ftop*(w*tel).mean(), ftop*tel.mean(), '/'.join('%g' % P for P in PRESSURES),
            ' '.join('%.3f' % R['thresholds'][m][str(P)]['above'] for P in PRESSURES),
            ' '.join('%.3f' % R['thresholds'][m][str(P)]['half_yield'] for P in PRESSURES), R['thresholds'][m]['aerogel']))
    R['p_threshold'] = {str(P): float(M_MU/np.sqrt(n_xe(HC/qe_band()[1], P)**2-1)) for P in PRESSURES}
    print('Xe threshold momentum at the UV edge of the QE band (%.0f nm): ' % (HC/qe_band()[1]) + ', '.join('%g bar %.2f GeV/c' % (P, R['p_threshold'][str(P)]) for P in PRESSURES))
    # overburden (information only)
    R['overburden'] = {}
    for X in (0, 50, 100, 200):
        spo = MuonSpectrum(overburden=X)
        pp, cc, _ = spo.sample(np.random.default_rng(args.seed+X), 400000)
        R['overburden'][str(X)] = dict(flux=float(spo.flux_per_m2), above_3bar=float(np.mean(pp > R['p_threshold']['3.0'])),
                                       above_2bar=float(np.mean(pp > R['p_threshold']['2.0'])), median_p=float(np.median(pp)))
        print('overburden %3d g/cm^2 concrete: flux %.1f /m^2/s (zenith < 75), median p %.2f GeV/c, above the 2 / 3 bar threshold %.3f / %.3f' % (
            X, spo.flux_per_m2, np.median(pp), R['overburden'][str(X)]['above_2bar'], R['overburden'][str(X)]['above_3bar']))
    # 3. aerogel angle vs momentum, range
    R['aerogel_shift'] = {}
    t1 = theta_c(AEROGEL_N_AIR)
    for pm in (0.5, 0.7, 1.0, 2.0, 3.0, 5.0):
        d = np.arccos(1/(AEROGEL_N_AIR*beta_of(pm, M_MU)))-t1
        R['aerogel_shift'][str(pm)] = float(1e3*d)
    print('\naerogel n = %.2f: theta(beta = 1) = %.2f mrad; shift vs beta = 1: ' % (AEROGEL_N_AIR, 1e3*t1) + ', '.join('%g GeV/c %.1f mrad' % (k, v) for k, v in zip((0.5, 0.7, 1, 2, 3, 5), R['aerogel_shift'].values())))
    R['range'] = {}
    for mat in ('lead', 'iron', 'concrete', 'polystyrene'):
        rho = muons.density(mat)
        R['range'][mat] = {str(t): float(muons.p_from_range(t*rho, mat)) for t in (1, 2, 5, 10, 20)}
        print('%-11s (rho %.2f): p cut [GeV/c] for 1/2/5/10/20 cm at normal incidence: %s' % (mat, rho, ' '.join('%.3f' % v for v in R['range'][mat].values())))
    # 4. section 5 rates: slope term and momentum
    tasks = [(P, D, h, s) for P in (2.0, 3.0) for D, h in SEC5_ROWS for s in range(args.nseed)]
    with mp.get_context('fork').Pool(args.nproc) as pool:
        res = pool.map(task_sec5, tasks)
        val = pool.map(task_val, [(k, z, x, pm, 3.0) for k, (z, x, pm) in enumerate(((0, 0, 10), (0, 30, 10), (3, 20, 10), (3, 25, 10), (3, 25, 2),
                                                                  (-3, 25, 3), (2, 15, 1.8), (5, 0, 5), (6, 0, 10), (4, -20, 2.5)))])
    R['sec5'] = {}
    print('\n== docs/design_options.md section 5 gas-ring rates [1/h], muons over a box at the mirror plane, L = %g mm, %d seeds (mean +- seed std) ==' % (SEC5_L, args.nseed))
    for P in (2.0, 3.0):
        for D, h in SEC5_ROWS:
            rows = [o for (PP, DD, hh, s), o in res if (PP, DD, hh) == (P, D, h)]
            agg = {k: (np.mean([r[k][0] for r in rows]), np.std([r[k][0] for r in rows], ddof=1),
                       np.mean([r[k][1] for r in rows]), np.std([r[k][1] for r in rows], ddof=1)) for k in rows[0]}
            R['sec5']['%g_%g_%g' % (P, D, h)] = {k: [float(x) for x in v] for k, v in agg.items()}
            print('P %g bar, mirror %5.1f, SiPM %4.1f cm: >=80%% legacy %5.1f+-%.1f | slope fixed %5.1f+-%.1f | beta=1 spectrum %5.1f | real (guan) %5.1f+-%.1f [reyna %5.1f shukla %5.1f eq10 %5.1f cos2 %5.1f] ; full: legacy %4.1f fixed %4.1f real %4.1f' % (
                P, D, 2*h/10, agg['legacy'][0], agg['legacy'][1], agg['fixed'][0], agg['fixed'][1], agg['beta1_spectrum'][0], agg['real_guan'][0], agg['real_guan'][1],
                agg['real_reyna'][0], agg['real_shukla'][0], agg['real_shukla-eq10'][0], agg['real_cos2'][0], agg['legacy'][2], agg['fixed'][2], agg['real_guan'][2]))
    R['validation'] = val
    print('\n== fast footprint model vs 3D tracer (3 bar, 3" f = 203.2 mm mirror, 5.08 cm SiPM) ==')
    print(' zenith  x[mm]  p[GeV/c] theta[mrad] | fm fast  legacy  tracer | fs fast  legacy  tracer')
    for v in val:
        print(' %5.1f  %5.1f  %5.1f   %6.2f      | %.3f   %.3f   %.3f  | %.3f   %.3f   %.3f' % (v['zen'], v['x'], v['p'], v['thc_mrad'], v['fast_fm'], v['legacy_fm'], v['tracer_fm'],
                                                                                  v['fast_fs'], v['legacy_fs'], v['tracer_fs']))
    with open(out/'cosmic_spectrum.json', 'w') as fo:
        json.dump(R, fo, indent=1)
    figures(out, sp, c, p, tel, W)
    print('cores used: %d' % args.nproc)


def figures(out, sp, c, p, tel, W):
    """Spectra with BESS data and the parametrisation band, zenith dependence and n(p), threshold fractions, aerogel angle."""
    pg = np.geomspace(0.2, 200, 400)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for m in muons.SPECTRUM_MODELS:
        ax[0].plot(pg, pg**2*intensity(pg, BESS_TSUKUBA_COS, m), color=COL[m], lw=2 if m in JOINT else 1.2, ls='-' if m in JOINT else ':', label=LAB[m])
    for name, T, mk in (('BESS Tsukuba (30 m, 11.4 GV)', BESS_TSUKUBA, 'o'), ('BESS Lynn Lake (360 m, 0.4 GV)', BESS_LYNN_LAKE, 's')):
        t = np.array(T)
        ax[0].errorbar(t[:, 2], t[:, 2]**2*t[:, 3], yerr=t[:, 2]**2*np.hypot(t[:, 4], t[:, 5]), fmt=mk, ms=4, color=INK, mfc='white' if mk == 's' else INK, label=name)
    ax[0].axvspan(0.2, 1.0, color=MUTED, alpha=0.08); ax[0].text(0.22, 6, 'below 1 GeV/c:\nReyna extrapolated', fontsize=8, color=MUTED)
    ax[0].set_xscale('log'); ax[0].set_xlabel('p [GeV/c]'); ax[0].set_ylabel('p² dI/dp dΩ [m⁻² s⁻¹ sr⁻¹ GeV/c]')
    ax[0].set_title('Vertical spectrum (cos θ = 0.99) vs BESS', loc='left', fontsize=10); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
    band = np.array([intensity(pg, BESS_TSUKUBA_COS, m) for m in JOINT])
    for name, T, cs, mk in (('Tsukuba', BESS_TSUKUBA, BESS_TSUKUBA_COS, 'o'), ('Lynn Lake', BESS_LYNN_LAKE, BESS_LYNN_LAKE_COS, 's')):
        t = np.array(T); ref = intensity(t[:, 2], cs, 'guan')
        ax[1].errorbar(t[:, 2], t[:, 3]/ref, yerr=np.hypot(t[:, 4], t[:, 5])/ref, fmt=mk, ms=4, color=INK, mfc='white' if mk == 's' else INK, label='BESS %s / Guan' % name)
    ref = intensity(pg, BESS_TSUKUBA_COS, 'guan')
    ax[1].fill_between(pg, band.min(0)/ref, band.max(0)/ref, color=COL['guan'], alpha=0.15, label='spread of the three joint parametrisations')
    for m in ('reyna', 'shukla'):
        ax[1].plot(pg, intensity(pg, BESS_TSUKUBA_COS, m)/ref, color=COL[m], lw=1.5, label=LAB[m]+' / Guan')
    ax[1].axhline(1, color=COL['guan'], lw=1.5); ax[1].set_xscale('log'); ax[1].set_ylim(0.2, 1.5)
    ax[1].set_xlabel('p [GeV/c]'); ax[1].set_ylabel('ratio to Guan et al.'); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
    ax[1].set_title('Ratios to the default parametrisation', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_spectrum_vertical.png', dpi=130, fig=fig); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for deg, ls in ((0, '-'), (30, '--'), (60, '-.'), (75, ':')):
        cs = np.cos(np.radians(deg))
        band = np.array([intensity(pg, cs, m) for m in JOINT])
        ax[0].fill_between(pg, band.min(0), band.max(0), color=COL['guan'], alpha=0.12)
        ax[0].plot(pg, intensity(pg, cs, 'guan'), color=COL['guan'], ls=ls, lw=1.8, label='Guan, θ = %d°' % deg)
        ax[0].plot(pg, intensity(pg, 1.0, 'guan')*cs**2, color=MUTED, ls=ls, lw=1, label='vertical × cos²θ, θ = %d°' % deg)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-3, 50)
    ax[0].set_xlabel('p [GeV/c]'); ax[0].set_ylabel('dI/dp dΩ [m⁻² s⁻¹ sr⁻¹ (GeV/c)⁻¹]'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title('Joint spectrum vs zenith (band: three parametrisations)', loc='left', fontsize=10)
    pq = np.geomspace(0.3, 100, 120)
    for m in muons.SPECTRUM_MODELS:
        ax[1].plot(pq, muons.angular_exponent(pq, m), color=COL[m], lw=2 if m in JOINT else 1.2, ls='-' if m in JOINT else ':', label=LAB[m])
    ax[1].axhline(2, color=MUTED, lw=1.2, ls='--', label='β = 1 toys: cos²θ (cos³θ through a plane)')
    for P, yt in ((2.0, 4.6), (3.0, 3.9)):
        pt = float(M_MU/np.sqrt(n_xe(HC/qe_band()[1], P)**2-1))
        ax[1].axvline(pt, color=INK, lw=0.8, ls=':')
        ax[1].text(pt*1.03, yt, 'Xe %g bar\nthreshold' % P, fontsize=7.5, color=INK)
    ax[1].axvline(P_AEROGEL, color=INK, lw=0.8, ls=':'); ax[1].text(1.03*P_AEROGEL, 4.3, 'aerogel\nthreshold', fontsize=7.5)
    ax[1].set_xscale('log'); ax[1].set_xlabel('p [GeV/c]'); ax[1].set_ylabel('effective exponent n, I ∝ cosⁿθ (fit over 0-75°)')
    ax[1].set_ylim(-0.5, 5); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7.5, loc='lower left')
    ax[1].set_title('Zenith law vs momentum', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_spectrum_zenith.png', dpi=130, fig=fig); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.geomspace(0.2, 200, 61)
    for m in list(JOINT)+['cos2']:
        w = W[m]*tel
        h, _ = np.histogram(p, bins, weights=w); h = h/w.sum()
        ax[0].stairs(h, bins, color=COL[m], lw=1.8 if m != 'cos2' else 1.2, label=LAB.get(m, 'Guan vertical × cos²θ (factorised)'))
    thr = [float(M_MU/np.sqrt(n_xe(HC/qe_band()[1], P)**2-1)) for P in PRESSURES]
    for P, t in zip(PRESSURES, thr):
        ax[0].axvline(t, color=INK, lw=0.7, ls=':')
    ytop = ax[0].get_ylim()[1]
    ax[0].text(thr[-1]*0.97, 0.97*ytop, '3 bar', fontsize=7.5, ha='right', va='top'); ax[0].text(thr[0]*1.03, 0.97*ytop, '1 bar', fontsize=7.5, va='top')
    ax[0].axvline(P_AEROGEL, color=INK, lw=0.7, ls='--'); ax[0].text(1.03*P_AEROGEL, 0.03*ytop, 'aerogel', fontsize=7.5)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('p [GeV/c]'); ax[0].set_ylabel('fraction of triggered muons per bin (60 log-spaced bins)'); ax[0].legend(fontsize=7.5, loc='center right'); ax[0].grid(alpha=.3)
    ax[0].set_title('Momentum of muons through the trigger telescope (dotted: Xe thresholds 1-3 bar)', loc='left', fontsize=9.5)
    for m in list(JOINT)+['cos2']:
        w = W[m]*tel; bb = beta_of(p, M_MU)
        fa = []; fh = []
        for P in PRESSURES:
            bt, th, Y = _G['tab'][P]; y = np.interp(bb, bt, Y)/Y[-1]
            fa.append(np.sum(w*(y > 0))/w.sum()); fh.append(np.sum(w*(y > 0.5))/w.sum())
        ax[1].plot(PRESSURES, fa, 'o-', color=COL[m], lw=1.8, label=LAB.get(m, 'factorised cos²') + ': any light')
        ax[1].plot(PRESSURES, fh, 's--', color=COL[m], lw=1.0, ms=4, label=LAB.get(m, 'factorised cos²') + ': ≥ 50 % of the β = 1 yield')
    ax[1].set_xlabel('Xe pressure [bar abs]'); ax[1].set_ylabel('fraction of triggered muons'); ax[1].set_ylim(0, 1); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=6.8); ax[1].set_title('Fraction above the Xe threshold (β = 1 toys: 1)', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_spectrum_thresholds.png', dpi=130, fig=fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    pa = np.geomspace(0.4, 20, 400); t1 = theta_c(AEROGEL_N_AIR)
    with np.errstate(invalid='ignore'):
        ax.plot(pa, 1e3*(np.arccos(np.minimum(1, 1/(AEROGEL_N_AIR*beta_of(pa, M_MU))))-t1), color=COL['reyna'], lw=2, label='aerogel n = 1.03 (in the tile)')
    for P, ls in ((2.0, '--'), (3.0, '-')):
        bt, th, Y = _G['tab'][P]
        tp = np.interp(beta_of(pa, M_MU), bt, th)
        ax.plot(pa, np.where(tp > 0, 1e3*(tp-th[-1]), np.nan), color=COL['guan'], ls=ls, lw=1.5, label='Xe %g bar, FT×QE mean (no line: no light)' % P)
    ps = [1e3*x for x in pore_signal(PRESSURES[-1])]
    ax.axhspan(*ps, color='#1baf7a', alpha=0.2, label='pore-gas signal, Xe 3 bar (+%.1f / +%.1f mrad)' % tuple(ps))
    ax.set_xscale('log'); ax.set_ylim(-60, 12); ax.set_xlabel('p [GeV/c]'); ax.set_ylabel('θ(p) − θ(β = 1) [mrad]')
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title('Cherenkov angle vs momentum', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_spectrum_angle_vs_p.png', dpi=130, fig=fig); plt.close(fig)


if __name__ == '__main__':
    main()
