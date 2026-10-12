"""Rates, photon counts, containment and angle-shift tables and figures from cosmic_rates.py (stdout, json, md, png).

Reads cosmic_rates_<group>.npz and cosmic_rates_meta.json from --input. Ring definitions: gas ring, full = >= 99 % of
the sampled photons on the mirror and of the ring on the SiPM, >= 80 % = >= 80 % on both (docs/design_options.md
section 5); aerogel ring, full / >= 80 % = >= 99 % / >= 80 % of the emitted photons on the SiPM (losses through the
tile sides count; the tracer treats the tile side faces as black); arc >= 50 % = half of the 36 azimuth bins (those
holding photons above threshold) with at least half of their emitted photons on the SiPM (arc*_any: a bin counts if one
sampled photon reaches the SiPM, the definition of the beta = 1 mirror toys). Rates per hour: muons through the top
counter plane (all sampled muons) and through the telescope (both counters), at beta = 1 with the cos^2 law and I0 (the
beta = 1 toys) and with the real momentum spectrum (default parametrisation; the others as weights). Mean +- standard
deviation over the seeds.
Aerogel angle in the tile, per seed, from the true emission angles (no ring resolution): plain mean, mean in a window
of -10 to +20 mrad around the beta = 1 angle of the plain n = 1.03 tile (fixed) or of the run's own, known tile index
(own window, truth level), and the 95 % quantile (beta -> 1 edge).
Absorber handles: range cut from the PDG CSDA range along the slant path, with the bottom counter left at its plane
and, as built, moved behind the absorber (geometry only, no multiple scattering), plus the counter side that keeps
90 % of the range-cut rings.
Photon counts are FT x QE photons on the SiPM without Rayleigh scatter: per muon from the matched run with scattering
(configuration 5 gas ring: geometry from the _geo run, photons from the _ray run), else the transmission factor of the
runs with scattering (configurations 2, 3 and the aerogel light of 5) or computed (configuration 4). Except in
configuration 4 they are Monte Carlo estimates from the photons sampled per muon (uniform in energy, QE as a weight),
so cuts on them are approximate for rings whose count is near the cut.
Precision per hour on the mean ring angle: sqrt(sigma_photon^2/<N> + rms_rings^2)/sqrt(rings per hour), sigma_photon
from the pixel pitch (and the Xe dispersion, the emission depth in configuration 4), rms_rings the spread of the true
ring angle over the rings (velocity).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, muons  # noqa: E402
from xerich.cherenkov import beta as beta_of, chromatic_rms, ftqe_angle_yield, pore_signal, qe_arc, qe_band, theta_c  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, AEROGEL_T, COSMIC_I0, COSMIC_I0_PMIN, HC, M_MU, MIRROR_R, MIRROR_SHORT_F, MUON_THMAX, PIXEL_PITCH  # noqa: E402
from xerich.layout import Phase1Layout  # noqa: E402
from xerich.media import N_AIR, ConstantIndex, Rayleigh, aerogel_index, aerogel_index_air, n_xe  # noqa: E402
from xerich.muons import MuonSpectrum, flux_through_square, intensity, sample_cos3  # noqa: E402
from xerich.plotting import MUTED, plt, save  # noqa: E402
from xerich.tracer import snell_angle  # noqa: E402

PRESSURES = (1.0, 1.5, 2.0, 2.5, 3.0)
MODELS = ('guan', 'reyna', 'shukla', 'shukla-eq10', 'cos2')
C_REAL = '#2a78d6'; C_B1 = MUTED; C_VAR = {'plain': '#2a78d6', 'pore-vac': '#eb6834', 'pore-air': '#1baf7a'}
NMIN = np.arange(0, 61)
WIN = (10.0, 20.0)           # peak window of the aerogel angle around a beta = 1 angle: -10 to +20 mrad
EDGE_Q = 0.95                # quantile of the aerogel angle taken as the beta -> 1 edge
ABSORBER_CM = (0, 5, 10, 20, 30, 50, 75, 100)
KEEP_REF = 0.9               # fraction of the range-cut rings the enlarged bottom counter keeps
SHORT_F_KEY = {'f%d' % f: k for k, (f, D) in MIRROR_SHORT_F.items()}   # run label -> Edmund stock number
Z_TILE_MID = Phase1Layout.gap+AEROGEL_T/2    # tile mid-plane below the SiPM plane [mm]
HIST_LO = -250.0             # lower edge of the aerogel-angle shift histograms [mrad] (below -theta_c of every tile index, >= -249.6)
_G = {}


def gas_label(P):
    return 'air' if P == 0 else 'xe%g' % P


def n_gas400(P):
    return N_AIR if P == 0 else float(n_xe(400.0, P))


def tile_n(P, v):
    ng = n_gas400(P)
    return {'plain': AEROGEL_N_AIR, 'pore-vac': float(aerogel_index(ng)), 'pore-air': float(aerogel_index_air(ng))}[v]


def xe_table(P):
    """(beta grid, FT x QE mean angle, yield) in Xe at P bar."""
    b = np.r_[1-np.geomspace(1e-6, 0.2, 3000)[::-1], 1.0]
    th, Y = ftqe_angle_yield(lambda lam: n_xe(lam, P), b)
    return b, th, Y


def xe_chromatic_rms(P):
    """FT x QE rms of the Xe Cherenkov angle at beta = 1 [rad] at P bar."""
    return chromatic_rms(lambda lam: n_xe(lam, P))


def rayleigh_prox(n=AEROGEL_N_AIR):
    """FT x QE mean transmission of aerogel photons from a uniform emission depth to the tile bottom (vertical muon)."""
    E = np.linspace(*qe_band(), 2001); w = qe_arc(E); L = Rayleigh().length(HC/E)
    s = AEROGEL_T/np.cos(theta_c(n))*(np.arange(200)+0.5)/200
    return float(np.sum(w*np.exp(-s[:, None]/L[None, :]).mean(0))/w.sum())


def weights(p, c):
    """{model: per-muon weight} for cos^3 muons with momenta drawn from the default spectrum at their zenith; 'b1': 1."""
    sp = _G['spec']; base = sp.intensity_above(c)/(COSMIC_I0*c*c); ig = intensity(p, c, sp.model)
    w = {m: intensity(p, c, m)/ig*base for m in muons.SPECTRUM_MODELS}
    w['cos2'] = intensity(p, 1.0, sp.model)*c*c/ig*base
    w['b1'] = np.ones(len(p))
    w['spectrum_p1'] = w[sp.model]*(p > COSMIC_I0_PMIN)   # spectrum weight of the muons above 1 GeV/c, for beta = 1 acceptances
    return w


def trigger_rates(top_half, z_top, z_bot, bot_half, n=1000000, seed=1):
    """Muon rates [1/h] through the top counter plane and through the telescope (zenith < 75 deg): beta = 1 law and spectrum."""
    rng = np.random.default_rng(seed)
    u, p0, _ = sample_cos3(rng, n, top_half, z_top, MUON_THMAX)
    bot = p0+((z_bot-z_top)/u[:, 2])[:, None]*u
    th_ = _G['top_half']
    tel = (np.abs(bot[:, 0]) < bot_half)&(np.abs(bot[:, 1]) < bot_half)&(np.abs(p0[:, 0]) < th_)&(np.abs(p0[:, 1]) < th_)
    w = _G['spec'].weight_vs_cos2(-u[:, 2])
    F = flux_through_square(2*top_half*1e-3, MUON_THMAX)*3600
    return dict(plane_b1=F, plane_real=F*w.mean(), tel_b1=F*tel.mean(), tel_real=F*np.mean(w*tel))


def acceptances(z, name, kind, origin, tag):
    """{definition: mask} for one run, origin and beta tag ('b1', 'br')."""
    g = lambda k: z['%s/%s_%s_%s' % (name, tag, origin, k)]
    if kind == 'c4':
        fd, arc, th = g('fd'), g('arc'), g('th')
        return dict(full=(fd >= .99)&(th > 0), r80=(fd >= .8)&(th > 0), arc50=(arc >= .5)&(th > 0), any=(fd > 0)&(th > 0))
    fm, fs, arc, e, fd, anyb = g('fm'), g('fs'), g('arc_half'), g('emit_qe'), g('fd'), g('arc')
    live = e > 0
    out = dict(arc50=(arc >= .5)&live, arc90=(arc >= .9)&live, arc50_any=(anyb >= .5)&live, arc90_any=(anyb >= .9)&live,
               any=(g('det_qe') > 0)&live)
    if origin == 'tile':
        out.update(full=(fd >= .99)&live, r80=(fd >= .8)&live)
    else:
        out.update(full=(fm >= .99)&(fs >= .99)&live, r80=(fm >= .8)&(fs >= .8)&live, fd80=(fd >= .8)&live)
    return out


def per_seed(F, w, m, seed, nseed):
    return np.array([F*np.sum((w*m)[seed == s])/np.sum(seed == s) for s in range(nseed)])


def photons(z, name, run, origin, tag, c, bb):
    """FT x QE photons on the SiPM per muon, Rayleigh transmission applied."""
    if run['group'] == 'c4':
        na = tile_n(run['P'], run['variant'])
        bg = np.r_[1-np.geomspace(1e-7, 0.1, 2000)[::-1], 1.0]
        Y = np.interp(bb, bg, ftqe_angle_yield(ConstantIndex(na), bg)[1])
        return Y*AEROGEL_T/c*z['%s/%s_tile_fd' % (name, tag)]*_G['ray_c4']
    det = z['%s/%s_%s_det_qe' % (name, tag, origin)]
    if name.endswith('_geo'):
        return z['%s/%s_%s_det_qe_clean' % (name.replace('_geo', '_ray'), tag, origin)]
    if run['rayleigh']:
        return z['%s/%s_%s_det_qe_clean' % (name, tag, origin)]
    if origin == 'tile':
        return det*_G['ray_tile'].get((run['group'], tuple(run['mirror']), 'air' if run['P'] == 0 else 'xe'), _G['ray_tile_default'])
    return det


def analyse_run(arg):
    """Rates, photon counts, radii, angle shifts and maps for one run of cosmic_rates.py."""
    g, run = arg
    z = np.load(_G['input']/('cosmic_rates_%s.npz' % g)); name = run['name']; kind = run['group']
    c = z[name+'/c']; p = z[name+'/p']; seed = z[name+'/seed']; nseed = int(seed.max())+1
    F = flux_through_square(2*run['top_half']*1e-3, np.radians(run['zmax']))*3600
    W = weights(p, c); b = beta_of(p, M_MU)
    norms = dict(plane=np.ones(len(c), bool), tel=z[name+'/top']&z[name+'/bot'])
    res = dict(name=name, group=kind, P=run['P'], variant=run['variant'], rayleigh=run['rayleigh'], F=F, n=int(len(c)), nseed=nseed)
    for origin in (run['origins'] if kind != 'c4' else ('tile',)):
        for tag in ('b1', 'br'):
            bb = np.ones(len(c)) if tag == 'b1' else b
            wk = 'b1' if tag == 'b1' else 'guan'; w = W[wk]
            A = acceptances(z, name, kind, origin, tag)
            nph = photons(z, name, run, origin, tag, c, bb)
            R = {}
            for nk, nm in norms.items():
                d = {}
                for dn, m in A.items():
                    for mk in (('b1', 'spectrum_p1') if tag == 'b1' else MODELS):
                        r = per_seed(F, W[mk], m&nm, seed, nseed)
                        d['%s_%s' % (dn, mk)] = [float(r.mean()), float(r.std(ddof=1))]
                sel = A['r80']&nm
                d['nmin_curve'] = [float(F*np.sum((w*sel)[nph >= k])/len(c)) for k in NMIN]
                d['photons_mean_r80'] = float(np.average(nph[sel], weights=w[sel])) if sel.any() else 0.0
                d['inv_photons_r80'] = float(np.average(1/np.maximum(nph[sel], 0.5), weights=w[sel])) if sel.any() else 0.0
                R[nk] = d
            sel = A['r80']
            R['photon_hist'] = np.histogram(nph[sel], np.arange(0, 82, 2), weights=w[sel])[0].tolist()
            R['photons_mean_all'] = float(np.average(nph, weights=w))
            R['emit_mean_all'] = float(np.average(z['%s/%s_%s_emit_qe' % (name, tag, origin)], weights=w)) if kind != 'c4' else 0.0
            if origin == 'tile':
                na = tile_n(run['P'], run['variant']); ng = n_gas400(run['P'])
                tha = np.arccos(np.minimum(1, 1/(na*bb))); thobs = snell_angle(tha, na, ng)
                dth = 1e3*(tha-theta_c(na))
                if sel.any():
                    R['theta_tile_mean'] = float(np.average(1e3*tha[sel], weights=w[sel]))
                    R['theta_tile_seed'] = [float(np.average(1e3*tha[sel&(seed == k)], weights=w[sel&(seed == k)])) if np.any(sel&(seed == k)) else np.nan
                                            for k in range(nseed)]
                    for key, t0 in (('win', theta_c(AEROGEL_N_AIR)), ('ownwin', theta_c(na))):
                        win = sel&(1e3*(tha-t0) > -WIN[0])&(1e3*(tha-t0) < WIN[1])
                        R['theta_tile_%s_seed' % key] = [float(np.average(1e3*tha[win&(seed == k)], weights=w[win&(seed == k)]))
                                                         if np.any(win&(seed == k)) else np.nan for k in range(nseed)]
                        R['%s_fraction' % key] = float(np.sum(w*win)/np.sum(w*sel))
                    R['theta_tile_edge_seed'] = [float(np.quantile(1e3*tha[sel&(seed == k)], EDGE_Q, weights=w[sel&(seed == k)], method='inverted_cdf'))
                                                 if np.any(sel&(seed == k)) else np.nan for k in range(nseed)]
                    hi = sel&(p > 2.0) if tag == 'br' else sel
                    R['theta_tile_mean_p2'] = float(np.average(1e3*tha[hi], weights=w[hi])) if hi.any() else None
                R['dtheta_hist'] = np.histogram(dth[sel], np.arange(HIST_LO, 3, 1.0), weights=w[sel])[0].tolist()
                if sel.any():
                    mu = np.average(dth[sel], weights=w[sel])
                    R['dtheta'] = dict(mean=float(mu), rms=float(np.sqrt(np.average((dth[sel]-mu)**2, weights=w[sel]))),
                                       q=[float(x) for x in np.quantile(dth[sel], (0.05, 0.16, 0.5, 0.84, 0.95), weights=w[sel], method='inverted_cdf')])
                if kind == 'c23' and not run['rayleigh'] and _G.get('rshift') is not None and sel.any():
                    R['offset_term'] = offset_term(z, name, run, c, w, sel, dth)
                if tag == 'br' and kind in ('c23', 'c4') and run['variant'] == 'plain' and not run['rayleigh']:
                    R['handles'] = handles(z, name, run, p, c, w, A['r80'], dth, norms['tel'], F)
            else:
                bt, th, _ = _G['xe'][run['P']]; thobs = np.interp(bb, bt, th)
            if sel.any():
                mu = np.average(thobs[sel], weights=w[sel])
                R['theta_obs'] = float(mu); R['theta_obs_rms'] = float(np.sqrt(np.average((thobs[sel]-mu)**2, weights=w[sel])))
            if kind != 'c4':
                f = run['mirror'][0]/2
                R['radius_hist'] = np.histogram(f*np.tan(thobs[sel]), np.arange(0, 30.25, 0.25), weights=w[sel])[0].tolist()
            if kind == 'c1':
                R['maps'] = containment_maps(z, name, run, c, w, A['r80'])
            res['%s_%s' % (origin, tag)] = R
    return res


def offset_term(z, name, run, c, w, sel, dth):
    """Ring-radius shift of the f/1 sphere with the impact offset (short-f tracer, vertical muon) as an angle [mrad].

    The offset is the muon's distance from the axis at the tile mid-plane (Z_TILE_MID below the SiPM plane); returns the
    mean and rms of that term and of its sum with the velocity term dth over the >= 80 % rings.
    """
    mk = 'f%d' % (run['mirror'][0]/2)
    off, dr = _G['rshift'][mk]
    phi = z[name+'/phi']; t = np.sqrt(1-c*c)/c
    rho = np.hypot(z[name+'/x0']+Z_TILE_MID*t*np.cos(phi), z[name+'/y0']+Z_TILE_MID*t*np.sin(phi))
    d = np.interp(rho, off, dr)/_G['dr_dtheta'][mk]
    ww = w[sel]

    def mr(v):
        m = np.average(v, weights=ww)
        return float(m), float(np.sqrt(np.average((v-m)**2, weights=ww)))
    return dict(offset=mr(d[sel]), total=mr(d[sel]+dth[sel]), velocity=mr(dth[sel]), rho_mean=float(np.average(rho[sel], weights=ww)))


def load_rshift(path):
    """{mirror: (|offset| [mm], radius shift [mm])} from short_f_angles.npz: air, clear 90 %, azimuth 0, parallel, zenith 0;
    and {mirror: dr/dtheta [mm/mrad] of the aerogel ring in air} from short_f_summary.json in the same directory."""
    zz = np.load(path)
    st = zz['stats']; f = list(zz['stats_fields']).index('fit_r_mm'); off = zz['offset_mm']
    with open(path.parent/'short_f_summary.json') as fs:
        terms = json.load(fs)['terms']
    drdt = {mk: terms['%s_air' % SHORT_F_KEY[mk]]['dr_dtheta_aero_mm_per_mrad'] for mk in SHORT_F_KEY}
    out = {}
    for i, mk in enumerate(SHORT_F_KEY):
        assert str(zz['mirror'][i]) == SHORT_F_KEY[mk]
        r = st[i, 0, 1, 0, 0, 0, :, f]
        a = np.abs(off); ua = np.unique(a)
        rr = np.array([np.nanmean(r[a == x]) for x in ua])
        out[mk] = (ua, rr-rr[0])
    return out, drdt


def containment_maps(z, name, run, c, w, acc):
    """Fraction of muons with a >= 80 % ring vs zenith and offset of the mirror-plane impact from the axis."""
    phi = z[name+'/phi']; x0 = z[name+'/x0']; y0 = z[name+'/y0']; f = run['mirror'][0]/2
    t = np.sqrt(1-c*c)/c
    rho = np.hypot(x0+f*t*np.cos(phi), y0+f*t*np.sin(phi))
    ze = np.degrees(np.arccos(c))
    zb = np.arange(0, 10.5, 0.5); rb = np.arange(0, 52, 4)
    num = np.histogram2d(ze, rho, (zb, rb), weights=w*acc)[0]; den = np.histogram2d(ze, rho, (zb, rb), weights=w)[0]
    m0 = rho < 8
    n1 = np.histogram(ze[m0], zb, weights=(w*acc)[m0])[0]; d1 = np.histogram(ze[m0], zb, weights=w[m0])[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        return dict(zb=zb.tolist(), rb=rb.tolist(), map=np.nan_to_num(num/den, nan=-1).tolist(), curve=np.nan_to_num(n1/d1, nan=-1).tolist())


def counter_coord(z, name, run, c, depth):
    """max(|x|, |y|) [mm] of each muon at depth [mm] below the bottom-counter plane of the run's telescope."""
    phi = z[name+'/phi']; t = np.sqrt(1-c*c)/c
    z_top, z_bot = (0.0, -AEROGEL_T-run['gap']) if run['group'] == 'c4' else (run['mirror'][0]/2, 0.0)
    dz = z_top-z_bot+depth
    return np.maximum(np.abs(z[name+'/x0']+dz*t*np.cos(phi)), np.abs(z[name+'/y0']+dz*t*np.sin(phi)))


def handles(z, name, run, p, c, w, sel, dth, tel, F):
    """Momentum handles on the aerogel-angle shift dth [mrad] of >= 80 % rings in the telescope.

    Absorber of t cm right below the bottom-counter plane, range cut along the slant path. keep / rate_h / mean / rms:
    the counter left at its plane (equivalent to a counter behind the absorber enlarged to keep the acceptance);
    *_counter: the same counter moved behind the absorber (geometry only, no multiple scattering); side90_mm: side of a
    square counter behind the absorber that keeps KEEP_REF of the range-cut rings of the first definition.
    """
    out = {}
    k0 = sel&tel; base = np.sum(w*k0)
    top = z[name+'/top']; half = _G['bot_half']['prox' if run['group'] == 'c4' else 'mirror']

    def stats(ww):
        if ww.sum() <= 0:
            return float('nan'), float('nan')
        mu = np.sum(ww*dth)/ww.sum()
        return float(mu), float(np.sqrt(np.sum(ww*(dth-mu)**2)/ww.sum()))
    for mat in ('lead', 'iron'):
        rho = muons.density(mat); rows = []
        for t in ABSORBER_CM:
            pc = muons.p_from_range(rho*t/c, mat) if t > 0 else np.zeros(len(c))
            ww = w*k0*(p > pc)
            mu, rms = stats(ww)
            m = counter_coord(z, name, run, c, 10.0*t)
            wc = w*sel*top*(p > pc)*(m < half); muc, rmsc = stats(wc)
            ws = w*sel*top*(p > pc); o = np.argsort(m); cw = np.cumsum(ws[o])
            i90 = min(np.searchsorted(cw, KEEP_REF*ww.sum()), len(cw)-1)
            rows.append(dict(t_cm=t, p_cut_vertical=float(muons.p_from_range(rho*t, mat)) if t > 0 else 0.0, keep=float(ww.sum()/base),
                             rate_h=float(F*ww.sum()/len(c)), mean=mu, rms=rms, keep_counter=float(wc.sum()/base),
                             rate_counter_h=float(F*wc.sum()/len(c)), mean_counter=muc, rms_counter=rmsc,
                             side90_mm=float(2*m[o][i90])))
        out['absorber_%s' % mat] = rows
    key = '%s/br_gas_det_qe' % name
    if key in z.files:
        ng = z[key]; rows = []
        for kmin in (1, 2, 3, 5):
            ptag = 1-np.exp(-ng)*sum(ng**j/np.prod(np.arange(1, j+1)) for j in range(kmin))
            ww = w*k0*ptag; mu, rms = stats(ww)
            rows.append(dict(kmin=kmin, keep=float(ww.sum()/base), rate_h=float(F*ww.sum()/len(c)), mean=mu, rms=rms,
                             mean_gas_photons=float(np.sum(w*k0*ng)/base)))
        out['gas_tag'] = rows
    mu, rms = stats(w*k0)
    out['none'] = dict(keep=1.0, rate_h=float(F*base/len(c)), mean=mu, rms=rms)
    return out


def sigma_photon(run, origin, theta_obs):
    """Per-photon angular error [rad]: pixel term (plus Xe dispersion; plus emission depth in configuration 4)."""
    pix = PIXEL_PITCH/np.sqrt(12)
    if run['group'] == 'c4':
        cos2 = np.cos(theta_obs)**2; tha = theta_c(tile_n(run['P'], run['variant']))
        return float(np.hypot(pix, AEROGEL_T*np.tan(tha)/np.sqrt(12))*cos2/run['gap'])
    s = pix/(run['mirror'][0]/2)
    if origin == 'gas':
        s = np.hypot(s, _G['chrom'][run['P']])
    return float(s)


def precision(run, origin, R, nk):
    """Statistical precision [mrad] on the mean ring angle after one hour of >= 80 % rings."""
    d = R[nk]; rate = d['r80_guan'][0] if 'r80_guan' in d else d['r80_b1'][0]
    if rate <= 0 or 'theta_obs' not in R:
        return None
    sp = sigma_photon(run, origin, R['theta_obs'])
    return float(1e3*np.sqrt(sp**2*d['inv_photons_r80']+R['theta_obs_rms']**2)/np.sqrt(rate)), float(1e3*sp/np.sqrt(rate*max(d['photons_mean_r80'], 1e-9)))


def main():
    ap = cli.parser(__doc__)
    ap.add_argument('--input', type=Path, required=True, help='directory with cosmic_rates_<group>.npz and cosmic_rates_meta.json')
    ap.add_argument('--nproc', type=int, default=16, help='worker processes (default 16)')
    ap.add_argument('--short-f', type=Path, default=None, help='short_f_angles.npz of the short-f tracer: adds the offset term of the ring radius')
    args = ap.parse_args(); out = cli.outdir(args)
    with open(args.input/'cosmic_rates_meta.json') as fm:
        meta = json.load(fm)
    runs = meta['runs']
    _G['spec'] = MuonSpectrum(meta['args'].get('spectrum', muons.SPECTRUM_DEFAULT))
    _G['input'] = args.input
    _G['top_half'] = meta['top_half']; _G['bot_half'] = meta['bot_half']
    _G['rshift'], _G['dr_dtheta'] = load_rshift(args.short_f) if args.short_f is not None else (None, None)
    with mp.get_context('fork').Pool(min(args.nproc, len(PRESSURES))) as pool:
        _G['xe'] = dict(zip(PRESSURES, pool.map(xe_table, PRESSURES)))
        _G['chrom'] = dict(zip(PRESSURES, pool.map(xe_chromatic_rms, PRESSURES)))
    _G['ray_c4'] = rayleigh_prox()
    # Rayleigh transmission of the aerogel photons on the SiPM: unscattered with scattering over all without, same population, beta = 1
    _G['ray_tile'] = {}
    for r in runs:
        if r['rayleigh'] and 'tile' in r['origins']:
            z = np.load(args.input/('cosmic_rates_%s.npz' % r['group'])); ref = r['name'].replace('_ray', '')
            t = z[r['name']+'/b1_tile_det_qe_clean'].mean()/z[ref+'/b1_tile_det_qe'].mean()
            _G['ray_tile'][(r['group'], tuple(r['mirror']), 'air' if r['P'] == 0 else 'xe')] = float(t)
    _G['ray_tile_default'] = float(np.mean([v for k, v in _G['ray_tile'].items() if k[2] == 'xe']))
    print('Rayleigh transmission of the aerogel ring photons (FT x QE): %s; configuration 4 (computed): %.3f; configuration 5 tile light uses %.3f'
          % (', '.join('%s %s %s: %.3f' % (k[0], k[1], k[2], v) for k, v in _G['ray_tile'].items()), _G['ray_c4'], _G['ray_tile_default']))
    with mp.get_context('fork').Pool(args.nproc) as pool:
        res = pool.map(analyse_run, [(r['group'], r) for r in runs])
    R = {r['name']: x for r, x in zip(runs, res)}
    run_of = {r['name']: r for r in runs}
    sh, th_, bh = meta['sipm_half'], meta['top_half'], meta['bot_half']
    trig = {'mirror_f%d' % f: trigger_rates(sh, f, 0.0, bh['mirror'])
            for f in [MIRROR_R/2]+sorted({r['mirror'][0]/2 for r in runs if r['group'] == 'c23'})}
    for gap in sorted({r['gap'] for r in runs if r['group'] == 'c4'}):
        trig['prox_g%g' % gap] = trigger_rates(th_, 0.0, -AEROGEL_T-gap, bh['prox'])
    for name, x in R.items():
        for k in [k for k in x if k.startswith(('gas_', 'tile_'))]:
            origin = k.split('_')[0]
            for nk in ('plane', 'tel'):
                pr = precision(run_of[name], origin, x[k], nk)
                x[k][nk]['precision_mrad_1h'] = pr
    with open(out/'cosmic_rates_summary.json', 'w') as fo:
        json.dump(dict(runs=R, trigger=trig, rayleigh=dict(tile={'%s %s %s' % k: v for k, v in _G['ray_tile'].items()}, c4=_G['ray_c4']),
                       chromatic_rms_mrad={str(k): 1e3*v for k, v in _G['chrom'].items()},
                       pore_signal_mrad=[1e3*x for x in pore_signal(PRESSURES[-1])],
                       dr_dtheta=_G['dr_dtheta']), fo, indent=1)
    tables(R, run_of, trig, out)
    figures(R, run_of, out)
    print('cores used: %d' % args.nproc)


def fmt(v, digits=1):
    return ('%.*f +- %.*f' % (digits, v[0], digits, v[1])) if v is not None else '-'


def rate(R, name, origin, tag, nk, d, model='guan'):
    x = R[name]['%s_%s' % (origin, tag)][nk]
    return x['%s_%s' % (d, 'b1' if tag == 'b1' else model)]


def tables(R, run_of, trig, out):
    """Task tables (stdout and cosmic_rates_tables.md)."""
    L = []
    L.append('## Configuration rates (rings per hour; beta = 1 = cos^2, I0 = %g, beta = 1; real = Guan spectrum and per-muon beta; mean +- seed std)\n' % COSMIC_I0)
    L.append('| Configuration | Light | Normalisation | Trigger /h (beta = 1 / real) | Definition | beta = 1 /h | real /h | real /day | real / beta = 1 | spectrum band /h | FT x QE photons per >= 80 % ring (real) | precision on the mean angle after 1 h [mrad] (real; pixel-only) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = []
    for P in PRESSURES:
        rows.append(('1, Xe %g bar' % P, 'c1_%s' % gas_label(P), 'gas', 'mirror_f203'))
    for mk in ('f50', 'f76'):
        rows.append(('2, air, %s' % mk, 'c2_%s_plain' % mk, 'tile', 'mirror_%s' % mk))
        for P in (1.0, 2.0, 3.0):
            rows.append(('3, Xe %g bar, %s' % (P, mk), 'c3_%s_%s_plain' % (mk, gas_label(P)), 'tile', 'mirror_%s' % mk))
    for gap in (30, 50, 70):
        rows.append(('4a, air, gap %d mm' % gap, 'c4_air_g%d_plain' % gap, 'tile', 'prox_g%d' % gap))
        for P in (1.0, 2.0, 3.0):
            rows.append(('4b, Xe %g bar, gap %d mm' % (P, gap), 'c4_%s_g%d_plain' % (gas_label(P), gap), 'tile', 'prox_g%d' % gap))
    for P in PRESSURES:
        rows.append(('5, Xe %g bar (gas ring)' % P, 'c5_%s_plain_geo' % gas_label(P), 'gas', 'mirror_f203'))
        rows.append(('5, Xe %g bar (aerogel light)' % P, 'c5_%s_plain' % gas_label(P), 'tile', 'mirror_f203'))
    summary = []
    for lab, name, origin, tk in rows:
        if name not in R:
            continue
        for nk in ('plane', 'tel'):
            t = trig[tk]
            for d in ('full', 'r80', 'arc50'):
                b1 = rate(R, name, origin, 'b1', nk, d); re = rate(R, name, origin, 'br', nk, d)
                band = [rate(R, name, origin, 'br', nk, d, m)[0] for m in ('guan', 'reyna', 'shukla')]
                xr = R[name]['%s_br' % origin][nk]
                pr = xr.get('precision_mrad_1h')
                L.append('| %s | %s | %s | %.0f / %.0f | %s | %s | %s | %.0f | %s | %.1f-%.1f | %s | %s |' % (
                    lab, origin, 'SiPM/tile plane' if nk == 'plane' else 'telescope', t['%s_b1' % nk], t['%s_real' % nk],
                    {'full': 'full', 'r80': '>= 80 %', 'arc50': 'arc >= 50 %'}[d], fmt(b1), fmt(re), 24*re[0],
                    ('%.2f' % (re[0]/b1[0])) if b1[0] > 0 else '-', min(band), max(band),
                    ('%.1f' % xr['photons_mean_r80']) if d == 'r80' else '',
                    ('%.2f; %.2f' % pr) if (pr and d == 'r80') else ''))
                summary.append(dict(config=lab, name=name, origin=origin, norm=nk, definition=d, beta1=b1, real=re, band=[min(band), max(band)],
                                    photons=xr['photons_mean_r80'], precision=pr if d == 'r80' else None))
    with open(out/'cosmic_rates_table.json', 'w') as fo:
        json.dump(summary, fo, indent=1)
    # larger mirror layouts (docs/design_options.md section 6.1)
    L.append('\n## Larger mirror layouts with the tile (Xe 3 bar, muons uniform over the 5 x 5 cm tile; per hour; beta = 1 / real)\n')
    L.append('| SiPM | Mirror | Tile | Aerogel arcs >= 50 % (any photon per bin, as the beta = 1 toys) | Aerogel arcs >= 50 % (half the bin light) | Full aerogel rings (arc >= 90 %, any photon) | Gas rings (>= 80 % of the photons on the SiPM) | Emitted / detected aerogel FT x QE per muon | Emitted / detected gas FT x QE per muon |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for name, x in R.items():
        if x['group'] != 's61':
            continue
        r = run_of[name]
        a50 = (rate(R, name, 'tile', 'b1', 'plane', 'arc50_any'), rate(R, name, 'tile', 'br', 'plane', 'arc50_any'))
        a90 = (rate(R, name, 'tile', 'b1', 'plane', 'arc90_any'), rate(R, name, 'tile', 'br', 'plane', 'arc90_any'))
        h50 = (rate(R, name, 'tile', 'b1', 'plane', 'arc50'), rate(R, name, 'tile', 'br', 'plane', 'arc50'))
        gas = ''; gph = ''
        if 'gas_b1' in x:
            g1 = rate(R, name, 'gas', 'b1', 'plane', 'fd80'); g2 = rate(R, name, 'gas', 'br', 'plane', 'fd80')
            gas = '%.1f / %.1f' % (g1[0], g2[0])
            gph = '%.1f / %.2f ; %.1f / %.2f' % (x['gas_b1']['emit_mean_all'], x['gas_b1']['photons_mean_all'], x['gas_br']['emit_mean_all'], x['gas_br']['photons_mean_all'])
        L.append('| %.1f cm | %.0f mm | %s | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %s | %.1f / %.2f ; %.1f / %.2f | %s |' % (
            2*r['sensor']/10, r['mirror'][1], r['variant'], a50[0][0], a50[1][0], h50[0][0], h50[1][0], a90[0][0], a90[1][0], gas,
            x['tile_b1']['emit_mean_all'], x['tile_b1']['photons_mean_all'], x['tile_br']['emit_mean_all'], x['tile_br']['photons_mean_all'], gph))
    # aerogel angle shifts
    L.append('\n## Aerogel-angle shift from the muon velocity, >= 80 % rings, real spectrum (mrad; in the tile)\n')
    L.append('| Run | mean | rms | 5 / 16 / 50 / 84 / 95 % |')
    L.append('|---|---|---|---|')
    for name, x in R.items():
        if x['group'] in ('c23', 'c4') and 'tile_br' in x and 'dtheta' in x['tile_br'] and not x['rayleigh']:
            d = x['tile_br']['dtheta']
            L.append('| %s | %.2f | %.2f | %s |' % (name, d['mean'], d['rms'], ' / '.join('%.1f' % q for q in d['q'])))
    L.append('\n## Expected Xe-minus-air difference of the aerogel angle in the tile, >= 80 % rings, matched muons (mrad; mean over the seeds'
             ' +- its error, seed std / sqrt(seeds))\n')
    L.append('| Geometry | Xe | Tile in Xe | beta = 1 (= expected signal) | real spectrum, plain mean | real, window -10 to +20 mrad around the beta = 1 angle'
             ' of the plain tile (fraction kept) | real, window around each run\'s own beta = 1 angle (fraction kept) | real, %g %% quantile'
             ' (beta -> 1 edge) | real, rings of muons with p > 2 GeV/c |' % (100*EDGE_Q))
    L.append('|---|---|---|---|---|---|---|---|---|')
    pairs = [('c2_%s_%%s' % mk, 'c3_%s_%%s_%%s' % mk, 'mirror %s' % mk) for mk in ('f50', 'f76')] + \
        [('c4_air_g%d_%%s' % g, 'c4_%%s_g%d_%%s' % g, 'proximity, gap %d mm' % g) for g in (30, 50, 70)]
    for rpat, pat, lab in pairs:
        for P in (1.0, 2.0, 3.0):
            for v in ('plain', 'pore-vac', 'pore-air'):
                # air run of the same tile variant; the pore-air tile in air is the plain tile
                nm = pat % (gas_label(P), v); ref = rpat % ('plain' if v == 'pore-air' else v)
                if nm not in R or ref not in R:
                    continue
                cells = []
                for tag, key, frac in (('b1', 'theta_tile_seed', None), ('br', 'theta_tile_seed', None), ('br', 'theta_tile_win_seed', 'win_fraction'),
                                       ('br', 'theta_tile_ownwin_seed', 'ownwin_fraction'), ('br', 'theta_tile_edge_seed', None)):
                    a = np.array(R[nm]['tile_%s' % tag][key]); b = np.array(R[ref]['tile_%s' % tag][key]); d = a-b
                    cells.append('%.2f +- %.2f' % (d.mean(), d.std(ddof=1)/np.sqrt(len(d)))+(' (%.2f)' % R[nm]['tile_br'][frac] if frac else ''))
                hp = (R[nm]['tile_br'].get('theta_tile_mean_p2'), R[ref]['tile_br'].get('theta_tile_mean_p2'))
                cells.append('%.2f' % (hp[0]-hp[1]) if None not in hp else '-')
                L.append('| %s | %g bar | %s | %s |' % (lab, P, v, ' | '.join(cells)))
    L.append('\n## Configurations 2 and 3: aerogel-angle spread of >= 80 % rings, velocity and impact-offset terms (mrad; mean / rms)\n')
    L.append('| Run | beta | velocity | offset (f/1 sphere) | sum | mean offset at the tile mid-plane [mm] |')
    L.append('|---|---|---|---|---|---|')
    for name, x in R.items():
        if x['group'] == 'c23' and not x['rayleigh'] and x['variant'] == 'plain':
            for tag in ('b1', 'br'):
                o = x.get('tile_%s' % tag, {}).get('offset_term')
                if o:
                    L.append('| %s | %s | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %.1f |' % (name, 'β = 1' if tag == 'b1' else 'real', *o['velocity'], *o['offset'], *o['total'], o['rho_mean']))
    txt = '\n'.join(L)
    with open(out/'cosmic_rates_tables.md', 'w') as fo:
        fo.write(txt+'\n')
    print(txt)


def figures(R, run_of, out):
    """Figures of the rates study."""
    # configuration 1 and 5 rates vs pressure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, (d, lab) in enumerate((('r80', '>= 80 % rings'), ('full', 'full rings'), ('arc50', 'arcs >= 50 %'))):
        for cfg, nm, ls in (('1', 'c1_%s', '-'), ('5', 'c5_%s_plain_geo', '--')):
            b1 = [rate(R, nm % gas_label(P), 'gas', 'b1', 'tel', d)[0] for P in PRESSURES]
            re = np.array([rate(R, nm % gas_label(P), 'gas', 'br', 'tel', d) for P in PRESSURES])
            band = np.array([[rate(R, nm % gas_label(P), 'gas', 'br', 'tel', d, m)[0] for m in ('guan', 'reyna', 'shukla')] for P in PRESSURES])
            ax[i].plot(PRESSURES, b1, 'o'+ls, color=C_B1, label='config %s, β = 1 (cos², I0 = %g)' % (cfg, COSMIC_I0))
            ax[i].errorbar(PRESSURES, re[:, 0], yerr=re[:, 1], fmt='s'+ls, color=C_REAL, label='config %s, real spectrum (Guan)' % cfg)
            ax[i].fill_between(PRESSURES, band.min(1), band.max(1), color=C_REAL, alpha=0.12)
        ax[i].set_xlabel('Xe pressure [bar abs]'); ax[i].set_ylabel('rings per hour (telescope)'); ax[i].grid(alpha=.3)
        ax[i].set_title('Gas %s' % lab, loc='left', fontsize=10); ax[i].set_ylim(0, None)
    ax[0].legend(fontsize=7.5)
    fig.suptitle('Configurations 1 and 5: gas-ring rate vs pressure (band: Guan / Reyna / Shukla; error bars: seed spread)', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94)); save(out, 'cosmic_rates_c15_vs_pressure.png', dpi=130, fig=fig); plt.close(fig)
    # radius and photon-count distributions, configuration 1
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(PRESSURES)))
    for P, col in zip(PRESSURES, cols):
        x = R['c1_%s' % gas_label(P)]
        rb = np.arange(0, 30.25, 0.25)[:len(x['gas_br']['radius_hist'])+1]
        ax[0].stairs(x['gas_br']['radius_hist'], rb, color=col, lw=1.6, label='%g bar, real' % P)
        h1 = np.array(x['gas_b1']['radius_hist']); ax[0].stairs(h1*np.sum(x['gas_br']['radius_hist'])/max(h1.sum(), 1e-12), rb, color=col, lw=0.8, ls=':')
        pb = np.arange(0, 82, 2)
        ax[1].stairs(x['gas_br']['photon_hist'], pb, color=col, lw=1.6, label='%g bar, real' % P)
        h1 = np.array(x['gas_b1']['photon_hist']); ax[1].stairs(h1*np.sum(x['gas_br']['photon_hist'])/max(h1.sum(), 1e-12), pb, color=col, lw=0.8, ls=':')
    ax[0].set_xlim(0, 16); ax[0].set_xlabel('ring radius f tan θ [mm]'); ax[0].set_ylabel('>= 80 % rings per hour per bin (SiPM plane)')
    ax[0].set_title('Ring radius (dotted: β = 1, same total)', loc='left', fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_xlabel('FT×QE photons on the SiPM per ring'); ax[1].set_ylabel('>= 80 % rings per hour per bin (SiPM plane)')
    ax[1].set_title('Photons per ring (dotted: β = 1, same total)', loc='left', fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); save(out, 'cosmic_rates_c1_radius_photons.png', dpi=130, fig=fig); plt.close(fig)
    # containment maps, configuration 1 at 3 and 2 bar
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, (P, tag, lab) in enumerate(((3.0, 'gas_br', 'real'), (3.0, 'gas_b1', 'β = 1'))):
        m = R['c1_xe3'][tag]['maps']
        im = ax[i].pcolormesh(m['rb'], m['zb'], np.ma.masked_less(np.array(m['map']), 0), vmin=0, vmax=1, cmap='Blues')
        ax[i].set_xlabel('mirror-plane impact offset from the axis [mm]'); ax[i].set_ylabel('muon zenith angle [deg]')
        ax[i].set_title('Config 1, Xe %g bar, %s: P(>= 80 %% ring)\n(all triggered muons, below threshold included)' % (P, lab), loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02)
    for P, col in zip(PRESSURES, cols):
        for tag, ls in (('gas_br', '-'), ('gas_b1', ':')):
            m = R['c1_%s' % gas_label(P)][tag]['maps']; zb = np.array(m['zb'])
            cv = np.array(m['curve']); cv[cv < 0] = np.nan
            ax[2].plot(0.5*(zb[1:]+zb[:-1]), cv, ls, color=col, label='%g bar%s' % (P, '' if tag == 'gas_br' else ' (β = 1)') if tag == 'gas_br' or P == 3.0 else None)
    ax[2].set_xlabel('muon zenith angle [deg]'); ax[2].set_ylabel('P(>= 80 % ring), offset < 8 mm'); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7.5)
    ax[2].set_title('Near the axis: real β (solid), β = 1 (dotted)', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_rates_c1_containment.png', dpi=130, fig=fig); plt.close(fig)
    # aerogel-angle shift distributions, configurations 2-4
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    edges = np.arange(HIST_LO, 3, 1.0); ps = [1e3*x for x in pore_signal(PRESSURES[-1])]
    for i, (names, title) in enumerate(((['c2_f50_plain', 'c2_f76_plain'], 'Config 2 (air), f = 50.8 / 76.2 mm'),
                                        (['c3_f50_xe3_plain', 'c3_f76_xe3_plain'], 'Config 3 (Xe 3 bar), f = 50.8 / 76.2 mm'),
                                        (['c4_air_g30_plain', 'c4_xe3_g30_plain', 'c4_air_g50_plain', 'c4_xe3_g50_plain'], 'Config 4, gaps 30 / 50 mm, air / Xe 3 bar'))):
        for nm, col in zip(names, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')):
            x = R[nm]['tile_br']; h = np.array(x['dtheta_hist']); h = h/max(h.sum(), 1e-12)
            ax[i].stairs(h, edges, color=col, lw=1.5, label='%s: mean %.1f, rms %.1f mrad' % (nm, x['dtheta']['mean'], x['dtheta']['rms']))
        ax[i].axvspan(*ps, color='#e87ba4', alpha=0.3, label='pore-gas signal at Xe 3 bar (+%.1f / +%.1f)' % tuple(ps))
        ax[i].set_xlim(HIST_LO, 12); ax[i].set_yscale('log'); ax[i].set_ylim(1e-5, 1)
        ax[i].set_xlabel('θ_aero(p) − θ_aero(β = 1) [mrad]'); ax[i].set_ylabel('fraction of >= 80 % rings per mrad'); ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=7); ax[i].set_title(title, loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_rates_aerogel_shift.png', dpi=130, fig=fig); plt.close(fig)
    # larger mirror layouts (docs/design_options.md section 6.1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    names = [n for n, x in R.items() if x['group'] == 's61' and x['variant'] == 'plain']
    labs = ['%.1f cm / %.0f mm' % (2*run_of[n]['sensor']/10, run_of[n]['mirror'][1]) for n in names]
    xx = np.arange(len(names))
    for j, (d, title) in enumerate((('arc50_any', 'Aerogel arcs >= 50 % per hour (any photon per azimuth bin)'), ('fd80', 'Gas rings (>= 80 % on the SiPM) per hour'))):
        org = 'tile' if d.startswith('arc') else 'gas'
        b1 = [rate(R, n, org, 'b1', 'plane', d)[0] for n in names]
        ax[j].bar(xx-0.3, b1, 0.2, color=C_B1, label='β = 1')
        for k, v in enumerate(('plain', 'pore-vac', 'pore-air')):
            nn = [n.replace('_plain', '_'+v) for n in names]
            if org == 'gas' and v != 'plain':
                continue
            re = np.array([rate(R, n, org, 'br', 'plane', d) for n in nn])
            ax[j].bar(xx-0.1+0.2*k, re[:, 0], 0.2, yerr=re[:, 1], color=C_VAR[v], label='real, tile %s' % v)
        ax[j].set_xticks(xx); ax[j].set_xticklabels(labs, rotation=45, ha='right', fontsize=8); ax[j].grid(alpha=.3, axis='y')
        ax[j].set_title(title, loc='left', fontsize=10); ax[j].legend(fontsize=7.5); ax[j].set_ylabel('per hour (SiPM / mirror)')
    plt.tight_layout(); save(out, 'cosmic_rates_s61_layouts.png', dpi=130, fig=fig); plt.close(fig)
    # momentum handles
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for j, (nm, title) in enumerate((('c3_f50_xe3_plain', 'Configuration 3, #%s (f = 50.8 mm), Xe 3 bar' % SHORT_F_KEY['f50']),
                                     ('c4_xe3_g30_plain', 'Configuration 4b, Xe 3 bar, gap 30 mm'))):
        h = R[nm]['tile_br'].get('handles')
        if h is None:
            continue
        for mat, mk in (('lead', 'o-'), ('iron', 's--')):
            rows = h['absorber_%s' % mat]
            ax[j].plot([r['rate_h'] for r in rows], [r['rms'] for r in rows], mk, color='#2a78d6' if mat == 'lead' else '#1baf7a',
                       label='%s absorber 0-100 cm (rms), counter enlarged' % mat)
            if mat == 'lead':
                ax[j].plot([r['rate_counter_h'] for r in rows], [r['rms_counter'] for r in rows], 'v:', color='#6d597a',
                           label='lead, same counter behind the absorber (rms)')
                for r in rows:
                    if r['t_cm'] in (20, 50, 100):
                        ax[j].annotate('%g cm' % r['t_cm'], (r['rate_counter_h'], r['rms_counter']), fontsize=7, xytext=(3, -9),
                                       textcoords='offset points', color='#6d597a')
            for r in rows:
                if r['t_cm'] in (0, 20, 50, 100) and mat == 'lead':
                    ax[j].annotate('%g cm' % r['t_cm'], (r['rate_h'], r['rms']), fontsize=7, xytext=(3, 3), textcoords='offset points')
            ax[j].plot([r['rate_h'] for r in rows], [abs(r['mean']) for r in rows], mk, color='#2a78d6' if mat == 'lead' else '#1baf7a', alpha=0.4, ms=3,
                       label='%s absorber (|mean|)' % mat)
        if 'gas_tag' in h:
            for r in h['gas_tag']:
                ax[j].plot(r['rate_h'], r['rms'], 'D', color='#eb6834', label='Xe gas-ring tag, >= %d photons' % r['kmin'] if r['kmin'] == 1 else None)
                ax[j].annotate('≥%d' % r['kmin'], (r['rate_h'], r['rms']), fontsize=7, xytext=(3, -9), textcoords='offset points')
        ax[j].axhspan(*[1e3*x for x in pore_signal(PRESSURES[-1])], color='#e87ba4', alpha=0.25, label='pore-gas signal, Xe 3 bar')
        ax[j].set_xscale('log'); ax[j].set_xlabel('>= 80 % rings per hour (telescope)'); ax[j].set_ylabel('velocity smear of the aerogel angle [mrad]')
        ax[j].set_yscale('log'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=7); ax[j].set_title(title, loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_rates_handles.png', dpi=130, fig=fig); plt.close(fig)
    # summary of the configurations
    rows = [('1 Xe %g' % P, 'c1_%s' % gas_label(P), 'gas') for P in (1.0, 2.0, 3.0)] + \
        [('2 air f%s' % k, 'c2_f%s_plain' % k, 'tile') for k in ('50', '76')] + \
        [('3 Xe %g f%s' % (P, k), 'c3_f%s_%s_plain' % (k, gas_label(P)), 'tile') for k in ('50', '76') for P in (1.0, 2.0, 3.0)] + \
        [('4a air g%d' % g, 'c4_air_g%d_plain' % g, 'tile') for g in (30, 50, 70)] + \
        [('4b Xe 3 g%d' % g, 'c4_xe3_g%d_plain' % g, 'tile') for g in (30, 50, 70)] + \
        [('5 Xe %g gas' % P, 'c5_%s_plain_geo' % gas_label(P), 'gas') for P in (1.0, 2.0, 3.0)]
    fig, ax = plt.subplots(figsize=(15, 5))
    xx = np.arange(len(rows))
    for k, (nk, off, lab) in enumerate((('tel', -0.2, 'telescope'), ('plane', 0.2, 'SiPM / tile plane'))):
        b1 = [rate(R, n, o, 'b1', nk, 'r80')[0] for _, n, o in rows]; re = np.array([rate(R, n, o, 'br', nk, 'r80') for _, n, o in rows])
        ax.bar(xx+off-0.1, b1, 0.2, color=C_B1, alpha=1 if nk == 'tel' else 0.5, label='β = 1, %s' % lab)
        ax.bar(xx+off+0.1, re[:, 0], 0.2, yerr=re[:, 1], color=C_REAL, alpha=1 if nk == 'tel' else 0.5, label='real, %s' % lab)
    ax.set_yscale('log'); ax.set_xticks(xx); ax.set_xticklabels([r[0] for r in rows], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('>= 80 % rings per hour'); ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=8, ncol=2)
    ax.set_title('Expected >= 80 % ring rates per configuration: β = 1 toys vs real momentum spectrum', loc='left', fontsize=10)
    plt.tight_layout(); save(out, 'cosmic_rates_summary.png', dpi=130, fig=fig); plt.close(fig)
    # rate vs minimum photon count
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    groups = ((rows[:3]+rows[-3:], 'Configurations 1 and 5 (gas ring)'), (rows[3:11], 'Configurations 2 and 3 (aerogel ring)'), (rows[11:17], 'Configuration 4'))
    for i, (rr, title) in enumerate(groups):
        cols = plt.cm.tab10(np.arange(len(rr)) % 10)
        for (lab, n, o), col in zip(rr, cols):
            ax[i].plot(NMIN, R[n]['%s_br' % o]['tel']['nmin_curve'], color=col, lw=1.6, label=lab)
            ax[i].plot(NMIN, R[n]['%s_b1' % o]['tel']['nmin_curve'], color=col, lw=0.8, ls=':')
        ax[i].set_yscale('log'); ax[i].set_ylim(0.1, None); ax[i].set_xlabel('minimum FT×QE photons on the SiPM per ring')
        ax[i].set_ylabel('>= 80 % rings per hour (telescope)'); ax[i].grid(alpha=.3); ax[i].legend(fontsize=7); ax[i].set_title(title+' (dotted: β = 1)', loc='left', fontsize=9.5)
    plt.tight_layout(); save(out, 'cosmic_rates_vs_min_photons.png', dpi=130, fig=fig); plt.close(fig)


if __name__ == '__main__':
    main()
