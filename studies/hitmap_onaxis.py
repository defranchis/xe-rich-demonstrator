"""Mean SiPM hit density for a vertical muon on the mirror axis, phase-1 layout with the aerogel tile.

Gases Xe 2 bar, Xe 3 bar (20 degC) and C4F10 1 bar fill the vessel; aerogel n = 1.03 (constant). Photons are
sampled from the Frank-Tamm spectrum over the band where the ARC SiPM QE is non-zero, traced with Snell,
Fresnel (polarised) and TIR at the tile faces, and counted with the FT x QE and the pure FT weighting
(stdout, hitmap_onaxis_<gas>_<weighting>.png, hitmap_onaxis_radial_gases.png, hitmap_onaxis_summary.csv/.md,
hitmap_onaxis.npz). An unpolarised-Fresnel rerun with the same seeds quantifies the polarisation effect.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from xerich import cli, sim  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band  # noqa: E402
from xerich.constants import (AEROGEL_HALF, AEROGEL_N_AIR, AEROGEL_T, HC, MIRROR_CLEAR_FRAC, MIRROR_D, MIRROR_R,  # noqa: E402
                              SENSOR_HALF_2IN)
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor  # noqa: E402
from xerich.media import C4F10_E_EV, aerogel_constant, c4f10, xenon  # noqa: E402
from xerich.plotting import AEROGEL, GAS, INK, MUTED, RADIATOR, plt, save  # noqa: E402
from xerich.tracer import CLASSES, REASONS  # noqa: E402

GASES = (('xe2', xenon(2)), ('xe3', xenon(3)), ('c4f10', c4f10()))
WEIGHTINGS = ('ftqe', 'ft')
WLABEL = {'ftqe': 'Frank-Tamm x ARC SiPM QE', 'ft': 'Frank-Tamm (no QE)'}
ORIGINS = ('gas', 'aerogel')
MAP_BIN = 0.2                      # [mm]
R_BIN = 0.05                       # [mm]
R_MAX = 36.0                       # beyond the sensor half diagonal [mm]
CLASS_STYLE = {'direct': '-', 'bottom': '--', 'top': '-.', 'TIR': ':', 'multiple': (0, (1, 1))}

_CFG = {}


def layout(gas):
    return Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), gas, Aerogel(AEROGEL_N_AIR),
                        tile_medium=aerogel_constant())


def run_chunk(task):
    """Trace one chunk; returns weight sums for (weighting, origin, ...) in photons per muon."""
    gi, pol, ci = task
    c = _CFG
    lay = c['layouts'][gi]
    rng = np.random.default_rng(c['seeds'][ci])
    ph, e = sim.onaxis_muon(lay, rng, c['spectra'][gi], c['n_gas'], c['n_tile'], c['max_refl'], pol)
    w0 = ph['weight']; qe = qe_arc(HC/ph['lam']); org = ph['origin']
    wsets = np.stack([w0*qe, w0])                                     # (weighting, photon)
    emit = np.array([[np.sum(ws[org == o]) for o in (0, 1)] for ws in wsets])
    o = org[e['i']].astype(np.intp); we = wsets[:, e['i']]*e['w']                     # (weighting, branch)
    reason = np.zeros((2, 2, len(REASONS)))
    key = o*len(REASONS)+e['reason']
    for k in range(2):
        reason[k] = np.bincount(key, we[k], minlength=2*len(REASONS)).reshape(2, -1)
    det = e['reason'] == 0
    x, y = e['p'][det, 0], e['p'][det, 1]; r = np.hypot(x, y); oc = o[det]*len(CLASSES)+e['cls'][det]; wd = we[:, det]
    nr = c['nr']; ir = np.minimum((r/R_BIN).astype(int), nr-1)
    radial = np.stack([np.bincount(oc*nr+ir, wd[k], minlength=2*len(CLASSES)*nr).reshape(2, len(CLASSES), nr)
                       for k in range(2)])
    cls = radial.sum(-1)
    out = dict(emit=emit, reason=reason, cls=cls, radial=radial, nbranch=len(e['i']))
    g = (oc == 0)                                                      # gas origin, direct class
    out['ring'] = np.array([[wd[k, g].sum(), (wd[k, g]*r[g]).sum(), (wd[k, g]*r[g]**2).sum()] for k in range(2)])
    if pol:
        nb = c['nb']; H = c['half']
        ix = np.clip(((x+H)/MAP_BIN).astype(int), 0, nb-1); iy = np.clip(((y+H)/MAP_BIN).astype(int), 0, nb-1)
        idx = (oc*nb+ix)*nb+iy
        out['map'] = np.stack([np.bincount(idx, wd[k], minlength=2*len(CLASSES)*nb*nb).reshape(2, len(CLASSES), nb, nb)
                               for k in range(2)])
    return out


def reduce(results):
    acc = {}
    for res in results:
        for k, v in res.items():
            acc[k] = acc[k]+v if k in acc else v
    return acc


def paraxial_radius(spec_w, f):
    """Mean and rms of f tan(theta_c(E)) over the weighted spectrum (no aberration, no tile)."""
    t = f*np.tan(np.arccos(1/spec_w.index(HC/spec_w.E))); p = spec_w.pdf
    m = np.trapezoid(p*t, spec_w.E)/np.trapezoid(p, spec_w.E)
    return m, np.sqrt(np.trapezoid(p*(t-m)**2, spec_w.E)/np.trapezoid(p, spec_w.E))


def summarise(acc):
    """Per-weighting summary dict from the reduced chunk sums."""
    out = {}
    for k, wn in enumerate(WEIGHTINGS):
        s = {'emit': acc['emit'][k], 'cls': acc['cls'][k], 'reason': acc['reason'][k]}
        sw, s1, s2 = acc['ring'][k]
        s['r_mean'] = s1/sw; s['r_rms'] = np.sqrt(s2/sw-(s1/sw)**2)
        s['lost_frac'] = acc['reason'][k][:, 1:]/acc['emit'][k][:, None]
        out[wn] = s
    return out


def sensor_panel(ax, img, half, norm, title, cmap):
    im = ax.imshow(img.T, origin='lower', extent=(-half, half, -half, half), norm=norm, cmap=cmap,
                   interpolation='nearest')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title(title, fontsize=10); ax.set_aspect('equal')
    return im


def masked(a):
    return np.ma.masked_less_equal(a, 0)


def case_figure(out, name, label, wn, maps, radial, s, rpar, lay):
    """2x3 figure: total map (linear, log), radial profile, gas Fresnel-reflected map, aerogel map, numbers."""
    half = SENSOR_HALF_2IN; area = MAP_BIN**2; rc = (np.arange(radial.shape[-1])+0.5)*R_BIN
    dens = maps/area; tot = dens.sum((0, 1))
    vmax = tot.max(); vmin = vmax*1e-9
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.2))
    cmap = plt.get_cmap('magma_r').copy(); cmap.set_bad('white')
    im = sensor_panel(ax[0, 0], tot, half, None, 'All photons, linear scale', cmap)
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.03, label='photons / mm² / muon')
    lnorm = LogNorm(vmin, vmax)
    for a, img, t in ((ax[0, 1], tot, 'All photons, log scale'),
                      (ax[1, 0], dens[0, 1:].sum(0), 'Gas photons reflected in or at the tile (log)'),
                      (ax[1, 1], dens[1].sum(0), 'Aerogel photons, all paths (log)')):
        im = sensor_panel(a, masked(img), half, lnorm, t, cmap)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label='photons / mm² / muon')
    a = ax[0, 2]
    for oi, (org, col) in enumerate(zip(ORIGINS, (GAS, AEROGEL))):
        for ci, cl in enumerate(CLASSES):
            y = radial[oi, ci]/R_BIN
            if y.sum() > 0:
                a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=col, ls=CLASS_STYLE[cl], lw=1.3,
                       label='%s, %s' % (org, cl))
    a.step(rc, radial.sum((0, 1))/R_BIN, where='mid', color=INK, lw=0.8, alpha=.5, label='all')
    a.set_yscale('log'); a.set_xlim(0, R_MAX); a.set_ylim(vmin*10, None)
    a.axvline(half, color=MUTED, lw=.8, ls=':'); a.text(half+.3, vmin*30, 'sensor\nedge', fontsize=7.5, color=MUTED)
    a.set_xlabel('radius r on the sensor [mm]'); a.set_ylabel('dN/dr [photons / mm / muon]')
    a.set_title('Radial profile by origin and path class', fontsize=10); a.legend(fontsize=7.5, loc='upper right')
    a.grid(alpha=.3)
    a = ax[1, 2]; a.axis('off')
    em, cls, lf = s['emit'], s['cls'], s['lost_frac']
    lines = ['%s, %s' % (label, WLABEL[wn]), 'vertical muon on the mirror axis, per muon', '',
             'emitted in the band: gas %.3g, aerogel %.3g' % (em[0], em[1]),
             'on the SiPM: gas %.4g, aerogel %.3g' % (cls[0].sum(), cls[1].sum())]
    for oi, org in enumerate(ORIGINS):
        lines.append('  %s: ' % org+', '.join('%s %.2g' % (cl, cls[oi, ci]) for ci, cl in enumerate(CLASSES) if cls[oi, ci] > 0))
    lines += ['', 'gas ring (direct): <r> = %.3f mm, rms = %.3f mm' % (s['r_mean'], s['r_rms']),
              '  paraxial f tan θc(λ): <r> = %.3f mm, rms = %.3f mm' % rpar, '',
              'lost fraction of the emitted photons:']
    for oi, org in enumerate(ORIGINS):
        lines.append('  %s: ' % org+', '.join('%s %.2g' % (REASONS[j+1], lf[oi, j]) for j in range(len(REASONS)-1) if lf[oi, j] > 0))
    lines += ['', 'mirror f = %.1f mm, Ø %.1f mm (clear %.0f %%), reflectivity 1;' % (lay.mirror.f, MIRROR_D, 100*MIRROR_CLEAR_FRAC),
              'aerogel %g×%g×%g mm, n = %.2f, %g mm below the %.1f mm SiPM;'
              % (2*AEROGEL_HALF, 2*AEROGEL_HALF, AEROGEL_T, lay.tile.n, lay.gap, 2*SENSOR_HALF_2IN),
              'no scattering or absorption; SiPM = perfect absorber;',
              'polarised Fresnel at the tile faces. Maps in %.1f mm bins.' % MAP_BIN]
    a.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', fontsize=9, color=INK, transform=a.transAxes, family='DejaVu Sans')
    fig.suptitle('Phase-1 SiPM hit density, %s, %s' % (label, WLABEL[wn]), fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    p = save(out, 'hitmap_onaxis_%s_%s.png' % (name, wn), dpi=130, fig=fig); plt.close(fig)
    return p


def gases_figure(out, profiles, labels):
    """Total dN/dr for the three gases: full range (log) and the gas-ring zoom (linear)."""
    rc = (np.arange(len(next(iter(profiles.values()))['ftqe']))+0.5)*R_BIN
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    styles = {'xe2': '-', 'xe3': '--', 'c4f10': '-.'}
    for name, prof in profiles.items():
        for wn, lw, al in (('ftqe', 1.6, 1.0), ('ft', 1.0, 0.45)):
            y = prof[wn]/R_BIN
            lab = '%s, %s' % (labels[name], 'FT x QE' if wn == 'ftqe' else 'FT')
            ax[0].step(rc, np.where(y > 0, y, np.nan), where='mid', color=RADIATOR[name], ls=styles[name], lw=lw, alpha=al, label=lab)
            ax[1].step(rc, y, where='mid', color=RADIATOR[name], ls=styles[name], lw=lw, alpha=al, label=lab)
    ax[0].set_yscale('log'); ax[0].set_xlim(0, R_MAX); ax[0].set_ylim(1e-7, None)
    ax[0].set_title('All photons on the SiPM (gas + aerogel, all paths)', fontsize=10)
    ax[1].set_xlim(9, 14.5); ax[1].set_title('Gas ring, linear scale', fontsize=10)
    for name in profiles:
        y = profiles[name]['ftqe']/R_BIN; i = np.argmax(y)
        ax[1].annotate(labels[name], (rc[i], y[i]), xytext=(4, 4), textcoords='offset points', fontsize=9, color=INK)
    for a in ax:
        a.set_xlabel('radius r on the sensor [mm]'); a.set_ylabel('dN/dr [photons / mm / muon]'); a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle('Vertical muon on the mirror axis: radial hit profiles (bins %.2f mm)' % R_BIN, fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    p = save(out, 'hitmap_onaxis_radial_gases.png', dpi=130, fig=fig); plt.close(fig)
    return p


def table_rows(summ, labels):
    head = (['gas', 'weighting', 'emitted_gas', 'emitted_aerogel']
            + ['sipm_%s_%s' % (o, c) for o in ORIGINS for c in CLASSES]
            + ['ring_r_mean_mm', 'ring_r_rms_mm']
            + ['lost_%s_%s' % (o, r.replace(' ', '_')) for o in ORIGINS for r in REASONS[1:]])
    rows = []
    for name in summ:
        for wn in WEIGHTINGS:
            s = summ[name][wn]
            rows.append([labels[name], wn, *s['emit'], *s['cls'].ravel(), s['r_mean'], s['r_rms'], *s['lost_frac'].ravel()])
    return head, rows


def write_tables(out, summ, labels):
    head, rows = table_rows(summ, labels)
    with open(out/'hitmap_onaxis_summary.csv', 'w') as f:
        f.write(','.join(head)+'\n')
        for r in rows:
            f.write(','.join([r[0], r[1]]+['%.6g' % v for v in r[2:]])+'\n')
    md = ['Photons per muon (emitted in the band; reaching the SiPM by origin and path class)', '',
          '| case | emitted gas | emitted aerogel | SiPM gas direct | gas bottom | gas top | gas TIR | gas multiple '
          '| SiPM aerogel direct | aerogel bottom | aerogel top | aerogel TIR | aerogel multiple |',
          '|---|' + '---|'*12]
    for r in rows:
        md.append('| %s, %s | ' % (r[0], r[1].upper().replace('FTQE', 'FT×QE')) + ' | '.join('%.3g' % v for v in r[2:14]) + ' |')
    md += ['', 'Gas ring radius (direct gas photons) and lost fractions of the emitted photons', '',
           '| case | <r> [mm] | rms r [mm] | gas: mirror aperture | gas: tile sides | gas: outside sensor | gas: cap | gas: escaped '
           '| aerogel: mirror aperture | aerogel: tile sides | aerogel: outside sensor | aerogel: cap | aerogel: escaped |',
           '|---|' + '---|'*12]
    for r in rows:
        md.append('| %s, %s | %.3f | %.3f | ' % (r[0], r[1].upper().replace('FTQE', 'FT×QE'), r[14], r[15])
                  + ' | '.join('%.3g' % v for v in r[16:]) + ' |')
    (out/'hitmap_onaxis_summary.md').write_text('\n'.join(md)+'\n')
    return md


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nphot', type=float, default=4e6, help='sampled photons per origin (gas, aerogel) and per radiator gas (default 4e6)')
    p.add_argument('--chunk', type=int, default=50000, help='sampled photons per origin and chunk (default 50000)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed; every gas uses the same chunk seeds')
    p.add_argument('--nproc', type=int, default=8, help='worker processes (default 8)')
    p.add_argument('--max-reflections', type=int, default=2, help='partial Fresnel reflections kept per photon path (default 2)')
    p.add_argument('--no-unpolarised', action='store_true', help='skip the unpolarised-Fresnel comparison run')
    args = p.parse_args(); out = cli.outdir(args)
    t0 = time.time()
    lo, hi = qe_band()
    nchunk = int(np.ceil(args.nphot/args.chunk))
    labels = {name: g.label.replace('C4F10', 'C₄F₁₀') for name, g in GASES}
    layouts = [layout(g) for _, g in GASES]
    spectra = [{'gas': PhotonSpectrum(lay.gas.index, lo, hi), 'tile': PhotonSpectrum(lay.tile_index.index, lo, hi)}
               for lay in layouts]
    nb = int(round(2*SENSOR_HALF_2IN/MAP_BIN))
    _CFG.update(layouts=layouts, spectra=spectra, seeds=np.random.SeedSequence(args.seed).spawn(nchunk),
                n_gas=args.chunk, n_tile=args.chunk, max_refl=args.max_reflections, nb=nb, half=SENSOR_HALF_2IN,
                nr=int(round(R_MAX/R_BIN)))
    pols = (True,) if args.no_unpolarised else (True, False)
    tasks = [(gi, pol, ci) for pol in pols for gi in range(len(GASES)) for ci in range(nchunk)]
    print('band %.2f-%.2f eV (%.1f-%.1f nm); %d chunks x %d photons per origin; %d tasks on %d processes'
          % (lo, hi, HC/hi, HC/lo, nchunk, args.chunk, len(tasks), args.nproc), flush=True)
    acc = {}
    with mp.get_context('fork').Pool(args.nproc) as pool:
        for t, res in zip(tasks, pool.imap(run_chunk, tasks)):
            key = (t[0], t[1])
            acc[key] = reduce([acc[key], res]) if key in acc else res
    for a in acc.values():
        for k in a:
            if k != 'nbranch':
                a[k] = a[k]/nchunk                                      # chunk sums -> per muon
    t_mc = time.time()-t0
    print('Monte Carlo wall time %.1f s, %d terminal branches' % (t_mc, sum(a['nbranch'] for a in acc.values())), flush=True)

    summ, profiles, npz = {}, {}, {}
    for gi, (name, gas) in enumerate(GASES):
        a = acc[(gi, True)]
        summ[name] = summarise(a)
        profiles[name] = {wn: a['radial'][k].sum((0, 1)) for k, wn in enumerate(WEIGHTINGS)}
        sp = spectra[gi]
        print('\n== %s: yield per mm in the band: gas %.4f (FT) %.4f (FT x QE); aerogel %.4f (FT) %.4f (FT x QE)' % (
            labels[name], sp['gas'].yield_per_mm, PhotonSpectrum(gas.index, lo, hi, weight=qe_arc).yield_per_mm,
            sp['tile'].yield_per_mm, PhotonSpectrum(layouts[gi].tile_index.index, lo, hi, weight=qe_arc).yield_per_mm))
        if name == 'c4f10':
            Em = C4F10_E_EV[0]
            for wn, spw in (('FT', sp['gas']), ('FT x QE', PhotonSpectrum(gas.index, lo, hi, weight=qe_arc))):
                frac = np.interp(Em, spw.E, spw.cdf)/spw.cdf[-1]
                print('   C4F10 photons below %.3f eV (above %.0f nm, index held at the table end): %.1f%% (%s)'
                      % (Em, HC/Em, 100*frac, wn))
        for k, wn in enumerate(WEIGHTINGS):
            s = summ[name][wn]
            spw = PhotonSpectrum(gas.index, lo, hi, weight=qe_arc) if wn == 'ftqe' else sp['gas']
            rpar = paraxial_radius(spw, layouts[gi].mirror.f)
            s['r_parax'] = rpar
            print(' %-6s emitted gas %.4g aerogel %.4g | SiPM gas %.5g aerogel %.4g | ring <r> %.4f rms %.4f mm (paraxial %.4f / %.4f)'
                  % (wn, *s['emit'], s['cls'][0].sum(), s['cls'][1].sum(), s['r_mean'], s['r_rms'], *rpar))
            for oi, org in enumerate(ORIGINS):
                print('   %-8s SiPM by class: %s' % (org, '  '.join('%s %.4g' % (c, s['cls'][oi, ci]) for ci, c in enumerate(CLASSES))))
                print('   %-8s lost fraction: %s' % (org, '  '.join('%s %.3g' % (r, s['lost_frac'][oi, j]) for j, r in enumerate(REASONS[1:]))))
            npz['%s_%s_map' % (name, wn)] = (a['map'][k]/MAP_BIN**2).astype(np.float32)
            npz['%s_%s_radial' % (name, wn)] = a['radial'][k]/R_BIN
            npz['%s_%s_emit' % (name, wn)] = s['emit']; npz['%s_%s_reason' % (name, wn)] = s['reason']
        if not args.no_unpolarised:
            u = acc[(gi, False)]
            print(' polarised / unpolarised Fresnel (FT x QE, photons per muon on the SiPM):')
            for oi, org in enumerate(ORIGINS):
                for ci, c in enumerate(CLASSES):
                    pp, uu = a['cls'][0, oi, ci], u['cls'][0, oi, ci]
                    if pp > 0 or uu > 0:
                        print('   %-8s %-8s %.5g / %.5g  ratio %.4f' % (org, c, pp, uu, pp/uu if uu > 0 else np.nan))
            npz['%s_unpol_cls' % name] = u['cls']
    md = write_tables(out, summ, labels)
    print('\n'+'\n'.join(md))
    edges = np.arange(nb+1)*MAP_BIN-SENSOR_HALF_2IN
    np.savez(out/'hitmap_onaxis.npz', map_edges=edges, r_edges=np.arange(int(round(R_MAX/R_BIN))+1)*R_BIN,
             origins=np.array(ORIGINS), classes=np.array(CLASSES), reasons=np.array(REASONS), **npz)
    figs = []
    for gi, (name, gas) in enumerate(GASES):
        a = acc[(gi, True)]
        for k, wn in enumerate(WEIGHTINGS):
            figs.append(case_figure(out, name, labels[name], wn, a['map'][k], a['radial'][k], summ[name][wn],
                                    summ[name][wn]['r_parax'], layouts[gi]))
    figs.append(gases_figure(out, profiles, labels))
    print('\nfigures:', ', '.join(f.name for f in figs))
    print('total wall time %.1f s' % (time.time()-t0))


if __name__ == '__main__':
    main()
