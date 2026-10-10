"""Mean SiPM hit density for a vertical muon on the mirror axis, phase-1 layout with the aerogel tile.

Gases Xe 2 bar, Xe 3 bar (20 degC) and C4F10 1 bar fill the vessel; aerogel n = 1.03 (constant by default),
with Rayleigh scattering. Photons are sampled from the Frank-Tamm spectrum over the band where the ARC SiPM QE
is non-zero, traced with Snell, Fresnel (polarised) and TIR at the tile faces, and counted with the FT x QE and
the pure FT weighting; the muon's own hit (Cherenkov light in the SiPM window) is a separate origin
(stdout, hitmap_onaxis_<gas>_<weighting>.png, hitmap_onaxis_radial_gases.png, hitmap_onaxis_halo.png,
hitmap_onaxis_scatter_lambda.png, hitmap_onaxis_summary.csv/.md, hitmap_onaxis.npz). An unpolarised-Fresnel
rerun with the same seeds quantifies the polarisation effect.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from xerich import cli, sim  # noqa: E402
from xerich.analysis import SCATTER_LAM_EDGES, scattered_fraction  # noqa: E402
from xerich.cherenkov import PhotonSpectrum, qe_arc, qe_band  # noqa: E402
from xerich.constants import (AEROGEL_HALF, AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_SURFACE_A, AEROGEL_T, HC,  # noqa: E402
                              MIP_DIRECT_PE, MIRROR_CLEAR_FRAC, MIRROR_D, MIRROR_R, RAYLEIGH_C_AEROGEL, SENSOR_HALF_2IN,
                              WINDOW_N, WINDOW_T)
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor, Window  # noqa: E402
from xerich.media import AEROGEL_MODELS, C4F10_E_EV, Rayleigh, aerogel_medium, c4f10, xenon  # noqa: E402
from xerich.plotting import AEROGEL, GAS, INK, MUTED, RADIATOR, SCINT_EDGE, plt, save  # noqa: E402
from xerich.tracer import CLASSES, REASONS  # noqa: E402

GAS_NAMES = ('xe2', 'xe3', 'c4f10')
WEIGHTINGS = ('ftqe', 'ft')
WLABEL = {'ftqe': 'Frank-Tamm x ARC SiPM QE', 'ft': 'Frank-Tamm (no QE)'}
ORIGINS = ('gas', 'aerogel', 'window')
ORIGIN_COLOUR = {'gas': GAS, 'aerogel': AEROGEL, 'window': SCINT_EDGE}
MAP_BIN = 0.2                      # [mm]
R_BIN = 0.05                       # [mm]
R_MAX = 36.0                       # beyond the sensor half diagonal [mm]
CLASS_STYLE = {'direct': '-', 'bottom': '--', 'top': '-.', 'TIR': ':', 'multiple': (0, (1, 1)), 'scattered': (0, (5, 1, 1, 1))}

_CFG = {}


def gases(model):
    """(name, Medium) of the three radiator gases, Xe with the given n_xe index model."""
    return tuple(zip(GAS_NAMES, (xenon(2, model=model), xenon(3, model=model), c4f10())))


def layout(gas, aerogel='constant', lam_ref=AEROGEL_LAMBDA_REF, scatter=None, window=None):
    return Phase1Layout(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), gas, Aerogel(AEROGEL_N_AIR),
                        tile_medium=aerogel_medium(aerogel, lam_ref=lam_ref), tile_scatter=scatter, window=window)


def run_chunk(task):
    """Trace one chunk; returns weight sums for (weighting, origin, ...) in photons per muon."""
    gi, pol, ci = task
    c = _CFG
    lay = c['layouts'][gi]
    rng = np.random.default_rng(c['seeds'][ci])
    nw = c['n_window'] if lay.window is not None and c['cluster'] is not None else 0
    ph, e = sim.onaxis_muon(lay, rng, c['spectra'][gi], c['n_gas'], c['n_tile'], c['max_refl'], pol, nw)
    if c['cluster'] is not None and lay.window is None:
        phw, ew = sim.window_cluster(lay, rng, c['spectra'][gi]['window'], c['n_window'], c['cluster'], pol)
        ew['i'] = ew['i']+len(ph['lam'])
        ph = {k: np.concatenate([ph[k], phw[k]]) for k in ph}; e = {k: np.concatenate([e[k], ew[k]]) for k in e}
    no = len(ORIGINS)
    w0 = ph['weight']; qe = qe_arc(HC/ph['lam']); org = ph['origin']
    wsets = np.stack([w0*qe, w0])                                     # (weighting, photon)
    emit = np.array([[np.sum(ws[org == o]) for o in range(no)] for ws in wsets])
    o = org[e['i']].astype(np.intp); we = wsets[:, e['i']]*e['w']                     # (weighting, branch)
    reason = np.zeros((2, no, len(REASONS)))
    key = o*len(REASONS)+e['reason']
    for k in range(2):
        reason[k] = np.bincount(key, we[k], minlength=no*len(REASONS)).reshape(no, -1)
    nl = len(SCATTER_LAM_EDGES)-1; il = np.clip(np.searchsorted(SCATTER_LAM_EDGES, ph['lam'][e['i']])-1, 0, nl-1)
    lkey = ((key*2+(e['nsc'] > 0))*nl+il)
    lam_end = np.stack([np.bincount(lkey, we[k], minlength=no*len(REASONS)*2*nl).reshape(no, len(REASONS), 2, nl)
                        for k in range(2)])
    det = e['reason'] == 0
    x, y = e['p'][det, 0], e['p'][det, 1]; r = np.hypot(x, y); oc = o[det]*len(CLASSES)+e['cls'][det]; wd = we[:, det]
    nr = c['nr']; ir = np.minimum((r/R_BIN).astype(int), nr-1)
    radial = np.stack([np.bincount(oc*nr+ir, wd[k], minlength=no*len(CLASSES)*nr).reshape(no, len(CLASSES), nr)
                       for k in range(2)])
    cls = radial.sum(-1)
    out = dict(emit=emit, reason=reason, cls=cls, radial=radial, nbranch=len(e['i']), lam_end=lam_end)
    g = (oc == 0)                                                      # gas origin, direct class
    out['ring'] = np.array([[wd[k, g].sum(), (wd[k, g]*r[g]).sum(), (wd[k, g]*r[g]**2).sum()] for k in range(2)])
    g = o[det] == 2                                                    # window origin: the muon's own hit
    out['spot'] = np.array([[wd[k, g].sum(), (wd[k, g]*r[g]).sum(), (wd[k, g]*r[g]**2).sum()] for k in range(2)])
    if pol:
        nb = c['nb']; H = c['half']
        ix = np.clip(((x+H)/MAP_BIN).astype(int), 0, nb-1); iy = np.clip(((y+H)/MAP_BIN).astype(int), 0, nb-1)
        idx = (oc*nb+ix)*nb+iy
        out['map'] = np.stack([np.bincount(idx, wd[k], minlength=no*len(CLASSES)*nb*nb).reshape(no, len(CLASSES), nb, nb)
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
        sw, s1, s2 = acc['spot'][k]
        s['spot_n'] = sw; s['spot_r_mean'] = s1/sw if sw > 0 else np.nan; s['spot_r_rms'] = np.sqrt(s2/sw) if sw > 0 else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
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
                      (ax[1, 0], dens[0, 1:].sum(0), 'Gas photons reflected or scattered in or at the tile (log)'),
                      (ax[1, 1], dens[1].sum(0), 'Aerogel photons, all paths (log)')):
        im = sensor_panel(a, masked(img), half, lnorm, t, cmap)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label='photons / mm² / muon')
    a = ax[0, 2]
    for oi, org in enumerate(ORIGINS):
        col = ORIGIN_COLOUR[org]
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
             'emitted in the band: gas %.3g, aerogel %.3g' % (em[0], em[1])
             + (', window %.3g' % em[2] if em[2] > 0 else ''),
             'on the SiPM: gas %.4g, aerogel %.3g' % (cls[0].sum(), cls[1].sum())
             + (', window %.3g (spot rms r %.2f mm)' % (cls[2].sum(), s['spot_r_rms']) if cls[2].sum() > 0 else '')]
    for oi, org in enumerate(ORIGINS):
        lines.append('  %s: ' % org+', '.join('%s %.2g' % (cl, cls[oi, ci]) for ci, cl in enumerate(CLASSES) if cls[oi, ci] > 0))
    lines += ['', 'gas ring (direct): <r> = %.3f mm, rms = %.3f mm' % (s['r_mean'], s['r_rms']),
              '  paraxial f tan θc(λ): <r> = %.3f mm, rms = %.3f mm' % rpar, '',
              'lost fraction of the emitted photons:']
    for oi, org in enumerate(ORIGINS[:2]):
        lines.append('  %s: ' % org+', '.join('%s %.2g' % (REASONS[j+1], lf[oi, j]) for j in range(len(REASONS)-1) if lf[oi, j] > 0))
    sc = lay.tile_scatter
    lines += ['', 'mirror f = %.1f mm, Ø %.1f mm (clear %.0f %%), reflectivity 1;' % (lay.mirror.f, MIRROR_D, 100*MIRROR_CLEAR_FRAC),
              'aerogel %g×%g×%g mm, n = %.2f, %g mm below the %.1f mm SiPM;'
              % (2*AEROGEL_HALF, 2*AEROGEL_HALF, AEROGEL_T, lay.tile.n, lay.gap, 2*SENSOR_HALF_2IN),
              ('Rayleigh in the tile, C = %g µm⁴/cm, A = %g; no absorption;' % (sc.C, sc.A) if sc is not None
               else 'no scattering or absorption;') + ' SiPM = perfect absorber;',
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


def halo_figure(out, radials, labels):
    """dN/dr (FT x QE, log scale) per gas: gas ring (direct), Rayleigh halo of gas and aerogel photons, other paths, window."""
    rc = (np.arange(next(iter(radials.values())).shape[-1])+0.5)*R_BIN
    fig, ax = plt.subplots(1, len(radials), figsize=(6*len(radials), 5.2), sharey=True)
    isc = CLASSES.index('scattered')
    for a, (name, rad) in zip(np.atleast_1d(ax), radials.items()):
        rad = rad[0]/R_BIN
        curves = (('gas, direct (ring)', rad[0, 0], GAS, '-'), ('gas, scattered (halo)', rad[0, isc], GAS, ':'),
                  ('gas, other paths', rad[0].sum(0)-rad[0, 0]-rad[0, isc], GAS, '--'),
                  ('aerogel, scattered', rad[1, isc], AEROGEL, ':'), ('aerogel, other paths', rad[1].sum(0)-rad[1, isc], AEROGEL, '--'),
                  ('window (muon spot)', rad[2].sum(0), ORIGIN_COLOUR['window'], '-'), ('all', rad.sum((0, 1)), INK, '-'))
        for lab, y, col, ls in curves:
            if y.sum() > 0:
                a.step(rc, np.where(y > 0, y, np.nan), where='mid', color=col, ls=ls, lw=1.0 if lab == 'all' else 1.4,
                       alpha=0.5 if lab == 'all' else 1, label=lab)
        a.set_yscale('log'); a.set_xlim(0, R_MAX); a.set_ylim(1e-6, None); a.grid(alpha=.3)
        a.axvline(SENSOR_HALF_2IN, color=MUTED, lw=.8, ls=':')
        a.set_title('%s: halo %.3g, ring %.4g photons / muon' % (labels[name], rad[0, isc].sum()*R_BIN, rad[0, 0].sum()*R_BIN), fontsize=10)
        a.set_xlabel('radius r on the sensor [mm]')
    np.atleast_1d(ax)[0].set_ylabel('dN/dr [photons / mm / muon], FT x QE'); np.atleast_1d(ax)[0].legend(fontsize=8)
    fig.suptitle('Rayleigh halo vs gas ring, vertical muon on the mirror axis (bins %.2f mm)' % R_BIN, fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    p = save(out, 'hitmap_onaxis_halo.png', dpi=130, fig=fig); plt.close(fig)
    return p


def scatter_figure(out, lam_ends, labels, scatter):
    """Scattered fraction vs wavelength for gas photons crossing the tile and for aerogel photons, with the single-pass expectation."""
    lc = 0.5*(SCATTER_LAM_EDGES[1:]+SCATTER_LAM_EDGES[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {'xe2': '-', 'xe3': '--', 'c4f10': '-.'}
    for name, le in lam_ends.items():
        for a, oi in ((ax[0], 0), (ax[1], 1)):
            a.step(lc, scattered_fraction(le[1], oi), where='mid', color=RADIATOR[name], ls=styles[name], lw=1.4, label=labels[name])
    if scatter is not None:
        L = scatter.length(lc)
        ax[0].plot(lc, 1-np.exp(-AEROGEL_T/L), color=INK, lw=0.9, alpha=.6, label='1 − exp(−t/Λ), t = %g mm' % AEROGEL_T)
        ax[1].plot(lc, 1-L/AEROGEL_T*(1-np.exp(-AEROGEL_T/L)), color=INK, lw=0.9, alpha=.6,
                   label='uniform emission depth, normal exit')
    ax[0].set_title('Gas photons that reach the tile', fontsize=10); ax[1].set_title('Aerogel photons', fontsize=10)
    for a in ax:
        a.set_xlabel('λ [nm]'); a.set_ylabel('fraction with ≥ 1 Rayleigh scatter'); a.grid(alpha=.3); a.legend(fontsize=8)
        a.set_xlim(SCATTER_LAM_EDGES[0], SCATTER_LAM_EDGES[-1]); a.set_ylim(0, None)
    fig.suptitle('Rayleigh scattering in the %g mm aerogel tile vs wavelength%s' % (
        AEROGEL_T, '' if scatter is None else ' (Λ = λ⁴/C, C = %g µm⁴/cm)' % scatter.C), fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    p = save(out, 'hitmap_onaxis_scatter_lambda.png', dpi=130, fig=fig); plt.close(fig)
    return p


def table_rows(summ, labels):
    head = (['gas', 'weighting', 'emitted_gas', 'emitted_aerogel']
            + ['sipm_%s_%s' % (o, c) for o in ORIGINS for c in CLASSES]
            + ['ring_r_mean_mm', 'ring_r_rms_mm']
            + ['lost_%s_%s' % (o, r.replace(' ', '_')) for o in ORIGINS[:2] for r in REASONS[1:]]
            + ['emitted_window', 'window_spot_r_mean_mm', 'window_spot_r_rms_mm'])
    rows = []
    for name in summ:
        for wn in WEIGHTINGS:
            s = summ[name][wn]
            rows.append([labels[name], wn, *s['emit'][:2], *s['cls'].ravel(), s['r_mean'], s['r_rms'],
                         *s['lost_frac'][:2].ravel(), s['emit'][2], s['spot_r_mean'], s['spot_r_rms']])
    return head, rows


def write_tables(out, summ, labels):
    head, rows = table_rows(summ, labels)
    with open(out/'hitmap_onaxis_summary.csv', 'w') as f:
        f.write(','.join(head)+'\n')
        for r in rows:
            f.write(','.join([r[0], r[1]]+['%.6g' % v for v in r[2:]])+'\n')
    case = lambda r: '| %s, %s | ' % (r[0], r[1].upper().replace('FTQE', 'FT×QE'))
    col = {h: i for i, h in enumerate(head)}
    cnt = ['emitted_gas', 'emitted_aerogel']+['sipm_%s_%s' % (o, c) for o in ORIGINS[:2] for c in CLASSES]
    md = ['Photons per muon (emitted in the band; reaching the SiPM by origin and path class)', '',
          '| case | emitted gas | emitted aerogel | ' + ' | '.join('SiPM %s %s' % (o, c) for o in ORIGINS[:2] for c in CLASSES) + ' |',
          '|---|' + '---|'*len(cnt)]
    for r in rows:
        md.append(case(r) + ' | '.join('%.3g' % r[col[h]] for h in cnt) + ' |')
    lost = ['lost_%s_%s' % (o, x.replace(' ', '_')) for o in ORIGINS[:2] for x in REASONS[1:]]
    md += ['', 'Gas ring radius (direct gas photons) and lost fractions of the emitted photons', '',
           '| case | <r> [mm] | rms r [mm] | ' + ' | '.join('%s: %s' % (o, x) for o in ORIGINS[:2] for x in REASONS[1:]) + ' |',
           '|---|' + '---|'*(len(lost)+2)]
    for r in rows:
        md.append(case(r) + '%.3f | %.3f | ' % (r[col['ring_r_mean_mm']], r[col['ring_r_rms_mm']])
                  + ' | '.join('%.3g' % r[col[h]] for h in lost) + ' |')
    if any(r[col['emitted_window']] > 0 for r in rows):
        md += ['', "The muon's own hit: Cherenkov photons in the SiPM window per muon (plus %g fired cell from the muon "
               'crossing the silicon)' % MIP_DIRECT_PE, '',
               '| case | emitted in the window | on the SiPM | spot <r> [mm] | spot rms r [mm] |', '|---|---|---|---|---|']
        for r in rows:
            md.append(case(r) + '%.3g | %.3g | %.3f | %.3f |' % (
                r[col['emitted_window']], sum(r[col['sipm_window_%s' % c]] for c in CLASSES),
                r[col['window_spot_r_mean_mm']], r[col['window_spot_r_rms_mm']]))
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
    cli.xe_index(p)
    p.add_argument('--rayleigh-c', type=float, default=RAYLEIGH_C_AEROGEL,
                   help='aerogel Rayleigh clarity C [um^4/cm], mean free path lambda^4/C; 0 switches scattering off (default %(default)g)')
    p.add_argument('--surface-a', type=float, default=AEROGEL_SURFACE_A,
                   help='aerogel surface factor A, sqrt(A) per face crossing, needs --rayleigh-c > 0 (default %(default)g)')
    p.add_argument('--aerogel-index', choices=AEROGEL_MODELS, default='constant',
                   help='aerogel n(lambda): constant, single-pole or k4geo, equal to 1.03 at --aerogel-lambda-ref (default %(default)s)')
    p.add_argument('--aerogel-lambda-ref', type=float, default=AEROGEL_LAMBDA_REF,
                   help='wavelength where the dispersive aerogel index equals 1.03 [nm] (default %(default)g)')
    p.add_argument('--muon-cluster', action=argparse.BooleanOptionalAction, default=True,
                   help="the muon's own hit: Cherenkov light of the muon in the SiPM window, origin 'window' (default on)")
    p.add_argument('--window-n', type=float, default=WINDOW_N, help='SiPM window refractive index (default %(default)g)')
    p.add_argument('--window-t', type=float, default=WINDOW_T, help='SiPM window thickness [mm] (default %(default)g)')
    p.add_argument('--window-external', action='store_true',
                   help='put the window in the optics of all photons (Fresnel at its bottom face); default: muon light only')
    args = p.parse_args(); out = cli.outdir(args)
    t0 = time.time()
    lo, hi = qe_band()
    nchunk = int(np.ceil(args.nphot/args.chunk))
    GASES = gases(args.xe_index)
    scatter = Rayleigh(args.rayleigh_c, args.surface_a) if args.rayleigh_c > 0 else None
    window = Window(args.window_n, args.window_t)
    cluster = window if args.muon_cluster else None
    labels = {name: g.label.replace('C4F10', 'C₄F₁₀') for name, g in GASES}
    layouts = [layout(g, args.aerogel_index, args.aerogel_lambda_ref, scatter, window if args.window_external else None)
               for _, g in GASES]
    spectra = [{'gas': PhotonSpectrum(lay.gas.index, lo, hi), 'tile': PhotonSpectrum(lay.tile_index.index, lo, hi),
                'window': PhotonSpectrum(window.medium.index, lo, hi)} for lay in layouts]
    nb = int(round(2*SENSOR_HALF_2IN/MAP_BIN))
    _CFG.update(layouts=layouts, spectra=spectra, seeds=np.random.SeedSequence(args.seed).spawn(nchunk),
                n_gas=args.chunk, n_tile=args.chunk, n_window=args.chunk, cluster=cluster, max_refl=args.max_reflections,
                nb=nb, half=SENSOR_HALF_2IN, nr=int(round(R_MAX/R_BIN)))
    pols = (True,) if args.no_unpolarised else (True, False)
    tasks = [(gi, pol, ci) for pol in pols for gi in range(len(GASES)) for ci in range(nchunk)]
    print('band %.2f-%.2f eV (%.1f-%.1f nm); %d chunks x %d photons per origin; %d tasks on %d processes'
          % (lo, hi, HC/hi, HC/lo, nchunk, args.chunk, len(tasks), args.nproc), flush=True)
    print('Xe index %s; aerogel %s; Rayleigh %s; muon cluster %s; window in the optics of all photons: %s'
          % (args.xe_index, layouts[0].tile_index.label, 'off' if scatter is None else 'C = %g um^4/cm, A = %g' % (scatter.C, scatter.A),
             'off' if cluster is None else 'n = %g, t = %g mm' % (window.n, window.thickness), 'yes' if args.window_external else 'no'),
          flush=True)
    acc, chunks = {}, {}
    with mp.get_context('fork').Pool(args.nproc) as pool:
        for t, res in zip(tasks, pool.imap(run_chunk, tasks)):
            key = (t[0], t[1])
            chunks.setdefault(key, []).append((res['cls'], res['ring'], res['spot']))
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
                  % (wn, *s['emit'][:2], s['cls'][0].sum(), s['cls'][1].sum(), s['r_mean'], s['r_rms'], *rpar))
            for oi, org in enumerate(ORIGINS[:2]):
                print('   %-8s SiPM by class: %s' % (org, '  '.join('%s %.4g' % (c, s['cls'][oi, ci]) for ci, c in enumerate(CLASSES))))
                print('   %-8s lost fraction: %s' % (org, '  '.join('%s %.3g' % (r, s['lost_frac'][oi, j]) for j, r in enumerate(REASONS[1:]))))
            npz['%s_%s_map' % (name, wn)] = (a['map'][k]/MAP_BIN**2).astype(np.float32)
            npz['%s_%s_radial' % (name, wn)] = a['radial'][k]/R_BIN
            npz['%s_%s_emit' % (name, wn)] = s['emit']; npz['%s_%s_reason' % (name, wn)] = s['reason']
            npz['%s_%s_lam_end' % (name, wn)] = a['lam_end'][k]
        if cluster is not None:
            for k, wn in enumerate(WEIGHTINGS):
                s = summ[name][wn]
                print(" %-6s muon's own hit: window photons on the SiPM %.4g (emitted %.4g), spot <r> %.4f rms %.4f mm; "
                      'plus %g fired cell from the silicon crossing' % (wn, s['spot_n'], s['emit'][2], s['spot_r_mean'],
                                                                         s['spot_r_rms'], MIP_DIRECT_PE))
        if scatter is not None:
            for oi in (0, 1):
                le = a['lam_end'][:, oi]; scat = le[:, :, 1].sum((1, 2))
                print(' %-8s Rayleigh: scattered fraction of the emitted photons %.4f (FT x QE) %.4f (FT); on the SiPM '
                      'after scattering %.4g (FT x QE) %.4g (FT)' % (ORIGINS[oi], *(scat/a['emit'][:, oi]),
                                                                  *a['cls'][:, oi, CLASSES.index('scattered')]))
        for key, nm in ((0, 'cls'), (1, 'ring'), (2, 'spot')):
            npz['%s_chunk_%s' % (name, nm)] = np.array([c[key] for c in chunks[(gi, True)]])
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
    figs.append(halo_figure(out, {name: acc[(gi, True)]['radial'] for gi, (name, _) in enumerate(GASES)}, labels))
    figs.append(scatter_figure(out, {name: acc[(gi, True)]['lam_end'] for gi, (name, _) in enumerate(GASES)}, labels, scatter))
    print('\nfigures:', ', '.join(f.name for f in figs))
    print('total wall time %.1f s' % (time.time()-t0))


if __name__ == '__main__':
    main()
