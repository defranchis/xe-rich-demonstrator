"""Cosmic ring rates of configurations 1-5 and of the larger mirror layouts with the muon momentum spectrum (npz, json).

Every run traces one muon sample twice with the 3D tracer (cosmic.trace_batch): at beta = 1 and at the velocity of a
momentum drawn from the joint momentum-zenith spectrum (muons.MuonSpectrum, default parametrisation) at the muon's own
zenith angle. Zenith angles follow the cos^3 law of the beta = 1 toys; the per-muon weight M(cos)/(I0 cos^2) turns the
sample into the spectrum (importance weights for the other parametrisations), so both columns share the muon geometry.
Configuration 4 uses sim.proximity_fractions with the same muons. Muons enter through the top counter plane; the
telescope flag marks those crossing the bottom counter. Results per run and seed go to cosmic_rates_<group>.npz;
cosmic_rates_plots.py draws and tabulates them. Runs with the same key (gases, pressures and tile variants of one
mirror, gap or layout set) share their muons seed by seed. Groups (--groups): c1, c5, c23 (configurations 1, 5, 2 and
3), s61 (the larger mirror layouts of docs/design_options.md section 6.1), c4 (configuration 4). The tile side faces
are black (see cosmic.py).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, cosmic, muons, sim  # noqa: E402
from xerich.cherenkov import beta as beta_of  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, M_MU, MIRROR_D, MIRROR_R, MIRROR_SHORT_F, MUON_THMAX  # noqa: E402
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor  # noqa: E402
from xerich.media import N_AIR, Rayleigh, air, aerogel_index, aerogel_index_air, n_xe, xenon  # noqa: E402

PRESSURES = (1.0, 1.5, 2.0, 2.5, 3.0)            # absolute [bar]
VARIANTS = ('plain', 'pore-vac', 'pore-air')     # tile n = 1.03; + 0.97 (n_gas - 1); + 0.935 (n_gas - n_air)
LAM_TILE = 400.0                                 # wavelength of the gas index in the pore shift [nm]
SHORT_F = {'f%d' % f: (2*f, D) for f, D in MIRROR_SHORT_F.values()}   # f50: Edmund #43-543, f76: #43-553; (R, D) [mm]
SIPM_HALF = 25.4             # 2" x 2" SiPM [mm]
TOP_HALF = 25.0              # top counter: 5 x 5 cm at the SiPM plane (configurations 1-3, 5) or the tile top face (4)
BOT_HALF = {'mirror': 40.0, 'prox': 25.0}        # bottom counter: 8 x 8 cm at the mirror vertex plane; 5 x 5 cm at the SiPM plane
GAPS = (30.0, 50.0, 70.0)
LAYOUTS_61 = [(h, D) for h in (25.6, 40.0, 51.2, 64.0) for D in (76.2, 152.4, 200.0)]
_G = {}


def gas_label(P):
    return 'air' if P == 0 else 'xe%g' % P


def gas_index400(P):
    return N_AIR if P == 0 else float(n_xe(LAM_TILE, P))


def tile_n(P, variant):
    """Tile index in gas P (0: air) for a pore-shift variant."""
    ng = gas_index400(P)
    return {'plain': AEROGEL_N_AIR, 'pore-vac': float(aerogel_index(ng)), 'pore-air': float(aerogel_index_air(ng))}[variant]


def runs(scale):
    """List of run definitions (dicts)."""
    R = []

    def add(**kw):
        kw.setdefault('variant', 'plain'); kw.setdefault('rayleigh', False); kw.setdefault('origins', ('gas', 'tile'))
        kw.setdefault('top_half', SIPM_HALF); kw.setdefault('zmax', float(np.degrees(MUON_THMAX)))
        kw['n_mu'] = int(kw['n_mu']*scale)
        R.append(kw)
    for P in PRESSURES:
        add(name='c1_%s' % gas_label(P), group='c1', mirror=(MIRROR_R, MIRROR_D), sensor=SIPM_HALF, P=P, tile=False,
            origins=('gas',), zmax=20.0, n_mu=100000, key='c15')
        add(name='c5_%s_plain' % gas_label(P), group='c5', mirror=(MIRROR_R, MIRROR_D), sensor=SIPM_HALF, P=P, tile=True, n_mu=40000, key='c5')
        add(name='c5_%s_plain_ray' % gas_label(P), group='c5', mirror=(MIRROR_R, MIRROR_D), sensor=SIPM_HALF, P=P, tile=True,
            rayleigh=True, origins=('gas',), zmax=20.0, n_mu=100000, key='c15')
        add(name='c5_%s_plain_geo' % gas_label(P), group='c5', mirror=(MIRROR_R, MIRROR_D), sensor=SIPM_HALF, P=P, tile=True,
            origins=('gas',), zmax=20.0, n_mu=100000, key='c15')
        if P in (1.0, 2.0, 3.0):
            for v in VARIANTS[1:]:
                add(name='c5_%s_%s' % (gas_label(P), v), group='c5', mirror=(MIRROR_R, MIRROR_D), sensor=SIPM_HALF, P=P, tile=True,
                    variant=v, origins=('tile',), n_mu=40000, key='c5')
    for mk, mir in SHORT_F.items():
        for v in ('plain', 'pore-vac'):
            add(name='c2_%s_%s' % (mk, v), group='c23', mirror=mir, sensor=SIPM_HALF, P=0, tile=True, variant=v, origins=('tile',), n_mu=40000,
                key='c23'+mk)
        add(name='c2_%s_plain_ray' % mk, group='c23', mirror=mir, sensor=SIPM_HALF, P=0, tile=True, rayleigh=True, origins=('tile',), n_mu=20000,
            key='c23ray'+mk)
        for P in PRESSURES:
            for v in VARIANTS:
                add(name='c3_%s_%s_%s' % (mk, gas_label(P), v), group='c23', mirror=mir, sensor=SIPM_HALF, P=P, tile=True, variant=v,
                    origins=('gas', 'tile') if v == 'plain' else ('tile',), n_mu=40000, key='c23'+mk)
        add(name='c3_%s_xe3_plain_ray' % mk, group='c23', mirror=mir, sensor=SIPM_HALF, P=3.0, tile=True, rayleigh=True, n_mu=20000, key='c23ray'+mk)
    for h, D in LAYOUTS_61:
        for v in VARIANTS:
            add(name='s61_h%g_D%g_%s' % (h, D, v), group='s61', mirror=(MIRROR_R, D), sensor=h, P=3.0, tile=True, variant=v,
                origins=('gas', 'tile') if v == 'plain' else ('tile',), top_half=TOP_HALF, n_mu=30000, key='s61')
    for P in (0.0,)+PRESSURES:
        for gap in GAPS:
            for v in (VARIANTS if P > 0 else ('plain', 'pore-vac')):
                add(name='c4_%s_g%g_%s' % (gas_label(P), gap, v), group='c4', P=P, gap=gap, variant=v, sensor=SIPM_HALF, tile=True,
                    top_half=TOP_HALF, n_mu=100000, key='c4')
    return R


def layout_of(run):
    tile = Aerogel(tile_n(run['P'], run['variant'])) if run['tile'] else None
    gas = air() if run['P'] == 0 else xenon(run['P'])
    return Phase1Layout(Mirror(*run['mirror']), Sensor(run['sensor']), gas, tile, tile_scatter=Rayleigh() if run['rayleigh'] else None)


def sample_muons(rng, n, half, z0, zmax):
    """cos^3 muons through the top counter (sample_cos3), momenta from the default spectrum at their zenith."""
    u, p0, zen = muons.sample_cos3(rng, n, half, z0, np.radians(zmax))
    p = _G['spec'].sample_p(rng, -u[:, 2])
    return u, p0, p


def work_mirror(run, rng, n):
    lay = layout_of(run)
    u, p0, p = sample_muons(rng, n, run['top_half'], lay.z_sensor, run['zmax'])
    bot = p0+((0.0-lay.z_sensor)/u[:, 2])[:, None]*u
    out = dict(c=-u[:, 2], phi=np.arctan2(u[:, 1], u[:, 0]), x0=p0[:, 0], y0=p0[:, 1], p=p,
               top=(np.abs(p0[:, 0]) < TOP_HALF)&(np.abs(p0[:, 1]) < TOP_HALF),
               bot=(np.abs(bot[:, 0]) < BOT_HALF['mirror'])&(np.abs(bot[:, 1]) < BOT_HALF['mirror']))
    for tag, b in (('b1', np.ones(n)), ('br', beta_of(p, M_MU))):
        acc = {k: [] for k in ('emit_ft', 'emit_qe', 'det_ft', 'det_qe', 'det_qe_clean', 'out_ft', 'mirror_ft', 'arc', 'arc_half')}
        for i0 in range(0, n, 2000):
            sl = slice(i0, i0+2000)
            r = cosmic.trace_batch(lay, rng, p0[sl], u[sl], b[sl], {'gas': _G['ngas'], 'tile': _G['ntile']}, run['origins'])
            for k in acc:
                acc[k].append(r[k])
        r = {k: np.concatenate(v) for k, v in acc.items()}
        for org in run['origins']:
            o = cosmic.ORIGINS.index(org); fm, fs, fd = cosmic.fractions(r, org)
            out.update({'%s_%s_%s' % (tag, org, k): v for k, v in
                        (('fm', fm), ('fs', fs), ('fd', fd), ('arc', r['arc'][:, o]), ('arc_half', r['arc_half'][:, o]), ('det_qe', r['det_qe'][:, o]),
                         ('det_qe_clean', r['det_qe_clean'][:, o]), ('emit_qe', r['emit_qe'][:, o]))})
    return out


def work_prox(run, rng, n):
    aero = Aerogel(tile_n(run['P'], run['variant'])); ng = gas_index400(run['P']); sens = Sensor(run['sensor'])
    u, p0, p = sample_muons(rng, n, run['top_half'], 0.0, run['zmax'])
    zs = -aero.thickness-run['gap']
    bot = p0+((zs-0.0)/u[:, 2])[:, None]*u
    out = dict(c=-u[:, 2], phi=np.arctan2(u[:, 1], u[:, 0]), x0=p0[:, 0], y0=p0[:, 1], p=p,
               top=np.ones(n, bool), bot=(np.abs(bot[:, 0]) < BOT_HALF['prox'])&(np.abs(bot[:, 1]) < BOT_HALF['prox']))
    for tag, b in (('b1', np.ones(n)), ('br', beta_of(p, M_MU))):
        th = np.arccos(np.minimum(1, 1/(aero.n*b)))
        fa, arc = sim.proximity_fractions(aero, sens, run['gap'], ng, u, p0, th)
        out.update({'%s_tile_fd' % tag: fa, '%s_tile_arc' % tag: arc, '%s_tile_th' % tag: th})
    return out


def task(arg):
    ri, si = arg
    run = _G['runs'][ri]; rng = np.random.default_rng([_G['seed'], zlib.crc32(run['key'].encode()), si]); n = run['n_mu']//_G['nseed']   # one muon sample per key
    t0 = time.time()
    out = (work_prox if run['group'] == 'c4' else work_mirror)(run, rng, n)
    out['seed'] = np.full(n, si, np.int16)
    return ri, si, out, time.time()-t0


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nphot', type=int, default=180, help='sampled tile photons per muon (gas: 0.4x); default 180')
    p.add_argument('--nmu-scale', type=float, default=1.0, help='scale of the muon numbers per run (default 1)')
    p.add_argument('--nseed', type=int, default=5, help='independent seeds per run (default 5)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed')
    p.add_argument('--nproc', type=int, default=20, help='worker processes (default 20)')
    p.add_argument('--groups', nargs='*', default=None, help='run groups (c1 c5 c23 s61 c4: configurations 1, 5, 2-3, the larger layouts, 4; default all)')
    p.add_argument('--spectrum', choices=muons.SPECTRUM_MODELS, default=muons.SPECTRUM_DEFAULT,
                   help='parametrisation the momenta are drawn from (default %(default)s)')
    args = p.parse_args(); out = cli.outdir(args)
    allr = runs(args.nmu_scale)
    sel = [i for i, r in enumerate(allr) if args.groups is None or r['group'] in args.groups]
    _G.update(runs=allr, seed=args.seed, nseed=args.nseed, ngas=int(round(0.4*args.nphot)), ntile=args.nphot,
              spec=muons.MuonSpectrum(args.spectrum))
    tasks = sorted([(ri, si) for ri in sel for si in range(args.nseed)], key=lambda t: -allr[t[0]]['n_mu'])
    t0 = time.time(); res = {}
    with mp.get_context('fork').Pool(args.nproc) as pool:
        for ri, si, o, dt in pool.imap_unordered(task, tasks):
            res[(ri, si)] = o
            print('%-28s seed %d: %6d muons %6.1f s' % (allr[ri]['name'], si, len(o['c']), dt), flush=True)
    for g in sorted({allr[i]['group'] for i in sel}):
        arrs = {}
        for ri in [i for i in sel if allr[i]['group'] == g]:
            parts = [res[(ri, si)] for si in range(args.nseed)]
            for k in parts[0]:
                arrs['%s/%s' % (allr[ri]['name'], k)] = np.concatenate([q[k] for q in parts])
        np.savez_compressed(out/('cosmic_rates_%s.npz' % g), **arrs)
    meta = dict(args={k: (str(v) if k == 'outdir' else v) for k, v in vars(args).items()}, runs=[allr[i] for i in sel],
                top_half=TOP_HALF, sipm_half=SIPM_HALF, bot_half=BOT_HALF, wall_s=time.time()-t0, nproc=args.nproc)
    with open(out/'cosmic_rates_meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print('DONE %d tasks in %.0f s with %d processes' % (len(tasks), time.time()-t0, args.nproc))


if __name__ == '__main__':
    main()
