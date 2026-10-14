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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli, cosmic, muons, provenance, sim  # noqa: E402
from xerich.analysis import pad_hits  # noqa: E402
from xerich.cherenkov import response  # noqa: E402
from xerich.cherenkov import beta as beta_of  # noqa: E402
from xerich.constants import AEROGEL_N_AIR, M_MU, MIRROR_D, MIRROR_R, MIRROR_SHORT_F, MUON_THMAX, P_REF  # noqa: E402
from xerich.layout import Aerogel, Mirror, Phase1Layout, Sensor  # noqa: E402
from xerich.media import N_AIR, Rayleigh, aerogel_index, aerogel_index_air, aerogel_medium, air, n_xe, xenon  # noqa: E402

PRESSURES = (1.0, 1.5, 2.0, 2.5, 3.0)            # absolute [bar]
VARIANTS = ('plain', 'pore-vac', 'pore-air')     # tile n = 1.03; + porosity_vac (n_gas - 1) (--porosity-vac); + 0.935 (n_gas - n_air)
LAM_TILE = 400.0                                 # wavelength of the gas index in the pore shift [nm]
SHORT_F = {'f%d' % f: (2*f, D) for f, D in MIRROR_SHORT_F.values()}   # f50: Edmund #43-543, f76: #43-553; (R, D) [mm]
SIPM_HALF = 25.4             # 2" x 2" SiPM [mm]
TOP_HALF = 25.0              # top counter: 5 x 5 cm at the SiPM plane (configurations 1-3, 5) or the tile top face (4)
BOT_HALF = {'mirror': 40.0, 'prox': 25.0}        # bottom counter: 8 x 8 cm at the mirror vertex plane; 5 x 5 cm at the SiPM plane
GAPS = (30.0, 50.0, 70.0)
NPHOT = cli.NPHOT_DEFAULT     # default sampled tile photons per muon (--nphot)
REFL_STREAM = 1              # last seed word of the board-reflection generator of a task
CHUNK = 2000                 # muons traced together in work_mirror at most (the random streams depend on the chunk)
CHUNK_PHOTONS = 3000000      # photons traced together at most: fewer muons per chunk at large --nphot (memory)
LAYOUTS_61 = [(h, D) for h in (25.6, 40.0, 51.2, 64.0) for D in (76.2, 152.4, 200.0)]
_G = {}


def gas_label(P):
    return 'air' if P == 0 else 'xe%g' % P


def gas_index400(P):
    return N_AIR if P == 0 else float(n_xe(LAM_TILE, P))


def tile_n(P, variant):
    """Tile index in gas P (0: air) for a pore-shift variant, Xe in the pores at the pore-gas temperature."""
    ng = N_AIR if P == 0 else float(n_xe(LAM_TILE, P, _G['inputs']['pore_t']))
    return {'plain': AEROGEL_N_AIR, 'pore-vac': float(aerogel_index(ng, porosity=_G['inputs']['porosity_vac'])),
            'pore-air': float(aerogel_index_air(ng))}[variant]


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
        add(name='c3_%s_%s_plain_ray' % (mk, gas_label(P_REF)), group='c23', mirror=mir, sensor=SIPM_HALF, P=P_REF, tile=True, rayleigh=True,
            n_mu=20000, key='c23ray'+mk)
    for h, D in LAYOUTS_61:
        for v in VARIANTS:
            add(name='s61_h%g_D%g_%s' % (h, D, v), group='s61', mirror=(MIRROR_R, D), sensor=h, P=P_REF, tile=True, variant=v,
                origins=('gas', 'tile') if v == 'plain' else ('tile',), top_half=TOP_HALF, n_mu=30000, key='s61')
    for P in (0.0,)+PRESSURES:
        for gap in GAPS:
            for v in (VARIANTS if P > 0 else ('plain', 'pore-vac')):
                add(name='c4_%s_g%g_%s' % (gas_label(P), gap, v), group='c4', P=P, gap=gap, variant=v, sensor=SIPM_HALF, tile=True,
                    top_half=TOP_HALF, n_mu=100000, key='c4')
    return R


def layout_of(run):
    tile = Aerogel(tile_n(run['P'], run['variant'])) if run['tile'] else None
    inp = _G['inputs']; disp = tile is not None and inp['aerogel_index'] != 'constant'
    tile_medium = aerogel_medium(inp['aerogel_index'], tile.n, inp['aerogel_lambda_ref']) if disp else None   # dispersion through n(lambda_ref)
    gas = air() if run['P'] == 0 else xenon(run['P'])
    model = inp['sensor_model'] if run['sensor'] == SIPM_HALF else 'square'   # the board replaces the 2" SiPM only
    return Phase1Layout(Mirror(*run['mirror'], reflectivity=inp['mirror_r']), Sensor(run['sensor'], model=model), gas, tile,
                        tile_medium=tile_medium, tile_scatter=Rayleigh() if run['rayleigh'] else None)


def sample_muons(rng, n, half, z0, zmax):
    """cos^3 muons through the top counter (sample_cos3), momenta from the default spectrum at their zenith."""
    u, p0, zen = muons.sample_cos3(rng, n, half, z0, np.radians(zmax))
    p = _G['spec'].sample_p(rng, -u[:, 2])
    return u, p0, p


def board_reflection_used(run):
    """Whether the board reflection of a run enters an output (the board table): configuration 1 and the configuration-5
    gas ring, whose photons come from the _ray runs; the other runs skip it (cost)."""
    return run['group'] == 'c1' or (run['group'] == 'c5' and run['rayleigh'])


def chunk_of(run):
    """Muons traced together in work_mirror: CHUNK, fewer when they would carry more than CHUNK_PHOTONS photons."""
    per_muon = sum({'gas': _G['ngas'], 'tile': _G['ntile']}[o] for o in run['origins'])
    return min(CHUNK, max(1, CHUNK_PHOTONS//per_muon))


def work_mirror(run, rng, n, refl_rng=None):
    lay = layout_of(run); chunk = chunk_of(run)
    u, p0, p = sample_muons(rng, n, run['top_half'], lay.z_sensor, run['zmax'])
    bot = p0+((0.0-lay.z_sensor)/u[:, 2])[:, None]*u
    out = dict(c=-u[:, 2], phi=np.arctan2(u[:, 1], u[:, 0]), x0=p0[:, 0], y0=p0[:, 1], p=p,
               top=(np.abs(p0[:, 0]) < TOP_HALF)&(np.abs(p0[:, 1]) < TOP_HALF),
               bot=(np.abs(bot[:, 0]) < BOT_HALF['mirror'])&(np.abs(bot[:, 1]) < BOT_HALF['mirror']))
    for tag, b in (('b1', np.ones(n)), ('br', beta_of(p, M_MU))):
        board = lay.sensor.board is not None
        refl = (_G['inputs']['board_r_spec'], _G['inputs']['board_r_diff']) if board_reflection_used(run) else (0.0, 0.0)
        rkeys = cosmic.REFL_KEYS if board and max(refl) > 0 else ()
        acc = {k: [] for k in ('emit_ft', 'emit_qe', 'det_ft', 'det_qe', 'det_qe_clean', 'out_ft', 'mirror_ft', 'arc', 'arc_half')
               + ((*cosmic.BOARD_KEYS, 'pads_pge') if board else ()) + rkeys}
        for i0 in range(0, n, chunk):
            sl = slice(i0, i0+chunk)
            r = cosmic.trace_batch(lay, rng, p0[sl], u[sl], b[sl], {'gas': _G['ngas'], 'tile': _G['ntile']}, run['origins'],
                                   response=response(_G['inputs']['response']), refl=refl, refl_rng=refl_rng)
            for k in acc:
                acc[k].append(r[k])
        r = {k: np.concatenate(v) for k, v in acc.items()}
        for org in run['origins']:
            o = cosmic.ORIGINS.index(org); fm, fs, fd = cosmic.fractions(r, org)
            out.update({'%s_%s_%s' % (tag, org, k): v for k, v in
                        (('fm', fm), ('fs', fs), ('fd', fd), ('arc', r['arc'][:, o]), ('arc_half', r['arc_half'][:, o]), ('det_qe', r['det_qe'][:, o]),
                         ('det_qe_clean', r['det_qe_clean'][:, o]), ('emit_qe', r['emit_qe'][:, o]))})
            if board:
                out.update({'%s_%s_%s' % (tag, org, k): r[k][:, o] for k in (*cosmic.BOARD_KEYS, 'pads_pge', *rkeys)})
    return out


def prox_photons(na, b, c):
    """FT x response photons per muon leaving the tile bottom of configuration 4 (velocities b, cos zenith c): yield over
    the tile path, Rayleigh transmission of a vertical muon (as cosmic_rates_plots.py)."""
    inp = _G['inputs']; qe = response(inp['response'])
    return sim.proximity_photons(na, b, c, qe, inp['aerogel_index'], inp['aerogel_lambda_ref'])*sim.proximity_rayleigh(AEROGEL_N_AIR, qe)


def ring_points(board, pts, acc, bot):
    """Pad index of the sensor-plane points pts (N, ns, nphi, 3) on the board, the mask of those on the sensor (acc) and on
    an active pad, and the ring mask: the same without the pad of the muon's crossing point bot (N, 3)."""
    pad = board.pad_index(pts[..., 0], pts[..., 1]); m = acc&(pad >= 0)
    return pad, m, m&(pad != board.pad_index(bot[:, 0], bot[:, 1])[:, None, None])


def work_prox(run, rng, n):
    aero = Aerogel(tile_n(run['P'], run['variant'])); ng = gas_index400(run['P']); sens = Sensor(run['sensor'])
    board = Sensor(run['sensor'], model=_G['inputs']['sensor_model']).board if run['sensor'] == SIPM_HALF else None
    if board is not None:
        sens = Sensor(board.half)                    # containment on the board footprint
    u, p0, p = sample_muons(rng, n, run['top_half'], 0.0, run['zmax'])
    zs = -aero.thickness-run['gap']
    bot = p0+((zs-0.0)/u[:, 2])[:, None]*u
    out = dict(c=-u[:, 2], phi=np.arctan2(u[:, 1], u[:, 0]), x0=p0[:, 0], y0=p0[:, 1], p=p,
               top=np.ones(n, bool), bot=(np.abs(bot[:, 0]) < BOT_HALF['prox'])&(np.abs(bot[:, 1]) < BOT_HALF['prox']))
    inp = _G['inputs']
    if inp['prox_rayleigh'] == 'per-photon':
        return work_prox_per_photon(run, aero, ng, sens, board, u, p0, p, bot, out)
    for tag, b in (('b1', np.ones(n)), ('br', beta_of(p, M_MU))):
        th = np.arccos(np.minimum(1, 1/(aero.n*b)))
        if board is None:
            fa, arc = sim.proximity_fractions(aero, sens, run['gap'], ng, u, p0, th)
        else:
            # binary pad hits: each of the ns x nphi photon directions stands for an equal share of the photons of the muon;
            # ring hits, without the pad the muon crosses on the sensor plane
            fa, arc, pts, acc = sim.proximity_fractions(aero, sens, run['gap'], ng, u, p0, th, points=True)
            pad, m, ring = ring_points(board, pts, acc, bot)
            mu = prox_photons(aero.n, b, -u[:, 2])/acc[0].size
            muon = np.broadcast_to(np.arange(n)[:, None, None], m.shape)
            npad, pge = pad_hits(muon[ring], pad[ring], np.broadcast_to(mu[:, None, None], m.shape)[ring], n)
            out.update({'%s_tile_act' % tag: m.sum((1, 2))/np.maximum(acc.sum((1, 2)), 1),
                        '%s_tile_act_ring' % tag: ring.sum((1, 2))/np.maximum(acc.sum((1, 2)), 1), '%s_tile_pads' % tag: npad,
                        '%s_tile_pads_pge' % tag: pge})
        out.update({'%s_tile_fd' % tag: fa, '%s_tile_arc' % tag: arc, '%s_tile_th' % tag: th})
    return out


def work_prox_per_photon(run, aero, ng, sens, board, u, p0, p, bot, out):
    """work_prox with the Rayleigh transmission of each photon direction along its path in the tile to the exit face
    (sim.rayleigh_transmission over the FT x response spectrum of the run's tile index): tile_tr is the mean transmission
    of the photons on the sensor; with the board, act and act_ring are transmission-weighted and the pad hits use the
    transmitted photons of each direction."""
    inp = _G['inputs']; qe = response(inp['response']); n = len(u)
    idx = None if inp['aerogel_index'] == 'constant' else aerogel_medium(inp['aerogel_index'], aero.n, inp['aerogel_lambda_ref']).index
    trans = sim.rayleigh_transmission_table(qe, idx)
    for tag, b in (('b1', np.ones(n)), ('br', beta_of(p, M_MU))):
        th = np.arccos(np.minimum(1, 1/(aero.n*b)))
        fa, arc, pts, acc, path = sim.proximity_fractions(aero, sens, run['gap'], ng, u, p0, th, points=True, paths=True)
        t = np.where(acc, trans(path), 0.0); ta = np.maximum(t.sum((1, 2)), 1e-300)
        out.update({'%s_tile_fd' % tag: fa, '%s_tile_arc' % tag: arc, '%s_tile_th' % tag: th,
                    '%s_tile_tr' % tag: np.where(acc.any((1, 2)), t.sum((1, 2))/np.maximum(acc.sum((1, 2)), 1), 0.0)})
        if board is not None:
            pad, m, ring = ring_points(board, pts, acc, bot)
            mu = (sim.proximity_photons(aero.n, b, -u[:, 2], qe, inp['aerogel_index'], inp['aerogel_lambda_ref'])/acc[0].size)[:, None, None]*t
            muon = np.broadcast_to(np.arange(n)[:, None, None], m.shape)
            npad, pge = pad_hits(muon[ring], pad[ring], mu[ring], n)
            out.update({'%s_tile_act' % tag: (t*m).sum((1, 2))/ta, '%s_tile_act_ring' % tag: (t*ring).sum((1, 2))/ta,
                        '%s_tile_pads' % tag: npad, '%s_tile_pads_pge' % tag: pge})
    return out


def task(arg):
    ri, si = arg
    run = _G['runs'][ri]; rng = np.random.default_rng([_G['seed'], zlib.crc32(run['key'].encode()), si]); n = run['n_mu']//_G['nseed']   # one muon sample per key
    t0 = time.time()
    if run['group'] == 'c4':
        out = work_prox(run, rng, n)
    else:   # the board reflection draws from its own stream, so the other outputs do not depend on it
        out = work_mirror(run, rng, n, np.random.default_rng([_G['seed'], zlib.crc32(run['key'].encode()), si, REFL_STREAM]))
    out['seed'] = np.full(n, si, np.int16)
    return ri, si, out, time.time()-t0


def nphot_of(args):
    """Sampled tile photons per muon: --nphot if given, else that of --preset (cli.PRESET_NPHOT), else NPHOT."""
    return args.nphot if args.nphot is not None else cli.PRESET_NPHOT.get(args.preset, NPHOT)


def main():
    p = cli.parser(__doc__)
    p.add_argument('--nphot', type=int, default=None,
                   help='sampled tile photons per muon (gas: 0.4x); default %d, with --preset %s. Muons are traced in chunks of '
                        'at most %d muons and %d photons, which bounds the memory of a worker' % (NPHOT, ', '.join(
                            '%s %d' % kv for kv in cli.PRESET_NPHOT.items()), CHUNK, CHUNK_PHOTONS))
    p.add_argument('--nmu-scale', type=float, default=1.0, help='scale of the muon numbers per run (default 1)')
    p.add_argument('--nseed', type=int, default=5, help='independent seeds per run (default 5)')
    p.add_argument('--seed', type=int, default=20261010, help='master seed')
    p.add_argument('--nproc', type=int, default=20, help='worker processes (default 20)')
    p.add_argument('--groups', nargs='*', default=None, help='run groups (c1 c5 c23 s61 c4: configurations 1, 5, 2-3, the larger layouts, 4; default all)')
    p.add_argument('--spectrum', choices=muons.SPECTRUM_MODELS, default=muons.SPECTRUM_DEFAULT,
                   help='parametrisation the momenta are drawn from (default %(default)s)')
    cli.detector_inputs(p)
    args = p.parse_args(); out = cli.outdir(args)
    args.nphot = nphot_of(args)
    allr = runs(args.nmu_scale)
    sel = [i for i, r in enumerate(allr) if args.groups is None or r['group'] in args.groups]
    _G.update(runs=allr, seed=args.seed, nseed=args.nseed, ngas=int(round(0.4*args.nphot)), ntile=args.nphot,
              spec=muons.MuonSpectrum(args.spectrum), inputs=cli.inputs(args))
    tasks = sorted([(ri, si) for ri in sel for si in range(args.nseed)], key=lambda t: -allr[t[0]]['n_mu'])
    t0 = time.time(); res = {}
    # a worker that dies (e.g. out of memory) raises BrokenProcessPool here instead of hanging
    with ProcessPoolExecutor(args.nproc, mp_context=mp.get_context('fork')) as ex:
        for fu in as_completed([ex.submit(task, t) for t in tasks]):
            ri, si, o, dt = fu.result()
            res[(ri, si)] = o
            print('%-28s seed %d: %6d muons %6.1f s' % (allr[ri]['name'], si, len(o['c']), dt), flush=True)
    for g in sorted({allr[i]['group'] for i in sel}):
        arrs = {}
        for ri in [i for i in sel if allr[i]['group'] == g]:
            parts = [res[(ri, si)] for si in range(args.nseed)]
            for k in parts[0]:
                arrs['%s/%s' % (allr[ri]['name'], k)] = np.concatenate([q[k] for q in parts])
        np.savez_compressed(out/('cosmic_rates_%s.npz' % g), **arrs, **provenance.npz(pressures=PRESSURES, gaps=GAPS))
    meta = dict(args={k: (str(v) if k == 'outdir' else v) for k, v in vars(args).items()}, inputs=_G['inputs'], runs=[allr[i] for i in sel],
                top_half=TOP_HALF, sipm_half=SIPM_HALF, bot_half=BOT_HALF, wall_s=time.time()-t0, nproc=args.nproc)
    with open(out/'cosmic_rates_meta.json', 'w') as f:
        json.dump(provenance.with_record(meta, pressures=PRESSURES, gaps=GAPS), f, indent=1)
    print('DONE %d tasks in %.0f s with %d processes' % (len(tasks), time.time()-t0, args.nproc))


if __name__ == '__main__':
    main()
