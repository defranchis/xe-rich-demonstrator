"""Command-line helpers shared by the study drivers."""
import argparse
import os
from pathlib import Path

from . import provenance
from .cherenkov import RESPONSE_DEFAULT, RESPONSES, theta_xe
from .cosmic import REFL_GENS
from .constants import (AEROGEL_LAMBDA_REF, AEROGEL_N_AIR, AEROGEL_POLE_NM, AEROGEL_POROSITY, AEROGEL_POROSITY_AIRREF, BOARD_NPAD,
                        BOARD_PAD, BOARD_R_DIFF, BOARD_R_SPEC, GAS_YIELD_REF_LEN, P_REF, PIXEL_PITCH, T_PORE, YIELD_BAND_NM)
from .layout import MIRROR_REFLECTIVITIES, SENSOR_MODELS, mirror_reflectivity
from .media import AEROGEL_MODELS, XE_INDEX_DEFAULT, XE_INDEX_MODELS

# Detector inputs of the validated baseline: S13361 PDE, nominal mirror reflectivity, board pixel map, single-pole aerogel,
# porosity of the air-referenced pore shift, pore gas at T_PORE, board reflectance; the defaults of detector_inputs()
BASELINE = dict(response='s13361', mirror_r='nominal', sensor_model='board', pore_t=T_PORE, aerogel_index='single-pole',
                aerogel_lambda_ref=AEROGEL_LAMBDA_REF, porosity_vac=AEROGEL_POROSITY_AIRREF, board_r_spec=BOARD_R_SPEC,
                board_r_diff=BOARD_R_DIFF, prox_rayleigh='per-photon')
INPUT_DEFAULTS = dict(BASELINE)
# Named sets of detector inputs (--preset): 'baseline-board' is the default (kept for existing commands); 'published' is the
# model of the tables published before the baseline: ARC QE, continuous 2" square, reflectivity 1, constant tile index,
# porosity 0.97, no board reflection
PRESETS = {'baseline-board': BASELINE,
           'published': dict(response=RESPONSE_DEFAULT, mirror_r='1', sensor_model='square', aerogel_index='constant',
                             porosity_vac=AEROGEL_POROSITY, board_r_spec=0.0, board_r_diff=0.0, prox_rayleigh='vertical')}
PROX_RAYLEIGH = ('per-photon', 'vertical')   # configuration-4 Rayleigh transmission: along each photon path / of a vertical muon
NPHOT_DEFAULT = 7200         # sampled tile photons per muon of cosmic_rates.py --nphot (binary pad hits need many, cosmic.trace_batch)
PRESET_NPHOT = {'baseline-board': NPHOT_DEFAULT, 'published': 180}   # --nphot implied by a preset


def preset(name):
    """Detector inputs of a named preset: INPUT_DEFAULTS updated with PRESETS[name]."""
    if name not in PRESETS:
        raise ValueError('unknown preset %r (choose from %s)' % (name, ', '.join(PRESETS)))
    return {**INPUT_DEFAULTS, **PRESETS[name]}


def default_outdir():
    """$XERICH_OUT if set, else xerich-output/ next to the repository checkout."""
    env = os.environ.get('XERICH_OUT')
    return Path(env) if env else Path(__file__).resolve().parents[3]/'xerich-output'


def parser(doc):
    """Argument parser with the common --outdir option."""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument('--outdir', type=Path, default=None,
                   help='output directory (default: $XERICH_OUT, else xerich-output/ next to the repository)')
    return p


def xe_index(p):
    """Add the --xe-index option (Xe refractive-index model of media.n_xe) to the parser p."""
    p.add_argument('--xe-index', choices=XE_INDEX_MODELS, default=XE_INDEX_DEFAULT,
                   help='Xe refractive-index model: borzsonyi (Boerzsoenyi 2008) or bm1981 (Bideau-Mehu 1981); default %(default)s')
    return p


def outdir(args):
    """Resolve and create the output directory of parsed arguments."""
    d = args.outdir if args.outdir is not None else default_outdir()
    d.mkdir(parents=True, exist_ok=True)
    provenance.set_args(args)                                   # every output of the driver records these arguments
    return d


def thc(p):
    """Add the --thc-mrad option (fixed gas Cherenkov angle in place of the Xe index model) to the parser p."""
    p.add_argument('--thc-mrad', type=float, default=None,
                   help='fixed Xe Cherenkov angle [mrad]; default: FT x QE-weighted mean over the ARC SiPM QE band from --xe-index')
    return p


def pressure(p):
    """Add the --pressure option (Xe pressure [bar abs], default constants.P_REF) to the parser p."""
    p.add_argument('--pressure', type=float, default=P_REF, help='Xe pressure [bar abs]; default %(default)g (constants.P_REF)')
    return p


def gas_yield(p):
    """Add the --gas-yield option (emitted Xe photons over constants.GAS_YIELD_REF_LEN of sim.MirrorToy) to the parser p."""
    p.add_argument('--gas-yield', type=float, default=None,
                   help='emitted Xe photons over %g mm of the gas light (beta = 1, %g-%g nm); default sim.gas_yield(--pressure)'
                        % (GAS_YIELD_REF_LEN, *YIELD_BAND_NM))
    return p


def thc_value(args, P_bar):
    """Gas Cherenkov angle [rad] for beta = 1 at P_bar [bar]: --thc-mrad if given, else cherenkov.theta_xe with --xe-index."""
    return args.thc_mrad/1e3 if args.thc_mrad is not None else theta_xe(P_bar, model=args.xe_index)


def slope_term(p):
    """Add the --slope-term option (footprint slope term of the legacy beta = 1 toys, see sim.footprint_fraction) to the parser p."""
    p.add_argument('--slope-term', action='store_true',
                   help='mirror footprint with the extra track-slope offset of the legacy scripts, which counts the track displacement twice')
    return p


def mirror_r(spec):
    """argparse type of --mirror-r: spec unchanged if layout.mirror_reflectivity accepts it."""
    try:
        mirror_reflectivity(spec)
    except ValueError:
        raise argparse.ArgumentTypeError('%r: choose %s or a flat value in (0, 1]' % (spec, ', '.join(MIRROR_REFLECTIVITIES))) from None
    return spec


def reflectance(spec):
    """argparse type of the board reflectances: a float in [0, 1)."""
    R = float(spec)
    if not 0 <= R < 1:
        raise argparse.ArgumentTypeError('%r: a reflectance in [0, 1)' % spec)
    return R


def preset_help():
    """Description of PRESETS for the --preset help, from its values."""
    return '; '.join('%s: %s' % (name, ', '.join(['--%s %s' % (k.replace('_', '-'), '%g' % v if isinstance(v, float) else v) for k, v in d.items()]
                                                  + (['--nphot %d in cosmic_rates.py' % PRESET_NPHOT[name]] if name in PRESET_NPHOT else [])))
                     for name, d in PRESETS.items())


def detector_inputs(p):
    """Add the detector-input options (--preset, --response, --mirror-r, --sensor-model, --pore-t, --aerogel-index,
    --aerogel-lambda-ref, --porosity-vac, --board-r-spec, --board-r-diff, --prox-rayleigh) to the parser p; inputs() resolves them."""
    p.add_argument('--preset', choices=tuple(PRESETS), default=None,
                   help='named set of detector inputs (%s). Options given explicitly override it; default: none (the defaults '
                        'below)' % preset_help())
    p.add_argument('--response', choices=tuple(RESPONSES), default=None,
                   help='photodetection response: arc_qe (ARC simulation SiPM QE) or s13361 (Hamamatsu S13361-3050 PDE, '
                        'through its window); default %s' % INPUT_DEFAULTS['response'])
    p.add_argument('--mirror-r', type=mirror_r, default=None,
                   help='mirror reflectivity: %s (nominal: assumed curve from the Edmund enhanced-Al spec, not measured; '
                        'constants.MIRROR_REFL_NOMINAL) or a flat value in (0, 1]; default %s' % (' or '.join(MIRROR_REFLECTIVITIES),
                                                                                                   INPUT_DEFAULTS['mirror_r']))
    n = 2*BOARD_NPAD
    p.add_argument('--sensor-model', choices=SENSOR_MODELS, default=None,
                   help='2" SiPM: square (continuous, fully active) or board (ARC prototype board: 2 x 2 S13361-3050NE-08 packages, '
                        '%d x %d pads of %g x %g mm at %g mm pitch, binary pad hits); default %s'
                        % (n, n, BOARD_PAD, BOARD_PAD, PIXEL_PITCH, INPUT_DEFAULTS['sensor_model']))
    p.add_argument('--pore-t', type=float, default=None,
                   help='temperature of the gas in the aerogel pores [K] for the pore-gas index shift (e.g. 274 for a cold box); '
                        'default %g' % INPUT_DEFAULTS['pore_t'])
    p.add_argument('--aerogel-index', choices=AEROGEL_MODELS, default=None,
                   help='aerogel dispersion (media.aerogel_medium): constant, single-pole (lambda0 = %g nm) or k4geo, equal to the '
                        'tile index at --aerogel-lambda-ref; default %s' % (AEROGEL_POLE_NM, INPUT_DEFAULTS['aerogel_index']))
    p.add_argument('--aerogel-lambda-ref', type=float, default=None,
                   help='wavelength [nm] at which the dispersive aerogel equals the tile index; default %g' % INPUT_DEFAULTS['aerogel_lambda_ref'])
    p.add_argument('--porosity-vac', type=float, default=None,
                   help='porosity of the pore-vac tile variant, n = %g + porosity (n_gas - 1) (the pore-air variant uses %g); '
                        'default %g' % (AEROGEL_N_AIR, AEROGEL_POROSITY_AIRREF, INPUT_DEFAULTS['porosity_vac']))
    for k, kind in (('spec', 'specular'), ('diff', 'diffuse (Lambertian)')):
        p.add_argument('--board-r-%s' % k, type=reflectance, default=None,
                       help='%s reflectance of the board, flat in wavelength, followed for %s reflections with --sensor-model board '
                            '(constants.BOARD_R_%s: %g, Congedo et al., arXiv:2602.20139); default %g'
                            % (kind, REFL_GENS, k.upper(), {'spec': BOARD_R_SPEC, 'diff': BOARD_R_DIFF}[k], INPUT_DEFAULTS['board_r_%s' % k]))
    p.add_argument('--prox-rayleigh', choices=PROX_RAYLEIGH, default=None,
                   help='Rayleigh transmission of the configuration-4 aerogel photons: per-photon (along each photon direction from '
                        'its emission depth to the tile exit) or vertical (one factor, that of a vertical muon); default %s'
                        % INPUT_DEFAULTS['prox_rayleigh'])
    return p


def inputs(args):
    """{name: value} of the detector inputs of parsed arguments: the option if given, else the --preset value, else its default."""
    base = preset(args.preset) if getattr(args, 'preset', None) else INPUT_DEFAULTS
    return {k: (getattr(args, k) if getattr(args, k, None) is not None else v) for k, v in base.items()}


def porosity(p):
    """Add the --porosity option (pore fraction of the vacuum-referenced pore-gas shift, media.aerogel_index) to the parser p."""
    p.add_argument('--porosity', type=float, default=AEROGEL_POROSITY,
                   help='aerogel porosity in n = %g + porosity (n_gas - 1) (%g: constants.AEROGEL_POROSITY_AIRREF); default %%(default)s'
                        % (AEROGEL_N_AIR, AEROGEL_POROSITY_AIRREF))
    return p
