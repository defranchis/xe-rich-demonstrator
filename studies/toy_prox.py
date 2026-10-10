"""Proximity-focusing aerogel toy: ring radius and rates vs aerogel-sensor gap, in air and in Xe (stdout).

Default: muons with momenta from the joint momentum-zenith spectrum (muons.MuonSpectrum) at their own zenith, the
aerogel angle at each muon's velocity, the plain n = 1.03 tile with the two pore-gas variants alongside; the printed ring
radii come from the same per-muon angles as the rates (median and 16-84 % of the muons with >= 80 % of the ring on the
SiPM). --legacy: the beta = 1 toy quoted in docs/design_options.md section 6.2 (Xe 3 bar, pore shift 0.97 (n_gas - 1),
radii from a fixed angle).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from xerich import cli  # noqa: E402
from xerich.cherenkov import beta  # noqa: E402
from xerich.constants import (AEROGEL_N_AIR, AEROGEL_T, COSMIC_I0, M_MU, MIRROR_R, MUON_THMAX, SENSOR_HALF_2IN, SENSOR_HALF_PROTO,  # noqa: E402
                              THETA_AERO_GAS)
from xerich.layout import Aerogel, Mirror, Radiator, Sensor  # noqa: E402
from xerich.media import N_AIR, aerogel_index, aerogel_index_air  # noqa: E402
from xerich.muons import SPECTRUM_DEFAULT, SPECTRUM_MODELS, MuonSpectrum, flux_through_square, redraw_xy, sample_cos3  # noqa: E402
from xerich.sim import proximity, proximity_fractions  # noqa: E402
from xerich.tracer import snell_angle  # noqa: E402

P_BAR = 3
LAM = 400                    # wavelength for the indices [nm]
GAPS = (30, 50, 70)
VARIANTS = ('plain', 'pore-vac', 'pore-air')
_G = {}


def legacy(model):
    rng_mu = np.random.default_rng(7); rng_xy = np.random.default_rng(5)
    ng = Radiator(P_BAR, model=model).n(LAM); aero = Aerogel(n=aerogel_index(ng)); sensor = Sensor(SENSOR_HALF_PROTO)
    z0 = Mirror(MIRROR_R).f

    def sampler(n):
        u, p0, th = sample_cos3(rng_mu, n, SENSOR_HALF_2IN, z0, MUON_THMAX)
        return u, redraw_xy(rng_xy, p0, aero.half), th
    for gap in GAPS:
        a, b = proximity(aero, sensor, gap, ng, sampler)
        print('gap %d mm: ring r ~ %.0f-%.0f mm; >=80%% of ring on 5.1 cm SiPM: %.0f /h ; arc>=50%%: %.0f /h'
              % (gap, gap*np.tan(THETA_AERO_GAS), (gap+aero.thickness)*np.tan(THETA_AERO_GAS), a, b))


def tile_index(ng, variant):
    return {'plain': AEROGEL_N_AIR, 'pore-vac': float(aerogel_index(ng)), 'pore-air': float(aerogel_index_air(ng))}[variant]


def run_case(case):
    """Rates and radii for one (gas index, gap, variant): real momenta and beta = 1 on the same muons."""
    ng, gap, variant = case
    u, p0, p, w = _G['u'], _G['p0'], _G['p'], _G['w']
    aero = Aerogel(tile_index(ng, variant)); sensor = Sensor(SENSOR_HALF_PROTO)
    out = {}
    for tag, b in (('real', beta(p, M_MU)), ('beta1', np.ones(len(p)))):
        th = np.arccos(np.minimum(1, 1/(aero.n*b)))
        fa, arc = proximity_fractions(aero, sensor, gap, ng, u, p0, th)
        ww = w if tag == 'real' else np.ones(len(p))
        acc = fa >= .8
        thg = snell_angle(th, aero.n, ng)
        out[tag] = dict(r80=_G['rate']*np.mean(ww*acc), r50=_G['rate']*np.mean(ww*(arc >= .5)),
                        thg=np.quantile(thg[acc], (0.16, 0.5, 0.84), weights=ww[acc], method='inverted_cdf') if acc.any() else np.full(3, np.nan))
    return (ng, gap, variant), aero.n, out


def main():
    p = cli.xe_index(cli.parser(__doc__))
    p.add_argument('--legacy', action='store_true', help='beta = 1 toy of docs/design_options.md section 6.2 (Xe 3 bar, pore shift 0.97, fixed angle for radii)')
    p.add_argument('--pressure', type=float, default=P_BAR, help='Xe pressure [bar abs] (default %(default)s)')
    p.add_argument('--spectrum', choices=SPECTRUM_MODELS, default=SPECTRUM_DEFAULT, help='muon spectrum (default %(default)s)')
    p.add_argument('--nmu', type=int, default=40000, help='muons (default 40000)')
    p.add_argument('--seed', type=int, default=7, help='seed of the muon directions (default 7)')
    p.add_argument('--nproc', type=int, default=8, help='worker processes (default 8)')
    args = p.parse_args()
    if args.legacy:
        legacy(args.xe_index)
        return
    rng_mu = np.random.default_rng(args.seed); rng_xy = np.random.default_rng(args.seed+1); rng_p = np.random.default_rng(args.seed+2)
    sp = MuonSpectrum(args.spectrum)
    u, p0, _ = sample_cos3(rng_mu, args.nmu, SENSOR_HALF_2IN, 0.0, MUON_THMAX)
    p0 = redraw_xy(rng_xy, p0, Aerogel(AEROGEL_N_AIR).half)
    c = -u[:, 2]; pm = sp.sample_p(rng_p, c)
    rate = flux_through_square(2*Aerogel(AEROGEL_N_AIR).half*1e-3, MUON_THMAX)*3600
    _G.update(u=u, p0=p0, p=pm, w=sp.weight_vs_cos2(c, COSMIC_I0), rate=rate)
    ngx = Radiator(args.pressure, model=args.xe_index).n(LAM)
    cases = [(ng, gap, v) for ng in (N_AIR, ngx) for gap in GAPS for v in VARIANTS]
    with mp.get_context('fork').Pool(args.nproc) as pool:
        res = pool.map(run_case, cases)
    print('muons through the 5 x 5 cm tile, zenith < 75 deg: %.0f /h (cos^2, I0 = %g, beta = 1) ; %.0f /h (%s spectrum, p > %g GeV/c)'
          % (rate, COSMIC_I0, rate*_G['w'].mean(), args.spectrum, sp.p_lo))
    print('gas        gap  tile          n_tile    theta_gas [mrad] 16/50/84 %  ring r [mm] (gap tan th_gas / + T tan th_tile, median)  >=80% on 5.1 cm SiPM /h  arc>=50% /h   (beta = 1: >=80%, arc>=50%)')
    for (ng, gap, v), na, o in res:
        r = o['real']; t = r['thg']
        print('%-9s %4d  %-9s  %.5f   %5.1f %5.1f %5.1f          %4.1f-%4.1f                               %6.1f        %6.1f        (%6.1f, %6.1f)'
              % ('air' if ng == N_AIR else 'Xe %g bar' % args.pressure, gap, v, na, *(1e3*t), gap*np.tan(t[1]), gap*np.tan(t[1])+AEROGEL_T*np.tan(snell_angle(t[1], ng, na)),
                 r['r80'], r['r50'], o['beta1']['r80'], o['beta1']['r50']))


if __name__ == '__main__':
    main()
