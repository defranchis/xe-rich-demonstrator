"""Acceptance of aerogel and gas light vs muon zenith angle, and SiPM hit maps for single muons.

Reads toy_76.npz from toy_aerogel.py (stdout, toy_aerogel_acceptance.png).
"""
from pathlib import Path

import numpy as np

from xerich import cli
from xerich.analysis import binned_mean
from xerich.constants import MIRROR_D, MIRROR_R, SENSOR_HALF_2IN, SENSOR_HALF_PROTO, SENSOR_MUON_RATE
from xerich.layout import Aerogel, Mirror, Radiator, Sensor
from xerich.media import aerogel_index
from xerich.plotting import AEROGEL, GAS, INK, LIGHT, MUTED, plt, save
from xerich.sim import MirrorToy

P_BAR = 3
LAM = 400                    # wavelength for the indices [nm]


def main():
    p = cli.xe_index(cli.parser(__doc__))
    p.add_argument('--input', type=Path, default=None, help='toy_76.npz from toy_aerogel.py (default: <outdir>/toy_76.npz)')
    args = p.parse_args(); out = cli.outdir(args)
    r = dict(np.load(args.input if args.input is not None else out/'toy_76.npz'))
    rate = SENSOR_MUON_RATE
    bins = np.arange(0, 32, 2.0); c = 0.5*(bins[1:]+bins[:-1]); idx = np.digitize(r['th'], bins)-1
    mean_na = binned_mean(r['na'], idx, len(c)); mean_ng = binned_mean(r['ng'], idx, len(c))
    print('arc fraction quantiles (aerogel, muons with any photon):', np.quantile(r['arc_a'][r['fa'] > 0], [.5, .9, .99, 1]))
    for thr in (1, 2, 3, 5):
        print('>= %d emitted aerogel photons on SiPM: %.1f%% of muons -> %.0f /h' % (thr, 100*np.mean(r['na'] >= thr), rate*np.mean(r['na'] >= thr)))
    print('mean emitted aerogel photons on SiPM for 8-18 deg muons: %.2f' % r['na'][(r['th'] > 8)&(r['th'] < 18)].mean())
    print('gas: >=80%% of cone on SiPM: %.1f%% -> %.0f /h' % (100*np.mean(r['fg'] >= .8), rate*np.mean(r['fg'] >= .8)))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    a = ax[0]
    a.plot(c, mean_ng, '-o', color=GAS, ms=4, label='gas (Xe 3 bar)')
    a.plot(c, mean_na, '-o', color=AEROGEL, ms=4, label='aerogel (n≈1.032 with Xe)')
    a.set_xlabel('muon zenith angle [deg]'); a.set_ylabel('emitted photons reaching SiPM / muon\n(320–450 nm, no QE, no scattering)')
    a.set_title('Geometric acceptance vs muon angle\n(3" mirror f=203 mm, 5.1×5.1 cm SiPM, μ uniform on SiPM)', fontsize=10)
    a2 = a.twinx(); h, _ = np.histogram(r['th'], bins); a2.bar(c, h/h.sum(), width=1.8, color=LIGHT, alpha=.3, zorder=0)
    a2.set_ylabel('fraction of muons', color=MUTED); a2.set_ylim(0, 0.4)
    a.set_zorder(a2.get_zorder()+1); a.patch.set_visible(False); a.legend(fontsize=8.5, loc='upper right'); a.grid(alpha=.3)

    ng = Radiator(P_BAR, model=args.xe_index).n(LAM)
    toy = MirrorToy(Mirror(MIRROR_R, MIRROR_D), Sensor(SENSOR_HALF_2IN), Aerogel(n=aerogel_index(ng)), ng)
    hs = SENSOR_HALF_PROTO
    rng = np.random.default_rng(11)
    for j, (thmu, lab) in enumerate(((3, 'μ at 3°'), (12, 'μ at 12°'))):
        b = ax[1+j]
        b.add_patch(plt.Rectangle((-hs, -hs), 2*hs, 2*hs, fill=False, ec=INK, lw=1.3))
        for k in range(4):
            ph = rng.uniform(0, 2*np.pi); th = np.radians(thmu)
            u = np.array([[np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), -np.cos(th)]])
            p0 = np.array([[rng.uniform(-15, 15), rng.uniform(-15, 15), toy.f]])
            phi = np.linspace(0, 2*np.pi, 360, endpoint=False)
            for kind, col in (('aero', AEROGEL), ('gas', GAS)):
                for frac in (0.25, 0.5, 0.75):
                    if kind == 'aero':
                        pe = p0+frac*toy.T/np.cos(th)*u
                        pb, dg, okb = toy.aerogel_exit(pe, u, phi); xy, ok = toy.image(pb, dg); ok &= okb
                    else:
                        pe = p0+(toy.T/np.cos(th)+frac*(toy.f-toy.T)/np.cos(th))*u
                        xy, ok = toy.gas_image(pe, u, phi)
                    b.plot(xy[0][ok[0], 0], xy[0][ok[0], 1], '.', ms=1.5, color=col)
        b.set_xlim(-30, 30); b.set_ylim(-30, 30); b.set_aspect('equal')
        b.set_title('Hit pattern on SiPM, %s\n(4 random muons; blue = gas, orange = aerogel)' % lab, fontsize=10)
        b.set_xlabel('x [mm]'); b.set_ylabel('y [mm]')
    plt.tight_layout(); save(out, 'toy_aerogel_acceptance.png', dpi=140)


if __name__ == '__main__':
    main()
