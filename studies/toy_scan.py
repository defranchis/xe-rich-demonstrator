"""Scan of SiPM size and mirror diameter for aerogel and gas light, muons uniform over the aerogel tile (stdout)."""
import numpy as np

from xerich import cli
from xerich.constants import MIRROR_R, MUON_THMAX, SENSOR_HALF_PROTO
from xerich.layout import Aerogel, Mirror, Radiator, Sensor
from xerich.media import aerogel_index
from xerich.muons import flux_through_square, redraw_xy, sample_cos3
from xerich.sim import MirrorToy

P_BAR = 3
LAM = 400                    # wavelength for the indices [nm]
N_MU = 12000


def main():
    model = cli.xe_index(cli.parser(__doc__)).parse_args().xe_index
    rng = np.random.default_rng(7)
    ng = Radiator(P_BAR, model=model).n(LAM); aero = Aerogel(n=aerogel_index(ng))
    rate = flux_through_square(2*aero.half*1e-3, MUON_THMAX)*3600
    print('muons through 5x5 cm tile: %.0f /h' % rate)
    print('sensor     mirror   | aerogel: <arc> | arc>=50%%/h  arc>=90%%/h  >=8 emitted/h  >=15 emitted/h | gas >=80%% cone /h')
    for sh, lab in ((SENSOR_HALF_PROTO, '5.1 cm'), (40, '8 cm'), (51.2, '10.2 cm'), (64, '12.8 cm')):
        for D in (76.2, 152.4, 200.0):
            sensor = Sensor(sh); toy = MirrorToy(Mirror(MIRROR_R, D), sensor, aero, ng)

            def sampler(n):
                u, p0, th = sample_cos3(rng, n, sensor.half, toy.f, MUON_THMAX)
                return u, redraw_xy(rng, p0, aero.half), th
            r = toy.run(N_MU, sampler)
            m = r['fa'] > 0
            print('%-8s %6.1f mm | %5.2f          | %7.0f   %8.0f   %10.0f   %12.0f   | %6.0f' % (
                lab, D, r['arc_a'][m].mean() if m.any() else 0, rate*np.mean(r['arc_a'] >= .5), rate*np.mean(r['arc_a'] >= .9),
                rate*np.mean(r['na'] >= 8), rate*np.mean(r['na'] >= 15), rate*np.mean(r['fg'] >= .8)))


if __name__ == '__main__':
    main()
