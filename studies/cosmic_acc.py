"""Cosmic gas-ring rates vs mirror and sensor size, 3 bar, ring fully contained on the sensor (stdout)."""
import numpy as np

from xerich import cli
from xerich.layout import Mirror, Sensor
from xerich.sim import ring_rate_full


def main():
    args = cli.thc(cli.xe_index(cli.parser(__doc__))).parse_args()
    thc = cli.thc_value(args, 3.0)
    rng = np.random.default_rng(2)
    print('3 bar (th_c=%.1f mrad), L=200 mm radiator, sensor 80x80 mm; rates per hour' % (1e3*thc))
    for (D, f) in ((50.8, 203.2), (76.2, 203.2), (101.6, 203.2), (152.4, 203.2), (200, 203.2), (260, 185)):
        mirror = Mirror.from_focal(f, D)
        r1, _ = ring_rate_full(rng, mirror, Sensor(40), 200, thc)
        r8, e8 = ring_rate_full(rng, mirror, Sensor(40), 200, thc, frac_req=0.8)
        print('D=%5.1f f=%5.1f : full ring %6.1f /h   >=80%% ring %6.1f /h (mean collected %.2f)' % (D, f, r1, r8, e8))
    print('sensor 40x40 mm')
    for (D, f) in ((76.2, 203.2), (152.4, 203.2), (200, 203.2)):
        r1, _ = ring_rate_full(rng, Mirror.from_focal(f, D), Sensor(20), 200, thc)
        print('D=%5.1f: full ring %6.1f /h' % (D, r1))
    print('6in f=610, L=600, sensor 80 / 160 mm')
    for sh in (40, 80):
        r1, _ = ring_rate_full(rng, Mirror.from_focal(609.6, 152.4), Sensor(sh), 600, thc)
        print('sensor half %d: %.2f /h' % (sh, r1))


if __name__ == '__main__':
    main()
