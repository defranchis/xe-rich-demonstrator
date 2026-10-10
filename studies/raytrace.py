"""Spherical-aberration smearing of the reconstructed Cherenkov angle vs track offset and tilt (stdout)."""
import numpy as np

from xerich import cli
from xerich.constants import MIRROR_R
from xerich.layout import Mirror
from xerich.sim import aberration

L = 200.0                    # radiator length [mm]
P_BAR = 3.0


def main():
    args = cli.thc(cli.xe_index(cli.parser(__doc__))).parse_args()
    thc = cli.thc_value(args, P_BAR)
    rng = np.random.default_rng(1)
    mirror = Mirror(MIRROR_R)
    print('h0[mm] tilt[deg]  max mirror radius used [mm]   <theta> [mrad]   rms spread [mrad]')
    for tilt in (0, 5, 10):
        for h0 in (0, 25, 50, 75):
            t, rho = aberration(h0, np.radians(tilt), rng, mirror, L, thc)
            print('%5d %4d   %6.1f   %6.2f   %5.2f' % (h0, tilt, rho.max(), 1e3*t.mean(), 1e3*t.std()))


if __name__ == '__main__':
    main()
