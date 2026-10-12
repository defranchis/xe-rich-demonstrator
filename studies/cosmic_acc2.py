"""Cosmic gas-ring rates at 2 and 3 bar with partial containment, aerogel ring size, pixel term and pore index shift (stdout)."""
import numpy as np

from xerich import cli
from xerich.analysis import pixel_sigma
from xerich.cherenkov import theta_c
from xerich.constants import AEROGEL_N_AIR, MIRROR_R, PIXEL_PITCH, SENSOR_HALF_PROTO
from xerich.layout import Mirror, Sensor
from xerich.media import n_xe, pore_index_shift
from xerich.sim import ring_rate_partial
from xerich.tracer import snell_angle

F = MIRROR_R/2               # mirror focal length [mm]
F_ROUND = round(F)           # focal length used in the printed estimates [mm]
L = 190                      # gas length [mm]
LAM = 400                    # wavelength for the indices [nm]


def main():
    args = cli.slope_term(cli.xe_index(cli.parser(__doc__))).parse_args(); model = args.xe_index
    rng = np.random.default_rng(3)
    for P in (2, 3):
        thc = theta_c(n_xe(LAM, P, model=model))
        print('P=%d bar theta_c=%.1f mrad, ring radius at f=203: %.1f mm' % (P, 1e3*thc, F_ROUND*np.tan(thc)))
        for D in (50.8, 76.2):
            for half, lab in ((SENSOR_HALF_PROTO, '5.1x5.1 cm (4x 8x8, 3.2mm pitch)'), (20, '4x4 cm'), (40, '8x8 cm')):
                r80, r100 = ring_rate_partial(rng, Mirror.from_focal(F, D), Sensor(half), L, thc, slope_term=args.slope_term)
                print('   mirror %.1f mm, sensor %-32s: >=80%% ring %5.1f/h   full ring %5.1f/h' % (D, lab, r80, r100))
    na = AEROGEL_N_AIR
    for P in (2, 3):
        ng = n_xe(LAM, P, model=model); tha = theta_c(na); thg = snell_angle(tha, na, ng)
        print('aerogel P=%d: theta in aerogel %.0f mrad, in gas %.0f mrad, ring radius at f=203: %.0f mm, footprint on mirror after 190mm: %.0f mm'
              % (P, 1e3*tha, 1e3*thg, F_ROUND*np.tan(thg), L*np.tan(thg)))
    for pitch in (PIXEL_PITCH, 0.8):
        print('pitch %.1f mm: per-photon sigma = %.2f mrad at f=203' % (pitch, pixel_sigma(pitch, F_ROUND)*1e3))
    for P in (2, 3):
        print('aerogel n shift from Xe in pores (porosity~0.97): +%.4f' % (pore_index_shift(n_xe(LAM, P, model=model))))


if __name__ == '__main__':
    main()
