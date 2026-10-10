"""Proximity-focusing aerogel toy: ring radius and rates vs aerogel-sensor gap, Xe at 3 bar (stdout)."""
import numpy as np

from xerich import cli
from xerich.constants import MIRROR_R, MUON_THMAX, SENSOR_HALF_2IN, SENSOR_HALF_PROTO, THETA_AERO_GAS
from xerich.layout import Aerogel, Mirror, Radiator, Sensor
from xerich.media import aerogel_index
from xerich.muons import redraw_xy, sample_cos3
from xerich.sim import proximity

P_BAR = 3
LAM = 400                    # wavelength for the indices [nm]


def main():
    model = cli.xe_index(cli.parser(__doc__)).parse_args().xe_index
    rng_mu = np.random.default_rng(7); rng_xy = np.random.default_rng(5)
    ng = Radiator(P_BAR, model=model).n(LAM); aero = Aerogel(n=aerogel_index(ng)); sensor = Sensor(SENSOR_HALF_PROTO)
    z0 = Mirror(MIRROR_R).f

    def sampler(n):
        u, p0, th = sample_cos3(rng_mu, n, SENSOR_HALF_2IN, z0, MUON_THMAX)
        return u, redraw_xy(rng_xy, p0, aero.half), th
    for gap in (30, 50, 70):
        a, b = proximity(aero, sensor, gap, ng, sampler)
        print('gap %d mm: ring r ~ %.0f-%.0f mm; >=80%% of ring on 5.1 cm SiPM: %.0f /h ; arc>=50%%: %.0f /h'
              % (gap, gap*np.tan(THETA_AERO_GAS), (gap+aero.thickness)*np.tan(THETA_AERO_GAS), a, b))


if __name__ == '__main__':
    main()
