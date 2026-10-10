"""3D geometric toy of the mirror setup with an aerogel tile under the SiPM, Xe at 3 bar.

Acceptance of aerogel and gas light for cosmic muons crossing the SiPM, for a 3" and a
6" mirror (stdout, toy_76.npz, toy_152.npz).
"""
import numpy as np

from xerich import cli
from xerich.cherenkov import theta_c
from xerich.constants import MIRROR_R, MUON_THMAX, SENSOR_HALF_2IN
from xerich.layout import Aerogel, Mirror, Radiator, Sensor
from xerich.media import aerogel_index
from xerich.muons import flux_through_square, sample_cos3
from xerich.sim import MirrorToy

P_BAR = 3
LAM = 400                    # wavelength for the indices [nm]
N_MU = 20000


def main():
    out = cli.outdir(cli.parser(__doc__).parse_args())
    rng = np.random.default_rng(7)
    ng = Radiator(P_BAR).n(LAM)
    aero = Aerogel(n=aerogel_index(ng)); sensor = Sensor(SENSOR_HALF_2IN)
    flux_hz = flux_through_square(2*sensor.half*1e-3, MUON_THMAX)
    print('n_gas=%.5f n_aero=%.4f theta_aero=%.1f mrad theta_gas=%.1f mrad; muons through 5.1x5.1 cm SiPM: %.0f /h'
          % (ng, aero.n, 1e3*theta_c(aero.n), 1e3*theta_c(ng), flux_hz*3600))
    for D in (76.2, 152.4):
        mirror = Mirror(MIRROR_R, D)
        toy = MirrorToy(mirror, sensor, aero, ng)
        r = toy.run(N_MU, lambda n: sample_cos3(rng, n, sensor.half, toy.f, MUON_THMAX))
        print('\nmirror %.1f mm (clear r=%.1f mm)' % (D, mirror.clear_radius))
        for lab, fr, arc, n in (('aerogel', r['fa'], r['arc_a'], r['na']), ('gas', r['fg'], r['arc_g'], r['ng'])):
            print(' %-8s any photon: %5.1f%%   arc>=25%%: %5.1f%%   arc>=50%%: %5.1f%%   >=5 emitted ph on SiPM: %5.1f%%   >=10: %5.1f%%   mean frac(all mu) %.3f' % (
                lab, 100*np.mean(fr > 0), 100*np.mean(arc >= .25), 100*np.mean(arc >= .5), 100*np.mean(n >= 5), 100*np.mean(n >= 10), fr.mean()))
        both = (r['arc_a'] >= .25)&(r['arc_g'] >= .8)
        print(' gas ring >=80%% AND aerogel arc >=25%%: %.1f%%  -> %.0f /h' % (100*both.mean(), both.mean()*flux_hz*3600))
        print(' aerogel arc>=25%% -> %.0f /h ; >=5 emitted aerogel photons on SiPM -> %.0f /h'
              % (np.mean(r['arc_a'] >= .25)*flux_hz*3600, np.mean(r['na'] >= 5)*flux_hz*3600))
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 75]
        idx = np.digitize(r['th'], bins)-1
        print(' theta_mu bin : frac(mu)  <aero frac>  P(arc>=25%)  <gas frac>')
        for b in range(len(bins)-1):
            m = idx == b
            if m.sum() == 0:
                continue
            print('  %2d-%2d deg   : %5.3f     %5.3f        %5.3f        %5.3f'
                  % (bins[b], bins[b+1], m.mean(), r['fa'][m].mean(), np.mean(r['arc_a'][m] >= .25), r['fg'][m].mean()))
        np.savez(out/('toy_%d.npz' % int(D)), **r)


if __name__ == '__main__':
    main()
