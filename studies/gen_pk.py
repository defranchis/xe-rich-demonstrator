"""Pythia 8 pi/K/p momenta in e+e- collisions at an FCC-ee energy (stdout, mom_<key>.npz).

Usage: gen_pk.py <91|160|240|365> <Nevt> [--outdir DIR]. Final-state hadrons with
p > 0.3 GeV and |cos(theta)| < 0.9 are kept; the seed is int(sqrt(s)).
"""
import numpy as np

from xerich import cli, provenance

CFG = {
    '91':  (91.188, ['WeakSingleBoson:ffbar2gmZ = on', '23:onMode = off', '23:onIfAny = 1 2 3 4 5'], 'Z→qq̄'),
    '160': (160.0,  ['WeakDoubleBoson:ffbar2WW = on'], 'WW'),
    '240': (240.0,  ['HiggsSM:ffbar2HZ = on'], 'ZH'),
    '365': (365.0,  ['Top:ffbar2ttbar(s:gmZ) = on'], 'tt̄'),
}
IDS = {211: 'pi', 321: 'K', 2212: 'p'}
P_MIN = 0.3                  # [GeV]
COS_MAX = 0.9


def main():
    p = cli.parser(__doc__)
    p.add_argument('key', choices=sorted(CFG)); p.add_argument('nevt', type=int)
    args = p.parse_args(); out = cli.outdir(args)
    import pythia8mc as p8
    key, N = args.key, args.nevt
    E, procs, lab = CFG[key]
    py = p8.Pythia("", False)
    for s in ['Beams:idA = 11', 'Beams:idB = -11', f'Beams:eCM = {E}', 'PDF:lepton = on',
              'Next:numberCount = 0', 'Random:setSeed = on', f'Random:seed = {int(E)}']+procs:
        py.readString(s)
    py.init()
    res = {'pi': [], 'K': [], 'p': []}
    nev = 0
    while nev < N:
        if not py.next():
            continue
        nev += 1
        for i in range(py.event.size()):
            q = py.event[i]
            if not q.isFinal():
                continue
            a = abs(q.id())
            if a in IDS:
                P = q.pAbs()
                if P < P_MIN or abs(q.pz()/P) > COS_MAX:
                    continue
                res[IDS[a]].append(P)
    np.savez(out/f'mom_{key}.npz', **provenance.npz(), N=nev, **{k: np.array(v) for k, v in res.items()})
    print(key, lab, nev, {k: len(v)/nev for k, v in res.items()})


if __name__ == '__main__':
    main()
