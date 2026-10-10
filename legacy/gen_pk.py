import sys, numpy as np, pythia8mc as p8
cfg = {
 '91':  (91.188, ['WeakSingleBoson:ffbar2gmZ = on','23:onMode = off','23:onIfAny = 1 2 3 4 5'], 'Z→qq̄'),
 '160': (160.0,  ['WeakDoubleBoson:ffbar2WW = on'], 'WW'),
 '240': (240.0,  ['HiggsSM:ffbar2HZ = on'], 'ZH'),
 '365': (365.0,  ['Top:ffbar2ttbar(s:gmZ) = on'], 'tt̄'),
}
key=sys.argv[1]; N=int(sys.argv[2])
E,procs,lab=cfg[key]
py=p8.Pythia("",False)
for s in ['Beams:idA = 11','Beams:idB = -11',f'Beams:eCM = {E}','PDF:lepton = on',
          'Next:numberCount = 0','Random:setSeed = on',f'Random:seed = {int(E)}']+procs:
    py.readString(s)
py.init()
out={'pi':[],'K':[],'p':[]}
ids={211:'pi',321:'K',2212:'p'}
nev=0
while nev<N:
    if not py.next(): continue
    nev+=1
    for i in range(py.event.size()):
        q=py.event[i]
        if not q.isFinal(): continue
        a=abs(q.id())
        if a in ids:
            P=q.pAbs()
            if P<0.3 or abs(q.pz()/P)>0.9: continue
            out[ids[a]].append(P)
np.savez(f'mom_{key}.npz',N=nev,**{k:np.array(v) for k,v in out.items()})
print(key,lab,nev,{k:len(v)/nev for k,v in out.items()})
