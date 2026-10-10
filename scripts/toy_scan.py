import numpy as np
import toy_aerogel as TA
def run_cfg(sens_half, mirror_D, N=12000):
    TA.SENS=sens_half
    # muons uniform over the aerogel tile (needed for aerogel light), trigger matched to tile
    orig=TA.sample_muons
    def sm(n,thmax=np.radians(75)):
        u,p0,th=orig(n,thmax); p0[:,0]=TA.rng.uniform(-TA.TILE,TA.TILE,n); p0[:,1]=TA.rng.uniform(-TA.TILE,TA.TILE,n); return u,p0,th
    TA.sample_muons=sm
    r,_=TA.run(N,0.9*mirror_D/2)
    TA.sample_muons=orig
    return r
rate=70.0*np.pi/2*(0.05**2)*(1-np.cos(np.radians(75))**4)*3600   # muons/h through 5x5 cm tile
print('muons through 5x5 cm tile: %.0f /h'%rate)
print('sensor     mirror   | aerogel: <arc> | arc>=50%%/h  arc>=90%%/h  >=8 emitted/h  >=15 emitted/h | gas >=80%% cone /h')
for sh,lab in ((25.6,'5.1 cm'),(40,'8 cm'),(51.2,'10.2 cm'),(64,'12.8 cm')):
    for D in (76.2,152.4,200.0):
        r=run_cfg(sh,D)
        m=r['fa']>0
        print('%-8s %6.1f mm | %5.2f          | %7.0f   %8.0f   %10.0f   %12.0f   | %6.0f'%(lab,D,r['arc_a'][m].mean() if m.any() else 0,
              rate*np.mean(r['arc_a']>=.5),rate*np.mean(r['arc_a']>=.9),rate*np.mean(r['na']>=8),rate*np.mean(r['na']>=15),rate*np.mean(r['fg']>=.8)))
