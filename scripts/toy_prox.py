import numpy as np
import toy_aerogel as TA
from toy_aerogel import cone, refract, na, ng, tha
rng=np.random.default_rng(5)
def prox(gap, sens=25.6, tile=25.0, T=10.0, N=40000):
    u,p0,th=TA.sample_muons(N); p0[:,0]=rng.uniform(-tile,tile,N); p0[:,1]=rng.uniform(-tile,tile,N)
    # p0 = entry at aerogel top face (z=0); aerogel z in [-T,0]; sensor at z=-(T+gap)
    p0[:,2]=0.0
    phi=np.linspace(0,2*np.pi,72,endpoint=False); acc=np.zeros((N,6,72),bool)
    for k,fr in enumerate((np.arange(6)+0.5)/6):
        pe=p0+(fr*T/(-u[:,2]))[:,None]*u
        d=cone(u,tha,phi); s=(pe[:,None,2]+T)/(-d[...,2]); pb=pe[:,None,:]+s[...,None]*d
        okb=(np.abs(pb[...,0])<tile)&(np.abs(pb[...,1])<tile)&(np.abs(pe[:,None,0])<tile)&(np.abs(pe[:,None,1])<tile)
        dg,okr=refract(d,np.array([0,0,1.]),na/ng)
        s2=gap/(-dg[...,2]); e=pb+s2[...,None]*dg
        acc[:,k]=okb&okr&(np.abs(e[...,0])<sens)&(np.abs(e[...,1])<sens)
    rate=70.0*np.pi/2*(2*tile*1e-3)**2*(1-np.cos(np.radians(75))**4)*3600
    fa=acc.mean((1,2)); arc=acc.any(1).mean(1)
    return rate*np.mean(fa>=.8), rate*np.mean(arc>=.5), np.degrees(tha)
for gap in (30,50,70):
    a,b,_=prox(gap); print('gap %d mm: ring r ~ %.0f-%.0f mm; >=80%% of ring on 5.1 cm SiPM: %.0f /h ; arc>=50%%: %.0f /h'%(gap,gap*np.tan(0.249),(gap+10)*np.tan(0.249),a,b))
