import numpy as np
from xe_optics import n_of
rng=np.random.default_rng(7)
R=406.4; f=R/2; T=10.0; TILE=25.0; SENS=25.4   # SiPM 2"x2" = 50.8 mm (confirmed)
P=3; ng=n_of(400,P); na=1.03+0.97*(ng-1)          # aerogel n with Xe-filled pores
tha=np.arccos(1/na); thg=np.arccos(1/ng)
C=np.array([0,0,R])
def refract(d,N,eta):
    cosi=-(d*N).sum(-1,keepdims=True); k=1-eta**2*(1-cosi**2)
    ok=(k[...,0]>0); t=eta*d+(eta*cosi-np.sqrt(np.clip(k,0,None)))*N
    return t/np.linalg.norm(t,axis=-1,keepdims=True), ok
def cone(u,th,phi):
    # orthonormal basis around u (N,3); phi (M,) -> (N,M,3)
    a=np.where(np.abs(u[:,2:3])<0.9,np.array([[0,0,1.]]),np.array([[1.,0,0]]))
    e1=np.cross(u,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True); e2=np.cross(u,e1)
    return (np.cos(th)*u[:,None,:]+np.sin(th)*(np.cos(phi)[None,:,None]*e1[:,None,:]+np.sin(phi)[None,:,None]*e2[:,None,:]))
def to_mirror_and_back(p,d,amir):
    # p,d (...,3) in gas going down. returns sensor hit xy and mask
    m=p-C; b=(m*d).sum(-1); c=(m*m).sum(-1)-R*R; disc=b*b-c
    s=-b+np.sqrt(np.clip(disc,0,None)); q=p+s[...,None]*d
    ok=(disc>0)&(np.hypot(q[...,0],q[...,1])<amir)&(d[...,2]<0)
    n=(q-C)/R; d2=d-2*(d*n).sum(-1,keepdims=True)*n
    ok&=d2[...,2]>0
    t=((f-T)-q[...,2])/np.where(d2[...,2]>0,d2[...,2],1); h=q+t[...,None]*d2
    intile=(np.abs(h[...,0])<TILE)&(np.abs(h[...,1])<TILE)
    d3,okr=refract(d2,np.array([0,0,-1.]),ng/na)
    d3=np.where(intile[...,None],d3,d2)
    t2=T/d3[...,2]; e=h+t2[...,None]*d3
    ok&=(np.abs(e[...,0])<SENS)&(np.abs(e[...,1])<SENS)
    return e[...,:2],ok
def sample_muons(N,thmax=np.radians(75)):
    # rate through horizontal plane ∝ cos^2(th)*cos(th) dΩ -> accept-reject on cos^3
    out=[]
    while sum(len(o) for o in out)<N:
        ct=rng.uniform(np.cos(thmax),1,4*N); keep=rng.uniform(0,1,4*N)<ct**3
        out.append(ct[keep])
    ct=np.concatenate(out)[:N]; st=np.sqrt(1-ct**2); ph=rng.uniform(0,2*np.pi,N)
    u=np.stack([st*np.cos(ph),st*np.sin(ph),-ct],1)
    x0=rng.uniform(-SENS,SENS,N); y0=rng.uniform(-SENS,SENS,N)
    return u,np.stack([x0,y0,np.full(N,f)],1),np.degrees(np.arccos(ct))
NPHI=90; NS=8
phi=np.arange(NPHI)/NPHI*2*np.pi
def run(N,amir,chunk=2000):
    res={'th':[],'fa':[],'arc_a':[],'na':[],'fg':[],'arc_g':[],'ng':[]}
    keep_hits=[]
    for i0 in range(0,N,chunk):
        u,p0,th=sample_muons(chunk)
        # ---- aerogel: emission depths in slab
        tt=((np.arange(NS)+0.5)/NS)[None,:]*T/(-u[:,2])[:,None]   # path params
        acc=np.zeros((chunk,NS,NPHI),bool); hits=[]
        for k in range(NS):
            pe=p0+tt[:,k:k+1]*u
            inside=(np.abs(pe[:,0])<TILE)&(np.abs(pe[:,1])<TILE)
            d=cone(u,tha,phi)                               # (N,M,3)
            # to bottom face
            s=(pe[:,None,2]-(f-T))/np.where(d[...,2]<0,-d[...,2],np.inf)
            pb=pe[:,None,:]+s[...,None]*d
            okb=(d[...,2]<0)&(np.abs(pb[...,0])<TILE)&(np.abs(pb[...,1])<TILE)&inside[:,None]
            dg,okr=refract(d,np.array([0,0,1.]),na/ng)
            xy,ok=to_mirror_and_back(pb,dg,amir)
            acc[:,k,:]=okb&okr&ok
            if k==NS//2: hits.append((xy,acc[:,k,:].copy()))
        L=T/(-u[:,2]); 
        nemit=24.7*L/10.0                                  # emitted photons, 320-450 nm, per track
        fa=acc.mean((1,2)); arc=acc.any(1).mean(1)
        # ---- gas
        zm=0.0; Lg=((f-T)-zm)/(-u[:,2]); tg=(T/(-u[:,2]))[:,None]+((np.arange(NS)+0.5)/NS)[None,:]*Lg[:,None]
        accg=np.zeros((chunk,NS,NPHI),bool)
        for k in range(NS):
            pe=p0+tg[:,k:k+1]*u
            d=cone(u,thg,phi)
            xy,ok=to_mirror_and_back(np.broadcast_to(pe[:,None,:],d.shape),d,amir)
            accg[:,k,:]=ok
        fg=accg.mean((1,2)); arcg=accg.any(1).mean(1); nemg=33.5*Lg/193.0
        for kk,v in zip(res,[th,fa,arc,fa*nemit,fg,arcg,fg*nemg]): res[kk].append(v)
        keep_hits.append((u,p0,th,hits[0]))
    return {k:np.concatenate(v) for k,v in res.items()},keep_hits
if __name__=='__main__':
    flux_hz=70.0*np.pi/2*(2*SENS*1e-3)**2*(1-np.cos(np.radians(75))**4)   # mu/s through sensor
    print('n_gas=%.5f n_aero=%.4f theta_aero=%.1f mrad theta_gas=%.1f mrad; muons through 5.1x5.1 cm SiPM: %.0f /h'%(ng,na,1e3*tha,1e3*thg,flux_hz*3600))
    for D in (76.2,152.4):
        r,_=run(20000,0.9*D/2)
        print('\nmirror %.1f mm (clear r=%.1f mm)'%(D,0.9*D/2))
        for lab,fr,arc,n in (('aerogel',r['fa'],r['arc_a'],r['na']),('gas',r['fg'],r['arc_g'],r['ng'])):
            print(' %-8s any photon: %5.1f%%   arc>=25%%: %5.1f%%   arc>=50%%: %5.1f%%   >=5 emitted ph on SiPM: %5.1f%%   >=10: %5.1f%%   mean frac(all mu) %.3f'%(
              lab,100*np.mean(fr>0),100*np.mean(arc>=.25),100*np.mean(arc>=.5),100*np.mean(n>=5),100*np.mean(n>=10),fr.mean()))
        both=(r['arc_a']>=.25)&(r['arc_g']>=.8)
        print(' gas ring >=80%% AND aerogel arc >=25%%: %.1f%%  -> %.0f /h'%(100*both.mean(),both.mean()*flux_hz*3600))
        print(' aerogel arc>=25%% -> %.0f /h ; >=5 emitted aerogel photons on SiPM -> %.0f /h'%(np.mean(r['arc_a']>=.25)*flux_hz*3600,np.mean(r['na']>=5)*flux_hz*3600))
        bins=[0,5,10,15,20,25,30,40,75]
        idx=np.digitize(r['th'],bins)-1
        print(' theta_mu bin : frac(mu)  <aero frac>  P(arc>=25%)  <gas frac>')
        for b in range(len(bins)-1):
            m=idx==b
            if m.sum()==0: continue
            print('  %2d-%2d deg   : %5.3f     %5.3f        %5.3f        %5.3f'%(bins[b],bins[b+1],m.mean(),r['fa'][m].mean(),np.mean(r['arc_a'][m]>=.25),r['fg'][m].mean()))
        np.savez('toy_%d.npz'%int(D),**r)
