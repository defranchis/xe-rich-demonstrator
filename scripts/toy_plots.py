import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import toy_aerogel as TA
from toy_aerogel import *
r=dict(np.load('toy_76.npz'))
rate=1033.
bins=np.arange(0,32,2.0); c=0.5*(bins[1:]+bins[:-1]); idx=np.digitize(r['th'],bins)-1
mean_na=np.array([r['na'][idx==b].mean() for b in range(len(c))]); mean_ng=np.array([r['ng'][idx==b].mean() for b in range(len(c))])
arc=np.array([r['arc_a'][idx==b].mean() for b in range(len(c))])
print('arc fraction quantiles (aerogel, muons with any photon):',np.quantile(r['arc_a'][r['fa']>0],[.5,.9,.99,1]))
for thr in (1,2,3,5):
    print('>= %d emitted aerogel photons on SiPM: %.1f%% of muons -> %.0f /h'%(thr,100*np.mean(r['na']>=thr),rate*np.mean(r['na']>=thr)))
print('mean emitted aerogel photons on SiPM for 8-18 deg muons: %.2f'%r['na'][(r['th']>8)&(r['th']<18)].mean())
print('gas: >=80%% of cone on SiPM: %.1f%% -> %.0f /h'%(100*np.mean(r['fg']>=.8),rate*np.mean(r['fg']>=.8)))
# event displays: muons at 12 deg, several azimuths/positions
fig,ax=plt.subplots(1,3,figsize=(14,4.6),gridspec_kw={'width_ratios':[1.25,1,1]})
a=ax[0]
a.plot(c,mean_ng,'-o',color='#2563eb',ms=4,label='gas (Xe 3 bar)')
a.plot(c,mean_na,'-o',color='#d97706',ms=4,label='aerogel (n≈1.032 with Xe)')
a.set_xlabel('muon zenith angle [deg]'); a.set_ylabel('emitted photons reaching SiPM / muon\n(320–450 nm, no QE, no scattering)')
a.set_title('Geometric acceptance vs muon angle\n(3" mirror f=203 mm, 5.1×5.1 cm SiPM, μ uniform on SiPM)',fontsize=10)
a2=a.twinx(); h,_=np.histogram(r['th'],bins); a2.bar(c,h/h.sum(),width=1.8,color='#9ca3af',alpha=.3,zorder=0); a2.set_ylabel('fraction of muons',color='#6b7280'); a2.set_ylim(0,0.4)
a.set_zorder(a2.get_zorder()+1); a.patch.set_visible(False); a.legend(fontsize=8.5,loc='upper right'); a.grid(alpha=.3)
# event displays
rng=np.random.default_rng(11)
for j,(thmu,lab) in enumerate(((3,'μ at 3°'),(12,'μ at 12°'))):
    b=ax[1+j]
    b.add_patch(plt.Rectangle((-25.6,-25.6),51.2,51.2,fill=False,ec='#1f2933',lw=1.3))
    for k in range(4):
        ph=rng.uniform(0,2*np.pi); th=np.radians(thmu)
        u=np.array([[np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),-np.cos(th)]]); p0=np.array([[rng.uniform(-15,15),rng.uniform(-15,15),f]])
        phi=np.linspace(0,2*np.pi,360,endpoint=False)
        for kind,col in (('aero','#d97706'),('gas','#2563eb')):
            for frac in (0.25,0.5,0.75):
                if kind=='aero':
                    pe=p0+frac*TA.T/np.cos(th)*u
                    d=cone(u,tha,phi); s=(pe[:,None,2]-(f-TA.T))/(-d[...,2]); pb=pe[:,None,:]+s[...,None]*d
                    okb=(np.abs(pb[...,0])<TILE)&(np.abs(pb[...,1])<TILE)
                    dg,okr=refract(d,np.array([0,0,1.]),na/ng); xy,ok=to_mirror_and_back(pb,dg,0.9*38.1); ok&=okb&okr
                else:
                    pe=p0+(TA.T/np.cos(th)+frac*(f-TA.T)/np.cos(th))*u
                    d=cone(u,thg,phi); xy,ok=to_mirror_and_back(np.broadcast_to(pe[:,None,:],d.shape),d,0.9*38.1)
                b.plot(xy[0][ok[0],0],xy[0][ok[0],1],'.',ms=1.5,color=col)
    b.set_xlim(-30,30); b.set_ylim(-30,30); b.set_aspect('equal')
    b.set_title('Hit pattern on SiPM, %s\n(4 random muons; blue = gas, orange = aerogel)'%lab,fontsize=10)
    b.set_xlabel('x [mm]'); b.set_ylabel('y [mm]')
plt.tight_layout(); plt.savefig('toy_aerogel_acceptance.png',dpi=140)
