import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from xe_optics import *
mpi,mK=0.13957,0.493677
keys=['91','160','240','365']; lab={'91':'Z (91 GeV)','160':'WW (160 GeV)','240':'ZH (240 GeV)','365':'tt̄ (365 GeV)'}
col={'91':'#264653','160':'#2a9d8f','240':'#e9c46a','365':'#e76f51'}
D={k:np.load(f'mom_{k}.npz') for k in keys}
def pth(m,P,lam=300):
    n=n_of(lam,P); return m/np.sqrt(n*n-1)
fig,ax=plt.subplots(2,2,figsize=(11,8.5))
bins=np.logspace(np.log10(0.3),np.log10(150),70); bc=np.sqrt(bins[1:]*bins[:-1]); dl=np.diff(np.log10(bins))
for j,(sp,m,nm) in enumerate((('pi',mpi,'π±'),('K',mK,'K±'))):
    a=ax[0,j]
    for k in keys:
        h,_=np.histogram(D[k][sp],bins); a.step(bc,h/D[k]['N']/dl,where='mid',color=col[k],lw=1.8,label=lab[k])
    for P,ls in ((2,'--'),(3,'-')):
        a.axvline(pth(m,P),color='k',ls=ls,lw=1); 
    a.text(pth(m,3),a.get_ylim()[1]*0.95 if False else 0,'')
    a.set_xscale('log'); a.set_xlabel('p [GeV]'); a.set_ylabel(f'{nm} per event / dlog₁₀p')
    a.set_title(f'{nm} momentum, |cosθ|<0.9  (lines: Xe threshold, -- 2 bar, — 3 bar)',fontsize=9.5)
    a.legend(fontsize=8); a.grid(alpha=.3,which='both')
E=np.linspace(HC/600,HC/200,300)
for j,(sp,m,nm) in enumerate((('pi',mpi,'π±'),('K',mK,'K±'))):
    a=ax[1,j]
    for k in keys:
        pp=D[k][sp]; pp=pp[pp>1.0]
        pp=np.random.default_rng(1).choice(pp,size=min(len(pp),20000),replace=False)
        for P,ls in ((2,'--'),(3,'-')):
            s=np.mean([dN_dxdE(E,x,m,P) for x in pp],axis=0)
            a.plot(E,s,color=col[k],ls=ls,lw=1.6,label=f'{lab[k]}, {P} bar')
    a.set_xlabel('photon energy [eV]'); a.set_ylabel('⟨d²N/dx dE⟩ per track [cm⁻¹ eV⁻¹]')
    a.set_title(f'Photon spectrum averaged over {nm} with p>1 GeV',fontsize=10)
    top=a.secondary_xaxis('top',functions=(lambda e:HC/np.maximum(e,1e-3),lambda l:HC/np.maximum(l,1e-3)))
    top.set_xticks([200,250,300,400,600]); top.set_xlabel('λ [nm]')
    a.legend(fontsize=6.5,ncol=2); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('xe_fccee_pik.png',dpi=140)
# table
rows=[]
for k in keys:
    for sp,m in (('pi',mpi),('K',mK)):
        pp=D[k][sp]; N=D[k]['N']
        r=[k,sp,len(pp)/N,np.median(pp),np.mean(pp)]
        for P in (2,3):
            r+=[np.mean(pp>pth(m,P)), np.mean((pp>pth(mpi,P))&(pp<pth(mK,P))) if sp=='K' else np.nan]
        r+=[np.mean(pp>20),np.mean(pp>40)]
        rows.append(r)
print('sqrt_s sp  n/ev  median mean | 2bar: f>th  f(thr-mode) | 3bar: f>th f(thr-mode) | f>20 f>40')
for r in rows: print('%4s %3s %6.2f %6.2f %6.2f | %.3f %.3f | %.3f %.3f | %.3f %.3f'%tuple(r))
