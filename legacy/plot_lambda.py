import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from xe_optics import *
mpi,mK=0.13957,0.493677
L=np.linspace(165,650,600); E=HC/L; J=HC/L**2   # dE/dlambda [eV/nm]
def dNdl(p,m,P): return dN_dxdE(E,p,m,P)*J
keys=['91','160','240','365']; lab={'91':'Z','160':'WW','240':'ZH','365':'tt̄'}
col={'91':'#264653','160':'#2a9d8f','240':'#e9c46a','365':'#e76f51'}
D={k:np.load(f'mom_{k}.npz') for k in keys}
fig,ax=plt.subplots(2,2,figsize=(11,8.5),sharex=True)
for i,P in enumerate((2,3)):
    a=ax[0,i]
    for p,ls in ((5,':'),(10,'--'),(20,'-'),(45,'-.')):
        a.plot(L,dNdl(p,mpi,P),color='#2a9d8f',ls=ls,lw=1.6,label=f'π {p} GeV')
        a.plot(L,dNdl(p,mK,P),color='#e76f51',ls=ls,lw=1.6,label=f'K {p} GeV')
    a.set_title(f'Xe {P} bar, fixed momentum'); a.set_ylabel('d²N/dx dλ [cm⁻¹ nm⁻¹]')
    a.set_yscale('log'); a.set_ylim(1e-4,2e-1); a.legend(ncol=2,fontsize=7.5); a.grid(alpha=.3,which='both')
rng=np.random.default_rng(1)
for j,(sp,m,nm) in enumerate((('pi',mpi,'π±'),('K',mK,'K±'))):
    a=ax[1,j]
    for k in keys:
        pp=D[k][sp]; pp=rng.choice(pp,size=min(len(pp),20000),replace=False)
        for P,ls in ((2,'--'),(3,'-')):
            s=np.mean([dNdl(x,m,P) for x in pp],axis=0)
            a.plot(L,s,color=col[k],ls=ls,lw=1.6,label=f'{lab[k]}, {P} bar')
    a.set_title(f'Averaged over FCC-ee {nm} (p>0.3 GeV, |cosθ|<0.9)',fontsize=10)
    a.set_ylabel('⟨d²N/dx dλ⟩ per track [cm⁻¹ nm⁻¹]'); a.set_yscale('log'); a.legend(ncol=2,fontsize=7.5); a.grid(alpha=.3,which='both')
for a in ax[1]: a.set_xlabel('λ [nm]')
for a in ax.flat: a.set_xlim(165,650)
plt.tight_layout(); plt.savefig('xe_cherenkov_lambda.png',dpi=140)
for P in (2,3):
    print(P,['%d:%.4f'%(l,dNdl(45,mpi,P)[np.argmin(abs(L-l))]) for l in (180,200,250,300,400,500,600)])
    # fraction of photons below 300 nm and in 200-600
    s=dNdl(45,mpi,P); m=(L>=200)
    print('  frac(200-300)/(200-600)=%.2f'%(np.trapezoid(s[(L>=200)&(L<=300)],L[(L>=200)&(L<=300)])/np.trapezoid(s[(L>=200)&(L<=600)],L[(L>=200)&(L<=600)])))
