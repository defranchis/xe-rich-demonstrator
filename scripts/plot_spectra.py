import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from xe_optics import *
mpi,mK,mp=0.13957,0.493677,0.938272
E=np.linspace(HC/650,HC/165,800)   # eV
lam=HC/E
cP={2:'#1f6fb4',3:'#d1495b'}
fig,ax=plt.subplots(2,2,figsize=(11,8.5))
# (a) dispersion
a=ax[0,0]
L=np.linspace(160,650,500)
for P in (2,3):
    a.plot(L,(n_of(L,P)-1)*1e3,color=cP[P],lw=2,label=f'Xe {P} bar, 20°C')
a.axvline(147,color='gray',ls=':'); a.text(150,3.9,'147 nm\nresonance',fontsize=8,color='gray')
a.set_xlabel('λ [nm]'); a.set_ylabel('(n−1) × 10³'); a.set_ylim(1,4.3); a.set_xlim(140,650)
a.legend(); a.set_title('Refractive index (Bideau-Mehu 1981, density-scaled)')
# (b),(c) spectra per energy
for k,(P) in enumerate((2,3)):
    a=ax[0,1] if P==2 else ax[1,0]
    for p,ls in ((5,':'),(10,'--'),(20,'-'),(40,'-.')):
        a.plot(E,dN_dxdE(E,p,mpi,P),color='#2a9d8f',ls=ls,lw=1.6,label=f'π {p} GeV')
        a.plot(E,dN_dxdE(E,p,mK,P),color='#e76f51',ls=ls,lw=1.6,label=f'K {p} GeV')
    a.set_xlabel('photon energy [eV]'); a.set_ylabel('d²N/dx dE  [cm⁻¹ eV⁻¹]')
    a.set_title(f'Frank–Tamm spectrum, Xe {P} bar'); a.set_xlim(E[0],E[-1])
    top=a.secondary_xaxis('top',functions=(lambda e:HC/np.maximum(e,1e-3),lambda l:HC/np.maximum(l,1e-3)))
    top.set_xticks([200,250,300,400,600]); top.set_xlabel('λ [nm]')
    a.legend(ncol=2,fontsize=7.5); a.grid(alpha=.3)
# (d) integrated yield vs momentum
a=ax[1,1]; p=np.logspace(np.log10(1.5),np.log10(100),400)
for P in (2,3):
    for m,c,nm in ((mpi,'#2a9d8f','π'),(mK,'#e76f51','K'),(mp,'#6d597a','p')):
        Ey=np.linspace(HC/600,HC/200,400)
        N=[np.trapezoid(dN_dxdE(Ey,pp,m,P),Ey) for pp in p]
        a.plot(p,N,color=c,ls='-' if P==3 else '--',lw=1.8,label=f'{nm}, {P} bar')
a.set_xscale('log'); a.set_xlabel('p [GeV]'); a.set_ylabel('N_γ / cm  (200–600 nm, emitted)')
a.set_title('Integrated yield (before window / QE)'); a.legend(fontsize=7.5,ncol=2); a.grid(alpha=.3,which='both')
plt.tight_layout(); plt.savefig('xe_cherenkov_spectra.png',dpi=140)
# numbers
Ey=np.linspace(HC/600,HC/200,400)
for P in (2,3):
    for pp in (5,10,20,45):
        print(P,pp,'pi %.3f'%np.trapezoid(dN_dxdE(Ey,pp,mpi,P),Ey),'K %.3f'%np.trapezoid(dN_dxdE(Ey,pp,mK,P),Ey),
              'dtheta(pi-K) mrad@300nm %.2f'%(1e3*(np.arccos(1/(n_of(300,P)*pp/np.hypot(pp,mpi)))-np.arccos(min(1,1/(n_of(300,P)*pp/np.hypot(pp,mK)))))))
    # chromatic spread of theta for beta=1 in 200-600 and 250-500 nm
    for lo,hi in ((200,600),(250,500),(300,600)):
        print('  chromatic theta_c range beta=1, %d-%d nm: %.1f -> %.1f mrad'%(lo,hi,1e3*np.arccos(1/n_of(hi,P)),1e3*np.arccos(1/n_of(lo,P))))
