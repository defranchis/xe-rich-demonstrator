import numpy as np
from xe_optics import n_of
rng=np.random.default_rng(3)
I0=70.0
def run(D,f,L,thc,half,req_m=0.8,req_s=0.8,N=300000,ca=0.9):
    a=ca*D/2; box=a+L*0.4
    x=rng.uniform(-box,box,N); y=rng.uniform(-box,box,N)
    tmax=np.radians(25); ct=rng.uniform(np.cos(tmax),1,N); st=np.sqrt(1-ct**2); ph=rng.uniform(0,2*np.pi,N)
    sx,sy=st*np.cos(ph)/ct, st*np.sin(ph)/ct; w=ct**3
    s=(np.arange(8)+0.5)/8*L; psi=np.arange(24)/24*2*np.pi
    S,P=np.meshgrid(s,psi,indexing='ij')
    hx=x[:,None,None]+S*sx[:,None,None]+S*np.tan(thc)*np.cos(P)
    hy=y[:,None,None]+S*sy[:,None,None]+S*np.tan(thc)*np.sin(P)
    fm=(hx**2+hy**2<a*a).mean(axis=(1,2))
    rr=f*np.tan(thc); cx,cy=f*sx,f*sy
    qx=cx[:,None]+rr*np.cos(psi); qy=cy[:,None]+rr*np.sin(psi)
    fs=((np.abs(qx)<half)&(np.abs(qy)<half)).mean(1)
    acc=(fm>=req_m)&(fs>=req_s)
    area=(2*box*1e-3)**2; dO=2*np.pi*(1-np.cos(tmax))
    return I0*area*dO*np.mean(w*acc)*3600, I0*area*dO*np.mean(w*((fm>=0.999)&(fs>=0.999)))*3600
for P in (2,3):
    thc=np.arccos(1/n_of(400,P))
    print('P=%d bar theta_c=%.1f mrad, ring radius at f=203: %.1f mm'%(P,1e3*thc,203*np.tan(thc)))
    for D in (50.8,76.2):
        for half,lab in ((25.6,'5.1x5.1 cm (4x 8x8, 3.2mm pitch)'),(20,'4x4 cm'),(40,'8x8 cm')):
            r80,r100=run(D,203.2,190,thc,half)
            print('   mirror %.1f mm, sensor %-32s: >=80%% ring %5.1f/h   full ring %5.1f/h'%(D,lab,r80,r100))
# aerogel ring
na=1.03; 
for P in (2,3):
    ng=n_of(400,P); tha=np.arccos(1/na); thg=np.arcsin(na*np.sin(tha)/ng)
    print('aerogel P=%d: theta in aerogel %.0f mrad, in gas %.0f mrad, ring radius at f=203: %.0f mm, footprint on mirror after 190mm: %.0f mm'%(P,1e3*tha,1e3*thg,203*np.tan(thg),190*np.tan(thg)))
# pixel errors
for pitch in (3.2,0.8):
    print('pitch %.1f mm: per-photon sigma = %.2f mrad at f=203'%(pitch,pitch/np.sqrt(12)/203*1e3))
# Xe in aerogel pores: n shift
for P in (2,3): print('aerogel n shift from Xe in pores (porosity~0.97): +%.4f'%(0.97*(n_of(400,P)-1)))
