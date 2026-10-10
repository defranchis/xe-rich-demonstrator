import numpy as np
rng=np.random.default_rng(2)
I0=70.0  # vertical muon intensity, m^-2 s^-1 sr^-1 (PDG, ~cos^2 law)
def rate(D_mm, f_mm, L_mm, th_c, sensor_half_mm, frac_req=1.0, N=400000, ca=0.9):
    a=ca*D_mm/2
    # sample track impact point on mirror plane uniformly in a box, direction with cos^3 weight (cos^2 flux x cos projection)
    box=a+L_mm*0.6
    x=rng.uniform(-box,box,N); y=rng.uniform(-box,box,N)
    # sample theta up to 35 deg uniformly in solid angle; weight later
    tmax=np.radians(35); ct=rng.uniform(np.cos(tmax),1,N); st=np.sqrt(1-ct**2); ph=rng.uniform(0,2*np.pi,N)
    sx,sy=st*np.cos(ph)/ct, st*np.sin(ph)/ct   # transverse slope
    w=ct**3
    # emission points s in [0,L] above mirror, photon azimuth psi; hit on mirror (flat approx, small angles)
    ns,npsi=8,16
    s=(np.arange(ns)+0.5)/ns*L_mm; psi=np.arange(npsi)/npsi*2*np.pi
    S,P=np.meshgrid(s,psi,indexing='ij')
    hx=x[:,None,None]+S*sx[:,None,None]+S*np.tan(th_c)*np.cos(P)
    hy=y[:,None,None]+S*sy[:,None,None]+S*np.tan(th_c)*np.sin(P)
    inside=(hx**2+hy**2<a*a).mean(axis=(1,2))
    # ring on sensor: centre at f*slope (centre of curvature on axis), radius f*tan(th_c)
    rr=f_mm*np.tan(th_c)
    cx,cy=f_mm*sx,f_mm*sy
    ok_sensor=(np.abs(cx)+rr<sensor_half_mm)&(np.abs(cy)+rr<sensor_half_mm)
    acc=(inside>=frac_req-1e-9)&ok_sensor
    area=(2*box*1e-3)**2; dOmega=2*np.pi*(1-np.cos(tmax))
    R=I0*area*dOmega*np.mean(w*acc)  # Hz
    eff=np.sum(w*acc*inside)/max(np.sum(w*acc),1e-12)
    return R*3600, eff
print('3 bar (th_c=63.6 mrad), L=200 mm radiator, sensor 80x80 mm; rates per hour')
for (D,f) in ((50.8,203.2),(76.2,203.2),(101.6,203.2),(152.4,203.2),(200,203.2),(260,185)):
    r1,_=rate(D,f,200,0.0636,40)
    r8,e8=rate(D,f,200,0.0636,40,frac_req=0.8)
    print('D=%5.1f f=%5.1f : full ring %6.1f /h   >=80%% ring %6.1f /h (mean collected %.2f)'%(D,f,r1,r8,e8))
print('sensor 40x40 mm')
for (D,f) in ((76.2,203.2),(152.4,203.2),(200,203.2)):
    r1,_=rate(D,f,200,0.0636,20); print('D=%5.1f: full ring %6.1f /h'%(D,r1))
print('6in f=610, L=600, sensor 80 / 160 mm')
for sh in (40,80):
    r1,_=rate(152.4,609.6,600,0.0636,sh); print('sensor half %d: %.2f /h'%(sh,r1))
