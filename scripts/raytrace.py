import numpy as np
R=406.4; f=R/2; L=200.0; th=0.0636   # mm, rad (3 bar, 320-450 nm)
C=np.array([0,0,0.0])                  # centre of curvature; mirror vertex at z=-R... use z axis toward mirror
# Mirror: sphere |x-C|=R, vertex at (0,0,R)? Put C at origin, mirror vertex at z=+R, detector plane z=R-f
def trace(h0,tilt=0.0,n=4000,rng=np.random.default_rng(1)):
    d0=np.array([np.sin(tilt),0,np.cos(tilt)])
    zs=rng.uniform(R-f-L*0+0, R-f+L, n)*0  # placeholder
    s=rng.uniform(0,L,n)                    # emission point: path length before detector plane... track starts at detector plane
    p0=np.array([h0,0,R-f])+s[:,None]*d0   # positions along track (from detector plane toward mirror)
    phi=rng.uniform(0,2*np.pi,n)
    # photon direction: angle th around d0
    e1=np.array([np.cos(tilt),0,-np.sin(tilt)]); e2=np.array([0,1,0])
    u=np.cos(th)*d0+np.sin(th)*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    # intersect with sphere |p+t u|=R, t>0
    b=np.sum(p0*u,1); c=np.sum(p0*p0,1)-R*R; t=-b+np.sqrt(b*b-c)
    q=p0+t[:,None]*u
    rho=np.hypot(q[:,0],q[:,1])
    nrm=q/R; v=u-2*np.sum(u*nrm,1)[:,None]*nrm
    t2=((R-f)-q[:,2])/v[:,2]; hit=q+t2[:,None]*v
    # reconstruct: ideal focus of direction w is point on plane where line through C with dir w hits: C + (R-f)/w_z * w
    # ring centre for track dir d0:
    c0=(R-f)/d0[2]*d0
    r=np.hypot(hit[:,0]-c0[0],hit[:,1]-c0[1])
    threc=np.arctan(r/f)
    return threc,rho
print('h0[mm] tilt[deg]  max mirror radius used [mm]   <theta> [mrad]   rms spread [mrad]')
for tilt in (0,5,10):
  for h0 in (0,25,50,75):
    t,rho=trace(h0,np.radians(tilt))
    print('%5d %4d   %6.1f   %6.2f   %5.2f'%(h0,tilt,rho.max(),1e3*t.mean(),1e3*t.std()))
