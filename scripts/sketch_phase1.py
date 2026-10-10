import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
R=406.4; f=R/2; Dm=76.2; a=Dm/2
thc=0.0632; tha=0.249              # 3 bar gas (400 nm), aerogel ring angle in gas
xt=6.0                              # track offset [mm]
INK='#1f2933'; MUT='#6b7280'; GAS='#2563eb'; AER='#d97706'; MIR='#475569'
z_sens=f; z_aer0=f-10; z_aer1=f
fig=plt.figure(figsize=(12,8.6))
ax=fig.add_axes([0.01,0.05,0.56,0.90]); bx=fig.add_axes([0.60,0.40,0.37,0.52])
def mirror_hit(p,d):
    C=np.array([0,R]); m=p-C; b=m@d; c=m@m-R*R; t=-b+np.sqrt(b*b-c)
    q=p+t*d; return q
def trace(x0,z0,ang,col,ls='-',lw=1.2):
    d=np.array([np.sin(ang),-np.cos(ang)]); p=np.array([x0,z0])
    q=mirror_hit(p,d)
    if abs(q[0])>a*0.95:   # misses mirror: draw to bottom
        t=(z0+12)/np.cos(ang); e=p+t*d
        ax.plot([p[0],e[0]],[p[1],e[1]],color=col,ls=ls,lw=lw,alpha=.9); return
    n=(q-np.array([0,R]))/R; d2=d-2*(d@n)*n; t2=(z_sens-q[1])/d2[1]; e=q+t2*d2
    ax.plot([p[0],q[0],e[0]],[p[1],q[1],e[1]],color=col,ls=ls,lw=lw,alpha=.9)
# vessel
ax.add_patch(Rectangle((-62,-22),124,262,fill=False,ec=MUT,lw=1.4,ls=(0,(6,3))))
ax.text(70,232,'pressure vessel (dashed), Xe 2–3 bar',color=MUT,fontsize=9,va='center')
# mirror
xs=np.linspace(-a,a,200); zs=R-np.sqrt(R*R-xs**2)
ax.fill_between(xs,zs-5,zs,color=MIR,alpha=.85)
ax.text(70,-4,'spherical mirror\n76.2 mm Ø, f = 203 mm\n(Edmund #43-584)',fontsize=8.5,color=INK,va='center')
# sensor + aerogel
ax.add_patch(Rectangle((-25.6,z_sens),51.2,3,fc=INK,ec=INK))
ax.add_patch(Rectangle((-30,z_sens+3),60,4,fc='#9ca3af',ec='none'))
ax.text(70,z_sens+1,'SiPM plane ≈ 5.1 × 5.1 cm (focal plane)',fontsize=8.5,va='center',color=INK)
ax.text(70,z_sens+10,'cooling plate / board',fontsize=8,color=MUT,va='center')
# scintillators
ax.add_patch(Rectangle((-26,252),52,6,fc='#10b981',alpha=.5,ec='#047857'))
ax.add_patch(Rectangle((-40,-40),80,6,fc='#10b981',alpha=.5,ec='#047857'))
ax.text(70,255,'trigger scint. ~5 × 5 cm',fontsize=8.5,va='center',color='#047857')
ax.text(70,-37,'trigger scint. ~8 × 8 cm',fontsize=8.5,va='center',color='#047857')
# muon track
ax.annotate('',xy=(xt,-46),xytext=(xt,270),arrowprops=dict(arrowstyle='-|>',color=INK,lw=2))
ax.text(xt+2,264,'cosmic μ',fontsize=9,color=INK)
# gas photons
for z0 in (40,100,160,190):
    for s in (+1,-1): trace(xt,z0,s*thc,GAS)
# aerogel photons
ax.text(-150,60,'gas photons\n(θ ≈ 63 mrad, 3 bar)\nfocused to a ring of\nr ≈ 13 mm',fontsize=8.5,color=GAS)
# dims
ax.annotate('',xy=(-72,0),xytext=(-72,z_sens),arrowprops=dict(arrowstyle='<->',color=MUT))
ax.text(-75,100,'f = 203 mm ≈ gas length',rotation=90,ha='right',va='center',fontsize=9,color=MUT)
ax.set_xlim(-155,175); ax.set_ylim(-55,280); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('Phase 1, gas-only ring imaging: side view (to scale, mm)',fontsize=11,color=INK,loc='left')
# top view of sensor
p=3.2
for i in range(17):
    v=-25.6+i*p
    bx.plot([v,v],[-25.6,25.6],color='#d1d5db',lw=.5); bx.plot([-25.6,25.6],[v,v],color='#d1d5db',lw=.5)
bx.plot([0,0],[-25.6,25.6],color=MUT,lw=1.2); bx.plot([-25.6,25.6],[0,0],color=MUT,lw=1.2)
bx.add_patch(Rectangle((-25.6,-25.6),51.2,51.2,fill=False,ec=INK,lw=1.5))
c0=(f*0+xt*0,0)  # vertical track: ring centred on mirror axis
bx.add_patch(Circle(c0,12.9,fill=False,ec=GAS,lw=2)); bx.text(13,15,'gas ring\n3 bar, r = 12.9 mm',color=GAS,fontsize=8.5,bbox=dict(fc='white',ec='none',pad=1))
bx.add_patch(Circle(c0,10.5,fill=False,ec=GAS,lw=1.2,ls=':')); bx.text(-24,-23,'2 bar, r = 10.5 mm',color=GAS,fontsize=8,bbox=dict(fc='white',ec='none',pad=1))
bx.plot(xt,0,'o',color=INK,ms=5); bx.text(xt+1.5,1.5,'μ',fontsize=9)
bx.text(-38,-36,'Ring centre is set by the track angle (f·tanθ), not its position',fontsize=8,color=MUT)
bx.set_xlim(-40,40); bx.set_ylim(-40,40); bx.set_aspect('equal'); bx.axis('off')
bx.set_title('Sensor plane, top view\nassumed 2×2 × 8×8 SiPMs, 3.2 mm pitch (to be confirmed)',fontsize=10,color=INK,loc='left')
fig.text(0.60,0.33,
 'Key numbers (3 bar Xe, vertical μ)\n'
 '• Ring centre on sensor = f·tan(θ_track): moves 3.5 mm per degree\n'
 '  → sensor-limited acceptance ≈ ±3–4°, mirror-limited offset ≈ ±2 cm\n'
 '• Expected ≥80%-contained rings: ~13/h (3 bar), ~18/h (2 bar)\n'
 '• Pixel term: 3.2 mm/√12/f = 4.5 mrad per photon (dominant)\n'
 '  vs 0.8 mm pitch: 1.1 mrad; chromatic: ~0.5 mrad\n'
 '• Aerogel left out in this phase (its ring, r ≈ 52 mm, would not be\n'
 '  imaged; it is measured later in a proximity-focusing configuration)',
 fontsize=9,color=INK,va='top',family='DejaVu Sans')
plt.savefig('arc_xe_demonstrator_phase1_gas_only.png',dpi=150)
