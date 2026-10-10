"""To-scale sketch of the mirror demonstrator: side view, sensor top view and key numbers."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .constants import AEROGEL_T, MIRROR_D, MIRROR_R, PIXEL_PITCH, SENSOR_HALF_PROTO, THC_GAS_3BAR_400, THETA_AERO_GAS
from .plotting import AEROGEL as AER, GAS, GRID, INK, LIGHT, MIRROR as MIR, MUTED as MUT, SCINT, SCINT_EDGE, plt

NOTES_AEROGEL = ('Key numbers (3 bar Xe, vertical μ)\n'
                 '• Ring centre on sensor = f·tan(θ_track): moves 3.5 mm per degree\n'
                 '  → sensor-limited acceptance ≈ ±3–4°, mirror-limited offset ≈ ±2 cm\n'
                 '• Expected ≥80%-contained rings: ~13/h (3 bar), ~18/h (2 bar)\n'
                 '• Pixel term: 3.2 mm/√12/f = 4.5 mrad per photon (dominant)\n'
                 '  vs 0.8 mm pitch: 1.1 mrad; chromatic: ~0.5 mrad\n'
                 '• Gas photons cross 1 cm aerogel before the SiPM\n'
                 '  (Rayleigh-scattering loss, roughly 25–40% at 350–400 nm)\n'
                 '• Xe fills the aerogel pores: n_aero shifts by ≈ +0.002 at 3 bar')
NOTES_GAS_ONLY = ('Key numbers (3 bar Xe, vertical μ)\n'
                  '• Ring centre on sensor = f·tan(θ_track): moves 3.5 mm per degree\n'
                  '  → sensor-limited acceptance ≈ ±3–4°, mirror-limited offset ≈ ±2 cm\n'
                  '• Expected ≥80%-contained rings: ~13/h (3 bar), ~18/h (2 bar)\n'
                  '• Pixel term: 3.2 mm/√12/f = 4.5 mrad per photon (dominant)\n'
                  '  vs 0.8 mm pitch: 1.1 mrad; chromatic: ~0.5 mrad\n'
                  '• Aerogel left out in this phase (its ring, r ≈ 52 mm, would not be\n'
                  '  imaged; it is measured later in a proximity-focusing configuration)')


def draw(aerogel=True):
    """Sketch with (aerogel=True) or without (gas-only phase 1) the aerogel tile; returns the figure."""
    R = MIRROR_R; f = R/2; a = MIRROR_D/2
    thc = THC_GAS_3BAR_400; tha = THETA_AERO_GAS
    hs = SENSOR_HALF_PROTO
    xt = 6.0                          # track offset [mm]
    z_sens = f; z_aer0 = f-AEROGEL_T
    fig = plt.figure(figsize=(12, 8.6))
    ax = fig.add_axes([0.01, 0.05, 0.56, 0.90]); bx = fig.add_axes([0.60, 0.40, 0.37, 0.52])

    def mirror_hit(p, d):
        C = np.array([0, R]); m = p-C; b = m@d; c = m@m-R*R; t = -b+np.sqrt(b*b-c)
        return p+t*d

    def trace(x0, z0, ang, col, ls='-', lw=1.2):
        d = np.array([np.sin(ang), -np.cos(ang)]); p = np.array([x0, z0])
        q = mirror_hit(p, d)
        if abs(q[0]) > a*0.95:        # misses the mirror: draw to the bottom
            t = (z0+12)/np.cos(ang); e = p+t*d
            ax.plot([p[0], e[0]], [p[1], e[1]], color=col, ls=ls, lw=lw, alpha=.9); return
        n = (q-np.array([0, R]))/R; d2 = d-2*(d@n)*n; t2 = (z_sens-q[1])/d2[1]; e = q+t2*d2
        ax.plot([p[0], q[0], e[0]], [p[1], q[1], e[1]], color=col, ls=ls, lw=lw, alpha=.9)

    # side view
    ax.add_patch(Rectangle((-62, -22), 124, 262, fill=False, ec=MUT, lw=1.4, ls=(0, (6, 3))))
    ax.text(70, 232, 'pressure vessel (dashed), Xe 2–3 bar', color=MUT, fontsize=9, va='center')
    xs = np.linspace(-a, a, 200); zs = R-np.sqrt(R*R-xs**2)
    ax.fill_between(xs, zs-5, zs, color=MIR, alpha=.85)
    ax.text(70, -4, 'spherical mirror\n76.2 mm Ø, f = 203 mm\n(Edmund #43-584)', fontsize=8.5, color=INK, va='center')
    if aerogel:
        ax.add_patch(Rectangle((-hs, z_aer0), 2*hs, AEROGEL_T, fc=AER, alpha=.25, ec=AER, lw=1))
    ax.add_patch(Rectangle((-hs, z_sens), 2*hs, 3, fc=INK, ec=INK))
    ax.add_patch(Rectangle((-30, z_sens+3), 60, 4, fc=LIGHT, ec='none'))
    ax.text(70, z_sens+1, 'SiPM plane ≈ 5.1 × 5.1 cm (focal plane)', fontsize=8.5, va='center', color=INK)
    if aerogel:
        ax.text(70, z_aer0+4, 'aerogel 5 × 5 × 1 cm, n ≈ 1.03', fontsize=8.5, va='center', color=AER)
    ax.text(70, z_sens+10, 'cooling plate / board', fontsize=8, color=MUT, va='center')
    ax.add_patch(Rectangle((-26, 252), 52, 6, fc=SCINT, alpha=.5, ec=SCINT_EDGE))
    ax.add_patch(Rectangle((-40, -40), 80, 6, fc=SCINT, alpha=.5, ec=SCINT_EDGE))
    ax.text(70, 255, 'trigger scint. ~5 × 5 cm', fontsize=8.5, va='center', color=SCINT_EDGE)
    ax.text(70, -37, 'trigger scint. ~8 × 8 cm', fontsize=8.5, va='center', color=SCINT_EDGE)
    ax.annotate('', xy=(xt, -46), xytext=(xt, 270), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2))
    ax.text(xt+2, 264, 'cosmic μ', fontsize=9, color=INK)
    for z0 in (40, 100, 160, 190):
        for s in (+1, -1):
            trace(xt, z0, s*thc, GAS)
    if aerogel:
        for s in (+1, -1):
            trace(xt, z_aer0+5, s*tha, AER, ls='--', lw=1.3)
        ax.text(-150, 150, 'aerogel photons\n(θ ≈ 250 mrad) miss\nthe 3" mirror and the\n5 cm sensor', fontsize=8.5, color=AER)
    ax.text(-150, 60, 'gas photons\n(θ ≈ 63 mrad, 3 bar)\nfocused to a ring of\nr ≈ 13 mm', fontsize=8.5, color=GAS)
    ax.annotate('', xy=(-72, 0), xytext=(-72, z_sens), arrowprops=dict(arrowstyle='<->', color=MUT))
    ax.text(-75, 100, 'f = 203 mm' if aerogel else 'f = 203 mm ≈ gas length', rotation=90, ha='right', va='center',
            fontsize=9, color=MUT)
    if aerogel:
        ax.annotate('', xy=(-52, 0), xytext=(-52, z_aer0), arrowprops=dict(arrowstyle='<->', color=MUT, lw=.8))
        ax.text(-50, 140, '~19 cm gas', rotation=90, va='center', fontsize=8, color=MUT)
    ax.set_xlim(-155, 175); ax.set_ylim(-55, 280); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Side view (to scale, mm)' if aerogel else 'Phase 1, gas-only ring imaging: side view (to scale, mm)',
                 fontsize=11, color=INK, loc='left')

    # sensor top view
    p = PIXEL_PITCH
    for i in range(17):
        v = -hs+i*p
        bx.plot([v, v], [-hs, hs], color=GRID, lw=.5); bx.plot([-hs, hs], [v, v], color=GRID, lw=.5)
    bx.plot([0, 0], [-hs, hs], color=MUT, lw=1.2); bx.plot([-hs, hs], [0, 0], color=MUT, lw=1.2)
    bx.add_patch(Rectangle((-hs, -hs), 2*hs, 2*hs, fill=False, ec=INK, lw=1.5))
    c0 = (0, 0)                       # vertical track: ring centred on the mirror axis
    bx.add_patch(Circle(c0, 12.9, fill=False, ec=GAS, lw=2))
    bx.text(13, 15, 'gas ring\n3 bar, r = 12.9 mm', color=GAS, fontsize=8.5, bbox=dict(fc='white', ec='none', pad=1))
    bx.add_patch(Circle(c0, 10.5, fill=False, ec=GAS, lw=1.2, ls=':'))
    bx.text(-24, -23, '2 bar, r = 10.5 mm', color=GAS, fontsize=8, bbox=dict(fc='white', ec='none', pad=1))
    if aerogel:
        bx.add_patch(Circle(c0, 52, fill=False, ec=AER, lw=1.5, ls='--')); bx.text(20, -45, 'aerogel ring r ≈ 52 mm', color=AER, fontsize=8.5)
    bx.plot(xt, 0, 'o', color=INK, ms=5); bx.text(xt+1.5, 1.5, 'μ', fontsize=9)
    lim, note_xy = (58, (-57, -58)) if aerogel else (40, (-38, -36))
    bx.text(*note_xy, 'Ring centre is set by the track angle (f·tanθ), not its position', fontsize=8, color=MUT)
    bx.set_xlim(-lim, lim); bx.set_ylim(-lim, lim); bx.set_aspect('equal'); bx.axis('off')
    bx.set_title('Sensor plane, top view: 2×2 matrices of 8×8 SiPMs, 3.2 mm pitch' if aerogel else
                 'Sensor plane, top view\nassumed 2×2 × 8×8 SiPMs, 3.2 mm pitch (to be confirmed)',
                 fontsize=10, color=INK, loc='left')
    fig.text(0.60, 0.33, NOTES_AEROGEL if aerogel else NOTES_GAS_ONLY, fontsize=9, color=INK, va='top', family='DejaVu Sans')
    return fig
