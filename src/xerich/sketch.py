"""To-scale sketch of the mirror pre-demonstrator: side view, sensor top view and key numbers."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .cherenkov import chromatic_rms, theta_c
from .constants import (AEROGEL_HALF, AEROGEL_N_AIR, AEROGEL_T, MIRROR_D, MIRROR_R, P_REF, PIXEL_PITCH,
                        PIXEL_PITCH_FINE, SENSOR_HALF_PROTO)
from .layout import Mirror, Phase1Layout
from .media import Rayleigh, aerogel_index_air, n_xe
from .plotting import AEROGEL as AER, GAS, GRID, INK, LIGHT, MIRROR as MIR, MUTED as MUT, SCINT, SCINT_EDGE, plt
from .tracer import snell_angle

P_GAS = P_REF     # Xe pressure of the drawn cones and rings [bar]
P_ALT = 3.0       # second pressure shown as a ring [bar]
LAM = 400.0       # wavelength of the drawn angles [nm]
GAP = Phase1Layout.gap   # tile top face to the SiPM plane [mm]


def rayleigh_loss(lam_nm, t=AEROGEL_T):
    """Rayleigh-scattered fraction 1 - exp(-t/Lambda) in a slab of thickness t [mm] (media.Rayleigh mean free path)."""
    return 1-np.exp(-t/Rayleigh().length(lam_nm))


def numbers(P=P_GAS, P2=P_ALT, lam=LAM):
    """Angles [rad], ring radii and lengths [mm] and the other quoted numbers, all from the package models."""
    f = MIRROR_R/2; hs = SENSOR_HALF_PROTO; clear = Mirror(MIRROR_R, MIRROR_D).clear_radius
    n = n_xe(lam, P); n2 = n_xe(lam, P2)
    thc = float(theta_c(n)); thc2 = float(theta_c(n2))
    tha = float(snell_angle(theta_c(AEROGEL_N_AIR), AEROGEL_N_AIR, n))   # constant-index tile, no pore-gas shift
    r_gas = f*np.tan(thc); gas_len = {True: f-GAP-AEROGEL_T, False: f}   # gas below the tile (with) or the SiPM (without)
    return dict(f=f, n=n, thc=thc, thc2=thc2, tha=tha, r_gas=r_gas, r_gas2=f*np.tan(thc2), r_aero=f*np.tan(tha),
                mm_per_deg=f*np.radians(1), pix=PIXEL_PITCH/np.sqrt(12)/f, pix_fine=PIXEL_PITCH_FINE/np.sqrt(12)/f,
                chrom=chromatic_rms(lambda x: n_xe(x, P)), loss350=rayleigh_loss(350.), loss400=rayleigh_loss(400.),
                gas_len=gas_len[True], acc_deg=np.degrees(np.arctan((hs-r_gas)/f)),
                offset={k: clear-L*np.tan(thc) for k, L in gas_len.items()},
                pore_air=float(aerogel_index_air(n)-AEROGEL_N_AIR))


def notes(k, aerogel):
    """Key-numbers text of the right-hand panel."""
    t = (f'Key numbers ({P_GAS:g} bar Xe, vertical μ)\n'
         f'• Ring centre on sensor = f·tan(θ_track): moves {k["mm_per_deg"]:.1f} mm per degree\n'
         f'  → sensor-limited acceptance ≈ ±{k["acc_deg"]:.1f}° (full ring, {LAM:.0f} nm),\n'
         f'  mirror-limited offset ≈ ±{k["offset"][aerogel]/10:.1f} cm (90 % clear aperture)\n'
         f'• Pixel term: {PIXEL_PITCH:g} mm/√12/f = {1e3*k["pix"]:.1f} mrad per photon (dominant)\n'
         f'  vs {PIXEL_PITCH_FINE:g} mm pitch: {1e3*k["pix_fine"]:.1f} mrad; chromatic (FT×QE rms): {1e3*k["chrom"]:.1f} mrad')
    if aerogel:
        return t+(f'\n• Gas photons cross {AEROGEL_T/10:g} cm aerogel before the SiPM\n'
                  f'  (Rayleigh-scattering loss {100*k["loss350"]:.0f}% at 350 nm, {100*k["loss400"]:.0f}% at 400 nm)\n'
                  f'• Xe in the aerogel pores: n_aero +{k["pore_air"]:.4f} vs air at {P_GAS:g} bar,\n'
                  f'  {LAM:.0f} nm; not drawn')
    return t+(f'\n• No aerogel: its ring (r ≈ {k["r_aero"]:.0f} mm) would not be imaged\n'
              '  (aerogel rings: configurations 2–4)')


def draw(aerogel=True):
    """Sketch with (aerogel=True) or without (configuration 1, Xe gas ring) the aerogel tile; returns the figure."""
    R = MIRROR_R; f = R/2; a = MIRROR_D/2
    k = numbers(); thc = k['thc']; tha = k['tha']
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
    ax.text(70, 232, f'pressure vessel (dashed), Xe {min(P_GAS, P_ALT):g}–{max(P_GAS, P_ALT):g} bar', color=MUT, fontsize=9, va='center')
    xs = np.linspace(-a, a, 200); zs = R-np.sqrt(R*R-xs**2)
    ax.fill_between(xs, zs-5, zs, color=MIR, alpha=.85)
    ax.text(70, -4, f'spherical mirror\n{MIRROR_D:.1f} mm Ø, f = {f:.0f} mm\n(Edmund #43-584)', fontsize=8.5, color=INK, va='center')
    if aerogel:
        ax.add_patch(Rectangle((-hs, z_aer0), 2*hs, AEROGEL_T, fc=AER, alpha=.25, ec=AER, lw=1))
    ax.add_patch(Rectangle((-hs, z_sens), 2*hs, 3, fc=INK, ec=INK))
    ax.add_patch(Rectangle((-30, z_sens+3), 60, 4, fc=LIGHT, ec='none'))
    ax.text(70, z_sens+1, f'SiPM plane ≈ {2*hs/10:.1f} × {2*hs/10:.1f} cm (focal plane)', fontsize=8.5, va='center', color=INK)
    if aerogel:
        ax.text(70, z_aer0+7, f'aerogel {2*AEROGEL_HALF/10:g} × {2*AEROGEL_HALF/10:g} × {AEROGEL_T/10:g} cm, n ≈ {AEROGEL_N_AIR:g}\n{GAP:g} mm below the SiPM', fontsize=8.5, va='top', color=AER)
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
        ax.text(-150, 150, f'aerogel photons\n(θ ≈ {1e3*tha:.0f} mrad) miss\nthe {MIRROR_D/25.4:.0f}" mirror and the\n{2*hs/10:.0f} cm sensor', fontsize=8.5, color=AER)
    ax.text(-150, 60, f'gas photons\n(θ ≈ {1e3*thc:.1f} mrad, {P_GAS:g} bar)\nfocused to a ring of\nr ≈ {k["r_gas"]:.1f} mm', fontsize=8.5, color=GAS)
    ax.annotate('', xy=(-72, 0), xytext=(-72, z_sens), arrowprops=dict(arrowstyle='<->', color=MUT))
    ax.text(-75, 100, f'f = {f:.0f} mm' if aerogel else f'f = {f:.0f} mm ≈ gas length', rotation=90, ha='right', va='center',
            fontsize=9, color=MUT)
    if aerogel:
        ax.annotate('', xy=(-52, 0), xytext=(-52, z_aer0), arrowprops=dict(arrowstyle='<->', color=MUT, lw=.8))
        ax.text(-50, 140, f'~{k["gas_len"]/10:.0f} cm gas', rotation=90, va='center', fontsize=8, color=MUT)
    ax.set_xlim(-155, 175); ax.set_ylim(-55, 280); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Configuration 5, Xe gas ring with aerogel tile: side view (to scale, mm)' if aerogel else
                 'Configuration 1, Xe gas ring without aerogel: side view (to scale, mm)',
                 fontsize=11, color=INK, loc='left')

    # sensor top view
    p = PIXEL_PITCH
    for i in range(17):
        v = -hs+i*p
        bx.plot([v, v], [-hs, hs], color=GRID, lw=.5); bx.plot([-hs, hs], [v, v], color=GRID, lw=.5)
    bx.plot([0, 0], [-hs, hs], color=MUT, lw=1.2); bx.plot([-hs, hs], [0, 0], color=MUT, lw=1.2)
    bx.add_patch(Rectangle((-hs, -hs), 2*hs, 2*hs, fill=False, ec=INK, lw=1.5))
    c0 = (0, 0)                       # vertical track: ring centred on the mirror axis
    bx.add_patch(Circle(c0, k['r_gas'], fill=False, ec=GAS, lw=2))
    bx.text(13, 15, f'gas ring\n{P_GAS:g} bar, r = {k["r_gas"]:.1f} mm', color=GAS, fontsize=8.5, bbox=dict(fc='white', ec='none', pad=1))
    bx.add_patch(Circle(c0, k['r_gas2'], fill=False, ec=GAS, lw=1.2, ls=':'))
    bx.text(-24, -23, f'{P_ALT:g} bar, r = {k["r_gas2"]:.1f} mm', color=GAS, fontsize=8, bbox=dict(fc='white', ec='none', pad=1))
    if aerogel:
        bx.add_patch(Circle(c0, k['r_aero'], fill=False, ec=AER, lw=1.5, ls='--')); bx.text(20, -45, f'aerogel ring r ≈ {k["r_aero"]:.0f} mm', color=AER, fontsize=8.5)
    bx.plot(xt, 0, 'o', color=INK, ms=5); bx.text(xt+1.5, 1.5, 'μ', fontsize=9)
    lim, note_xy = (58, (-57, -58)) if aerogel else (40, (-38, -36))
    bx.text(*note_xy, 'Ring centre is set by the track angle (f·tanθ), not its position', fontsize=8, color=MUT)
    bx.set_xlim(-lim, lim); bx.set_ylim(-lim, lim); bx.set_aspect('equal'); bx.axis('off')
    bx.set_title(f'Sensor plane, top view\n2×2 arrays of 8×8 SiPMs, {PIXEL_PITCH:g} mm pitch\n(S13361-3050NE-08, to be confirmed)',
                 fontsize=10, color=INK, loc='left')
    fig.text(0.60, 0.33, notes(k, aerogel), fontsize=9, color=INK, va='top', family='DejaVu Sans')
    return fig
