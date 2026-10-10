"""Cosmic muons: sampling for a cos^2(zenith) intensity, surface momentum-zenith spectra, CSDA range.

Each sampler draws from the generator in a fixed order, so a given seed always
produces the same muons.
"""
import numpy as np

from .constants import COSMIC_I0, COSMIC_I0_PMIN, M_MU, MUON_THMAX


def sample_cos3(rng, N, half, z0, thmax):
    """N downward muons crossing the square |x|,|y| < half at height z0.

    The zenith angle follows cos^3 (cos^2 intensity times projected area) up to thmax,
    by accept-reject in batches of 4N. Returns directions (N,3), entry points (N,3)
    and zenith angles [deg].
    """
    out = []
    while sum(len(o) for o in out) < N:
        ct = rng.uniform(np.cos(thmax), 1, 4*N); keep = rng.uniform(0, 1, 4*N) < ct**3
        out.append(ct[keep])
    ct = np.concatenate(out)[:N]; st = np.sqrt(1-ct**2); ph = rng.uniform(0, 2*np.pi, N)
    u = np.stack([st*np.cos(ph), st*np.sin(ph), -ct], 1)
    x0 = rng.uniform(-half, half, N); y0 = rng.uniform(-half, half, N)
    return u, np.stack([x0, y0, np.full(N, z0)], 1), np.degrees(np.arccos(ct))


def redraw_xy(rng, p0, half):
    """Redraw the entry points p0 (N,3) in place, uniform over |x|,|y| < half (x first)."""
    n = len(p0)
    p0[:, 0] = rng.uniform(-half, half, n); p0[:, 1] = rng.uniform(-half, half, n)
    return p0


def sample_box(rng, N, box, thmax):
    """N muons with impact points uniform over |x|,|y| < box and directions uniform in solid angle up to thmax.

    Returns x, y, the transverse slopes sx, sy and the weight cos^3(zenith) that
    turns the uniform solid-angle sampling into the flux through a horizontal plane.
    """
    x = rng.uniform(-box, box, N); y = rng.uniform(-box, box, N)
    ct = rng.uniform(np.cos(thmax), 1, N); st = np.sqrt(1-ct**2); ph = rng.uniform(0, 2*np.pi, N)
    sx, sy = st*np.cos(ph)/ct, st*np.sin(ph)/ct
    w = ct**3
    return x, y, sx, sy, w


def flux_through_square(side_m, thmax, I0=COSMIC_I0):
    """Muon rate [Hz] through a horizontal square of side side_m [m] with zenith < thmax.

    Integral of I0 cos^2 x cos over the solid angle: I0 pi/2 A (1 - cos^4 thmax).
    """
    return I0*np.pi/2*side_m**2*(1-np.cos(thmax)**4)


# ---- Momentum and zenith spectrum of surface muons -------------------------------------------------------------
# Intensities are dI/dp dOmega [m^-2 s^-1 sr^-1 (GeV/c)^-1] at momentum p [GeV/c] and cos(zenith) c.

REYNA_C = (0.00253, 0.2455, 1.288, -0.2555, 0.0209)        # Reyna, hep-ph/0604145, best-fit c1-c5 of Eq. (3) [cm^-2 s^-1 sr^-1 GeV^-1]
GUAN_A, GUAN_GAMMA = 0.14, 2.7                             # Guan et al., arXiv:1509.06176, Eq. (3) [cm^-2 s^-1 sr^-1 GeV^-1]
GUAN_EPS_PI, GUAN_EPS_K, GUAN_RK = 115.0, 850.0, 0.054     # pion and kaon critical energies [GeV], kaon term weight
GUAN_E1, GUAN_K = 3.64, 1.29                               # low-energy modification of Eq. (3) [GeV], cos(theta*) power
CHIRKIN_P = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)   # cos(theta*) of Eq. (2) (Chirkin, hep-ph/0407078, Table 1)
SHUKLA_DURHAM = dict(I0=72.5, n=3.06, E0=3.87, eps=854.0, Ec=0.5)  # Shukla & Sankrith, arXiv:1606.06907, Table 1, Durham
SHUKLA_R_D = 174.0                                          # Earth radius / production height, ibid. Table 2

SPECTRUM_MODELS = ('guan', 'reyna', 'shukla', 'shukla-eq10')
SPECTRUM_DEFAULT = 'guan'


def intensity_reyna(p, c):
    """Reyna (hep-ph/0604145) Eqs. (2)-(3): I(p, theta) = cos^3(theta) I_v(p cos(theta)), I_v = c1 z^-(c2 + c3 x + c4 x^2 + c5 x^3), x = log10 z.

    Fitted for 1 < p < 2000/cos(theta) GeV/c; below 1 GeV/c it is an extrapolation.
    """
    c1, c2, c3, c4, c5 = REYNA_C
    z = np.asarray(p, float)*c; x = np.log10(z)
    return 1e4*c**3*c1*z**-(c2+c3*x+c4*x*x+c5*x**3)


def cos_theta_star(c):
    """cos of the zenith angle at the muon production point, Chirkin's fit (Guan et al. Eq. (2))."""
    p1, p2, p3, p4, p5 = CHIRKIN_P
    return np.sqrt((c*c+p1*p1+p2*c**p3+p4*c**p5)/(1+p1*p1+p2+p4))


def intensity_guan(p, c, m=M_MU):
    """Guan et al. (arXiv:1509.06176) Eq. (3), modified Gaisser formula, times dE/dp = p/E.

    E_mu is taken as the total energy (the paper does not specify it; the kinetic-energy reading lies 2-10 % above the
    BESS Tsukuba data at 0.8-5 GeV/c, the total-energy reading within -3.5 to +2.7 %). 1.1 E cos(theta*)/eps: Gaisser formula.
    """
    p = np.asarray(p, float); E = np.hypot(p, m); cs = cos_theta_star(c)
    dIdE = GUAN_A*(E*(1+GUAN_E1/(E*cs**GUAN_K)))**-GUAN_GAMMA*(1/(1+1.1*E*cs/GUAN_EPS_PI)+GUAN_RK/(1+1.1*E*cs/GUAN_EPS_K))
    return 1e4*dIdE*p/E


def path_ratio_shukla(c, r_d=SHUKLA_R_D):
    """Inclined over vertical path length D(theta) in a curved atmosphere, Shukla & Sankrith Eq. (7)."""
    return np.sqrt((r_d*c)**2+2*r_d+1)-r_d*c


def intensity_shukla(p, c, factorised=False, par=SHUKLA_DURHAM):
    """Shukla & Sankrith (arXiv:1606.06907), with E taken as the momentum p (their fits are to momentum spectra).

    factorised=False: energy-loss form I0 N (E0 D(theta) + p)^-n (1 + p/eps)^-1, the integrand of their derivation,
    Eqs. (8)-(9) (the vertical energy loss E0 scaled by the path ratio D; a construction of this code, not an equation of
    the paper). factorised=True: their Eq. (10), I0 N (E0 + p)^-n (1 + p/eps)^-1 D(theta)^-(n-1).
    N = (n - 1)(E0 + Ec)^(n-1) (text after Eq. (1)).
    """
    I0, n, E0, eps, Ec = (par[k] for k in ('I0', 'n', 'E0', 'eps', 'Ec'))
    p = np.asarray(p, float); D = path_ratio_shukla(c); N = (n-1)*(E0+Ec)**(n-1)
    if factorised:
        return I0*N*(E0+p)**-n/(1+p/eps)*D**-(n-1)
    return I0*N*(E0*D+p)**-n/(1+p/eps)


def intensity(p, c, model=SPECTRUM_DEFAULT, factorise=False):
    """Surface muon intensity dI/dp dOmega [m^-2 s^-1 sr^-1 (GeV/c)^-1] of the parametrisation model (SPECTRUM_MODELS).

    factorise=True replaces the zenith dependence by the vertical spectrum times cos^2(theta) (the law of the beta = 1 toys).
    """
    if factorise:
        return intensity(p, 1.0, model)*np.asarray(c, float)**2
    if model == 'guan':
        return intensity_guan(p, c)
    if model == 'reyna':
        return intensity_reyna(p, c)
    if model == 'shukla':
        return intensity_shukla(p, c)
    if model == 'shukla-eq10':
        return intensity_shukla(p, c, factorised=True)
    raise ValueError('unknown spectrum model %r (choose from %s)' % (model, ', '.join(SPECTRUM_MODELS)))


def vertical_integral(model=SPECTRUM_DEFAULT, p_lo=COSMIC_I0_PMIN, p_hi=1e4, n=20001):
    """Vertical intensity above p_lo [m^-2 s^-1 sr^-1] of a parametrisation (trapezoid rule in log p)."""
    u = np.linspace(np.log(p_lo), np.log(p_hi), n); p = np.exp(u)
    return np.trapezoid(intensity(p, 1.0, model)*p, u)


def angular_exponent(p, model=SPECTRUM_DEFAULT, thmax=MUON_THMAX, n=200):
    """Effective exponent n(p) of I(p, theta) ~ cos^n(theta): least-squares slope of ln I(p, theta)/I(p, 0) vs ln cos(theta),
    theta uniform in [0, thmax]."""
    th = np.linspace(0, thmax, n)[1:]; x = np.log(np.cos(th))
    p = np.atleast_1d(np.asarray(p, float))
    y = np.log(intensity(p[:, None], np.cos(th)[None, :], model)/intensity(p, 1.0, model)[:, None])
    return (y@x)/(x@x)


def altitude_factor(*args, **kw):
    """Correction of the sea-level spectrum to the site altitude: no primary-source parametrisation verified."""
    raise NotImplementedError('altitude correction of the sea-level parametrisations: no verified primary source; '
                              'compare with the BESS Lynn Lake (360 m) spectrum in muon_data instead')


def _range_table(material):
    from .muon_data import RANGE_TABLES
    _, rho, rows = RANGE_TABLES[material]
    t = np.array(rows)
    return t[:, 0], t[:, 1], rho


def _loglog(x, xt, yt):
    """exp of the linear interpolation of log y in log x; above the last row a power law through the last two rows."""
    lx = np.log(x); lxt = np.log(xt); lyt = np.log(yt)
    y = np.interp(lx, lxt, lyt)
    hi = lx > lxt[-1]
    if np.any(hi):
        y = np.where(hi, lyt[-1]+(lyt[-1]-lyt[-2])/(lxt[-1]-lxt[-2])*(lx-lxt[-1]), y)
    return np.exp(y)


def csda_range(p, material):
    """CSDA range [g/cm^2] of a muon of momentum p [GeV/c] in material (muon_data.RANGE_TABLES), log-log interpolation;
    above the table (100 GeV kinetic energy) a power law through its last two rows."""
    pt, Rt, _ = _range_table(material)
    return _loglog(np.asarray(p, float), pt, Rt)


def p_from_range(R, material):
    """Momentum [GeV/c] whose CSDA range in material is R [g/cm^2] (inverse of csda_range); 0 for R <= 0."""
    pt, Rt, _ = _range_table(material)
    R = np.asarray(R, float)
    p = _loglog(np.maximum(R, 1e-30), Rt, pt)
    return np.where(R > 0, p, 0.0)


def density(material):
    """Density [g/cm^3] of a material of muon_data.RANGE_TABLES."""
    return _range_table(material)[2]


class MuonSpectrum:
    """Joint momentum and zenith distribution of the muon flux through a horizontal plane, tabulated for sampling.

    The density in (ln p, cos theta) is I(p, theta) cos(theta) p (azimuth uniform), with I from intensity(model) on an
    n_p x n_c grid of cells, p_lo < p < p_hi [GeV/c], theta < thmax. norm: None keeps the parametrisation's own
    normalisation, 'I0' scales it so that its vertical intensity above COSMIC_I0_PMIN equals COSMIC_I0. overburden
    [g/cm^2] of overburden_material above the plane slows the muons by CSDA range along the slant path (stopped ones
    are removed; csda_range is a power law above its table); the tabulated p is the momentum after it. factorise: see
    intensity. flux_per_m2 is the flux [Hz/m^2] through the plane.
    """

    def __init__(self, model=SPECTRUM_DEFAULT, p_lo=0.2, p_hi=1000.0, thmax=MUON_THMAX, norm=None, overburden=0.0,
                 overburden_material='concrete', factorise=False, n_p=800, n_c=300):
        self.model = model; self.thmax = thmax; self.p_lo = p_lo; self.p_hi = p_hi
        self.scale = COSMIC_I0/vertical_integral(model) if norm == 'I0' else 1.0
        self.u_edges = np.linspace(np.log(p_lo), np.log(p_hi), n_p+1)
        self.c_edges = np.linspace(np.cos(thmax), 1.0, n_c+1)
        u = 0.5*(self.u_edges[1:]+self.u_edges[:-1]); c = 0.5*(self.c_edges[1:]+self.c_edges[:-1])
        P, C = np.meshgrid(np.exp(u), c, indexing='ij')
        if overburden > 0:
            Ps = p_from_range(csda_range(P, overburden_material)+overburden/C, overburden_material)
            h = 1e-4
            jac = (p_from_range(csda_range(P*(1+h), overburden_material)+overburden/C, overburden_material)-Ps)/(P*h)
            dens = intensity(Ps, C, model, factorise)*jac
        else:
            dens = intensity(P, C, model, factorise)
        per_sr = self.scale*dens*P*np.diff(self.u_edges)[:, None]
        cell = per_sr*C*np.diff(self.c_edges)[None, :]*2*np.pi
        self.flux_per_m2 = cell.sum()
        self.cdf = np.cumsum(cell.ravel())/self.flux_per_m2
        self.n_c = n_c
        self.intensity_c = per_sr.sum(0)
        self.cdf_p = np.cumsum(per_sr, 0)/self.intensity_c

    def sample(self, rng, n):
        """n muons: momentum [GeV/c], cos(zenith), azimuth [rad]."""
        k = np.minimum(np.searchsorted(self.cdf, rng.random(n), side='right'), len(self.cdf)-1)
        i, j = np.divmod(k, self.n_c)
        u = self.u_edges[i]+rng.random(n)*(self.u_edges[i+1]-self.u_edges[i])
        c = self.c_edges[j]+rng.random(n)*(self.c_edges[j+1]-self.c_edges[j])
        return np.exp(u), c, rng.uniform(0, 2*np.pi, n)

    def column(self, c):
        """Index of the cos(zenith) cell holding each c."""
        return np.clip(np.searchsorted(self.c_edges, c, side='right')-1, 0, self.n_c-1)

    def intensity_above(self, c):
        """Intensity integrated over p_lo < p < p_hi [m^-2 s^-1 sr^-1] at cos(zenith) c (cell value)."""
        return self.intensity_c[self.column(c)]

    def sample_p(self, rng, c):
        """Momenta [GeV/c] drawn from the spectrum at fixed cos(zenith) c (one per entry of c)."""
        j = self.column(c); r = rng.random(len(j)); i = np.empty(len(j), int)
        for jj in np.unique(j):
            m = j == jj
            i[m] = np.searchsorted(self.cdf_p[:, jj], r[m], side='right')
        i = np.minimum(i, len(self.u_edges)-2)
        return np.exp(self.u_edges[i]+rng.random(len(j))*(self.u_edges[i+1]-self.u_edges[i]))

    def weight_vs_cos2(self, c, I0=COSMIC_I0):
        """Per-muon weight turning a cos^2 law of vertical intensity I0 (cos^3 through a plane) into this spectrum."""
        c = np.asarray(c, float)
        return self.intensity_above(c)/(I0*c*c)

    def rate_through_square(self, half_mm):
        """Muon rate [Hz] through a horizontal square of half side half_mm [mm]."""
        return self.flux_per_m2*(2*half_mm*1e-3)**2
