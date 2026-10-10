import numpy as np
R=8.314462618; T0=273.15; P0=101325.
def B_virial(T):
    # Xe second virial coefficient [m^3/mol], approx (Dymond & Smith): ~ -130 cm3/mol at 293 K
    return (-130.0 + 0.9*(T-293.15))*1e-6
def molar_density(P,T):
    # solve P = rho R T (1 + B rho)
    B=B_virial(T); a=B*R*T; b=R*T
    return (-b+np.sqrt(b*b+4*a*P))/(2*a) if a!=0 else P/b
rho0 = P0/(R*T0)*(1/(1+B_virial(T0)*P0/(R*T0)))  # reference density of Bideau-Mehu (0C, 1 atm)
def nm1_ref(lam_nm):
    l=lam_nm*1e-3; x=l**-2
    return 0.00322869/(46.301-x)+0.00355393/(59.578-x)+0.0606764/(112.74-x)
def n_of(lam_nm, P_bar, T=293.15):
    # Lorentz-Lorenz scaling with density
    n0=1+nm1_ref(lam_nm); LL0=(n0**2-1)/(n0**2+2)
    LL=LL0*molar_density(P_bar*1e5,T)/rho0
    return np.sqrt((1+2*LL)/(1-LL))
ALPHA=1/137.035999; HC=1239.84198  # eV nm
def dN_dxdE(E_eV, p, m, P_bar, T=293.15):
    lam=HC/E_eV; n=n_of(lam,P_bar,T); beta=p/np.hypot(p,m)
    s2=1-1/(beta*n)**2
    return np.where(s2>0, ALPHA/(1.97326980e-5)*s2, 0.0)  # per eV per cm (hbar c = 1.973e-5 eV cm)
