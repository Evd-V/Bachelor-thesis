import numpy as np
from scipy.constants import G

import general as ge

def vir_radius(z, virM, H0=70, omegaM0=0.3):
    """ Virial radius """
    
    hubble = ge.hubble_para(z, H0, omegaM0)         # H(z)
    denom = ge.crit_vir_dens(z) * hubble * hubble   # Denominator
    virRad3 = 2 * G * virM / denom                  # virial radius ^3
    
    return np.power(virRad3, 1/3)

def concentration(z, rS, virM):
    """ Concentration at redshift z; rS=scale radius """
    return vir_radius(z, virM) / rS

def conc_nfw(M, z):
    """ (logarithmic) concentration for Delta_c=200 """
    
    aPara = 0.520 + 0.385 * np.exp(-0.617 * np.power(z, 1.21))  # a
    bPara = -0.101 + 0.026 * z                                  # b
    logC = aPara + bPara * np.log10(M / 1e12)       # M in units h^-1 M_sun
    
    return logC

def rho_s(z, rS, virM):
    """ Density at the scale radius """
    
    conc = concentration(z, rS, virM)               # Concentration
    denom = 16 * np.pi * rS**3 * (np.log(1 + conc) - conc/(1+conc))
    rhoS = virM / denom                             # Density at r=rS
    
    return rhoS

def grav_pot(r, z, rS, virM):
    """ The gravitational potential for the NFW profile """
    
    rhoS = rho_s(z, rS, virM)                       # Density at r=rS
    phi = -16 * np.pi * G * rhoS * rS * np.log(1 + r/rS) / (r/rS)   # Potential
    
    return phi

def gamma_nfw(r, rS):
    """ Logarithmic density slope of NFW profile """
    return -(1 + 3 * r/rS) / (1 + r/rS)

def NFW_profile(r, rS, virM, z=0):
    """ Simple form of NFW profile """
    
    denom = r * np.power(1+r/rS, 2) / rS            # Denominator of profile
    rhoS = rho_s(z, rS, virM)                       # Density at r=rS
    rho = 4 * rhoS / denom                          # Density profile
    
    return rho

def test_nfw(r):
    """ NFW profile as a function of r/r_s, returns rho/rho_s """
    return 4 / (r * np.power(1+r, 2))

def new_nfw_profile(M, z):
    """ New and updated NFW profile """
    
    conc = conc_nfw(M, z)                           # Concentration
    rho = 1 / (conc * np.power((1 + conc), 2))      # rho/rho_s
    
    return rho

