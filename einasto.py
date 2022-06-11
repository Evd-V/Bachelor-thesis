import numpy as np
from scipy.constants import G
from scipy.integrate import solve_ivp, odeint
from scipy.special import gamma, gammainc

import nfw_profile as nf


def einasto_profile(r, rS, rhoS, alpha):
    """ Density according to Einasto profile """
    
    part1 = -2 * (np.power(r/rS, alpha) - 1) / alpha    # First part
    rho = rhoS * np.exp(part1)                          # Density profile
    
    return rho

def gamma_einasto(r, rS, alpha):
    """ Logarithmic density slope of Einasto profile """
    return -2 * np.power(r/rs, alpha)

def mass_einasto(rS, rhoS, alpha):
    """ Mass profile of an Einasto profile """
    
    eta = np.power(2/alpha, 1/alpha) * r/rS             # Parameter
    gamm = gamma(3/alpha) - gammainc(3/alpha, np.power(eta, alpha)) # Gamma func
    part1 = np.power(alpha/2, 3/alpha) * np.exp(2/alpha)# First part
    part2 = 4 * np.pi * rhoS * np.power(rS, 3) / alpha  # Second part
    massR = part1 * part2 * gam                         # Mass profile
    
    return massR

def rho_s_einasto(r, rS, alpha, virM):
    """ Density at the scale radius: r=rS, virial mass is required """
    
    eta = np.power(2/alpha, 1/alpha) * r/rS                     # Parameter
    gam = gamma(3/alpha) - gammainc(3/alpha, np.power(eta, alpha)) # Gamma func
    part1 = alpha * np.power(2/alpha, alpha/3) / (4 * np.pi)    # First part
    part2 = np.exp(-2/alpha) * np.power(rS, 3)                  # Second part
    rhoS = part1 * part2 / gam                                  # rhoS
    
    return rhoS

def einasto_virial(r, rS, alpha, virM):
    """ Einasto profile using the virial mass """
    
    rhoS = rho_s_einasto(r, rS, alpha, virM)            # rho at r=rS
    rho = einasto_profile(r, rS, rhoS, alpha)           # Density profile
    
    return rho


def vel_disp(M, z):
    """ Velocity dispersion for dark matter, M is the mass for Delta_c=200 in 
        units of solar masses. For values of fitting parameters, see Ervard 
        et al. (2008).
    """
    
    alpha = 0.3361                                      # Slope, fitting para
    sigNorm = 1082.9                                    # normalization, km s^-1
    h = nf.hubble_para(z) / 100                         # Hubble's constant at z
    
    # Testing, add h or not
    velDisp = sigNorm * np.power(M / 1e15, alpha)       # Velocity dispersion
    
    return velDisp

def crit_delta(z):
    """ The linear density threshold for collapse at redshift z """
    return 2.18     # Change later, make z dependent

def peak_height(M, z):
    """ Dimensionless peak-height parameter, nu """
    return crit_delta(z) / vel_disp(M, z)

def peak_new(M, z):
    """ Numerical approximation for peak-height, see Dutton """
    
    m = np.log10(M / 1e12)                   # M in units of h^-1 M_sun
    
        # log10 of nu(M, 0)
    nu0 = -0.11 + 0.146 * m + 0.0138 * np.power(m, 2) * 0.00123 * np.power(m, 3)
        # Ratio: nu(M, z)/nu(M, 0)
    ratio = 0.033 + 0.79 * (1 + z) + 0.176 * np.exp(-1.356 * z)
    
    nuFull = np.power(10, nu0) * ratio       # Peak-height
    
    return nuFull

def alpha_einasto(M, z):
    """ Best fit alpha value for the Einasto profile """
    return 0.0095 * np.power(peak_new(M, z), 2) + 0.155

def conc_einasto(M, z):
    """ Concentration for the Einasto profile. Returns logarithm base 10 """
    
    aPara = 0.459 + (0.997-0.459) * np.exp(-0.49 * np.power(z, 1.303))  # a
    bPara = -0.130 + 0.029 * z                                          # b
    
    logC = aPara + bPara * np.log10(M / 1e12)       # M in units of h^-1 M_sun
    
    return logC

def new_einasto_profile(M, z):
    """ New improved Einasto density profile """
    
    alpha = alpha_einasto(M, z)                     # Alpha parameter
    conc = conc_einasto(M, z)                       # Concentration (=r/r_s)
    rho = np.exp(-2 * (np.power(conc, alpha) - 1) / alpha)  # rho/rho_s
    
    return rho

def test_einasto(r, M, z=0):
    """ Einasto profile as function of r/r_s, returns rho/rho_s """
    
    alpha = alpha_einasto(M, z)                     # Alpha
    logRho = - 2 * (np.power(r, alpha) - 1) / alpha # Logarithm of rho
    
    return np.exp(logRho)


def poisson_einasto(r, phi, rS, alpha, virM):
    """ Poisson equation for the Einasto profile """
    return 4 * np.pi * G * einasto_virial(r, rS, alpha, virM)


def poisson(phi, r):
    return (phi[1], 4 * np.pi * G * einasto_virial(r, 1, 0.2, 1e12))

def solv_poisson(phiS, rRange):
    solved = odeint(poisson, phiS, rRange)
    return solved[:,0]


