import numpy as np
from astropy.constants import M_sun
from matplotlib.pyplot import figure, show


def matter_density(z, omegaM0=0.3089):
    """ Calculate the matter density at redshift z """
    
    omegaL0 = 1 - omegaM0            # Assuming OmegaR0 = 0 (no radiation)
    power = np.power(1+z, 3)         # (1+z)^3, z can be an array of values
    omegaM = omegaM0 * power / (omegaM0 * power + omegaL0)  # Matter density
    
    return omegaM

def crit_vir_dens(z, omegaM0=0.3089):
    """ Compute the critical overdensity for virialisation (Delta_c) """
    
    omegaM = matter_density(z, omegaM0=omegaM0)     # Matter density at z
    y = omegaM - 1                                  # Parameter
    DeltaC = 18 * np.pi * np.pi + 82*y - 39*np.power(y, 2)
    
    return DeltaC

def hubble_para(z, H0=67.66, omegaM0=0.3089):
    """ Hubble parameter in a matter-lambda dominated Universe """
    
    omegaL0 = 1 - omegaM0                           # Dark energy
    a = 1 / (1 + z)                                 # Acceleration
    hOverH0 = omegaM0 * np.power(a, -3) + omegaL0   # (H/H0)^2
    hubble = H0 * np.sqrt(hOverH0)                  # H(z)
    
    return hubble

def conv_dens(dens):
    """ Convert from kg/m^3 to M_sun/pc^3 """
    return dens / M_sun.value * 1 / np.power(3.2408e-17, 3)


def conv_inv_dens(dens):
    """ Convert from M_sun/pc^3 to kg/m^3 """
    return dens * M_sun.value * np.power(3.2408e-17, 3)

def conv_year_sec(time):
    """ Convert years to seconds """
    return time * 3600 * 24 * 365.25


def conv_sec_year(time):
    return time / (3600 * 24 * 365.25)


def conv_m_kpc(distance):
    """ Convert meters to kpc """
    return distance * 3.2408e-20

def conv_kpc_m(distance):
    """ Convert kpc to meters """
    return distance / 3.2408e-20


def diff_func(y, t):
    """ Differentiate a function """
    return (y[1:] - y[:-1]) / (t[1:] - t[:-1])


def find_closest(array, value):
    """ Find closest element to value in array """
    ind = (np.abs(array - value)).argmin()
    return ind, array[ind]


def lin_func(x):
    return x


def runga_kuta_solver(func, h, t0, y0, *args):
    """ Runga kuta solver for differential equations """
    
    k1 = func(t0, y0, *args)                            # First coefficient
    
    y1 = y0 + h * k1 / 2                                # New evaluation point
    k2 = func(t0 + h/2, y1, *args)                      # Second coefficient
    
    y2 = y0 + h * k2 / 2                                # New evaluation point
    k3 = func(t0 + h/2, y2, *args)                      # Third coefficient
    
    y3 = y0 + h * k3                                    # New evaluation point
    k4 = func(t0 + h, y3, *args)                        # Fourth coefficient
    
    yN = y0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6           # Next y value
    
    return yN


def execute_solver(func, h, tLim, y0, *args):
    """ Execute the Runga kuta solver """
    
    tNow = tLim[0]                                      # Lower time boundary
    
    tRange = np.arange(tLim[0], tLim[1]+h, h)           # Range of time values
    yResults = [y0]                                     # List with results
    
    for ind in range(len(tRange)-1):
        yNew = runga_kuta_solver(func, h, tRange[ind], yResults[ind], *args)
        yResults.append(yNew)
    
    return tRange, yResults
