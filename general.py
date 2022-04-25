import numpy as np



def matter_density(z, omegaM0=0.3):
    """ Calculate the matter density at redshift z """
    
    omegaL0 = 1 - omegaM0            # Assuming OmegaR0 = 0 (no radiation)
    power = np.power(1+z, 3)         # (1+z)^3, z can be an array of values
    omegaM = omegaM0 * power / (omegaM0 * power + omegaL0)  # Matter density
    
    return omegaM

def crit_vir_dens(z, omegaM0=0.3):
    """ Compute the critical overdensity for virialisation (Delta_c) """
    
    omegaM = matter_density(z, omegaM0=omegaM0)     # Matter density at z
    y = omegaM - 1                                  # Parameter
    DeltaC = 18 * np.pi * np.pi + 82*y - 39*np.power(y, 2)
    
    return DeltaC

def hubble_para(z, H0=70, omegaM0=0.3):
    """ Hubble parameter in a matter-lambda dominated Universe """
    
    omegaL0 = 1 - omegaM0                           # Dark energy
    a = 1 / (1 + z)                                 # Acceleration
    hOverH0 = omegaM0 * np.power(a, -3) + omegaL0   # (H/H0)^2
    hubble = H0 * np.sqrt(hOverH0)                  # H(z)
    
    return hubble