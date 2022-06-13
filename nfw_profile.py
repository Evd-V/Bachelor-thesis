import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure, show

import general as ge
import mass_change as mc



def conc_delta_nfw(M, z):
    """ (logarithmic) concentration for Delta_c=200 """
    
    aPara = 0.520 + 0.385 * np.exp(-0.617 * np.power(z, 1.21))  # a
    bPara = -0.101 + 0.026 * z                                  # b
    logC = aPara + bPara * np.log10(M / 1e12)       # M in units h^-1 M_sun
    
    return logC


def conc_vir_nfw(z, M):
    """ Concentration for the virial mass """
    
    aPara = 0.537 + 0.488 * np.exp(-0.718 * np.power(z, 1.08))  # a
    bPara = -0.097 + 0.024 * z                                  # b
    logC = aPara + bPara * np.log10(M / 1e12)       # M in units h^-1 M_sun
    
    return np.power(10, logC)


def scale_radius_nfw(z, M0, omegaM0=0.3, H0=70):
    """ Find the scale radius at redshift z """
    
    if type(z) == np.ndarray: i = 0                         # Index for mass
    else: i = 1                                             # z = float or int
    
    virM = mc.virial_mass(z, M0, omegaM0=omegaM0, H0=H0)[i] # Virial mass at z
    virR = mc.virial_radius(z, M0, omegaM0=omegaM0, H0=H0)  # Virial radius at z
    conc = conc_vir_nfw(z, virM)                            # Concentration
    rS = virR / conc                                        # Scale radius
    
    return rS


def rho_s_z(z, M0, omegaM0=0.3, H0=70):
    """ Density at r=r_s as function of z """
    
    if type(z) == np.ndarray: i = 0                         # Index for mass
    else: i = 1                                             # z = float or int
    
    rS = scale_radius_nfw(z, M0, omegaM0=omegaM0, H0=H0)    # Scale radius
    virM = mc.virial_mass(z, M0, omegaM0=omegaM0, H0=H0)[i] # Virial mass at z
    conc = conc_vir_nfw(z, virM)                            # Concentration
    
    part1 = np.log(1 + conc) - conc / (1 + conc)            # First part
    denom = 16 * np.pi * np.power(rS, 3) * part1            # Denominator
    rhoS = virM * 2e30 / denom                              # rho_s
    
    return rhoS


def rs_nfw(z, virM, *args):
    """ New function for scale length, assumes you know the virial mass """
    
    virR = mc.virial_rad(z, virM, *args)                    # Virial radius
    conc = conc_vir_nfw(z, virM)                            # Concentration
    rS = virR / conc                                        # Scale length
    
    return rS


def rhos_nfw(z, virM, *args):
    """ New function to find rho_s, assumes you know the virial mass """
    
    rS = rs_nfw(z, virM, *args)                             # Scale length
    conc = conc_vir_nfw(z, virM)                            # Concentration
    
    part1 = np.log(1 + conc) - conc / (1 + conc)            # First part
    denom = 16 * np.pi * np.power(rS, 3) * part1            # Denominator
    rhoS = virM * 2e30 / denom                              # rho_s
    
    return rhoS


def gamma_nfw(r, rS):
    """ Logarithmic density slope of NFW profile """
    return -(1 + 3 * r/rS) / (1 + r/rS)


def NFW_profile(r, virM, z=0, *args):
    """ The full NFW profile """
    
    rS = rs_nfw(z, virM, *args)                         # Scale length
    denom = r * np.power(1+r/rS, 2) / rS                # Denominator
    
    rhoS = rhos_nfw(z, virM, *args)                     # Density at r=rS
    rho = 4 * rhoS / denom                              # Density profile
    
    return rho


def no_param_nfw(r):
    """ NFW profile as a function of r/r_s, returns rho/rho_s """
    return 4 / (r * np.power(1+r, 2))


def nfw_profile(r, z, virM, *args):
    """ Complete NFW profile as function of redshift """
    
    rhoS = rhos_nfw(z, virM,*args)                          # rho_s
    rS = rs_nfw(z, virM, *args)                             # r_s
    
    if type(z) == np.ndarray:
        c = np.zeros((len(r), len(z)))                      # Array for c
        
        for ind in range(len(r)):                           # Looping over r
            c[:,ind] = r[ind] / rS
    
    else: c = r / rS                                        # Concentration
    
    denom = c * np.power(1+c, 2)                            # Denominator
    rho = 4 * rhoS / denom                                  # Density profile
    
    return rho


def pot_nfw(r, z, virM, *args):
    """ New gravitational potential, assumes you know the virial mass """
    
    rS = rs_nfw(z, virM, *args)                             # r_s
    rhoS = rhos_nfw(z, virM, *args)                         # rho_s
    
    if type(z) == np.ndarray:                               # Array of z values
        c = np.zeros((len(r), len(z)))                      # Array for c
        
        for ind in range(len(r)):                           # Looping over r
            c[:,ind] = r[ind] / rS
    
    else: c = r / rS                                        # Single z value
    
    part1 = -16 * np.pi * G * rhoS * rS * rS                # First part
    part2 = np.log(1 + c) / c                               # Second part
    
    phi = part1 * part2                                     # Potential
    
    return phi


def find_m_z(redRange, massRange, redValue, *args):
    """ Find the virial mass for a given redshift value """
    
    interpMass = interp1d(redRange, massRange, *args)       # Interpolation
    massAtz = interpMass(redValue)                          # Mass at z
    
    return massAtz


def diff_grav_pot(r, z, virM, *args):
    """ Gravitational potential differentiated w.r.t. r """
    
    rhoS = rhos_nfw(z, M0, *args)                           # rho_s
    rS = rs_nfw(z, M0, *args)                               # r_s
    
    c = r / rS                                              # Concentration at r
    B = -16 * np.pi * G * rhoS * rS * rS                    # Constant
    part1 = c * 1 / (rS + r) - c * r * np.log(1 + c)
    
    dif = B * part1                                         # Fully diff.
    
    return dif


def mass_z_nfw(r, z, M, *args):
    """ Mass of halo (in kg) as function of r & z """
    
    zRange = np.linspace(0, z)                              # Redshift range
    rhoS = rhos_nfw(zRange, M, *args)                       # rho_s
    rS = scale_radius_nfw(zRange, M, *args)                 # r_s
    c = r / rS                                              # Concentration
    
    part1 = np.log(1 + c) - c / (1 + z)                     # First part
    massR = 16 * np.pi * rhoS * np.power(rS, 3) * part1     # Mass at r
    
    return massR


def find_vir_mass(z, virM, *args):
    """ Find the virial mass """
    
    rhoS = rhos_nfw(z, virM, *args)                         # rho_s
    rS = rs_nfw(z, virM, *args)                             # r_s
    virR = mc.virial_rad(z, virM, *args)                    # Virial radius
    
    const = 16 * np.pi * rhoS * np.power(rS, 3)             # Constant factor
    bracket = np.log(virR + rS) + rS / (virR + rS)          # Term in brackets
    
    conc = virR / rS                                        # Concentration
    bracket2 = np.log(1 + conc) - conc / (1 + conc)
    
    fullM = const * bracket2                                # Full mass
    
    return rhoS, rS, fullM


def find_conc(r, z, virM, *args):
    """ Find concentration """
    
    rhoS = rhos_nfw(z, virM,*args)                          # rho_s
    rS = rs_nfw(z, virM, *args)                             # r_s
    
    if type(z) == np.ndarray:
        c = np.zeros((len(r), len(z)))                      # Array for c
        
        for ind in range(len(r)):                           # Looping over r
            c[:,ind] = r[ind] / rS
    
    else: c = r / rS                                        # Concentration
    
    return c

def main():
    
    M0 = 1e12 / 0.6766
    fName = "./getPWGH/PWGH_average_125e12.dat"
    boschRed, boschMass, boschRate = mc.mah_bosch(fName, M0)    # van den Bosch
    
    
    rRange = np.linspace(1e19, 1e22, 1000)
    cVals = find_conc(rRange, boschRed, boschMass)
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(rRange, cVals[0], label="z=0")
    ax.plot(rRange, cVals[-1], label="z=5")
    
    ax.legend()
    ax.grid()
    
    show()

# main()
