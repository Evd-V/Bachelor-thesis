import numpy as np
from scipy.constants import G
from scipy.integrate import solve_ivp
from matplotlib.pyplot import figure, show


import general as ge


def mass(z, M0, beta=0.10, gamma=0.69):
    """ Mass of halos as function of redshift """
    return M0 * np.power((1 + z), beta) * np.exp(-gamma * z)


def red_time(z, omegaM0=0.3, H0=70):
    """ Time for a given redshift according to Concordance model """
    
    a = 1 / (1 + z)                                 # Acceleration
    aML = np.power(omegaM0 / (1 - omegaM0), 1/3)    # Matter -> lambda
    zML = 1 / aML - 1                               # a = 1/(1+z)
    
    part1 = 2 / (3 * np.sqrt(1 - omegaM0))          # Part 1
    part2 = np.power(a / aML, 1.5)                  # Part 2
    part3 = np.sqrt(1 + np.power(a / aML, 3))       # Part 3
    
    H0units = H0 * 3600 * 24 * 365.25 * 1e3/3.0857e22   # Units: yr^-1
    time = part1 * np.log(part2 + part3) / H0units      # Time in years
    
    t0 = 13.7e9                                     # t0 in years
    
    return t0 - time


def time_red(t, omegaM0=0.3, H0=70):
    """ Find redshift for a given look-back time, uses Concordance model """
    
    H0units = H0 * 3600 * 24 * 365.25 * 1e3/3.0857e22           # Units: yr^-1
    
    omegaL0 = 1 - omegaM0                           # According to Concordance
    aML = np.power(omegaM0 / omegaL0, 1/3)          # Matter -> lambda
    
    B = H0units * t * 3 * np.sqrt(omegaL0) / 2      # Parameter
    num = aML * np.exp(-2*B/3) * np.power(np.exp(2*B) - 1, 2/3) # Numerator
    denom = np.power(2, 2/3)                                    # Denominator
    
    a = num / denom                                 # Acceleration
    z = 1 / a - 1                                   # Redshift
    
    return z


def mean_accretion(z, M0, omegaM0=0.3):
    """ Mean accretion rate in units of M_sun yr^-1, assumes Omega_r=0 """
    
    omegaL0 = 1 - omegaM0                               # Dark energy
    sqrtPart = np.sqrt(omegaM0 * np.power(1 + z, 3) + omegaL0)  # Square root
    firstPart = 42 * np.power(M0 / 1e12, 1.127)         # M0 in units of M_sun
    meanRate = firstPart * (1 + 1.17*z) * sqrtPart      # Mean accretion rate
    
    return meanRate


def median_accretion(z, M0, omegaM0=0.3):
    """ Median accretion rate in units of M_sun yr^-1, assumes Omega_r=0 """
    
    omegaL0 = 1 - omegaM0                               # Dark energy
    sqrtPart = np.sqrt(omegaM0 * np.power(1 + z, 3) + omegaL0)  # Square root
    firstPart = 24.1 * np.power(M0 / 1e12, 1.094)       # M0 in units of M_sun
    medianRate = firstPart * (1 + 1.75*z) * sqrtPart    # Median accretion rate
    
    return medianRate


def mass_redshift(z, M0, omegaM0=0.3, H0=70):
    """ Mean virial mass as function of redshift """
    
    timeRange = red_time(z, omegaM0=omegaM0, H0=H0)         # Time range
    deltaT = timeRange[1:] - timeRange[:-1]                 # Time steps
    
    massList = [M0]                                         # List with masses
    
    for ind in range(len(z)-1):
        curM = massList[ind]                                # Current mass
        curRate = mean_accretion(z[ind], curM, omegaM0=omegaM0)     # Accretion
        massList.append(curM - deltaT[ind]*curRate)         # New mass
    
    return massList


def virial_radius(z, M0, omegaM0=0.3, H0=70):
    """ Virial radius as function of z and M"""
    
    if type(z) == float or type(z) == int:              # z is not an array
        redRange = np.linspace(0, z)                    # Redshift range
        massChange = mass_redshift(redRange, M0, omegaM0=omegaM0, H0=H0)
        virMass = massChange[-1]                        # M at redshift z
    
    else:                                               # z is an array
        massChange = mass_redshift(z, M0, omegaM0=omegaM0, H0=H0)
        virMass = massChange[-1]
    
    deltaC = ge.crit_vir_dens(z, omegaM0=omegaM0)       # Critial overdensity
    hubble = ge.hubble_para(z, omegaM0=omegaM0, H0=H0)  # Hubble parameter
    
    part1 = 2 * G * virMass / (deltaC * np.power(hubble, 2))    # First part
    virRad = np.power(part1, 1/3)                               # Virial radius
    
    return virRad


# Finding mass functions
redRange = np.linspace(0, 5, 1000)                      # Redshift range
initMass = [1e10, 1e11, 1e12, 1e13, 1e14]               # Units of solar mass

masses = [mass(redRange, m0, beta=-.04, gamma=.54) for m0 in initMass]   # Mass function

# Mass accretion
meanChange = mean_accretion(redRange, 1e12)
medChange = median_accretion(redRange, 1e12)

# Using mass accretion
massZ = [mass_redshift(redRange, m0) for m0 in initMass]
radZ = [virial_radius(redRange, m0) for m0 in initMass]

# Plotting
fig = figure(figsize=(14,7))
ax = fig.add_subplot(1,1,1)

# ax.plot(redRange, massZ, label="Accretion")

# ax.plot(redRange, meanChange, label="Mean")
# ax.plot(redRange, medChange, label="Median")

for ind, R in enumerate(radZ):
    ax.plot(redRange, R, label=f"{initMass[ind]:.2e}")

ax.set_xlabel("z")
ax.set_ylabel("M")

# ax.set_xlim(0,5)
# ax.set_ylim(1e9, 5e12)

ax.legend()
ax.grid()

ax.set_yscale("log")

show()