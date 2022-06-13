import numpy as np
from scipy.constants import G
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from astropy.constants import M_sun
from astropy.cosmology import FlatLambdaCDM
from matplotlib.pyplot import figure, show

import general as ge


def red_time(z, omegaM0=0.3089, H0=67.66):
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


def time_red(t, omegaM0=0.3089, H0=67.66):
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


def astropy_time(z, H0=67.66, omegaM0=0.3089):
    """ Astropy conversion redshit to time"""
    
    cosmo = FlatLambdaCDM(H0=H0, Om0=omegaM0)           # Cosmological model
    timeCosm = cosmo.lookback_time(z).value * 1e9       # Time in years
    
    return timeCosm


def exp_mah(z, M0, alpha):
    """ Exponential mass accretion history (MAH) """
    return M0 * np.exp(-alpha * z)


def fit_parameter_mah(z, beta, gamma):
    """ Parameterized mass accretion, returns M/M0 """
    return np.power(1 + z, beta) * np.exp(-gamma * z)


def mah_bosch(fName, M0):
    """ New mass accretion history, uses generated data """
    
    with open(fName) as f:                               # Opening data
        data = np.loadtxt((x.replace('     ', '  ') for x in f))
    
    redRange = data[:,1]                                # Redshift values
    massRange = np.power(10, data[:,3]) * M0            # Mass range
    rateRange = data[:,7]                               # Mass accretion rate
    
    return redRange, massRange, rateRange


def read_zhao(fName):
    """ MAH from code of Zhao (2009) """
    
    with open(fName) as f:
        data = np.loadtxt((x.replace('   ', '  ') for x in f), skiprows=1)
    
        # Unpacking data
    redRange = data[:,0]                                # Redshift
    virM = data[:,1]                                    # Virial mass (Msun*h)
    conc = data[:,2]                                    # Concentration
    virR = data[:,4]                                    # Virial radius (Mpc*h)
    time = data[:,-1]                                   # Time (yr*h)
    
    return redRange, virM, conc, virR, time




def mean_accretion(z, M, H0=70, omegaM0=0.3):
    """ New mean accretion rate """
    
    hubble = ge.hubble_para(z, H0=H0, omegaM0=omegaM0) / 100
    
    omegaL0 = 1 - omegaM0                               # Dark energy
    sqrtPart = np.sqrt(omegaM0 * np.power(1 + z, 3) + omegaL0)  # Square root
    firstPart = 46.1 * np.power(M / (1e12), 1.1)        # First part
    meanRate = firstPart * (1 + 1.11*z) * sqrtPart      # Mean accretion rate
    
    return meanRate


def median_accretion(z, M, omegaM0=0.3):
    """ Median accretion rate in units of M_sun yr^-1, assumes Omega_r=0 """
    
    omegaL0 = 1 - omegaM0                               # Dark energy
    sqrtPart = np.sqrt(omegaM0 * np.power(1 + z, 3) + omegaL0)  # Square root
    firstPart = 25.3 * np.power(M / 1e12, 1.1)          # M0 in units of M_sun/h
    medianRate = firstPart * (1 + 1.65*z) * sqrtPart    # Median accretion rate
    
    return medianRate


def virial_mass(z, M0, omegaM0=0.3, H0=70):
    """ Mean virial mass as function of redshift """
    
    if type(z) != np.ndarray:                               # z is not an array
        if z == 0: return M0, M0                            # Current mass
        z = np.linspace(0, z, 1000)                         # Create range of z
    
    timeRange = astropy_time(z, H0=H0, omegaM0=omegaM0)     # Time range
    deltaT = timeRange[1:] - timeRange[:-1]                 # Time steps
    
    massList = [M0]                                         # List with masses
    cRate = []                                              # List with rates
    
    for ind in range(len(z)-1):
        curM = massList[ind]                                    # Current mass
        curRate = mean_accretion(z[ind], curM, omegaM0=omegaM0) # Accretion
        
        massList.append(curM - deltaT[ind]*curRate)             # New mass
        cRate.append(curRate)                                   # Current rate
    
    return np.asarray(massList), massList[-1], cRate


def virial_rad(z, M, *args):
    """ Calculate the virial radius in m, assumes you know the virial mass """
    
    virM = M * M_sun.value                              # Virial mass in kg
    
    deltaC = ge.crit_vir_dens(z, *args)                 # Critial overdensity
    hubble = ge.hubble_para(z, *args)                   # Hubble parameter
    hUnits = hubble * 1e3 / 3.0857e22                   # Units: s^-1
    
    part1 = 2 * G * virM / (deltaC * np.power(hUnits, 2))       # First part
    virR = np.power(part1, 1/3)                                 # Virial radius
    
    return virR


def virial_vel(z, M, *args):
    """ Circular speed at the virial radius, assumes you know the vir. mass """
    
    virRad = virial_rad(z, M, *args)                        # Virial radius
    virVel = np.sqrt(G * M * 2e30 / virRad)                 # Virial velocity
    
    return virVel


def main():
    """ Main function that will be executed """
    
        # Finding mass functions
    redRange = np.linspace(0, 9, 1000)                      # Redshift range
    initMass = [1e11, 1e12, 1e13, 1e14, 1e15]               # Units of solar mass
    
    logzRange = np.linspace(0, 1, 1000)                     # log10(1+z)
    zRange = np.power(10, logzRange) - 1                    # z
    
    timeRange = red_time(redRange)[::-1]
    
    tRange = astropy_time(redRange, H0=70, omegaM0=0.3)     # Correct times
    tStep = np.mean(tRange[1:] - tRange[:-1])               # Time step
    
    
        # Mass accretion
    meanChange = mean_accretion(redRange, 1e12)
    medChange = median_accretion(redRange, 1e12)
    
        # Using mass accretion
    massZ = [virial_mass(redRange, m0) for m0 in initMass]
    radZ = [virial_rad(redRange, m0) for m0 in initMass]
    velZ = [virial_vel(redRange, m0) for m0 in initMass]
    
    # Parameterized
    M0 = 10**12
#     alpha = np.log(2)
    betas = (-0.04, -0.9, 0.62, 1.42)
    gammas = (0.54, 0.35, 0.88, 1.39)
    L = range(len(betas))
    
        # Filenames for van den Bosch and Zhao data
    fName = "./getPWGH/PWGH_average_12.dat"
    fNames = [f"./getPWGH/PWGH_average_{i}.dat" for i in range(11, 16, 1)]
    fileZhao = "./mandc_m12/mandcoutput.m12"
    
        # MAH
    boschRed, boschMah, boschRate = mah_bosch(fName, M0)    # van den Bosch
    fakMass, fakFinMass, fakRateOld = virial_mass(redRange, M0) # Fak old
    fakhouriNew = mean_accretion(boschRed, boschMah)        # Fakhouri new
    
        # Mass ranges
    boschResults = [mah_bosch(fN, initMass[ind]) for ind, fN in enumerate(fNames)]
    fakOld = [virial_mass(redRange, m)[2] for m in initMass]
    fakResults = [mean_accretion(b[0], b[1]) for b in boschResults]
    
        # Find parameter alpha
    closeInd, closeVal = ge.find_closest(boschMah, M0/2)
    alpha = np.log(2) / boschRed[closeInd]
    
        # Other M(z) models
    singMah = exp_mah(redRange, M0, alpha)
    
    para, covMat = curve_fit(fit_parameter_mah, boschRed, boschMah/M0)
    optFit = fit_parameter_mah(redRange, para[0], para[1]) * M0
    
    zhao = read_zhao(fileZhao)                  # Zhao model
    zhaoRed, zhaoMass = zhao[0], zhao[1]        # Redshift & mass for Zhao
    zhaoTime = astropy_time(zhaoRed)            # Time
    
    cutInd, cutRedVal = ge.find_closest(zhao[0], max(redRange))
    
        # Finding dM/dt
    derivExp = ge.diff_func(singMah, tRange)                        # Exp.
    derivOpt = ge.diff_func(optFit, tRange)                         # 2 para.
    derivZhao = ge.diff_func(zhaoMass[:cutInd], zhaoTime[:cutInd])  # Zhao
