import numpy as np
from scipy.interpolate import interp1d
from astropy.constants import M_sun, pc, kpc


def matter_density(z, omegaM0=0.3089):
    """ Calculate the matter density at redshift z, assume a LCDM 
        cosmology.

        Input:
            z:       Redshift at which the matter density is calculated 
                     (float or numpy array).
            omegaM0: Matter density at redshift z=0, default: 0.3089, 
                     according to the latest results of the Planck 
                     collaboration (float).
        
        Returns:
            omegaM:  The matter density at the given redshift(s) (float or 
                     numpy array).
    """
    
    omegaL0 = 1 - omegaM0            # Assuming OmegaR0 = 0 (no radiation)
    power = np.power(1+z, 3)         # (1+z)^3, z can be an array of values
    omegaM = omegaM0 * power / (omegaM0 * power + omegaL0)  # Matter density
    
    return omegaM

def crit_vir_dens(z, omegaM0=0.3089):
    """ Compute the critical overdensity for virialisation (Delta_c) at 
        given redshift(s).

        Input:
            z:       Redshift at which the matter density is calculated 
                     (float or numpy array).
            omegaM0: Matter density at redshift z=0, default: 0.3089, 
                     according to the latest results of the Planck 
                     collaboration (float).
        
        Returns:
            DeltaC:  The critical overdensity for virialisation at the 
                     input redshift(s) (float or numpy array).
    """
    
    omegaM = matter_density(z, omegaM0=omegaM0)     # Matter density at z
    y = omegaM - 1                                  # Parameter
    DeltaC = 18 * np.pi * np.pi + 82*y - 39*np.power(y, 2)
    
    return DeltaC

def hubble_para(z, H0=67.66, omegaM0=0.3089):
    """ Hubble parameter as function of redshift in a LCDM cosmology.

        Input:
            z:       Redshift(s) at which the matter density is calculated 
                     (float or numpy array).
            H0:      The Hubble parameter at redshift z=0, default: 67.77
                     km/s/Mpc according to the latest results of the Plank 
                     collaboration (float).
            omegaM0: Matter density at redshift z=0, default: 0.3089, 
                     according to the latest results of the Planck 
                     collaboration (float).
        
        Returns:
            hubble:  The hubble parameter at the given redshift(s) (float 
                     or numpy array).
    
    """
    
    omegaL0 = 1 - omegaM0                           # Dark energy
    a = 1 / (1 + z)                                 # Acceleration
    hOverH0 = omegaM0 * np.power(a, -3) + omegaL0   # (H/H0)^2
    hubble = H0 * np.sqrt(hOverH0)                  # H(z)
    
    return hubble

def conv_dens(dens):
    """ Convert density from kg/m^3 to M_sun/pc^3.

        Input:      dens:   density in kg/m^3 (float or numpy array).
        Returns:    density in M_sun/pc^3 (float or numpy array).
    """
    return dens / M_sun.value * np.power(pc.value, 3)


def conv_inv_dens(dens):
    """ Convert density from M_sun/pc^3 to kg/m^3 

        Input:      dens:   density in M_sun/pc^3 (float or numpy array).
        Returns:    density in kg/m^3 (float or numpy array).
    """
    return dens * M_sun.value * np.power(pc.value, -3)

def conv_year_sec(time):
    """ Convert years to seconds.

        Input:      time:   time in years (float or numpy array).
        Returns:    time in seconds (float or numpy array).
    """
    return time * 3600 * 24 * 365.25

def conv_sec_year(time):
    """ Convert seconds to years.
    
        Input:      time:   time in seconds (float or numpy array).
        Returns:    time in years (float or numpy array).
    """
    return time / (3600 * 24 * 365.25)

def conv_m_kpc(dist):
    """ Convert meters to kpc.
    
        Input:      dist:   distance in meters (float or numpy array).
        Returns:    distance in kpc (float or numpy array).
    """
    return dist / kpc.value

def conv_kpc_m(distance):
    """ Convert kpc to meters.
    
        Input:      dist:   distance in kpc (float or numpy array).
        Returns:    distance in meters (float or numpy array).
    """
    return distance * kpc.value


def diff_func(y, t):
    """ Differentiate a function f(t) w.r.t. t
    
        Input:
            y:      y values of function (numpy array).
            t:      time values at which the function is defined 
                    (numpy array).

        Returns:
            Differentiated function values (numpy array).
    """
    return (y[1:] - y[:-1]) / (t[1:] - t[:-1])


def find_closest(array, value):
    """ Find closest element to value in array.

        Input:
            array:      array with the values (numpy array).
            value:      value to which closest element in array will be 
                        determined (float or integer).
        
        Returns:
            ind:        index of closest element in array (integer).
            array[ind]: value of closest element in array (float or integer).
    """
    ind = (np.abs(array - value)).argmin()
    return ind, array[ind]


def lin_func(x):
    """ Linear function of x. """
    return x

def find_m_z(redRange, massRange, redValue):
    """ Interpolate the virial mass for a given redshift value.
    
        Input:
            redRange:   redshift range at which the mass is defined 
                        (numpy array).
            massRange:  mass values corresponding to redshifts (numpy array).
            redValue:   redshift at which mass will be interpolated (float).
        
        Returns:
            massAtz:    interpolated mass at redValue (float).
    """
    
    interpMass = interp1d(redRange, massRange)              # Interpolation
    massAtz = interpMass(redValue)                          # Mass at z
    
    return massAtz


def runga_kuta_solver(func, h, t0, y0, *args):
    """ Runge-kuta solver for first order differential equations. 

        Input:
            func:   Function that has to be integrated, with as first 
                    argument time, as second the integration variable 
                    (e.g. f(t, y)) (function).
            h:      Step size for iterative scheme (float).
            t0:     Initial time (float).
            y0:     Initial value of y at time t=i (float).
            *args:  Arguments passed to the function.
    
        Returns:
            yN:     The value of y at time t=i+1 (float).
    """
    
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
    """ Use te Runge kuta method to solve ordinary differential equations.

        Input:
            func:   Function that has to be integrated, with as first 
                    argument time, as second the integration variable 
                    (e.g. f(t, y)) (function).
            h:      Step size for iterative scheme (float).
            tLim:   start and end times of integration (tuple or list).
            y0:     Initial value of y at starting time (float).
            *args:  Arguments passed to the function.
    
        Returns:
            tRange:     The time range at which the integration is done 
                        (numpy array).
            yResults:   The solved equation values (numpy array).
    """
    
    tRange = np.arange(tLim[0], tLim[1]+h, h)           # Range of time values
    yResults = [y0]                                     # List with results
    
    for ind in range(len(tRange)-1):
        yNew = runga_kuta_solver(func, h, tRange[ind], yResults[ind], *args)
        yResults.append(yNew)
    
    return tRange, yResults
