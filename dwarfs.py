import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from matplotlib.pyplot import figure, show

import general as ge
import nfw_profile as nf
import mass_change as mc
import initial_cond as ic
from galaxy import dwarf_galaxy


# --------------------- #
#  Initial conditions   #
# --------------------- #

class initial_conds(object):
    """ Class containging the initial conditions for dwarf galaxies """

    def __init__(self):
        """ Initializing """

            # Data of galaxies
        self.draco = self.draco()                       # Draco I
        self.sculp = self.sculptor()                    # Sculptor
        self.ursa = self.ursa_minor()                   # Ursa minor
        self.sext = self.sextans()                      # Sextans I
        self.car = self.carina()                        # Carina I


    def find_cond(self, pmRa, pmDec, rVel, dm, ra, dec):
        """ Find initial position and velocity in Cartesian galactocentric
            coordinate system.
        """

            # Coordinate systems
        helio = ic.obtain_pos(ra, dec, dm, pmRa, pmDec, rVel)
        gal = ic.transf_frame(helio)                    # Galactocentric
        p0, v0 = ic.pos_vel(gal)                        # Position & velocity

        return p0, v0


    def draco(self):
        """ Initial conditions for the Draco I dwarf galaxy """

            # Data by gaia
        pmRaDra = 0.039 * u.mas/u.yr                    # RA in mas/yr
        pmDecDra = -0.181 * u.mas/u.yr                  # Declination in mas/yr
        rVelDra = -292.1 * u.km/u.s                     # Radial velocity (km/s)
        dmDra = 19.57                                   # Distance modulus

        raDra = 260.051625 * u.deg                      # Right ascension
        decDra = (57 + 54/60 + 55/3600) * u.deg         # Declination

        p0Dra, v0Dra = self.find_cond(pmRaDra, pmDecDra, rVelDra, dmDra, raDra,
                                      decDra)

        pSpher = (ge.conv_m_kpc(81.8), 55.3, 273.3)     # Starting pos in sph.
        vSpher = (-97.4*1e3, 136.9*1e3, -66.2*1e3)      # Startign vel in sph.

        pDra = p0Dra * 3.0857e19                      # Starting position in m
        vDra = ic.conv_vel_frame(vSpher, pSpher)      # Starting velocity in m/s

        return pDra, vDra

    def sculptor(self):
        """ Initial conditions for the Draco I dwarf galaxy """

            # Initial conditions for Sculptor dwarf
        pmRaScu = 0.096 * u.mas/u.yr                    # RA in mas/yr
        pmDecScu = -0.159 * u.mas/u.yr                  # Declination in mas/yr
        rVelScu = 111.5 * u.km/u.s                      # Radial velocity (km/s)
        dmScu = 19.67                                   # Distance modulus
        
        raScu = 15.038984 * u.deg                       # Right ascension
        decScu = -33.709029 * u.deg                     # Declination
        
        p0Scu, v0Scu = self.find_cond(pmRaScu, pmDecScu, rVelScu, dmScu, raScu,
                                      decScu)
        
        pScSpher = (ge.conv_m_kpc(86.0), 172.7, 62.7)   # Starting pos in sph.
        vScSpher = (75.9*1e3, 154.2*1e3, -54.3*1e3)     # Startign vel in sph.
        
        pScu = p0Scu * 3.0857e19                        # Starting position in m
        vScu = ic.conv_vel_frame(vScSpher, pScSpher)    # Starting velocity in m/s
    
        return pScu, vScu
    
    def ursa_minor(self):
        """ Initial conditiosn for the Ursa minor dwarf galaxy """
        
            # Initial conditions for Sextans
        pmRaUrs = -0.114 * u.mas/u.yr                   # RA in mas/yr
        pmDecUrs = 0.069 * u.mas/u.yr                   # Declination in mas/yr
        rVelUrs = -245.1 * u.km/u.s                     # Radial velocity (km/s)
        dmUrs = 19.4                                    # Distance modulus
        
        raUrs = 227.285379 * u.deg                      # Right ascension
        decUrs = 67.222605 * u.deg                      # Declination
        
        p0Urs, v0Urs = self.find_cond(pmRaUrs, pmDecUrs, rVelUrs, dmUrs, raUrs,
                                      decUrs)
        
        pUrSpher = (ge.conv_kpc_m(77.8), 46.5, 293.0)   # Starting pos in sph.
        vUrSpher = (-76.9*1e3, 143.9*1e3, -31.9*1e3)    # Startign vel in sph.
        
        pUrs = p0Urs * 3.0857e19                        # Starting pos in m
        vUrs = ic.conv_vel_frame(vUrSpher, pUrSpher)    # Starting vel in m/s
        
        return pUrs, vUrs
    
    def carina(self):
        """ Initial conditions for the Carina dwarf galaxy """
        
            # Initial conditions for Sextans
        pmRaCar = 0.533 * u.mas/u.yr                    # RA in mas/yr
        pmDecCar = 0.12 * u.mas/u.yr                    # Declination in mas/yr
        rVelCar = 221.8 * u.km/u.s                      # Radial velocity (km/s)
        dmCar = 20.13                                   # Distance modulus
        
        raCar = 100.402888 * u.deg                      # Right ascension
        decCar = -50.966196 * u.deg                     # Declination
        
        p0Car, v0Car = self.find_cond(pmRaCar, pmDecCar, rVelCar, dmCar, raCar,
                                      decCar)
        
        pCaSpher = (ge.conv_kpc_m(107.6), 111.9, 75.5)   # Starting pos in sph.
        vCaSpher = (0.8*1e3, -191.6*1e3, -8.6*1e3)     # Startign vel in sph.
        
        pCar = p0Car * 3.0857e19                        # Starting pos in m
        vCar = ic.conv_vel_frame(vCaSpher, pCaSpher)    # Starting vel in m/s
        
        return pCar, vCar
    
    
    def sextans(self):
        """ Initial conditions for the Sextans I dwarf galaxy """
        
            # Initial conditions for Sextans
        pmRaSxt = -0.403 * u.mas/u.yr                    # RA in mas/yr
        pmDecSxt = 0.029 * u.mas/u.yr                  # Declination in mas/yr
        rVelSxt = 224.9 * u.km/u.s                      # Radial velocity (km/s)
        dmSxt = 19.89                                   # Distance modulus
        
        raSxt = 153.262319 * u.deg                       # Right ascension
        decSxt = -1.614602 * u.deg                     # Declination
        
        p0Sxt, v0Sxt = self.find_cond(pmRaSxt, pmDecSxt, rVelSxt, dmSxt, raSxt,
                                      decSxt)
        
        pSxSpher = (ge.conv_kpc_m(98.1), 49.3, 57.9)   # Starting pos in sph.
        vSxSpher = (83.3*1e3, -8.8*1e3, -219.7*1e3)     # Startign vel in sph.
        
        pSxt = p0Sxt * 3.0857e19                        # Starting pos in m
        vSxt = ic.conv_vel_frame(vSxSpher, pSxSpher)    # Starting vel in m/s
        
        return pSxt, vSxt


class calculate_prop(object):
    """ Class to find the position and velocity for dwarf galaxies """
    
    def __init__(self, galName):
        """ Initialization """
        
        self.name = galName                             # Name of dwarf galaxy
        self.initCond = initial_conds()                 # Initial conditions
    
    
    def dict_init(self):
        """ Dictionary containing names corresponding to initial conditions"""
        
        init = self.initCond                            # Loading inital cond.
        
            # Dictionary containing the data
        galData = {
                   "Draco": init.draco,
                   "Sculptor": init.sculp,
                   "Ursa minor": init.ursa,
                   "Sextans": init.sext,
                   "Carina": init.car
                  }
        
        return galData
    
    def load_dict(self):
        """ Load correct dictionary data """
        return self.dict_init()[self.name]
    
    def pos_vel(self, timeRange, fZhao, fBosch):
        """ Find full position and velocity for dwarf """
        
        galData = self.load_dict()
        
        zhaoGal = dwarf_galaxy("Zhao", galData[0], galData[1], fZhao)
        zhaoP, zhaoV = zhaoGal.integ_time(timeRange)
        zhaoPos, zhaoVel = zhaoGal.dist_time_vel(zhaoP, zhaoV)
        
        boschGal = dwarf_galaxy("Bosch", galData[0], galData[1], fBosch)
        boschP, boschV = boschGal.integ_time(timeRange)
        boschPos, boschVel = boschGal.dist_time_vel(boschP, boschV)
        
        return zhaoPos, zhaoVel, boschPos, boschVel
