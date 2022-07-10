from re import A
import numpy as np
from scipy.interpolate import interp1d
from astropy.constants import kpc
import astropy.units as u
import matplotlib
from matplotlib.pyplot import figure, show

import general as ge
import initial_cond as ic
import derivatives as dv
from galaxy import dwarf_galaxy


# --------------------- #
#  Initial conditions   #
# --------------------- #

class initial_conds(object):
    """ Class containging the initial conditions for dwarf galaxies. 
        The initial conditions for each dwarf are taken from the 
        early data release 3 of the Gaia satellite (Li et al. 2021). 
        The data used here are: the proper motions in both right 
        ascension and declination, the radial velocity and the 
        distance modulus. Next to this we need the right ascension 
        and declination, these are taken from the NED database.

        The input parameter 'sig' (only available for draco and 
        sculptor) gives the ability to add or subtract 1 sigma 
        uncertainty in all the initial conditions.

        Attributes:

            draco:  The initial conditions for the Draco dwarf 
                    spheroidal galaxy.
            
            sculp:  The initial conditions for the Sculptor dwarf 
                    spheroidal galaxy.
            
            ursa:   The initial conditions for the Ursa Minor dwarf 
                    spheroidal galaxy.

            sext:   The initial conditions for the Sextans dwarf 
                    spheroidal galaxy.
            
            car:    The initial conditions for the Carina dwarf 
                    spheroidal galaxy.
    """

    def __init__(self, sig=None):
        """ Initializing the initial conditions for the dwarf galaxies 
        
            Input:
                sig     (None or string)

            Returns:
                initial_conds   (object)
        """

            # Data of galaxies
        self.draco = self.draco(sigma=sig)              # Draco I
        self.sculp = self.sculptor(sigma=sig)           # Sculptor
        self.ursa = self.ursa_minor()                   # Ursa minor
        self.sext = self.sextans()                      # Sextans I
        self.car = self.carina()                        # Carina I


    def find_cond(self, pmRa, pmDec, rVel, dm, ra, dec):
        """ Find initial position and velocity in Cartesian galactocentric
            coordinate system, from the initial conditions in an heliocentric 
            coordinate system.

            Input:
                pmRa:   The proper motion in the right ascension in mas/yr 
                        (float).
                pmDec:  The proper motion in the declination in mas/yr (float).
                rVel:   The radial velocity in km/s (float).
                dm:     The distance modulus (float).
                ra:     Right ascension coordinate in degrees (float).
                dec:    Declination in degrees (float).
            
            Returns:
                p0:     3D position of the dwarf in a galactocentric Cartesian 
                        coordinate system (numpy array).
                v0:     3D velocity of the dwarf in a galactocentric Cartesian 
                        coordinate system (numpy array).
        """

            # Coordinate systems
        helio = ic.obtain_pos(ra, dec, dm, pmRa, pmDec, rVel)
        gal = ic.transf_frame(helio)                    # Galactocentric
        p0, v0 = ic.pos_vel(gal)                        # Position & velocity

        return p0, v0


    def draco(self, sigma=None):
        """ Initial conditions for the Draco I dwarf spheroidal galaxy """

            # Data by gaia
        pmRaDra = 0.039 * u.mas/u.yr                    # RA in mas/yr
        pmDecDra = -0.181 * u.mas/u.yr                  # Declination in mas/yr
        rVelDra = -292.1 * u.km/u.s                     # Radial velocity (km/s)
        dmDra = 19.57                                   # Distance modulus

        raDra = 260.051625 * u.deg                      # Right ascension
        decDra = 57.915361 * u.deg                      # Declination

        err = [0.02 * u.mas/u.yr, 0.02*u.mas/u.yr, 0*u.km/u.s, .16]

        if sigma == "+":
            p0Dra, v0Dra = self.find_cond(pmRaDra+err[0], pmDecDra+err[1], 
                                          rVelDra+err[2], dmDra+err[3], raDra,
                                          decDra)

        elif sigma == "-":
            p0Dra, v0Dra = self.find_cond(pmRaDra-err[0], pmDecDra-err[1], 
                                          rVelDra-err[2], dmDra-err[3], raDra,
                                          decDra)                

        else:
            p0Dra, v0Dra = self.find_cond(pmRaDra, pmDecDra, rVelDra, dmDra, raDra,
                                        decDra)
        
        return p0Dra * kpc.value, v0Dra * 1e3

    def sculptor(self, sigma=None):
        """ Initial conditions for the Sculptor dwarf spheroidal galaxy """

            # Initial conditions for Sculptor dwarf
        pmRaScu = 0.096 * u.mas/u.yr                    # RA in mas/yr
        pmDecScu = -0.159 * u.mas/u.yr                  # Declination in mas/yr
        rVelScu = 111.5 * u.km/u.s                      # Radial velocity (km/s)
        dmScu = 19.67                                   # Distance modulus
        
        raScu = 15.038984 * u.deg                       # Right ascension
        decScu = -33.709029 * u.deg                     # Declination
        
        err = [0.019 * u.mas/u.yr, 0.019*u.mas/u.yr, 0*u.km/u.s, .13]

        if sigma == "+":
            p0Scu, v0Scu = self.find_cond(pmRaScu+err[0], pmDecScu+err[1], 
                                          rVelScu+err[2], dmScu+err[3], raScu,
                                          decScu)

        elif sigma == "-":
            p0Scu, v0Scu = self.find_cond(pmRaScu-err[0], pmDecScu-err[1], 
                                          rVelScu-err[2], dmScu-err[3], raScu,
                                          decScu)                

        else:
            p0Scu, v0Scu = self.find_cond(pmRaScu, pmDecScu, rVelScu, dmScu, raScu,
                                        decScu)
            
        return p0Scu * kpc.value, v0Scu * 1e3
    
    def ursa_minor(self):
        """ Initial conditiosn for the Ursa Minor dwarf spheroidal galaxy """
        
            # Initial conditions for Sextans
        pmRaUrs = -0.114 * u.mas/u.yr                   # RA in mas/yr
        pmDecUrs = 0.069 * u.mas/u.yr                   # Declination in mas/yr
        rVelUrs = -245.1 * u.km/u.s                     # Radial velocity (km/s)
        dmUrs = 19.4                                    # Distance modulus
        
        raUrs = 227.285379 * u.deg                      # Right ascension
        decUrs = 67.222605 * u.deg                      # Declination
        
        p0Urs, v0Urs = self.find_cond(pmRaUrs, pmDecUrs, rVelUrs, dmUrs, raUrs,
                                      decUrs)

        return p0Urs * kpc.value, v0Urs*1e3
    
    def carina(self):
        """ Initial conditions for the Carina dwarf spheroidal galaxy """
        
            # Initial conditions for Sextans
        pmRaCar = 0.533 * u.mas/u.yr                    # RA in mas/yr
        pmDecCar = 0.12 * u.mas/u.yr                    # Declination in mas/yr
        rVelCar = 221.8 * u.km/u.s                      # Radial velocity (km/s)
        dmCar = 20.13                                   # Distance modulus
        
        raCar = 100.402888 * u.deg                      # Right ascension
        decCar = -50.966196 * u.deg                     # Declination
        
        p0Car, v0Car = self.find_cond(pmRaCar, pmDecCar, rVelCar, dmCar, raCar,
                                      decCar)

        return p0Car * kpc.value, v0Car * 1e3
    
    
    def sextans(self):
        """ Initial conditions for the Sextans I dwarf spheroidal galaxy """
        
            # Initial conditions for Sextans
        pmRaSxt = -0.403 * u.mas/u.yr                   # RA in mas/yr
        pmDecSxt = 0.029 * u.mas/u.yr                   # Declination in mas/yr
        rVelSxt = 224.9 * u.km/u.s                      # Radial velocity (km/s)
        dmSxt = 19.89                                   # Distance modulus
        
        raSxt = 153.262319 * u.deg                      # Right ascension
        decSxt = -1.614602 * u.deg                      # Declination
        
        p0Sxt, v0Sxt = self.find_cond(pmRaSxt, pmDecSxt, rVelSxt, dmSxt, raSxt,
                                      decSxt)
                
        return p0Sxt * kpc.value, v0Sxt * 1e3


class calculate_prop(object):
    """ Class to find the position and velocity for dwarf galaxies as a function 
        of time. There are 5 dwarf galaxies can be entered: Draco, Sculptor, 
        Ursa Minor, Sextans and Carina. More can be added if their initial 
        conditions are added to the class 'initial_conds'.

        For Draco and Sculptor it is possible to select the -1 or +1 sigma 
        uncertainties in the initial conditions. The others can be added later in 
        the class 'initial_conds'.

        Attributes:

            name:       The name of the dwarf galaxy (one of the five named above).

            initCond:   The initial conditions of the dwarf galaxy specified by 
                        the name entered.
            
            zhaoGal:    A dark matter halo whose evolution is given by the model 
                        of Zhao (2009).
            
            boschGal:   A dark matter halo whose evolution is given by the model 
                        of van den Bosch (2014).
    """
    
    def __init__(self, galName, fZhao, fBosch, sig=None):
        """ Initialization the dwarf galaxy of choice and the evolution model of 
            the dark matter halo of the Milky Way.

            Input:
                galName     (string)
                fZhao       (string)
                fBosch      (string)
                sig         (None or string)
            
            Returns:
                calculate_prop  (object)
        """
        
        self.name = galName                             # Name of dwarf galaxy
        self.initCond = initial_conds(sig=sig)          # Initial conditions

        galaxies = self.dwarf_object(fZhao, fBosch)     # Initializing dwarf gal.

        self.zhaoGal = galaxies[0]                      # Zhao model
        self.boschGal = galaxies[1]                     # Bosch model
    
    def dwarf_object(self, fZhao, fBosch):
        """ Initializing the evolution models of the dark matter halo.

            Input:
                fZhao:  Filename containing the data of the model of Zhao (string).
                fBosch: Filename containing the data of the model of van den 
                        Bosch (string).
            Returns:
                zhaoGal:    Zhao model for dark matter halo of the MW 
                            (dwarf_galaxy object).
                boschGal:   van den Bosch model for dark matter halo of the MW 
                            (dwarf_galaxy object).
        """

        galData = self.load_dict()                      # Loading data
        zhaoGal = dwarf_galaxy("Zhao", galData[0], galData[1], fZhao)
        boschGal = dwarf_galaxy("Bosch", galData[0], galData[1], fBosch)

        return zhaoGal, boschGal

    def red_time(self):
        """ Retrieve redshift and time for the two dark matter halo models. 

            Input:
                -
            
            Returns:
                Redshift for model of Zhao (numpy array).
                Lookback time for model of Zhao (numpy array).
                Redshift for model of van den Bosch (numpy array).
                Lookback time for model of van den Bosch (numpy array).
        """

        zhaoGal = self.zhaoGal                      # Model of Zhao
        boschGal = self.boschGal                    # Model of van den Bosch

        return zhaoGal.red, zhaoGal.time, boschGal.red, boschGal.time


    def dict_init(self):
        """ Dictionary containing the initial conditions for the five different 
            dwarf spheroidal galaxies.

            Input:
                -
            
            Returns:
                galData:    Initial conditions for all dwarfs (dictionary).
        """
        
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
        """ Load dictionary data for correct dwarf galaxy """
        return self.dict_init()[self.name]
    
    def pos_vel(self, timeRange):
        """ Find the norm of the position and velocity vectors for the 
            dwarf as a function of time. This is done by integrating the 
            equation of motion using the two halo models.

            Input:
                timeRange:  Time at which the orbits are evaluated (numpy
                            array).
            
            Returns:
                zhaoPos:    Position for model of Zhao (numpy array).
                zhaoVel:    Velocity for model of Zhao (numpy array).
                boschPos:   Position for model of Bosch (numpy array).
                boschVel:   Velocity for model of Bosch (numpy array).
        """
                
            # Model of Zhao
        zhaoGal = self.zhaoGal                                  # Loading dm halo
        zhaoP, zhaoV = zhaoGal.integ_time(timeRange)            # Integrating orbit
        zhaoPos, zhaoVel = zhaoGal.dist_time_vel(zhaoP, zhaoV)  # Norm of vectors
        
            # Model of van den Bosch
        boschGal = self.boschGal                                # Loading dm halo
        boschP, boschV = boschGal.integ_time(timeRange)         # Integrating orbit
        boschPos, boschVel = boschGal.dist_time_vel(boschP, boschV) # Norm of vectors
        
        return zhaoPos, zhaoVel, boschPos, boschVel
    
    def orbit_tindep(self, tRange, z=0):
        """ Find the orbit of a dwarf galaxy for a time independent 
            potential. We use the model of Zhao to find the halo 
            properties; at low redshift (z < 2) the difference between 
            the models is negligible. At higher z the difference grows 
            which can potentially lead to problems. However, it is not 
            common to integrate with a static potential at high z.

            Input:
                tRange: Time steps at which orbit will be evaluated 
                        (numpy array).
                z:      Redshift at which the halo properties are taken;
                        default: z=0 (float).
            
            Returns:
                sPos:   3D position vector (2D numpy array).
                sVel:   3D velocity vector (2D numpy array).
                fullP:  Norm of position vector (numpy array).
                fullV:  Norm of velocity vector (numpy array).
        """

        tIndepGal = self.zhaoGal                            # dm halo
        sPos, sVel = tIndepGal.time_indep(tRange, z=z)      # Vectors
        fullP, fullV = tIndepGal.dist_time_vel(sPos, sVel)  # Norm

        return sPos, sVel, fullP, fullV
    
    def orbit_properties(self, fullPos):
        """ Calculate some properties of a time independent orbit, 
            these are: the pericenter and apocenter distance, and the 
            eccentricity of the orbit.

            Input:
                fullPos:    Norm of position vector as a function of 
                            time from a time independent potential 
                            model (numpy array).
            
            Returns:
                rPeri:  Distance to pericenter in meter (float).
                rApo:   Distance to apocenter in meter (float).
                ecc:    Eccentricity of orbit (float).
        """

        kpcPos = ge.conv_m_kpc(fullPos)             # Position in kpc

        rPeri = min(kpcPos)                         # Pericenter (kpc)
        rApo = max(kpcPos)                          # Apocenter (kpc)

        ecc = (rApo - rPeri) / (rApo + rPeri)       # Eccentricity

        return rPeri, rApo, ecc

    def energy(self, potNme, tRange, fullP, fullV):
        """ Calculate the specific kinetic and potential energies of 
            a time dependent orbit. Units: J/kg.

            Input:
                potNme: The name of the dark matter halo model used 
                        (Zhao or Bosch) (string).
                tRange: Time range where the energies are calculated 
                        (numpy array).
                fullP:  Norm of position vectors (numpy array).
                fullV:  Norm of velocity vectors (numpy array).

            Returns:
                eKin:   Specific kinetic energy (numpy array).
                ePot:   Specific potential energy (numpy array).
        """

        eKin = .5 * np.power(fullV, 2)            # Kinetic energy

            # Potential energy
        if potNme == "Zhao":
            ePot = self.zhaoGal.pot_energy(tRange, fullP)
        elif potNme == "Bosch":
            ePot = self.boschGal.pot_energy(tRange, fullP)
        else:
            raise ValueError("Invalid potential model name")

        return eKin, ePot
    
    def energy_tindep(self, fullP, fullV, z=0):
        """ Specific kinetic and potential energy for a time 
            independent potential dark matter halo model.

            Input:
                fullP:  Norm of position vectors (numpy array).
                fullV:  Norm of velocity vectors (numpy array).
                z:      Redshift at which the potential is taken 
                        (float).
            
            Returns:
                eKin:   Specific kinetic energy (numpy array).
                ePot:   Specific potential energy (numpy array).
        """

        eKin = .5 * np.power(fullV, 2)            # Kinetic energy
        ePot = self.zhaoGal.tindep_pot(fullP, z=z)  # Pot energy
        
        return eKin, ePot


# --------------------- #
#   Orbit integration   #
# --------------------- #

def main():
    """ Main function that will be executed. """

        # File names for models
    fZhao = "./mandc_m125_final/mandcoutput.m125_final"
    fBosch = "./getPWGH/PWGH_median.dat"
    
        # Time integration range
    timeRange = np.linspace(-ge.conv_year_sec(1e7), -ge.conv_year_sec(13.5e9), int(1e3))
    yearRange = ge.conv_sec_year(timeRange) / 1e9                   # Time in Gyr

    t0, tS, tF = 13.8, 1e-2, 13.5
    adjRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]            # Time in Gyr

    
        # Draco
    draco = calculate_prop("Draco", fZhao, fBosch)
    draZP, draZV, draBP, draBV = draco.pos_vel(timeRange)

    dracoPlus = calculate_prop("Draco", fZhao, fBosch, sig="+")
    draPZP, draPZV = dracoPlus.pos_vel(timeRange)[0:2]

    dracoMin = calculate_prop("Draco", fZhao, fBosch, sig="-")
    draMZP, draMZV = dracoMin.pos_vel(timeRange)[0:2]

    draTime = (2.9, 4.6)

    eDraZ = draco.energy("Zhao", timeRange, draZP[:-1], draZV[:-1])
    eDraB = draco.energy("Bosch", timeRange, draBP[:-1], draBV[:-1])
    
            # Time independent
    draSP, draSV = draco.orbit_tindep(timeRange, fZhao)[2:]
    eKinS, ePotS = draco.energy_tindep(draSP[:-1], draSV[:-1])
    # staticProp = draco.orbit_properties(draSP[:-1])

            # Derivatives
    histDraco, derivDraco = dv.take_deriv("Draco")
    edgesDraco = dv.stair_edges(derivDraco[0], histDraco[0])

        # Printing properties 
    # print(f"Pericenter = {staticProp[0]}")
    # print(f"Apocenter = {staticProp[1]}")
    # print(f"Eccentricity = {staticProp[2]}")

    # minInd = [ge.find_closest(adjRange, dT)[0] for dT in draTime]
    # minVal = min(draZP[minInd[1]:minInd[0]])
    # anotherInd = ge.find_closest(draZP, minVal)[0]

    # print(ge.conv_m_kpc(minVal))
    # print(adjRange[anotherInd])

        # Sculptor
    sculptor = calculate_prop("Sculptor", fZhao, fBosch)
    scuZP, scuZV, scuBP, scuBV = sculptor.pos_vel(timeRange)
    scuTime = (3.4, 6.2)

    scuSP, scuSV = sculptor.orbit_tindep(timeRange, fZhao)[2:]

    scuPlus = calculate_prop("Sculptor", fZhao, fBosch, sig="+")
    scuPZP, scuPZV = scuPlus.pos_vel(timeRange)[0:2]

    scuMin = calculate_prop("Sculptor", fZhao, fBosch, sig="-")
    scuMZP, scuMZV = scuMin.pos_vel(timeRange)[0:2]

            # Derivatives
    histScu, derivScu = dv.take_log_deriv("Sculptor")
    edgesScu = dv.stair_edges(derivScu[0], histScu[0])
    
        # Carina
    carina = calculate_prop("Carina", fZhao, fBosch)
    carZP, carZV, carBP, carBV = carina.pos_vel(timeRange)
    car1Time = (5, 6)
    car2Time = (9, 12.75)

    carSP, carSV = carina.orbit_tindep(timeRange, fZhao)[2:]

            # Derivatives
    histCar, derivCar = dv.take_log_deriv("Carina")
    edgesCar = dv.stair_edges(derivCar[0], histCar[0])
    
        # Sextans
    sextans = calculate_prop("Sextans", fZhao, fBosch)
    sxtZP, sxtZV, sxtBP, sxtBV = sextans.pos_vel(timeRange)
    
        # Ursa minor
    ursaMin = calculate_prop("Ursa minor", fZhao, fBosch)
    ursZP, ursZV, ursBP, ursBV = ursaMin.pos_vel(timeRange)
    ursTime = (2, 4.4)

    ursSP, ursSV = ursaMin.orbit_tindep(timeRange, fZhao)[2:]

            # Derivatives
    histUrsa, derivUrsa = dv.take_deriv("Ursa Minor")
    edgesUrsa = dv.stair_edges(derivUrsa[0], histUrsa[0])

    
        # MW virial radius
    initCond = initial_conds()
    draPos, draVel = initCond.draco                         # Draco

    draZhao = dwarf_galaxy("Zhao", draPos, draVel, fZhao)

    interTime = interp1d(draZhao.time, draZhao.red)
    corrZVals = interTime(-yearRange*1e9)
    
    zhaoIntO = interp1d(draZhao.red, draZhao.virR)          # Interp for Zhao
    zhaoIntR = ge.conv_m_kpc(zhaoIntO(corrZVals))

        # Redshift on top axis
    redVals = np.unique(np.floor(corrZVals))                # Selecting z values
    redInd = [ge.find_closest(corrZVals, zV)[0] for zV in redVals]
    locs = [adjRange[ind] for ind in redInd]               # Tick locations


    matplotlib.rcParams['font.family'] = ['Times']

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twiny()
    # ax3 = fig.add_subplot(2,2,1)
    # ax4 = fig.add_subplot(2,2,2)

        # Time independent
    # ax.plot(adjRange[:-1], ge.conv_m_kpc(draSP[:-1]), label="$t$ indep.", 
    #         ls="-.", lw=2, color="magenta")
    # ax3.plot(adjRange[:-1], draSV[:-1]/1e3, label="Velocity", 
    #         ls="--", lw=2, color="red")

        # Time dependent
    ax.plot(adjRange[:-1], ge.conv_m_kpc(scuMZP[:-1]), color="magenta", ls="-.", 
            label=r"$-\sigma$", lw=2, zorder=2.7)

    # ax.plot(adjRange[:-1], ge.conv_m_kpc(draBP[:-1]), label="van den Bosch (2014)",
    #         color="red", lw=2, zorder=3)
    # ax.plot(adjRange[:-1], ge.conv_m_kpc(draZP[:-1]), color="navy", ls="-", 
    #         label="Normal", lw=2, zorder=2.7)
    # ax.axvline(draTime[0], color="black", ls="-.", lw=2)
    # ax.axvline(draTime[1], color="black", ls="-.", lw=2)
    # ax.axvspan(draTime[0], draTime[1], color="lightgreen", alpha=.3)

        # Ursa Minor
    # ax.plot(adjRange[:-1], ge.conv_m_kpc(ursZP[:-1]), color="magenta", lw=2, 
    #         label="Orbit")
    # # ax.plot(adjRange[:-1], ge.conv_m_kpc(ursSP[:-1]), ls=":")
    # ax.axvline(ursTime[0], color="black", ls="-.", lw=2)
    # ax.axvline(ursTime[1], color="black", ls="-.", lw=2)
    # ax.axvspan(ursTime[0], ursTime[1], color="lightgreen", alpha=.3)

        # Sculptor
    ax.plot(adjRange[:-1], ge.conv_m_kpc(scuZP[:-1]), color="navy", lw=2, 
            label="Normal")
    # ax.plot(adjRange[:-1], ge.conv_m_kpc(scuSP[:-1]), ls=":", color="navy")
    ax.axvline(scuTime[0], color="black", ls="-.", lw=2)
    ax.axvline(scuTime[1], color="black", ls="-.", lw=2)
    ax.axvspan(scuTime[0], scuTime[1], color="lightgreen", alpha=.3)

    ax.plot(adjRange[:-1], ge.conv_m_kpc(scuPZP[:-1]), color="red", ls="--", 
            label=r"$+\sigma$", lw=2, zorder=2.7)

        # Carina
    # ax.plot(adjRange[:-1], ge.conv_m_kpc(carZP[:-1]), color="teal", lw=2, 
    #         label="Orbit", ls="-")
    # # ax.plot(adjRange[:-1], ge.conv_m_kpc(carSP[:-1]), ls=":")

    # ax.axvline(car1Time[0], color="black", ls="-.", lw=2)
    # ax.axvline(car1Time[1], color="black", ls="-.", lw=2)
    # ax.axvspan(car1Time[0], car1Time[1], color="lightgreen", alpha=.3)

    # ax.axvline(car2Time[0], color="black", ls="-.", lw=2)
    # ax.axvline(car2Time[1], color="black", ls="-.", lw=2)
    # ax.axvspan(car2Time[0], car2Time[1], color="lightgreen", alpha=.3)
    # ax.plot(adjRange[:-1], carZV[:-1], color="navy", lw=2)


        # Energies
    # ax.plot(adjRange[:-1], eKinS+ePotS, color="magenta", ls=":", lw=2)
    # ax.plot(adjRange[:-1], eKinS, color="navy", ls=":", lw=2)
    # ax.plot(adjRange[:-1], ePotS, color="red", ls=":", lw=2)
    # label=r"$\mathcal{E}$", label=r"$\mathcal{K}$" label=r"$\mathcal{U}$"

    # ax.plot(adjRange[:-1], eDraZ[0]+eDraZ[1], color="magenta", ls="--", lw=2, zorder=3)
    # ax.plot(adjRange[:-1], eDraB[0]+eDraB[1], label=r"$\mathcal{E}$", color="magenta", 
    #         lw=2, zorder=2.9)

    # ax.plot(adjRange[:-1], eDraZ[0], color="navy", ls="--", lw=2, zorder=2.8)
    # ax.plot(adjRange[:-1], eDraB[0], label=r"$\mathcal{K}$", color="navy", lw=2, 
    #         zorder=2.7)

    # ax.plot(adjRange[:-1], eDraZ[1], color="red", ls="--", lw=2, zorder=2.6)
    # ax.plot(adjRange[:-1], eDraB[1], label=r"$\mathcal{U}$", color="red", lw=2, 
    #         zorder=2.5)

        # SFH
    # ax3.stairs(histScu[1]*1e4, edgesScu, baseline=None, color="red", lw=2)
    # ax3.scatter(histScu[0], histScu[1]*1e4, color="red", marker="X", alpha=.3, s=50)

    # ax4.axhline(-12, color="k", lw=2)
    # ax4.plot(derivScu[0], derivScu[1]*1e5, color="red", marker="X", ms=7, lw=2)

    # ax2.plot(adjRange, zhaoIntR/max(zhaoIntR), color="white", alpha=0)
    ax2.plot(adjRange, zhaoIntR, color="slateblue", lw=2, ls=":", 
             label=r"$r_\Delta$")
    # ax2.plot(adjRange, tIndepR, color="slateblue", ls=":", label=r"$r_\Delta$ MW")

    ax.set_xlabel(r"$t_0 - t$ (Gyr)", fontsize=22)
    ax.set_ylabel(r"$r$ (kpc)", fontsize=22)
    ax.tick_params(axis="both", labelsize=24)
    # ax.set_title("Dashed = Zhao (2009), solid = van den Bosch (2014)",
    #              fontsize=18)
    # ax.yaxis.offsetText.set_fontsize(24)

    # ax.grid(zorder=2.1)
    # ax.legend(fontsize=15, loc="best")

    indices = (0, 1, 2, 4)
    placeLocs = [locs[i] for i in indices]
    placeRedVals = [redVals[i] for i in indices]

    ax2.set_xticks(placeLocs, placeRedVals)
    ax2.set_xlabel(r"$z$", fontsize=22)
    ax2.tick_params(axis="x", labelsize=24)

    # ax3.set_xlabel(r"$t_0 - t$ (Gyr)", fontsize=22)
    # ax3.set_ylabel(r"$\psi$ ($10^{-4}$ M$_\odot$ yr$^{-1}$)", fontsize=22)
    # ax3.tick_params(axis="both", labelsize=24)
    # ax3.yaxis.offsetText.set_fontsize(24)

    # ax4.set_xlabel(r"$t_0 - t$ (Gyr)", fontsize=22)
    # ax4.set_ylabel(r"d $\psi$ / dt ($10^{-5}$ M$_\odot$ yr$^{-1}$ Gyr$^{-1}$)", 
    #                fontsize=20)
    # ax4.tick_params(axis="both", labelsize=24)

    # ax.legend(bbox_to_anchor=(1.18, 1.05), fontsize=20, frameon=False, ncol=1)
    ax.legend(bbox_to_anchor=(.65, 1.3), fontsize=20, frameon=False, ncol=3)
    # ax3.legend(bbox_to_anchor=(0.7, 1.25), fontsize=20, frameon=False)
    ax2.legend(bbox_to_anchor=(.77, 1.3), fontsize=20, frameon=False)

    # fig.suptitle("Dashed = Zhao (2009), solid = van den Bosch (2014)", fontsize=22)
    fig.tight_layout()
    fig.savefig("Sculptor_sigma_orbit.png")

    show()


if __name__ == "__main__":
    main()
