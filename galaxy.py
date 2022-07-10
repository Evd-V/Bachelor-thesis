import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from astropy.constants import M_sun
import astropy.units as u
from matplotlib.pyplot import figure, show
import matplotlib

import general as ge
import initial_cond as ic
import leapfrog as lf

from zhao import zhao
from bosch import bosch


class dwarf_galaxy(object):
    """ Class for intializing a dark matter halo and its properties.
        There are two models available for the evolution of the dark 
        matter halo: Zhao (2009) and van den Bosch (2014).

        Attributes:

            name:   The name of the evolution model of the dark matter 
                    halo; there are 2 options: Zhao and Bosch, corresponding 
                    to the models of Zhao (2009) and van den Bosch (2014), 
                    respectively.
            
            fName:  The name of the file containing the data of the model.

            pos:    The 3D initial position of the dwarf galaxy of interest. 
                    This should preferabaly be given in a galactocentric 
                    Cartesian coordinate system.
            
            vel:    The 3D initial velocity of the dwarf galaxy of interest. 
                    Defined in the same coordinate system as the position.
            
            t0:     The age of the Universe according to a LCDM cosmology.

            M0:     The virial mass of the dark matter halo at redshift z=0
                    in units of solar masses. Default: 1.25*10^12.

            prof:   The evolution profile of the darkr matter halo, 
                    determined by the model chosen (Zhao or Bosch).
            
            red:    The redshift at which the profile is defined.
            Mass:   The virial halo mass of the profile in solar masses.
            virR:   The virial radius of the dark matter halo in meters.
            time:   The lookback time corresponding to the redshift red in yr.
            rS:     The scale radius of the dark matter halo in meters
            rhoS:   The density at the scale radius of the dark matter halo 
                    in kg/m^3.
    
    """

    def __init__(self, name, pos, vel, fName, M0=1.25e12):
        """ Initializing the dark matter halo and the correct profile.

            Input:
                name    (string)
                pos     (numpy array)
                vel     (numpy array)
                fName   (string)
                M0      (float)
            
            Returns:
                dwarf_galaxy    (object)
        """
        
        self.name = name                        # Name of model
        self.fName = fName                      # File name
        self.pos = pos                          # 3D position
        self.vel = vel                          # 3D velocity
        
        self.t0 = 13.78e9                       # Age of Universe
        self.M0 = M0                            # Initial halo mass
        
            # Creating profile depending on model
        if name == "Bosch":                     # van den Bosch model
            self.prof = bosch(fName, M0)
        
        elif name == "Zhao":                    # Zhao model
            self.prof = zhao(fName)
        
        else: raise ValueError("Invalid model name")
        
        self.red = self.red()                   # Redshift
        self.mass = self.mass()                 # Mass (M_\odot)
        self.virR = self.virR()                 # Virial radius (m)
        self.time = self.time()                 # Lookback time (yr)
        self.rS, self.rhoS = self.rs_rhos()     # r_s and rho_s in SI units
    
    
    def norm_time(self, tRange):
        return self.t0 + ge.conv_sec_year(tRange)
    
    def red(self):
        return self.prof.red
    
    def mass(self):
        return self.prof.mass
    
    def time(self):
        return self.prof.time
    
    def rs_rhos(self):
        """ Find r_s and rho_s in SI units.

            Input:
                -

            Returns:
                scale radius in meters (numpy array).
                density at the scale radius in kg/m^3 (numpy array)
        """
    
        prof = self.prof                            # Profile
    
        if self.name == "Bosch": return prof.rS(), prof.rhoS()
        return prof.rS, prof.rhoS
    
    def virR(self):
        """ Virial radius in meters """
        if self.name == "Bosch": return self.prof.vir_rad()
        return self.prof.virR
    
    def find_rs_rhoS(self, zV):
        """ Find r_s and rho_s at a given redshift value 
        
            Input:
                zV:     Redshift at whicht r_s and rho_s are found 
                        (float)
        """
        redInd = ge.find_closest(self.red, zV)[0]
        return self.rS[redInd], self.rhoS[redInd]
    
    def correct_red(self, tRange):
        """ Find redshift corresponding to a given time range.

            Input:
                tRange: time (in years) at which the redshift has to be 
                        found (numpy array).
            
            Returns:
                interpolated redshift values (numpy array)
        """
        interTime = interp1d(self.time, self.red)
        return interTime(tRange)

    
    def diff_pot(self, p, z):
        """ Equation of motion that has to be solved to find the position 
            and velocity of dwarf galaxies in a potential of a dark matter 
            halo given by the NFW profile.

            Input:
                p:      Position vector of the dwarf galaxy (2D numpy array).
                z:      Redshift at which the equation of motion is calculated 
                        (float).
                
            Returns:
                The equation of motion (numpy array).
        """
        
        r = np.linalg.norm(p)                           # Distance
        rS, rhoS = self.find_rs_rhoS(z)                 # r_s & rho_s
        
        term1 = 16 * np.pi * G * rhoS * rS              # Term in front
        term2 = 1 + r / rS                              # Recurring term
        term3 = rS / (r * term2) - np.power(rS/r, 2) * np.log(term2)  # Brackets
        
        fRad = term1 * term3                            # Force in radial direc.
        
        return p * fRad / r
    
    
    def integ_time(self, tRange):
        """ Compute the orbit of the dwarf galaxy by integrating the second 
            order differential equation given by the equation of motion. 
            The equation of motion is found with a time dependent 
            gravitational potential.

            Input:
                tRange: Time steps at which the orbit has to be calculated,  
                        in seconds (numpy array).

            Returns:
                timePos: The 3D position of the dwarf galaxy as a function  
                         of time (2D numpy array).
                timeVel: The 3D velocity of the dwarf galaxy as a function 
                         of time (2D numpy array).
        """
        
        yearRange = ge.conv_sec_year(-tRange)   # Convert time to years
        corrRed = self.correct_red(yearRange)   # Interpolate redshift
        
            # Integrating the orbit
        timePos, timeVel = lf.time_leap(self.diff_pot, tRange, self.pos,
                                        self.vel, corrRed)
        
        return timePos, timeVel
    
    
    def time_indep(self, tRange, z=0):
        """ Compute the orbit of a dwarf galaxy using a time independent 
            gravitational potential. The equation of motion is solved, and 
            the properties (e.g. r_s and rho_s) of the dark matter halo 
            are held constant
        
            Input:
                tRange: Time steps at which the orbit has to be calculated,  
                        in seconds (numpy array).
                z:      Redshift value at which the orbit has to be 
                        computed, influences the halo parameters (mass, r_s, 
                        rho_s, ...). Default: z=0 (float).
            
            Returns:
                statPos: The 3D position of the dwarf galaxy as a function  
                         of time in a static potential (!) (2D numpy array).
                statVel: The 3D velocity of the dwarf galaxy as a function 
                         of time in a static potential (!) (2D numpy array).
        """
    
        redV = z * np.ones((len(self.red)))                 # Redshift values

            # Static potential integration
        statPos, statVel = lf.execute_leap(self.diff_pot, tRange, self.pos,
                                           self.vel, redV)
        
        return statPos, statVel
    
    
    def dist_time_vel(self, posV, velV):
        """ Find the norm of the position and velocity vectors. These vectors 
            can be in a 2D array containing a large number of vectors.

            Input:
                posV:   3D position vectors (numpy array).
                velV:   3D velocity vectors (numpy array).
            
            Returns:
                The norm of the position vectors (numpy array).
                The norm of the velocity vecotrs (numpy array).
        """
        
        pNorm = [np.linalg.norm(pos) for pos in posV]       # Norm of position
        vNorm = [np.linalg.norm(vel) for vel in velV]       # Norm of velocity
        
        return np.asarray(pNorm), np.asarray(vNorm)
    
    
    def pot_energy(self, tRange, pos):
        """ Gravitational potential energy per unit mass using the given 
            position vector(s).

            Input:
                tRange: Time steps at which the potential energy is calculated
                        in seconds (numpy array).
                pos:    Norm of the position vector(s) in meter (numpy array).
            
            Returns:
                potEn:  The gravitational potential energy per unit mass in 
                        units of J/kg (numpy array).
        """

            # Converting time range to years
        yearRange = ge.conv_sec_year(-tRange)               # Time in years
        corrRed = self.correct_red(yearRange)               # Corresponding z

        profile = self.prof                                 # Loading profile

            # Potential energy per unit mass
        potEn = [profile.pot_nfw(zV, pos[ind]) 
                   for ind, zV in enumerate(corrRed[:-1])]

        return potEn
    
    def tindep_pot(self, pos, z=0):
        """ Time independent potential energy. """
        return self.prof.pot_nfw(z, pos)
    
    def kin_energy(self, vel):
        """ Kinetic energy per unit mass. """
        return 0.5 * np.power(vel, 2)


def exp_mah(z, alpha, M0=1.25e12):
    """ Simple exponential MAH """
    return M0 * np.exp(-alpha * z)

def two_param_mah(z, beta, gamma, M0=1.25e12):
    """ Two parameter MAH """
    return M0 * np.power(1+z, beta) * np.exp(-gamma * z)


def main():
    """ Main function that will be executed """
    
        # File names for models
    pathName = "~/Documents/Evan/Studie/Year 3/Bachelor Thesis/"
    fZhao = pathName + "mandc_m125_final/mandcoutput.m125_final"
    fBosch = pathName + "getPWGH/PWGH_median.dat"    
    
        # Initial conditions Draco I dwarf
    pmRaDra = 0.039 * u.mas/u.yr                        # RA in mas/yr
    pmDecDra = -0.181 * u.mas/u.yr                      # Declination in mas/yr
    radVelDra = -292.1 * u.km/u.s                       # Radial velocity (km/s)
    dmDra = 19.57                                       # Distance modulus
    
    raDra = 260.051625 * u.deg                          # Right ascension
    decDra = (57 + 54/60 + 55/3600) * u.deg             # Declination
    
         # Coordinate systems
    helioDra = ic.obtain_pos(raDra, decDra, dmDra, pmRaDra, pmDecDra, radVelDra)
    galDra = ic.transf_frame(helioDra)                  # Galactocentric
    p0Dra, v0Dra = ic.pos_vel(galDra)                   # Position & velocity

    pSpher = (ge.conv_m_kpc(81.8), 55.3, 273.3)     # Starting pos in spherical
    vSpher = (-97.4*1e3, 136.9*1e3, -66.2*1e3)      # Startign vel in spherical

    pStart = p0Dra * 3.0857e19                      # Starting position in m
    vStart = ic.conv_vel_frame(vSpher, pSpher)      # Starting velocity in m/s

        # Time integration range
    tS = 1e-2                   # Start integration time (Gyr)
    tF = 13.2                   # End integration time (Gyr)
    t0 = 13.8                   # Age of Universe (Gyr)
    
        # Time in s
    timeRange = np.linspace(-ge.conv_year_sec(tS*1e9), -ge.conv_year_sec(tF*1e9), 
                            int(1e3))
    yearRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]       # Time in Gyr

    distRange = np.linspace(ge.conv_kpc_m(0.1), ge.conv_kpc_m(300), 1000)

        # Creating the dwarf_galaxy objects
    zhaoDwarf = dwarf_galaxy("Zhao", pStart, vStart, fZhao)     # Zhao
    boschDwarf = dwarf_galaxy("Bosch", pStart, vStart, fBosch)  # van den Bosch

    dracoZhao = dwarf_galaxy("Zhao", pStart, vStart, fZhao)
    dracoBosch = dwarf_galaxy("Bosch", pStart, vStart, fBosch)

        # Profiles
    draZProf = dracoZhao.prof
    draBProf = dracoBosch.prof

        # Dark matter profile (r/r_s vs rho/rho_s)
    xPara = distRange / dracoZhao.rS[0]
    sZhao = draZProf.simp_profile(xPara)
    sBosch = draBProf.simp_profile(xPara)

        # Redshifts
    zhaoRed, boschRed = dracoZhao.red, dracoBosch.red

    zVals = [0, 2.5, 5]
    colors = ["navy", "red", "magenta"]

    cutBInd = ge.find_closest(boschRed, 10)[0]
    boschRed = boschRed[:cutBInd+1]

    cutZInd = ge.find_closest(zhaoRed, boschRed[-1])[0]
    zhaoRed = zhaoRed[:cutZInd+1]

    zhaoConc = draZProf.conc[:cutZInd+1]
    boschConc = draBProf.conc[:cutBInd+1]

        # Virial masses and radii
    draZMass, draBMass = dracoZhao.mass[:cutZInd+1], dracoBosch.mass[:cutBInd+1]
    draZRad, draBRad = dracoZhao.virR[:cutZInd+1], dracoBosch.virR[:cutBInd+1]

    findDraZM = draZProf.virial_mass()[:cutZInd+1]
    findDraBM = draBProf.virial_mass()[:cutBInd+1]


    indAlpha = ge.find_closest(draZMass, 0.5*1.25e12)[0]
    alph = np.log(2) / zhaoRed[indAlpha]
    expMah = exp_mah(zhaoRed, alph)

    beta, gamma = 0.1, 0.69
    twoParamMah = two_param_mah(zhaoRed, beta, gamma)

        # r_s and rho_s
    draZRs, draBRs = draZProf.rS[:cutZInd+1], draBProf.rS()[:cutBInd+1]
    draZRhos, draBRhos = draZProf.rhoS[:cutZInd+1], draBProf.rhoS()[:cutBInd+1]

    denom = 16 * np.pi * np.power(draZRs, 3) * (np.log(1 + zhaoConc)
            - zhaoConc / (1 + zhaoConc))
    calcRho = draZMass * M_sun.value / denom

    calcRs = draZRad / zhaoConc
    
        # Density profile
    draZDens = draZProf.nfw_profile(zVals, distRange)
    draBDens = draBProf.nfw_profile(zVals, distRange)

        # Gravitational potential
    draZPot = draZProf.pot_nfw(zVals, distRange)
    draBPot = draBProf.pot_nfw(zVals, distRange)

    indicesZ = [ge.find_closest(zhaoRed, z)[0] for z in zVals]
    radii = [draZRad[i] for i in indicesZ]

        # Integrating the orbit
    zhaoP, zhaoV = zhaoDwarf.integ_time(timeRange)
    zhaoPos, zhaoVel = zhaoDwarf.dist_time_vel(zhaoP, zhaoV)

    boschP, boschV = boschDwarf.integ_time(timeRange)
    boschPos, boschVel = boschDwarf.dist_time_vel(boschP, boschV)

        # Time independent
    statP, statV = zhaoDwarf.time_indep(timeRange)
    statPos, statVel = zhaoDwarf.dist_time_vel(statP, statV)

    #     # MW virial radius
    # corrZVals = mc.time_red(yearRange*1e9)                  # Redshift for time

    # zhaoIntO = interp1d(zhaoDwarf.red, zhaoDwarf.virR)      # Interp for Zhao
    # boschIntO = interp1d(boschDwarf.red, boschDwarf.virR)   # Interp for Bosch

    # zhaoIntR = ge.conv_m_kpc(zhaoIntO(corrZVals))           # vir. rad Zhao
    # boschIntR = ge.conv_m_kpc(boschIntO(corrZVals))         # vir. rad Bosch

    #         # Redshift on top axis
    # redVals = np.unique(np.floor(corrZVals))                # Selecting z values
    # redInd = [ge.find_closest(corrZVals, zV)[0] for zV in redVals]
    # locs = [yearRange[ind] for ind in redInd]               # Tick locations


    #     # Finding properties of first infall
    # infalT = (3.5, 4.5)
    # inds = [ge.find_closest(yearRange, val)[0] for val in infalT][::-1]

    # cutBosch = boschPos[inds[0]:inds[1]]
    # cutZhao = zhaoPos[inds[0]:inds[1]]

    # boschInd, boschVal = ge.find_closest(boschPos, min(cutBosch))
    # zhaoInd, zhaoVal = ge.find_closest(zhaoPos, min(cutZhao))

    # print("Zhao position =", ge.conv_m_kpc(zhaoVal))
    # print("Bosch position =", ge.conv_m_kpc(boschVal))

    # print("Zhao velocity =", zhaoVel[zhaoInd])
    # print("Bosch velocity =", boschVel[boschInd])

    # print("Time Zhao =", yearRange[zhaoInd])
    # print("Time Bosch =", yearRange[boschInd])


    # Plotting
    matplotlib.rcParams['font.family'] = ['Times']

    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    # ax2 = ax.twiny()

        # DM profile (r/r_s vs rho/rho_s)
    # ax.plot(xPara, sZhao, color="navy", lw=2)

        # Masses & radii
    ax.plot(zhaoRed, draZMass, color="navy", label="Zhao", lw=2, ls="--")
    ax.plot(boschRed, draBMass, color="red", label="van den Bosch", lw=2)

    # ax.plot(zhaoRed, expMah, color="magenta", label="Exponential", lw=2, ls="-.")
    # ax.plot(zhaoRed, twoParamMah, color="lime", lw=2, ls=":", 
    #         label="Two parameter")

    # ax.plot(zhaoRed, findDraZM, color="navy", ls=":", alpha=0.7)
    # ax.plot(boschRed, findDraBM, color="red", ls=":", alpha=0.7)

    ax2.plot(zhaoRed, ge.conv_m_kpc(draZRad), color="navy", lw=2, ls="--")
    ax2.plot(boschRed, ge.conv_m_kpc(draBRad), color="red", lw=2)

        # r_s and rho_s
    # ax.plot(zhaoRed, ge.conv_m_kpc(draZRs), color="navy", lw=2, ls="--")
    # ax.plot(boschRed, ge.conv_m_kpc(draBRs), color="red", lw=2)

    # ax2.plot(zhaoRed, ge.conv_dens(draZRhos), color="navy", lw=2, ls="--")
    # ax2.plot(boschRed, ge.conv_dens(draBRhos), color="red", lw=2)

        # c and r_s vs r_delta
    # ax.plot(zhaoRed, zhaoConc, color="navy", lw=2, ls="--")
    # ax.plot(boschRed, boschConc, color="red", lw=2)

    # ax2.plot(ge.conv_m_kpc(draZRad), ge.conv_m_kpc(draZRs), color="navy", lw=2, ls="--")
    # ax2.plot(ge.conv_m_kpc(draBRad), ge.conv_m_kpc(draBRs), color="red", lw=2)

        # Density
    # for ind, dens in enumerate(draZDens):
    #     ax.plot(ge.conv_m_kpc(distRange), ge.conv_dens(dens), ls="--", lw=2, 
    #             color=colors[ind])
    #     ax.plot(ge.conv_m_kpc(distRange), ge.conv_dens(draBDens[ind]), 
    #             color=colors[ind], label=f"z = {zVals[ind]}", lw=2)

    #     ax.axvline(ge.conv_m_kpc(radii[ind]), ls=":", alpha=0.7, 
    #                color=colors[ind], lw=2)


    #     # Potential
    # for ind, pot in enumerate(draZPot):
    #     ax.plot(ge.conv_m_kpc(distRange), pot, ls="--", lw=2, 
    #             color=colors[ind])
    #     ax.plot(ge.conv_m_kpc(distRange), draBPot[ind], 
    #             color=colors[ind], label=f"z = {zVals[ind]}", lw=2)
        
    #     ax.axvline(ge.conv_m_kpc(radii[ind]), ls=":", alpha=0.7, 
    #                color=colors[ind], lw=2)

    
    ax.set_xlabel(r"$z$", fontsize=22)
    ax.set_ylabel(r"$M_\Delta$ (M$_\odot$)", fontsize=22)
    ax.tick_params(axis="both", labelsize=24)
    # ax.legend(fontsize=22, frameon=False)
    
    ax2.set_xlabel(r"z", fontsize=22)
    ax2.set_ylabel(r"$r_\Delta$ (kpc)", fontsize=22)
    ax2.tick_params(axis="both", labelsize=24)
    
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # ax.yaxis.offsetText.set_fontsize(22)
    # ax.legend(bbox_to_anchor=(1.23, 1.05), frameon=False, fontsize=22)

    # ax2.set_yscale("log")

        # r(t)
    ax.plot(yearRange[:-1], ge.conv_m_kpc(statPos[:-1]), color="teal", ls="--",
            label="Time independent")

    ax.plot(yearRange[:-1], ge.conv_m_kpc(zhaoPos[:-1]), label="Zhao (2009)",
            color="navy")

    ax.plot(yearRange[:-1], ge.conv_m_kpc(boschPos[:-1]), color="red",
            label="van den Bosch (2014)")

    # ax2.plot(yearRange, zhaoIntR, color="slateblue", ls=":")
    # ax2.plot(yearRange, boschIntR, color="tomato", ls=":")

    # ax.set_xlabel(r"$t_0 - t$ (Gyr)", fontsize=15)
    # ax.set_ylabel(r"$r$ (kpc)", fontsize=15)
    # ax.tick_params(axis="both", labelsize=15)

    # ax.grid()
    # ax.legend(loc="lower right", fontsize=15)

    # ax2.set_xticks(locs[:6], redVals[:6])
    # ax2.set_xlabel(r"$z$", fontsize=15)
    # ax2.tick_params(axis="x", labelsize=12)

    fig.suptitle("Dashed = Zhao (2009), solid = van den Bosch (2014)", fontsize=22)

    fig.tight_layout()
    # fig.savefig("virrad_virmass_slides.png")

    show()


if __name__ == "__main__":
    main()
