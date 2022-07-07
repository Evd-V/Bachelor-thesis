import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from astropy.constants import M_sun
import astropy.units as u
from matplotlib.pyplot import figure, show
import matplotlib

import general as ge
import mass_change as mc
import initial_cond as ic
import leapfrog as lf

from zhao import zhao
from bosch import bosch


class dwarf_galaxy(object):
    """ Dwarf galaxy object """

    def __init__(self, name, pos, vel, fName, M0=1.25e12):
        """ Initializing """
        
        self.name = name                        # Name of model
        self.fName = fName                      # File name
        self.pos = pos                          # 3D position
        self.vel = vel                          # 3D velocity
        
        self.t0 = 13.78e9                        # Age of Universe
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
        """ Find r_s and rho_s in SI units """
    
        prof = self.prof                            # Profile
    
        if self.name == "Bosch": return prof.rS(), prof.rhoS()
        return prof.rS, prof.rhoS
    
    def virR(self):
        """ Virial radius in meters """
        if self.name == "Bosch": return self.prof.vir_rad()
        return self.prof.virR
    
    def find_rs_rhoS(self, zV):
        """ Find correct rS """
        redInd = ge.find_closest(self.red, zV)[0]
        return self.rS[redInd], self.rhoS[redInd]
    
    def correct_red(self, tRange):
        """ Find corrected redshift """
        interTime = interp1d(self.time, self.red)
        return interTime(tRange)

    
    def diff_pot(self, p, z, M, *args):
        """ Differential equation """
        
        r = np.linalg.norm(p)                           # Distance
        rS, rhoS = self.find_rs_rhoS(z)                 # r_s & rho_s
        
        term1 = 16 * np.pi * G * rhoS * rS              # Term in front
        term2 = 1 + r / rS                              # Recurring term
        term3 = rS / (r * term2) - np.power(rS/r, 2) * np.log(term2)  # Brackets
        
        fRad = term1 * term3                            # Force in radial direc.
        
        return p * fRad / r
    
    
    def integ_time(self, tRange, *args):
        """ Integrate the orbit of the dwarf galaxy """
        
            # Converting time range to years
        yearRange = ge.conv_sec_year(-tRange)
        
            # Interpolate mass and redshift values
        # interTime = interp1d(self.time, self.red)
        # corrRed = interTime(yearRange)

        corrRed = self.correct_red(yearRange)
        
        interMass = interp1d(self.red, self.mass)
        corrMass = interMass(corrRed)
        
            # Integrating the orbit
        timePos, timeVel = lf.time_leap(self.diff_pot, tRange, self.pos,
                                        self.vel, corrRed, corrMass, *args)
        
        return timePos, timeVel
    
    
    def time_indep(self, tRange, z=0, *args):
        """ Integrate orbit with time independent potential """
    
        redV = z * np.ones((len(self.red)))                 # Redshift range
        zInd, zVal = ge.find_closest(self.red, z)
        massV = self.mass[zInd] * np.ones((len(self.red)))  # Mass range
        
            # Static potential integration
        statPos, statVel = lf.execute_leap(self.diff_pot, tRange, self.pos,
                                           self.vel, redV, massV, *args)
        
        return statPos, statVel
    
    
    def dist_time_vel(self, tPos, tVel):
        """ Find norm of position and velocity vectors, time dependent """
        
        pNorm = [np.linalg.norm(pos) for pos in tPos]       # Norm of position
        vNorm = [np.linalg.norm(vel) for vel in tVel]       # Norm of velocity
        
        return np.asarray(pNorm), np.asarray(vNorm)
    
    
    # def pot_energy(self, tRange, tPos, *args):
    #     """ Gravitational potential energy per unit mass """
        
    #         # Interpolate mass and redshift values
    #     normTime = self.norm_time(tRange)
    #     corrRed, corrMass = lf.correct_values(self.red, self.mass, normTime)
        
    #     tDist = np.asarray([np.linalg.norm(pos) for pos in tPos])   # Distance
        
    #     prof = self.profile
    #     pot = prof.pot_z_r(corrRed, tDist, *args)
        
    #     return pot
    
    def pot_energy(self, tRange, pos):
        """ Gravitational potential energy per unit mass """

            # Converting time range to years
        yearRange = ge.conv_sec_year(-tRange)               # Time in years
        corrRed = self.correct_red(yearRange)               # Corresponding z

        profile = self.prof                                 # Loading profile

        # fullPot = profile.pot_nfw(corrRed, pos)
        fullPot = [profile.pot_nfw(zV, pos[ind]) 
                   for ind, zV in enumerate(corrRed[:-1])]

        return fullPot # profile.pot_nfw(corrRed, pos)
    
    def tindep_pot(self, pos, z=0):
        """ Time independent potential energy """
        return self.prof.pot_nfw(z, pos)
    
    def kin_energy(self, tVel, *args):
        """ Kinetic energy per unit mass """
        return 0.5 * np.power(tVel, 2)


def exp_mah(z, alpha, M0=1.25e12):
    """ Simple exponential MAH """
    return M0 * np.exp(-alpha * z)

def two_param_mah(z, beta, gamma, M0=1.25e12):
    """ Two parameter MAH """
    return M0 * np.power(1+z, beta) * np.exp(-gamma * z)


def main():
    """ Main function that will be executed """
    
        # File names for models
    fZhao = "./mandc_m125_final/mandcoutput.m125_final"
    fBosch = "./getPWGH/PWGH_median.dat"

    M0 = 1.25e12                                        # Initial mass, M_sun
    
    
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
    tS = 1e-2                  # Start integration time (Gyr)
    tF = 13.2               # End integration time (Gyr)
    t0 = 13.8             # Age of Universe (Gyr)
    
        # Time in s
    timeRange = np.linspace(-ge.conv_year_sec(tS*1e9), -ge.conv_year_sec(tF*1e9), 
                            int(1e3))
    yearRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]       # Time in Gyr

    distRange = np.linspace(ge.conv_kpc_m(0.1), ge.conv_kpc_m(300), 1000)

        # Creating the dwarf_galaxy objects
    zhaoDwarf = dwarf_galaxy("Zhao", pStart, vStart, fZhao)     # Zhao
    boschDwarf = dwarf_galaxy("Bosch", pStart, vStart, fBosch)  # van den Bosch

    # dracoCond = initial_conds().draco
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

    #     # Integrating the orbit
    # zhaoP, zhaoV = zhaoDwarf.integ_time(timeRange)
    # zhaoPos, zhaoVel = zhaoDwarf.dist_time_vel(zhaoP, zhaoV)

    # boschP, boschV = boschDwarf.integ_time(timeRange)
    # boschPos, boschVel = boschDwarf.dist_time_vel(boschP, boschV)

    #     # Time independent
    # statP, statV = zhaoDwarf.time_indep(timeRange)
    # statPos, statVel = zhaoDwarf.dist_time_vel(statP, statV)

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
    ax.set_yscale("log")

    # ax.yaxis.offsetText.set_fontsize(22)
    # ax.legend(bbox_to_anchor=(1.23, 1.05), frameon=False, fontsize=22)

    ax2.set_yscale("log")

        # r(t)
    # ax.plot(yearRange[:-1], ge.conv_m_kpc(statPos[:-1]), color="teal", ls="--",
    #         label="Time independent")

    # ax.plot(yearRange[:-1], ge.conv_m_kpc(zhaoPos[:-1]), label="Zhao (2009)",
    #         color="navy")

    # ax.plot(yearRange[:-1], ge.conv_m_kpc(boschPos[:-1]), color="red",
    #         label="van den Bosch (2014)")

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
    fig.savefig("virrad_virmass_slides.png")

    show()


if __name__ == "__main__":
    main()