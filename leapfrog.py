import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
import astropy.units as u
from matplotlib.pyplot import figure, show, cm

import general as ge
import nfw_profile as nf
import mass_change as mc
import initial_cond as ic
from classes import nfw_potential



def leap_frog(func, dt, x0, vh, *args):
    """ Leapfrog method """
    
    fN = func(x0, *args)                    # f(x_0)
    vHalf = vh + fN * dt                    # v_{i-1/2}
    xN = x0 + vHalf * dt                    # x_{i+1}
    
    return xN, vHalf


def alt_leap(func, dt, x0, v0, *args):
    """ Alternative form of leapfrog method """
    
    a0 = func(x0, *args)                    # a_i
    xN = x0 + v0 * dt + 0.5 * a0 * dt * dt  # x_{i+1}
    
    aN = func(xN, *args)                    # a_{i+1}
    vN = v0 + 0.5 * (a0 + aN) * dt          # v_{i+1}
    
    return xN, vN


def execute_leap(func, tRange, p0, v0, z, M, *args):
    """ Execute the leapfrog integration method """
    
    tSteps = tRange[1:] - tRange[:-1]               # Time steps
    h = np.mean(tSteps)
    pV = np.zeros((len(tRange), 3))                 # Array for x values
    vV = np.zeros((len(tRange), 3))                 # Array for vel. values
    
    pV[0] = p0                                      # Initial position vector
    vV[0] = v0                                      # Initial velocity vector
    
    for i in range(len(tSteps)-1):
        pV[i+1], vV[i+1] = alt_leap(func, tSteps[i], pV[i], vV[i], z, M, *args)
    
    return pV, vV


def time_leap(func, tRange, p0, v0, z, M, *args):
    """ Time dependent leapfrog method """
    
    tSteps = tRange[1:] - tRange[:-1]               # Time steps
    pV = np.zeros((len(tRange), 3))                 # Array for x values
    vV = np.zeros((len(tRange), 3))                 # Array for vel. values
    
        # Checking input
    if len(p0) != len(v0):
        raise Exception("Position and velocity must have the same length")
    
    if len(z) != len(tRange):                       # Interpolate mass & z range
        newZ = np.linspace(min(z), max(z), len(tRange))
        interpZ = interp1d(ge.lin_func(z), z)
        
        M = nf.find_m_z(z, M, newZ, *args)          # Full mass range
        z = interpZ(newZ)                           # Full redshift range
    
    
    pV[0] = p0                                      # Initial position vector
    vV[0] = v0                                      # Initial velocity vector
    
    for i in range(len(tSteps)-1):
        pV[i+1], vV[i+1] = alt_leap(func, tSteps[i], pV[i], vV[i], z[i], M[i], 
                                    *args)
    
    return pV, vV


def diff_eq_pot(p, z, M, *args):
    """ Second order differential equation that has to be solved """
    
    r = np.linalg.norm(p)                               # Distance
    
    rS = nf.rs_nfw(z, M, *args)                         # Scale length
    rhoS = nf.rhos_nfw(z, M, *args)                     # Density at r_s
    
    term1 = -16 * np.pi * G * rhoS * rS                  # Term in front
    term2 = 1 + r / rS                                  # Recurring term
    term3 = rS / (r * term2) - np.power(rS/r, 2) * np.log(term2)    # Brackets
    
    fRad = -term1 * term3                               # Force in radial direc.
    
    return p * fRad / r


def new_diff_pot(p, z, M, *args):
    """ New differential equation """
    
    profile = nfw_potential(z, M)                       # Initializing pot.
    
    r = np.linalg.norm(p)                               # Distance
    
    rS = profile.scale_rad()                            # Scale radius
    rhoS = profile.rho_rs()                             # rho_s
    
    term1 = -16 * np.pi * G * rhoS * rS                 # Term in front
    term2 = 1 + r / rS                                  # Recurring term
    term3 = rS / (r * term2) - np.power(rS/r, 2) * np.log(term2)    # Brackets
    
    fRad = -term1 * term3                               # Force in radial direc.
    
    return p * fRad / r


def correct_values(redRange, massRange, timeRange):
    """ Correcting redshift and mass for time dependent potential """
    
    corrRed = mc.time_red(timeRange)                    # z range of interest
    corrMass = nf.find_m_z(redRange, massRange, corrRed)    # Corresponding M
    
    return corrRed, corrMass


def kin_energy(vel):
    """ Kinetic energy per unit mass """
    return 0.5 * vel * vel


def pot_energy(dist, z=0, M=1e12, *args):
    """ Gravitational potential energy per unit mass """
    return nf.pot_nfw(dist, z, M, *args)


def plot_mult_3d(pos, linestyles, labels, cmap=cm.cool, step=10, saveFig=None):
    """ Plot multiple 3d orbits """
    
    startPos = pos[0][0]                                # Starting position
    galCen = np.zeros((3))                              # Galactic center
    timeLen = len(pos[0])                               # Number of points
    
    xV = [p[:,0] for p in pos]
    yV = [p[:,1] for p in pos]
    zV = [p[:,2] for p in pos]
    
    # Plotting
    fig = figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    ax.scatter(*startPos, label="Start", marker="o", color="crimson")
    ax.scatter(*galCen, label="GC", marker="o", color="chartreuse")
    
    for j in range(len(xV)):
        for i in range(0, timeLen-step, step):
            ax.plot(xV[j][i:i+step+1], yV[j][i:i+step+1], zV[j][i:i+step+1], 
                    color=cmap(i/timeLen), alpha=0.9, 
                    ls=linestyles[j])
    
        # Setting axes
    ax.set_xlabel(r"$x$ (kpc)", fontsize=18)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=18)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    
    titl = ""
    for ind in range(len(labels)):
        titl += f"{linestyles[ind]} = {labels[ind]}\n"
    
    ax.legend(ncol=2, frameon=False, fontsize=15)
    ax.set_title(titl)
    
    if saveFig: fig.savefig(str(saveFig))
    else: show()


def plot_orbit_3d(pos, cmap=cm.cool, step=10, saveFig=None):
    """ Plot 3d orbit """
    
    startPos = pos[0]                                   # Starting position
    galCen = np.zeros((3))                              # Galactic center
    timeLen = len(pos)                                  # Number of points
    
    xV, yV, zV = pos[:,0], pos[:,1], pos[:,2]           # Unpacking position
    
    # Plotting
    fig = figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d', azim=-83, elev=49)
    
    ax.scatter(*startPos, label="Start", marker="o", color="crimson")
    ax.scatter(*galCen, label="GC", marker="o", color="chartreuse")
    ax.scatter(100, 100, 100, marker="x")
    
        # Plotting the orbit
    for i in range(0, timeLen-step, step):
        ax.plot(xV[i:i+step+1], yV[i:i+step+1], zV[i:i+step+1], 
                color=cmap(i/timeLen), alpha=0.9)
    
        # Setting axes
    ax.set_xlabel(r"$x$ (kpc)", fontsize=18)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=18)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    
    ax.legend(fontsize=15)
    
    fig.tight_layout()
    
    if saveFig: fig.savefig(str(saveFig))
    else: show()



def main():
    """ Main function that will be executed """
    
        # Time integration range
    t0 = 13.2e9                                         # Age of Universe
    
    timeRange = np.linspace(0, -ge.conv_year_sec(12e9), int(1e3))  # Time in s
    yearRange = ge.conv_sec_year(timeRange) / 1e9                  # Time in Gyr
    
    normTime = t0 + yearRange * 1e9                                # Normal time
    
        # For Draco dwarf galaxy
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
    
    pSpher = (ge.conv_kpc_m(81.8), 55.3, 273.3)     # Starting pos in spherical
    vSpher = (-97.4*1e3, 136.9*1e3, -66.2*1e3)      # Startign vel in spherical
    
    pStart = p0Dra * 3.0857e19                      # Starting position in m
    vStart = ic.conv_vel_frame(vSpher, pSpher)      # Starting velocity in m/s
    
    
        # Retrieving redshift and mass
    fName = "./getPWGH/PWGH_average_12.dat"
    newFB = "./getPWGH/PWGH_average_125e12.dat"
#     fileZhao = "./mandc_m12/mandcoutput.m12"
    fileZhao = "./mandc_m125e12/mandcoutput.m125e12"
    
    M0 = 1e12                                               # MW initial mass
    zV = 0                                                  # Redshift
    
        # Time dependent
    boschRed, boschMass, boschRate = mc.mah_bosch(fName, M0)    # van den Bosch
    corrBoschRed, corrBoschMass = correct_values(boschRed, boschMass, normTime)
    
    boschPos, boschVel = time_leap(diff_eq_pot, timeRange, pStart, vStart, 
                                   corrBoschRed, corrBoschMass)
    
    posBosch = np.asarray([np.linalg.norm(pos) for pos in boschPos])
    velBosch = np.asarray([np.linalg.norm(vel) for vel in boschVel])
    
        # New Bosch model
    bR, bM, bRate = mc.mah_bosch(newFB, 1.25e12)
    cBR, cBM = correct_values(bR, bM, normTime)
    
    newBP, newBV = time_leap(new_diff_pot, timeRange, pStart, vStart, cBR, cBM)
    pB = np.asarray([np.linalg.norm(pos) for pos in newBP])
    vB = np.asarray([np.linalg.norm(vel) for vel in newBV])
    
        # Zhao
    zhao = mc.read_zhao(fileZhao)                   # Zhao data
    corrZhaoRed, corrZhaoMass = correct_values(zhao[0], zhao[1]/.6766, normTime)
    
    zhaoPos, zhaoVel = time_leap(new_diff_pot, timeRange, pStart, vStart, 
                                 corrZhaoRed, corrZhaoMass)
    
    posZhao = np.asarray([np.linalg.norm(pos) for pos in zhaoPos])
    velZhao = np.asarray([np.linalg.norm(vel) for vel in zhaoVel])
    
        # Integrating the orbit, time independent
    integPos, integVel = execute_leap(diff_eq_pot, timeRange, pStart, vStart, zV, M0)
    
    fullPos = np.asarray([np.linalg.norm(pos) for pos in integPos])
    fullVel = np.asarray([np.linalg.norm(vel) for vel in integVel])
    
    scalePos = ge.conv_m_kpc(integPos)
    
#     rApo = max(fullPos[:-1])                        # Apocenter
#     rPeri = min(fullPos[:-1])                       # Pericenter
#     ecc = (rApo - rPeri) / (rApo + rPeri)           # Eccentricity
#     
#     
#     kinEnergy = kin_energy(velBosch)                       # Kinetic energy
#     potEnergy = pot_energy(posBosch)                       # Potential energy
#     totEnergy = kinEnergy + potEnergy                      # Total energy
#     
#     zhaoKin = kin_energy(velZhao)
#     zhaoPot = pot_energy(posZhao)
#     zhaoTot = zhaoKin + zhaoPot
    
#     plot_orbit_3d(ge.conv_m_kpc(integPos[:-1]), saveFig=None)
#     labels = ["van den Bosch", "Zhao", "Time independent"]
#     plot3d = [ge.conv_m_kpc(boschPos), ge.conv_m_kpc(zhaoPos), scalePos]
#     plot_mult_3d(plot3d, linestyles=["solid", "dashed", "dotted"], labels=labels)
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
#     ax2 = ax.twinx()
    
    ax.scatter(0, ge.conv_m_kpc(fullPos[0]), marker="o", color="springgreen", zorder=3)
    
#     ax.plot(yearRange[:-1], kinEnergy[:-1], label="Kinetic energy", color="navy")
#     ax.plot(yearRange, potEnergy, label="Potential energy", color="green")
#     ax.plot(yearRange, totEnergy, label="Total energy", color="red")
#     
#     ax.plot(yearRange[:-1], zhaoKin[:-1], ls="--", color="navy")
#     ax.plot(yearRange, zhaoPot, ls="--", color="green")
#     ax.plot(yearRange, zhaoTot, ls="--", color="red")
    
    ax.plot(yearRange[:-1], ge.conv_m_kpc(fullPos[:-1]), label="Time independent", 
            color="navy")
    
    ax.plot(yearRange[:-1], ge.conv_m_kpc(posBosch[:-1]), label="van den Bosch", 
            color="red")
    
    ax.plot(yearRange[:-1], ge.conv_m_kpc(pB[:-1]), label="New vdB", ls="--",
            color="teal")
    
    ax.plot(yearRange[:-1], ge.conv_m_kpc(posZhao[:-1]), label="Zhao",
            color="lime")
    
#     ax2.plot(yearRange[:-1], fullVel[:-1]*1e-3, color="navy", ls="--")
#     ax2.plot(yearRange[:-1], boschVel[:-1]*1e-3, color="red", ls="--")
#     ax2.plot(yearRange[:-1], boschVel[:-1]*1e-3, color="lime", ls="--")
    
#     for ind in range(len(integVel[0])):
#         ax.plot(yearRange[:-1], integVel[:-1,ind]*1e-3, label=coordinates[ind])
    
    ax.set_xlabel(r"$t$ (Gyr)", fontsize=18)
#     ax.set_ylabel(r"$\epsilon$ (J/kg)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
#     ax.set_title("Solid = van den Bosch, dashed = Zhao", fontsize=15)
    
#     ax2.set_ylabel(r"$v$(km/s)", fontsize=18)
#     ax2.tick_params(axis="y", labelsize=15)
#     ax2.legend(bbox_to_anchor=(0.7, 1.1), frameon=False, fontsize=15)
    
#     ax.legend(bbox_to_anchor=(1.25, 1), ncol=1, frameon=False, fontsize=15)
    
    ax.legend(fontsize=15)
    ax.grid()
    
#     fig.tight_layout()
#     fig.savefig("time_dep_energy.png")
    show()

if __name__ == "__main__":
    main()
