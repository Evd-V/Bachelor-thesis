import numpy as np
from astropy.constants import kpc
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure, show, cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

import general as ge
from galaxy import dwarf_galaxy
from dwarfs import initial_conds

def retrieve_3d(galName, modelName, fName, tRange, sig=None):
    """ Retrieve 3D position and velocity """

    init = initial_conds(sig=sig)           # Initializing conditions

            # Dictionary containing the data
    galData = {
                "Draco": init.draco,
                "Sculptor": init.sculp,
                "Ursa minor": init.ursa,
                "Sextans": init.sext,
                "Carina": init.car
                }

    initialDwarf = galData[galName]                  # Loading data

    dwarfObject = dwarf_galaxy(modelName, initialDwarf[0], 
                               initialDwarf[1], fName)
    
    pos3D, vel3D = dwarfObject.integ_time(tRange)   # Integrating

        # MW virial radius
    yearRange = ge.conv_sec_year(tRange) / 1e9           # Time in Gyr
    interTime = interp1d(dwarfObject.time, dwarfObject.red)
    corrZVals = interTime(-yearRange*1e9)
    
    IntObj = interp1d(dwarfObject.red, dwarfObject.virR) # Interp object
    IntRad = IntObj(corrZVals)                           # Interpolating

    return pos3D, vel3D, IntRad

def plot_3D_data(pos3D, virR):
    """ Plot 3D position (and possibly velocity) """

    pos3D /= kpc.value                  # Position in kpc
    x, y, z = pos3D[:-1,0], pos3D[:-1,1], pos3D[:-1,2]  # Unpacking

    virR = virR[0] / kpc.value                   # R_vir in kpc

    # draw sphere
    phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xS = virR*np.cos(phi)*np.sin(theta)
    yS = virR*np.sin(phi)*np.sin(theta)
    zS = virR*np.cos(theta)

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1, projection="3d")

    ax.plot(x, y, z, color="navy")
    ax.plot_surface(xS, yS, zS, color="crimson", alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    show()

def get_virR(virR):
    """ Virial radius in Cartesian coords """
    
    phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xS = virR*np.cos(phi)*np.sin(theta)
    yS = virR*np.sin(phi)*np.sin(theta)
    zS = virR*np.cos(theta)

    return np.asarray([xS, yS, zS])


def update_data(its, pos, lines):
    """ Function for animation """
    lines.set_data(pos[:its,0], pos[:its,1])
    lines.set_3d_properties(pos[:its,2])
    lines.set_c("navy")
    lines.set_lw(2)
    return lines

def animate_orbit(pos):
    """ Animate the orbit of a dwarf galaxy """

    pos /= kpc.value        # Position
    GC = (0, 0, 0)          # Galactic center

    fig = figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

        # Increase figure size
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), 
                          np.diag([1, 1, 1, 1]))

    lines = ax.plot([], [], [])[0]

    ax.scatter(*GC, marker="o", color="red", s=50)
    ax.scatter(*pos[0,:], marker="*", color="forestgreen", s=50)

    ax.set_xlim(-45, 50)
    ax.set_ylim(-125, 175)
    ax.set_zlim(-100, 200)

    ax.set_xlabel(r"$x$ (kpc)", fontsize=20)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=20)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=20)

    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20

    ax.tick_params(axis="both", labelsize=20)

    ani = animation.FuncAnimation(
            fig, update_data, len(pos), 
            fargs=(pos, lines), interval=5, 
            repeat=False)
    
        
    show()

    # ani.save("Draco_orbit.mp4", fps=30)

def main():
    """ Main function that will be executed"""

        # File names for models
    fZhao = "mandc_m125_final/mandcoutput.m125_final"

        # Time integration range
    tS = 1e-2                   # Start integration time (Gyr)
    tF = 13.2                   # End integration time (Gyr)
    t0 = 13.8                   # Age of Universe (Gyr)
    
        # Time in s
    timeRange = np.linspace(-ge.conv_year_sec(tS*1e9), -ge.conv_year_sec(tF*1e9), 
                            int(1e3))
    yearRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]       # Time in Gyr


        # Draco
    posVec, velVec, virR = retrieve_3d("Draco", "Zhao", fZhao, timeRange)

    posVec = posVec[:-1,:]

    # plot_3D_data(posVec, virR)
    animate_orbit(posVec[::-3])


if __name__ == "__main__":
    main()
