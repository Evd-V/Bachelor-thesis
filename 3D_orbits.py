import numpy as np
from astropy.constants import kpc
from matplotlib.pyplot import figure, show
import matplotlib.animation as animation

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

    return pos3D, vel3D

def plot_3D_data(pos3D):
    """ Plot 3D position (and possibly velocity) """

    pos3D /= kpc.value
    x, y, z = pos3D[:-1,0], pos3D[:-1,1], pos3D[:-1,2]

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1, projection="3d")

    ax.plot(x, y, z, color="navy")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    show()

def update_data(its, pos, lines):
    """ Function for animation """
    lines.set_data(pos[:its,0], pos[:its,1])
    lines.set_3d_properties(pos[:its,2])
    return lines

def animate_orbit(pos):
    """ Animate the orbit of a dwarf galaxy """

    pos /= kpc.value        # Position
    GC = (0, 0, 0)          # Galactic center

    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(projection='3d')

    lines = ax.plot([], [], [])[0]

    ax.set_xlim(-45, 50)
    ax.set_ylim(-125, 175)
    ax.set_zlim(-100, 200)

    ax.set_xlabel(r"$x$ (kpc)", fontsize=22)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=22)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=22)

    ax.tick_params(axis="both", labelsize=24)

    ani = animation.FuncAnimation(
            fig, update_data, len(pos), 
            fargs=(pos, lines), interval=5)
        
    show()


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
    posVec, velVec = retrieve_3d("Draco", "Zhao", fZhao, timeRange)

    # plot_3D_data(posVec)
    animate_orbit(posVec[:-1:-3])


if __name__ == "__main__":
    main()
