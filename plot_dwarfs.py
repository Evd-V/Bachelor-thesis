import numpy as np
from matplotlib.pyplot import figure, show

import general as ge
from galaxy import dwarf_galaxy
from dwarfs import initial_conds, calculate_prop


def plot_graphs():
    """ Test function to plot some dwarfs """
    
        # File names for models
    fZhao = "./mandc_m125e12/mandcoutput.m125e12"
    fBosch = "./getPWGH/PWGH_average_125e12_test.dat"

        # Time integration range
    timeRange = np.linspace(0, -ge.conv_year_sec(13.5e9), int(1e3))  # Time in s
    yearRange = ge.conv_sec_year(timeRange) / 1e9                  # Time in Gyr

    t0, tS, tF = 13.78, 0, 13.5
    adjRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]       # Time in Gyr
    
    
        # Draco
    draco = calculate_prop("Draco")
    draZP, draZV, draBP, draBV = draco.pos_vel(timeRange, fZhao, fBosch)
    
        # Sculptor
    sculptor = calculate_prop("Sculptor")
    scuZP, scuZV, scuBP, scuBV = sculptor.pos_vel(timeRange, fZhao, fBosch)
    
        # Carina
    carina = calculate_prop("Carina")
    carZP, carZV, carBP, carBV = carina.pos_vel(timeRange, fZhao, fBosch)
    
        # Sextans
    sextans = calculate_prop("Sextans")
    sxtZP, sxtZV, sxtBP, sxtBV = sextans.pos_vel(timeRange, fZhao, fBosch)
    
        # Ursa minor
    ursaMin = calculate_prop("Ursa minor")
    ursZP, ursZV, ursBP, ursBV = ursaMin.pos_vel(timeRange, fZhao, fBosch)
    
    
    allZPos = [draZP, scuZP, ursZP, carZP]          # Zhao
    allBPos = [draBP, scuBP, ursBP, carBP]          # van den Bosch
    
        # Corresponding names
    names = ("Draco", "Sculptor", "Ursa Minor", "Carina")
    colors = ("navy", "red", "magenta", "teal")
    
    
    # Plotting
    fig = figure(figsize=(14,7), constrained_layout=False)
    outer_grid = fig.add_gridspec(2, 2, wspace=.15, hspace=0)
    
    axs = outer_grid.subplots()
    
    ind = 0
    for i, ax in np.ndenumerate(axs):
        ax.plot(adjRange[:-1], ge.conv_m_kpc(allZPos[ind][:-1]), ls="--", 
                color=colors[ind])
                
        ax.plot(adjRange[:-1], ge.conv_m_kpc(allBPos[ind][:-1]), label=names[ind], 
                color=colors[ind])
        
        ax.grid()
        ax.legend(fontsize=18)
        ax.set_xlabel(r"$t-t_0$ (Gyr)", fontsize=18)
        
        ind += 1
    
    show()

if __name__ == "__main__":
    plot_graphs()
