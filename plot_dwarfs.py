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
    
        # MW virial radius
    dwarfGal = dwarf_galaxy("Zhao", sextans.load_dict()[0], sextans.load_dict()[1], fZhao)
    
    interTime = interp1d(dwarfGal.time, dwarfGal.red)
    corrZVals = interTime(ge.conv_sec_year(-timeRange))
    
    zhaoIntO = interp1d(dwarfGal.red, dwarfGal.virR)        # Interp for Zhao
    zhaoIntR = ge.conv_m_kpc(zhaoIntO(corrZVals))
    
    timeLims = ((3.5, 4.5), (4, 7), (1.6, 3.6), (5.5, 7))
    
    # Plotting
    fig = figure(figsize=(14,7), constrained_layout=False)
    outer_grid = fig.add_gridspec(2, 2, wspace=.15, hspace=0)
    axs = outer_grid.subplots()
    
    ind = 0                                         # Index to keep track
    
    for i, ax in np.ndenumerate(axs):
        
            # Zhao model
        ax.plot(adjRange[:-1], ge.conv_m_kpc(allZPos[ind][:-1]), ls="--", 
                color=colors[ind])
        
            # van den Bosch model
        ax.plot(adjRange[:-1], ge.conv_m_kpc(allBPos[ind][:-1]), label=names[ind], 
                color=colors[ind])
        
            # MW virial radius
        ax.plot(adjRange, zhaoIntR, color="slateblue", ls=":")
        
        ax.axvline(timeLims[ind][0], color="black", ls="-.")
        ax.axvline(timeLims[ind][1], color="black", ls="-.")
        
            # Setting axes
        ax.grid()
        ax.legend(frameon=False, loc="upper right", fontsize=18)
        
        if ind == 0 or ind == 1:    # Very ugly solution...
            ax.tick_params(axis="x", labelsize=18, labelcolor="white")
            ax.tick_params(axis="y", labelsize=18, labelcolor="black")
            
        else:
            ax.tick_params(axis="both", labelsize=18)
        
        ind += 1                                    # To the next plot
    
        # Setting axes and title
    fig.supxlabel(r"$t-t_0$ (Gyr)", fontsize=18)
    fig.supylabel(r"$r$ (kpc)", fontsize=18)
    fig.suptitle("Dashed = Zhao (2009), solid = van den Bosch (2014)", 
                 fontsize=18)
    
#     fig.savefig("galaxy_orbits.png")
    
    show()

if __name__ == "__main__":
    plot_graphs()
