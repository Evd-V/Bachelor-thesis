import numpy as np
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.pyplot import figure, vlines, show

import general as ge
import mass_change as mc
from galaxy import dwarf_galaxy
from dwarfs import initial_conds, calculate_prop



def plot_graphs():
    """ Test function to plot some dwarfs """
    
        # File names for models
    fZhao = "./mandc_m125e12/mandcoutput.m125e12"
    # fZhao = "./mandc_m125_final/mandcoutput.m125_final"
    # fZhao = "./mandc_m10/mandcoutput.m10"
    # fBosch = "./getPWGH/PWGH_average_125e12_test.dat"
    fBosch = "./getPWGH/PWGH_median.dat"
    # fBosch = "./getPWGH/PWGH_median_m12.dat"
    # fBosch = "./getPWGH/PWGH_average_125e12_average.dat"

        # Time integration range
        # Time in s
    timeRange = np.linspace(-ge.conv_year_sec(1e7), -ge.conv_year_sec(13.5e9), int(1e3))
    yearRange = ge.conv_sec_year(timeRange) / 1e9                   # Time in Gyr
    
    t0, tS, tF = 13.78, 0, 13.5
    adjRange = np.linspace(t0-tF, t0-tS, int(1e3))[::-1]            # Time in Gyr
    
    
        # Draco
    draco = calculate_prop("Draco", fZhao, fBosch)
    draZP, draZV, draBP, draBV = draco.pos_vel(timeRange)
    
        # Sculptor
    sculptor = calculate_prop("Sculptor", fZhao, fBosch)
    scuZP, scuZV, scuBP, scuBV = sculptor.pos_vel(timeRange)
    
        # Carina
    carina = calculate_prop("Carina", fZhao, fBosch)
    carZP, carZV, carBP, carBV = carina.pos_vel(timeRange)
    
        # Sextans
    sextans = calculate_prop("Sextans", fZhao, fBosch)
    sxtZP, sxtZV, sxtBP, sxtBV = sextans.pos_vel(timeRange)
    
        # Ursa minor
    ursaMin = calculate_prop("Ursa minor", fZhao, fBosch)
    ursZP, ursZV, ursBP, ursBV = ursaMin.pos_vel(timeRange)
    
        # Combining all dwarf galaxies
    allZPos = [draZP, scuZP, ursZP, carZP]          # Zhao
    allBPos = [draBP, scuBP, ursBP, carBP]          # van den Bosch
    
    allZVel = [draZV, scuZV, ursZV, carZV]
    allBVel = [draBV, scuBV, ursBV, carBV]
    
        # Corresponding names
    names = ("Draco", "Sculptor", "Ursa Minor", "Carina")
    colors = ("navy", "red", "magenta", "teal")
    
        # MW virial radius
#     corrZVals = mc.time_red(adjRange*1e9)                   # Redshift for time
    dwarfGal = dwarf_galaxy("Zhao", sextans.load_dict()[0], sextans.load_dict()[1], fZhao)
    
    interTime = interp1d(dwarfGal.time, dwarfGal.red)
    corrZVals = interTime(ge.conv_sec_year(-timeRange))
    
    zhaoIntO = interp1d(dwarfGal.red, dwarfGal.virR)        # Interp for Zhao
    zhaoIntR = ge.conv_m_kpc(zhaoIntO(corrZVals))
    
    timeLims = ((2.9, 4.6), (3.4, 6.2), (2, 4.4), (5, 6), (9, 12.8))
    
    matplotlib.rcParams['font.family'] = ['Times']

    # Plotting
    fig = figure(figsize=(14,7), constrained_layout=False)
    outer_grid = fig.add_gridspec(2, 2, wspace=.15, hspace=0)
    axs = outer_grid.subplots()
    
    ind = 0                                         # Index to keep track of plots
    
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
        ax.axvspan(timeLims[ind][0], timeLims[ind][1], color="lightgreen", alpha=0.3)

        if ind == 3:
            ax.axvline(timeLims[ind+1][0], color="black", ls="-.")
            ax.axvline(timeLims[ind+1][1], color="black", ls="-.")
            ax.axvspan(timeLims[ind+1][0], timeLims[ind+1][1], color="lightgreen", alpha=0.3)
        
            # Setting axes
        # ax.grid()
        ax.legend(frameon=False, loc="upper right", fontsize=22)
        
        if ind == 0 or ind == 1:    # Very ugly solution...
            ax.tick_params(axis="x", labelsize=24, labelcolor="white")
            ax.tick_params(axis="y", labelsize=24, labelcolor="black")
            
        else:
            ax.tick_params(axis="both", labelsize=24)
        
        ind += 1                                    # To the next plot
    
        # Setting axes and title
    fig.supxlabel(r"$t_0-t$ (Gyr)", fontsize=22)
    fig.supylabel(r"$r$ (kpc)", fontsize=22)
    # fig.suptitle("Dashed = Zhao (2009), solid = van den Bosch (2014)", 
    #              fontsize=22)
    
    fig.savefig("galaxy_orbits_comp.png")
    
    show()

if __name__ == "__main__":
    plot_graphs()