import numpy as np
import matplotlib
from matplotlib.pyplot import figure, show

import general as ge

def hist_draco():
    """ Histogram SFH Draco dwarf """

    tRange = (0.643, 2.022, 3.6765, 5.5148, 7.353)      # Time values
    psiInner = (0.67, 0.685, 0.18, 0.03, 0)             # Inner part
    psiOuter = (0.615, 0.63, 0.39, 0.03, 0)             # Outer part

    psiRangeIn = np.asarray(psiInner) * 1e-4
    psiRangeOut = np.asarray(psiOuter) * 1e-4

    return np.asarray(tRange), psiRangeIn + psiRangeOut

def hist_ursa():
    """ Histogram SFH Ursa Minor """

        # Assume BP stars are blue stragglers!!
    tRange = (0.919, 3.217, 5.5148)             # Questionable last points
    psiInner = (1.48, 0.13, 0)                  # Inner region
    psiOuter = (1.23, 0.23, 0)                  # Outer region

    psiRangeIn = np.asarray(psiInner) * 1e-4
    psiRangeOut = np.asarray(psiOuter) * 1e-4

    return np.asarray(tRange), psiRangeIn + psiRangeOut

def hist_sculptor():
    """ Histogram SFH Sculptor """

    # tRange = range(0.3, 10.3, 0.5)          # Time range
    psiTuple = (.98, 1, .99, .95, .92, .87, .82, .75, .62, 
                .57, .5, .39, .28, .21, .15, .12, .1, .08, 
                .07, .07, .08, .14)             # Psi values
    psiRange = np.asarray(psiTuple) * 1e-3# * 1e-4

    tRange = np.linspace(0.3, 10.3, len(psiRange))

    return tRange, psiRange

def hist_carina():
    """ Histogram SFH Carina (Savino 2015) """

    # tRange = range(0.05, 11.8, 0.5)         # Time range
    psiTuple = (1.95, 1.33, .93, .95, 1.18, 1.18, 1.57, 1.2, 
                1.18, 1., .82, .6, .47, .56, 1.18, 1.38, 2., 
                2.18, 1.82, 1.44, 1.24, .82, .6, .28)
    psiRange = np.asarray(psiTuple) * 1e-4  # Psi values

    tRange = np.linspace(0.05, 11.8, len(psiRange))

    return tRange, psiRange

def accurate_sculptor():
    """ Accurate SFH values for Sculptor (de Boer(?)) """

    fName = "./SFHs/Sculptor_SFH/SFHage_all"            # Data file name
    data = np.loadtxt(fName, comments="#")              # Loading data

        # Unpacking
    timeVals = data[:,0]
    one, two, three, four, five = data[:,2], data[:,4], data[:,6], data[:,8], data[:,10]

    totSFH = one + two + three + four + five

    return 14 - timeVals, totSFH
    

def accurate_carina():
    """ Accurate SFH values for Carina (de Boer 2014) """

    fName = "./SFHs/Carina_SFH/SFHage_all"               # Data file name
    data = np.loadtxt(fName, comments="#")              # Loading data

    timeVals = data[:,0]
    inner, middle, outer = data[:,2], data[:,4], data[:,6]  # Unpacking
    totSFH = inner + middle + outer

    return 14 - timeVals, totSFH


def diff_complete(y, t):
    """ Differentiate function, 0 if delta y = 0 """
    deltaT = t[1:] - t[:-1]
    return np.where(deltaT == 0, 0, (y[1:] - y[:-1]) / deltaT)

def take_deriv(name):
    """ Take derivative """

        # Dictionary containing histogram data
    dataDict = {
                "Draco": hist_draco(),
                "Ursa Minor": hist_ursa(),
                "Sculptor": hist_sculptor(),
                "Carina": accurate_carina()
               }
    
    dwarfData = dataDict[name]                  # Picking correct dwarf
    tVals, psiVals = dwarfData[0], dwarfData[1] # Unpacking

    dT = (tVals[1:] + tVals[:-1]) * .5       # Time to take derivative

    return (tVals, psiVals), (dT, diff_complete(psiVals, tVals))

def take_log_deriv(name):
    """ Take derivative of ln(psi_ """

            # Dictionary containing histogram data
    dataDict = {
                "Draco": hist_draco(),
                "Ursa Minor": hist_ursa(),
                "Sculptor": hist_sculptor(),
                "Carina": accurate_carina()
               }
    
    dwarfData = dataDict[name]                  # Selecting correct dwarf
    tVals, psiData = dwarfData[0], dwarfData[1] # Unpacking data

    dT = (tVals[1:] + tVals[:-1]) * .5                      # time step
    logPsi = np.where(psiData != 0., np.log(psiData), 0)     # ln(psi)

    return (tVals, psiData), (dT, diff_complete(logPsi, tVals)*psiData[1:])

def stair_edges(dT, tVals):
    """ Determine edges of stairs function """

    final = 2 * tVals[-1] - dT[-1]              # Final edge location
    edges1 = np.append([0], [dT])               # First append 0 at beginning

    return np.append([edges1], [final])

def plot_hist(name, saveFig=None):
    """ Plot histogram of SFH for given dwarf galaxy """

        # Dictionary containing histogram data
    dataDict = {
                "Draco": hist_draco(),
                "Ursa Minor": hist_ursa(),
                "Sculptor": hist_sculptor(),
                "Carina": accurate_carina()
               }
    
    # data = dataDict[name]

    data = take_deriv(name)
    tVals, histVals = data[0]
    dT, derivVals = data[1]


    # tVals, histVals = data[0], np.where(data[1]!=0, np.log(data[1]), 0)
    # dT, derivVals = take_log_deriv(name)

    histEdges1 = np.append([0], [dT])

    dist = tVals[-1] - dT[-1]
    histEdges = np.append([histEdges1], [tVals[-1]+dist])

    matplotlib.rcParams['font.family'] = ['Times']

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # ax.step(tVals, histVals, where="mid", color="navy")
    ax.stairs(histVals, histEdges, color="navy", lw=2)
    ax.scatter(tVals, histVals, marker="o", color="navy", s=100, alpha=.3, zorder=3)

    # ax2.axhline(-1.25, color="k", lw=2)
    ax2.plot(dT, derivVals, color="navy", marker="o", ms=10, lw=2)

    ax.yaxis.offsetText.set_fontsize(22)
    ax.set_ylabel(r"$\psi$ ($10^{-4}$ M$_\odot$ yr$^{-1}$)", fontsize=22)

    ax2.yaxis.offsetText.set_fontsize(22)
    ax2.set_ylabel(r"d$\psi$/dt ($10^{-5}$ M$_\odot$ yr$^{-1}$ Gyr$^{-1}$)", fontsize=22)

    ax.tick_params(axis="both", labelsize=24)
    ax2.tick_params(axis="both", labelsize=24)

    fig.supxlabel(r"$t_0 - t$ (Gyr)", fontsize=22)

    fig.tight_layout()
    if saveFig: fig.savefig(saveFig)

    show()

# "Normal derivative"
# Draco: -1e-5
# Ursa: -5e-5
# Sculptor: -1.3e-4
# Carina: -3.5e-5

# log derivtive * hist data
# Draco: -5e-6
# Ursa: -1e-5, but anywhere between -2e-6 and -2.5e-5 is fine
# Sculptor: -1.1e-4
# Carina: -4e-5



def main():

    # plot_hist("Ursa Minor")

    names = ["Draco", "Ursa Minor", "Sculptor", "Carina"]
    colors = ["navy", "magenta", "red", "teal"]
    lineStyles = ["-", "--", "-.", ":"]
    markers = ["o", "x", "+", "*"]

    derivDwarf = take_deriv("Draco")
    xVals = range(0, len(derivDwarf))

    # derivDwarfs = [take_deriv(name) for name in names]
    # xDwarfs = [range(0, len(deriv)) for deriv in derivDwarfs]

    # plot_hist("Ursa Minor", saveFig=None)

    derivDwarfs = [take_log_deriv(name)[1] for name in names]

        # Draco -0.6e-5
        # Ursa Minor -0.5e-5
        # Sculptor -1.4e-5
        # Carina -7e-5

    matplotlib.rcParams['font.family'] = ['Times']

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)

    for ind, dwarf in enumerate(derivDwarfs):
        ax.plot(dwarf[0], dwarf[1], marker=markers[ind], label=names[ind],  
                color=colors[ind], ls=lineStyles[ind], ms=10, lw=2)
    
    # ax.axhline(-.5, color="k")

    ax.set_xlabel(r"$t - t_0$ (Gyr)", fontsize=22)
    ax.set_ylabel(r"d$\psi$ / dt ($10^{-4}$ M$_\odot$ yr$^{-1}$ Gyr$^{-1}$)", fontsize=22)
    ax.tick_params(axis="both", labelsize=24)
    ax.legend(fontsize=22, frameon=False)

    # ax.grid()

    fig.tight_layout()
    # fig.savefig("derivatives.png")
    show()

if __name__ == "__main__":
    main()