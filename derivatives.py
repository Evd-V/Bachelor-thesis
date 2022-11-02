import numpy as np
import matplotlib
from matplotlib.pyplot import figure, show
from scipy.interpolate import CubicSpline


class sfh_dwarfs(object):
    """ Class for storing and analysing SFHs of several dwarfs

        Attributes:

            dataDict:   Dictionary containing the SFH of the 4 dwarf 
                        galaxies: Draco, Ursa Minor, Sculptor and 
                        Carina. The values in the dictionary are a tuple 
                        containing the time values and the SFRs at the 
                        corresponding times.
    """

    def __init__(self) -> None:
        """ Initializing the SFHs 

            Input:
                -
            
            Returns:
                sfh_dwarfs  (object)
        """
    
                # Dictionary containing histogram data
        self.dataDict = {
            "Draco": self.hist_draco(),
            "Ursa Minor": self.hist_ursa(),
            "Sculptor": self.hist_sculptor(),
            "Carina": self.accurate_carina()
            }

    def hist_draco(self):
        """ Histogram SFH Draco dwarf. """

        tRange = (0.643, 2.022, 3.6765, 5.5148, 7.353)      # Time values
        psiInner = (0.67, 0.685, 0.18, 0.03, 0)             # Inner part
        psiOuter = (0.615, 0.63, 0.39, 0.03, 0)             # Outer part

        psiRangeIn = np.asarray(psiInner) * 1e-4
        psiRangeOut = np.asarray(psiOuter) * 1e-4

        return np.asarray(tRange), psiRangeIn + psiRangeOut

    def hist_ursa(self):
        """ Histogram SFH Ursa Minor. """

            # Assume BP stars are blue stragglers!!
        tRange = (0.919, 3.217, 5.5148)             # Questionable last points
        psiInner = (1.48, 0.13, 0)                  # Inner region
        psiOuter = (1.23, 0.23, 0)                  # Outer region

        psiRangeIn = np.asarray(psiInner) * 1e-4
        psiRangeOut = np.asarray(psiOuter) * 1e-4

        return np.asarray(tRange), psiRangeIn + psiRangeOut

    def hist_sculptor(self):
        """ Histogram SFH Sculptor. """

        # tRange = range(0.3, 10.3, 0.5)          # Time range
        psiTuple = (.98, 1, .99, .95, .92, .87, .82, .75, .62, 
                    .57, .5, .39, .28, .21, .15, .12, .1, .08, 
                    .07, .07, .08, .14)             # Psi values
        psiRange = np.asarray(psiTuple) * 1e-3# * 1e-4

        tRange = np.linspace(0.3, 10.3, len(psiRange))

        return tRange, psiRange

    def hist_carina(self):
        """ Histogram SFH Carina (Savino 2015). """

        # tRange = range(0.05, 11.8, 0.5)         # Time range
        psiTuple = (1.95, 1.33, .93, .95, 1.18, 1.18, 1.57, 1.2, 
                    1.18, 1., .82, .6, .47, .56, 1.18, 1.38, 2., 
                    2.18, 1.82, 1.44, 1.24, .82, .6, .28)
        psiRange = np.asarray(psiTuple) * 1e-4  # Psi values

        tRange = np.linspace(0.05, 11.8, len(psiRange))

        return tRange, psiRange

    def accurate_sculptor(self):
        """ Accurate SFH values for Sculptor (de Boer(?)). """

        fName = "./SFHs/Sculptor_SFH/SFHage_all"            # Data file name
        data = np.loadtxt(fName, comments="#")              # Loading data

            # Unpacking
        timeVals = data[:,0]
        one, two, three, four, five = data[:,2], data[:,4], data[:,6], data[:,8], data[:,10]

        totSFH = one + two + three + four + five

        return 14 - timeVals, totSFH

    def accurate_carina(self):
        """ Accurate SFH values for Carina (de Boer 2014). """

        fName = "./SFHs/Carina_SFH/SFHage_all"               # Data file name
        data = np.loadtxt(fName, comments="#")              # Loading data

        timeVals = data[:,0]
        inner, middle, outer = data[:,2], data[:,4], data[:,6]  # Unpacking
        totSFH = inner + middle + outer

        return 14 - timeVals, totSFH


    def diff_complete(self, y, t):
        """ Differentiate function, 0 if delta t = 0. """
        deltaT = t[1:] - t[:-1]
        return np.where(deltaT == 0, 0, (y[1:] - y[:-1]) / deltaT)


    def take_deriv(self, name, norm=True):
        """ Take derivative of the SFH for one of four dwarf 
            galaxies: Draco, Ursa Minor, Sculptor or Carina.

            Input:
                name:   name of dwarf (string).
            
            Returns:
                Tuple containing the time values and the SFH of 
                the dwarf (tuple containing 2 numpy arrays).
                
                Tuple containing the time values where the 
                derivates are taken and the derivative of the SFH
                (tuple containing 2 numpy arrays).
        """

            # Dictionary containing histogram data
        
        dwarfData = self.dataDict[name]             # Picking dwarf
        tVals, psiVals = dwarfData[0], dwarfData[1] # Unpacking

        if norm: psiVals /= max(psiVals)            # Normalizing

        dT = (tVals[1:] + tVals[:-1]) * .5          # Time for derivatives

        return (tVals, psiVals), (dT, self.diff_complete(psiVals, tVals))

    def take_log_deriv(self, name):
        """ Take derivative of the log of the SFH for one of four 
            dwarf galaxies: Draco, Ursa Minor, Sculptor or Carina.

            Input:
                name:   name of dwarf (string).
            
            Returns:
                Tuple containing the time values and the SFH of 
                the dwarf (tuple containing 2 numpy arrays).
                
                Tuple containing the time values where the 
                derivates are taken and the derivative of the log 
                of the SFH (tuple containing 2 numpy arrays).
        """

                # Dictionary containing histogram data
        dwarfData = self.dataDict[name]             # Selecting dwarf
        tVals, psiData = dwarfData[0], dwarfData[1] # Unpacking data

        dT = (tVals[1:] + tVals[:-1]) * .5                      # time step
        logPsi = np.where(psiData != 0., np.log(psiData), 0)    # ln(psi)

        return (tVals, psiData), (dT, self.diff_complete(logPsi, tVals)*psiData[1:])

    def stair_edges(self, dT, tVals):
        """ Determine edges of stairs function used for plotting SFH.

            Input:
                dT:     Time at which derivatives are taken (numpy array).
                tVals:  Time where the SFH is defined (numpy array).
            
            Returns:
                Location of edges for stairs function (numpy array).
        """

        final = 2 * tVals[-1] - dT[-1]              # Final edge location
        edges1 = np.append([0], [dT])               # Append 0 at beginning

        return np.append([edges1], [final])


    def hist_interp(self, name):
        """ Cubic spline interpolation for histogram data """

        # dwarfData = self.dataDict[name]             # Selecting dwarf
        # tVals, psiData = dwarfData[0], dwarfData[1] # Unpacking data

        data = self.take_deriv(name)        # Retrieve data of dwarf
        tVals, histVals = data[0]           # SFH itself
        dT, derivVals = data[1]             # Derivative of the SFH

        histEdges = self.stair_edges(dT, tVals)  # Edges of histogram

        spline = CubicSpline(tVals, histVals)
        tRange = np.linspace(min(tVals), max(tVals), 100)

            # Plotting
        fig = figure(figsize=(14,7))
        ax = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ax.stairs(histVals, histEdges, color="navy", lw=2, zorder=2.9)
        ax.scatter(tVals, histVals, marker="o", color="navy", s=100, alpha=.3, zorder=3)
        ax.plot(tRange, spline(tRange), color="red", zorder=2.8)

        ax2.plot(dT, derivVals, color="navy", marker="o", ms=10, lw=2)
        ax2.plot(tRange, spline(tRange, 1), color="green")

        ax.grid(zorder=2)
        ax2.grid()

        show()

    

    def plot_hist(self, name, norm=True, saveFig=None):
        """ Plot histogram of SFH for given dwarf galaxy along with the 
            derivative of the SFH.

            Input:
                saveFig:    Option to save figure (None or string).

            Returns:
                -
        """

        data = self.take_deriv(name, norm=norm)  # Retrieve data of dwarf
        tVals, histVals = data[0]           # SFH itself
        dT, derivVals = data[1]             # Derivative of the SFH

        if norm: histVals /= max(histVals)
        
        histEdges = self.stair_edges(dT, tVals)  # Edges of histogram

        matplotlib.rcParams['font.family'] = ['Times']

        # Plotting
        fig = figure(figsize=(14,7))
        ax = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

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
    
    def plot_hist_dwarfs(self, logD=False, *args) -> None:
        """ Plot the SFH histogram data of the 4 dwarfs """
        return None

    def plot_deriv_dwarfs(self, logD=False, *args):
        """ Plot the derivatives of the histogram of the 4 dwarf galaxies.

            Input:
                logD:   use logarithm histogram values (Boolean)

                args:
                    norm:   use normalized histogram data (Boolean)
            
            Returns:
                Plot showing the derivatives of the dwarf galaxies
        """

            # Initialize plot properties for 4 dwarfs
        names = ["Draco", "Ursa Minor", "Sculptor", "Carina"]
        colors = ["navy", "magenta", "red", "teal"]
        lineStyles = ["-", "--", "-.", ":"]
        markers = ["o", "x", "+", "*"]

        if logD:                    # Derivatives of logarithm
            derivDwarfs = [self.take_log_deriv(name, *args)[1] 
                           for name in names]
        
        else:                       # Normal derivatives
            derivDwarfs = [self.take_deriv(name, *args)[1] 
                           for name in names]

        matplotlib.rcParams['font.family'] = ['Times']

        # Plotting
        fig = figure(figsize=(14,7))
        ax = fig.add_subplot(1,1,1)

        for ind, dwarf in enumerate(derivDwarfs):
            ax.plot(dwarf[0], dwarf[1], marker=markers[ind], label=names[ind],  
                    color=colors[ind], ls=lineStyles[ind], ms=10, lw=2)
        
            # Threshold
        # ax.axhline(-.5, color="k")

        ax.set_xlabel(r"$t - t_0$ (Gyr)", fontsize=22)
        ax.set_ylabel(r"d$\psi$ / dt ($10^{-4}$ M$_\odot$ yr$^{-1}$ Gyr$^{-1}$)", fontsize=22)
        ax.tick_params(axis="both", labelsize=24)
        ax.legend(fontsize=22, frameon=False)

        fig.tight_layout()

            # Uncomment to save figure
        # fig.savefig("derivatives.png")
        show()


def main():
    """ Main function that will be executed """

    sfhClass = sfh_dwarfs()     # Initializing class for SHFs

    # sfhClass.plot_hist("Sculptor")
    # sfhClass.plot_all_dwarfs()
    sfhClass.hist_interp("Draco")

if __name__ == "__main__":    
    main()