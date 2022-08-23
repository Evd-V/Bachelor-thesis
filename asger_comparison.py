import numpy as np
from astropy.constants import M_sun
from matplotlib.pyplot import figure, show
import matplotlib

from zhao import zhao
import general as ge


def load_asger(fName="haloevol.dat"):
    """ Load data from file of Asger """

    fData = np.loadtxt(fName)                   # Reading data
    h0 = .674                                   # Value of h_0

        # Unpacking data
    asTime, asRed = fData[:,0]/h0, fData[:,1]
    asVirM, asConc, asVirR = fData[:,2]/h0, fData[:,3], 1e3*fData[:,4]/h0

    asRs = ge.conv_kpc_m(asVirR) / asConc       # Scale radius

        # Calculating rho_s
    const = 16 * np.pi * np.power(asRs, 3)      # Constant
    brackets = np.log(1+asConc) - asConc / (1 + asConc)

    asRhoS = asVirM * M_sun.value / (const * brackets)  # Density at r_s
    rhoSUnits = ge.conv_dens(asRhoS)             # SI units

    return asTime, asRed, asVirM, asConc, asVirR, asVirR/asConc, rhoSUnits


def load_evan(fName="./mandc_asger/mandcoutput.asger"):
    """ Load data of my own file """

    zData = zhao(fName)                 # Initialising object

        # Loading data
    time, red = zData.time, zData.red
    mass, rad = zData.mass, ge.conv_m_kpc(zData.virR)
    conc = zData.conc
    rS, rhoS = ge.conv_m_kpc(zData.rS), ge.conv_dens(zData.rhoS)

    return time, red, mass, conc, rad, rS, rhoS


def compare(varX, varY, *args):
    """ Compare data of Asger and Evan """

        # Names of variables
    variables = {
                 "Time" : 0,
                 "Redshift" : 1,
                 "Mass" : 2,
                 "Concentration" : 3,
                 "Radius" : 4,
                 "Rs" : 5,
                 "RhoS" : 6
                 }

        # Assigning ints to variable names
    indX = variables[varX]      # Int for varX
    indY = variables[varY]      # Int for varY

        # Retrieving data
    if len(args) == 0:
        asgerData = load_asger()
        evanData = load_evan()

    elif len(args) == 2:
        asgerData = load_asger(fName=args[0])
        evanData = load_evan(fName=args[1])
    
    else:
        raise Exception("Invalid number of arguments")

        # Selecting correct data
    asgerVars = (asgerData[indX], asgerData[indY])
    evanVars = (evanData[indX], evanData[indY])

    return asgerVars, evanVars

def plot_comparison(varX, varY, saveFig=None, *args):
    """ Plot a comparison of the data """

        # Retrieving data
    asgData, evData = compare(varX, varY, *args)

    matplotlib.rcParams['font.family'] = ['Times']

        # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)

    ax.plot(*asgData, color="navy", label="Asger")
    ax.plot(*evData, color="r", label="Evan")

    ax.set_xlabel(varX, fontsize=20)
    ax.set_ylabel(varY + r"(kpc)", fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    ax.yaxis.offsetText.set_fontsize(22)

    # ax.set_yscale("log")
    # ax.set_xlim(0, 200)
    ax.set_ylim(0, 30)

    ax.grid()
    ax.legend(fontsize=20)

    if saveFig: fig.savefig(saveFig)
    show()

plot_comparison("Radius", "Rs", saveFig="rs_rvir_comp.png")

