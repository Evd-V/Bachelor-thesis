import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.constants import M_sun
from matplotlib.pyplot import figure, vlines, show

import mass_change as mc
import concentration as co
import general as ge
import nfw_profile as nf


class nfw_potential(object):
    """ Class for the NFW profile """
    
    def __init__(self, redshift, mass):
        """ Initializing """
        
        self.red = redshift
        self.mass = mass
        
        self.littleh = 0.6766
    
    
    def get_vir_rad(self, *args):
        """ Virial radius """
        return mc.virial_rad(self.red, self.mass, *args)
    
#     def conc_nfw(self):
#         """ Concentration """
#         
#         z = self.red
#         M = self.mass / self.littleh
#         
#         aPara = 0.537 + 0.488 * np.exp(-0.718 * np.power(z, 1.08))  # a
#         bPara = -0.097 + 0.024 * z                                  # b
#         logC = aPara + bPara * np.log10(M / 1e12)       # M in units h^-1 M_sun
#         
#         return np.power(10, logC)
    
    def table(self):
        """ Table values """
        
        zVals = [0, 0.35, 0.5, 1, 1.44, 2.15, 2.5, 2.9, 4.1, 5.4]
#         CVals = [9.75, 7.25, 6.5, 4.75, 3.8, 3, 2.65, 2.42, 2.1, 1.86]
#         gVals = [0.11, 0.107, 0.105, 0.1, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
#         M0Vals = [5e5, 2.2e4, 1e4, 1e3, 210, 43, 18, 9, 1.9, .42]
        
        CVals = [10.2, 7.85, 7.16, 5.45, 4.55, 3.55, 3.24, 2.92, 2.6, 2.3]
        gVals = [0.1, 0.095, 0.092, 0.088, 0.085, 0.08, 0.08, 0.08, 0.08, 0.08]
        M0Vals = [1e5, 1.2e4, 5.5e3, 700, 180, 30, 15, 7.0, 1.9, 0.36]
        
        return np.asarray(zVals), np.asarray(CVals), np.asarray(gVals), np.asarray(M0Vals)
    
    
    def interp_vals(self):
        """ Interpolate table values """
        
        tabVals = self.table()
        redshift = self.red
        
        maxInd = ge.find_closest(redshift, max(tabVals[0]))[0]
        
        if maxInd == len(redshift):
            interpObj = [interp1d(tabVals[0], tab, kind="cubic") for tab in tabVals[1:]]
            interpVals = [iObj(redshift) for iObj in interpObj]
            return np.asarray(interpVals)
        
        interpObj = [interp1d(tabVals[0], tab, fill_value='extrapolate', 
                     kind="cubic") for tab in tabVals[1:]]
        
        intVals = [iObj(redshift) for iObj in interpObj]
        
        return np.asarray(intVals)
    
#     def conc_nfw(self):
#         """ Another test """
#         
#         intVals = self.interp_vals()
#         cV, gV, M0V = intVals[0], intVals[1], 1e12 * intVals[2] * self.littleh
#         M = self.mass / self.littleh
#         
#         part1 = cV * np.power(M/1e12, -gV)
#         part2 = 1 + np.power(self.mass/M0V, 0.4)
#         
#         return part1 * part2
    
    def conc_nfw(self, *args):
        """ Third test """
        
        red = self.red
        xV = co.x(red, *args)
        return co.B0(xV) * co.c_func(red, self.mass, *args)
    
    
#     def conc_nfw(self):
#         """ Test concentration """
#         
#         M = self.mass * self.littleh
#         
#         return 9.6 * np.power(M/1e12, -0.075)
    
    def scale_rad(self, *args):
        """ Scale radius for the NFW profile """
        
        virR = self.get_vir_rad(*args)                      # Virial radius
        conc = self.conc_nfw()                              # Concentration
        
        return virR / conc
    
    def rho_rs(self, *args):
        
        rS = self.scale_rad(*args)                          # Scale length
        conc = self.conc_nfw()                              # Concentration
        virM = self.mass                                    # Virial mass
        
        part1 = np.log(1 + conc) - conc / (1 + conc)        # First part
        denom = 16 * np.pi * np.power(rS, 3) * part1        # Denominator
        rhoS = virM * M_sun.value / denom                   # rho_s
        
        return rhoS
    
    
    def rs_at_z(self, zV, *args):
        """ Find the scale length at a given redshift """
        
        rSTot = self.scale_rad(*args)                       # r_s at all z
        zInd, zVal = ge.find_closest(self.red, zV)          # Correct z index
        
        return rSTot[zInd]
    
    def rhos_at_z(self, zV, *args):
        """ Find rho_s at a given redshift """
        
        zInd = self.find_z_ind(zV)
        rhoS = self.rho_rs(*args)
        
        return rhoS[zInd]
    
    
    def find_z_ind(self, zV):
        """ Find indices corresponding to redshift values """
        
        if type(zV) != np.ndarray and type(zV) != list and type(zV) != tuple:
            return ge.find_closest(self.red, zV)[0]
        
        return np.asarray([ge.find_closest(self.red, z)[0] for z in zV])
    
    
    def simp_profile(self, x):
        """ Simple NFW profile """
        return 4 / (x * np.power(1+x, 2))
    
    
    def nfw_profile(self, zV, r, *args):
        """ Full NFW profile """
        
        rS = self.scale_rad(*args)                          # Scale length
        rhoS = self.rho_rs(*args)                           # rho_s
        c = self.conc_nfw()                                 # Concentration
        
        zInd = self.find_z_ind(zV)                          # Finding correct z
        selrS, selrhoS = rS[zInd], rhoS[zInd]               # Slicing lists
        
        if type(zInd) != np.ndarray:                        # Single z value
            denom = r * np.power(1 + r/rS[zInd], 2) / rS[zInd]
            return 4 * rhoS[zInd] / denom
        
            # Multiple z values
        frac = [selrhoS[ind] / (r * np.power(1 + r/selrS[ind], 2) / selrS[ind]) 
                for ind in range(len(selrS))]
        
        return 4 * np.asarray(frac)
    
    
    def pot_nfw(self, zV, r, *args):
        """ Potential of the NFW profile """
        
        rhoS = self.rho_rs(*args)                           # rho_s
        rS = self.scale_rad(*args)                          # r_s
        c = self.conc_nfw()                                 # Concentration
        
        zInd = self.find_z_ind(zV)                          # Finding correct z
        selrS, selrhoS = rS[zInd], rhoS[zInd]               # Slicing lists
        
        if type(zInd) != np.ndarray:                        # Single z value
            part1 = -16 * np.pi * G * selrhoS * selrS * selrS
            part2 = np.log(1 + r/selrS) / (r/selrS)
            return part1 * part2
        
            # Multiple z values
        part1 = -16 * np.pi * G * selrhoS * selrS * selrS       # First part
        part2 = [np.log(1 + r/rsV) / (r/rsV) for rsV in selrS]  # Second part
        
        phi = [part1[ind] * part2[ind] for ind in range(len(zInd))] # Potential
        
        return np.asarray(phi)
    
    
    def pot_z_r(self, zV, rV, *args):
        """ Potential at a specific z and r values """
        
        rS = self.rs_at_z(zV, *args)                        # Scale length
        rhoS = self.rhos_at_z(zV, *args)                    # rhos_s
        
#         rS = self.scale_rad(*args)
#         rhoS = self.rho_rs(*args)
#         conc = self.conc_nfw()
        
        part1 = -16 * np.pi * G * rhoS * rS * rS            # First part
        part2 = np.log(1 + c) / c                           # Second part
        
        phi = part1 * part2                                 # Potential
        
        return phi
    
    
    def rho_delta(self, *args):
        """ Find the density at the virial radius """
        
        z = self.red                                        # Redshift
        virR = self.get_vir_rad(*args)                      # Virial radii
        rS = self.scale_rad()                               # Scale rad
        rhoS = self.rho_rs()                                # rho_s
        
        dens = []
        
        for ind, zV in enumerate(z):
            
            c = virR[ind] / rS[ind]                         # Concentration
            denom = c * np.power(1+c, 2)                    # Denominator
            dens.append(4 * rhoS[ind] / denom)              # Density profile
        
        return dens
    
    def mass_range(self, zV, r, *args):
        """ Find the mass as function of radius """
        
        dens = self.nfw_profile(zV, r, *args)               # Density profiles
        vol = 4 * np.pi * np.power(r, 3) / 3                # Volume
        
        return dens * vol
    
    
    def test_mr(self, zV, r, *args):
        """ New mass as function of r """
        
        dens = self.nfw_profile(zV, r, *args)               # Density profiles
        
        def some_func(rV, densV):
            return rV * rV * densV
        
        integ = quad(some_func, 0, r, args=(dens))
        
        return 4 * np.pi * integ[0]
    
    def test_mrange(self, zV, rRange, *args):
        """ M(r) """
        
        masses = [self.test_mr(zV, r, *args) for r in rRange]
        return masses
    
    
    def find_vir_mass(self, zV, r, *args):
        """ Find virial mass as check """
        
        rS = self.scale_rad(*args)                          # Scale length
        rhoS = self.rho_rs(*args)                           # rho_s
        virR = self.get_vir_rad(*args)                      # Virial radius
        
        zInd = self.find_z_ind(zV)                          # Finding correct z
        selrS, selrhoS = rS[zInd], rhoS[zInd]               # Slicing lists
        
        if type(zInd) != np.ndarray:
            const = 16 * np.pi * rhoS[zInd] * np.power(rS[zInd], 3)
            bracket = np.log(1 + r/selrS) - r / (selrS * (1 + r/selrS))
            return const * bracket
        
            # Constant factor
        const = [selrhoS[i] * np.power(selrS[i], 3) for i in range(len(selrS))]
        bracket = [np.log(1 + r/rsV) - r / (rsV * (1 + r/rsV)) for rsV in selrS]
        
        massTerm = [const[i] * bracket[i] for i in range(len(const))]
        
        return 16 * np.pi * np.asarray(massTerm)


def main():

    M0 = 1.25e12
    
        # Van den Bosch
    fName = "./getPWGH/PWGH_average_125e12.dat"
    boschRed, boschMass, boschRate = mc.mah_bosch(fName, M0)    # van den Bosch
    
#     boschRed = boschRed[:500]
#     boschMass = boschMass[:500]
    
        # Zhao
    zhaoFile = "./mandc_m125e12/mandcoutput.m125e12"
    zhao = mc.read_zhao(zhaoFile)                               # Zhao data
    
    cutInd, cutRedVal = ge.find_closest(zhao[0], max(boschRed))
    zhaoRed, zhaoMass = zhao[0][:cutInd+1], zhao[1][:cutInd+1]    # Redshift & M
    
    zhaoMass = nf.find_m_z(zhao[0], zhao[1]/.6766, boschRed)  # Interpolating
    
        # r values and redshift values
    rRange = np.linspace(0.5e19, 6e21, 500)
    zRange = np.linspace(0, 5, 3)
    colors = ["navy", "crimson", "green"]
    
        # Potential properties for van den Bosch model
    boschProfile = nfw_potential(boschRed, boschMass)
    
    rSBosch = boschProfile.scale_rad()                          # Scale radius
    rhoSB = boschProfile.rho_rs()                               # rho_s
    virRadBosch = boschProfile.get_vir_rad()                    # Virial rad.
    rhoDelBosch = boschProfile.rho_delta()                      # rho_delta
    cBosch = boschProfile.conc_nfw()                            # Concentration
    
    potBosch = boschProfile.pot_nfw(zRange, rRange)             # Grav. pot.
    profileBosch = boschProfile.nfw_profile(zRange, rRange)     # Density
    massBosch = boschProfile.find_vir_mass(zRange, rRange)      # M(r)
    
    
        # Potential properties for Zhao model
    zhaoProfile = nfw_potential(boschRed, zhaoMass)
    
    rSZhao = zhaoProfile.scale_rad()
    rhoSZ = zhaoProfile.rho_rs()                                # rho_s
    virRadZhao = zhaoProfile.get_vir_rad()                      # Virial rad.
    rhoDelZhao = zhaoProfile.rho_delta()                        # rho_delta
    cZhao = zhaoProfile.conc_nfw()                              # Concentration
    
    potZhao = zhaoProfile.pot_nfw(zRange, rRange)               # Grav. pot.
    profileZhao = zhaoProfile.nfw_profile(zRange, rRange)       # Density
    massZhao = zhaoProfile.find_vir_mass(zRange, rRange)        # M(r)
    
        # Time independent
    xRange = rRange / rSBosch[0]
    tIndepRho = boschProfile.simp_profile(xRange)
    
    
        # Zhao checking stuff
    with open(zhaoFile) as f:
        data = np.loadtxt((x.replace('   ', '  ') for x in f), skiprows=1)
    
    conc, virR = data[:,2], data[:,4]
    rS, rhoS = data[:,8], data[:,7]
    
    newRS = nf.find_m_z(zhao[0], rS/.6766, boschRed)*1e6
    newRhoS = nf.find_m_z(zhao[0], rhoS*.6766*.6766, boschRed)/1e18
    newConc = nf.find_m_z(zhao[0], zhao[2], boschRed)
    
    kpcRange = ge.conv_m_kpc(rRange)*1e3
    indic = [ge.find_closest(boschRed, z)[0] for z in zRange]
    
    newDensZhao = [4 * newRhoS[i] / ((kpcRange/newRS[i]) * np.power(1+kpcRange/newRS[i], 2))
                   for i in indic]
    
    with open(fName) as f:                               # Opening data
        Bdata = np.loadtxt((x.replace('     ', '  ') for x in f))
    
    newCB = Bdata[:,6]
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
#     ax2 = fig.add_subplot(1,2,2)
    
        # Time indep: rho/rho_s (r/r_s)
#     ax.plot(xRange, tIndepRho, color="navy")
    
        # Virial radius
#     ax.plot(boschRed, ge.conv_m_kpc(virRadZhao), label="Zhao (2009)", ls="--", 
#             color="navy")
#     ax.plot(boschRed, ge.conv_m_kpc(virRadBosch), label="van den Bosch (2014)", 
#             color="red")
    
        # r_s
#     ax.plot(boschRed, ge.conv_m_kpc(rSZhao), label="Zhao (2009)", color="navy")
#     ax.plot(boschRed, ge.conv_m_kpc(rSBosch), label="van den Bosch (2014)", 
#             color="red")
#     
#     ax.plot(boschRed, newRS/1e3, label="Direct", color="green")
    
        # rho_s
#     ax2.plot(boschRed, ge.conv_dens(rhoSZ), color="red", ls="--")
#     ax2.plot(boschRed, ge.conv_dens(rhoSB), color="navy", ls="--")
    
#     ax.plot(boschRed, ge.conv_dens(rhoSZ), label="Zhao (2009)", color="red", ls="--")
#     ax.plot(boschRed, ge.conv_dens(rhoSB), label="van den Bosch (2014)", color="navy", ls="--")
#     
#     ax.plot(boschRed, newRhoS, label="Direct", color="green")
    
        # r_s vs vir rad
#     ax.plot(ge.conv_m_kpc(rSZhao), ge.conv_m_kpc(virRadZhao), color="navy", 
#             label="Zhao (2009)")
#     ax.plot(ge.conv_m_kpc(rSBosch), ge.conv_m_kpc(virRadBosch), color="red", 
#             label="van den Bosch (2014)")
    
        # rho(r)
#     for ind, prof in enumerate(profileBosch):
#         ax.plot(ge.conv_m_kpc(rRange), ge.conv_dens(prof), color=colors[ind], 
#                 label=f"z={zRange[ind]}")
#         
#         ax.plot(ge.conv_m_kpc(rRange), ge.conv_dens(profileZhao[ind]), 
#                 color=colors[ind], ls="--")
#         
#         
#         ax.plot(ge.conv_m_kpc(rRange), newDensZhao[ind], 
#                 color=colors[ind], ls=":")
    
        # rho_delta (z)
#     ax.plot(boschRed, ge.conv_dens(rhoDelZhao), label="Zhao (2009)", color="navy")
#     ax.plot(boschRed, ge.conv_dens(rhoDelBosch), label="van den Bosch (2014)", 
#             color="red")
    
        # c(z)
    ax.plot(boschRed, cZhao, label="Zhao (2009)", color="navy")
    ax.plot(boschRed, cBosch, label="van den Bosch (2014)", color="red")
    
    ax.plot(boschRed, newConc, label="Direct Zhao", color="green")
    ax.plot(boschRed, newCB, label="Direct Bosch", color="black")
    
    
#     print(ge.conv_m_kpc(virRadBosch[0]))
#     print(ge.conv_m_kpc(virRadBosch[277]))
#     print(ge.conv_m_kpc(virRadBosch[555]))
    
        # M(r)
#     for ind, mass in enumerate(massBosch):
#         ax.plot(ge.conv_m_kpc(rRange), mass / M_sun.value, color=colors[ind],
#                 label=f"z={zRange[ind]}")
#         ax.plot(ge.conv_m_kpc(rRange), massZhao[ind] / M_sun.value, 
#                 color=colors[ind], ls="--")
    
        # M(z)
#     ax2.plot(boschRed, zhaoMass, ls="--", color="navy", label="Zhao (2009)")
#     ax2.plot(boschRed, boschMass, color="red", label="van den Bosch (2014)")
    
        # phi(r)
#     for ind, pot in enumerate(potBosch):
#         ax.plot(ge.conv_m_kpc(rRange), potZhao[ind], color=colors[ind], ls="--")
#         ax.plot(ge.conv_m_kpc(rRange), pot, color=colors[ind], 
#                 label=f"z={zRange[ind]}")
    
    
    ax.set_xlabel(r"$z$", fontsize=15)
    ax.set_ylabel(r"$\rho$ (M$_\odot$ / pc$^3$)", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
#     ax.set_title("Dashed = Zhao (2009), solid = van den Bosch (2014)", 
#                  fontsize=15)
    
#     ax2.set_xlabel(r"$z$", fontsize=15)
#     ax2.set_ylabel(r"$M_\Delta$ (M$_\odot$)", fontsize=15)
#     ax2.tick_params(axis="both", labelsize=15)
    
#     ax2.set_ylabel(r"$\rho_s$ (M$_\odot$ / pc$^3$)", fontsize=15)
#     ax2.tick_params(axis="y", labelsize=15)
    
#     ax.set_yscale("log")
#     ax2.set_yscale("log")
    
    ax.grid()
    ax.legend(fontsize=15)
#     ax.legend(bbox_to_anchor=(.8, 1.1), ncol=2, frameon=False, fontsize=15)
    
#     ax2.grid()
#     ax2.legend(fontsize=15)
    
    fig.tight_layout()
#     fig.savefig("rs_virrad.png")
    
    show()

if __name__ == "__main__":
    main()
