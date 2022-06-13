import numpy as np
from scipy.constants import G
from astropy.constants import M_sun

import mass_change as mc
import general as ge


class bosch(object):
    """ Class for van den Bosch model """
    
    def __init__(self, fName, M0):
        """ Initializing """
        
        self.fName = fName
        self.M0 = M0
        
        data = self.load_data()                         # Loading data
        
            # Unpacking data
        self.red = data[:,1]                            # Redshift
        self.mass = np.power(10, data[:,3]) * self.M0   # Virial mass
        self.conc = data[:,6]                           # Concentration
        self.time = data[:,2] * 1e9                     # Lookback time (yr)
    
    
    def load_data(self):
        """ Loading data """
        
        with open(self.fName) as f:                     # Opening data
            data = np.loadtxt((x.replace('     ', '  ') for x in f))
        return data
    
    
    def vir_rad(self, *args):
        """ Find the virial radius """
        
        red = self.red                                  # Redshift
        deltaC = ge.crit_vir_dens(red)                  # Delta_c
        hubble = ge.hubble_para(red, *args)             # Hubble parameter
        
        hUnits = hubble * 1e3/3.0857e22            # Units: s-1
        
        num = 2 * G * self.mass * M_sun.value
        denom = deltaC * np.power(hUnits, 2)
        
        return np.power(num / denom, 1/3)
    
    
    def rS(self, *args):
        """ Find the scale length """
        return self.vir_rad(*args) / self.conc
    
    def rhoS(self, *args):
        """ Density at scale length """
        
        conc = self.conc                                    # Concentration
        const = 16 * np.pi * np.power(self.rS(*args), 3)    # Constant term
        brack = np.log(1 + conc) - conc / (1 + conc)        # Bracket term
        
        return self.mass * M_sun.value / (const * brack)
    
    def find_z_ind(self, zV):
        """ Find indices corresponding to redshift values """
        
        if type(zV) != np.ndarray and type(zV) != list and type(zV) != tuple:
            return ge.find_closest(self.red, zV)[0]
        
        return np.asarray([ge.find_closest(self.red, z)[0] for z in zV])
    
    def rs_rhos_at_z(self, zV, *args):
        """ Find scale length and density at r_s at specific z value """
#         zInd = ge.find_closest(self.red, zV)[0]     # Correct z index
        zInd = self.find_z_ind(zV)
        return self.rS(*args)[zInd], self.rhoS(*args)[zInd]
    
    
    def mass_at_r(self, zV, r, *args):
        """ Function M(r) """
        
        rS, rhoS = self.rs_rhos_at_z(zV, *args)         # r_s and rho_s
        
        mass = [rhoS[i] * rS[i] * (np.log(1+r/rS[i]) - r / (rS[i] + r))
                for i in range(len(rS))]
        
        return 16 * np.pi * np.asarray(mass)
    
    
    def simp_profile(self, x):
        """ Simple NFW profile """
        return 4 / (x * np.power(1+x, 2))
    
    def nfw_profile(self, zV, r, *args):
        """ Full NFW profile """
        
        zInd = self.find_z_ind(zV)                          # Selecting z ind
        
        rS = self.rS(*args)
        rhoS = self.rhoS(*args)
        
        if type(zInd) != np.ndarray:                        # Single z value
            denom = r * np.power(1 + r/rS[zInd], 2) / rS[zInd]
            return 4 * rhoS[zInd] / denom
        
        selrS, selrhoS = rS[zInd], rhoS[zInd]               # Slicing lists
        
                    # Multiple z values
        frac = [selrhoS[ind] / (r * np.power(1 + r/selrS[ind], 2) / selrS[ind]) 
                for ind in range(len(selrS))]
        
        return 4 * np.asarray(frac)
    
    def pot_nfw(self, zV, r, *args):
        """ Potential of the NFW profile """
        
        rhoS = self.rhoS(*args)                             # rho_s
        rS = self.rS(*args)                                 # r_s
        
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




