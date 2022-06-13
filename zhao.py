import numpy as np
from scipy.constants import G

import mass_change as mc
import general as ge


class zhao(object):
    """ Class for the zhao model """
    
    
    def __init__(self, fName):
        """ Initializing """
        
        self.fName = fName                      # File name
        
        data = self.load_data()
        h0 = 0.6766
        
            # Unpacking data
        self.red = data[:,0]                            # Redshift
        self.mass = data[:,1] / h0                      # Virial mass
        self.conc = data[:,2]                           # Concentration
        self.virR = 3.0857e22 * data[:,4] / h0          # Virial radius
        self.rhoS = ge.conv_inv_dens(data[:,7]/1e18) * h0 * h0   # rho_s
        self.rS = 3.0857e22 * data[:,8] / h0                     # r_s
        self.time = data[:,-1] / h0                     # Age of Universe
    
    
    def load_data(self):
        """ Load data from file """
        
        with open(self.fName) as f:
            data = np.loadtxt((x.replace('   ', '  ') for x in f), skiprows=1)
        return data
    
    
    def find_z_ind(self, zV):
        """ Find indices corresponding to redshift values """
        
        if type(zV) != np.ndarray and type(zV) != list and type(zV) != tuple:
            return ge.find_closest(self.red, zV)[0]
        
        return np.asarray([ge.find_closest(self.red, z)[0] for z in zV])

    
    def rs_rhos_at_z(self, zV):
        """ Find scale length and density at r_s at specific z value """
#         zInd = ge.find_closest(self.red, zV)[0]     # Correct z index
        zInd = self.find_z_ind(zV)
        return self.rS[zInd], self.rhoS[zInd]
    
    def mass_at_r(self, zV, r):
        """ Function M(r) """
        
        rS, rhoS = self.rs_rhos_at_z(zV)                # r_s and rho_s
        
        mass = [rhoS[i] * rS[i] * (np.log(1+r/rS[i]) - r / (rS[i] + r))
                for i in range(len(rS))]
        
        return 16 * np.pi * np.asarray(mass)
    
    
    def simp_profile(self, x):
        """ Simple NFW profile """
        return 4 / (x * np.power(1+x, 2))
    
    
    def nfw_profile(self, zV, r, *args):
        """ Full NFW profile """
        
        zInd = self.find_z_ind(zV)                          # Selecting z ind
        
        rS = self.rS
        rhoS = self.rhoS
        
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
        
        rhoS = self.rhoS                                    # rho_s
        rS = self.rS                                        # r_s
        
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



