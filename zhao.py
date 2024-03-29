import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from astropy.constants import kpc

import general as ge


class zhao(object):
    """ Class for generating a potential for a spherical dark matter halo, 
        using the data generated by the model of Zhao (2009). 

        Attributes:

            fName:  The name of the file containing the data generated by the 
                    model of Zhao. The data should be generated using the code 
                    provided at: http://202.127.29.4/dhzhao/mandc.html
            
            data:   The loaded data from the text file; the data is stored in 
                    a 2D numpy array. For more information on which data is 
                    stored, see the README file generated using the code.
            
            h0:     Little h, hubble parameter at z=0 divided by 100, unitless.
                    Value from the Planck collaboration (2020).
            
            red:    The redshift values at which the properties of the dark 
                    matter halo are computed.
            
            mass:   The virial mass of the dark matter halo in units of solar 
                    masses, computed at the redshifts given by red.
            
            conc:   The concentration of the halo (r_vir / r_s) at the given 
                    redshifts.
            
            virR:   The virial radius at the redshift values of the dark 
                    matter halo in meters.

            rhoS:   The density at the scale radius in kg/m^3.

            rS:     The scale radius of the dark matter halo in meters.
            
            time:   The lookback time corresponding to the redshift values in 
                    years.
    """    
    
    def __init__(self, fName):
        """ Initializing the potential according to the model of Zhao
        
            Input:
                fName   (string)
        
            Returns:
                zhao    (object)
        """
        
        self.fName = fName                      # File name
        
        data = self.load_data()
        h0 = 0.6766
        t0 = data[:,-1][0] / h0
        
            # Unpacking data
        self.red = data[:,0]                            # Redshift
        self.mass = data[:,1] / h0                      # Virial mass
        self.conc = data[:,2]                           # Concentration
        self.virR = 1e3 * kpc.value * data[:,4] / h0    # Virial radius
        self.rhoS = ge.conv_inv_dens(data[:,7]/1e18) * h0 * h0   # rho_s
        self.rS = 1e3 * kpc.value * data[:,8] / h0      # r_s
        self.time = t0 - data[:,-1] / h0                # Age of Universe
    
    
    def load_data(self):
        """ Loading data from a generated data file 

            Input:
                -
            
            Returns:
                data:   array containing the properties of the dark matter 
                        halo (2D numpy array).
        """
        with open(self.fName) as f:
            data = np.loadtxt((x.replace('   ', '  ') for x in f), skiprows=1)
        return data
    
    
    def find_z_ind(self, zV):
        """ Input a redshift and find the indices corresponding to the closest 
            redshift value(s) of the generated data. If zV is an array then 
            the closest indices for all the redshift values are determined and 
            returned.

            Input:
                zV:     The redshift value(s) for which the closest index has 
                        to be found (float or numpy array).
            
            Returns:
                The indices corresponding to the closest redshift values 
                (integer or numpy array).
        """
        
        if type(zV) != np.ndarray and type(zV) != list and type(zV) != tuple:
            return ge.find_closest(self.red, zV)[0]
        
        return np.asarray([ge.find_closest(self.red, z)[0] for z in zV])

    
    def rs_rhos_at_z(self, zV):
        """ Find the scale radius (r_s) and the density at the scale radius 
            (rho_s) for a given redshift value. This is done by finding the 
            closest redshift value to the input redshift value(s), NOT by 
            interpolating.

            Input:
                zV:     redshift(s) at which r_s and rho_s will be determined
                        (float or numpy array).
            
            Returns:
                Value of r_s at zV (float or numpy array).
                Value of rho_s at zV (float or numpy array).
        """
        zInd = self.find_z_ind(zV)                      # Correct z index
        return self.rS[zInd], self.rhoS[zInd]
    
    
    def mass_at_r(self, zV, r):
        """ The mass of the dark matter halo as function of distance from the 
            center of the dark matter halo at a given redshift.
        
            Input:
                zV:     redshift(s) at which the mass as function of radius
                        is determined (float or numpy array).
                r:      the distances from the center of the halo at which 
                        the mass will be calculated (float or numpy array).

            Returns:
                mass as function of r and z (float or numpy array (1D or 2D))
        """
        
        rS, rhoS = self.rs_rhos_at_z(zV)                # r_s and rho_s
        
        mass = [rhoS[i] * rS[i] * (np.log(1+r/rS[i]) - r / (rS[i] + r))
                for i in range(len(rS))]
        
        return 16 * np.pi * np.asarray(mass)
    
    
    def simp_profile(self, x):
        """ The NFW profile density profile for the dark matter halo. This 
            function gives rho/rho_s as function of r/r_s. Therefore you do 
            not need to specify the parameters r_s and rho_s. Moreover, this 
            profile is time independent.

            Input:
                x:      r/r_s values, dimensionless (numpy array).
            
            Returns:
                rho/rho_s for the given x values (numpy array).
        """
        return 4 / (x * np.power(1+x, 2))
    
    
    def nfw_profile(self, zV, r):
        """ A time dependent NFW density profile. With the input of the 
            desired redshift value(s), a time dependent density profile 
            as function of radius is output. r_s and rho_s are determined 
            using the model of van den Bosch.
            
            Input:
                zV:     the redshift values at which the density profile 
                        is computed (float or numpy array).
                r:      distance from the center of the halo (float or 
                        numpy array).

            Returns:
                time dependent NFW profile (float or numpy array(1D or 2D))
        """
        
        zInd = self.find_z_ind(zV)                          # Selecting z ind
        
        rS = self.rS                                        # r_s
        rhoS = self.rhoS                                    # rho_s
        
        if type(zInd) != np.ndarray:                        # Single z value
            denom = r * np.power(1 + r/rS[zInd], 2) / rS[zInd]
            return 4 * rhoS[zInd] / denom
        
        selrS, selrhoS = rS[zInd], rhoS[zInd]               # Slicing lists
        
            # Multiple z values
        frac = [selrhoS[ind] / (r * np.power(1 + r/selrS[ind], 2) / selrS[ind]) 
                for ind in range(len(selrS))]
        
        return 4 * np.asarray(frac)
    
    def pot_nfw(self, zV, r):
        """ The gravitational potential corresponding to the NFW 
            density profile. This is obtained by solving the Poisson 
            equation. For the NFW profile there exists an analytical 
            solution.
        
            Input:
                zV:     the redshift values at which the potential is 
                        computed (float or numpy array).
                r:      distance from the center of the halo (float or 
                        numpy array).

            Returns:
                gravitational potential (float or numpy array(1D or 2D))
        """
        
        rhoS = self.rhoS                                    # rho_s
        rS = self.rS                                        # r_s
        
        zInd = self.find_z_ind(zV)                          # Finding correct z
        
        if type(zInd) != np.ndarray:                        # Single z value
            
               # Need to interpolate due to coarse grid
            interpRS = interp1d(self.red, self.rS)      # Creating r_s int. object
            selrS = interpRS(zV)                        # Interpolating r_s

            interpRhoS = interp1d(self.red, self.rhoS)  # Creating rho_s int. object
            selrhoS = interpRhoS(zV)                    # Interpolating rho_s
            
            part1 = -16 * np.pi * G * selrhoS * selrS * selrS
            part2 = np.log(1 + r/selrS) / (r/selrS)
            
            return part1 * part2
        
            # Multiple z values
        selrS, selrhoS = rS[zInd], rhoS[zInd]                   # Slicing lists
        
        part1 = -16 * np.pi * G * selrhoS * selrS * selrS       # First part
        part2 = [np.log(1 + r/rsV) / (r/rsV) for rsV in selrS]  # Second part
        
        phi = [part1[ind] * part2[ind] for ind in range(len(zInd))] # Potential
        
        return np.asarray(phi)



