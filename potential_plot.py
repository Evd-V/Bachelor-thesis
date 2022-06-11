import numpy as np
from matplotlib.pyplot import figure, show

# import galpy.potential as gp
import nfw_profile as nf
import einasto as ei


# Galpy density profile
# NFW = gp.NFWPotential(amp=1, a=rS)
# densities = gp.evaluateDensities(NFW, r, 0)


# Obtaining NFW density profile
r = np.linspace(0.01, 10, int(1e3))                 # Radius values
rS = 1                                              # Scale length
virM = 1e12                                         # Virial mass
alpha = 0.2                                         # alpha for Einasto

rOverRs = np.linspace(0.01, 2, int(1e3))

alphRange = np.linspace(0.1, 1, 5)

# Density profiles
rhoNFW = nf.NFW_profile(r, rS, virM)                # NFW profile
rhoSNFW = nf.rho_s(0, rS, virM)                     # rhoS for NFW

newNFW = nf.test_nfw(rOverRs)
newEinasto = ei.test_einasto(rOverRs, 1e14)

rhoEinasto = ei.einasto_virial(r, rS, alpha, virM)  # Einasto profile
rhoSEinasto = ei.rho_s_einasto(r, rS, alpha, virM)  # rhoS for Einasto

alphRhoEin = [ei.einasto_virial(r, rS, a, virM) / ei.rho_s_einasto(r, rS, a, virM)
              for a in alphRange]


# Gravitational potential
redshift = 0
redRange = np.linspace(0, 5, 6)
nfwGrav = [nf.grav_pot(r, z, rS, virM) for z in redRange]


# Critical overdensity for virilisation
zRange = np.linspace(0, 10, 250)
cDens = nf.crit_vir_dens(zRange)

distVals = [0.1, 0.5, 1, 1.5]
densZ = [nf.NFW_profile(rV, rS, virM, z=zRange) for rV in distVals]

phiS = [nfwGrav[0][0], 8e3]
einPotential = ei.solv_poisson(phiS, r)

# Plotting
fig = figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)

# --------------
# Einasto profile
# --------------

# massRange = np.linspace(10, 15.5, 1000)
# masses = np.power(10, massRange)
# redRange = np.linspace(0, 5, 6)
# 
# redShifts = np.linspace(0, 5, 1000)     # Redshift range
# mTest = 1e14                            # Units of h^-1 M_sun
# 
# alphas = [ei.alpha_einasto(masses, z) for z in redRange]
# concen = [ei.conc_einasto(masses, z) for z in redRange]
# densEin = ei.new_einasto_profile(mTest, redShifts)
# densNFW = nf.new_nfw_profile(mTest, redShifts)
# 
# ax.plot(redShifts, densEin, label="Einasto")
# ax.plot(redShifts, densNFW, label="NFW")


# for ind, alph in enumerate(alphas):
# #    ax.plot(massRange, alph, label=f"z={redRange[ind]}")
#     ax.plot(massRange, concen[ind], label=f"z={redRange[ind]}")


# -----------
# Gravitational potentials
# -----------
# for ind, grav in enumerate(nfwGrav):
#     ax.plot(np.log10(r), grav, label=f"z={redRange[ind]}")
# 
# ax.plot(np.log10(r), nfwGrav[0], label="NFW", color="Navy")
# ax.plot(np.log10(r), einPotential, label="Einasto potential")

#-------------
# Densities
#-------------
# ax.plot(np.log10(r), np.log10(rhoNFW/rhoSNFW), label="NFW", color="navy", 
#         zorder=3, lw=2)

ax.plot(rOverRs, newNFW, label="NFW")
ax.plot(rOverRs, newEinasto, label="Einasto")

# 
# for ind, p in enumerate(alphRhoEin):
#     ax.plot(np.log10(r), np.log10(p), label=f"Einasto {alphRange[ind]:.2f}")

# ax.plot(r, densities/densities[0], label="Galpy", color="crimson")

#----------
# Other stuff
#----------
#ax.plot(r, nf.gamma_nfw(r, rS), color="seagreen", label=r"$\gamma$")

# for ind, den in enumerate(densZ):
#     ax.plot(zRange, den, label=f"r={distVals[ind]}")
#ax.plot(zRange, cDens, label="Critical density", color="forestgreen")

#ax.set_xlabel(r"$\log_{10} M_{200}$", fontsize=20)
ax.set_xlabel(r"$r / r_s$", fontsize=20)
ax.set_ylabel(r"$\rho / \rho_s$", fontsize=20)


ax.set_xscale("log")
# ax.set_yscale("log")

ax.grid()
ax.legend(fontsize=18)

show()