import numpy as np
from scipy.constants import G
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure, show, cm

import general as ge


def leap_frog(func, dt, x0, vh, *args):
    """ The iterative scheme for the leapfrog integration method.
        Solves an ordinary second order differential equation of the 
        form d^2 y / dt = f(y).

        Input:
            func:   Function that will be integrated, should take the 
                    integration variable as first input (function).
            dt:     The time step for the iterative scheme (float).
            x0:     Initial condition of x at t=i (float).
            vh:     Initial condition of v (dx/dt) at t=i-1/2 (float).
            *args:  Extra arguments passed to the function.
        
        Returns:
            xN:     The value of x at the time t=i+1 (float).
            vHalf:  The value of dx/dt (v) at time t=i+1/2 (float).
    """
    
    fN = func(x0, *args)                    # f(x_0)
    vHalf = vh + fN * dt                    # v_{i-1/2}
    xN = x0 + vHalf * dt                    # x_{i+1}
    
    return xN, vHalf


def alt_leap(func, dt, x0, v0, *args):
    """ Alternative form of leapfrog method to solve ordinary 
        second order differential equations. In contrast to the 
        method above, this method does not require the initial 
        velocity at t=i-1/2, but at t=i, which is generally more 
        convenient.

        Input:
            func:   Function that will be integrated, should take the 
                    integration variable as first input (function).
            dt:     The time step for the iterative scheme (float).
            x0:     Initial condition of x at t=i (float).
            v0:     Initial condition of v (dx/dt) at t=i (float).
            *args:  Extra arguments passed to the function.
        
        Returns:
            xN:     The value of x at the time t=i+1 (float).
            vN:     The value of dx/dt (v) at time t=i+1 (float).
            
    """
    
    a0 = func(x0, *args)                    # a_i
    xN = x0 + v0 * dt + 0.5 * a0 * dt * dt  # x_{i+1}
    
    aN = func(xN, *args)                    # a_{i+1}
    vN = v0 + 0.5 * (a0 + aN) * dt          # v_{i+1}
    
    return xN, vN


def execute_leap(func, tRange, p0, v0, z, *args):
    """ Execute the leapfrog integration method by calling the iterative 
        scheme in a loop. Here the "alternative" leapfrog method is used 
        as it does not require the initial velocity at t=i-1/2. This 
        function can be used to differentiate the orbits of dwarf 
        galaxies back in time for a static gravitational potential.

        Input:
            func:   Function that will be integrated, should take the 
                    integration variable as first input (function).
            tRange: The points in time where the second order differential 
                    equation has to be solved (numpy array).
            p0:     Initial 3D position, e.g. in Cartesian coordinates, at 
                    the time of the first entry of tRange (3D numpy array).
            v0:     Initial 3D velocity, e.g. in Cartesian coordinates, at 
                    the time of the first entry of tRange (3D numpy array).
            z:      Redshift for a given time at which the equation will be 
                    solved (numpy array).
        
        Returns:
            pV:     The 3D position vector at the times given by tRange 
                    (numpy array with shape (len(tRange), 3)).
            vV:     The 3D velocity vector at the times given by tRange 
                    (numpy array with shape (len(tRange), 3)).
    """
    
    tSteps = tRange[1:] - tRange[:-1]               # Time steps
    pV = np.zeros((len(tRange), 3))                 # Array for x values
    vV = np.zeros((len(tRange), 3))                 # Array for vel. values
    
    pV[0] = p0                                      # Initial position vector
    vV[0] = v0                                      # Initial velocity vector
    
    for i in range(len(tSteps)-1):
        pV[i+1], vV[i+1] = alt_leap(func, tSteps[i], pV[i], vV[i], z, *args)
    
    return pV, vV


def time_leap(func, tRange, p0, v0, z, *args):
    """ Execute the leapfrog integration method by calling the iterative 
        scheme in a loop. Here the "alternative" leapfrog method is used 
        as it does not require the initial velocity at t=i-1/2. This 
        function can be used to differentiate the orbits of dwarf 
        galaxies back in time for a time dependent gravitational 
        potential.

        Input:
            func:   Function that will be integrated, should take the 
                    integration variable as first input (function).
            tRange: The points in time where the second order differential 
                    equation has to be solved (numpy array).
            p0:     Initial 3D position, e.g. in Cartesian coordinates, at 
                    the time of the first entry of tRange (3D numpy array).
            v0:     Initial 3D velocity, e.g. in Cartesian coordinates, at 
                    the time of the first entry of tRange (3D numpy array).
            z:      Redshifts corresponding to the time range at which the 
                    equation will be solved (numpy array).
        
        Returns:
            pV:     The 3D position vector at the times given by tRange 
                    (numpy array with shape (len(tRange), 3)).
            vV:     The 3D velocity vector at the times given by tRange 
                    (numpy array with shape (len(tRange), 3)).    
    """
    
    tSteps = tRange[1:] - tRange[:-1]               # Time steps
    pV = np.zeros((len(tRange), 3))                 # Array for x values
    vV = np.zeros((len(tRange), 3))                 # Array for vel. values
    
        # Checking input
    if len(p0) != len(v0):
        raise Exception("Position and velocity must have the same length")
    
    if len(z) != len(tRange):                       # Interpolate mass & z range
        newZ = np.linspace(min(z), max(z), len(tRange))
        interpZ = interp1d(ge.lin_func(z), z)
        
        z = interpZ(newZ)                           # Full redshift range
    
    pV[0] = p0                                      # Initial position vector
    vV[0] = v0                                      # Initial velocity vector
    
    for i in range(len(tSteps)-1):
        pV[i+1], vV[i+1] = alt_leap(func, tSteps[i], pV[i], vV[i], z[i], *args)
    
    return pV, vV


def plot_mult_3d(pos, linestyles, labels, cmap=cm.cool, step=10, saveFig=None):
    """ Plot multiple 3D orbits of, for example, dwarf galaxies orbiting the 
        Milky Way.

        Input:
            pos:    3D position vectors that will be plotted (numpy array).
            linestyles: Matplotlib linestyles that will be used for the plot 
                        (list or tuple).
            labels: Labels of the different 3D orbits (list or tuple).
            cmap:   Colormap used to color the orbits and represent the time 
                    (matplotlib colormap).
            step:   Step size on how gradual the colormap changes (integer).
            saveFig:Option to save the generated figure, if None then the 
                    figure is not save. If anything else, the figure is saved 
                    to that name (None or string).
        
        Returns:
            -
    """
    
    startPos = pos[0][0]                                # Starting position
    galCen = np.zeros((3))                              # Galactic center
    timeLen = len(pos[0])                               # Number of points
    
        # Unpacking position vector
    xV = [p[:,0] for p in pos]
    yV = [p[:,1] for p in pos]
    zV = [p[:,2] for p in pos]
    
    # Plotting
    fig = figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    ax.scatter(*startPos, label="Start", marker="o", color="crimson")
    ax.scatter(*galCen, label="GC", marker="o", color="chartreuse")
    
    for j in range(len(xV)):
        for i in range(0, timeLen-step, step):
            ax.plot(xV[j][i:i+step+1], yV[j][i:i+step+1], zV[j][i:i+step+1], 
                    color=cmap(i/timeLen), alpha=0.9, 
                    ls=linestyles[j])
    
        # Setting axes
    ax.set_xlabel(r"$x$ (kpc)", fontsize=18)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=18)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    
    titl = ""
    for ind in range(len(labels)):
        titl += f"{linestyles[ind]} = {labels[ind]}\n"
    
    ax.legend(ncol=2, frameon=False, fontsize=15)
    ax.set_title(titl)
    
    if saveFig: fig.savefig(str(saveFig))
    else: show()


def plot_orbit_3d(pos, cmap=cm.cool, step=10, saveFig=None):
    """ Plot the 3D orbit of, for example, a dwarf galaxy orbiting the 
        Milky Way.

        Input:
            pos:    3D position vector that will be plotted (numpy array).
            cmap:   Colormap used to color the orbit and represent the time 
                    (matplotlib colormap).
            step:   Step size on how gradual the colormap changes (integer).
            saveFig:Option to save the generated figure, if None then the 
                    figure is not save. If anything else, the figure is saved 
                    to that name (None or string).
        
        Returns:
            -
    """
    
    startPos = pos[0]                                   # Starting position
    galCen = np.zeros((3))                              # Galactic center
    timeLen = len(pos)                                  # Number of points
    
    xV, yV, zV = pos[:,0], pos[:,1], pos[:,2]           # Unpacking position
    
    # Plotting
    fig = figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1, projection='3d', azim=-83, elev=49)
    
    ax.scatter(*startPos, label="Start", marker="o", color="crimson")
    ax.scatter(*galCen, label="GC", marker="o", color="chartreuse")
    ax.scatter(100, 100, 100, marker="x")
    
        # Plotting the orbit
    for i in range(0, timeLen-step, step):
        ax.plot(xV[i:i+step+1], yV[i:i+step+1], zV[i:i+step+1], 
                color=cmap(i/timeLen), alpha=0.9)
    
        # Setting axes
    ax.set_xlabel(r"$x$ (kpc)", fontsize=18)
    ax.set_ylabel(r"$y$ (kpc)", fontsize=18)
    ax.set_zlabel(r"$z$ (kpc)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    
    ax.legend(fontsize=15)
    
    fig.tight_layout()
    
    if saveFig: fig.savefig(str(saveFig))
    else: show()

