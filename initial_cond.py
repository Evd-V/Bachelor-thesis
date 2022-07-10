import numpy as np
import astropy.coordinates as coord
import astropy.units as u


def spher_to_cart(spher):
    """ Transform from spherical to Cartesian coordinates """

    r, theta, phi = spher                               # Unpacking
    theta = np.radians(theta)
    phi = np.radians(phi)

    x = r * np.cos(phi) * np.sin(theta)                 # x
    y = r * np.sin(phi) * np.sin(theta)                 # y
    z = r * np.cos(theta)                               # z

    return x, y, z


def cart_to_helio(cart):
    """ From cartesian to right ascension and declination """

    x, y, z = cart                                      # Unpacking
    alpha = np.arctan(y / x)                            # Right ascension
    delta = np.arcsin(z)                                # Declination

    return alpha, delta


def spher_to_helio(spher):
    """ From spherical to right ascension and declination """

    cart = spher_to_cart(spher)                         # Cartesian
    helio = cart_to_helio(cart)                         # Heliocentric

    return helio


def obtain_pos(ra, dec, dm, pmRa, pmDec, radVel):
    """ Obtain the position for a dwarf galaxy. Input has to have units!

        Input:
            ra:     Right ascension (float);
            dec:    Declination (float);
            dm:     Distance modulus (float);
            pmRa:   Proper motion of right ascension (float);
            pmDec:  Proper motion of declination (float);
            radVel: Radial velocity (float).

        Returns:
            i:      Position in heliocentric coordiates (astropy SkyCoord).
    """

    d = np.power(10, dm/5 - 2) * u.kpc                   # Distance in kpc

        # Obtain the position of the dwarf galaxy
    i = coord.SkyCoord(ra=ra, dec=dec, distance=d, pm_ra_cosdec=pmRa,
                       pm_dec=pmDec, radial_velocity=radVel, frame="icrs")

    return i


def transf_frame(i, rSun=8.178, U=10, V=11, W=7, vLSR=229):
    """ Transform position from heliocentric to galactocentric coordinates

        Input:
            i:      Position in heliocentric coord. (astropy.coord.Skycoord);
            rSun:   Distance Sun to Galactic Center in kpc (float);
            U:      Sun's velocity to Galactic center in km/s (float);
            V:      Sun's velocity to Galactic rotation in km/s (float);
            W:      Sun's velocity to North Galactic pole in km/s (float);
            vLSR:   Sun's velocity w.r.t. local standard of rest in km/s (float).

        Returns:
            g:      Position in galactocentric coordinates (astropy SkyCoord).
    """

        # Transform to galactocentric coordinates using astropy
    g = i.transform_to(coord.Galactocentric
                      (galcen_distance=rSun * u.kpc,
                       galcen_v_sun=coord.CartesianDifferential
                       (U * u.km/u.s, (V + vLSR) * u.km/u.s, W * u.km/u.s)))

    return g


def pos_vel(g):
    """ Obtain the position and velocity of the dwarf galaxy.

        Input:
            g:      Position in galactocentric coordinates.

        Returns:
            p0:     Position coordinates in kpc (numpy array);
            v0:     Velocity values in km/s (numpy array).
    """

    p0 = np.asarray([-g.x.value, g.y.value, g.z.value])
    v0 = np.asarray([-g.v_x.value, g.v_y.value, g.v_z.value])

    return p0, v0


def conv_vel_frame(vSpher, pSpher):
    """ Convert velocity in spherical coorinates to cartesian.
        Decapricated: Astropy keeps better track of signs and units.
    """

    vR, vTheta, vPhi = vSpher                       # Unpacking velocity
    r, theta, phi = pSpher                          # Unpacking position
    theta, phi = np.radians(theta), np.radians(phi)

    sT, cT = np.sin(theta), np.cos(theta)           # sin & cos for theta
    sP, cP = np.sin(phi), np.cos(phi)               # sin & cos for phi

    vX = vR * sT * cP + vTheta * cT * cP + vPhi * -sP   # v_x
    vY = vR * sT * sP + vTheta * cT * sP + vPhi * cP    # v_y
    vZ = vR * cT - vTheta * sT                          # v_z

    return np.asarray((vX, vY, vZ))
