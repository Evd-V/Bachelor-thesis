import numpy as np
import astropy.coordinates as coord
import astropy.units as u


def spher_to_cart(spher):
    """ From spherical to Cartesian coordinates """

    r, theta, phi = spher                                   # Unpacking
    theta = np.radians(theta)
    phi = np.radians(phi)

    x = r * np.cos(phi) * np.sin(theta)                     # x
    y = r * np.sin(phi) * np.sin(theta)                     # y
    z = r * np.cos(theta)                                   # z

    return x, y, z


def cart_to_helio(cart):
    """ From cartesian to right ascension and declination """

    x, y, z = cart                                          # Unpacking
    alpha = np.arctan(y / x)                                # Right ascension
    delta = np.arcsin(z)                                    # Declination

    return alpha, delta


def spher_to_helio(spher):
    """ From spherical to right ascension and declination """

    cart = spher_to_cart(spher)                             # Cartesian
    helio = cart_to_helio(cart)                             # Heliocentric

    return helio


def cart_to_gal(cart, dm, R0=8.122e3, z0=25):
    """ From Cartesian to galactocentric coordinates """

    x, y, z = cart                                          # Unpacking
    d = np.power(10, dm/5 - 2)                              # Distance

    b = np.arctan(y / (x - R0))                             # b coordinate
    l = np.arcsin((z - z0) / d)                             # l coordinate

    return (b, l)


def spher_to_gal(spher, dm, R0=8.122e3, z0=25):
    """ From spherical to galactocentric coordinates """

    cart = spher_to_cart(spher)                             # Cartesian
    gal = cart_to_gal(cart, dm, R0=R0, z0=z0)               # Galactic

    return gal


def obtain_pos(ra, dec, dm, pmRa, pmDec, radVel):
    """ Obtain the position for a dwarf galaxy.Input has to have units!

        Input:
            ra     = right ascension (float);
            dec    = declination (float);
            dm     = distance modulus (float);
            pmRa   = proper motion of right ascension (float);
            pmDec  = proper motion of declination (float);
            radVel = radial velocity (float).

        Returns:
            i      = position in heliocentric coordiates.
    """

    d = np.power(10, dm/5 - 2) * u.kpc                   # Distance in kpc

        # Obtain the position of the dwarf galaxy
    i = coord.SkyCoord(ra=ra, dec=dec, distance=d, pm_ra_cosdec=pmRa,
                       pm_dec=pmDec, radial_velocity=radVel, frame="icrs")

    return i


def transf_frame(i, rSun=8.178, U=10, V=11, W=7, vLSR=229):
    """ Transform position from heliocentric to galactocentric coordinates

        Input:
            i    = position in heliocentric coord. (astropy.coord.Skycoord);
            rSun = distance Sun to Galactic Center in kpc (float);
            U    = Sun's velocity to Galactic center in km/s (float);
            V    = Sun's velocity to Galactic rotation in km/s (float);
            W    = Sun's velocity to North Galactic pole in km/s (float);
            vLSR = Sun's velocity w.r.t. local standard of rest in km/s (float).

        Returns:
            g    = position in galactocentric coordinates.
    """

        # Transform to galactocentric coordinates using astropy
    g = i.transform_to(coord.Galactocentric
                      (galcen_distance=rSun * u.kpc,
                       galcen_v_sun=coord.CartesianDifferential
                       (U * u.km/u.s, (V + vLSR) * u.km/u.s, W * u.km/u.s)))

    return g


def find_gal_coord(spher, dm, a0=192.8595, del0=27.1284, l0=122.9320):
    """ Find the galactocentric coordinates based on spherical coordinates """

    cart = spher_to_cart(spher)                             # To Cartesian
    b, l = np.asarray(cart_to_gal(cart, dm)) * np.pi / 180  # Galactic coord.

    a0 *= np.pi / 180                                       # Degrees to rad
    del0 *= np.pi / 180                                     # Degrees to rad
    l0 *= np.pi / 180                                       # Degrees to rad

    part1 = np.sin(del0) * np.sin(b)
    part2 = np.cos(del0) * np.cos(b) * np.cos(l0 - l)

    delta = np.arcsin(part1 + part2)                        # Declination

    part3 = np.cos(b) * np.sin(l0 - l) / np.cos(delta)

    ra = a0 + np.arcsin(part3)                              # Right ascension

    return ra * 180 / np.pi, delta * 180 / np.pi


def pos_vel(g):
    """ Obtain the position and velocity of the dwarf galaxy.

        Input:
            g   = position in galactocentric coordinates.

        Returns:
            p0 = position coordinates in kpc (numpy array);
            v0 = velocity values in km/s (numpy array).
    """

    p0 = np.asarray([-g.x.value, g.y.value, g.z.value])
    v0 = np.asarray([-g.v_x.value, g.v_y.value, g.v_z.value])

    return p0, v0


def conv_vel_frame(vSpher, pSpher):
    """ Convert velocity in spherical coorinates to cartesian """

    vR, vTheta, vPhi = vSpher                       # Unpacking velocity
    r, theta, phi = pSpher                          # Unpacking position
    theta, phi = np.radians(theta), np.radians(phi)

    sT, cT = np.sin(theta), np.cos(theta)           # sin & cos for theta
    sP, cP = np.sin(phi), np.cos(phi)               # sin & cos for phi

    vX = vR * sT * cP + vTheta * cT * cP + vPhi * -sP   # v_x
    vY = vR * sT * sP + vTheta * cT * sP + vPhi * cP    # v_y
    vZ = vR * cT - vTheta * sT                          # v_z

    return np.asarray((vX, vY, vZ))


def main():
    """ Main function that will be executed """

    # DraI
    coordDraI = (81.8, 55.3, 273.3)
    cartDra = spher_to_cart(coordDraI)      # Cartesian coordinates

    print(cartDra)
    pmRaDra = (0.039 * 1e-3 / 3600) * u.mas/u.yr        # Right ascension in mas/yr
    pmDecDra = (-0.181 * 1e-3 / 3600) * u.mas/u.yr      # declination in mas/yr
    radVelDra = -292.1 * u.km/u.s                       # Radial velocity (km/s)
    dmDra = 19.57                                       # Distance modulus

    raDra = (17*15 + 20*0.25 + 12.4/240) * u.deg
    decDra = (57 + 54/60 + 55/3600) * u.deg


    helioDra = obtain_pos(raDra, decDra, dmDra, pmRaDra, pmDecDra, radVelDra)
    galDra = transf_frame(helioDra)
    p0Dra, v0Dra = pos_vel(galDra)

    print(p0Dra)
    print(v0Dra)

    # print(find_gal_coord(coordDraI, d))
