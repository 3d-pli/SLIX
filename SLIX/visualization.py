import numpy
import pymp
from matplotlib import pyplot as plt
from matplotlib import colors

CPU_COUNT = 8


def downsample(image, sample_size=10):
    # downsample image
    x, y = image.shape
    nx = numpy.ceil(x / sample_size).astype('int')
    ny = numpy.ceil(y / sample_size).astype('int')
    small_img = pymp.shared.array((nx, ny), dtype='float32')

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, nx):
            for j in range(0, ny):
                roi = image[sample_size * i:sample_size * i + sample_size, sample_size * j:sample_size * j + sample_size]
                small_img[i, j] = numpy.round(numpy.median(roi))

    return small_img


def unit_vectors(directions):
    """
    Calculates the unit vector from direction and inclination

    Parameters
    ----------
    directions : 3d-array
        direction in radian

    Returns
    -------
    res : 3d-array, 3d-array, 3d-array
        x-, y- and z-vector component in arrays
    """
    directions_rad = numpy.deg2rad(directions)
    UnitX = numpy.sin(0.5 * numpy.pi) * numpy.cos(directions_rad)
    UnitY = numpy.sin(0.5 * numpy.pi) * numpy.sin(directions_rad)

    UnitX[numpy.isclose(directions, -1)] = 0
    UnitY[numpy.isclose(directions, -1)] = 0

    return UnitX, UnitY


def visualize_unit_vectors(UnitX, UnitY, thinout=1):
    #UnitX = UnitX[400:600, 400:600]
    #UnitY = UnitY[400:600, 400:600]
    #print(UnitX.max(), UnitY.max())

    for i in range(UnitX.shape[-1]):
        X, Y = numpy.meshgrid(numpy.arange(0, UnitX.shape[1]), numpy.arange(0, UnitX.shape[0]))
        U = thinout * UnitX[:, :, i]
        V = thinout * UnitY[:, :, i]
        mask = numpy.logical_or(U != 0, V != 0)
        skip = slice(None, None, thinout)
        X = X[mask]
        Y = Y[mask]
        U = U[mask]
        V = V[mask]

        # Normalize the arrows:
        U_normed = U / numpy.sqrt(U ** 2 + V ** 2)
        V_normed = V / numpy.sqrt(U ** 2 + V ** 2)

        plt.quiver(X[skip], Y[skip], V_normed[skip], U_normed[skip], numpy.arctan2(V_normed[skip], U_normed[skip]),
                   cmap='hsv', angles='xy', scale_units='xy', scale=1,
                   headwidth=0, headlength=0, headaxislength=0, minlength=0, pivot='mid')
    plt.show()

