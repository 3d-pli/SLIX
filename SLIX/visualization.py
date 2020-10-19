import numpy
import pymp
from matplotlib import pyplot as plt
import copy

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


def visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1, cmap='viridis', vmin=0, vmax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    cmap_mod = copy.copy(plt.get_cmap(cmap))
    cmap_mod.set_under('black')  # Color for values less than vmin
    cmap_mod.set_over('white')  # Color for values more than vmax

    im = ax.imshow(parameter_map, interpolation='nearest', origin='lower', cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    return fig, ax


def visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=0.8):
    if ax is None:
        ax = plt.gca()
    skip = (slice(None, None, thinout), slice(None, None, thinout))
    for i in range(UnitX.shape[-1]):
        X, Y = numpy.meshgrid(numpy.arange(0, UnitX.shape[1]), numpy.arange(0, UnitX.shape[0]))
        U = UnitX[:, :, i]
        V = UnitY[:, :, i]

        mask = numpy.logical_or(U[skip] != 0, V[skip] != 0)
        X = X[skip][mask]
        Y = Y[skip][mask]
        U = U[skip][mask]
        V = V[skip][mask]

        # Normalize the arrows:
        U_normed = thinout * U / numpy.sqrt(U ** 2 + V ** 2)
        V_normed = thinout * V / numpy.sqrt(U ** 2 + V ** 2)

        ax.quiver(X, Y, U_normed, V_normed, numpy.arctan2(V_normed, U_normed),
                   cmap='hsv', angles='xy', scale_units='xy', scale=1, alpha=alpha,
                   headwidth=0, headlength=0, headaxislength=0, minlength=0, pivot='mid')
    return ax

