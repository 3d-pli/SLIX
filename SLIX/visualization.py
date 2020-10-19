import numpy
import pymp
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image
import copy
from . import toolbox

CPU_COUNT = toolbox.CPU_COUNT


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
    UnitX = -numpy.sin(0.5 * numpy.pi) * numpy.cos(directions_rad)
    UnitY = numpy.sin(0.5 * numpy.pi) * numpy.sin(directions_rad)

    UnitX[numpy.isclose(directions, -1)] = 0
    UnitY[numpy.isclose(directions, -1)] = 0

    return UnitX, UnitY


def downsample(image, sample_size=10, background_value=-1):
    # downsample image
    if len(image.shape) == 2:
        x, y = image.shape
        z = 1
    else:
        x, y, z = image.shape

    nx = numpy.ceil(x / sample_size).astype('int')
    ny = numpy.ceil(y / sample_size).astype('int')
    small_img = pymp.shared.array((nx, ny, z), dtype='float32')

    for sub_image in range(z):
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    roi = image[sample_size * i:sample_size * i + sample_size,
                                sample_size * j:sample_size * j + sample_size, sub_image]
                    if numpy.count_nonzero(roi != background_value) > 0.1 * roi.size:
                        small_img[i, j, sub_image] = numpy.median(roi[roi != background_value])
                    else:
                        small_img[i, j, sub_image] = background_value

    if z == 1:
        small_img = small_img.reshape((nx, ny))

    return small_img


def visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1,
                            cmap='viridis', vmin=0, vmax=None, colorbar=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    cmap_mod = copy.copy(plt.get_cmap(cmap))
    im = ax.imshow(parameter_map, interpolation='nearest', origin='lower', cmap=cmap_mod, alpha=alpha)
    im.cmap.set_under(color='k')  # Color for values less than vmin
    im.cmap.set_over(color='w')  # Color for values more than vmax
    im.set_clim(vmin, vmax)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=0.8):
    if ax is None:
        ax = plt.gca()

    thinout = int(thinout)
    if thinout <= 1:
        thinout = 1
    else:
        original_size = UnitX.shape[:-1]
        small_unit_x = downsample(UnitX, thinout, background_value=0)
        for i in range(UnitX.shape[-1]):
            UnitX[:, :, i] = numpy.array(Image.fromarray(small_unit_x[:, :, i])
                                         .resize(original_size[::-1], resample=Image.NEAREST))
        small_unit_y = downsample(UnitY, thinout, background_value=0)
        for i in range(UnitY.shape[-1]):
            UnitY[:, :, i] = numpy.array(Image.fromarray(small_unit_y[:, :, i])
                                         .resize(original_size[::-1], resample=Image.NEAREST))
        del original_size
        del small_unit_y
        del small_unit_x
    skip = (slice(None, None, thinout), slice(None, None, thinout))

    for i in range(UnitX.shape[-1]):
        mesh_x, mesh_y = numpy.meshgrid(numpy.arange(0, UnitX.shape[1]), numpy.arange(0, UnitX.shape[0]))
        mesh_u = UnitX[:, :, i]
        mesh_v = UnitY[:, :, i]

        mask = numpy.logical_or(mesh_u[skip] != 0, mesh_v[skip] != 0)
        mesh_x = mesh_x[skip][mask]
        mesh_y = mesh_y[skip][mask]
        mesh_u = mesh_u[skip][mask]
        mesh_v = mesh_v[skip][mask]

        # Normalize the arrows:
        mesh_u_normed = thinout * mesh_u / numpy.sqrt(mesh_u ** 2 + mesh_v ** 2)
        mesh_v_normed = thinout * mesh_v / numpy.sqrt(mesh_u ** 2 + mesh_v ** 2)

        ax.quiver(mesh_x, mesh_y, mesh_u_normed, mesh_v_normed, numpy.arctan2(mesh_v_normed, mesh_u_normed),
                   cmap='hsv', angles='xy', scale_units='xy', scale=1, alpha=alpha,
                   headwidth=0, headlength=0, headaxislength=0, minlength=0, pivot='mid', clim=(0, numpy.pi))
    return ax

