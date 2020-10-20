import numpy
import pymp
from matplotlib import pyplot as plt
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
    res : 3d-array, 3d-array
        x- and y-vector component in arrays
    """
    directions_rad = numpy.deg2rad(directions)
    UnitX = -numpy.sin(0.5 * numpy.pi) * numpy.cos(directions_rad)
    UnitY = numpy.sin(0.5 * numpy.pi) * numpy.sin(directions_rad)

    UnitX[numpy.isclose(directions, -1)] = 0
    UnitY[numpy.isclose(directions, -1)] = 0

    return UnitX, UnitY


def downsample(image, sample_size, background_value=-1, background_threshold=0.5):
    """
    Reduce image dimensions of a parameter map by applying a median filter in each image in the z-axis.
    The background will not be considered for the median filter except when the magnitude of it is above the given
    threshold.

    Parameters
    ----------
    image: 2D or 3D parameter map calculated with SLIX.toolbox.
    sample_size: Down sampling parameter.
    background_value: Background value in this parameter map. This is generally -1 but can differ for unit vectors.
    background_threshold: If magnitude of the background values exceeds this value, the downsampled image will have
                          background_value as it's resulting pixel value.

    Returns
    -------
    2D or 3D Numpy array with reduced image dimensions
    """
    image = numpy.array(image)
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
                    if numpy.count_nonzero(roi == background_value) < background_threshold * roi.size:
                        small_img[i, j, sub_image] = numpy.median(roi[roi != background_value])
                    else:
                        small_img[i, j, sub_image] = background_value

    if z == 1:
        small_img = small_img.reshape((nx, ny))

    return small_img


def visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1,
                            cmap='viridis', vmin=0, vmax=None, colorbar=True):
    """
    This method will create a Matplotlib plot based on imshow to represent the given parameter map.
    Here, the parameter map be plotted to the current axis and figure. If none is applied, the method will create a new
    subfigure. To show the results, please use pyplot.show().

    Parameters
    ----------
    parameter_map: 2D parameter map calculated with SLIX.toolbox.
    fig: Matplotlib figure. If None a new subfigure will be created for fig and ax.
    ax: Matplotlib axis. If None a new subfigure will be created for fig and ax.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other plots like the original measurement.
    cmap: Matplotlib color map which is used for the shown image.
    vmin: Minimum value in the resulting plot. If any value is below vmin it will be displayed in black.
    vmax: Maximum value in the resulting plot. If any value is above vmax it will be displayed in white.
    colorbar: Boolean value controlling if a color bar will be displayed in the current subplot.

    Returns
    -------
    The current Matplotlib figure and axis. The image can be shown with pyplot.show().
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    cmap_mod = copy.copy(plt.get_cmap(cmap))
    im = ax.imshow(parameter_map, interpolation='nearest', cmap=cmap_mod, alpha=alpha)
    im.cmap.set_under(color='k')  # Color for values less than vmin
    im.cmap.set_over(color='w')  # Color for values more than vmax
    im.set_clim(vmin, vmax)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=1, background_threshold=0.5):
    """
    This method will create a Matplotlib plot based on quiver to represent the given unit vectors in a more readable
    way. Parameters like thinout can be used to reduce the computing load. If thinout = 1 the resulting vectors might
    not be visible without zooming in significantly.
    Here, the vectors will only be plotted to the current axis. To show the results, please use pyplot.show().

    Parameters
    ----------
    UnitX: Unit vectors in x-axis
    UnitY: Unit vectors in y-axis
    thinout: Unit vectors will be thinned out using downsampling and thinning in combination. This will increase the
             vector size in the resulting image but will also reduce the information density. Please use with caution.
    ax: Matplotlib axis. If none, the current context axis will be used.
    alpha: Apply alpha to Matplotlib plots to overlay them with some other plots like the original measurement.
    background_threshold: If magnitude of the background values exceeds this value, the downsampled image will have
                          background_value as it's resulting pixel value.

    Returns
    -------
    The current Matplotlib axis. The image can be shown with pyplot.show().

    """
    if ax is None:
        ax = plt.gca()

    thinout = int(thinout)
    if thinout <= 1:
        thinout = 1
    else:
        original_size = UnitX.shape[:-1]
        small_unit_x = downsample(UnitX, thinout, background_value=0, background_threshold=background_threshold)
        for i in range(UnitX.shape[-1]):
            UnitX[:, :, i] = numpy.array(Image.fromarray(small_unit_x[:, :, i])
                                         .resize(original_size[::-1], resample=Image.NEAREST))
        small_unit_y = downsample(UnitY, thinout, background_value=0, background_threshold=background_threshold)
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

