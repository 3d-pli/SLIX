import numpy as _numpy
from matplotlib.colors import hsv_to_rgb as _hsv_to_rgb
from matplotlib import pyplot as _plt
from PIL import Image as _Image
import copy as _copy


def downsample(image, kernel_size, background_value=-1,
               background_threshold=0.5):
    """
    Reduce image dimensions of a parameter map by replacing (N x N) pixels by
    their median value for each image. Image pixels with undefined values
    (background) will not be considered for computing the median,
    except when the fraction of background pixels lies above the defined
    threshold.

    Args:

        image: 2D or 3D parameter map (single image or image stack) calculated with
         SLIX.toolbox.

        kernel_size: Downsampling parameter N (defines how many image pixels
        (N x N) are replaced by their median value).

        background_value: Background value of the parameter map. This is generally
        -1 but can differ for unit vector maps.

        background_threshold: Fraction of background pixels in the considered
        (N x N) area for which the image pixels are set to background_value.
        If the fraction of background pixels lies above this defined threshold,
        background pixels will not be considered for computing the median.

    Returns:

        2D or 3D NumPy array with reduced image dimensions.
    """
    image = _numpy.array(image)
    # downsample image
    if len(image.shape) == 2:
        x, y = image.shape
        z = 1
    else:
        x, y, z = image.shape

    nx = _numpy.ceil(x / kernel_size).astype('int')
    ny = _numpy.ceil(y / kernel_size).astype('int')
    small_img = _numpy.empty((nx, ny, z), dtype='float32')

    for sub_image in range(z):
        for i in range(0, nx):
            for j in range(0, ny):
                roi = image[kernel_size * i:kernel_size * i + kernel_size,
                            kernel_size * j:kernel_size * j + kernel_size,
                            sub_image]
                if _numpy.count_nonzero(roi != background_value) >= \
                        background_threshold * roi.size:
                    small_img[i, j, sub_image] = _numpy.median(
                        roi[roi != background_value])
                else:
                    small_img[i, j, sub_image] = background_value

    if z == 1:
        small_img = small_img.reshape((nx, ny))

    return small_img


def visualize_parameter_map(parameter_map, fig=None, ax=None, alpha=1,
                            cmap='viridis', vmin=0, vmax=None, colorbar=True):
    """
    This method will create a Matplotlib plot based on imshow to display the
    given parameter map in different colors. The parameter map is plotted to
    the current axis and figure. If neither is given, the method will
    create a new subfigure. To show the results, please use pyplot.show().

    Args:

        parameter_map: 2D parameter map calculated with SLIX.toolbox.

        fig: Matplotlib figure. If None, a new subfigure will be created for fig
        and ax.

        ax: Matplotlib axis. If None, a new subfigure will be created for fig
        and ax.

        alpha: Apply alpha to Matplotlib plots to overlay them with some other
        image like the averaged transmitted light intensity.

        cmap: Matplotlib color map which is used for displaying the image.

        vmin: Minimum value in the resulting plot. If any value is below vmin,
        it will be displayed in black.

        vmax: Maximum value in the resulting plot. If any value is above vmax,
        it will be displayed in white.

        colorbar: Boolean value controlling if a color bar will be displayed in
        the current subplot.

    Returns:

        The current Matplotlib figure and axis. The image can be shown with
        pyplot.show().
    """
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    cmap_mod = _copy.copy(_plt.get_cmap(cmap))
    im = ax.imshow(parameter_map, interpolation='nearest', cmap=cmap_mod,
                   alpha=alpha)
    im.cmap.set_under(color='k')  # Color for values less than vmin
    im.cmap.set_over(color='w')  # Color for values more than vmax
    im.set_clim(vmin, vmax)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def visualize_unit_vectors(UnitX, UnitY, thinout=1, ax=None, alpha=1,
                           background_threshold=0.5):
    """
    This method will create a Matplotlib plot based on quiver to represent the
    given unit vectors as colored lines (vector map).
    Parameters like thinout can be used to reduce the computing load. If
    thinout = 1, the resulting vectors might not be visible
    without zooming in significantly. Here, the vectors will only be plotted
    to the current axis. To show the results, please use pyplot.show().

    Args:

        UnitX: Unit vector components along the x-axis (3D NumPy array).

        UnitY: Unit vector components along the y-axis (3D NumPy array).

        thinout: Downsampling parameter N (defines how many vectors N x N are
        replaced by one vector using the downsample function).
        Unit vectors will be thinned out using downsampling and thinning in
        combination. This will increase the
        vector size in the resulting image but will also reduce the information
        density. Please use with caution.

        ax: Matplotlib axis. If None, the current context axis will be used.

        alpha: Apply alpha to Matplotlib plots to overlay them with some other
        other image like the averaged transmitted light intensity.
        background_threshold: If the fraction of background pixels (number of
        pixels without vector within N x N pixels) is below this threshold,
        the downsampled pixel will not show a vector.

    Returns:

        The current Matplotlib axis. The image can be shown with pyplot.show().

    """
    if ax is None:
        ax = _plt.gca()

    thinout = int(thinout)
    if thinout <= 1:
        thinout = 1
    else:
        UnitX = UnitX.copy()
        UnitY = UnitY.copy()
        original_size = UnitX.shape[:-1]
        small_unit_x = downsample(UnitX, thinout, background_value=0,
                                  background_threshold=background_threshold)
        for i in range(UnitX.shape[-1]):
            UnitX[:, :, i] = _numpy.array(_Image.fromarray(small_unit_x[:, :, i])
                                         .resize(original_size[::-1],
                                                 resample=_Image.NEAREST))

        small_unit_y = downsample(UnitY, thinout, background_value=0,
                                  background_threshold=background_threshold)
        for i in range(UnitY.shape[-1]):
            UnitY[:, :, i] = _numpy.array(_Image.fromarray(small_unit_y[:, :, i])
                                         .resize(original_size[::-1],
                                                 resample=_Image.NEAREST))
        del original_size
        del small_unit_y
        del small_unit_x
    skip = (slice(None, None, thinout), slice(None, None, thinout))

    for i in range(UnitX.shape[-1]):
        mesh_x, mesh_y = _numpy.meshgrid(_numpy.arange(0, UnitX.shape[1]),
                                        _numpy.arange(0, UnitX.shape[0]))
        mesh_u = UnitX[:, :, i]
        mesh_v = UnitY[:, :, i]

        mask = _numpy.logical_or(mesh_u[skip] != 0, mesh_v[skip] != 0)
        mesh_x = mesh_x[skip][mask]
        mesh_y = mesh_y[skip][mask]
        mesh_u = mesh_u[skip][mask]
        mesh_v = mesh_v[skip][mask]

        # Normalize the arrows:
        mesh_u_normed = thinout * mesh_u / _numpy.sqrt(mesh_u**2 + mesh_v**2)
        mesh_v_normed = thinout * mesh_v / _numpy.sqrt(mesh_u**2 + mesh_v**2)

        ax.quiver(mesh_x, mesh_y, mesh_u_normed, mesh_v_normed,
                  _numpy.arctan2(mesh_v_normed, mesh_u_normed),
                  cmap='hsv', angles='xy', scale_units='xy', scale=1,
                  alpha=alpha, headwidth=0, headlength=0, headaxislength=0,
                  minlength=0, pivot='mid', clim=(0, _numpy.pi))
    return ax


def visualize_direction(direction):
    """
    Generate a 2D colorized direction image in the HSV color space based on
    the original direction. Value and saturation of the color will always be
    one. The hue is determined by the direction.

    If the direction parameter is only a 2D numpy array, the result will be
    a simple orientation map where each pixel contains the HSV value
    corresponding to the direction angle.

    When a 3D stack with max. three directions is used, the result will be
    different. The resulting image will have two times the width and height.
    Each 2x2 square will show the direction angle of up to three directions.
    Depending on the number of directions, the following pattern is used to
    show the different direction angles.

    1 direction:

        1 1
        1 1

    2 directions:

        1 2
        2 1

    3 directions:

        1 2
        3 0

    Args:

        direction: 2D or 3D Numpy array containing the direction of the image
                   stack

    Returns:

        numpy.ndarray: 2D image containing the resulting HSV orientation map

    """
    direction = _numpy.array(direction)
    direction_shape = direction.shape

    h = direction
    s = _numpy.ones(direction.shape)
    v = _numpy.ones(direction.shape)

    hsv_stack = _numpy.stack((1 - h / 180.0, s, v))
    hsv_stack = _numpy.moveaxis(hsv_stack, 0, -1)
    rgb_stack = _hsv_to_rgb(hsv_stack)

    if len(direction_shape) > 2:
        return _visualize_multiple_direction(direction, rgb_stack)
    else:
        return _visualize_one_direction(direction, rgb_stack)


def _visualize_one_direction(direction, rgb_stack):
    output_image = rgb_stack
    output_image[direction == -1] = 0

    return output_image.astype('float32')


def _visualize_multiple_direction(direction, rgb_stack):
    output_image = _numpy.empty((direction.shape[0] * 2,
                                direction.shape[1] * 2,
                                3))
    # count valid directions
    valid_directions = _numpy.count_nonzero(direction > -1, axis=-1)

    r = rgb_stack[..., 0]
    g = rgb_stack[..., 1]
    b = rgb_stack[..., 2]

    # Now we need to place them in the right pixel on our output image
    for x in range(direction.shape[0]):
        for y in range(direction.shape[1]):
            if valid_directions[x, y] == 0:
                output_image[x * 2:x * 2 + 1, y * 2:y * 2 + 1] = 0
            elif valid_directions[x, y] == 1:
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 0] = r[x, y, 0]
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 1] = g[x, y, 0]
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 2] = b[x, y, 0]
            else:
                output_image[x * 2, y * 2, 0] = r[x, y, 0]
                output_image[x * 2, y * 2, 1] = g[x, y, 0]
                output_image[x * 2, y * 2, 2] = b[x, y, 0]

                output_image[x * 2 + 1, y * 2, 0] = r[x, y, 1]
                output_image[x * 2 + 1, y * 2, 1] = g[x, y, 1]
                output_image[x * 2 + 1, y * 2, 2] = b[x, y, 1]

                if valid_directions[x, y] == 2:
                    output_image[x * 2, y * 2 + 1, 0] = r[x, y, 1]
                    output_image[x * 2, y * 2 + 1, 1] = g[x, y, 1]
                    output_image[x * 2, y * 2 + 1, 2] = b[x, y, 1]

                    output_image[x * 2 + 1, y * 2 + 1, 0] = r[x, y, 0]
                    output_image[x * 2 + 1, y * 2 + 1, 1] = g[x, y, 0]
                    output_image[x * 2 + 1, y * 2 + 1, 2] = b[x, y, 0]
                else:
                    output_image[x * 2, y * 2 + 1, 0] = r[x, y, 2]
                    output_image[x * 2, y * 2 + 1, 1] = g[x, y, 2]
                    output_image[x * 2, y * 2 + 1, 2] = b[x, y, 2]

                    if valid_directions[x, y] == 3:
                        output_image[x * 2 + 1, y * 2 + 1, 0] = 0
                        output_image[x * 2 + 1, y * 2 + 1, 1] = 0
                        output_image[x * 2 + 1, y * 2 + 1, 2] = 0
                    if valid_directions[x, y] == 4:
                        output_image[x * 2 + 1, y * 2 + 1, 0] = r[x, y, 3]
                        output_image[x * 2 + 1, y * 2 + 1, 1] = g[x, y, 3]
                        output_image[x * 2 + 1, y * 2 + 1, 2] = b[x, y, 3]

    return output_image.astype('float32')
