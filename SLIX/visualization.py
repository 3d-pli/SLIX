import numpy
from matplotlib.colors import hsv_to_rgb
from matplotlib import pyplot as plt
from PIL import Image
import copy

from SLIX._visualization import _downsample, _downsample_2d, _count_nonzero

__all__ = ['visualize_parameter_map',
           'visualize_unit_vectors',
           'visualize_direction']


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
        fig, ax = plt.subplots(1, 1)

    cmap_mod = copy.copy(plt.get_cmap(cmap))
    im = ax.imshow(parameter_map, interpolation='nearest', cmap=cmap_mod,
                   alpha=alpha)
    im.cmap.set_under(color='k')  # Color for values less than vmin
    im.cmap.set_over(color='w')  # Color for values more than vmax
    im.set_clim(vmin, vmax)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def visualize_unit_vectors(UnitX, UnitY, ax=None, thinout=20,
                           scale=-1, vector_width=1,
                           alpha=0.8, background_threshold=0.5,
                           background_value=0):
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

        thinout: Downscaling parameter N (defines how many vectors N x N are
        replaced by one vector).
        Unit vectors will be thinned out using downscaling and thinning in
        combination. This will increase the
        vector size in the resulting image but will also reduce the information
        density. Please use with caution.

        scale: Increase the vector length by the given scale. Vectors will be
               longer and might overlap if the scale is too high.

        ax: Matplotlib axis. If None, the current context axis will be used.

        vector_width: When choosing a high scale, the vectors might appear
        quite thin which results in hard to read images. This option allows to
        increase the vector thickness to improve visibility.

        alpha: Apply alpha to Matplotlib plots to overlay them with some other
        other image like the averaged transmitted light intensity.
        background_threshold: If the fraction of background pixels (number of
        pixels without vector within N x N pixels) is below this threshold,
        the downscaled pixel will not show a vector.

        background_value: Background value of the parameter map. This is
        generally 0 in both axes for unit vector maps
        but can differ if another threshold was set.

        background_value: Fraction of background pixels in the considered
        (N x N) area for which the image pixels are set to background_value.
        If the fraction of background pixels lies above this defined threshold,
        background pixels will not be considered for computing the median.


    Returns:

        The current Matplotlib axis. The image can be shown with pyplot.show().

    """
    if ax is None:
        ax = plt.gca()

    while len(UnitX.shape) < 3:
        UnitX = UnitX[..., numpy.newaxis]
    while len(UnitY.shape) < 3:
        UnitY = UnitY[..., numpy.newaxis]

    # The default scale is below zero to allow the user to define his own scale
    # A scale below zero isn't valid for visualization. If the user
    # defines no scale, we suspect that the user wants an image
    # where each vector has a scale of one. Therefore we set the scale to
    # the same as our thinout when we draw the image.
    if scale < 0:
        scale = thinout

    if thinout > 1:
        downscaled_unit_x = _downsample(UnitX, thinout,
                                        background_threshold, background_value)
        downscaled_unit_y = _downsample(UnitY, thinout,
                                        background_threshold, background_value)
        # Rescale images to original dimensions

        for i in range(UnitX.shape[2]):
            UnitX[:, :, i] = numpy.array(
                Image.fromarray(downscaled_unit_x[:, :, i]) \
                .resize(UnitX.shape[:2][::-1], Image.NEAREST)
            )
            UnitY[:, :, i] = numpy.array(
                Image.fromarray(downscaled_unit_y[:, :, i]) \
                .resize(UnitY.shape[:2][::-1], Image.NEAREST)
            )

        del downscaled_unit_y
        del downscaled_unit_x
    for i in range(UnitX.shape[2]):
        mesh_x, mesh_y = numpy.meshgrid(numpy.arange(0, UnitX.shape[1], thinout),
                                        numpy.arange(0, UnitX.shape[0], thinout))
        mesh_u = UnitX[::thinout, ::thinout, i]
        mesh_v = UnitY[::thinout, ::thinout, i]

        # Normalize the arrows:
        mesh_u_normed = mesh_u / numpy.sqrt(numpy.maximum(1e-15,
                                            mesh_u ** 2 + mesh_v ** 2))
        mesh_v_normed = mesh_v / numpy.sqrt(numpy.maximum(1e-15,
                                            mesh_u ** 2 + mesh_v ** 2))
        mesh_u_normed[numpy.isclose(mesh_u, 0) &
                      numpy.isclose(mesh_v, 0)] = numpy.nan
        mesh_v_normed[numpy.isclose(mesh_u, 0) &
                      numpy.isclose(mesh_v, 0)] = numpy.nan

        normed_angle = numpy.arctan2(mesh_v_normed, -mesh_u_normed)
        color_rgb = visualize_direction(normed_angle * 180.0 / numpy.pi)
        color_rgb = numpy.clip(color_rgb.reshape(color_rgb.shape[0] *
                                                 color_rgb.shape[1], 3), 0, 1)

        # 1/scale to increase vector length for scale > 1
        ax.quiver(mesh_x, mesh_y, mesh_u_normed, mesh_v_normed,
                  color=color_rgb, angles='xy', scale_units='xy',
                  scale=1.0/scale, headwidth=0, headlength=0, headaxislength=0,
                  minlength=0, pivot='mid', alpha=alpha,
                  width=vector_width, units='xy', edgecolors=color_rgb)
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
    direction = numpy.array(direction)
    direction_shape = direction.shape

    h = direction
    s = numpy.ones(direction.shape)
    v = numpy.ones(direction.shape)

    hsv_stack = numpy.stack((h / 180.0, s, v))
    hsv_stack = numpy.moveaxis(hsv_stack, 0, -1)
    rgb_stack = hsv_to_rgb(hsv_stack)

    if len(direction_shape) > 2:
        return _visualize_multiple_direction(direction, rgb_stack)
    else:
        return _visualize_one_direction(direction, rgb_stack)


def _visualize_one_direction(direction, rgb_stack):
    output_image = rgb_stack
    output_image[direction == -1] = 0

    return output_image.astype('float32')


def _visualize_multiple_direction(direction, rgb_stack):
    output_image = numpy.zeros((direction.shape[0] * 2,
                                direction.shape[1] * 2,
                                3))
    # count valid directions
    valid_directions = numpy.count_nonzero(direction > -1, axis=-1)

    r = rgb_stack[..., 0]
    g = rgb_stack[..., 1]
    b = rgb_stack[..., 2]

    # Now we need to place them in the right pixel on our output image
    for x in range(direction.shape[0]):
        for y in range(direction.shape[1]):
            if valid_directions[x, y] == 0:
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2] = 0
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
