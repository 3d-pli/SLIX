import copy

import numpy
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from SLIX._visualization import _downsample, _plot_axes_unit_vectors, \
    _visualize_multiple_direction, \
    _visualize_one_direction

__all__ = ['parameter_map',
           'unit_vectors',
           'unit_vector_distribution',
           'direction',
           'Colormap']


class Colormap:
    @staticmethod
    def prepare(direction: numpy.ndarray, inclination: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        if direction.max(axis=None) > numpy.pi and not numpy.isclose(direction.max(axis=None), numpy.pi):
            direction = numpy.deg2rad(direction)
        if inclination.max(axis=None) > numpy.pi and not numpy.isclose(inclination.max(axis=None), numpy.pi):
            inclination = numpy.deg2rad(inclination)

        # If inclination is only 2D and direction is 3D, we need to make sure that the
        # inclination matches the shape of the direction.
        if inclination.ndim == 2 and direction.ndim == 3:
            inclination = inclination[..., numpy.newaxis]
        if inclination.ndim == 3 and inclination.shape[-1] != direction.shape[-1]:
            inclination = numpy.repeat(inclination, direction.shape[-1], axis=-1)

        return direction, inclination

    @staticmethod
    def hsv_white(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)

        hsv_stack = numpy.stack((direction / numpy.pi,
                                 1.0 - (2 * inclination / numpy.pi),
                                 numpy.ones(direction.shape)))
        hsv_stack = numpy.moveaxis(hsv_stack, 0, -1)
        return numpy.clip(hsv_to_rgb(hsv_stack), 0, 1)

    @staticmethod
    def hsv_black(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)

        hsv_stack = numpy.stack((direction / numpy.pi,
                                 numpy.ones(direction.shape),
                                 1.0 - (2 * inclination / numpy.pi)))
        hsv_stack = numpy.moveaxis(hsv_stack, 0, -1)
        return numpy.clip(hsv_to_rgb(hsv_stack), 0, 1)

    @staticmethod
    def rgb(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)

        direction[direction > numpy.pi / 2] = numpy.pi - direction[direction > numpy.pi / 2]
        rgb_stack = numpy.stack((
            numpy.cos(inclination) * numpy.cos(direction),
            numpy.cos(inclination) * numpy.sin(direction),
            numpy.sin(inclination)
        ))

        rgb_stack = numpy.moveaxis(rgb_stack, 0, -1)

        return numpy.clip(rgb_stack, 0, 1)

    @staticmethod
    def hsv_black_reverse(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)
        direction = numpy.clip(numpy.abs(-numpy.pi + direction), 0, numpy.pi)

        return Colormap.hsv_black(direction, inclination)

    @staticmethod
    def hsv_white_reverse(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)
        direction = numpy.clip(numpy.abs(-numpy.pi + direction), 0, numpy.pi)

        return Colormap.hsv_white(direction, inclination)

    @staticmethod
    def rgb_reverse(direction: numpy.ndarray, inclination: numpy.ndarray) -> numpy.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination)
        direction = numpy.clip(numpy.abs(-numpy.pi + direction), 0, numpy.pi)

        return Colormap.rgb(direction, inclination)


def parameter_map(parameter_map, fig=None, ax=None, alpha=1,
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


def color_bubble(colormap: Colormap, shape=(1000, 1000, 3)) -> numpy.ndarray:
    """
    Based on the chosen colormap in methods like unit_vectors or
    direction, the user might want to see the actual color bubble to understand
    the shown orientations. This method creates an empty numpy array and fills
    it with values based on the circular orientation from the middle point.
    The color can be directed from the colormap argument

    Args:
        colormap: Colormap which will be used to create the color bubble
        shape: Shape of the resulting color bubble.

    Returns: NumPy array containing the color bubble

    """

    # create a meshgrid of the shape with the position of each pixel
    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    # center of our color_bubble
    center = numpy.array([shape[0]/2, shape[1]/2])
    # radius where a full circle is still visible
    radius = numpy.minimum(numpy.minimum(center[0], center[1]),
                           numpy.minimum(shape[0] - center[0], shape[1] - center[1]))
    # calculate the direction as the angle between the center and the pixel
    direction = numpy.pi - numpy.arctan2(y - center[0], x - center[1]) % numpy.pi

    # calculate the inclination as the distance between the center and the pixel
    inclination = numpy.sqrt((y - center[0])**2 + (x - center[1])**2)
    # normalize the inclination to a range of 0 to 90 degrees where 0 degree is at a distance of radius
    # and 90 degree is at a distance of 0
    inclination = 90 - inclination / radius * 90

    # create the color bubble
    color_bubble = colormap(direction, inclination)
    color_bubble[inclination < 0] = 0

    return (255.0 * color_bubble).astype('uint8')


def unit_vectors(unit_x, unit_y, ax=None, thinout=20,
                 scale=-1, vector_width=1,
                 alpha=0.8, background_threshold=0.5,
                 background_value=0, colormap=Colormap.hsv_black,
                 weighting=None):
    """
    This method will create a Matplotlib plot based on quiver to represent the
    given unit vectors as colored lines (vector map).
    Parameters like thinout can be used to reduce the computing load. If
    thinout = 1, the resulting vectors might not be visible
    without zooming in significantly. Here, the vectors will only be plotted
    to the current axis. To show the results, please use pyplot.show().

    Args:

        unit_x: Unit vector components along the x-axis (3D NumPy array).

        unit_y: Unit vector components along the y-axis (3D NumPy array).

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

        colormap: The colormap to use. Default is HSV black. The available color maps
                  can be found in the colormap class.

        weighting: Weighting of the vectors. If None, the vectors will be
                   weighted by a value of one, resulting in normal unit vectors.

    Returns:

        The current Matplotlib axis. The image can be shown with pyplot.show().

    """
    if ax is None:
        ax = plt.gca()

    while len(unit_x.shape) < 3:
        unit_x = unit_x[..., numpy.newaxis]
    while len(unit_y.shape) < 3:
        unit_y = unit_y[..., numpy.newaxis]

    # The default scale is below zero to allow the user to define his own scale
    # A scale below zero isn't valid for visualization. If the user
    # defines no scale, we suspect that the user wants an image
    # where each vector has a scale of one. Therefor we set the scale to
    # the same as our thinout when we draw the image.
    if scale < 0:
        scale = thinout

    if thinout > 1:
        downscaled_unit_x = _downsample(unit_x, thinout,
                                        background_threshold, background_value)
        downscaled_unit_y = _downsample(unit_y, thinout,
                                        background_threshold, background_value)

        while len(downscaled_unit_x.shape) < 3:
            downscaled_unit_x = downscaled_unit_x[..., numpy.newaxis]
        while len(downscaled_unit_y.shape) < 3:
            downscaled_unit_y = downscaled_unit_y[..., numpy.newaxis]

        # Rescale images to original dimensions
        for i in range(unit_x.shape[2]):
            unit_x[:, :, i] = numpy.array(
                Image.fromarray(downscaled_unit_x[:, :, i])
                     .resize(unit_x.shape[:2][::-1], Image.NEAREST)
            )
            unit_y[:, :, i] = numpy.array(
                Image.fromarray(downscaled_unit_y[:, :, i])
                     .resize(unit_y.shape[:2][::-1], Image.NEAREST)
            )

        del downscaled_unit_y
        del downscaled_unit_x

        if weighting is not None:
            while len(weighting.shape) < 3:
                weighting = weighting[..., numpy.newaxis]
            downscaled_weighting = _downsample(weighting, thinout, 0, 0)
            weighting = numpy.array(
                Image.fromarray(downscaled_weighting)
                     .resize(weighting.shape[:2][::-1], Image.NEAREST)
            )
            weighting = weighting[::thinout, ::thinout]
            weighting = weighting.flatten()

    for i in range(unit_x.shape[2]):
        mesh_x, mesh_y = numpy.meshgrid(numpy.arange(0, unit_x.shape[1],
                                                     thinout),
                                        numpy.arange(0, unit_x.shape[0],
                                                     thinout))
        mesh_u = unit_x[::thinout, ::thinout, i]
        mesh_v = unit_y[::thinout, ::thinout, i]

        _plot_axes_unit_vectors(ax,
                                mesh_x.flatten(),
                                mesh_y.flatten(),
                                mesh_u.flatten(),
                                mesh_v.flatten(),
                                scale, alpha, vector_width,
                                weighting, colormap)
    return ax


def unit_vector_distribution(unit_x, unit_y, ax=None, thinout=20,
                             scale=-1, vector_width=1,
                             alpha=0.01, colormap=Colormap.hsv_black,
                             weighting=None):
    """
    This method will create a Matplotlib plot based on quiver to represent the
    given unit vectors as colored lines (vector map).
    Instead of showing a single vector like in unit_vector, here each vector
    will be shown in the resulting image. The thinout parameter will determine
    how many vectors will be overlapping. It is recommended to use a very small
    alpha value to see which directions in the resulting plot are dominant.
    Here, the vectors will only be plotted
    to the current axis. To show the results, please use pyplot.show(). The
    result might need some time to show depending on the input image size.

    Args:

        unit_x: Unit vector components along the x-axis (3D NumPy array).

        unit_y: Unit vector components along the y-axis (3D NumPy array).

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

        colormap: The colormap to use. Default is HSV black. The available color maps
                  can be found in the colormap class.

        weighting: Weighting of the vectors. If None, the vectors will be
        weighted by a value of one, resulting in normal unit vectors.

    Returns:

        The current Matplotlib axis. The image can be shown with pyplot.show().

    """
    if ax is None:
        ax = plt.gca()

    while len(unit_x.shape) < 3:
        unit_x = unit_x[..., numpy.newaxis]
    while len(unit_y.shape) < 3:
        unit_y = unit_y[..., numpy.newaxis]

    # The default scale is below zero to allow the user to define his own scale
    # A scale below zero isn't valid for visualization. If the user
    # defines no scale, we suspect that the user wants an image
    # where each vector has a scale of one. Therefore we set the scale to
    # the same as our thinout when we draw the image.
    if scale < 0:
        scale = thinout

    mesh_x = numpy.empty(unit_x.size)
    mesh_y = numpy.empty(unit_x.size)
    mesh_u = numpy.empty(unit_x.size)
    mesh_v = numpy.empty(unit_x.size)
    mesh_weighting = numpy.empty(unit_x.size)
    idx = 0

    progress_bar = tqdm.tqdm(total=thinout * thinout,
                             desc='Creating unit vectors.')
    for offset_x in range(thinout):
        for offset_y in range(thinout):
            progress_bar.update(1)
            for i in range(unit_x.shape[2]):
                mesh_x_it, mesh_y_it = numpy.meshgrid(
                    numpy.arange(0, unit_x.shape[1] - offset_x, thinout),
                    numpy.arange(0, unit_x.shape[0] - offset_y, thinout)
                )
                mesh_x_it = mesh_x_it.flatten()
                mesh_y_it = mesh_y_it.flatten()
                mesh_u_it = unit_x[offset_y::thinout, offset_x::thinout, i] \
                    .flatten()
                mesh_v_it = unit_y[offset_y::thinout, offset_x::thinout, i] \
                    .flatten()

                if weighting is not None:
                    mesh_weighting_it = weighting[offset_y::thinout,
                                                  offset_x::thinout] \
                        .flatten()
                else:
                    mesh_weighting_it = numpy.ones(mesh_u_it.size)

                mesh_x[idx:idx + len(mesh_x_it)] = mesh_x_it
                mesh_y[idx:idx + len(mesh_y_it)] = mesh_y_it
                mesh_u[idx:idx + len(mesh_u_it)] = mesh_u_it
                mesh_v[idx:idx + len(mesh_v_it)] = mesh_v_it
                mesh_weighting[idx:idx + len(mesh_weighting_it)] = mesh_weighting_it

                idx = idx + len(mesh_x_it)

    progress_bar.set_description('Finished. Plotting unit vectors.')
    _plot_axes_unit_vectors(ax, mesh_x, mesh_y, mesh_u, mesh_v,
                            scale, alpha, vector_width, mesh_weighting, colormap)
    progress_bar.set_description('Done')
    progress_bar.close()
    return ax


def direction(direction, inclination=None, saturation=None, value=None, colormap=Colormap.hsv_black):
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

        inclination: Optional inclination of the image in degrees. If none is set, an inclination of 0° is assumed.

        saturation: Weight image by using the saturation value. Use either a 2D image
                    or a 3D image with the same shape as the direction. If no image
                    is used, the saturation for all image pixels will be set to 1

        value:  Weight image by using the value. Use either a 2D image
                or a 3D image with the same shape as the direction. If no image
                is used, the value for all image pixels will be set to 1

        colormap: The colormap to use. Default is HSV black. The available color maps
                  can be found in the colormap class.

    Returns:

        numpy.ndarray: 2D image containing the resulting HSV orientation map

    """
    direction = numpy.array(direction)
    direction_shape = direction.shape
    if inclination is None:
        inclination = numpy.zeros_like(direction)

    colors = colormap(direction, inclination)
    hsv_colors = rgb_to_hsv(colors)

    # If no saturation is given, create an "empty" saturation image that will be used
    if saturation is None:
        saturation = numpy.ones(direction.shape)
    # Normalize saturation image
    saturation = saturation / saturation.max(axis=None)
    # If we have a saturation image, check if the shape matches (3D) and correct accordingly
    while len(saturation.shape) < len(direction.shape):
        saturation = saturation[..., numpy.newaxis]
    if not saturation.shape[-1] == direction_shape[-1]:
        saturation = numpy.repeat(saturation, direction_shape[-1], axis=-1)

    # If no value is given, create an "empty" value image that will be used
    if value is None:
        value = numpy.ones(direction.shape)
    # Normalize value image
    value = value / value.max(axis=None)
    # If we have a value image, check if the shape matches (3D) and correct accordingly
    while len(value.shape) < len(direction.shape):
        value = value[..., numpy.newaxis]
    if not value.shape[-1] == direction_shape[-1]:
        value = numpy.repeat(value, direction_shape[-1], axis=-1)

    hsv_colors[..., 1] *= saturation
    hsv_colors[..., 2] *= value
    colors = hsv_to_rgb(hsv_colors)

    if len(direction_shape) > 2:
        return (255.0 * _visualize_multiple_direction(direction, colors)).astype(numpy.uint8)
    else:
        return (255.0 * _visualize_one_direction(direction, colors)).astype(numpy.uint8)
