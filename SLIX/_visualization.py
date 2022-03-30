import numba
import numpy


@numba.njit()
def _count_nonzero(image):
    iterator_image = image.flatten()
    number_of_pixels = 0

    for i in range(len(iterator_image)):
        if iterator_image[i] != 0:
            number_of_pixels += 1

    return number_of_pixels


@numba.jit(parallel=True, nopython=True)
def _downsample_2d(image, kernel_size,
                   background_threshold, background_value):
    nx = int(numpy.ceil(image.shape[0] / kernel_size))
    ny = int(numpy.ceil(image.shape[1] / kernel_size))
    output_image = numpy.empty((nx, ny))

    output_image[:, :] = background_value

    for i in numba.prange(0, nx):
        for j in numba.prange(0, ny):
            roi = image[kernel_size * i:kernel_size * i + kernel_size,
                        kernel_size * j:kernel_size * j + kernel_size]
            roi = roi.flatten()
            number_of_valid_vectors = _count_nonzero(roi != background_value)

            if number_of_valid_vectors >= background_threshold * roi.size:
                valid_vectors = 0
                roi.sort()

                for idx in range(roi.size):
                    if roi[idx] != background_value:
                        valid_vectors += 1

                    if valid_vectors == number_of_valid_vectors // 2:
                        if number_of_valid_vectors % 2 == 0:
                            output_image[i, j] = roi[idx]
                        else:
                            output_image[i, j] = (roi[idx+1] + roi[idx]) / 2

    return output_image


def _downsample(image, kernel_size, background_threshold=0,
                background_value=0):
    nx = int(numpy.ceil(image.shape[0] / kernel_size))
    ny = int(numpy.ceil(image.shape[1] / kernel_size))
    if len(image.shape) < 3:
        z = 1
    else:
        z = image.shape[2]
    result_image = numpy.empty((nx, ny, z))

    for sub_image in range(z):
        result_image[:, :, sub_image] = \
            _downsample_2d(image[:, :, sub_image], kernel_size,
                           background_threshold, background_value)

    result_image = numpy.squeeze(result_image)
    return result_image


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


def _plot_axes_unit_vectors(ax, mesh_x, mesh_y, mesh_u, mesh_v,
                            scale, alpha, vector_width, weighting, colormap):
    # Normalize the arrows:
    mesh_u_normed = mesh_u / numpy.sqrt(numpy.maximum(1e-15,
                                                      mesh_u ** 2 +
                                                      mesh_v ** 2))
    mesh_v_normed = mesh_v / numpy.sqrt(numpy.maximum(1e-15,
                                                      mesh_u ** 2 +
                                                      mesh_v ** 2))

    # Convert to RGB colors
    normed_angle = numpy.abs(numpy.arctan2(mesh_v_normed, -mesh_u_normed))
    color_rgb = colormap(normed_angle, numpy.zeros_like(normed_angle))

    # Apply weighting
    if weighting is not None:
        mesh_u_normed = weighting * mesh_u_normed
        mesh_v_normed = weighting * mesh_v_normed

    # Apply scaling
    mesh_u_normed *= scale
    mesh_v_normed *= scale

    mesh_u_normed[numpy.isclose(mesh_u, 0) &
                  numpy.isclose(mesh_v, 0)] = numpy.nan
    mesh_v_normed[numpy.isclose(mesh_u, 0) &
                  numpy.isclose(mesh_v, 0)] = numpy.nan

    # 1/scale to increase vector length for scale > 1
    ax.quiver(mesh_x, mesh_y, mesh_u_normed, mesh_v_normed,
              color=color_rgb, units='xy',
              angles='xy', scale_units='xy',
              scale=1, headwidth=0, headlength=0,
              headaxislength=0, minlength=0, pivot='mid',
              alpha=alpha, width=vector_width,
              edgecolors=color_rgb)
