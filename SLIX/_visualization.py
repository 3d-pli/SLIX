import numba
import numpy


@numba.jit
def _count_nonzero(image):
    iterator_image = image.flatten()
    number_of_pixels = 0

    for i in range(len(iterator_image)):
        if iterator_image[i] != 0:
            number_of_pixels += 1

    return number_of_pixels


@numba.jit(parallel=False, nopython=True)
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
