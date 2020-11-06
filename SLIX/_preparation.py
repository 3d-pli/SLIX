import numba
import numpy


@numba.jit(nopython=True)
def _thin_out_plain(image, factor):
    return image[::factor, ::factor, :]


@numba.jit(nopython=True, parallel=True)
def _thin_out_average(image, factor):
    nx = int(numpy.ceil(image.shape[0] / factor))
    ny = int(numpy.ceil(image.shape[1] / factor))
    result_image = numpy.empty((nx, ny, image.shape[2]), dtype=numpy.float64)

    for i in numba.prange(0, nx):
        for j in numba.prange(0, ny):
            for k in numba.prange(0, image.shape[2]):
                roi = image[i * factor:(i+1) * factor,
                            j * factor:(j+1) * factor,
                            k]
                result_image[i, j, k] = numpy.mean(roi)

    return result_image


@numba.jit(nopython=True, parallel=True)
def _thin_out_median(image, factor):
    nx = int(numpy.ceil(image.shape[0] / factor))
    ny = int(numpy.ceil(image.shape[1] / factor))
    result_image = numpy.empty((nx, ny, image.shape[2]), dtype=numpy.float64)

    for i in numba.prange(0, nx):
        for j in numba.prange(0, ny):
            for k in numba.prange(0, image.shape[2]):
                roi = image[i * factor:(i+1) * factor,
                            j * factor:(j+1) * factor,
                            k]
                result_image[i, j, k] = numpy.median(roi)

    return result_image