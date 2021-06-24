import numba
import numpy

_multiprocessing_worker_fourier_var_dict = {}


def _init_worker_fourier_smoothing(X, X_shape):
    _multiprocessing_worker_fourier_var_dict['X'] = X
    _multiprocessing_worker_fourier_var_dict['X_shape'] = X_shape


def _worker_function_fourier_smoothing(i, threshold, window):
    x = i % _multiprocessing_worker_fourier_var_dict['X_shape'][0]
    y = i // _multiprocessing_worker_fourier_var_dict['X_shape'][0]
    image = numpy.frombuffer(_multiprocessing_worker_fourier_var_dict['X'])\
        .reshape(_multiprocessing_worker_fourier_var_dict['X_shape'])

    fft = numpy.fft.fft(image[x, y, :])
    frequencies = numpy.fft.fftfreq(len(fft))
    frequencies = frequencies / frequencies.max()

    multiplier = 1 - (0.5 + 0.5 * numpy.tanh(
        (numpy.abs(frequencies) - threshold) / window))
    fft = numpy.multiply(fft, multiplier[numpy.newaxis, numpy.newaxis, ...])

    image[x, y, :] = numpy.real(numpy.fft.ifft(fft)).astype(image.dtype)


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
