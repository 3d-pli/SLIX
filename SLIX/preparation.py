import multiprocessing
from functools import partial
import numpy
from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray
import os
import scipy.signal as signal
from SLIX._preparation import _thin_out_median, _thin_out_plain, \
    _thin_out_average, _init_worker_fourier_smoothing, \
    _worker_function_fourier_smoothing

__all__ = ['thin_out', 'savitzky_golay_smoothing',
           'low_pass_fourier_smoothing']


def low_pass_fourier_smoothing(image, threshold=0.2, smoothing_factor=0.025):
    """
    Applies Low Pass fourier filter to given line profiles / image
    and returns the smoothened measurement.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

        threshold: Threshold percentage of low frequencies
                   which will completely pass

        smoothing_factor: Apply a smoothing factor which will smooth out the 
                          applied multiplication factor for the low pass filter.
                          A higher value will result in more smoothing of the curve.
                          Values between 1e-15 and 1 are accepted. Other values 
                          might result in an error.

    Returns:

        Complete SLI measurement image with applied Low Pass fourier filter
        and the same shape as the original image.
    """
    X_shape = image.shape
    X = RawArray('d', X_shape[0] * X_shape[1] * X_shape[2])
    X_np = numpy.frombuffer(X).reshape(X_shape)

    numpy.copyto(X_np, image)

    partial_worker_function = partial(_worker_function_fourier_smoothing,
                                      threshold=threshold, window=smoothing_factor)

    with Pool(processes=os.cpu_count(),
              initializer=_init_worker_fourier_smoothing,
              initargs=(X, X_shape)) as pool:
        pool.map(partial_worker_function,
                 range(X_shape[0] * X_shape[1]))

    return X_np.astype(image.dtype)



def savitzky_golay_smoothing(image, window_length=45, polyorder=2):
    """
    Applies Savitzky-Golay filter to given line profiles / image
    and returns the smoothened measurement.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

        window_length: Used window length for the Savitzky-Golay filter

        polyorder: Used polynomial order for the Savitzky-Golay filter

    Returns:

        Complete SLI measurement image with applied Savitzky-Golay filter
        and the same shape as the original image.
    """
    conc_image = numpy.concatenate((image[:, :, -window_length:],
                                    image,
                                    image[:, :, :window_length]), axis=-1)
    conc_image = signal.savgol_filter(conc_image, window_length,
                                      polyorder, axis=-1)
    return conc_image[:, :, window_length:-window_length]


def thin_out(image, factor=2, strategy='plain'):
    """
    Thin out the image stack used for SLIX. This can be useful when the image
    stack is quite large and should be processed quickly. This can also prove
    useful if there is a lot of noise that could be filtered by using a lower
    resolution image.

    Args:

        image: Image that should be thinned out.

        factor: Factor which will be used for thinning the image. A factor
                of N means that every N-th pixel will be kept.

        strategy: Strategy used for thinning out the image. Available methods:
                  'plain' (keep the pixel),
                  'average' (calculate average of area),
                  'median' (calculate the median of area)

    Returns:

        numpy.ndarray with the thinned out image

    """
    strategy = strategy.lower()
    if strategy == 'plain':
        return _thin_out_plain(image, factor)
    elif strategy == 'average':
        return _thin_out_average(image, factor)
    elif strategy == 'median':
        return _thin_out_median(image, factor)
    else:
        raise ValueError('Strategy not implemented. Known strategies are:'
                         ' plain, average, median.')
