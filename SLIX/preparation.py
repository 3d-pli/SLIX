import numpy
import scipy.signal
from SLIX._preparation import _thin_out_median, _thin_out_plain, \
    _thin_out_average


def low_pass_fourier_smoothing(image, threshold_low=10, threshold_high=25):
    fft = numpy.fft.fft(image, axis=-1)

    # Define thresholds for low pass filter
    threshold_start_position = image.shape[-1] * threshold_low // 100
    threshold_end_position = image.shape[-1] * threshold_high // 100

    magnitude = numpy.abs(fft)
    magnitude_copy = -magnitude.copy()
    magnitude_threshold_low = -1.0 * numpy.sort(magnitude_copy, axis=-1) \
        [:, :, threshold_end_position][..., numpy.newaxis]
    magnitude_threshold_high = -1.0 * numpy.sort(magnitude_copy, axis=-1) \
        [:, :, threshold_start_position][..., numpy.newaxis]

    interval = numpy.maximum(1e-15,
                             magnitude_threshold_high - magnitude_threshold_low)
    middle_point = magnitude_threshold_low + 0.5 * magnitude_threshold_high

    # Calculate low pass filter and apply it to our original signal
    multiplier = 0.5 + 0.5 * numpy.tanh((magnitude-middle_point) / interval)
    fft = numpy.multiply(multiplier, fft)

    # Apply inverse fourier transform
    image = numpy.real(numpy.fft.ifft(fft)).astype(numpy.float32)
    return image


def savitzky_golay_smoothing(image, window_length=45, polyorder=2):
    """
    Applies Savitzky-Golay filter to given roiset and returns the
    smoothened measurement.

    Args:
        image: Complete SLI measurement image stack as a 2D/3D Numpy array
        window_length: Used window length for the Savitzky-Golay filter
        polyorder: Used polynomial order for the Savitzky-Golay filter

    Returns: Complete SLI measurement image with applied Savitzky-Golay filter
    and the same shape as the original image.
    """
    print(image.shape)
    conc_image = numpy.concatenate((image[:, :, -window_length:],
                                    image,
                                    image[:, :, :window_length]), axis=-1)
    print(conc_image.shape)
    conc_image = scipy.signal.savgol_filter(conc_image, window_length,
                                            polyorder, axis=2)
    return conc_image[:, :, window_length:-window_length]


def thin_out(image, factor=2, strategy='plain'):
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
