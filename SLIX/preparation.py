import numpy
import scipy.signal
from SLIX._preparation import _thin_out_median, _thin_out_plain, \
                              _thin_out_average


def apply_smoothing(image, window_length=45, polyorder=2):
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
