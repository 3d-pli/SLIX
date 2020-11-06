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

    conc_image = numpy.concatenate((image[:, :, image.shape[2]//2:],
                                    image,
                                    image[:, :, :image.shape[2]//2]), axis=-1)
    conc_image = scipy.signal.savgol_filter(conc_image, window_length,
                                            polyorder, axis=-1)
    return conc_image[:, :, image.shape[2]//2:-image.shape[2]//2]


def thin_out(image, factor=2, strategy='plain'):
    if strategy == 'plain':
        return _thin_out_plain(image, factor)
    elif strategy == 'average':
        return _thin_out_average(image, factor)
    elif strategy == 'median':
        return _thin_out_median(image, factor)
