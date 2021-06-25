import numpy

import SLIX
from SLIX.CPU._toolbox import _direction, _prominence, _peakwidth, \
    _peakdistance, _centroid, _centroid_correction_bases, _peaks

__all__ = ['TARGET_PROMINENCE', 'peaks',
           'peak_width', 'peak_prominence',
           'peak_distance', 'mean_peak_distance',
           'background_mask', 'mean_peak_width',
           'direction', 'num_peaks', 'mean_peak_prominence',
           'unit_vectors', 'centroid_correction', 'normalize']

TARGET_PROMINENCE = 0.08


def background_mask(image):
    """
    Creates a background mask by setting all image pixels with low scattering
    signals to zero. As all background pixels are near zero for all images in
    the SLI image stack, this method should remove most of the background
    allowing for better approximations using the available features.
    It is advised to use this function.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

        threshold: Threshhold for mask creation (default: 10)

    Returns:

        numpy.array: 1D/2D-image which masks the background as True and
                     foreground as False
    """
    avg_image = numpy.average(image, axis=-1)
    # Set histogram to a range of 0 to 1 ignoring any outliers.
    hist_avg_image = avg_image / numpy.percentile(avg_image, 99)
    # Generate histogram in range of 0 to 1 to ignore outliers again. We search for values at the beginning anyway.
    avg_hist, avg_bins = numpy.histogram(hist_avg_image, bins=256, range=(0, 1))
    # Use SLIX to search for significant peaks in the histogram
    avg_hist = avg_hist[numpy.newaxis, numpy.newaxis, ...]
    peaks = SLIX.toolbox.significant_peaks(image=avg_hist).flatten()
    # Reverse the histogram to search for minimal values with SLIX (again)
    avg_hist = -avg_hist
    reversed_peaks = SLIX.toolbox.significant_peaks(image=avg_hist).flatten()

    # We can now calculate the index of our background threshold using the reversed_peaks
    index = numpy.argmax(reversed_peaks)
    # Reverse from 0 to 1 to original image scale and calculate the threshold position
    threshold = avg_bins[index] * numpy.percentile(avg_image, 99)

    # Return a mask with the calculated background image
    return avg_image < threshold


def peaks(image):
    """
Detect all peaks from a full SLI measurement. Peaks will not be filtered
    in any way. To detect only significant peaks, filter the peaks by using
    the prominence as a threshold.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

    Returns:

    2D/3D boolean image containing masking the peaks with `True`
    """
    image = numpy.array(image, dtype=numpy.float32)
    
    reshape = False
    if len(image.shape) == 3:
        reshape = True
        [image_x, image_y, image_z] = image.shape
        image = image.reshape(image_x * image_y, image_z)

    resulting_image = _peaks(image)

    if reshape:
        image = image.reshape(image_x, image_y, image_z)
        resulting_image = resulting_image.reshape(image_x, image_y, image_z)
    return resulting_image.astype('bool')


def num_peaks(image=None, peak_image=None):
    """
Calculate the number of peaks from each line profile in an SLI image series
    by detecting all peaks and applying thresholds to remove unwanted peaks.

    Args:

        image: Full SLI measurement (series of images) which is prepared for the
               pipeline using the SLIX toolbox methods.

        peak_image: Boolean NumPy array specifying the peak positions in the full SLI stack

    Returns:

        Array where each entry corresponds to the number of detected peaks within
        the first dimension of the SLI image series.
    """
    if peak_image is None and image is not None:
        peak_image = peaks(image)
    elif peak_image is not None:
        peak_image = numpy.array(peak_image)
    else:
        raise ValueError('Either image or peak_image has to be defined.')

    return numpy.count_nonzero(peak_image, axis=-1).astype(numpy.uint16)


def normalize(image, kind_of_normalization=0):
    """
    Normalize given line profile by using a normalization technique based on
    the kind_of_normalization parameter.

    0 : Scale line profile to be between 0 and 1

    1 : Divide line profile through its mean value

    Arguments:
        image: Full SLI measurement (series of images) which is
               prepared for the pipeline using the SLIX toolbox methods.
        kind_of_normalization: Normalization technique which will be used for
        the calculation

    Returns:
        numpy.array -- Image where each pixel is normalized by the last axis
        of the image
    """

    image = numpy.array(image, dtype=numpy.float32)
    if kind_of_normalization == 0:
        image = (image - image.min(axis=-1)[..., None]) \
                / numpy.maximum(1e-15, image.max(axis=-1)[..., None] -
                                image.min(axis=-1)[..., None])
    else:
        image = image / \
                numpy.maximum(1e-15, numpy.mean(image, axis=-1)[..., None])
    return image


def peak_prominence(image, peak_image=None, kind_of_normalization=0):
    """
    Calculate the peak prominence of all given peak positions within a line
    profile. The line profile will be normalized by dividing the line profile
    through its mean value. Therefore, values above 1 are possible.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        kind_of_normalization: Normalize given line profile by using a
        normalization technique based on the kind_of_normalization parameter.
           0 : Scale line profile to be between 0 and 1
           1 : Divide line profile through its mean value

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees. the mean peak prominence of the line
    profile in degrees.
    """
    image = numpy.array(image, dtype=numpy.float32)
    if peak_image is None:
        peak_image = peaks(image).astype('uint8')
    else:
        peak_image = numpy.array(peak_image).astype('uint8')
    image = normalize(image, kind_of_normalization)

    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    result_img = _prominence(image, peak_image)

    result_img = result_img.reshape((image_x, image_y, image_z))
    return result_img


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0):
    """
        Calculate the mean peak prominence of all given peak positions within a
    line profile. The line profile will be normalized by dividing the line
    profile through its mean value. Therefore, values above 1 are possible.

    Args:

        image: Original line profile used to detect all peaks. This array will be
            further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        kind_of_normalization: Normalize given line profile by using a
        normalization technique based on the kind_of_normalization parameter.
           0 : Scale line profile to be between 0 and 1
           1 : Divide line profile through its mean value

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees.
    """
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')
    result_img = peak_prominence(image, peak_image, kind_of_normalization)
    result_img = numpy.sum(result_img, axis=-1) / \
                 numpy.maximum(1, numpy.count_nonzero(peak_image, axis=-1))
    return result_img.astype('float32')


def peak_width(image, peak_image=None, target_height=0.5):
    """
    Calculate the peak width of all given peak positions within a line profile.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        target_height: Relative peak height in relation to the prominence of the
        given peak.

    Returns:

        NumPy array where each entry corresponds to the peak width of the line
        profile. The values are in degree.
    """
    image = numpy.array(image, dtype='float32')
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')

    [image_x, image_y, image_z] = image.shape

    image = image.reshape(image_x * image_y, image_z)
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    prominence = _prominence(image, peak_image)
    result_image = _peakwidth(image, peak_image, prominence, target_height)

    result_image = result_image.reshape((image_x, image_y, image_z))
    result_image = result_image * 360.0 / image_z

    return result_image


def mean_peak_width(image, peak_image=None, target_height=0.5):
    """
    Calculate the mean peak width of all given peak positions within a line
    profile.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        target_height: Relative peak height in relation to the prominence of the
        given peak.

    Returns:

        NumPy array where each entry corresponds to the mean peak width of the
        line profile. The values are in degree.
    """
    if peak_image is not None:
        peak_image = numpy.array(peak_image).astype('uint8')
    else:
        peak_image = peaks(image).astype('uint8')
    result_img = peak_width(image, peak_image, target_height)
    result_img = numpy.sum(result_img, axis=-1) / \
                 numpy.maximum(1, numpy.count_nonzero(peak_image, axis=-1))

    return result_img


def peak_distance(peak_image, centroids):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        centroids: Use centroid calculation to better determine the peak position
        regardless of the number of
        measurements / illumination angles used.

    Returns:

        NumPy array of floating point values containing the peak distance of the
        line profiles in degrees in their respective peak position. The first peak
        of each peak pair will show the distance between peak_1 and peak_2 while
        the second peak will show 360 - (peak_2 - peak_1).
    """
    peak_image = numpy.array(peak_image).astype('uint8')
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('uint8')
    centroids = centroids.reshape(image_x * image_y, image_z).astype('float32')

    result_img = _peakdistance(peak_image, centroids, number_of_peaks)
    result_img = result_img.reshape((image_x, image_y, image_z))

    return result_img


def mean_peak_distance(peak_image, centroids):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        centroids: Use centroid calculation to better determine the peak position
        regardless of the number of
        measurements / illumination angles used.

    Returns:

        NumPy array of floating point values containing the mean peak distance of
        the line profiles in degrees.
    """
    result_image = peak_distance(peak_image, centroids)
    result_image[result_image > 180] = 0
    result_image = numpy.sum(result_image, axis=-1) / \
                   numpy.maximum(1, numpy.count_nonzero(result_image, axis=-1))
    return result_image


def direction(peak_image, centroids, correction_angle=0,
              number_of_directions=3):
    """
    Calculate up to `number_of_directions` direction angles based on the given
    peak positions. If more than `number_of_directions*2` peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a
    direction angle of BACKGROUND_COLOR. The peak positions are determined by
    the position of the corresponding peak pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the
    direction angle will be considered as invalid, resulting in a direction
    angle of BACKGROUND_COLOR.

    Args:

        correction_angle: Correct the resulting direction angle by the value.
        This is useful when the stack or camera was rotated.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        centroids: Centroids resulting from `centroid_correction` for more accurate
                   results

        number_of_directions: Number of directions which shall be generated.

    Returns:

        NumPy array with the shape (x, y, `number_of_directions`) containing up to
        `number_of_directions` direction angles. x equals the number of pixels of
        the SLI image series. If a direction angle is invalid or missing, the
        array entry will be BACKGROUND_COLOR instead.
    """
    peak_image = numpy.array(peak_image).astype('uint8')
    [image_x, image_y, image_z] = peak_image.shape

    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')
    centroids = centroids.reshape(image_x * image_y, image_z).astype('float32')
    number_of_peaks = numpy.count_nonzero(peak_image, axis=-1).astype('uint8')

    result_img = _direction(peak_image, centroids, number_of_peaks,
                            number_of_directions, correction_angle)
    result_img = result_img.reshape((image_x, image_y, number_of_directions))

    return result_img


def centroid_correction(image, peak_image, low_prominence=TARGET_PROMINENCE,
                        high_prominence=None):
    """
    Correct peak positions from a line profile by looking at only the peak
    with a given threshold using a centroid calculation. If a minimum is found
    in the considered interval, this minimum will be used as the limit instead.
    The range for the peak correction is limited by
    MAX_DISTANCE_FOR_CENTROID_ESTIMATION.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        low_prominence: Lower prominence bound for detecting a peak.

        high_prominence: Higher prominence bound for detecting a peak.

    Returns:

        NumPy array with the positions of all detected peak positions corrected
        with the centroid calculation.
    """
    if peak_image is None:
        peak_image = peaks(image).astype('uint8')
    if low_prominence is None:
        low_prominence = -numpy.inf
    if high_prominence is None:
        high_prominence = -numpy.inf

    [image_x, image_y, image_z] = image.shape
    image = normalize(image)
    image = image.reshape(image_x * image_y, image_z).astype('float32')
    peak_image = peak_image.reshape(image_x * image_y, image_z).astype('uint8')

    reverse_image = -1 * image
    reverse_peaks = peaks(reverse_image.reshape((image_x, image_y, image_z))) \
        .astype('uint8') \
        .reshape(image_x * image_y, image_z)
    reverse_prominence = _prominence(reverse_image, reverse_peaks)

    reverse_peaks[reverse_prominence < low_prominence] = False
    reverse_peaks[reverse_prominence > high_prominence] = False

    left_bases, right_bases = _centroid_correction_bases(image, peak_image,
                                                         reverse_peaks)
    # Centroid calculation based on left_bases and right_bases
    centroid_peaks = _centroid(image, peak_image, left_bases, right_bases)
    centroid_peaks = centroid_peaks.reshape((image_x, image_y, image_z))

    return centroid_peaks


def unit_vectors(direction):
    """
    Calculate the unit vectors (UnitX, UnitY) from a given direction angle.

    Args:

        direction: 3D NumPy array - direction angles in degrees

    Returns:

        UnitX, UnitY: 3D NumPy array, 3D NumPy array
            x- and y-vector component in arrays
    """
    directions_rad = numpy.deg2rad(direction)
    UnitX = -numpy.sin(0.5 * numpy.pi) * numpy.cos(directions_rad)
    UnitY = numpy.sin(0.5 * numpy.pi) * numpy.sin(directions_rad)

    UnitX[numpy.isclose(direction, -1)] = 0
    UnitY[numpy.isclose(direction, -1)] = 0

    return UnitX, UnitY
