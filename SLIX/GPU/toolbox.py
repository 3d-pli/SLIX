import cupy
import numpy
from numba import cuda

import SLIX
from SLIX.GPU._toolbox import _direction, _prominence, _peakwidth, \
    _peakdistance, _centroid_correction_bases, _centroid, \
    _peaks

__all__ = ['TARGET_PROMINENCE', 'peaks',
           'peak_width', 'peak_prominence',
           'peak_distance', 'mean_peak_distance',
           'background_mask', 'mean_peak_width',
           'direction', 'num_peaks', 'mean_peak_prominence',
           'unit_vectors', 'centroid_correction', 'normalize']

TARGET_PROMINENCE = 0.08


def background_mask(image, return_numpy=True):
    """
    Creates a background mask by setting all image pixels with low scattering
    signals to zero. As all background pixels are near zero for all images in
    the SLI image stack, this method should remove most of the background
    allowing for better approximations using the available features.
    It is advised to use this function.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

        threshold: Threshhold for mask creation (default: 10)

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      Numpy array will be returned.

    Returns:

        numpy.array: 1D/2D-image which masks the background as True and
                     foreground as False
    """
    gpu_image = cupy.array(image, dtype='float32')
    gpu_average = cupy.average(gpu_image, axis=-1)

    # Set histogram to a range of 0 to 1 ignoring any outliers.
    hist_avg_image = gpu_average / cupy.percentile(gpu_image, 99)
    # Generate histogram in range of 0 to 1 to ignore outliers again. We search for values at the beginning anyway.
    avg_hist, avg_bins = cupy.histogram(hist_avg_image, bins=256, range=(0, 1))
    # Use SLIX to search for significant peaks in the histogram
    avg_hist = avg_hist[numpy.newaxis, numpy.newaxis, ...]
    peaks = SLIX.toolbox.significant_peaks(image=avg_hist).flatten()
    # Reverse the histogram to search for minimal values with SLIX (again)
    avg_hist = -avg_hist
    reversed_peaks = SLIX.toolbox.significant_peaks(image=avg_hist).flatten()

    # We can now calculate the index of our background threshold using the reversed_peaks
    index = numpy.argmax(reversed_peaks)
    # Reverse from 0 to 1 to original image scale and calculate the threshold position
    threshold = avg_bins[index] * numpy.percentile(gpu_average, 99)

    # Return a mask with the calculated background image
    gpu_mask = gpu_average < threshold

    if return_numpy:
        cpu_mask = cupy.asnumpy(gpu_mask)
        del gpu_image
        del gpu_mask
        return cpu_mask
    else:
        return gpu_mask


def peaks(image, return_numpy=True):
    """
    Detect all peaks from a full SLI measurement. Peaks will not be filtered
    in any way. To detect only significant peaks, filter the peaks by using
    the prominence as a threshold.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D Numpy array

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      Numpy array will be returned.

    Returns:

    2D/3D boolean image containing masking the peaks with `True`
    """

    gpu_image = cupy.array(image, dtype='float32')

    resulting_peaks = cupy.zeros(gpu_image.shape, dtype='int8')
    threads_per_block = (1, 1)
    blocks_per_grid = image.shape[:-1]
    _peaks[blocks_per_grid, threads_per_block](gpu_image, resulting_peaks)
    cuda.synchronize()

    if return_numpy:
        peaks_cpu = cupy.asnumpy(resulting_peaks)
        del resulting_peaks

        return peaks_cpu.astype('bool')
    else:
        return resulting_peaks.astype('bool')


def num_peaks(image=None, peak_image=None, return_numpy=True):
    """
    Calculate the number of peaks from each line profile in an SLI image series
    by detecting all peaks and applying thresholds to remove unwanted peaks.

    Args:

        image: Full SLI measurement (series of images) which is prepared for the
               pipeline using the SLIX toolbox methods.

        peak_image: Boolean NumPy array specifying the peak positions in the full SLI stack

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
                      array will be returned.

    Returns:

        Array where each entry corresponds to the number of detected peaks within
        the first dimension of the SLI image series.
    """

    if peak_image is None and image is not None:
        peak_image = peaks(image, return_numpy=False)
    elif peak_image is not None:
        peak_image = cupy.array(peak_image)
    else:
        raise ValueError('Either image or peak_image has to be defined.')

    resulting_image = cupy.count_nonzero(peak_image, axis=-1)\
                          .astype(cupy.uint16)
    if return_numpy:
        resulting_image_cpu = cupy.asnumpy(resulting_image)
        del resulting_image
        return resulting_image_cpu
    else:
        return resulting_image


def normalize(image, kind_of_normalization=0, return_numpy=True):
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
        return_numpy:  Specifies if a CuPy or Numpy array will be returned.

    Returns:
        numpy.array -- Image where each pixel is normalized by the last axis
        of the image
    """
    gpu_image = cupy.array(image, dtype='float32')
    if kind_of_normalization == 0:
        gpu_image = (gpu_image - cupy.min(gpu_image, axis=-1)[..., None]) / \
                    cupy.maximum(1e-15, (cupy.max(gpu_image, axis=-1)
                                         [..., None] -
                                         cupy.min(gpu_image, axis=-1)
                                         [..., None]))
    else:
        gpu_image = gpu_image / cupy.mean(gpu_image, axis=-1)[..., None]

    if return_numpy:
        gpu_image_cpu = cupy.asnumpy(gpu_image)
        del gpu_image
        return gpu_image_cpu
    else:
        return gpu_image


def peak_prominence(image, peak_image=None, kind_of_normalization=0,
                    return_numpy=True):
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

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees.
    """
    gpu_image = cupy.array(image, dtype='float32')
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image).astype('uint8')
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False).astype('uint8')
    gpu_image = normalize(gpu_image, kind_of_normalization, return_numpy=False)

    result_img_gpu = cupy.zeros(gpu_image.shape, dtype='float32')

    threads_per_block = (1, 1)
    blocks_per_grid = gpu_peak_image.shape[:-1]
    _prominence[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image,
                                                    result_img_gpu)
    cuda.synchronize()

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        if isinstance(image, type(numpy.zeros(0))):
            del gpu_image
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=0,
                         return_numpy=True):
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

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees.
    """
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image).astype('uint8')
    else:
        gpu_peak_image = peaks(image, return_numpy=False).astype('uint8')
    peak_prominence_gpu = peak_prominence(image, peak_image,
                                          kind_of_normalization,
                                          return_numpy=False)
    peak_prominence_gpu = cupy.sum(peak_prominence_gpu, axis=-1) / \
                          cupy.maximum(1, cupy.count_nonzero(gpu_peak_image,
                                                             axis=-1))
    peak_prominence_gpu = peak_prominence_gpu.astype('float32')

    del gpu_peak_image
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_prominence_gpu)
        del peak_prominence_gpu
        return peak_width_cpu
    else:
        return peak_prominence_gpu


def peak_width(image, peak_image=None, target_height=0.5, return_numpy=True):
    """
    Calculate the peak width of all given peak positions within a line profile.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        target_height: Relative peak height in relation to the prominence of the
        given peak.

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array where each entry corresponds to the peak width of the line
        profile. The values are in degree.
    """
    gpu_image = cupy.array(image, dtype='float32')
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image).astype('uint8')
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False).astype('uint8')

    threads_per_block = (1, 1)
    blocks_per_grid = gpu_peak_image.shape[:-1]

    gpu_prominence = cupy.empty(gpu_image.shape, dtype='float32')
    _prominence[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image,
                                                    gpu_prominence)
    cuda.synchronize()

    result_image_gpu = cupy.zeros(gpu_image.shape, dtype='float32')
    _peakwidth[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image,
                                                   gpu_prominence,
                                                   result_image_gpu,
                                                   target_height)
    cuda.synchronize()

    del gpu_prominence
    if peak_image is None:
        del gpu_peak_image

    result_image_gpu = result_image_gpu * 360.0 / gpu_image.shape[-1]

    if return_numpy:
        if isinstance(image, type(numpy.zeros(0))):
            del gpu_image
        result_image_cpu = cupy.asnumpy(result_image_gpu)
        del result_image_gpu
        return result_image_cpu
    else:
        return result_image_gpu


def mean_peak_width(image, peak_image=None, target_height=0.5,
                    return_numpy=True):
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

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array where each entry corresponds to the mean peak width of the
        line profile. The values are in degree.
    """
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image).astype('uint8')
    else:
        gpu_peak_image = peaks(image, return_numpy=False).astype('uint8')
    peak_width_gpu = peak_width(image, gpu_peak_image, target_height,
                                return_numpy=False)
    peak_width_gpu = cupy.sum(peak_width_gpu, axis=-1) / \
                     cupy.maximum(1, cupy.count_nonzero(gpu_peak_image,
                                                        axis=-1))

    del gpu_peak_image
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_width_gpu)
        del peak_width_gpu
        return peak_width_cpu
    else:
        return peak_width_gpu


def peak_distance(peak_image, centroids, return_numpy=True):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        centroids: Use centroid calculation to better determine the peak position
        regardless of the number of
        measurements / illumination angles used.

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array of floating point values containing the peak distance of the
        line profiles in degrees in their respective peak position. The first peak
        of each peak pair will show the distance between peak_1 and peak_2 while
        the second peak will show 360 - (peak_2 - peak_1).
    """
    gpu_peak_image = cupy.array(peak_image).astype('uint8')
    gpu_centroids = cupy.array(centroids).astype('float32')

    number_of_peaks = num_peaks(peak_image=gpu_peak_image,
                                return_numpy=False).astype('int8')
    result_image_gpu = cupy.zeros(gpu_peak_image.shape, dtype='float32')

    threads_per_block = (1, 1)
    blocks_per_grid = gpu_peak_image.shape[:-1]
    _peakdistance[blocks_per_grid, threads_per_block](gpu_peak_image,
                                                      gpu_centroids,
                                                      number_of_peaks,
                                                      result_image_gpu)
    cuda.synchronize()

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        result_image_cpu = cupy.asnumpy(result_image_gpu)
        del result_image_gpu
        return result_image_cpu
    else:
        return result_image_gpu


def mean_peak_distance(peak_image, centroids, return_numpy=True):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        centroids: Use centroid calculation to better determine the peak position
        regardless of the number of
        measurements / illumination angles used.

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array of floating point values containing the mean peak distance of
        the line profiles in degrees.
    """
    peak_distance_gpu = peak_distance(peak_image, centroids,
                                      return_numpy=False)
    peak_distance_gpu[peak_distance_gpu > 180] = 0
    peak_distance_gpu = cupy.sum(peak_distance_gpu, axis=-1) / \
                        cupy.maximum(1, cupy.count_nonzero(peak_distance_gpu,
                                                           axis=-1))
    if return_numpy:
        peak_width_cpu = cupy.asnumpy(peak_distance_gpu)
        del peak_distance_gpu
        return peak_width_cpu
    else:
        return peak_distance_gpu


def direction(peak_image, centroids, correction_angle=0,
              number_of_directions=3, return_numpy=True):
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

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array with the shape (x, y, `number_of_directions`) containing up to
        `number_of_directions` direction angles. x equals the number of pixels of
        the SLI image series. If a direction angle is invalid or missing, the
        array entry will be BACKGROUND_COLOR instead.
    """
    gpu_peak_image = cupy.array(peak_image).astype('int8')
    gpu_centroids = cupy.array(centroids).astype('float32')

    result_img_gpu = cupy.empty(
        (gpu_peak_image.shape[0], gpu_peak_image.shape[1],
         number_of_directions), dtype='float32')
    number_of_peaks = cupy.count_nonzero(gpu_peak_image, axis=-1).astype(
        'int8')

    threads_per_block = (1, 1)
    blocks_per_grid = gpu_peak_image.shape[:-1]
    _direction[blocks_per_grid, threads_per_block](gpu_peak_image,
                                                   gpu_centroids,
                                                   number_of_peaks,
                                                   result_img_gpu,
                                                   correction_angle)
    cuda.synchronize()
    del number_of_peaks

    if peak_image is None:
        del gpu_peak_image

    if return_numpy:
        result_img_cpu = cupy.asnumpy(result_img_gpu)
        del result_img_gpu
        return result_img_cpu
    else:
        return result_img_gpu


def centroid_correction(image, peak_image, low_prominence=TARGET_PROMINENCE,
                        high_prominence=None, return_numpy=True):
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

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        NumPy array with the positions of all detected peak positions corrected
        with the centroid calculation.
    """
    gpu_image = normalize(cupy.array(image, dtype='float32'))
    if peak_image is not None:
        gpu_peak_image = cupy.array(peak_image, dtype='uint8')
    else:
        gpu_peak_image = peaks(gpu_image, return_numpy=False).astype('uint8')
    if low_prominence is None:
        low_prominence = -cupy.inf
    if high_prominence is None:
        high_prominence = -cupy.inf

    gpu_reverse_image = (-1 * gpu_image).astype('float32')
    gpu_reverse_peaks = peaks(gpu_reverse_image, return_numpy=False).astype(
        'uint8')
    gpu_reverse_prominence = cupy.empty(gpu_reverse_image.shape,
                                        dtype='float32')

    threads_per_block = (1, 1)
    blocks_per_grid = gpu_peak_image.shape[:-1]
    _prominence[blocks_per_grid, threads_per_block](gpu_reverse_image,
                                                    gpu_reverse_peaks,
                                                    gpu_reverse_prominence)
    cuda.synchronize()
    del gpu_reverse_image

    gpu_reverse_peaks[gpu_reverse_prominence < low_prominence] = False
    gpu_reverse_peaks[gpu_reverse_prominence > high_prominence] = False
    del gpu_reverse_prominence

    gpu_left_bases = cupy.empty(gpu_image.shape, dtype='int8')
    gpu_right_bases = cupy.empty(gpu_image.shape, dtype='int8')
    _centroid_correction_bases[blocks_per_grid,
                               threads_per_block](gpu_image,
                                                  gpu_peak_image,
                                                  gpu_reverse_peaks,
                                                  gpu_left_bases,
                                                  gpu_right_bases)
    cuda.synchronize()
    del gpu_reverse_peaks

    # Centroid calculation based on left_bases and right_bases
    gpu_centroid_peaks = cupy.empty(gpu_image.shape, dtype='float32')
    _centroid[blocks_per_grid, threads_per_block](gpu_image, gpu_peak_image,
                                                  gpu_left_bases,
                                                  gpu_right_bases,
                                                  gpu_centroid_peaks)
    cuda.synchronize()
    if peak_image is None:
        del gpu_peak_image
    del gpu_right_bases
    del gpu_left_bases

    if return_numpy:
        result_img_cpu = cupy.asnumpy(gpu_centroid_peaks)
        del gpu_centroid_peaks
        return result_img_cpu
    else:
        return gpu_centroid_peaks


def unit_vectors(direction, return_numpy=True):
    """
    Calculate the unit vectors (UnitX, UnitY) from a given direction angle.

    Args:

        direction: 3D NumPy array - direction angles in degrees

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or Numpy
        array will be returned.

    Returns:

        UnitX, UnitY: 3D NumPy array, 3D NumPy array
            x- and y-vector component in arrays
    """
    direction_gpu = cupy.array(direction)
    direction_gpu_rad = cupy.deg2rad(direction_gpu)
    UnitX = -cupy.sin(0.5 * cupy.pi) * cupy.cos(direction_gpu_rad)
    UnitY = cupy.sin(0.5 * cupy.pi) * cupy.sin(direction_gpu_rad)
    del direction_gpu_rad

    UnitX[cupy.isclose(direction_gpu, -1)] = 0
    UnitY[cupy.isclose(direction_gpu, -1)] = 0
    del direction_gpu

    if return_numpy:
        return UnitX.get(), UnitY.get()
    return UnitX, UnitY
