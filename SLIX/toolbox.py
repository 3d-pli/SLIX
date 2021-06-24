try:
    try:
        import cupy
        from numba import cuda

        cupy.empty((0), dtype=float)
        from SLIX.GPU import toolbox as gpu_toolbox

        gpu_available = True
    except cupy.cuda.runtime.CUDARuntimeError:
        print('[WARNING] CuPy is installed but an error was thrown by the '
              'runtime. SLIX will fall back to the CPU variant.')
        gpu_available = False
    except (cuda.cudadrv.driver.CudaAPIError,
            cuda.cudadrv.driver.LinkerError):
        print("[WARNING] Numba CUDA couldn't be initialized. "
              "Please check if there are problems with your CUDA / Numba "
              "version. SLIX will fall back to the CPU variant.")
        gpu_available = False
except (ModuleNotFoundError, NameError):
    gpu_available = False
    print('[WARNING] CuPy is not installed. The toolbox will use the CPU '
          'variant instead. If you want to use the GPU variant, please run '
          '`pip install cupy`.')
    
from SLIX.CPU import toolbox as cpu_toolbox
import numpy

__all__ = ['background_mask', 'centroid_correction',
           'direction', 'unit_vectors', 'num_peaks',
           'mean_peak_prominence', 'peaks',
           'peak_prominence', 'peak_width',
           'mean_peak_distance', 'peak_distance',
           'mean_peak_width', 'significant_peaks']


def background_mask(image, use_gpu=gpu_available,
                    return_numpy=True):
    """
    Creates a background mask by setting all image pixels with low scattering
    signals to zero. As all background pixels are near zero for all images in
    the SLI image stack, this method should remove most of the background
    allowing for better approximations using the available features.
    It is advised to use this function.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D NumPy array

        threshold: Threshhold for mask creation (default: 10)

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        numpy.array: 1D/2D-image which masks the background as True and
                     foreground as False
    """

    if use_gpu:
        return gpu_toolbox.background_mask(image, return_numpy)
    else:
        return cpu_toolbox.background_mask(image)


def peaks(image, use_gpu=gpu_available, return_numpy=True):
    """
    Detect all peaks from a full SLI measurement. Peaks will not be filtered
    in any way. To detect only significant peaks, filter the peaks by using
    the prominence as a threshold.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D NumPy array

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

    2D/3D boolean image containing masking the peaks with `True`
    """

    if use_gpu:
        return gpu_toolbox.peaks(image, return_numpy)
    else:
        return cpu_toolbox.peaks(image)


def significant_peaks(image, low_prominence=cpu_toolbox.TARGET_PROMINENCE,
                      high_prominence=numpy.inf,
                      use_gpu=gpu_available, return_numpy=True):
    """
    Detect all peaks from a full SLI measurement and filter them by passing
    thresholds.

    Args:

        image: Complete SLI measurement image stack as a 2D/3D NumPy array

        low_prominence: Minimum prominence needed by peak to count as a peak.
                        Peaks below this threshold will not be considered as a
                        peak.

        high_prominence: Maximum prominence needed by peak to count as a peak.
                         Peaks below this threshold will not be considered as a
                         peak.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        2D/3D boolean image containing masking the peaks with `True`

    """
    if use_gpu:
        peaks = gpu_toolbox.peaks(image, return_numpy=return_numpy)
        prominences = gpu_toolbox.peak_prominence(image, peaks,
                                                  return_numpy=return_numpy)
        peaks[prominences < low_prominence] = False
        peaks[prominences > high_prominence] = False
    else:
        peaks = cpu_toolbox.peaks(image)
        prominences = cpu_toolbox.peak_prominence(image, peaks)
        peaks[prominences < low_prominence] = False
        peaks[prominences > high_prominence] = False
    return peaks


def num_peaks(image, low_prominence=cpu_toolbox.TARGET_PROMINENCE,
              high_prominence=numpy.inf,
              use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the number of peaks from each line profile in an SLI image series
    by detecting all peaks and applying thresholds to remove unwanted peaks.

    Args:

        image: Full SLI measurement (series of images) which is prepared for
               the pipeline using the SLIX toolbox methods.

        low_prominence: Lower prominence bound for detecting a peak.

        high_prominence: Higher prominence bound for detecting a peak.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        Array where each entry corresponds to the number of detected peaks within
        the first dimension of the SLI image series.
    """
    if use_gpu:
        peaks = significant_peaks(image, low_prominence, high_prominence,
                                  return_numpy=False)
        return gpu_toolbox.num_peaks(peak_image=peaks,
                                     return_numpy=return_numpy)
    else:
        peaks = significant_peaks(image, low_prominence, high_prominence,
                                  use_gpu=False)
        return cpu_toolbox.num_peaks(peak_image=peaks)


def direction(peak_image, centroids, correction_angle=0,
              number_of_directions=3,
              use_gpu=gpu_available, return_numpy=True):
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

        peak_image: Boolean NumPy array specifying the peak positions in the
                    full SLI stack

        centroids: Centroids resulting from `centroid_correction` for more accurate
                   results

        number_of_directions: Number of directions which shall be generated.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        NumPy array with the shape (x, y, `number_of_directions`) containing
        up to `number_of_directions` direction angles. x equals the number
        of pixels of the SLI image series. If a direction angle is invalid
        or missing, the array entry will be BACKGROUND_COLOR instead.
    """
    if use_gpu:
        return gpu_toolbox.direction(peak_image, centroids, correction_angle,
                                     number_of_directions, return_numpy)
    else:
        return cpu_toolbox.direction(peak_image, centroids, correction_angle,
                                     number_of_directions)


def peak_distance(peak_image, centroids, use_gpu=gpu_available,
                  return_numpy=True):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the
                    full SLI stack

        centroids: Use centroid calculation to better determine the peak
        position regardless of the number of measurements / illumination
        angles used.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        NumPy array of floating point values containing the peak distance of
        the line profiles in degrees in their respective peak position. The
        first peak of each peak pair will show the distance between peak_1
        and peak_2 while the second peak will show 360 - (peak_2 - peak_1).
    """
    if use_gpu:
        return gpu_toolbox.peak_distance(peak_image, centroids, return_numpy)
    else:
        return cpu_toolbox.peak_distance(peak_image, centroids)


def mean_peak_distance(peak_image, centroids, use_gpu=gpu_available,
                       return_numpy=True):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks
    for each line profile in an SLI image series.

    Args:

        peak_image: Boolean NumPy array specifying the peak positions in the
                    full SLI stack

        centroids: Use centroid calculation to better determine the peak position
        regardless of the number of measurements / illumination angles used.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        NumPy array of floating point values containing the mean peak distance
        of the line profiles in degrees.
    """

    if use_gpu:
        return gpu_toolbox.mean_peak_distance(peak_image, centroids,
                                              return_numpy)
    else:
        return cpu_toolbox.mean_peak_distance(peak_image, centroids)


def peak_prominence(image, peak_image=None, kind_of_normalization=1,
                    use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the peak prominence of all given peak positions within a line
    profile. The line profile will be normalized by dividing the line profile
    through its mean value. Therefore, values above 1 are possible.

    Args:

        image: Original line profile used to detect all peaks. This array will be
        further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the
                    full SLI stack

        kind_of_normalization: Normalize given line profile by using a
        normalization technique based on the kind_of_normalization parameter.
           0 : Scale line profile to be between 0 and 1
           1 : Divide line profile through its mean value

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
                      NumPy array will be returned.

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees.
    """
    if use_gpu:
        return gpu_toolbox.peak_prominence(image, peak_image,
                                           kind_of_normalization, return_numpy)
    else:
        return cpu_toolbox.peak_prominence(image, peak_image,
                                           kind_of_normalization)


def mean_peak_prominence(image, peak_image=None, kind_of_normalization=1,
                         use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the mean peak prominence of all given peak positions within a
    line profile. The line profile will be normalized by dividing the line
    profile through its mean value. Therefore, values above 1 are possible.

    Args:

        image: Original line profile used to detect all peaks. This array
               will be further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the
                    full SLI stack

        kind_of_normalization: Normalize given line profile by using a
        normalization technique based on the kind_of_normalization parameter.
           0 : Scale line profile to be between 0 and 1
           1 : Divide line profile through its mean value

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
        NumPy array will be returned.

    Returns:

        Floating point value containing the mean peak prominence of the line
        profile in degrees.
    """
    if use_gpu:
        return gpu_toolbox.mean_peak_prominence(image, peak_image,
                                                kind_of_normalization,
                                                return_numpy)
    else:
        return cpu_toolbox.mean_peak_prominence(image, peak_image,
                                                kind_of_normalization)


def peak_width(image, peak_image=None, target_height=0.5,
               use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the peak width of all given peak positions within a line profile.

    Args:

        image: Original line profile used to detect all peaks. This array will
        be further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the full
        SLI stack

        target_height: Relative peak height in relation to the prominence of the
        given peak.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
        NumPy array will be returned.

    Returns:

        NumPy array where each entry corresponds to the peak width of the line
        profile. The values are in degree.
    """
    if use_gpu:
        return gpu_toolbox.peak_width(image, peak_image, target_height,
                                      return_numpy=return_numpy)
    else:
        return cpu_toolbox.peak_width(image, peak_image, target_height)


def mean_peak_width(image, peak_image=None, target_height=0.5,
                    use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the mean peak width of all given peak positions within a line
    profile.

    Args:

        image: Original line profile used to detect all peaks. This array will
        be further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the
        full SLI stack

        target_height: Relative peak height in relation to the prominence of
        the given peak.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
        NumPy array will be returned.

    Returns:

        NumPy array where each entry corresponds to the mean peak width of the
        line profile. The values are in degree.
    """
    if use_gpu:
        return gpu_toolbox.mean_peak_width(image, peak_image, target_height,
                                           return_numpy=return_numpy)
    else:
        return cpu_toolbox.mean_peak_width(image, peak_image, target_height)


def centroid_correction(image, peak_image,
                        low_prominence=cpu_toolbox.TARGET_PROMINENCE,
                        high_prominence=numpy.inf,
                        use_gpu=gpu_available, return_numpy=True):
    """
    Correct peak positions from a line profile by looking at only the peak
    with a given threshold using a centroid calculation. If a minimum is found
    in the considered interval, this minimum will be used as the limit instead.
    The range for the peak correction is limited by
    MAX_DISTANCE_FOR_CENTROID_ESTIMATION.

    Args:

        image: Original line profile used to detect all peaks. This array will
        be further analyzed to better determine the peak positions.

        peak_image: Boolean NumPy array specifying the peak positions in the
        full SLI stack

        low_prominence: Lower prominence bound for detecting a peak.

        high_prominence: Higher prominence bound for detecting a peak.

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
        NumPy array will be returned.

    Returns:

        _numpy array with the positions of all detected peak positions corrected
        with the centroid calculation.
    """
    if use_gpu:
        return gpu_toolbox.centroid_correction(image, peak_image,
                                               low_prominence,
                                               high_prominence, return_numpy)
    else:
        return cpu_toolbox.centroid_correction(image, peak_image,
                                               low_prominence,
                                               high_prominence)


def unit_vectors(direction, use_gpu=gpu_available, return_numpy=True):
    """
    Calculate the unit vectors (UnitX, UnitY) from a given direction angle.

    Args:

        direction: 3D NumPy array - direction angles in degrees

        use_gpu: If available use the GPU for calculation

        return_numpy: Necessary if using `use_gpu`. Specifies if a CuPy or
        NumPy array will be returned.

    Returns:

        UnitX, UnitY: 3D NumPy array, 3D NumPy array
            x- and y-vector component in arrays
    """
    if use_gpu:
        return gpu_toolbox.unit_vectors(direction, return_numpy=return_numpy)
    else:
        return cpu_toolbox.unit_vectors(direction)
