import multiprocessing

import nibabel
import numpy
import pymp
import tifffile
from scipy.signal import peak_widths, savgol_filter, find_peaks, peak_prominences

pymp.config.nested = True

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
CPU_COUNT = min(16, multiprocessing.cpu_count())
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08


def all_peaks(line_profile, cut_edges=True):
    """
    Detect all peaks from a given line profile in an SLI measurement. Peaks will not be filtered in any way.
    To detect only significant peaks, use the 'peak_positions' method and apply thresholds.

    Parameters
    ----------
    line_profile: 1D-NumPy array with all measurements of a single pixel.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    List with the positions of all detected peak positions.
    """
    number_of_measurements = line_profile.shape[0] // 2

    # Generate peaks
    maxima, _ = find_peaks(line_profile)

    # Only consider peaks which are in bounds
    if cut_edges:
        maxima = maxima[(maxima >= number_of_measurements // 2) & (maxima <= len(line_profile) -
                                                                   number_of_measurements // 2)]
        # Filter double peak
        if numpy.all(numpy.isin([number_of_measurements // 2,
                                 len(line_profile) - number_of_measurements // 2], maxima)):
            maxima = maxima[1:]

    return maxima


def num_peaks_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=numpy.inf, cut_edges=True):
    """
    Calculate the number of peaks from each line profile in an SLI image series by detecting all peaks and applying thresholds to
    remove unwanted peaks.

    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    NumPy array where each entry corresponds to the number of detected peaks within the first dimension of the SLI image series.
    """
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.int32)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            return_value[i] = len(accurate_peak_positions(peaks, roi, low_prominence, high_prominence, False))
    return return_value


def accurate_peak_positions(peak_positions, line_profile, low_prominence=TARGET_PROMINENCE, high_prominence=numpy.inf,
                            centroid_calculation=True):
    """
    Post-processing method after peaks have been calculated using the 'all_peaks' method. The peak are filtered based
    on their peak prominence. Additionally, peak positions can be corrected by applying centroid corrections based on the
    line profile.

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    centroid_calculation: Use centroid calculation to better determine the peak position regardless of the number of
    measurements / illumination angles used.

    Returns
    -------
    NumPy array with the positions of all detected peak positions.
    """
    n_roi = normalize(line_profile)
    peak_prominence = numpy.array(peak_prominences(n_roi, peak_positions)[0])
    selected_peaks = peak_positions[(peak_prominence > low_prominence) & (peak_prominence < high_prominence)]

    if centroid_calculation:
        return centroid_correction(n_roi, selected_peaks, low_prominence, high_prominence)

    return selected_peaks


def peakdistance(peak_positions, number_of_measurements):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks within a line profile.

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in one line profile.

    Returns
    -------
    Floating point value containing the mean peak distance of the line profile in degrees.
    """
    # Scale peaks correctly for direction
    peak_positions = (peak_positions - number_of_measurements // 2) * (360.0 / number_of_measurements)
    num_peaks = len(peak_positions)

    # Compute peak distance for curves with 1-2 detected peaks
    if num_peaks == 1:  # distance for one peak = 0
        return 0
    if num_peaks >= 2 and num_peaks % 2 == 0:
        distances = numpy.abs(peak_positions[::2] - peak_positions[1::2])
        dist = distances.mean()
        if dist > 180:
            dist = 360 - dist
        return dist
    else:
        return BACKGROUND_COLOR


def peakdistance_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True,
                       centroid_calculation=True):
    """
    Calculate the mean peak distance in degrees between two corresponding peaks for each line profile in an SLI image series.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.

    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.
    centroid_calculation: Use centroid calculation to better determine the peak position regardless of the number of
    measurements / illumination angles used.

    Returns
    -------
    NumPy array of floating point values containing the mean peak distance of the line profiles in degrees.
    """
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = accurate_peak_positions(peaks, roi, low_prominence, high_prominence, cut_edges,
                                            centroid_calculation)
            return_value[i] = peakdistance(peaks, len(peaks), len(roi) // 2)
    return return_value


def prominence(peak_positions, line_profile):
    """
    Calculate the mean peak prominence of all given peak positions within a line profile. The line profile will be
    normalized by dividing the line profile through its mean value. Therefore, values above 1 are possible.

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.

    Returns
    -------
    NumPy array where each entry corresponds to the mean peak prominence of the line profile.
    """
    num_peaks = len(peak_positions)
    prominence_roi = normalize(line_profile, kind_of_normalization=1)
    return 0 if num_peaks == 0 else numpy.mean(peak_prominences(prominence_roi, peak_positions)[0])


def prominence_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    """
    Calculate the mean peak prominence of all given peak positions for each line profile in an SLI image series. Each line
    profile will be normalized by dividing the line profile through its mean value. Therefore, values above 1 are
    possible.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.

    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    NumPy array where each entry corresponds to the mean peak prominence of the line profile.
    """
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = accurate_peak_positions(peaks, roi, low_prominence, high_prominence, cut_edges, False)
            return_value[i] = prominence(peaks, roi)
    return return_value


def peakwidth(peak_positions, line_profile, number_of_measurements):
    """

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in one line profile.

    Returns
    -------
    NumPy array where each entry corresponds to the mean peak width of the line profile.
    """
    num_peaks = len(peak_positions)
    if num_peaks > 0:
        widths = peak_widths(line_profile, peak_positions, rel_height=0.5)
        return numpy.mean(widths[0]) * (360.0 / number_of_measurements)
    else:
        return 0


def peakwidth_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    """
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.

    Parameters
    ----------
    roiset: Full SLI measurement (series of images) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    NumPy array where each entry corresponds to the mean peak width of the line profile.
    """
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = accurate_peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i] = peakwidth(peaks, roi, len(roi) // 2)
    return return_value


def crossing_direction(peak_positions, number_of_measurements):
    """
    Calculate up to three direction angles based on the given peak positions. If more than six peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the direction angle will be considered as invalid,
    resulting in a direction angle of BACKGROUND_COLOR.

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in the line profile.

    Returns
    -------
    NumPy array with the shape (3,) containing up to three direction angles. If a direction angle is invalid or missing,
    the array entry will be BACKGROUND_COLOR instead.
    """
    num_peaks = len(peak_positions)
    # Scale peaks correctly for direction
    peak_positions = (peak_positions - number_of_measurements // 2) * (360.0 / number_of_measurements)
    # Change behaviour based on amount of peaks (steep, crossing, ...)
    ret_val = numpy.full(3, BACKGROUND_COLOR, dtype=numpy.float)

    if num_peaks == 1:
        ret_val[0] = (270.0 - peak_positions[0]) % 180
    elif num_peaks % 2 == 0 and num_peaks <= 6:
        ret_val[:num_peaks // 2] = (270.0 - ((peak_positions[num_peaks // 2:] +
                                              peak_positions[:num_peaks // 2]) / 2.0)) % 180
        if num_peaks > 2:
            distances = peak_positions[num_peaks // 2:] - peak_positions[:num_peaks // 2]
            ret_val[:len(distances)][numpy.abs(distances - 180) > 35] = BACKGROUND_COLOR
    return ret_val


def crossing_direction_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    """
    Calculate up to three direction angles based on the given peak positions. If more than six peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pairs (i.e. 6 peaks: 1+4, 2+5, 3+6).
    If two peaks are too far away or too near (outside of 180°±35°), the direction angle will be considered as invalid,
    resulting in a direction angle of BACKGROUND_COLOR.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in 'SLIXParameterGenerator'.

    Parameters
    ----------
    roiset: Full SLI measurement (image series) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    NumPy array with the shape (x, 3) containing up to three direction angles. 
    x equals the number of pixels of the SLI image series. If a direction angle is invalid or missing, the array entry will be BACKGROUND_COLOR instead.

    """
    return_value = pymp.shared.array((roiset.shape[0], 3), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = accurate_peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i, :] = crossing_direction(peaks, len(roi) // 2)
    return return_value


def non_crossing_direction(peak_positions, number_of_measurements):
    """
    Calculate one direction angle based on the given peak positions. If more than two peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pair.

    Parameters
    ----------
    peak_positions: Detected peak positions of the 'all_peaks' method.
    number_of_measurements: Number of measurements during a full SLI measurement, i.e. the number of points in the line profile.

    Returns
    -------
    Floating point value containing the direction angle in degrees.
    If a direction angle is invalid or missing, the returned value will be BACKGROUND_COLOR instead.
    """
    num_peaks = len(peak_positions)
    # Scale peaks correctly for direction
    peak_positions = (peak_positions - number_of_measurements // 2) * (360.0 / number_of_measurements)
    # Change behaviour based on amount of peaks (steep, crossing, ...)
    if num_peaks == 1:
        return (270 - peak_positions[0]) % 180
    elif num_peaks == 2:
        return (270 - ((peak_positions[1] + peak_positions[0]) / 2.0)) % 180
    else:
        return BACKGROUND_COLOR


def non_crossing_direction_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    """
    Calculate one direction angle based on the given peak positions. If more than two peaks are present, no
    direction angle will be calculated to avoid errors. This will result in a direction angle of BACKGROUND_COLOR.
    The peak positions are determined by the position of the corresponding peak pair.
    Note: Please do not use this method when evaluating many line profiles while generating most if not all of the
    parameter maps. In this case, it is faster to write a simple pipeline as seen in SLIXParameterGenerator.

    Parameters
    ----------
    roiset: Full SLI measurement (image series) which is prepared for the pipeline using the SLIX toolbox methods.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.
    cut_edges: When True only consider peaks within the second third of all detected peaks.

    Returns
    -------
    NumPy array of floating point values containing the direction angle in degree.
    If a direction angle is invalid or missing, the returned value will be BACKGROUND_COLOR instead.
    """
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = accurate_peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i] = non_crossing_direction(peaks, len(roi) // 2)
    return return_value


def create_sampling(line_profile, peak_positions, left_bound, right_bound, target_peak_height,
                    number_of_samples=NUMBER_OF_SAMPLES):
    """
    Parameters
    ----------
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    peak_positions: Detected peak positions of the 'all_peaks' method.
    left_bound: Left bound for linear interpolation
    right_bound: Right bound for linear interpolation
    target_peak_height: Targeted peak height for centroid calculation
    number_of_samples: Number of samples used for linear interpolation

    Returns
    -------
    Linear interpolated array, new left bound, new right bound for centroid calculation.

    """
    sampling = numpy.interp(numpy.arange(left_bound - 1, right_bound + 1, 1 / 100),
                            numpy.arange(left_bound - 1, right_bound + 1), line_profile[left_bound - 1:right_bound + 1])
    if line_profile[left_bound] > target_peak_height:
        _left_bound = number_of_samples
    else:
        choices = numpy.argwhere(sampling[:(peak_positions - left_bound + 1) * number_of_samples] < target_peak_height)
        if len(choices) > 0:
            _left_bound = choices.max()
        else:
            _left_bound = number_of_samples
    if line_profile[right_bound] > target_peak_height:
        _right_bound = len(sampling) - number_of_samples
    else:
        choices = numpy.argwhere(sampling[(peak_positions - left_bound + 1) * number_of_samples:] < target_peak_height)
        if len(choices) > 0:
            _right_bound = (peak_positions - left_bound + 1) * number_of_samples + choices.min()
        else:
            _right_bound = len(sampling) - number_of_samples

    return sampling, _left_bound, _right_bound


def centroid_correction(line_profile, peak_positions, low_prominence=TARGET_PROMINENCE, high_prominence=None):
    """
    Correct peak positions from a line profile by looking at only the peak with a given threshold using a centroid
    calculation. If a minimum is found in the considered interval, this minimum will be used as the limit instead.
    The range for the peak correction is limited by MAX_DISTANCE_FOR_CENTROID_ESTIMATION.

    Parameters
    ----------
    line_profile: Original line profile used to detect all peaks. This array will be further
    analyzed to better determine the peak positions.
    peak_positions: Detected peak positions of the 'all_peaks' method.
    low_prominence: Lower prominence bound for detecting a peak.
    high_prominence: Higher prominence bound for detecting a peak.

    Returns
    -------
    NumPy array with the positions of all detected peak positions corrected with the centroid calculation.
    """
    reverse_roi = -1 * line_profile
    minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
    centroid_maxima = peak_positions.copy().astype('float32')

    for i in range(peak_positions.shape[0]):
        peak = peak_positions[i]
        target_peak_height = line_profile[peak_positions[i]] - line_profile[peak_positions].max() * \
                             (1 - TARGET_PEAK_HEIGHT)
        minima_distances = peak - minima

        left_position = right_position = peak

        # Check for minima in left and set left position accordingly
        target_distances = (minima_distances <= MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances > 0)
        if target_distances.any():
            left_position = peak - minima_distances[target_distances].min()
        # Look for peak height
        below_target_peak_height = numpy.argwhere(
            line_profile[peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION: peak] < target_peak_height)
        if len(below_target_peak_height) > 0:
            below_target_peak_height = below_target_peak_height.max()
            temp_left_position = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION + below_target_peak_height
            if temp_left_position < left_position:
                left_position = temp_left_position
        else:
            temp_left_position = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION
            if temp_left_position < left_position:
                left_position = temp_left_position

        # Repeat for right bound
        target_distances = (minima_distances >= -MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances < 0)
        if target_distances.any():
            right_position = peak - minima_distances[target_distances].min()
        # Look for 80% of the peak height
        below_target_peak_height = numpy.argwhere(
            line_profile[peak: peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION] < target_peak_height)
        if len(below_target_peak_height) > 0:
            below_target_peak_height = below_target_peak_height.min()
            temp_right_position = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION - below_target_peak_height
            if temp_right_position > right_position:
                right_position = temp_right_position
        else:
            temp_right_position = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION
            if temp_right_position > right_position:
                right_position = temp_right_position

        sampling, left_bound, right_bound = create_sampling(line_profile, peak, left_position, right_position,
                                                            target_peak_height)
        integer_left_pos = (left_position - 1) + 1 / NUMBER_OF_SAMPLES * left_bound
        integer_right_pos = (left_position - 1) + 1 / NUMBER_OF_SAMPLES * right_bound
        # Move at max one step size on the x-coordinate axis to the left or right to prevent too much movement
        centroid = numpy.sum(numpy.arange(integer_left_pos, integer_right_pos - 1e-10, 0.01) *
                             sampling[left_bound:right_bound]) / numpy.sum(sampling[left_bound:right_bound])
        if numpy.abs(centroid - peak) > 1:
            centroid = peak + numpy.sign(centroid - peak)
        centroid_maxima[i] = centroid

    return centroid_maxima


def read_image(FILEPATH):
    """
    Reads image file and returns it.
    Supported file formats: NIfTI, Tiff.

    Arguments:
        FILEPATH: Path to image

    Returns:
        numpy.array: Image with shape [x, y, z] where [x, y] is the size of a single image and z specifies the number
                     of measurements
    """
    # Load NIfTI dataset
    if FILEPATH.endswith('.nii'):
        data = nibabel.load(FILEPATH).get_fdata()
        data = numpy.squeeze(numpy.swapaxes(data, 0, 1))
    else:
        data = tifffile.imread(FILEPATH)
        data = numpy.squeeze(numpy.moveaxis(data, 0, -1))

    return data


def create_background_mask(IMAGE, threshold=10):
    """
    Creates a background mask based on given threshold. As all background pixels are near zero when looking through
    the z-axis plot, this method should remove most of the background allowing for better approximations using the available
    features. It is advised to use this function.

    Arguments:
        IMAGE: 2D/3D-image containing the z-axis in the last dimension

    Keyword Arguments:
        threshold: Threshhold for mask creation (default: {10})

    Returns:
        numpy.array: 1D/2D-image which masks the background as True and foreground as False
    """
    mask = numpy.max(IMAGE < threshold, axis=-1)
    return mask


def create_roiset(IMAGE, ROISIZE, extend=True):
    """
    Create roi set of given image by creating an image containing the average value of pixels within the
    specified ROISIZE. The returned image will have twice the size in the third axis as the both halfs will be doubled
    for the peak detection.

    Arguments:
        IMAGE: Image containing multiple images in a 3D-stack
        ROISIZE: Size in pixels which are used to create the region of interest image

    Returns:
        numpy.array: Image with shape [x/ROISIZE, y/ROISIZE, 2*'number of measurements'] containing the average value
        of the given roi for each image in z-axis.
    """
    # Get image dimensions
    x = IMAGE.shape[0]
    y = IMAGE.shape[1]
    number_of_measurements = IMAGE.shape[2]
    nx = numpy.ceil(x / ROISIZE).astype('int')
    ny = numpy.ceil(y / ROISIZE).astype('int')

    if extend:
        roi_set = pymp.shared.array((nx * ny, 2 * number_of_measurements), dtype='float32')
    else:
        roi_set = pymp.shared.array((nx * ny, number_of_measurements), dtype='float32')

    # ROISIZE == 1 is exactly the same as the original image
    if ROISIZE > 1:
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    # Create average of selected ROI and append two halfs to the front and back
                    roi = IMAGE[ROISIZE * i:ROISIZE * i + ROISIZE, ROISIZE * j:ROISIZE * j + ROISIZE, :]
                    average_per_dimension = numpy.average(numpy.average(roi, axis=1), axis=0).flatten()
                    if extend:
                        average_per_dimension = numpy.concatenate(
                            (average_per_dimension[-number_of_measurements // 2:], average_per_dimension,
                             average_per_dimension[:number_of_measurements // 2]))
                    roi_set[i * ny + j] = average_per_dimension
    else:
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    roi = IMAGE[i, j, :]
                    if extend:
                        roi = numpy.concatenate((roi[-number_of_measurements // 2:], roi,
                                                 roi[:number_of_measurements // 2]))
                    roi_set[i * ny + j] = roi

    return roi_set


def smooth_roiset(roiset, range=45, polynom_order=2):
    """
    Applies Savitzky-Golay filter to given roiset and returns the smoothened measurement.

    Args:
        roiset: Flattened image with the dimensions [x*y, z] where z equals the number of measurements
        range: Used window length for the Savitzky-Golay filter
        polynom_order: Used polynomial order for the Savitzky-Golay filter

    Returns: Line profiles with applied Savitzky-Golay filter and the same shape as the original roi set.

    """
    roiset_rolled = pymp.shared.array(roiset.shape, dtype='float32')
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(len(roiset)):
            roi = roiset[i]
            # Extension of the range to include circularity.
            roi_c = numpy.concatenate((roi, roi, roi))
            roi_rolled = savgol_filter(roi_c, range, polynom_order)
            # Shrink array back down to it's original size
            roi_rolled = roi_rolled[len(roi):-len(roi)]
            roiset_rolled[i] = roi_rolled
    return roiset_rolled


def normalize(roi, kind_of_normalization=0):
    """
    Normalize given line profile by using a normalization technique based on the kind_of_normalization parameter.

    0 : Scale line profile to be between 0 and 1
    1 : Divide line profile through its mean value

    Arguments:
        roi: Line profile of a single pixel / region of interest
        kind_of_normalization: Normalization technique which will be used for the calculation

    Returns:
        numpy.array -- Normalized line profile of the given roi parameter
    """
    roi = roi.copy().astype('float32')
    if not numpy.all(roi == 0):
        if roi.max() == roi.min():
            normalized_roi = numpy.ones(roi.shape)
        else:
            if kind_of_normalization == 0:
                normalized_roi = (roi - roi.min()) / (roi.max() - roi.min())
            elif kind_of_normalization == 1:
                normalized_roi = roi / numpy.mean(roi)
        return normalized_roi
    return roi


def reshape_array_to_image(image, x, ROISIZE):
    """
    Convert array back to image keeping the lower resolution based on the ROISIZE.

    Arguments:
        image: Array created by other methods with lower resolution based on ROISIZE
        x: Size of original image in x-dimension
        ROISIZE: Size of the ROI used for evaluating the roiset

    Returns:
        numpy.array -- Reshaped image based on the input array
    """
    image_reshaped = image.reshape(
        (numpy.ceil(x / ROISIZE).astype('int'), image.shape[0] // numpy.ceil(x / ROISIZE).astype('int')))
    return image_reshaped
