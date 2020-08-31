import multiprocessing

import nibabel
import numpy
import pymp
import tifffile
from read_roi import read_roi_zip
from scipy.signal import peak_widths, savgol_filter, find_peaks, peak_prominences

pymp.config.nested = True

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
CPU_COUNT = min(16, multiprocessing.cpu_count())
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08


def all_peaks(roi, cut_edges=True):
    number_of_measurements = roi.shape[0] // 2

    roi = normalize(roi)
    # Generate peaks
    maxima, _ = find_peaks(roi)

    # Only consider peaks which are in bounds
    if cut_edges:
        maxima = maxima[(maxima >= number_of_measurements // 2) & (maxima <= len(roi) - number_of_measurements // 2)]
        # Filter double peak
        if numpy.all(numpy.isin([number_of_measurements // 2, len(roi) - number_of_measurements // 2], maxima)):
            maxima = maxima[1:]

    return maxima


def num_peaks_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=numpy.inf, cut_edges=True):
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.int32)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            return_value[i] = len(peak_positions(peaks, roi, low_prominence, high_prominence, False))
    return return_value


def peak_positions(peaks, roi, low_prominence=TARGET_PROMINENCE, high_prominence=numpy.inf, centroid_calculation=True):
    n_roi = normalize(roi)
    peak_prominence = numpy.array(peak_prominences(n_roi, peaks)[0])
    selected_peaks = peaks[(peak_prominence > low_prominence) & (peak_prominence < high_prominence)]

    if centroid_calculation:
        return centroid_correction(n_roi, selected_peaks, low_prominence, high_prominence)

    return selected_peaks


def peakdistance(peaks, num_peaks, number_of_images):
    # Scale peaks correctly for direction
    peaks = (peaks - number_of_images // 2) * (360.0 / number_of_images)

    # Compute peak distance for curves with 1-2 detected peaks
    if num_peaks == 1:  # distance for one peak = 0
        return 0
    if num_peaks >= 2 and num_peaks % 2 == 0:
        distances = numpy.abs(peaks[::2] - peaks[1::2])
        dist = distances.mean()
        if dist > 180:
            dist = 360 - dist
        return dist
    else:
        return BACKGROUND_COLOR


def peakdistance_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True,
                       centroid_calculation=True):
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = peak_positions(peaks, roi, low_prominence, high_prominence, cut_edges, centroid_calculation)
            return_value[i] = peakdistance(peaks, len(peaks), len(roi) // 2)
    return return_value


def prominence(peaks, line_profile, num_peaks):
    prominence_roi = normalize(line_profile, kind_of_normalizaion=1)
    return 0 if num_peaks == 0 else numpy.mean(peak_prominences(prominence_roi, peaks)[0])


def prominence_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = peak_positions(peaks, roi, low_prominence, high_prominence, cut_edges, False)
            return_value[i] = prominence(peaks, roi, len(peaks))
    return return_value


def peakwidth(peaks, line_profile, num_peaks, number_of_images):
    if num_peaks > 0:
        widths = peak_widths(line_profile, peaks, rel_height=0.5)
        return numpy.mean(widths[0]) * (360.0 / number_of_images)
    else:
        return 0


def peakwidth_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i] = peakwidth(peaks, roi, len(peaks), len(roi) // 2)
    return return_value


def crossing_direction(peaks, num_peaks, number_of_images):
    # Scale peaks correctly for direction
    peaks = (peaks - number_of_images // 2) * (360.0 / number_of_images)
    # Change behaviour based on amount of peaks (steep, crossing, ...)
    ret_val = numpy.full(3, BACKGROUND_COLOR, dtype=numpy.float)

    if num_peaks == 1:
        ret_val[0] = (270.0 - peaks[0]) % 180
    elif num_peaks % 2 == 0 and num_peaks <= 6:
        ret_val[:num_peaks // 2] = (270.0 - ((peaks[num_peaks // 2:] + peaks[:num_peaks // 2]) / 2.0)) % 180
        if num_peaks > 2:
            distances = peaks[num_peaks // 2:] - peaks[:num_peaks // 2]
            ret_val[:len(distances)][numpy.abs(distances - 180) > 35] = BACKGROUND_COLOR
    return ret_val


def crossing_direction_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    return_value = pymp.shared.array((roiset.shape[0], 3), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i, :] = crossing_direction(peaks, len(peaks), len(roi) // 2)
    return return_value


def non_crossing_direction(peaks, num_peaks, number_of_images):
    # Scale peaks correctly for direction
    peaks = (peaks - number_of_images // 2) * (360.0 / number_of_images)
    # Change behaviour based on amount of peaks (steep, crossing, ...)
    if num_peaks == 1:
        return (270 - peaks[0]) % 180
    elif num_peaks == 2:
        return (270 - ((peaks[1] + peaks[0]) / 2.0)) % 180
    else:
        return BACKGROUND_COLOR


def non_crossing_direction_image(roiset, low_prominence=TARGET_PROMINENCE, high_prominence=None, cut_edges=True):
    return_value = pymp.shared.array((roiset.shape[0], 1), dtype=numpy.float)
    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(0, len(roiset)):
            roi = roiset[i]
            peaks = all_peaks(roi, cut_edges)
            peaks = peak_positions(peaks, roi, low_prominence, high_prominence, False)
            return_value[i] = non_crossing_direction(peaks, len(peaks), len(roi) // 2)
    return return_value


# Create sampling to get exact 80% of peak height
def create_sampling(roi, peak, left, right, target_peak_height, number_of_samples=NUMBER_OF_SAMPLES):
    sampling = numpy.interp(numpy.arange(left - 1, right + 1, 1 / 100),
                            numpy.arange(left - 1, right + 1), roi[left - 1:right + 1])
    if roi[left] > target_peak_height:
        left_bound = number_of_samples
    else:
        choices = numpy.argwhere(sampling[:(peak - left + 1) * number_of_samples] < target_peak_height)
        if len(choices) > 0:
            left_bound = choices.max()
        else:
            left_bound = number_of_samples
    if roi[right] > target_peak_height:
        right_bound = len(sampling) - number_of_samples
    else:
        choices = numpy.argwhere(sampling[(peak - left + 1) * number_of_samples:] < target_peak_height)
        if len(choices) > 0:
            right_bound = (peak - left + 1) * number_of_samples + choices.min()
        else:
            right_bound = len(sampling) - number_of_samples

    return sampling, left_bound, right_bound


def centroid_correction(roi, high_peaks, low_prominence=TARGET_PROMINENCE, high_prominence=None):
    reverse_roi = -1 * roi
    minima, _ = find_peaks(reverse_roi, prominence=(low_prominence, high_prominence))
    centroid_maxima = high_peaks.copy().astype('float32')

    for i in range(high_peaks.shape[0]):
        peak = high_peaks[i]
        target_peak_height = roi[high_peaks[i]] - roi[high_peaks].max() * (1 - TARGET_PEAK_HEIGHT)
        minima_distances = peak - minima

        lpos = rpos = peak

        # Check for minima in left and set left position accordingly
        target_distances = (minima_distances <= MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances > 0)
        if target_distances.any():
            lpos = peak - minima_distances[target_distances].min()
        # Look for peak height
        below_target_peak_height = numpy.argwhere(
            roi[peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION: peak] < target_peak_height)
        if len(below_target_peak_height) > 0:
            below_target_peak_height = below_target_peak_height.max()
            tlpos = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION + below_target_peak_height
            if tlpos < lpos:
                lpos = tlpos
        else:
            tlpos = peak - MAX_DISTANCE_FOR_CENTROID_ESTIMATION
            if tlpos < lpos:
                lpos = tlpos

        # Repeat for right bound
        target_distances = (minima_distances >= -MAX_DISTANCE_FOR_CENTROID_ESTIMATION) & (minima_distances < 0)
        if target_distances.any():
            rpos = peak - minima_distances[target_distances].min()
        # Look for 80% of the peak height
        below_target_peak_height = numpy.argwhere(
            roi[peak: peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION] < target_peak_height)
        if len(below_target_peak_height) > 0:
            below_target_peak_height = below_target_peak_height.min()
            trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION - below_target_peak_height
            if trpos > rpos:
                rpos = trpos
        else:
            trpos = peak + MAX_DISTANCE_FOR_CENTROID_ESTIMATION
            if trpos > rpos:
                rpos = trpos

        sampling, lbound, rbound = create_sampling(roi, peak, lpos, rpos, target_peak_height)
        int_lpos = (lpos - 1) + 1 / NUMBER_OF_SAMPLES * lbound
        int_rpos = (lpos - 1) + 1 / NUMBER_OF_SAMPLES * rbound
        # Move at max one entry on the x-coordinate axis to the left or right to prevent too much movement
        centroid = numpy.sum(numpy.arange(int_lpos, int_rpos - 1e-10, 0.01) *
                             sampling[lbound:rbound]) / numpy.sum(sampling[lbound:rbound])
        if numpy.abs(centroid - peak) > 1:
            centroid = peak + numpy.sign(centroid - peak)
        centroid_maxima[i] = centroid

    return centroid_maxima


def read_image(FILEPATH):
    """
    Reads iamge file and returns it.

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
    the z-axis plot this method should remove most of the background allowing better approximations using the available
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


def zaxis_from_imagej_roiset(IMAGE, PATH_TO_ROISET, extend=True):
    rois = read_roi_zip(PATH_TO_ROISET)
    number_of_measurements = IMAGE.shape[2]

    if extend:
        roi_set = pymp.shared.array((len(rois.items()), 2 * number_of_measurements), dtype='float32')
    else:
        roi_set = pymp.shared.array((len(rois.items()), number_of_measurements), dtype='float32')

    roi_values = list(dict(rois.items()).values())
    name_set = list(dict(rois.items()).keys())

    with pymp.Parallel(CPU_COUNT) as p:
        for i in p.range(1, len(rois)):
            value = roi_values[i]
            name = value['name']
            width, height = value['width'], value['height']
            roi_type = value['type']
            left, top = value['left'], value['top']
            center = (left + width / 2, top + height / 2)

            if width == height and roi_type == 'oval':
                x_indices = numpy.arange(top, top + height + 1)
                y_indices = numpy.arange(left, left + width + 1)
                rectangle_indices = numpy.array(numpy.meshgrid(x_indices, y_indices)).T.reshape(-1, 2)
                rectangle_indices = rectangle_indices[(rectangle_indices[:, 0] - center[1]) ** 2 + (
                        rectangle_indices[:, 1] - center[0]) ** 2 < width * height]

                roi = IMAGE[rectangle_indices[:, 0], rectangle_indices[:, 1], :]
                average_per_dimension = numpy.average(roi, axis=0).flatten()
                if extend:
                    average_per_dimension = numpy.concatenate(
                        (average_per_dimension[-number_of_measurements // 2:], average_per_dimension, average_per_dimension[:number_of_measurements // 2]))
                name_set[i] = name
                roi_set[i] = average_per_dimension
            else:
                continue

    return roi_set[1:], name_set[1:]


def zaxis_roiset(IMAGE, ROISIZE, extend=True):
    """
    Create z-axis profile of given image by creating a roiset image containing the average value of pixels within the
    specified ROISIZE. The returned image will have twice the size in the third axis as the both half will be doubled for
    the peak detection.

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
                            (average_per_dimension[-number_of_measurements // 2:], average_per_dimension, average_per_dimension[:number_of_measurements // 2]))
                    roi_set[i * ny + j] = average_per_dimension
    else:
        with pymp.Parallel(CPU_COUNT) as p:
            for i in p.range(0, nx):
                for j in range(0, ny):
                    roi = IMAGE[i, j, :]
                    if extend:
                        roi = numpy.concatenate((roi[-number_of_measurements // 2:], roi, roi[:number_of_measurements // 2]))
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
    Normalize given line profile by using a normalization technique based on the kind_of_normalization parameter

    0 : Scale line profile to be between 0 and 1
    1 : Divide line profile through it's mean value

    Arguments:
        roi: Line profile of a singular pixel / region of interest
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
    else:
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
