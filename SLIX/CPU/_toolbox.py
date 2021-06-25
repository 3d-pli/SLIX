from numba import jit, prange
import numpy

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 2

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.06


@jit(nopython=True, parallel=True)
def _peaks(image):
    peaks = image.copy()
    resulting_peaks = numpy.zeros(peaks.shape, dtype=numpy.uint8)
    for idx in prange(peaks.shape[0]):
        sub_image = image[idx]

        pos = 0
        for loop_pos in range(0, len(sub_image)):
            if loop_pos < pos:
                continue
            if sub_image[pos] - sub_image[pos - 1] <= 0:
                pos = pos + 1
                continue

            pos_ahead = pos + 1
            while pos_ahead < 2 * len(sub_image) and \
                    sub_image[pos_ahead % len(sub_image)] == \
                    sub_image[pos]:
                pos_ahead = pos_ahead + 1

            if sub_image[pos] - sub_image[pos_ahead % len(sub_image)] > 0:
                left = pos
                right = pos_ahead - 1
                resulting_peaks[idx, (left + right) // 2] = 1

            pos = pos_ahead
    return resulting_peaks


@jit(nopython=True, parallel=True)
def _prominence(image, peak_image):
    result_image = numpy.empty(image.shape, dtype=numpy.float32)
    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peak_array = peak_image[idx]
        for pos in prange(len(sub_peak_array)):
            if sub_peak_array[pos] == 1:
                i_min = -len(sub_peak_array) / 2
                i_max = int(len(sub_peak_array) * 1.5)

                i = pos
                left_min = sub_image[pos]
                wlen = len(sub_peak_array) - 1
                while i_min <= i and sub_image[i] <= sub_image[pos] and \
                        wlen > 0:
                    if sub_image[i] < left_min:
                        left_min = sub_image[i]
                    i = i - 1
                    wlen = wlen - 1

                i = pos
                right_min = sub_image[pos]
                wlen = len(sub_peak_array) - 1
                while i <= i_max and \
                        sub_image[i % len(sub_peak_array)] <= sub_image[pos] \
                        and wlen > 0:
                    if sub_image[i % len(sub_peak_array)] < right_min:
                        right_min = sub_image[i % len(sub_peak_array)]
                    i = i + 1
                    wlen = wlen - 1

                result_image[idx, pos] = sub_image[pos] - \
                                         max(left_min, right_min)
            else:
                result_image[idx, pos] = 0
    return result_image


@jit(nopython=True, parallel=True)
def _peakwidth(image, peak_image, prominence, target_height):
    result_image = numpy.zeros(image.shape).astype(numpy.float32)
    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peak_array = peak_image[idx]
        sub_prominece = prominence[idx]

        for pos in prange(len(sub_peak_array)):
            if sub_peak_array[pos] == 1:
                height = sub_image[pos] - sub_prominece[pos] * target_height
                i_min = -len(sub_peak_array) // 2
                i_max = 1.5 * len(sub_peak_array)

                i = int(pos)
                while i_min < i and \
                        sub_image[i % len(sub_peak_array)] - height > 1e-7:
                    i -= 1
                left_ip = numpy.float32(i)
                if sub_image[i % len(sub_peak_array)] < height:
                    left_ip += (height - sub_image[i % len(sub_peak_array)]) \
                               / (sub_image[(i + 1) % len(sub_peak_array)] -
                                  sub_image[i % len(sub_peak_array)])

                # Find intersection point on right side
                i = int(pos)
                while i < i_max and \
                        sub_image[i % len(sub_peak_array)] - height > 1e-7:
                    i += 1
                right_ip = numpy.float32(i)
                if sub_image[i % len(sub_peak_array)] < height:
                    right_ip -= (height - sub_image[i % len(sub_peak_array)]) \
                                / (sub_image[(i - 1) % len(sub_peak_array)] -
                                   sub_image[i % len(sub_peak_array)])

                result_image[idx, pos] = right_ip - left_ip
    return result_image


@jit(nopython=True, parallel=True)
def _peakdistance(peak_image, centroids, number_of_peaks):
    result_image = numpy.zeros(peak_image.shape, dtype=numpy.float32)
    for idx in prange(peak_image.shape[0]):
        sub_peak_array = peak_image[idx]
        sub_centroid_array = centroids[idx]
        current_pair = 0

        for i in prange(len(sub_peak_array)):
            if sub_peak_array[i] == 1:
                if number_of_peaks[idx] == 1:
                    result_image[idx, i] = 360.0
                    break
                elif number_of_peaks[idx] % 2 == 0:
                    left = (i + sub_centroid_array[i]) * 360.0 / \
                           len(sub_peak_array)
                    right_side_peak = number_of_peaks[idx]//2
                    current_position = i
                    while right_side_peak > 0 and \
                            current_position < len(sub_peak_array):
                        current_position = current_position + 1
                        if sub_peak_array[current_position] == 1:
                            right_side_peak = right_side_peak - 1
                    if right_side_peak > 0:
                        result_image[idx, i] = 0
                    else:
                        right = (current_position +
                                 sub_centroid_array[current_position]) * \
                                360.0 / len(sub_peak_array)
                        result_image[idx, i] = right - left
                        result_image[idx, current_position] = 360 - \
                                                              (right - left)

                    current_pair += 1

                if current_pair == number_of_peaks[idx]//2:
                    break
    return result_image


@jit(nopython=True, parallel=True)
def _direction(peak_array, centroids, number_of_peaks, num_directions, correctdir):
    result_image = numpy.empty((peak_array.shape[0], num_directions),
                               dtype=numpy.float32)
    for idx in prange(peak_array.shape[0]):
        sub_peak_array = peak_array[idx]
        sub_centroid_array = centroids[idx]
        current_direction = 0

        # Set the whole pixel in the direction to background.
        # That just in case if not all directions are calculated
        # when only two or four peaks are present.
        result_image[idx, :] = BACKGROUND_COLOR
        # Check if the line profile has less peaks than the number
        # of directions which shall be calculated.
        if number_of_peaks[idx] // 2 <= num_directions:
            for i in prange(len(sub_peak_array)):
                # If one of our line profile pixels is a peak
                if sub_peak_array[i] == 1:
                    # Mark the position as the left position of our peak
                    left = (i + sub_centroid_array[i]) * \
                           360.0 / len(sub_peak_array) + \
                           numpy.float32(correctdir)
                    # If there is only one peak present, convert the left
                    # position to our direction
                    if number_of_peaks[idx] == 1:
                        result_image[idx, current_direction] = \
                            (270.0 - left) % 180
                        break
                    # If we got an even number of peaks, we can calculate
                    # each direction without any problems.
                    elif number_of_peaks[idx] % 2 == 0:
                        # We search for a peak which is around 180° away.
                        # We will search for it using the following distance
                        # as the number of peaks we need to pass.
                        right_side_peak = number_of_peaks[idx] // 2
                        current_position = i
                        # Check for peaks until we find the corresponding peak
                        while right_side_peak > 0 and \
                                current_position < len(sub_peak_array):
                            current_position = current_position + 1
                            if sub_peak_array[current_position] == 1:
                                right_side_peak = right_side_peak - 1

                        if right_side_peak == 0:
                            right = (current_position +
                                     sub_centroid_array[current_position]) * \
                                     360.0 / len(sub_peak_array) + \
                                     numpy.float32(correctdir)
                            # If our peaks are around 180° ± 35° apart,
                            # we can calculate the direction.
                            if number_of_peaks[idx] > 2 and \
                                    abs(180 - (right - left)) >= 35:
                                result_image[idx] = BACKGROUND_COLOR
                                break

                            result_image[idx, current_direction] = \
                                (270.0 - ((left + right) / 2.0)) % 180
                            current_direction += 1

                    if current_direction == number_of_peaks[idx]//2:
                        break
    return result_image


@jit(nopython=True, parallel=True)
def _centroid_correction_bases(image, peak_image, reverse_peaks):
    left_bases = numpy.empty(image.shape, dtype=numpy.int8)
    right_bases = numpy.empty(image.shape, dtype=numpy.int8)

    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peaks = peak_image[idx]
        sub_reverse_peaks = reverse_peaks[idx]

        max_pos = 0
        for pos in range(len(sub_image)):
            if sub_image[pos] > max_pos:
                max_pos = sub_image[pos]

        for pos in range(len(sub_peaks)):
            if sub_peaks[pos] == 1:

                target_peak_height = max(0, sub_image[pos] - max_pos *
                                         TARGET_PEAK_HEIGHT)
                left_position = MAX_DISTANCE_FOR_CENTROID_ESTIMATION
                right_position = MAX_DISTANCE_FOR_CENTROID_ESTIMATION

                # Check for minima in range
                for offset in range(1, MAX_DISTANCE_FOR_CENTROID_ESTIMATION):
                    if sub_reverse_peaks[pos - offset] == 1:
                        left_position = offset
                        break
                    if sub_reverse_peaks[(pos + offset) %
                                         len(sub_reverse_peaks)] == 1:
                        right_position = offset
                        break

                # Check for peak height
                for offset in range(left_position):
                    if sub_image[pos - offset] < target_peak_height:
                        left_position = offset
                        break
                for offset in range(right_position):
                    if sub_image[(pos + offset) % len(sub_image)] < \
                            target_peak_height:
                        right_position = offset
                        break

                left_bases[idx, pos] = left_position
                right_bases[idx, pos] = right_position
            else:
                left_bases[idx, pos] = 0
                right_bases[idx, pos] = 0
    return left_bases, right_bases


@jit(nopython=True, parallel=True)
def _centroid(image, peak_image, left_bases, right_bases):
    centroid_peaks = numpy.zeros(image.shape, dtype=numpy.float32)

    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peaks = peak_image[idx]
        sub_left_bases = left_bases[idx]
        sub_right_bases = right_bases[idx]

        max_pos = 0
        for pos in range(len(sub_image)):
            if sub_image[pos] > max_pos:
                max_pos = sub_image[pos]

        for pos in range(len(sub_peaks)):
            if sub_peaks[pos] == 1:
                centroid_sum_top = 0.0
                centroid_sum_bottom = 1e-15
                target_peak_height = max(0, sub_image[pos] - max_pos *
                                         TARGET_PEAK_HEIGHT)

                for x in range(-sub_left_bases[pos], sub_right_bases[pos]):
                    img_pixel = sub_image[(pos + x) % len(sub_image)]
                    next_img_pixel = sub_image[(pos + x + 1) % len(sub_image)]
                    for interp in range(NUMBER_OF_SAMPLES):
                        step = interp / NUMBER_OF_SAMPLES
                        func_val = img_pixel + \
                                   (next_img_pixel - img_pixel) * step

                        if func_val >= target_peak_height:
                            centroid_sum_top += (x + step) * func_val
                            centroid_sum_bottom += func_val
                centroid = centroid_sum_top / centroid_sum_bottom
                if abs(centroid) > 1:
                    centroid = numpy.sign(centroid)
                centroid_peaks[idx, pos] = centroid
    return centroid_peaks
