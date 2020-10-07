from numba import jit, prange
import numpy

# DEFAULT PARAMETERS
BACKGROUND_COLOR = -1
MAX_DISTANCE_FOR_CENTROID_ESTIMATION = 3

NUMBER_OF_SAMPLES = 100
TARGET_PEAK_HEIGHT = 0.94
TARGET_PROMINENCE = 0.08


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
                while i_min <= i and sub_image[i] <= sub_image[pos] and wlen > 0:
                    if sub_image[i] < left_min:
                        left_min = sub_image[i]
                    i = i - 1
                    wlen = wlen - 1

                i = pos
                right_min = sub_image[pos]
                wlen = len(sub_peak_array) - 1
                while i <= i_max and sub_image[i % len(sub_peak_array)] <= sub_image[pos] and wlen > 0:
                    if sub_image[i % len(sub_peak_array)] < right_min:
                        right_min = sub_image[i % len(sub_peak_array)]
                    i = i + 1
                    wlen = wlen - 1

                result_image[idx, pos] = sub_image[pos] - max(left_min, right_min)
            else:
                result_image[idx, pos] = 0
    return result_image


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def _peakwidth(image, peak_image, prominence, target_height):
    result_image = numpy.empty(image.shape).astype(numpy.float32)
    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peak_array = peak_image[idx]
        sub_prominece = prominence[idx]

        for pos in prange(len(sub_peak_array)):
            if sub_peak_array[pos] == 1:
                height = sub_image[pos] - sub_prominece[pos] * target_height
                i_min = -len(sub_peak_array) / 2
                i_max = int(len(sub_peak_array) * 1.5)

                i = int(pos)
                while i_min < i and height < sub_image[i]:
                    i -= 1
                left_ip = float(i)
                if sub_image[i] < height:
                    # Interpolate if true intersection height is between samples
                    left_ip += (height - sub_image[i]) / (sub_image[i + 1] - sub_image[i])

                # Find intersection point on right side
                i = int(pos)
                while i < i_max and height < sub_image[i]:
                    i += 1
                right_ip = float(i)
                if sub_image[i] < height:
                    # Interpolate if true intersection height is between samples
                    right_ip -= (height - sub_image[i]) / (sub_image[i - 1] - sub_image[i])

                result_image[idx, pos] = right_ip - left_ip
            else:
                result_image[idx, pos] = 0
    return result_image


@jit(nopython=True, parallel=True)
def _peakdistance(peak_image, centroids, number_of_peaks):
    result_image = numpy.empty(peak_image.shape).astype(numpy.float32)
    for idx in prange(peak_image.shape[0]):
        sub_peak_array = peak_image[idx]
        current_pair = 0

        for i in prange(len(sub_peak_array)):
            if sub_peak_array[i] == 1:
                if number_of_peaks[idx] == 1:
                    result_image[idx, i] = 360.0
                    break
                elif number_of_peaks[idx] % 2 == 0:
                    left = i * 360.0 / len(sub_peak_array)
                    right_side_peak = number_of_peaks[idx]//2
                    current_position = i+1
                    while right_side_peak > 0 and current_position < len(sub_peak_array):
                        if sub_peak_array[current_position] == 1:
                            right_side_peak = right_side_peak - 1
                        current_position = current_position + 1
                    right = (current_position-1) * 360.0 / len(sub_peak_array)
                    result_image[idx, i] = right - left
                    result_image[idx, current_position-1] = 360 - (right - left)

                    current_pair += 1

                if current_pair == number_of_peaks[idx]//2:
                    break
    return result_image


@jit(nopython=True, parallel=True)
def _direction(peak_array, centroids, number_of_peaks, num_directions):
    result_image = numpy.empty((peak_array.shape[0], num_directions)).astype(numpy.float32)
    for idx in prange(peak_array.shape[0]):
        sub_peak_array = peak_array[idx]
        current_direction = 0

        result_image[idx, :] = BACKGROUND_COLOR
        if number_of_peaks[idx] // 2 <= num_directions:
            for i in prange(len(sub_peak_array)):
                if sub_peak_array[i] == 1:
                    left = i * 360.0 / len(sub_peak_array)
                    if number_of_peaks[idx] == 1:
                        result_image[idx, current_direction] = (270.0 - left) % 180
                        break
                    elif number_of_peaks[idx] % 2 == 0:
                        right_side_peak = number_of_peaks[idx]//2
                        current_position = i+1
                        while right_side_peak > 0 and current_position < len(sub_peak_array):
                            if sub_peak_array[current_position] == 1:
                                right_side_peak = right_side_peak - 1
                            current_position = current_position + 1
                        right = (current_position-1) * 360.0 / len(sub_peak_array)
                        if number_of_peaks[idx] == 2 or abs(180 - (right - left)) < 35:
                            result_image[idx, current_direction] = (270.0 - ((left + right) / 2.0)) % 180
                        current_direction += 1

                    if current_direction == number_of_peaks[idx]//2:
                        break
    return result_image


@jit(nopython=True, parallel=True)
def _centroid_correction_bases(image, peak_image, reverse_peaks):
    left_bases = numpy.empty(image.shape)
    right_bases = numpy.empty(image.shape)

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

                target_peak_height = max(0, sub_image[pos] - max_pos * (1 - TARGET_PEAK_HEIGHT))
                left_position = MAX_DISTANCE_FOR_CENTROID_ESTIMATION
                right_position = MAX_DISTANCE_FOR_CENTROID_ESTIMATION

                # Check for minima in range
                for offset in range(MAX_DISTANCE_FOR_CENTROID_ESTIMATION):
                    if sub_reverse_peaks[pos - offset] == 1:
                        left_position = offset
                    if sub_reverse_peaks[(pos + offset) % len(sub_reverse_peaks)] == 1:
                        right_position = offset

                # Check for peak height
                for offset in range(abs(left_position)):
                    if sub_image[pos - offset] < target_peak_height:
                        left_position = offset
                        break
                for offset in range(right_position):
                    if sub_image[(pos + offset) % len(sub_image)] < target_peak_height:
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
    centroid_peaks = numpy.empty(image.shape)

    for idx in prange(image.shape[0]):
        sub_image = image[idx]
        sub_peaks = peak_image[idx]
        sub_left_bases = left_bases[idx]
        sub_right_bases = right_bases[idx]

        for pos in range(len(sub_peaks)):
            if sub_peaks[pos] == 1:
                centroid_sum_top = 0.0
                centroid_sum_bottom = 0.0
                for x in range(-sub_left_bases[pos], sub_right_bases[pos]):
                    img_pixel = sub_image[(pos + x) % len(sub_image)]
                    next_img_pixel = sub_image[(pos + x + 1) % len(sub_image)]
                    for interp in range(NUMBER_OF_SAMPLES+1):
                        step = interp / NUMBER_OF_SAMPLES
                        func_val = img_pixel + (next_img_pixel - img_pixel) * step
                        if func_val > sub_peaks[pos] * TARGET_PEAK_HEIGHT:
                            centroid_sum_top += (x + step) * func_val
                            centroid_sum_bottom += func_val

                centroid = centroid_sum_top / centroid_sum_bottom
                if centroid > 1:
                    centroid = 1
                if centroid < -1:
                    centroid = -1
                centroid_peaks[idx, pos] = centroid
            else:
                centroid_peaks[idx, pos] = 0
    return centroid_peaks
